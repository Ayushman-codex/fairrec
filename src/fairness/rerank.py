import pandas as pd


def rerank(cands: pd.DataFrame, lam: float, k: int = 10, target=None) -> pd.DataFrame:
    """lam=0: only relevance. lam=1: fairness gain dominates.
    target: dict group -> desired exposure share (default: equal per group)."""
    out = []
    groups = cands["group"].unique()
    if target is None:
        target = {grp: 1 / len(groups) for grp in groups}
    for u, g in cands.groupby("user_id"):
        g = g.copy()
        rng_ = g["score"].max() - g["score"].min() + 1e-9
        g["rel"] = (g["score"] - g["score"].min()) / rng_
        chosen_idx, counts = [], {grp: 0 for grp in g["group"].unique()}
        for pos in range(min(k, len(g))):
            share = {grp: counts[grp] / max(pos, 1) for grp in counts}
            rest = g.loc[~g.index.isin(chosen_idx)].copy()
            rest["gain"] = rest["group"].map(lambda x: max(target.get(x, 0) - share[x], 0))
            rest["final"] = (1 - lam) * rest["rel"] + lam * rest["gain"]
            best = rest.sort_values("final", ascending=False).iloc[0]
            chosen_idx.append(best.name)
            counts[best["group"]] += 1
        r = g.loc[chosen_idx, ["user_id", "item_id", "score", "group"]].copy()
        r["rank"] = range(1, len(r) + 1)
        out.append(r)
    return pd.concat(out, ignore_index=True)