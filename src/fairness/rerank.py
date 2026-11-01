import pandas as pd


def rerank(cands: pd.DataFrame, lam: float, k: int = 10) -> pd.DataFrame:
    """lam=0: sirf relevance. lam=1: zyada fairness."""
    out = []
    target = 1 / cands["group"].nunique()
    for u, g in cands.groupby("user_id"):
        g = g.copy()
        rng_ = g["score"].max() - g["score"].min() + 1e-9
        g["rel"] = (g["score"] - g["score"].min()) / rng_
        chosen_idx, counts = [], {grp: 0 for grp in g["group"].unique()}
        for pos in range(min(k, len(g))):
            share = {grp: counts[grp] / max(pos, 1) for grp in counts}
            rest = g.loc[~g.index.isin(chosen_idx)].copy()
            rest["gain"] = rest["group"].map(lambda x: max(target - share[x], 0))
            rest["final"] = (1 - lam) * rest["rel"] + lam * rest["gain"]
            best = rest.sort_values("final", ascending=False).iloc[0]
            chosen_idx.append(best.name)
            counts[best["group"]] += 1
        r = g.loc[chosen_idx, ["user_id", "item_id", "score", "group"]].copy()
        r["rank"] = range(1, len(r) + 1)
        out.append(r)
    return pd.concat(out, ignore_index=True)