import numpy as np


def gini_exposure(recs_df) -> float:
    """0 = sab items ko barabar exposure, 1 = sab exposure ek item ko."""
    counts = np.sort(recs_df["item_id"].value_counts().values.astype(float))
    n = len(counts)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(counts) / (n * counts.sum()))


def catalog_coverage(recs_df, n_items: int) -> float:
    """Kitne % items kisi ko kabhi recommend hue."""
    return recs_df["item_id"].nunique() / n_items


def group_exposure(recs_df) -> dict:
    return recs_df["group"].value_counts(normalize=True).round(3).to_dict()


def ndcg_at_k(recs_df, truth: dict, k: int = 10) -> float:
    """truth = {user_id: set(relevant item_ids)}"""
    vals = []
    for u, g in recs_df.groupby("user_id"):
        rel = truth.get(u, set())
        if not rel:
            continue
        items = g.sort_values("rank")["item_id"].tolist()[:k]
        dcg = sum(1 / np.log2(i + 2) for i, x in enumerate(items) if x in rel)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), k)))
        vals.append(dcg / idcg)
    return float(np.mean(vals)) if vals else 0.0