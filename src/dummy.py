import numpy as np
import pandas as pd


def make_dummy_candidates(n_users=200, n_items=500, k=100, seed=0):
    rng = np.random.default_rng(seed)
    pop = rng.zipf(1.5, n_items).astype(float)
    pop = pop / pop.sum()
    groups = np.where(pop > np.quantile(pop, 0.9), "head",
              np.where(pop > np.quantile(pop, 0.5), "mid", "tail"))
    rows = []
    for u in range(n_users):
        items = rng.choice(n_items, size=k, replace=False, p=pop)
        scores = np.sort(rng.random(k))[::-1] + pop[items] * 5
        for it, s in zip(items, scores):
            rows.append((u, int(it), float(s), groups[it]))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "score", "group"])