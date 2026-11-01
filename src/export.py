import json
from pathlib import Path

import numpy as np

from src.fairness.rerank import rerank
from src.eval.metrics import gini_exposure, catalog_coverage, group_exposure, ndcg_at_k
from src.bandit.linucb import simulate
from src.ope.estimators import ips, snips, doubly_robust, bootstrap_ci


def _save(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def build_all(cands, truth_df, n_items, out_dir="results",
              lams=(0.0, 0.25, 0.5, 0.75, 1.0), k=10, demo_users=100):
    truth = {u: set(g["item_id"]) for u, g in truth_df.groupby("user_id")}
    demo = sorted(cands["user_id"].unique())[:demo_users]

    # 1) Fairness trade-off table + sample recommendations
    rows = []
    for lam in lams:
        r = rerank(cands, lam=lam, k=k)
        rows.append({
            "lam": lam,
            "ndcg": round(ndcg_at_k(r, truth, k), 4),
            "gini": round(gini_exposure(r), 4),
            "coverage": round(catalog_coverage(r, n_items), 4),
            "group_share": group_exposure(r),
        })
        sample = r[r["user_id"].isin(demo)]
        recs = {
            str(u): g.sort_values("rank")[["item_id", "rank", "group", "score"]].to_dict("records")
            for u, g in sample.groupby("user_id")
        }
        _save(recs, f"{out_dir}/recs_lam_{lam}.json")
    _save(rows, f"{out_dir}/tradeoff.json")

    # 2) Bandit (SIMULATION, not real users)
    log = simulate(n_rounds=3000)
    curve = {
        "round": log["round"].tolist()[::30],
        "linucb": log["reward"].cumsum().tolist()[::30],
        "random": log["random_reward"].cumsum().tolist()[::30],
        "note": "Simulation on synthetic users, not a real online experiment.",
    }
    _save(curve, f"{out_dir}/bandit.json")

    # 3) Off-policy evaluation (SIMULATION)
    truth_val = float(log["random_reward"].mean())
    ope = {"true_value": round(truth_val, 4), "estimators": []}
    for name, fn in [("IPS", ips), ("SNIPS", snips), ("DR", doubly_robust)]:
        lo, hi = bootstrap_ci(fn, log)
        ope["estimators"].append({"name": name, "value": round(fn(log), 4),
                                  "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    ope["note"] = "Logged-bandit data is simulated with known propensities."
    _save(ope, f"{out_dir}/ope.json")
    print("Saved JSON files in", out_dir)