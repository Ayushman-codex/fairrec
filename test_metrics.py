from src.dummy import make_dummy_candidates
from src.fairness.rerank import rerank
from src.eval.metrics import gini_exposure, catalog_coverage, group_exposure, ndcg_at_k

cands = make_dummy_candidates()
truth = {u: set(g.nlargest(5, "score")["item_id"]) for u, g in cands.groupby("user_id")}

print("lam | ndcg@10 | gini | coverage | groups")
for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
    r = rerank(cands, lam=lam, k=10)
    print(lam, "|", round(ndcg_at_k(r, truth), 3), "|",
          round(gini_exposure(r), 3), "|",
          round(catalog_coverage(r, 500), 3), "|",
          group_exposure(r))