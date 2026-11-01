from src.dummy import make_dummy_candidates
from src.fairness.rerank import rerank

cands = make_dummy_candidates()
for lam in [0.0, 0.5, 1.0]:
    r = rerank(cands, lam=lam, k=10)
    print("lam =", lam)
    print(r["group"].value_counts(normalize=True).round(3).to_dict())