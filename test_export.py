from src.dummy import make_dummy_candidates
from src.export import build_all

cands = make_dummy_candidates()
truth = cands.sort_values("score", ascending=False).groupby("user_id").head(5)[["user_id", "item_id"]]
build_all(cands, truth, n_items=500, out_dir="results_demo")