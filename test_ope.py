from src.bandit.linucb import simulate
from src.ope.estimators import ips, snips, doubly_robust, bootstrap_ci

log = simulate(n_rounds=3000)
truth = log["random_reward"].mean()   # random policy ka asli reward (approx)
print("Truth (random policy):", round(truth, 3))

for name, fn in [("IPS", ips), ("SNIPS", snips), ("DR", doubly_robust)]:
    est = fn(log)
    lo, hi = bootstrap_ci(fn, log)
    print(f"{name:6s} estimate={est:.3f}  95% CI=({lo:.3f}, {hi:.3f})")

est = ips(log, clip=5)
print("IPS clip=5", round(est, 3))