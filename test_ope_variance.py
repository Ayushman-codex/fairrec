import numpy as np
from src.bandit.linucb import simulate
from src.ope.estimators import ips, doubly_robust

print("eps | max_weight | IPS std | DR std")
for eps in [0.5, 0.2, 0.1, 0.02]:
    v_ips, v_dr, mw = [], [], []
    for seed in range(15):
        log = simulate(n_rounds=2000, eps=eps, seed=seed)
        v_ips.append(ips(log))
        v_dr.append(doubly_robust(log))
        mw.append((0.1 / log["propensity"]).max())
    print(eps, "|", round(np.mean(mw), 1), "|",
          round(np.std(v_ips), 3), "|", round(np.std(v_dr), 3))