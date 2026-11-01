import numpy as np
import pandas as pd


def _weights(log, n_arms, clip=None):
    # target policy = uniform random: har arm ki probability 1/n_arms
    w = (1.0 / n_arms) / log["propensity"].values
    if clip is not None:
        w = np.minimum(w, clip)
    return w


def ips(log, n_arms=10, clip=None):
    w = _weights(log, n_arms, clip)
    return float((w * log["reward"].values).mean())


def snips(log, n_arms=10, clip=None):
    w = _weights(log, n_arms, clip)
    return float((w * log["reward"].values).sum() / w.sum())


def doubly_robust(log, n_arms=10, clip=None):
    w = _weights(log, n_arms, clip)
    q = (log.groupby("action")["reward"].mean()
            .reindex(range(n_arms)).fillna(0.0).values)
    direct = q.mean()
    correction = w * (log["reward"].values - q[log["action"].values])
    return float((direct + correction).mean())


def bootstrap_ci(fn, log, n_boot=200, seed=0, **kw):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        s = log.sample(len(log), replace=True,
                       random_state=int(rng.integers(1_000_000_000)))
        vals.append(fn(s, **kw))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)