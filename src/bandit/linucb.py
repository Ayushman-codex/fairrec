import numpy as np
import pandas as pd


class LinUCB:
    def __init__(self, n_arms, d, alpha=1.0):
        self.alpha = alpha
        self.A = [np.eye(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]

    def scores(self, x):
        out = []
        for A, b in zip(self.A, self.b):
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            out.append(theta @ x + self.alpha * np.sqrt(x @ A_inv @ x))
        return np.array(out)

    def update(self, arm, x, reward):
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x


def simulate(n_rounds=3000, n_arms=10, d=5, alpha=1.0, eps=0.1, seed=0):
    """Nakli users pe bandit chalata hai. Har decision ki propensity log karta hai."""
    rng = np.random.default_rng(seed)
    true_theta = rng.normal(size=(n_arms, d))
    model = LinUCB(n_arms, d, alpha)
    rows = []
    for t in range(n_rounds):
        x = rng.normal(size=d)
        x = x / np.linalg.norm(x)
        p_reward = 1 / (1 + np.exp(-2 * (true_theta @ x)))

        best = int(np.argmax(model.scores(x)))
        probs = np.full(n_arms, eps / n_arms)
        probs[best] += 1 - eps
        arm = int(rng.choice(n_arms, p=probs))
        reward = int(rng.binomial(1, p_reward[arm]))
        model.update(arm, x, reward)

        rand_arm = int(rng.integers(n_arms))
        rand_reward = int(rng.binomial(1, p_reward[rand_arm]))
        rows.append((t, arm, probs[arm], reward, rand_reward))
    return pd.DataFrame(rows, columns=["round", "action", "propensity", "reward", "random_reward"])