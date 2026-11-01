from src.bandit.linucb import simulate

log = simulate()
print("LinUCB total reward :", log["reward"].sum())
print("Random total reward :", log["random_reward"].sum())
print("Propensity min/max  :", log["propensity"].min().round(3), log["propensity"].max().round(3))