from env import BlackBox
from stable_baselines3 import PPO


model = PPO.load("Best")
game = BlackBox()

R = 0
t = 0
sims = 5000

s = game.get_state()
done = False
#continue
while not done:
    action, _ = model.predict(s, deterministic=True)

    s, r, done, info = game.step(action)
    print("Found max:", round((info["pred_max"]/info["true_max"]).item(), 4), "Action:", action, "Reward:", round(r, 4))
    game.render()
    R += r
    t += 1
