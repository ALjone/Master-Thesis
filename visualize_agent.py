from game import BlackBox
from stable_baselines3 import PPO


model = PPO.load("Best")
game = BlackBox()

R = 0
t = 0
sims = 5000

s = game.reset()
done = False
#continue
while not done:
    #action, _states = model.predict(s)
    action = []
    for i in range(2):
        action.append(float(input()))

    s, r, done, info = game.step(action)
    print("Found max:", (info["pred_max"]/info["true_max"]).item(), action)
    game.render()
    R += r
    t += 1
