from env import BlackBox

env = BlackBox()
while True:
    env.reset()
    env.step((0.1, 0.15))
    env.render()