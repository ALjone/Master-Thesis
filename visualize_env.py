from env import BlackBox

env = BlackBox()
while True:
    env.reset()
    env.render()