import torch
from tanh_agent import Agent
from batched_env import BlackBox
env = BlackBox(batch_size=8, dims = 2, domain=(-1, 1))
agent = Agent(env.observation_space, dims = 2).to(torch.device("cuda"))

loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(agent.parameters(), lr = 1e-3)
s, t = env.reset()
for i in range(100):
    opt.zero_grad()
    action_mean, _, _ = agent.forward(s, t)
    action = torch.tanh(action_mean)
    target = env.GP.get_next_point(return_idx=False)
    loss = loss_fn(action_mean, target)
    loss.backward()
    opt.step()

    env.step(target)
    print(f"Epoch: {i} Loss: {round(loss.item(), 4)}")


torch.save(agent, "Pretrained_tanh_agent.t")