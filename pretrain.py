import torch
from tanh_agent import Agent
from batched_env import BlackBox
env = BlackBox(batch_size=1024, dims = 2, T = 60)

loss_fn = torch.nn.MSELoss()
print("Running with batch size:", env.batch_size)
for attempt in range(10):
    print("Attempt:", attempt, "\n")
    s, t = env.reset()
    agent = Agent(env.observation_space, dims = 2).to(torch.device("cuda"))
    opt = torch.optim.Adam(agent.parameters(), lr = 0.5e-4)
    losses = []
    should_break = False
    for i in range(2000):
        for _ in range(10):
            opt.zero_grad()
            action_mean, _, _ = agent.forward(s, t)
            action = torch.tanh(action_mean)
            target = env.GP.get_next_point(return_idx=False)
            loss: torch.Tensor = loss_fn(action, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            opt.step()
            losses.append(loss.item())
            if torch.isnan(target).any():
                print("Target is nan????")
            if torch.isnan(loss).any():
                print("Loss is NaN. Stopping training.")
                should_break = True
                break
            (s, t), _, _, _ = env.step(target)
        
        if should_break:
            break
        print(f"Epoch: {i+1} Loss: {round(sum(losses)/len(losses), 4)}")

        torch.save(agent, f"Pretrained_tanh_agent_{attempt}.t")
    if i > 100: #Make sure at least one attempt goes to 100
        break