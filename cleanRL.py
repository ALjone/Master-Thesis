# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import time
from env.batched_env import BlackBox
from agents.pix_2_pix_agent import Agent as pix_agent

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils import load_config

def get_config():
    # fmt: off
    config = load_config("configs/training_config.yml")

    config.minibatch_size = int(config.batch_size // config.num_minibatches)


    return config


if __name__ == "__main__":
    config = get_config()
    run_name = "With positional encoding, medium entropy"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )


    device = torch.device("cuda")

    # env setup
    env: BlackBox = BlackBox(config)

    #agent = torch.load("Pretrained_tanh_agent.t") if args.pretrained else tanh_Agent(env.observation_space, args.dims).to(device)
    agent = pix_agent(env.observation_space, config.dims).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5, weight_decay=config.weight_decay)

    # ALGO Logic: Storage setup
    img_obs = torch.zeros((config.num_steps, ) + env.observation_space.shape).to(device)
    time_obs = torch.zeros((config.num_steps, config.batch_size)).to(device)
    actions_before_tanh = torch.zeros((config.num_steps, ) + env.action_space.shape).to(device)
    logprobs = torch.zeros((config.num_steps, config.batch_size)).to(device)
    rewards = torch.zeros((config.num_steps, config.batch_size)).to(device)
    dones = torch.zeros((config.num_steps, config.batch_size)).to(device)
    values = torch.zeros((config.num_steps, config.batch_size)).to(device)
    stds = torch.zeros((config.num_steps, config.batch_size)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_img_obs, next_time_obs = env.reset()
    next_done = torch.zeros(config.batch_size).to(device)
    num_updates = config.total_timesteps // config.batch_size
    for update in range(1, num_updates + 1):
        episodic_lengths = []
        episodic_returns = []
        episodic_peaks = []

        agent.eval()
        # Annealing the rate if instructed to do so.
        if config.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, config.num_steps):
            global_step += 1 * config.batch_size
            img_obs[step] = next_img_obs
            time_obs[step] = next_time_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action_before_tanh, logprob, _, value = agent.get_action_and_value(next_img_obs, next_time_obs)
                values[step] = value.flatten()
            actions_before_tanh[step] = action_before_tanh
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            (next_img_obs, next_time_obs), reward, next_done, info = env.step(action_before_tanh) #, False if isinstance(agent, tanh_Agent) else True)
            rewards[step] = reward.view(-1)

            if torch.sum(next_done) > 0:
                returns = torch.mean(info["episodic_returns"][next_done])
                lengths = torch.mean(info["episodic_length"][next_done])
                peak = torch.mean(info["peak"][next_done])
                episodic_lengths.append(lengths.item())
                episodic_returns.append(returns.item())
                episodic_peaks.append(peak.item())

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_img_obs, next_time_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.to(torch.long)
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_img_obs = img_obs.reshape((-1,) + env.observation_space.shape[1:])
        b_time_obs = time_obs.reshape(-1, )
        b_logprobs = logprobs.reshape(-1)
        b_actions_before_tanh = actions_before_tanh.reshape((-1,) + env.action_space.shape[1:])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(config.batch_size)
        clipfracs = []
        agent.train()
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_img_obs[mb_inds], b_time_obs[mb_inds], b_actions_before_tanh.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

            if config.target_kl is not None:
                if approx_kl > config.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/average_action", torch.mean(actions_before_tanh).item(), global_step)
        writer.add_scalar("charts/action_std", torch.std(stds).item(), global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/max_observation", next_img_obs.max(), global_step)
        writer.add_scalar("charts/min_observation", next_img_obs.min(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", (global_step / (time.time() - start_time)), "Global step:", global_step)#, "Log std:", ", ".join([str(param.item()) for param in agent.action_logstd]))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        writer.add_scalar("performance/episodic_return", np.mean(episodic_returns), global_step)
        writer.add_scalar("performance/episodic_length", np.mean(episodic_lengths), global_step)
        writer.add_scalar("performance/portion_of_max", np.mean(episodic_peaks), global_step)
        torch.save(agent.state_dict(), config.save_path)

    writer.close()