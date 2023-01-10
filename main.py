from multiprocessing import Pool
from agent import DQAgent
import torch
from game import BlackBox
from simple_network import simple_network
from datetime import datetime
import numpy as np
from hyperparams import Hyperparams
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from stable_baselines import PPO2



def play_episodes(episodes_to_play, model: simple_network, hyperparams: Hyperparams, env: BlackBox):
    memories = []
    reward = 0
    for _ in range(episodes_to_play):
        next_state = env.reset()
        done = False
        while(not done):
            state = next_state
            action = model.predict(state, env.valid_moves())
            next_state, reward, done = env.step(action)
            memories.append((action, state, next_state, reward, done))
    return memories

class Trainer:
    def __init__(self, hyperparams: Hyperparams) -> None:
        # params
        self.max_episodes: int = hyperparams.max_episodes
        self.game: BlackBox = BlackBox(hyperparams.resolution)

        self.agent: DQAgent = DQAgent(hyperparams)
        
        self.update_rate: int = 1
        self.test_games: int = hyperparams.test_games

        self.hyperparams = hyperparams

        self.writer = SummaryWriter()

    def update_writer(self, games_per_second, actions_per_second, episodes):
        self.writer.add_scalar("Other/Games per second", games_per_second, episodes//self.update_rate)
        self.writer.add_scalar("Other/Actions per second", actions_per_second, episodes//self.update_rate)


    def test(self, episodes):
        #Having a function to print is bad, should be fixed
        #Also make it plot or something 
        #This is very hacky...
        self.agent.testing = True
        actions = np.zeros((self.game.resolution, self.game.resolution))
        V = []
        moves = 0
        score = 0

        for i in tqdm(range(self.test_games), leave=False):
            next_state = self.game.reset()
            done = False
            while(not done):
                state = next_state
                move = self.agent.get_move(state, self.game.valid_moves())
                actions[move] += 1
                state, reward, done = self.game.step(move)
                moves += 1

                if i == self.test_games-1: #Only do this for last game
                    state = next_state
                    state = torch.tensor(state) if type(state) == np.ndarray else state
                    _, value, _ = self.agent.trainer.model(state.to(self.agent.trainer.device), return_separate = True)
                    V.append(value.item())

            score += reward

        print(f"\tAverage reward: {round(score/self.test_games, 2)}\n\tAverage actions: {int(moves/self.test_games)} moves\n\t")
        self.agent.testing = False

        self.writer.add_scalar("Average/Average reward", score/self.test_games, episodes//self.update_rate)
        self.writer.add_scalar("Average/Average actions", moves/self.test_games, episodes//self.update_rate)

        plt.plot(V)
        self.writer.add_figure("V and A/V as function of time", plt.gcf(), episodes//self.update_rate)
        plt.cla()
        plt.close()
        plt.imshow(actions)
        plt.colorbar()
        self.writer.add_figure("Actions done", plt.gcf(), episodes//self.update_rate)
        plt.cla()
        plt.close()

    def get_benchmark(self):
        start_time = time.time()
        sims = 100
        score = 0
        actions = 0
        start_score = 0
        self.agent.testing = True

        for _ in tqdm(range(sims), leave=False):
            self.game.reset()
            start_score += self.game.get_reward(True)
            done = False
            while(not done):
                move = self.agent._get_random(self.game.valid_moves())
                _, reward, done = self.game.step(move)
                actions += 1
            score += reward

        print(f"Benchmark: \n\tAverage reward: {round(score/sims, 2)}\n\tAverage score at start: {round(start_score/sims, 2)}\n\tAverage actions: {int(actions/sims)} moves\n\tPlayed {round(sims/(time.time()-start_time), 2)} g/s\n\tDid {round(actions/(time.time()-start_time), 2)} a/s\n\t")
        self.agent.testing = False

    def formate_time(self, seconds):
        #https://stackoverflow.com/a/775075
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if m == 0:
            return f'{int(s)} seconds' 
        if h == 0: 
            return f'{int(m)} minutes and {int(s)} seconds' 
        else:
            return f'{int(h)} hours, {int(m)} minutes and {int(s)} seconds'

    def save(self):
        torch.save(self.agent.trainer.model, "checkpoints/last") #For easily getting it
        torch.save(self.agent.trainer.model, 'models/model_'+ datetime.now().strftime("%m_%d_%Y%H_%M_%S"))

    def experience(self, models, hyperparams, envs, num_per_core):
        memories = 0
        exp_time = time.time()
        for model in models:
            model.reload_model(self.agent.trainer.model.state_dict())
        with Pool(processes=self.hyperparams.workers) as pool:
            multiple_results = [pool.apply_async(play_episodes, (n, m, h, e)) for n, m, h, e in zip(num_per_core, models, hyperparams, envs)]
            results = [res.get() for res in multiple_results]

        for res in results:
            memories += len(res)
            self.agent.make_memory(res)

        return memories, time.time()-exp_time

    def train(self, memories):
        train_time = time.time()
        self.agent.train((memories//self.hyperparams.batch_size)*self.hyperparams.train_per_memory)
        return time.time()-train_time


    def run(self, benchmark = False):
        start_time = time.time() 
        print(f"GPU available: {torch.cuda.is_available()}")
        if benchmark:
            self.get_benchmark()
        episodes = 0
        prev_time = start_time
        
        #Initialize distribution
        num_per_core = [self.hyperparams.game_batch_size for _ in range(self.hyperparams.workers)]
        hyperparams = [self.hyperparams for _ in range(self.hyperparams.workers)]
        envs = [BlackBox(self.hyperparams.resolution) for _ in range(self.hyperparams.workers)]
        models = [simple_network(self.hyperparams.resolution, self.hyperparams.action_space, self.hyperparams.action_masking, self.hyperparams.frame_stack, self.agent.trainer.model.state_dict(), envs[0].resolution) for _ in range(self.hyperparams.workers)]


        total_memories = 0
        #Main loop
        while (episodes < self.max_episodes):
            memories, exp_time = self.experience(models, hyperparams, envs, num_per_core)

            train_time = self.train(memories)

            total_memories += memories

            torch.save(self.agent.trainer.model, "checkpoints/last_checkpoint_in_case_of_crash")

            eps = sum(num_per_core)
            episodes += eps

            #time_left = (time.time()-prev_time)*(self.max_episodes-episodes)
            print(f"\nAt {round(episodes/1000, 1)}k games played.", #/{round(self.max_episodes/1000, 1)}k games played.",# ETA: {self.formate_time(int(time_left))}.",
            f"Playing {round(eps/(time.time()-prev_time), 2)} g/s and doing {round(memories/(time.time()-prev_time), 2)} a/s. Spent {self.formate_time(exp_time)} experiencing and {self.formate_time(train_time)} training.", 
            f"Trained on {int(self.agent.trained_times/1000)}k samples so far, done {int(total_memories/1000)}k actions")

            self.test(episodes)
            self.update_writer(games_per_second=eps/(time.time()-prev_time), actions_per_second=memories/(time.time()-prev_time), episodes=episodes)
            prev_time = time.time() 

        self.save()

        print(f"Finished training. Took {self.formate_time(int(time.time()-start_time))}.")



if __name__ == "__main__":

    hyperparams = Hyperparams() 
    #hyperparams.set_load_path("checkpoints\\last_checkpoint_in_case_of_crash")

    trainer = Trainer(hyperparams = hyperparams)
    trainer.run(True)