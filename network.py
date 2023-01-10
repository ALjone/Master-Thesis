from typing import List
from prioritized_replay_memory import PrioritizedReplayBuffer
import torch
from hyperparams import Hyperparams
from model import SnakeBrain
#from experimental_model import SnakeBrain
import numpy as np

class DQTrainer:
    def __init__(self, hyperparams: Hyperparams) -> None:
        self.model: SnakeBrain = SnakeBrain(hyperparams.resolution, hyperparams.action_space, hyperparams.frame_stack)
        #TODO FP16!!
        
        #https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
        if hyperparams.load_path is not None:   
            weights = torch.load(hyperparams.load_path)
            try:
                self.model.load_state_dict(weights.state_dict())
                print("Successfully loaded model")
            except:
                own_state = self.model.state_dict()
                for name, param in weights.state_dict().items():
                    if "conv" not in name:
                        continue
                    own_state[name].copy_(param)
                print("Successfully loaded conv layers")
            

        #Try high
        self.gamma: float = hyperparams.gamma
        self.optim: torch.optim.Adam = torch.optim.Adam(self.model.parameters(), lr=hyperparams.lr, weight_decay=hyperparams.reg)

        self.device = hyperparams.device

        self.action_space = hyperparams.action_space

        self.target_model: SnakeBrain = SnakeBrain(hyperparams.resolution, hyperparams.action_space, hyperparams.frame_stack)
        self.target_model.load_state_dict(self.model.state_dict())

        self.model.to(self.device)
        self.target_model.to(self.device)

        self.batch_size = hyperparams.batch_size
        self.clip: int = hyperparams.clip

        self.tau = hyperparams.tau

        self.hyperparams = hyperparams

    def __rand_argmax(self, tens):
        argmaxes = torch.where(tens == tens.max())[1]
        return np.random.choice(argmaxes.cpu())

    def reload_model(self):
        if self.hyperparams.load_path is not None:
            self.model.load_state_dict(torch.load(self.hyperparams.load_path).state_dict())
        self.model.eval()
        self.model.to(self.device)


    def predict(self, features: torch.Tensor, valid_moves: torch.Tensor):
        with torch.no_grad():
            features = features.to(self.device)
            self.model.eval()
            pred = self.model(features)
            if not self.hyperparams.action_masking:
                return self.__rand_argmax(pred)
            #NOTE this is action masked unless the line above is uncommented
            #https://discuss.pytorch.org/t/masked-argmax-in-pytorch/105341
            valid_moves = valid_moves.to(self.device)
            large = torch.finfo (pred.dtype).max
            return self.__rand_argmax((pred - large * (1 - valid_moves) - large * (1 - valid_moves)))

    def random(self, valid_moves: torch.Tensor):
        valid_moves = valid_moves.cpu()#.to(self.device)
        max_inds, = torch.where(valid_moves == valid_moves.max())
        return np.random.choice(max_inds).item()

    def soft_update(self):
        for tp, sp in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)
    
    def train(self, bank: PrioritizedReplayBuffer) -> None:
        if len(bank) < self.hyperparams.wait_frames:
            return [], []

        self.model.train()
        self.soft_update()

        samples = bank.sample(min(len(bank), self.batch_size))
        return self._train_batch(samples)


    def _train_batch(self, samples):
        batch, weights, tree_idxs = samples

        self.optim.zero_grad()

        state, action, reward, next_state, done = batch

        with torch.no_grad():
            Q_next = self.target_model(next_state).max(dim=1).values
        Q_target = reward + self.gamma * (1 - done) * Q_next
        Q = self.model(state)[torch.arange(len(action)), action.to(torch.long).flatten()]

        td_error = torch.abs(Q - Q_target).detach()
        loss = torch.mean((Q - Q_target)**2 * weights)

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optim.step()

        return tree_idxs, td_error
