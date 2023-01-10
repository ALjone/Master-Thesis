import numpy as np
import torch
from hyperparams import Hyperparams
from network import DQTrainer
from prioritized_replay_memory import PrioritizedReplayBuffer

class DQAgent:
    def __init__(self, hyperparams: Hyperparams) -> None:
        
        self.action_space = hyperparams.action_space
        self.bank: PrioritizedReplayBuffer = PrioritizedReplayBuffer((3*hyperparams.frame_stack, hyperparams.resolution, hyperparams.resolution), 1, hyperparams.replay_size, hyperparams.device, beta =hyperparams.beta, alpha = hyperparams.alpha)
        self.device = hyperparams.device
        self.testing: bool = False

        self.trainer: DQTrainer = DQTrainer(hyperparams)
        self.trained_times = 0
        self.bsz = hyperparams.batch_size
        self.resolution = hyperparams.resolution

    def get_move(self, state: np.ndarray, valid_moves: np.ndarray) -> int:
        state = torch.tensor(state) if type(state) == np.ndarray else state
        valid_moves = torch.tensor(valid_moves) if type(valid_moves) == np.ndarray else valid_moves
        pred = self._predict(state, valid_moves)
        return divmod(pred, self.resolution)

    def make_memory(self, memories: list):
        for memory in memories:
            self._make_memory(*memory)

    def _make_memory(self, action: int, state: np.ndarray, next_state: np.ndarray, reward: float, done: bool) -> None:
        state = torch.tensor(state) if type(state) == np.ndarray else state
        next_state = torch.tensor(next_state) if type(next_state) == np.ndarray else next_state
        self.bank.add((state, torch.tensor(action), torch.tensor(reward), next_state if not done else torch.zeros(next_state.shape), int(done)))

    def train(self, train_times):
        """Call this after an episode is finished."""
        print("Train times: ", train_times)
        self.trained_times += train_times*self.bsz
        for _ in range(train_times):
            idxs, error = self.trainer.train(self.bank)
            self.bank.update_priorities(idxs, error)


    def _predict(self, state: torch.Tensor, valid_moves: torch.Tensor):
        return self.trainer.predict(state, valid_moves)

    def _get_random(self, valid_moves: torch.Tensor):
        valid_moves = torch.flatten(torch.tensor(valid_moves) if type(valid_moves) == np.ndarray else valid_moves)

        return divmod(self.trainer.random(valid_moves), self.resolution)