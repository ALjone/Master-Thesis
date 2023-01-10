import os.path 
import torch

class Hyperparams:
    def __init__(self) -> None:
        
        self.device: torch.DeviceObjType = torch.device("cuda")
        
        #trainer
        self.gamma: float = 1#0.99 #TRY HIGH AT SOME POINT
        self.lr: float = 0.0002 #0.0000625 Recommended from https://arxiv.org/pdf/1710.02298.pdf
        self.reg: float = 1e-4
        self.batch_size: int = 1024
        self.clip: int = 10 # 10 was suggested in Dueling heads paper
        self.tau = 0.001#0.0005
        self.wait_frames = 1000 #Wait at least this many frames before starting, according to https://arxiv.org/pdf/1710.02298.pdf
        self.load_path: str = None
        self.frame_stack = 1


        #replay
        self.beta = 0.7
        self.alpha = 0.5
        
        #main
        self.max_episodes: int = 5000000
        self.replay_size: int = 100000
        self.workers = 4
        #How many games per core to train
        self.game_batch_size: int = 1000
        self.test_games: int = max(self.batch_size//10, 1)
        #Important, as we need to train _at least_ once per move we make, otherwise priorization is dumb
        self.train_per_memory: int = 3

        #game
        self.resolution: int = 20
        self.lifespan: int = 100

        self.action_space = self.resolution**2
        #Note that this only affects the moves done by the network. Random moves will still be action masked
        self.action_masking = False
        


    def set_load_path(self, load_path: str) -> None:
        print("Loading a model, this is not a fresh run.")
        if os.path.isfile(load_path):
            self.load_path = load_path
        else:
            print("Couldn't find a file named", load_path)
