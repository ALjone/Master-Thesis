from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image

class LoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_rollout_end(self):
        image = self.training_env.render(mode="rgb_array")
        # "HWC" specify the dataformat of the image, here channel last
        # (H for height, W for width, C for channel)
        # See https://pytorch.org/docs/stable/tensorboard.html
        # for supported formats
        self.logger.record("rollout/action_dist", Image(image, "HWC"), exclude=("stdout", "log", "json", "csv"))
        return True


    def _on_step(self):
        pass