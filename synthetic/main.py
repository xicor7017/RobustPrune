import warnings
import hydra
from omegaconf import DictConfig, OmegaConf

import omegaconf
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@hydra.main(config_path="config", config_name="config")
def run_experiment(cfg: DictConfig):
    set_seed(cfg.seed)
    if cfg.mode.lower() == "overfit":
        from overfit import overfit
        overfit(cfg)
    elif cfg.mode.lower() == "prune":
        from prune import Prune
        Prune(cfg)
    else:
        raise ValueError("Invalid mode. Choose either 'overfit' or 'prune'.")

if __name__ == "__main__":
    run_experiment()