import random
import numpy as np
import torch

def torch_fix_seed(seed=19990210):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms = True

def seed_worker(seed=19990210):
    worker_seed = torch.initial_seed(seed) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)