import numpy as np
import torch
import random
import jiwer



def wer(a, b):
    return jiwer.wer(a, b, reference_transform=jiwer.wer_standardize, hypothesis_transform=jiwer.wer_standardize)


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
