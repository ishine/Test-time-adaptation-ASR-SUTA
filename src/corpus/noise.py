import os
from pathlib import Path
import numpy as np
import torch
import json
import librosa

from . import Define


class MUSANNoiseCorpus(object):
    """ free-sound noises from MUSAN """
    def __init__(self) -> None:
        self.root = f"{Define.MUSAN}/noise/free-sound"

    def __len__(self):
        return 843
    
    def get(self, idx: int) -> np.ndarray:
        basename = f"noise-free-sound-{idx % 843:04d}"
        noise, _ = librosa.load(f"{self.root}/{basename}.wav", sr=16000)

        return noise
