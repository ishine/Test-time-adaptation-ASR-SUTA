from torch.utils.data import Dataset
import random

from ..corpus.corpus import StandardCorpus


class SingleAccentSequence(Dataset):
    def __init__(self, accent: str="aus") -> None:
        self.corpus = StandardCorpus(root=f"_cache/CommonVoice-accent/{accent}")
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class AUSSequence(Dataset):
    def __new__(cls):
        return SingleAccentSequence(accent="aus")


class ENGSequence(Dataset):
    def __new__(cls):
        return SingleAccentSequence(accent="eng")
    

class INDSequence(Dataset):
    def __new__(cls):
        return SingleAccentSequence(accent="ind")
    

class IRESequence(Dataset):
    def __new__(cls):
        return SingleAccentSequence(accent="ire")
    

class SCOSequence(Dataset):
    def __new__(cls):
        return SingleAccentSequence(accent="sco")
    

class USSequence(Dataset):
    def __new__(cls):
        return SingleAccentSequence(accent="us")
