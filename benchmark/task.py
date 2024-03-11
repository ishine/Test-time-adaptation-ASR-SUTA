from torch.utils.data import Dataset
import random

from corpus.corpus import (
    LibriSpeechCorpus, SutaCorpus, CommonVoiceCorpus,
    ESDCorpus, L2ArcticCorpus, NoisyL2ArcticCorpus,
    CHIMECorpus
)


class Task1(Dataset):
    def __init__(self) -> None:
        self.corpus = SutaCorpus()
        self.idx_seq = list(range(len(self.corpus)))
        random.shuffle(self.idx_seq)
    
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class Task2(Dataset):
    def __init__(self) -> None:
        self.corpus = SutaCorpus()
        self.idx_seq = []
        self.task_boundaries = []
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    self.idx_seq.append(i * 100 + j * 10 + k)
            self.task_boundaries.append(len(self.idx_seq))
    
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class Task3(Dataset):
    def __init__(self) -> None:
        self.corpus = SutaCorpus()
        self.idx_seq = []
        self.task_boundaries = []
        for j in range(10):
            for i in range(10):
                for k in range(10):
                    self.idx_seq.append(i * 100 + j * 10 + k)
            self.task_boundaries.append(len(self.idx_seq))
        
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class Task4(Dataset):
    def __init__(self) -> None:
        self.corpus = SutaCorpus()
        self.idx_seq = []
        self.task_boundaries = []
        for k in range(10):
            for i in range(10):
                for j in range(10):
                    self.idx_seq.append(i * 100 + j * 10 + k)
            self.task_boundaries.append(len(self.idx_seq))

    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class Task5(Dataset):
    def __init__(self, extra_noise=0.0) -> None:
        self.corpus = ESDCorpus(extra_noise=extra_noise)
        self.idx_seq = list(range(len(self.corpus)))
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class Task6(Dataset):
    def __init__(self, extra_noise=0.0) -> None:
        self.corpus = ESDCorpus(extra_noise=extra_noise)
        self.idx_seq = []
        self.task_boundaries = []
        for i in range(10):
            for j in range(5):
                for k in range(30):
                    self.idx_seq.append(i * 150 + j * 30 + k)
            self.task_boundaries.append(len(self.idx_seq))

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class Task7(Dataset):
    def __init__(self, extra_noise=0.0) -> None:
        self.corpus = ESDCorpus(extra_noise=extra_noise)
        self.idx_seq = []
        self.task_boundaries = []
        for j in range(5):
            for i in range(10):
                for k in range(30):
                    self.idx_seq.append(i * 150 + j * 30 + k)
            self.task_boundaries.append(len(self.idx_seq))

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class Task8(Dataset):
    def __init__(self, extra_noise=0.0) -> None:
        self.corpus = ESDCorpus(extra_noise=extra_noise)
        self.idx_seq = []
        self.task_boundaries = []
        for k in range(30):
            for i in range(10):
                for j in range(5):
                    self.idx_seq.append(i * 150 + j * 30 + k)
            self.task_boundaries.append(len(self.idx_seq))

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class Trial(Dataset):
    def __init__(self, extra_noise=0.0) -> None:
        self.corpus = ESDCorpus(extra_noise=extra_noise)
        self.idx_seq = []
        for j in range(5):
            if j == 2:
                continue
            for i in range(10):
                for k in range(30):
                    self.idx_seq.append(i * 150 + j * 30 + k)

    def __len__(self):
        return len(self.idx_seq)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class Task13(Dataset):
    def __init__(self, extra_noise=0.0) -> None:
        self.corpus = LibriSpeechCorpus(extra_noise=extra_noise)
        self.idx_seq = list(range(len(self.corpus)))
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class Task14(Dataset):
    def __init__(self) -> None:
        self.corpus = CHIMECorpus()
        self.idx_seq = list(range(len(self.corpus)))
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class Task9(Dataset):
    def __init__(self) -> None:
        self.corpus = NoisyL2ArcticCorpus()
        self.idx_seq = list(range(len(self.corpus)))
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class Task10(Dataset):
    def __init__(self) -> None:
        self.corpus = NoisyL2ArcticCorpus()
        self.idx_seq = []
        self.task_boundaries = []
        for k, accent_name in self.corpus.accent2str.items():
            spk_idxs = [self.corpus.spks.index(spk) for spk in self.corpus.spks if accent_name in spk]
            for spk_idx in spk_idxs:
                self.idx_seq.extend([spk_idx * self.corpus.n_per_spk + j for j in range(self.corpus.n_per_spk)])
            self.task_boundaries.append(len(self.idx_seq))
    
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class Task11(Dataset):
    def __init__(self) -> None:
        self.corpus = NoisyL2ArcticCorpus()
        self.idx_seq = []
        self.task_boundaries = []
        for j in range(self.corpus.n_per_spk):
            for i in range(24):
                self.idx_seq.append(i * self.corpus.n_per_spk + j)
            self.task_boundaries.append(len(self.idx_seq))

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class Task12(Dataset):
    def __init__(self) -> None:
        self.corpus = NoisyL2ArcticCorpus()
        self.idx_seq = []
        self.task_boundaries = []
        for i in range(24):
            for j in range(self.corpus.n_per_spk):
                self.idx_seq.append(i * self.corpus.n_per_spk + j)
            self.task_boundaries.append(len(self.idx_seq))

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
