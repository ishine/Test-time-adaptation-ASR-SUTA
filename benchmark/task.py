from torch.utils.data import Dataset
import random

from corpus.corpus import LibriSpeechCorpus, SutaCorpus, CommonVoiceCorpus


def suta_corpus_idx_transform(idx):
    assert idx < 1000
    res = [0, 0, 0]
    cnt = 2
    while idx > 0:
        res[cnt] = idx % 10
        idx = idx // 10
        cnt -= 1
    return res


class Task1(Dataset):
    def __init__(self) -> None:
        self.corpus = SutaCorpus()
        self.idx_seq = list(range(len(self.corpus)))
        random.shuffle(self.idx_seq)
    
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        return self.corpus.get(*suta_corpus_idx_transform(self.idx_seq[idx]))


class Task2(Dataset):
    def __init__(self) -> None:
        self.corpus = SutaCorpus()
        self.idx_seq = []
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    self.idx_seq.append(i * 100 + j * 10 + k)
    
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        return self.corpus.get(*suta_corpus_idx_transform(self.idx_seq[idx]))


class Task3(Dataset):
    def __init__(self) -> None:
        self.corpus = SutaCorpus()
        self.idx_seq = []
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    self.idx_seq.append(j * 100 + i * 10 + k)

    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        return self.corpus.get(*suta_corpus_idx_transform(self.idx_seq[idx]))
    

class Task4(Dataset):
    def __init__(self) -> None:
        self.corpus = SutaCorpus()
        self.idx_seq = []
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    self.idx_seq.append(j * 100 + k * 10 + i)

    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        return self.corpus.get(*suta_corpus_idx_transform(self.idx_seq[idx]))


class Task5(Dataset):
    def __init__(self, extra_noise=0.0) -> None:
        self.corpus = LibriSpeechCorpus(extra_noise=extra_noise)
        self.idx_seq = list(range(len(self.corpus)))
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
