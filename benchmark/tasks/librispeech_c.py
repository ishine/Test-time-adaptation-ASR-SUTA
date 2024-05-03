from torch.utils.data import Dataset
import random

from corpus.corpus import LibriSpeechCCorpus


class RandomSequence(Dataset):
    def __init__(self, noise_type: str, snr_level=10) -> None:
        root = f"_cache/LibriSpeech-c/{noise_type}/snr={snr_level}"
        self.corpus = LibriSpeechCCorpus(root=root)
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
