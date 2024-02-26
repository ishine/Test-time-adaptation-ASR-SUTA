from torch.utils.data import Dataset
import yaml
from tqdm import tqdm

from systems.suta1 import SUTASystem
from utils.tool import wer


class BaseStrategy(object):
    def run(self, ds: Dataset):
        raise NotImplementedError
    
    def get_adapt_count(self) -> int:
        return 0


class NoStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system = SUTASystem(self.config)
    
    def run(self, ds: Dataset):
        n_words = []
        errs = []
        for sample in tqdm(ds):
            n_words.append(len(sample["text"].split(" ")))
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
        
        return {
            "wers": errs,
            "n_words": n_words
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count


class SUTAStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system = SUTASystem(config)
    
    def run(self, ds: Dataset):
        n_words = []
        errs = []
        self.system.snapshot("init")
        for sample in tqdm(ds):
            for _ in range(self.config["steps"]):
                res = self.system.adapt(
                    [sample["wav"]],
                    em_coef=self.config["em_coef"],
                    reweight=self.config["reweight"],
                    temp=self.config["temp"],
                    not_blank=self.config["non_blank"]
                )
                if not res:
                    self.system.load_snapshot("init")
                    break

            n_words.append(len(sample["text"].split(" ")))
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)

            self.system.load_snapshot("init")
        
        return {
            "wers": errs,
            "n_words": n_words
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count
            

class CSUTAStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system = SUTASystem(config)
    
    def run(self, ds: Dataset):
        n_words = []
        errs = []
        self.system.snapshot("init")
        for sample in tqdm(ds):
            for _ in range(self.config["steps"]):
                res = self.system.adapt(
                    [sample["wav"]],
                    em_coef=self.config["em_coef"],
                    reweight=self.config["reweight"],
                    temp=self.config["temp"],
                    not_blank=self.config["non_blank"]
                )
                if not res:
                    self.system.load_snapshot("init")
                    break

            n_words.append(len(sample["text"].split(" ")))
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
        
        return {
            "wers": errs,
            "n_words": n_words
        }

    def get_adapt_count(self):
        return self.system.adapt_count
