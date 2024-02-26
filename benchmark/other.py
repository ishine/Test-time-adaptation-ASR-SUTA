from torch.utils.data import Dataset
from tqdm import tqdm

from .basic import BaseStrategy, CSUTAStrategy
from systems.suta1 import SUTASystem
from utils.tool import wer


class CSUTAResetStrategy(CSUTAStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def run(self, ds: Dataset):
        n_words = []
        errs = []
        self.system.snapshot("init")
        for idx, sample in tqdm(enumerate(ds), total=len(ds)):
            if idx % 100 == 0:
                self.system.load_snapshot("init")
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
