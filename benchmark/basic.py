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
    
    def fix_adapt(self, sample):
        for _ in range(self.config["steps"]):
            res = self.system.adapt(
                [sample["wav"]],
                em_coef=self.config["em_coef"],
                reweight=self.config["reweight"],
                temp=self.config["temp"],
                not_blank=self.config["non_blank"],
                l2_coef=0  # SUTA default no regularization
            )
            if not res:
                break
        return res
    
    def _update(self, sample):
        self.system.load_snapshot("init")
        res = self.fix_adapt(sample)
        if not res:  # suta failed
            print("oh no")
            self.system.load_snapshot("init")
    
    def run(self, ds: Dataset):
        n_words = []
        errs, losses = [], []
        self.system.snapshot("init")
        for sample in tqdm(ds):
            n_words.append(len(sample["text"].split(" ")))

            # update
            self._update(sample)

            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
            loss = self.system.calc_loss(
                [sample["wav"]],
                em_coef=self.config["em_coef"],
                reweight=self.config["reweight"],
                temp=self.config["temp"],
                not_blank=self.config["non_blank"]
            )
            losses.append(loss)

            self.system.load_snapshot("init")
        
        return {
            "wers": errs,
            "n_words": n_words,
            "losses": losses,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count
            

class CSUTAStrategy(SUTAStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def fix_adapt(self, sample):
        for _ in range(self.config["steps"]):
            res = self.system.adapt(
                [sample["wav"]],
                em_coef=self.config["em_coef"],
                reweight=self.config["reweight"],
                temp=self.config["temp"],
                not_blank=self.config["non_blank"],
                l2_coef=self.config["l2_coef"],
            )
            if not res:
                break
        return res
    
    def _update(self, sample):
        self.system.snapshot("checkpoint")
        res = self.fix_adapt(sample)
        if not res:  # suta failed
            print("oh no")
            self.system.load_snapshot("checkpoint")
