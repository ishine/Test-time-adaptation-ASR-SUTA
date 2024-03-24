from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm

from systems.suta import SUTASystem
from utils.tool import wer
from .basic import BaseStrategy

from dlhlp_lib.utils.data_structure import Queue


class SUTAStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system = SUTASystem(config)

        self.queue = Queue(max_size=5)

    def fix_adapt(self, sample):
        self.queue.update(sample)
        for _ in range(self.config["steps"]):
            res = self.system.adapt(
                [s["wav"] for s in self.queue.data],
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
        self.system.load_snapshot("init")
        res = self.fix_adapt(sample)
        if not res:  # suta failed
            print("oh no")
            self.system.load_snapshot("init")
    
    def run(self, ds: Dataset):
        n_words = []
        errs, losses = [], []
        transcriptions = []
        dataloader = DataLoader(
            ds,
            batch_size=1,
            num_workers=4,
            collate_fn=lambda x: x
        )
        self.system.snapshot("init")
        for sample in tqdm(dataloader):
            sample = sample[0]
            n_words.append(len(sample["text"].split(" ")))

            # update
            self._update(sample)

            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
            transcriptions.append((sample["text"], trans[0]))
            loss = self.system.calc_loss(
                [sample["wav"]],
                em_coef=self.config["em_coef"],
                reweight=self.config["reweight"],
                temp=self.config["temp"],
                not_blank=self.config["non_blank"]
            )
            losses.append(loss)
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "losses": losses,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count
            

class CSUTAStrategy(SUTAStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def _update(self, sample):
        self.system.snapshot("checkpoint")
        res = self.fix_adapt(sample)
        if not res:  # suta failed
            print("oh no")
            self.system.load_snapshot("checkpoint")
