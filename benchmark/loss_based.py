from torch.utils.data import Dataset
from tqdm import tqdm

from .basic import BaseStrategy
from systems.suta1 import SUTASystem
from utils.tool import wer


class MultiExpertStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system = SUTASystem(config)

        # expert queue
        self.expert_keys = []
        self.expert_cnt = 0
        self.max_size = 10
    
    def add_expert(self):
        self.system.snapshot(str(self.expert_cnt))
        self.expert_keys.append(str(self.expert_cnt))
        self.expert_cnt += 1
        if len(self.expert_keys) > self.max_size:
            popped_key = self.expert_keys.pop(0)
            self.system.delete_snapshot(popped_key)

    def select_best_key_from_ref(self, refs) -> str:
        best_loss = 2e9
        for key in refs:
            if refs[key]["total_loss"] < best_loss:
                best_loss = refs[key]["total_loss"]
                best_key = key
        return best_key

    def run(self, ds: Dataset):
        n_words = []
        errs = []
        self.system.snapshot("init")
        for sample in tqdm(ds):
            n_words.append(len(sample["text"].split(" ")))

            # Query experts
            refs = {}
            for key in self.expert_keys + ["init"]:
                self.system.load_snapshot(key)
                loss = self.system.calc_loss(
                    [sample["wav"]],
                    em_coef=self.config["em_coef"],
                    reweight=self.config["reweight"],
                    temp=self.config["temp"],
                    not_blank=self.config["non_blank"]
                )
                refs[key] = loss
            
            best_key = self.select_best_key_from_ref(refs)
            if best_key == "init":  # use suta if "init" is the best
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
                if res:  # suta successed, add new expert
                    self.add_expert()
            else:
                self.system.load_snapshot(best_key)
            
            # inference
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
        
        return {
            "wers": errs,
            "n_words": n_words
        }

    def get_adapt_count(self):
        return self.system.adapt_count
