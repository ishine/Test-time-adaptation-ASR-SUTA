from torch.utils.data import Dataset
import yaml
from tqdm import tqdm

from systems.suta import SUTASystem
from utils.tool import wer
from .basic import BaseStrategy


class EMAStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system = SUTASystem(config)

        self.ema_task_vector = None
        self.alpha = 0.999

    def _ema_update(self):
        origin_model_state = self.system.history["init"][0]
        task_vector = self._get_task_vector()

        if self.ema_task_vector is None:
            assert "merged" not in self.system.history
            self.ema_task_vector = {}
            for name in origin_model_state:
                self.ema_task_vector[name] = (1 - self.alpha) * task_vector[name]
        else:
            for name in origin_model_state:
                self.ema_task_vector[name] = self.alpha * self.ema_task_vector[name] + (1 - self.alpha) * task_vector[name]
        
        merged_model_state = {
            name: origin_model_state[name] + self.ema_task_vector[name]
        for name in origin_model_state}
        self.system.history["merged"] = (merged_model_state, None, None)
        self.system.load_snapshot("merged")
        # print("merge tv")

    def _get_task_vector(self):
        model_state = self.system.model.state_dict()
        origin_model_state = self.system.history["init"][0]
        task_vector = {
            name: model_state[name] - origin_model_state[name]
        for name in model_state}
        return task_vector
    
    def adapt(self, sample):
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
            self._ema_update()
        return res
    
    def _update(self, sample):
        self.system.snapshot("checkpiont")
        res = self.adapt(sample)
        if not res:  # suta failed
            print("oh no")
            self.system.load_snapshot("checkpiont")
            self.ema_task_vector = self._get_task_vector()
    
    def run(self, ds: Dataset):
        n_words = []
        errs, losses = [], []
        task_boundaries = getattr(ds, "task_boundaries", [])
        self.system.snapshot("init")
        for idx, sample in tqdm(enumerate(ds), total=len(ds)):
            n_words.append(len(sample["text"].split(" ")))

            # update
            # if idx in task_boundaries:
            #     print("reset") 
            #     self.system.load_snapshot("init")
            #     self.ema_task_vector = None
            #     self.system.delete_snapshot("merged")
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
            # input()
        
        return {
            "wers": errs,
            "n_words": n_words,
            "losses": losses,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count
    

class TIESStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system = SUTASystem(config)

        self.ema_task_vector = None
        self.alpha = 0.999

    def _update_merged_model(self):
        model_state = self.system.model.state_dict()
        origin_model_state = self.system.history["init"][0]
        task_vector = {
            name: model_state[name] - origin_model_state[name]
        for name in model_state}
        # print("get tv")

        if self.ema_task_vector is None:
            assert "merged" not in self.system.history
            self.ema_task_vector = {}
            for name in model_state:
                self.ema_task_vector[name] = (1 - self.alpha) * task_vector[name]
        else:
            for name in model_state:
                self.ema_task_vector[name] = self.alpha * self.ema_task_vector[name] + (1 - self.alpha) * task_vector[name]
        
        merged_model_state = {
            name: origin_model_state[name] + self.ema_task_vector[name]
        for name in model_state}
        self.system.history["merged"] = (merged_model_state, None, None)
        # print("merge tv")
    
    def adapt(self, sample):
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
        start_point = "init" if self.ema_task_vector is None else "merged"
        self.system.load_snapshot(start_point)
        res = self.adapt(sample)
        if not res:  # suta failed
            print("oh no")
            self.system.load_snapshot(start_point)
        self._update_merged_model()
    
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
            # input()
        
        return {
            "wers": errs,
            "n_words": n_words,
            "losses": losses,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count