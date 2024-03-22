from torch.utils.data import Dataset
import yaml
from tqdm import tqdm

from systems.suta import SUTASystem
from utils.tool import wer
from .basic import BaseStrategy


class AWMCStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.anchor = SUTASystem(config)
        self.system = SUTASystem(config)  # chaser
        self.leader = SUTASystem(config)

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
            self.ema_task_vector = {}
            for name in model_state:
                self.ema_task_vector[name] = (1 - self.alpha) * task_vector[name]
        else:
            for name in model_state:
                self.ema_task_vector[name] = self.alpha * self.ema_task_vector[name] + (1 - self.alpha) * task_vector[name]
        
        merged_model_state = {
            name: origin_model_state[name] + self.ema_task_vector[name]
        for name in model_state}
        self.leader.history["merged"] = (merged_model_state, None, None)
        self.leader.load_snapshot("merged")
        # print("merge tv")
    
    def adapt(self, sample):  # AWMC use PL instead of unsupervised objectives
        anchor_pl_target = self.anchor.inference([sample["wav"]])[0]
        leader_pl_target = self.leader.inference([sample["wav"]])[0]
        for _ in range(self.config["steps"]):
            res = self.system.pl_adapt(
                [sample["wav"], sample["wav"]],
                transcriptions=[anchor_pl_target, leader_pl_target],
            )
            if not res:
                break
        return res
    
    def _update(self, sample):
        self.system.snapshot("checkpoint")
        res = self.adapt(sample)
        if not res:  # suta failed
            print("oh no")
            self.system.snapshot("checkpoint")
        self._update_merged_model()
    
    def run(self, ds: Dataset):
        n_words = []
        errs, losses = [], []
        self.system.snapshot("init")
        self.leader.snapshot("init")
        self.leader.snapshot("merged")
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