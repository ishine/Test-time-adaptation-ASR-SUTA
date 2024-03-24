import torch
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
        for param in self.anchor.model.parameters():
            param.detach_()
        for param in self.leader.model.parameters():
            param.detach_()

        self.ema_task_vector = None
        self.alpha = 0.999

    @torch.no_grad
    def _update_leader(self):
        origin_model_state = self.system.history["init"][0]
        task_vector = self._get_task_vector(leader=False)

        if self.ema_task_vector is None:
            self.ema_task_vector = {}
            for name in origin_model_state:
                self.ema_task_vector[name] = (1 - self.alpha) * task_vector[name]
        else:
            for name in origin_model_state:
                self.ema_task_vector[name] = self.alpha * self.ema_task_vector[name] + (1 - self.alpha) * task_vector[name]
        
        merged_model_state = {
            name: origin_model_state[name] + self.ema_task_vector[name]
        for name in origin_model_state}
        self.leader.history["merged"] = (merged_model_state, None, None)
        self.leader.load_snapshot("merged")

        # for name in origin_model_state:
        #     param_to_update = self.leader.model.state_dict()[name]
        #     param_to_update.data = (origin_model_state[name] + self.ema_task_vector[name]).data

    @torch.no_grad
    def _get_task_vector(self, leader=False):
        if leader:
            model_state = self.leader.model.state_dict()
        else:
            model_state = self.system.model.state_dict()
        origin_model_state = self.system.history["init"][0]
        task_vector = {
            name: model_state[name] - origin_model_state[name]
        for name in model_state}
        return task_vector
    
    def adapt(self, sample):  # AWMC use PL instead of unsupervised objectives
        anchor_pl_target = self.anchor.inference([sample["wav"]])[0]
        for _ in range(self.config["steps"]):
            leader_pl_target = self.leader.inference([sample["wav"]])[0]
            res = self.system.pl_adapt(
                [sample["wav"], sample["wav"]],
                transcriptions=[anchor_pl_target, leader_pl_target],
            )
            if not res:
                break
            self._update_leader()
        return res
    
    def _update(self, sample):
        self.system.snapshot("checkpoint")
        self.leader.snapshot("checkpoint")
        res = self.adapt(sample)
        if not res:  # suta failed
            print("oh no")
            self.system.load_snapshot("checkpoint")
            self.leader.load_snapshot("checkpoint")
            self.ema_task_vector = self._get_task_vector(leader=True)
    
    def run(self, ds: Dataset):
        n_words = []
        errs, losses = [], []
        transcriptions = []
        task_boundaries = getattr(ds, "task_boundaries", [])
        self.system.snapshot("init")
        self.leader.snapshot("init")
        for idx, sample in tqdm(enumerate(ds), total=len(ds)):
            n_words.append(len(sample["text"].split(" ")))

            # update
            if idx in task_boundaries:  # Original AWMC only test on single domain, therefore we should reset at task boundary
                print("reset")
                self.system.load_snapshot("init")
                self.leader.load_snapshot("init")
                self.ema_task_vector = None
            self._update(sample)

            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
            transcriptions.append((sample["text"], trans[0]))
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "losses": losses,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count