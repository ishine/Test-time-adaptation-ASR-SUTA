import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm

from systems.suta import SUTASystem
from utils.tool import wer
from .basic import BaseStrategy


class CoTTAStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system = SUTASystem(config)
        self.teacher = SUTASystem(config)
        for param in self.teacher.model.parameters():
            param.detach_()

        self.ema_task_vector = None
        self.alpha = 0.999
        self.restore_ratio = config["restore_ratio"]

    @torch.no_grad
    def _ema_update_and_stochastic_restore(self):
        origin_model_state = self.system.history["init"][0]
        task_vector = self._get_task_vector(teacher=False)  # use system's task vector to update teacher

        if self.ema_task_vector is None:
            self.ema_task_vector = {}
            for name in origin_model_state:
                self.ema_task_vector[name] = (1 - self.alpha) * task_vector[name]
        else:
            for name in origin_model_state:
                self.ema_task_vector[name] = self.alpha * self.ema_task_vector[name] + (1 - self.alpha) * task_vector[name]
        
        for name in origin_model_state:
            param_to_update = self.teacher.model.state_dict()[name]
            param_to_update.data = (origin_model_state[name] + self.ema_task_vector[name]).data

        # stochastic_restore
        for nm, m in self.system.model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape) < self.restore_ratio).float().to(self.system.model.device)
                    p.data = origin_model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)

    def _get_task_vector(self, teacher=False):
        if teacher:
            model_state = self.teacher.model.state_dict()
        else:
            model_state = self.system.model.state_dict()
        model_state = self.system.model.state_dict()
        origin_model_state = self.system.history["init"][0]
        task_vector = {
            name: model_state[name] - origin_model_state[name]
        for name in model_state}
        return task_vector
    
    def adapt(self, sample):
        teacher_pl_target = self.teacher.inference([sample["wav"]])[0]
        # print("PL: ", teacher_pl_target)
        for _ in range(self.config["steps"]):
            res = self.system.pl_adapt(
                [sample["wav"]],
                transcriptions=[teacher_pl_target],
            )
            if not res:
                break
            self._ema_update_and_stochastic_restore()
        return res
    
    def _update(self, sample):
        self.system.snapshot("checkpoint")
        self.teacher.snapshot("checkpoint")
        res = self.adapt(sample)
        if not res:  # suta failed
            print("oh no")
            self.system.load_snapshot("checkpoint")
            self.teacher.load_snapshot("checkpoint")
            self.ema_task_vector = self._get_task_vector(teacher=True)
    
    def run(self, ds: Dataset):
        n_words = []
        errs, losses = [], []
        transcriptions = []
        self.system.snapshot("init")
        for idx, sample in tqdm(enumerate(ds), total=len(ds)):
            n_words.append(len(sample["text"].split(" ")))

            # update
            self._update(sample)

            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
            transcriptions.append((sample["text"], trans[0]))
            # print(sample["text"])
            # print(trans[0])
            # input()
        
        return {
            "wers": errs,
            "n_words": n_words,
            "losses": losses,
            "transcriptions": transcriptions,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count