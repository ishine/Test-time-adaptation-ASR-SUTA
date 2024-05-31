
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
import scipy.stats
import statistics

from systems.suta import SUTASystem
from utils.tool import wer

from dlhlp_lib.utils.data_structure import Queue

from .basic import BaseStrategy


class GaussianStatModel(object):
    def __init__(self, data: list[float]) -> None:
        self.mu = statistics.mean(data)
        self.std = statistics.stdev(data)
        # print(f"Gaussian model: mu={self.mu}, std={self.std}")
    
    def get_prob(self, x: float, reduction: int=1) -> float:  # reduce variance by multiple sampling
        return scipy.stats.norm(self.mu, self.std / (reduction ** 0.5)).cdf(x)
    
    def get_deviation(self, x: float, reduction: int=1) -> float:  # reduce variance by multiple sampling
        return (x - self.mu) / (self.std / (reduction ** 0.5))


class FixDomainExpert(object):
    def __init__(self, config) -> None:
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)
        self.timestep = 0

        # create stat (gaussian model)
        self.stat_model = None
        self.domain_update_freq = 50
        self.next_update_step = 2 * self.domain_update_freq
        self.domain_stats = []
        
        self.system.snapshot("start")

    def reset(self) -> None:
        self.timestep = 0
        self.system.load_snapshot("init")
        self.system.history = {}
        self.system.snapshot("init")

        self.stat_model = None
        self.domain_update_freq = 50
        self.next_update_step = 2 * self.domain_update_freq
        self.domain_stats = []
        self.system.snapshot("start")
    
    def collect_improvement_stats(self, data, snapshot_name: str) -> list[float]:
        init_losses, domain_losses = [], []
        self.system.load_snapshot("init")
        self.system.eval()
        for sample in data:
            loss = self.system.calc_suta_loss([sample["wav"]])
            init_losses.append(loss["total_loss"])
        
        self.system.load_snapshot(snapshot_name)
        self.system.eval()
        for sample in data:
            loss = self.system.calc_suta_loss([sample["wav"]])
            domain_losses.append(loss["total_loss"])
        return [x - y for x, y in zip(domain_losses, init_losses)]

    def build_stat_model(self, data):
        if self.stat_model is not None:  # FixDomainExpert's stat model will not change over time
            return
        
        if "domain" not in self.system.history:
            if self.timestep >= self.domain_update_freq:  # first creation
                self.system.snapshot("domain")
            else:
                return
        
        if self.timestep >= self.next_update_step:
            self.stat_model = GaussianStatModel(self.domain_stats)  # stat is collected from "domain"
            self.domain_stats = []
            return

        # Collect "loss improvement" stat for domain detector
        self.domain_stats.extend(self.collect_improvement_stats(data, "domain"))

    def is_distribution_shift(self, data) -> bool:
        if self.stat_model is None:
            return False
        
        # Collect current loss improvement stat and then judged by domain detector (Gaussian)
        data_stat = statistics.mean(self.collect_improvement_stats(data, "domain"))
        # p = self.stat_model.get_prob(data_stat, reduction=len(data))
        deviation = self.stat_model.get_deviation(data_stat, reduction=len(data))
        # print(f"Stat: {data_stat}, deviation={deviation}")
        
        # if p > 0.9999366575:  # +4std
        # if p > 0.9544997361: # +2std
        if deviation > 2:
            return True
        return False

    def update(self, data):
        self.system.load_snapshot("start")
        self.system.eval()
        record = {}
        self.system.suta_adapt_auto(
            wavs=[s["wav"] for s in data],
            batch_size=1,
            record=record,
        )
        if record.get("collapse", False):
            print("oh no")
        self.system.snapshot("start")
        self.timestep += len(data)


class DynamicDomainExpert(object):
    def __init__(self, config) -> None:
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)
        self.timestep = 0

        # create stat (gaussian model)
        self.stat_model = None
        self.domain_update_freq = 50
        self.next_update_step = 2 * self.domain_update_freq
        self.domain_stats = []
        
        self.system.snapshot("start")
    
    def reset(self) -> None:
        self.timestep = 0
        self.system.load_snapshot("init")
        self.system.history = {}
        self.system.snapshot("init")

        self.stat_model = None
        self.domain_update_freq = 50
        self.next_update_step = 2 * self.domain_update_freq
        self.domain_stats = []
        self.system.snapshot("start")
    
    def collect_improvement_stats(self, data, snapshot_name: str) -> list[float]:
        init_losses, domain_losses = [], []
        self.system.load_snapshot("init")
        self.system.eval()
        for sample in data:
            loss = self.system.calc_suta_loss([sample["wav"]])
            init_losses.append(loss["total_loss"])
        
        self.system.load_snapshot(snapshot_name)
        self.system.eval()
        for sample in data:
            loss = self.system.calc_suta_loss([sample["wav"]])
            domain_losses.append(loss["total_loss"])
        return [x - y for x, y in zip(domain_losses, init_losses)]

    def build_stat_model(self, data):
        if "domain" not in self.system.history:
            if self.timestep >= self.domain_update_freq:  # first creation
                self.system.snapshot("domain")
            else:
                return
        
        if self.timestep >= self.next_update_step:
            if "running_domain" in self.system.history:  # domain <- running_domain, running_domain <- current
                self.system.history["domain"] = self.system.history["running_domain"]  # update "domain"
            self.system.snapshot("running_domain")  # snapshot current status to "running_domain"
            self.stat_model = GaussianStatModel(self.domain_stats)  # stat is collected from "domain"
            self.domain_stats = []
            self.next_update_step += self.domain_update_freq

        # Collect "loss improvement" stat for domain detector
        compared_snapshot_name = "running_domain" if "running_domain" in self.system.history else "domain"
        self.domain_stats.extend(self.collect_improvement_stats(data, compared_snapshot_name))

    def is_distribution_shift(self, data) -> bool:
        if self.stat_model is None:
            return False
        
        # Collect current loss improvement stat and then judged by domain detector (Gaussian)
        data_stat = statistics.mean(self.collect_improvement_stats(data, "domain"))
        # p = self.stat_model.get_prob(data_stat, reduction=len(data))
        deviation = self.stat_model.get_deviation(data_stat, reduction=len(data))
        # print(f"Stat: {data_stat}, deviation={deviation}")
        
        # if p > 0.9999366575:  # +4std
        # if p > 0.9544997361: # +2std
        if deviation > 2:
            return True
        return False

    def update(self, data):
        self.system.load_snapshot("start")
        self.system.eval()
        record = {}
        self.system.suta_adapt_auto(
            wavs=[s["wav"] for s in data],
            batch_size=1,
            record=record,
        )
        if record.get("collapse", False):
            print("oh no")
        self.system.snapshot("start")
        self.timestep += len(data)



class FixFreqDomainExpert(object):
    def __init__(self, config) -> None:
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)
        self.timestep = 0

        self.domain_update_freq = 50
        
        self.system.snapshot("start")

    def reset(self) -> None:
        self.timestep = 0
        self.system.load_snapshot("init")
        self.system.history = {}
        self.system.snapshot("init")
        self.system.snapshot("start")
    
    def build_stat_model(self, data):
        pass

    def is_distribution_shift(self, data) -> bool:
        if self.timestep >= self.domain_update_freq:
            return True
        return False
        
    def update(self, data):
        self.system.load_snapshot("start")
        self.system.eval()
        record = {}
        self.system.suta_adapt_auto(
            wavs=[s["wav"] for s in data],
            batch_size=1,
            record=record,
        )
        if record.get("collapse", False):
            print("oh no")
        self.system.snapshot("start")
        self.timestep += len(data)


class DualDomainResetStrategy(BaseStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system_config = config["config"]
        self.system = SUTASystem(self.system_config)
        self.system.snapshot("start")

        # Set expert
        if config["strategy_config"]["reset"] == "fix":
            self.expert_cls = FixDomainExpert
        elif config["strategy_config"]["reset"] == "dynamic":
            self.expert_cls = DynamicDomainExpert
        elif config["strategy_config"]["reset"] == "fix-freq":
            self.expert_cls = FixFreqDomainExpert
        else:
            raise NotImplementedError
        self.expert = self.expert_cls(config)
        self.fail_cnt, self.patience = 0, config["strategy_config"]["patience"]

        self.timestep = 0
        self.update_freq = config["strategy_config"]["update_freq"]
        self.memory = Queue(max_size=config["strategy_config"]["memory"])

    def init_start(self, sample) -> None:
        if self.timestep % self.update_freq == 0 and self.timestep > 0:  # check reset
            self.expert.build_stat_model(self.memory.data)
            if self.expert.is_distribution_shift(self.memory.data):
                self.fail_cnt += 1
            else:
                self.fail_cnt = 0

            # handle reset
            if self.fail_cnt == self.patience:
                print("========== reset ==========")
                self.expert.reset()  # reset expert
                self.reset_record.append(self.timestep)
                self.fail_cnt = 0 
            
            self.expert.update(self.memory.data)
            self.memory.clear()
        
        # init
        self.system.history["start"] = self.expert.system.history["start"]  # fetch start point from slow system
        self.system.load_snapshot("start")
        self.memory.update(sample)
    
    def adapt(self, sample):
        self.system.eval()
        is_collapse = False
        for _ in range(self.system_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=[sample["wav"]],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True
        if is_collapse:
            print("oh no")
    
    def update(self, sample):
        self.init_start(sample)
        self.adapt(sample)
    
    def run(self, ds: Dataset):
        self.reset_record = []
        n_words = []
        errs, losses = [], []
        transcriptions = []
        for sample in tqdm(ds):
            n_words.append(len(sample["text"].split(" ")))

            # update
            self.update(sample)

            self.system.eval()
            trans = self.system.inference([sample["wav"]])[0]
            err = wer(sample["text"], trans)
            errs.append(err)
            transcriptions.append((sample["text"], trans))
            
            # loss
            loss = self.system.calc_suta_loss([sample["wav"]])
            ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            loss["ctc_loss"] = ctc_loss["ctc_loss"]
            losses.append(loss)

            self.timestep += 1
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "losses": losses,
            "reset_step": self.reset_record,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count




# Deprecated
# class DomainExpert(object):
#     def __init__(self, config) -> None:
#         self.system_config = config["config"]
#         self.system = SUTASystem(self.system_config)
#         self.timestep = 0

#         # create stat (gaussian model)
#         self.create_anchor_step = 50
#         self.create_stat_step = 100
#         self.in_domain_stats = []
#         self.running_domain_stats = Queue(max_size=50)
        
#         self.system.snapshot("start")
#         self.stage = "create_anchor"
    
#     def build_stat_model(self, data):
#         if self.stage != "create_stat":
#             return
        
#         if self.timestep >= self.create_stat_step:  # stat created
#             ws = 10  # mean window
#             in_domain_stats = [(sum(self.in_domain_stats[i-ws:i]) / ws) for i in range(ws, len(self.in_domain_stats))]
#             self.mu = statistics.mean(in_domain_stats)
#             self.std = statistics.stdev(in_domain_stats)
#             print(f"Gaussian model: mu={self.mu}, std={self.std}")
#             self.stage = "stat created"
#             print(self.stage)
#             return

#         # Calculate difference
#         init_losses, in_domain_losses, running_domain_losses = [], [], []
#         self.system.load_snapshot("init")
#         self.system.eval()
#         for sample in data:
#             loss = self.system.calc_suta_loss([sample["wav"]])
#             init_losses.append(loss["total_loss"])
#         self.system.load_snapshot("domain")
#         self.system.eval()
#         for sample in data:
#             loss = self.system.calc_suta_loss([sample["wav"]])
#             in_domain_losses.append(loss["total_loss"])
#         self.in_domain_stats.extend([x - y for x, y in zip(in_domain_losses, init_losses)])

#         # self.system.load_snapshot("start")
#         # self.system.eval()
#         # for sample in data:
#         #     loss = self.system.calc_suta_loss([sample["wav"]])
#         #     running_domain_losses.append(loss["total_loss"])
#         # for x, y in zip(running_domain_losses, init_losses):
#         #     self.running_domain_stats.update(x - y)

#     def is_distribution_shift(self, data) -> bool:
#         if self.stage != "stat created":
#             return False
        
#         # Calculate difference
#         init_losses, domain_losses = [], []
#         self.system.load_snapshot("init")
#         self.system.eval()
#         for sample in data:
#             loss = self.system.calc_suta_loss([sample["wav"]])
#             init_losses.append(loss["total_loss"])
#         self.system.load_snapshot("domain")
#         self.system.eval()
#         for sample in data:
#             loss = self.system.calc_suta_loss([sample["wav"]])
#             domain_losses.append(loss["total_loss"])
#         domain_diff = [x - y for x, y in zip(domain_losses, init_losses)]
#         domain_shift = statistics.mean(domain_diff)
#         p = scipy.stats.norm(self.mu, self.std).cdf(domain_shift)
#         # print(f"Shift: {domain_shift}, p={p}")

#         if p > 0.9999366575:  # +4std
#             return True
        
#         # # track running shift
#         # self.system.load_snapshot("start")
#         # self.system.eval()
#         # for sample in data:
#         #     loss = self.system.calc_suta_loss([sample["wav"]])
#         #     domain_losses.append(loss["total_loss"])
#         # running_domain_diff = [x - y for x, y in zip(domain_losses, init_losses)]
#         # running_domain_shift = statistics.mean(running_domain_diff)
#         # mu = statistics.mean(self.running_domain_stats.data)
#         # std = statistics.variance(self.running_domain_stats.data)
#         # p = scipy.stats.norm(mu, std).cdf(running_domain_shift)
#         # print(f"Running model: mu={mu}, std={std}")
#         # print(f"Running Shift: {running_domain_shift}, p={p}")

#         # if p > 0.9999366575:  # >4std
#         #     return True
#         # # running update if <=2std
#         # if p <= 0.9544997361:
#         #     for x in running_domain_diff:
#         #         self.running_domain_stats.update(x)

#     def update(self, data):
#         self.system.load_snapshot("start")
#         self.system.eval()
#         record = {}
#         self.system.suta_adapt_auto(
#             wavs=[s["wav"] for s in data],
#             batch_size=1,
#             record=record,
#         )
#         if record.get("collapse", False):
#             print("oh no")
#         self.system.snapshot("start")

#         self.timestep += len(data)
#         if self.timestep >= self.create_anchor_step and self.stage == "create_anchor":
#             self.system.snapshot("domain")
#             self.stage = "create_stat"
#             print(self.stage)

