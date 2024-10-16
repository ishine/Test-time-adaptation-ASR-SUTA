from torch.utils.data import Dataset
import yaml
from tqdm import tqdm

from ..system.suta import SUTASystem
from ..utils.tool import wer
from .base import IStrategy
from .dsuta import Buffer, DSUTAStrategy


class OverfitStrategy(DSUTAStrategy):
    def _update(self, sample):
        self.memory.update(sample)
        if (self.timestep + 1) % self.update_freq == 0:
            self.slow_system.load_snapshot("start")
            self.slow_system.eval()
            record = {}
            self.slow_system.ctc_adapt_auto(
                wavs=[s["wav"] for s in self.memory.data],
                texts=[s["text"] for s in self.memory.data], 
                batch_size=1,
                record=record,
            )
            if record.get("collapse", False):
                print("oh no")
            self.slow_system.snapshot("start")
            self.memory.clear()
        self.system.history["start"] = self.slow_system.history["start"]  # fetch start point from slow system
