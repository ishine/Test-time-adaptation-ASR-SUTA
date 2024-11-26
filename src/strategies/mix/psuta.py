import os
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
from openai import OpenAI
import json
from collections import defaultdict

from ...system.suta import SUTASystem
from ...utils.tool import wer, call_llm_OpenAI
from ...utils.prompter import Prompter
from ..base import IStrategy
from .suta import SUTARescoreStrategy


class PSUTARescoreStrategy(SUTARescoreStrategy):    
    def _adapt(self, sample):
        self.system.eval()
        is_collapse = False
        for _ in range(self.strategy_config["steps"]):
            record = {}
            self.system.psuta_adapt(
                wavs=[sample["wav"]],
                record=record,
                mode="partial"
            )
            if record.get("collapse", False):
                is_collapse = True
        if is_collapse:
            print("oh no")
