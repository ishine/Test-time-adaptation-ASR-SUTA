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
from visplot.utils import load_results


class SSUTARescoreStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()

        self._log = None

        self.info = load_results(exp_root=f"suta-rescore/benchmark/{self.config['task_name']}")

        self.temperature = 2.5
    
    def _init_start(self, sample) -> None:
        self.system.load_snapshot("init")
    
    def _adapt(self, sample):
        self.system.eval()
        is_collapse = False
        for _ in range(self.strategy_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=[sample["wav"]],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True
        if is_collapse:
            print("oh no")

    def _update(self, sample):
        pass
    
    def inference(self, idx: int) -> str:
        self.system.eval()
        with torch.no_grad():
            logits = self.info["logits"][idx]
            smoothed_logits = logits / self.temperature
            self._log["logits"].append(smoothed_logits)
            res = self.system.processor.batch_decode(np.expand_dims(smoothed_logits, axis=0), n_best=5, alpha=0.5, beta=0.0)
        merged_score = list(res.lm_score)[0]
        self._log["merged_score"].append(merged_score)
        nbest_trans = list(res.text)[0]
        self._log["nbest_trans"].append(nbest_trans)  # not exactly n results due to deduplication
        return nbest_trans[0]

    def run(self, ds: Dataset):
        long_cnt = 0
        self._log = defaultdict(list)
        for idx, sample in tqdm(enumerate(ds), total=len(ds)):
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
            self._log["n_words"].append(len(sample["text"].split(" ")))

            # self._init_start(sample)
            # self._adapt(sample)

            trans = self.inference(idx - long_cnt)
            err = wer(sample["text"], trans)
            self._log["wers"].append(err)
            self._log["transcriptions"].append((sample["text"], trans))
            self._log["basenames"].append(sample["id"])

            # loss
            # loss = self.system.calc_suta_loss([sample["wav"]])
            # ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            # loss["ctc_loss"] = ctc_loss["ctc_loss"]
            # self._log["losses"].append(loss)

            # self._log["logits"].append(self.system.calc_logits([sample["wav"]])[0])

            # self._update(sample)
            
        print("#Too long: ", long_cnt)
        
        return self._log
        
    def get_adapt_count(self):
        return self.system.adapt_count


class SSUTALLMStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self._log = None

        # LLM setup
        self.prompter = Prompter("nbest")
        self.llm_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        f = open('vocab.json')
        self.vocab = json.load(f)

        self.info = load_results(exp_root=f"ssuta-rescore/temp=1.6/{self.config['task_name']}")

    def _parse_res(self, res) -> str:
        llm_response = res.choices[0].message.content
        self._log["LLM"].append(llm_response)
        # print(llm_response)

        # normalize
        prefix = "The corrected transcription is: "
        idx = llm_response.find(prefix)
        try:
            assert idx >= 0
            ans = llm_response[idx+len(prefix):].strip().upper()
            trans = ""
            for c in ans:
                if c == " " or c in self.vocab:
                    trans += c
            return trans
        except:
            print(f"LLM format error: {llm_response}")
            raise
    
    def inference(self, idx: int) -> str:
        nbest_trans = self.info["nbest_trans"][idx]
        msg = []
        if "system_prompt" in self.prompter.template:
            msg.append({"role": "system", "content": self.prompter.template['system_prompt']})
        msg.append({"role": "user", "content": self.prompter.generate_prompt({"nbest": '\n'.join(nbest_trans)})})
        
        res = call_llm_OpenAI(self.llm_client, model_name="gpt-3.5-turbo-0125", msg=msg, max_retries=5)
        try:
            trans = self._parse_res(res)
        except:
            trans = nbest_trans[0]
        return trans

    def run(self, ds: Dataset):
        long_cnt = 0
        self._log = defaultdict(list)
        for idx, sample in tqdm(enumerate(ds), total=len(ds)):
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
            self._log["n_words"].append(len(sample["text"].split(" ")))

            trans = self.inference(idx - long_cnt)
            err = wer(sample["text"], trans)
            self._log["wers"].append(err)
            self._log["transcriptions"].append((sample["text"], trans))
            self._log["basenames"].append(sample["id"])
            
        print("#Too long: ", long_cnt)
        
        return self._log
    
    def get_adapt_count(self):
        return 0
