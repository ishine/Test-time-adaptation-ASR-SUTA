import os
import scipy.special
import torch
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
import json
from collections import defaultdict
import pickle
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from ...utils.tool import wer, call_llm_AsyncOpenAI, call_llm_OpenAI
from ...utils.prompter import Prompter
from ..base import IStrategy


class V0Strategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]

        self._log = None
        
        # LLM setup
        self.prompter = Prompter("select")
        self.llm_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        f = open('vocab.json')
        self.vocab = json.load(f)

        self._init_info()

    def _init_info(self):
        self.task_name = self.config["task_name"]
        try:
            with open(f"results/benchmark/LLM/benchmark/{self.task_name}/result/results.pkl", "rb") as f:
                sys1_results = pickle.load(f)
        except:
            with open(f"results/benchmark/aLLM/v0/{self.task_name}/result/results.pkl", "rb") as f:
                sys1_results = pickle.load(f)
        with open(f"results/benchmark/suta-LLM/benchmark/{self.task_name}/result/results.pkl", "rb") as f:
            sys2_results = pickle.load(f)
        # sanity check
        assert sys1_results["basenames"] == sys2_results["basenames"], "Mismatch! Please rerun previous expermients..."
        self.h1 = [x[1] for x in sys1_results["transcriptions"]]
        self.h2 = [x[1] for x in sys2_results["transcriptions"]]

        # load selected results
        self.selected = self._load_selected()
    
    def _load_selected(self) -> list[str]:
        tgt_file = f"results/benchmark/v0/_cache/{self.task_name}/selected.json"
        if os.path.exists(tgt_file):  # load from cache
            with open(tgt_file, "r") as f:
                res = json.load(f)
            return res

        os.makedirs(os.path.dirname(tgt_file), exist_ok=True)
        selected = []
        for (h1, h2) in tqdm(zip(self.h1, self.h2), total=len(self.h1), desc="Selecting"):
            if h1 == h2:
                ans = "h0"
                selected.append(ans)
                continue
            msg = []
            if "system_prompt" in self.prompter.template:
                msg.append({"role": "system", "content": self.prompter.template['system_prompt']})
            msg.append({"role": "user", "content": self.prompter.generate_prompt({"transcriptions": '\n'.join([h1, h2])})})
            
            res = call_llm_OpenAI(self.llm_client, model_name="gpt-3.5-turbo-0125", msg=msg, max_retries=5)
            try:
                trans = self._parse_res(res)
                if trans == h1:
                    ans = "h1"
                elif trans == h2:
                    ans = "h2"
                else:
                    print("LLM generates new transcription.")
                    raise NotImplementedError
            except:
                ans = "h0"
            # print(ans)
            selected.append(ans)
        with open(tgt_file, "w") as f:
            json.dump(selected, f, indent=4)
        return selected

    def _parse_res(self, res) -> str:
        llm_response = res.choices[0].message.content
        # print(llm_response)

        # normalize
        prefix = "The most reasonable transcription is: "
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
        if self.selected[idx] == "h0":
            trans = self.h2[idx]
        elif self.selected[idx] == "h1":
            trans = self.h1[idx]
        elif self.selected[idx] == "h2":
            trans = self.h2[idx]
        else:
            raise NotImplementedError
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


class V0PPLStrategy(V0Strategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]

        self._log = None
        
        # LLM setup
        self.scorer = PPL()
        self._init_info()
    
    def _load_selected(self) -> list[str]:
        tgt_file = f"results/benchmark/v0-ppl/_cache/{self.task_name}/selected.json"
        if os.path.exists(tgt_file):  # load from cache
            with open(tgt_file, "r") as f:
                res = json.load(f)
            return res

        os.makedirs(os.path.dirname(tgt_file), exist_ok=True)
        selected = []
        for (h1, h2) in tqdm(zip(self.h1, self.h2), total=len(self.h1), desc="Selecting"):
            if h1 == h2:
                ans = "h0"
                selected.append(ans)
                continue
            try:
                ans = "h2" if self.scorer.calc_ppl(h1) >= self.scorer.calc_ppl(h2) else "h1"
            except:
                ans = "h0"
            selected.append(ans)
        with open(tgt_file, "w") as f:
            json.dump(selected, f, indent=4)
        return selected


class PPL(object):
    def __init__(self):
        model_id="openai-community/gpt2"
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to("cuda")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    def calc_ppl(self, text) -> float:
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to("cuda")
        # print(input_ids)
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss
        ppl = torch.exp(neg_log_likelihood.mean())
        return ppl.item()


class V1Strategy(V0Strategy):
    # suta or LLM by average confidence
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]

        self._log = None
        
        # LLM setup
        self.scorer = PPL()
        self._init_info()

    def _init_info(self):
        self.task_name = self.config["task_name"]
        with open(f"results/benchmark/none/benchmark/{self.task_name}/result/results.pkl", "rb") as f:
            orig_results = pickle.load(f)
        try:
            with open(f"results/benchmark/LLM/benchmark/{self.task_name}/result/results.pkl", "rb") as f:
                sys1_results = pickle.load(f)
        except:
            with open(f"results/benchmark/aLLM/v0/{self.task_name}/result/results.pkl", "rb") as f:
                sys1_results = pickle.load(f)
        with open(f"results/benchmark/suta/benchmark-0/{self.task_name}/result/results.pkl", "rb") as f:
            sys2_results = pickle.load(f)
        # sanity check
        # assert sys1_results["basenames"] == sys2_results["basenames"], "Mismatch! Please rerun previous expermients..."
        self.h1 = [x[1] for x in sys1_results["transcriptions"]]
        self.h2 = [x[1] for x in sys2_results["transcriptions"]]
        self.logits = orig_results["logits"]

        # load selected results
        self.selected = self._load_selected()
    
    def _get_average_confidence(self, logit) -> float:
        logit = scipy.special.softmax(logit, axis=-1)
        return logit.max(axis=-1).mean()

    def _load_selected(self) -> list[str]:
        tgt_file = f"results/benchmark/v1/_cache/{self.task_name}/selected.json"
        if os.path.exists(tgt_file):  # load from cache
            with open(tgt_file, "r") as f:
                res = json.load(f)
            return res

        os.makedirs(os.path.dirname(tgt_file), exist_ok=True)
        selected = []
        for idx, (h1, h2) in tqdm(enumerate(zip(self.h1, self.h2)), total=len(self.h1), desc="Selecting"):
            if h1 == h2:
                ans = "h0"
                selected.append(ans)
                continue
            conf = self._get_average_confidence(self.logits[idx])
            ans = "h1" if conf > 0.9 else "h2"
            selected.append(ans)
        with open(tgt_file, "w") as f:
            json.dump(selected, f, indent=4)
        return selected


class V0AStrategy(IStrategy):
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

        self._init_info()

    def _init_info(self):
        self.task_name = self.config["task_name"]
        with open(f"results/benchmark/rescore/benchmark/{self.task_name}/result/results.pkl", "rb") as f:
            sys1_results = pickle.load(f)
        with open(f"results/benchmark/suta-rescore/benchmark/{self.task_name}/result/results.pkl", "rb") as f:
            sys2_results = pickle.load(f)
        # sanity check
        assert sys1_results["basenames"] == sys2_results["basenames"], "Mismatch! Please rerun previous expermients..."
        self.nbest1 = sys1_results["nbest_trans"]
        self.nbest2 = sys2_results["nbest_trans"]

    def _parse_res(self, res) -> str:
        llm_response = res.choices[0].message.content
        self._log["LLM"].append(llm_response)
        # print(res)

        # normalize
        prefix = "The corrected transcription is: "  # v0
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
        nbest_trans = self.nbest1[idx] + self.nbest2[idx]
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
