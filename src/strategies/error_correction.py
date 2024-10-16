import os
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
from openai import OpenAI
import json
from collections import defaultdict

from ..system.suta import SUTASystem
from ..utils.tool import wer
from ..utils.prompter import Prompter
from .base import IStrategy


class RescoreStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()
    
    def run(self, ds: Dataset):
        long_cnt = 0
        basenames = []
        n_words = []
        errs, losses = [], []
        transcriptions = []
        logits = []
        for sample in tqdm(ds):
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
            n_words.append(len(sample["text"].split(" ")))
            trans = self.system.beam_inference([sample["wav"]], n_best=1)
            err = wer(sample["text"], trans[0])
            errs.append(err)
            transcriptions.append((sample["text"], trans[0]))
            basenames.append(sample["id"])

            # loss
            loss = self.system.calc_suta_loss([sample["wav"]])
            ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            loss["ctc_loss"] = ctc_loss["ctc_loss"]
            losses.append(loss)

            logits.append(self.system.calc_logits([sample["wav"]])[0])
        
        print("#Too long: ", long_cnt)
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "basenames": basenames,
            "losses": losses,
            "logits": logits,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count


class LLMStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()
        self._log = None

        # LLM setup
        # self.prompter = Prompter("LI-TTA")
        self.prompter = Prompter("GenSEC")
        self.llm_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        f = open('vocab.json')
        self.vocab = json.load(f)
    
    def _llm(self, nbest_trans):
        msg = []
        if "system_prompt" in self.prompter.template:
            msg.append({"role": "system", "content": self.prompter.template['system_prompt']})
        msg.append({"role": "user", "content": self.prompter.generate_prompt({"5best": '\n'.join(nbest_trans)})})
        res = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=msg,
            temperature=0,
        )
        res = res.choices[0].message.content
        # print(res)
        self._log["LLM"].append(res)

        # normalize
        prefix = "The true transcription from the 5-best hypotheses is: "
        try:
            assert res.startswith(prefix)
        except:
            print(f"LLM format error: {res}")
            return nbest_trans[0]
        new_str = ""
        x = res[len(prefix):].upper()
        for c in x:
            if c == " " or c in self.vocab:
                new_str += c
        return new_str

    # def _llm(self, nbest_trans):
    #     msg = []
    #     if "system_prompt" in self.prompter.template:
    #         msg.append({"role": "system", "content": self.prompter.template['system_prompt']})
    #     msg.append({"role": "user", "content": self.prompter.generate_prompt({"transcription": nbest_trans[0]})})
    #     res = self.llm_client.chat.completions.create(
    #         model="gpt-3.5-turbo-0125",
    #         messages=msg,
    #     )
    #     res = res.choices[0].message.content
    #     # print(res)
    #     self._log["LLM"].append(res)

    #     # normalize
    #     new_str = ""
    #     x = res.upper()
    #     for c in x:
    #         if c == " " or c in self.vocab:
    #             new_str += c
    #     return new_str

    def inference(self, sample) -> str:
        nbest_trans = self.system.beam_inference([sample["wav"]], n_best=5)[0]
        res = self._llm(nbest_trans)
        # print(res)

        return res

    def run(self, ds: Dataset):
        long_cnt = 0
        self._log = defaultdict(list)
        for sample in tqdm(ds):
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
            self._log["n_words"].append(len(sample["text"].split(" ")))

            self.system.eval()
            trans = self.inference(sample)
            err = wer(sample["text"], trans)
            self._log["wers"].append(err)
            self._log["transcriptions"].append((sample["text"], trans))
            
            # loss
            loss = self.system.calc_suta_loss([sample["wav"]])
            ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            loss["ctc_loss"] = ctc_loss["ctc_loss"]
            self._log["losses"].append(loss)

            self._log["logits"].append(self.system.calc_logits([sample["wav"]])[0])
            
        print("#Too long: ", long_cnt)
        
        return self._log
    
    def get_adapt_count(self):
        return self.system.adapt_count
