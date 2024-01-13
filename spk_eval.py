import os 
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from typing import Dict, List
import random

from dlhlp_lib.parsers.raw_parsers import LibriTTSRawParser, LibriTTSInstance

from data import collect_audio_batch
from systems.suta import SUTASystem
from utils.tool import wer, seed_everything


def infer(system: SUTASystem, instances) -> List[str]:
    res = []
    for instance in instances:
        data_batch = [(instance.wav_path, instance.text)]
        lens, wavs, texts, files = collect_audio_batch(data_batch, extra_noise=0.005)
        system.inference(wavs)
        trans = system.inference(wavs)
        res.extend(trans)
    return res


def adapt(system: SUTASystem, instances) -> None:
    for instance in instances:
        data_batch = [(instance.wav_path, instance.text)]
        lens, wavs, texts, files = collect_audio_batch(data_batch, extra_noise=0.005)
        system.adapt(wavs)


def main(args):
    src_parser = LibriTTSRawParser(args.dataset_dir)
    spk2instances = {}
    for instance in src_parser.test_other:
        if instance.speaker not in spk2instances:
            spk2instances[instance.speaker] = []
        spk2instances[instance.speaker].append(instance)
    # sexs = [src_parser.speaker_metadata[k]["sex"] for k in spk2instances]
    # print(list(zip(spk2instances.keys(), sexs)))
    # input()
    male_spks = ["3005", "6432", "8131", "4198", "7105"]
    female_spks = ["5484", "6070", "5442", "3528", "8280"]
    system = SUTASystem(args)
    system.snapshot("init")

    lines = []
    spks = male_spks + female_spks
    for tgt in tqdm(spks):
        gt = [instance.text for instance in spk2instances[tgt]]
        preds = infer(system, spk2instances[tgt])
        lines.append(("None", tgt, wer(gt, preds)))
    for src in spks:
        instances = []
        instance = random.sample(spk2instances[src], 1)[0]
        for _ in range(10):
            instances.append(instance)
        adapt(system, instances)
        for tgt in tqdm(spks):
            gt = [instance.text for instance in spk2instances[tgt]]
            preds = infer(system, spk2instances[tgt])
            lines.append((src, tgt, wer(gt, preds)))
        system.load_snapshot("init")
        # break
    # print(lines)
    with open("spk_results-n.txt", "w") as f:
        for line in lines:
            f.write(f"{line[0]}|{line[1]}|{line[2] * 100:.2f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTA ASR")
    parser.add_argument('--asr', type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--episodic', action='store_true')
    parser.add_argument('--div_coef', type=float, default=0.)
    parser.add_argument('--opt', type=str, default='AdamW')
    parser.add_argument('--dataset_name', type=str, default='libritts')
    parser.add_argument('--dataset_dir', type=str, default='/work/u7663915/Data/LibriTTS')
    parser.add_argument('--split', default=['test-other'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--em_coef', type=float, default=1.)
    parser.add_argument('--reweight', action='store_true')
    parser.add_argument('--bias_only', action='store_true')
    parser.add_argument('--train_feature', action='store_true')
    parser.add_argument('--train_LN', default=True)
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--temp', type=float, default=2.5)
    parser.add_argument('--non_blank', action='store_true')
    parser.add_argument('--log_dir', type=str, default='./exps')
    parser.add_argument('--extra_noise', type=float, default=0.)
    parser.add_argument('--scheduler', default=None)

    args = parser.parse_args()
    # fix to the best setting
    args.non_blank = True
    args.train_feature = True
    args.reweight = True
    args.lr = 2e-5
    args.em_coef = 0.3

    seed_everything(666)
    main(args)
