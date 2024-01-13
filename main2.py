import os 
import torch
import pandas as pd
import argparse
from tqdm import tqdm

from data import load_dataset
from systems.suta import SUTASystem
from utils.tool import wer


def main(args):
    # dataset and model
    dataset = load_dataset(args.split, args.dataset_name, args.dataset_dir, args.batch_size, args.extra_noise)
    system = SUTASystem(args)

    # Start
    log = True
    check_steps = [1, 3, 5, 10, 20, 40]
    transcriptions = {"gt": [], "origin": []}
    for i in check_steps:
        transcriptions[i] = []

    system.snapshot("init")
    for batch_idx, batch in enumerate(dataset):
        lens, wavs, texts, files = batch
        texts = list(texts)
        transcriptions["gt"].extend(texts)

        trans = system.inference(wavs)
        transcriptions["origin"].extend(trans)

        if log:
            ori_wer = wer(texts, trans)
            print("original WER: ", ori_wer)
            # print(texts, trans)

        for i in range(args.steps):
            system.adapt(wavs)
            if i + 1 in check_steps:
                trans = system.inference(wavs)
                transcriptions[i + 1].extend(trans)
                if log:
                    ada_wer = wer(texts, trans)
                    print(f"adapt-{i+1} WER:  ", ada_wer)
                    # print(texts, trans)
        system.load_snapshot("init")
        if batch_idx > 10:
            break
    
    print('------------------------------------')
    print("original WER:", wer(transcriptions["gt"], transcriptions["origin"]))
    for t in check_steps:
        if t > args.steps:
            break
        print(f"TTA-{t} WER:", wer(transcriptions["gt"], transcriptions[t]))
    print('------------------------------------')


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
    main(args)
