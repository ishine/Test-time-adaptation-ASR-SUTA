import os
import argparse
import yaml

from utils.tool import seed_everything
from benchmark import get_strategy, get_task


def main(args):
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    strategy = get_strategy(args.strategy_name)(config)
    task = get_task(args.task_name)
    results = strategy.run(task)

    err = 0
    for i in range(len(task)):
        err += results["wers"][i] * results["n_words"][i]
    denom = sum(results["n_words"])
    wer = err / denom
    print("Strategy: ", args.strategy_name)
    print("Task: ", args.task_name)
    print(f"WER: {wer * 100:.2f}%")
    print("Step count: ", strategy.get_adapt_count())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTA ASR")
    parser.add_argument('-s', '--strategy_name', type=str)
    parser.add_argument('-t', '--task_name', type=str)
    parser.add_argument('--config', type=str, default="benchmark/config.yaml")
    
    args = parser.parse_args()
    seed_everything(666)
    main(args)
