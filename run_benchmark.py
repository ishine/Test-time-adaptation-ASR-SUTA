import os
import argparse
import yaml
import json
import pickle

from utils.tool import seed_everything
from benchmark import get_strategy, get_task


def main(args):
    if args.exp_name == "":
        args.exp_name = args.strategy_name
    output_dir = f"results/benchmark/{args.task_name}/{args.exp_name}"
    os.makedirs(output_dir, exist_ok=True)

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    strategy = get_strategy(args.strategy_name)(config)
    task = get_task(args.task_name)
    results = strategy.run(task)

    assert len(results["n_words"]) == len(results["wers"])  # please do this
    err = 0
    for i in range(len(task)):
        err += results["wers"][i] * results["n_words"][i]
    denom = sum(results["n_words"])
    wer = err / denom
    print("Strategy: ", args.strategy_name)
    print("Task: ", args.task_name)
    print(f"WER: {wer * 100:.2f}%")
    print("Step count: ", strategy.get_adapt_count())

    # log
    log = {
        "exp_name": args.exp_name,
        "strategy": args.strategy_name,
        "task_name": args.task_name,
        "config": config,
        "results": {
            "wer": wer,
            "step_count": strategy.get_adapt_count()
        }
    }
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(log, f, indent=4)
    with open(f"{output_dir}/results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTA ASR")
    parser.add_argument('-s', '--strategy_name', type=str)
    parser.add_argument('-t', '--task_name', type=str)
    parser.add_argument('-n', '--exp_name', type=str, default="")
    parser.add_argument('--config', type=str, default="benchmark/config.yaml")
    
    args = parser.parse_args()
    seed_everything(666)
    main(args)
