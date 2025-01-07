from typing import TypedDict, Any
from enum import Enum
import json
import os
import argparse
from multiprocessing import Queue, Process
from pathlib import Path
import traceback

from tqdm import tqdm
from dotenv import load_dotenv # type: ignore

from utils import *


def run_single_chunk(
    queue: Queue,
    config: RunConfig,
    chunk: tuple[int, int]
) -> None:
    dataset, agent = load(config, chunk)

    buffer_path = buffer_chunk_path(config, chunk)
    results: list[dict] = []\
        if not buffer_path.exists()\
        else json.load(open(buffer_path, "r"))
    n_pass = len(results)

    for input, label in dataset:
        if n_pass == 0:
            try:
                output = agent.run(input)
                result = agent.evaluate(dataset.evaluate_output, label, output)
                cost = output["initial_cost"] + sum([
                    i["critic_cost"] + i["refiner_cost"]
                    for i in output["iteration"]
                ])
            except:
                output = {
                    "error": traceback.format_exc()
                }
                cost = 0
                result = [False] * config["n_iter"]

            results.append({
                "input": input,
                "label": label,
                "output": output,
                "result": result,
                "cost": cost
            })
            json.dump(results, open(buffer_path, "w"), indent=2)
        else:
            n_pass -= 1

        queue.put(None, block=False)


def run_single_config(
    config: RunConfig,
    n_process: int,
    result_dir_path: Path
) -> None:
    chunks = generate_chunks(config["bench_config"]["slice"], n_process)
    n_queue = (lambda x: x[1] - x[0])(config["bench_config"]["slice"])

    queue = Queue()
    queue.cancel_join_thread()

    for chunk in chunks:
        p = Process(target=run_single_chunk, args=(queue, config, chunk))
        p.start()

    for _ in tqdm(range(n_queue)):
        queue.get()

    _load = lambda chunk: json.load(open(buffer_chunk_path(config, chunk), "r"))

    full: list[dict] = sum([_load(chunk) for chunk in chunks], [])
    full_score = calc_full_score(config, [*map(lambda x: x["result"], full)])

    save_results(
        config,
        result_dir_path,
        {
            "final_score": full_score,
            "generations": full
        }
    )

    for chunk in chunks:
        os.remove(buffer_chunk_path(config, chunk))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--queue_path", type=str)
    parser.add_argument("-n", "--n_process", type=int)
    parser.add_argument("-r", "--result_dir_path", type=str)
    parser.add_argument("--buffer_path", type=str, default=".cache/buffer")

    args = parser.parse_args()
    return args

def main() -> None:
    args = parse_args()
    queue_path: str = args.queue_path
    n_process: int = args.n_process
    result_dir_path = Path(args.result_dir_path)
    result_dir_path.mkdir(parents=True, exist_ok=True)
    BUFFER_PATH = Path(args.buffer_path)
    BUFFER_PATH.mkdir(parents=True, exist_ok=True)

    queue = load_queue(queue_path)

    for config in tqdm(queue, desc="configs"):
        if not save_path(config, result_dir_path).exists():
            run_single_config(config, n_process, result_dir_path)

if __name__ == "__main__":
    load_dotenv()
    main()