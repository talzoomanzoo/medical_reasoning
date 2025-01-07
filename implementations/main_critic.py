from typing import TypedDict, Any, TypeVar
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
from runbox.benchmarks import Benchmark
from runbox.agents.self_refine_base import SelfRefineBase


def run_refinement(
    input: Any,
    label: Any,
    initial_response: str,
    initial_prediction: Any,
    initial_cost: float,
    agent: SelfRefineBase
) -> dict:
    output: dict = {
        "initial_response": initial_response,
        "initial_prediction": initial_prediction,
        "initial_cost": initial_cost,
        "iteration": []
    }

    stop = False
    response = initial_response
    prediction = initial_prediction
    critic_response: Any | None
    for _ in range(agent.n_iter):
        if not stop:
            critic_response, stop, critic_cost\
                = agent.run_critic(input, response, label)
        else:
            critic_response = None
            critic_cost = 0

        if not stop and critic_response is not None:
            refiner_response, refiner_prediction, refiner_cost\
                = agent.run_refiner(input, response, critic_response)

            output["iteration"].append({ # type: ignore
                "critic_response": critic_response,
                "refiner_response": refiner_response,
                "refiner_prediction": refiner_prediction,
                "critic_cost": critic_cost,
                "refiner_cost": refiner_cost
            })
            response = refiner_response
            prediction = refiner_prediction
        else:
            refiner_response = "-"
            refiner_cost = 0
            output["iteration"].append({ # type: ignore
                "critic_response": critic_response,
                "refiner_response": refiner_response,
                "refiner_prediction": prediction,
                "critic_cost": critic_cost,
                "refiner_cost": refiner_cost
            })

    return output

def run_single_chunk(
    queue: Queue,
    config: RunConfig,
    initial_infos: list[dict],
    chunk: tuple[int, int]
) -> None:
    dataset, agent = load(config, chunk)

    buffer_path = buffer_chunk_path(config, chunk)
    results: list[dict] = []\
        if not buffer_path.exists()\
        else json.load(open(buffer_path, "r"))
    n_pass = len(results)

    for (input, label), initial_info in zip(dataset, initial_infos):
        if n_pass == 0:
            try:
                output = run_refinement(
                    input,
                    label,
                    initial_info["initial_response"], # type: ignore
                    initial_info["initial_prediction"], # type: ignore
                    initial_info["initial_cost"],
                    agent
                )
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
    initial_infos: list[dict],
    n_process: int,
    result_dir_path: Path
) -> None:
    chunks = generate_chunks(config["bench_config"]["slice"], n_process)
    n_queue = (lambda x: x[1] - x[0])(config["bench_config"]["slice"])

    queue = Queue()
    queue.cancel_join_thread()

    for chunk in chunks:
        p = Process(target=run_single_chunk, args=(queue, config, initial_infos, chunk))
        p.start()

    for _ in tqdm(range(n_queue)):
        queue.get()

    _load = lambda chunk: json.load(open(buffer_chunk_path(config, chunk), "r"))

    full: list[dict] = sum([_load(chunk) for chunk in chunks], [])
    full_score = calc_full_score(config, [*map(lambda x: x["result"], full)])
    total_cost = sum(map(lambda x: x["cost"], full))

    save_results(
        config,
        result_dir_path,
        {
            "config": config,
            "final_score": full_score,
            "total_cost": total_cost,
            "generations": full
        }
    )

    for chunk in chunks:
        os.remove(buffer_chunk_path(config, chunk))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--initial_response_path", type=str)
    parser.add_argument("-q", "--queue_path", type=str)
    parser.add_argument("-n", "--n_process", type=int)
    parser.add_argument("-r", "--result_dir_path", type=str)
    parser.add_argument("--buffer_path", type=str, default=".cache/buffer")

    args = parser.parse_args()
    return args

def load_initial_infos(path: str) -> list[dict]:
    return json.load(open(path, "r"))["generations"]

def main() -> None:
    args = parse_args()
    initial_response_path: str = args.initial_response_path
    queue_path: str = args.queue_path
    n_process: int = args.n_process
    result_dir_path = Path(args.result_dir_path)
    result_dir_path.mkdir(parents=True, exist_ok=True)
    BUFFER_PATH = Path(args.buffer_path)
    BUFFER_PATH.mkdir(parents=True, exist_ok=True)

    queue = load_queue(queue_path)
    initial_infos = load_initial_infos(initial_response_path)

    for config in tqdm(queue, desc="configs"):
        if not save_path(config, result_dir_path).exists():
            run_single_config(config, initial_infos, n_process, result_dir_path)

if __name__ == "__main__":
    load_dotenv()
    main()