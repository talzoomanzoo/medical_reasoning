from typing import TypedDict, Any, Callable, Literal, TypeVar, Mapping
from enum import Enum
from pathlib import Path
import json

from runbox.benchmarks import Benchmark, SupportsBenchmark
from config_genrm import prepare as prepare_genrm
from config_mcsr import prepare as prepare_mcsr
from config_sr import prepare as prepare_sr


BUFFER_PATH = Path(".cache/buffer")


class BenchConfig(TypedDict):
    split: str
    slice: tuple[int, int]

class RunConfig(TypedDict):
    benchmark: str
    bench_config: BenchConfig
    method: Literal["sr", "mcsr", "genrm"]
    models: list[str]
    n_iter: int


_BenchInput = TypeVar("_BenchInput", bound=Mapping[str, Any])
_BenchOutput = TypeVar("_BenchOutput", contravariant=True)
_BenchEvalResult = TypeVar("_BenchEvalResult")
_AgentRowResult = TypeVar("_AgentRowResult", covariant=True)
type Prepare = Callable[
    [str, list[str], int],
    tuple[
        type[Benchmark[_BenchInput, _BenchOutput, _BenchEvalResult]],
        SupportsBenchmark[_BenchInput, _BenchOutput, _BenchEvalResult, _AgentRowResult]
    ]
]
PREPARES: dict[str, Prepare] = { # type: ignore
    "sr": prepare_sr,
    "mcsr": prepare_mcsr,
    "genrm": prepare_genrm
}


def generate_chunks(slice: tuple[int, int], n_process: int) -> list[tuple[int, int]]:
    r = (slice[1] - slice[0]) // n_process

    chunks = [
        (
            slice[0] + i*r,
            slice[0] + (i+1)*r
        )
        for i in range(n_process - 1)
    ]
    chunks.append((slice[0] + (n_process-1) * r, slice[1]))

    return chunks

def buffer_chunk_path(config: RunConfig, chunk: tuple[int, int]) -> Path:
    return BUFFER_PATH / Path(f"{config['method']}-{config['benchmark']}-{'-'.join(config['models'])}-{chunk}.json")

def calc_full_score(
    config: RunConfig,
    full: list[dict]
) -> Any:
    prepare = PREPARES[config["method"]]
    _, agent = prepare(config["benchmark"], config["models"], config["n_iter"])
    return agent.calc_full_score(full)

def save_path(
    config: RunConfig,
    result_dir_path: Path
) -> Path:
    file_name = f"{config['method']}-{config['benchmark']}-{'-'.join(config['models'])}.json"
    file_path = result_dir_path / Path(file_name)
    return file_path

def save_results(
    config: RunConfig,
    result_dir_path: Path,
    result: dict
) -> None:
    file_path = save_path(config, result_dir_path)
    with open(file_path, "w") as f:
        json.dump(result, f, indent=2)

def load_queue(path: str) -> list[RunConfig]:
    queue: list[RunConfig] = json.load(open(path, "r"))
    return queue