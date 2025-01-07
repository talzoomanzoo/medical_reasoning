from typing import TypedDict, Any, Callable, Literal, TypeVar, Mapping, NotRequired
from enum import Enum
from pathlib import Path
import json

from runbox.benchmarks import Benchmark
from runbox.agents.self_refine_base import SelfRefineBase
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
    cheat: NotRequired[bool]


_BenchInput = TypeVar("_BenchInput", bound=Mapping[str, Any])
_BenchOutput = TypeVar("_BenchOutput", contravariant=True)
_BenchEvalResult = TypeVar("_BenchEvalResult")
_CriticOutput = TypeVar("_CriticOutput")
type _Benchmark = Benchmark[_BenchInput, _BenchOutput, _BenchEvalResult]
type _SelfRefineBase = SelfRefineBase[_BenchInput, _BenchOutput, _BenchEvalResult, _CriticOutput]
type _Settings = tuple[type[_Benchmark], _SelfRefineBase]
type Prepare = Callable[[str, list[str], int, bool], _Settings]
PREPARES: dict[str, Prepare] = {
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

def path_str(config: RunConfig) -> str:
    cheat = config['cheat'] if 'cheat' in config else False
    return "_".join([
        config['method'],
        config['benchmark'],
        f"{config['n_iter']}i",
        *(("cheat",) if cheat else ()),
        *config['models']
    ])

def buffer_chunk_path(config: RunConfig, chunk: tuple[int, int]) -> Path:
    return BUFFER_PATH / Path(f"{path_str(config)}_{chunk}.json")

def calc_full_score(
    config: RunConfig,
    full: list[dict]
) -> Any:
    prepare = PREPARES[config["method"]]
    _, agent = prepare(config["benchmark"], config["models"], 0, False)
    return agent.calc_full_score(full)

def save_path(
    config: RunConfig,
    result_dir_path: Path
) -> Path:
    file_name = f"{path_str(config)}.json"
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

def load(
    config: RunConfig,
    chunk: tuple[int, int] | None = None
) -> tuple[_Benchmark, _SelfRefineBase]:
    prepare = PREPARES[config["method"]]
    cheat = config["cheat"] if "cheat" in config else False
    _Benchmark_, agent = prepare(config["benchmark"], config["models"], config["n_iter"], cheat)
    dataset = _Benchmark_( # type: ignore
        split=config["bench_config"]["split"],
        slice=(chunk if chunk is not None else config["bench_config"]["slice"])
    )
    return dataset, agent