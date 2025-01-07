import os
from typing import Callable, TypeVar, Mapping, Any
import json

from runbox.agents import *
from runbox.benchmarks import *
from runbox.benchmarks.benchmarks.medqa import MedQAEvalResult, MedQAInput, MedQAOutput
from runbox.utils import ChatOpenAIConfig, create_4o_mini_extractor
from runbox.agents.self_refine._agent import SelfRefineAgent


_BenchInput = TypeVar("_BenchInput", bound=Mapping[str, Any])
_BenchOutput = TypeVar("_BenchOutput")
_BenchEvalResult = TypeVar("_BenchEvalResult")


type SelfRefineAgentCreator[_BenchInput, _BenchOutput, _BenchEvalResult]\
    = Callable[
        [ChatOpenAIConfig, ChatOpenAIConfig, ChatOpenAIConfig, int, bool],
        SelfRefineAgent[_BenchInput, _BenchOutput, _BenchEvalResult]
    ]

def sr_prompt_paths(benchmark: str, cheat: bool) -> tuple[str, str, str, str]:
    return (
        f"runbox/prompts/{benchmark}/main.json",
        f"runbox/prompts/{benchmark}/self_refine/critic.json"\
            if not cheat\
            else f"runbox/prompts/{benchmark}/self_refine/critics_cheat",
        f"runbox/prompts/{benchmark}/self_refine/refiner.json",
        f"runbox/prompts/{benchmark}/extractor.json"
    )

def create_sr_agent(
    benchmark: str,
    AgentType: type[SelfRefineAgent[_BenchInput, _BenchOutput, _BenchEvalResult]],
) -> SelfRefineAgentCreator[_BenchInput, _BenchOutput, _BenchEvalResult]: # type: ignore

    def f(
        main_config: ChatOpenAIConfig,
        critic_config: ChatOpenAIConfig,
        refiner_config: ChatOpenAIConfig,
        n_iter: int,
        cheat: bool
    ) -> SelfRefineAgent[_BenchInput, _BenchOutput, _BenchEvalResult]:
        paths = sr_prompt_paths(benchmark, cheat)

        return AgentType( # type: ignore
            main_config=main_config,
            critic_config=critic_config,
            refiner_config=refiner_config,
            main_prompt_path=paths[0],
            critic_prompt_path=paths[1],
            refiner_prompt_path=paths[2],
            add_extractor=create_4o_mini_extractor(paths[3]),
            n_iter=n_iter
        )

    return f

benchmark_configs: dict[str, tuple[type[Benchmark], SelfRefineAgentCreator]] = {
    "medqa": (MedQA, create_sr_agent("medqa", MedQASelfRefineAgent)),
    "medmcqa": (MedMCQA, create_sr_agent("medmcqa", MedMCQASelfRefineAgent)),
    "pubmedqa": (PubMedQA, create_sr_agent("pubmedqa", PubMedQASelfRefineAgent)),
    "ddxplus": (DDXPlus, create_sr_agent("ddxplus", DDXPlusSelfRefineAgent))
}


MODEL_CONFIGS_PATH = "models.json"
try:
    model_configs: dict = json.load(open(MODEL_CONFIGS_PATH, "r"))
except:
    raise Exception("model config file required")
def prepare(
    benchmark: str,
    models: list[str],
    n_iter: int,
    cheat: bool
) -> tuple[type[Benchmark], SelfRefineAgent]:
    benchmark_, create_agent_ = benchmark_configs[benchmark]

    return (
        benchmark_,
        create_agent_(
            model_configs[models[0]],
            model_configs[models[1]],
            model_configs[models[2]],
            n_iter,
            cheat
        )
    )