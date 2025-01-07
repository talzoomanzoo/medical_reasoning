from typing import Callable, TypeVar, Mapping, Any, TypedDict
import os
from pathlib import Path
from statistics import mean
from functools import reduce
import asyncio

from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

from ..self_refine_base import SelfRefineBase
from runbox.utils import (
    ChatOpenAIConfig,
    load_chat_prompt_template_json,
    invoke,
    ExtractorAdder,
    ainvoke
)


_BenchInput = TypeVar("_BenchInput", bound=Mapping[str, Any])
_BenchOutput = TypeVar("_BenchOutput")
_BenchEvalResult = TypeVar("_BenchEvalResult")

class MultiCriticSelfRefineCriticOutput(TypedDict):
    responses: list[str]
    scores: list[int]

class MultiCriticSelfRefineAgent[_BenchInput, _BenchOutput, _BenchEvalResult](
    SelfRefineBase[_BenchInput, _BenchOutput, _BenchEvalResult, MultiCriticSelfRefineCriticOutput]
):
    def __init__(
        self,
        main_config: ChatOpenAIConfig,
        critic_config: ChatOpenAIConfig,
        refiner_config: ChatOpenAIConfig,
        main_prompt_path: str,
        critic_prompts_dir_path: str,
        refiner_prompt_path: str,
        add_extractor: ExtractorAdder,
        n_iter: int = 3,
        cheat: bool = False
    ) -> None:
        super().__init__(
            main_config=main_config,
            main_prompt_path=main_prompt_path,
            add_extractor=add_extractor,
            n_iter=n_iter,
            cheat=cheat
        )

        self.critics = [
            load_chat_prompt_template_json(critic_prompts_dir_path / Path(prompt_path))\
                | ChatOpenAI(**critic_config)
            for prompt_path in os.listdir(critic_prompts_dir_path)
        ]
        self.refiner = load_chat_prompt_template_json(refiner_prompt_path)\
            | ChatOpenAI(**refiner_config)

    def run_critic(
        self,
        input: _BenchInput,
        initial_response: str,
        label: _BenchOutput | None = None
    ) -> tuple[MultiCriticSelfRefineCriticOutput, bool, float]:
        responses = []
        scores = []
        total_cost = 0

        if self.cheat:
            assert label is not None

        async def _arun_critics() -> list[tuple[str, float]]:
            return list(await asyncio.gather(*(
                asyncio.create_task(
                    ainvoke(
                        critic,
                        {
                            **input,
                            "initial_response": initial_response,
                            **({ "label": label } if self.cheat else {})
                        } # type: ignore
                    )
                )
                for critic in self.critics
            )))
        results = asyncio.run(_arun_critics())

        for critic_response, critic_cost in results:
            feedback = "\n".join(filter(
                lambda x: '- GOOD:' in x or '- BAD:' in x,
                critic_response.splitlines()
            ))
            responses.append(feedback)

            s = self.parser(critic_response)
            score = int(s) if s is not None else 0
            scores.append(score)

            total_cost += critic_cost

        agg_response = "\n\n".join(responses)
        stop = reduce(lambda a, b: a and b, map(lambda x: x >= 4, scores))

        return (
            {
                "responses": responses,
                "scores": scores
            },
            stop,
            total_cost
        )

    def run_refiner(
        self,
        input: _BenchInput,
        initial_response: str,
        critic_response: MultiCriticSelfRefineCriticOutput
    ) -> tuple[str, _BenchOutput, float]:
        refiner_response, refiner_cost = invoke(
            self.refiner,
            {
                **input,
                "initial_response": initial_response,
                "critic_response": "\n\n".join(critic_response["responses"])
            } # type: ignore
        )
        refiner_prediction = self.parser(refiner_response)
        return refiner_response, refiner_prediction, refiner_cost