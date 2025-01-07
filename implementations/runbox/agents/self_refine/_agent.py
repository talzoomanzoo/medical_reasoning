from typing import Callable, TypeVar, Mapping, Any, TypedDict

from langchain_openai import ChatOpenAI

from ..self_refine_base import SelfRefineBase
from runbox.utils import ChatOpenAIConfig, load_chat_prompt_template_json, invoke, ExtractorAdder


_BenchInput = TypeVar("_BenchInput", bound=Mapping[str, Any])
_BenchOutput = TypeVar("_BenchOutput")
_BenchEvalResult = TypeVar("_BenchEvalResult")

class SelfRefineCriticOutput(TypedDict):
    response: str

def _stop(response: str) -> bool:
    response_ = response.lower()
    return "`stop`" in response_\
        or "'stop'" in response_\
        or '"stop"' in response_

class SelfRefineAgent[_BenchInput, _BenchOutput, _BenchEvalResult](
    SelfRefineBase[_BenchInput, _BenchOutput, _BenchEvalResult, SelfRefineCriticOutput]
):
    def __init__(
        self,
        main_config: ChatOpenAIConfig,
        critic_config: ChatOpenAIConfig,
        refiner_config: ChatOpenAIConfig,
        main_prompt_path: str,
        critic_prompt_path: str,
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

        self.critic = load_chat_prompt_template_json(critic_prompt_path)\
            | ChatOpenAI(**critic_config)
        self.refiner = load_chat_prompt_template_json(refiner_prompt_path)\
            | ChatOpenAI(**refiner_config)

    def run_critic(
        self,
        input: _BenchInput,
        initial_response: str,
        label: _BenchOutput | None = None
    ) -> tuple[SelfRefineCriticOutput, bool, float]:
        if self.cheat:
            assert label is not None

        critic_response, critic_cost = invoke(
            self.critic,
            {
                **input,
                "initial_response": initial_response,
                **({ "label": label } if self.cheat else {})
            } # type: ignore
        )
        stop = _stop(critic_response)
        return { "response": critic_response }, stop, critic_cost

    def run_refiner(
        self,
        input: _BenchInput,
        initial_response: str,
        critic_response: SelfRefineCriticOutput
    ) -> tuple[str, _BenchOutput, float]:
        refiner_response, refiner_cost = invoke(
            self.refiner,
            {
                **input,
                "initial_response": initial_response,
                "critic_response": critic_response["response"]
            } # type: ignore
        )
        refiner_prediction = self.parser(refiner_response)
        return refiner_response, refiner_prediction, refiner_cost