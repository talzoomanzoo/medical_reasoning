from abc import abstractmethod
from typing import Callable, TypeVar, Mapping, Any, Generic

from langchain_openai import ChatOpenAI

from runbox.benchmarks import SupportsBenchmark
from runbox.utils import ChatOpenAIConfig, load_chat_prompt_template_json, invoke, ExtractorAdder


_BenchInput = TypeVar("_BenchInput", bound=Mapping[str, Any])
_BenchOutput = TypeVar("_BenchOutput")
_BenchEvalResult = TypeVar("_BenchEvalResult")
_CriticOutput = TypeVar("_CriticOutput")

type _SelfRefineRowResult = list[_BenchEvalResult]


class SelfRefineBase(
    SupportsBenchmark[_BenchInput, _BenchOutput, _BenchEvalResult, _SelfRefineRowResult],
    Generic[_BenchInput, _BenchOutput, _BenchEvalResult, _CriticOutput]
):
    def __init__(
        self,
        main_config: ChatOpenAIConfig,
        main_prompt_path: str,
        add_extractor: ExtractorAdder,
        n_iter: int,
        cheat: bool
    ) -> None:
        self.main = load_chat_prompt_template_json(main_prompt_path)\
            | ChatOpenAI(**main_config)
        self.parser = add_extractor(self.parse)
        self.n_iter = n_iter
        self.cheat = cheat

    def run_main(self, input: _BenchInput) -> tuple[str, _BenchOutput, float]:
        initial_response, initial_cost = invoke(self.main, input)
        initial_prediction = self.parser(initial_response)
        return initial_response, initial_prediction, initial_cost

    @abstractmethod
    def run_critic(
        self,
        input: _BenchInput,
        initial_response: str,
        label: _BenchOutput | None = None
    ) -> tuple[_CriticOutput, bool, float]:
        # critic_response, stop, critic_cost
        ...

    @abstractmethod
    def run_refiner(
        self,
        input: _BenchInput,
        initial_response: str,
        critic_response: _CriticOutput
    ) -> tuple[str, _BenchOutput, float]:
        # refiner_response, refiner_prediction, refiner_cost
        ...

    def run(
        self,
        input: _BenchInput,
        label: _BenchOutput | None = None
    ) -> dict:
        initial_response, initial_prediction, initial_cost = self.run_main(input)
        output: dict = {
            "initial_response": initial_response,
            "initial_prediction": initial_prediction,
            "initial_cost": initial_cost,
            "iteration": []
        }

        stop = False
        response = initial_response
        prediction = initial_prediction
        critic_response: _CriticOutput | None
        for _ in range(self.n_iter):
            if not stop:
                critic_response, stop, critic_cost\
                    = self.run_critic(input, response, label)
            else:
                critic_response = None
                critic_cost = 0

            if not stop and critic_response is not None:
                refiner_response, refiner_prediction, refiner_cost\
                    = self.run_refiner(input, response, critic_response)

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

    def evaluate(
        self,
        evaluator: Callable[[_BenchOutput, _BenchOutput | None], _BenchEvalResult],
        label: _BenchOutput,
        output: dict
    ) -> _SelfRefineRowResult:
        predictions = [output["initial_prediction"]]
        for o in output["iteration"]:
            predictions.append(o["refiner_prediction"])
        return [evaluator(label, prediction) for prediction in predictions]