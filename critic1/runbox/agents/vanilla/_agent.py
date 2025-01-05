from typing import Callable, TypeVar, Mapping, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain import hub

from runbox.benchmarks import SupportsBenchmark
from runbox.utils import ChatOpenAIConfig, load_chat_prompt_template_json, invoke, ExtractorAdder


_BenchInput = TypeVar("_BenchInput", bound=Mapping[str, Any])
_BenchOutput = TypeVar("_BenchOutput")
_BenchEvalResult = TypeVar("_BenchEvalResult")

type VanillaRowResult = _BenchEvalResult

# Pull the prompt from the Langchain hub
prompt_cds = hub.pull("public-med-critic-prompt/ddxplus-initial") #five shots
prompt_kqa = hub.pull("public-med-critic-prompt/medqa_initial-response") #five shots

class VanillaAgent[_BenchInput, _BenchOutput, _BenchEvalResult](
    SupportsBenchmark[_BenchInput, _BenchOutput, _BenchEvalResult, VanillaRowResult]
):
    def __init__(
        self,
        client_config: ChatOpenAIConfig,
        #prompt_path: str,
        add_extractor: ExtractorAdder,
        benchmark: str
    ) -> None:
        # Select the appropriate prompt based on the benchmark
        if benchmark == "ddxplus":
            prompt = prompt_cds
        elif benchmark == "medqa":
            prompt = prompt_kqa
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")

        self.client = prompt | ChatOpenAI(**client_config)
        self.parser = add_extractor(self.parse)

         #load_chat_prompt_template_json(prompt_path)
    def run(self, input: _BenchInput) -> dict:
        content, cost = invoke(self.client, input)
        return { "prediction": self.parser(content), "output": content, "cost": cost }

    def evaluate(
        self,
        evaluator: Callable[[_BenchOutput, _BenchOutput | None], _BenchEvalResult],
        label: _BenchOutput,
        output: dict
    ) -> _BenchEvalResult:
        return evaluator(label, output["prediction"])