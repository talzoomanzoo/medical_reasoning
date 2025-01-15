import argparse
import asyncio
import json
import os
import random
import yaml
import copy
import re
from transformers.trainer_utils import set_seed
import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAIChat, OpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import os
import tiktoken

class DefaultModel:
    """
    Default model class for vllm inference
    """
    def __init__(self, model_name):
        self.model_name = model_name
        
class LocalModel(DefaultModel):
    """
    Local model class for vllm inference
    """
    def __init__(self, model_name, openai_api_base, openai_api_key="EMPTY", max_tokens=1024, top_p=0.95, temperature=0.1, frequency_penalty=0.0, stop=None, n=None):
        super().__init__(model_name)
        self.openai_api_base = openai_api_base
        self.openai_api_key = openai_api_key
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.stop = stop
        self.n = n
        self.model = OpenAI(model_name=model_name, openai_api_base=openai_api_base, openai_api_key=openai_api_key, max_tokens=max_tokens, top_p=top_p, temperature=temperature, frequency_penalty=frequency_penalty, stop=stop, n=n)
        
    async def multiple_generate(self, model_input):
        while True:
            try:
                response = await self.model.agenerate(prompts=[model_input])  # if you need it
                break
            except Exception as e:
                print(f"Exception occurred: {e}")
                return None
        response_text = []
        for i in range(len(response.generations[0])):
            response_text.append(response.generations[0][i].text)
        result = {
            "prediction": response_text,
            "model_input": model_input
        }
        return result

    async def single_generate(self, model_input):
        while True:
            try:
                response = await self.model.ainvoke(model_input)
                break
            except Exception as e:
                print(f"Exception occurred: {e}")
                return None
        result = {
            "prediction": [response],
            "model_input": model_input
        }
        return result

class OpenAIModel(DefaultModel):
    def __init__(self, model_name, max_tokens=1024, top_p=0.95, temperature=0.1, frequency_penalty=0.0, stop=None, n=None):
        super().__init__(model_name)
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.stop = stop
        self.n = n
        self.total_cost = 0
        self.model = ChatOpenAI(model_name=model_name, max_tokens=max_tokens, top_p=top_p, temperature=temperature, frequency_penalty=frequency_penalty, stop=stop, n=n)
        
    async def multiple_generate(self, model_input):
        with get_openai_callback() as cb:
            human_message = HumanMessage(content=model_input)
            while True:
                try:
                    response = await self.model.agenerate([[human_message]])
                    break
                except Exception as e:
                    print(f"Exception occurred: {e}")
                    return None
            response_text = []
            for i in range(len(response.generations[0])):
                response_text.append(response.generations[0][i].text)
            result = {
                "prediction": response_text,
                "model_input": model_input
            }
            self.total_cost += cb.total_cost
            return result

    async def single_generate(self, model_input):
        with get_openai_callback() as cb:
            human_message = HumanMessage(content=model_input)
            while True:
                try:
                    response = await self.model.ainvoke([human_message])
                    print(response)
                    break
                except Exception as e:
                    print(f"Exception occurred: {e}")
                    return None
            result = {
                "prediction": [response.content],
                "model_input": model_input
            }
            self.total_cost += cb.total_cost
            return result

    def get_token_used(self):
        return self.total_cost
