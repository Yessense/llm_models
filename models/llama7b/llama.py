from infrastructure.base_model import BaseLLMModel
import torch
import pprint

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Union
from transformers import AutoModelForCausalLM, LlamaTokenizer
from transformers import pipeline


class Llama(BaseLLMModel):
    REPOSITORY = "decapoda-research/"

    def __init__(self,
                 device: Union[int, Literal["auto"]],
                 name: str = 'llama',
                 model_size: Literal["7B", "13B"] = "7B",
                 ) -> None:

        if isinstance(device, int):
            self.device_map = {"": device}
        elif isinstance(device, str):
            self.device_map = device
        else:
            raise TypeError("device must be of type int or str, got %s" % type(device))
        self.model_size = model_size
        self.model_path = f"{self.REPOSITRY}/llama-{self.model_size.lower()}-hf"

        name = f"{name}-{model_size}"
        super().__init__(name=self.name)

    def _load(self) -> None:
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map=self.device_map
        )
        self.model.eval()
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        self.generation_config = GenerationConfig.from_pretrained(self.model_path)


    def generate(self, prompt: str) -> str:
        # generate text
        output = self.generation_config(str,
                                          do_sample=False,
                                          return_full_text=False,
                                          max_new_tokens=self.max_new_tokens)
        output = output[0]['generated_text']
        return output