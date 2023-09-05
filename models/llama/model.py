from infrastructure.base_model import BaseLLMModel
import torch
import pprint

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Union
from transformers import AutoModelForCausalLM, LlamaTokenizer, GenerationConfig


class LLaMA(BaseLLMModel):
    REPOSITORY = "decapoda-research"

    def __init__(self,
                 device: Union[int, Literal["auto"]],
                 name: str = 'llama',
                 model_size: Literal["7B", "13B"] = "7B",
                 max_new_tokens: int = 50,
                 ) -> None:

        if isinstance(device, int):
            self.device_map = {"": device}
        elif isinstance(device, str):
            self.device_map = device
        else:
            raise TypeError("device must be of type int or str, got %s" % type(device))
        self.model_size = model_size
        self.model_path = f"{self.REPOSITORY}/llama-{self.model_size.lower()}-hf"
        self.max_new_tokens = max_new_tokens

        name = f"{name}-{model_size}"
        super().__init__(name=name)
        self._load()

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
        self.generation_config.max_new_tokens = self.max_new_tokens


    def generate(self, prompt: str) -> str:
        data = self.tokenizer(prompt, return_tensors="pt")
        data = {k: v.to(self.model.device) for k, v in data.items()}
        # generate text
        with torch.no_grad():
            output_ids = self.model.generate(
                **data,
                generation_config=self.generation_config
            )[0]
        output_ids = output_ids[len(data["input_ids"][0]):]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output
    
    def score(self):
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str,choices = ["7B", "13B"], default="7B")
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    
    model = LLaMA(device=args.device, model_size=args.model_size)