from typing import Literal, Union

import torch
from infrastructure.base_model import BaseLLMModel
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    AutoTokenizer,
    pipeline
)

# TODO: secure auth_token value
# Ampiro's authentication HF token
AUTH_TOKEN = "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"


class Vicuna(BaseLLMModel):
    REPOSITORY = "lmsys"

    def __init__(
        self,
        device: Union[int, Literal["auto"]],
        name: str = 'vicuna',
        model_size: Literal["7b", "13b"] = "13b",
        max_new_tokens: int = 50,
    ) -> None:

        if isinstance(device, int):
            self.device_map = {"": device}
        elif isinstance(device, str):
            self.device_map = device
        else:
            raise TypeError(
                f"device must be of type int or str, got {type(device)}"
            )

        self.model_size = model_size
        self.model_path = f"{self.REPOSITORY}/vicuna-{self.model_size.lower()}-v1.5"
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path
        )

        self.pipe = pipeline(
            model=self.model, 
            tokenizer=self.tokenizer,
            task='text-generation',
            max_new_tokens=self.max_new_tokens
        )
        
    def generate(self, promt: str) -> str:
        output = self.pipe(promt)[0]['generated_text']
        return output

    def score(self):
        pass


if __name__ == "__main__":

    device = 1
    model_size = "7b"
    model = Vicuna(device=device, model_size=model_size)
    
    promt = """
        How would you hide a red apple on the table?
        1. move_to("red apple"), 2. pick_up("red apple", "unspecified"), 3. move_to("table"), 4. put("red apple", "table"), 5. done().
        How would you throw a red apple from the box on the chair?
        1. move_to("red apple"), 2. pick_up("red apple", "box"), 3. move_to("chair"), 4. put("red apple", "chair"), 5. done().
        How would you move a red apple from the table to the drawer?
         
    """
    output = model.generate(promt=promt)
    print(output)
    
