from infrastructure.base_model import BaseLLMModel
from ctransformers import AutoModelForCausalLM
from typing import Any



class Vicuna33B(BaseLLMModel):
    MODEL_NAME = "lmsys/vicuna-33b-v1.3"

    def __init__(
        self,
        name: str = "vicuna33b",
    ) -> None:
        super().__init__(name=name)
        self._load()
        
    def _load(self) -> None: 
        self.llm = AutoModelForCausalLM.from_pretrained(
            "TheBloke/vicuna-33B-GGUF",
            model_file="vicuna-33b.Q5_K_M.gguf",
            model_type="llama",
            context_length=1280,
            gpu_layers=128
        )

    def generate(self, text: str, **kwargs) -> str:
        output = self.llm(text)
        return output

    def score(self, option: str) -> float:
        """Score one option"""
        raise NotImplementedError