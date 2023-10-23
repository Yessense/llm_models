from infrastructure.base_model import BaseLLMModel, BaseInput, BaseOutput, ScoringInput
from transformers import pipeline
from fastchat.model.model_adapter import load_model
from typing import Any


class Vicuna13B(BaseLLMModel):
    MODEL_NAME = "lmsys/vicuna-13b-v1.5"

    def __init__(
        self,
        num_devices: int = 2,
        name: str = "vicuna13b",
        max_new_tokens: int = 120,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.num_devices = num_devices
        super().__init__(name=name)
        self._load()

    def _load(self) -> None:
        self.model, self.tokenizer = load_model(
            model_path=self.MODEL_NAME,
            device="cuda",
            num_gpus=self.num_devices,
            load_8bit=False,
            revision="main",
            debug=False,
        )
        self._prepare_for_generation()

    def _prepare_for_generation(self) -> None:
        self.generation_pipeline = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer
        )

    def generate(self, text: str, **kwargs) -> BaseOutput:
        output = self.generation_pipeline(
            text,
            do_sample=False,
            return_full_text=False,
            max_new_tokens=self.max_new_tokens,
        )
        output = BaseOutput(output[0]["generated_text"])
        return output

    def score_text(self, inputs: ScoringInput, option_start: str = "\n", **kwargs) -> Any:
        raise NotImplementedError

    def score_option(self, query, option):
        raise NotImplementedError
    
    def score(self, option: str) -> float:
        """Score one option"""
        raise NotImplementedError
