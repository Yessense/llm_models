from abc import ABC


class BaseLLMModel(ABC):
    """ Base class for LLM models"""
    _name: str

    @property
    def name(self):
        return self._name

    def __init__(self,
                 name: str = "Base Model"):
        self._name = name
        
    def _load(self) -> None:
        """Load model, tokenizer, etc"""