from abc import ABC, abstractmethod



class BaseLLMModel(ABC):
    """ Base class for LLM models"""
    _name: str

    @property
    def name(self):
        return self._name

    def __init__(self,
                 name: str = "Base Model"):
        self._name = name
        
    @abstractmethod
    def _load(self) -> None:
        """Load model, tokenizer, etc"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text"""
    
    @abstractmethod
    def score(self, option: str) -> float:
        """Score one option"""