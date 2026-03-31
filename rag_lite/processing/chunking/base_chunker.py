from abc import ABC, abstractmethod
from typing import List

class BaseChunker(ABC):
    @abstractmethod
    def split_text(self, text: str, separators: List[str]) -> List[str]:
        pass