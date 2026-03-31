from enum import Enum
from typing import List, Dict

class ChunkingStrategy(Enum):
    DOCUMENT = "document" 
    CODE = "code"          
    MARKDOWN = "markdown"


class SeparatorProvider:
    _STRATEGY_MAP: Dict[str, List[str]] = {
        "document": ["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        
        "markdown": ["\n# ", "\n## ", "\n### ", "\n---", "\n\n", "\n", ". ", " ", ""],
        
        "code": ["\nclass ", "\ndef ", "\nfunction ", "\n\n", "\n", " {", "(", "[", "; ", " ", ""],
        
    }

    # Mapeo de extensiones a estrategias
    _EXTENSION_TO_STRATEGY: Dict[str, str] = {
        "md": "markdown",
        "py": "code", # ! Future implementation
        "js": "code",
        "cpp": "code",
        "pdf": "document",  
        "docx": "document", 
        "odt": "document",
        "txt": "document"
    }

    @classmethod
    def get_separators(cls, extension: str) -> List[str]:
        """
        Busca los separadores ideales para una extensión.
        Si no hay específicos, usa el fallback 'document'.
        """
        ext = extension.lower()
        strategy_name = cls._EXTENSION_TO_STRATEGY.get(ext, "document")
        return cls._STRATEGY_MAP.get(strategy_name, cls._STRATEGY_MAP["document"])