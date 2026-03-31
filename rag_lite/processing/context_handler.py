import json
from typing import List, Dict

class ContextHandler:
    """
    Handles the transformation of chat history between 
    structured JSON and natural language for embeddings.
    """

    @staticmethod
    def to_embedding_text(messages: List[Dict[str, str]]) -> str:
        """
        PARSER: Converts chat history to a clean string.
        Format: 'user: [content]\n\nassistant: [content]'
        This removes JSON overhead while preserving role context.
        """
        lines = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '').strip()
            # We use 'user' and 'assistant' exactly as they are
            lines.append(f"{role}: {content}")
        
        # Double newline helps the Recursive Chunker keep turns together
        return "\n\n".join(lines)

    @staticmethod
    def serialize_metadata(messages: List[Dict[str, str]]) -> str:
        """
        SERIALIZER: Converts the list to a JSON string for storage in 
        ChromaDB metadata, allowing Pepe to read the full structure later.
        """
        return json.dumps(messages)