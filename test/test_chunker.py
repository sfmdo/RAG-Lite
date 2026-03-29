import os
from dotenv import load_dotenv
load_dotenv()
import sys
import logging
from pathlib import Path


root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from processing.chunking.chunker_controller import ChunkerController

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)

def run_unified_test():
    print("\n" + "="*75)
    print(" RAG-LITE UNIFIED CHUNKING TEST (CONTEXT, TXT, MARKDOWN) ")
    print("="*75)

    # Initialize Controller
    try:
        tokenizer = os.getenv("TOKENIZER_NAME")
        if tokenizer is None:
            raise ValueError("ERROR: La variable de entorno 'MODEL_NAME' no está definida. "
                    "Asegúrate de configurar tu archivo .env con la ruta al modelo.") 
        controller = ChunkerController(tokenizer_name=tokenizer)
        print("Chunker Controller initialized successfully.")
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Define Test Cases
    test_cases = [
        {
            "name": "CONVERSATIONAL CONTEXT (JSON List -> .context)",
            "ext": "context",
            "data": [
                {'role': 'user', 'content': 'Hola Pepe, ¿qué puedes hacer?'},
                {'role': 'assistant', 'content': 'Puedo ayudarte con:\n1. Responder preguntas.\n2. Analizar datos.\n3. Charlar un rato.'},
                {'role': 'user', 'content': 'Ando chill de cojones'}
            ]
        },
        {
            "name": "MARKDOWN DOCUMENT (Headers & Structure -> .md)",
            "ext": "md",
            "data": """
# RAG-Lite Project
## Overview
This system is designed for local efficiency using e5-small-v2.
It supports multiple file formats like PDF, DOCX, and MD.

## Technical Details
### Chunking Strategy
The strategy for Markdown prioritizes headers to keep sections together.
*   **Size:** 200 tokens.
*   **Overlap:** 50 tokens.

### Optimization
By using local tokenizers, we ensure 100% offline privacy on Ryzen and i5 CPUs.
            """ * 3 # Repeat to force multiple chunks
        },
        {
            "name": "PLAIN TEXT (Standard Prose -> .txt)",
            "ext": "txt",
            "data": """
            The e5-small-v2 model is a highly efficient text encoder. It is specifically designed for information retrieval tasks and semantic search. 
            Retrieval-Augmented Generation (RAG) systems benefit significantly from having coherent text fragments. 
            If we cut a sentence in the middle, we lose semantic meaning.
            """ * 6
        }
    ]

    # Export Path
    export_file = root_path / "debug_chunks_unified.txt"
    
    with open(export_file, "w", encoding="utf-8") as f:
        f.write("=== RAG-LITE UNIFIED DEBUG EXPORT ===\n\n")

        for case in test_cases:
            print(f"\n>>> Processing: {case['name']}")
            
            # The controller automatically chooses the strategy based on the extension
            chunks = controller.process(case['data'], extension=case['ext'])
            
            f.write(f"--- TEST CASE: {case['name']} ---\n")
            f.write(f"Total Chunks: {len(chunks)}\n\n")

            for i, chunk in enumerate(chunks):
                tokens = controller.chunker._length_function(chunk)
                
                # Console Output (Minimal Preview)
                header = f" CHUNK #{i+1} ({tokens} tokens) "
                print(f"\033[93m{header:-^60}\033[0m")
                # Show first 100 chars and a bit of the end to see the cut
                preview = f"{chunk.strip()[:100]} [...] {chunk.strip()[-40:]}"
                print(preview.replace('\n', ' ')) 
                
                # File Output (Full Content for auditing)
                f.write(f"CHUNK #{i+1} [{tokens} tokens]:\n{chunk.strip()}\n")
                f.write("-" * 40 + "\n\n")
            
            f.write("\n" + "="*60 + "\n\n")

    print("\n" + "="*75)
    print(f"UNIFIED TEST COMPLETE")
    print(f"Detailed results saved to: {export_file}")
    print("="*75)

if __name__ == "__main__":
    run_unified_test()