import os
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.ingestion.document_loader import serveDocument

# Define paths relative to this script
BASE_DIR = Path(__file__).resolve().parent
INPUT_FOLDER = BASE_DIR / "inputs"

def run_loader_tests(folder_path: Path):
    """
    Iterates through all files in the input folder and 
    attempts to extract text using the unified serveDocument function.
    """
    if not folder_path.exists():
        print(f"Error: Input folder not found at {folder_path}")
        return

    print(f"Starting tests on folder: {folder_path}\n")

    for filename in os.listdir(folder_path):
        file_path = folder_path / filename
        
        if file_path.is_dir():
            continue

        print(f"--- Testing File: {filename} ---")
        try:
            content = serveDocument(str(file_path))
            
            if content:
                print(content)
                print("\n" + "="*50 + "\n")
            else:
                print("Warning: Document returned empty string.\n")

        except Exception as e:
            print(f"FAILED {filename}: {str(e)}\n")

if __name__ == "__main__":
    run_loader_tests(INPUT_FOLDER)