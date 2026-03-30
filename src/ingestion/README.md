# Module 1: Ingestion & Document Loading

## Overview
The **Ingestion** module serves as the gateway for the RAG-Lite system. Its primary responsibility is to accept various file formats, identify their type, and route them to the appropriate parser to extract raw text. By centralizing the loading logic, it ensures that the rest of the pipeline receives a clean, standardized string regardless of whether the source was a PDF, a Word document, or a simple text file.

---

## 1. Supported Formats & Parsers
The module utilizes a registry pattern (the `loaders` dictionary) to map file extensions to specialized parsing functions:

| Extension | Parser Function | Source File |
| :--- | :--- | :--- |
| `.pdf` | `load_pdf` | `src/ingestion/pdf_parser.py` |
| `.md` | `load_md` | `src/ingestion/markdown_parser.py` |
| `.txt` | `load_txt` | `src/ingestion/txt_parser.py` |
| `.docx` | `load_docx` | `src/ingestion/docs_loader.py` |
| `.odt` | `load_odt` | `src/ingestion/docs_loader.py` |

---

## 2. Core Logic

### `extractExtension(path: str) -> str`
**Purpose:** A utility function to isolate the file extension from a given file path.
*   **Logic:** It iterates backward from the end of the string until it finds a dot (`.`).
*   **Returns:** The characters following the dot (e.g., `"pdf"`, `"docx"`). If no dot is found or the path is invalid, it returns an empty string.

### `serveDocument(path: str) -> str`
**Purpose:** The main entry point for the Ingestion module.
*   **Action:** 
    1.  Calls `extractExtension` to determine the file type.
    2.  Validates if the extension exists in the supported `loaders`.
    3.  **Error Handling:** If the extension is unsupported or empty, it raises an `Exception("Extension not supported")`.
    4.  **Execution:** Triggers the mapped parser function and returns the extracted text as a single string.

---

## 3. Workflow Example

1.  **Input:** A user uploads `report_2024.pdf`.
2.  **Extraction:** `extractExtension` identifies the type as `pdf`.
3.  **Routing:** `serveDocument` finds `load_pdf` in the `loaders` dictionary.
4.  **Parsing:** `load_pdf("./path/to/report_2024.pdf")` is executed.
5.  **Output:** A raw string containing all text from the PDF is passed to the **Processing Module**.

---

## 4. Internal Dependencies
This module acts as a router for the following internal sub-modules:
*   **`pdf_parser.py`**: Handles complex PDF layouts and text extraction.
*   **`markdown_parser.py`**: Cleans Markdown syntax while preserving content.
*   **`txt_parser.py`**: Simple stream reading for plain text files.
*   **`docs_loader.py`**: Utilizes specialized libraries to handle XML-based document formats (DOCX/ODT).