from src.ingestion.pdf_parser import load_pdf
from src.ingestion.markdown_parser import load_md
from src.ingestion.txt_parser import load_txt
from src.ingestion.docs_loader import load_docx, load_odt

loaders = {
    "md": load_md,
    "pdf": load_pdf,
    "txt": load_txt,
    "docx": load_docx,
    "odt": load_odt,
}

def extractExtension(path: str) -> str:
    extension=""
    for i in range(len(path)-1, -1, -1):
        if path[i] == '.':
            extension = path[i+1:]
            return extension
    if extension == "" or extension == None:
        return ""
    return extension

def serveDocument(path) -> str:
    extension = extractExtension(path)
    if extension not in loaders or extension == "":
        raise Exception("Extension not supported")
    text = loaders[extension](path)
    return text