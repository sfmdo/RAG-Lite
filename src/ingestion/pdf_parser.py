import re
from PyPDF2 import PdfReader

def load_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                page_text = page_text.replace('\xa0', ' ')
                text += page_text + "\n"
        
        text = re.sub(r'\n\s*\n', '\n', text)
        text = text.replace('\n', ' ') 

        return text.strip()
    except Exception as e:
        print(f"Error procesando PDF: {e}")
        return ""