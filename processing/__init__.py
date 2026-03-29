import re
import unicodedata

def normalize_text(text: str) -> str:
    """
    Standardizes raw text for embedding models.
    Removes noise while preserving semantic structure.
    """
    if not text:
        return ""

    # Unicode Normalization (NFC)
    text = unicodedata.normalize('NFC', text)

    # Clean invisible characters and non-breaking spaces
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    text = text.replace('\xa0', ' ')

    # Standardize Line Breaks
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Horizontal Whitespace
    text = re.sub(r'[ \t]+', ' ', text)

    # Trim
    return text.strip()

