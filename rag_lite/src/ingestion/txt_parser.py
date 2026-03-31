def load_txt(path: str) -> str:
    """
    read and extract the content for a txt
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read().strip()
    except Exception as e:
        return f"Error processing TXT: {str(e)}"