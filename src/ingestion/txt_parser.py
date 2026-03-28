def load_txt(path: str) -> str:
    """
    Lee un archivo de texto plano de forma segura.
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read().strip()
    except Exception as e:
        return f"Error al leer el archivo TXT: {str(e)}"