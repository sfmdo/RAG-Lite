import frontmatter

def load_md(path: str) -> str:
    """
    Carga un archivo MD manteniendo su estructura.
    Si tiene metadatos (YAML), los convierte en un encabezado legible.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)
        
        header = ""
        if post.metadata:
            header = "METADATA\n"
            for key, value in post.metadata.items():
                header += f"- {key}: {value}\n"
            header += "---\n\n"
        
        return header + post.content
        
    except Exception as e:
        return f"Error al cargar el archivo MD: {e}"

