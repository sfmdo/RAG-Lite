import docx
from odf.opendocument import load
from odf import teletype, text


def load_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_odt(path):
    textdoc = load(path)
    paragraphs = textdoc.getElementsByType(text.P)
    return "\n".join([teletype.extractText(p) for p in paragraphs])