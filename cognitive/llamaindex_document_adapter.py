from llama_index.core.schema import Document as LlamaindexDocument
from llama_index.core import SimpleDirectoryReader


class LlamaindexDocumentAdapter:
    """
    Adapter for Llamaindex document

        document: the document to be adapted
    """

    def __init__(self, path: str):
        self.path = path

    def read(self):
        documents = SimpleDirectoryReader().load_file(self.path)

