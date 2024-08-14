from ragas.testset.docstore import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import BaseDocumentTransformer, Document
from typing import List
import fitz


class LangchainDocumentAdapter:
    """
    Adapter for Langchain document

        path: the path to the document
    """
    num_questions = 0

    def __init__(self, path: str):
        self.path = path
        self.num_questions = 0

    def __pdf_image_detection(self) -> int:
        """
        Check if in the document there are images

        Returns:
            number of document if there are images, 0 otherwise
        """
        doc = fitz.open(self.path)
        has_image = False
        text_content = ""

        for page in doc:
            # Check for images
            if page.get_images(full=True):
                has_image = True

            # Extract text
            text_content += page.get_text()

        # Count the number of words in the text
        word_count = len(text_content.split())

        if has_image and word_count == 0:
            # If the PDF has images and no text, do nothing
            return 0
        elif word_count > 0:
            # If the PDF has text (with or without images), return the number of words
            return word_count
        else:
            # If the PDF has neither text nor images, return nothing
            return 0

    def __get_filename(self) -> str:
        """
        Get the filename of the document

        Returns:
            the filename of the document
        """
        return self.path.split("/")[-1]

    def read(self) -> List[Document]:
        """
        Read the document

        Returns:
            the content of the document
        """
        # Check if the document has images
        word_count = self.__pdf_image_detection()
        documents = []
        if word_count > 5:
            loader = PyPDFLoader(self.path)
            page = loader.load()

            # Split the text into sentences
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
            )

            documents = splitter.split_documents(page)

            # for the chunked documents, add the filename as metadata
            for doc in documents:
                doc.metadata['filename'] = self.__get_filename()

        return documents

    def read_and_summary(self) -> List[Document]:
        """
        Read the document

        Returns:
            the content of the document
        """
        # Check if the document has images
        word_count = self.__pdf_image_detection()

        documents = []

        if word_count != 0:
            loader = PyPDFLoader(self.path)
            page = loader.load()

            # Split the text into sentences
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
            )

            documents = splitter.split_documents(page)

            # for the chunked documents, add the filename as metadata
            for doc in documents:
                doc.metadata['filename'] = self.__get_filename()
                # TODO add summarization
                doc.page_content = """Summary of the document api call"""

        return documents

    def generate_questions(self) -> int:
        """
        Generate questions based on the content of the document

        Returns:
            the number of questions generated
        """
        word_count = self.__pdf_image_detection()
        ratio = 20
        self.num_questions = word_count / ratio
        self.num_questions = round(self.num_questions)

        if self.num_questions > 20:
            self.num_questions = 20

        return self.num_questions