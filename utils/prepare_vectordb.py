import os

from gym.envs.registration import namespace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from PyPDF2 import PdfFileReader
from langchain_pinecone import PineconeVectorStore
from io import BytesIO
import requests
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

class Document:
    """
    Document class to hold the content and metadata of each document.
    """
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata

class PrepareVectorDB:
    """
    PrepareVectorDB class to prepare a vector database from a given document.
    """
    def __init__(self, link: str, chunk_size: int, chunk_overlap: int, clerkId) -> None:
        """
        Initialize the PrepareVectorDB class with the document link, chunk size, and chunk overlap.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.link = link
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        self.clerkId = clerkId

    def load_documents(self) -> List[Document]:
        """
        Load the documents from the given link and return a list of Document objects.
        """
        print("Loading the uploaded documents...")
        docs = []
        response = requests.get(self.link)
        pdf = PdfFileReader(BytesIO(response.content))
        for page in range(pdf.getNumPages()):
            page_content = pdf.getPage(page).extractText()
            metadata = {"page_number": page}
            docs.append(Document(page_content, metadata))
        return docs

    def chunk_documents(self, docs: List[Document]):
        """
        Chunk the documents and return a list of chunked documents.
        """
        print("Chunking documents...")
        return self.text_splitter.split_documents(docs)

    def prepare_vectordb(self, chunked_documents):
        """
        Prepare the vector database from the chunked documents and return the vector store.
        """
        print("Preparing vectordb...")
        return PineconeVectorStore.from_documents(
            documents=chunked_documents,
            index_name="raggpt",
            embedding=self.embedding,
            namespace=self.clerkId
        )

    def query_vectordb(self, vectorstore, query: str):
        """
        Query the vector database with the given query and print the results.
        """
        print("Querying the vectordb with the query:", query)
        results = vectorstore.similarity_search(query)
        for i, document in enumerate(results, start=1):
            print(f"{i}: Answer: {document.page_content}\n")

    def run(self):
        """
        Run the entire process of preparing the vector database and querying it.
        """
        docs = self.load_documents()
        print("Documents loaded successfully!")

        chunked_documents = self.chunk_documents(docs)
        print("Documents chunked successfully!")

        vectorstore = self.prepare_vectordb(chunked_documents)
        print("Vector database prepared successfully!")

        # self.query_vectordb(vectorstore, "What are transformers?")
