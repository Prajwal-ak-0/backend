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
    def __init__(self, link: str, clerkId) -> None:
        """
        Initialize the PrepareVectorDB class with the document link, chunk size, and chunk overlap.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
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

    def prepare_vectordb_to_store(self, chunked_documents):
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

    def format_results(self, results):
        """
        Format the results from the similarity search and return them as a string.
        """
        formatted_results = []
        for i, document in enumerate(results, start=1):
            formatted_results.append(f"Result {i}:")
            formatted_results.append(f"Page Number: {document.metadata['page_number']}")
            formatted_results.append(f"Page Content: {document.page_content}\n")
        return "\n".join(formatted_results)

    def create_pinecone_instance_and_query(self, query: str):
        """
        Create a Pinecone instance, perform a similarity search, and return the formatted results.
        """
        vectordb = PineconeVectorStore(namespace=self.clerkId, embedding=self.embedding, index_name="raggpt")
        results = vectordb.similarity_search(query=query, k=2)

        # formatting the results
        formatted_results = self.format_results(results)
        print("Results formatted successfully!")
        print(formatted_results)

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