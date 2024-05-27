import os
import requests
from PyPDF2 import PdfFileReader
from io import BytesIO
from dotenv import load_dotenv
from typing import List
from enum import Enum
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, desc
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from models import ChatHistory, Sender
from database import SessionLocal
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_structured_output_runnable
import re
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from transformers import AutoProcessor, AutoModelForPreTraining, pipeline
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["HUGGINGFACEHUB_API_KEY"] = os.getenv("HUGGINGFACEHUB_API_KEY")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = os.getenv("TF_ENABLE_ONEDNN_OPTS")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

class SenderType(Enum):
    USER = "USER"
    BOT = "BOT"


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
        self.embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.clerkId = clerkId

        # Create a new session
        self.db = SessionLocal()

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

        print("Embedding Started")
        model_name = "maidalun1020/bce-embedding-base_v1"
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'batch_size': 64, 'normalize_embeddings': True, 'show_progress_bar': False}

        embedding = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv("HUGGINGFACEHUB_API_KEY"),
            model_name = model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        print("Preparing vectordb Started.")
        vectorstore = PineconeVectorStore(index_name="raggpt", embedding=embedding, namespace=self.clerkId)
        vectorstore.add_documents(chunked_documents)
        Response = vectorstore.similarity_search(query="What are encoders?", k=1)
        print(Response)

    def create_pinecone_instance_and_query(self, query: str):
        print("Creating Pinecone instance and querying...")
        updatedQuery = self.summarize_histrory(self.retrieve_history(self.clerkId), query)
        vectorstore = PineconeVectorStore(index_name="raggpt", embedding=self.embedding, namespace=self.clerkId)
        res = vectorstore.similarity_search(query=updatedQuery, k=2)

        # Create a ChatGroq model
        chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

        # Prepare the prompt
        system = "You are an assistant for question-answering tasks. \
            Use the following pieces of retrieved context to answer the question. \
            If you don't know the answer, just say that you don't know. \
            Use three sentences maximum and keep the answer concise."
        human = f"{res}\n\n{updatedQuery}"
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

        # Generate the response
        chain = prompt | chat
        response = chain.invoke({})

        # Update the chat history
        self.update_history_in_db(self.clerkId, query, Sender.BOT)
        self.update_history_in_db(self.clerkId, response.content, Sender.BOT)

        print("Results Printing: ", response.content)
        return response.content

    def summarize_histrory(self, history: List[ChatHistory], query: str):
        """
        Summarize the history of the chat messages.
        """
        print("Summarizing the chat history...")

        # Create a ChatGroq model
        chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

        # Create a chat custom prompt template
        template = """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is.\n\nChat History:\n\n{chat_history}\n\nUser Question:\n\n{user_question}"""

        # Prepare the chat history
        chat_history = "\n\n".join([f"{message.sender.value}: {message.message}" for message in history])

        # Prepare the user question
        user_question = query

        prompt = ChatPromptTemplate.from_template(template)

        # Generate the response
        chain = prompt | chat
        response = chain.invoke({"chat_history": chat_history, "user_question": user_question})

        print("Summarized successfully: ")
        return response.content

    def retrieve_history(self, user_id: str) -> List[ChatHistory]:
        print("Retrieving history...")
        history = self.db.query(ChatHistory) \
            .filter(ChatHistory.clerkId == user_id) \
            .order_by(ChatHistory.id.desc()) \
            .limit(10) \
            .all()
        history.reverse()
        return history

    def update_history_in_db(self, user_id: str, message: str, sender: Sender) -> None:
        print("Updating history in database...")
        new_message = ChatHistory(
            clerkId=user_id,
            message=message,
            sender=sender
        )
        self.db.add(new_message)
        self.db.commit()