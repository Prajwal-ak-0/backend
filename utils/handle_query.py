from utils.prepare_vectordb import PrepareVectorDB
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")


class HandleQuery:
    def __init__(self, query: str, clerkId: str) -> None:
        self.query = query
        self.clerkId = clerkId
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    def create_pinecone_instance_and_query(self):
        """
        Create a Pinecone instance, perform a similarity search, and return the formatted results.
        """
        vectordb = PineconeVectorStore(index_name="raggpt", pinecone_api_key=os.getenv("PINECONE_API_KEY"),
                                       namespace=self.clerkId, embedding=self.embedding)
        results = vectordb.similarity_search(self.query, k=3)
        print("Results:", results)

        # format the results
        formatted_results = []
        for result in results:
            formatted_result = {
                "clerkId": result.id,
                "score": result.score
            }
            formatted_results.append(formatted_result)

        print("Formatted Results:", formatted_results)
