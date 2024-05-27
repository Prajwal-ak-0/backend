import os
import uvicorn
from utils.prepare_vectordb import PrepareVectorDB

if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 8000))
    # uvicorn.run("app.api:app", host="0.0.0.0", port=port, reload=True)

    vectordb = PrepareVectorDB("https://firebasestorage.googleapis.com/v0/b/rag-gpt.appspot.com/o/Transformer.pdf?alt=media&token=22be1cad-ab0b-4a1a-a767-8d7bdff3cf81", clerkId="user_2g5KfdZ6FcmY6FgjNNRKqbCvioE")
    docs = vectordb.load_documents()
    chunked_documents = vectordb.chunk_documents(docs)
    vectordb.prepare_vectordb_to_store(chunked_documents)
    print("Successfully Prepared VectorDB...")

    vectordb = PrepareVectorDB(link=None, clerkId="user_2g5KfdZ6FcmY6FgjNNRKqbCvioE")
    results = vectordb.create_pinecone_instance_and_query(query="What are encoders?")
    print("Results Printing: ", results)
