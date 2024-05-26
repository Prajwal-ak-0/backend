import uvicorn
# from utils.prepare_vectordb import PrepareVectorDB

if __name__ == "__main__":
    # prepare_vectordb = PrepareVectorDB(link="https://arxiv.org/pdf/2106.04561.pdf", clerkId="user_2g5KfdZ6FcmY6FgjNNRKqbCvioE")
    # docs = prepare_vectordb.create_pinecone_instance_and_query(query="Explain both of them.")

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)