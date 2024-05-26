import uvicorn
from utils.prepare_vectordb import PrepareVectorDB

if __name__ == "__main__":
    prepare_vectordb = PrepareVectorDB(
        link="https://firebasestorage.googleapis.com/v0/b/rag-gpt.appspot.com/o/Transformer.pdf?alt=media&token=f2f4845d-c7b0-4f61-a1e9-67cf7aa188d2",
        chunk_size=1000,
        chunk_overlap=100
    )
    prepare_vectordb.prepare_and_save_vectordb()
    # uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
