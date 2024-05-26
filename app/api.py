from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from database import SessionLocal, engine
import models
from utils.prepare_vectordb import PrepareVectorDB

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class UserCreate(BaseModel):
    clerkId: str
    username: str
    email: str

@app.post("/api/users/", response_model=UserCreate)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.clerkId == user.clerkId).first()
    if db_user is None:
        db_user = models.User(
            clerkId=user.clerkId,
            email=user.email,
            username=user.username,
        )
        db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

class DocumentCreate(BaseModel):
    clerkId: str
    link: str

@app.post("/api/link", response_model=DocumentCreate)
def create_document(document: DocumentCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.clerkId == document.clerkId).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    db_document = models.Document(
        clerkId=document.clerkId,
        link=document.link,
        user=db_user
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)


    # Prepare vectordb
    vectordb = PrepareVectorDB(document.link, clerkId=document.clerkId)
    docs = vectordb.load_documents()
    chunked_documents = vectordb.chunk_documents(docs)
    vectorstore = vectordb.prepare_vectordb_to_store(chunked_documents)

    return db_document

class QueryCreate(BaseModel):
    clerkId: str
    query: str

@app.post("/api/query", response_model=QueryCreate)
def create_query(query: QueryCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.clerkId == query.clerkId).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    db_query = models.Query(
        clerkId=query.clerkId,
        query=query.query,
        user=db_user
    )
    db.add(db_query)
    db.commit()
    db.refresh(db_query)

    # Query vectordb
    vectordb = PrepareVectorDB(link=None, clerkId=query.clerkId)
    results = vectordb.create_pinecone_instance_and_query(query=query.query)

    return {"clerkId": query.clerkId, "query": query.query, "results": results}