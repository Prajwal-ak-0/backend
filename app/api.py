from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from database import SessionLocal, engine
import models
from typing import List

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
    links: List[str]

@app.post("/api/users/", response_model=UserCreate)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    print(f"Recieved user: {user}")
    db_user = db.query(models.User).filter(models.User.clerkId == user.clerkId).first()
    if db_user is None:
        # If the user doesn't exist, create a new one
        db_user = models.User(
            clerkId=user.clerkId,
            email=user.email,
            username=user.username,
            links=user.links
        )
        db.add(db_user)
        print("User created Successfully")
    else:
        # If the user does exist, append the new link to their links field
        db_user.links.extend(user.links)
        print("Added Link to user Successfully")
    db.commit()
    db.refresh(db_user)
    return db_user