from sqlalchemy import Column, String, Integer, DateTime, Enum, ForeignKey, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import enum
from datetime import datetime
from sqlalchemy import Boolean
from database import Base
from pydantic import BaseModel, EmailStr
from typing import List
from sqlalchemy import ARRAY

# Define the Sender Enum
class Sender(enum.Enum):
    USER = "USER"
    BOT = "BOT"


# Define the UserCreate model

class UserCreate(BaseModel):
    clerkId: str
    email: str
    username: str

# Define the User model

class User(Base):
    __tablename__ = 'users'

    clerkId = Column(String, primary_key=True, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    documents = relationship("Document", back_populates="user")
    queries = relationship("Query", back_populates="user")
    chatHistory = relationship("ChatHistory", back_populates="user")


# Define the Document model
class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    clerkId = Column(String, ForeignKey('users.clerkId'))
    title = Column(String, nullable=True)
    link = Column(String, nullable=True)
    user = relationship("User", back_populates="documents")


# Define the Query model
class Query(Base):
    __tablename__ = 'queries'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    clerkId = Column(String, ForeignKey('users.clerkId'))
    query = Column(String)
    user = relationship("User", back_populates="queries")


# Define the ChatHistory model
class ChatHistory(Base):
    __tablename__ = 'chat_history'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    clerkId = Column(String, ForeignKey('users.clerkId'))
    message = Column(String)
    sender = Column(Enum(Sender))
    user = relationship("User", back_populates="chatHistory")