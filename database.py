import os
from sqlalchemy import create_engine, URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv

load_dotenv()

DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

connection_string = URL.create(
    'postgresql+psycopg2',
    username=DB_USERNAME,
    password=DB_PASSWORD,
    host=DB_HOST,
    database=DB_NAME,
    query={
        "sslmode": "require",
    }
)

try:
    engine = create_engine(connection_string)
    print(engine)
    print("Database connection successful")
except Exception as e:
    print(f"Database connection failed: {e}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
