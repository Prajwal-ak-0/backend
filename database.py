import os
from sqlalchemy import create_engine, URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Use environment variables to store sensitive information
DB_USERNAME = os.getenv('DB_USERNAME', 'rag-gpt_owner')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'A3xQJVRj1SEC')
DB_HOST = os.getenv('DB_HOST', 'ep-restless-hall-a5vpjlwg.us-east-2.aws.neon.tech')
DB_NAME = os.getenv('DB_NAME', 'rag-gpt')

# Extract endpoint ID from the host for use in the connection string
ENDPOINT_ID = DB_HOST.split('.')[0]

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
