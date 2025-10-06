from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine
from datetime import datetime
from pydantic import BaseModel, EmailStr
import os
from dotenv import load_dotenv
from sqlalchemy.ext.mutable import MutableDict


# Load environment variables
load_dotenv()

# Database configuration
DB_USER = os.getenv("DB_USER", "saasdb_owner")
DB_PASSWORD = os.getenv("DB_PASSWORD", "hZm1Ql3RgJjs")
DB_HOST = os.getenv("DB_HOST", "ep-holy-voice-a2p4hd0z-pooler.eu-central-1.aws.neon.tech")
DB_NAME = os.getenv("DB_NAME", "saasdb")

# Create Database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}?sslmode=require"

# Configure SQLAlchemy engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    echo=True,
    pool_size=5,  # Maximum number of connections in the pool
    max_overflow=10,  # Maximum number of connections that can be created beyond pool_size
    pool_timeout=30,  # Timeout for getting a connection from the pool
    pool_recycle=1800,  # Recycle connections after 30 minutes
    pool_pre_ping=True,  # Add this to detect disconnections

)

# Session configuration
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserDB(Base):
    __tablename__ = "user"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_admin = Column(Boolean, default=False)

class ContactDB(Base):
    __tablename__ = "contact"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    company = Column(String, nullable=True)
    collaborator = Column(String, nullable=True)
    message = Column(String, nullable=True)
    image_url = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class PromptDB(Base):
    __tablename__ = "prompts"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)



class ContactCreate(BaseModel):
    name: str
    email: EmailStr
    company: str | None = None
    collaborator: str | None = None
    message: str | None = None

class ContactResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    company: str | None
    collaborator: str | None
    message: str | None
    image_url: str | None
    created_at: datetime

    class Config:
        from_attributes = True

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class User(BaseModel):
    id: int
    name: str
    email: EmailStr
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str



class PromptCreate(BaseModel):
    content: str

class PromptResponse(BaseModel):
    id: int
    content: str
    created_at: datetime

    class Config:
        from_attributes = True




def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    Base.metadata.create_all(bind=engine)


# New model: BreadSession for persisting bread conversation state
class BreadSession(Base):
    __tablename__ = "bread_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, nullable=False)
    mode = Column(String, nullable=True)  # "all-at-once" | "one-by-one"
    # answers = Column(JSON, nullable=True)  # normalized answers collected so far
    answers = Column(MutableDict.as_mutable(JSON), nullable=True)
    completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


# New model: RecipeSession for general-purpose recipe conversations
class RecipeSession(Base):
    __tablename__ = "recipe_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, nullable=False)
    mode = Column(String, nullable=True)  # "all-at-once" | "one-by-one"
    answers = Column(JSON, nullable=True)
    completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)



# from sqlalchemy import Column, Integer, String, DateTime, Boolean
# from sqlalchemy.orm import sessionmaker, declarative_base
# from sqlalchemy import create_engine
# from datetime import datetime
# from pydantic import BaseModel, EmailStr
# DATABASE_URL = "postgresql://saasdb_owner:hZm1Ql3RgJjs@ep-holy-voice-a2p4hd0z-pooler.eu-central-1.aws.neon.tech/saasdb?sslmode=require"

# engine = create_engine(DATABASE_URL, echo=True)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()
# class UserDB(Base):
#     __tablename__ = "user"
    
#     id = Column(Integer, primary_key=True, index=True)
#     name = Column(String, nullable=False)
#     email = Column(String, unique=True, nullable=False)
#     hashed_password = Column(String, nullable=False)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     is_admin = Column(Boolean, default=False)

# class ContactDB(Base):
#     __tablename__ = "contact"

#     id = Column(Integer, primary_key=True, index=True)
#     name = Column(String, nullable=False)
#     email = Column(String, nullable=False)
#     company = Column(String, nullable=True)
#     collaborator = Column(String, nullable=True)
#     message = Column(String, nullable=True)
#     image_url = Column(String, nullable=False)
#     created_at = Column(DateTime, default=datetime.utcnow)

# class ContactCreate(BaseModel):
#     name: str
#     email: EmailStr
#     company: str | None = None
#     collaborator: str | None = None
#     message: str | None = None

# class ContactResponse(BaseModel):
#     id: int
#     name: str
#     email: EmailStr
#     company: str | None
#     collaborator: str | None
#     message: str | None
#     image_url: str | None
#     created_at: datetime

#     class Config:
#         from_attributes = True

# class UserCreate(BaseModel):
#     name: str
#     email: EmailStr
#     password: str

# class User(BaseModel):
#     id: int
#     name: str
#     email: EmailStr
#     created_at: datetime

#     class Config:
#         from_attributes = True

# class Token(BaseModel):
#     access_token: str
#     token_type: str

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# def create_tables():
#     Base.metadata.create_all(bind=engine)