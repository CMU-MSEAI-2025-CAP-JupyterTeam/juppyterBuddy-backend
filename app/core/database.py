# app/config/py
#.................... Databse Configuration ....................
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Database Configuration (Change to your preferred database)
DATABASE_URL = "sqlite:///./jupyterbuddy.db"  # where db is stored. default is Local SQLite file (Change to PostgreSQL if needed)
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)  # connect to database
SessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine
)  # Creates a session factory for interacting with the database.
Base = declarative_base()  # Allows us to define database tables as Python classes.

