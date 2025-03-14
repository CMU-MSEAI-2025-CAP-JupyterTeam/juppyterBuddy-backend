from fastapi import FastAPI
from app.core.database import init_db

# Initialize Database on Startup
init_db()

app = FastAPI()

@app.get("/")
def root():
    return {"message": "JupyterBuddy API is running!"}
