from fastapi import FastAPI

app = FastAPI()  # Create a FastAPI instance

@app.get("/")  # Define a GET route
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/items/{item_id}")  # `{item_id}` is a variable
def read_item(item_id: int):
    return {"item_id": item_id}