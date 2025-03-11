# app/api/routes.py
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/")
async def root():
    return {"message": "Welcome to JupyterBuddy API"}