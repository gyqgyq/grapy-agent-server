from fastapi import APIRouter

router = APIRouter(prefix="/agent", tags=["agent"])

@router.get("/")
def read_root():
    return {"message": "Hello, agent!"}