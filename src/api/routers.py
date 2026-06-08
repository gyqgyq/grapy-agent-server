from fastapi import APIRouter

from src.agent.router import router as agent_router

router = APIRouter()

router.include_router(agent_router)