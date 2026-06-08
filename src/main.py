from fastapi import FastAPI, Depends

from src.api.routers import router

app = FastAPI()
app.include_router(router)