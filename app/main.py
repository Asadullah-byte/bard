from fastapi import FastAPI, HTTPException
from app.core.config import settings
from app.core.errors import http_exception_handler

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Register exception handler
app.add_exception_handler(HTTPException, http_exception_handler)

from app.api.v1.api import api_router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {"message": "Hello World", "project": settings.PROJECT_NAME}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
