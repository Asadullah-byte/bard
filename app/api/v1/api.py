from fastapi import APIRouter
from app.api.v1.endpoints import auth, users
from app.api.v1.endpoints import images
from app.api.v1.endpoints import onboarding
from app.api.v1.endpoints import journey

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(images.router, prefix="/users/images", tags=["images"])
api_router.include_router(onboarding.router, prefix="/users/onboarding", tags=["onboarding"])
api_router.include_router(journey.router, prefix="/users/journey", tags=["journey"])