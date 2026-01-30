from typing import Optional
from pydantic import BaseModel
from app.schemas.user import User


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    sub: Optional[str] = None


class AuthResponse(BaseModel):
    """Combined auth response with token and user details."""
    access_token: str
    token_type: str
    user: User
