import uuid
from typing import Generator, Annotated
from fastapi import Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.errors import ForbiddenException, NotFoundException, UnauthorizedException
from app.core.rbac import Role
from app.db.session import async_session
from app.models.user import User
from app.schemas.token import TokenPayload

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login"
)

async def get_db() -> Generator[AsyncSession, None, None]:
    async with async_session() as session:
        yield session

SessionDep = Annotated[AsyncSession, Depends(get_db)]
TokenDep = Annotated[str, Depends(reusable_oauth2)]

async def get_current_user(session: SessionDep, token: TokenDep) -> User:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
    except (JWTError, ValidationError):
        raise UnauthorizedException(detail="Unauthorized")
    user = await session.get(User, uuid.UUID(token_data.sub))
    if not user:
        raise NotFoundException(detail="User not found")
    return user

CurrentUser = Annotated[User, Depends(get_current_user)]

async def get_current_active_user(current_user: CurrentUser) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

CurrentActiveUser = Annotated[User, Depends(get_current_active_user)]

class RoleChecker:
    def __init__(self, allowed_roles: list[Role]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: Annotated[User, Depends(get_current_active_user)]) -> User:
        if user.role not in self.allowed_roles:
            raise ForbiddenException(detail="Insufficient permissions")
        return user

def check_role(roles: list[Role]):
    return Depends(RoleChecker(roles))
