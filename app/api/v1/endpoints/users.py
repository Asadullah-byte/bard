from typing import Any, List
from fastapi import APIRouter, status
from sqlalchemy import select
from app.api import deps
from app.models.user import User
from app.schemas.user import User as UserSchema
from app.core.rbac import Role

from app.schemas.response import APIResponse
from app.core.errors import InternalServerErrorException

router = APIRouter()

@router.get("/me", response_model=APIResponse[UserSchema], status_code=status.HTTP_200_OK)
async def read_user_me(
    current_user: deps.CurrentUser,
) -> Any:
    try:
        return APIResponse.success_response(data=current_user)
    except Exception as e:
        raise InternalServerErrorException(detail="Failed to retrieve user information")

@router.get("", response_model=APIResponse[List[UserSchema]], dependencies=[deps.check_role([Role.ADMIN])], status_code=status.HTTP_200_OK)
async def read_users(
    session: deps.SessionDep,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve users. Only for admins.
    """
    try:
        stmt = select(User).offset(skip).limit(limit)
        result = await session.execute(stmt)
        users = result.scalars().all()
        return APIResponse.success_response(data=list(users))
    except Exception as e:
        raise InternalServerErrorException(detail="Failed to retrieve users")
