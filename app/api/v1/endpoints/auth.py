import logging
from typing import Any
from datetime import timedelta
from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from app.core import security
from app.core.config import settings
from app.api.deps import SessionDep
from app.models.user import User
from app.schemas.token import AuthResponse
from app.schemas.user import User as UserSchema, UserCreate, UserLogin, UserUpdate
from app.schemas.response import APIResponse
from app.core.errors import BadRequestException, ForbiddenException, InternalServerErrorException, UnauthorizedException
from app.api import deps 

logger = logging.getLogger(__name__)

router = APIRouter()

async def verify_user_update_data(
    session: deps.SessionDep, 
    current_user: User, 
    user_in: UserUpdate
) -> None:
    """
    Verify user update logic. (Currently no-op as email verification is removed)
    """
    pass


@router.post("/login", response_model=APIResponse[AuthResponse], status_code=status.HTTP_200_OK)
async def login_access_token(
    session: SessionDep, login_data: UserLogin
) -> Any:
    try:
        stmt = select(User).where(User.email == login_data.email)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            raise UnauthorizedException(detail="Incorrect email or password")
        elif not user.is_active:
            raise ForbiddenException(detail="Inactive user")
            
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = security.create_access_token(
            user.id, expires_delta=access_token_expires
        )
        
        auth_data = AuthResponse(
            access_token=access_token,
            token_type="bearer",
            user=UserSchema.model_validate(user)
        )
        return APIResponse.success_response(data=auth_data, message="Login successful")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise InternalServerErrorException(detail="Login failed")


@router.post("/register", response_model=APIResponse[AuthResponse], status_code=status.HTTP_201_CREATED)
async def register_user(
    *,
    session: deps.SessionDep,
    user_in: UserCreate,
) -> Any:
    try:
        stmt = select(User).where(User.email == user_in.email)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        if user:
            raise BadRequestException(
                detail="The user with this email already exists in the system"
            )

        user = User(
            first_name=user_in.first_name,
            last_name=user_in.last_name,
            email=user_in.email,
            hashed_password=security.get_password_hash(user_in.password),
            role=user_in.role,
            is_active=user_in.is_active,
            is_onboarded=False,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        
        # Generate access token for immediate login after registration
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = security.create_access_token(
            user.id, expires_delta=access_token_expires
        )
        
        auth_data = AuthResponse(
            access_token=access_token,
            token_type="bearer",
            user=UserSchema.model_validate(user)
        )
        return APIResponse.success_response(data=auth_data, message="User registered successfully")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise InternalServerErrorException(detail="Failed to create user")


@router.put("/update-me", response_model=APIResponse[UserSchema], status_code=status.HTTP_200_OK)
async def update_user_me(
    *,
    session: deps.SessionDep,
    user_in: UserUpdate,
    current_user: deps.CurrentUser,
) -> Any:

    try:
        # Verify update logic
        await verify_user_update_data(session=session, current_user=current_user, user_in=user_in)
        
        update_data = user_in.model_dump(exclude_unset=True)
        
        # Only update password if provided and not empty
        if "password" in update_data:
            password = update_data.pop("password")
            if password and password.strip():
                hashed_password = security.get_password_hash(password)
                current_user.hashed_password = hashed_password
            
        for field, value in update_data.items():
            setattr(current_user, field, value)

        session.add(current_user)
        await session.commit()
        await session.refresh(current_user)
        
        return APIResponse.success_response(
            data=UserSchema.model_validate(current_user), 
            message="User information updated successfully"
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error updating user profile: {e}", exc_info=True)
        raise InternalServerErrorException(detail="Failed to update user information")
