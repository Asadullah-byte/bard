from typing import Optional
from datetime import datetime
import uuid
from pydantic import BaseModel, EmailStr
from app.core.rbac import Role


class UserBase(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    is_active: bool = True
    role: Role = Role.USER


class UserCreate(UserBase):
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    password: Optional[str] = None


class User(UserBase):
    id: uuid.UUID
    is_onboarded: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
