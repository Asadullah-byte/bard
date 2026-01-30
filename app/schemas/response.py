from typing import Optional, Any, Dict, Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class APIResponse(BaseModel, Generic[T]):

    success: bool
    message: str
    data: Optional[T] = None
    error: Optional[Dict[str, Any]] = None

    @classmethod
    def success_response(cls, data: T, message: str = "Success") -> "APIResponse[T]":

        return cls(success=True, message=message, data=data)

    @classmethod
    def error_response(cls, message: str, error: Optional[Dict[str, Any]] = None) -> "APIResponse[T]":
        return cls(success=False, message=message, error=error)
