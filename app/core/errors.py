from fastapi import status, HTTPException, Request
from fastapi.responses import JSONResponse
from app.schemas.response import APIResponse


async def http_exception_handler(request: Request, exc: HTTPException):
    error_code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "CONFLICT",
        500: "INTERNAL_SERVER_ERROR"
    }
    error_code = error_code_map.get(exc.status_code, f"HTTP_{exc.status_code}")
    
    response_data = APIResponse.error_response(
        message=str(exc.detail),
        error={"code": error_code}
    ).model_dump()
    return JSONResponse(status_code=exc.status_code, content=response_data)


class NotFoundException(HTTPException):
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


class BadRequestException(HTTPException):
    def __init__(self, detail: str = "Bad request"):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


class UnauthorizedException(HTTPException):
    def __init__(self, detail: str = "Unauthorized"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


class ForbiddenException(HTTPException):
    def __init__(self, detail: str = "Forbidden"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)
class InternalServerErrorException(HTTPException):
    def __init__(self, detail: str = "Internal Server Error"):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)
