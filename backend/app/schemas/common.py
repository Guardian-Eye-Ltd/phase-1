from pydantic import BaseModel, Field
from typing import Any

class BaseResponse(BaseModel):
    """
    Standard envelope format wrapper for all API success responses.
    """
    status: str = Field("success", description="Status code indicator ('success' or 'fail')")
    message: str | None = Field(None, description="Detailed text message describing action results")
    data: Any | None = Field(None, description="Output payload data matching request parameters")

class ErrorResponse(BaseModel):
    """
    Standard format layout representing system or endpoint exceptions.
    """
    status: str = Field("error", description="Error status indicator")
    detail: str = Field(..., description="Message containing exact reason for validation/runtime failure")
