from pydantic import BaseModel, Field
from typing import Any, Dict

class AuditLogResponse(BaseModel):
    id: int
    user_id: int | None
    user_name: str
    username: str
    action: str
    resource_type: str
    resource_id: str | None
    timestamp: str
    metadata: Dict[str, Any] | None = None

class AuditLogListResponse(BaseModel):
    items: list[AuditLogResponse]
    total: int
    page: int
    size: int
