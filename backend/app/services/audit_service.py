import json
from typing import Any, Dict, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.models.audit import AuditLog
from app.models.user import User

class AuditService:
    """
    Service for creating and querying chain-of-custody audit logs for all investigator activities.
    """

    @staticmethod
    async def log_action(
        db: AsyncSession,
        user_id: int | None,
        action: str,
        resource_type: str,
        resource_id: str | None = None,
        metadata: Dict[str, Any] | None = None
    ) -> AuditLog:
        """
        Record a new forensic audit log entry.
        """
        metadata_str = json.dumps(metadata) if metadata else None
        audit_entry = AuditLog(
            user_id=user_id,
            action=action.upper(),
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id is not None else None,
            metadata_json=metadata_str
        )
        db.add(audit_entry)
        await db.commit()
        await db.refresh(audit_entry)
        return audit_entry

    @staticmethod
    async def get_audit_logs(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        action_filter: str | None = None,
        user_id_filter: int | None = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Query audit logs with optional filtering and joined user information.
        """
        query = select(AuditLog, User.full_name, User.username).outerjoin(User, AuditLog.user_id == User.id)

        if action_filter:
            query = query.where(AuditLog.action == action_filter.upper())
        if user_id_filter:
            query = query.where(AuditLog.user_id == user_id_filter)

        # Count total matches
        from sqlalchemy import func
        count_query = select(func.count()).select_from(AuditLog)
        if action_filter:
            count_query = count_query.where(AuditLog.action == action_filter.upper())
        if user_id_filter:
            count_query = count_query.where(AuditLog.user_id == user_id_filter)
        
        total_result = await db.execute(count_query)
        total = total_result.scalar_one()

        query = query.order_by(desc(AuditLog.timestamp)).offset(skip).limit(limit)
        results = await db.execute(query)

        logs = []
        for log, full_name, username in results.all():
            metadata = None
            if log.metadata_json:
                try:
                    metadata = json.loads(log.metadata_json)
                except Exception:
                    metadata = {"raw": log.metadata_json}
                    
            logs.append({
                "id": log.id,
                "user_id": log.user_id,
                "user_name": full_name or "System / Anonymous",
                "username": username or "system",
                "action": log.action,
                "resource_type": log.resource_type,
                "resource_id": log.resource_id,
                "timestamp": log.timestamp.isoformat(),
                "metadata": metadata
            })

        return logs, total
