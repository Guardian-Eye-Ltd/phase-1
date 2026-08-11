from enum import Enum
from typing import List, Union
from fastapi import HTTPException, status, Depends
from app.models.user import User

class SystemRole(str, Enum):
    """Enumeration of system authorization roles."""
    ADMIN = "Admin"
    OPERATOR = "Operator"
    INVESTIGATOR = "Investigator"
    VIEWER = "Viewer"

class PermissionChecker:
    """
    FastAPI dependency for Role-Based Access Control (RBAC) enforcement.
    
    Verifies that the authenticated user possesses an authorized system role.
    """
    def __init__(self, allowed_roles: Union[List[str], List[SystemRole], str, SystemRole]):
        if isinstance(allowed_roles, (str, SystemRole)):
            roles_list = [allowed_roles]
        else:
            roles_list = allowed_roles

        self.allowed_roles = [
            r.value if hasattr(r, "value") else str(r) for r in roles_list
        ]


    def __call__(self, current_user: User) -> User:
        """
        Enforce role validation against the current user context.
        
        Args:
            current_user: The authenticated User object (injected by get_current_user).
            
        Returns:
            The validated User object.
            
        Raises:
            HTTPException: 403 Forbidden if user role is not authorized.
        """
        if not current_user.role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User has no assigned role and cannot perform this operation."
            )
            
        user_role = current_user.role.role_name
        
        if user_role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation forbidden. Required role: {self.allowed_roles}, provided role: '{user_role}'."
            )
            
        return current_user

# Pre-instantiated reusable permission guards
RequireAdmin = PermissionChecker([SystemRole.ADMIN])
RequireOperator = PermissionChecker([SystemRole.ADMIN, SystemRole.OPERATOR])
RequireInvestigator = PermissionChecker([SystemRole.ADMIN, SystemRole.OPERATOR, SystemRole.INVESTIGATOR])
RequireViewer = PermissionChecker([SystemRole.ADMIN, SystemRole.OPERATOR, SystemRole.INVESTIGATOR, SystemRole.VIEWER])
