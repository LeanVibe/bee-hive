"""
RBAC (Role-Based Access Control) API Endpoints

Enterprise-grade role and permission management system for Epic 6 Phase 2.
Provides comprehensive RBAC functionality with role management, permission
matrix visualization, and bulk user operations for 60%+ admin overhead reduction.

Epic 6: Advanced User Experience & Adoption - Phase 2
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query, Path, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, or_

from ..core.database import get_async_session
from ..core.logging_service import get_component_logger
from ..core.auth import (
    get_current_user, require_permission, Permission, UserRole, 
    get_auth_service, User
)

logger = get_component_logger("rbac_api")

router = APIRouter(prefix="/api/rbac", tags=["rbac"])

# Pydantic models for RBAC operations

class RoleCreate(BaseModel):
    """Create new role request."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    permissions: List[Permission]
    is_system_role: bool = False
    parent_role: Optional[str] = None  # For role hierarchy

class RoleUpdate(BaseModel):
    """Update role request."""
    name: Optional[str] = None
    description: Optional[str] = None
    permissions: Optional[List[Permission]] = None
    is_active: Optional[bool] = None

class RoleResponse(BaseModel):
    """Role response model."""
    id: str
    name: str
    description: Optional[str]
    permissions: List[Permission]
    is_system_role: bool
    is_active: bool
    user_count: int
    created_at: datetime
    updated_at: datetime
    parent_role: Optional[str] = None

class UserRoleAssignment(BaseModel):
    """User role assignment model."""
    user_id: str
    role_ids: List[str]
    assigned_by: Optional[str] = None
    expires_at: Optional[datetime] = None

class BulkRoleAssignment(BaseModel):
    """Bulk role assignment model."""
    user_ids: List[str]
    role_ids: List[str]
    assigned_by: Optional[str] = None
    expires_at: Optional[datetime] = None

class PermissionMatrixEntry(BaseModel):
    """Permission matrix entry."""
    role_name: str
    permission: Permission
    granted: bool
    inherited: bool = False
    source_role: Optional[str] = None

class PermissionMatrix(BaseModel):
    """Complete permission matrix."""
    roles: List[str]
    permissions: List[Permission]
    matrix: List[PermissionMatrixEntry]
    last_updated: datetime

class RoleHierarchy(BaseModel):
    """Role hierarchy structure."""
    role_id: str
    role_name: str
    parent_id: Optional[str]
    children: List['RoleHierarchy']
    depth: int
    user_count: int

RoleHierarchy.model_rebuild()  # Rebuild for forward reference

# In-memory role store (should be replaced with database models)
_roles_store: Dict[str, Dict[str, Any]] = {}
_user_roles: Dict[str, List[str]] = {}

# Initialize default roles if not exist
def _ensure_default_roles():
    """Ensure default system roles exist."""
    if not _roles_store:
        default_roles = [
            {
                "id": str(uuid4()),
                "name": "Super Admin",
                "description": "Full system access",
                "permissions": list(Permission),
                "is_system_role": True,
                "is_active": True,
                "user_count": 0,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            },
            {
                "id": str(uuid4()),
                "name": "Enterprise Admin", 
                "description": "Enterprise management access",
                "permissions": [
                    Permission.CREATE_PILOT, Permission.VIEW_PILOT, Permission.UPDATE_PILOT,
                    Permission.VIEW_ROI_METRICS, Permission.CREATE_ROI_METRICS,
                    Permission.MANAGE_USERS
                ],
                "is_system_role": True,
                "is_active": True,
                "user_count": 0,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            },
            {
                "id": str(uuid4()),
                "name": "Developer",
                "description": "Development access",
                "permissions": [
                    Permission.VIEW_PILOT, Permission.CREATE_DEVELOPMENT_TASK,
                    Permission.VIEW_DEVELOPMENT_TASK, Permission.EXECUTE_DEVELOPMENT_TASK
                ],
                "is_system_role": True,
                "is_active": True,
                "user_count": 0,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        ]
        
        for role in default_roles:
            _roles_store[role["id"]] = role

_ensure_default_roles()

@router.get("/roles", response_model=List[RoleResponse])
async def get_roles(
    include_inactive: bool = Query(default=False),
    search: Optional[str] = Query(default=None),
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Get all roles with filtering options."""
    try:
        roles = []
        for role_data in _roles_store.values():
            if not include_inactive and not role_data.get("is_active", True):
                continue
                
            if search and search.lower() not in role_data["name"].lower():
                continue
                
            # Count users with this role
            user_count = sum(
                1 for user_role_list in _user_roles.values() 
                if role_data["id"] in user_role_list
            )
            role_data["user_count"] = user_count
            
            roles.append(RoleResponse(**role_data))
        
        logger.info(f"Retrieved {len(roles)} roles for user {current_user.id}")
        return roles
        
    except Exception as e:
        logger.error(f"Error retrieving roles: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve roles")

@router.post("/roles", response_model=RoleResponse)
async def create_role(
    role_data: RoleCreate,
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Create a new role."""
    try:
        # Check if role name already exists
        existing_role = next(
            (r for r in _roles_store.values() if r["name"].lower() == role_data.name.lower()),
            None
        )
        if existing_role:
            raise HTTPException(
                status_code=400, 
                detail=f"Role with name '{role_data.name}' already exists"
            )
        
        # Create new role
        role_id = str(uuid4())
        now = datetime.utcnow()
        
        new_role = {
            "id": role_id,
            "name": role_data.name,
            "description": role_data.description,
            "permissions": role_data.permissions,
            "is_system_role": role_data.is_system_role,
            "is_active": True,
            "user_count": 0,
            "created_at": now,
            "updated_at": now,
            "parent_role": role_data.parent_role
        }
        
        _roles_store[role_id] = new_role
        
        logger.info(f"Created role '{role_data.name}' by user {current_user.id}")
        return RoleResponse(**new_role)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating role: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create role")

@router.get("/roles/{role_id}", response_model=RoleResponse)
async def get_role(
    role_id: str = Path(...),
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Get specific role by ID."""
    try:
        role_data = _roles_store.get(role_id)
        if not role_data:
            raise HTTPException(status_code=404, detail="Role not found")
        
        # Count users with this role
        user_count = sum(
            1 for user_role_list in _user_roles.values() 
            if role_id in user_role_list
        )
        role_data["user_count"] = user_count
        
        return RoleResponse(**role_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving role {role_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve role")

@router.put("/roles/{role_id}", response_model=RoleResponse)
async def update_role(
    role_id: str = Path(...),
    role_update: RoleUpdate = ...,
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Update existing role."""
    try:
        role_data = _roles_store.get(role_id)
        if not role_data:
            raise HTTPException(status_code=404, detail="Role not found")
        
        # Prevent modification of system roles by non-super-admins
        if role_data.get("is_system_role") and current_user.role != UserRole.SUPER_ADMIN:
            raise HTTPException(
                status_code=403, 
                detail="Cannot modify system roles"
            )
        
        # Update fields
        update_data = role_update.dict(exclude_none=True)
        for field, value in update_data.items():
            role_data[field] = value
        
        role_data["updated_at"] = datetime.utcnow()
        
        logger.info(f"Updated role {role_id} by user {current_user.id}")
        
        # Count users with this role
        user_count = sum(
            1 for user_role_list in _user_roles.values() 
            if role_id in user_role_list
        )
        role_data["user_count"] = user_count
        
        return RoleResponse(**role_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating role {role_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update role")

@router.delete("/roles/{role_id}")
async def delete_role(
    role_id: str = Path(...),
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Delete a role (soft delete - deactivate)."""
    try:
        role_data = _roles_store.get(role_id)
        if not role_data:
            raise HTTPException(status_code=404, detail="Role not found")
        
        # Prevent deletion of system roles
        if role_data.get("is_system_role"):
            raise HTTPException(
                status_code=403, 
                detail="Cannot delete system roles"
            )
        
        # Check if role is assigned to users
        user_count = sum(
            1 for user_role_list in _user_roles.values() 
            if role_id in user_role_list
        )
        
        if user_count > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete role assigned to {user_count} users"
            )
        
        # Soft delete (deactivate)
        role_data["is_active"] = False
        role_data["updated_at"] = datetime.utcnow()
        
        logger.info(f"Deleted role {role_id} by user {current_user.id}")
        
        return JSONResponse({
            "status": "success", 
            "message": "Role deleted successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting role {role_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete role")

@router.post("/assign-roles")
async def assign_user_roles(
    assignment: UserRoleAssignment,
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Assign roles to a user."""
    try:
        # Validate role IDs exist
        invalid_roles = []
        for role_id in assignment.role_ids:
            if role_id not in _roles_store:
                invalid_roles.append(role_id)
        
        if invalid_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role IDs: {', '.join(invalid_roles)}"
            )
        
        # Assign roles
        _user_roles[assignment.user_id] = assignment.role_ids
        
        logger.info(
            f"Assigned {len(assignment.role_ids)} roles to user {assignment.user_id} "
            f"by {current_user.id}"
        )
        
        return JSONResponse({
            "status": "success",
            "message": f"Assigned {len(assignment.role_ids)} roles to user",
            "user_id": assignment.user_id,
            "role_ids": assignment.role_ids
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning roles: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to assign roles")

@router.post("/bulk-assign-roles")
async def bulk_assign_roles(
    assignment: BulkRoleAssignment,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Bulk assign roles to multiple users."""
    try:
        # Validate role IDs exist
        invalid_roles = []
        for role_id in assignment.role_ids:
            if role_id not in _roles_store:
                invalid_roles.append(role_id)
        
        if invalid_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role IDs: {', '.join(invalid_roles)}"
            )
        
        # Process bulk assignment in background
        background_tasks.add_task(
            process_bulk_role_assignment,
            assignment.user_ids,
            assignment.role_ids,
            current_user.id
        )
        
        return JSONResponse({
            "status": "success",
            "message": f"Bulk role assignment started for {len(assignment.user_ids)} users",
            "user_count": len(assignment.user_ids),
            "role_count": len(assignment.role_ids)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error bulk assigning roles: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to bulk assign roles")

@router.get("/permission-matrix", response_model=PermissionMatrix)
async def get_permission_matrix(
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Get complete permission matrix for all roles."""
    try:
        roles = list(_roles_store.keys())
        permissions = list(Permission)
        matrix_entries = []
        
        for role_id, role_data in _roles_store.items():
            if not role_data.get("is_active", True):
                continue
                
            role_permissions = role_data.get("permissions", [])
            
            for permission in permissions:
                granted = permission in role_permissions
                matrix_entries.append(PermissionMatrixEntry(
                    role_name=role_data["name"],
                    permission=permission,
                    granted=granted,
                    inherited=False,  # TODO: Implement role inheritance
                    source_role=None
                ))
        
        matrix = PermissionMatrix(
            roles=[role_data["name"] for role_data in _roles_store.values() 
                   if role_data.get("is_active", True)],
            permissions=permissions,
            matrix=matrix_entries,
            last_updated=datetime.utcnow()
        )
        
        logger.info(f"Retrieved permission matrix for user {current_user.id}")
        return matrix
        
    except Exception as e:
        logger.error(f"Error retrieving permission matrix: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve permission matrix")

@router.get("/user-roles/{user_id}")
async def get_user_roles(
    user_id: str = Path(...),
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Get roles assigned to specific user."""
    try:
        user_role_ids = _user_roles.get(user_id, [])
        user_roles = []
        
        for role_id in user_role_ids:
            role_data = _roles_store.get(role_id)
            if role_data:
                user_roles.append(RoleResponse(**role_data))
        
        return JSONResponse({
            "status": "success",
            "user_id": user_id,
            "roles": [role.dict() for role in user_roles],
            "role_count": len(user_roles)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving user roles: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user roles")

@router.get("/permissions", response_model=List[Dict[str, Any]])
async def get_all_permissions(
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Get all available permissions with descriptions."""
    try:
        permission_info = []
        
        # Group permissions by category
        permission_categories = {
            "Pilot Management": [
                Permission.CREATE_PILOT, Permission.VIEW_PILOT, 
                Permission.UPDATE_PILOT, Permission.DELETE_PILOT
            ],
            "Analytics & ROI": [
                Permission.VIEW_ROI_METRICS, Permission.CREATE_ROI_METRICS
            ],
            "Executive Engagement": [
                Permission.VIEW_EXECUTIVE_ENGAGEMENT, 
                Permission.CREATE_EXECUTIVE_ENGAGEMENT,
                Permission.UPDATE_EXECUTIVE_ENGAGEMENT
            ],
            "Development": [
                Permission.CREATE_DEVELOPMENT_TASK, Permission.VIEW_DEVELOPMENT_TASK,
                Permission.EXECUTE_DEVELOPMENT_TASK
            ],
            "System Administration": [
                Permission.MANAGE_USERS, Permission.VIEW_SYSTEM_LOGS,
                Permission.CONFIGURE_SYSTEM
            ]
        }
        
        for category, perms in permission_categories.items():
            for perm in perms:
                permission_info.append({
                    "value": perm.value,
                    "name": perm.value.replace("_", " ").title(),
                    "category": category,
                    "description": f"Permission to {perm.value.replace('_', ' ').lower()}"
                })
        
        return permission_info
        
    except Exception as e:
        logger.error(f"Error retrieving permissions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve permissions")

@router.get("/hierarchy", response_model=List[RoleHierarchy])
async def get_role_hierarchy(
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Get role hierarchy structure."""
    try:
        # Build hierarchy tree (simplified - assuming flat structure for now)
        hierarchy = []
        
        for role_id, role_data in _roles_store.items():
            if not role_data.get("is_active", True):
                continue
                
            user_count = sum(
                1 for user_role_list in _user_roles.values() 
                if role_id in user_role_list
            )
            
            hierarchy.append(RoleHierarchy(
                role_id=role_id,
                role_name=role_data["name"],
                parent_id=role_data.get("parent_role"),
                children=[],  # TODO: Implement hierarchy children
                depth=0,
                user_count=user_count
            ))
        
        return hierarchy
        
    except Exception as e:
        logger.error(f"Error retrieving role hierarchy: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve role hierarchy")

# Background task functions

async def process_bulk_role_assignment(
    user_ids: List[str],
    role_ids: List[str],
    assigned_by: str
):
    """Process bulk role assignment in background."""
    try:
        success_count = 0
        failed_assignments = []
        
        for user_id in user_ids:
            try:
                _user_roles[user_id] = role_ids
                success_count += 1
            except Exception as e:
                failed_assignments.append({"user_id": user_id, "error": str(e)})
        
        logger.info(
            f"Bulk role assignment completed: {success_count} successful, "
            f"{len(failed_assignments)} failed by user {assigned_by}"
        )
        
        # TODO: Send notification/email about bulk assignment results
        
    except Exception as e:
        logger.error(f"Error in bulk role assignment background task: {str(e)}")

# Export router
__all__ = ["router"]