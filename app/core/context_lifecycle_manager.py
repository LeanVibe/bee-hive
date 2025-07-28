"""
Context Lifecycle Manager - Complete Context Version and Recovery System.

Provides comprehensive lifecycle management for contexts with:
- Context versioning and history tracking
- Recovery and restoration capabilities
- Lifecycle state management
- Audit trail and change tracking
- Rollback and migration support
- Data integrity and consistency checks
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, asdict
from enum import Enum

from sqlalchemy import select, and_, or_, func, update, delete, desc, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.context import Context, ContextType
from ..models.agent import Agent
from ..core.database import get_async_session
from ..core.redis import get_redis_client
from ..core.config import get_settings


logger = logging.getLogger(__name__)


class ContextLifecycleState(Enum):
    """Context lifecycle states."""
    DRAFT = "draft"
    ACTIVE = "active"
    CONSOLIDATING = "consolidating"
    CONSOLIDATED = "consolidated"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    DELETED = "deleted"
    RECOVERED = "recovered"


class VersionAction(Enum):
    """Types of version actions."""
    CREATE = "create"
    UPDATE = "update"
    CONSOLIDATE = "consolidate"
    RESTORE = "restore"
    MIGRATE = "migrate"
    ROLLBACK = "rollback"
    ARCHIVE = "archive"
    DELETE = "delete"


@dataclass
class ContextVersion:
    """Represents a version of a context."""
    version_id: str
    context_id: UUID
    version_number: int
    action: VersionAction
    content_hash: str
    content_snapshot: Dict[str, Any]
    metadata_snapshot: Dict[str, Any]
    created_at: datetime
    created_by: Optional[str]
    parent_version_id: Optional[str]
    changes_summary: str
    size_bytes: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert version to dictionary."""
        return {
            "version_id": self.version_id,
            "context_id": str(self.context_id),
            "version_number": self.version_number,
            "action": self.action.value,
            "content_hash": self.content_hash,
            "content_snapshot": self.content_snapshot,
            "metadata_snapshot": self.metadata_snapshot,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "parent_version_id": self.parent_version_id,
            "changes_summary": self.changes_summary,
            "size_bytes": self.size_bytes
        }


@dataclass
class LifecycleAuditEntry:
    """Audit entry for lifecycle events."""
    audit_id: str
    context_id: UUID
    agent_id: Optional[UUID]
    action: str
    previous_state: ContextLifecycleState
    new_state: ContextLifecycleState
    timestamp: datetime
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str]


@dataclass
class RecoveryPoint:
    """Recovery point for context restoration."""
    recovery_id: str
    context_id: UUID
    version_id: str
    created_at: datetime
    recovery_type: str
    metadata: Dict[str, Any]
    data_integrity_hash: str


class ContextLifecycleManager:
    """
    Comprehensive context lifecycle management system.
    
    Features:
    - Complete version history tracking
    - State-based lifecycle management
    - Recovery and rollback capabilities
    - Audit trail with detailed logging
    - Data integrity verification
    - Migration and backup support
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = get_redis_client()
        
        # Configuration
        self.max_versions_per_context = 50
        self.version_retention_days = 365
        self.recovery_point_interval_hours = 24
        self.audit_retention_days = 90
        
        # State tracking
        self.lifecycle_states: Dict[UUID, ContextLifecycleState] = {}
        self.version_cache: Dict[str, ContextVersion] = {}
        self.audit_cache: List[LifecycleAuditEntry] = []
        
        # Performance metrics
        self.operation_metrics = {
            "versions_created": 0,
            "restorations_performed": 0,
            "rollbacks_executed": 0,
            "integrity_checks": 0,
            "audit_entries": 0
        }
    
    async def create_version(
        self,
        context: Context,
        action: VersionAction,
        changes_summary: str = "",
        created_by: Optional[str] = None,
        parent_version_id: Optional[str] = None
    ) -> ContextVersion:
        """
        Create a new version of a context.
        
        Args:
            context: Context to version
            action: Action that triggered versioning
            changes_summary: Summary of changes made
            created_by: Who created the version
            parent_version_id: Parent version if this is a branch
            
        Returns:
            Created ContextVersion
        """
        try:
            # Generate version ID and get next version number
            version_id = str(uuid4())
            version_number = await self._get_next_version_number(context.id)
            
            # Create content and metadata snapshots
            content_snapshot = self._create_content_snapshot(context)
            metadata_snapshot = self._create_metadata_snapshot(context)
            
            # Calculate content hash for integrity
            content_hash = self._calculate_content_hash(content_snapshot)
            
            # Calculate size
            size_bytes = len(json.dumps(content_snapshot, default=str))
            
            # Create version object
            version = ContextVersion(
                version_id=version_id,
                context_id=context.id,
                version_number=version_number,
                action=action,
                content_hash=content_hash,
                content_snapshot=content_snapshot,
                metadata_snapshot=metadata_snapshot,
                created_at=datetime.utcnow(),
                created_by=created_by,
                parent_version_id=parent_version_id,
                changes_summary=changes_summary or f"Version created via {action.value}",
                size_bytes=size_bytes
            )
            
            # Store version in Redis
            await self._store_version(version)
            
            # Update context metadata with version info
            await self._update_context_version_metadata(context, version)
            
            # Create audit entry
            await self._create_audit_entry(
                context_id=context.id,
                agent_id=context.agent_id,
                action=f"version_created_{action.value}",
                previous_state=self._get_context_state(context),
                new_state=self._get_context_state(context),
                metadata={
                    "version_id": version_id,
                    "version_number": version_number,
                    "changes_summary": changes_summary
                },
                success=True
            )
            
            # Update metrics
            self.operation_metrics["versions_created"] += 1
            
            logger.info(f"Created version {version_number} for context {context.id}")
            return version
            
        except Exception as e:
            logger.error(f"Error creating version for context {context.id}: {e}")
            raise
    
    async def get_version_history(
        self,
        context_id: UUID,
        limit: int = 20
    ) -> List[ContextVersion]:
        """
        Get version history for a context.
        
        Args:
            context_id: Context ID to get history for
            limit: Maximum number of versions to return
            
        Returns:
            List of context versions in reverse chronological order
        """
        try:
            versions = []
            
            # Get version list from Redis
            version_list_key = f"context_versions:{context_id}"
            version_ids = await self.redis_client.lrange(version_list_key, 0, limit - 1)
            
            # Retrieve each version
            for version_id in version_ids:
                version = await self._get_version(version_id.decode())
                if version:
                    versions.append(version)
            
            # Sort by version number (descending)
            versions.sort(key=lambda v: v.version_number, reverse=True)
            
            return versions
            
        except Exception as e:
            logger.error(f"Error getting version history for context {context_id}: {e}")
            return []
    
    async def restore_context_version(
        self,
        context_id: UUID,
        version_id: str,
        create_backup: bool = True
    ) -> Context:
        """
        Restore a context to a specific version.
        
        Args:
            context_id: Context ID to restore
            version_id: Version ID to restore to
            create_backup: Whether to create backup before restore
            
        Returns:
            Restored context
        """
        try:
            # Get current context
            async with get_async_session() as session:
                current_context = await session.get(Context, context_id)
                if not current_context:
                    raise ValueError(f"Context {context_id} not found")
                
                # Create backup version if requested
                if create_backup:
                    await self.create_version(
                        current_context,
                        VersionAction.UPDATE,
                        f"Backup before restore to version {version_id}",
                        created_by="system"
                    )
                
                # Get target version
                target_version = await self._get_version(version_id)
                if not target_version:
                    raise ValueError(f"Version {version_id} not found")
                
                # Restore context from version snapshot
                restored_context = await self._restore_from_snapshot(
                    current_context,
                    target_version
                )
                
                # Update context state
                previous_state = self._get_context_state(current_context)
                self._set_context_state(context_id, ContextLifecycleState.RECOVERED)
                
                # Create audit entry
                await self._create_audit_entry(
                    context_id=context_id,
                    agent_id=current_context.agent_id,
                    action="context_restored",
                    previous_state=previous_state,
                    new_state=ContextLifecycleState.RECOVERED,
                    metadata={
                        "restored_from_version": version_id,
                        "target_version_number": target_version.version_number,
                        "backup_created": create_backup
                    },
                    success=True
                )
                
                # Create new version for the restoration
                await self.create_version(
                    restored_context,
                    VersionAction.RESTORE,
                    f"Restored from version {target_version.version_number}",
                    created_by="system"
                )
                
                await session.commit()
                
                # Update metrics
                self.operation_metrics["restorations_performed"] += 1
                
                logger.info(f"Restored context {context_id} to version {target_version.version_number}")
                return restored_context
                
        except Exception as e:
            logger.error(f"Error restoring context {context_id} to version {version_id}: {e}")
            raise
    
    async def rollback_to_previous_version(
        self,
        context_id: UUID,
        steps_back: int = 1
    ) -> Context:
        """
        Rollback context to previous version.
        
        Args:
            context_id: Context ID to rollback
            steps_back: Number of versions to roll back
            
        Returns:
            Rolled back context
        """
        try:
            # Get version history
            versions = await self.get_version_history(context_id, limit=steps_back + 5)
            
            if len(versions) <= steps_back:
                raise ValueError(f"Not enough versions to rollback {steps_back} steps")
            
            # Get target version (skip current version)
            target_version = versions[steps_back]
            
            # Restore to target version
            restored_context = await self.restore_context_version(
                context_id,
                target_version.version_id,
                create_backup=True
            )
            
            # Update audit with rollback info
            await self._create_audit_entry(
                context_id=context_id,
                agent_id=restored_context.agent_id,
                action="context_rollback",
                previous_state=ContextLifecycleState.ACTIVE,
                new_state=ContextLifecycleState.RECOVERED,
                metadata={
                    "steps_back": steps_back,
                    "target_version": target_version.version_number,
                    "rollback_reason": "manual_rollback"
                },
                success=True
            )
            
            self.operation_metrics["rollbacks_executed"] += 1
            
            logger.info(f"Rolled back context {context_id} {steps_back} versions")
            return restored_context
            
        except Exception as e:
            logger.error(f"Error rolling back context {context_id}: {e}")
            raise
    
    async def create_recovery_point(
        self,
        context_id: UUID,
        recovery_type: str = "scheduled",
        metadata: Optional[Dict[str, Any]] = None
    ) -> RecoveryPoint:
        """
        Create a recovery point for a context.
        
        Args:
            context_id: Context ID to create recovery point for
            recovery_type: Type of recovery point
            metadata: Additional metadata
            
        Returns:
            Created recovery point
        """
        try:
            # Get current context
            async with get_async_session() as session:
                context = await session.get(Context, context_id)
                if not context:
                    raise ValueError(f"Context {context_id} not found")
            
            # Create version for recovery point
            version = await self.create_version(
                context,
                VersionAction.CREATE,
                f"Recovery point ({recovery_type})",
                created_by="system"
            )
            
            # Create recovery point
            recovery_point = RecoveryPoint(
                recovery_id=str(uuid4()),
                context_id=context_id,
                version_id=version.version_id,
                created_at=datetime.utcnow(),
                recovery_type=recovery_type,
                metadata=metadata or {},
                data_integrity_hash=version.content_hash
            )
            
            # Store recovery point
            await self._store_recovery_point(recovery_point)
            
            logger.info(f"Created recovery point for context {context_id}")
            return recovery_point
            
        except Exception as e:
            logger.error(f"Error creating recovery point for context {context_id}: {e}")
            raise
    
    async def verify_data_integrity(
        self,
        context_id: UUID,
        version_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify data integrity for a context or version.
        
        Args:
            context_id: Context ID to verify
            version_id: Specific version to verify (current if None)
            
        Returns:
            Integrity verification results
        """
        try:
            result = {
                "context_id": str(context_id),
                "version_id": version_id,
                "integrity_valid": False,
                "checks_performed": [],
                "issues_found": [],
                "verification_time": datetime.utcnow().isoformat()
            }
            
            if version_id:
                # Verify specific version
                version = await self._get_version(version_id)
                if not version:
                    result["issues_found"].append(f"Version {version_id} not found")
                    return result
                
                # Check content hash
                current_hash = self._calculate_content_hash(version.content_snapshot)
                if current_hash != version.content_hash:
                    result["issues_found"].append("Content hash mismatch")
                else:
                    result["checks_performed"].append("content_hash_verification")
                
                # Verify snapshot structure
                if self._verify_snapshot_structure(version.content_snapshot):
                    result["checks_performed"].append("snapshot_structure_verification")
                else:
                    result["issues_found"].append("Invalid snapshot structure")
                    
            else:
                # Verify current context
                async with get_async_session() as session:
                    context = await session.get(Context, context_id)
                    if not context:
                        result["issues_found"].append("Context not found")
                        return result
                
                # Verify context data consistency
                if self._verify_context_consistency(context):
                    result["checks_performed"].append("context_consistency_verification")
                else:
                    result["issues_found"].append("Context data inconsistency")
                
                # Verify version metadata
                if await self._verify_version_metadata(context_id):
                    result["checks_performed"].append("version_metadata_verification")
                else:
                    result["issues_found"].append("Version metadata inconsistency")
            
            # Determine overall integrity status
            result["integrity_valid"] = len(result["issues_found"]) == 0
            
            # Update metrics
            self.operation_metrics["integrity_checks"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error verifying data integrity for context {context_id}: {e}")
            return {
                "context_id": str(context_id),
                "integrity_valid": False,
                "error": str(e)
            }
    
    async def get_lifecycle_audit_trail(
        self,
        context_id: UUID,
        limit: int = 50
    ) -> List[LifecycleAuditEntry]:
        """
        Get audit trail for a context's lifecycle.
        
        Args:
            context_id: Context ID to get audit trail for
            limit: Maximum number of entries to return
            
        Returns:
            List of audit entries
        """
        try:
            audit_entries = []
            
            # Get audit entries from Redis
            audit_key = f"context_audit:{context_id}"
            entries = await self.redis_client.lrange(audit_key, 0, limit - 1)
            
            for entry_json in entries:
                try:
                    entry_data = json.loads(entry_json.decode())
                    audit_entry = LifecycleAuditEntry(
                        audit_id=entry_data["audit_id"],
                        context_id=UUID(entry_data["context_id"]),
                        agent_id=UUID(entry_data["agent_id"]) if entry_data.get("agent_id") else None,
                        action=entry_data["action"],
                        previous_state=ContextLifecycleState(entry_data["previous_state"]),
                        new_state=ContextLifecycleState(entry_data["new_state"]),
                        timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                        metadata=entry_data["metadata"],
                        success=entry_data["success"],
                        error_message=entry_data.get("error_message")
                    )
                    audit_entries.append(audit_entry)
                except Exception as e:
                    logger.warning(f"Error parsing audit entry: {e}")
            
            return audit_entries
            
        except Exception as e:
            logger.error(f"Error getting audit trail for context {context_id}: {e}")
            return []
    
    async def cleanup_old_versions(
        self,
        retention_days: Optional[int] = None,
        max_versions_per_context: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Clean up old versions based on retention policies.
        
        Args:
            retention_days: Days to retain versions (use default if None)
            max_versions_per_context: Max versions per context (use default if None)
            
        Returns:
            Cleanup statistics
        """
        try:
            retention_days = retention_days or self.version_retention_days
            max_versions = max_versions_per_context or self.max_versions_per_context
            
            cleanup_stats = {
                "contexts_processed": 0,
                "versions_deleted": 0,
                "bytes_freed": 0,
                "errors": 0
            }
            
            # Get all contexts with versions
            async with get_async_session() as session:
                contexts_with_versions = await session.execute(
                    select(Context.id).where(
                        Context.context_metadata.op('?')('versions')
                    )
                )
                
                context_ids = [row[0] for row in contexts_with_versions.all()]
            
            # Process each context
            for context_id in context_ids:
                try:
                    versions = await self.get_version_history(context_id, limit=1000)
                    
                    # Identify versions to delete
                    cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                    versions_to_delete = []
                    
                    # Delete by age
                    for version in versions:
                        if version.created_at < cutoff_date:
                            versions_to_delete.append(version)
                    
                    # Delete excess versions (keep most recent)
                    if len(versions) > max_versions:
                        excess_versions = versions[max_versions:]
                        versions_to_delete.extend(excess_versions)
                    
                    # Remove duplicates
                    unique_versions_to_delete = list({v.version_id: v for v in versions_to_delete}.values())
                    
                    # Delete versions
                    for version in unique_versions_to_delete:
                        await self._delete_version(version)
                        cleanup_stats["versions_deleted"] += 1
                        cleanup_stats["bytes_freed"] += version.size_bytes
                    
                    cleanup_stats["contexts_processed"] += 1
                    
                except Exception as e:
                    logger.error(f"Error cleaning versions for context {context_id}: {e}")
                    cleanup_stats["errors"] += 1
            
            logger.info(
                f"Version cleanup completed: {cleanup_stats['versions_deleted']} versions deleted, "
                f"{cleanup_stats['bytes_freed']} bytes freed"
            )
            
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Error during version cleanup: {e}")
            return {"error": str(e)}
    
    async def get_lifecycle_statistics(self) -> Dict[str, Any]:
        """Get comprehensive lifecycle management statistics."""
        try:
            stats = {
                "operation_metrics": self.operation_metrics.copy(),
                "state_distribution": {},
                "version_statistics": {},
                "recovery_statistics": {},
                "audit_statistics": {}
            }
            
            # Calculate state distribution
            state_counts = defaultdict(int)
            for state in self.lifecycle_states.values():
                state_counts[state.value] += 1
            stats["state_distribution"] = dict(state_counts)
            
            # Version statistics
            total_versions = await self.redis_client.eval(
                "return #redis.call('keys', 'context_version:*')",
                0
            )
            stats["version_statistics"] = {
                "total_versions": total_versions,
                "cached_versions": len(self.version_cache)
            }
            
            # Recovery point statistics
            total_recovery_points = await self.redis_client.eval(
                "return #redis.call('keys', 'recovery_point:*')",
                0
            )
            stats["recovery_statistics"] = {
                "total_recovery_points": total_recovery_points
            }
            
            # Audit statistics
            stats["audit_statistics"] = {
                "cached_audit_entries": len(self.audit_cache),
                "audit_retention_days": self.audit_retention_days
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting lifecycle statistics: {e}")
            return {"error": str(e)}
    
    # Private Methods
    
    async def _get_next_version_number(self, context_id: UUID) -> int:
        """Get next version number for a context."""
        try:
            versions = await self.get_version_history(context_id, limit=1)
            if versions:
                return versions[0].version_number + 1
            return 1
        except Exception:
            return 1
    
    def _create_content_snapshot(self, context: Context) -> Dict[str, Any]:
        """Create content snapshot of context."""
        return {
            "id": str(context.id),
            "title": context.title,
            "content": context.content,
            "context_type": context.context_type.value,
            "importance_score": context.importance_score,
            "consolidation_summary": context.consolidation_summary,
            "is_consolidated": context.is_consolidated,
            "tags": context.tags,
            "related_context_ids": context.related_context_ids
        }
    
    def _create_metadata_snapshot(self, context: Context) -> Dict[str, Any]:
        """Create metadata snapshot of context."""
        return {
            "agent_id": str(context.agent_id) if context.agent_id else None,
            "session_id": str(context.session_id) if context.session_id else None,
            "parent_context_id": str(context.parent_context_id) if context.parent_context_id else None,
            "access_count": context.access_count,
            "relevance_decay": context.relevance_decay,
            "context_metadata": context.context_metadata,
            "created_at": context.created_at.isoformat() if context.created_at else None,
            "updated_at": context.updated_at.isoformat() if context.updated_at else None,
            "accessed_at": context.accessed_at.isoformat() if context.accessed_at else None,
            "consolidated_at": context.consolidated_at.isoformat() if context.consolidated_at else None
        }
    
    def _calculate_content_hash(self, content_snapshot: Dict[str, Any]) -> str:
        """Calculate hash of content snapshot for integrity checking."""
        content_string = json.dumps(content_snapshot, sort_keys=True, default=str)
        return hashlib.sha256(content_string.encode()).hexdigest()
    
    async def _store_version(self, version: ContextVersion) -> None:
        """Store version in Redis."""
        try:
            # Store version data
            version_key = f"context_version:{version.version_id}"
            await self.redis_client.setex(
                version_key,
                self.version_retention_days * 86400,  # Convert days to seconds
                json.dumps(version.to_dict(), default=str)
            )
            
            # Add to version list for context
            version_list_key = f"context_versions:{version.context_id}"
            await self.redis_client.lpush(version_list_key, version.version_id)
            
            # Cache in memory
            self.version_cache[version.version_id] = version
            
        except Exception as e:
            logger.error(f"Error storing version {version.version_id}: {e}")
            raise
    
    async def _get_version(self, version_id: str) -> Optional[ContextVersion]:
        """Get version from cache or Redis."""
        try:
            # Check memory cache first
            if version_id in self.version_cache:
                return self.version_cache[version_id]
            
            # Get from Redis
            version_key = f"context_version:{version_id}"
            version_data = await self.redis_client.get(version_key)
            
            if version_data:
                version_dict = json.loads(version_data.decode())
                version = ContextVersion(
                    version_id=version_dict["version_id"],
                    context_id=UUID(version_dict["context_id"]),
                    version_number=version_dict["version_number"],
                    action=VersionAction(version_dict["action"]),
                    content_hash=version_dict["content_hash"],
                    content_snapshot=version_dict["content_snapshot"],
                    metadata_snapshot=version_dict["metadata_snapshot"],
                    created_at=datetime.fromisoformat(version_dict["created_at"]),
                    created_by=version_dict.get("created_by"),
                    parent_version_id=version_dict.get("parent_version_id"),
                    changes_summary=version_dict["changes_summary"],
                    size_bytes=version_dict["size_bytes"]
                )
                
                # Cache in memory
                self.version_cache[version_id] = version
                return version
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting version {version_id}: {e}")
            return None
    
    async def _delete_version(self, version: ContextVersion) -> None:
        """Delete version from storage."""
        try:
            # Remove from Redis
            version_key = f"context_version:{version.version_id}"
            await self.redis_client.delete(version_key)
            
            # Remove from version list
            version_list_key = f"context_versions:{version.context_id}"
            await self.redis_client.lrem(version_list_key, 0, version.version_id)
            
            # Remove from memory cache
            self.version_cache.pop(version.version_id, None)
            
        except Exception as e:
            logger.error(f"Error deleting version {version.version_id}: {e}")
    
    async def _restore_from_snapshot(
        self,
        context: Context,
        version: ContextVersion
    ) -> Context:
        """Restore context from version snapshot."""
        try:
            # Restore content fields
            content_snapshot = version.content_snapshot
            context.title = content_snapshot.get("title", context.title)
            context.content = content_snapshot.get("content", context.content)
            context.importance_score = content_snapshot.get("importance_score", context.importance_score)
            context.consolidation_summary = content_snapshot.get("consolidation_summary")
            context.is_consolidated = content_snapshot.get("is_consolidated", context.is_consolidated)
            context.tags = content_snapshot.get("tags", context.tags)
            context.related_context_ids = content_snapshot.get("related_context_ids", context.related_context_ids)
            
            # Restore metadata fields
            metadata_snapshot = version.metadata_snapshot
            context.access_count = metadata_snapshot.get("access_count", context.access_count)
            context.relevance_decay = metadata_snapshot.get("relevance_decay", context.relevance_decay)
            
            # Update context metadata to include restoration info
            if not context.context_metadata:
                context.context_metadata = {}
            
            context.context_metadata["restored_from_version"] = version.version_id
            context.context_metadata["restored_at"] = datetime.utcnow().isoformat()
            context.context_metadata["restoration_summary"] = version.changes_summary
            
            return context
            
        except Exception as e:
            logger.error(f"Error restoring context from snapshot: {e}")
            raise
    
    async def _update_context_version_metadata(
        self,
        context: Context,
        version: ContextVersion
    ) -> None:
        """Update context metadata with version information."""
        try:
            if not context.context_metadata:
                context.context_metadata = {}
            
            # Initialize versions list if not exists
            if "versions" not in context.context_metadata:
                context.context_metadata["versions"] = []
            
            # Add version info
            version_info = {
                "version_id": version.version_id,
                "version_number": version.version_number,
                "action": version.action.value,
                "created_at": version.created_at.isoformat(),
                "changes_summary": version.changes_summary,
                "size_bytes": version.size_bytes
            }
            
            context.context_metadata["versions"].append(version_info)
            
            # Keep only recent versions in metadata
            if len(context.context_metadata["versions"]) > 10:
                context.context_metadata["versions"] = context.context_metadata["versions"][-10:]
            
            # Update version metadata
            context.context_metadata["current_version"] = version.version_id
            context.context_metadata["total_versions"] = version.version_number
            context.context_metadata["last_versioned_at"] = version.created_at.isoformat()
            
        except Exception as e:
            logger.error(f"Error updating context version metadata: {e}")
    
    def _get_context_state(self, context: Context) -> ContextLifecycleState:
        """Get current lifecycle state of context."""
        if context.id in self.lifecycle_states:
            return self.lifecycle_states[context.id]
        
        # Infer state from context properties
        if context.context_metadata and context.context_metadata.get("archived"):
            return ContextLifecycleState.ARCHIVED
        elif context.is_consolidated == "true":
            return ContextLifecycleState.CONSOLIDATED
        else:
            return ContextLifecycleState.ACTIVE
    
    def _set_context_state(self, context_id: UUID, state: ContextLifecycleState) -> None:
        """Set lifecycle state for context."""
        self.lifecycle_states[context_id] = state
    
    async def _create_audit_entry(
        self,
        context_id: UUID,
        agent_id: Optional[UUID],
        action: str,
        previous_state: ContextLifecycleState,
        new_state: ContextLifecycleState,
        metadata: Dict[str, Any],
        success: bool,
        error_message: Optional[str] = None
    ) -> None:
        """Create audit entry for lifecycle event."""
        try:
            audit_entry = LifecycleAuditEntry(
                audit_id=str(uuid4()),
                context_id=context_id,
                agent_id=agent_id,
                action=action,
                previous_state=previous_state,
                new_state=new_state,
                timestamp=datetime.utcnow(),
                metadata=metadata,
                success=success,
                error_message=error_message
            )
            
            # Store in Redis
            audit_key = f"context_audit:{context_id}"
            audit_data = {
                "audit_id": audit_entry.audit_id,
                "context_id": str(audit_entry.context_id),
                "agent_id": str(audit_entry.agent_id) if audit_entry.agent_id else None,
                "action": audit_entry.action,
                "previous_state": audit_entry.previous_state.value,
                "new_state": audit_entry.new_state.value,
                "timestamp": audit_entry.timestamp.isoformat(),
                "metadata": audit_entry.metadata,
                "success": audit_entry.success,
                "error_message": audit_entry.error_message
            }
            
            await self.redis_client.lpush(audit_key, json.dumps(audit_data, default=str))
            
            # Set expiration for audit data
            await self.redis_client.expire(audit_key, self.audit_retention_days * 86400)
            
            # Cache in memory (limited)
            self.audit_cache.append(audit_entry)
            if len(self.audit_cache) > 1000:
                self.audit_cache = self.audit_cache[-500:]  # Keep last 500
            
            self.operation_metrics["audit_entries"] += 1
            
        except Exception as e:
            logger.error(f"Error creating audit entry: {e}")
    
    async def _store_recovery_point(self, recovery_point: RecoveryPoint) -> None:
        """Store recovery point in Redis."""
        try:
            recovery_key = f"recovery_point:{recovery_point.recovery_id}"
            recovery_data = {
                "recovery_id": recovery_point.recovery_id,
                "context_id": str(recovery_point.context_id),
                "version_id": recovery_point.version_id,
                "created_at": recovery_point.created_at.isoformat(),
                "recovery_type": recovery_point.recovery_type,
                "metadata": recovery_point.metadata,
                "data_integrity_hash": recovery_point.data_integrity_hash
            }
            
            await self.redis_client.setex(
                recovery_key,
                self.version_retention_days * 86400,
                json.dumps(recovery_data, default=str)
            )
            
            # Add to recovery list for context
            recovery_list_key = f"context_recovery_points:{recovery_point.context_id}"
            await self.redis_client.lpush(recovery_list_key, recovery_point.recovery_id)
            
        except Exception as e:
            logger.error(f"Error storing recovery point: {e}")
    
    def _verify_snapshot_structure(self, snapshot: Dict[str, Any]) -> bool:
        """Verify that snapshot has required structure."""
        required_fields = ["id", "title", "content", "context_type"]
        return all(field in snapshot for field in required_fields)
    
    def _verify_context_consistency(self, context: Context) -> bool:
        """Verify context data consistency."""
        try:
            # Check required fields
            if not context.title or not context.content:
                return False
            
            # Check data types
            if not isinstance(context.importance_score, (int, float)):
                return False
            
            # Check enum values
            if context.context_type not in ContextType:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def _verify_version_metadata(self, context_id: UUID) -> bool:
        """Verify version metadata consistency."""
        try:
            # Get context and check version metadata
            async with get_async_session() as session:
                context = await session.get(Context, context_id)
                if not context or not context.context_metadata:
                    return True  # No metadata to verify
                
                # Check if version information is consistent
                versions_meta = context.context_metadata.get("versions", [])
                if not versions_meta:
                    return True
                
                # Verify each version exists
                for version_info in versions_meta[-5:]:  # Check last 5 versions
                    version_id = version_info.get("version_id")
                    if version_id:
                        version = await self._get_version(version_id)
                        if not version:
                            return False
                
                return True
                
        except Exception as e:
            logger.error(f"Error verifying version metadata: {e}")
            return False


# Global instance for application use
_lifecycle_manager: Optional[ContextLifecycleManager] = None


def get_context_lifecycle_manager() -> ContextLifecycleManager:
    """
    Get singleton context lifecycle manager instance.
    
    Returns:
        ContextLifecycleManager instance
    """
    global _lifecycle_manager
    
    if _lifecycle_manager is None:
        _lifecycle_manager = ContextLifecycleManager()
    
    return _lifecycle_manager