"""
Access Control Manager for Context Engine.

Implements RBAC (Role-Based Access Control) for cross-agent context sharing
with privacy-preserving boundaries and security audit capabilities.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
import logging

from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.context import Context
from ..models.agent import Agent


logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Context access levels for RBAC."""
    PRIVATE = "PRIVATE"  # Only the creating agent
    AGENT_SHARED = "AGENT_SHARED"  # Specific agents
    SESSION_SHARED = "SESSION_SHARED"  # All agents in session
    PUBLIC = "PUBLIC"  # All agents in the system


class Permission(Enum):
    """Context permissions."""
    READ = "READ"
    WRITE = "WRITE"
    SHARE = "SHARE"
    DELETE = "DELETE"


class AccessAuditEvent:
    """Represents an access audit event."""
    
    def __init__(
        self,
        context_id: uuid.UUID,
        requesting_agent_id: uuid.UUID,
        action: str,
        granted: bool,
        reason: str,
        timestamp: Optional[datetime] = None
    ):
        self.context_id = context_id
        self.requesting_agent_id = requesting_agent_id
        self.action = action
        self.granted = granted
        self.reason = reason
        self.timestamp = timestamp or datetime.utcnow()


class AccessControlManager:
    """
    Manages access control for Context Engine with RBAC capabilities.
    
    Features:
    - Role-based access control for contexts
    - Privacy-preserving cross-agent sharing
    - Security audit logging
    - Permission validation
    - Access pattern analysis
    """
    
    def __init__(self, db_session: AsyncSession):
        """
        Initialize access control manager.
        
        Args:
            db_session: Database session for access control operations
        """
        self.db = db_session
        self.audit_events: List[AccessAuditEvent] = []
        
        # Cache for permission lookups
        self._permission_cache: Dict[str, Tuple[bool, datetime]] = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def check_context_access(
        self,
        context_id: uuid.UUID,
        requesting_agent_id: uuid.UUID,
        permission: Permission,
        session_id: Optional[uuid.UUID] = None
    ) -> bool:
        """
        Check if an agent has access to a specific context.
        
        Args:
            context_id: Context to check access for
            requesting_agent_id: Agent requesting access
            permission: Type of permission required
            session_id: Current session context (optional)
            
        Returns:
            True if access is granted, False otherwise
        """
        try:
            # Check cache first
            cache_key = f"{context_id}:{requesting_agent_id}:{permission.value}:{session_id}"
            cached_result = self._get_cached_permission(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Get context from database
            context = await self.db.get(Context, context_id)
            if not context:
                self._audit_access(context_id, requesting_agent_id, permission.value, False, "Context not found")
                return False
            
            # Check access based on context access level
            access_granted = False
            reason = ""
            
            # Get access level from metadata or default to PRIVATE
            access_level_str = getattr(context, 'access_level', None) or context.context_metadata.get('access_level', 'PRIVATE')
            
            try:
                access_level = AccessLevel(access_level_str)
            except ValueError:
                access_level = AccessLevel.PRIVATE
            
            if access_level == AccessLevel.PRIVATE:
                # Only the creating agent can access
                access_granted = context.agent_id == requesting_agent_id
                reason = "Private context - owner only" if access_granted else "Access denied - private context"
                
            elif access_level == AccessLevel.AGENT_SHARED:
                # Check if agent is in shared list
                shared_agents = context.context_metadata.get('shared_agents', [])
                access_granted = (
                    context.agent_id == requesting_agent_id or
                    str(requesting_agent_id) in shared_agents
                )
                reason = "Agent in shared list" if access_granted else "Agent not in shared list"
                
            elif access_level == AccessLevel.SESSION_SHARED:
                # Check if agents are in the same session
                if session_id and context.session_id:
                    access_granted = context.session_id == session_id
                    reason = "Same session access" if access_granted else "Different session"
                else:
                    # Fall back to agent check
                    access_granted = context.agent_id == requesting_agent_id
                    reason = "No session context - owner only"
                    
            elif access_level == AccessLevel.PUBLIC:
                # Check importance threshold for public access
                min_importance = 0.7  # High-importance contexts only for public access
                access_granted = context.importance_score >= min_importance
                reason = f"Public context with importance {context.importance_score}" if access_granted else f"Public context below importance threshold ({context.importance_score} < {min_importance})"
            
            # Additional permission checks
            if access_granted and permission in [Permission.WRITE, Permission.DELETE]:
                # Write/delete permissions require ownership
                access_granted = context.agent_id == requesting_agent_id
                if not access_granted:
                    reason = f"{permission.value} permission requires ownership"
            
            # Cache the result
            self._cache_permission(cache_key, access_granted)
            
            # Audit the access attempt
            self._audit_access(context_id, requesting_agent_id, permission.value, access_granted, reason)
            
            return access_granted
            
        except Exception as e:
            logger.error(f"Error checking context access: {e}")
            self._audit_access(context_id, requesting_agent_id, permission.value, False, f"Error: {str(e)}")
            return False
    
    async def filter_contexts_by_access(
        self,
        contexts: List[Context],
        requesting_agent_id: uuid.UUID,
        permission: Permission = Permission.READ,
        session_id: Optional[uuid.UUID] = None
    ) -> List[Context]:
        """
        Filter a list of contexts based on access permissions.
        
        Args:
            contexts: List of contexts to filter
            requesting_agent_id: Agent requesting access
            permission: Permission level required
            session_id: Current session context
            
        Returns:
            Filtered list of contexts the agent can access
        """
        accessible_contexts = []
        
        for context in contexts:
            has_access = await self.check_context_access(
                context_id=context.id,
                requesting_agent_id=requesting_agent_id,
                permission=permission,
                session_id=session_id
            )
            
            if has_access:
                accessible_contexts.append(context)
        
        return accessible_contexts
    
    async def share_context(
        self,
        context_id: uuid.UUID,
        owner_agent_id: uuid.UUID,
        target_agent_ids: List[uuid.UUID],
        access_level: AccessLevel
    ) -> bool:
        """
        Share a context with specific agents.
        
        Args:
            context_id: Context to share
            owner_agent_id: Agent that owns the context
            target_agent_ids: Agents to share with
            access_level: Level of access to grant
            
        Returns:
            True if sharing was successful
        """
        try:
            # Verify ownership
            has_share_permission = await self.check_context_access(
                context_id=context_id,
                requesting_agent_id=owner_agent_id,
                permission=Permission.SHARE
            )
            
            if not has_share_permission:
                logger.warning(f"Agent {owner_agent_id} cannot share context {context_id}")
                return False
            
            # Get and update context
            context = await self.db.get(Context, context_id)
            if not context:
                return False
            
            # Update access level and shared agents
            context.context_metadata = context.context_metadata or {}
            context.context_metadata['access_level'] = access_level.value
            
            if access_level == AccessLevel.AGENT_SHARED:
                current_shared = context.context_metadata.get('shared_agents', [])
                new_shared = list(set(current_shared + [str(agent_id) for agent_id in target_agent_ids]))
                context.context_metadata['shared_agents'] = new_shared
            
            # Update access level attribute if it exists
            if hasattr(context, 'access_level'):
                context.access_level = access_level.value
            
            await self.db.commit()
            
            # Clear cache for affected agents
            self._clear_cache_for_context(context_id)
            
            # Audit the sharing action
            for target_agent_id in target_agent_ids:
                self._audit_access(
                    context_id, target_agent_id, "SHARE_GRANTED", True,
                    f"Shared by {owner_agent_id} with access level {access_level.value}"
                )
            
            logger.info(f"Context {context_id} shared with {len(target_agent_ids)} agents")
            return True
            
        except Exception as e:
            logger.error(f"Error sharing context: {e}")
            await self.db.rollback()
            return False
    
    async def revoke_context_access(
        self,
        context_id: uuid.UUID,
        owner_agent_id: uuid.UUID,
        target_agent_ids: Optional[List[uuid.UUID]] = None
    ) -> bool:
        """
        Revoke access to a context.
        
        Args:
            context_id: Context to revoke access for
            owner_agent_id: Agent that owns the context
            target_agent_ids: Specific agents to revoke (None for all)
            
        Returns:
            True if revocation was successful
        """
        try:
            # Verify ownership
            has_permission = await self.check_context_access(
                context_id=context_id,
                requesting_agent_id=owner_agent_id,
                permission=Permission.SHARE
            )
            
            if not has_permission:
                return False
            
            context = await self.db.get(Context, context_id)
            if not context:
                return False
            
            if target_agent_ids is None:
                # Revoke all access - make private
                context.context_metadata = context.context_metadata or {}
                context.context_metadata['access_level'] = AccessLevel.PRIVATE.value
                context.context_metadata.pop('shared_agents', None)
                
                if hasattr(context, 'access_level'):
                    context.access_level = AccessLevel.PRIVATE.value
                
                audit_message = "All access revoked - context made private"
            else:
                # Revoke specific agents
                current_shared = context.context_metadata.get('shared_agents', [])
                remaining_shared = [
                    agent_id for agent_id in current_shared 
                    if agent_id not in [str(aid) for aid in target_agent_ids]
                ]
                
                if remaining_shared:
                    context.context_metadata['shared_agents'] = remaining_shared
                else:
                    # No agents left, make private
                    context.context_metadata['access_level'] = AccessLevel.PRIVATE.value
                    context.context_metadata.pop('shared_agents', None)
                    
                    if hasattr(context, 'access_level'):
                        context.access_level = AccessLevel.PRIVATE.value
                
                audit_message = f"Access revoked for {len(target_agent_ids)} agents"
            
            await self.db.commit()
            
            # Clear cache
            self._clear_cache_for_context(context_id)
            
            # Audit revocation
            revoked_agents = target_agent_ids or []
            for agent_id in revoked_agents:
                self._audit_access(
                    context_id, agent_id, "ACCESS_REVOKED", True, audit_message
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error revoking context access: {e}")
            await self.db.rollback()
            return False
    
    async def get_access_patterns(
        self,
        agent_id: Optional[uuid.UUID] = None,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze access patterns for security monitoring.
        
        Args:
            agent_id: Specific agent to analyze (optional)
            days_back: Number of days to analyze
            
        Returns:
            Access pattern analysis
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days_back)
        
        # Filter audit events
        recent_events = [
            event for event in self.audit_events
            if event.timestamp >= cutoff_time and
            (agent_id is None or event.requesting_agent_id == agent_id)
        ]
        
        # Analyze patterns
        total_attempts = len(recent_events)
        granted_attempts = sum(1 for event in recent_events if event.granted)
        denied_attempts = total_attempts - granted_attempts
        
        # Access by action type
        action_stats = {}
        for event in recent_events:
            if event.action not in action_stats:
                action_stats[event.action] = {'total': 0, 'granted': 0, 'denied': 0}
            
            action_stats[event.action]['total'] += 1
            if event.granted:
                action_stats[event.action]['granted'] += 1
            else:
                action_stats[event.action]['denied'] += 1
        
        # Suspicious patterns
        suspicious_patterns = []
        
        # High denial rate
        if total_attempts > 10 and (denied_attempts / total_attempts) > 0.3:
            suspicious_patterns.append(f"High denial rate: {denied_attempts}/{total_attempts}")
        
        # Multiple failed access attempts to same context
        context_failures = {}
        for event in recent_events:
            if not event.granted:
                context_failures[event.context_id] = context_failures.get(event.context_id, 0) + 1
        
        for context_id, failures in context_failures.items():
            if failures > 5:
                suspicious_patterns.append(f"Multiple failures on context {context_id}: {failures} attempts")
        
        return {
            "analysis_period_days": days_back,
            "total_attempts": total_attempts,
            "granted_attempts": granted_attempts,
            "denied_attempts": denied_attempts,
            "success_rate": granted_attempts / max(1, total_attempts),
            "action_breakdown": action_stats,
            "suspicious_patterns": suspicious_patterns,
            "agent_id": str(agent_id) if agent_id else "all_agents"
        }
    
    async def get_security_audit_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive security audit report.
        
        Returns:
            Security audit report with recommendations
        """
        # Get recent access patterns
        recent_patterns = await self.get_access_patterns(days_back=30)
        
        # Query database for context access statistics
        public_contexts_count = await self._count_contexts_by_access_level(AccessLevel.PUBLIC)
        shared_contexts_count = await self._count_contexts_by_access_level(AccessLevel.AGENT_SHARED)
        private_contexts_count = await self._count_contexts_by_access_level(AccessLevel.PRIVATE)
        
        # Security recommendations
        recommendations = []
        
        if public_contexts_count > private_contexts_count * 0.1:
            recommendations.append(
                "High ratio of public contexts detected. Review public context criteria."
            )
        
        if recent_patterns["success_rate"] < 0.7:
            recommendations.append(
                "Low access success rate detected. Review access control policies."
            )
        
        if len(recent_patterns["suspicious_patterns"]) > 0:
            recommendations.append(
                "Suspicious access patterns detected. Investigate failed access attempts."
            )
        
        return {
            "report_generated_at": datetime.utcnow().isoformat(),
            "context_distribution": {
                "private": private_contexts_count,
                "agent_shared": shared_contexts_count,
                "public": public_contexts_count
            },
            "recent_access_patterns": recent_patterns,
            "security_recommendations": recommendations,
            "audit_events_count": len(self.audit_events)
        }
    
    def _audit_access(
        self,
        context_id: uuid.UUID,
        requesting_agent_id: uuid.UUID,
        action: str,
        granted: bool,
        reason: str
    ) -> None:
        """Record access audit event."""
        event = AccessAuditEvent(
            context_id=context_id,
            requesting_agent_id=requesting_agent_id,
            action=action,
            granted=granted,
            reason=reason
        )
        
        self.audit_events.append(event)
        
        # Keep only recent events to prevent memory issues
        if len(self.audit_events) > 10000:
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            self.audit_events = [
                event for event in self.audit_events
                if event.timestamp >= cutoff_time
            ]
        
        logger.debug(f"Access audit: {action} on {context_id} by {requesting_agent_id}: {'GRANTED' if granted else 'DENIED'} - {reason}")
    
    def _get_cached_permission(self, cache_key: str) -> Optional[bool]:
        """Get cached permission result if not expired."""
        if cache_key in self._permission_cache:
            result, timestamp = self._permission_cache[cache_key]
            if datetime.utcnow() - timestamp < timedelta(seconds=self._cache_ttl):
                return result
            else:
                del self._permission_cache[cache_key]
        return None
    
    def _cache_permission(self, cache_key: str, result: bool) -> None:
        """Cache permission result."""
        self._permission_cache[cache_key] = (result, datetime.utcnow())
        
        # Simple cache cleanup
        if len(self._permission_cache) > 1000:
            # Remove oldest 20% of entries
            sorted_keys = sorted(
                self._permission_cache.keys(),
                key=lambda k: self._permission_cache[k][1]
            )
            for key in sorted_keys[:200]:
                del self._permission_cache[key]
    
    def _clear_cache_for_context(self, context_id: uuid.UUID) -> None:
        """Clear cached permissions for a specific context."""
        keys_to_remove = [
            key for key in self._permission_cache.keys()
            if key.startswith(str(context_id))
        ]
        for key in keys_to_remove:
            del self._permission_cache[key]
    
    async def _count_contexts_by_access_level(self, access_level: AccessLevel) -> int:
        """Count contexts by access level."""
        try:
            # Check both the access_level column and metadata
            result = await self.db.execute(
                select(func.count(Context.id)).where(
                    or_(
                        getattr(Context, 'access_level', None) == access_level.value,
                        Context.context_metadata.op('->>')('access_level') == access_level.value
                    )
                )
            )
            return result.scalar() or 0
        except Exception as e:
            logger.error(f"Error counting contexts by access level: {e}")
            return 0


# Singleton instance for application use
_access_control_manager: Optional[AccessControlManager] = None


async def get_access_control_manager(db_session: AsyncSession) -> AccessControlManager:
    """
    Get access control manager instance.
    
    Args:
        db_session: Database session
        
    Returns:
        AccessControlManager instance
    """
    return AccessControlManager(db_session)