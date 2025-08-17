"""
Backward Compatibility Adapters for Unified Manager Migration

This module provides backward compatibility for the Epic 1 Phase 1.2 consolidation
that merged 338 core files into 7 unified managers. These adapters ensure zero
breaking changes during the transition period.

Usage:
    The adapters automatically proxy old imports to new unified managers,
    maintaining the exact same interface while leveraging the new architecture.
    
Migration Strategy:
    1. Phase 1: Install compatibility layer (this file)
    2. Phase 2: Update imports gradually across codebase  
    3. Phase 3: Remove compatibility layer after validation
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
import structlog

# Import all unified managers
from .agent_manager import AgentManager
from .context_manager_unified import ContextManagerUnified  
from .resource_manager import ResourceManager
from .communication_manager import CommunicationManager
from .security_manager import SecurityManager
from .workflow_manager import WorkflowManager
from .storage_manager import StorageManager

logger = structlog.get_logger()

# Global instances of unified managers for backward compatibility
_agent_manager: Optional[AgentManager] = None
_context_manager: Optional[ContextManagerUnified] = None
_resource_manager: Optional[ResourceManager] = None
_communication_manager: Optional[CommunicationManager] = None
_security_manager: Optional[SecurityManager] = None
_workflow_manager: Optional[WorkflowManager] = None
_storage_manager: Optional[StorageManager] = None


async def _get_agent_manager() -> AgentManager:
    """Get or create agent manager instance."""
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = AgentManager()
        await _agent_manager.initialize()
    return _agent_manager


async def _get_context_manager() -> ContextManagerUnified:
    """Get or create context manager instance."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManagerUnified()
        await _context_manager.initialize()
    return _context_manager


async def _get_resource_manager() -> ResourceManager:
    """Get or create resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
        await _resource_manager.initialize()
    return _resource_manager


async def _get_communication_manager() -> CommunicationManager:
    """Get or create communication manager instance."""
    global _communication_manager
    if _communication_manager is None:
        _communication_manager = CommunicationManager()
        await _communication_manager.initialize()
    return _communication_manager


async def _get_security_manager() -> SecurityManager:
    """Get or create security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
        await _security_manager.initialize()
    return _security_manager


async def _get_workflow_manager() -> WorkflowManager:
    """Get or create workflow manager instance."""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = WorkflowManager()
        await _workflow_manager.initialize()
    return _workflow_manager


async def _get_storage_manager() -> StorageManager:
    """Get or create storage manager instance."""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
        await _storage_manager.initialize()
    return _storage_manager


# =============================================================================
# AGENT-RELATED COMPATIBILITY ADAPTERS
# =============================================================================

class AgentSpawnerAdapter:
    """Compatibility adapter for agent_spawner.py -> AgentManager."""
    
    def __init__(self):
        self._manager = None
    
    async def _get_manager(self):
        if self._manager is None:
            self._manager = await _get_agent_manager()
        return self._manager
    
    async def spawn_agent(self, agent_spec: Dict[str, Any], **kwargs):
        """Spawn agent using new unified manager."""
        manager = await self._get_manager()
        return await manager.spawn_agent(agent_spec, **kwargs)
    
    async def stop_agent(self, agent_id: str, **kwargs):
        """Stop agent using new unified manager."""
        manager = await self._get_manager()
        return await manager.stop_agent(agent_id, **kwargs)
    
    async def get_agent_status(self, agent_id: str):
        """Get agent status using new unified manager."""
        manager = await self._get_manager()
        return await manager.get_agent_status(agent_id)


class AgentRegistryAdapter:
    """Compatibility adapter for agent_registry.py -> AgentManager."""
    
    def __init__(self):
        self._manager = None
    
    async def _get_manager(self):
        if self._manager is None:
            self._manager = await _get_agent_manager()
        return self._manager
    
    async def register_agent(self, agent_id: str, agent_info: Dict[str, Any]):
        """Register agent using new unified manager."""
        manager = await self._get_manager()
        return await manager.register_agent(agent_id, agent_info)
    
    async def unregister_agent(self, agent_id: str):
        """Unregister agent using new unified manager."""
        manager = await self._get_manager()
        return await manager.unregister_agent(agent_id)
    
    async def get_agent_info(self, agent_id: str):
        """Get agent info using new unified manager."""
        manager = await self._get_manager()
        return await manager.get_agent_info(agent_id)
    
    async def list_agents(self, **filters):
        """List agents using new unified manager."""
        manager = await self._get_manager()
        return await manager.list_agents(**filters)


class AgentLifecycleManagerAdapter:
    """Compatibility adapter for agent_lifecycle_manager.py -> AgentManager."""
    
    def __init__(self):
        self._manager = None
    
    async def _get_manager(self):
        if self._manager is None:
            self._manager = await _get_agent_manager()
        return self._manager
    
    async def start_lifecycle(self, agent_id: str):
        """Start agent lifecycle using new unified manager."""
        manager = await self._get_manager()
        return await manager.start_agent_lifecycle(agent_id)
    
    async def stop_lifecycle(self, agent_id: str):
        """Stop agent lifecycle using new unified manager."""
        manager = await self._get_manager()
        return await manager.stop_agent_lifecycle(agent_id)


# =============================================================================
# COMMUNICATION-RELATED COMPATIBILITY ADAPTERS
# =============================================================================

class MessagingServiceAdapter:
    """Compatibility adapter for messaging_service.py -> CommunicationManager."""
    
    def __init__(self):
        self._manager = None
    
    async def _get_manager(self):
        if self._manager is None:
            self._manager = await _get_communication_manager()
        return self._manager
    
    async def send_message(self, message: Dict[str, Any], **kwargs):
        """Send message using new unified manager."""
        manager = await self._get_manager()
        return await manager.send_message(message, **kwargs)
    
    async def receive_message(self, agent_id: str, **kwargs):
        """Receive message using new unified manager."""
        manager = await self._get_manager()
        return await manager.receive_message(agent_id, **kwargs)
    
    async def broadcast_message(self, message: Dict[str, Any], **kwargs):
        """Broadcast message using new unified manager."""
        manager = await self._get_manager()
        return await manager.broadcast_message(message, **kwargs)


class RedisPubSubManagerAdapter:
    """Compatibility adapter for redis_pubsub_manager.py -> CommunicationManager."""
    
    def __init__(self):
        self._manager = None
    
    async def _get_manager(self):
        if self._manager is None:
            self._manager = await _get_communication_manager()
        return self._manager
    
    async def publish(self, channel: str, message: Any):
        """Publish to Redis using new unified manager."""
        manager = await self._get_manager()
        return await manager.publish_to_channel(channel, message)
    
    async def subscribe(self, channels: List[str], callback: Callable):
        """Subscribe to Redis channels using new unified manager."""
        manager = await self._get_manager()
        return await manager.subscribe_to_channels(channels, callback)


# =============================================================================
# CONTEXT-RELATED COMPATIBILITY ADAPTERS
# =============================================================================

class ContextCompressionAdapter:
    """Compatibility adapter for context_compression.py -> ContextManagerUnified."""
    
    def __init__(self):
        self._manager = None
    
    async def _get_manager(self):
        if self._manager is None:
            self._manager = await _get_context_manager()
        return self._manager
    
    async def compress_context(self, context_data: Dict[str, Any], **kwargs):
        """Compress context using new unified manager."""
        manager = await self._get_manager()
        return await manager.compress_context(context_data, **kwargs)
    
    async def decompress_context(self, compressed_data: str, **kwargs):
        """Decompress context using new unified manager."""
        manager = await self._get_manager()
        return await manager.decompress_context(compressed_data, **kwargs)


class ContextLifecycleManagerAdapter:
    """Compatibility adapter for context_lifecycle_manager.py -> ContextManagerUnified."""
    
    def __init__(self):
        self._manager = None
    
    async def _get_manager(self):
        if self._manager is None:
            self._manager = await _get_context_manager()
        return self._manager
    
    async def create_context(self, context_spec: Dict[str, Any]):
        """Create context using new unified manager."""
        manager = await self._get_manager()
        return await manager.create_context(context_spec)
    
    async def destroy_context(self, context_id: str):
        """Destroy context using new unified manager."""
        manager = await self._get_manager()
        return await manager.destroy_context(context_id)


# =============================================================================
# WORKFLOW-RELATED COMPATIBILITY ADAPTERS
# =============================================================================

class WorkflowEngineAdapter:
    """Compatibility adapter for workflow_engine.py -> WorkflowManager."""
    
    def __init__(self):
        self._manager = None
    
    async def _get_manager(self):
        if self._manager is None:
            self._manager = await _get_workflow_manager()
        return self._manager
    
    async def execute_workflow(self, workflow_def: Dict[str, Any], **kwargs):
        """Execute workflow using new unified manager."""
        manager = await self._get_manager()
        return await manager.execute_workflow(workflow_def, **kwargs)
    
    async def pause_workflow(self, workflow_id: str):
        """Pause workflow using new unified manager."""
        manager = await self._get_manager()
        return await manager.pause_workflow(workflow_id)
    
    async def resume_workflow(self, workflow_id: str):
        """Resume workflow using new unified manager."""
        manager = await self._get_manager()
        return await manager.resume_workflow(workflow_id)


class TaskSchedulerAdapter:
    """Compatibility adapter for task_scheduler.py -> WorkflowManager."""
    
    def __init__(self):
        self._manager = None
    
    async def _get_manager(self):
        if self._manager is None:
            self._manager = await _get_workflow_manager()
        return self._manager
    
    async def schedule_task(self, task_spec: Dict[str, Any], **kwargs):
        """Schedule task using new unified manager."""
        manager = await self._get_manager()
        return await manager.schedule_task(task_spec, **kwargs)
    
    async def cancel_task(self, task_id: str):
        """Cancel task using new unified manager."""
        manager = await self._get_manager()
        return await manager.cancel_task(task_id)


# =============================================================================
# SECURITY-RELATED COMPATIBILITY ADAPTERS
# =============================================================================

class SecurityAuditAdapter:
    """Compatibility adapter for security_audit.py -> SecurityManager."""
    
    def __init__(self):
        self._manager = None
    
    async def _get_manager(self):
        if self._manager is None:
            self._manager = await _get_security_manager()
        return self._manager
    
    async def log_security_event(self, event: Dict[str, Any]):
        """Log security event using new unified manager."""
        manager = await self._get_manager()
        return await manager.log_security_event(event)
    
    async def audit_access(self, user_id: str, resource: str, action: str):
        """Audit access using new unified manager."""
        manager = await self._get_manager()
        return await manager.audit_access(user_id, resource, action)


class AuthAdapter:
    """Compatibility adapter for auth.py -> SecurityManager."""
    
    def __init__(self):
        self._manager = None
    
    async def _get_manager(self):
        if self._manager is None:
            self._manager = await _get_security_manager()
        return self._manager
    
    async def authenticate(self, credentials: Dict[str, Any]):
        """Authenticate using new unified manager."""
        manager = await self._get_manager()
        return await manager.authenticate(credentials)
    
    async def authorize(self, user_id: str, resource: str, action: str):
        """Authorize using new unified manager."""
        manager = await self._get_manager()
        return await manager.authorize(user_id, resource, action)


# =============================================================================
# PERFORMANCE-RELATED COMPATIBILITY ADAPTERS
# =============================================================================

class PerformanceOptimizerAdapter:
    """Compatibility adapter for performance_optimizer.py -> ResourceManager."""
    
    def __init__(self):
        self._manager = None
    
    async def _get_manager(self):
        if self._manager is None:
            self._manager = await _get_resource_manager()
        return self._manager
    
    async def optimize_performance(self, target: str, **kwargs):
        """Optimize performance using new unified manager."""
        manager = await self._get_manager()
        return await manager.optimize_performance(target, **kwargs)
    
    async def get_performance_metrics(self):
        """Get performance metrics using new unified manager."""
        manager = await self._get_manager()
        return await manager.get_performance_metrics()


class ResourceOptimizerAdapter:
    """Compatibility adapter for resource_optimizer.py -> ResourceManager."""
    
    def __init__(self):
        self._manager = None
    
    async def _get_manager(self):
        if self._manager is None:
            self._manager = await _get_resource_manager()
        return self._manager
    
    async def allocate_resources(self, resource_spec: Dict[str, Any]):
        """Allocate resources using new unified manager."""
        manager = await self._get_manager()
        return await manager.allocate_resources(resource_spec)
    
    async def deallocate_resources(self, allocation_id: str):
        """Deallocate resources using new unified manager."""
        manager = await self._get_manager()
        return await manager.deallocate_resources(allocation_id)


# =============================================================================
# STORAGE-RELATED COMPATIBILITY ADAPTERS
# =============================================================================

class DatabaseAdapter:
    """Compatibility adapter for database.py -> StorageManager."""
    
    def __init__(self):
        self._manager = None
    
    async def _get_manager(self):
        if self._manager is None:
            self._manager = await _get_storage_manager()
        return self._manager
    
    async def get_session(self):
        """Get database session using new unified manager."""
        manager = await self._get_manager()
        return await manager.get_database_session()
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None):
        """Execute query using new unified manager."""
        manager = await self._get_manager()
        return await manager.execute_query(query, params)


class RedisAdapter:
    """Compatibility adapter for redis.py -> StorageManager."""
    
    def __init__(self):
        self._manager = None
    
    async def _get_manager(self):
        if self._manager is None:
            self._manager = await _get_storage_manager()
        return self._manager
    
    async def get_redis(self):
        """Get Redis client using new unified manager."""
        manager = await self._get_manager()
        return await manager.get_redis_client()
    
    async def set_cache(self, key: str, value: Any, ttl: int = None):
        """Set cache value using new unified manager."""
        manager = await self._get_manager()
        return await manager.set_cache(key, value, ttl)
    
    async def get_cache(self, key: str):
        """Get cache value using new unified manager."""
        manager = await self._get_manager()
        return await manager.get_cache(key)


# =============================================================================
# ADAPTER REGISTRY
# =============================================================================

# Global registry of adapter instances for reuse
_adapter_registry: Dict[str, Any] = {}


def get_adapter(adapter_name: str):
    """Get or create adapter instance."""
    if adapter_name not in _adapter_registry:
        adapter_classes = {
            'agent_spawner': AgentSpawnerAdapter,
            'agent_registry': AgentRegistryAdapter,
            'agent_lifecycle_manager': AgentLifecycleManagerAdapter,
            'messaging_service': MessagingServiceAdapter,
            'redis_pubsub_manager': RedisPubSubManagerAdapter,
            'context_compression': ContextCompressionAdapter,
            'context_lifecycle_manager': ContextLifecycleManagerAdapter,
            'workflow_engine': WorkflowEngineAdapter,
            'task_scheduler': TaskSchedulerAdapter,
            'security_audit': SecurityAuditAdapter,
            'auth': AuthAdapter,
            'performance_optimizer': PerformanceOptimizerAdapter,
            'resource_optimizer': ResourceOptimizerAdapter,
            'database': DatabaseAdapter,
            'redis': RedisAdapter,
        }
        
        if adapter_name in adapter_classes:
            _adapter_registry[adapter_name] = adapter_classes[adapter_name]()
        else:
            raise ValueError(f"Unknown adapter: {adapter_name}")
    
    return _adapter_registry[adapter_name]


# =============================================================================
# CLEANUP FUNCTIONS
# =============================================================================

async def cleanup_adapters():
    """Cleanup all adapter instances."""
    global _agent_manager, _context_manager, _resource_manager
    global _communication_manager, _security_manager, _workflow_manager, _storage_manager
    
    # Cleanup unified managers
    managers = [
        _agent_manager, _context_manager, _resource_manager,
        _communication_manager, _security_manager, _workflow_manager, _storage_manager
    ]
    
    for manager in managers:
        if manager is not None:
            try:
                await manager.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up manager: {e}")
    
    # Reset global variables
    _agent_manager = None
    _context_manager = None
    _resource_manager = None
    _communication_manager = None
    _security_manager = None
    _workflow_manager = None
    _storage_manager = None
    
    # Clear adapter registry
    _adapter_registry.clear()
    
    logger.info("Compatibility adapters cleaned up successfully")