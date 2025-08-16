"""
Messaging Migration Utilities
Provides compatibility and migration helpers for transitioning from legacy messaging services
to the unified messaging service.

Epic 1, Phase 1 Week 2: Messaging Service Migration Support
"""

from typing import Dict, Any, Optional, List, Callable, Union
import asyncio
import warnings
import functools
from datetime import datetime

from .messaging_service import (
    MessagingService, Message, MessageType, MessagePriority, RoutingStrategy,
    get_messaging_service, send_agent_message, send_task_assignment, 
    send_heartbeat_request, publish_system_event
)
from .logging_service import get_component_logger

logger = get_component_logger("messaging_migration")


def deprecated_messaging_service(replacement_info: str):
    """Decorator to mark messaging services as deprecated"""
    def decorator(cls):
        original_init = cls.__init__
        
        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            warnings.warn(
                f"{cls.__name__} is deprecated. {replacement_info}",
                DeprecationWarning,
                stacklevel=2
            )
            logger.warning(f"Using deprecated messaging service {cls.__name__}. "
                          f"Migration info: {replacement_info}")
            return original_init(self, *args, **kwargs)
        
        cls.__init__ = new_init
        return cls
    return decorator


class LegacyMessageAdapter:
    """Adapter to convert legacy message formats to unified Message format"""
    
    @staticmethod
    def from_agent_message(agent_msg) -> Message:
        """Convert AgentMessage to unified Message"""
        # Handle different legacy AgentMessage formats
        try:
            if hasattr(agent_msg, 'message_type'):
                # From agent_messaging_service.py
                return Message(
                    id=agent_msg.message_id,
                    type=MessageType(agent_msg.message_type.value),
                    priority=MessagePriority(agent_msg.priority.value) if hasattr(agent_msg.priority, 'value') else MessagePriority.NORMAL,
                    sender=agent_msg.from_agent,
                    recipient=agent_msg.to_agent,
                    payload=agent_msg.payload,
                    correlation_id=agent_msg.correlation_id,
                    created_at=agent_msg.created_at if hasattr(agent_msg, 'created_at') else datetime.utcnow(),
                    metadata=agent_msg.metadata if hasattr(agent_msg, 'metadata') else {}
                )
            elif hasattr(agent_msg, 'type'):
                # From agent_communication_service.py
                return Message(
                    id=agent_msg.id,
                    type=MessageType(agent_msg.type.value),
                    priority=MessagePriority(agent_msg.priority.value) if hasattr(agent_msg.priority, 'value') else MessagePriority.NORMAL,
                    sender=agent_msg.from_agent,
                    recipient=agent_msg.to_agent,
                    payload=agent_msg.payload,
                    correlation_id=agent_msg.correlation_id,
                    timestamp=agent_msg.timestamp if hasattr(agent_msg, 'timestamp') else None
                )
            else:
                # Generic message format
                return Message(
                    id=getattr(agent_msg, 'id', str(agent_msg.get('id', 'unknown'))),
                    type=MessageType.EVENT,
                    sender=getattr(agent_msg, 'from_agent', str(agent_msg.get('from_agent', 'unknown'))),
                    recipient=getattr(agent_msg, 'to_agent', agent_msg.get('to_agent')),
                    payload=getattr(agent_msg, 'payload', agent_msg.get('payload', {}))
                )
        except Exception as e:
            logger.error(f"Failed to convert legacy message: {e}")
            # Fallback generic message
            return Message(
                type=MessageType.EVENT,
                sender="legacy_system",
                payload={"error": "Failed to convert legacy message", "original": str(agent_msg)}
            )
    
    @staticmethod
    def from_stream_message(stream_msg) -> Message:
        """Convert StreamMessage to unified Message"""
        try:
            return Message(
                id=stream_msg.id,
                type=MessageType(stream_msg.message_type.value),
                priority=MessagePriority(stream_msg.priority.value) if hasattr(stream_msg.priority, 'value') else MessagePriority.NORMAL,
                sender=stream_msg.from_agent,
                recipient=stream_msg.to_agent,
                payload=stream_msg.payload,
                timestamp=stream_msg.timestamp,
                ttl=stream_msg.ttl,
                correlation_id=stream_msg.correlation_id,
                signature=stream_msg.signature if hasattr(stream_msg, 'signature') else None
            )
        except Exception as e:
            logger.error(f"Failed to convert stream message: {e}")
            return Message(
                type=MessageType.EVENT,
                sender="legacy_system",
                payload={"error": "Failed to convert stream message", "original": str(stream_msg)}
            )
    
    @staticmethod
    def to_legacy_format(message: Message, format_type: str = "agent_message") -> Dict[str, Any]:
        """Convert unified Message to legacy format for backward compatibility"""
        if format_type == "agent_message":
            return {
                "message_id": message.id,
                "message_type": message.type.value,
                "from_agent": message.sender,
                "to_agent": message.recipient,
                "payload": message.payload,
                "priority": message.priority.value,
                "correlation_id": message.correlation_id,
                "created_at": message.created_at.isoformat(),
                "metadata": message.metadata
            }
        elif format_type == "stream_message":
            return {
                "id": message.id,
                "from_agent": message.sender,
                "to_agent": message.recipient,
                "message_type": message.type.value,
                "payload": message.payload,
                "priority": message.priority.value,
                "timestamp": message.timestamp,
                "ttl": message.ttl,
                "correlation_id": message.correlation_id,
                "signature": message.signature
            }
        else:
            return message.to_redis_dict()


class MessagingServiceAdapter:
    """Adapter for legacy messaging service interfaces"""
    
    def __init__(self):
        self.messaging_service = get_messaging_service()
        self._legacy_handlers: Dict[str, Callable] = {}
    
    async def send_lifecycle_message(self, 
                                   message_type,
                                   from_agent: str,
                                   to_agent: str,
                                   payload: Dict[str, Any],
                                   priority=None,
                                   correlation_id: Optional[str] = None,
                                   expires_in_seconds: Optional[int] = None) -> str:
        """Legacy interface for sending lifecycle messages"""
        # Convert legacy message type to unified format
        try:
            unified_type = MessageType(message_type.value if hasattr(message_type, 'value') else str(message_type))
        except ValueError:
            unified_type = MessageType.EVENT
        
        # Convert legacy priority
        unified_priority = MessagePriority.NORMAL
        if priority:
            try:
                if hasattr(priority, 'value'):
                    priority_value = priority.value
                else:
                    priority_value = int(priority)
                
                # Map legacy priority values to unified priorities
                if priority_value >= 10:
                    unified_priority = MessagePriority.CRITICAL
                elif priority_value >= 8:
                    unified_priority = MessagePriority.URGENT
                elif priority_value >= 5:
                    unified_priority = MessagePriority.HIGH
                else:
                    unified_priority = MessagePriority.NORMAL
            except:
                unified_priority = MessagePriority.NORMAL
        
        message = Message(
            type=unified_type,
            sender=from_agent,
            recipient=to_agent,
            payload=payload,
            priority=unified_priority,
            correlation_id=correlation_id,
            ttl=expires_in_seconds
        )
        
        success = await self.messaging_service.send_message(message)
        return message.id if success else None
    
    async def register_message_handler(self, message_type, handler: Callable):
        """Legacy interface for registering message handlers"""
        handler_id = f"legacy_{message_type.value if hasattr(message_type, 'value') else str(message_type)}"
        self._legacy_handlers[handler_id] = handler
        
        # Create adapter handler that converts messages
        async def adapter_handler(message: Message) -> Optional[Message]:
            try:
                # Convert to legacy format
                legacy_message = LegacyMessageAdapter.to_legacy_format(message, "agent_message")
                
                # Call legacy handler
                if asyncio.iscoroutinefunction(handler):
                    await handler(legacy_message)
                else:
                    handler(legacy_message)
                
                return None
            except Exception as e:
                logger.error(f"Legacy handler {handler_id} failed", error=str(e))
                return None
        
        # Register with messaging service
        from .messaging_service import MessageHandler
        
        class LegacyHandlerWrapper(MessageHandler):
            def __init__(self, handler_id: str, message_type, adapter_handler):
                try:
                    unified_type = MessageType(message_type.value if hasattr(message_type, 'value') else str(message_type))
                    message_types = [unified_type]
                except ValueError:
                    message_types = []
                
                super().__init__(handler_id, pattern="legacy.*", message_types=message_types)
                self.adapter_handler = adapter_handler
            
            async def _process_message(self, message: Message) -> Optional[Message]:
                return await self.adapter_handler(message)
        
        wrapper = LegacyHandlerWrapper(handler_id, message_type, adapter_handler)
        self.messaging_service.register_handler(wrapper)
        
        logger.info(f"Legacy handler registered: {handler_id}")
    
    async def send_message(self, message) -> bool:
        """Legacy interface for sending messages"""
        # Convert legacy message to unified format
        unified_message = LegacyMessageAdapter.from_agent_message(message)
        return await self.messaging_service.send_message(unified_message)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Legacy interface for getting metrics"""
        metrics = self.messaging_service.get_service_metrics()
        return metrics.to_dict()


class MessageBrokerAdapter:
    """Adapter for legacy MessageBroker interface"""
    
    def __init__(self):
        self.messaging_service = get_messaging_service()
        self._connected = False
    
    async def connect(self) -> None:
        """Legacy connect interface"""
        if not self.messaging_service._connected:
            await self.messaging_service.connect()
        await self.messaging_service.start_service()
        self._connected = True
        logger.info("Legacy MessageBroker adapter connected")
    
    async def disconnect(self) -> None:
        """Legacy disconnect interface"""
        await self.messaging_service.stop_service()
        await self.messaging_service.disconnect()
        self._connected = False
        logger.info("Legacy MessageBroker adapter disconnected")
    
    async def send_message(self, message) -> str:
        """Legacy send_message interface"""
        unified_message = LegacyMessageAdapter.from_stream_message(message)
        success = await self.messaging_service.send_message(unified_message)
        return unified_message.id if success else None
    
    async def consume_messages(self, stream_name: str, group_name: str, 
                             consumer_name: str, handler: Callable, **kwargs) -> None:
        """Legacy consume_messages interface"""
        # Create adapter handler
        async def adapter_handler(message: Message) -> Optional[Message]:
            try:
                # Convert to legacy stream message format
                legacy_message = LegacyMessageAdapter.to_legacy_format(message, "stream_message")
                
                # Call legacy handler (usually returns bool for success)
                if asyncio.iscoroutinefunction(handler):
                    success = await handler(legacy_message)
                else:
                    success = handler(legacy_message)
                
                return None  # No response expected
            except Exception as e:
                logger.error(f"Legacy message handler failed", error=str(e))
                return None
        
        # Register handler with messaging service
        from .messaging_service import MessageHandler
        
        class LegacyConsumerHandler(MessageHandler):
            def __init__(self, handler_id: str, adapter_handler):
                super().__init__(handler_id, pattern="*")
                self.adapter_handler = adapter_handler
            
            async def _process_message(self, message: Message) -> Optional[Message]:
                return await self.adapter_handler(message)
        
        handler_id = f"legacy_consumer_{stream_name}_{group_name}_{consumer_name}"
        wrapper = LegacyConsumerHandler(handler_id, adapter_handler)
        self.messaging_service.register_handler(wrapper)
        
        # Subscribe to relevant topics
        if "broadcast" in stream_name:
            await self.messaging_service.subscribe_to_topic("broadcast", handler_id)
        elif "agent_messages:" in stream_name:
            agent_id = stream_name.replace("agent_messages:", "")
            await self.messaging_service.subscribe_to_topic(agent_id, handler_id)
        
        logger.info(f"Legacy consumer registered: {handler_id}")


# Backward compatibility functions
async def create_legacy_communication_service(**kwargs) -> MessagingServiceAdapter:
    """Create legacy communication service adapter"""
    warnings.warn(
        "create_legacy_communication_service is deprecated. Use get_messaging_service() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    adapter = MessagingServiceAdapter()
    if not adapter.messaging_service._connected:
        await adapter.messaging_service.connect()
        await adapter.messaging_service.start_service()
    return adapter


async def create_legacy_message_broker(**kwargs) -> MessageBrokerAdapter:
    """Create legacy message broker adapter"""
    warnings.warn(
        "create_legacy_message_broker is deprecated. Use get_messaging_service() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    adapter = MessageBrokerAdapter()
    await adapter.connect()
    return adapter


# Migration verification functions
async def verify_messaging_migration() -> Dict[str, Any]:
    """Verify that messaging migration is working correctly"""
    messaging = get_messaging_service()
    
    # Test basic connectivity
    if not messaging._connected:
        await messaging.connect()
        await messaging.start_service()
    
    # Perform health check
    health = await messaging.health_check()
    
    # Get service metrics
    metrics = messaging.get_service_metrics()
    
    # Test message sending
    test_message = Message(
        type=MessageType.EVENT,
        sender="migration_test",
        payload={"test": "migration_verification"},
        routing_strategy=RoutingStrategy.BROADCAST
    )
    
    send_success = await messaging.send_message(test_message)
    
    return {
        "health_check": health,
        "metrics": metrics.to_dict(),
        "test_message_sent": send_success,
        "migration_status": "success" if health.get("status") == "healthy" and send_success else "issues_detected",
        "timestamp": datetime.utcnow().isoformat()
    }


# Migration status tracking
_migration_status = {
    "messaging_service_created": True,
    "legacy_adapters_ready": True,
    "orchestrator_migrated": False,
    "redis_migrated": False,
    "api_endpoints_migrated": False,
    "tests_updated": False
}


def get_migration_status() -> Dict[str, Any]:
    """Get current migration status"""
    completed = sum(1 for status in _migration_status.values() if status)
    total = len(_migration_status)
    progress = (completed / total) * 100
    
    return {
        "progress_percent": progress,
        "completed_tasks": completed,
        "total_tasks": total,
        "status_details": _migration_status,
        "next_steps": [
            key for key, status in _migration_status.items() 
            if not status
        ]
    }


def mark_migration_complete(component: str) -> None:
    """Mark a component as migrated"""
    if component in _migration_status:
        _migration_status[component] = True
        logger.info(f"Migration completed for component: {component}")
    else:
        logger.warning(f"Unknown migration component: {component}")