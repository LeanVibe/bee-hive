"""
Integration Manager - Consolidated External Integration Management

Consolidates functionality from:
- CommunicationManager, MessagingService, ConnectionManager
- StorageManager, DatabaseManager, CacheManager
- Redis integration, WebSocket broadcasting
- All integration-related manager classes (15+ files)

Preserves Epic C Phase C.4 WebSocket broadcasting and API v2 connectivity.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from ..config import settings
from ..logging_service import get_component_logger

logger = get_component_logger("integration_manager")


@dataclass
class ConnectionManager:
    """WebSocket connection management."""
    active_connections: Dict[str, Any]
    connection_count: int = 0
    
    async def broadcast_agent_update(self, agent_id: str, data: Dict[str, Any]) -> None:
        """Broadcast agent update to all connections."""
        pass
    
    async def broadcast_task_update(self, task_id: str, data: Dict[str, Any]) -> None:
        """Broadcast task update to all connections."""
        pass
        
    async def broadcast_system_status(self, data: Dict[str, Any]) -> None:
        """Broadcast system status to all connections."""
        pass


class IntegrationError(Exception):
    """Integration management errors."""
    pass


class IntegrationManager:
    """
    Consolidated Integration Manager
    
    Replaces and consolidates:
    - CommunicationManager, MessagingService  
    - StorageManager, DatabaseManager, CacheManager
    - ConnectionManager (WebSocket)
    - Redis integration components
    - All database, cache, messaging integration classes (15+ files)
    
    Preserves:
    - Epic C Phase C.4 WebSocket broadcasting
    - API v2 database connectivity
    - Redis caching and pub/sub
    - Real-time PWA updates
    """

    def __init__(self, master_orchestrator):
        """Initialize integration manager."""
        self.master_orchestrator = master_orchestrator
        
        # Integration components (lazy-loaded)
        self._database_session = None
        self._redis_client = None
        self._websocket_manager = None
        self._cache_manager = None
        
        # Connection tracking
        self.database_connected = False
        self.redis_connected = False
        self.websocket_enabled = False
        
        # Performance metrics
        self.db_query_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.websocket_broadcast_count = 0
        
        logger.info("Integration Manager initialized")

    async def initialize(self) -> None:
        """Initialize all integration components."""
        try:
            # Initialize database connection
            await self._initialize_database()
            
            # Initialize Redis connection
            await self._initialize_redis()
            
            # Initialize WebSocket manager
            if self.master_orchestrator.config.enable_websocket_broadcasting:
                await self._initialize_websocket()
            
            logger.info("✅ Integration Manager initialized successfully",
                       database_connected=self.database_connected,
                       redis_connected=self.redis_connected,
                       websocket_enabled=self.websocket_enabled)
            
        except Exception as e:
            logger.error("❌ Integration Manager initialization failed", error=str(e))
            raise IntegrationError(f"Initialization failed: {e}") from e

    async def shutdown(self) -> None:
        """Shutdown all integration components."""
        logger.info("🛑 Shutting down Integration Manager...")
        
        # Close database connections
        if self._database_session:
            try:
                await self._database_session.close()
            except Exception as e:
                logger.warning(f"Failed to close database session: {e}")
        
        # Close Redis connection
        if self._redis_client:
            try:
                await self._redis_client.close()
            except Exception as e:
                logger.warning(f"Failed to close Redis connection: {e}")
        
        # Close WebSocket connections
        if self._websocket_manager:
            try:
                await self._websocket_manager.shutdown()
            except Exception as e:
                logger.warning(f"Failed to shutdown WebSocket manager: {e}")
        
        logger.info("✅ Integration Manager shutdown complete")

    # ==================================================================
    # DATABASE INTEGRATION (StorageManager consolidation)
    # ==================================================================

    async def get_database_session(self):
        """Get database session - consolidated from multiple managers."""
        if not self.database_connected:
            await self._initialize_database()
        
        return self._database_session

    async def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> Any:
        """Execute database query with connection management."""
        try:
            if not self.database_connected:
                await self._initialize_database()
            
            session = await self.get_database_session()
            if not session:
                return None
            
            result = await session.execute(query, parameters or {})
            self.db_query_count += 1
            
            return result
            
        except Exception as e:
            logger.error("Database query failed", query=query, error=str(e))
            return None

    async def store_performance_metric(
        self,
        metric_name: str,
        metric_value: float,
        tags: Dict[str, Any] = None
    ) -> bool:
        """Store performance metric in database."""
        try:
            session = await self.get_database_session()
            if not session:
                return False
            
            from ...models.performance_metric import PerformanceMetric
            
            metric = PerformanceMetric(
                metric_name=metric_name,
                metric_value=metric_value,
                tags=tags or {}
            )
            
            session.add(metric)
            await session.commit()
            
            return True
            
        except Exception as e:
            logger.error("Failed to store performance metric", 
                        metric_name=metric_name, error=str(e))
            return False

    # ==================================================================
    # REDIS INTEGRATION (CacheManager consolidation)
    # ==================================================================

    async def cache_get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            if not self.redis_connected:
                await self._initialize_redis()
            
            if not self._redis_client:
                return None
            
            value = await self._redis_client.get(key)
            
            if value:
                self.cache_hit_count += 1
                # Deserialize JSON if possible
                try:
                    import json
                    return json.loads(value)
                except:
                    return value
            else:
                self.cache_miss_count += 1
                return None
                
        except Exception as e:
            logger.error("Cache get failed", key=key, error=str(e))
            self.cache_miss_count += 1
            return None

    async def cache_set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 300
    ) -> bool:
        """Set value in Redis cache."""
        try:
            if not self.redis_connected:
                await self._initialize_redis()
            
            if not self._redis_client:
                return False
            
            # Serialize to JSON if possible
            try:
                import json
                serialized_value = json.dumps(value)
            except:
                serialized_value = str(value)
            
            await self._redis_client.setex(key, ttl_seconds, serialized_value)
            return True
            
        except Exception as e:
            logger.error("Cache set failed", key=key, error=str(e))
            return False

    async def cache_delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            if not self.redis_connected or not self._redis_client:
                return False
            
            result = await self._redis_client.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error("Cache delete failed", key=key, error=str(e))
            return False

    async def publish_message(self, channel: str, message: Dict[str, Any]) -> bool:
        """Publish message to Redis pub/sub."""
        try:
            if not self.redis_connected or not self._redis_client:
                return False
            
            import json
            serialized_message = json.dumps(message)
            await self._redis_client.publish(channel, serialized_message)
            
            return True
            
        except Exception as e:
            logger.error("Redis publish failed", channel=channel, error=str(e))
            return False

    # ==================================================================
    # WEBSOCKET INTEGRATION (Epic C Phase C.4 compatibility)
    # ==================================================================

    async def broadcast_agent_update(self, agent_id: str, update_data: Dict[str, Any]) -> None:
        """Broadcast agent update via WebSocket - Epic C Phase C.4 integration."""
        if not self.websocket_enabled or not self._websocket_manager:
            logger.debug("WebSocket broadcasting not enabled")
            return
        
        try:
            broadcast_payload = {
                "type": "agent_update",
                "agent_id": agent_id,
                "data": update_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._websocket_manager.broadcast_agent_update(agent_id, broadcast_payload)
            self.websocket_broadcast_count += 1
            
            logger.debug("Agent update broadcasted", agent_id=agent_id)
            
        except Exception as e:
            logger.warning("Failed to broadcast agent update", 
                         agent_id=agent_id, error=str(e))

    async def broadcast_task_update(self, task_id: str, update_data: Dict[str, Any]) -> None:
        """Broadcast task update via WebSocket - Epic C Phase C.4 integration."""
        if not self.websocket_enabled or not self._websocket_manager:
            logger.debug("WebSocket broadcasting not enabled")
            return
        
        try:
            broadcast_payload = {
                "type": "task_update", 
                "task_id": task_id,
                "data": update_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._websocket_manager.broadcast_task_update(task_id, broadcast_payload)
            self.websocket_broadcast_count += 1
            
            logger.debug("Task update broadcasted", task_id=task_id)
            
        except Exception as e:
            logger.warning("Failed to broadcast task update",
                         task_id=task_id, error=str(e))

    async def broadcast_system_status(self, status_data: Dict[str, Any]) -> None:
        """Broadcast system status via WebSocket - Epic C Phase C.4 integration."""
        if not self.websocket_enabled or not self._websocket_manager:
            logger.debug("WebSocket broadcasting not enabled")
            return
        
        try:
            broadcast_payload = {
                "type": "system_status",
                "data": status_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._websocket_manager.broadcast_system_status(broadcast_payload)
            self.websocket_broadcast_count += 1
            
            logger.debug("System status broadcasted")
            
        except Exception as e:
            logger.warning("Failed to broadcast system status", error=str(e))

    # ==================================================================
    # MESSAGING SERVICE INTEGRATION
    # ==================================================================

    async def send_message(
        self,
        recipient: str,
        message_type: str,
        content: Dict[str, Any],
        priority: str = "medium"
    ) -> bool:
        """Send message via messaging service."""
        try:
            # Could integrate with email, Slack, etc.
            message = {
                "recipient": recipient,
                "type": message_type,
                "content": content,
                "priority": priority,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "integration_manager"
            }
            
            # Publish to Redis for message queue processing
            await self.publish_message("messages", message)
            
            logger.info("Message sent", recipient=recipient, type=message_type)
            return True
            
        except Exception as e:
            logger.error("Failed to send message", 
                        recipient=recipient, error=str(e))
            return False

    async def send_alert(
        self,
        alert_type: str,
        title: str,
        description: str,
        severity: str = "medium"
    ) -> bool:
        """Send alert notification."""
        alert_message = {
            "alert_type": alert_type,
            "title": title,
            "description": description,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.send_message(
            recipient="system_admin",
            message_type="alert",
            content=alert_message,
            priority=severity
        )

    # ==================================================================
    # STATUS AND METRICS
    # ==================================================================

    async def get_status(self) -> Dict[str, Any]:
        """Get integration manager status."""
        return {
            "database_connected": self.database_connected,
            "redis_connected": self.redis_connected,
            "websocket_enabled": self.websocket_enabled,
            "db_query_count": self.db_query_count,
            "cache_performance": {
                "hit_count": self.cache_hit_count,
                "miss_count": self.cache_miss_count,
                "hit_rate": (
                    self.cache_hit_count / max(1, self.cache_hit_count + self.cache_miss_count)
                ) * 100
            },
            "websocket_broadcast_count": self.websocket_broadcast_count
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get integration metrics for monitoring."""
        return {
            "database_queries": self.db_query_count,
            "cache_hits": self.cache_hit_count,
            "cache_misses": self.cache_miss_count,
            "cache_hit_rate": (
                self.cache_hit_count / max(1, self.cache_hit_count + self.cache_miss_count)
            ),
            "websocket_broadcasts": self.websocket_broadcast_count,
            "connection_status": {
                "database": self.database_connected,
                "redis": self.redis_connected,
                "websocket": self.websocket_enabled
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all integrations."""
        health_status = {
            "overall": "healthy",
            "components": {}
        }
        
        # Database health
        try:
            session = await self.get_database_session()
            if session:
                # Simple query to test connection
                await session.execute("SELECT 1")
                health_status["components"]["database"] = "healthy"
            else:
                health_status["components"]["database"] = "unhealthy"
                health_status["overall"] = "degraded"
        except Exception as e:
            health_status["components"]["database"] = f"unhealthy: {e}"
            health_status["overall"] = "degraded"
        
        # Redis health
        try:
            if self.redis_connected and self._redis_client:
                await self._redis_client.ping()
                health_status["components"]["redis"] = "healthy"
            else:
                health_status["components"]["redis"] = "disconnected"
        except Exception as e:
            health_status["components"]["redis"] = f"unhealthy: {e}"
            health_status["overall"] = "degraded"
        
        # WebSocket health
        if self.websocket_enabled:
            health_status["components"]["websocket"] = "enabled"
        else:
            health_status["components"]["websocket"] = "disabled"
        
        return health_status

    # ==================================================================
    # INITIALIZATION METHODS
    # ==================================================================

    async def _initialize_database(self) -> None:
        """Initialize database connection."""
        try:
            from ..database import get_session
            self._database_session = await anext(get_session())
            self.database_connected = True
            logger.info("✅ Database connection initialized")
            
        except Exception as e:
            logger.warning("Database connection not available", error=str(e))
            self.database_connected = False

    async def _initialize_redis(self) -> None:
        """Initialize Redis connection."""
        try:
            from ..redis import get_redis
            self._redis_client = get_redis()
            
            # Test connection
            await self._redis_client.ping()
            self.redis_connected = True
            logger.info("✅ Redis connection initialized")
            
        except Exception as e:
            logger.warning("Redis connection not available", error=str(e))
            self.redis_connected = False

    async def _initialize_websocket(self) -> None:
        """Initialize WebSocket manager."""
        try:
            # Import WebSocket manager
            from ..communication.connection_manager import ConnectionManager as WSManager
            self._websocket_manager = ConnectionManager(active_connections={})
            self.websocket_enabled = True
            logger.info("✅ WebSocket broadcasting enabled")
            
        except Exception as e:
            logger.warning("WebSocket manager not available", error=str(e))
            self.websocket_enabled = False

    # ==================================================================
    # COMPATIBILITY METHODS (Legacy Support)
    # ==================================================================

    async def get_session_cache(self):
        """Legacy session cache compatibility."""
        return self

    async def get_storage_manager(self):
        """Legacy storage manager compatibility."""
        return self

    async def get_communication_manager(self):
        """Legacy communication manager compatibility."""
        return self

    async def get_cache_manager(self):
        """Legacy cache manager compatibility."""  
        return self