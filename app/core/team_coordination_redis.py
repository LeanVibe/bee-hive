"""
Redis Integration Service for Team Coordination

Enterprise-grade Redis integration providing:
- Real-time messaging and pub/sub
- Distributed coordination state management
- Performance metrics caching
- Event streaming for multi-agent workflows
- Circuit breaker patterns for resilience
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Callable, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio import ConnectionPool
import structlog
from pydantic import BaseModel

from ..core.redis import get_redis
from ..schemas.team_coordination import (
    AgentStatusUpdate, TaskStatusUpdate, SystemMetricsUpdate, CoordinationAlert
)


logger = structlog.get_logger()


# =====================================================================================
# REDIS CHANNEL CONSTANTS AND PATTERNS
# =====================================================================================

class RedisChannels:
    """Redis channel constants for team coordination."""
    
    # Agent channels
    AGENT_REGISTRATIONS = "coordination:agent:registrations"
    AGENT_STATUS_UPDATES = "coordination:agent:status"
    AGENT_HEARTBEATS = "coordination:agent:heartbeats"
    AGENT_WORKLOAD_UPDATES = "coordination:agent:workload"
    
    # Task channels
    TASK_ASSIGNMENTS = "coordination:task:assignments"
    TASK_STATUS_UPDATES = "coordination:task:status"
    TASK_COMPLETIONS = "coordination:task:completions"
    TASK_REASSIGNMENTS = "coordination:task:reassignments"
    
    # System channels
    SYSTEM_METRICS = "coordination:system:metrics"
    SYSTEM_ALERTS = "coordination:system:alerts"
    COORDINATION_EVENTS = "coordination:system:events"
    
    # Performance channels
    PERFORMANCE_METRICS = "coordination:performance:metrics"
    BOTTLENECK_ALERTS = "coordination:performance:bottlenecks"
    
    # WebSocket broadcast channels
    WEBSOCKET_BROADCAST = "coordination:websocket:broadcast"
    
    @classmethod
    def agent_specific(cls, agent_id: str) -> str:
        """Get agent-specific channel."""
        return f"coordination:agent:{agent_id}"
    
    @classmethod
    def task_specific(cls, task_id: str) -> str:
        """Get task-specific channel."""
        return f"coordination:task:{task_id}"
    
    @classmethod
    def team_specific(cls, team_id: str) -> str:
        """Get team-specific channel."""
        return f"coordination:team:{team_id}"


class RedisKeys:
    """Redis key patterns for coordination data."""
    
    # Agent data
    AGENT_STATUS = "coordination:agent_status:{agent_id}"
    AGENT_CAPABILITIES = "coordination:agent_capabilities:{agent_id}"
    AGENT_WORKLOAD = "coordination:agent_workload:{agent_id}"
    AGENT_PERFORMANCE = "coordination:agent_performance:{agent_id}"
    
    # Task data
    TASK_ASSIGNMENT_QUEUE = "coordination:task_queue"
    TASK_ASSIGNMENTS = "coordination:task_assignments"
    TASK_PROGRESS = "coordination:task_progress:{task_id}"
    
    # System state
    ACTIVE_AGENTS = "coordination:active_agents"
    SYSTEM_METRICS = "coordination:system_metrics"
    COORDINATION_STATE = "coordination:system_state"
    
    # Performance data
    METRICS_BUFFER = "coordination:metrics_buffer"
    PERFORMANCE_HISTORY = "coordination:performance_history:{period}"
    
    # Lock patterns
    ASSIGNMENT_LOCK = "coordination:lock:assignment:{task_id}"
    AGENT_UPDATE_LOCK = "coordination:lock:agent:{agent_id}"


# =====================================================================================
# DATA MODELS AND STRUCTURES
# =====================================================================================

@dataclass  
class CoordinationEvent:
    """Base coordination event structure."""
    event_id: str
    event_type: str
    timestamp: datetime
    source_agent_id: Optional[str] = None
    target_agent_id: Optional[str] = None
    payload: Dict[str, Any] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoordinationEvent':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class AgentCoordinationState:
    """Agent coordination state in Redis."""
    agent_id: str
    status: str
    current_workload: float
    available_capacity: float
    active_tasks: List[str]
    capabilities: List[str]
    last_heartbeat: datetime
    performance_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        data = asdict(self)
        data['last_heartbeat'] = self.last_heartbeat.isoformat()
        return data


@dataclass
class TaskCoordinationState:
    """Task coordination state in Redis."""
    task_id: str
    status: str
    assigned_agent_id: Optional[str]
    priority: str
    estimated_completion: Optional[datetime]
    progress_percentage: float
    required_capabilities: List[str]
    assignment_timestamp: Optional[datetime]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        data = asdict(self)
        if self.estimated_completion:
            data['estimated_completion'] = self.estimated_completion.isoformat()
        if self.assignment_timestamp:
            data['assignment_timestamp'] = self.assignment_timestamp.isoformat()
        return data


# =====================================================================================
# CIRCUIT BREAKER PATTERN FOR REDIS RESILIENCE
# =====================================================================================

class CircuitBreakerState:
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class RedisCircuitBreaker:
    """Circuit breaker for Redis operations."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        
    def should_allow_request(self) -> bool:
        """Check if request should be allowed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and \
               (datetime.utcnow() - self.last_failure_time).seconds > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        
        # HALF_OPEN state - allow one request to test
        return True
    
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
        
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


# =====================================================================================
# MAIN REDIS COORDINATION SERVICE
# =====================================================================================

class TeamCoordinationRedisService:
    """Enterprise Redis service for team coordination."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub_client: Optional[redis.Redis] = None
        self.circuit_breaker = RedisCircuitBreaker()
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.active_subscriptions: Set[str] = set()
        self.metrics_buffer: List[Dict[str, Any]] = []
        self.background_tasks: Set[asyncio.Task] = set()
        
    async def initialize(self) -> None:
        """Initialize Redis connections and background tasks."""
        try:
            self.redis_client = await get_redis()
            self.pubsub_client = await get_redis()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Team coordination Redis service initialized")
            
        except Exception as e:
            logger.error("Failed to initialize Redis service", error=str(e))
            raise
    
    async def cleanup(self) -> None:
        """Clean up connections and background tasks."""
        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close connections
        if self.pubsub_client:
            await self.pubsub_client.close()
        
        logger.info("Team coordination Redis service cleaned up")
    
    async def _start_background_tasks(self) -> None:
        """Start background processing tasks."""
        # Metrics flush task
        task = asyncio.create_task(self._metrics_flush_loop())
        self.background_tasks.add(task)
        
        # System monitoring task
        task = asyncio.create_task(self._system_monitoring_loop())
        self.background_tasks.add(task)
        
        # Cleanup task for expired data
        task = asyncio.create_task(self._cleanup_expired_data_loop())
        self.background_tasks.add(task)
    
    # =====================================================================================
    # AGENT COORDINATION METHODS
    # =====================================================================================
    
    async def register_agent(self, agent_state: AgentCoordinationState) -> None:
        """Register agent in Redis with coordination state."""
        if not self.circuit_breaker.should_allow_request():
            logger.warning("Circuit breaker open, skipping agent registration")
            return
        
        try:
            # Store agent state
            agent_key = RedisKeys.AGENT_STATUS.format(agent_id=agent_state.agent_id)
            await self.redis_client.hset(agent_key, mapping=agent_state.to_dict())
            await self.redis_client.expire(agent_key, timedelta(hours=24))
            
            # Add to active agents set
            await self.redis_client.sadd(RedisKeys.ACTIVE_AGENTS, agent_state.agent_id)
            
            # Publish registration event
            event = CoordinationEvent(
                event_id=str(uuid.uuid4()),
                event_type="agent_registered",
                timestamp=datetime.utcnow(),
                source_agent_id=agent_state.agent_id,
                payload={
                    "agent_name": agent_state.agent_id,  # Would get name from full agent data
                    "capabilities": agent_state.capabilities,
                    "initial_capacity": agent_state.available_capacity
                }
            )
            
            await self.publish_event(RedisChannels.AGENT_REGISTRATIONS, event)
            
            self.circuit_breaker.record_success()
            logger.info("Agent registered in Redis", agent_id=agent_state.agent_id)
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error("Failed to register agent", agent_id=agent_state.agent_id, error=str(e))
            raise
    
    async def update_agent_status(self, agent_id: str, status: str, workload: float) -> None:
        """Update agent status and workload."""
        if not self.circuit_breaker.should_allow_request():
            return
        
        try:
            # Use distributed lock for agent updates
            async with self._distributed_lock(RedisKeys.AGENT_UPDATE_LOCK.format(agent_id=agent_id)):
                # Update agent state
                agent_key = RedisKeys.AGENT_STATUS.format(agent_id=agent_id)
                updates = {
                    "status": status,
                    "current_workload": workload,
                    "available_capacity": 1.0 - workload,
                    "last_heartbeat": datetime.utcnow().isoformat()
                }
                
                await self.redis_client.hset(agent_key, mapping=updates)
                
                # Publish status update
                status_update = AgentStatusUpdate(
                    agent_id=agent_id,
                    agent_name=agent_id,  # Would get actual name
                    old_status=status,  # Would get from previous state
                    new_status=status,
                    current_workload=workload,
                    active_tasks=0  # Would calculate from task assignments
                )
                
                await self.publish_websocket_message(status_update)
            
            self.circuit_breaker.record_success()
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error("Failed to update agent status", agent_id=agent_id, error=str(e))
    
    async def get_agent_state(self, agent_id: str) -> Optional[AgentCoordinationState]:
        """Get current agent coordination state."""
        if not self.circuit_breaker.should_allow_request():
            return None
        
        try:
            agent_key = RedisKeys.AGENT_STATUS.format(agent_id=agent_id)
            data = await self.redis_client.hgetall(agent_key)
            
            if not data:
                return None
            
            # Convert Redis data back to AgentCoordinationState
            state_data = {
                "agent_id": agent_id,
                "status": data.get("status", "unknown"),
                "current_workload": float(data.get("current_workload", 0.0)),
                "available_capacity": float(data.get("available_capacity", 1.0)),
                "active_tasks": json.loads(data.get("active_tasks", "[]")),
                "capabilities": json.loads(data.get("capabilities", "[]")),
                "last_heartbeat": datetime.fromisoformat(data.get("last_heartbeat", datetime.utcnow().isoformat())),
                "performance_score": float(data.get("performance_score", 0.8))
            }
            
            return AgentCoordinationState(**state_data)
            
        except Exception as e:
            logger.error("Failed to get agent state", agent_id=agent_id, error=str(e))
            return None
    
    # =====================================================================================
    # TASK COORDINATION METHODS
    # =====================================================================================
    
    async def queue_task_for_assignment(self, task_state: TaskCoordinationState) -> None:
        """Queue task for intelligent assignment."""
        if not self.circuit_breaker.should_allow_request():
            return
        
        try:
            # Add task to assignment queue with priority scoring
            priority_score = self._calculate_priority_score(task_state.priority)
            task_data = {
                "task_id": task_state.task_id,
                "data": json.dumps(task_state.to_dict()),
                "queued_at": datetime.utcnow().isoformat()
            }
            
            # Use sorted set for priority-based queuing
            await self.redis_client.zadd(
                RedisKeys.TASK_ASSIGNMENT_QUEUE,
                {json.dumps(task_data): priority_score}
            )
            
            # Publish task queued event
            event = CoordinationEvent(
                event_id=str(uuid.uuid4()),
                event_type="task_queued",
                timestamp=datetime.utcnow(),
                payload={
                    "task_id": task_state.task_id,
                    "priority": task_state.priority,
                    "required_capabilities": task_state.required_capabilities
                }
            )
            
            await self.publish_event(RedisChannels.TASK_ASSIGNMENTS, event)
            
            logger.info("Task queued for assignment", task_id=task_state.task_id)
            
        except Exception as e:
            logger.error("Failed to queue task", task_id=task_state.task_id, error=str(e))
    
    async def assign_task_to_agent(self, task_id: str, agent_id: str) -> None:
        """Assign task to specific agent with coordination."""
        if not self.circuit_breaker.should_allow_request():
            return
        
        try:
            # Use distributed lock to prevent double assignment
            async with self._distributed_lock(RedisKeys.ASSIGNMENT_LOCK.format(task_id=task_id)):
                # Update task assignment
                assignment_data = {
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "assigned_at": datetime.utcnow().isoformat(),
                    "status": "assigned"
                }
                
                await self.redis_client.hset(
                    RedisKeys.TASK_ASSIGNMENTS,
                    task_id,
                    json.dumps(assignment_data)
                )
                
                # Update agent's active tasks
                agent_key = RedisKeys.AGENT_STATUS.format(agent_id=agent_id)
                active_tasks_data = await self.redis_client.hget(agent_key, "active_tasks")
                active_tasks = json.loads(active_tasks_data or "[]")
                
                if task_id not in active_tasks:
                    active_tasks.append(task_id)
                    await self.redis_client.hset(agent_key, "active_tasks", json.dumps(active_tasks))
                
                # Remove from assignment queue
                queue_items = await self.redis_client.zrange(RedisKeys.TASK_ASSIGNMENT_QUEUE, 0, -1)
                for item in queue_items:
                    item_data = json.loads(item)
                    if item_data.get("task_id") == task_id:
                        await self.redis_client.zrem(RedisKeys.TASK_ASSIGNMENT_QUEUE, item)
                        break
                
                # Publish assignment event
                assignment_event = CoordinationEvent(
                    event_id=str(uuid.uuid4()),
                    event_type="task_assigned",
                    timestamp=datetime.utcnow(),
                    source_agent_id=agent_id,
                    payload=assignment_data
                )
                
                await self.publish_event(RedisChannels.TASK_ASSIGNMENTS, assignment_event)
                
                # WebSocket notification
                task_update = TaskStatusUpdate(
                    task_id=task_id,
                    task_title=task_id,  # Would get actual title
                    old_status="pending",
                    new_status="assigned",
                    assigned_agent_id=agent_id,
                    assigned_agent_name=agent_id  # Would get actual name
                )
                
                await self.publish_websocket_message(task_update)
            
            logger.info("Task assigned to agent", task_id=task_id, agent_id=agent_id)
            
        except Exception as e:
            logger.error("Failed to assign task", task_id=task_id, agent_id=agent_id, error=str(e))
    
    # =====================================================================================
    # REAL-TIME MESSAGING AND EVENTS
    # =====================================================================================
    
    async def publish_event(self, channel: str, event: CoordinationEvent) -> None:
        """Publish coordination event to Redis channel."""
        if not self.circuit_breaker.should_allow_request():
            return
        
        try:
            await self.redis_client.publish(channel, json.dumps(event.to_dict()))
            logger.debug("Event published", channel=channel, event_type=event.event_type)
            
        except Exception as e:
            logger.error("Failed to publish event", channel=channel, error=str(e))
    
    async def publish_websocket_message(self, message: Union[AgentStatusUpdate, TaskStatusUpdate, SystemMetricsUpdate]) -> None:
        """Publish message for WebSocket broadcast."""
        try:
            message_data = {
                "type": message.type,
                "data": message.dict(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.publish(RedisChannels.WEBSOCKET_BROADCAST, json.dumps(message_data))
            
        except Exception as e:
            logger.error("Failed to publish WebSocket message", error=str(e))
    
    async def subscribe_to_channel(self, channel: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to Redis channel with message handler."""
        try:
            if channel not in self.message_handlers:
                self.message_handlers[channel] = []
            
            self.message_handlers[channel].append(handler)
            
            if channel not in self.active_subscriptions:
                # Start subscription task
                task = asyncio.create_task(self._channel_subscription_loop(channel))
                self.background_tasks.add(task)
                self.active_subscriptions.add(channel)
            
            logger.info("Subscribed to channel", channel=channel)
            
        except Exception as e:
            logger.error("Failed to subscribe to channel", channel=channel, error=str(e))
    
    async def _channel_subscription_loop(self, channel: str) -> None:
        """Background loop for channel subscription."""
        try:
            pubsub = self.pubsub_client.pubsub()
            await pubsub.subscribe(channel)
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        
                        # Call all handlers for this channel
                        for handler in self.message_handlers.get(channel, []):
                            try:
                                await handler(data) if asyncio.iscoroutinefunction(handler) else handler(data)
                            except Exception as handler_error:
                                logger.error("Message handler error", 
                                           channel=channel, error=str(handler_error))
                        
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in message", channel=channel)
                        
        except Exception as e:
            logger.error("Channel subscription error", channel=channel, error=str(e))
        finally:
            await pubsub.unsubscribe(channel)
            self.active_subscriptions.discard(channel)
    
    # =====================================================================================
    # PERFORMANCE METRICS AND MONITORING
    # =====================================================================================
    
    async def record_performance_metric(self, metric_type: str, agent_id: Optional[str], value: float, metadata: Dict[str, Any] = None) -> None:
        """Record performance metric for later aggregation."""
        metric_data = {
            "metric_type": metric_type,
            "agent_id": agent_id,
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        # Add to metrics buffer
        self.metrics_buffer.append(metric_data)
        
        # Flush if buffer is full
        if len(self.metrics_buffer) >= 100:
            await self._flush_metrics_buffer()
    
    async def _flush_metrics_buffer(self) -> None:
        """Flush metrics buffer to Redis."""
        if not self.metrics_buffer:
            return
        
        try:
            # Store metrics in Redis with timestamp-based keys
            timestamp = datetime.utcnow()
            period_key = RedisKeys.PERFORMANCE_HISTORY.format(
                period=timestamp.strftime("%Y%m%d%H")  # Hourly buckets
            )
            
            # Use Redis pipeline for efficiency
            pipe = self.redis_client.pipeline()
            
            for metric in self.metrics_buffer:
                pipe.lpush(period_key, json.dumps(metric))
            
            # Set expiration for old metrics (7 days)
            pipe.expire(period_key, timedelta(days=7))
            
            await pipe.execute()
            
            logger.debug("Flushed metrics buffer", count=len(self.metrics_buffer))
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error("Failed to flush metrics buffer", error=str(e))
    
    async def _metrics_flush_loop(self) -> None:
        """Background loop to flush metrics periodically."""
        while True:
            try:
                await asyncio.sleep(30)  # Flush every 30 seconds
                await self._flush_metrics_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics flush loop error", error=str(e))
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system coordination metrics."""
        try:
            # Get active agents count
            active_agents = await self.redis_client.scard(RedisKeys.ACTIVE_AGENTS)
            
            # Get queued tasks count
            queued_tasks = await self.redis_client.zcard(RedisKeys.TASK_ASSIGNMENT_QUEUE)
            
            # Get assigned tasks count
            assigned_tasks = await self.redis_client.hlen(RedisKeys.TASK_ASSIGNMENTS)
            
            # Calculate average workload
            agent_keys = await self.redis_client.smembers(RedisKeys.ACTIVE_AGENTS)
            total_workload = 0.0
            agent_count = 0
            
            for agent_id in agent_keys:
                agent_key = RedisKeys.AGENT_STATUS.format(agent_id=agent_id)
                workload = await self.redis_client.hget(agent_key, "current_workload")
                if workload:
                    total_workload += float(workload)
                    agent_count += 1
            
            avg_workload = total_workload / agent_count if agent_count > 0 else 0.0
            
            return {
                "active_agents": active_agents,
                "queued_tasks": queued_tasks,
                "assigned_tasks": assigned_tasks,
                "average_workload": avg_workload,
                "system_utilization": avg_workload * 100,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get system metrics", error=str(e))
            return {}
    
    # =====================================================================================
    # UTILITY METHODS
    # =====================================================================================
    
    async def _system_monitoring_loop(self) -> None:
        """Background system monitoring and alerting."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                metrics = await self.get_system_metrics()
                
                # Check for alerts
                alerts = []
                
                # High system utilization alert
                if metrics.get("system_utilization", 0) > 90:
                    alerts.append("High system utilization detected")
                
                # Queue backup alert
                if metrics.get("queued_tasks", 0) > 50:
                    alerts.append("Task queue backup detected")
                
                # No active agents alert
                if metrics.get("active_agents", 0) == 0:
                    alerts.append("No active agents available")
                
                # Publish alerts if any
                for alert_msg in alerts:
                    alert = CoordinationAlert(
                        severity="warning",
                        alert_type="system_monitoring",
                        message=alert_msg,
                        timestamp=datetime.utcnow()
                    )
                    await self.publish_websocket_message(alert)
                
                # Publish system metrics
                metrics_update = SystemMetricsUpdate(
                    metrics=metrics,
                    alerts=alerts
                )
                await self.publish_websocket_message(metrics_update)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("System monitoring loop error", error=str(e))
    
    async def _cleanup_expired_data_loop(self) -> None:
        """Background cleanup of expired coordination data."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up expired agent states
                agent_keys = await self.redis_client.keys(RedisKeys.AGENT_STATUS.format(agent_id="*"))
                expired_agents = []
                
                for key in agent_keys:
                    ttl = await self.redis_client.ttl(key)
                    if ttl < 0:  # No expiration set or expired
                        agent_id = key.split(":")[-1]
                        expired_agents.append(agent_id)
                
                # Remove expired agents from active set
                if expired_agents:
                    await self.redis_client.srem(RedisKeys.ACTIVE_AGENTS, *expired_agents)
                    logger.info("Cleaned up expired agents", count=len(expired_agents))
                
                # Clean up old metrics
                current_hour = datetime.utcnow().strftime("%Y%m%d%H")
                pattern = RedisKeys.PERFORMANCE_HISTORY.format(period="*")
                metric_keys = await self.redis_client.keys(pattern)
                
                for key in metric_keys:
                    # Extract period from key
                    period = key.split(":")[-1]
                    try:
                        period_dt = datetime.strptime(period, "%Y%m%d%H")
                        if (datetime.utcnow() - period_dt).days > 7:
                            await self.redis_client.delete(key)
                    except ValueError:
                        continue
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))
    
    @asynccontextmanager
    async def _distributed_lock(self, lock_key: str, timeout: int = 30):
        """Distributed lock implementation using Redis."""
        lock_value = str(uuid.uuid4())
        acquired = False
        
        try:
            # Try to acquire lock
            acquired = await self.redis_client.set(
                lock_key, lock_value, nx=True, ex=timeout
            )
            
            if not acquired:
                raise Exception(f"Could not acquire lock: {lock_key}")
            
            yield
            
        finally:
            if acquired:
                # Release lock only if we own it
                lua_script = """
                if redis.call("get", KEYS[1]) == ARGV[1] then
                    return redis.call("del", KEYS[1])
                else
                    return 0
                end
                """
                await self.redis_client.eval(lua_script, 1, lock_key, lock_value)
    
    def _calculate_priority_score(self, priority: str) -> float:
        """Calculate numeric priority score for task queuing."""
        priority_scores = {
            "low": 1.0,
            "medium": 5.0, 
            "high": 8.0,
            "critical": 10.0
        }
        return priority_scores.get(priority.lower(), 5.0)


# =====================================================================================
# GLOBAL SERVICE INSTANCE
# =====================================================================================

# Global service instance
_coordination_redis_service: Optional[TeamCoordinationRedisService] = None


async def get_coordination_redis_service() -> TeamCoordinationRedisService:
    """Get the global coordination Redis service instance."""
    global _coordination_redis_service
    
    if _coordination_redis_service is None:
        _coordination_redis_service = TeamCoordinationRedisService()
        await _coordination_redis_service.initialize()
    
    return _coordination_redis_service


async def cleanup_coordination_redis_service() -> None:
    """Cleanup the global coordination Redis service."""
    global _coordination_redis_service
    
    if _coordination_redis_service is not None:
        await _coordination_redis_service.cleanup()
        _coordination_redis_service = None