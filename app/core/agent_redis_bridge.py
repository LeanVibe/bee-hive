"""
Agent Redis Bridge for LeanVibe Agent Hive 2.0

Provides Redis Stream communication bridge between the orchestrator and
individual agents running in tmux sessions. Handles message routing,
task distribution, and result collection.

Features:
- Agent registration and heartbeat management
- Task distribution to appropriate agents
- Result collection and status updates
- Load balancing across available agents
- Automatic failover and recovery
- Message persistence and replay
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

from .config import settings
from .enhanced_redis_streams_manager import EnhancedRedisStreamsManager, ConsumerGroupType
from ..models.agent import Agent, AgentStatus

logger = structlog.get_logger()


class MessageType(Enum):
    """Types of messages exchanged between orchestrator and agents."""
    # Orchestrator to Agent
    TASK_ASSIGNMENT = "task_assignment"
    CONFIG_UPDATE = "config_update"
    SHUTDOWN_REQUEST = "shutdown_request"
    HEALTH_CHECK = "health_check"
    
    # Agent to Orchestrator
    AGENT_READY = "agent_ready"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    HEARTBEAT = "heartbeat"
    LOG_MESSAGE = "log_message"
    STATUS_UPDATE = "status_update"


class Priority(Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class AgentMessage:
    """Message structure for agent communication."""
    id: str
    type: MessageType
    sender_id: str
    target_id: Optional[str]
    payload: Dict[str, Any]
    priority: Priority = Priority.NORMAL
    timestamp: datetime = None
    correlation_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["type"] = self.type.value
        result["priority"] = self.priority.value
        result["timestamp"] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        data["type"] = MessageType(data["type"])
        data["priority"] = Priority(data["priority"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class AgentRegistration:
    """Agent registration information."""
    agent_id: str
    agent_type: str
    session_name: str
    capabilities: List[str]
    consumer_group: str
    workspace_path: str
    registered_at: datetime = None
    last_heartbeat: datetime = None
    status: AgentStatus = AgentStatus.ACTIVE
    
    def __post_init__(self):
        if self.registered_at is None:
            self.registered_at = datetime.utcnow()
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["registered_at"] = self.registered_at.isoformat()
        result["last_heartbeat"] = self.last_heartbeat.isoformat()
        result["status"] = self.status.value
        return result


class AgentRedisBridge:
    """
    Redis Stream bridge for agent communication with the orchestrator.
    
    Manages bidirectional communication between the orchestrator and individual
    agents running in tmux sessions using Redis Streams for reliable message delivery.
    """
    
    def __init__(self, redis_manager: EnhancedRedisStreamsManager):
        self.redis_manager = redis_manager
        self.redis_client = redis_manager.redis_client
        
        # Stream names
        self.orchestrator_stream = "orchestrator_commands"
        self.agent_responses_stream = "agent_responses"
        self.agent_heartbeats_stream = "agent_heartbeats"
        
        # Agent registry
        self.registered_agents: Dict[str, AgentRegistration] = {}
        
        # Message handlers
        self.message_handlers: Dict[MessageType, Callable] = {}
        
        # Task routing
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        
        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.message_processor_task: Optional[asyncio.Task] = None
        self.health_monitor_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "failed_deliveries": 0,
            "active_agents": 0,
            "tasks_distributed": 0,
            "average_response_time": 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize the Redis bridge."""
        logger.info("🌉 Initializing Agent Redis Bridge...")
        
        try:
            # Ensure Redis streams exist
            await self._ensure_streams_exist()
            
            # Start background tasks
            self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor_loop())
            self.message_processor_task = asyncio.create_task(self._message_processor_loop())
            self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            
            # Register default message handlers
            self._register_default_handlers()
            
            logger.info("✅ Agent Redis Bridge initialized successfully")
            
        except Exception as e:
            logger.error("❌ Failed to initialize Agent Redis Bridge", error=str(e))
            raise
    
    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        session_name: str,
        capabilities: List[str],
        consumer_group: str,
        workspace_path: str
    ) -> bool:
        """
        Register an agent with the bridge.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (claude-code, cursor, etc.)
            session_name: Tmux session name
            capabilities: List of agent capabilities
            consumer_group: Redis consumer group for the agent
            workspace_path: Agent workspace directory
            
        Returns:
            True if registration successful
        """
        try:
            registration = AgentRegistration(
                agent_id=agent_id,
                agent_type=agent_type,
                session_name=session_name,
                capabilities=capabilities,
                consumer_group=consumer_group,
                workspace_path=workspace_path
            )
            
            self.registered_agents[agent_id] = registration
            self.metrics["active_agents"] = len(self.registered_agents)
            
            # Send agent ready message
            await self.send_message_to_orchestrator(AgentMessage(
                id=str(uuid.uuid4()),
                type=MessageType.AGENT_READY,
                sender_id=agent_id,
                target_id="orchestrator",
                payload={
                    "agent_type": agent_type,
                    "capabilities": capabilities,
                    "session_name": session_name,
                    "workspace_path": workspace_path
                }
            ))
            
            logger.info(
                "🤖 Agent registered successfully",
                agent_id=agent_id,
                agent_type=agent_type,
                session_name=session_name,
                capabilities=capabilities
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "❌ Failed to register agent",
                agent_id=agent_id,
                error=str(e)
            )
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the bridge."""
        try:
            if agent_id in self.registered_agents:
                del self.registered_agents[agent_id]
                self.metrics["active_agents"] = len(self.registered_agents)
                
                # Clean up any pending task assignments
                tasks_to_remove = [
                    task_id for task_id, assigned_agent_id in self.task_assignments.items()
                    if assigned_agent_id == agent_id
                ]
                for task_id in tasks_to_remove:
                    del self.task_assignments[task_id]
                
                logger.info("Agent unregistered successfully", agent_id=agent_id)
                return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to unregister agent", agent_id=agent_id, error=str(e))
            return False
    
    async def send_message_to_agent(
        self,
        agent_id: str,
        message: AgentMessage
    ) -> bool:
        """
        Send a message to a specific agent.
        
        Args:
            agent_id: Target agent identifier
            message: Message to send
            
        Returns:
            True if message was sent successfully
        """
        try:
            if agent_id not in self.registered_agents:
                logger.warning("Attempted to send message to unregistered agent", agent_id=agent_id)
                return False
            
            agent_reg = self.registered_agents[agent_id]
            
            # Add message to the appropriate stream
            message_data = message.to_dict()
            message_data["target_agent_id"] = agent_id
            message_data["consumer_group"] = agent_reg.consumer_group
            
            await self.redis_client.xadd(
                self.orchestrator_stream,
                message_data
            )
            
            self.metrics["messages_sent"] += 1
            
            logger.debug(
                "Message sent to agent",
                agent_id=agent_id,
                message_type=message.type.value,
                message_id=message.id
            )
            
            return True
            
        except Exception as e:
            self.metrics["failed_deliveries"] += 1
            logger.error(
                "Failed to send message to agent",
                agent_id=agent_id,
                message_type=message.type.value,
                error=str(e)
            )
            return False
    
    async def send_message_to_orchestrator(self, message: AgentMessage) -> bool:
        """
        Send a message from an agent to the orchestrator.
        
        Args:
            message: Message to send
            
        Returns:
            True if message was sent successfully
        """
        try:
            message_data = message.to_dict()
            
            await self.redis_client.xadd(
                self.agent_responses_stream,
                message_data
            )
            
            self.metrics["messages_sent"] += 1
            
            logger.debug(
                "Message sent to orchestrator",
                sender_id=message.sender_id,
                message_type=message.type.value,
                message_id=message.id
            )
            
            return True
            
        except Exception as e:
            self.metrics["failed_deliveries"] += 1
            logger.error(
                "Failed to send message to orchestrator",
                sender_id=message.sender_id,
                message_type=message.type.value,
                error=str(e)
            )
            return False
    
    async def assign_task_to_agent(
        self,
        task_id: str,
        task_description: str,
        required_capabilities: List[str],
        priority: Priority = Priority.NORMAL,
        preferred_agent_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Assign a task to the most suitable available agent.
        
        Args:
            task_id: Unique task identifier
            task_description: Description of the task
            required_capabilities: Capabilities required for the task
            priority: Task priority level
            preferred_agent_id: Preferred agent ID (if any)
            
        Returns:
            Agent ID if task was assigned successfully, None otherwise
        """
        try:
            # Find suitable agent
            target_agent_id = await self._find_suitable_agent(
                required_capabilities,
                preferred_agent_id
            )
            
            if not target_agent_id:
                logger.warning(
                    "No suitable agent found for task",
                    task_id=task_id,
                    required_capabilities=required_capabilities
                )
                return None
            
            # Create task assignment message
            task_message = AgentMessage(
                id=str(uuid.uuid4()),
                type=MessageType.TASK_ASSIGNMENT,
                sender_id="orchestrator",
                target_id=target_agent_id,
                payload={
                    "task_id": task_id,
                    "description": task_description,
                    "required_capabilities": required_capabilities,
                    "assigned_at": datetime.utcnow().isoformat()
                },
                priority=priority,
                correlation_id=task_id
            )
            
            # Send message to agent
            success = await self.send_message_to_agent(target_agent_id, task_message)
            
            if success:
                self.task_assignments[task_id] = target_agent_id
                self.metrics["tasks_distributed"] += 1
                
                logger.info(
                    "Task assigned to agent",
                    task_id=task_id,
                    agent_id=target_agent_id,
                    priority=priority.value
                )
                
                return target_agent_id
            
            return None
            
        except Exception as e:
            logger.error(
                "Failed to assign task to agent",
                task_id=task_id,
                error=str(e)
            )
            return None
    
    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a specific agent."""
        if agent_id not in self.registered_agents:
            return None
        
        registration = self.registered_agents[agent_id]
        
        # Check if agent is responsive
        time_since_heartbeat = (datetime.utcnow() - registration.last_heartbeat).total_seconds()
        is_responsive = time_since_heartbeat < 60  # 1 minute threshold
        
        # Get assigned tasks
        assigned_tasks = [
            task_id for task_id, assigned_agent_id in self.task_assignments.items()
            if assigned_agent_id == agent_id
        ]
        
        return {
            "agent_id": agent_id,
            "registration": registration.to_dict(),
            "is_responsive": is_responsive,
            "time_since_heartbeat": time_since_heartbeat,
            "assigned_tasks": assigned_tasks,
            "task_count": len(assigned_tasks)
        }
    
    async def get_bridge_metrics(self) -> Dict[str, Any]:
        """Get comprehensive bridge metrics."""
        return {
            "metrics": self.metrics,
            "registered_agents": len(self.registered_agents),
            "active_tasks": len(self.task_assignments),
            "agent_details": [
                await self.get_agent_status(agent_id)
                for agent_id in self.registered_agents.keys()
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Private helper methods
    
    async def _ensure_streams_exist(self) -> None:
        """Ensure required Redis streams exist."""
        streams = [
            self.orchestrator_stream,
            self.agent_responses_stream,
            self.agent_heartbeats_stream
        ]
        
        for stream_name in streams:
            try:
                # Try to get stream info, create if doesn't exist
                await self.redis_client.xinfo_stream(stream_name)
            except RedisError:
                # Stream doesn't exist, create it
                await self.redis_client.xadd(
                    stream_name,
                    {"initialized": "true", "timestamp": datetime.utcnow().isoformat()}
                )
                logger.debug(f"Created Redis stream: {stream_name}")
    
    async def _find_suitable_agent(
        self,
        required_capabilities: List[str],
        preferred_agent_id: Optional[str] = None
    ) -> Optional[str]:
        """Find the most suitable agent for a task."""
        # If preferred agent is specified and available, use it
        if preferred_agent_id and preferred_agent_id in self.registered_agents:
            agent_reg = self.registered_agents[preferred_agent_id]
            if agent_reg.status == AgentStatus.ACTIVE:
                return preferred_agent_id
        
        # Find best match based on capabilities
        best_match = None
        best_score = 0.0
        
        for agent_id, registration in self.registered_agents.items():
            if registration.status != AgentStatus.ACTIVE:
                continue
            
            # Calculate capability match score
            score = self._calculate_capability_match(
                registration.capabilities,
                required_capabilities
            )
            
            # Consider current workload (fewer tasks = higher score)
            assigned_task_count = sum(1 for aid in self.task_assignments.values() if aid == agent_id)
            workload_penalty = assigned_task_count * 0.1
            adjusted_score = max(0, score - workload_penalty)
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_match = agent_id
        
        return best_match
    
    def _calculate_capability_match(
        self,
        agent_capabilities: List[str],
        required_capabilities: List[str]
    ) -> float:
        """Calculate how well agent capabilities match requirements."""
        if not required_capabilities:
            return 1.0  # No specific requirements
        
        if not agent_capabilities:
            return 0.0  # No capabilities
        
        matches = 0
        for req_cap in required_capabilities:
            for agent_cap in agent_capabilities:
                if req_cap.lower() in agent_cap.lower():
                    matches += 1
                    break
        
        return matches / len(required_capabilities)
    
    async def _heartbeat_monitor_loop(self) -> None:
        """Monitor agent heartbeats and update status."""
        while True:
            try:
                current_time = datetime.utcnow()
                
                for agent_id, registration in list(self.registered_agents.items()):
                    time_since_heartbeat = (current_time - registration.last_heartbeat).total_seconds()
                    
                    # Mark agents as inactive if no heartbeat for 2 minutes
                    if time_since_heartbeat > 120:
                        if registration.status == AgentStatus.ACTIVE:
                            registration.status = AgentStatus.INACTIVE
                            logger.warning(
                                "Agent marked as inactive due to missed heartbeats",
                                agent_id=agent_id,
                                time_since_heartbeat=time_since_heartbeat
                            )
                    
                    # Remove agents that haven't been seen for 10 minutes
                    elif time_since_heartbeat > 600:
                        await self.unregister_agent(agent_id)
                        logger.info(
                            "Agent unregistered due to prolonged inactivity",
                            agent_id=agent_id
                        )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor loop: {e}")
                await asyncio.sleep(60)
    
    async def _message_processor_loop(self) -> None:
        """Process incoming messages from agents."""
        while True:
            try:
                # Read messages from agent responses stream
                messages = await self.redis_client.xread(
                    {self.agent_responses_stream: "$"},
                    count=10,
                    block=1000
                )
                
                for stream_name, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        await self._process_agent_message(message_id, fields)
                        self.metrics["messages_received"] += 1
                
            except Exception as e:
                logger.error(f"Error in message processor loop: {e}")
                await asyncio.sleep(5)
    
    async def _process_agent_message(self, message_id: str, fields: Dict[str, str]) -> None:
        """Process a single message from an agent."""
        try:
            # Reconstruct message object
            message_data = {k.decode() if isinstance(k, bytes) else k: 
                          v.decode() if isinstance(v, bytes) else v 
                          for k, v in fields.items()}
            
            message_type = MessageType(message_data.get("type"))
            sender_id = message_data.get("sender_id")
            
            # Update agent heartbeat if this is a heartbeat message
            if message_type == MessageType.HEARTBEAT:
                if sender_id in self.registered_agents:
                    self.registered_agents[sender_id].last_heartbeat = datetime.utcnow()
                    self.registered_agents[sender_id].status = AgentStatus.ACTIVE
            
            # Handle other message types
            elif message_type in self.message_handlers:
                await self.message_handlers[message_type](message_data)
            
            logger.debug(
                "Processed agent message",
                message_id=message_id,
                message_type=message_type.value,
                sender_id=sender_id
            )
            
        except Exception as e:
            logger.error(
                "Failed to process agent message",
                message_id=message_id,
                error=str(e)
            )
    
    async def _health_monitor_loop(self) -> None:
        """Monitor overall system health and performance."""
        while True:
            try:
                # Update metrics
                self.metrics["active_agents"] = len([
                    r for r in self.registered_agents.values()
                    if r.status == AgentStatus.ACTIVE
                ])
                
                # Log health summary
                if self.metrics["active_agents"] > 0:
                    logger.debug(
                        "Agent bridge health summary",
                        active_agents=self.metrics["active_agents"],
                        messages_sent=self.metrics["messages_sent"],
                        messages_received=self.metrics["messages_received"],
                        failed_deliveries=self.metrics["failed_deliveries"]
                    )
                
                await asyncio.sleep(60)  # Health check every minute
                
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(60)
    
    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        async def handle_task_progress(message_data: Dict[str, Any]) -> None:
            logger.info(
                "Task progress update",
                task_id=message_data.get("payload", {}).get("task_id"),
                sender_id=message_data.get("sender_id"),
                progress=message_data.get("payload", {}).get("progress")
            )
        
        async def handle_task_completed(message_data: Dict[str, Any]) -> None:
            task_id = message_data.get("payload", {}).get("task_id")
            if task_id in self.task_assignments:
                del self.task_assignments[task_id]
            
            logger.info(
                "Task completed",
                task_id=task_id,
                sender_id=message_data.get("sender_id")
            )
        
        async def handle_task_failed(message_data: Dict[str, Any]) -> None:
            task_id = message_data.get("payload", {}).get("task_id")
            if task_id in self.task_assignments:
                del self.task_assignments[task_id]
            
            logger.warning(
                "Task failed",
                task_id=task_id,
                sender_id=message_data.get("sender_id"),
                error=message_data.get("payload", {}).get("error")
            )
        
        self.message_handlers = {
            MessageType.TASK_PROGRESS: handle_task_progress,
            MessageType.TASK_COMPLETED: handle_task_completed,
            MessageType.TASK_FAILED: handle_task_failed
        }
    
    async def shutdown(self) -> None:
        """Shutdown the Redis bridge."""
        logger.info("🛑 Shutting down Agent Redis Bridge...")
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.message_processor_task:
            self.message_processor_task.cancel()
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
        
        # Send shutdown messages to all registered agents
        for agent_id in list(self.registered_agents.keys()):
            shutdown_message = AgentMessage(
                id=str(uuid.uuid4()),
                type=MessageType.SHUTDOWN_REQUEST,
                sender_id="orchestrator",
                target_id=agent_id,
                payload={"reason": "orchestrator_shutdown"}
            )
            await self.send_message_to_agent(agent_id, shutdown_message)
        
        logger.info("✅ Agent Redis Bridge shutdown complete")


# Factory function
async def create_agent_redis_bridge(
    redis_manager: Optional[EnhancedRedisStreamsManager] = None
) -> AgentRedisBridge:
    """Create and initialize AgentRedisBridge."""
    if redis_manager is None:
        redis_manager = EnhancedRedisStreamsManager()
    
    bridge = AgentRedisBridge(redis_manager)
    await bridge.initialize()
    return bridge