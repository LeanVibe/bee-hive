"""
Python-based Hook Lifecycle System for LeanVibe Agent Hive 2.0.

This system provides comprehensive lifecycle event tracking with Python-based hooks,
security safeguards, event aggregation, and real-time streaming capabilities.

Features:
- Python-based hook processing with async support
- SecurityValidator for dangerous command detection
- Event aggregation and batching for high-frequency events
- Real-time WebSocket streaming for dashboard integration
- Redis Streams integration for event buffering
- Comprehensive performance monitoring and logging
- Configurable dangerous command blocking
"""

import asyncio
import json
import time
import uuid
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging

import structlog
import redis.asyncio as redis
from fastapi import WebSocket

from ..models.observability import EventType, AgentEvent
from ..core.database import get_async_session
from ..core.redis import get_redis
from ..core.config import get_settings
from .enhanced_lifecycle_hooks import EnhancedEventType, LifecycleEventData
from .enhanced_security_safeguards import ControlDecision, SecurityContext, AgentBehaviorState

logger = structlog.get_logger()


class HookType(str, Enum):
    """Types of hooks supported by the system."""
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"
    NOTIFICATION = "Notification"
    SUBAGENT_STOP = "SubagentStop"
    AGENT_START = "AgentStart"
    AGENT_STOP = "AgentStop"
    ERROR = "Error"


class SecurityRisk(str, Enum):
    """Security risk levels for commands."""
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class DangerousCommand:
    """Configuration for dangerous command detection."""
    pattern: str
    risk_level: SecurityRisk
    description: str
    block_execution: bool = True
    require_approval: bool = False
    
    def matches(self, command: str) -> bool:
        """Check if command matches this dangerous pattern."""
        try:
            return bool(re.search(self.pattern, command, re.IGNORECASE))
        except re.error:
            logger.error(f"Invalid regex pattern: {self.pattern}")
            return False


@dataclass
class HookEvent:
    """Standardized hook event data structure."""
    hook_type: HookType
    agent_id: uuid.UUID
    session_id: Optional[uuid.UUID]
    timestamp: datetime
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    priority: int = 5  # 1=highest, 10=lowest
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hook_type": self.hook_type.value,
            "agent_id": str(self.agent_id),
            "session_id": str(self.session_id) if self.session_id else None,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "priority": self.priority,
            "metadata": self.metadata
        }


@dataclass
class HookProcessingResult:
    """Result of hook processing."""
    success: bool
    processing_time_ms: float
    error: Optional[str] = None
    security_decision: Optional[ControlDecision] = None
    blocked_reason: Optional[str] = None
    event_id: Optional[str] = None


class SecurityValidator:
    """
    Security validator for dangerous command detection and blocking.
    
    Provides configurable rules for detecting and blocking dangerous commands
    with support for different risk levels and response actions.
    """
    
    def __init__(self):
        self.dangerous_commands: List[DangerousCommand] = []
        self.custom_validators: List[Callable[[str], Tuple[bool, SecurityRisk, str]]] = []
        self.validation_cache: Dict[str, Tuple[bool, SecurityRisk, str, datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)
        
        # Performance metrics
        self.metrics = {
            "validations_performed": 0,
            "commands_blocked": 0,
            "approvals_required": 0,
            "cache_hits": 0,
            "avg_validation_time_ms": 0.0
        }
        
        # Initialize default dangerous commands
        self._initialize_dangerous_commands()
    
    def _initialize_dangerous_commands(self) -> None:
        """Initialize default dangerous command patterns."""
        
        # Critical risk commands
        self.dangerous_commands.extend([
            DangerousCommand(
                pattern=r'rm\s+-rf\s*/',
                risk_level=SecurityRisk.CRITICAL,
                description="Recursive delete from root directory",
                block_execution=True
            ),
            DangerousCommand(
                pattern=r'sudo\s+rm\s+-rf',
                risk_level=SecurityRisk.CRITICAL,
                description="Sudo recursive delete",
                block_execution=True
            ),
            DangerousCommand(
                pattern=r'mkfs\.',
                risk_level=SecurityRisk.CRITICAL,
                description="Format filesystem",
                block_execution=True
            ),
            DangerousCommand(
                pattern=r'dd\s+if=.*of=/dev/',
                risk_level=SecurityRisk.CRITICAL,
                description="Direct disk write operations",
                block_execution=True
            ),
            DangerousCommand(
                pattern=r'fork\s*\(\s*\)|while\s*:\s*fork',
                risk_level=SecurityRisk.CRITICAL,
                description="Fork bomb patterns",
                block_execution=True
            )
        ])
        
        # High risk commands
        self.dangerous_commands.extend([
            DangerousCommand(
                pattern=r'sudo\s+',
                risk_level=SecurityRisk.HIGH,
                description="Sudo commands",
                block_execution=False,
                require_approval=True
            ),
            DangerousCommand(
                pattern=r'chmod\s+777',
                risk_level=SecurityRisk.HIGH,
                description="Dangerous permission changes",
                block_execution=False,
                require_approval=True
            ),
            DangerousCommand(
                pattern=r'iptables|ufw|firewall',
                risk_level=SecurityRisk.HIGH,
                description="Firewall modifications",
                block_execution=False,
                require_approval=True
            ),
            DangerousCommand(
                pattern=r'curl.*\|\s*sh|wget.*\|\s*sh',
                risk_level=SecurityRisk.HIGH,
                description="Download and execute scripts",
                block_execution=False,
                require_approval=True
            )
        ])
        
        # Medium risk commands
        self.dangerous_commands.extend([
            DangerousCommand(
                pattern=r'rm\s+-f\s+/.*',
                risk_level=SecurityRisk.MEDIUM,
                description="Force delete system files",
                block_execution=False,
                require_approval=True
            ),
            DangerousCommand(
                pattern=r'crontab\s+-r',
                risk_level=SecurityRisk.MEDIUM,
                description="Remove all cron jobs",
                block_execution=False,
                require_approval=True
            ),
            DangerousCommand(
                pattern=r'kill\s+-9\s+1',
                risk_level=SecurityRisk.MEDIUM,
                description="Kill init process",
                block_execution=False,
                require_approval=True
            )
        ])
        
        logger.info(f"Initialized {len(self.dangerous_commands)} dangerous command patterns")
    
    def add_dangerous_command(self, command: DangerousCommand) -> None:
        """Add a new dangerous command pattern."""
        self.dangerous_commands.append(command)
        # Sort by risk level (critical first)
        risk_order = {
            SecurityRisk.CRITICAL: 1,
            SecurityRisk.HIGH: 2,
            SecurityRisk.MEDIUM: 3,
            SecurityRisk.LOW: 4,
            SecurityRisk.SAFE: 5
        }
        self.dangerous_commands.sort(key=lambda cmd: risk_order[cmd.risk_level])
        logger.info(f"Added dangerous command pattern: {command.description}")
    
    def add_custom_validator(
        self,
        validator: Callable[[str], Tuple[bool, SecurityRisk, str]]
    ) -> None:
        """Add a custom validator function."""
        self.custom_validators.append(validator)
        logger.info("Added custom security validator")
    
    async def validate_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, SecurityRisk, str]:
        """
        Validate a command for security risks.
        
        Args:
            command: The command to validate
            context: Optional context information
            
        Returns:
            Tuple of (is_safe, risk_level, reason)
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(command, context)
            cached_result = self._get_cached_validation(cache_key)
            if cached_result:
                self.metrics["cache_hits"] += 1
                return cached_result[0], cached_result[1], cached_result[2]
            
            # Check against dangerous command patterns
            for dangerous_cmd in self.dangerous_commands:
                if dangerous_cmd.matches(command):
                    result = (not dangerous_cmd.block_execution, dangerous_cmd.risk_level, dangerous_cmd.description)
                    self._cache_validation(cache_key, result)
                    self._update_metrics(result, start_time)
                    return result
            
            # Run custom validators
            for validator in self.custom_validators:
                try:
                    is_dangerous, risk_level, reason = validator(command)
                    if is_dangerous:
                        result = (False, risk_level, reason)
                        self._cache_validation(cache_key, result)
                        self._update_metrics(result, start_time)
                        return result
                except Exception as e:
                    logger.error(f"Custom validator error: {e}")
            
            # Command appears safe
            result = (True, SecurityRisk.SAFE, "No security risks detected")
            self._cache_validation(cache_key, result)
            self._update_metrics(result, start_time)
            return result
            
        except Exception as e:
            logger.error(f"Command validation error: {e}")
            result = (False, SecurityRisk.HIGH, f"Validation error: {str(e)}")
            self._update_metrics(result, start_time)
            return result
    
    def _generate_cache_key(self, command: str, context: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for validation caching."""
        import hashlib
        key_data = {
            "command": command.strip().lower(),
            "context_hash": hash(str(sorted(context.items()))) if context else None
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _get_cached_validation(self, cache_key: str) -> Optional[Tuple[bool, SecurityRisk, str]]:
        """Get cached validation result if still valid."""
        if cache_key in self.validation_cache:
            is_safe, risk_level, reason, cached_at = self.validation_cache[cache_key]
            if datetime.utcnow() - cached_at < self.cache_ttl:
                return is_safe, risk_level, reason
            else:
                del self.validation_cache[cache_key]
        return None
    
    def _cache_validation(self, cache_key: str, result: Tuple[bool, SecurityRisk, str]) -> None:
        """Cache validation result with timestamp."""
        self.validation_cache[cache_key] = (*result, datetime.utcnow())
        
        # Limit cache size
        if len(self.validation_cache) > 1000:
            sorted_cache = sorted(
                self.validation_cache.items(),
                key=lambda x: x[1][3]  # Sort by timestamp
            )
            self.validation_cache = dict(sorted_cache[-800:])
    
    def _update_metrics(self, result: Tuple[bool, SecurityRisk, str], start_time: float) -> None:
        """Update validation metrics."""
        self.metrics["validations_performed"] += 1
        
        is_safe, risk_level, reason = result
        if not is_safe:
            if risk_level == SecurityRisk.CRITICAL:
                self.metrics["commands_blocked"] += 1
            else:
                self.metrics["approvals_required"] += 1
        
        # Update average validation time
        validation_time = (time.time() - start_time) * 1000
        current_avg = self.metrics["avg_validation_time_ms"]
        total_validations = self.metrics["validations_performed"]
        self.metrics["avg_validation_time_ms"] = (
            (current_avg * (total_validations - 1) + validation_time) / total_validations
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get security validator metrics."""
        return {
            "security_validator": self.metrics.copy(),
            "dangerous_patterns_count": len(self.dangerous_commands),
            "custom_validators_count": len(self.custom_validators),
            "cache_size": len(self.validation_cache)
        }


class EventAggregator:
    """
    Event aggregation and batching system for high-frequency events.
    
    Provides intelligent batching and aggregation to handle high-frequency
    events efficiently while maintaining real-time capabilities.
    """
    
    def __init__(self, batch_size: int = 100, flush_interval_ms: int = 1000):
        self.batch_size = batch_size
        self.flush_interval = timedelta(milliseconds=flush_interval_ms)
        
        # Event batches by priority
        self.event_batches: Dict[int, List[HookEvent]] = defaultdict(list)
        self.last_flush: Dict[int, datetime] = {}
        
        # Aggregation rules
        self.aggregation_rules: Dict[str, Callable[[List[HookEvent]], HookEvent]] = {}
        
        # Performance metrics
        self.metrics = {
            "events_aggregated": 0,
            "batches_processed": 0,
            "aggregation_rules_applied": 0,
            "avg_batch_size": 0.0,
            "flush_operations": 0
        }
        
        # Background flush task
        self.flush_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
    
    async def start(self) -> None:
        """Start the event aggregator."""
        if self.flush_task is None:
            self.flush_task = asyncio.create_task(self._background_flush())
            logger.info("Event aggregator started")
    
    async def stop(self) -> None:
        """Stop the event aggregator."""
        if self.flush_task:
            self.shutdown_event.set()
            await self.flush_task
            self.flush_task = None
            logger.info("Event aggregator stopped")
    
    def add_aggregation_rule(
        self,
        event_type: str,
        aggregator: Callable[[List[HookEvent]], HookEvent]
    ) -> None:
        """Add an aggregation rule for specific event types."""
        self.aggregation_rules[event_type] = aggregator
        logger.info(f"Added aggregation rule for event type: {event_type}")
    
    async def add_event(self, event: HookEvent) -> None:
        """Add an event to the aggregation system."""
        priority = event.priority
        self.event_batches[priority].append(event)
        self.metrics["events_aggregated"] += 1
        
        # Check if batch is ready for processing
        if len(self.event_batches[priority]) >= self.batch_size:
            await self._flush_batch(priority)
    
    async def flush_all(self) -> List[List[HookEvent]]:
        """Flush all batches immediately."""
        all_batches = []
        for priority in list(self.event_batches.keys()):
            if self.event_batches[priority]:
                batch = await self._flush_batch(priority)
                if batch:
                    all_batches.append(batch)
        return all_batches
    
    async def _flush_batch(self, priority: int) -> Optional[List[HookEvent]]:
        """Flush a specific priority batch."""
        if not self.event_batches[priority]:
            return None
        
        batch = self.event_batches[priority].copy()
        self.event_batches[priority].clear()
        self.last_flush[priority] = datetime.utcnow()
        
        # Apply aggregation rules
        aggregated_batch = await self._apply_aggregation_rules(batch)
        
        # Update metrics
        self.metrics["batches_processed"] += 1
        self.metrics["flush_operations"] += 1
        batch_size = len(aggregated_batch)
        current_avg = self.metrics["avg_batch_size"]
        total_batches = self.metrics["batches_processed"]
        self.metrics["avg_batch_size"] = (
            (current_avg * (total_batches - 1) + batch_size) / total_batches
        )
        
        logger.debug(f"Flushed batch with {batch_size} events (priority {priority})")
        return aggregated_batch
    
    async def _apply_aggregation_rules(self, events: List[HookEvent]) -> List[HookEvent]:
        """Apply aggregation rules to event batch."""
        if not self.aggregation_rules:
            return events
        
        # Group events by type
        events_by_type = defaultdict(list)
        for event in events:
            events_by_type[event.hook_type.value].append(event)
        
        aggregated_events = []
        
        for event_type, type_events in events_by_type.items():
            if event_type in self.aggregation_rules:
                try:
                    aggregated_event = self.aggregation_rules[event_type](type_events)
                    aggregated_events.append(aggregated_event)
                    self.metrics["aggregation_rules_applied"] += 1
                except Exception as e:
                    logger.error(f"Aggregation rule error for {event_type}: {e}")
                    aggregated_events.extend(type_events)
            else:
                aggregated_events.extend(type_events)
        
        return aggregated_events
    
    async def _background_flush(self) -> None:
        """Background task to flush batches based on time intervals."""
        while not self.shutdown_event.is_set():
            try:
                now = datetime.utcnow()
                
                for priority in list(self.event_batches.keys()):
                    if not self.event_batches[priority]:
                        continue
                    
                    last_flush_time = self.last_flush.get(priority, datetime.min)
                    if now - last_flush_time >= self.flush_interval:
                        await self._flush_batch(priority)
                
                # Wait for next check or shutdown
                await asyncio.wait_for(
                    self.shutdown_event.wait(),
                    timeout=self.flush_interval.total_seconds()
                )
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Background flush error: {e}")
                await asyncio.sleep(1)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregator metrics."""
        return {
            "event_aggregator": self.metrics.copy(),
            "current_batch_sizes": {
                str(priority): len(events) 
                for priority, events in self.event_batches.items()
            },
            "aggregation_rules_count": len(self.aggregation_rules)
        }


class WebSocketStreamer:
    """
    WebSocket streaming service for real-time dashboard events.
    
    Manages WebSocket connections and streams events to connected
    dashboard clients with filtering and priority handling.
    """
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_filters: Dict[WebSocket, Dict[str, Any]] = {}
        
        # Performance metrics
        self.metrics = {
            "active_connections": 0,
            "messages_sent": 0,
            "connection_errors": 0,
            "filtered_messages": 0,
            "avg_broadcast_time_ms": 0.0
        }
    
    async def connect(self, websocket: WebSocket, filters: Optional[Dict[str, Any]] = None) -> None:
        """Connect a new WebSocket client."""
        await websocket.accept()
        self.active_connections.add(websocket)
        if filters:
            self.connection_filters[websocket] = filters
        
        self.metrics["active_connections"] = len(self.active_connections)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """Disconnect a WebSocket client."""
        self.active_connections.discard(websocket)
        self.connection_filters.pop(websocket, None)
        
        self.metrics["active_connections"] = len(self.active_connections)
        logger.info(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast_event(self, event: HookEvent) -> None:
        """Broadcast an event to all connected clients."""
        if not self.active_connections:
            return
        
        start_time = time.time()
        message = {
            "type": "hook_event",
            "data": event.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        disconnected_clients = set()
        messages_sent = 0
        filtered_count = 0
        
        for websocket in self.active_connections:
            try:
                # Apply filters if configured
                if websocket in self.connection_filters:
                    if not self._should_send_to_client(event, self.connection_filters[websocket]):
                        filtered_count += 1
                        continue
                
                await websocket.send_text(json.dumps(message))
                messages_sent += 1
                
            except Exception as e:
                logger.error(f"WebSocket send error: {e}")
                disconnected_clients.add(websocket)
                self.metrics["connection_errors"] += 1
        
        # Remove disconnected clients
        for client in disconnected_clients:
            await self.disconnect(client)
        
        # Update metrics
        self.metrics["messages_sent"] += messages_sent
        self.metrics["filtered_messages"] += filtered_count
        
        broadcast_time = (time.time() - start_time) * 1000
        current_avg = self.metrics["avg_broadcast_time_ms"]
        total_broadcasts = self.metrics["messages_sent"] // len(self.active_connections) if self.active_connections else 1
        self.metrics["avg_broadcast_time_ms"] = (
            (current_avg * (total_broadcasts - 1) + broadcast_time) / total_broadcasts
        )
        
        if messages_sent > 0:
            logger.debug(f"Broadcasted event to {messages_sent} clients ({filtered_count} filtered)")
    
    def _should_send_to_client(self, event: HookEvent, filters: Dict[str, Any]) -> bool:
        """Check if event should be sent to client based on filters."""
        
        # Agent ID filter
        if "agent_ids" in filters:
            if str(event.agent_id) not in filters["agent_ids"]:
                return False
        
        # Hook type filter
        if "hook_types" in filters:
            if event.hook_type.value not in filters["hook_types"]:
                return False
        
        # Priority filter
        if "min_priority" in filters:
            if event.priority < filters["min_priority"]:
                return False
        
        # Session ID filter
        if "session_ids" in filters and event.session_id:
            if str(event.session_id) not in filters["session_ids"]:
                return False
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get WebSocket streaming metrics."""
        return {
            "websocket_streamer": self.metrics.copy(),
            "connection_count": len(self.active_connections),
            "filtered_connections": len(self.connection_filters)
        }


class HookLifecycleSystem:
    """
    Python-based Hook Lifecycle System with comprehensive event processing.
    
    Orchestrates all hook processing components including security validation,
    event aggregation, WebSocket streaming, and Redis integration.
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Core components
        self.security_validator = SecurityValidator()
        self.event_aggregator = EventAggregator(
            batch_size=self.settings.HOOK_BATCH_SIZE or 100,
            flush_interval_ms=self.settings.HOOK_FLUSH_INTERVAL_MS or 1000
        )
        self.websocket_streamer = WebSocketStreamer()
        
        # Hook processors
        self.hook_processors: Dict[HookType, List[Callable]] = defaultdict(list)
        
        # Redis integration
        self.redis_client: Optional[redis.Redis] = None
        self.redis_streams = {
            "hook_events": "hook_lifecycle_events",
            "security_events": "hook_security_events",
            "performance_events": "hook_performance_events"
        }
        
        # Performance metrics
        self.metrics = {
            "hooks_processed": 0,
            "hooks_blocked": 0,
            "processing_errors": 0,
            "avg_processing_time_ms": 0.0,
            "performance_threshold_violations": 0
        }
        
        # Configuration
        self.config = {
            "enable_security_validation": True,
            "enable_event_aggregation": True,
            "enable_websocket_streaming": True,
            "enable_redis_streaming": True,
            "performance_threshold_ms": 50.0,
            "max_payload_size": 100000  # 100KB
        }
        
        # Initialize system
        self._initialize_default_aggregation_rules()
    
    async def initialize(self) -> None:
        """Initialize the hook lifecycle system."""
        try:
            # Initialize Redis connection
            if self.config["enable_redis_streaming"]:
                self.redis_client = get_redis()
                logger.info("Connected to Redis for hook streaming")
            
            # Start event aggregator
            if self.config["enable_event_aggregation"]:
                await self.event_aggregator.start()
                logger.info("Event aggregator started")
            
            logger.info("Hook Lifecycle System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hook Lifecycle System: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the hook lifecycle system."""
        try:
            # Stop event aggregator
            if self.config["enable_event_aggregation"]:
                await self.event_aggregator.stop()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Hook Lifecycle System shutdown completed")
            
        except Exception as e:
            logger.error(f"Hook Lifecycle System shutdown error: {e}")
    
    def register_hook_processor(
        self,
        hook_type: HookType,
        processor: Callable[[HookEvent], Any]
    ) -> None:
        """Register a hook processor for specific hook types."""
        self.hook_processors[hook_type].append(processor)
        logger.info(f"Registered hook processor for {hook_type.value}")
    
    async def process_hook(
        self,
        hook_type: HookType,
        agent_id: uuid.UUID,
        session_id: Optional[uuid.UUID],
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        priority: int = 5
    ) -> HookProcessingResult:
        """
        Process a hook event through the complete lifecycle.
        
        Args:
            hook_type: Type of hook event
            agent_id: Agent identifier
            session_id: Optional session identifier
            payload: Hook event payload
            correlation_id: Optional correlation identifier
            priority: Event priority (1=highest, 10=lowest)
            
        Returns:
            HookProcessingResult with processing details
        """
        start_time = time.time()
        
        try:
            # Create hook event
            event = HookEvent(
                hook_type=hook_type,
                agent_id=agent_id,
                session_id=session_id,
                timestamp=datetime.utcnow(),
                payload=payload,
                correlation_id=correlation_id,
                priority=priority
            )
            
            # Security validation
            security_decision = None
            blocked_reason = None
            
            if self.config["enable_security_validation"]:
                is_safe, security_decision, blocked_reason = await self._validate_security(event)
                if not is_safe:
                    self.metrics["hooks_blocked"] += 1
                    processing_time = (time.time() - start_time) * 1000
                    return HookProcessingResult(
                        success=False,
                        processing_time_ms=processing_time,
                        security_decision=security_decision,
                        blocked_reason=blocked_reason
                    )
            
            # Process through registered processors
            await self._run_hook_processors(event)
            
            # Add to event aggregation
            if self.config["enable_event_aggregation"]:
                await self.event_aggregator.add_event(event)
            
            # Stream to WebSocket clients
            if self.config["enable_websocket_streaming"]:
                await self.websocket_streamer.broadcast_event(event)
            
            # Stream to Redis
            if self.config["enable_redis_streaming"] and self.redis_client:
                await self._stream_to_redis(event)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_processing_metrics(processing_time)
            
            # Generate event ID for tracking
            event_id = str(uuid.uuid4())
            
            logger.debug(
                f"Hook processed successfully",
                hook_type=hook_type.value,
                agent_id=str(agent_id),
                processing_time_ms=processing_time,
                event_id=event_id
            )
            
            return HookProcessingResult(
                success=True,
                processing_time_ms=processing_time,
                event_id=event_id
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.metrics["processing_errors"] += 1
            
            logger.error(
                f"Hook processing error",
                hook_type=hook_type.value if hook_type else "unknown",
                agent_id=str(agent_id),
                error=str(e),
                processing_time_ms=processing_time
            )
            
            return HookProcessingResult(
                success=False,
                processing_time_ms=processing_time,
                error=str(e)
            )
    
    async def _validate_security(self, event: HookEvent) -> Tuple[bool, Optional[ControlDecision], Optional[str]]:
        """Validate event for security risks."""
        
        # Extract command from payload if present
        command = None
        if "command" in event.payload:
            command = event.payload["command"]
        elif "tool_name" in event.payload and "parameters" in event.payload:
            # For tool usage events
            tool_name = event.payload["tool_name"]
            parameters = event.payload["parameters"]
            command = f"{tool_name} {json.dumps(parameters)}"
        
        if command:
            is_safe, risk_level, reason = await self.security_validator.validate_command(
                command=command,
                context={
                    "hook_type": event.hook_type.value,
                    "agent_id": str(event.agent_id),
                    "timestamp": event.timestamp.isoformat()
                }
            )
            
            if not is_safe:
                if risk_level == SecurityRisk.CRITICAL:
                    return False, ControlDecision.DENY, reason
                elif risk_level == SecurityRisk.HIGH:
                    return False, ControlDecision.REQUIRE_APPROVAL, reason
                else:
                    return False, ControlDecision.ESCALATE, reason
        
        return True, None, None
    
    async def _run_hook_processors(self, event: HookEvent) -> None:
        """Run registered hook processors for the event."""
        processors = self.hook_processors.get(event.hook_type, [])
        
        if processors:
            tasks = []
            for processor in processors:
                try:
                    if asyncio.iscoroutinefunction(processor):
                        tasks.append(processor(event))
                    else:
                        # Run synchronous processor in thread pool
                        tasks.append(asyncio.get_event_loop().run_in_executor(None, processor, event))
                except Exception as e:
                    logger.error(f"Hook processor setup error: {e}")
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Hook processor execution error: {result}")
    
    async def _stream_to_redis(self, event: HookEvent) -> None:
        """Stream event data to Redis streams."""
        try:
            # Determine target stream based on event type and content
            stream_name = self.redis_streams["hook_events"]
            
            if event.hook_type in [HookType.ERROR] or event.metadata.get("security_alert"):
                stream_name = self.redis_streams["security_events"]
            elif event.metadata.get("performance_metric"):
                stream_name = self.redis_streams["performance_events"]
            
            # Stream event data
            await self.redis_client.xadd(
                stream_name,
                event.to_dict(),
                maxlen=10000  # Keep last 10k events
            )
            
        except Exception as e:
            logger.error(f"Redis streaming error: {e}")
    
    def _update_processing_metrics(self, processing_time_ms: float) -> None:
        """Update hook processing metrics."""
        self.metrics["hooks_processed"] += 1
        
        # Check performance threshold
        if processing_time_ms > self.config["performance_threshold_ms"]:
            self.metrics["performance_threshold_violations"] += 1
            logger.warning(
                f"Hook processing exceeded performance threshold",
                processing_time_ms=processing_time_ms,
                threshold_ms=self.config["performance_threshold_ms"]
            )
        
        # Update average processing time
        current_avg = self.metrics["avg_processing_time_ms"]
        total_hooks = self.metrics["hooks_processed"]
        self.metrics["avg_processing_time_ms"] = (
            (current_avg * (total_hooks - 1) + processing_time_ms) / total_hooks
        )
    
    def _initialize_default_aggregation_rules(self) -> None:
        """Initialize default event aggregation rules."""
        
        # Tool usage aggregation rule
        def aggregate_tool_usage(events: List[HookEvent]) -> HookEvent:
            """Aggregate multiple tool usage events."""
            if len(events) == 1:
                return events[0]
            
            # Create aggregated event
            first_event = events[0]
            aggregated_payload = {
                "aggregated_count": len(events),
                "tool_usage_summary": {},
                "time_range": {
                    "start": min(e.timestamp for e in events).isoformat(),
                    "end": max(e.timestamp for e in events).isoformat()
                }
            }
            
            # Summarize tool usage
            for event in events:
                tool_name = event.payload.get("tool_name", "unknown")
                if tool_name not in aggregated_payload["tool_usage_summary"]:
                    aggregated_payload["tool_usage_summary"][tool_name] = {
                        "count": 0,
                        "success_count": 0,
                        "error_count": 0
                    }
                
                summary = aggregated_payload["tool_usage_summary"][tool_name]
                summary["count"] += 1
                
                if event.payload.get("success", True):
                    summary["success_count"] += 1
                else:
                    summary["error_count"] += 1
            
            return HookEvent(
                hook_type=first_event.hook_type,
                agent_id=first_event.agent_id,
                session_id=first_event.session_id,
                timestamp=datetime.utcnow(),
                payload=aggregated_payload,
                correlation_id=first_event.correlation_id,
                priority=min(e.priority for e in events),  # Use highest priority
                metadata={"aggregated": True, "original_count": len(events)}
            )
        
        # Register aggregation rules
        self.event_aggregator.add_aggregation_rule("PreToolUse", aggregate_tool_usage)
        self.event_aggregator.add_aggregation_rule("PostToolUse", aggregate_tool_usage)
    
    # Convenience methods for common hook types
    
    async def process_pre_tool_use(
        self,
        agent_id: uuid.UUID,
        session_id: Optional[uuid.UUID],
        tool_name: str,
        parameters: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> HookProcessingResult:
        """Process PreToolUse hook event."""
        payload = {
            "tool_name": tool_name,
            "parameters": parameters,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.process_hook(
            hook_type=HookType.PRE_TOOL_USE,
            agent_id=agent_id,
            session_id=session_id,
            payload=payload,
            correlation_id=correlation_id,
            priority=3  # High priority for tool usage
        )
    
    async def process_post_tool_use(
        self,
        agent_id: uuid.UUID,
        session_id: Optional[uuid.UUID],
        tool_name: str,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        correlation_id: Optional[str] = None
    ) -> HookProcessingResult:
        """Process PostToolUse hook event."""
        payload = {
            "tool_name": tool_name,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if result is not None:
            # Truncate large results
            if isinstance(result, str) and len(result) > self.config["max_payload_size"]:
                payload["result"] = result[:self.config["max_payload_size"]] + "... (truncated)"
                payload["result_truncated"] = True
                payload["full_result_size"] = len(result)
            else:
                payload["result"] = result
        
        if error:
            payload["error"] = error
        
        if execution_time_ms is not None:
            payload["execution_time_ms"] = execution_time_ms
        
        return await self.process_hook(
            hook_type=HookType.POST_TOOL_USE,
            agent_id=agent_id,
            session_id=session_id,
            payload=payload,
            correlation_id=correlation_id,
            priority=3  # High priority for tool usage
        )
    
    async def process_stop(
        self,
        agent_id: uuid.UUID,
        session_id: Optional[uuid.UUID],
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ) -> HookProcessingResult:
        """Process Stop hook event."""
        payload = {
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if details:
            payload["details"] = details
        
        return await self.process_hook(
            hook_type=HookType.STOP,
            agent_id=agent_id,
            session_id=session_id,
            payload=payload,
            priority=2  # Very high priority for stop events
        )
    
    async def process_notification(
        self,
        agent_id: uuid.UUID,
        session_id: Optional[uuid.UUID],
        level: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> HookProcessingResult:
        """Process Notification hook event."""
        payload = {
            "level": level,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if details:
            payload["details"] = details
        
        return await self.process_hook(
            hook_type=HookType.NOTIFICATION,
            agent_id=agent_id,
            session_id=session_id,
            payload=payload,
            priority=5  # Medium priority for notifications
        )
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all components."""
        return {
            "hook_lifecycle_system": self.metrics.copy(),
            "security_validator": self.security_validator.get_metrics(),
            "event_aggregator": self.event_aggregator.get_metrics(),
            "websocket_streamer": self.websocket_streamer.get_metrics(),
            "configuration": self.config.copy()
        }


# Global hook lifecycle system instance
_hook_lifecycle_system: Optional[HookLifecycleSystem] = None


async def get_hook_lifecycle_system() -> HookLifecycleSystem:
    """Get global hook lifecycle system instance."""
    global _hook_lifecycle_system
    
    if _hook_lifecycle_system is None:
        _hook_lifecycle_system = HookLifecycleSystem()
        await _hook_lifecycle_system.initialize()
    
    return _hook_lifecycle_system


# Convenience functions for hook processing
async def process_pre_tool_use_hook(
    agent_id: uuid.UUID,
    session_id: Optional[uuid.UUID],
    tool_name: str,
    parameters: Dict[str, Any],
    correlation_id: Optional[str] = None
) -> HookProcessingResult:
    """Convenience function for processing PreToolUse hooks."""
    system = await get_hook_lifecycle_system()
    return await system.process_pre_tool_use(
        agent_id=agent_id,
        session_id=session_id,
        tool_name=tool_name,
        parameters=parameters,
        correlation_id=correlation_id
    )


async def process_post_tool_use_hook(
    agent_id: uuid.UUID,
    session_id: Optional[uuid.UUID],
    tool_name: str,
    success: bool,
    result: Any = None,
    error: Optional[str] = None,
    execution_time_ms: Optional[float] = None,
    correlation_id: Optional[str] = None
) -> HookProcessingResult:
    """Convenience function for processing PostToolUse hooks."""
    system = await get_hook_lifecycle_system()
    return await system.process_post_tool_use(
        agent_id=agent_id,
        session_id=session_id,
        tool_name=tool_name,
        success=success,
        result=result,
        error=error,
        execution_time_ms=execution_time_ms,
        correlation_id=correlation_id
    )


async def process_stop_hook(
    agent_id: uuid.UUID,
    session_id: Optional[uuid.UUID],
    reason: str,
    details: Optional[Dict[str, Any]] = None
) -> HookProcessingResult:
    """Convenience function for processing Stop hooks."""
    system = await get_hook_lifecycle_system()
    return await system.process_stop(
        agent_id=agent_id,
        session_id=session_id,
        reason=reason,
        details=details
    )


async def process_notification_hook(
    agent_id: uuid.UUID,
    session_id: Optional[uuid.UUID],
    level: str,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> HookProcessingResult:
    """Convenience function for processing Notification hooks."""
    system = await get_hook_lifecycle_system()
    return await system.process_notification(
        agent_id=agent_id,
        session_id=session_id,
        level=level,
        message=message,
        details=details
    )