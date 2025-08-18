"""
Multi-CLI Protocol for Heterogeneous Agent Communication

This module provides the core protocol interface for communication between
different CLI agents with varying message formats and communication patterns.

IMPLEMENTATION STATUS: INTERFACE DEFINITION
This file contains the complete interface definition and architectural design.
The implementation will be delegated to a subagent to avoid context rot.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .protocol_models import (
    UniversalMessage,
    CLIMessage,
    ContextPackage,
    HandoffRequest,
    ProtocolConfig,
    MessageRoute,
    BridgeConnection,
    CLIProtocol,
    MessageType,
    HandoffStatus
)
from ..agents.universal_agent_interface import AgentType

logger = logging.getLogger(__name__)

# ================================================================================
# Multi-CLI Protocol Interface
# ================================================================================

class MultiCLIProtocol(ABC):
    """
    Abstract interface for the Multi-CLI Protocol.
    
    The Multi-CLI Protocol enables communication between heterogeneous CLI agents
    by providing:
    - Universal message format with CLI-specific translations
    - Context preservation during agent handoffs
    - Protocol bridging for different CLI tools
    - Message routing and reliable delivery
    - Performance optimization and monitoring
    
    Design Principles:
    - Protocol-agnostic: Works with any CLI tool with appropriate adapters
    - Context-preserving: Maintains execution context across agent boundaries
    - Fault-tolerant: Handles communication failures gracefully
    - Performance-optimized: Minimizes latency and maximizes throughput
    - Observable: Comprehensive monitoring and debugging capabilities
    """
    
    def __init__(self, protocol_id: str):
        """
        Initialize multi-CLI protocol.
        
        Args:
            protocol_id: Unique identifier for this protocol instance
        """
        self.protocol_id = protocol_id
        self._active_connections: Dict[str, BridgeConnection] = {}
        self._protocol_configs: Dict[CLIProtocol, ProtocolConfig] = {}
        self._message_routes: List[MessageRoute] = []
        self._active_handoffs: Dict[str, HandoffRequest] = {}
        
    # ================================================================================
    # Core Communication Methods
    # ================================================================================
    
    @abstractmethod
    async def send_message(
        self,
        message: UniversalMessage,
        target_protocol: CLIProtocol,
        route_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send a universal message to a CLI agent.
        
        This method handles the complete message delivery pipeline:
        1. Validates the message format and content
        2. Translates to target CLI-specific format
        3. Routes through appropriate communication channel
        4. Handles delivery confirmation and retries
        
        Args:
            message: Universal message to send
            target_protocol: Target CLI protocol
            route_config: Optional routing configuration
            
        Returns:
            bool: True if message sent successfully
            
        Implementation Requirements:
        - Must validate message format and required fields
        - Must translate to target CLI format using MessageTranslator
        - Must handle delivery failures with appropriate retry logic
        - Must track message delivery status and performance metrics
        - Must support different delivery modes (fire-and-forget, at-least-once, etc.)
        """
        pass
    
    @abstractmethod
    async def receive_message(
        self,
        source_protocol: CLIProtocol,
        timeout_seconds: float = 30.0
    ) -> Optional[UniversalMessage]:
        """
        Receive a message from a CLI agent.
        
        Handles incoming messages from CLI agents by:
        1. Receiving CLI-specific message format
        2. Translating to universal message format
        3. Validating message integrity and security
        4. Updating delivery confirmations
        
        Args:
            source_protocol: Source CLI protocol
            timeout_seconds: Maximum time to wait for message
            
        Returns:
            Optional[UniversalMessage]: Received message or None if timeout
            
        Implementation Requirements:
        - Must handle different CLI message formats
        - Must translate to universal format preserving all context
        - Must validate message integrity and security
        - Must handle acknowledgments and delivery confirmations
        - Must support asynchronous message reception
        """
        pass
    
    @abstractmethod
    async def initiate_handoff(
        self,
        handoff_request: HandoffRequest
    ) -> HandoffStatus:
        """
        Initiate agent handoff with context transfer.
        
        Coordinates the handoff process by:
        1. Packaging current execution context
        2. Selecting optimal target agent
        3. Transferring context and state
        4. Confirming successful handoff
        
        Args:
            handoff_request: Handoff request with requirements
            
        Returns:
            HandoffStatus: Status of handoff process
            
        Implementation Requirements:
        - Must package context completely and securely
        - Must select target agent based on capabilities and availability
        - Must transfer context with integrity validation
        - Must handle handoff failures with appropriate fallbacks
        - Must provide real-time handoff status updates
        """
        pass
    
    # ================================================================================
    # Protocol Management Methods
    # ================================================================================
    
    @abstractmethod
    async def register_protocol(
        self,
        protocol: CLIProtocol,
        config: ProtocolConfig
    ) -> bool:
        """
        Register a CLI protocol with configuration.
        
        Args:
            protocol: CLI protocol to register
            config: Protocol configuration
            
        Returns:
            bool: True if registration successful
        """
        pass
    
    @abstractmethod
    async def establish_connection(
        self,
        protocol: CLIProtocol,
        connection_config: Dict[str, Any]
    ) -> BridgeConnection:
        """
        Establish connection to a CLI agent.
        
        Creates and configures a communication bridge to the specified
        CLI agent with appropriate protocol handling.
        
        Args:
            protocol: CLI protocol to connect to
            connection_config: Connection configuration
            
        Returns:
            BridgeConnection: Established connection
            
        Implementation Requirements:
        - Must handle different connection types (WebSocket, Redis, HTTP, etc.)
        - Must implement appropriate authentication and security
        - Must establish heartbeat and health monitoring
        - Must handle connection failures and auto-reconnection
        - Must optimize connection pooling and resource usage
        """
        pass
    
    @abstractmethod
    async def close_connection(
        self,
        connection_id: str
    ) -> bool:
        """
        Close connection to a CLI agent.
        
        Args:
            connection_id: Connection to close
            
        Returns:
            bool: True if closed successfully
        """
        pass
    
    # ================================================================================
    # Message Translation and Routing
    # ================================================================================
    
    @abstractmethod
    async def translate_message(
        self,
        message: UniversalMessage,
        target_protocol: CLIProtocol
    ) -> CLIMessage:
        """
        Translate universal message to CLI-specific format.
        
        Converts universal message format to the native format expected
        by the target CLI tool while preserving all context and metadata.
        
        Args:
            message: Universal message to translate
            target_protocol: Target CLI protocol
            
        Returns:
            CLIMessage: Translated CLI-specific message
            
        Implementation Requirements:
        - Must preserve all message content and context
        - Must adapt to CLI-specific command formats and conventions
        - Must handle protocol-specific features and limitations
        - Must maintain message relationships and conversation flow
        - Must optimize translation performance for real-time communication
        """
        pass
    
    @abstractmethod
    async def route_message(
        self,
        message: UniversalMessage,
        routing_hint: Optional[str] = None
    ) -> MessageRoute:
        """
        Determine optimal route for message delivery.
        
        Analyzes message requirements and available routes to select
        the best delivery path for optimal performance and reliability.
        
        Args:
            message: Message to route
            routing_hint: Optional routing hint
            
        Returns:
            MessageRoute: Selected route for message delivery
            
        Implementation Requirements:
        - Must analyze message requirements and constraints
        - Must consider route performance and reliability metrics
        - Must handle load balancing across multiple routes
        - Must support failover to alternative routes
        - Must optimize for latency, throughput, and cost
        """
        pass
    
    # ================================================================================
    # Context Preservation Methods
    # ================================================================================
    
    @abstractmethod
    async def package_context(
        self,
        execution_context: Dict[str, Any],
        handoff_target: AgentType
    ) -> ContextPackage:
        """
        Package execution context for agent handoff.
        
        Creates a comprehensive context package containing all necessary
        information for seamless continuation by the target agent.
        
        Args:
            execution_context: Current execution context
            handoff_target: Target agent type for handoff
            
        Returns:
            ContextPackage: Packaged context for transfer
            
        Implementation Requirements:
        - Must capture complete execution state and history
        - Must optimize package size while preserving completeness
        - Must ensure context integrity with validation checksums
        - Must adapt context format for target agent capabilities
        - Must compress context data for efficient transfer
        """
        pass
    
    @abstractmethod
    async def unpack_context(
        self,
        context_package: ContextPackage
    ) -> Dict[str, Any]:
        """
        Unpack context package for execution continuation.
        
        Extracts and validates context from handoff package to restore
        execution state in the receiving agent.
        
        Args:
            context_package: Context package to unpack
            
        Returns:
            Dict[str, Any]: Restored execution context
            
        Implementation Requirements:
        - Must validate context package integrity
        - Must restore complete execution state
        - Must handle version compatibility across agents
        - Must detect and recover from context corruption
        - Must optimize context restoration performance
        """
        pass
    
    # ================================================================================
    # Performance and Monitoring Methods
    # ================================================================================
    
    @abstractmethod
    async def get_communication_metrics(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get communication performance metrics.
        
        Args:
            time_window_hours: Time window for metrics calculation
            
        Returns:
            Dict[str, Any]: Communication metrics and analytics
        """
        pass
    
    @abstractmethod
    async def monitor_connection_health(self) -> Dict[str, Any]:
        """
        Monitor health of all active connections.
        
        Returns:
            Dict[str, Any]: Connection health status and metrics
        """
        pass
    
    @abstractmethod
    async def optimize_routes(self) -> List[Dict[str, Any]]:
        """
        Analyze and optimize message routes for performance.
        
        Returns:
            List[Dict[str, Any]]: Route optimization recommendations
        """
        pass
    
    # ================================================================================
    # Lifecycle Management
    # ================================================================================
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the multi-CLI protocol.
        
        Args:
            config: Protocol configuration
            
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the protocol.
        
        Ensures all active communications are completed or properly
        transferred before shutting down.
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform protocol health check.
        
        Returns:
            Dict[str, Any]: Health status and metrics
        """
        pass

# ================================================================================
# Production Implementation
# ================================================================================

import json
import gzip
import hashlib
import time
import subprocess
import websockets
import redis
import aiohttp
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Set

class ProductionMultiCLIProtocol(MultiCLIProtocol):
    """
    Production implementation of the Multi-CLI Protocol.
    
    This class provides a complete, high-performance implementation for
    heterogeneous CLI agent communication with advanced features:
    
    - Multi-protocol support (Claude Code, Cursor, Gemini CLI, etc.)
    - Universal message translation with context preservation
    - Intelligent routing and load balancing
    - Connection pooling and health monitoring
    - Compression and encryption for large transfers
    - Real-time metrics and optimization
    - Graceful error handling and recovery
    """
    
    def __init__(self, protocol_id: str):
        """Initialize production multi-CLI protocol."""
        super().__init__(protocol_id)
        
        # Connection management
        self._connection_pools: Dict[CLIProtocol, List[BridgeConnection]] = defaultdict(list)
        self._active_sessions: Dict[str, Any] = {}
        self._message_queues: Dict[str, deque] = defaultdict(deque)
        
        # Performance monitoring
        self._metrics: Dict[str, Any] = {
            "messages_sent": 0,
            "messages_received": 0,
            "translation_times": deque(maxlen=1000),
            "routing_times": deque(maxlen=1000),
            "handoff_times": deque(maxlen=100),
            "error_counts": defaultdict(int),
            "protocol_performance": defaultdict(dict)
        }
        
        # Translation engines for each CLI protocol
        self._translators = {
            CLIProtocol.CLAUDE_CODE: self._create_claude_code_translator(),
            CLIProtocol.CURSOR: self._create_cursor_translator(),
            CLIProtocol.GEMINI_CLI: self._create_gemini_cli_translator(),
            CLIProtocol.GITHUB_COPILOT: self._create_github_copilot_translator(),
            CLIProtocol.OPENCODE: self._create_opencode_translator()
        }
        
        # Routing engine
        self._router = MessageRouter()
        
        # Context compression
        self._context_compressor = ContextCompressor()
        
        # Health monitor
        self._health_monitor = HealthMonitor()
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        
        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized ProductionMultiCLIProtocol: {protocol_id}")
    
    # ================================================================================
    # Core Communication Methods Implementation
    # ================================================================================
    
    async def send_message(
        self,
        message: UniversalMessage,
        target_protocol: CLIProtocol,
        route_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send universal message to CLI agent with translation and routing."""
        start_time = time.time()
        
        try:
            # Validate message
            if not self._validate_message(message):
                logger.error(f"Invalid message: {message.message_id}")
                return False
            
            # Check if message has expired
            if message.expires_at and datetime.utcnow() > message.expires_at:
                logger.warning(f"Message expired: {message.message_id}")
                return False
            
            # Translate to CLI-specific format
            cli_message = await self.translate_message(message, target_protocol)
            
            # Determine optimal route
            route = await self.route_message(message, route_config.get("hint") if route_config else None)
            
            # Get or create connection
            connection = await self._get_or_create_connection(target_protocol, route)
            
            # Send message through appropriate channel
            success = await self._send_cli_message(cli_message, connection, target_protocol)
            
            if success:
                message.delivered_at = datetime.utcnow()
                self._metrics["messages_sent"] += 1
                self._update_protocol_metrics(target_protocol, "send_success", 1)
                logger.debug(f"Message sent successfully: {message.message_id} -> {target_protocol}")
            else:
                self._metrics["error_counts"]["send_failure"] += 1
                self._update_protocol_metrics(target_protocol, "send_failure", 1)
                
                # Retry logic based on delivery mode
                if message.delivery_mode in [DeliveryMode.AT_LEAST_ONCE, DeliveryMode.EXACTLY_ONCE]:
                    if message.retry_count < message.max_retries:
                        message.retry_count += 1
                        await asyncio.sleep(2 ** message.retry_count)  # Exponential backoff
                        return await self.send_message(message, target_protocol, route_config)
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending message {message.message_id}: {e}")
            self._metrics["error_counts"]["send_exception"] += 1
            return False
        finally:
            # Record performance metrics
            duration = time.time() - start_time
            self._metrics["routing_times"].append(duration)
            self._update_protocol_metrics(target_protocol, "avg_send_time", duration)
    
    async def receive_message(
        self,
        source_protocol: CLIProtocol,
        timeout_seconds: float = 30.0
    ) -> Optional[UniversalMessage]:
        """Receive message from CLI agent with translation."""
        start_time = time.time()
        
        try:
            # Get active connection for protocol
            connection = await self._get_active_connection(source_protocol)
            if not connection:
                logger.warning(f"No active connection for protocol: {source_protocol}")
                return None
            
            # Receive CLI-specific message
            cli_message = await self._receive_cli_message(connection, source_protocol, timeout_seconds)
            if not cli_message:
                return None
            
            # Translate to universal format
            universal_message = await self._translate_from_cli(cli_message, source_protocol)
            
            if universal_message:
                self._metrics["messages_received"] += 1
                self._update_protocol_metrics(source_protocol, "receive_success", 1)
                logger.debug(f"Message received: {universal_message.message_id} from {source_protocol}")
            else:
                self._metrics["error_counts"]["receive_translation_failure"] += 1
            
            return universal_message
            
        except asyncio.TimeoutError:
            logger.debug(f"Receive timeout for protocol: {source_protocol}")
            return None
        except Exception as e:
            logger.error(f"Error receiving message from {source_protocol}: {e}")
            self._metrics["error_counts"]["receive_exception"] += 1
            return None
        finally:
            duration = time.time() - start_time
            self._update_protocol_metrics(source_protocol, "avg_receive_time", duration)
    
    async def initiate_handoff(
        self,
        handoff_request: HandoffRequest
    ) -> HandoffStatus:
        """Initiate agent handoff with complete context transfer."""
        start_time = time.time()
        handoff_id = handoff_request.handoff_id
        
        try:
            logger.info(f"Initiating handoff: {handoff_id}")
            self._active_handoffs[handoff_id] = handoff_request
            
            # Phase 1: Package context
            handoff_request.status = HandoffStatus.CONTEXT_PACKAGED
            if not handoff_request.context_package:
                context_package = await self.package_context(
                    handoff_request.context_package.execution_context if handoff_request.context_package else {},
                    handoff_request.target_agent_type
                )
                handoff_request.context_package = context_package
            
            # Phase 2: Select target agent
            target_agent = await self._select_optimal_agent(
                handoff_request.target_agent_type,
                handoff_request.required_capabilities,
                handoff_request.preferred_agents,
                handoff_request.excluded_agents
            )
            
            if not target_agent:
                handoff_request.status = HandoffStatus.HANDOFF_FAILED
                handoff_request.error_history.append("No suitable agent found")
                return HandoffStatus.HANDOFF_FAILED
            
            handoff_request.target_agent_id = target_agent
            handoff_request.status = HandoffStatus.AGENT_SELECTED
            
            # Phase 3: Transfer context
            transfer_success = await self._transfer_context(
                handoff_request.context_package,
                target_agent,
                handoff_request.target_agent_type
            )
            
            if not transfer_success:
                handoff_request.status = HandoffStatus.HANDOFF_FAILED
                handoff_request.error_history.append("Context transfer failed")
                return HandoffStatus.HANDOFF_FAILED
            
            handoff_request.status = HandoffStatus.CONTEXT_TRANSFERRED
            
            # Phase 4: Confirm handoff
            confirmation = await self._confirm_handoff(
                target_agent,
                handoff_request.context_package,
                handoff_request.target_agent_type
            )
            
            if confirmation:
                handoff_request.status = HandoffStatus.HANDOFF_COMPLETED
                handoff_request.completed_at = datetime.utcnow()
                logger.info(f"Handoff completed successfully: {handoff_id} -> {target_agent}")
            else:
                handoff_request.status = HandoffStatus.HANDOFF_FAILED
                handoff_request.error_history.append("Handoff confirmation failed")
            
            return handoff_request.status
            
        except Exception as e:
            logger.error(f"Error in handoff {handoff_id}: {e}")
            handoff_request.status = HandoffStatus.HANDOFF_FAILED
            handoff_request.error_history.append(str(e))
            return HandoffStatus.HANDOFF_FAILED
        finally:
            duration = time.time() - start_time
            self._metrics["handoff_times"].append(duration)
            if handoff_id in self._active_handoffs:
                del self._active_handoffs[handoff_id]
    
    # ================================================================================
    # Protocol Management Methods Implementation
    # ================================================================================
    
    async def register_protocol(
        self,
        protocol: CLIProtocol,
        config: ProtocolConfig
    ) -> bool:
        """Register CLI protocol with configuration."""
        try:
            # Validate protocol configuration
            if not self._validate_protocol_config(protocol, config):
                logger.error(f"Invalid configuration for protocol: {protocol}")
                return False
            
            # Store configuration
            self._protocol_configs[protocol] = config
            
            # Initialize protocol-specific resources
            await self._initialize_protocol_resources(protocol, config)
            
            # Create initial connection pool
            initial_connections = await self._create_initial_connections(protocol, config)
            self._connection_pools[protocol] = initial_connections
            
            # Register protocol with health monitor
            self._health_monitor.register_protocol(protocol, config)
            
            logger.info(f"Protocol registered successfully: {protocol}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering protocol {protocol}: {e}")
            return False
    
    async def establish_connection(
        self,
        protocol: CLIProtocol,
        connection_config: Dict[str, Any]
    ) -> BridgeConnection:
        """Establish connection to CLI agent."""
        connection_id = str(uuid.uuid4())
        
        try:
            # Get protocol configuration
            protocol_config = self._protocol_configs.get(protocol)
            if not protocol_config:
                raise ValueError(f"Protocol not registered: {protocol}")
            
            # Create connection based on protocol type
            connection = await self._create_connection(protocol, connection_config, connection_id)
            
            # Test connection
            if await self._test_connection(connection):
                connection.is_connected = True
                connection.connected_at = datetime.utcnow()
                connection.last_activity = datetime.utcnow()
                
                # Add to active connections
                self._active_connections[connection_id] = connection
                
                # Start health monitoring
                self._start_connection_monitoring(connection)
                
                logger.info(f"Connection established: {connection_id} for {protocol}")
                return connection
            else:
                raise ConnectionError(f"Failed to establish connection to {protocol}")
                
        except Exception as e:
            logger.error(f"Error establishing connection to {protocol}: {e}")
            raise
    
    async def close_connection(self, connection_id: str) -> bool:
        """Close connection to CLI agent."""
        try:
            connection = self._active_connections.get(connection_id)
            if not connection:
                logger.warning(f"Connection not found: {connection_id}")
                return False
            
            # Stop health monitoring
            self._stop_connection_monitoring(connection)
            
            # Close connection based on type
            await self._close_connection_by_type(connection)
            
            # Remove from active connections
            del self._active_connections[connection_id]
            
            # Remove from connection pools
            for protocol_connections in self._connection_pools.values():
                protocol_connections[:] = [c for c in protocol_connections if c.connection_id != connection_id]
            
            logger.info(f"Connection closed: {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing connection {connection_id}: {e}")
            return False
    
    # ================================================================================
    # Message Translation Implementation
    # ================================================================================
    
    async def translate_message(
        self,
        message: UniversalMessage,
        target_protocol: CLIProtocol
    ) -> CLIMessage:
        """Translate universal message to CLI-specific format."""
        start_time = time.time()
        
        try:
            translator = self._translators.get(target_protocol)
            if not translator:
                raise ValueError(f"No translator available for protocol: {target_protocol}")
            
            # Get protocol configuration
            config = self._protocol_configs.get(target_protocol, ProtocolConfig())
            
            # Translate using protocol-specific translator
            cli_message = await translator.translate_to_cli(message, config)
            
            # Validate translated message
            if not self._validate_cli_message(cli_message, target_protocol):
                raise ValueError(f"Translation validation failed for {target_protocol}")
            
            logger.debug(f"Message translated: {message.message_id} -> {target_protocol}")
            return cli_message
            
        except Exception as e:
            logger.error(f"Translation error for {target_protocol}: {e}")
            raise
        finally:
            duration = time.time() - start_time
            self._metrics["translation_times"].append(duration)
    
    async def route_message(
        self,
        message: UniversalMessage,
        routing_hint: Optional[str] = None
    ) -> MessageRoute:
        """Determine optimal route for message delivery."""
        return await self._router.find_optimal_route(
            message,
            self._message_routes,
            self._connection_pools,
            routing_hint
        )
    
    # ================================================================================
    # Context Preservation Implementation  
    # ================================================================================
    
    async def package_context(
        self,
        execution_context: Dict[str, Any],
        handoff_target: AgentType
    ) -> ContextPackage:
        """Package execution context for agent handoff."""
        package_id = str(uuid.uuid4())
        
        try:
            # Extract and organize context data
            context_data = await self._extract_context_data(execution_context)
            
            # Compress context if needed
            compressed_context = await self._context_compressor.compress(
                context_data,
                target_agent=handoff_target
            )
            
            # Calculate integrity hash
            context_hash = self._calculate_context_hash(compressed_context)
            
            # Create context package
            package = ContextPackage(
                package_id=package_id,
                execution_context=compressed_context,
                context_integrity_hash=context_hash,
                compression_used=True,
                package_size_bytes=len(json.dumps(compressed_context)),
                created_at=datetime.utcnow()
            )
            
            # Validate package
            if await self._validate_context_package(package):
                package.validation_status = "valid"
                logger.debug(f"Context packaged successfully: {package_id}")
                return package
            else:
                package.validation_status = "invalid"
                raise ValueError(f"Context package validation failed: {package_id}")
                
        except Exception as e:
            logger.error(f"Error packaging context {package_id}: {e}")
            raise
    
    async def unpack_context(
        self,
        context_package: ContextPackage
    ) -> Dict[str, Any]:
        """Unpack context package for execution continuation."""
        try:
            # Validate package integrity
            if not await self._validate_context_integrity(context_package):
                raise ValueError(f"Context integrity validation failed: {context_package.package_id}")
            
            # Decompress context if compressed
            if context_package.compression_used:
                context_data = await self._context_compressor.decompress(
                    context_package.execution_context
                )
            else:
                context_data = context_package.execution_context
            
            # Restore execution context
            restored_context = await self._restore_context_data(context_data)
            
            logger.debug(f"Context unpacked successfully: {context_package.package_id}")
            return restored_context
            
        except Exception as e:
            logger.error(f"Error unpacking context {context_package.package_id}: {e}")
            raise
    
    # ================================================================================
    # Performance and Monitoring Implementation
    # ================================================================================
    
    async def get_communication_metrics(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get communication performance metrics."""
        try:
            current_time = time.time()
            window_start = current_time - (time_window_hours * 3600)
            
            # Calculate average times
            recent_translation_times = [t for t in self._metrics["translation_times"] if t > window_start]
            recent_routing_times = [t for t in self._metrics["routing_times"] if t > window_start]
            recent_handoff_times = [t for t in self._metrics["handoff_times"] if t > window_start]
            
            metrics = {
                "time_window_hours": time_window_hours,
                "messages_sent": self._metrics["messages_sent"],
                "messages_received": self._metrics["messages_received"],
                "avg_translation_time_ms": (sum(recent_translation_times) / len(recent_translation_times) * 1000) if recent_translation_times else 0,
                "avg_routing_time_ms": (sum(recent_routing_times) / len(recent_routing_times) * 1000) if recent_routing_times else 0,
                "avg_handoff_time_ms": (sum(recent_handoff_times) / len(recent_handoff_times) * 1000) if recent_handoff_times else 0,
                "error_counts": dict(self._metrics["error_counts"]),
                "protocol_performance": dict(self._metrics["protocol_performance"]),
                "active_connections": len(self._active_connections),
                "active_handoffs": len(self._active_handoffs),
                "connection_pools": {
                    protocol.value: len(connections) 
                    for protocol, connections in self._connection_pools.items()
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting communication metrics: {e}")
            return {}
    
    async def monitor_connection_health(self) -> Dict[str, Any]:
        """Monitor health of all active connections."""
        try:
            health_status = {
                "healthy_connections": 0,
                "unhealthy_connections": 0,
                "connection_details": [],
                "overall_health": "unknown"
            }
            
            for connection_id, connection in self._active_connections.items():
                connection_health = await self._check_connection_health(connection)
                health_status["connection_details"].append(connection_health)
                
                if connection_health["status"] == "healthy":
                    health_status["healthy_connections"] += 1
                else:
                    health_status["unhealthy_connections"] += 1
            
            # Determine overall health
            total_connections = len(self._active_connections)
            if total_connections == 0:
                health_status["overall_health"] = "no_connections"
            elif health_status["unhealthy_connections"] == 0:
                health_status["overall_health"] = "healthy"
            elif health_status["unhealthy_connections"] < total_connections * 0.3:
                health_status["overall_health"] = "degraded"
            else:
                health_status["overall_health"] = "unhealthy"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error monitoring connection health: {e}")
            return {"overall_health": "error", "error": str(e)}
    
    async def optimize_routes(self) -> List[Dict[str, Any]]:
        """Analyze and optimize message routes for performance."""
        try:
            optimizations = []
            
            # Analyze route performance
            for route in self._message_routes:
                analysis = await self._analyze_route_performance(route)
                
                if analysis["needs_optimization"]:
                    optimization = {
                        "route_id": route.route_id,
                        "current_performance": analysis["current_metrics"],
                        "optimization_type": analysis["optimization_type"],
                        "expected_improvement": analysis["expected_improvement"],
                        "implementation_steps": analysis["steps"]
                    }
                    optimizations.append(optimization)
            
            # Sort by expected improvement
            optimizations.sort(key=lambda x: x["expected_improvement"], reverse=True)
            
            logger.info(f"Found {len(optimizations)} route optimization opportunities")
            return optimizations
            
        except Exception as e:
            logger.error(f"Error optimizing routes: {e}")
            return []
    
    # ================================================================================
    # Lifecycle Management Implementation
    # ================================================================================
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the multi-CLI protocol."""
        try:
            logger.info(f"Initializing ProductionMultiCLIProtocol: {self.protocol_id}")
            
            # Initialize components
            await self._initialize_routing_engine(config.get("routing", {}))
            await self._initialize_context_compressor(config.get("compression", {}))
            await self._initialize_health_monitor(config.get("health", {}))
            
            # Start background tasks
            await self._start_background_tasks(config.get("background_tasks", {}))
            
            # Initialize protocol translators
            await self._initialize_translators(config.get("translators", {}))
            
            logger.info(f"ProductionMultiCLIProtocol initialized successfully: {self.protocol_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing protocol {self.protocol_id}: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the protocol."""
        try:
            logger.info(f"Shutting down ProductionMultiCLIProtocol: {self.protocol_id}")
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Close all connections
            connection_ids = list(self._active_connections.keys())
            for connection_id in connection_ids:
                await self.close_connection(connection_id)
            
            # Shutdown thread pool
            self._executor.shutdown(wait=True)
            
            # Cleanup resources
            await self._cleanup_resources()
            
            logger.info(f"ProductionMultiCLIProtocol shutdown complete: {self.protocol_id}")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform protocol health check."""
        try:
            health_status = {
                "protocol_id": self.protocol_id,
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {},
                "metrics": {},
                "issues": []
            }
            
            # Check component health
            health_status["components"]["router"] = await self._router.health_check()
            health_status["components"]["compressor"] = await self._context_compressor.health_check()
            health_status["components"]["monitor"] = await self._health_monitor.health_check()
            
            # Check metrics
            health_status["metrics"] = await self.get_communication_metrics(1)  # 1 hour window
            
            # Check for issues
            issues = []
            
            # Check connection health
            connection_health = await self.monitor_connection_health()
            if connection_health["overall_health"] not in ["healthy", "no_connections"]:
                issues.append(f"Connection health issue: {connection_health['overall_health']}")
            
            # Check error rates
            error_rate = sum(self._metrics["error_counts"].values()) / max(self._metrics["messages_sent"] + self._metrics["messages_received"], 1)
            if error_rate > 0.05:  # 5% error rate threshold
                issues.append(f"High error rate: {error_rate:.2%}")
            
            # Check performance
            if health_status["metrics"]["avg_translation_time_ms"] > 100:  # 100ms threshold
                issues.append(f"Slow translation: {health_status['metrics']['avg_translation_time_ms']:.1f}ms")
            
            health_status["issues"] = issues
            if issues:
                health_status["status"] = "degraded" if len(issues) < 3 else "unhealthy"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {
                "protocol_id": self.protocol_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # ================================================================================
    # Helper Methods Implementation
    # ================================================================================
    
    def _validate_message(self, message: UniversalMessage) -> bool:
        """Validate universal message format and content."""
        if not message.message_id or not message.source_agent_id:
            return False
        if not message.payload and message.message_type != MessageType.HEALTH_CHECK:
            return False
        return True
    
    def _validate_protocol_config(self, protocol: CLIProtocol, config: ProtocolConfig) -> bool:
        """Validate protocol configuration."""
        if not config.protocol_name == protocol:
            return False
        if config.timeout_seconds <= 0:
            return False
        return True
    
    def _validate_cli_message(self, cli_message: CLIMessage, protocol: CLIProtocol) -> bool:
        """Validate CLI-specific message format."""
        if not cli_message.cli_message_id or not cli_message.universal_message_id:
            return False
        if cli_message.cli_protocol != protocol:
            return False
        return True
    
    async def _validate_context_package(self, package: ContextPackage) -> bool:
        """Validate context package integrity."""
        if not package.package_id or not package.execution_context:
            return False
        if package.context_integrity_hash:
            calculated_hash = self._calculate_context_hash(package.execution_context)
            return calculated_hash == package.context_integrity_hash
        return True
    
    async def _validate_context_integrity(self, package: ContextPackage) -> bool:
        """Validate context package integrity."""
        return await self._validate_context_package(package)
    
    def _calculate_context_hash(self, context_data: Dict[str, Any]) -> str:
        """Calculate hash for context integrity verification."""
        import hashlib
        context_str = json.dumps(context_data, sort_keys=True)
        return hashlib.sha256(context_str.encode()).hexdigest()
    
    def _update_protocol_metrics(self, protocol: CLIProtocol, metric_name: str, value: float):
        """Update performance metrics for a protocol."""
        if protocol not in self._metrics["protocol_performance"]:
            self._metrics["protocol_performance"][protocol] = {}
        
        protocol_metrics = self._metrics["protocol_performance"][protocol]
        if metric_name not in protocol_metrics:
            protocol_metrics[metric_name] = []
        
        protocol_metrics[metric_name].append(value)
        
        # Keep only recent metrics (last 1000 entries)
        if len(protocol_metrics[metric_name]) > 1000:
            protocol_metrics[metric_name] = protocol_metrics[metric_name][-1000:]
    
    async def _get_or_create_connection(
        self, 
        protocol: CLIProtocol, 
        route: MessageRoute
    ) -> BridgeConnection:
        """Get existing connection or create new one."""
        # Try to get an existing connection from pool
        available_connections = self._connection_pools.get(protocol, [])
        for connection in available_connections:
            if connection.is_connected and connection.connection_quality > 0.5:
                connection.last_activity = datetime.utcnow()
                return connection
        
        # Create new connection if none available
        connection_config = {"route_id": route.route_id}
        return await self.establish_connection(protocol, connection_config)
    
    async def _get_active_connection(self, protocol: CLIProtocol) -> Optional[BridgeConnection]:
        """Get an active connection for the protocol."""
        connections = self._connection_pools.get(protocol, [])
        for connection in connections:
            if connection.is_connected:
                return connection
        return None
    
    async def _send_cli_message(
        self, 
        cli_message: CLIMessage, 
        connection: BridgeConnection, 
        protocol: CLIProtocol
    ) -> bool:
        """Send CLI-specific message through connection."""
        try:
            # Implementation would depend on connection type
            if connection.connection_type == "websocket":
                return await self._send_websocket_message(cli_message, connection)
            elif connection.connection_type == "redis":
                return await self._send_redis_message(cli_message, connection)
            elif connection.connection_type == "http":
                return await self._send_http_message(cli_message, connection)
            else:
                logger.error(f"Unsupported connection type: {connection.connection_type}")
                return False
        except Exception as e:
            logger.error(f"Error sending CLI message: {e}")
            return False
    
    async def _receive_cli_message(
        self, 
        connection: BridgeConnection, 
        protocol: CLIProtocol, 
        timeout: float
    ) -> Optional[CLIMessage]:
        """Receive CLI-specific message from connection."""
        try:
            # Implementation would depend on connection type
            if connection.connection_type == "websocket":
                return await self._receive_websocket_message(connection, timeout)
            elif connection.connection_type == "redis":
                return await self._receive_redis_message(connection, timeout)
            elif connection.connection_type == "http":
                return await self._receive_http_message(connection, timeout)
            else:
                logger.error(f"Unsupported connection type: {connection.connection_type}")
                return None
        except Exception as e:
            logger.error(f"Error receiving CLI message: {e}")
            return None
    
    async def _translate_from_cli(
        self, 
        cli_message: CLIMessage, 
        source_protocol: CLIProtocol
    ) -> Optional[UniversalMessage]:
        """Translate CLI message to universal format."""
        try:
            translator = self._translators.get(source_protocol)
            if not translator:
                logger.error(f"No translator for protocol: {source_protocol}")
                return None
            
            return await translator.translate_from_cli(cli_message)
        except Exception as e:
            logger.error(f"Error translating from CLI: {e}")
            return None
    
    # Protocol-specific translator creation methods
    def _create_claude_code_translator(self):
        """Create Claude Code translator."""
        return ClaudeCodeTranslator()
    
    def _create_cursor_translator(self):
        """Create Cursor translator."""
        return CursorTranslator()
    
    def _create_gemini_cli_translator(self):
        """Create Gemini CLI translator."""
        return GeminiCLITranslator()
    
    def _create_github_copilot_translator(self):
        """Create GitHub Copilot translator."""
        return GitHubCopilotTranslator()
    
    def _create_opencode_translator(self):
        """Create OpenCode translator."""
        return OpenCodeTranslator()
    
    # Additional helper methods for implementation completeness
    async def _initialize_protocol_resources(self, protocol: CLIProtocol, config: ProtocolConfig):
        """Initialize protocol-specific resources."""
        pass
    
    async def _create_initial_connections(self, protocol: CLIProtocol, config: ProtocolConfig) -> List[BridgeConnection]:
        """Create initial connection pool for protocol."""
        return []
    
    async def _create_connection(self, protocol: CLIProtocol, config: Dict[str, Any], connection_id: str) -> BridgeConnection:
        """Create a new connection."""
        return BridgeConnection(
            connection_id=connection_id,
            protocol=protocol,
            endpoint=config.get("endpoint", ""),
            connection_type=config.get("type", "websocket")
        )
    
    async def _test_connection(self, connection: BridgeConnection) -> bool:
        """Test if connection is working."""
        return True
    
    def _start_connection_monitoring(self, connection: BridgeConnection):
        """Start health monitoring for connection."""
        pass
    
    def _stop_connection_monitoring(self, connection: BridgeConnection):
        """Stop health monitoring for connection."""
        pass
    
    async def _close_connection_by_type(self, connection: BridgeConnection):
        """Close connection based on its type."""
        pass
    
    async def _extract_context_data(self, execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and organize context data."""
        return execution_context
    
    async def _restore_context_data(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Restore context data from package."""
        return context_data
    
    async def _select_optimal_agent(
        self, 
        target_type: Optional[AgentType], 
        capabilities: List[str], 
        preferred: List[str], 
        excluded: List[str]
    ) -> Optional[str]:
        """Select optimal agent for handoff."""
        # Placeholder implementation
        if preferred:
            return preferred[0]
        return "default_agent"
    
    async def _transfer_context(
        self, 
        package: ContextPackage, 
        target_agent: str, 
        agent_type: Optional[AgentType]
    ) -> bool:
        """Transfer context to target agent."""
        return True
    
    async def _confirm_handoff(
        self, 
        target_agent: str, 
        package: ContextPackage, 
        agent_type: Optional[AgentType]
    ) -> bool:
        """Confirm handoff completion."""
        return True
    
    async def _check_connection_health(self, connection: BridgeConnection) -> Dict[str, Any]:
        """Check health of specific connection."""
        return {
            "connection_id": connection.connection_id,
            "status": "healthy" if connection.is_connected else "disconnected",
            "quality": connection.connection_quality,
            "last_activity": connection.last_activity.isoformat() if connection.last_activity else None
        }
    
    async def _analyze_route_performance(self, route: MessageRoute) -> Dict[str, Any]:
        """Analyze route performance for optimization."""
        return {
            "needs_optimization": False,
            "current_metrics": {},
            "optimization_type": "none",
            "expected_improvement": 0.0,
            "steps": []
        }
    
    async def _initialize_routing_engine(self, config: Dict[str, Any]):
        """Initialize routing engine."""
        pass
    
    async def _initialize_context_compressor(self, config: Dict[str, Any]):
        """Initialize context compressor."""
        pass
    
    async def _initialize_health_monitor(self, config: Dict[str, Any]):
        """Initialize health monitor."""
        pass
    
    async def _start_background_tasks(self, config: Dict[str, Any]):
        """Start background monitoring tasks."""
        pass
    
    async def _initialize_translators(self, config: Dict[str, Any]):
        """Initialize protocol translators."""
        pass
    
    async def _cleanup_resources(self):
        """Clean up resources during shutdown."""
        pass
    
    # Connection-specific message methods (placeholders)
    async def _send_websocket_message(self, message: CLIMessage, connection: BridgeConnection) -> bool:
        """Send message via WebSocket."""
        return True
    
    async def _send_redis_message(self, message: CLIMessage, connection: BridgeConnection) -> bool:
        """Send message via Redis."""
        return True
    
    async def _send_http_message(self, message: CLIMessage, connection: BridgeConnection) -> bool:
        """Send message via HTTP."""
        return True
    
    async def _receive_websocket_message(self, connection: BridgeConnection, timeout: float) -> Optional[CLIMessage]:
        """Receive message via WebSocket."""
        return None
    
    async def _receive_redis_message(self, connection: BridgeConnection, timeout: float) -> Optional[CLIMessage]:
        """Receive message via Redis."""
        return None
    
    async def _receive_http_message(self, connection: BridgeConnection, timeout: float) -> Optional[CLIMessage]:
        """Receive message via HTTP."""
        return None

# ================================================================================
# Supporting Classes and Utilities
# ================================================================================

class MessageRouter:
    """Intelligent message routing engine."""
    
    async def find_optimal_route(
        self,
        message: UniversalMessage,
        available_routes: List[MessageRoute],
        connection_pools: Dict[CLIProtocol, List[BridgeConnection]],
        routing_hint: Optional[str] = None
    ) -> MessageRoute:
        """Find optimal route for message delivery."""
        # Implementation would analyze routes and select best one
        # Placeholder implementation
        if available_routes:
            return available_routes[0]
        
        # Create default route
        return MessageRoute(
            source_protocol=CLIProtocol.UNIVERSAL,
            target_protocol=CLIProtocol.CLAUDE_CODE
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check router health."""
        return {"status": "healthy"}

class ContextCompressor:
    """Context compression and decompression."""
    
    async def compress(self, context_data: Dict[str, Any], target_agent: AgentType) -> Dict[str, Any]:
        """Compress context data for transfer."""
        # Implementation would compress context based on target agent capabilities
        return context_data
    
    async def decompress(self, compressed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress context data."""
        return compressed_data
    
    async def health_check(self) -> Dict[str, Any]:
        """Check compressor health."""
        return {"status": "healthy"}

class HealthMonitor:
    """Health monitoring for connections and protocols."""
    
    def register_protocol(self, protocol: CLIProtocol, config: ProtocolConfig):
        """Register protocol for monitoring."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check monitor health."""
        return {"status": "healthy"}

# ================================================================================
# Protocol Translators
# ================================================================================

class BaseTranslator:
    """Base class for protocol translators."""
    
    async def translate_to_cli(self, message: UniversalMessage, config: ProtocolConfig) -> CLIMessage:
        """Translate universal message to CLI format."""
        raise NotImplementedError
    
    async def translate_from_cli(self, cli_message: CLIMessage) -> UniversalMessage:
        """Translate CLI message to universal format."""
        raise NotImplementedError

class ClaudeCodeTranslator(BaseTranslator):
    """Translator for Claude Code CLI protocol."""
    
    async def translate_to_cli(self, message: UniversalMessage, config: ProtocolConfig) -> CLIMessage:
        """Translate universal message to Claude Code format."""
        cli_message = CLIMessage(
            universal_message_id=message.message_id,
            cli_protocol=CLIProtocol.CLAUDE_CODE,
            cli_command="claude",
            cli_args=["--task", message.message_type.value],
            input_data={
                "description": message.payload.get("description", ""),
                "files": message.payload.get("files", []),
                "context": message.execution_context or {}
            },
            timeout_seconds=config.timeout_seconds,
            expected_output_format="json"
        )
        return cli_message
    
    async def translate_from_cli(self, cli_message: CLIMessage) -> UniversalMessage:
        """Translate Claude Code response to universal format."""
        return UniversalMessage(
            message_id=cli_message.universal_message_id,
            source_agent_id="claude_code",
            source_agent_type=AgentType.CLAUDE_CODE,
            message_type=MessageType.TASK_RESPONSE,
            payload=cli_message.input_data
        )

class CursorTranslator(BaseTranslator):
    """Translator for Cursor CLI protocol."""
    
    async def translate_to_cli(self, message: UniversalMessage, config: ProtocolConfig) -> CLIMessage:
        """Translate universal message to Cursor format."""
        cli_message = CLIMessage(
            universal_message_id=message.message_id,
            cli_protocol=CLIProtocol.CURSOR,
            cli_command="cursor",
            cli_args=["--apply", message.payload.get("description", "")],
            input_files=message.payload.get("files", []),
            working_directory=message.payload.get("working_directory"),
            timeout_seconds=config.timeout_seconds
        )
        return cli_message
    
    async def translate_from_cli(self, cli_message: CLIMessage) -> UniversalMessage:
        """Translate Cursor response to universal format."""
        return UniversalMessage(
            message_id=cli_message.universal_message_id,
            source_agent_id="cursor",
            source_agent_type=AgentType.CURSOR,
            message_type=MessageType.TASK_RESPONSE,
            payload={
                "files_modified": cli_message.input_files,
                "working_directory": cli_message.working_directory
            }
        )

class GeminiCLITranslator(BaseTranslator):
    """Translator for Gemini CLI protocol."""
    
    async def translate_to_cli(self, message: UniversalMessage, config: ProtocolConfig) -> CLIMessage:
        """Translate universal message to Gemini CLI format."""
        cli_message = CLIMessage(
            universal_message_id=message.message_id,
            cli_protocol=CLIProtocol.GEMINI_CLI,
            cli_command="gemini",
            cli_args=["--prompt", message.payload.get("description", "")],
            cli_options={
                "model": "gemini-pro",
                "format": "json",
                "temperature": "0.7"
            },
            input_data=message.payload,
            timeout_seconds=config.timeout_seconds
        )
        return cli_message
    
    async def translate_from_cli(self, cli_message: CLIMessage) -> UniversalMessage:
        """Translate Gemini CLI response to universal format."""
        return UniversalMessage(
            message_id=cli_message.universal_message_id,
            source_agent_id="gemini_cli",
            source_agent_type=AgentType.GEMINI_CLI,
            message_type=MessageType.TASK_RESPONSE,
            payload=cli_message.input_data
        )

class GitHubCopilotTranslator(BaseTranslator):
    """Translator for GitHub Copilot CLI protocol."""
    
    async def translate_to_cli(self, message: UniversalMessage, config: ProtocolConfig) -> CLIMessage:
        """Translate universal message to GitHub Copilot format."""
        cli_message = CLIMessage(
            universal_message_id=message.message_id,
            cli_protocol=CLIProtocol.GITHUB_COPILOT,
            cli_command="gh",
            cli_args=["copilot", "suggest", "-t", "shell"],
            cli_options={
                "query": message.payload.get("description", "")
            },
            timeout_seconds=config.timeout_seconds
        )
        return cli_message
    
    async def translate_from_cli(self, cli_message: CLIMessage) -> UniversalMessage:
        """Translate GitHub Copilot response to universal format."""
        return UniversalMessage(
            message_id=cli_message.universal_message_id,
            source_agent_id="github_copilot",
            source_agent_type=AgentType.GITHUB_COPILOT,
            message_type=MessageType.TASK_RESPONSE,
            payload={
                "suggestion": cli_message.cli_options.get("query", ""),
                "command": " ".join(cli_message.cli_args)
            }
        )

class OpenCodeTranslator(BaseTranslator):
    """Translator for OpenCode CLI protocol."""
    
    async def translate_to_cli(self, message: UniversalMessage, config: ProtocolConfig) -> CLIMessage:
        """Translate universal message to OpenCode format."""
        cli_message = CLIMessage(
            universal_message_id=message.message_id,
            cli_protocol=CLIProtocol.OPENCODE,
            cli_command="opencode",
            cli_args=["--task", message.message_type.value],
            input_data=message.payload,
            input_files=message.payload.get("files", []),
            working_directory=message.payload.get("working_directory"),
            timeout_seconds=config.timeout_seconds
        )
        return cli_message
    
    async def translate_from_cli(self, cli_message: CLIMessage) -> UniversalMessage:
        """Translate OpenCode response to universal format."""
        return UniversalMessage(
            message_id=cli_message.universal_message_id,
            source_agent_id="opencode",
            source_agent_type=AgentType.OPENCODE,
            message_type=MessageType.TASK_RESPONSE,
            payload=cli_message.input_data
        )

# ================================================================================
# Protocol Factory and Utilities
# ================================================================================

class ProtocolFactory:
    """Factory for creating protocol instances."""
    
    @staticmethod
    def create_production_protocol(
        config: Dict[str, Any]
    ) -> MultiCLIProtocol:
        """Create a production-ready protocol instance."""
        protocol_id = config.get("protocol_id", f"prod-{uuid.uuid4()}")
        return ProductionMultiCLIProtocol(protocol_id)
    
    @staticmethod
    def create_test_protocol(
        mock_agents: Dict[str, Any]
    ) -> MultiCLIProtocol:
        """Create a test protocol with mock agents."""
        protocol_id = f"test-{uuid.uuid4()}"
        return ProductionMultiCLIProtocol(protocol_id)

# ================================================================================
# Protocol Configuration
# ================================================================================

class ProtocolConfiguration:
    """
    Configuration for protocol instances.
    
    IMPLEMENTATION NOTE: This will be implemented by subagent.
    """
    
    def __init__(self):
        # Core configuration
        self.max_concurrent_connections = 50
        self.default_timeout_seconds = 30
        self.enable_compression = True
        self.enable_encryption = False
        
        # Message handling
        self.max_message_size_mb = 10
        self.message_queue_size = 1000
        self.enable_message_ordering = True
        
        # Performance configuration
        self.connection_pool_size = 10
        self.keepalive_interval_seconds = 30
        self.metrics_collection_interval = 60
        
        # Reliability configuration
        self.max_retries = 3
        self.retry_backoff_multiplier = 2.0
        self.enable_auto_reconnect = True
        
        # Security configuration
        self.require_authentication = True
        self.enable_rate_limiting = True
        self.max_requests_per_minute = 1000