#!/usr/bin/env python3
"""
Communication Protocol Testing Framework

Comprehensive testing for multi-CLI agent communication protocols.
Tests message standardization, protocol translation, Redis queue management,
WebSocket coordination, and cross-agent communication reliability.

This framework validates:
- Message format standardization across CLI types
- Protocol translation between different agent formats
- Redis queue reliability and performance
- WebSocket real-time coordination
- Error handling and recovery in communication
- Message ordering and delivery guarantees
"""

import asyncio
import json
import time
import uuid
import redis
import websockets
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import pytest
from unittest.mock import Mock, AsyncMock, patch
import threading
import queue
import msgpack
from contextlib import asynccontextmanager

class MessageType(Enum):
    """Types of messages in the communication protocol."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    CONTEXT_SHARE = "context_share"
    ERROR_REPORT = "error_report"
    HEARTBEAT = "heartbeat"
    COORDINATION = "coordination"

class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class ProtocolVersion(Enum):
    """Supported protocol versions."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"

@dataclass
class StandardMessage:
    """Standardized message format for multi-CLI coordination."""
    message_id: str
    message_type: MessageType
    protocol_version: ProtocolVersion
    source_agent: str
    target_agent: Optional[str]
    timestamp: float
    priority: MessagePriority
    content: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl: int = 300  # Time to live in seconds
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        data = asdict(self)
        # Convert enums to strings
        data['message_type'] = self.message_type.value
        data['protocol_version'] = self.protocol_version.value
        data['priority'] = self.priority.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StandardMessage':
        """Create message from dictionary."""
        # Convert string enums back to enum types
        data['message_type'] = MessageType(data['message_type'])
        data['protocol_version'] = ProtocolVersion(data['protocol_version'])
        data['priority'] = MessagePriority(data['priority'])
        return cls(**data)

class ProtocolTranslator:
    """Translates between standard message format and CLI-specific formats."""
    
    def __init__(self):
        self.translators = {
            'claude_code': self._translate_claude_code,
            'cursor': self._translate_cursor,
            'gemini_cli': self._translate_gemini_cli,
            'github_copilot': self._translate_github_copilot,
            'opencode': self._translate_opencode
        }
    
    def translate_to_cli(self, message: StandardMessage, cli_type: str) -> str:
        """Translate standard message to CLI-specific format."""
        if cli_type not in self.translators:
            raise ValueError(f"Unsupported CLI type: {cli_type}")
        
        return self.translators[cli_type](message)
    
    def translate_from_cli(self, cli_message: str, cli_type: str, source_agent: str) -> StandardMessage:
        """Translate CLI-specific format back to standard message."""
        # Parse the CLI message (this would be CLI-specific)
        try:
            data = json.loads(cli_message)
        except json.JSONDecodeError:
            # Handle non-JSON formats
            data = {"raw_content": cli_message}
        
        # Create standard message
        return StandardMessage(
            message_id=data.get('id', str(uuid.uuid4())),
            message_type=MessageType(data.get('type', 'task_request')),
            protocol_version=ProtocolVersion.V2_0,
            source_agent=source_agent,
            target_agent=data.get('target'),
            timestamp=time.time(),
            priority=MessagePriority(data.get('priority', 2)),
            content=data.get('content', {}),
            context=data.get('context', {}),
            metadata={'cli_type': cli_type, 'original_format': cli_message}
        )
    
    def _translate_claude_code(self, message: StandardMessage) -> str:
        """Translate to Claude Code CLI format."""
        claude_format = {
            "id": message.message_id,
            "type": "command",
            "command": message.content.get('task_type', 'general'),
            "args": message.content.get('args', []),
            "options": {
                "priority": message.priority.value,
                "timeout": message.ttl,
                "context": message.context
            }
        }
        return json.dumps(claude_format)
    
    def _translate_cursor(self, message: StandardMessage) -> str:
        """Translate to Cursor CLI format."""
        cursor_format = {
            "action": message.content.get('task_type', 'edit'),
            "request_id": message.message_id,
            "priority": message.priority.value,
            "payload": message.content,
            "context": message.context,
            "metadata": {
                "source": message.source_agent,
                "timestamp": message.timestamp
            }
        }
        return json.dumps(cursor_format)
    
    def _translate_gemini_cli(self, message: StandardMessage) -> str:
        """Translate to Gemini CLI format."""
        gemini_format = {
            "query_id": message.message_id,
            "query_type": message.content.get('task_type', 'analyze'),
            "input": message.content,
            "config": {
                "priority": message.priority.value,
                "max_tokens": message.content.get('max_tokens', 1000),
                "context": message.context
            }
        }
        return json.dumps(gemini_format)
    
    def _translate_github_copilot(self, message: StandardMessage) -> str:
        """Translate to GitHub Copilot format."""
        copilot_format = {
            "request": {
                "id": message.message_id,
                "type": message.content.get('task_type', 'completion'),
                "input": message.content.get('prompt', ''),
                "context": message.context
            },
            "options": {
                "priority": message.priority.value,
                "model": message.content.get('model', 'default')
            }
        }
        return json.dumps(copilot_format)
    
    def _translate_opencode(self, message: StandardMessage) -> str:
        """Translate to OpenCode format."""
        opencode_format = {
            "task": {
                "id": message.message_id,
                "type": message.content.get('task_type', 'code_gen'),
                "description": message.content.get('description', ''),
                "context": message.context,
                "priority": message.priority.value
            }
        }
        return json.dumps(opencode_format)

class RedisQueueManager:
    """Manages Redis queues for multi-agent communication."""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.queue_prefix = "multi_cli_agent"
        self.dlq_prefix = "dlq"
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_failed": 0,
            "queue_sizes": {}
        }
    
    async def send_message(self, queue_name: str, message: StandardMessage) -> bool:
        """Send message to Redis queue."""
        try:
            queue_key = f"{self.queue_prefix}:{queue_name}"
            message_data = msgpack.packb(message.to_dict())
            
            # Add to queue with priority support
            score = message.priority.value * 1000000 + time.time()
            self.redis_client.zadd(queue_key, {message_data: score})
            
            # Set TTL for message
            if message.ttl > 0:
                expiry_key = f"{queue_key}:expiry:{message.message_id}"
                self.redis_client.setex(expiry_key, message.ttl, message.message_id)
            
            self.stats["messages_sent"] += 1
            return True
        
        except Exception as e:
            self.stats["messages_failed"] += 1
            print(f"Failed to send message: {e}")
            return False
    
    async def receive_message(self, queue_name: str, timeout: int = 5) -> Optional[StandardMessage]:
        """Receive message from Redis queue."""
        try:
            queue_key = f"{self.queue_prefix}:{queue_name}"
            
            # Get highest priority message
            result = self.redis_client.bzpopmax(queue_key, timeout)
            
            if result:
                _, message_data, score = result
                message_dict = msgpack.unpackb(message_data)
                message = StandardMessage.from_dict(message_dict)
                
                # Check if message has expired
                expiry_key = f"{queue_key}:expiry:{message.message_id}"
                if self.redis_client.exists(expiry_key):
                    self.redis_client.delete(expiry_key)
                    self.stats["messages_received"] += 1
                    return message
                else:
                    # Message expired, move to DLQ
                    await self._move_to_dlq(message, "expired")
                    return None
            
            return None
        
        except Exception as e:
            self.stats["messages_failed"] += 1
            print(f"Failed to receive message: {e}")
            return None
    
    async def _move_to_dlq(self, message: StandardMessage, reason: str):
        """Move message to dead letter queue."""
        dlq_key = f"{self.dlq_prefix}:failed_messages"
        dlq_entry = {
            "message": message.to_dict(),
            "reason": reason,
            "timestamp": time.time()
        }
        self.redis_client.lpush(dlq_key, msgpack.packb(dlq_entry))
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        # Update queue sizes
        for key in self.redis_client.scan_iter(match=f"{self.queue_prefix}:*"):
            queue_name = key.decode().split(":")[-1]
            self.stats["queue_sizes"][queue_name] = self.redis_client.zcard(key)
        
        return self.stats.copy()
    
    def clear_queues(self):
        """Clear all queues for testing."""
        for key in self.redis_client.scan_iter(match=f"{self.queue_prefix}:*"):
            self.redis_client.delete(key)
        for key in self.redis_client.scan_iter(match=f"{self.dlq_prefix}:*"):
            self.redis_client.delete(key)

class WebSocketCoordinator:
    """Manages WebSocket connections for real-time coordination."""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.server = None
        self.connections = {}
        self.message_handlers = {}
        self.stats = {
            "connections": 0,
            "messages_broadcasted": 0,
            "connection_errors": 0
        }
    
    async def start_server(self):
        """Start WebSocket server."""
        try:
            self.server = await websockets.serve(
                self.handle_connection,
                "localhost",
                self.port
            )
            print(f"WebSocket server started on port {self.port}")
        except Exception as e:
            print(f"Failed to start WebSocket server: {e}")
            raise
    
    async def stop_server(self):
        """Stop WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
    
    async def handle_connection(self, websocket, path):
        """Handle new WebSocket connection."""
        agent_id = None
        try:
            # Wait for agent identification
            async for message in websocket:
                data = json.loads(message)
                
                if data.get("type") == "identify":
                    agent_id = data.get("agent_id")
                    self.connections[agent_id] = websocket
                    self.stats["connections"] += 1
                    
                    await websocket.send(json.dumps({
                        "type": "identified",
                        "agent_id": agent_id,
                        "status": "connected"
                    }))
                else:
                    # Handle coordination messages
                    await self.handle_coordination_message(data, agent_id)
        
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            self.stats["connection_errors"] += 1
            print(f"WebSocket error: {e}")
        finally:
            if agent_id and agent_id in self.connections:
                del self.connections[agent_id]
                self.stats["connections"] -= 1
    
    async def handle_coordination_message(self, data: Dict[str, Any], sender_id: str):
        """Handle coordination message from agent."""
        message_type = data.get("type")
        
        if message_type in self.message_handlers:
            await self.message_handlers[message_type](data, sender_id)
        
        # Broadcast to relevant agents
        target = data.get("target")
        if target and target in self.connections:
            await self.send_to_agent(target, data)
        elif not target:
            # Broadcast to all
            await self.broadcast_message(data, exclude=sender_id)
    
    async def send_to_agent(self, agent_id: str, message: Dict[str, Any]):
        """Send message to specific agent."""
        if agent_id in self.connections:
            try:
                await self.connections[agent_id].send(json.dumps(message))
                return True
            except Exception as e:
                print(f"Failed to send to {agent_id}: {e}")
                return False
        return False
    
    async def broadcast_message(self, message: Dict[str, Any], exclude: str = None):
        """Broadcast message to all connected agents."""
        for agent_id, websocket in self.connections.items():
            if agent_id != exclude:
                try:
                    await websocket.send(json.dumps(message))
                    self.stats["messages_broadcasted"] += 1
                except Exception as e:
                    print(f"Failed to broadcast to {agent_id}: {e}")
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register message handler."""
        self.message_handlers[message_type] = handler

class CommunicationProtocolTester:
    """Main tester for communication protocol validation."""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.translator = ProtocolTranslator()
        self.redis_manager = RedisQueueManager(redis_host, redis_port)
        self.websocket_coordinator = WebSocketCoordinator()
        self.test_results = []
    
    async def setup(self):
        """Setup testing environment."""
        # Test Redis connection
        try:
            self.redis_manager.redis_client.ping()
            print("✅ Redis connection established")
        except redis.ConnectionError:
            raise Exception("❌ Redis server not available")
        
        # Clear existing queues
        self.redis_manager.clear_queues()
        
        # Start WebSocket server
        await self.websocket_coordinator.start_server()
    
    async def cleanup(self):
        """Cleanup testing environment."""
        self.redis_manager.clear_queues()
        await self.websocket_coordinator.stop_server()
    
    async def test_message_standardization(self) -> Dict[str, Any]:
        """Test message format standardization across CLI types."""
        test_results = {
            "test_name": "Message Standardization",
            "start_time": time.time(),
            "cli_tests": [],
            "translation_errors": 0,
            "success_rate": 0.0
        }
        
        # Test message for each CLI type
        test_message = StandardMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.TASK_REQUEST,
            protocol_version=ProtocolVersion.V2_0,
            source_agent="test_orchestrator",
            target_agent="test_agent",
            timestamp=time.time(),
            priority=MessagePriority.NORMAL,
            content={
                "task_type": "code_analysis",
                "file_path": "src/main.py",
                "analysis_type": "complexity"
            },
            context={"project_root": "/path/to/project"}
        )
        
        cli_types = ['claude_code', 'cursor', 'gemini_cli', 'github_copilot', 'opencode']
        
        for cli_type in cli_types:
            cli_test = {
                "cli_type": cli_type,
                "translation_success": False,
                "round_trip_success": False,
                "format_preserved": False,
                "error": None
            }
            
            try:
                # Test translation to CLI format
                cli_format = self.translator.translate_to_cli(test_message, cli_type)
                cli_test["translation_success"] = True
                cli_test["cli_format_length"] = len(cli_format)
                
                # Test parsing CLI format back
                parsed_message = self.translator.translate_from_cli(
                    cli_format, cli_type, test_message.source_agent
                )
                cli_test["round_trip_success"] = True
                
                # Check format preservation (key fields)
                cli_test["format_preserved"] = (
                    parsed_message.message_id == test_message.message_id and
                    parsed_message.source_agent == test_message.source_agent
                )
            
            except Exception as e:
                cli_test["error"] = str(e)
                test_results["translation_errors"] += 1
            
            test_results["cli_tests"].append(cli_test)
        
        # Calculate success rate
        successful_tests = sum(1 for test in test_results["cli_tests"] 
                             if test["translation_success"] and test["round_trip_success"])
        test_results["success_rate"] = successful_tests / len(cli_types)
        test_results["end_time"] = time.time()
        
        return test_results
    
    async def test_redis_queue_reliability(self) -> Dict[str, Any]:
        """Test Redis queue reliability and performance."""
        test_results = {
            "test_name": "Redis Queue Reliability",
            "start_time": time.time(),
            "messages_sent": 0,
            "messages_received": 0,
            "messages_lost": 0,
            "average_latency": 0.0,
            "queue_performance": {},
            "priority_ordering": True
        }
        
        # Test 1: Basic send/receive
        test_queue = "test_queue"
        test_messages = []
        
        # Send test messages with different priorities
        priorities = [MessagePriority.LOW, MessagePriority.NORMAL, MessagePriority.HIGH, MessagePriority.CRITICAL]
        
        for i, priority in enumerate(priorities * 5):  # 20 messages total
            message = StandardMessage(
                message_id=f"test_msg_{i}",
                message_type=MessageType.TASK_REQUEST,
                protocol_version=ProtocolVersion.V2_0,
                source_agent="test_sender",
                target_agent="test_receiver",
                timestamp=time.time(),
                priority=priority,
                content={"test_data": f"message_{i}"},
                ttl=60
            )
            
            send_time = time.time()
            success = await self.redis_manager.send_message(test_queue, message)
            
            if success:
                test_results["messages_sent"] += 1
                test_messages.append((message, send_time))
        
        # Receive messages and check ordering
        received_messages = []
        latencies = []
        
        for _ in range(len(test_messages)):
            received = await self.redis_manager.receive_message(test_queue, timeout=2)
            receive_time = time.time()
            
            if received:
                test_results["messages_received"] += 1
                received_messages.append(received)
                
                # Find corresponding sent message to calculate latency
                for sent_msg, send_time in test_messages:
                    if sent_msg.message_id == received.message_id:
                        latency = receive_time - send_time
                        latencies.append(latency)
                        break
        
        # Check priority ordering
        for i in range(1, len(received_messages)):
            if received_messages[i-1].priority.value < received_messages[i].priority.value:
                test_results["priority_ordering"] = False
                break
        
        # Calculate metrics
        test_results["messages_lost"] = test_results["messages_sent"] - test_results["messages_received"]
        test_results["average_latency"] = sum(latencies) / len(latencies) if latencies else 0.0
        test_results["queue_performance"] = self.redis_manager.get_queue_stats()
        test_results["end_time"] = time.time()
        
        return test_results
    
    async def test_websocket_coordination(self) -> Dict[str, Any]:
        """Test WebSocket real-time coordination."""
        test_results = {
            "test_name": "WebSocket Coordination",
            "start_time": time.time(),
            "connections_established": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "coordination_latency": [],
            "connection_stability": True
        }
        
        # Simulate multiple agent connections
        async def mock_agent_client(agent_id: str, test_messages: List[Dict[str, Any]]):
            uri = f"ws://localhost:{self.websocket_coordinator.port}"
            
            try:
                async with websockets.connect(uri) as websocket:
                    # Identify agent
                    await websocket.send(json.dumps({
                        "type": "identify",
                        "agent_id": agent_id
                    }))
                    
                    # Wait for identification confirmation
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    if data.get("type") == "identified":
                        test_results["connections_established"] += 1
                    
                    # Send test messages
                    for message in test_messages:
                        send_time = time.time()
                        await websocket.send(json.dumps(message))
                        test_results["messages_sent"] += 1
                    
                    # Listen for responses
                    try:
                        while True:
                            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                            receive_time = time.time()
                            test_results["messages_received"] += 1
                            
                            # Calculate coordination latency (simplified)
                            data = json.loads(response)
                            if "timestamp" in data:
                                latency = receive_time - data["timestamp"]
                                test_results["coordination_latency"].append(latency)
                    
                    except asyncio.TimeoutError:
                        pass  # Normal timeout
            
            except Exception as e:
                test_results["connection_stability"] = False
                print(f"Agent {agent_id} connection error: {e}")
        
        # Create test agents
        agent_tasks = []
        for i in range(3):
            agent_id = f"test_agent_{i}"
            test_messages = [
                {
                    "type": "coordination",
                    "action": "task_update",
                    "agent_id": agent_id,
                    "timestamp": time.time(),
                    "data": {"status": "working", "progress": 50}
                },
                {
                    "type": "coordination",
                    "action": "request_help",
                    "agent_id": agent_id,
                    "timestamp": time.time(),
                    "target": f"test_agent_{(i+1)%3}",
                    "data": {"task": "code_review"}
                }
            ]
            
            agent_tasks.append(mock_agent_client(agent_id, test_messages))
        
        # Run agent simulations
        await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Calculate average coordination latency
        if test_results["coordination_latency"]:
            test_results["average_coordination_latency"] = (
                sum(test_results["coordination_latency"]) / 
                len(test_results["coordination_latency"])
            )
        else:
            test_results["average_coordination_latency"] = 0.0
        
        test_results["end_time"] = time.time()
        return test_results
    
    async def test_error_handling_recovery(self) -> Dict[str, Any]:
        """Test error handling and recovery mechanisms."""
        test_results = {
            "test_name": "Error Handling and Recovery",
            "start_time": time.time(),
            "error_scenarios_tested": 0,
            "recovery_successful": 0,
            "recovery_failed": 0,
            "scenario_results": []
        }
        
        # Test scenarios
        scenarios = [
            {
                "name": "Invalid message format",
                "action": lambda: self.translator.translate_to_cli(
                    StandardMessage(
                        message_id="invalid",
                        message_type="invalid_type",  # This will cause error
                        protocol_version=ProtocolVersion.V2_0,
                        source_agent="test",
                        target_agent=None,
                        timestamp=time.time(),
                        priority=MessagePriority.NORMAL,
                        content={}
                    ), "claude_code"
                )
            },
            {
                "name": "Redis connection failure",
                "action": lambda: self._simulate_redis_failure()
            },
            {
                "name": "Message TTL expiry",
                "action": lambda: self._test_message_expiry()
            },
            {
                "name": "Queue overflow",
                "action": lambda: self._test_queue_overflow()
            }
        ]
        
        for scenario in scenarios:
            scenario_result = {
                "scenario_name": scenario["name"],
                "error_triggered": False,
                "recovery_attempted": False,
                "recovery_successful": False,
                "error_details": None
            }
            
            try:
                # Attempt the action that should cause an error
                await scenario["action"]()
                scenario_result["error_triggered"] = False
            except Exception as e:
                scenario_result["error_triggered"] = True
                scenario_result["error_details"] = str(e)
                
                # Attempt recovery
                scenario_result["recovery_attempted"] = True
                try:
                    # Implement basic recovery logic
                    await self._attempt_recovery(scenario["name"])
                    scenario_result["recovery_successful"] = True
                    test_results["recovery_successful"] += 1
                except Exception as recovery_error:
                    scenario_result["recovery_successful"] = False
                    scenario_result["recovery_error"] = str(recovery_error)
                    test_results["recovery_failed"] += 1
            
            test_results["scenario_results"].append(scenario_result)
            test_results["error_scenarios_tested"] += 1
        
        test_results["end_time"] = time.time()
        return test_results
    
    async def _simulate_redis_failure(self):
        """Simulate Redis connection failure."""
        # Temporarily break Redis connection
        original_client = self.redis_manager.redis_client
        self.redis_manager.redis_client = redis.Redis(host="invalid_host", port=9999)
        
        try:
            message = StandardMessage(
                message_id="test_failure",
                message_type=MessageType.TASK_REQUEST,
                protocol_version=ProtocolVersion.V2_0,
                source_agent="test",
                target_agent=None,
                timestamp=time.time(),
                priority=MessagePriority.NORMAL,
                content={}
            )
            await self.redis_manager.send_message("test_queue", message)
        finally:
            # Restore original client
            self.redis_manager.redis_client = original_client
    
    async def _test_message_expiry(self):
        """Test message TTL expiry."""
        message = StandardMessage(
            message_id="expiry_test",
            message_type=MessageType.TASK_REQUEST,
            protocol_version=ProtocolVersion.V2_0,
            source_agent="test",
            target_agent=None,
            timestamp=time.time(),
            priority=MessagePriority.NORMAL,
            content={},
            ttl=1  # 1 second TTL
        )
        
        await self.redis_manager.send_message("expiry_test_queue", message)
        await asyncio.sleep(2)  # Wait for expiry
        
        # Try to receive expired message
        received = await self.redis_manager.receive_message("expiry_test_queue", timeout=1)
        if received:
            raise Exception("Expired message was still received")
    
    async def _test_queue_overflow(self):
        """Test queue overflow handling."""
        # Send many messages quickly
        for i in range(1000):
            message = StandardMessage(
                message_id=f"overflow_test_{i}",
                message_type=MessageType.TASK_REQUEST,
                protocol_version=ProtocolVersion.V2_0,
                source_agent="test",
                target_agent=None,
                timestamp=time.time(),
                priority=MessagePriority.NORMAL,
                content={"index": i}
            )
            
            success = await self.redis_manager.send_message("overflow_test_queue", message)
            if not success:
                raise Exception(f"Failed to send message {i} due to overflow")
    
    async def _attempt_recovery(self, scenario_name: str):
        """Attempt recovery from error scenario."""
        if scenario_name == "Redis connection failure":
            # Test if Redis is back online
            self.redis_manager.redis_client.ping()
        elif scenario_name == "Message TTL expiry":
            # Clear expired messages from DLQ
            self.redis_manager.redis_client.delete("dlq:failed_messages")
        elif scenario_name == "Queue overflow":
            # Clear overflow queue
            self.redis_manager.redis_client.delete("multi_cli_agent:overflow_test_queue")
        else:
            # Generic recovery
            await asyncio.sleep(0.1)
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive communication protocol tests."""
        suite_results = {
            "test_suite": "Communication Protocol",
            "start_time": time.time(),
            "tests_executed": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "detailed_results": []
        }
        
        try:
            await self.setup()
            
            # Test 1: Message Standardization
            print("🔄 Testing message standardization...")
            result1 = await self.test_message_standardization()
            suite_results["detailed_results"].append(result1)
            suite_results["tests_executed"] += 1
            if result1["success_rate"] >= 0.8:
                suite_results["tests_passed"] += 1
                print("✅ Message standardization - PASSED")
            else:
                suite_results["tests_failed"] += 1
                print("❌ Message standardization - FAILED")
            
            # Test 2: Redis Queue Reliability
            print("📨 Testing Redis queue reliability...")
            result2 = await self.test_redis_queue_reliability()
            suite_results["detailed_results"].append(result2)
            suite_results["tests_executed"] += 1
            if result2["messages_lost"] == 0 and result2["priority_ordering"]:
                suite_results["tests_passed"] += 1
                print("✅ Redis queue reliability - PASSED")
            else:
                suite_results["tests_failed"] += 1
                print("❌ Redis queue reliability - FAILED")
            
            # Test 3: WebSocket Coordination
            print("🌐 Testing WebSocket coordination...")
            result3 = await self.test_websocket_coordination()
            suite_results["detailed_results"].append(result3)
            suite_results["tests_executed"] += 1
            if result3["connections_established"] >= 3 and result3["connection_stability"]:
                suite_results["tests_passed"] += 1
                print("✅ WebSocket coordination - PASSED")
            else:
                suite_results["tests_failed"] += 1
                print("❌ WebSocket coordination - FAILED")
            
            # Test 4: Error Handling and Recovery
            print("🔧 Testing error handling and recovery...")
            result4 = await self.test_error_handling_recovery()
            suite_results["detailed_results"].append(result4)
            suite_results["tests_executed"] += 1
            if result4["recovery_successful"] >= result4["recovery_failed"]:
                suite_results["tests_passed"] += 1
                print("✅ Error handling and recovery - PASSED")
            else:
                suite_results["tests_failed"] += 1
                print("❌ Error handling and recovery - FAILED")
        
        except Exception as e:
            suite_results["fatal_error"] = str(e)
            print(f"❌ Fatal error in communication tests: {e}")
        
        finally:
            suite_results["end_time"] = time.time()
            suite_results["total_duration"] = suite_results["end_time"] - suite_results["start_time"]
            await self.cleanup()
        
        return suite_results

# Pytest integration
@pytest.fixture
async def protocol_tester():
    """Pytest fixture for protocol testing."""
    tester = CommunicationProtocolTester()
    yield tester

@pytest.mark.asyncio
async def test_message_translation(protocol_tester):
    """Test message translation between formats."""
    tester = protocol_tester
    
    message = StandardMessage(
        message_id="test_123",
        message_type=MessageType.TASK_REQUEST,
        protocol_version=ProtocolVersion.V2_0,
        source_agent="test_agent",
        target_agent=None,
        timestamp=time.time(),
        priority=MessagePriority.NORMAL,
        content={"task": "test"}
    )
    
    # Test translation to different CLI formats
    claude_format = tester.translator.translate_to_cli(message, "claude_code")
    cursor_format = tester.translator.translate_to_cli(message, "cursor")
    
    assert claude_format != cursor_format  # Should be different formats
    assert "test_123" in claude_format  # Should contain message ID

@pytest.mark.asyncio 
async def test_redis_message_flow(protocol_tester):
    """Test Redis message send/receive flow."""
    tester = protocol_tester
    await tester.setup()
    
    try:
        message = StandardMessage(
            message_id="redis_test",
            message_type=MessageType.TASK_REQUEST,
            protocol_version=ProtocolVersion.V2_0,
            source_agent="sender",
            target_agent="receiver",
            timestamp=time.time(),
            priority=MessagePriority.HIGH,
            content={"data": "test"}
        )
        
        # Send message
        success = await tester.redis_manager.send_message("test_queue", message)
        assert success
        
        # Receive message
        received = await tester.redis_manager.receive_message("test_queue", timeout=5)
        assert received is not None
        assert received.message_id == message.message_id
        
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    async def main():
        """Run communication protocol tests standalone."""
        print("📡 Communication Protocol Testing Suite")
        print("=" * 60)
        
        tester = CommunicationProtocolTester()
        
        try:
            results = await tester.run_comprehensive_tests()
            
            print("\n" + "=" * 60)
            print("📊 COMMUNICATION PROTOCOL TEST RESULTS")
            print("=" * 60)
            print(f"Tests Executed: {results['tests_executed']}")
            print(f"Tests Passed: {results['tests_passed']}")
            print(f"Tests Failed: {results['tests_failed']}")
            print(f"Success Rate: {results['tests_passed']/results['tests_executed']*100:.1f}%")
            print(f"Total Duration: {results['total_duration']:.2f}s")
            
            # Save detailed results
            with open('communication_protocol_test_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n📄 Detailed results saved to: communication_protocol_test_results.json")
            
        except Exception as e:
            print(f"❌ Test suite error: {str(e)}")
    
    asyncio.run(main())