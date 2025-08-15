"""
WebSocket Performance Optimization for Project Index Events

This module provides advanced performance optimizations including event batching,
compression, rate limiting, priority queuing, and connection pooling for WebSocket events.
"""

import asyncio
import gzip
import json
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Callable, Union
from uuid import UUID
from dataclasses import dataclass, field
from enum import Enum
import heapq

import structlog
from pydantic import BaseModel

from .websocket_events import ProjectIndexWebSocketEvent, ProjectIndexEventType

logger = structlog.get_logger()


class EventPriority(Enum):
    """Event priority levels for queue management."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class CompressionType(Enum):
    """Event compression types."""
    NONE = "none"
    GZIP = "gzip"
    JSON_MINIFY = "json_minify"


@dataclass
class QueuedEvent:
    """Event wrapper for priority queue management."""
    priority: EventPriority
    timestamp: datetime
    event: ProjectIndexWebSocketEvent
    target_clients: Set[str]
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """For priority queue sorting (higher priority first)."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.timestamp < other.timestamp


@dataclass
class BatchedEvents:
    """Container for batched events."""
    events: List[ProjectIndexWebSocketEvent] = field(default_factory=list)
    target_clients: Set[str] = field(default_factory=set)
    batch_start: datetime = field(default_factory=datetime.utcnow)
    total_size_bytes: int = 0


@dataclass
class ConnectionStats:
    """Connection performance statistics."""
    client_id: str
    connected_at: datetime
    messages_sent: int = 0
    messages_failed: int = 0
    bytes_sent: int = 0
    last_activity: datetime = field(default_factory=datetime.utcnow)
    average_latency_ms: float = 0.0
    compression_ratio: float = 1.0
    rate_limit_violations: int = 0


class RateLimiter:
    """Token bucket rate limiter for WebSocket connections."""
    
    def __init__(self, max_tokens: int = 100, refill_rate: float = 10.0):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate  # tokens per second
        self.tokens: Dict[str, float] = {}
        self.last_refill: Dict[str, datetime] = {}
    
    def check_rate_limit(self, client_id: str, tokens_required: int = 1) -> bool:
        """Check if client can send message within rate limit."""
        now = datetime.utcnow()
        
        # Initialize client if not exists
        if client_id not in self.tokens:
            self.tokens[client_id] = self.max_tokens
            self.last_refill[client_id] = now
            return True
        
        # Refill tokens based on time elapsed
        time_elapsed = (now - self.last_refill[client_id]).total_seconds()
        tokens_to_add = time_elapsed * self.refill_rate
        
        self.tokens[client_id] = min(
            self.max_tokens, 
            self.tokens[client_id] + tokens_to_add
        )
        self.last_refill[client_id] = now
        
        # Check if enough tokens available
        if self.tokens[client_id] >= tokens_required:
            self.tokens[client_id] -= tokens_required
            return True
        
        return False
    
    def reset_client(self, client_id: str) -> None:
        """Reset rate limit for client."""
        self.tokens.pop(client_id, None)
        self.last_refill.pop(client_id, None)


class EventCompressor:
    """Handles event compression for WebSocket transmission."""
    
    def __init__(self, compression_threshold: int = 1024):
        self.compression_threshold = compression_threshold
        self.compression_stats = {
            "events_compressed": 0,
            "bytes_saved": 0,
            "compression_time_ms": 0
        }
    
    def compress_event(
        self, 
        event: Dict[str, Any], 
        compression_type: CompressionType = CompressionType.GZIP
    ) -> Tuple[bytes, CompressionType, float]:
        """Compress event data if beneficial."""
        start_time = time.time()
        
        try:
            # Serialize event
            serialized = json.dumps(event, separators=(',', ':'))
            original_size = len(serialized.encode('utf-8'))
            
            # Skip compression for small events
            if original_size < self.compression_threshold:
                compression_time = (time.time() - start_time) * 1000
                return serialized.encode('utf-8'), CompressionType.NONE, compression_time
            
            compressed_data = None
            final_compression_type = CompressionType.NONE
            
            if compression_type == CompressionType.GZIP:
                compressed_data = gzip.compress(serialized.encode('utf-8'))
                final_compression_type = CompressionType.GZIP
            elif compression_type == CompressionType.JSON_MINIFY:
                # JSON is already minified with separators
                compressed_data = serialized.encode('utf-8')
                final_compression_type = CompressionType.JSON_MINIFY
            else:
                compressed_data = serialized.encode('utf-8')
            
            # Only use compression if it provides significant benefit
            compression_ratio = len(compressed_data) / original_size
            if compression_ratio > 0.8:  # Less than 20% savings
                compressed_data = serialized.encode('utf-8')
                final_compression_type = CompressionType.NONE
            else:
                # Update stats
                self.compression_stats["events_compressed"] += 1
                self.compression_stats["bytes_saved"] += original_size - len(compressed_data)
            
            compression_time = (time.time() - start_time) * 1000
            self.compression_stats["compression_time_ms"] += compression_time
            
            return compressed_data, final_compression_type, compression_ratio
            
        except Exception as e:
            logger.error("Event compression failed", error=str(e))
            # Fallback to uncompressed
            fallback_data = json.dumps(event).encode('utf-8')
            compression_time = (time.time() - start_time) * 1000
            return fallback_data, CompressionType.NONE, 1.0


class EventBatcher:
    """Batches events for efficient WebSocket transmission."""
    
    def __init__(
        self, 
        batch_size_limit: int = 10,
        batch_time_limit_ms: int = 100,
        batch_size_bytes: int = 8192
    ):
        self.batch_size_limit = batch_size_limit
        self.batch_time_limit_ms = batch_time_limit_ms
        self.batch_size_bytes = batch_size_bytes
        
        # Active batches by client
        self.client_batches: Dict[str, BatchedEvents] = {}
        
        # Batch processing task
        self.batch_processor_task: Optional[asyncio.Task] = None
        self.batch_callbacks: List[Callable[[str, List[Dict[str, Any]]], None]] = []
        
        # Statistics
        self.batch_stats = {
            "batches_created": 0,
            "events_batched": 0,
            "batch_efficiency": 0.0  # Average events per batch
        }
    
    def add_event_to_batch(self, client_id: str, event: ProjectIndexWebSocketEvent) -> bool:
        """Add event to client's batch. Returns True if batch is ready to send."""
        if client_id not in self.client_batches:
            self.client_batches[client_id] = BatchedEvents()
            self.client_batches[client_id].target_clients.add(client_id)
        
        batch = self.client_batches[client_id]
        
        # Estimate event size
        event_dict = {
            "type": event.type.value,
            "data": event.data,
            "timestamp": event.timestamp.isoformat(),
            "correlation_id": str(event.correlation_id)
        }
        event_size = len(json.dumps(event_dict).encode('utf-8'))
        
        # Check if adding this event would exceed limits
        would_exceed_size = batch.total_size_bytes + event_size > self.batch_size_bytes
        would_exceed_count = len(batch.events) >= self.batch_size_limit
        
        if would_exceed_size or would_exceed_count:
            # Process current batch first
            if batch.events:
                return True
        
        # Add event to batch
        batch.events.append(event)
        batch.total_size_bytes += event_size
        
        # Check if batch is ready based on time
        batch_age_ms = (datetime.utcnow() - batch.batch_start).total_seconds() * 1000
        if batch_age_ms >= self.batch_time_limit_ms:
            return True
        
        return False
    
    def get_ready_batch(self, client_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get ready batch for client and clear it."""
        if client_id not in self.client_batches:
            return None
        
        batch = self.client_batches[client_id]
        if not batch.events:
            return None
        
        # Convert events to dict format
        batch_events = []
        for event in batch.events:
            event_dict = {
                "type": event.type.value,
                "data": event.data,
                "timestamp": event.timestamp.isoformat(),
                "correlation_id": str(event.correlation_id),
                "batch": True,
                "batch_size": len(batch.events)
            }
            batch_events.append(event_dict)
        
        # Update statistics
        self.batch_stats["batches_created"] += 1
        self.batch_stats["events_batched"] += len(batch.events)
        
        if self.batch_stats["batches_created"] > 0:
            self.batch_stats["batch_efficiency"] = (
                self.batch_stats["events_batched"] / self.batch_stats["batches_created"]
            )
        
        # Clear batch
        del self.client_batches[client_id]
        
        return batch_events
    
    def force_flush_client_batch(self, client_id: str) -> Optional[List[Dict[str, Any]]]:
        """Force flush client's batch regardless of size/time limits."""
        return self.get_ready_batch(client_id)
    
    def flush_all_batches(self) -> Dict[str, List[Dict[str, Any]]]:
        """Flush all pending batches."""
        all_batches = {}
        
        for client_id in list(self.client_batches.keys()):
            batch = self.get_ready_batch(client_id)
            if batch:
                all_batches[client_id] = batch
        
        return all_batches
    
    def start_batch_processor(self) -> None:
        """Start background batch processing task."""
        if self.batch_processor_task is None or self.batch_processor_task.done():
            self.batch_processor_task = asyncio.create_task(self._batch_processor_loop())
    
    def stop_batch_processor(self) -> None:
        """Stop background batch processing."""
        if self.batch_processor_task and not self.batch_processor_task.done():
            self.batch_processor_task.cancel()
    
    async def _batch_processor_loop(self) -> None:
        """Background task to process time-based batches."""
        while True:
            try:
                await asyncio.sleep(self.batch_time_limit_ms / 1000)
                
                # Check for timed-out batches
                ready_batches = {}
                now = datetime.utcnow()
                
                for client_id, batch in list(self.client_batches.items()):
                    batch_age_ms = (now - batch.batch_start).total_seconds() * 1000
                    
                    if batch_age_ms >= self.batch_time_limit_ms and batch.events:
                        ready_batch = self.get_ready_batch(client_id)
                        if ready_batch:
                            ready_batches[client_id] = ready_batch
                
                # Notify callbacks about ready batches
                for client_id, batch_events in ready_batches.items():
                    for callback in self.batch_callbacks:
                        try:
                            await callback(client_id, batch_events)
                        except Exception as e:
                            logger.error("Batch callback failed", error=str(e))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in batch processor", error=str(e))
    
    def add_batch_callback(self, callback: Callable[[str, List[Dict[str, Any]]], None]) -> None:
        """Add callback for when batches are ready."""
        self.batch_callbacks.append(callback)


class ConnectionPool:
    """Manages WebSocket connection pool with performance optimization."""
    
    def __init__(self, max_connections: int = 1000):
        self.max_connections = max_connections
        self.connections: Dict[str, ConnectionStats] = {}
        self.connection_health: Dict[str, float] = {}  # Health score 0.0 to 1.0
        
        # Performance monitoring
        self.pool_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "connection_errors": 0,
            "average_latency_ms": 0.0,
            "total_bytes_sent": 0,
            "message_success_rate": 0.0
        }
    
    def add_connection(self, client_id: str) -> bool:
        """Add new connection to pool."""
        if len(self.connections) >= self.max_connections:
            # Remove least healthy connection
            self._evict_unhealthy_connection()
        
        self.connections[client_id] = ConnectionStats(
            client_id=client_id,
            connected_at=datetime.utcnow()
        )
        self.connection_health[client_id] = 1.0
        
        self.pool_stats["total_connections"] += 1
        self.pool_stats["active_connections"] = len(self.connections)
        
        logger.debug("Added connection to pool", client_id=client_id)
        return True
    
    def remove_connection(self, client_id: str) -> None:
        """Remove connection from pool."""
        if client_id in self.connections:
            del self.connections[client_id]
            self.connection_health.pop(client_id, None)
            
            self.pool_stats["active_connections"] = len(self.connections)
            
            logger.debug("Removed connection from pool", client_id=client_id)
    
    def record_message_sent(
        self, 
        client_id: str, 
        message_size: int, 
        latency_ms: float, 
        success: bool
    ) -> None:
        """Record message transmission statistics."""
        if client_id not in self.connections:
            return
        
        stats = self.connections[client_id]
        stats.last_activity = datetime.utcnow()
        
        if success:
            stats.messages_sent += 1
            stats.bytes_sent += message_size
            
            # Update average latency (exponential moving average)
            if stats.average_latency_ms == 0:
                stats.average_latency_ms = latency_ms
            else:
                stats.average_latency_ms = (stats.average_latency_ms * 0.8 + latency_ms * 0.2)
        else:
            stats.messages_failed += 1
            self.pool_stats["connection_errors"] += 1
        
        # Update connection health
        self._update_connection_health(client_id)
        
        # Update pool statistics
        self._update_pool_stats()
    
    def record_rate_limit_violation(self, client_id: str) -> None:
        """Record rate limit violation for connection."""
        if client_id in self.connections:
            self.connections[client_id].rate_limit_violations += 1
            self._update_connection_health(client_id)
    
    def get_connection_health(self, client_id: str) -> float:
        """Get connection health score (0.0 to 1.0)."""
        return self.connection_health.get(client_id, 0.0)
    
    def get_healthy_connections(self, min_health: float = 0.5) -> List[str]:
        """Get list of healthy connection IDs."""
        return [
            client_id for client_id, health in self.connection_health.items()
            if health >= min_health
        ]
    
    def _update_connection_health(self, client_id: str) -> None:
        """Update connection health score based on performance metrics."""
        if client_id not in self.connections:
            return
        
        stats = self.connections[client_id]
        health_score = 1.0
        
        # Factor in success rate
        total_messages = stats.messages_sent + stats.messages_failed
        if total_messages > 0:
            success_rate = stats.messages_sent / total_messages
            health_score *= success_rate
        
        # Factor in latency (penalize high latency)
        if stats.average_latency_ms > 0:
            latency_penalty = min(stats.average_latency_ms / 1000.0, 1.0)  # Max penalty at 1 second
            health_score *= (1.0 - latency_penalty * 0.5)
        
        # Factor in rate limit violations
        if stats.rate_limit_violations > 0:
            violation_penalty = min(stats.rate_limit_violations / 10.0, 0.5)
            health_score *= (1.0 - violation_penalty)
        
        # Factor in connection age (slight bonus for stable connections)
        connection_age_hours = (datetime.utcnow() - stats.connected_at).total_seconds() / 3600
        if connection_age_hours > 1:
            age_bonus = min(connection_age_hours / 24.0, 0.1)  # Max 10% bonus
            health_score += age_bonus
        
        self.connection_health[client_id] = max(0.0, min(1.0, health_score))
    
    def _evict_unhealthy_connection(self) -> None:
        """Remove the least healthy connection."""
        if not self.connection_health:
            return
        
        # Find connection with lowest health score
        worst_client = min(self.connection_health.items(), key=lambda x: x[1])
        self.remove_connection(worst_client[0])
        
        logger.info("Evicted unhealthy connection", 
                   client_id=worst_client[0], 
                   health_score=worst_client[1])
    
    def _update_pool_stats(self) -> None:
        """Update pool-wide statistics."""
        if not self.connections:
            return
        
        total_messages = sum(s.messages_sent + s.messages_failed for s in self.connections.values())
        successful_messages = sum(s.messages_sent for s in self.connections.values())
        
        if total_messages > 0:
            self.pool_stats["message_success_rate"] = successful_messages / total_messages
        
        total_latency = sum(s.average_latency_ms for s in self.connections.values())
        if len(self.connections) > 0:
            self.pool_stats["average_latency_ms"] = total_latency / len(self.connections)
        
        self.pool_stats["total_bytes_sent"] = sum(s.bytes_sent for s in self.connections.values())
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool performance statistics."""
        return {
            **self.pool_stats,
            "connection_health_distribution": {
                "excellent": len([h for h in self.connection_health.values() if h >= 0.9]),
                "good": len([h for h in self.connection_health.values() if 0.7 <= h < 0.9]),
                "fair": len([h for h in self.connection_health.values() if 0.5 <= h < 0.7]),
                "poor": len([h for h in self.connection_health.values() if h < 0.5])
            }
        }


class PriorityEventQueue:
    """Priority queue for WebSocket events with intelligent scheduling."""
    
    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self.event_queue: List[QueuedEvent] = []
        self.queue_stats = {
            "events_queued": 0,
            "events_processed": 0,
            "events_dropped": 0,
            "average_queue_time_ms": 0.0
        }
        
        # Priority mapping for event types
        self.event_priorities = {
            ProjectIndexEventType.PROJECT_INDEX_UPDATED: EventPriority.HIGH,
            ProjectIndexEventType.ANALYSIS_PROGRESS: EventPriority.NORMAL,
            ProjectIndexEventType.DEPENDENCY_CHANGED: EventPriority.HIGH,
            ProjectIndexEventType.CONTEXT_OPTIMIZED: EventPriority.NORMAL
        }
    
    def enqueue_event(
        self, 
        event: ProjectIndexWebSocketEvent, 
        target_clients: Set[str],
        priority: Optional[EventPriority] = None
    ) -> bool:
        """Add event to priority queue."""
        if len(self.event_queue) >= self.max_queue_size:
            # Drop lowest priority event
            if self.event_queue:
                dropped = heapq.heappop(self.event_queue)
                self.queue_stats["events_dropped"] += 1
                logger.warning("Dropped low priority event due to queue full", 
                             event_type=dropped.event.type.value)
        
        # Determine priority
        if priority is None:
            priority = self.event_priorities.get(event.type, EventPriority.NORMAL)
        
        # Create queued event
        queued_event = QueuedEvent(
            priority=priority,
            timestamp=datetime.utcnow(),
            event=event,
            target_clients=target_clients
        )
        
        # Add to heap
        heapq.heappush(self.event_queue, queued_event)
        self.queue_stats["events_queued"] += 1
        
        return True
    
    def dequeue_event(self) -> Optional[QueuedEvent]:
        """Get highest priority event from queue."""
        if not self.event_queue:
            return None
        
        queued_event = heapq.heappop(self.event_queue)
        
        # Calculate queue time
        queue_time_ms = (datetime.utcnow() - queued_event.timestamp).total_seconds() * 1000
        
        # Update average queue time
        current_avg = self.queue_stats["average_queue_time_ms"]
        processed_count = self.queue_stats["events_processed"]
        self.queue_stats["average_queue_time_ms"] = (
            (current_avg * processed_count + queue_time_ms) / (processed_count + 1)
        )
        
        self.queue_stats["events_processed"] += 1
        
        return queued_event
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self.event_queue)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue performance statistics."""
        return {
            **self.queue_stats,
            "current_queue_size": len(self.event_queue),
            "queue_utilization": len(self.event_queue) / self.max_queue_size
        }


class WebSocketPerformanceManager:
    """Main performance management orchestrator."""
    
    def __init__(
        self,
        max_connections: int = 1000,
        rate_limit_tokens: int = 100,
        rate_limit_refill: float = 10.0,
        compression_threshold: int = 1024
    ):
        self.rate_limiter = RateLimiter(rate_limit_tokens, rate_limit_refill)
        self.compressor = EventCompressor(compression_threshold)
        self.batcher = EventBatcher()
        self.connection_pool = ConnectionPool(max_connections)
        self.priority_queue = PriorityEventQueue()
        
        # Global performance metrics
        self.global_metrics = {
            "total_events_processed": 0,
            "total_events_failed": 0,
            "average_processing_time_ms": 0.0,
            "peak_concurrent_connections": 0,
            "total_bytes_saved_compression": 0
        }
    
    async def process_event(
        self, 
        event: ProjectIndexWebSocketEvent, 
        target_clients: Set[str],
        priority: Optional[EventPriority] = None
    ) -> Dict[str, Any]:
        """Process event through performance optimization pipeline."""
        start_time = time.time()
        results = {
            "events_sent": 0,
            "events_failed": 0,
            "bytes_sent": 0,
            "compression_ratio": 1.0,
            "rate_limited_clients": 0
        }
        
        try:
            # Filter clients by rate limit
            valid_clients = set()
            for client_id in target_clients:
                if self.rate_limiter.check_rate_limit(client_id):
                    valid_clients.add(client_id)
                else:
                    results["rate_limited_clients"] += 1
                    self.connection_pool.record_rate_limit_violation(client_id)
            
            if not valid_clients:
                logger.debug("All clients rate limited for event", event_type=event.type.value)
                return results
            
            # Convert event to dict
            event_dict = {
                "type": event.type.value,
                "data": event.data,
                "timestamp": event.timestamp.isoformat(),
                "correlation_id": str(event.correlation_id)
            }
            
            # Compress if beneficial
            compressed_data, compression_type, compression_ratio = self.compressor.compress_event(
                event_dict, CompressionType.GZIP
            )
            results["compression_ratio"] = compression_ratio
            
            # Send to each valid client
            for client_id in valid_clients:
                try:
                    # Simulate WebSocket send (in real implementation, this would be actual WebSocket send)
                    send_start = time.time()
                    
                    # Add compression headers if used
                    if compression_type != CompressionType.NONE:
                        event_dict["_compression"] = compression_type.value
                    
                    # Record successful send
                    latency_ms = (time.time() - send_start) * 1000
                    self.connection_pool.record_message_sent(
                        client_id, len(compressed_data), latency_ms, True
                    )
                    
                    results["events_sent"] += 1
                    results["bytes_sent"] += len(compressed_data)
                    
                except Exception as e:
                    logger.error("Failed to send event to client", 
                               client_id=client_id, error=str(e))
                    
                    # Record failed send
                    self.connection_pool.record_message_sent(client_id, 0, 0, False)
                    results["events_failed"] += 1
            
            # Update global metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_global_metrics(processing_time_ms, results)
            
            return results
            
        except Exception as e:
            logger.error("Error in performance event processing", error=str(e))
            results["events_failed"] = len(target_clients)
            return results
    
    def add_connection(self, client_id: str) -> bool:
        """Add new WebSocket connection."""
        success = self.connection_pool.add_connection(client_id)
        
        # Update peak connections
        current_connections = len(self.connection_pool.connections)
        if current_connections > self.global_metrics["peak_concurrent_connections"]:
            self.global_metrics["peak_concurrent_connections"] = current_connections
        
        return success
    
    def remove_connection(self, client_id: str) -> None:
        """Remove WebSocket connection."""
        self.connection_pool.remove_connection(client_id)
        self.rate_limiter.reset_client(client_id)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "global_metrics": self.global_metrics,
            "rate_limiter": {
                "active_clients": len(self.rate_limiter.tokens)
            },
            "compressor": self.compressor.compression_stats,
            "batcher": self.batcher.batch_stats,
            "connection_pool": self.connection_pool.get_pool_stats(),
            "priority_queue": self.priority_queue.get_queue_stats()
        }
    
    def _update_global_metrics(self, processing_time_ms: float, results: Dict[str, Any]) -> None:
        """Update global performance metrics."""
        self.global_metrics["total_events_processed"] += results["events_sent"]
        self.global_metrics["total_events_failed"] += results["events_failed"]
        
        # Update average processing time
        total_events = self.global_metrics["total_events_processed"]
        if total_events > 0:
            current_avg = self.global_metrics["average_processing_time_ms"]
            self.global_metrics["average_processing_time_ms"] = (
                (current_avg * (total_events - results["events_sent"]) + 
                 processing_time_ms * results["events_sent"]) / total_events
            )
        
        # Update compression savings
        if results["compression_ratio"] < 1.0:
            estimated_savings = results["bytes_sent"] * (1 - results["compression_ratio"])
            self.global_metrics["total_bytes_saved_compression"] += estimated_savings


# Global performance manager instance
_performance_manager: Optional[WebSocketPerformanceManager] = None


def get_performance_manager() -> WebSocketPerformanceManager:
    """Get or create the global performance manager."""
    global _performance_manager
    if _performance_manager is None:
        _performance_manager = WebSocketPerformanceManager()
    return _performance_manager