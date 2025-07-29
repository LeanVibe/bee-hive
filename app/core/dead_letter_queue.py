"""
Enhanced Dead Letter Queue (DLQ) Management for LeanVibe Agent Hive 2.0 - VS 4.3

Provides comprehensive retry handling, monitoring, and replay mechanisms for failed messages
with production-grade reliability and observability. Integrates with VS 3.3 Error Handling
Framework and new VS 4.3 components.

Performance targets:
- >99.9% eventual delivery rate
- <100ms message processing overhead
- Handle 10k+ poison messages without system impact

New VS 4.3 Features:
- Advanced poison message detection and isolation
- Intelligent retry scheduling with adaptive strategies  
- Comprehensive monitoring and alerting integration
- Enhanced observability hooks integration
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

import structlog
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ..models.message import StreamMessage, MessageStatus, MessageType, MessagePriority
from ..core.config import settings
from .dlq_retry_scheduler import DLQRetryScheduler, RetryPriority, SchedulingStrategy
from .poison_message_detector import PoisonMessageDetector, PoisonDetectionResult, IsolationAction
from .dlq_monitoring import DLQMonitor
from .error_handling_integration import get_error_handling_integration

logger = structlog.get_logger()


class DLQPolicy(str, Enum):
    """Dead letter queue handling policies."""
    IMMEDIATE = "immediate"  # Move to DLQ on first failure
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Retry with exponential backoff
    LINEAR_BACKOFF = "linear_backoff"  # Retry with linear backoff
    CIRCUIT_BREAKER = "circuit_breaker"  # Circuit breaker pattern


@dataclass
class DLQConfiguration:
    """Configuration for dead letter queue behavior."""
    
    max_retries: int = 3
    initial_retry_delay_ms: int = 1000  # 1 second
    max_retry_delay_ms: int = 60000  # 1 minute
    dlq_max_size: int = 100000  # Maximum DLQ entries
    dlq_ttl_hours: int = 72  # DLQ entries TTL
    policy: DLQPolicy = DLQPolicy.EXPONENTIAL_BACKOFF
    circuit_breaker_threshold: float = 0.5  # 50% failure rate triggers circuit breaker
    circuit_breaker_window_minutes: int = 5
    
    # Monitoring and alerting
    monitor_enabled: bool = True
    alert_threshold: int = 1000  # Alert when DLQ size exceeds this
    alert_callback: Optional[Callable[[int, str], None]] = None


@dataclass
class DLQEntry:
    """Represents an entry in the dead letter queue."""
    
    original_stream: str
    original_message_id: str
    message: StreamMessage
    failure_reason: str
    retry_count: int
    first_failure_time: float
    last_failure_time: float
    next_retry_time: Optional[float] = None
    dlq_entry_id: Optional[str] = None
    
    def to_redis_dict(self) -> Dict[str, str]:
        """Convert DLQ entry to Redis-compatible dictionary."""
        return {
            "original_stream": self.original_stream,
            "original_message_id": self.original_message_id,
            "message_data": self.message.json(),
            "failure_reason": self.failure_reason,
            "retry_count": str(self.retry_count),
            "first_failure_time": str(self.first_failure_time),
            "last_failure_time": str(self.last_failure_time),
            "next_retry_time": str(self.next_retry_time) if self.next_retry_time else "",
            "entry_timestamp": str(time.time())
        }
    
    @classmethod
    def from_redis_dict(cls, data: Dict[str, str], entry_id: str) -> "DLQEntry":
        """Create DLQ entry from Redis dictionary."""
        message = StreamMessage.parse_raw(data["message_data"])
        
        return cls(
            original_stream=data["original_stream"],
            original_message_id=data["original_message_id"],
            message=message,
            failure_reason=data["failure_reason"],
            retry_count=int(data["retry_count"]),
            first_failure_time=float(data["first_failure_time"]),
            last_failure_time=float(data["last_failure_time"]),
            next_retry_time=float(data["next_retry_time"]) if data["next_retry_time"] else None,
            dlq_entry_id=entry_id
        )


class DeadLetterQueueManager:
    """
    Enhanced Dead Letter Queue management for Redis Streams - VS 4.3.
    
    Handles failed message retry logic, DLQ storage, monitoring, and replay mechanisms
    with production-grade reliability and observability features.
    
    New VS 4.3 Features:
    - Integrated poison message detection and quarantine
    - Intelligent retry scheduling with adaptive strategies
    - Real-time monitoring and alerting
    - Enhanced error handling framework integration
    - Performance optimization for >99.9% delivery rate
    """
    
    def __init__(
        self,
        redis_client: Redis,
        config: Optional[DLQConfiguration] = None,
        enable_poison_detection: bool = True,
        enable_intelligent_retry: bool = True,
        enable_monitoring: bool = True
    ):
        self.redis = redis_client
        self.config = config or DLQConfiguration()
        self.enable_poison_detection = enable_poison_detection
        self.enable_intelligent_retry = enable_intelligent_retry
        self.enable_monitoring = enable_monitoring
        
        # DLQ stream names
        self.dlq_stream = "dead_letter_queue"
        self.retry_stream = "retry_queue" 
        self.quarantine_stream = "poison_quarantine"
        self.dlq_stats_key = "dlq:stats"
        
        # VS 4.3 Components Integration
        self.retry_scheduler: Optional[DLQRetryScheduler] = None
        self.poison_detector: Optional[PoisonMessageDetector] = None
        self.dlq_monitor: Optional[DLQMonitor] = None
        self.error_integration = get_error_handling_integration()
        
        # Circuit breaker state
        self._circuit_breaker_state: Dict[str, Dict] = {}
        
        # Enhanced performance metrics
        self._metrics = {
            "messages_retried": 0,
            "messages_moved_to_dlq": 0,
            "successful_replays": 0,
            "failed_replays": 0,
            "dlq_size": 0,
            "retry_queue_size": 0,
            "poison_messages_detected": 0,
            "poison_messages_quarantined": 0,
            "average_processing_time_ms": 0.0,
            "eventual_delivery_rate": 0.0
        }
        
        # Background tasks
        self._retry_processor_task: Optional[asyncio.Task] = None
        self._dlq_monitor_task: Optional[asyncio.Task] = None
        self._circuit_breaker_task: Optional[asyncio.Task] = None
        self._poison_scanner_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start DLQ background processing tasks with VS 4.3 enhancements."""
        try:
            # Initialize VS 4.3 components
            await self._initialize_vs43_components()
            
            # Start enhanced retry processor (now uses intelligent scheduling)
            if self.enable_intelligent_retry and self.retry_scheduler:
                await self.retry_scheduler.start()
                logger.info("✅ Intelligent retry scheduler started")
            else:
                # Fallback to basic retry processor
                self._retry_processor_task = asyncio.create_task(
                    self._retry_processor_loop()
                )
            
            # Start enhanced DLQ monitor with real-time alerting
            if self.enable_monitoring and self.dlq_monitor:
                await self.dlq_monitor.start()
                logger.info("✅ Enhanced DLQ monitoring started")
            elif self.config.monitor_enabled:
                # Fallback to basic monitor
                self._dlq_monitor_task = asyncio.create_task(
                    self._dlq_monitor_loop()
                )
            
            # Start circuit breaker monitor
            self._circuit_breaker_task = asyncio.create_task(
                self._circuit_breaker_monitor_loop()
            )
            
            # Start poison message scanner
            if self.enable_poison_detection:
                self._poison_scanner_task = asyncio.create_task(
                    self._poison_scanner_loop()
                )
                logger.info("✅ Poison message scanner started")
            
            logger.info(
                "🚀 Enhanced DLQ Manager started with VS 4.3 features",
                poison_detection=self.enable_poison_detection,
                intelligent_retry=self.enable_intelligent_retry,
                monitoring=self.enable_monitoring
            )
            
        except Exception as e:
            logger.error("Failed to start Enhanced DLQ Manager", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop DLQ background processing tasks."""
        tasks = [
            self._retry_processor_task,
            self._dlq_monitor_task,
            self._circuit_breaker_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        completed_tasks = [t for t in tasks if t is not None]
        if completed_tasks:
            await asyncio.gather(*completed_tasks, return_exceptions=True)
        
        logger.info("DLQ Manager stopped")
    
    async def _initialize_vs43_components(self) -> None:
        """Initialize VS 4.3 components."""
        try:
            # Initialize poison message detector
            if self.enable_poison_detection:
                self.poison_detector = PoisonMessageDetector(
                    max_message_size_bytes=settings.MAX_MESSAGE_SIZE_BYTES,
                    detection_timeout_ms=100,  # Fast detection for performance
                    enable_adaptive_learning=True
                )
                logger.info("🔍 Poison message detector initialized")
            
            # Initialize intelligent retry scheduler
            if self.enable_intelligent_retry:
                self.retry_scheduler = DLQRetryScheduler(
                    redis_client=self.redis,
                    config=None  # Use default config
                )
                logger.info("⚡ Intelligent retry scheduler initialized")
            
            # Initialize DLQ monitor
            if self.enable_monitoring:
                self.dlq_monitor = DLQMonitor(
                    redis_client=self.redis,
                    monitoring_interval_seconds=30,
                    enable_alerting=True,
                    enable_trend_analysis=True
                )
                logger.info("📊 Enhanced DLQ monitor initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize VS 4.3 components: {e}")
            raise
    
    async def _poison_scanner_loop(self) -> None:
        """Background loop to scan for poison messages in DLQ."""
        while True:
            try:
                if not self.poison_detector:
                    await asyncio.sleep(60)
                    continue
                
                # Scan recent DLQ entries for poison characteristics
                recent_entries = await self._get_recent_dlq_entries(limit=100)
                
                for entry in recent_entries:
                    try:
                        # Analyze message for poison characteristics
                        detection_result = await self.poison_detector.analyze_message(
                            message=entry.message,
                            context={
                                "original_stream": entry.original_stream,
                                "failure_reason": entry.failure_reason,
                                "retry_count": entry.retry_count
                            }
                        )
                        
                        # Handle poison messages based on detection result
                        if detection_result.is_poison:
                            await self._handle_detected_poison_message(entry, detection_result)
                        
                    except Exception as e:
                        logger.error(f"Error analyzing message for poison: {e}")
                
                # Sleep before next scan
                await asyncio.sleep(300)  # Scan every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in poison scanner loop: {e}")
                await asyncio.sleep(300)
    
    async def _get_recent_dlq_entries(self, limit: int = 100) -> List[DLQEntry]:
        """Get recent DLQ entries for analysis."""
        try:
            # Get recent entries from DLQ stream
            entries = await self.redis.xrevrange(
                self.dlq_stream,
                count=limit
            )
            
            dlq_entries = []
            for entry_id, fields in entries:
                try:
                    # Convert bytes to strings
                    str_fields = {
                        k.decode() if isinstance(k, bytes) else k: 
                        v.decode() if isinstance(v, bytes) else v 
                        for k, v in fields.items()
                    }
                    
                    dlq_entry = DLQEntry.from_redis_dict(str_fields, entry_id.decode())
                    dlq_entries.append(dlq_entry)
                    
                except Exception as e:
                    logger.error(f"Error parsing DLQ entry {entry_id}: {e}")
            
            return dlq_entries
            
        except Exception as e:
            logger.error(f"Error getting recent DLQ entries: {e}")
            return []
    
    async def _handle_detected_poison_message(
        self,
        dlq_entry: DLQEntry,
        detection_result: PoisonDetectionResult
    ) -> None:
        """Handle a detected poison message based on detection result."""
        try:
            if detection_result.suggested_action == IsolationAction.IMMEDIATE_QUARANTINE:
                await self._quarantine_poison_message(dlq_entry, detection_result)
                self._metrics["poison_messages_quarantined"] += 1
                
            elif detection_result.suggested_action == IsolationAction.TRANSFORM_AND_RETRY:
                # Attempt to transform and reschedule
                if detection_result.is_recoverable:
                    await self._transform_and_retry_message(dlq_entry, detection_result)
                else:
                    await self._quarantine_poison_message(dlq_entry, detection_result)
                    
            elif detection_result.suggested_action == IsolationAction.HUMAN_REVIEW:
                await self._flag_for_human_review(dlq_entry, detection_result)
                
            # Update metrics
            self._metrics["poison_messages_detected"] += 1
            
            # Emit detection event to error handling framework
            await self.error_integration.emit_error_handling_failure(
                error_type="poison_message_detected",
                error_message=f"Poison message detected: {detection_result.detection_reason}",
                component="dlq_poison_scanner",
                context={
                    "poison_type": detection_result.poison_type.value if detection_result.poison_type else "unknown",
                    "confidence": detection_result.confidence.value,
                    "risk_score": detection_result.risk_score,
                    "suggested_action": detection_result.suggested_action.value,
                    "dlq_entry_id": dlq_entry.dlq_entry_id,
                    "original_stream": dlq_entry.original_stream
                }
            )
            
            logger.warning(
                "🦠 Poison message detected and handled",
                dlq_entry_id=dlq_entry.dlq_entry_id,
                poison_type=detection_result.poison_type.value if detection_result.poison_type else "unknown",
                confidence=detection_result.confidence.value,
                action_taken=detection_result.suggested_action.value
            )
            
        except Exception as e:
            logger.error(f"Error handling detected poison message: {e}")
    
    async def _quarantine_poison_message(
        self,
        dlq_entry: DLQEntry,
        detection_result: PoisonDetectionResult
    ) -> None:
        """Quarantine a poison message."""
        try:
            # Create quarantine entry
            quarantine_data = {
                "original_dlq_entry": json.dumps(dlq_entry.to_redis_dict()),
                "detection_result": json.dumps(detection_result.to_dict()),
                "quarantined_at": str(time.time()),
                "quarantine_reason": detection_result.detection_reason,
                "poison_type": detection_result.poison_type.value if detection_result.poison_type else "unknown",
                "risk_score": str(detection_result.risk_score)
            }
            
            # Add to quarantine stream
            await self.redis.xadd(
                self.quarantine_stream,
                quarantine_data,
                maxlen=10000,  # Keep up to 10k quarantined messages
                approximate=True
            )
            
            # Remove from main DLQ (optional - could keep for analysis)
            # await self.redis.xdel(self.dlq_stream, dlq_entry.dlq_entry_id)
            
        except Exception as e:
            logger.error(f"Error quarantining poison message: {e}")
    
    async def _transform_and_retry_message(
        self,
        dlq_entry: DLQEntry,
        detection_result: PoisonDetectionResult
    ) -> None:
        """Attempt to transform and retry a recoverable poison message."""
        try:
            # Apply basic transformations based on recovery suggestions
            transformed_message = dlq_entry.message
            
            for suggestion in detection_result.recovery_suggestions:
                if "fix json syntax" in suggestion.lower():
                    # Attempt basic JSON repair
                    transformed_message = await self._attempt_json_repair(transformed_message)
                elif "clean encoding" in suggestion.lower():
                    # Attempt encoding cleanup
                    transformed_message = await self._clean_message_encoding(transformed_message)
            
            # Schedule retry with lower priority and increased delay
            if self.retry_scheduler:
                retry_id = await self.retry_scheduler.schedule_retry(
                    original_stream=dlq_entry.original_stream,
                    original_message_id=dlq_entry.original_message_id,
                    message=transformed_message,
                    failure_reason=f"Transformed poison message: {detection_result.detection_reason}",
                    retry_count=dlq_entry.retry_count,
                    max_retries=max(1, self.config.max_retries - 2),  # Reduce max retries
                    priority=RetryPriority.LOW,  # Lower priority for transformed messages
                    strategy=SchedulingStrategy.LINEAR_BACKOFF  # Conservative strategy
                )
                
                logger.info(
                    "🔄 Poison message transformed and scheduled for retry",
                    retry_id=retry_id,
                    original_dlq_entry=dlq_entry.dlq_entry_id
                )
            
        except Exception as e:
            logger.error(f"Error transforming and retrying poison message: {e}")
            # Fallback: quarantine the message
            await self._quarantine_poison_message(dlq_entry, detection_result)
    
    async def _flag_for_human_review(
        self,
        dlq_entry: DLQEntry,
        detection_result: PoisonDetectionResult
    ) -> None:
        """Flag a message for human review."""
        try:
            # Add to human review queue with high priority alert
            review_data = {
                "dlq_entry": json.dumps(dlq_entry.to_redis_dict()),
                "detection_result": json.dumps(detection_result.to_dict()),
                "flagged_at": str(time.time()),
                "review_priority": "high" if detection_result.risk_score > 0.7 else "medium",
                "requires_human_review": "true"
            }
            
            await self.redis.xadd(
                "human_review_queue",
                review_data
            )
            
            logger.warning(
                "👥 Message flagged for human review",
                dlq_entry_id=dlq_entry.dlq_entry_id,
                risk_score=detection_result.risk_score,
                detection_reason=detection_result.detection_reason
            )
            
        except Exception as e:
            logger.error(f"Error flagging message for human review: {e}")
    
    async def _attempt_json_repair(self, message: StreamMessage) -> StreamMessage:
        """Attempt basic JSON repair for malformed messages."""
        try:
            # This is a simplified JSON repair - in production you'd use a more sophisticated library
            payload_str = json.dumps(message.payload)
            
            # Basic repairs
            payload_str = payload_str.replace(',}', '}')  # Remove trailing commas
            payload_str = payload_str.replace(',]', ']')  # Remove trailing commas in arrays
            
            # Try to parse repaired JSON
            repaired_payload = json.loads(payload_str)
            message.payload = repaired_payload
            
            return message
            
        except Exception:
            # If repair fails, return original message
            return message
    
    async def _clean_message_encoding(self, message: StreamMessage) -> StreamMessage:
        """Clean message encoding issues."""
        try:
            # Convert payload to string and clean encoding
            payload_str = json.dumps(message.payload)
            
            # Remove replacement characters and other encoding artifacts
            payload_str = payload_str.replace('\ufffd', '')  # Remove replacement character
            payload_str = payload_str.replace('\ufeff', '')  # Remove BOM
            
            # Re-encode as UTF-8
            payload_str = payload_str.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Parse cleaned JSON
            cleaned_payload = json.loads(payload_str)
            message.payload = cleaned_payload
            
            return message
            
        except Exception:
            # If cleaning fails, return original message
            return message
    
    async def handle_failed_message(
        self,
        original_stream: str,
        original_message_id: str,
        message: StreamMessage,
        failure_reason: str,
        current_retry_count: int = 0
    ) -> bool:
        """
        Enhanced failed message handling with VS 4.3 poison detection and intelligent retry.
        
        Returns True if message should be retried, False if moved to DLQ.
        """
        processing_start_time = time.time()
        
        try:
            # Step 1: Pre-processing poison detection (fast check)
            if self.enable_poison_detection and self.poison_detector:
                detection_result = await self.poison_detector.analyze_message(
                    message=message,
                    context={
                        "original_stream": original_stream,
                        "failure_reason": failure_reason,
                        "retry_count": current_retry_count
                    }
                )
                
                # Handle immediate quarantine cases
                if (detection_result.is_poison and 
                    detection_result.suggested_action == IsolationAction.IMMEDIATE_QUARANTINE):
                    
                    await self._quarantine_poison_message_immediate(
                        original_stream, original_message_id, message, detection_result
                    )
                    self._metrics["poison_messages_quarantined"] += 1
                    return False
            
            # Step 2: Check circuit breaker state
            if self._is_circuit_breaker_open(original_stream):
                logger.warning(
                    "Circuit breaker open, moving message to DLQ",
                    stream=original_stream,
                    message_id=original_message_id
                )
                await self._move_to_dlq(
                    original_stream, original_message_id, message, 
                    f"Circuit breaker open: {failure_reason}", current_retry_count
                )
                return False
            
            # Step 3: Intelligent retry decision
            should_retry = current_retry_count < self.config.max_retries
            
            if should_retry and self.enable_intelligent_retry and self.retry_scheduler:
                # Use intelligent retry scheduler
                retry_priority = self._determine_retry_priority(
                    original_stream, failure_reason, current_retry_count
                )
                
                retry_id = await self.retry_scheduler.schedule_retry(
                    original_stream=original_stream,
                    original_message_id=original_message_id,
                    message=message,
                    failure_reason=failure_reason,
                    retry_count=current_retry_count,
                    max_retries=self.config.max_retries,
                    priority=retry_priority
                )
                
                self._metrics["messages_retried"] += 1
                
                logger.info(
                    "✅ Message scheduled for intelligent retry",
                    stream=original_stream,
                    message_id=original_message_id,
                    retry_id=retry_id,
                    retry_count=current_retry_count + 1,
                    priority=retry_priority.value
                )
                
                return True
                
            elif should_retry:
                # Fallback to basic retry logic
                next_retry_time = self._calculate_next_retry_time(
                    current_retry_count, self.config.policy
                )
                
                await self._add_to_retry_queue(
                    original_stream, original_message_id, message,
                    failure_reason, current_retry_count, next_retry_time
                )
                
                self._metrics["messages_retried"] += 1
                
                logger.info(
                    "Message scheduled for basic retry",
                    stream=original_stream,
                    message_id=original_message_id,
                    retry_count=current_retry_count + 1,
                    next_retry=datetime.fromtimestamp(next_retry_time).isoformat()
                )
                
                return True
            else:
                # Max retries exceeded, move to DLQ
                await self._move_to_dlq(
                    original_stream, original_message_id, message,
                    f"Max retries exceeded: {failure_reason}", current_retry_count
                )
                
                self._metrics["messages_moved_to_dlq"] += 1
                
                logger.warning(
                    "Message moved to DLQ after max retries",
                    stream=original_stream,
                    message_id=original_message_id,
                    retry_count=current_retry_count
                )
                
                return False
                
        except Exception as e:
            logger.error(
                "Error in enhanced failed message handling",
                stream=original_stream,
                message_id=original_message_id,
                error=str(e)
            )
            
            # Emit error event to error handling framework
            await self.error_integration.emit_error_handling_failure(
                error_type="dlq_processing_error",
                error_message=f"Failed to process failed message: {str(e)}",
                component="dlq_manager",
                context={
                    "original_stream": original_stream,
                    "original_message_id": original_message_id,
                    "failure_reason": failure_reason,
                    "retry_count": current_retry_count,
                    "processing_time_ms": (time.time() - processing_start_time) * 1000
                }
            )
            
            # Fallback: move to DLQ
            await self._move_to_dlq(
                original_stream, original_message_id, message,
                f"DLQ handler error: {str(e)}", current_retry_count
            )
            return False
        
        finally:
            # Update processing time metrics
            processing_time_ms = (time.time() - processing_start_time) * 1000
            self._update_processing_time_metrics(processing_time_ms)
    
    async def _quarantine_poison_message_immediate(
        self,
        original_stream: str,
        original_message_id: str,
        message: StreamMessage,
        detection_result: PoisonDetectionResult
    ) -> None:
        """Immediately quarantine a poison message during failed message processing."""
        try:
            # Create quarantine entry with immediate priority
            quarantine_data = {
                "original_stream": original_stream,
                "original_message_id": original_message_id,
                "message_data": message.json(),
                "detection_result": json.dumps(detection_result.to_dict()),
                "quarantined_at": str(time.time()),
                "quarantine_reason": f"Immediate quarantine: {detection_result.detection_reason}",
                "poison_type": detection_result.poison_type.value if detection_result.poison_type else "unknown",
                "risk_score": str(detection_result.risk_score),
                "quarantine_source": "failed_message_processing"
            }
            
            # Add to quarantine stream with high priority
            await self.redis.xadd(
                self.quarantine_stream,
                quarantine_data,
                maxlen=10000,
                approximate=True
            )
            
            # Emit quarantine event
            await self.error_integration.emit_error_handling_failure(
                error_type="poison_message_quarantined",
                error_message=f"Poison message immediately quarantined: {detection_result.detection_reason}",
                component="dlq_manager",
                context={
                    "poison_type": detection_result.poison_type.value if detection_result.poison_type else "unknown",
                    "confidence": detection_result.confidence.value,
                    "risk_score": detection_result.risk_score,
                    "original_stream": original_stream,
                    "original_message_id": original_message_id
                }
            )
            
            logger.warning(
                "🦠 Poison message immediately quarantined",
                original_stream=original_stream,
                original_message_id=original_message_id,
                poison_type=detection_result.poison_type.value if detection_result.poison_type else "unknown",
                risk_score=detection_result.risk_score
            )
            
        except Exception as e:
            logger.error(f"Error immediately quarantining poison message: {e}")
    
    def _determine_retry_priority(
        self,
        original_stream: str,
        failure_reason: str,
        retry_count: int
    ) -> RetryPriority:
        """Determine appropriate retry priority based on context."""
        
        # High priority streams
        high_priority_streams = ["critical_agent_messages", "system_notifications"]
        if any(priority_stream in original_stream for priority_stream in high_priority_streams):
            return RetryPriority.HIGH
        
        # Critical failure types get higher priority
        critical_failures = ["timeout", "network_error", "connection_refused"]
        if any(critical_failure in failure_reason.lower() for critical_failure in critical_failures):
            if retry_count == 0:
                return RetryPriority.HIGH
            elif retry_count < 2:
                return RetryPriority.MEDIUM
            else:
                return RetryPriority.LOW
        
        # Validation errors get lower priority
        validation_failures = ["validation_error", "parsing_error", "invalid_format"]
        if any(validation_failure in failure_reason.lower() for validation_failure in validation_failures):
            return RetryPriority.LOW
        
        # Default priority based on retry count
        if retry_count == 0:
            return RetryPriority.MEDIUM
        elif retry_count < 2:
            return RetryPriority.MEDIUM
        else:
            return RetryPriority.LOW
    
    def _update_processing_time_metrics(self, processing_time_ms: float) -> None:
        """Update processing time metrics for performance monitoring."""
        try:
            # Update average processing time (rolling average)
            current_avg = self._metrics["average_processing_time_ms"]
            if current_avg == 0.0:
                self._metrics["average_processing_time_ms"] = processing_time_ms
            else:
                # Simple rolling average (could be improved with more sophisticated methods)
                self._metrics["average_processing_time_ms"] = (current_avg * 0.9 + processing_time_ms * 0.1)
            
            # Store processing time in Redis for monitoring
            asyncio.create_task(self._store_processing_time_metric(processing_time_ms))
            
        except Exception as e:
            logger.error(f"Error updating processing time metrics: {e}")
    
    async def _store_processing_time_metric(self, processing_time_ms: float) -> None:
        """Store processing time metric in Redis for monitoring."""
        try:
            metrics_key = "dlq:performance_metrics"
            
            # Store current processing time
            await self.redis.set(
                f"{metrics_key}:last_processing_time_ms",
                str(processing_time_ms),
                ex=3600  # Expire after 1 hour
            )
            
            # Update average processing time in Redis
            await self.redis.set(
                f"{metrics_key}:avg_processing_time_ms",
                str(self._metrics["average_processing_time_ms"]),
                ex=3600
            )
            
        except Exception as e:
            logger.error(f"Error storing processing time metric in Redis: {e}")
    
    async def _add_to_retry_queue(
        self,
        original_stream: str,
        original_message_id: str,
        message: StreamMessage,
        failure_reason: str,
        retry_count: int,
        next_retry_time: float
    ) -> None:
        """Add message to retry queue with scheduled retry time."""
        current_time = time.time()
        
        dlq_entry = DLQEntry(
            original_stream=original_stream,
            original_message_id=original_message_id,
            message=message,
            failure_reason=failure_reason,
            retry_count=retry_count + 1,
            first_failure_time=current_time,
            last_failure_time=current_time,
            next_retry_time=next_retry_time
        )
        
        # Add to retry stream with score as retry time for sorted retrieval
        await self.redis.zadd(
            self.retry_stream,
            {json.dumps(dlq_entry.to_redis_dict()): next_retry_time}
        )
    
    async def _move_to_dlq(
        self,
        original_stream: str,
        original_message_id: str,
        message: StreamMessage,
        failure_reason: str,
        retry_count: int
    ) -> None:
        """Move message to dead letter queue."""
        current_time = time.time()
        
        dlq_entry = DLQEntry(
            original_stream=original_stream,
            original_message_id=original_message_id,
            message=message,
            failure_reason=failure_reason,
            retry_count=retry_count,
            first_failure_time=current_time,
            last_failure_time=current_time
        )
        
        # Add to DLQ stream
        dlq_id = await self.redis.xadd(
            self.dlq_stream,
            dlq_entry.to_redis_dict(),
            maxlen=self.config.dlq_max_size,
            approximate=True
        )
        
        # Update DLQ size metric
        self._metrics["dlq_size"] = await self.redis.xlen(self.dlq_stream)
        
        # Update failure stats for circuit breaker
        await self._update_failure_stats(original_stream)
    
    def _calculate_next_retry_time(self, retry_count: int, policy: DLQPolicy) -> float:
        """Calculate next retry time based on policy."""
        current_time = time.time()
        
        if policy == DLQPolicy.IMMEDIATE:
            return current_time
        elif policy == DLQPolicy.LINEAR_BACKOFF:
            delay_ms = self.config.initial_retry_delay_ms * (retry_count + 1)
        elif policy == DLQPolicy.EXPONENTIAL_BACKOFF:
            delay_ms = self.config.initial_retry_delay_ms * (2 ** retry_count)
        else:  # CIRCUIT_BREAKER
            delay_ms = self.config.initial_retry_delay_ms * (2 ** retry_count)
        
        # Cap at max delay
        delay_ms = min(delay_ms, self.config.max_retry_delay_ms)
        
        return current_time + (delay_ms / 1000.0)
    
    async def _retry_processor_loop(self) -> None:
        """Background loop to process retry queue."""
        while True:
            try:
                current_time = time.time()
                
                # Get messages ready for retry (score <= current_time)
                retry_entries = await self.redis.zrangebyscore(
                    self.retry_stream,
                    min=0,
                    max=current_time,
                    withscores=True,
                    start=0,
                    num=100  # Process up to 100 retries per batch
                )
                
                for entry_data, score in retry_entries:
                    try:
                        # Parse retry entry
                        entry_dict = json.loads(entry_data)
                        dlq_entry = DLQEntry.from_redis_dict(entry_dict, "")
                        
                        # Attempt to replay message
                        success = await self._replay_message(dlq_entry)
                        
                        # Remove from retry queue
                        await self.redis.zrem(self.retry_stream, entry_data)
                        
                        if success:
                            self._metrics["successful_replays"] += 1
                            logger.debug(
                                "Message replay successful",
                                stream=dlq_entry.original_stream,
                                message_id=dlq_entry.original_message_id
                            )
                        else:
                            # Failed again, handle according to policy
                            await self.handle_failed_message(
                                dlq_entry.original_stream,
                                dlq_entry.original_message_id,
                                dlq_entry.message,
                                f"Retry failed: {dlq_entry.failure_reason}",
                                dlq_entry.retry_count
                            )
                            self._metrics["failed_replays"] += 1
                    
                    except Exception as e:
                        logger.error(f"Error processing retry entry: {e}")
                        # Remove malformed entry
                        await self.redis.zrem(self.retry_stream, entry_data)
                
                # Update retry queue size metric
                self._metrics["retry_queue_size"] = await self.redis.zcard(self.retry_stream)
                
                # Sleep before next iteration
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in retry processor loop: {e}")
                await asyncio.sleep(5)
    
    async def _replay_message(self, dlq_entry: DLQEntry) -> bool:
        """
        Attempt to replay a message to its original stream.
        
        Returns True if successful, False if failed.
        """
        try:
            # Re-add message to original stream
            await self.redis.xadd(
                dlq_entry.original_stream,
                dlq_entry.message.to_redis_dict(),
                maxlen=settings.REDIS_STREAM_MAX_LEN,
                approximate=True
            )
            
            logger.debug(
                "Message replayed to original stream",
                stream=dlq_entry.original_stream,
                message_id=dlq_entry.original_message_id
            )
            
            return True
            
        except RedisError as e:
            logger.error(
                "Failed to replay message",
                stream=dlq_entry.original_stream,
                message_id=dlq_entry.original_message_id,
                error=str(e)
            )
            return False
    
    async def _dlq_monitor_loop(self) -> None:
        """Background loop to monitor DLQ size and trigger alerts."""
        while True:
            try:
                # Update DLQ metrics
                self._metrics["dlq_size"] = await self.redis.xlen(self.dlq_stream)
                
                # Check alert threshold
                if (self._metrics["dlq_size"] > self.config.alert_threshold and 
                    self.config.alert_callback):
                    self.config.alert_callback(
                        self._metrics["dlq_size"],
                        f"DLQ size ({self._metrics['dlq_size']}) exceeded threshold ({self.config.alert_threshold})"
                    )
                
                # Clean up old DLQ entries
                await self._cleanup_old_dlq_entries()
                
                # Sleep before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in DLQ monitor loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_dlq_entries(self) -> None:
        """Remove old DLQ entries based on TTL configuration."""
        try:
            cutoff_time = time.time() - (self.config.dlq_ttl_hours * 3600)
            
            # Get old entries
            old_entries = await self.redis.xrange(
                self.dlq_stream,
                min="-",
                max=f"{int(cutoff_time * 1000)}-0",
                count=1000  # Process up to 1000 entries per cleanup
            )
            
            if old_entries:
                # Delete old entries
                entry_ids = [entry_id for entry_id, _ in old_entries]
                await self.redis.xdel(self.dlq_stream, *entry_ids)
                
                logger.info(f"Cleaned up {len(entry_ids)} old DLQ entries")
                
        except Exception as e:
            logger.error(f"Error cleaning up old DLQ entries: {e}")
    
    async def _circuit_breaker_monitor_loop(self) -> None:
        """Monitor circuit breaker state and recovery."""
        while True:
            try:
                current_time = time.time()
                window_start = current_time - (self.config.circuit_breaker_window_minutes * 60)
                
                # Check each stream's failure rate
                for stream_name in list(self._circuit_breaker_state.keys()):
                    cb_state = self._circuit_breaker_state[stream_name]
                    
                    if cb_state["state"] == "open":
                        # Check if circuit breaker should transition to half-open
                        if current_time >= cb_state["next_attempt_time"]:
                            cb_state["state"] = "half-open"
                            cb_state["half_open_attempts"] = 0
                            logger.info(f"Circuit breaker half-open for stream {stream_name}")
                    
                    elif cb_state["state"] == "half-open":
                        # Reset to closed if enough successful attempts
                        if cb_state["half_open_attempts"] >= 5:  # 5 successful attempts
                            cb_state["state"] = "closed"
                            cb_state["failure_count"] = 0
                            logger.info(f"Circuit breaker closed for stream {stream_name}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in circuit breaker monitor: {e}")
                await asyncio.sleep(30)
    
    async def _update_failure_stats(self, stream_name: str) -> None:
        """Update failure statistics for circuit breaker."""
        current_time = time.time()
        
        if stream_name not in self._circuit_breaker_state:
            self._circuit_breaker_state[stream_name] = {
                "state": "closed",  # closed, open, half-open
                "failure_count": 0,
                "last_failure_time": current_time,
                "next_attempt_time": None,
                "half_open_attempts": 0
            }
        
        cb_state = self._circuit_breaker_state[stream_name]
        cb_state["failure_count"] += 1
        cb_state["last_failure_time"] = current_time
        
        # Check if circuit breaker should open
        if (cb_state["state"] == "closed" and 
            cb_state["failure_count"] >= 10):  # 10 failures trigger circuit breaker
            
            cb_state["state"] = "open"
            cb_state["next_attempt_time"] = current_time + 60  # Try again in 1 minute
            
            logger.warning(f"Circuit breaker opened for stream {stream_name}")
    
    def _is_circuit_breaker_open(self, stream_name: str) -> bool:
        """Check if circuit breaker is open for a stream."""
        if stream_name not in self._circuit_breaker_state:
            return False
        
        return self._circuit_breaker_state[stream_name]["state"] == "open"
    
    async def get_dlq_stats(self) -> Dict[str, Any]:
        """Get comprehensive DLQ statistics."""
        dlq_size = await self.redis.xlen(self.dlq_stream)
        retry_size = await self.redis.zcard(self.retry_stream)
        
        return {
            "dlq_size": dlq_size,
            "retry_queue_size": retry_size,
            "metrics": self._metrics.copy(),
            "circuit_breaker_states": {
                stream: state["state"] 
                for stream, state in self._circuit_breaker_state.items()
            },
            "configuration": {
                "max_retries": self.config.max_retries,
                "policy": self.config.policy.value,
                "dlq_max_size": self.config.dlq_max_size,
                "dlq_ttl_hours": self.config.dlq_ttl_hours
            }
        }
    
    async def replay_dlq_messages(
        self,
        stream_filter: Optional[str] = None,
        message_type_filter: Optional[MessageType] = None,
        max_messages: int = 100
    ) -> int:
        """
        Manually replay messages from DLQ back to their original streams.
        
        Returns number of messages successfully replayed.
        """
        replayed_count = 0
        
        try:
            # Get DLQ entries
            entries = await self.redis.xrange(
                self.dlq_stream,
                min="-",
                max="+",
                count=max_messages
            )
            
            for entry_id, fields in entries:
                try:
                    # Convert fields to DLQ entry
                    str_fields = {k.decode() if isinstance(k, bytes) else k: 
                                v.decode() if isinstance(v, bytes) else v 
                                for k, v in fields.items()}
                    
                    dlq_entry = DLQEntry.from_redis_dict(str_fields, entry_id.decode())
                    
                    # Apply filters
                    if stream_filter and stream_filter not in dlq_entry.original_stream:
                        continue
                    
                    if (message_type_filter and 
                        dlq_entry.message.message_type != message_type_filter):
                        continue
                    
                    # Attempt replay
                    if await self._replay_message(dlq_entry):
                        # Remove from DLQ on successful replay
                        await self.redis.xdel(self.dlq_stream, entry_id)
                        replayed_count += 1
                        
                        logger.info(
                            "Manual DLQ replay successful",
                            stream=dlq_entry.original_stream,
                            message_id=dlq_entry.original_message_id
                        )
                
                except Exception as e:
                    logger.error(f"Error replaying DLQ entry {entry_id}: {e}")
            
            logger.info(f"Manual DLQ replay completed: {replayed_count} messages replayed")
            return replayed_count
            
        except Exception as e:
            logger.error(f"Error in manual DLQ replay: {e}")
            return replayed_count
    
    async def get_dlq_entries(
        self,
        start: str = "-",
        end: str = "+",
        count: int = 100
    ) -> List[DLQEntry]:
        """Get DLQ entries for inspection."""
        entries = []
        
        try:
            redis_entries = await self.redis.xrange(
                self.dlq_stream,
                min=start,
                max=end,
                count=count
            )
            
            for entry_id, fields in redis_entries:
                str_fields = {k.decode() if isinstance(k, bytes) else k: 
                            v.decode() if isinstance(v, bytes) else v 
                            for k, v in fields.items()}
                
                dlq_entry = DLQEntry.from_redis_dict(str_fields, entry_id.decode())
                entries.append(dlq_entry)
            
        except Exception as e:
            logger.error(f"Error getting DLQ entries: {e}")
        
        return entries