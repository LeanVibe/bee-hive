"""
Dead Letter Queue Handler for LeanVibe Agent Hive 2.0 - Vertical Slice 4.2

Provides comprehensive dead letter queue management for poison messages,
failed task recovery, and message replay capabilities with detailed analysis.

Key Features:
- Automatic poison message detection and quarantine
- DLQ monitoring with alerting and analysis
- Manual and automatic message replay capabilities
- Poison message pattern analysis for system improvement
- Failed workflow recovery with context preservation
- Comprehensive metrics and observability for DLQ operations
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
from sqlalchemy import select, update, and_, or_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from .enhanced_redis_streams_manager import EnhancedRedisStreamsManager
from .database import get_async_session
from .config import settings
from ..models.message import StreamMessage, MessageType, MessagePriority, MessageStatus

logger = structlog.get_logger()


class DLQMessageStatus(str, Enum):
    """Status of messages in DLQ."""
    QUARANTINED = "quarantined"  # Newly added to DLQ
    ANALYZED = "analyzed"  # Pattern analysis completed
    REPLAYABLE = "replayable"  # Ready for replay
    REPLAYED = "replayed"  # Successfully replayed
    PERMANENT_FAILURE = "permanent_failure"  # Cannot be recovered
    ARCHIVED = "archived"  # Archived for historical analysis


class FailureCategory(str, Enum):
    """Categories of message failures."""
    TIMEOUT = "timeout"
    PARSING_ERROR = "parsing_error"
    VALIDATION_ERROR = "validation_error"
    HANDLER_EXCEPTION = "handler_exception"
    DEPENDENCY_FAILURE = "dependency_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(str, Enum):
    """Strategies for message recovery."""
    IMMEDIATE_RETRY = "immediate_retry"
    DELAYED_RETRY = "delayed_retry"
    REQUEUE_WITH_PRIORITY = "requeue_with_priority"
    MANUAL_INTERVENTION = "manual_intervention"
    SKIP_AND_CONTINUE = "skip_and_continue"
    WORKFLOW_RESTART = "workflow_restart"
    PERMANENT_DISCARD = "permanent_discard"


@dataclass
class DLQMessage:
    """Enhanced message representation in DLQ."""
    original_message: StreamMessage
    failure_count: int
    first_failure_time: datetime
    last_failure_time: datetime
    failure_category: FailureCategory
    failure_details: Dict[str, Any]
    original_stream: str
    original_consumer_group: str
    dlq_status: DLQMessageStatus = DLQMessageStatus.QUARANTINED
    recovery_strategy: Optional[RecoveryStrategy] = None
    retry_count: int = 0
    max_retries: int = 3
    workflow_context: Optional[Dict[str, Any]] = None
    
    @property
    def dlq_id(self) -> str:
        return f"dlq_{self.original_message.id}_{int(self.first_failure_time.timestamp())}"
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['original_message'] = self.original_message.to_dict() if hasattr(self.original_message, 'to_dict') else str(self.original_message)
        result['first_failure_time'] = self.first_failure_time.isoformat()
        result['last_failure_time'] = self.last_failure_time.isoformat()
        return result


@dataclass
class DLQAnalytics:
    """Analytics for DLQ patterns and trends."""
    total_messages: int = 0
    messages_by_category: Dict[FailureCategory, int] = None
    messages_by_stream: Dict[str, int] = None
    messages_by_consumer_group: Dict[str, int] = None
    top_failure_patterns: List[Dict[str, Any]] = None
    recovery_success_rate: float = 0.0
    average_time_in_dlq_hours: float = 0.0
    messages_by_status: Dict[DLQMessageStatus, int] = None
    
    def __post_init__(self):
        if self.messages_by_category is None:
            self.messages_by_category = {}
        if self.messages_by_stream is None:
            self.messages_by_stream = {}
        if self.messages_by_consumer_group is None:
            self.messages_by_consumer_group = {}
        if self.top_failure_patterns is None:
            self.top_failure_patterns = []
        if self.messages_by_status is None:
            self.messages_by_status = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DLQMetrics:
    """Operational metrics for DLQ system."""
    messages_processed: int = 0
    messages_recovered: int = 0
    messages_permanently_failed: int = 0
    messages_replayed: int = 0
    automatic_recoveries: int = 0
    manual_interventions: int = 0
    pattern_analyses_performed: int = 0
    avg_recovery_time_hours: float = 0.0
    dlq_growth_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DLQError(Exception):
    """Base exception for DLQ operations."""
    pass


class MessageAnalysisError(DLQError):
    """Error in message analysis."""
    pass


class RecoveryError(DLQError):
    """Error in message recovery."""
    pass


class DeadLetterQueueHandler:
    """
    Comprehensive Dead Letter Queue Handler for poison message management.
    
    Provides:
    - Automatic poison message detection and quarantine
    - Intelligent failure analysis and categorization
    - Multiple recovery strategies with workflow awareness
    - Pattern analysis for system improvement
    - Manual and automatic replay capabilities
    - Comprehensive monitoring and alerting
    """
    
    def __init__(
        self,
        streams_manager: EnhancedRedisStreamsManager,
        dlq_stream_suffix: str = ":dlq",
        analysis_interval: int = 300,  # 5 minutes
        cleanup_interval: int = 3600,  # 1 hour
        max_dlq_age_days: int = 7,
        enable_automatic_recovery: bool = True,
        recovery_batch_size: int = 10
    ):
        """
        Initialize Dead Letter Queue Handler.
        
        Args:
            streams_manager: Enhanced Redis Streams Manager
            dlq_stream_suffix: Suffix for DLQ streams
            analysis_interval: Interval for pattern analysis in seconds
            cleanup_interval: Interval for cleanup operations in seconds
            max_dlq_age_days: Maximum age for messages in DLQ
            enable_automatic_recovery: Enable automatic recovery attempts
            recovery_batch_size: Batch size for recovery operations
        """
        self.streams_manager = streams_manager
        self.dlq_stream_suffix = dlq_stream_suffix
        self.analysis_interval = analysis_interval
        self.cleanup_interval = cleanup_interval
        self.max_dlq_age_days = max_dlq_age_days
        self.enable_automatic_recovery = enable_automatic_recovery
        self.recovery_batch_size = recovery_batch_size
        
        # Message storage and tracking
        self._dlq_messages: Dict[str, DLQMessage] = {}
        self._dlq_streams: Set[str] = set()
        self._failure_patterns: Dict[str, int] = defaultdict(int)
        
        # Analytics and metrics
        self._analytics = DLQAnalytics()
        self._metrics = DLQMetrics()
        
        # Background tasks
        self._analysis_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._recovery_task: Optional[asyncio.Task] = None
        
        # Recovery strategies configuration
        self._recovery_strategies = {
            FailureCategory.TIMEOUT: RecoveryStrategy.DELAYED_RETRY,
            FailureCategory.PARSING_ERROR: RecoveryStrategy.MANUAL_INTERVENTION,
            FailureCategory.VALIDATION_ERROR: RecoveryStrategy.MANUAL_INTERVENTION,
            FailureCategory.HANDLER_EXCEPTION: RecoveryStrategy.IMMEDIATE_RETRY,
            FailureCategory.DEPENDENCY_FAILURE: RecoveryStrategy.DELAYED_RETRY,
            FailureCategory.RESOURCE_EXHAUSTION: RecoveryStrategy.DELAYED_RETRY,
            FailureCategory.NETWORK_ERROR: RecoveryStrategy.IMMEDIATE_RETRY,
            FailureCategory.UNKNOWN: RecoveryStrategy.MANUAL_INTERVENTION
        }
        
        # Pattern analysis rules
        self._analysis_rules = [
            self._detect_timeout_patterns,
            self._detect_validation_patterns,
            self._detect_resource_patterns,
            self._detect_workflow_patterns
        ]
    
    async def start(self) -> None:
        """Start the DLQ handler with background tasks."""
        # Start background tasks
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.enable_automatic_recovery:
            self._recovery_task = asyncio.create_task(self._recovery_loop())
        
        # Initialize existing DLQ streams
        await self._discover_existing_dlq_streams()
        
        logger.info(
            "Dead Letter Queue Handler started",
            extra={
                "dlq_stream_suffix": self.dlq_stream_suffix,
                "analysis_interval": self.analysis_interval,
                "automatic_recovery_enabled": self.enable_automatic_recovery,
                "existing_dlq_streams": len(self._dlq_streams)
            }
        )
    
    async def stop(self) -> None:
        """Stop the DLQ handler and cleanup resources."""
        tasks = [self._analysis_task, self._cleanup_task, self._recovery_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
        
        active_tasks = [task for task in tasks if task and not task.done()]
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        
        logger.info("Dead Letter Queue Handler stopped")
    
    async def process_failed_message(
        self,
        original_message: StreamMessage,
        original_stream: str,
        consumer_group: str,
        failure_details: Dict[str, Any]
    ) -> str:
        """
        Process a failed message and add it to the appropriate DLQ.
        
        Args:
            original_message: The original message that failed
            original_stream: Original stream name
            consumer_group: Consumer group where failure occurred
            failure_details: Details about the failure
            
        Returns:
            DLQ message ID
        """
        try:
            # Categorize the failure
            failure_category = await self._categorize_failure(failure_details)
            
            # Create DLQ message
            current_time = datetime.utcnow()
            dlq_message = DLQMessage(
                original_message=original_message,
                failure_count=1,
                first_failure_time=current_time,
                last_failure_time=current_time,
                failure_category=failure_category,
                failure_details=failure_details,
                original_stream=original_stream,
                original_consumer_group=consumer_group,
                recovery_strategy=self._recovery_strategies.get(
                    failure_category, RecoveryStrategy.MANUAL_INTERVENTION
                )
            )
            
            # Check if this is a repeat failure
            existing_dlq_id = await self._find_existing_dlq_message(original_message.id)
            if existing_dlq_id and existing_dlq_id in self._dlq_messages:
                # Update existing DLQ message
                existing_msg = self._dlq_messages[existing_dlq_id]
                existing_msg.failure_count += 1
                existing_msg.last_failure_time = current_time
                existing_msg.failure_details = failure_details
                dlq_message = existing_msg
            else:
                # Store new DLQ message
                self._dlq_messages[dlq_message.dlq_id] = dlq_message
            
            # Add to DLQ stream
            dlq_stream_name = f"{original_stream}{self.dlq_stream_suffix}"
            await self._add_to_dlq_stream(dlq_stream_name, dlq_message)
            
            # Update metrics and analytics
            self._metrics.messages_processed += 1
            await self._update_analytics(dlq_message)
            
            # Check for automatic recovery eligibility
            if (self.enable_automatic_recovery and 
                dlq_message.recovery_strategy in [
                    RecoveryStrategy.IMMEDIATE_RETRY, 
                    RecoveryStrategy.DELAYED_RETRY
                ]):
                await self._schedule_automatic_recovery(dlq_message)
            
            logger.info(
                f"Processed failed message into DLQ",
                extra={
                    "dlq_id": dlq_message.dlq_id,
                    "original_message_id": original_message.id,
                    "failure_category": failure_category.value,
                    "recovery_strategy": dlq_message.recovery_strategy.value if dlq_message.recovery_strategy else None
                }
            )
            
            return dlq_message.dlq_id
            
        except Exception as e:
            logger.error(f"Failed to process failed message: {e}")
            raise DLQError(f"DLQ processing failed: {e}")
    
    async def replay_message(
        self,
        dlq_message_id: str,
        target_stream: Optional[str] = None,
        priority_boost: bool = False
    ) -> bool:
        """
        Replay a message from DLQ back to its original stream or specified target.
        
        Args:
            dlq_message_id: DLQ message identifier
            target_stream: Optional target stream (defaults to original)
            priority_boost: Whether to boost message priority
            
        Returns:
            True if replay was successful
        """
        try:
            if dlq_message_id not in self._dlq_messages:
                raise DLQError(f"DLQ message {dlq_message_id} not found")
            
            dlq_message = self._dlq_messages[dlq_message_id]
            
            # Check if message is eligible for replay
            if dlq_message.dlq_status == DLQMessageStatus.PERMANENT_FAILURE:
                raise DLQError(f"Message {dlq_message_id} marked as permanent failure")
            
            # Prepare message for replay
            replay_message = dlq_message.original_message
            
            # Boost priority if requested
            if priority_boost:
                replay_message.priority = MessagePriority.HIGH
            
            # Add replay metadata
            replay_message.payload['_dlq_replay'] = {
                "dlq_id": dlq_message_id,
                "replay_time": time.time(),
                "retry_count": dlq_message.retry_count + 1,
                "failure_category": dlq_message.failure_category.value
            }
            
            # Determine target stream
            target = target_stream or dlq_message.original_stream
            
            # Send message back to stream
            message_id = await self.streams_manager._base_manager.send_stream_message(
                stream_name=target,
                message=replay_message
            )
            
            # Update DLQ message status
            dlq_message.dlq_status = DLQMessageStatus.REPLAYED
            dlq_message.retry_count += 1
            
            # Update metrics
            self._metrics.messages_replayed += 1
            
            logger.info(
                f"Successfully replayed message {dlq_message_id}",
                extra={
                    "dlq_id": dlq_message_id,
                    "target_stream": target,
                    "new_message_id": message_id,
                    "retry_count": dlq_message.retry_count
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to replay message {dlq_message_id}: {e}")
            raise RecoveryError(f"Message replay failed: {e}")
    
    async def replay_batch(
        self,
        filter_criteria: Optional[Dict[str, Any]] = None,
        max_messages: int = 10
    ) -> Dict[str, Any]:
        """
        Replay a batch of messages based on filter criteria.
        
        Args:
            filter_criteria: Optional filters (failure_category, stream, etc.)
            max_messages: Maximum number of messages to replay
            
        Returns:
            Dictionary with replay results
        """
        try:
            # Find eligible messages
            eligible_messages = await self._find_eligible_messages(
                filter_criteria, max_messages
            )
            
            results = {
                "total_attempted": len(eligible_messages),
                "successful": 0,
                "failed": 0,
                "errors": []
            }
            
            # Replay each message
            for dlq_message_id in eligible_messages:
                try:
                    success = await self.replay_message(dlq_message_id)
                    if success:
                        results["successful"] += 1
                    else:
                        results["failed"] += 1
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append({
                        "dlq_id": dlq_message_id,
                        "error": str(e)
                    })
            
            logger.info(
                f"Batch replay completed",
                extra={
                    "attempted": results["total_attempted"],
                    "successful": results["successful"],
                    "failed": results["failed"]
                }
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Batch replay failed: {e}")
            raise RecoveryError(f"Batch replay failed: {e}")
    
    async def analyze_failure_patterns(self) -> Dict[str, Any]:
        """
        Analyze failure patterns to identify systemic issues.
        
        Returns:
            Dictionary with analysis results and recommendations
        """
        try:
            analysis_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_messages_analyzed": len(self._dlq_messages),
                "patterns": {},
                "recommendations": []
            }
            
            # Run analysis rules
            for rule in self._analysis_rules:
                try:
                    pattern_result = await rule()
                    if pattern_result:
                        analysis_results["patterns"][rule.__name__] = pattern_result
                except Exception as e:
                    logger.error(f"Error in analysis rule {rule.__name__}: {e}")
            
            # Generate recommendations based on patterns
            recommendations = await self._generate_recommendations(
                analysis_results["patterns"]
            )
            analysis_results["recommendations"] = recommendations
            
            # Update analytics
            self._metrics.pattern_analyses_performed += 1
            
            logger.info(
                f"Completed failure pattern analysis",
                extra={
                    "messages_analyzed": len(self._dlq_messages),
                    "patterns_found": len(analysis_results["patterns"]),
                    "recommendations": len(recommendations)
                }
            )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            raise MessageAnalysisError(f"Pattern analysis failed: {e}")
    
    async def get_dlq_statistics(self) -> Dict[str, Any]:
        """Get comprehensive DLQ statistics."""
        # Update analytics
        await self._refresh_analytics()
        
        return {
            "dlq_metrics": self._metrics.to_dict(),
            "dlq_analytics": self._analytics.to_dict(),
            "dlq_streams_count": len(self._dlq_streams),
            "total_dlq_messages": len(self._dlq_messages),
            "messages_by_age": await self._get_messages_by_age(),
            "top_failure_sources": await self._get_top_failure_sources(),
            "recovery_recommendations": await self._get_recovery_recommendations()
        }
    
    async def _categorize_failure(self, failure_details: Dict[str, Any]) -> FailureCategory:
        """Categorize failure based on details."""
        error_type = failure_details.get("error_type", "").lower()
        error_message = failure_details.get("error_message", "").lower()
        
        if "timeout" in error_message or "timeout" in error_type:
            return FailureCategory.TIMEOUT
        elif "parsing" in error_message or "json" in error_message:
            return FailureCategory.PARSING_ERROR
        elif "validation" in error_message or "invalid" in error_message:
            return FailureCategory.VALIDATION_ERROR
        elif "network" in error_message or "connection" in error_message:
            return FailureCategory.NETWORK_ERROR
        elif "memory" in error_message or "resource" in error_message:
            return FailureCategory.RESOURCE_EXHAUSTION
        elif "dependency" in error_message:
            return FailureCategory.DEPENDENCY_FAILURE
        elif error_type == "exception":
            return FailureCategory.HANDLER_EXCEPTION
        else:
            return FailureCategory.UNKNOWN
    
    async def _add_to_dlq_stream(self, dlq_stream_name: str, dlq_message: DLQMessage) -> None:
        """Add message to DLQ stream."""
        # Convert DLQ message to stream format
        stream_data = {
            "dlq_id": dlq_message.dlq_id,
            "original_message": json.dumps(dlq_message.original_message.to_dict()),
            "failure_category": dlq_message.failure_category.value,
            "failure_details": json.dumps(dlq_message.failure_details),
            "failure_count": str(dlq_message.failure_count),
            "first_failure_time": dlq_message.first_failure_time.isoformat(),
            "last_failure_time": dlq_message.last_failure_time.isoformat(),
            "original_stream": dlq_message.original_stream,
            "original_consumer_group": dlq_message.original_consumer_group,
            "dlq_status": dlq_message.dlq_status.value,
            "recovery_strategy": dlq_message.recovery_strategy.value if dlq_message.recovery_strategy else ""
        }
        
        # Add to Redis stream
        await self.streams_manager._base_manager._redis.xadd(
            dlq_stream_name, stream_data
        )
        
        self._dlq_streams.add(dlq_stream_name)
    
    async def _analysis_loop(self) -> None:
        """Background task for pattern analysis."""
        while True:
            try:
                await asyncio.sleep(self.analysis_interval)
                await self.analyze_failure_patterns()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Background task for cleanup operations."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_old_messages()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _recovery_loop(self) -> None:
        """Background task for automatic recovery."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._perform_automatic_recovery()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in recovery loop: {e}")
    
    async def _cleanup_old_messages(self) -> None:
        """Cleanup old messages from DLQ."""
        cutoff_time = datetime.utcnow() - timedelta(days=self.max_dlq_age_days)
        
        messages_to_archive = []
        for dlq_id, dlq_message in self._dlq_messages.items():
            if dlq_message.first_failure_time < cutoff_time:
                messages_to_archive.append(dlq_id)
        
        for dlq_id in messages_to_archive:
            dlq_message = self._dlq_messages[dlq_id]
            dlq_message.dlq_status = DLQMessageStatus.ARCHIVED
            # In a full implementation, would move to long-term storage
        
        if messages_to_archive:
            logger.info(f"Archived {len(messages_to_archive)} old DLQ messages")
    
    # Placeholder methods for complete implementation
    
    async def _discover_existing_dlq_streams(self) -> None:
        """Discover existing DLQ streams."""
        pass
    
    async def _find_existing_dlq_message(self, message_id: str) -> Optional[str]:
        """Find existing DLQ message by original message ID."""
        return None
    
    async def _update_analytics(self, dlq_message: DLQMessage) -> None:
        """Update analytics with new DLQ message."""
        pass
    
    async def _schedule_automatic_recovery(self, dlq_message: DLQMessage) -> None:
        """Schedule automatic recovery for eligible message."""
        pass
    
    async def _find_eligible_messages(self, filter_criteria: Optional[Dict[str, Any]], max_messages: int) -> List[str]:
        """Find messages eligible for replay."""
        return []
    
    async def _refresh_analytics(self) -> None:
        """Refresh analytics data."""
        pass
    
    async def _get_messages_by_age(self) -> Dict[str, int]:
        """Get message count by age groups."""
        return {}
    
    async def _get_top_failure_sources(self) -> List[Dict[str, Any]]:
        """Get top failure sources."""
        return []
    
    async def _get_recovery_recommendations(self) -> List[str]:
        """Get recovery recommendations."""
        return []
    
    async def _perform_automatic_recovery(self) -> None:
        """Perform automatic recovery for eligible messages."""
        pass
    
    async def _detect_timeout_patterns(self) -> Optional[Dict[str, Any]]:
        """Detect timeout patterns."""
        return None
    
    async def _detect_validation_patterns(self) -> Optional[Dict[str, Any]]:
        """Detect validation error patterns."""
        return None
    
    async def _detect_resource_patterns(self) -> Optional[Dict[str, Any]]:
        """Detect resource exhaustion patterns."""
        return None
    
    async def _detect_workflow_patterns(self) -> Optional[Dict[str, Any]]:
        """Detect workflow-related failure patterns."""
        return None
    
    async def _generate_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on patterns."""
        return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        dlq_size = len(self._dlq_messages)
        is_healthy = dlq_size < 10000  # Arbitrary threshold
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "dlq_message_count": dlq_size,
            "dlq_streams_count": len(self._dlq_streams),
            "background_tasks_running": sum(1 for task in [
                self._analysis_task, self._cleanup_task, self._recovery_task
            ] if task and not task.done()),
            "automatic_recovery_enabled": self.enable_automatic_recovery,
            "timestamp": datetime.utcnow().isoformat()
        }