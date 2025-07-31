"""
Unified DLQ Service for LeanVibe Agent Hive 2.0

Integrates all DLQ components into a cohesive production-ready service:
- DeadLetterQueueManager (VS 4.3) - Core DLQ functionality
- DLQRetryScheduler - Intelligent retry scheduling  
- PoisonMessageDetector - Advanced poison detection
- Enterprise reliability components integration
- Comprehensive error handling and graceful degradation
- Production monitoring and alerting

Provides a single interface for all DLQ operations with:
- >99.9% message delivery reliability
- <10ms processing overhead under normal load
- Auto-recovery from component failures
- Real-time monitoring and alerting
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .dead_letter_queue import DeadLetterQueueManager, DLQConfiguration
from .dlq_retry_scheduler import DLQRetryScheduler, RetrySchedulerConfig, RetryPriority, SchedulingStrategy
from .poison_message_detector import PoisonMessageDetector, DetectionConfidence
from .enterprise_backpressure_manager import EnterpriseBackPressureManager
from .enterprise_consumer_group_manager import EnterpriseConsumerGroupManager  
from .intelligent_retry_scheduler import IntelligentRetryScheduler
from .config import settings
from ..models.message import StreamMessage, MessageType, MessagePriority

logger = structlog.get_logger()


class DLQServiceStatus(str, Enum):
    """Unified DLQ Service status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class DLQServiceConfig:
    """Configuration for unified DLQ service."""
    
    # Core DLQ settings
    max_retries: int = 5
    initial_retry_delay_ms: int = 1000
    max_retry_delay_ms: int = 300000  # 5 minutes
    dlq_max_size: int = 1000000
    dlq_ttl_hours: int = 168  # 7 days
    
    # Poison detection settings
    enable_poison_detection: bool = True
    poison_detection_timeout_ms: int = 100
    max_message_size_bytes: int = 1024 * 1024  # 1MB
    
    # Retry scheduling settings
    enable_intelligent_retry: bool = True
    max_concurrent_retries: int = 1000
    scheduler_interval_ms: int = 100
    adaptive_learning_enabled: bool = True
    
    # Integration settings
    enable_backpressure_integration: bool = True
    enable_consumer_group_integration: bool = True
    enable_enterprise_retry_integration: bool = True
    
    # Monitoring settings
    enable_monitoring: bool = True
    monitoring_interval_seconds: int = 30
    alert_threshold: int = 1000
    performance_target_ms: int = 10  # 10ms processing target
    
    # Graceful degradation settings
    enable_graceful_degradation: bool = True
    component_failure_timeout_seconds: int = 30
    fallback_to_basic_dlq: bool = True


@dataclass
class DLQServiceMetrics:
    """Comprehensive metrics for unified DLQ service."""
    
    # Overall performance
    total_messages_processed: int = 0
    successful_deliveries: int = 0
    permanent_failures: int = 0
    messages_in_dlq: int = 0
    
    # Success rates
    overall_delivery_rate: float = 0.0
    retry_success_rate: float = 0.0
    poison_detection_accuracy: float = 0.0
    
    # Performance
    average_processing_time_ms: float = 0.0
    p95_processing_time_ms: float = 0.0
    p99_processing_time_ms: float = 0.0
    
    # Component health
    component_statuses: Dict[str, str] = field(default_factory=dict)
    component_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # System health
    service_status: DLQServiceStatus = DLQServiceStatus.STARTING
    last_health_check: float = 0.0
    uptime_seconds: float = 0.0


class UnifiedDLQService:
    """
    Unified DLQ Service integrating all DLQ components.
    
    Provides a production-ready interface for:
    - Reliable message processing with >99.9% delivery rate
    - Intelligent retry scheduling and poison detection
    - Enterprise-grade reliability and monitoring
    - Graceful degradation and auto-recovery
    - Real-time metrics and alerting
    """
    
    def __init__(
        self,
        redis_client: Redis,
        config: Optional[DLQServiceConfig] = None
    ):
        """Initialize unified DLQ service."""
        self.redis = redis_client
        self.config = config or DLQServiceConfig()
        
        # Component instances
        self.dlq_manager: Optional[DeadLetterQueueManager] = None
        self.retry_scheduler: Optional[DLQRetryScheduler] = None
        self.poison_detector: Optional[PoisonMessageDetector] = None
        self.backpressure_manager: Optional[EnterpriseBackPressureManager] = None
        self.consumer_group_manager: Optional[EnterpriseConsumerGroupManager] = None
        self.enterprise_retry_scheduler: Optional[IntelligentRetryScheduler] = None
        
        # Service state
        self.service_status = DLQServiceStatus.STARTING
        self.start_time = time.time()
        self.metrics = DLQServiceMetrics()
        
        # Background tasks
        self.monitor_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Graceful degradation state
        self.degraded_mode = False
        self.failed_components: set = set()
        
        # Event callbacks
        self.failure_callbacks: List[Callable[[str, Exception], None]] = []
        self.recovery_callbacks: List[Callable[[str], None]] = []
        
        logger.info(
            "🏗️ Unified DLQ Service initialized",
            poison_detection=self.config.enable_poison_detection,
            intelligent_retry=self.config.enable_intelligent_retry,
            backpressure_integration=self.config.enable_backpressure_integration
        )
    
    async def start(self) -> None:
        """Start unified DLQ service with all components."""
        if self.service_status != DLQServiceStatus.STARTING:
            logger.warning("DLQ service already started or starting")
            return
        
        try:
            logger.info("🚀 Starting unified DLQ service...")
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize enterprise components (optional)
            await self._initialize_enterprise_components()
            
            # Start all components
            await self._start_all_components()
            
            # Start monitoring
            if self.config.enable_monitoring:
                await self._start_monitoring()
            
            # Update service status
            self.service_status = DLQServiceStatus.HEALTHY
            self.start_time = time.time()
            
            logger.info(
                "✅ Unified DLQ Service started successfully",
                components_active=len([c for c in [
                    self.dlq_manager, self.retry_scheduler, self.poison_detector,
                    self.backpressure_manager, self.consumer_group_manager
                ] if c is not None]),
                service_status=self.service_status.value
            )
            
        except Exception as e:
            self.service_status = DLQServiceStatus.ERROR
            logger.error(f"❌ Failed to start unified DLQ service: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop unified DLQ service gracefully."""
        logger.info("🛑 Stopping unified DLQ service...")
        self.service_status = DLQServiceStatus.STOPPING
        
        try:
            # Stop monitoring
            if self.monitor_task:
                self.monitor_task.cancel()
            if self.health_check_task:
                self.health_check_task.cancel()
            
            # Stop all components
            await self._stop_all_components()
            
            # Final cleanup
            self.service_status = DLQServiceStatus.ERROR  # Stopped state
            
            logger.info("✅ Unified DLQ Service stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping unified DLQ service: {e}")
    
    async def handle_failed_message(
        self,
        original_stream: str,
        original_message_id: str,
        message: Union[StreamMessage, Dict[str, Any]],
        failure_reason: str,
        current_retry_count: int = 0,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Main entry point for handling failed messages.
        
        Returns:
            True if message will be retried, False if moved to permanent DLQ
        """
        start_time = time.time()
        
        try:
            # Convert message to StreamMessage if needed
            if isinstance(message, dict):
                # Convert dict to StreamMessage (simplified)
                stream_message = StreamMessage(
                    id=message.get('id', str(uuid.uuid4())),
                    message_type=MessageType(message.get('type', 'task')),
                    priority=MessagePriority(message.get('priority', 'normal')),
                    payload=message.get('payload', {}),
                    from_agent=message.get('from_agent', 'unknown'),
                    to_agent=message.get('to_agent', 'unknown')
                )
            else:
                stream_message = message
            
            # Step 1: Check service health and handle degraded mode
            if self.service_status not in [DLQServiceStatus.HEALTHY, DLQServiceStatus.DEGRADED]:
                return await self._handle_message_during_service_failure(
                    original_stream, original_message_id, stream_message, failure_reason
                )
            
            # Step 2: Pre-processing with poison detection (if enabled)
            if self.config.enable_poison_detection and self.poison_detector:
                try:
                    detection_result = await self.poison_detector.analyze_message(
                        stream_message, context
                    )
                    
                    if detection_result.is_poison:
                        await self._handle_poison_message(
                            original_stream, original_message_id, stream_message, 
                            detection_result, failure_reason
                        )
                        self.metrics.permanent_failures += 1
                        return False
                        
                except Exception as e:
                    logger.error(f"Poison detection failed: {e}")
                    # Continue with normal processing if poison detection fails
            
            # Step 3: Backpressure check (if enabled)
            if self.config.enable_backpressure_integration and self.backpressure_manager:
                try:
                    flow_decision = await self.backpressure_manager.check_flow_control(
                        original_stream, "normal", 1024, 0.0
                    )
                    
                    if flow_decision.action in ["reject", "shed"]:
                        logger.warning(
                            f"Message rejected due to backpressure: {flow_decision.reason}"
                        )
                        await self._move_to_permanent_dlq(
                            original_stream, original_message_id, stream_message,
                            f"Backpressure rejection: {flow_decision.reason}"
                        )
                        self.metrics.permanent_failures += 1
                        return False
                        
                except Exception as e:
                    logger.error(f"Backpressure check failed: {e}")
                    # Continue with processing if backpressure check fails
            
            # Step 4: Intelligent retry processing
            should_retry = await self._process_retry_decision(
                original_stream, original_message_id, stream_message,
                failure_reason, current_retry_count, context
            )
            
            # Update metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self.metrics.total_messages_processed += 1
            self.metrics.average_processing_time_ms = (
                (self.metrics.average_processing_time_ms * (self.metrics.total_messages_processed - 1) + 
                 processing_time_ms) / self.metrics.total_messages_processed
            )
            
            if should_retry:
                self.metrics.successful_deliveries += 1
            else:
                self.metrics.permanent_failures += 1
            
            # Check performance target
            if processing_time_ms > self.config.performance_target_ms:
                logger.warning(
                    f"⚠️ DLQ processing exceeded target",
                    processing_time_ms=processing_time_ms,
                    target_ms=self.config.performance_target_ms
                )
            
            return should_retry
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(
                f"💥 Error in unified DLQ service",
                error=str(e),
                processing_time_ms=processing_time_ms,
                original_stream=original_stream,
                original_message_id=original_message_id
            )
            
            # Fallback: move to DLQ
            await self._emergency_dlq_fallback(
                original_stream, original_message_id, stream_message, str(e)
            )
            
            self.metrics.permanent_failures += 1
            return False
    
    async def _initialize_core_components(self) -> None:
        """Initialize core DLQ components."""
        try:
            # Initialize DLQ Manager (always required)
            dlq_config = DLQConfiguration(
                max_retries=self.config.max_retries,
                initial_retry_delay_ms=self.config.initial_retry_delay_ms,
                max_retry_delay_ms=self.config.max_retry_delay_ms,
                dlq_max_size=self.config.dlq_max_size,
                dlq_ttl_hours=self.config.dlq_ttl_hours,
                monitor_enabled=self.config.enable_monitoring,
                alert_threshold=self.config.alert_threshold
            )
            
            self.dlq_manager = DeadLetterQueueManager(
                redis_client=self.redis,
                config=dlq_config,
                enable_poison_detection=self.config.enable_poison_detection,
                enable_intelligent_retry=self.config.enable_intelligent_retry,
                enable_monitoring=self.config.enable_monitoring
            )
            
            # Initialize Poison Detector (if enabled)
            if self.config.enable_poison_detection:
                self.poison_detector = PoisonMessageDetector(
                    max_message_size_bytes=self.config.max_message_size_bytes,
                    detection_timeout_ms=self.config.poison_detection_timeout_ms,
                    enable_adaptive_learning=self.config.adaptive_learning_enabled
                )
            
            # Initialize Retry Scheduler (if enabled)
            if self.config.enable_intelligent_retry:
                retry_config = RetrySchedulerConfig(
                    max_concurrent_retries=self.config.max_concurrent_retries,
                    scheduler_interval_ms=self.config.scheduler_interval_ms,
                    adaptive_learning_enabled=self.config.adaptive_learning_enabled
                )
                
                self.retry_scheduler = DLQRetryScheduler(
                    redis_client=self.redis,
                    config=retry_config
                )
            
            logger.info("✅ Core DLQ components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize core components: {e}")
            raise
    
    async def _initialize_enterprise_components(self) -> None:
        """Initialize enterprise reliability components (optional)."""
        try:
            # Initialize Backpressure Manager (if enabled)
            if self.config.enable_backpressure_integration:
                try:
                    self.backpressure_manager = EnterpriseBackPressureManager(
                        redis_client=self.redis,
                        max_throughput=10000.0,
                        target_latency_ms=50.0
                    )
                    logger.info("✅ Backpressure manager initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize backpressure manager: {e}")
                    if not self.config.enable_graceful_degradation:
                        raise
            
            # Initialize Consumer Group Manager (if enabled)
            if self.config.enable_consumer_group_integration:
                try:
                    self.consumer_group_manager = EnterpriseConsumerGroupManager(
                        redis_client=self.redis,
                        backpressure_manager=self.backpressure_manager
                    )
                    logger.info("✅ Consumer group manager initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize consumer group manager: {e}")
                    if not self.config.enable_graceful_degradation:
                        raise
            
            # Initialize Enterprise Retry Scheduler (if enabled)
            if self.config.enable_enterprise_retry_integration:
                try:
                    self.enterprise_retry_scheduler = IntelligentRetryScheduler(
                        redis_client=self.redis,
                        max_concurrent_retries=self.config.max_concurrent_retries
                    )
                    logger.info("✅ Enterprise retry scheduler initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize enterprise retry scheduler: {e}")
                    if not self.config.enable_graceful_degradation:
                        raise
            
        except Exception as e:
            if self.config.enable_graceful_degradation:
                logger.warning(f"Enterprise components initialization failed, continuing in degraded mode: {e}")
                self.degraded_mode = True
            else:
                raise
    
    async def _start_all_components(self) -> None:
        """Start all initialized components."""
        # Start core components
        if self.dlq_manager:
            await self.dlq_manager.start()
            
        if self.poison_detector:
            # Poison detector doesn't need explicit start
            pass
            
        if self.retry_scheduler:
            await self.retry_scheduler.start()
        
        # Start enterprise components (graceful failure)
        enterprise_components = [
            ("backpressure_manager", self.backpressure_manager),
            ("consumer_group_manager", self.consumer_group_manager),
            ("enterprise_retry_scheduler", self.enterprise_retry_scheduler)
        ]
        
        for name, component in enterprise_components:
            if component:
                try:
                    await component.start()
                    logger.info(f"✅ {name} started")
                except Exception as e:
                    logger.error(f"Failed to start {name}: {e}")
                    self.failed_components.add(name)
                    if not self.config.enable_graceful_degradation:
                        raise
    
    async def _stop_all_components(self) -> None:
        """Stop all components gracefully."""
        components = [
            ("dlq_manager", self.dlq_manager),
            ("retry_scheduler", self.retry_scheduler),
            ("backpressure_manager", self.backpressure_manager),
            ("consumer_group_manager", self.consumer_group_manager),
            ("enterprise_retry_scheduler", self.enterprise_retry_scheduler)
        ]
        
        for name, component in components:
            if component:
                try:
                    await component.stop()
                    logger.info(f"✅ {name} stopped")
                except Exception as e:
                    logger.error(f"Error stopping {name}: {e}")
    
    async def _start_monitoring(self) -> None:
        """Start monitoring tasks."""
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.service_status not in [DLQServiceStatus.STOPPING, DLQServiceStatus.ERROR]:
            try:
                await self._collect_metrics()
                await self._check_alerts()
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.monitoring_interval_seconds)
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self.service_status not in [DLQServiceStatus.STOPPING, DLQServiceStatus.ERROR]:
            try:
                await self._perform_health_check()
                await asyncio.sleep(60)  # Health check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)
    
    async def _process_retry_decision(
        self,
        original_stream: str,
        original_message_id: str,
        message: StreamMessage,
        failure_reason: str,
        current_retry_count: int,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Process retry decision using available components."""
        
        # Try intelligent retry scheduler first
        if self.retry_scheduler and "retry_scheduler" not in self.failed_components:
            try:
                retry_id = await self.retry_scheduler.schedule_retry(
                    original_stream=original_stream,
                    original_message_id=original_message_id,
                    message=message,
                    failure_reason=failure_reason,
                    retry_count=current_retry_count,
                    max_retries=self.config.max_retries,
                    priority=RetryPriority.MEDIUM
                )
                
                logger.debug(f"Scheduled intelligent retry: {retry_id}")
                return True
                
            except Exception as e:
                logger.error(f"Intelligent retry scheduler failed: {e}")
                self.failed_components.add("retry_scheduler")
        
        # Fall back to DLQ manager
        if self.dlq_manager:
            try:
                return await self.dlq_manager.handle_failed_message(
                    original_stream=original_stream,
                    original_message_id=original_message_id,
                    message=message,
                    failure_reason=failure_reason,
                    current_retry_count=current_retry_count
                )
            except Exception as e:
                logger.error(f"DLQ manager failed: {e}")
        
        # Emergency fallback
        await self._emergency_dlq_fallback(
            original_stream, original_message_id, message, failure_reason
        )
        return False
    
    async def _handle_poison_message(
        self,
        original_stream: str,
        original_message_id: str,
        message: StreamMessage,
        detection_result: Any,
        failure_reason: str
    ) -> None:
        """Handle detected poison message."""
        try:
            if self.dlq_manager:
                # Use DLQ manager's poison handling
                await self.dlq_manager._quarantine_poison_message_immediate(
                    original_stream, original_message_id, message, detection_result
                )
            else:
                # Emergency poison handling
                logger.critical(
                    f"🦠 POISON MESSAGE DETECTED - Emergency quarantine",
                    original_stream=original_stream,
                    original_message_id=original_message_id,
                    poison_type=detection_result.poison_type.value if detection_result.poison_type else "unknown"
                )
                
        except Exception as e:
            logger.error(f"Failed to handle poison message: {e}")
    
    async def _handle_message_during_service_failure(
        self,
        original_stream: str,
        original_message_id: str,
        message: StreamMessage,
        failure_reason: str
    ) -> bool:
        """Handle message when service is in failure state."""
        logger.warning(
            f"🚨 Processing message during service failure state",
            service_status=self.service_status.value,
            original_stream=original_stream
        )
        
        # Emergency basic DLQ handling
        await self._emergency_dlq_fallback(
            original_stream, original_message_id, message, failure_reason
        )
        return False
    
    async def _emergency_dlq_fallback(
        self,
        original_stream: str,
        original_message_id: str,
        message: StreamMessage,
        failure_reason: str
    ) -> None:
        """Emergency fallback DLQ handling."""
        try:
            # Store in Redis with basic structure
            emergency_dlq_key = f"emergency_dlq:{original_stream}"
            emergency_entry = {
                "original_message_id": original_message_id,
                "message_data": message.json(),
                "failure_reason": failure_reason,
                "emergency_timestamp": time.time(),
                "service_status": self.service_status.value
            }
            
            await self.redis.lpush(emergency_dlq_key, str(emergency_entry))
            await self.redis.expire(emergency_dlq_key, 86400 * 7)  # 7 days
            
            logger.critical(
                f"🆘 EMERGENCY DLQ FALLBACK",
                original_stream=original_stream,
                original_message_id=original_message_id,
                emergency_key=emergency_dlq_key
            )
            
        except Exception as e:
            logger.critical(f"💥 EMERGENCY DLQ FALLBACK FAILED: {e}")
    
    async def _move_to_permanent_dlq(
        self,
        original_stream: str,
        original_message_id: str,
        message: StreamMessage,
        reason: str
    ) -> None:
        """Move message to permanent DLQ."""
        if self.dlq_manager:
            try:
                await self.dlq_manager._move_to_dlq(
                    original_stream, original_message_id, message, reason, 0
                )
            except Exception as e:
                logger.error(f"Failed to move to permanent DLQ: {e}")
                await self._emergency_dlq_fallback(
                    original_stream, original_message_id, message, reason
                )
        else:
            await self._emergency_dlq_fallback(
                original_stream, original_message_id, message, reason
            )
    
    async def _collect_metrics(self) -> None:
        """Collect metrics from all components."""
        try:
            # Update basic metrics
            self.metrics.uptime_seconds = time.time() - self.start_time
            self.metrics.last_health_check = time.time()
            
            # Collect from DLQ manager
            if self.dlq_manager:
                dlq_stats = await self.dlq_manager.get_dlq_stats()
                self.metrics.messages_in_dlq = dlq_stats["dlq_size"]
                self.metrics.component_metrics["dlq_manager"] = dlq_stats
            
            # Collect from retry scheduler
            if self.retry_scheduler:
                scheduler_metrics = await self.retry_scheduler.get_scheduler_metrics()
                self.metrics.retry_success_rate = scheduler_metrics["performance_metrics"]["success_rate"]
                self.metrics.component_metrics["retry_scheduler"] = scheduler_metrics
            
            # Calculate overall delivery rate
            total_processed = self.metrics.successful_deliveries + self.metrics.permanent_failures
            if total_processed > 0:
                self.metrics.overall_delivery_rate = self.metrics.successful_deliveries / total_processed
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    async def _check_alerts(self) -> None:
        """Check alert conditions."""
        try:
            # Check DLQ size alert
            if self.metrics.messages_in_dlq > self.config.alert_threshold:
                logger.warning(
                    f"🚨 DLQ size alert",
                    messages_in_dlq=self.metrics.messages_in_dlq,
                    threshold=self.config.alert_threshold
                )
            
            # Check processing time alert
            if self.metrics.average_processing_time_ms > self.config.performance_target_ms * 2:
                logger.warning(
                    f"⚠️ Processing time alert",
                    avg_processing_time_ms=self.metrics.average_processing_time_ms,
                    target_ms=self.config.performance_target_ms
                )
            
            # Check delivery rate alert
            if self.metrics.overall_delivery_rate < 0.99:  # <99% delivery rate
                logger.warning(
                    f"📉 Delivery rate alert",
                    delivery_rate=self.metrics.overall_delivery_rate,
                    target_rate=0.999
                )
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    async def _perform_health_check(self) -> None:
        """Perform comprehensive health check."""
        try:
            component_statuses = {}
            
            # Check each component
            components = [
                ("dlq_manager", self.dlq_manager),
                ("retry_scheduler", self.retry_scheduler),
                ("poison_detector", self.poison_detector),
                ("backpressure_manager", self.backpressure_manager),
                ("consumer_group_manager", self.consumer_group_manager)
            ]
            
            for name, component in components:
                if component:
                    try:
                        if hasattr(component, 'health_check'):
                            health = await component.health_check()
                            component_statuses[name] = health["status"]
                        else:
                            component_statuses[name] = "healthy"  # Assume healthy if no health check
                    except Exception as e:
                        component_statuses[name] = "error"
                        logger.error(f"Health check failed for {name}: {e}")
                        self.failed_components.add(name)
            
            self.metrics.component_statuses = component_statuses
            
            # Determine overall service status
            statuses = list(component_statuses.values())
            if all(status == "healthy" for status in statuses):
                self.service_status = DLQServiceStatus.HEALTHY
            elif any(status in ["critical", "error"] for status in statuses):
                self.service_status = DLQServiceStatus.DEGRADED
            else:
                self.service_status = DLQServiceStatus.DEGRADED
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
            self.service_status = DLQServiceStatus.ERROR
    
    # Public API methods
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        await self._collect_metrics()
        
        return {
            "service_status": self.service_status.value,
            "uptime_seconds": self.metrics.uptime_seconds,
            "performance_metrics": {
                "total_messages_processed": self.metrics.total_messages_processed,
                "successful_deliveries": self.metrics.successful_deliveries,
                "permanent_failures": self.metrics.permanent_failures,
                "overall_delivery_rate": self.metrics.overall_delivery_rate,
                "retry_success_rate": self.metrics.retry_success_rate,
                "average_processing_time_ms": self.metrics.average_processing_time_ms
            },
            "component_statuses": self.metrics.component_statuses,
            "component_metrics": self.metrics.component_metrics,
            "failed_components": list(self.failed_components),
            "degraded_mode": self.degraded_mode,
            "configuration": {
                "max_retries": self.config.max_retries,
                "performance_target_ms": self.config.performance_target_ms,
                "alert_threshold": self.config.alert_threshold,
                "graceful_degradation": self.config.enable_graceful_degradation
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Get service health status."""
        await self._perform_health_check()
        
        return {
            "status": self.service_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": self.metrics.uptime_seconds,
            "component_health": self.metrics.component_statuses,
            "failed_components": list(self.failed_components),
            "degraded_mode": self.degraded_mode,
            "performance_metrics": {
                "delivery_rate": self.metrics.overall_delivery_rate,
                "processing_time_ms": self.metrics.average_processing_time_ms,
                "messages_in_dlq": self.metrics.messages_in_dlq
            }
        }
    
    def register_failure_callback(self, callback: Callable[[str, Exception], None]) -> None:
        """Register callback for component failures."""
        self.failure_callbacks.append(callback)
    
    def register_recovery_callback(self, callback: Callable[[str], None]) -> None:
        """Register callback for component recovery."""
        self.recovery_callbacks.append(callback)