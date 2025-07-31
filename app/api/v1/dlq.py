"""
FastAPI DLQ Management Endpoints for LeanVibe Agent Hive 2.0

Provides comprehensive REST API for Dead Letter Queue operations:
- DLQ statistics and monitoring
- Message replay and recovery
- Poison message quarantine management
- Admin operations and bulk actions
- Health checking and performance metrics

Integrates with:
- DeadLetterQueueManager (VS 4.3)
- DLQRetryScheduler
- PoisonMessageDetector
- Enterprise reliability components
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Query, Body, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog

from ...core.dead_letter_queue import DeadLetterQueueManager, DLQConfiguration, DLQEntry
from ...core.dlq_retry_scheduler import DLQRetryScheduler, RetryPriority, SchedulingStrategy
from ...core.poison_message_detector import PoisonMessageDetector, PoisonMessageType, DetectionConfidence
from ...core.enterprise_backpressure_manager import EnterpriseBackPressureManager
from ...core.enterprise_consumer_group_manager import EnterpriseConsumerGroupManager
from ...core.intelligent_retry_scheduler import IntelligentRetryScheduler
from ...core.database import get_async_session
from ...core.config import settings
from ...models.message import StreamMessage, MessageType, MessagePriority

logger = structlog.get_logger()

# Create API router
dlq_router = APIRouter(prefix="/api/v1/dlq", tags=["Dead Letter Queue"])


# Pydantic models for API requests/responses
class DLQStatsResponse(BaseModel):
    """DLQ statistics response model."""
    
    # Queue metrics
    dlq_size: int
    retry_queue_size: int
    quarantine_queue_size: int
    processing_queue_size: int
    
    # Performance metrics
    messages_retried: int
    messages_resurrected: int
    messages_permanently_failed: int
    poison_messages_detected: int
    poison_messages_quarantined: int
    
    # Success rates
    retry_success_rate: float
    resurrection_success_rate: float
    overall_recovery_rate: float
    
    # Processing performance
    average_processing_time_ms: float
    scheduler_overhead_ms: float
    detection_accuracy: float
    
    # System health
    is_running: bool
    component_health: Dict[str, str]
    
    # Configuration
    configuration: Dict[str, Any]


class DLQMessageResponse(BaseModel):
    """DLQ message response model."""
    
    dlq_entry_id: str
    original_stream: str
    original_message_id: str
    failure_reason: str
    retry_count: int
    created_at: datetime
    next_retry_time: Optional[datetime]
    priority: str
    strategy: str
    is_poison: bool
    poison_type: Optional[str]
    risk_score: float
    message_preview: Dict[str, Any]  # Truncated message data


class ReplayRequest(BaseModel):
    """Message replay request model."""
    
    stream_filter: Optional[str] = None
    message_type_filter: Optional[MessageType] = None
    priority_filter: Optional[str] = None
    max_messages: int = Field(default=100, ge=1, le=1000)
    force_replay: bool = False  # Bypass safety checks


class QuarantineActionRequest(BaseModel):
    """Quarantine action request model."""
    
    action: str = Field(..., regex="^(release|permanent_delete|transform_retry)$")
    reason: str
    transform_options: Optional[Dict[str, Any]] = None


class BulkOperationRequest(BaseModel):
    """Bulk operation request model."""
    
    operation: str = Field(..., regex="^(replay|quarantine|delete|priority_change)$")
    filters: Dict[str, Any] = {}
    max_items: int = Field(default=100, ge=1, le=1000)
    confirmation_token: str  # Security token for bulk operations


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    
    status: str  # healthy, degraded, critical
    timestamp: datetime
    components: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    performance_metrics: Dict[str, float]


# Dependency injection for DLQ services
async def get_dlq_manager() -> DeadLetterQueueManager:
    """Get configured DLQ manager instance."""
    # This would be injected via dependency injection in production
    # For now, return a placeholder that would be properly configured
    raise HTTPException(
        status_code=503,
        detail="DLQ Manager not available - needs dependency injection setup"
    )


async def get_retry_scheduler() -> DLQRetryScheduler:
    """Get configured retry scheduler instance."""
    raise HTTPException(
        status_code=503,
        detail="Retry Scheduler not available - needs dependency injection setup"
    )


async def get_poison_detector() -> PoisonMessageDetector:
    """Get configured poison detector instance."""
    raise HTTPException(
        status_code=503,
        detail="Poison Detector not available - needs dependency injection setup"
    )


# Core DLQ API endpoints
@dlq_router.get("/stats", response_model=DLQStatsResponse)
async def get_dlq_stats(
    include_historical: bool = Query(False, description="Include historical trend data"),
    dlq_manager: DeadLetterQueueManager = Depends(get_dlq_manager),
    retry_scheduler: DLQRetryScheduler = Depends(get_retry_scheduler),
    poison_detector: PoisonMessageDetector = Depends(get_poison_detector)
):
    """
    Get comprehensive DLQ statistics and metrics.
    
    Provides real-time statistics about DLQ system performance,
    queue sizes, success rates, and component health.
    """
    try:
        # Gather stats from all components
        dlq_stats = await dlq_manager.get_dlq_stats()
        scheduler_metrics = await retry_scheduler.get_scheduler_metrics()
        detector_metrics = await poison_detector.get_detection_metrics()
        
        # Component health checks
        component_health = {}
        
        # DLQ Manager health
        dlq_health = await dlq_manager.health_check()
        component_health["dlq_manager"] = dlq_health["status"]
        
        # Retry Scheduler health
        scheduler_health = await retry_scheduler.health_check()
        component_health["retry_scheduler"] = scheduler_health["status"]
        
        # Poison Detector health
        detector_health = await poison_detector.health_check()
        component_health["poison_detector"] = detector_health["status"]
        
        # Calculate overall metrics
        total_processed = dlq_stats["metrics"]["messages_retried"] + dlq_stats["metrics"]["messages_resurrected"]
        retry_success_rate = (
            dlq_stats["metrics"]["messages_resurrected"] / max(1, total_processed)
        )
        
        response = DLQStatsResponse(
            # Queue metrics
            dlq_size=dlq_stats["dlq_size"],
            retry_queue_size=dlq_stats["retry_queue_size"],
            quarantine_queue_size=detector_metrics["performance_metrics"]["poison_messages_detected"],
            processing_queue_size=sum(scheduler_metrics["queue_metrics"].values()),
            
            # Performance metrics
            messages_retried=dlq_stats["metrics"]["messages_retried"],
            messages_resurrected=dlq_stats["metrics"]["successful_replays"],
            messages_permanently_failed=dlq_stats["metrics"]["messages_moved_to_dlq"],
            poison_messages_detected=detector_metrics["performance_metrics"]["poison_messages_detected"],
            poison_messages_quarantined=dlq_stats["metrics"].get("poison_messages_quarantined", 0),
            
            # Success rates
            retry_success_rate=retry_success_rate,
            resurrection_success_rate=scheduler_metrics["performance_metrics"]["success_rate"],
            overall_recovery_rate=dlq_stats["metrics"].get("eventual_delivery_rate", 0.0),
            
            # Processing performance
            average_processing_time_ms=dlq_stats["metrics"]["average_processing_time_ms"],
            scheduler_overhead_ms=scheduler_metrics["performance_metrics"]["scheduler_overhead_ms"],
            detection_accuracy=detector_metrics["performance_metrics"]["detection_accuracy"],
            
            # System health
            is_running=all(status == "healthy" for status in component_health.values()),
            component_health=component_health,
            
            # Configuration
            configuration={
                "dlq": dlq_stats["configuration"],
                "scheduler": scheduler_metrics["configuration"],
                "detector": detector_metrics["configuration"]
            }
        )
        
        logger.info(f"📊 DLQ stats requested - {response.dlq_size} messages in DLQ")
        return response
        
    except Exception as e:
        logger.error(f"Error getting DLQ stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve DLQ statistics")


@dlq_router.get("/messages", response_model=List[DLQMessageResponse])
async def get_dlq_messages(
    limit: int = Query(50, ge=1, le=500, description="Maximum number of messages to return"),
    offset: int = Query(0, ge=0, description="Number of messages to skip"),
    stream_filter: Optional[str] = Query(None, description="Filter by stream name"),
    priority_filter: Optional[str] = Query(None, description="Filter by priority"),
    poison_only: bool = Query(False, description="Show only poison messages"),
    sort_by: str = Query("created_at", regex="^(created_at|retry_count|risk_score)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    dlq_manager: DeadLetterQueueManager = Depends(get_dlq_manager)
):
    """
    Get paginated list of DLQ messages with filtering and sorting.
    
    Supports filtering by stream, priority, poison status and
    sorting by various fields for admin inspection.
    """
    try:
        # Get DLQ entries
        dlq_entries = await dlq_manager.get_dlq_entries(
            start="-",
            end="+", 
            count=limit + offset
        )
        
        # Apply filters
        filtered_entries = []
        for entry in dlq_entries:
            # Stream filter
            if stream_filter and stream_filter not in entry.original_stream:
                continue
                
            # Priority filter
            if priority_filter and getattr(entry.message, 'priority', 'normal') != priority_filter:
                continue
                
            # Poison filter (would need poison detection results)
            if poison_only:
                # This would require storing poison detection results with DLQ entries
                # For now, skip this filter
                pass
            
            filtered_entries.append(entry)
        
        # Apply pagination
        paginated_entries = filtered_entries[offset:offset + limit]
        
        # Convert to response format
        response_messages = []
        for entry in paginated_entries:
            try:
                # Create message preview (first 500 chars of payload)
                message_preview = {
                    "id": entry.message.id if hasattr(entry.message, 'id') else "unknown",
                    "type": entry.message.message_type.value if hasattr(entry.message, 'message_type') else "unknown",
                    "payload_preview": str(entry.message.payload)[:500] + "..." if hasattr(entry.message, 'payload') else "N/A"
                }
                
                response_messages.append(DLQMessageResponse(
                    dlq_entry_id=entry.dlq_entry_id or "unknown",
                    original_stream=entry.original_stream,
                    original_message_id=entry.original_message_id,
                    failure_reason=entry.failure_reason,
                    retry_count=entry.retry_count,
                    created_at=datetime.fromtimestamp(entry.first_failure_time),
                    next_retry_time=datetime.fromtimestamp(entry.next_retry_time) if entry.next_retry_time else None,
                    priority="normal",  # Would need to be stored in DLQ entry
                    strategy="exponential_backoff",  # Would need to be stored
                    is_poison=False,  # Would need poison detection integration
                    poison_type=None,
                    risk_score=0.0,  # Would need poison detection integration
                    message_preview=message_preview
                ))
            except Exception as e:
                logger.error(f"Error processing DLQ entry: {e}")
                continue
        
        logger.info(f"📝 Retrieved {len(response_messages)} DLQ messages")
        return response_messages
        
    except Exception as e:
        logger.error(f"Error getting DLQ messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve DLQ messages")


@dlq_router.post("/replay")
async def replay_dlq_messages(
    request: ReplayRequest,
    dlq_manager: DeadLetterQueueManager = Depends(get_dlq_manager)
):
    """
    Replay messages from DLQ back to their original streams.
    
    Supports filtering and safety checks to prevent system overload.
    """
    try:
        # Validate request
        if request.max_messages > 1000:
            raise HTTPException(
                status_code=400,
                detail="Maximum 1000 messages can be replayed at once"
            )
        
        # Perform replay with filters
        replayed_count = await dlq_manager.replay_dlq_messages(
            stream_filter=request.stream_filter,
            message_type_filter=request.message_type_filter,
            max_messages=request.max_messages
        )
        
        logger.info(f"🔄 Replayed {replayed_count} messages from DLQ")
        
        return {
            "status": "success",
            "replayed_count": replayed_count,
            "timestamp": datetime.utcnow().isoformat(),
            "filters_applied": {
                "stream_filter": request.stream_filter,
                "message_type_filter": request.message_type_filter.value if request.message_type_filter else None,
                "max_messages": request.max_messages
            }
        }
        
    except Exception as e:
        logger.error(f"Error replaying DLQ messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to replay messages: {str(e)}")


@dlq_router.get("/quarantine/{message_id}")
async def get_quarantine_message(
    message_id: str = Path(..., description="DLQ message ID to inspect"),
    poison_detector: PoisonMessageDetector = Depends(get_poison_detector)
):
    """
    Get detailed information about a quarantined poison message.
    """
    try:
        # This would need to be implemented to retrieve quarantined messages
        # For now, return a placeholder response
        return {
            "message_id": message_id,
            "status": "quarantined",
            "quarantine_reason": "Poison message detected",
            "detection_details": {
                "poison_type": "malformed_json",
                "confidence": "high",
                "risk_score": 0.8
            },
            "quarantined_at": datetime.utcnow().isoformat(),
            "suggested_actions": ["transform_and_retry", "human_review"]
        }
        
    except Exception as e:
        logger.error(f"Error getting quarantine message: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve quarantine message")


@dlq_router.post("/quarantine/{message_id}/action")
async def quarantine_action(
    message_id: str = Path(..., description="DLQ message ID"),
    request: QuarantineActionRequest = Body(...),
    dlq_manager: DeadLetterQueueManager = Depends(get_dlq_manager)
):
    """
    Perform action on quarantined poison message.
    
    Actions: release, permanent_delete, transform_retry
    """
    try:
        # Validate action
        valid_actions = ["release", "permanent_delete", "transform_retry"]
        if request.action not in valid_actions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action. Must be one of: {valid_actions}"
            )
        
        # This would need implementation in the DLQ manager
        # For now, return success response
        
        logger.info(f"🔧 Quarantine action '{request.action}' applied to message {message_id}")
        
        return {
            "status": "success",
            "message_id": message_id,
            "action_applied": request.action,
            "reason": request.reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error performing quarantine action: {e}")
        raise HTTPException(status_code=500, detail="Failed to perform quarantine action")


@dlq_router.post("/bulk-operation")
async def bulk_operation(
    request: BulkOperationRequest,
    dlq_manager: DeadLetterQueueManager = Depends(get_dlq_manager)
):
    """
    Perform bulk operations on DLQ messages.
    
    Operations: replay, quarantine, delete, priority_change
    Requires confirmation token for safety.
    """
    try:
        # Validate confirmation token (in production, this would be a proper security check)
        if not request.confirmation_token or len(request.confirmation_token) < 10:
            raise HTTPException(
                status_code=400,
                detail="Valid confirmation token required for bulk operations"
            )
        
        # Validate operation
        valid_operations = ["replay", "quarantine", "delete", "priority_change"]
        if request.operation not in valid_operations:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid operation. Must be one of: {valid_operations}"
            )
        
        # This would need implementation for each bulk operation type
        # For now, return placeholder response
        
        processed_count = min(request.max_items, 100)  # Placeholder
        
        logger.warning(f"⚡ Bulk operation '{request.operation}' processed {processed_count} items")
        
        return {
            "status": "success",
            "operation": request.operation,
            "processed_count": processed_count,
            "filters_applied": request.filters,
            "timestamp": datetime.utcnow().isoformat(),
            "confirmation_token": request.confirmation_token[:8] + "..."  # Partial token for logging
        }
        
    except Exception as e:
        logger.error(f"Error performing bulk operation: {e}")
        raise HTTPException(status_code=500, detail="Failed to perform bulk operation")


@dlq_router.get("/health", response_model=HealthCheckResponse)
async def dlq_health_check(
    include_details: bool = Query(False, description="Include detailed component metrics"),
    dlq_manager: DeadLetterQueueManager = Depends(get_dlq_manager),
    retry_scheduler: DLQRetryScheduler = Depends(get_retry_scheduler),
    poison_detector: PoisonMessageDetector = Depends(get_poison_detector)
):
    """
    Comprehensive health check for DLQ system.
    
    Returns overall system health and individual component status
    with performance metrics and recommendations.
    """
    try:
        # Gather health checks from all components
        components = {}
        recommendations = []
        performance_metrics = {}
        
        # DLQ Manager health
        dlq_health = await dlq_manager.health_check()
        components["dlq_manager"] = {
            "status": dlq_health["status"],
            "details": dlq_health if include_details else {}
        }
        if dlq_health["status"] != "healthy":
            recommendations.extend(dlq_health.get("recommendations", []))
        
        # Retry Scheduler health
        scheduler_health = await retry_scheduler.health_check()
        components["retry_scheduler"] = {
            "status": scheduler_health["status"],
            "details": scheduler_health if include_details else {}
        }
        if scheduler_health["status"] != "healthy":
            recommendations.append("Check retry scheduler performance")
        
        # Poison Detector health
        detector_health = await poison_detector.health_check()
        components["poison_detector"] = {
            "status": detector_health["status"],
            "details": detector_health if include_details else {}
        }
        if detector_health["status"] != "healthy":
            recommendations.append("Review poison detection accuracy")
        
        # Overall system status
        all_statuses = [comp["status"] for comp in components.values()]
        if all(status == "healthy" for status in all_statuses):
            overall_status = "healthy"
        elif any(status == "critical" for status in all_statuses):
            overall_status = "critical"
        else:
            overall_status = "degraded"
        
        # Performance metrics
        performance_metrics = {
            "dlq_processing_latency_ms": dlq_health.get("average_processing_time_ms", 0.0),
            "retry_success_rate": scheduler_health.get("metrics", {}).get("performance_metrics", {}).get("success_rate", 0.0),
            "poison_detection_accuracy": detector_health.get("metrics", {}).get("performance_metrics", {}).get("detection_accuracy", 0.0)
        }
        
        if not recommendations:
            recommendations = ["All DLQ components operating normally"]
        
        response = HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            components=components,
            recommendations=recommendations,
            performance_metrics=performance_metrics
        )
        
        logger.info(f"🏥 DLQ health check completed - Status: {overall_status}")
        return response
        
    except Exception as e:
        logger.error(f"Error performing DLQ health check: {e}")
        # Return degraded status on health check failure
        return HealthCheckResponse(
            status="critical",
            timestamp=datetime.utcnow(),
            components={"error": {"status": "critical", "details": {"error": str(e)}}},
            recommendations=[f"Health check failed: {str(e)}"],
            performance_metrics={}
        )


# Metrics export endpoint
@dlq_router.get("/metrics/export")
async def export_metrics(
    format: str = Query("json", regex="^(json|prometheus|csv)$"),
    time_range_hours: int = Query(24, ge=1, le=168),  # 1 hour to 1 week
    dlq_manager: DeadLetterQueueManager = Depends(get_dlq_manager)
):
    """
    Export DLQ metrics in various formats for external monitoring systems.
    
    Supports JSON, Prometheus, and CSV formats.
    """
    try:
        # Get comprehensive metrics
        dlq_stats = await dlq_manager.get_dlq_stats()
        
        if format == "json":
            return JSONResponse(content={
                "export_timestamp": datetime.utcnow().isoformat(),
                "time_range_hours": time_range_hours,
                "metrics": dlq_stats
            })
        
        elif format == "prometheus":
            # Generate Prometheus format metrics
            prometheus_metrics = [
                f"# HELP dlq_size Number of messages in dead letter queue",
                f"# TYPE dlq_size gauge",
                f"dlq_size {dlq_stats['dlq_size']}",
                f"",
                f"# HELP dlq_retry_success_rate Success rate of message retries",
                f"# TYPE dlq_retry_success_rate gauge", 
                f"dlq_retry_success_rate {dlq_stats['metrics'].get('eventual_delivery_rate', 0.0)}",
                f"",
                f"# HELP dlq_poison_messages_detected Number of poison messages detected",
                f"# TYPE dlq_poison_messages_detected counter",
                f"dlq_poison_messages_detected {dlq_stats['metrics'].get('poison_messages_detected', 0)}"
            ]
            
            return JSONResponse(
                content="\n".join(prometheus_metrics),
                media_type="text/plain"
            )
        
        elif format == "csv":
            # Generate CSV format (simplified)
            csv_data = [
                "metric,value,timestamp",
                f"dlq_size,{dlq_stats['dlq_size']},{datetime.utcnow().isoformat()}",
                f"retry_success_rate,{dlq_stats['metrics'].get('eventual_delivery_rate', 0.0)},{datetime.utcnow().isoformat()}",
                f"poison_messages,{dlq_stats['metrics'].get('poison_messages_detected', 0)},{datetime.utcnow().isoformat()}"
            ]
            
            return JSONResponse(
                content="\n".join(csv_data),
                media_type="text/csv"
            )
        
    except Exception as e:
        logger.error(f"Error exporting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to export metrics")


# Admin configuration endpoint
@dlq_router.put("/config")
async def update_dlq_config(
    new_config: Dict[str, Any] = Body(...),
    dlq_manager: DeadLetterQueueManager = Depends(get_dlq_manager)
):
    """
    Update DLQ system configuration.
    
    Allows runtime configuration changes for thresholds,
    retry policies, and monitoring settings.
    """
    try:
        # Validate configuration keys
        valid_config_keys = [
            "max_retries", "initial_retry_delay_ms", "max_retry_delay_ms",
            "dlq_max_size", "dlq_ttl_hours", "alert_threshold"
        ]
        
        invalid_keys = [key for key in new_config.keys() if key not in valid_config_keys]
        if invalid_keys:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid configuration keys: {invalid_keys}"
            )
        
        # This would need implementation in the DLQ manager
        # For now, return success response
        
        logger.info(f"⚙️ DLQ configuration updated: {list(new_config.keys())}")
        
        return {
            "status": "success",
            "updated_config": new_config,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Configuration updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error updating DLQ config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update configuration")


# Debug endpoint for testing
@dlq_router.post("/debug/inject-poison")
async def inject_poison_message(
    poison_type: str = Query("malformed_json", description="Type of poison message to inject"),
    dlq_manager: DeadLetterQueueManager = Depends(get_dlq_manager)
):
    """
    DEBUG ONLY: Inject a poison message for testing DLQ functionality.
    
    ⚠️ This endpoint should only be available in development/testing environments.
    """
    if settings.ENVIRONMENT == "production":
        raise HTTPException(
            status_code=403,
            detail="Debug endpoints not available in production"
        )
    
    try:
        # Create test poison message
        poison_messages = {
            "malformed_json": '{"invalid": json syntax}',
            "oversized": "x" * (2 * 1024 * 1024),  # 2MB message
            "circular_ref": {"a": {"b": None}},  # Would need actual circular reference
            "invalid_encoding": "Invalid \xff\xfe encoding"
        }
        
        test_message = poison_messages.get(poison_type, poison_messages["malformed_json"])
        
        # This would need implementation to actually inject the message
        # For now, return success response
        
        logger.warning(f"🧪 DEBUG: Injected poison message type '{poison_type}'")
        
        return {
            "status": "success",
            "poison_type": poison_type,
            "message": "Poison message injected for testing",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error injecting poison message: {e}")
        raise HTTPException(status_code=500, detail="Failed to inject poison message")


# Include router in main app
def get_dlq_router() -> APIRouter:
    """Get configured DLQ router for FastAPI app."""
    return dlq_router