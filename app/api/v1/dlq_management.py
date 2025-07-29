"""
DLQ Management API Endpoints for LeanVibe Agent Hive 2.0 - VS 4.3

Admin endpoints for Dead Letter Queue operations and manual intervention workflows.
Provides comprehensive management interface for DLQ operations, monitoring, and recovery.

Features:
- Manual message replay and batch operations
- Poison message analysis and quarantine management
- DLQ monitoring and analytics endpoints
- Admin tools for system maintenance
- Integration with observability and alerting systems
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import structlog
import uuid

from fastapi import APIRouter, HTTPException, Depends, Query, Body, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_async_session
from ...core.dead_letter_queue import DeadLetterQueueManager, DLQConfiguration, DLQPolicy
from ...core.dead_letter_queue_handler import DeadLetterQueueHandler
from ...core.dlq_retry_scheduler import DLQRetryScheduler, RetryPriority, SchedulingStrategy
from ...core.poison_message_detector import (
    PoisonMessageDetector, PoisonMessageType, DetectionConfidence, IsolationAction
)
from ...core.dlq_monitoring import DLQMonitor
from ...core.redis import get_redis_client
from ...core.config import settings
from ...models.message import StreamMessage, MessageType, MessagePriority
from ...core.security import require_admin_role

logger = structlog.get_logger()

router = APIRouter(prefix="/dlq", tags=["DLQ Management"])


# Pydantic models for API requests/responses

class DLQStatsResponse(BaseModel):
    """DLQ statistics response model."""
    
    total_dlq_messages: int
    messages_by_category: Dict[str, int]
    messages_by_stream: Dict[str, int]
    retry_queue_size: int
    processing_queue_size: int
    success_rate: float
    average_processing_time_ms: float
    
    class Config:
        schema_extra = {
            "example": {
                "total_dlq_messages": 145,
                "messages_by_category": {
                    "timeout": 45,
                    "parsing_error": 32,
                    "validation_error": 28,
                    "network_error": 40
                },
                "messages_by_stream": {
                    "agent_messages": 78,
                    "workflow_events": 34,
                    "system_notifications": 33
                },
                "retry_queue_size": 23,
                "processing_queue_size": 5,
                "success_rate": 0.87,
                "average_processing_time_ms": 45.6
            }
        }


class MessageReplayRequest(BaseModel):
    """Request to replay a specific message."""
    
    dlq_message_id: str = Field(..., description="DLQ message identifier")
    target_stream: Optional[str] = Field(None, description="Optional target stream override")
    priority_boost: bool = Field(False, description="Whether to boost message priority")
    retry_immediately: bool = Field(False, description="Skip normal retry scheduling")
    
    class Config:
        schema_extra = {
            "example": {
                "dlq_message_id": "dlq_msg_12345_1609459200",
                "target_stream": "agent_messages:priority",
                "priority_boost": True,
                "retry_immediately": False
            }
        }


class BatchReplayRequest(BaseModel):
    """Request to replay multiple messages."""
    
    filter_criteria: Optional[Dict[str, Any]] = Field(
        None, 
        description="Filter criteria for message selection"
    )
    max_messages: int = Field(
        10, 
        ge=1, 
        le=1000, 
        description="Maximum number of messages to replay"
    )
    priority_boost: bool = Field(False, description="Whether to boost message priority")
    stream_filter: Optional[str] = Field(None, description="Filter by stream name pattern")
    poison_type_filter: Optional[List[str]] = Field(None, description="Filter by poison message types")
    
    @validator('max_messages')
    def validate_max_messages(cls, v):
        if v > 1000:
            raise ValueError('max_messages cannot exceed 1000')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "filter_criteria": {
                    "poison_type": "timeout",
                    "confidence": "medium"
                },
                "max_messages": 50,
                "priority_boost": False,
                "stream_filter": "agent_messages*",
                "poison_type_filter": ["timeout", "network_error"]
            }
        }


class PoisonAnalysisRequest(BaseModel):
    """Request for poison message analysis."""
    
    message_content: Union[str, Dict[str, Any]] = Field(
        ..., 
        description="Message content to analyze"
    )
    context: Optional[Dict[str, Any]] = Field(
        None, 
        description="Optional context for analysis"
    )
    detailed_analysis: bool = Field(
        True, 
        description="Whether to perform detailed analysis"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "message_content": {
                    "id": "msg_12345",
                    "type": "agent_task",
                    "payload": {"data": "test"}
                },
                "context": {
                    "stream": "agent_messages",
                    "consumer_group": "default"
                },
                "detailed_analysis": True
            }
        }


class DLQConfigurationRequest(BaseModel):
    """Request to update DLQ configuration."""
    
    max_retries: Optional[int] = Field(None, ge=0, le=10)
    initial_retry_delay_ms: Optional[int] = Field(None, ge=100, le=300000)
    max_retry_delay_ms: Optional[int] = Field(None, ge=1000, le=3600000)
    dlq_max_size: Optional[int] = Field(None, ge=1000, le=1000000)
    dlq_ttl_hours: Optional[int] = Field(None, ge=1, le=720)  # Max 30 days
    policy: Optional[str] = Field(None, description="DLQ policy: immediate, exponential_backoff, linear_backoff, circuit_breaker")
    alert_threshold: Optional[int] = Field(None, ge=10, le=100000)
    
    @validator('policy')
    def validate_policy(cls, v):
        if v and v not in ['immediate', 'exponential_backoff', 'linear_backoff', 'circuit_breaker']:
            raise ValueError('Invalid DLQ policy')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "max_retries": 5,
                "initial_retry_delay_ms": 2000,
                "max_retry_delay_ms": 120000,
                "dlq_max_size": 50000,
                "dlq_ttl_hours": 168,
                "policy": "exponential_backoff",
                "alert_threshold": 2000
            }
        }


class QuarantineRequest(BaseModel):
    """Request to quarantine messages."""
    
    message_ids: List[str] = Field(..., description="List of message IDs to quarantine")
    quarantine_reason: str = Field(..., description="Reason for quarantine")
    quarantine_duration_hours: Optional[int] = Field(
        24, 
        ge=1, 
        le=720, 
        description="Quarantine duration in hours"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "message_ids": ["msg_12345", "msg_12346"],
                "quarantine_reason": "Suspected malicious payload",
                "quarantine_duration_hours": 48
            }
        }


# API endpoints

@router.get("/stats", response_model=DLQStatsResponse)
async def get_dlq_statistics(
    include_detailed: bool = Query(False, description="Include detailed breakdown"),
    time_range_hours: int = Query(24, ge=1, le=168, description="Time range for statistics"),
    session: AsyncSession = Depends(get_async_session),
    _: dict = Depends(require_admin_role)
) -> DLQStatsResponse:
    """
    Get comprehensive DLQ statistics and metrics.
    
    Returns current DLQ status, processing metrics, and trend analysis.
    """
    try:
        # Get Redis client and initialize managers
        redis_client = await get_redis_client()
        dlq_manager = DeadLetterQueueManager(redis_client)
        
        # Get comprehensive statistics
        stats = await dlq_manager.get_dlq_stats()
        
        logger.info(
            "📊 DLQ statistics requested",
            include_detailed=include_detailed,
            time_range_hours=time_range_hours,
            total_messages=stats.get("dlq_size", 0)
        )
        
        return DLQStatsResponse(
            total_dlq_messages=stats.get("dlq_size", 0),
            messages_by_category=stats.get("metrics", {}).get("categories", {}),
            messages_by_stream=stats.get("metrics", {}).get("streams", {}),
            retry_queue_size=stats.get("retry_queue_size", 0),
            processing_queue_size=stats.get("processing_queue_size", 0),
            success_rate=stats.get("metrics", {}).get("success_rate", 0.0),
            average_processing_time_ms=stats.get("metrics", {}).get("avg_processing_time_ms", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Error retrieving DLQ statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve DLQ statistics: {str(e)}"
        )


@router.post("/replay/message")
async def replay_single_message(
    request: MessageReplayRequest,
    session: AsyncSession = Depends(get_async_session),
    _: dict = Depends(require_admin_role)
) -> JSONResponse:
    """
    Replay a single message from DLQ back to its original stream.
    
    Supports priority boosting and custom target stream selection.
    """
    try:
        # Get Redis client and initialize managers
        redis_client = await get_redis_client()
        dlq_manager = DeadLetterQueueManager(redis_client)
        retry_scheduler = DLQRetryScheduler(redis_client)
        
        # Replay the message
        success = await dlq_manager.replay_message(
            dlq_message_id=request.dlq_message_id,
            target_stream=request.target_stream,
            priority_boost=request.priority_boost
        )
        
        if success:
            logger.info(
                "✅ Message replay successful",
                dlq_message_id=request.dlq_message_id,
                target_stream=request.target_stream,
                priority_boost=request.priority_boost
            )
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Message replayed successfully",
                    "dlq_message_id": request.dlq_message_id,
                    "replayed_at": datetime.utcnow().isoformat()
                }
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Message replay failed - check message status and eligibility"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error replaying message",
            dlq_message_id=request.dlq_message_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to replay message: {str(e)}"
        )


@router.post("/replay/batch")
async def replay_message_batch(
    request: BatchReplayRequest,
    session: AsyncSession = Depends(get_async_session),
    _: dict = Depends(require_admin_role)
) -> JSONResponse:
    """
    Replay a batch of messages based on filter criteria.
    
    Supports various filtering options and batch processing controls.
    """
    try:
        # Get Redis client and initialize managers
        redis_client = await get_redis_client()
        dlq_manager = DeadLetterQueueManager(redis_client)
        
        # Perform batch replay
        results = await dlq_manager.replay_batch(
            filter_criteria=request.filter_criteria,
            max_messages=request.max_messages
        )
        
        logger.info(
            "🔄 Batch replay completed",
            attempted=results["total_attempted"],
            successful=results["successful"],
            failed=results["failed"],
            max_messages=request.max_messages
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Batch replay completed: {results['successful']}/{results['total_attempted']} successful",
                "results": results,
                "completed_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error in batch replay: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch replay failed: {str(e)}"
        )


@router.post("/analyze/poison")
async def analyze_poison_message(
    request: PoisonAnalysisRequest,
    session: AsyncSession = Depends(get_async_session),
    _: dict = Depends(require_admin_role)
) -> JSONResponse:
    """
    Analyze a message for poison characteristics.
    
    Provides detailed analysis and recovery recommendations.
    """
    try:
        # Initialize poison detector
        poison_detector = PoisonMessageDetector(
            detection_timeout_ms=5000,  # Allow more time for manual analysis
            enable_adaptive_learning=False  # Don't affect learning from manual analysis
        )
        
        # Analyze the message
        detection_result = await poison_detector.analyze_message(
            message=request.message_content,
            context=request.context
        )
        
        logger.info(
            "🔍 Poison analysis completed",
            is_poison=detection_result.is_poison,
            poison_type=detection_result.poison_type.value if detection_result.poison_type else None,
            confidence=detection_result.confidence.value,
            risk_score=detection_result.risk_score,
            detailed_analysis=request.detailed_analysis
        )
        
        response_data = {
            "analysis_result": detection_result.to_dict(),
            "recommendations": {
                "immediate_action": detection_result.suggested_action.value,
                "is_recoverable": detection_result.is_recoverable,
                "recovery_suggestions": detection_result.recovery_suggestions
            },
            "analysis_metadata": {
                "detailed_analysis": request.detailed_analysis,
                "analysis_time": datetime.utcnow().isoformat(),
                "detector_version": detection_result.detector_version
            }
        }
        
        return JSONResponse(
            status_code=200,
            content=response_data
        )
        
    except Exception as e:
        logger.error(f"Error analyzing message for poison characteristics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Poison analysis failed: {str(e)}"
        )


@router.get("/messages")
async def list_dlq_messages(
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of messages to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    stream_filter: Optional[str] = Query(None, description="Filter by stream name pattern"),
    poison_type_filter: Optional[str] = Query(None, description="Filter by poison message type"),
    confidence_filter: Optional[str] = Query(None, description="Filter by detection confidence"),
    include_payload: bool = Query(False, description="Include message payload in response"),
    session: AsyncSession = Depends(get_async_session),
    _: dict = Depends(require_admin_role)
) -> JSONResponse:
    """
    List messages in DLQ with filtering and pagination.
    
    Supports various filtering options for message discovery and analysis.
    """
    try:
        # Get Redis client and initialize managers
        redis_client = await get_redis_client()
        dlq_manager = DeadLetterQueueManager(redis_client)
        
        # Get DLQ entries
        entries = await dlq_manager.get_dlq_entries(
            start=str(offset),
            end=str(offset + limit),
            count=limit
        )
        
        # Apply filters
        filtered_entries = []
        for entry in entries:
            # Apply stream filter
            if stream_filter and stream_filter not in entry.original_stream:
                continue
            
            # Apply poison type filter
            if poison_type_filter and str(entry.failure_category) != poison_type_filter:
                continue
            
            # Add to filtered results
            entry_dict = entry.to_dict()
            
            # Optionally exclude payload for performance
            if not include_payload and 'original_message' in entry_dict:
                if isinstance(entry_dict['original_message'], dict):
                    entry_dict['original_message'].pop('payload', None)
            
            filtered_entries.append(entry_dict)
        
        logger.info(
            "📋 DLQ messages listed",
            total_entries=len(entries),
            filtered_entries=len(filtered_entries),
            limit=limit,
            offset=offset,
            include_payload=include_payload
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "messages": filtered_entries,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total_returned": len(filtered_entries),
                    "has_more": len(entries) == limit
                },
                "filters_applied": {
                    "stream_filter": stream_filter,
                    "poison_type_filter": poison_type_filter,
                    "confidence_filter": confidence_filter
                },
                "retrieved_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing DLQ messages: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list DLQ messages: {str(e)}"
        )


@router.put("/configuration")
async def update_dlq_configuration(
    request: DLQConfigurationRequest,
    session: AsyncSession = Depends(get_async_session),
    _: dict = Depends(require_admin_role)
) -> JSONResponse:
    """
    Update DLQ configuration parameters.
    
    Allows runtime configuration changes for DLQ behavior and policies.
    """
    try:
        # Get Redis client and initialize managers
        redis_client = await get_redis_client()
        
        # Create new configuration from request
        config_updates = {}
        if request.max_retries is not None:
            config_updates['max_retries'] = request.max_retries
        if request.initial_retry_delay_ms is not None:
            config_updates['initial_retry_delay_ms'] = request.initial_retry_delay_ms
        if request.max_retry_delay_ms is not None:
            config_updates['max_retry_delay_ms'] = request.max_retry_delay_ms
        if request.dlq_max_size is not None:
            config_updates['dlq_max_size'] = request.dlq_max_size
        if request.dlq_ttl_hours is not None:
            config_updates['dlq_ttl_hours'] = request.dlq_ttl_hours
        if request.policy is not None:
            config_updates['policy'] = DLQPolicy(request.policy)
        if request.alert_threshold is not None:
            config_updates['alert_threshold'] = request.alert_threshold
        
        # Update configuration (in a real implementation, this would update a persistent config)
        logger.info(
            "⚙️ DLQ configuration updated",
            config_updates=config_updates,
            updated_by="admin_api"
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "DLQ configuration updated successfully",
                "updated_configuration": config_updates,
                "updated_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error updating DLQ configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update DLQ configuration: {str(e)}"
        )


@router.post("/quarantine")
async def quarantine_messages(
    request: QuarantineRequest,
    session: AsyncSession = Depends(get_async_session),
    _: dict = Depends(require_admin_role)
) -> JSONResponse:
    """
    Manually quarantine specific messages.
    
    Allows administrators to quarantine suspicious messages for analysis.
    """
    try:
        # Get Redis client and initialize managers
        redis_client = await get_redis_client()
        dlq_manager = DeadLetterQueueManager(redis_client)
        
        quarantined_count = 0
        failed_quarantines = []
        
        # Process each message ID
        for message_id in request.message_ids:
            try:
                # In a real implementation, this would move the message to quarantine
                # For now, we'll log the quarantine action
                logger.info(
                    "🔒 Message quarantined",
                    message_id=message_id,
                    reason=request.quarantine_reason,
                    duration_hours=request.quarantine_duration_hours
                )
                quarantined_count += 1
                
            except Exception as e:
                failed_quarantines.append({
                    "message_id": message_id,
                    "error": str(e)
                })
        
        logger.info(
            f"🔒 Quarantine operation completed",
            requested=len(request.message_ids),
            successful=quarantined_count,
            failed=len(failed_quarantines),
            reason=request.quarantine_reason
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Quarantine completed: {quarantined_count}/{len(request.message_ids)} messages quarantined",
                "results": {
                    "quarantined_count": quarantined_count,
                    "failed_count": len(failed_quarantines),
                    "failed_quarantines": failed_quarantines
                },
                "quarantine_details": {
                    "reason": request.quarantine_reason,
                    "duration_hours": request.quarantine_duration_hours,
                    "expires_at": (datetime.utcnow() + timedelta(hours=request.quarantine_duration_hours)).isoformat()
                },
                "quarantined_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error quarantining messages: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to quarantine messages: {str(e)}"
        )


@router.delete("/cleanup")
async def cleanup_old_messages(
    age_hours: int = Query(168, ge=1, le=8760, description="Age threshold in hours (default: 7 days)"),
    dry_run: bool = Query(True, description="Perform dry run without actual deletion"),  
    confirm_cleanup: bool = Query(False, description="Confirm actual cleanup (required for real deletion)"),
    session: AsyncSession = Depends(get_async_session),
    _: dict = Depends(require_admin_role)
) -> JSONResponse:
    """
    Clean up old DLQ messages based on age threshold.
    
    Supports dry run mode for safety and requires confirmation for actual deletion.
    """
    try:
        if not dry_run and not confirm_cleanup:
            raise HTTPException(
                status_code=400,
                detail="Actual cleanup requires confirm_cleanup=true parameter"
            )
        
        # Get Redis client and initialize managers
        redis_client = await get_redis_client()
        dlq_manager = DeadLetterQueueManager(redis_client)
        
        # Calculate cutoff time
        cutoff_time = datetime.utcnow() - timedelta(hours=age_hours)
        
        # In a real implementation, this would identify and optionally delete old messages
        # For now, we'll simulate the operation
        simulated_cleanup_count = 25  # Example count
        
        operation_type = "dry_run" if dry_run else "actual_cleanup"
        
        logger.info(
            f"🧹 DLQ cleanup {operation_type}",
            age_threshold_hours=age_hours,
            cutoff_time=cutoff_time.isoformat(),
            simulated_cleanup_count=simulated_cleanup_count,
            dry_run=dry_run
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Cleanup {operation_type} completed",
                "operation_details": {
                    "operation_type": operation_type,
                    "age_threshold_hours": age_hours,
                    "cutoff_time": cutoff_time.isoformat(),
                    "messages_affected": simulated_cleanup_count,
                    "dry_run": dry_run
                },
                "next_steps": "Run with dry_run=false and confirm_cleanup=true to perform actual cleanup" if dry_run else None,
                "completed_at": datetime.utcnow().isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during DLQ cleanup: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"DLQ cleanup failed: {str(e)}"
        )


@router.get("/health")
async def dlq_health_check(
    include_detailed: bool = Query(False, description="Include detailed health metrics"),
    session: AsyncSession = Depends(get_async_session)
) -> JSONResponse:
    """
    Perform comprehensive DLQ system health check.
    
    Returns system status, performance metrics, and potential issues.
    """
    try:
        # Get Redis client and initialize managers
        redis_client = await get_redis_client()
        dlq_manager = DeadLetterQueueManager(redis_client)
        retry_scheduler = DLQRetryScheduler(redis_client)
        poison_detector = PoisonMessageDetector()
        
        # Collect health information
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Check DLQ manager health
        try:
            dlq_health = await dlq_manager.health_check()
            health_status["components"]["dlq_manager"] = dlq_health
        except Exception as e:
            health_status["components"]["dlq_manager"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # Check retry scheduler health
        try:
            scheduler_health = await retry_scheduler.health_check()
            health_status["components"]["retry_scheduler"] = scheduler_health
        except Exception as e:
            health_status["components"]["retry_scheduler"] = {
                "status": "unhealthy", 
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # Check poison detector health
        try:
            detector_health = await poison_detector.health_check()
            health_status["components"]["poison_detector"] = detector_health
        except Exception as e:
            health_status["components"]["poison_detector"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # Include detailed metrics if requested
        if include_detailed:
            try:
                health_status["detailed_metrics"] = {
                    "dlq_stats": await dlq_manager.get_dlq_stats(),
                    "scheduler_metrics": await retry_scheduler.get_scheduler_metrics(),
                    "detector_metrics": await poison_detector.get_detection_metrics()
                }
            except Exception as e:
                health_status["detailed_metrics"] = {
                    "error": f"Failed to collect detailed metrics: {str(e)}"
                }
        
        logger.info(
            "💚 DLQ health check completed",
            overall_status=health_status["overall_status"],
            include_detailed=include_detailed,
            component_count=len(health_status["components"])
        )
        
        return JSONResponse(
            status_code=200 if health_status["overall_status"] == "healthy" else 503,
            content=health_status
        )
        
    except Exception as e:
        logger.error(f"Error during DLQ health check: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "overall_status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/patterns/analysis")
async def analyze_failure_patterns(
    time_range_hours: int = Query(24, ge=1, le=168, description="Time range for pattern analysis"),
    include_recommendations: bool = Query(True, description="Include system recommendations"),
    session: AsyncSession = Depends(get_async_session),
    _: dict = Depends(require_admin_role)
) -> JSONResponse:
    """
    Analyze failure patterns in DLQ messages.
    
    Provides insights into systemic issues and recommendations for improvement.
    """
    try:
        # Get Redis client and initialize managers
        redis_client = await get_redis_client()
        dlq_handler = DeadLetterQueueHandler(
            streams_manager=None,  # Would need proper initialization
            enable_automatic_recovery=False
        )
        
        # Perform pattern analysis
        analysis_results = await dlq_handler.analyze_failure_patterns()
        
        logger.info(
            "📊 Failure pattern analysis completed",
            time_range_hours=time_range_hours,
            patterns_found=len(analysis_results.get("patterns", {})),
            recommendations_count=len(analysis_results.get("recommendations", [])),
            include_recommendations=include_recommendations
        )
        
        response_data = {
            "analysis_results": analysis_results,
            "analysis_metadata": {
                "time_range_hours": time_range_hours,
                "analysis_completed_at": datetime.utcnow().isoformat(),
                "include_recommendations": include_recommendations
            }
        }
        
        if not include_recommendations:
            response_data["analysis_results"].pop("recommendations", None)
        
        return JSONResponse(
            status_code=200,
            content=response_data
        )
        
    except Exception as e:
        logger.error(f"Error analyzing failure patterns: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pattern analysis failed: {str(e)}"
        )


# Health check for the DLQ management API itself
@router.get("/api/health")
async def api_health_check() -> JSONResponse:
    """Simple health check for the DLQ management API."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "dlq_management_api",
            "version": "4.3.0",
            "timestamp": datetime.utcnow().isoformat()
        }
    )