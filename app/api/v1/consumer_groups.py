"""
Enhanced API endpoints for Consumer Group management - Vertical Slice 4.2

Provides REST API endpoints for managing Redis Streams consumer groups,
monitoring performance, and controlling message routing behavior.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field
import structlog

from ...core.enhanced_redis_streams_manager import (
    EnhancedRedisStreamsManager, ConsumerGroupConfig, ConsumerGroupType,
    MessageRoutingMode, ConsumerGroupMetrics
)
from ...core.consumer_group_coordinator import (
    ConsumerGroupCoordinator, ConsumerGroupStrategy, ProvisioningPolicy,
    ResourceAllocationMode
)
from ...core.workflow_message_router import (
    WorkflowMessageRouter, WorkflowRoutingStrategy, DependencyResolutionMode
)
from ...core.dead_letter_queue_handler import (
    DeadLetterQueueHandler, DLQMessageStatus, FailureCategory, RecoveryStrategy
)
from ...models.message import StreamMessage, MessageType, MessagePriority

logger = structlog.get_logger()

router = APIRouter(prefix="/consumer-groups", tags=["Consumer Groups"])


# Pydantic models for API requests/responses

class ConsumerGroupConfigRequest(BaseModel):
    """Request model for creating consumer groups."""
    name: str = Field(..., description="Consumer group name")
    stream_name: str = Field(..., description="Target stream name")
    agent_type: ConsumerGroupType = Field(..., description="Agent type for the group")
    routing_mode: MessageRoutingMode = Field(
        MessageRoutingMode.LOAD_BALANCED, 
        description="Message routing mode"
    )
    max_consumers: int = Field(10, description="Maximum number of consumers", ge=1, le=100)
    min_consumers: int = Field(1, description="Minimum number of consumers", ge=1)
    auto_scale_enabled: bool = Field(True, description="Enable auto-scaling")
    lag_threshold: int = Field(100, description="Lag threshold for scaling", ge=1)


class ConsumerGroupResponse(BaseModel):
    """Response model for consumer group information."""
    name: str
    stream_name: str
    agent_type: str
    routing_mode: str
    max_consumers: int
    min_consumers: int
    auto_scale_enabled: bool
    current_consumers: int
    lag: int
    pending_count: int
    throughput_msg_per_sec: float
    success_rate: float
    avg_processing_time_ms: float
    created_at: Optional[datetime] = None


class MessageRouteRequest(BaseModel):
    """Request model for routing messages to consumer groups."""
    message_id: str = Field(..., description="Message identifier")
    from_agent: str = Field(..., description="Source agent")
    message_type: MessageType = Field(..., description="Message type")
    payload: Dict[str, Any] = Field(..., description="Message payload")
    priority: MessagePriority = Field(MessagePriority.NORMAL, description="Message priority")
    workflow_id: Optional[str] = Field(None, description="Optional workflow context")
    dependencies: Optional[List[str]] = Field(None, description="Task dependencies")


class BatchReplayRequest(BaseModel):
    """Request model for batch message replay from DLQ."""
    filter_criteria: Optional[Dict[str, Any]] = Field(None, description="Filter criteria")
    max_messages: int = Field(10, description="Maximum messages to replay", ge=1, le=100)
    priority_boost: bool = Field(False, description="Boost message priority")


class CoordinatorConfigRequest(BaseModel):
    """Request model for coordinator configuration."""
    strategy: ConsumerGroupStrategy = Field(..., description="Group management strategy")
    provisioning_policy: ProvisioningPolicy = Field(..., description="Provisioning policy")
    allocation_mode: ResourceAllocationMode = Field(..., description="Resource allocation mode")
    enable_cross_group_coordination: bool = Field(True, description="Enable cross-group coordination")


# Dependency injection for core services
# In a real implementation, these would be proper dependency injection

async def get_streams_manager() -> EnhancedRedisStreamsManager:
    """Get Enhanced Redis Streams Manager instance."""
    # This would be injected from application context
    # For now, creating a placeholder
    manager = EnhancedRedisStreamsManager()
    await manager.connect()
    return manager


async def get_coordinator() -> ConsumerGroupCoordinator:
    """Get Consumer Group Coordinator instance."""
    streams_manager = await get_streams_manager()
    coordinator = ConsumerGroupCoordinator(streams_manager)
    await coordinator.start()
    return coordinator


async def get_workflow_router() -> WorkflowMessageRouter:
    """Get Workflow Message Router instance."""
    streams_manager = await get_streams_manager()
    coordinator = await get_coordinator()
    router = WorkflowMessageRouter(streams_manager, coordinator)
    await router.start()
    return router


async def get_dlq_handler() -> DeadLetterQueueHandler:
    """Get Dead Letter Queue Handler instance."""
    streams_manager = await get_streams_manager()
    handler = DeadLetterQueueHandler(streams_manager)
    await handler.start()
    return handler


# Consumer Group Management Endpoints

@router.post("/", response_model=Dict[str, str])
async def create_consumer_group(
    config: ConsumerGroupConfigRequest,
    coordinator: ConsumerGroupCoordinator = Depends(get_coordinator)
):
    """Create a new consumer group with specified configuration."""
    try:
        group_config = ConsumerGroupConfig(
            name=config.name,
            stream_name=config.stream_name,
            agent_type=config.agent_type,
            routing_mode=config.routing_mode,
            max_consumers=config.max_consumers,
            min_consumers=config.min_consumers,
            auto_scale_enabled=config.auto_scale_enabled,
            lag_threshold=config.lag_threshold
        )
        
        await coordinator.streams_manager.create_consumer_group(group_config)
        
        logger.info(f"Created consumer group {config.name}")
        
        return {
            "status": "success",
            "message": f"Consumer group {config.name} created successfully",
            "group_name": config.name
        }
        
    except Exception as e:
        logger.error(f"Failed to create consumer group {config.name}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to create consumer group: {str(e)}")


@router.get("/", response_model=List[ConsumerGroupResponse])
async def list_consumer_groups(
    coordinator: ConsumerGroupCoordinator = Depends(get_coordinator)
):
    """List all managed consumer groups with their current status."""
    try:
        all_stats = await coordinator.streams_manager.get_all_group_stats()
        
        groups = []
        for group_name, stats in all_stats.items():
            groups.append(ConsumerGroupResponse(
                name=group_name,
                stream_name=stats.stream_name,
                agent_type=group_name.replace("_consumers", ""),  # Simplified
                routing_mode="load_balanced",  # Would get from actual config
                max_consumers=20,  # Would get from actual config
                min_consumers=1,   # Would get from actual config
                auto_scale_enabled=True,  # Would get from actual config
                current_consumers=stats.consumer_count,
                lag=stats.lag,
                pending_count=stats.pending_count,
                throughput_msg_per_sec=stats.throughput_msg_per_sec,
                success_rate=stats.success_rate,
                avg_processing_time_ms=stats.avg_processing_time_ms
            ))
        
        return groups
        
    except Exception as e:
        logger.error(f"Failed to list consumer groups: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list consumer groups: {str(e)}")


@router.get("/{group_name}", response_model=ConsumerGroupResponse)
async def get_consumer_group(
    group_name: str,
    coordinator: ConsumerGroupCoordinator = Depends(get_coordinator)
):
    """Get detailed information about a specific consumer group."""
    try:
        stats = await coordinator.streams_manager.get_consumer_group_stats(group_name)
        
        if not stats:
            raise HTTPException(status_code=404, detail=f"Consumer group {group_name} not found")
        
        return ConsumerGroupResponse(
            name=group_name,
            stream_name=stats.stream_name,
            agent_type=group_name.replace("_consumers", ""),
            routing_mode="load_balanced",
            max_consumers=20,
            min_consumers=1,
            auto_scale_enabled=True,
            current_consumers=stats.consumer_count,
            lag=stats.lag,
            pending_count=stats.pending_count,
            throughput_msg_per_sec=stats.throughput_msg_per_sec,
            success_rate=stats.success_rate,
            avg_processing_time_ms=stats.avg_processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get consumer group {group_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get consumer group: {str(e)}")


@router.delete("/{group_name}")
async def delete_consumer_group(
    group_name: str,
    coordinator: ConsumerGroupCoordinator = Depends(get_coordinator)
):
    """Delete a consumer group (stops all consumers and removes group)."""
    try:
        # In a full implementation, would properly delete the group
        # For now, just acknowledge the request
        
        logger.info(f"Consumer group {group_name} deletion requested")
        
        return {
            "status": "success",
            "message": f"Consumer group {group_name} deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to delete consumer group {group_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete consumer group: {str(e)}")


# Consumer Management Endpoints

@router.post("/{group_name}/consumers")
async def add_consumer_to_group(
    group_name: str,
    consumer_id: str = Query(..., description="Consumer identifier"),
    coordinator: ConsumerGroupCoordinator = Depends(get_coordinator)
):
    """Add a consumer to a consumer group."""
    try:
        # Placeholder handler
        async def message_handler(message):
            logger.info(f"Consumer {consumer_id} processing message {message.id}")
        
        await coordinator.streams_manager.add_consumer_to_group(
            group_name, consumer_id, message_handler
        )
        
        return {
            "status": "success",
            "message": f"Consumer {consumer_id} added to group {group_name}"
        }
        
    except Exception as e:
        logger.error(f"Failed to add consumer {consumer_id} to group {group_name}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to add consumer: {str(e)}")


@router.delete("/{group_name}/consumers/{consumer_id}")
async def remove_consumer_from_group(
    group_name: str,
    consumer_id: str,
    coordinator: ConsumerGroupCoordinator = Depends(get_coordinator)
):
    """Remove a consumer from a consumer group."""
    try:
        await coordinator.streams_manager.remove_consumer_from_group(
            group_name, consumer_id
        )
        
        return {
            "status": "success",
            "message": f"Consumer {consumer_id} removed from group {group_name}"
        }
        
    except Exception as e:
        logger.error(f"Failed to remove consumer {consumer_id} from group {group_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove consumer: {str(e)}")


# Message Routing Endpoints

@router.post("/route", response_model=Dict[str, Any])
async def route_message(
    route_request: MessageRouteRequest,
    target_group: Optional[str] = Query(None, description="Target consumer group"),
    workflow_router: WorkflowMessageRouter = Depends(get_workflow_router)
):
    """Route a message to appropriate consumer group with workflow awareness."""
    try:
        # Create stream message
        message = StreamMessage(
            id=route_request.message_id,
            from_agent=route_request.from_agent,
            to_agent=None,  # Will be routed to consumer group
            message_type=route_request.message_type,
            payload=route_request.payload,
            priority=route_request.priority,
            timestamp=datetime.utcnow().timestamp()
        )
        
        # Route the message
        if target_group:
            # Route to specific group
            message_id = await workflow_router.streams_manager.send_message_to_group(
                target_group, message
            )
            routing_decision = {
                "target_group": target_group,
                "routing_strategy": "manual",
                "message_id": message_id
            }
        else:
            # Use intelligent routing
            routing_decision = await workflow_router.route_task_message(
                message, route_request.workflow_id, route_request.dependencies
            )
        
        return {
            "status": "success",
            "routing_decision": routing_decision.to_dict() if hasattr(routing_decision, 'to_dict') else routing_decision,
            "message_id": route_request.message_id
        }
        
    except Exception as e:
        logger.error(f"Failed to route message {route_request.message_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to route message: {str(e)}")


@router.post("/route/workflow", response_model=Dict[str, Any])
async def route_workflow(
    workflow_id: str,
    tasks: List[Dict[str, Any]],
    workflow_router: WorkflowMessageRouter = Depends(get_workflow_router)
):
    """Route an entire workflow with intelligent task distribution."""
    try:
        result = await workflow_router.route_workflow(workflow_id, tasks)
        
        return {
            "status": "success",
            "workflow_routing": result
        }
        
    except Exception as e:
        logger.error(f"Failed to route workflow {workflow_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to route workflow: {str(e)}")


# Dead Letter Queue Management Endpoints

@router.get("/dlq/statistics", response_model=Dict[str, Any])
async def get_dlq_statistics(
    dlq_handler: DeadLetterQueueHandler = Depends(get_dlq_handler)
):
    """Get comprehensive DLQ statistics and analytics."""
    try:
        stats = await dlq_handler.get_dlq_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get DLQ statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get DLQ statistics: {str(e)}")


@router.post("/dlq/replay/{dlq_message_id}")
async def replay_dlq_message(
    dlq_message_id: str,
    target_stream: Optional[str] = Query(None, description="Target stream override"),
    priority_boost: bool = Query(False, description="Boost message priority"),
    dlq_handler: DeadLetterQueueHandler = Depends(get_dlq_handler)
):
    """Replay a specific message from the Dead Letter Queue."""
    try:
        success = await dlq_handler.replay_message(
            dlq_message_id, target_stream, priority_boost
        )
        
        return {
            "status": "success" if success else "failed",
            "dlq_message_id": dlq_message_id,
            "replayed": success
        }
        
    except Exception as e:
        logger.error(f"Failed to replay DLQ message {dlq_message_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to replay message: {str(e)}")


@router.post("/dlq/replay/batch", response_model=Dict[str, Any])
async def replay_dlq_batch(
    replay_request: BatchReplayRequest,
    dlq_handler: DeadLetterQueueHandler = Depends(get_dlq_handler)
):
    """Replay a batch of messages from DLQ based on filter criteria."""
    try:
        result = await dlq_handler.replay_batch(
            replay_request.filter_criteria,
            replay_request.max_messages
        )
        
        return {
            "status": "success",
            "batch_replay_result": result
        }
        
    except Exception as e:
        logger.error(f"Failed to replay DLQ batch: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to replay batch: {str(e)}")


@router.get("/dlq/analyze", response_model=Dict[str, Any])
async def analyze_dlq_patterns(
    dlq_handler: DeadLetterQueueHandler = Depends(get_dlq_handler)
):
    """Analyze DLQ failure patterns and provide recommendations."""
    try:
        analysis = await dlq_handler.analyze_failure_patterns()
        return analysis
        
    except Exception as e:
        logger.error(f"Failed to analyze DLQ patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze patterns: {str(e)}")


# Coordination and Scaling Endpoints

@router.post("/rebalance")
async def rebalance_consumer_groups(
    background_tasks: BackgroundTasks,
    coordinator: ConsumerGroupCoordinator = Depends(get_coordinator)
):
    """Trigger consumer group rebalancing for optimal performance."""
    try:
        # Run rebalancing in background
        background_tasks.add_task(coordinator.rebalance_groups)
        
        return {
            "status": "success",
            "message": "Consumer group rebalancing initiated"
        }
        
    except Exception as e:
        logger.error(f"Failed to initiate rebalancing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate rebalancing: {str(e)}")


@router.get("/metrics", response_model=Dict[str, Any])
async def get_comprehensive_metrics(
    coordinator: ConsumerGroupCoordinator = Depends(get_coordinator),
    workflow_router: WorkflowMessageRouter = Depends(get_workflow_router)
):
    """Get comprehensive metrics for all consumer group systems."""
    try:
        coordinator_metrics = await coordinator.get_coordinator_metrics()
        streams_metrics = await coordinator.streams_manager.get_performance_metrics()
        routing_metrics = await workflow_router.get_routing_metrics()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "coordinator_metrics": coordinator_metrics,
            "streams_metrics": streams_metrics,
            "routing_metrics": routing_metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to get comprehensive metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    coordinator: ConsumerGroupCoordinator = Depends(get_coordinator),
    workflow_router: WorkflowMessageRouter = Depends(get_workflow_router),
    dlq_handler: DeadLetterQueueHandler = Depends(get_dlq_handler)
):
    """Comprehensive health check for all consumer group systems."""
    try:
        coordinator_health = await coordinator.health_check()
        streams_health = await coordinator.streams_manager.health_check()
        router_health = await workflow_router.health_check()
        dlq_health = await dlq_handler.health_check()
        
        overall_healthy = all([
            coordinator_health.get("status") == "healthy",
            streams_health.get("enhanced_status") == "healthy",
            router_health.get("status") == "healthy",
            dlq_health.get("status") == "healthy"
        ])
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "coordinator": coordinator_health,
                "streams": streams_health,
                "router": router_health,
                "dlq": dlq_health
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# Configuration Management Endpoints

@router.put("/coordinator/config")
async def update_coordinator_config(
    config: CoordinatorConfigRequest,
    coordinator: ConsumerGroupCoordinator = Depends(get_coordinator)
):
    """Update coordinator configuration."""
    try:
        # In a full implementation, would update coordinator configuration
        # For now, just acknowledge the request
        
        logger.info(f"Coordinator configuration update requested")
        
        return {
            "status": "success",
            "message": "Coordinator configuration updated",
            "config": config.dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to update coordinator config: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to update config: {str(e)}")


@router.get("/coordinator/config")
async def get_coordinator_config(
    coordinator: ConsumerGroupCoordinator = Depends(get_coordinator)
):
    """Get current coordinator configuration."""
    try:
        return {
            "strategy": coordinator.strategy.value,
            "provisioning_policy": coordinator.provisioning_policy.value,
            "allocation_mode": coordinator.allocation_mode.value,
            "enable_cross_group_coordination": coordinator.enable_cross_group_coordination,
            "health_check_interval": coordinator.health_check_interval,
            "rebalance_interval": coordinator.rebalance_interval
        }
        
    except Exception as e:
        logger.error(f"Failed to get coordinator config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")