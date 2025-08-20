"""
Enhanced API endpoints for intelligent sleep management.

Provides comprehensive REST API for sleep-wake orchestration, analytics,
and intelligent agent management capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from pydantic import BaseModel, Field, validator

from ..core.sleep_wake_manager import get_sleep_wake_manager
from ..core.simple_orchestrator import SimpleOrchestrator, create_simple_orchestrator
from ..core.recovery_manager import get_recovery_manager
from ..core.sleep_analytics import get_sleep_analytics_engine
from ..core.security import get_current_user, require_analytics_access
from ..models.sleep_wake import SleepState


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sleep", tags=["Sleep Management"])


# Request/Response Models
class SleepCycleRequest(BaseModel):
    """Request model for initiating sleep cycles."""
    agent_id: UUID = Field(..., description="Agent ID to put to sleep")
    cycle_type: str = Field(default="manual", description="Type of sleep cycle")
    expected_wake_time: Optional[datetime] = Field(
        default=None, 
        description="Expected wake time (ISO format)"
    )
    force: bool = Field(
        default=False, 
        description="Force sleep even if agent is busy"
    )
    
    @validator('cycle_type')
    def validate_cycle_type(cls, v):
        allowed_types = ["manual", "scheduled", "emergency", "maintenance"]
        if v not in allowed_types:
            raise ValueError(f"cycle_type must be one of {allowed_types}")
        return v


class WakeCycleRequest(BaseModel):
    """Request model for wake cycles."""
    agent_id: UUID = Field(..., description="Agent ID to wake up")
    verify_health: bool = Field(
        default=True, 
        description="Perform health verification after wake"
    )
    recovery_mode: bool = Field(
        default=False, 
        description="Use recovery mode for problematic wake"
    )


class BulkSleepRequest(BaseModel):
    """Request model for bulk sleep operations."""
    agent_ids: List[UUID] = Field(..., description="List of agent IDs")
    cycle_type: str = Field(default="manual", description="Type of sleep cycle")
    stagger_delay_seconds: int = Field(
        default=0, 
        description="Delay between operations in seconds"
    )
    max_concurrent: int = Field(
        default=5, 
        description="Maximum concurrent operations"
    )


class ScheduleRequest(BaseModel):
    """Request model for scheduling sleep operations."""
    agent_id: Optional[UUID] = Field(
        default=None, 
        description="Agent ID, None for system-wide"
    )
    schedule_time: datetime = Field(..., description="When to schedule the operation")
    operation_type: str = Field(..., description="Type of operation (sleep/wake)")
    recurring: bool = Field(default=False, description="Recurring schedule")
    recurrence_pattern: Optional[str] = Field(
        default=None, 
        description="Cron expression for recurring"
    )
    
    @validator('operation_type')
    def validate_operation_type(cls, v):
        if v not in ["sleep", "wake", "consolidate", "optimize"]:
            raise ValueError("operation_type must be sleep, wake, consolidate, or optimize")
        return v


class SystemConfigRequest(BaseModel):
    """Request model for system configuration updates."""
    enable_intelligent_scheduling: bool = Field(
        default=True, 
        description="Enable intelligent scheduling"
    )
    auto_recovery_enabled: bool = Field(
        default=True, 
        description="Enable automatic recovery"
    )
    consolidation_threshold_percent: int = Field(
        default=80, 
        ge=50, 
        le=95,
        description="Context usage threshold for consolidation"
    )
    max_sleep_duration_hours: int = Field(
        default=24, 
        ge=1, 
        le=168,
        description="Maximum sleep duration in hours"
    )
    background_optimization: bool = Field(
        default=True, 
        description="Enable background optimization"
    )


class SleepCycleResponse(BaseModel):
    """Response model for sleep cycle operations."""
    success: bool
    message: str
    cycle_id: Optional[UUID] = None
    agent_id: UUID
    estimated_completion_time: Optional[datetime] = None
    warnings: List[str] = []


class SystemStatusResponse(BaseModel):
    """Response model for system status."""
    timestamp: datetime
    system_healthy: bool
    total_agents: int
    active_sleep_cycles: int
    pending_operations: int
    recent_performance: Dict[str, Any]
    error_summary: List[str] = []


# Core Sleep Management Endpoints
@router.post("/cycle/start", response_model=SleepCycleResponse)
async def start_sleep_cycle(
    request: SleepCycleRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Initiate a sleep cycle for an agent with intelligent optimization.
    
    Features:
    - Pre-sleep validation and health checks
    - Intelligent checkpoint creation
    - Automated consolidation during sleep
    - Performance monitoring and optimization
    """
    try:
        logger.info(f"Starting sleep cycle for agent {request.agent_id}")
        
        # Get sleep-wake manager
        sleep_manager = await get_sleep_wake_manager()
        
        # Pre-flight checks
        if not request.force:
            # Check if agent is in a suitable state for sleep
            system_status = await sleep_manager.get_system_status()
            agent_status = system_status.get("agents", {}).get(str(request.agent_id))
            
            if not agent_status:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent {request.agent_id} not found"
                )
            
            if agent_status["sleep_state"] != SleepState.AWAKE.value:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Agent {request.agent_id} is not awake (current state: {agent_status['sleep_state']})"
                )
        
        # Initiate sleep cycle
        success = await sleep_manager.initiate_sleep_cycle(
            agent_id=request.agent_id,
            cycle_type=request.cycle_type,
            expected_wake_time=request.expected_wake_time
        )
        
        if success:
            # Estimate completion time
            estimated_completion = request.expected_wake_time
            if not estimated_completion:
                # Default to 1 hour if no wake time specified
                estimated_completion = datetime.utcnow() + timedelta(hours=1)
            
            return SleepCycleResponse(
                success=True,
                message=f"Sleep cycle initiated successfully for agent {request.agent_id}",
                agent_id=request.agent_id,
                estimated_completion_time=estimated_completion,
                warnings=[]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initiate sleep cycle"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting sleep cycle: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error starting sleep cycle: {str(e)}"
        )


@router.post("/cycle/wake")
async def wake_agent(
    request: WakeCycleRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Wake an agent from sleep with optional recovery and verification.
    
    Features:
    - Intelligent state restoration
    - Health verification and validation
    - Automatic recovery on wake failure
    - Performance optimization after wake
    """
    try:
        logger.info(f"Waking agent {request.agent_id}")
        
        sleep_manager = await get_sleep_wake_manager()
        
        # Standard wake cycle
        if not request.recovery_mode:
            success = await sleep_manager.initiate_wake_cycle(request.agent_id)
            
            if success:
                return {
                    "success": True,
                    "message": f"Agent {request.agent_id} woken successfully",
                    "agent_id": request.agent_id,
                    "recovery_used": False
                }
            else:
                # Fall back to recovery mode
                request.recovery_mode = True
        
        # Recovery mode wake
        if request.recovery_mode:
            recovery_manager = get_recovery_manager()
            success, checkpoint = await recovery_manager.initiate_recovery(
                agent_id=request.agent_id,
                recovery_type="manual"
            )
            
            if success:
                return {
                    "success": True,
                    "message": f"Agent {request.agent_id} woken using recovery mode",
                    "agent_id": request.agent_id,
                    "recovery_used": True,
                    "checkpoint_id": str(checkpoint.id) if checkpoint else None
                }
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to wake agent {request.agent_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error waking agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error waking agent: {str(e)}"
        )


@router.post("/bulk/sleep")
async def bulk_sleep_operation(
    request: BulkSleepRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Perform bulk sleep operations with intelligent staggering and concurrency control.
    
    Features:
    - Configurable concurrency limits
    - Intelligent staggering to prevent resource conflicts
    - Individual operation tracking and error handling
    - Comprehensive success/failure reporting
    """
    try:
        logger.info(f"Starting bulk sleep operation for {len(request.agent_ids)} agents")
        
        sleep_manager = await get_sleep_wake_manager()
        results = []
        
        # Process agents in batches
        semaphore = asyncio.Semaphore(request.max_concurrent)
        
        async def sleep_single_agent(agent_id: UUID, delay: float) -> Dict[str, Any]:
            async with semaphore:
                try:
                    # Apply stagger delay
                    if delay > 0:
                        await asyncio.sleep(delay)
                    
                    success = await sleep_manager.initiate_sleep_cycle(
                        agent_id=agent_id,
                        cycle_type=request.cycle_type
                    )
                    
                    return {
                        "agent_id": str(agent_id),
                        "success": success,
                        "message": "Sleep cycle initiated" if success else "Failed to initiate sleep cycle",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                except Exception as e:
                    return {
                        "agent_id": str(agent_id),
                        "success": False,
                        "message": f"Error: {str(e)}",
                        "timestamp": datetime.utcnow().isoformat()
                    }
        
        # Create tasks with staggered delays
        tasks = []
        for i, agent_id in enumerate(request.agent_ids):
            delay = i * request.stagger_delay_seconds
            tasks.append(sleep_single_agent(agent_id, delay))
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "agent_id": "unknown",
                    "success": False,
                    "message": f"Task exception: {str(result)}",
                    "timestamp": datetime.utcnow().isoformat()
                })
            else:
                processed_results.append(result)
        
        # Summary statistics
        successful_count = sum(1 for r in processed_results if r["success"])
        failed_count = len(processed_results) - successful_count
        
        return {
            "bulk_operation_id": f"bulk_sleep_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "total_agents": len(request.agent_ids),
            "successful": successful_count,
            "failed": failed_count,
            "success_rate": successful_count / len(request.agent_ids) if request.agent_ids else 0,
            "results": processed_results,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in bulk sleep operation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk operation failed: {str(e)}"
        )


# Intelligent Scheduling Endpoints
@router.post("/schedule/create")
async def create_schedule(
    request: ScheduleRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Create intelligent scheduling for sleep-wake operations.
    
    Features:
    - Flexible scheduling with cron expressions
    - Agent-specific or system-wide scheduling
    - Recurring patterns with intelligent optimization
    - Conflict detection and resolution
    """
    try:
        logger.info(f"Creating schedule for {request.operation_type} operation")
        
        orchestrator = create_simple_orchestrator()
        
        # Create schedule
        schedule_id = await orchestrator.create_schedule(
            operation_type=request.operation_type,
            agent_id=request.agent_id,
            schedule_time=request.schedule_time,
            recurring=request.recurring,
            recurrence_pattern=request.recurrence_pattern
        )
        
        return {
            "success": True,
            "schedule_id": str(schedule_id),
            "message": f"Schedule created for {request.operation_type} operation",
            "next_execution": request.schedule_time.isoformat(),
            "recurring": request.recurring
        }
        
    except Exception as e:
        logger.error(f"Error creating schedule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create schedule: {str(e)}"
        )


@router.get("/schedule/list")
async def list_schedules(
    agent_id: Optional[UUID] = Query(None, description="Filter by agent ID"),
    active_only: bool = Query(True, description="Show only active schedules"),
    current_user: dict = Depends(get_current_user)
):
    """List all scheduled operations with filtering options."""
    try:
        orchestrator = create_simple_orchestrator()
        
        schedules = await orchestrator.get_schedules(
            agent_id=agent_id,
            active_only=active_only
        )
        
        return {
            "schedules": schedules,
            "total_count": len(schedules),
            "filtered_by_agent": agent_id is not None,
            "active_only": active_only
        }
        
    except Exception as e:
        logger.error(f"Error listing schedules: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list schedules: {str(e)}"
        )


# System Management Endpoints
@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    include_detailed_metrics: bool = Query(False, description="Include detailed performance metrics"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get comprehensive system status with health monitoring.
    
    Features:
    - Real-time agent state monitoring
    - Performance metrics and trends
    - Error detection and alerting
    - Resource utilization tracking
    """
    try:
        sleep_manager = await get_sleep_wake_manager()
        orchestrator = create_simple_orchestrator()
        
        # Get base system status
        status = await sleep_manager.get_system_status()
        
        # Enhance with orchestrator status
        orchestrator_status = await orchestrator.get_system_health()
        
        # Calculate summary metrics
        total_agents = len(status.get("agents", {}))
        active_sleep_cycles = sum(
            1 for agent in status.get("agents", {}).values()
            if agent["sleep_state"] in ["SLEEPING", "PREPARING_SLEEP"]
        )
        
        response = SystemStatusResponse(
            timestamp=datetime.utcnow(),
            system_healthy=status.get("system_healthy", False) and orchestrator_status.get("healthy", False),
            total_agents=total_agents,
            active_sleep_cycles=active_sleep_cycles,
            pending_operations=orchestrator_status.get("pending_operations", 0),
            recent_performance=status.get("metrics", {}),
            error_summary=status.get("errors", []) + orchestrator_status.get("errors", [])
        )
        
        if include_detailed_metrics:
            # Add detailed analytics
            analytics_engine = get_sleep_analytics_engine()
            detailed_metrics = await analytics_engine.get_comprehensive_dashboard()
            response.recent_performance.update(detailed_metrics)
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )


@router.post("/configuration/update")
async def update_system_configuration(
    request: SystemConfigRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Update system configuration for intelligent sleep management.
    
    Features:
    - Runtime configuration updates without restart
    - Validation and safety checks
    - Configuration history and rollback
    - Impact assessment and warnings
    """
    try:
        logger.info("Updating system configuration")
        
        orchestrator = create_simple_orchestrator()
        
        # Update configuration
        config_result = await orchestrator.update_configuration({
            "intelligent_scheduling": request.enable_intelligent_scheduling,
            "auto_recovery": request.auto_recovery_enabled,
            "consolidation_threshold": request.consolidation_threshold_percent,
            "max_sleep_duration": request.max_sleep_duration_hours,
            "background_optimization": request.background_optimization
        })
        
        return {
            "success": True,
            "message": "System configuration updated successfully",
            "updated_settings": request.dict(),
            "restart_required": config_result.get("restart_required", False),
            "warnings": config_result.get("warnings", [])
        }
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {str(e)}"
        )


@router.post("/emergency/shutdown")
async def emergency_shutdown(
    agent_id: Optional[UUID] = Query(None, description="Specific agent ID, None for system-wide"),
    current_user: dict = Depends(get_current_user)
):
    """
    Perform emergency shutdown with immediate recovery.
    
    Features:
    - Immediate state preservation
    - Graceful resource cleanup
    - Automatic recovery initiation
    - Emergency contact notification
    """
    try:
        logger.warning(f"Emergency shutdown requested for agent {agent_id}")
        
        sleep_manager = await get_sleep_wake_manager()
        
        # Perform emergency shutdown
        success = await sleep_manager.emergency_shutdown(agent_id)
        
        return {
            "success": success,
            "message": f"Emergency shutdown {'completed' if success else 'failed'}",
            "agent_id": str(agent_id) if agent_id else "system-wide",
            "timestamp": datetime.utcnow().isoformat(),
            "recovery_initiated": success
        }
        
    except Exception as e:
        logger.error(f"Error during emergency shutdown: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Emergency shutdown failed: {str(e)}"
        )


@router.post("/optimization/run")
async def run_system_optimization(
    target_metric: str = Query("performance", description="Optimization target"),
    current_user: dict = Depends(get_current_user)
):
    """
    Run intelligent system optimization.
    
    Features:
    - Performance-based optimization
    - Resource utilization improvements
    - Predictive optimization strategies
    - Non-disruptive optimization execution
    """
    try:
        logger.info(f"Running system optimization targeting {target_metric}")
        
        sleep_manager = await get_sleep_wake_manager()
        orchestrator = create_simple_orchestrator()
        
        # Run optimization
        optimization_results = await sleep_manager.optimize_performance()
        orchestrator_optimization = await orchestrator.optimize_system()
        
        # Combine results
        combined_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "target_metric": target_metric,
            "sleep_manager_optimizations": optimization_results,
            "orchestrator_optimizations": orchestrator_optimization,
            "overall_success": True
        }
        
        return combined_results
        
    except Exception as e:
        logger.error(f"Error running optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )


# Recovery and Diagnostics Endpoints
@router.get("/diagnostics/health")
async def comprehensive_health_check(
    agent_id: Optional[UUID] = Query(None, description="Specific agent for health check"),
    current_user: dict = Depends(get_current_user)
):
    """
    Perform comprehensive health diagnostics.
    
    Features:
    - Multi-component health validation
    - Performance bottleneck identification
    - Predictive failure detection
    - Actionable recommendations
    """
    try:
        recovery_manager = get_recovery_manager()
        
        # Perform comprehensive health check
        readiness = await recovery_manager.validate_recovery_readiness(agent_id)
        
        # Add system-wide health metrics
        sleep_manager = await get_sleep_wake_manager()
        system_status = await sleep_manager.get_system_status()
        
        health_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": str(agent_id) if agent_id else "system-wide",
            "overall_healthy": readiness["ready"] and system_status["system_healthy"],
            "recovery_readiness": readiness,
            "system_metrics": system_status["metrics"],
            "recommendations": [],
            "warnings": []
        }
        
        # Generate recommendations
        if not readiness["ready"]:
            health_report["recommendations"].extend([
                "System not ready for recovery operations",
                "Review error conditions and resolve issues"
            ])
        
        if len(readiness.get("errors", [])) > 0:
            health_report["warnings"].extend(readiness["errors"])
        
        return health_report
        
    except Exception as e:
        logger.error(f"Error performing health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/recovery/history")
async def get_recovery_history(
    agent_id: Optional[UUID] = Query(None, description="Filter by agent ID"),
    limit: int = Query(50, ge=1, le=200, description="Maximum records to return"),
    current_user: dict = Depends(get_current_user)
):
    """Get recovery operation history with filtering and pagination."""
    try:
        recovery_manager = get_recovery_manager()
        
        history = await recovery_manager.get_recovery_history(
            agent_id=agent_id,
            limit=limit
        )
        
        return {
            "history": history,
            "total_records": len(history),
            "filtered_by_agent": agent_id is not None,
            "agent_id": str(agent_id) if agent_id else None
        }
        
    except Exception as e:
        logger.error(f"Error getting recovery history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recovery history: {str(e)}"
        )