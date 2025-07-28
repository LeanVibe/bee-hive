"""
Advanced Orchestration API Endpoints for LeanVibe Agent Hive 2.0

Vertical Slice 2.1: API endpoints for advanced orchestration features including
enhanced load balancing, intelligent routing, failure recovery, and workflow management.

Endpoints:
- /orchestration/workflow/execute - Execute advanced workflows
- /orchestration/task/assign - Assign tasks with advanced routing
- /orchestration/failure/handle - Handle system failures  
- /orchestration/metrics - Get comprehensive metrics
- /orchestration/performance/validate - Validate performance targets
- /orchestration/load-balancing/optimize - Optimize load distribution
- /orchestration/circuit-breakers - Manage circuit breakers
- /orchestration/recovery/predict - Predict failure probabilities
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import json

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_async_session
from ...core.vertical_slice_2_1_integration import (
    VerticalSlice21Integration, get_vs21_integration, IntegrationMode,
    VS21PerformanceTargets, VS21Metrics
)
from ...core.enhanced_workflow_engine import (
    EnhancedWorkflowDefinition, EnhancedTaskDefinition, WorkflowTemplate,
    EnhancedExecutionMode, ResourceAllocationStrategy, WorkflowOptimizationGoal
)
from ...core.enhanced_intelligent_task_router import (
    EnhancedTaskRoutingContext, EnhancedRoutingStrategy
)
from ...core.enhanced_failure_recovery_manager import (
    FailureEvent, FailureType, FailureSeverity, RecoveryStrategy
)
from ...models.agent import Agent, AgentStatus, AgentType
from ...models.task import Task, TaskStatus, TaskPriority, TaskType
from ...models.workflow import Workflow, WorkflowStatus
from ...schemas.task import TaskCreate, TaskResponse
from ...schemas.agent import AgentResponse
from ...schemas.workflow import WorkflowCreate, WorkflowResponse

router = APIRouter(prefix="/orchestration", tags=["Advanced Orchestration"])


# Request/Response Models

class AdvancedTaskAssignmentRequest(BaseModel):
    """Request model for advanced task assignment."""
    model_config = ConfigDict(from_attributes=True)
    
    task_id: str
    task_type: str
    priority: str = "medium"
    required_capabilities: List[str] = Field(default_factory=list)
    
    # Enhanced routing preferences
    routing_strategy: str = "hybrid_intelligence"
    preferred_cognitive_style: Optional[str] = None
    creativity_requirements: float = Field(default=0.5, ge=0.0, le=1.0)
    analytical_depth: float = Field(default=0.5, ge=0.0, le=1.0)
    collaboration_intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Performance expectations
    expected_quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_acceptable_delay_minutes: Optional[int] = Field(default=None, ge=1)
    
    # Context and constraints
    contextual_factors: Dict[str, Any] = Field(default_factory=dict)
    resource_constraints: Dict[str, Any] = Field(default_factory=dict)


class AdvancedTaskAssignmentResponse(BaseModel):
    """Response model for advanced task assignment."""
    model_config = ConfigDict(from_attributes=True)
    
    success: bool
    assigned_agent: Optional[Dict[str, Any]] = None
    assignment_metrics: Dict[str, Any] = Field(default_factory=dict)
    assignment_time_ms: float
    routing_context: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = None
    timestamp: str


class EnhancedWorkflowRequest(BaseModel):
    """Request model for enhanced workflow execution."""
    model_config = ConfigDict(from_attributes=True)
    
    workflow_id: str
    name: str
    description: str
    template: Optional[str] = None
    
    # Task definitions
    tasks: List[Dict[str, Any]]
    
    # Execution configuration
    execution_mode: str = "adaptive"
    resource_allocation_strategy: str = "optimized"
    optimization_goal: str = "balance_all"
    
    # Constraints
    max_concurrent_tasks: int = Field(default=10, ge=1, le=100)
    max_agents: int = Field(default=20, ge=1, le=100)
    max_duration_minutes: Optional[int] = Field(default=None, ge=1)
    
    # Quality and performance
    quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    failure_tolerance: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Context
    context: Dict[str, Any] = Field(default_factory=dict)


class EnhancedWorkflowResponse(BaseModel):
    """Response model for enhanced workflow execution."""
    model_config = ConfigDict(from_attributes=True)
    
    workflow_result: Dict[str, Any]
    execution_metrics: Dict[str, Any] = Field(default_factory=dict)
    performance_analysis: Dict[str, Any] = Field(default_factory=dict)
    optimization_recommendations: List[str] = Field(default_factory=list)
    integration_metadata: Dict[str, Any] = Field(default_factory=dict)


class FailureEventRequest(BaseModel):
    """Request model for failure event handling."""
    model_config = ConfigDict(from_attributes=True)
    
    failure_type: str
    severity: str
    
    # Affected components
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None
    
    # Failure details
    error_message: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class FailureEventResponse(BaseModel):
    """Response model for failure event handling."""
    model_config = ConfigDict(from_attributes=True)
    
    recovery_success: bool
    recovery_metrics: Dict[str, Any] = Field(default_factory=dict)
    system_impact: Dict[str, Any] = Field(default_factory=dict)
    improvement_recommendations: List[str] = Field(default_factory=list)
    handled_timestamp: str


class OrchestrationMetricsResponse(BaseModel):
    """Response model for orchestration metrics."""
    model_config = ConfigDict(from_attributes=True)
    
    timestamp: str
    orchestration_metrics: Dict[str, Any]
    system_metrics: Dict[str, Any] = Field(default_factory=dict)
    performance_score: float
    targets_met: int
    total_targets: int


class PerformanceValidationResponse(BaseModel):
    """Response model for performance validation."""
    model_config = ConfigDict(from_attributes=True)
    
    overall_score: float
    targets_met: int
    total_targets: int
    target_results: Dict[str, bool]
    current_metrics: Dict[str, Any]
    performance_targets: Dict[str, Any]
    improvement_recommendations: List[str] = Field(default_factory=list)
    validation_timestamp: str


class CircuitBreakerStatusResponse(BaseModel):
    """Response model for circuit breaker status."""
    model_config = ConfigDict(from_attributes=True)
    
    resource_id: str
    resource_type: str
    state: str
    failure_count: int
    success_count: int
    total_requests: int
    failure_rate: float
    last_failure: Optional[str] = None


class FailurePredictionResponse(BaseModel):
    """Response model for failure predictions."""
    model_config = ConfigDict(from_attributes=True)
    
    predictions: Dict[str, float]  # agent_id -> failure_probability
    time_horizon_minutes: int
    high_risk_agents: List[str]
    prediction_timestamp: str


# API Endpoints

@router.post("/workflow/execute", response_model=EnhancedWorkflowResponse)
async def execute_advanced_workflow(
    request: EnhancedWorkflowRequest,
    background_tasks: BackgroundTasks,
    integration: VerticalSlice21Integration = Depends(get_vs21_integration)
):
    """
    Execute an advanced workflow with enhanced orchestration capabilities.
    
    This endpoint provides sophisticated workflow execution with:
    - Advanced dependency management
    - Intelligent resource allocation
    - Real-time optimization
    - Comprehensive monitoring
    """
    try:
        # Convert request to enhanced workflow definition
        task_definitions = []
        for task_data in request.tasks:
            task_def = EnhancedTaskDefinition(
                task_id=task_data.get("task_id"),
                task_type=TaskType(task_data.get("task_type", "code_generation")),
                name=task_data.get("name", ""),
                description=task_data.get("description", ""),
                dependencies=task_data.get("dependencies", []),
                required_capabilities=task_data.get("required_capabilities", []),
                estimated_duration_minutes=task_data.get("estimated_duration_minutes", 30),
                priority=TaskPriority(task_data.get("priority", "medium"))
            )
            task_definitions.append(task_def)
        
        workflow_definition = EnhancedWorkflowDefinition(
            workflow_id=request.workflow_id,
            name=request.name,
            description=request.description,
            template=WorkflowTemplate(request.template) if request.template else None,
            tasks=task_definitions,
            execution_mode=EnhancedExecutionMode(request.execution_mode),
            resource_allocation_strategy=ResourceAllocationStrategy(request.resource_allocation_strategy),
            optimization_goal=WorkflowOptimizationGoal(request.optimization_goal),
            max_concurrent_tasks=request.max_concurrent_tasks,
            max_agents=request.max_agents,
            max_duration_minutes=request.max_duration_minutes,
            quality_threshold=request.quality_threshold,
            failure_tolerance=request.failure_tolerance,
            context=request.context
        )
        
        # Execute workflow through integration service
        result = await integration.execute_advanced_workflow(workflow_definition, request.context)
        
        return EnhancedWorkflowResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")


@router.post("/task/assign", response_model=AdvancedTaskAssignmentResponse)
async def assign_task_with_advanced_routing(
    request: AdvancedTaskAssignmentRequest,
    integration: VerticalSlice21Integration = Depends(get_vs21_integration),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Assign a task using advanced routing with persona-based matching.
    
    This endpoint provides sophisticated task assignment with:
    - Persona-based agent matching
    - Performance history analysis
    - Contextual suitability scoring
    - Real-time load balancing
    """
    try:
        # Get task from database
        task_result = await db.get(Task, request.task_id)
        if not task_result:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task = task_result
        
        # Create routing preferences
        routing_preferences = {
            "strategy": request.routing_strategy,
            "preferred_cognitive_style": request.preferred_cognitive_style,
            "creativity_requirements": request.creativity_requirements,
            "analytical_depth": request.analytical_depth,
            "collaboration_intensity": request.collaboration_intensity,
            "expected_quality_threshold": request.expected_quality_threshold,
            "max_acceptable_delay_minutes": request.max_acceptable_delay_minutes,
            "contextual_factors": request.contextual_factors,
            "resource_constraints": request.resource_constraints
        }
        
        # Assign task through integration service
        result = await integration.assign_task_with_orchestration(task, routing_preferences)
        
        return AdvancedTaskAssignmentResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task assignment failed: {str(e)}")


@router.post("/failure/handle", response_model=FailureEventResponse)
async def handle_system_failure(
    request: FailureEventRequest,
    background_tasks: BackgroundTasks,
    integration: VerticalSlice21Integration = Depends(get_vs21_integration)
):
    """
    Handle a system failure with automatic recovery.
    
    This endpoint provides comprehensive failure handling with:
    - Automatic task reassignment
    - Circuit breaker management
    - Recovery strategy execution
    - Impact assessment
    """
    try:
        # Create failure event
        failure_event = FailureEvent(
            event_id=str(uuid.uuid4()),
            failure_type=FailureType(request.failure_type),
            severity=FailureSeverity(request.severity),
            timestamp=datetime.utcnow(),
            agent_id=request.agent_id,
            task_id=request.task_id,
            session_id=request.session_id,
            workflow_id=request.workflow_id,
            error_message=request.error_message,
            context=request.context
        )
        
        # Handle failure through integration service
        result = await integration.handle_system_failure(failure_event)
        
        return FailureEventResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failure handling failed: {str(e)}")


@router.get("/metrics", response_model=OrchestrationMetricsResponse)
async def get_orchestration_metrics(
    integration: VerticalSlice21Integration = Depends(get_vs21_integration)
):
    """
    Get comprehensive orchestration metrics.
    
    Returns detailed metrics including:
    - Load balancing efficiency
    - Routing accuracy
    - Failure recovery performance
    - Workflow execution statistics
    - System resource utilization
    """
    try:
        metrics = await integration.get_comprehensive_metrics()
        
        # Calculate performance score
        performance_score = metrics.calculate_overall_score(VS21PerformanceTargets())
        target_results = metrics.meets_targets(VS21PerformanceTargets())
        
        return OrchestrationMetricsResponse(
            timestamp=metrics.timestamp.isoformat(),
            orchestration_metrics=metrics.orchestration_metrics.__dict__,
            system_metrics={
                "active_agents": metrics.active_agents,
                "total_tasks": metrics.total_tasks,
                "completed_tasks": metrics.completed_tasks,
                "failed_tasks": metrics.failed_tasks,
                "active_workflows": metrics.active_workflows,
                "cpu_utilization_percent": metrics.cpu_utilization_percent,
                "memory_utilization_percent": metrics.memory_utilization_percent
            },
            performance_score=performance_score,
            targets_met=sum(1 for met in target_results.values() if met),
            total_targets=len(target_results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/performance/validate", response_model=PerformanceValidationResponse)
async def validate_performance_targets(
    integration: VerticalSlice21Integration = Depends(get_vs21_integration)
):
    """
    Validate current performance against VS 2.1 targets.
    
    Returns detailed analysis of:
    - Target achievement status
    - Performance gaps
    - Improvement recommendations
    - Trending analysis
    """
    try:
        validation_result = await integration.validate_performance_targets()
        
        return PerformanceValidationResponse(**validation_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance validation failed: {str(e)}")


@router.post("/load-balancing/optimize")
async def optimize_load_distribution(
    background_tasks: BackgroundTasks,
    integration: VerticalSlice21Integration = Depends(get_vs21_integration)
):
    """
    Trigger load distribution optimization.
    
    This endpoint initiates:
    - Load analysis across all agents
    - Optimization plan generation
    - Task rebalancing execution
    - Performance impact assessment
    """
    try:
        # Trigger optimization in background
        background_tasks.add_task(
            integration.orchestration_engine.optimize_load_distribution
        )
        
        return {
            "message": "Load distribution optimization initiated",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load optimization failed: {str(e)}")


@router.get("/circuit-breakers", response_model=List[CircuitBreakerStatusResponse])
async def get_circuit_breaker_status(
    integration: VerticalSlice21Integration = Depends(get_vs21_integration)
):
    """
    Get status of all circuit breakers.
    
    Returns information about:
    - Circuit breaker states
    - Failure/success counts
    - Performance statistics
    - Recent activity
    """
    try:
        circuit_breakers = []
        
        # Get circuit breaker information from recovery manager
        if integration.recovery_manager:
            for resource_id, cb in integration.recovery_manager.circuit_breakers.items():
                status = CircuitBreakerStatusResponse(
                    resource_id=cb.resource_id,
                    resource_type=cb.resource_type,
                    state=cb.state.value,
                    failure_count=cb.failure_count,
                    success_count=cb.success_count,
                    total_requests=cb.total_requests,
                    failure_rate=cb.total_failures / cb.total_requests if cb.total_requests > 0 else 0.0,
                    last_failure=cb.last_failure_time.isoformat() if cb.last_failure_time else None
                )
                circuit_breakers.append(status)
        
        return circuit_breakers
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get circuit breaker status: {str(e)}")


@router.post("/circuit-breakers/{resource_id}/reset")
async def reset_circuit_breaker(
    resource_id: str,
    integration: VerticalSlice21Integration = Depends(get_vs21_integration)
):
    """
    Manually reset a circuit breaker.
    
    This action:
    - Resets failure counts
    - Restores normal operation
    - Logs the manual intervention
    """
    try:
        if not integration.recovery_manager:
            raise HTTPException(status_code=503, detail="Recovery manager not available")
        
        success = await integration.recovery_manager.reset_circuit_breaker(resource_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Circuit breaker not found")
        
        return {
            "message": f"Circuit breaker {resource_id} successfully reset",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Circuit breaker reset failed: {str(e)}")


@router.get("/recovery/predict", response_model=FailurePredictionResponse)
async def predict_failure_probabilities(
    time_horizon_minutes: int = Query(default=30, ge=1, le=1440),
    integration: VerticalSlice21Integration = Depends(get_vs21_integration)
):
    """
    Predict failure probabilities for active agents.
    
    This endpoint provides:
    - Machine learning-based failure prediction
    - Risk assessment for each agent
    - Recommended preventive actions
    - Trend analysis
    """
    try:
        if not integration.recovery_manager:
            raise HTTPException(status_code=503, detail="Recovery manager not available")
        
        predictions = await integration.recovery_manager.predict_agent_failures(time_horizon_minutes)
        
        # Identify high-risk agents (>70% failure probability)
        high_risk_agents = [agent_id for agent_id, prob in predictions.items() if prob > 0.7]
        
        return FailurePredictionResponse(
            predictions=predictions,
            time_horizon_minutes=time_horizon_minutes,
            high_risk_agents=high_risk_agents,
            prediction_timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failure prediction failed: {str(e)}")


@router.get("/health")
async def get_orchestration_health(
    integration: VerticalSlice21Integration = Depends(get_vs21_integration)
):
    """
    Get overall health status of the orchestration system.
    
    Returns:
    - Component health status
    - System availability
    - Recent performance metrics
    - Active alerts
    """
    try:
        # Basic health check
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "integration_mode": integration.mode.value,
            "running": integration.running,
            "components": {
                "orchestration_engine": integration.orchestration_engine is not None,
                "task_router": integration.task_router is not None,
                "recovery_manager": integration.recovery_manager is not None,
                "workflow_engine": integration.workflow_engine is not None,
                "persona_system": integration.persona_system is not None
            },
            "uptime_seconds": (datetime.utcnow() - integration.integration_start_time).total_seconds()
        }
        
        # Check component health
        unhealthy_components = [name for name, healthy in health_status["components"].items() if not healthy]
        if unhealthy_components:
            health_status["status"] = "degraded"
            health_status["issues"] = f"Unhealthy components: {', '.join(unhealthy_components)}"
        
        return health_status
        
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }