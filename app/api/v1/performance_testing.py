"""
Performance Testing API Endpoints

Provides comprehensive API endpoints for orchestrating and managing
performance testing across all system components.

Features:
- Orchestrated performance testing execution
- Real-time monitoring and status tracking  
- Regression testing and baseline comparison
- Comprehensive reporting and analytics
- CI/CD integration support
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks, status
from pydantic import BaseModel, Field, validator

from ...core.performance_orchestrator import (
    PerformanceOrchestrator,
    OrchestrationConfig,
    OrchestrationResult,
    TestCategory,
    TestStatus,
    create_performance_orchestrator
)
from ...core.context_manager import ContextManager
from ...core.database import get_session
from ...core.security import get_current_user

router = APIRouter(prefix="/api/v1/performance", tags=["Performance Testing"])


# Request/Response Models

class OrchestrationConfigRequest(BaseModel):
    """Request model for orchestration configuration."""
    
    # Test execution settings
    enable_context_engine_tests: bool = Field(default=True, description="Enable Context Engine tests")
    enable_redis_streams_tests: bool = Field(default=True, description="Enable Redis Streams tests")
    enable_vertical_slice_tests: bool = Field(default=True, description="Enable Vertical Slice tests")
    enable_system_integration_tests: bool = Field(default=True, description="Enable System Integration tests")
    
    # Test parameters
    context_engine_iterations: int = Field(default=5, ge=1, le=20, description="Context Engine test iterations")
    redis_streams_duration_minutes: int = Field(default=10, ge=1, le=60, description="Redis Streams test duration")
    vertical_slice_scenarios: int = Field(default=3, ge=1, le=10, description="Vertical Slice test scenarios")
    integration_test_scenarios: int = Field(default=2, ge=1, le=5, description="Integration test scenarios")
    
    # Execution settings
    parallel_execution: bool = Field(default=True, description="Enable parallel test execution")
    max_concurrent_tests: int = Field(default=3, ge=1, le=10, description="Maximum concurrent tests")
    timeout_minutes: int = Field(default=60, ge=10, le=240, description="Test timeout in minutes")
    
    # Reporting settings
    generate_detailed_reports: bool = Field(default=True, description="Generate detailed reports")
    export_metrics_to_prometheus: bool = Field(default=True, description="Export metrics to Prometheus")
    enable_real_time_monitoring: bool = Field(default=True, description="Enable real-time monitoring")
    
    # CI/CD integration
    fail_on_critical_failures: bool = Field(default=True, description="Fail on critical failures")
    fail_on_regression_percent: float = Field(default=15.0, ge=0.0, le=50.0, description="Regression failure threshold")
    baseline_comparison_enabled: bool = Field(default=True, description="Enable baseline comparison")


class TestExecutionRequest(BaseModel):
    """Request model for test execution."""
    test_suite_name: Optional[str] = Field(default=None, description="Optional test suite name")
    config_override: Optional[OrchestrationConfigRequest] = Field(default=None, description="Configuration override")
    baseline_comparison: Optional[bool] = Field(default=None, description="Enable baseline comparison")
    tags: Dict[str, str] = Field(default_factory=dict, description="Custom tags for the test run")


class RegressionTestRequest(BaseModel):
    """Request model for regression testing."""
    baseline_orchestration_id: str = Field(..., description="Baseline orchestration ID for comparison")
    regression_threshold_percent: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=50.0, 
        description="Regression threshold percentage"
    )
    test_suite_name: Optional[str] = Field(default=None, description="Optional test suite name")
    config_override: Optional[OrchestrationConfigRequest] = Field(default=None, description="Configuration override")


class ContinuousMonitoringRequest(BaseModel):
    """Request model for continuous monitoring."""
    monitoring_duration_minutes: int = Field(default=60, ge=5, le=480, description="Monitoring duration")
    sampling_interval_seconds: int = Field(default=30, ge=5, le=300, description="Sampling interval")
    alert_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {"cpu_percent": 80.0, "memory_percent": 85.0},
        description="Alert thresholds"
    )


class TestStatusResponse(BaseModel):
    """Response model for test status."""
    orchestration_id: str
    status: str
    progress_percent: int
    running_tests: int
    completed_tests: int
    failed_tests: int
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    performance_score: Optional[float] = None


class PerformanceTargetsResponse(BaseModel):
    """Response model for performance targets."""
    targets: List[Dict[str, Any]]
    categories: List[str]
    critical_targets_count: int
    total_targets_count: int


# Global orchestrator instance
_orchestrator: Optional[PerformanceOrchestrator] = None


async def get_orchestrator() -> PerformanceOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = await create_performance_orchestrator()
    return _orchestrator


# Performance Testing Endpoints

@router.post("/orchestration/start", response_model=Dict[str, Any])
async def start_comprehensive_testing(
    request: TestExecutionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Start comprehensive performance testing across all components.
    
    Features:
    - Coordinated testing of Context Engine, Redis Streams, and Vertical Slice
    - Configurable test parameters and execution modes
    - Real-time progress tracking and monitoring
    - Automated baseline comparison and regression detection
    """
    try:
        orchestrator = await get_orchestrator()
        
        # Apply configuration override if provided
        if request.config_override:
            config = OrchestrationConfig(**request.config_override.model_dump())
            orchestrator.config = config
        
        # Start testing in background
        orchestration_future = asyncio.create_task(
            orchestrator.run_comprehensive_testing(
                test_suite_name=request.test_suite_name,
                baseline_comparison=request.baseline_comparison
            )
        )
        
        # Get initial status
        orchestration_id = orchestrator.current_orchestration.orchestration_id if orchestrator.current_orchestration else "unknown"
        
        return {
            "orchestration_id": orchestration_id,
            "status": "started",
            "message": "Comprehensive performance testing started",
            "test_suite_name": request.test_suite_name,
            "configuration": orchestrator.config.__dict__,
            "estimated_duration_minutes": orchestrator.config.timeout_minutes,
            "status_url": f"/api/v1/performance/orchestration/{orchestration_id}/status",
            "results_url": f"/api/v1/performance/orchestration/{orchestration_id}/results",
            "started_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start performance testing: {str(e)}"
        )


@router.post("/regression/start", response_model=Dict[str, Any])
async def start_regression_testing(
    request: RegressionTestRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Start regression testing against a baseline orchestration.
    
    Features:
    - Automated comparison against historical baselines
    - Configurable regression thresholds
    - Detailed regression analysis and reporting
    - Performance trend identification
    """
    try:
        orchestrator = await get_orchestrator()
        
        # Apply configuration override if provided
        if request.config_override:
            config = OrchestrationConfig(**request.config_override.model_dump())
            orchestrator.config = config
        
        # Start regression testing in background
        regression_future = asyncio.create_task(
            orchestrator.run_regression_testing(
                baseline_orchestration_id=request.baseline_orchestration_id,
                regression_threshold_percent=request.regression_threshold_percent
            )
        )
        
        # Get initial status
        orchestration_id = orchestrator.current_orchestration.orchestration_id if orchestrator.current_orchestration else "unknown"
        
        return {
            "orchestration_id": orchestration_id,
            "status": "started",
            "message": "Regression testing started",
            "baseline_orchestration_id": request.baseline_orchestration_id,
            "regression_threshold": request.regression_threshold_percent or orchestrator.config.fail_on_regression_percent,
            "test_suite_name": request.test_suite_name,
            "status_url": f"/api/v1/performance/orchestration/{orchestration_id}/status",
            "results_url": f"/api/v1/performance/orchestration/{orchestration_id}/results",
            "started_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start regression testing: {str(e)}"
        )


@router.post("/monitoring/start", response_model=Dict[str, Any])
async def start_continuous_monitoring(
    request: ContinuousMonitoringRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Start continuous performance monitoring.
    
    Features:
    - Real-time performance metrics collection
    - Configurable sampling intervals and thresholds
    - Automated alerting on performance degradation
    - Trend analysis and performance insights
    """
    try:
        orchestrator = await get_orchestrator()
        
        # Start monitoring in background
        monitoring_future = asyncio.create_task(
            orchestrator.run_continuous_monitoring(
                monitoring_duration_minutes=request.monitoring_duration_minutes,
                sampling_interval_seconds=request.sampling_interval_seconds
            )
        )
        
        monitoring_id = f"monitoring_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "monitoring_id": monitoring_id,
            "status": "started",
            "message": "Continuous monitoring started",
            "duration_minutes": request.monitoring_duration_minutes,
            "sampling_interval_seconds": request.sampling_interval_seconds,
            "alert_thresholds": request.alert_thresholds,
            "estimated_samples": request.monitoring_duration_minutes * 60 // request.sampling_interval_seconds,
            "status_url": f"/api/v1/performance/monitoring/{monitoring_id}/status",
            "started_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start continuous monitoring: {str(e)}"
        )


@router.get("/orchestration/{orchestration_id}/status", response_model=TestStatusResponse)
async def get_orchestration_status(
    orchestration_id: str = Path(..., description="Orchestration ID"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get the current status of a performance testing orchestration.
    
    Provides real-time updates on test progress, completion status,
    and preliminary performance metrics.
    """
    try:
        orchestrator = await get_orchestrator()
        
        status_info = await orchestrator.get_orchestration_status(orchestration_id)
        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Orchestration {orchestration_id} not found"
            )
        
        return TestStatusResponse(**status_info)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get orchestration status: {str(e)}"
        )


@router.get("/orchestration/{orchestration_id}/results", response_model=Dict[str, Any])
async def get_orchestration_results(
    orchestration_id: str = Path(..., description="Orchestration ID"),
    include_detailed_metrics: bool = Query(False, description="Include detailed metrics"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get comprehensive results from a performance testing orchestration.
    
    Returns detailed performance metrics, target validation results,
    recommendations, and comparative analysis.
    """
    try:
        orchestrator = await get_orchestrator()
        
        # Check if this is the current orchestration
        if (orchestrator.current_orchestration and 
            orchestrator.current_orchestration.orchestration_id == orchestration_id):
            
            if orchestrator.current_orchestration.overall_status == TestStatus.RUNNING:
                raise HTTPException(
                    status_code=status.HTTP_202_ACCEPTED,
                    detail="Orchestration is still running. Check status endpoint for progress."
                )
            
            results = orchestrator.current_orchestration.to_dict()
        else:
            # Try to load from storage
            results = await orchestrator._load_orchestration_results(orchestration_id)
            if not results:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Results for orchestration {orchestration_id} not found"
                )
        
        # Filter detailed metrics if not requested
        if not include_detailed_metrics:
            # Remove detailed metrics from test results
            for test_result in results.get("test_results", []):
                if "metrics" in test_result and isinstance(test_result["metrics"], dict):
                    # Keep only summary metrics
                    test_result["metrics"] = {
                        k: v for k, v in test_result["metrics"].items()
                        if k in ["overall_status", "performance_score", "targets_validation"]
                    }
        
        return {
            "orchestration_id": orchestration_id,
            "results": results,
            "retrieved_at": datetime.utcnow().isoformat(),
            "includes_detailed_metrics": include_detailed_metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get orchestration results: {str(e)}"
        )


@router.get("/targets", response_model=PerformanceTargetsResponse)
async def get_performance_targets(
    category_filter: Optional[TestCategory] = Query(None, description="Filter by test category"),
    critical_only: bool = Query(False, description="Return only critical targets"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get all defined performance targets and their specifications.
    
    Returns comprehensive information about performance targets,
    including thresholds, criticality, and validation criteria.
    """
    try:
        orchestrator = await get_orchestrator()
        
        targets = orchestrator.performance_targets
        
        # Apply filters
        if category_filter:
            targets = [t for t in targets if t.category == category_filter]
        
        if critical_only:
            targets = [t for t in targets if t.critical]
        
        # Convert to response format
        targets_data = []
        for target in targets:
            targets_data.append({
                "name": target.name,
                "category": target.category.value,
                "target_value": target.target_value,
                "unit": target.unit,
                "description": target.description,
                "critical": target.critical,
                "tolerance_percent": target.tolerance_percent
            })
        
        categories = list(set(t.category.value for t in orchestrator.performance_targets))
        critical_count = len([t for t in orchestrator.performance_targets if t.critical])
        
        return PerformanceTargetsResponse(
            targets=targets_data,
            categories=categories,
            critical_targets_count=critical_count,
            total_targets_count=len(orchestrator.performance_targets)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance targets: {str(e)}"
        )


@router.post("/orchestration/{orchestration_id}/cancel")
async def cancel_orchestration(
    orchestration_id: str = Path(..., description="Orchestration ID"),
    current_user: dict = Depends(get_current_user)
):
    """
    Cancel a running performance testing orchestration.
    
    Gracefully stops all running tests and provides partial results
    for completed test components.
    """
    try:
        orchestrator = await get_orchestrator()
        
        # Check if this is the current orchestration
        if (not orchestrator.current_orchestration or 
            orchestrator.current_orchestration.orchestration_id != orchestration_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Active orchestration {orchestration_id} not found"
            )
        
        if orchestrator.current_orchestration.overall_status != TestStatus.RUNNING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Orchestration {orchestration_id} is not running"
            )
        
        # Cancel running tests
        for test_id, task in orchestrator.running_tests.items():
            task.cancel()
        
        # Update orchestration status
        orchestrator.current_orchestration.overall_status = TestStatus.CANCELLED
        orchestrator.current_orchestration.end_time = datetime.utcnow()
        orchestrator.current_orchestration.duration_seconds = (
            orchestrator.current_orchestration.end_time - 
            orchestrator.current_orchestration.start_time
        ).total_seconds()
        
        # Store partial results
        await orchestrator._store_orchestration_results()
        
        return {
            "orchestration_id": orchestration_id,
            "status": "cancelled",
            "message": "Orchestration cancelled successfully",
            "cancelled_at": datetime.utcnow().isoformat(),
            "partial_results_available": len(orchestrator.current_orchestration.test_results) > 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel orchestration: {str(e)}"
        )


@router.get("/history", response_model=Dict[str, Any])
async def get_orchestration_history(
    limit: int = Query(50, ge=1, le=200, description="Maximum results to return"),
    category_filter: Optional[TestCategory] = Query(None, description="Filter by test category"),
    status_filter: Optional[TestStatus] = Query(None, description="Filter by status"),
    days_back: int = Query(30, ge=1, le=365, description="Days of history to include"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get historical performance testing orchestration results.
    
    Provides access to historical test results for trend analysis,
    baseline comparison, and performance regression tracking.
    """
    try:
        # This would typically query a database of stored orchestrations
        # For now, return a placeholder structure
        
        history = {
            "orchestrations": [],
            "summary": {
                "total_orchestrations": 0,
                "success_rate": 0.0,
                "average_performance_score": 0.0,
                "trend_analysis": {
                    "performance_trending": "stable",
                    "recent_regressions": 0,
                    "improvements_detected": 0
                }
            },
            "filters_applied": {
                "limit": limit,
                "category_filter": category_filter.value if category_filter else None,
                "status_filter": status_filter.value if status_filter else None,
                "days_back": days_back
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return history
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get orchestration history: {str(e)}"
        )


@router.get("/metrics/realtime", response_model=Dict[str, Any])
async def get_realtime_metrics(
    current_user: dict = Depends(get_current_user)
):
    """
    Get real-time performance metrics from the system.
    
    Provides live system performance data including resource utilization,
    throughput metrics, and current system health indicators.
    """
    try:
        orchestrator = await get_orchestrator()
        
        # Collect real-time metrics
        sample = await orchestrator._collect_performance_sample()
        
        # Add system health indicators
        realtime_metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_resources": sample,
            "performance_indicators": {
                "system_healthy": sample.get("cpu_percent", 0) < 80 and sample.get("memory_percent", 0) < 85,
                "performance_status": "optimal" if sample.get("cpu_percent", 0) < 50 else "degraded",
                "resource_utilization": "normal" if sample.get("memory_percent", 0) < 75 else "high"
            },
            "active_testing": {
                "orchestration_running": orchestrator.current_orchestration is not None,
                "current_orchestration_id": (
                    orchestrator.current_orchestration.orchestration_id 
                    if orchestrator.current_orchestration else None
                ),
                "running_tests": len(orchestrator.running_tests)
            }
        }
        
        return realtime_metrics
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get realtime metrics: {str(e)}"
        )


@router.post("/configuration/validate")
async def validate_configuration(
    config: OrchestrationConfigRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Validate a performance testing configuration.
    
    Checks configuration parameters for validity, estimates resource
    requirements, and provides recommendations for optimization.
    """
    try:
        # Validate configuration parameters
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": [],
            "estimated_duration_minutes": 0,
            "estimated_resource_usage": {}
        }
        
        # Validate test combinations
        enabled_tests = [
            config.enable_context_engine_tests,
            config.enable_redis_streams_tests,
            config.enable_vertical_slice_tests,
            config.enable_system_integration_tests
        ]
        
        if not any(enabled_tests):
            validation_results["errors"].append("At least one test category must be enabled")
            validation_results["valid"] = False
        
        # Estimate duration
        duration_estimate = 0
        if config.enable_context_engine_tests:
            duration_estimate += config.context_engine_iterations * 2  # ~2 minutes per iteration
        if config.enable_redis_streams_tests:
            duration_estimate += config.redis_streams_duration_minutes + 5  # +5 for setup/teardown
        if config.enable_vertical_slice_tests:
            duration_estimate += config.vertical_slice_scenarios * 3  # ~3 minutes per scenario
        if config.enable_system_integration_tests:
            duration_estimate += config.integration_test_scenarios * 5  # ~5 minutes per scenario
        
        validation_results["estimated_duration_minutes"] = duration_estimate
        
        # Add recommendations
        if duration_estimate > 60:
            validation_results["warnings"].append(
                f"Estimated duration ({duration_estimate} minutes) exceeds 1 hour"
            )
            validation_results["recommendations"].append(
                "Consider reducing test iterations or enabling parallel execution"
            )
        
        if config.max_concurrent_tests > 5:
            validation_results["warnings"].append(
                "High concurrency may impact system performance during testing"
            )
        
        return {
            "configuration": config.model_dump(),
            "validation": validation_results,
            "validated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate configuration: {str(e)}"
        )


# Health and Status Endpoints

@router.get("/health")
async def get_performance_system_health():
    """
    Get performance testing system health status.
    
    Returns health information for all performance testing components
    and their current operational status.
    """
    try:
        orchestrator = await get_orchestrator()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "performance_orchestrator": {
                    "status": "healthy",
                    "initialized": orchestrator is not None,
                    "current_orchestration": orchestrator.current_orchestration is not None
                },
                "context_engine_tests": {
                    "status": "healthy" if orchestrator.benchmark_suite else "not_initialized",
                    "available": orchestrator.config.enable_context_engine_tests
                },
                "redis_streams_tests": {
                    "status": "healthy" if orchestrator.load_test_framework else "not_initialized",
                    "available": orchestrator.config.enable_redis_streams_tests
                },
                "vertical_slice_tests": {
                    "status": "healthy" if orchestrator.performance_validator else "not_initialized",
                    "available": orchestrator.config.enable_vertical_slice_tests
                }
            },
            "system_ready": True,
            "performance_targets_loaded": len(orchestrator.performance_targets) > 0
        }
        
        # Check for any unhealthy components
        unhealthy_components = [
            name for name, component in health_status["components"].items()
            if component["status"] != "healthy"
        ]
        
        if unhealthy_components:
            health_status["status"] = "degraded"
            health_status["unhealthy_components"] = unhealthy_components
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }