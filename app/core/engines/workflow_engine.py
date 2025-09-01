"""
WorkflowEngine - Consolidated Workflow Orchestration for LeanVibe Agent Hive 2.0

Consolidates 8+ workflow implementations into a single, high-performance DAG engine:
- workflow_engine.py (1,960 LOC) - Core DAG workflow
- enhanced_workflow_engine.py (906 LOC) - Advanced features  
- advanced_orchestration_engine.py (761 LOC) - Orchestration
- workflow_engine_error_handling.py (904 LOC) - Error handling
- strategic_implementation_engine.py (1,017 LOC) - Strategic planning

Performance Targets:
- <2s workflow compilation for complex DAGs
- Parallel execution optimization
- Real-time dependency resolution  
- Checkpoint-based recovery
"""

import asyncio
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .base_engine import BaseEngine, EngineConfig, EngineRequest, EngineResponse

# Import production workflow engine for consolidation
try:
    from ..workflow_engine import WorkflowEngine as ProductionWorkflowEngine
    from ..workflow_engine import WorkflowResult, TaskResult, ExecutionPlan, ExecutionMode
    PRODUCTION_ENGINE_AVAILABLE = True
    logger.info("Production workflow engine available for consolidation")
except ImportError:
    PRODUCTION_ENGINE_AVAILABLE = False
    logger.warning("Production workflow engine not available - using placeholder")


class WorkflowRequestType(str, Enum):
    """Workflow engine request types."""
    EXECUTE_WORKFLOW = "execute_workflow"
    EXECUTE_DAG_WORKFLOW = "execute_dag_workflow" 
    EXECUTE_SEMANTIC_WORKFLOW = "execute_semantic_workflow"
    PAUSE_WORKFLOW = "pause_workflow"
    CANCEL_WORKFLOW = "cancel_workflow"
    GET_WORKFLOW_STATUS = "get_workflow_status"
    GET_CRITICAL_PATH = "get_critical_path"
    OPTIMIZE_WORKFLOW = "optimize_workflow"
    ADD_TASK_TO_WORKFLOW = "add_task_to_workflow"
    REMOVE_TASK_FROM_WORKFLOW = "remove_task_from_workflow"


@dataclass
class ConsolidatedWorkflowResult:
    """Consolidated workflow execution result."""
    workflow_id: str
    success: bool
    status: str
    execution_time_ms: float
    completed_tasks: int
    failed_tasks: int
    total_tasks: int
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class WorkflowEngine(BaseEngine):
    """
    Consolidated Workflow Engine - Advanced DAG-based orchestration.
    
    Consolidates multiple workflow implementations into a single, high-performance engine
    with support for:
    - DAG dependency resolution and parallel execution
    - Semantic memory integration
    - Dynamic workflow modification  
    - Checkpoint-based recovery
    - Critical path analysis and optimization
    """
    
    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.production_engine: Optional[ProductionWorkflowEngine] = None
        self.active_workflows: Dict[str, asyncio.Task] = {}
        
        # Configuration from engine config
        orchestrator_config = config.custom_settings.get("orchestrator")
        agent_registry_config = config.custom_settings.get("agent_registry") 
        communication_config = config.custom_settings.get("communication_service")
        semantic_memory_enabled = config.custom_settings.get("enable_semantic_memory", True)
        
        # Performance configuration
        self.performance_targets = config.custom_settings.get("performance_targets", {
            "workflow_compilation_ms": 2000,
            "dependency_resolution_ms": 500,
            "batch_execution_latency_ms": 100
        })
        
    async def _engine_initialize(self) -> None:
        """Initialize consolidated workflow engine."""
        logger.info("Initializing ConsolidatedWorkflowEngine")
        
        if PRODUCTION_ENGINE_AVAILABLE:
            # Initialize production workflow engine for delegation
            orchestrator = self.config.custom_settings.get("orchestrator")
            agent_registry = self.config.custom_settings.get("agent_registry")
            communication_service = self.config.custom_settings.get("communication_service")
            enable_semantic_memory = self.config.custom_settings.get("enable_semantic_memory", True)
            
            self.production_engine = ProductionWorkflowEngine(
                orchestrator=orchestrator,
                agent_registry=agent_registry,
                communication_service=communication_service,
                enable_semantic_memory=enable_semantic_memory
            )
            
            await self.production_engine.initialize()
            logger.info("✅ Production workflow engine initialized and integrated")
        else:
            logger.warning("⚠️ Production workflow engine not available - using mock implementation")
    
    async def _engine_process(self, request: EngineRequest) -> EngineResponse:
        """Process consolidated workflow request."""
        request_type = request.request_type
        
        try:
            if request_type == WorkflowRequestType.EXECUTE_WORKFLOW:
                return await self._handle_execute_workflow(request)
            elif request_type == WorkflowRequestType.EXECUTE_DAG_WORKFLOW:
                return await self._handle_execute_dag_workflow(request)
            elif request_type == WorkflowRequestType.EXECUTE_SEMANTIC_WORKFLOW:
                return await self._handle_execute_semantic_workflow(request)
            elif request_type == WorkflowRequestType.PAUSE_WORKFLOW:
                return await self._handle_pause_workflow(request)
            elif request_type == WorkflowRequestType.CANCEL_WORKFLOW:
                return await self._handle_cancel_workflow(request)
            elif request_type == WorkflowRequestType.GET_WORKFLOW_STATUS:
                return await self._handle_get_workflow_status(request)
            elif request_type == WorkflowRequestType.GET_CRITICAL_PATH:
                return await self._handle_get_critical_path(request)
            elif request_type == WorkflowRequestType.OPTIMIZE_WORKFLOW:
                return await self._handle_optimize_workflow(request)
            elif request_type == WorkflowRequestType.ADD_TASK_TO_WORKFLOW:
                return await self._handle_add_task_to_workflow(request)
            elif request_type == WorkflowRequestType.REMOVE_TASK_FROM_WORKFLOW:
                return await self._handle_remove_task_from_workflow(request)
            else:
                return EngineResponse(
                    request_id=request.request_id,
                    success=False,
                    error=f"Unknown workflow request type: {request_type}",
                    error_code="UNKNOWN_REQUEST_TYPE"
                )
                
        except Exception as e:
            logger.error(f"Error processing workflow request: {e}")
            return EngineResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
                error_code="WORKFLOW_PROCESSING_ERROR"
            )
    
    async def _handle_execute_workflow(self, request: EngineRequest) -> EngineResponse:
        """Handle standard workflow execution."""
        workflow_id = request.payload.get("workflow_id")
        if not workflow_id:
            return EngineResponse(
                request_id=request.request_id,
                success=False,
                error="workflow_id is required",
                error_code="MISSING_WORKFLOW_ID"
            )
        
        start_time = time.time()
        
        try:
            if self.production_engine:
                # Delegate to production engine
                result = await self.production_engine.execute_workflow(workflow_id)
                
                consolidated_result = self._convert_to_consolidated_result(result)
                processing_time_ms = (time.time() - start_time) * 1000
                
                return EngineResponse(
                    request_id=request.request_id,
                    success=consolidated_result.success,
                    result={
                        "workflow_id": consolidated_result.workflow_id,
                        "status": consolidated_result.status,
                        "execution_time_ms": consolidated_result.execution_time_ms,
                        "completed_tasks": consolidated_result.completed_tasks,
                        "failed_tasks": consolidated_result.failed_tasks,
                        "total_tasks": consolidated_result.total_tasks
                    },
                    error=consolidated_result.error,
                    processing_time_ms=processing_time_ms
                )
            else:
                # Mock implementation
                await asyncio.sleep(0.1)  # Simulate processing
                processing_time_ms = (time.time() - start_time) * 1000
                
                return EngineResponse(
                    request_id=request.request_id,
                    success=True,
                    result={
                        "workflow_id": workflow_id,
                        "status": "completed",
                        "execution_time_ms": processing_time_ms,
                        "completed_tasks": 5,
                        "failed_tasks": 0,
                        "total_tasks": 5,
                        "mock": True
                    },
                    processing_time_ms=processing_time_ms
                )
                
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            return EngineResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
                error_code="WORKFLOW_EXECUTION_ERROR",
                processing_time_ms=processing_time_ms
            )
    
    async def _handle_execute_dag_workflow(self, request: EngineRequest) -> EngineResponse:
        """Handle DAG workflow execution with advanced features."""
        workflow_id = request.payload.get("workflow_id")
        execution_strategy = request.payload.get("execution_strategy", "adaptive")
        max_parallel_tasks = request.payload.get("max_parallel_tasks", 20)
        enable_recovery = request.payload.get("enable_recovery", True)
        
        if not workflow_id:
            return EngineResponse(
                request_id=request.request_id,
                success=False,
                error="workflow_id is required",
                error_code="MISSING_WORKFLOW_ID"
            )
        
        start_time = time.time()
        
        try:
            if self.production_engine:
                from ..task_batch_executor import BatchExecutionStrategy
                
                # Map string to enum
                strategy_mapping = {
                    "adaptive": BatchExecutionStrategy.ADAPTIVE,
                    "parallel": BatchExecutionStrategy.PARALLEL_OPTIMIZED,
                    "sequential": BatchExecutionStrategy.SEQUENTIAL,
                    "resource_aware": BatchExecutionStrategy.RESOURCE_AWARE
                }
                strategy = strategy_mapping.get(execution_strategy, BatchExecutionStrategy.ADAPTIVE)
                
                # Delegate to production engine
                result = await self.production_engine.execute_workflow_with_dag(
                    workflow_id=workflow_id,
                    execution_strategy=strategy,
                    max_parallel_tasks=max_parallel_tasks,
                    enable_recovery=enable_recovery
                )
                
                consolidated_result = self._convert_to_consolidated_result(result)
                processing_time_ms = (time.time() - start_time) * 1000
                
                return EngineResponse(
                    request_id=request.request_id,
                    success=consolidated_result.success,
                    result={
                        "workflow_id": consolidated_result.workflow_id,
                        "status": consolidated_result.status,
                        "execution_time_ms": consolidated_result.execution_time_ms,
                        "completed_tasks": consolidated_result.completed_tasks,
                        "failed_tasks": consolidated_result.failed_tasks,
                        "total_tasks": consolidated_result.total_tasks,
                        "dag_features": {
                            "execution_strategy": execution_strategy,
                            "max_parallel_tasks": max_parallel_tasks,
                            "recovery_enabled": enable_recovery
                        }
                    },
                    error=consolidated_result.error,
                    processing_time_ms=processing_time_ms
                )
            else:
                # Enhanced mock implementation for DAG features
                await asyncio.sleep(0.2)  # Simulate DAG processing
                processing_time_ms = (time.time() - start_time) * 1000
                
                return EngineResponse(
                    request_id=request.request_id,
                    success=True,
                    result={
                        "workflow_id": workflow_id,
                        "status": "completed",
                        "execution_time_ms": processing_time_ms,
                        "completed_tasks": 8,
                        "failed_tasks": 0,
                        "total_tasks": 8,
                        "dag_features": {
                            "execution_strategy": execution_strategy,
                            "max_parallel_tasks": max_parallel_tasks,
                            "recovery_enabled": enable_recovery,
                            "critical_path_duration_ms": 1200,
                            "parallelization_efficiency": 0.85
                        },
                        "mock": True
                    },
                    processing_time_ms=processing_time_ms
                )
                
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            return EngineResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
                error_code="DAG_WORKFLOW_EXECUTION_ERROR",
                processing_time_ms=processing_time_ms
            )
    
    async def _handle_execute_semantic_workflow(self, request: EngineRequest) -> EngineResponse:
        """Handle semantic workflow execution with memory integration."""
        workflow_id = request.payload.get("workflow_id")
        agent_id = request.payload.get("agent_id")
        context_data = request.payload.get("context_data", {})
        enable_context_injection = request.payload.get("enable_context_injection", True)
        enable_knowledge_learning = request.payload.get("enable_knowledge_learning", True)
        
        if not workflow_id or not agent_id:
            return EngineResponse(
                request_id=request.request_id,
                success=False,
                error="workflow_id and agent_id are required",
                error_code="MISSING_REQUIRED_FIELDS"
            )
        
        start_time = time.time()
        
        try:
            if self.production_engine:
                # Delegate to production engine
                result = await self.production_engine.execute_semantic_workflow(
                    workflow_id=workflow_id,
                    agent_id=agent_id,
                    context_data=context_data,
                    enable_context_injection=enable_context_injection,
                    enable_knowledge_learning=enable_knowledge_learning
                )
                
                consolidated_result = self._convert_to_consolidated_result(result)
                processing_time_ms = (time.time() - start_time) * 1000
                
                return EngineResponse(
                    request_id=request.request_id,
                    success=consolidated_result.success,
                    result={
                        "workflow_id": consolidated_result.workflow_id,
                        "status": consolidated_result.status,
                        "execution_time_ms": consolidated_result.execution_time_ms,
                        "completed_tasks": consolidated_result.completed_tasks,
                        "failed_tasks": consolidated_result.failed_tasks,
                        "total_tasks": consolidated_result.total_tasks,
                        "semantic_features": {
                            "agent_id": agent_id,
                            "context_injection_enabled": enable_context_injection,
                            "knowledge_learning_enabled": enable_knowledge_learning,
                            "context_data_provided": bool(context_data)
                        }
                    },
                    error=consolidated_result.error,
                    processing_time_ms=processing_time_ms
                )
            else:
                # Mock semantic execution
                await asyncio.sleep(0.15)  # Simulate semantic processing
                processing_time_ms = (time.time() - start_time) * 1000
                
                return EngineResponse(
                    request_id=request.request_id,
                    success=True,
                    result={
                        "workflow_id": workflow_id,
                        "status": "completed",
                        "execution_time_ms": processing_time_ms,
                        "completed_tasks": 6,
                        "failed_tasks": 0,
                        "total_tasks": 6,
                        "semantic_features": {
                            "agent_id": agent_id,
                            "context_injection_enabled": enable_context_injection,
                            "knowledge_learning_enabled": enable_knowledge_learning,
                            "context_items_injected": 3,
                            "knowledge_items_learned": 2,
                            "context_retrieval_time_ms": 45
                        },
                        "mock": True
                    },
                    processing_time_ms=processing_time_ms
                )
                
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            return EngineResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
                error_code="SEMANTIC_WORKFLOW_EXECUTION_ERROR",
                processing_time_ms=processing_time_ms
            )
    
    async def _handle_pause_workflow(self, request: EngineRequest) -> EngineResponse:
        """Handle workflow pause request."""
        workflow_id = request.payload.get("workflow_id")
        
        if self.production_engine:
            success = await self.production_engine.pause_workflow(workflow_id)
            return EngineResponse(
                request_id=request.request_id,
                success=success,
                result={"workflow_id": workflow_id, "status": "paused" if success else "pause_failed"}
            )
        else:
            return EngineResponse(
                request_id=request.request_id,
                success=True,
                result={"workflow_id": workflow_id, "status": "paused", "mock": True}
            )
    
    async def _handle_cancel_workflow(self, request: EngineRequest) -> EngineResponse:
        """Handle workflow cancellation request."""
        workflow_id = request.payload.get("workflow_id")
        reason = request.payload.get("reason", "User requested cancellation")
        
        if self.production_engine:
            success = await self.production_engine.cancel_workflow(workflow_id, reason)
            return EngineResponse(
                request_id=request.request_id,
                success=success,
                result={"workflow_id": workflow_id, "status": "cancelled" if success else "cancel_failed", "reason": reason}
            )
        else:
            return EngineResponse(
                request_id=request.request_id,
                success=True,
                result={"workflow_id": workflow_id, "status": "cancelled", "reason": reason, "mock": True}
            )
    
    async def _handle_get_workflow_status(self, request: EngineRequest) -> EngineResponse:
        """Handle workflow status request."""
        workflow_id = request.payload.get("workflow_id")
        
        if self.production_engine:
            status = await self.production_engine.get_execution_status(workflow_id)
            return EngineResponse(
                request_id=request.request_id,
                success=True,
                result=status
            )
        else:
            return EngineResponse(
                request_id=request.request_id,
                success=True,
                result={
                    "workflow_id": workflow_id,
                    "is_active": False,
                    "status": "completed",
                    "mock": True
                }
            )
    
    async def _handle_get_critical_path(self, request: EngineRequest) -> EngineResponse:
        """Handle critical path analysis request."""
        workflow_id = request.payload.get("workflow_id")
        
        if self.production_engine:
            critical_path = await self.production_engine.get_workflow_critical_path(workflow_id)
            if critical_path:
                return EngineResponse(
                    request_id=request.request_id,
                    success=True,
                    result={
                        "workflow_id": workflow_id,
                        "critical_path": {
                            "total_duration_ms": critical_path.total_duration,
                            "task_count": len(critical_path.task_sequence),
                            "bottleneck_tasks": critical_path.bottleneck_tasks,
                            "optimization_opportunities": critical_path.optimization_opportunities
                        }
                    }
                )
            else:
                return EngineResponse(
                    request_id=request.request_id,
                    success=False,
                    error="Critical path not available for workflow",
                    error_code="CRITICAL_PATH_NOT_FOUND"
                )
        else:
            return EngineResponse(
                request_id=request.request_id,
                success=True,
                result={
                    "workflow_id": workflow_id,
                    "critical_path": {
                        "total_duration_ms": 1500,
                        "task_count": 4,
                        "bottleneck_tasks": ["task_3", "task_7"],
                        "optimization_opportunities": ["parallel_task_2_and_4", "optimize_task_3_processing"]
                    },
                    "mock": True
                }
            )
    
    async def _handle_optimize_workflow(self, request: EngineRequest) -> EngineResponse:
        """Handle workflow optimization request."""
        workflow_id = request.payload.get("workflow_id")
        
        if self.production_engine:
            optimization_result = await self.production_engine.optimize_workflow_execution(workflow_id)
            return EngineResponse(
                request_id=request.request_id,
                success=True,
                result=optimization_result
            )
        else:
            return EngineResponse(
                request_id=request.request_id,
                success=True,
                result={
                    "workflow_id": workflow_id,
                    "optimizations": {
                        "parallelization_improvement": "25%",
                        "execution_time_reduction": "18%",
                        "resource_utilization_improvement": "30%"
                    },
                    "recommendations": [
                        "Enable parallel execution for tasks 2-4",
                        "Increase resource allocation for bottleneck task 3",
                        "Cache intermediate results for task 7"
                    ],
                    "mock": True
                }
            )
    
    async def _handle_add_task_to_workflow(self, request: EngineRequest) -> EngineResponse:
        """Handle dynamic task addition request."""
        workflow_id = request.payload.get("workflow_id")
        task_id = request.payload.get("task_id")
        dependencies = request.payload.get("dependencies", [])
        
        if self.production_engine:
            success = await self.production_engine.add_task_to_workflow(workflow_id, task_id, dependencies)
            return EngineResponse(
                request_id=request.request_id,
                success=success,
                result={
                    "workflow_id": workflow_id,
                    "task_id": task_id,
                    "added": success,
                    "dependencies": dependencies
                }
            )
        else:
            return EngineResponse(
                request_id=request.request_id,
                success=True,
                result={
                    "workflow_id": workflow_id,
                    "task_id": task_id,
                    "added": True,
                    "dependencies": dependencies,
                    "mock": True
                }
            )
    
    async def _handle_remove_task_from_workflow(self, request: EngineRequest) -> EngineResponse:
        """Handle dynamic task removal request."""
        workflow_id = request.payload.get("workflow_id")
        task_id = request.payload.get("task_id")
        
        if self.production_engine:
            success = await self.production_engine.remove_task_from_workflow(workflow_id, task_id)
            return EngineResponse(
                request_id=request.request_id,
                success=success,
                result={
                    "workflow_id": workflow_id,
                    "task_id": task_id,
                    "removed": success
                }
            )
        else:
            return EngineResponse(
                request_id=request.request_id,
                success=True,
                result={
                    "workflow_id": workflow_id,
                    "task_id": task_id,
                    "removed": True,
                    "mock": True
                }
            )
    
    def _convert_to_consolidated_result(self, production_result: 'WorkflowResult') -> ConsolidatedWorkflowResult:
        """Convert production workflow result to consolidated format."""
        return ConsolidatedWorkflowResult(
            workflow_id=production_result.workflow_id,
            success=production_result.status.value in ["completed", "paused"],
            status=production_result.status.value,
            execution_time_ms=production_result.execution_time * 1000,
            completed_tasks=production_result.completed_tasks,
            failed_tasks=production_result.failed_tasks,
            total_tasks=production_result.total_tasks,
            error=production_result.error,
            metadata={
                "task_results_count": len(production_result.task_results),
                "consolidated_engine": True
            }
        )
    
    async def _engine_shutdown(self) -> None:
        """Shutdown consolidated workflow engine."""
        logger.info("Shutting down ConsolidatedWorkflowEngine")
        
        # Shutdown production engine if available
        if self.production_engine:
            await self.production_engine.cleanup()
        
        # Cancel active workflows
        for workflow_id, task in self.active_workflows.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled active workflow {workflow_id}")
        
        self.active_workflows.clear()
        logger.info("ConsolidatedWorkflowEngine shutdown complete")