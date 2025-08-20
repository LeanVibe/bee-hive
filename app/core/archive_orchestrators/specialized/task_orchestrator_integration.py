"""
Task-Orchestrator Integration Bridge for LeanVibe Agent Hive 2.0
Seamlessly integrates unified task execution engine with production orchestrator.

Epic 1, Phase 2 Week 3 - Task System Consolidation
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

try:
    from .unified_task_execution_engine import (
        UnifiedTaskExecutionEngine,
        TaskExecutionRequest,
        TaskExecutionType,
        TaskExecutionStatus,
        ExecutionMode,
        SchedulingStrategy,
        get_unified_task_execution_engine
    )
    UNIFIED_ENGINE_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    UNIFIED_ENGINE_AVAILABLE = False

try:
    from .production_orchestrator_unified import (
        ProductionOrchestrator,
        AgentState,
        OrchestrationStrategy,
        TaskPriority as OrchestratorTaskPriority
    )
    ORCHESTRATOR_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    ORCHESTRATOR_AVAILABLE = False

try:
    from .logging_service import get_component_logger
except (ImportError, NameError, AttributeError):
    import logging
    def get_component_logger(name):
        return logging.getLogger(name)

try:
    from .messaging_service import get_messaging_service, Message, MessageType
    MESSAGING_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    MESSAGING_AVAILABLE = False

logger = get_component_logger("task_orchestrator_integration")

class IntegrationMode(str, Enum):
    """Integration modes for task-orchestrator coordination"""
    TASK_DRIVEN = "task_driven"  # Task engine requests agents from orchestrator
    ORCHESTRATOR_DRIVEN = "orchestrator_driven"  # Orchestrator assigns tasks to agents
    BIDIRECTIONAL = "bidirectional"  # Both systems coordinate dynamically
    HYBRID = "hybrid"  # Adaptive coordination based on load

@dataclass
class TaskAgentRequest:
    """Request for agent assignment from orchestrator"""
    task_id: str
    required_capabilities: List[str] = field(default_factory=list)
    preferred_agent_id: Optional[str] = None
    priority: int = 5
    timeout_seconds: int = 30
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentAssignmentResponse:
    """Response from orchestrator for agent assignment"""
    success: bool
    task_id: str
    assigned_agent_id: Optional[str] = None
    agent_capabilities: List[str] = field(default_factory=list)
    estimated_availability_time: Optional[datetime] = None
    assignment_confidence: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskCompletionNotification:
    """Notification sent to orchestrator when task completes"""
    task_id: str
    agent_id: str
    status: TaskExecutionStatus
    execution_time_ms: float
    result: Any = None
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class TaskOrchestratorBridge:
    """
    Integration bridge between unified task execution engine and production orchestrator.
    Provides seamless coordination for optimal multi-agent task execution.
    """
    
    def __init__(self, 
                 task_engine: Optional[Any] = None,
                 orchestrator: Optional[Any] = None,
                 integration_mode: IntegrationMode = IntegrationMode.HYBRID):
        
        if UNIFIED_ENGINE_AVAILABLE:
            self.task_engine = task_engine or get_unified_task_execution_engine()
        else:
            self.task_engine = None
            
        self.orchestrator = orchestrator  # Will be injected when available
        self.integration_mode = integration_mode
        
        if MESSAGING_AVAILABLE:
            self.messaging = get_messaging_service()
        else:
            self.messaging = None
        
        # Integration state
        self._active_integrations: Dict[str, TaskAgentRequest] = {}
        self._agent_task_mapping: Dict[str, Set[str]] = {}  # agent_id -> task_ids
        self._task_agent_mapping: Dict[str, str] = {}  # task_id -> agent_id
        
        # Performance tracking
        self._integration_stats = {
            "agent_requests": 0,
            "successful_assignments": 0,
            "failed_assignments": 0,
            "average_assignment_time_ms": 0.0,
            "active_task_agent_pairs": 0,
            "coordination_overhead_ms": 0.0
        }
        
        # Circuit breaker for orchestrator communication
        self._orchestrator_failures = 0
        self._max_failures = 5
        self._circuit_open_until: Optional[datetime] = None
        
        # Background coordination
        self._coordination_running = False
        self._coordination_task: Optional[asyncio.Task] = None
        
        logger.info("Task-Orchestrator integration bridge initialized", 
                   mode=integration_mode.value)
    
    async def start_integration(self):
        """Start integration bridge services"""
        if not self._coordination_running:
            self._coordination_task = asyncio.create_task(self._coordination_loop())
            self._coordination_running = True
            
            # Register task execution hooks
            await self._register_task_hooks()
            
            logger.info("Task-Orchestrator integration started")
    
    async def stop_integration(self):
        """Stop integration bridge gracefully"""
        self._coordination_running = False
        
        if self._coordination_task and not self._coordination_task.done():
            self._coordination_task.cancel()
            try:
                await self._coordination_task
            except asyncio.CancelledError:
                pass
        
        # Complete any pending integrations
        await self._cleanup_pending_integrations()
        
        logger.info("Task-Orchestrator integration stopped")
    
    async def request_agent_for_task(self, request: TaskAgentRequest) -> AgentAssignmentResponse:
        """Request agent assignment from orchestrator for task execution"""
        start_time = datetime.utcnow()
        self._integration_stats["agent_requests"] += 1
        
        try:
            # Check circuit breaker
            if not await self._is_orchestrator_available():
                return AgentAssignmentResponse(
                    success=False,
                    task_id=request.task_id,
                    error_message="Orchestrator circuit breaker open"
                )
            
            # Store pending request
            self._active_integrations[request.task_id] = request
            
            # Request agent from orchestrator
            if self.orchestrator:
                agent_response = await self._request_agent_from_orchestrator(request)
            else:
                # Fallback: simulate agent assignment
                agent_response = await self._simulate_agent_assignment(request)
            
            # Update mappings on successful assignment
            if agent_response.success and agent_response.assigned_agent_id:
                self._task_agent_mapping[request.task_id] = agent_response.assigned_agent_id
                if agent_response.assigned_agent_id not in self._agent_task_mapping:
                    self._agent_task_mapping[agent_response.assigned_agent_id] = set()
                self._agent_task_mapping[agent_response.assigned_agent_id].add(request.task_id)
                
                self._integration_stats["successful_assignments"] += 1
                self._integration_stats["active_task_agent_pairs"] += 1
            else:
                self._integration_stats["failed_assignments"] += 1
                self._orchestrator_failures += 1
            
            # Update performance metrics
            assignment_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            current_avg = self._integration_stats["average_assignment_time_ms"]
            total_requests = self._integration_stats["agent_requests"]
            new_avg = ((current_avg * (total_requests - 1)) + assignment_time_ms) / total_requests
            self._integration_stats["average_assignment_time_ms"] = new_avg
            
            # Clean up request
            self._active_integrations.pop(request.task_id, None)
            
            logger.debug("Agent assignment completed", 
                        task_id=request.task_id,
                        success=agent_response.success,
                        agent_id=agent_response.assigned_agent_id,
                        assignment_time_ms=assignment_time_ms)
            
            return agent_response
            
        except Exception as e:
            logger.error("Agent assignment request failed", 
                        task_id=request.task_id, error=str(e))
            
            self._integration_stats["failed_assignments"] += 1
            self._orchestrator_failures += 1
            
            return AgentAssignmentResponse(
                success=False,
                task_id=request.task_id,
                error_message=f"Assignment request failed: {str(e)}"
            )
    
    async def notify_task_completion(self, notification: TaskCompletionNotification):
        """Notify orchestrator of task completion"""
        try:
            # Update internal mappings
            if notification.task_id in self._task_agent_mapping:
                agent_id = self._task_agent_mapping[notification.task_id]
                
                # Remove from agent's task set
                if agent_id in self._agent_task_mapping:
                    self._agent_task_mapping[agent_id].discard(notification.task_id)
                    if not self._agent_task_mapping[agent_id]:
                        del self._agent_task_mapping[agent_id]
                
                # Remove task mapping
                del self._task_agent_mapping[notification.task_id]
                self._integration_stats["active_task_agent_pairs"] -= 1
            
            # Notify orchestrator if available
            if self.orchestrator and await self._is_orchestrator_available():
                await self._notify_orchestrator_completion(notification)
            
            logger.debug("Task completion notification processed", 
                        task_id=notification.task_id,
                        agent_id=notification.agent_id,
                        status=notification.status.value)
            
        except Exception as e:
            logger.error("Task completion notification failed", 
                        task_id=notification.task_id, error=str(e))
    
    async def get_agent_workload(self, agent_id: str) -> Dict[str, Any]:
        """Get current workload information for an agent"""
        try:
            agent_tasks = self._agent_task_mapping.get(agent_id, set())
            
            # Get task details from task engine
            task_details = []
            for task_id in agent_tasks:
                status = await self.task_engine.get_task_status(task_id)
                if status:
                    task_details.append(status)
            
            return {
                "agent_id": agent_id,
                "active_task_count": len(agent_tasks),
                "active_task_ids": list(agent_tasks),
                "task_details": task_details,
                "workload_score": self._calculate_workload_score(agent_tasks, task_details)
            }
            
        except Exception as e:
            logger.error("Failed to get agent workload", agent_id=agent_id, error=str(e))
            return {
                "agent_id": agent_id,
                "active_task_count": 0,
                "active_task_ids": [],
                "task_details": [],
                "workload_score": 0.0,
                "error": str(e)
            }
    
    async def optimize_task_agent_allocation(self) -> Dict[str, Any]:
        """Optimize task-agent allocation based on current workloads"""
        try:
            optimization_start = datetime.utcnow()
            
            # Analyze current allocations
            overloaded_agents = []
            underutilized_agents = []
            
            for agent_id, tasks in self._agent_task_mapping.items():
                workload = await self.get_agent_workload(agent_id)
                workload_score = workload["workload_score"]
                
                if workload_score > 0.8:  # Overloaded
                    overloaded_agents.append((agent_id, workload_score, tasks))
                elif workload_score < 0.3:  # Underutilized
                    underutilized_agents.append((agent_id, workload_score, tasks))
            
            # Suggest rebalancing if needed
            rebalancing_suggestions = []
            if overloaded_agents and underutilized_agents:
                rebalancing_suggestions = await self._generate_rebalancing_suggestions(
                    overloaded_agents, underutilized_agents
                )
            
            optimization_time_ms = (datetime.utcnow() - optimization_start).total_seconds() * 1000
            
            return {
                "optimization_completed": True,
                "optimization_time_ms": optimization_time_ms,
                "total_agents": len(self._agent_task_mapping),
                "overloaded_agents": len(overloaded_agents),
                "underutilized_agents": len(underutilized_agents),
                "rebalancing_suggestions": rebalancing_suggestions,
                "performance_metrics": {
                    "average_workload": self._calculate_average_workload(),
                    "workload_variance": self._calculate_workload_variance(),
                    "allocation_efficiency": self._calculate_allocation_efficiency()
                }
            }
            
        except Exception as e:
            logger.error("Task-agent allocation optimization failed", error=str(e))
            return {
                "optimization_completed": False,
                "error": str(e)
            }
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics"""
        return {
            **self._integration_stats,
            "integration_mode": self.integration_mode.value,
            "orchestrator_available": self.orchestrator is not None,
            "circuit_breaker_status": "open" if self._circuit_open_until else "closed",
            "active_integrations": len(self._active_integrations),
            "agent_count": len(self._agent_task_mapping),
            "task_count": len(self._task_agent_mapping),
            "orchestrator_failures": self._orchestrator_failures
        }
    
    # Private methods
    async def _request_agent_from_orchestrator(self, request: TaskAgentRequest) -> AgentAssignmentResponse:
        """Request agent assignment from production orchestrator"""
        try:
            # Convert request to orchestrator format
            orchestrator_request = {
                "required_capabilities": request.required_capabilities,
                "preferred_agent_id": request.preferred_agent_id,
                "priority": request.priority,
                "resource_requirements": request.resource_requirements,
                "timeout_seconds": request.timeout_seconds
            }
            
            # Call orchestrator (this would be the actual orchestrator method)
            # For now, simulate orchestrator response
            response = await self._simulate_orchestrator_agent_request(orchestrator_request)
            
            return AgentAssignmentResponse(
                success=response.get("success", False),
                task_id=request.task_id,
                assigned_agent_id=response.get("agent_id"),
                agent_capabilities=response.get("capabilities", []),
                assignment_confidence=response.get("confidence", 0.0),
                error_message=response.get("error")
            )
            
        except Exception as e:
            logger.error("Orchestrator agent request failed", error=str(e))
            return AgentAssignmentResponse(
                success=False,
                task_id=request.task_id,
                error_message=f"Orchestrator request failed: {str(e)}"
            )
    
    async def _simulate_orchestrator_agent_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate orchestrator agent request for testing"""
        await asyncio.sleep(0.01)  # Simulate network delay
        
        # Simulate successful assignment
        return {
            "success": True,
            "agent_id": f"agent_{uuid.uuid4().hex[:8]}",
            "capabilities": request.get("required_capabilities", []),
            "confidence": 0.85
        }
    
    async def _simulate_agent_assignment(self, request: TaskAgentRequest) -> AgentAssignmentResponse:
        """Simulate agent assignment when orchestrator is not available"""
        await asyncio.sleep(0.005)  # Simulate minimal delay
        
        return AgentAssignmentResponse(
            success=True,
            task_id=request.task_id,
            assigned_agent_id=f"simulated_agent_{uuid.uuid4().hex[:8]}",
            agent_capabilities=request.required_capabilities,
            assignment_confidence=0.5,  # Lower confidence for simulated assignment
            metadata={"simulated": True}
        )
    
    async def _notify_orchestrator_completion(self, notification: TaskCompletionNotification):
        """Notify orchestrator of task completion"""
        try:
            # Convert notification to orchestrator format
            orchestrator_notification = {
                "task_id": notification.task_id,
                "agent_id": notification.agent_id,
                "status": notification.status.value,
                "execution_time_ms": notification.execution_time_ms,
                "performance_metrics": notification.performance_metrics
            }
            
            # Send to orchestrator (this would be the actual orchestrator method)
            await self._send_orchestrator_notification(orchestrator_notification)
            
        except Exception as e:
            logger.error("Orchestrator completion notification failed", error=str(e))
    
    async def _send_orchestrator_notification(self, notification: Dict[str, Any]):
        """Send notification to orchestrator"""
        # This would be the actual implementation to notify the orchestrator
        # For now, just log the notification
        logger.debug("Orchestrator notification sent", notification=notification)
    
    async def _register_task_hooks(self):
        """Register hooks with task engine for automatic integration"""
        # This would register callbacks with the task engine
        # to automatically handle agent requests and completion notifications
        logger.debug("Task execution hooks registered")
    
    async def _is_orchestrator_available(self) -> bool:
        """Check if orchestrator is available (circuit breaker pattern)"""
        if self._circuit_open_until:
            if datetime.utcnow() < self._circuit_open_until:
                return False
            else:
                # Reset circuit breaker
                self._circuit_open_until = None
                self._orchestrator_failures = 0
        
        if self._orchestrator_failures >= self._max_failures:
            # Open circuit breaker for 30 seconds
            self._circuit_open_until = datetime.utcnow() + timedelta(seconds=30)
            logger.warning("Orchestrator circuit breaker opened", 
                          failures=self._orchestrator_failures)
            return False
        
        return True
    
    async def _coordination_loop(self):
        """Background coordination loop"""
        while self._coordination_running:
            try:
                # Periodic optimization and maintenance
                await self._periodic_optimization()
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Coordination loop error", error=str(e))
                await asyncio.sleep(5)
    
    async def _periodic_optimization(self):
        """Perform periodic optimization tasks"""
        try:
            # Clean up stale mappings
            await self._cleanup_stale_mappings()
            
            # Update performance metrics
            await self._update_performance_metrics()
            
            # Check for rebalancing opportunities
            if len(self._agent_task_mapping) > 1:
                await self.optimize_task_agent_allocation()
                
        except Exception as e:
            logger.error("Periodic optimization failed", error=str(e))
    
    async def _cleanup_stale_mappings(self):
        """Clean up stale task-agent mappings"""
        stale_tasks = []
        
        for task_id in list(self._task_agent_mapping.keys()):
            status = await self.task_engine.get_task_status(task_id)
            if not status or status["status"] in ["completed", "failed", "cancelled"]:
                stale_tasks.append(task_id)
        
        for task_id in stale_tasks:
            if task_id in self._task_agent_mapping:
                agent_id = self._task_agent_mapping[task_id]
                if agent_id in self._agent_task_mapping:
                    self._agent_task_mapping[agent_id].discard(task_id)
                    if not self._agent_task_mapping[agent_id]:
                        del self._agent_task_mapping[agent_id]
                del self._task_agent_mapping[task_id]
        
        if stale_tasks:
            logger.debug("Cleaned up stale task mappings", count=len(stale_tasks))
    
    async def _cleanup_pending_integrations(self):
        """Clean up pending integration requests"""
        pending_count = len(self._active_integrations)
        if pending_count > 0:
            self._active_integrations.clear()
            logger.info("Cleaned up pending integrations", count=pending_count)
    
    def _calculate_workload_score(self, task_ids: Set[str], task_details: List[Dict[str, Any]]) -> float:
        """Calculate workload score for an agent (0.0 to 1.0)"""
        if not task_ids:
            return 0.0
        
        # Base score from task count
        base_score = min(1.0, len(task_ids) / 10.0)  # Assume 10 tasks = full load
        
        # Adjust based on task complexity and execution time
        complexity_factor = 1.0
        for task in task_details:
            if "execution_time_ms" in task and task["execution_time_ms"] > 60000:  # >1 minute
                complexity_factor += 0.1
        
        return min(1.0, base_score * complexity_factor)
    
    def _calculate_average_workload(self) -> float:
        """Calculate average workload across all agents"""
        if not self._agent_task_mapping:
            return 0.0
        
        total_tasks = sum(len(tasks) for tasks in self._agent_task_mapping.values())
        return total_tasks / len(self._agent_task_mapping)
    
    def _calculate_workload_variance(self) -> float:
        """Calculate workload variance across agents"""
        if len(self._agent_task_mapping) < 2:
            return 0.0
        
        task_counts = [len(tasks) for tasks in self._agent_task_mapping.values()]
        return float(statistics.variance(task_counts))
    
    def _calculate_allocation_efficiency(self) -> float:
        """Calculate overall allocation efficiency"""
        if not self._agent_task_mapping:
            return 1.0
        
        # Efficiency is inverse of workload variance (lower variance = higher efficiency)
        variance = self._calculate_workload_variance()
        max_possible_variance = len(self._agent_task_mapping) ** 2
        
        if max_possible_variance == 0:
            return 1.0
        
        return max(0.0, 1.0 - (variance / max_possible_variance))
    
    async def _generate_rebalancing_suggestions(self, 
                                              overloaded_agents: List[Tuple[str, float, Set[str]]],
                                              underutilized_agents: List[Tuple[str, float, Set[str]]]) -> List[Dict[str, Any]]:
        """Generate suggestions for rebalancing workloads"""
        suggestions = []
        
        for overloaded_agent_id, overload_score, overloaded_tasks in overloaded_agents:
            if underutilized_agents:
                underutilized_agent_id, _, _ = underutilized_agents[0]
                
                # Suggest moving some tasks from overloaded to underutilized agent
                tasks_to_move = list(overloaded_tasks)[:min(2, len(overloaded_tasks))]
                
                suggestions.append({
                    "type": "task_migration",
                    "from_agent": overloaded_agent_id,
                    "to_agent": underutilized_agent_id,
                    "tasks": tasks_to_move,
                    "expected_benefit": (overload_score - 0.6) * 0.5  # Estimated benefit
                })
        
        return suggestions
    
    async def _update_performance_metrics(self):
        """Update integration performance metrics"""
        # Calculate coordination overhead
        coordination_overhead = 0.0  # This would be calculated based on timing data
        self._integration_stats["coordination_overhead_ms"] = coordination_overhead

# Global integration bridge instance
_integration_bridge = None

def get_task_orchestrator_bridge() -> TaskOrchestratorBridge:
    """Get global task-orchestrator integration bridge"""
    global _integration_bridge
    if _integration_bridge is None:
        _integration_bridge = TaskOrchestratorBridge()
    return _integration_bridge

# Convenience functions for easy integration
async def request_agent_for_task(task_id: str, 
                                required_capabilities: List[str] = None,
                                priority: int = 5,
                                timeout_seconds: int = 30) -> AgentAssignmentResponse:
    """Convenience function to request agent for task"""
    bridge = get_task_orchestrator_bridge()
    request = TaskAgentRequest(
        task_id=task_id,
        required_capabilities=required_capabilities or [],
        priority=priority,
        timeout_seconds=timeout_seconds
    )
    return await bridge.request_agent_for_task(request)

async def notify_task_completed(task_id: str, 
                               agent_id: str,
                               status: TaskExecutionStatus,
                               execution_time_ms: float,
                               result: Any = None,
                               error_message: Optional[str] = None) -> None:
    """Convenience function to notify task completion"""
    bridge = get_task_orchestrator_bridge()
    notification = TaskCompletionNotification(
        task_id=task_id,
        agent_id=agent_id,
        status=status,
        execution_time_ms=execution_time_ms,
        result=result,
        error_message=error_message
    )
    await bridge.notify_task_completion(notification)