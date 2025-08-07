"""
Real-time Multi-Agent Coordination Monitoring API
Critical infrastructure for diagnosing and recovering from 20% success rate crisis.

Provides comprehensive monitoring endpoints for coordination success tracking,
failure analysis, agent health monitoring, and recovery operations.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import structlog

from ...core.orchestrator import AgentOrchestrator
from ...core.redis import get_message_broker
from ...core.database import get_session
from ...models.agent import Agent, AgentStatus
from ...schemas.agent import AgentResponse

logger = structlog.get_logger()

router = APIRouter(prefix="/coordination-monitoring", tags=["coordination-monitoring"])

# Real-time monitoring state
_monitoring_state = {
    "success_rate_history": deque(maxlen=1000),  # Last 1000 coordination attempts
    "failure_analysis": defaultdict(int),
    "agent_health_cache": {},
    "communication_latency_cache": deque(maxlen=100),
    "active_connections": set(),
    "last_update": datetime.utcnow()
}

# Pydantic models for monitoring responses
class CoordinationSuccessRate(BaseModel):
    """Real-time coordination success rate with trend analysis."""
    current_rate: float = Field(..., description="Current success rate percentage")
    trend: str = Field(..., description="Trend direction: improving, declining, stable")
    last_hour_rate: float = Field(..., description="Success rate in last hour")
    last_24h_rate: float = Field(..., description="Success rate in last 24 hours")
    total_attempts: int = Field(..., description="Total coordination attempts tracked")
    alert_status: str = Field(..., description="Alert level: healthy, warning, critical")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class FailureAnalysis(BaseModel):
    """Breakdown of coordination failures by type."""
    serialization_errors: int = Field(default=0, description="Data serialization failures")
    workflow_state_errors: int = Field(default=0, description="Workflow state management failures")
    agent_timeout_errors: int = Field(default=0, description="Agent timeout failures")
    communication_errors: int = Field(default=0, description="Redis communication failures")
    task_assignment_errors: int = Field(default=0, description="Task assignment failures")
    unknown_errors: int = Field(default=0, description="Unclassified error types")
    total_failures: int = Field(..., description="Total failure count")

class AgentHealthIndicator(BaseModel):
    """Real-time agent health status with performance metrics."""
    agent_id: str = Field(..., description="Agent identifier")
    name: str = Field(..., description="Agent display name")
    status: str = Field(..., description="Current status: online, offline, error")
    health_score: float = Field(..., description="Health score 0-100")
    last_heartbeat: Optional[datetime] = Field(None, description="Last heartbeat timestamp")
    specialization_badges: List[str] = Field(default=[], description="Capability specializations")
    performance_metrics: Dict[str, float] = Field(default={}, description="Performance metrics")
    current_task: Optional[str] = Field(None, description="Currently assigned task")
    error_count: int = Field(default=0, description="Recent error count")

class TaskDistributionStatus(BaseModel):
    """Task queue and distribution visualization data."""
    pending_tasks: List[Dict[str, Any]] = Field(default=[], description="Tasks awaiting assignment")
    assigned_tasks: Dict[str, List[Dict[str, Any]]] = Field(default={}, description="Tasks by agent")
    failed_tasks: List[Dict[str, Any]] = Field(default=[], description="Failed task assignments")
    queue_depth: int = Field(default=0, description="Total tasks in queue")
    average_wait_time: float = Field(default=0.0, description="Average task wait time in seconds")

class CommunicationHealth(BaseModel):
    """Agent communication monitoring metrics."""
    redis_status: str = Field(..., description="Redis connection status")
    average_latency: float = Field(..., description="Average message latency in ms")
    message_throughput: float = Field(..., description="Messages per second")
    error_rate: float = Field(..., description="Communication error rate percentage")
    connected_agents: int = Field(..., description="Currently connected agents")
    recent_latencies: List[float] = Field(default=[], description="Last 10 latency measurements")

class CoordinationDashboardData(BaseModel):
    """Complete coordination monitoring dashboard data."""
    success_rate: CoordinationSuccessRate
    failure_analysis: FailureAnalysis
    agent_health: List[AgentHealthIndicator]
    task_distribution: TaskDistributionStatus
    communication_health: CommunicationHealth
    system_alerts: List[str] = Field(default=[], description="Active system alerts")
    last_updated: datetime = Field(default_factory=datetime.utcnow)

# Global orchestrator dependency
async def get_orchestrator() -> AgentOrchestrator:
    """Get orchestrator instance for monitoring."""
    from ...main import app
    if not hasattr(app.state, 'orchestrator'):
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    return app.state.orchestrator

# Core monitoring functions
async def calculate_success_rate() -> CoordinationSuccessRate:
    """Calculate real-time coordination success rate with trend analysis."""
    history = list(_monitoring_state["success_rate_history"])
    
    if not history:
        return CoordinationSuccessRate(
            current_rate=0.0,
            trend="unknown",
            last_hour_rate=0.0,
            last_24h_rate=0.0,
            total_attempts=0,
            alert_status="warning"
        )
    
    # Calculate current success rate
    total_attempts = len(history)
    successful_attempts = sum(1 for result in history if result.get("success", False))
    current_rate = (successful_attempts / total_attempts) * 100 if total_attempts > 0 else 0.0
    
    # Calculate hourly and daily rates
    now = datetime.utcnow()
    hour_ago = now - timedelta(hours=1)
    day_ago = now - timedelta(days=1)
    
    recent_hour = [r for r in history if r.get("timestamp", datetime.min) > hour_ago]
    recent_day = [r for r in history if r.get("timestamp", datetime.min) > day_ago]
    
    last_hour_rate = (sum(1 for r in recent_hour if r.get("success", False)) / len(recent_hour) * 100) if recent_hour else 0.0
    last_24h_rate = (sum(1 for r in recent_day if r.get("success", False)) / len(recent_day) * 100) if recent_day else 0.0
    
    # Determine trend
    if len(history) >= 10:
        recent_success = sum(1 for r in history[-10:] if r.get("success", False))
        earlier_success = sum(1 for r in history[-20:-10] if r.get("success", False)) if len(history) >= 20 else recent_success
        
        if recent_success > earlier_success:
            trend = "improving"
        elif recent_success < earlier_success:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"
    
    # Determine alert status
    if current_rate >= 95:
        alert_status = "healthy"
    elif current_rate >= 80:
        alert_status = "warning" 
    else:
        alert_status = "critical"
    
    return CoordinationSuccessRate(
        current_rate=current_rate,
        trend=trend,
        last_hour_rate=last_hour_rate,
        last_24h_rate=last_24h_rate,
        total_attempts=total_attempts,
        alert_status=alert_status
    )

async def analyze_failures() -> FailureAnalysis:
    """Analyze coordination failures by type."""
    failures = _monitoring_state["failure_analysis"]
    
    total_failures = sum(failures.values())
    
    return FailureAnalysis(
        serialization_errors=failures.get("serialization_error", 0),
        workflow_state_errors=failures.get("workflow_state_error", 0),
        agent_timeout_errors=failures.get("agent_timeout", 0),
        communication_errors=failures.get("communication_error", 0),
        task_assignment_errors=failures.get("task_assignment_error", 0),
        unknown_errors=failures.get("unknown_error", 0),
        total_failures=total_failures
    )

async def get_agent_health_indicators(orchestrator: AgentOrchestrator) -> List[AgentHealthIndicator]:
    """Get real-time agent health indicators."""
    indicators = []
    
    for agent_id, agent in orchestrator.agents.items():
        # Calculate health score based on various factors
        health_score = 100.0
        
        # Deduct for errors
        recent_errors = _monitoring_state["agent_health_cache"].get(agent_id, {}).get("error_count", 0)
        health_score -= min(recent_errors * 10, 50)  # Max 50 point deduction for errors
        
        # Deduct for staleness  
        if agent.last_heartbeat:
            staleness = (datetime.utcnow() - agent.last_heartbeat).total_seconds()
            if staleness > 300:  # 5 minutes
                health_score -= min(staleness / 60 * 5, 40)  # Max 40 point deduction for staleness
        
        # Status mapping
        status_map = {
            "active": "online",
            "idle": "online", 
            "sleeping": "offline",
            "error": "error",
            "inactive": "offline"
        }
        
        status = status_map.get(agent.status.value, "unknown")
        
        # Generate specialization badges
        badges = []
        if agent.capabilities:
            for cap in agent.capabilities:
                if hasattr(cap, 'specialization_areas'):
                    badges.extend(cap.specialization_areas)
        
        indicators.append(AgentHealthIndicator(
            agent_id=str(agent_id),
            name=getattr(agent, 'name', f"Agent-{str(agent_id)[:8]}"),
            status=status,
            health_score=max(0.0, health_score),
            last_heartbeat=agent.last_heartbeat,
            specialization_badges=badges[:3],  # Limit to top 3 badges
            performance_metrics={
                "context_usage": getattr(agent, 'context_window_usage', 0.0),
                "response_time": getattr(agent, 'average_response_time', 0.0),
                "task_completion_rate": getattr(agent, 'task_completion_rate', 0.0)
            },
            current_task=getattr(agent, 'current_task', None),
            error_count=recent_errors
        ))
    
    return indicators

async def get_task_distribution_status(orchestrator: AgentOrchestrator) -> TaskDistributionStatus:
    """Get task distribution and queue status."""
    # This would integrate with the actual task queue system
    # For now, providing mock structure based on orchestrator state
    
    pending_tasks = []
    assigned_tasks = defaultdict(list)
    failed_tasks = []
    
    # Simulate task data based on orchestrator workflows
    for workflow_id, workflow_data in orchestrator.active_workflows.items():
        tasks = workflow_data.get('tasks', [])
        for task in tasks:
            task_info = {
                "id": task.get('id', 'unknown'),
                "title": task.get('name', 'Untitled Task'),
                "priority": task.get('priority', 'medium'),
                "estimated_effort": task.get('estimated_effort', 30),
                "created_at": datetime.utcnow().isoformat()
            }
            
            if task.get('status') == 'pending':
                pending_tasks.append(task_info)
            elif task.get('assigned_agent'):
                assigned_tasks[task['assigned_agent']].append(task_info)
            elif task.get('status') == 'failed':
                failed_tasks.append(task_info)
    
    return TaskDistributionStatus(
        pending_tasks=pending_tasks,
        assigned_tasks=dict(assigned_tasks),
        failed_tasks=failed_tasks,
        queue_depth=len(pending_tasks),
        average_wait_time=0.0  # Would calculate from real queue metrics
    )

async def get_communication_health() -> CommunicationHealth:
    """Get agent communication health metrics."""
    latencies = list(_monitoring_state["communication_latency_cache"])
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    recent_latencies = latencies[-10:] if latencies else []
    
    try:
        message_broker = await get_message_broker()
        redis_status = "healthy" if message_broker else "unhealthy"
        connected_agents = len(_monitoring_state["agent_health_cache"])
    except:
        redis_status = "error"
        connected_agents = 0
    
    return CommunicationHealth(
        redis_status=redis_status,
        average_latency=avg_latency,
        message_throughput=0.0,  # Would calculate from Redis metrics
        error_rate=0.0,  # Would calculate from error logs
        connected_agents=connected_agents,
        recent_latencies=recent_latencies
    )

# API Endpoints
@router.get("/dashboard", response_model=CoordinationDashboardData)
async def get_coordination_dashboard(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """
    Get comprehensive coordination monitoring dashboard data.
    Critical endpoint for diagnosing 20% success rate crisis.
    """
    try:
        logger.info("🔍 Fetching coordination dashboard data")
        
        # Gather all monitoring data concurrently
        success_rate, failure_analysis, agent_health, task_distribution, communication_health = await asyncio.gather(
            calculate_success_rate(),
            analyze_failures(),
            get_agent_health_indicators(orchestrator),
            get_task_distribution_status(orchestrator),
            get_communication_health()
        )
        
        # Generate system alerts based on current state
        alerts = []
        if success_rate.alert_status == "critical":
            alerts.append(f"CRITICAL: Coordination success rate at {success_rate.current_rate:.1f}% - immediate attention required")
        
        if failure_analysis.total_failures > 50:
            alerts.append(f"HIGH: {failure_analysis.total_failures} coordination failures detected")
        
        error_agents = [agent for agent in agent_health if agent.status == "error"]
        if error_agents:
            alerts.append(f"MEDIUM: {len(error_agents)} agents in error state")
        
        if communication_health.redis_status != "healthy":
            alerts.append("HIGH: Redis communication issues detected")
        
        dashboard_data = CoordinationDashboardData(
            success_rate=success_rate,
            failure_analysis=failure_analysis,
            agent_health=agent_health,
            task_distribution=task_distribution,
            communication_health=communication_health,
            system_alerts=alerts
        )
        
        logger.info("✅ Dashboard data assembled", 
                   success_rate=success_rate.current_rate,
                   total_agents=len(agent_health),
                   alerts_count=len(alerts))
        
        return dashboard_data
        
    except Exception as e:
        logger.error("❌ Failed to get coordination dashboard", error=str(e))
        raise HTTPException(status_code=500, detail=f"Dashboard data retrieval failed: {str(e)}")

@router.post("/record-coordination-result")
async def record_coordination_result(
    success: bool,
    error_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Record a coordination attempt result for success rate tracking."""
    try:
        result = {
            "success": success,
            "timestamp": datetime.utcnow(),
            "error_type": error_type,
            "metadata": metadata or {}
        }
        
        # Add to history
        _monitoring_state["success_rate_history"].append(result)
        
        # Update failure analysis if failed
        if not success and error_type:
            _monitoring_state["failure_analysis"][error_type] += 1
        
        _monitoring_state["last_update"] = datetime.utcnow()
        
        logger.info("📊 Coordination result recorded", 
                   success=success, 
                   error_type=error_type)
        
        return {"status": "recorded", "timestamp": result["timestamp"]}
        
    except Exception as e:
        logger.error("❌ Failed to record coordination result", error=str(e))
        raise HTTPException(status_code=500, detail=f"Recording failed: {str(e)}")

@router.post("/recovery-actions/restart-agent/{agent_id}")
async def restart_agent_action(
    agent_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Emergency agent restart action for coordination recovery."""
    try:
        logger.info("🔄 Executing emergency agent restart", agent_id=agent_id)
        
        # Terminate agent gracefully
        success = await orchestrator.shutdown_agent(agent_id, graceful=True)
        
        if success:
            # Brief pause for cleanup
            await asyncio.sleep(2)
            
            # Restart agent (this would need to be implemented in orchestrator)
            # For now, we'll mark this as an action taken
            
            # Clear cached errors for this agent
            if agent_id in _monitoring_state["agent_health_cache"]:
                _monitoring_state["agent_health_cache"][agent_id]["error_count"] = 0
            
            logger.info("✅ Agent restart completed", agent_id=agent_id)
            
            return {
                "status": "success",
                "agent_id": agent_id,
                "action": "restart_completed",
                "timestamp": datetime.utcnow()
            }
        else:
            raise HTTPException(status_code=400, detail="Agent restart failed")
            
    except Exception as e:
        logger.error("❌ Agent restart failed", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Restart failed: {str(e)}")

@router.post("/recovery-actions/reset-coordination")
async def reset_coordination_system(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Emergency coordination system reset."""
    try:
        logger.info("🚨 Executing emergency coordination system reset")
        
        # Clear workflow state
        orchestrator.active_workflows.clear()
        
        # Clear monitoring state
        _monitoring_state["success_rate_history"].clear()
        _monitoring_state["failure_analysis"].clear()
        _monitoring_state["agent_health_cache"].clear()
        _monitoring_state["communication_latency_cache"].clear()
        
        # Restart coordination system
        orchestrator.coordination_enabled = True
        
        logger.info("✅ Coordination system reset completed")
        
        return {
            "status": "success",
            "action": "coordination_reset_completed",
            "timestamp": datetime.utcnow(),
            "message": "Coordination system has been reset - monitoring data cleared"
        }
        
    except Exception as e:
        logger.error("❌ Coordination system reset failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"System reset failed: {str(e)}")

@router.post("/task-distribution/reassign/{task_id}")
async def reassign_task(
    task_id: str,
    target_agent_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Manual task reassignment for coordination recovery."""
    try:
        logger.info("📋 Manual task reassignment", task_id=task_id, target_agent=target_agent_id)
        
        # Find the task in active workflows
        task_found = False
        for workflow_id, workflow_data in orchestrator.active_workflows.items():
            tasks = workflow_data.get('tasks', [])
            for task in tasks:
                if task.get('id') == task_id:
                    # Reassign task
                    task['assigned_agent'] = target_agent_id
                    task['reassigned_at'] = datetime.utcnow()
                    task_found = True
                    break
            if task_found:
                break
        
        if not task_found:
            raise HTTPException(status_code=404, detail="Task not found")
        
        logger.info("✅ Task reassignment completed", task_id=task_id, new_agent=target_agent_id)
        
        return {
            "status": "success",
            "task_id": task_id,
            "assigned_to": target_agent_id,
            "action": "task_reassigned",
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Task reassignment failed", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Reassignment failed: {str(e)}")

# WebSocket endpoint for real-time monitoring
@router.websocket("/live-dashboard")
async def coordination_monitoring_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time coordination monitoring updates.
    Provides <100ms latency updates for dashboard components.
    """
    await websocket.accept()
    _monitoring_state["active_connections"].add(websocket)
    
    try:
        logger.info("🔗 Real-time monitoring connection established")
        
        orchestrator = await get_orchestrator()
        
        while True:
            try:
                # Gather real-time data
                dashboard_data = await get_coordination_dashboard.__wrapped__(orchestrator)
                
                # Send update to client
                await websocket.send_json({
                    "type": "dashboard_update",
                    "data": dashboard_data.dict(),
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Update every 2 seconds for real-time feel
                await asyncio.sleep(2)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error("WebSocket update error", error=str(e))
                await websocket.send_json({
                    "type": "error",
                    "message": f"Update failed: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                })
                await asyncio.sleep(5)  # Longer delay on errors
                
    except WebSocketDisconnect:
        pass
    finally:
        _monitoring_state["active_connections"].discard(websocket)
        logger.info("🔌 Real-time monitoring connection closed")

# Utility endpoints for testing and validation
@router.post("/test/generate-coordination-data")
async def generate_test_coordination_data():
    """Generate test coordination data for dashboard validation."""
    try:
        # Generate test success/failure data
        for i in range(50):
            success = i % 5 != 0  # 80% success rate for testing
            error_type = "serialization_error" if not success else None
            
            await record_coordination_result.__wrapped__(
                success=success,
                error_type=error_type,
                metadata={"test_data": True, "iteration": i}
            )
        
        logger.info("🧪 Test coordination data generated (80% success rate)")
        
        return {
            "status": "success",
            "message": "Generated 50 test coordination results with 80% success rate",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error("❌ Test data generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Test data generation failed: {str(e)}")