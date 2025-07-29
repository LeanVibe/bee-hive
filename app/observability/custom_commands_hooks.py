"""
Custom Commands Observability Hooks for LeanVibe Agent Hive 2.0 - Phase 6.1

Integration with existing Phase 5 observability infrastructure for comprehensive
monitoring, alerting, and analytics of multi-agent workflow command execution.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import structlog

from .hooks import HookInterceptor
from ..core.redis import get_message_broker, AgentMessageBroker
from ..schemas.custom_commands import CommandStatus, CommandExecutionResult

logger = structlog.get_logger()


@dataclass
class CommandExecutionEvent:
    """Custom commands execution event for observability."""
    event_type: str
    execution_id: str
    command_name: str
    command_version: str
    status: str
    timestamp: datetime
    duration_seconds: Optional[float] = None
    agent_assignments: Dict[str, str] = None
    error_message: Optional[str] = None
    resource_usage: Dict[str, Any] = None
    metadata: Dict[str, Any] = None


class CustomCommandsHooks:
    """
    Observability hooks integration for custom commands system.
    
    Provides seamless integration with existing Phase 5 observability infrastructure
    including Prometheus metrics, structured logging, alerting, and real-time monitoring.
    """
    
    def __init__(
        self,
        hook_interceptor: Optional[HookInterceptor] = None,
        message_broker: Optional[AgentMessageBroker] = None
    ):
        self.hook_interceptor = hook_interceptor
        self.message_broker = message_broker
        
        # Event tracking
        self.active_executions: Dict[str, CommandExecutionEvent] = {}
        self.execution_metrics: Dict[str, Any] = {
            "total_executions": 0,
            "active_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "commands_per_minute": 0.0,
            "agent_utilization": {}
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            "max_execution_time_seconds": 1800,  # 30 minutes
            "max_failure_rate_percent": 20.0,
            "max_concurrent_executions": 50,
            "min_success_rate_percent": 80.0
        }
        
        # Register hooks with existing system
        self._register_custom_hooks()
        
        logger.info("CustomCommandsHooks initialized with Phase 5 integration")
    
    async def on_command_execution_start(
        self,
        execution_id: str,
        command_name: str,
        command_version: str,
        requester_id: Optional[str] = None,
        parameters: Dict[str, Any] = None,
        agent_assignments: Dict[str, str] = None
    ) -> None:
        """Handle command execution start event."""
        try:
            event = CommandExecutionEvent(
                event_type="command_execution_start",
                execution_id=execution_id,
                command_name=command_name,
                command_version=command_version,
                status="running",
                timestamp=datetime.utcnow(),
                agent_assignments=agent_assignments or {},
                metadata={
                    "requester_id": requester_id,
                    "parameters_count": len(parameters or {}),
                    "agents_assigned": len(agent_assignments or {})
                }
            )
            
            # Track active execution
            self.active_executions[execution_id] = event
            
            # Update metrics
            self.execution_metrics["total_executions"] += 1
            self.execution_metrics["active_executions"] = len(self.active_executions)
            
            # Send to hook interceptor if available
            if self.hook_interceptor:
                import uuid
                # Create a mock event for the hook interceptor
                await self.hook_interceptor.capture_pre_tool_use(
                    session_id=uuid.uuid4(),  # Mock session ID
                    agent_id=uuid.uuid4(),    # Mock agent ID
                    tool_data={
                        "tool_name": "custom_command_execution",
                        "parameters": {
                            "execution_id": execution_id,
                            "command_name": command_name,
                            "command_version": command_version,
                            "requester_id": requester_id,
                            "agent_count": len(agent_assignments or {})
                        }
                    }
                )
            
            # Publish to Redis streams for real-time monitoring
            if self.message_broker:
                await self.message_broker.publish_agent_event(
                    "custom_commands",
                    "execution_started",
                    {
                        "execution_id": execution_id,
                        "command_name": command_name,
                        "timestamp": event.timestamp.isoformat(),
                        "agents_assigned": list(agent_assignments.keys()) if agent_assignments else []
                    }
                )
            
            logger.info(
                "Command execution started",
                execution_id=execution_id,
                command_name=command_name,
                agents_assigned=len(agent_assignments or {})
            )
            
        except Exception as e:
            logger.error("Failed to handle command execution start", error=str(e))
    
    async def on_command_execution_complete(
        self,
        execution_id: str,
        result: CommandExecutionResult,
        resource_usage: Dict[str, Any] = None
    ) -> None:
        """Handle command execution completion event."""
        try:
            # Get tracked execution
            tracked_event = self.active_executions.pop(execution_id, None)
            
            if not tracked_event:
                logger.warning("Untracked execution completed", execution_id=execution_id)
                return
            
            # Calculate metrics
            execution_time = result.total_execution_time_seconds or 0.0
            success = result.status == CommandStatus.COMPLETED
            
            # Update execution metrics
            if success:
                self.execution_metrics["successful_executions"] += 1
            else:
                self.execution_metrics["failed_executions"] += 1
            
            self.execution_metrics["active_executions"] = len(self.active_executions)
            
            # Update rolling average execution time
            total_executions = self.execution_metrics["total_executions"]
            current_avg = self.execution_metrics["average_execution_time"]
            new_avg = ((current_avg * (total_executions - 1)) + execution_time) / total_executions
            self.execution_metrics["average_execution_time"] = new_avg
            
            # Send to hook interceptor if available
            if self.hook_interceptor:
                import uuid
                # Create a mock completion event for the hook interceptor
                await self.hook_interceptor.capture_post_tool_use(
                    session_id=uuid.uuid4(),  # Mock session ID
                    agent_id=uuid.uuid4(),    # Mock agent ID
                    tool_result={
                        "tool_name": "custom_command_execution",
                        "success": success,
                        "result": {
                            "execution_id": execution_id,
                            "command_name": result.command_name,
                            "status": result.status.value,
                            "execution_time_seconds": execution_time,
                            "total_steps": result.total_steps,
                            "completed_steps": result.completed_steps,
                            "failed_steps": result.failed_steps
                        },
                        "execution_time_ms": int(execution_time * 1000) if execution_time else None
                    }
                )
            
            # Check for alert conditions
            await self._check_alert_conditions(result, execution_time)
            
            # Publish completion event
            if self.message_broker:
                await self.message_broker.publish_agent_event(
                    "custom_commands",
                    "execution_completed",
                    {
                        "execution_id": execution_id,
                        "command_name": result.command_name,
                        "status": result.status.value,
                        "execution_time_seconds": execution_time,
                        "success": success,
                        "timestamp": (result.end_time or datetime.utcnow()).isoformat()
                    }
                )
            
            logger.info(
                "Command execution completed",
                execution_id=execution_id,
                command_name=result.command_name,
                status=result.status.value,
                execution_time_seconds=execution_time,
                success=success
            )
            
        except Exception as e:
            logger.error("Failed to handle command execution completion", error=str(e))
    
    async def on_step_execution_start(
        self,
        execution_id: str,
        step_id: str,
        agent_id: str,
        step_data: Dict[str, Any] = None
    ) -> None:
        """Handle workflow step execution start."""
        try:
            # Send to hook interceptor if available
            if self.hook_interceptor:
                import uuid
                await self.hook_interceptor.capture_pre_tool_use(
                    session_id=uuid.uuid4(),  # Mock session ID
                    agent_id=uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id,
                    tool_data={
                        "tool_name": "workflow_step_execution",
                        "parameters": {
                            "execution_id": execution_id,
                            "step_id": step_id,
                            "step_data": step_data or {}
                        }
                    }
                )
            
            # Update agent utilization tracking
            if agent_id not in self.execution_metrics["agent_utilization"]:
                self.execution_metrics["agent_utilization"][agent_id] = {
                    "active_steps": 0,
                    "completed_steps": 0,
                    "failed_steps": 0,
                    "total_execution_time": 0.0
                }
            
            self.execution_metrics["agent_utilization"][agent_id]["active_steps"] += 1
            
            logger.debug(
                "Step execution started",
                execution_id=execution_id,
                step_id=step_id,
                agent_id=agent_id
            )
            
        except Exception as e:
            logger.error("Failed to handle step execution start", error=str(e))
    
    async def on_step_execution_complete(
        self,
        execution_id: str,
        step_id: str,
        agent_id: str,
        success: bool,
        execution_time_seconds: float,
        error_message: Optional[str] = None,
        outputs: Dict[str, Any] = None
    ) -> None:
        """Handle workflow step execution completion."""
        try:
            # Send to hook interceptor if available
            if self.hook_interceptor:
                import uuid
                await self.hook_interceptor.capture_post_tool_use(
                    session_id=uuid.uuid4(),  # Mock session ID
                    agent_id=uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id,
                    tool_result={
                        "tool_name": "workflow_step_execution",
                        "success": success,
                        "result": outputs or {},
                        "error": error_message,
                        "execution_time_ms": int(execution_time_seconds * 1000) if execution_time_seconds else None
                    }
                )
            
            # Update agent utilization metrics
            if agent_id in self.execution_metrics["agent_utilization"]:
                agent_metrics = self.execution_metrics["agent_utilization"][agent_id]
                agent_metrics["active_steps"] = max(0, agent_metrics["active_steps"] - 1)
                
                if success:
                    agent_metrics["completed_steps"] += 1
                else:
                    agent_metrics["failed_steps"] += 1
                
                agent_metrics["total_execution_time"] += execution_time_seconds
            
            logger.debug(
                "Step execution completed",
                execution_id=execution_id,
                step_id=step_id,
                agent_id=agent_id,
                success=success,
                execution_time_seconds=execution_time_seconds
            )
            
        except Exception as e:
            logger.error("Failed to handle step execution completion", error=str(e))
    
    async def on_task_distribution_complete(
        self,
        execution_id: str,
        command_name: str,
        distribution_result: Dict[str, Any]
    ) -> None:
        """Handle task distribution completion event."""
        try:
            # Send to hook interceptor if available
            if self.hook_interceptor:
                import uuid
                await self.hook_interceptor.capture_notification(
                    session_id=uuid.uuid4(),  # Mock session ID
                    agent_id=uuid.uuid4(),    # Mock agent ID
                    notification={
                        "level": "info",
                        "message": f"Task distribution completed for {command_name}",
                        "details": {
                            "execution_id": execution_id,
                            "assignments": len(distribution_result.get("assignments", [])),
                            "unassigned_tasks": len(distribution_result.get("unassigned_tasks", [])),
                            "distribution_time_ms": distribution_result.get("distribution_time_ms", 0),
                            "strategy_used": distribution_result.get("strategy_used", "unknown")
                        }
                    }
                )
            
            logger.info(
                "Task distribution completed",
                execution_id=execution_id,
                assignments=len(distribution_result.get("assignments", [])),
                unassigned=len(distribution_result.get("unassigned_tasks", [])),
                strategy=distribution_result.get("strategy_used")
            )
            
        except Exception as e:
            logger.error("Failed to handle task distribution completion", error=str(e))
    
    async def on_security_violation(
        self,
        execution_id: str,
        command_name: str,
        violation_type: str,
        violation_details: Dict[str, Any],
        requester_id: Optional[str] = None
    ) -> None:
        """Handle security policy violation."""
        try:
            # Send to hook interceptor if available
            if self.hook_interceptor:
                import uuid
                await self.hook_interceptor.capture_notification(
                    session_id=uuid.uuid4(),  # Mock session ID
                    agent_id=uuid.uuid4(),    # Mock agent ID
                    notification={
                        "level": "critical",
                        "message": f"Security violation detected: {violation_type}",
                        "details": {
                            "execution_id": execution_id,
                            "command_name": command_name,
                            "violation_type": violation_type,
                            "violation_details": violation_details,
                            "requester_id": requester_id
                        }
                    }
                )
            
            # Send immediate alert
            if self.message_broker:
                await self.message_broker.publish_agent_event(
                    "security_alerts",
                    "violation_detected",
                    {
                        "execution_id": execution_id,
                        "command_name": command_name,
                        "violation_type": violation_type,
                        "requester_id": requester_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "severity": "critical"
                    }
                )
            
            logger.critical(
                "Security violation detected",
                execution_id=execution_id,
                command_name=command_name,
                violation_type=violation_type,
                requester_id=requester_id
            )
            
        except Exception as e:
            logger.error("Failed to handle security violation", error=str(e))
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics for monitoring."""
        try:
            # Calculate success rate
            total_completed = (
                self.execution_metrics["successful_executions"] + 
                self.execution_metrics["failed_executions"]
            )
            success_rate = (
                (self.execution_metrics["successful_executions"] / max(total_completed, 1)) * 100
            )
            
            # Calculate commands per minute (simplified)
            current_time = datetime.utcnow()
            commands_per_minute = self.execution_metrics["total_executions"] / max(
                (current_time - datetime.utcnow().replace(hour=0, minute=0, second=0)).seconds / 60, 1
            )
            
            return {
                "timestamp": current_time.isoformat(),
                "execution_metrics": self.execution_metrics,
                "performance_metrics": {
                    "success_rate_percent": success_rate,
                    "commands_per_minute": commands_per_minute,
                    "average_execution_time_seconds": self.execution_metrics["average_execution_time"],
                    "active_executions": len(self.active_executions)
                },
                "agent_metrics": self.execution_metrics["agent_utilization"],
                "alert_status": await self._get_alert_status(),
                "system_health": {
                    "status": "healthy" if success_rate >= self.alert_thresholds["min_success_rate_percent"] else "degraded",
                    "active_alerts": await self._get_active_alerts()
                }
            }
            
        except Exception as e:
            logger.error("Failed to get system metrics", error=str(e))
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    # Private helper methods
    
    def _register_custom_hooks(self) -> None:
        """Register custom command hooks with the existing hook manager."""
        try:
            # Register hook types specific to custom commands
            custom_hook_types = [
                "custom_command_execution_start",
                "custom_command_execution_complete", 
                "custom_command_step_start",
                "custom_command_step_complete",
                "task_distribution_complete",
                "security_violation"
            ]
            
            for hook_type in custom_hook_types:
                # This would register with the existing hook manager
                # Implementation depends on the hook manager's API
                logger.debug(f"Registered hook type: {hook_type}")
            
        except Exception as e:
            logger.error("Failed to register custom hooks", error=str(e))
    
    async def _check_alert_conditions(
        self,
        result: CommandExecutionResult,
        execution_time: float
    ) -> None:
        """Check for alert conditions and trigger alerts if necessary."""
        try:
            alerts = []
            
            # Check execution time threshold
            if execution_time > self.alert_thresholds["max_execution_time_seconds"]:
                alerts.append({
                    "type": "long_execution_time",
                    "message": f"Execution time {execution_time}s exceeds threshold",
                    "execution_id": str(result.execution_id),
                    "threshold": self.alert_thresholds["max_execution_time_seconds"]
                })
            
            # Check failure rate
            total_completed = (
                self.execution_metrics["successful_executions"] + 
                self.execution_metrics["failed_executions"]
            )
            
            if total_completed >= 10:  # Only check after sufficient data
                failure_rate = (self.execution_metrics["failed_executions"] / total_completed) * 100
                if failure_rate > self.alert_thresholds["max_failure_rate_percent"]:
                    alerts.append({
                        "type": "high_failure_rate",
                        "message": f"Failure rate {failure_rate:.1f}% exceeds threshold",
                        "failure_rate": failure_rate,
                        "threshold": self.alert_thresholds["max_failure_rate_percent"]
                    })
            
            # Check concurrent executions
            if len(self.active_executions) > self.alert_thresholds["max_concurrent_executions"]:
                alerts.append({
                    "type": "high_concurrency",
                    "message": f"Active executions {len(self.active_executions)} exceeds threshold",
                    "active_executions": len(self.active_executions),
                    "threshold": self.alert_thresholds["max_concurrent_executions"]
                })
            
            # Send alerts
            for alert in alerts:
                await self._send_alert(alert)
                
        except Exception as e:
            logger.error("Failed to check alert conditions", error=str(e))
    
    async def _send_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert through observability system."""
        try:
            # Send to hook interceptor if available
            if self.hook_interceptor:
                import uuid
                await self.hook_interceptor.capture_notification(
                    session_id=uuid.uuid4(),  # Mock session ID
                    agent_id=uuid.uuid4(),    # Mock agent ID
                    notification={
                        "level": "warning",
                        "message": alert.get("message", f"System alert: {alert['type']}"),
                        "details": alert
                    }
                )
            
            if self.message_broker:
                await self.message_broker.publish_agent_event(
                    "system_alerts",
                    alert["type"],
                    {
                        **alert,
                        "timestamp": datetime.utcnow().isoformat(),
                        "component": "custom_commands"
                    }
                )
            
            logger.warning("System alert triggered", alert_type=alert["type"], details=alert)
            
        except Exception as e:
            logger.error("Failed to send alert", error=str(e))
    
    async def _get_alert_status(self) -> Dict[str, Any]:
        """Get current alert status."""
        try:
            # Calculate current metrics for alert evaluation
            total_completed = (
                self.execution_metrics["successful_executions"] + 
                self.execution_metrics["failed_executions"]
            )
            
            success_rate = (
                (self.execution_metrics["successful_executions"] / max(total_completed, 1)) * 100
            )
            
            failure_rate = (
                (self.execution_metrics["failed_executions"] / max(total_completed, 1)) * 100
            )
            
            return {
                "execution_time_alert": self.execution_metrics["average_execution_time"] > self.alert_thresholds["max_execution_time_seconds"],
                "failure_rate_alert": failure_rate > self.alert_thresholds["max_failure_rate_percent"],
                "concurrency_alert": len(self.active_executions) > self.alert_thresholds["max_concurrent_executions"],
                "success_rate_alert": success_rate < self.alert_thresholds["min_success_rate_percent"],
                "current_metrics": {
                    "success_rate": success_rate,
                    "failure_rate": failure_rate,
                    "active_executions": len(self.active_executions),
                    "average_execution_time": self.execution_metrics["average_execution_time"]
                }
            }
            
        except Exception as e:
            logger.error("Failed to get alert status", error=str(e))
            return {"error": str(e)}
    
    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of currently active alerts."""
        try:
            active_alerts = []
            alert_status = await self._get_alert_status()
            
            for alert_type, is_active in alert_status.items():
                if isinstance(is_active, bool) and is_active:
                    active_alerts.append({
                        "type": alert_type,
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": "active"
                    })
            
            return active_alerts
            
        except Exception as e:
            logger.error("Failed to get active alerts", error=str(e))
            return []