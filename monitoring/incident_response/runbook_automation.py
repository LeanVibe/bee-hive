"""
Epic 7 Phase 3: Runbook Automation for Incident Response

Automated incident response system with runbook execution:
- Automated response to common incident scenarios
- Self-healing system capabilities with safe automation boundaries
- Runbook execution with approval workflows for critical actions
- Incident escalation and human handoff when automation fails
- Comprehensive logging and audit trail for all automated actions
- Integration with monitoring and alerting systems
"""

import asyncio
import json
import time
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
import tempfile
import os

logger = structlog.get_logger()


class RunbookStepType(Enum):
    """Types of runbook steps."""
    COMMAND = "command"
    API_CALL = "api_call"
    DATABASE_QUERY = "database_query"
    NOTIFICATION = "notification"
    WAIT = "wait"
    APPROVAL = "approval"
    CONDITIONAL = "conditional"
    HEALTH_CHECK = "health_check"


class RunbookStepStatus(Enum):
    """Status of runbook step execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING_APPROVAL = "waiting_approval"


class IncidentSeverity(Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RunbookStep:
    """Individual step in a runbook."""
    id: str
    name: str
    description: str
    step_type: RunbookStepType
    command: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 3
    retry_delay_seconds: int = 5
    required_approval: bool = False
    approval_timeout_minutes: int = 15
    on_success: Optional[str] = None  # Next step if successful
    on_failure: Optional[str] = None  # Next step if failed
    conditions: Dict[str, Any] = field(default_factory=dict)
    safe_to_automate: bool = True


@dataclass
class RunbookExecution:
    """Runbook execution instance."""
    id: str
    runbook_name: str
    incident_id: str
    triggered_by: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"
    current_step: Optional[str] = None
    step_executions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_log: List[str] = field(default_factory=list)
    approval_requests: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Runbook:
    """Complete runbook definition."""
    name: str
    description: str
    version: str
    trigger_conditions: List[str]
    steps: List[RunbookStep]
    metadata: Dict[str, Any] = field(default_factory=dict)
    auto_execute: bool = False
    require_approval: bool = False
    max_execution_time_minutes: int = 60
    incident_types: List[str] = field(default_factory=list)


class RunbookAutomationSystem:
    """
    Comprehensive runbook automation system for Epic 7 Phase 3.
    
    Provides automated incident response, self-healing capabilities,
    and safe automation with human oversight for production systems.
    """
    
    def __init__(self):
        self.runbooks: Dict[str, Runbook] = {}
        self.active_executions: Dict[str, RunbookExecution] = {}
        self.execution_history: List[RunbookExecution] = []
        
        # Configuration
        self.automation_enabled = True
        self.safe_mode = True  # Only execute steps marked as safe
        self.approval_required_for_critical = True
        self.max_concurrent_executions = 5
        
        # Statistics
        self.automation_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "manual_interventions": 0,
            "avg_execution_time_minutes": 0,
            "automation_success_rate": 0
        }
        
        self.setup_default_runbooks()
        logger.info("🤖 Runbook Automation System initialized for Epic 7 Phase 3")
        
    def setup_default_runbooks(self):
        """Setup default runbooks for common incident scenarios."""
        
        # High CPU Usage Response Runbook
        self.add_runbook(Runbook(
            name="high_cpu_response",
            description="Automated response to high CPU usage alerts",
            version="1.0.0",
            trigger_conditions=["alert.rule_name == 'high_cpu_usage'"],
            auto_execute=True,
            steps=[
                RunbookStep(
                    id="check_system_status",
                    name="Check System Status",
                    description="Verify system health and current load",
                    step_type=RunbookStepType.HEALTH_CHECK,
                    command="system_health_check",
                    safe_to_automate=True,
                    on_success="identify_cpu_hogs"
                ),
                RunbookStep(
                    id="identify_cpu_hogs",
                    name="Identify CPU-Heavy Processes",
                    description="Find processes consuming the most CPU",
                    step_type=RunbookStepType.COMMAND,
                    command="ps aux --sort=-%cpu | head -10",
                    safe_to_automate=True,
                    on_success="check_process_legitimacy"
                ),
                RunbookStep(
                    id="check_process_legitimacy",
                    name="Verify Process Legitimacy",
                    description="Check if high-CPU processes are legitimate",
                    step_type=RunbookStepType.CONDITIONAL,
                    command="validate_running_processes",
                    safe_to_automate=True,
                    on_success="restart_high_cpu_services",
                    on_failure="escalate_to_human"
                ),
                RunbookStep(
                    id="restart_high_cpu_services",
                    name="Restart High CPU Services",
                    description="Restart services consuming excessive CPU",
                    step_type=RunbookStepType.COMMAND,
                    command="systemctl restart {service_name}",
                    required_approval=True,
                    safe_to_automate=False,
                    approval_timeout_minutes=10,
                    on_success="verify_cpu_reduction"
                ),
                RunbookStep(
                    id="verify_cpu_reduction",
                    name="Verify CPU Usage Reduction",
                    description="Confirm CPU usage has returned to normal",
                    step_type=RunbookStepType.HEALTH_CHECK,
                    command="check_cpu_usage",
                    safe_to_automate=True,
                    on_success="notify_success",
                    on_failure="escalate_to_human"
                ),
                RunbookStep(
                    id="escalate_to_human",
                    name="Escalate to Human",
                    description="Hand off incident to human operator",
                    step_type=RunbookStepType.NOTIFICATION,
                    command="send_escalation_notification",
                    safe_to_automate=True
                ),
                RunbookStep(
                    id="notify_success",
                    name="Notify Resolution Success",
                    description="Send success notification",
                    step_type=RunbookStepType.NOTIFICATION,
                    command="send_success_notification",
                    safe_to_automate=True
                )
            ],
            incident_types=["high_cpu_usage", "performance_degradation"]
        ))
        
        # Database Connection Pool Exhaustion Runbook
        self.add_runbook(Runbook(
            name="database_connection_recovery",
            description="Automated recovery for database connection pool issues",
            version="1.0.0",
            trigger_conditions=["alert.rule_name == 'database_connection_exhaustion'"],
            auto_execute=True,
            steps=[
                RunbookStep(
                    id="check_db_connections",
                    name="Check Database Connections",
                    description="Analyze current database connection usage",
                    step_type=RunbookStepType.DATABASE_QUERY,
                    command="SELECT count(*) FROM pg_stat_activity WHERE state = 'active'",
                    safe_to_automate=True,
                    on_success="identify_long_running_queries"
                ),
                RunbookStep(
                    id="identify_long_running_queries",
                    name="Identify Long-Running Queries",
                    description="Find queries that have been running for too long",
                    step_type=RunbookStepType.DATABASE_QUERY,
                    command="SELECT pid, query, state_change FROM pg_stat_activity WHERE state = 'active' AND state_change < now() - interval '5 minutes'",
                    safe_to_automate=True,
                    on_success="terminate_long_queries"
                ),
                RunbookStep(
                    id="terminate_long_queries",
                    name="Terminate Long-Running Queries",
                    description="Kill queries running longer than threshold",
                    step_type=RunbookStepType.DATABASE_QUERY,
                    command="SELECT pg_terminate_backend({pid})",
                    required_approval=True,
                    safe_to_automate=False,
                    approval_timeout_minutes=5,
                    on_success="restart_connection_pool"
                ),
                RunbookStep(
                    id="restart_connection_pool",
                    name="Restart Connection Pool",
                    description="Restart PgBouncer to reset connection pool",
                    step_type=RunbookStepType.COMMAND,
                    command="systemctl restart pgbouncer",
                    required_approval=True,
                    safe_to_automate=False,
                    on_success="verify_connection_recovery"
                ),
                RunbookStep(
                    id="verify_connection_recovery",
                    name="Verify Connection Recovery",
                    description="Confirm database connections are healthy",
                    step_type=RunbookStepType.HEALTH_CHECK,
                    command="test_database_connectivity",
                    safe_to_automate=True,
                    on_success="notify_recovery_success",
                    on_failure="escalate_db_issue"
                ),
                RunbookStep(
                    id="escalate_db_issue",
                    name="Escalate Database Issue",
                    description="Escalate to database team",
                    step_type=RunbookStepType.NOTIFICATION,
                    command="escalate_to_database_team",
                    safe_to_automate=True
                ),
                RunbookStep(
                    id="notify_recovery_success",
                    name="Notify Recovery Success",
                    description="Send recovery success notification",
                    step_type=RunbookStepType.NOTIFICATION,
                    command="send_recovery_notification",
                    safe_to_automate=True
                )
            ],
            incident_types=["database_connection_exhaustion", "database_performance"]
        ))
        
        # API Error Rate Spike Response Runbook
        self.add_runbook(Runbook(
            name="api_error_rate_response",
            description="Automated response to API error rate spikes",
            version="1.0.0",
            trigger_conditions=["alert.rule_name == 'high_api_error_rate'"],
            auto_execute=True,
            steps=[
                RunbookStep(
                    id="analyze_error_patterns",
                    name="Analyze Error Patterns",
                    description="Identify the most common API errors",
                    step_type=RunbookStepType.API_CALL,
                    command="GET /api/v2/monitoring/errors/analysis",
                    safe_to_automate=True,
                    on_success="check_service_health"
                ),
                RunbookStep(
                    id="check_service_health",
                    name="Check Service Health",
                    description="Verify health of all API services",
                    step_type=RunbookStepType.HEALTH_CHECK,
                    command="api_health_check_all_services",
                    safe_to_automate=True,
                    on_success="restart_unhealthy_services"
                ),
                RunbookStep(
                    id="restart_unhealthy_services",
                    name="Restart Unhealthy Services",
                    description="Restart services reporting unhealthy status",
                    step_type=RunbookStepType.COMMAND,
                    command="docker-compose restart {unhealthy_services}",
                    required_approval=True,
                    safe_to_automate=False,
                    on_success="verify_error_rate_improvement"
                ),
                RunbookStep(
                    id="verify_error_rate_improvement",
                    name="Verify Error Rate Improvement",
                    description="Check if API error rate has improved",
                    step_type=RunbookStepType.HEALTH_CHECK,
                    command="check_api_error_rate",
                    safe_to_automate=True,
                    on_success="notify_api_recovery",
                    on_failure="escalate_api_issue"
                ),
                RunbookStep(
                    id="escalate_api_issue",
                    name="Escalate API Issue",
                    description="Escalate to engineering team",
                    step_type=RunbookStepType.NOTIFICATION,
                    command="escalate_to_engineering_team",
                    safe_to_automate=True
                ),
                RunbookStep(
                    id="notify_api_recovery",
                    name="Notify API Recovery",
                    description="Send API recovery notification",
                    step_type=RunbookStepType.NOTIFICATION,
                    command="send_api_recovery_notification",
                    safe_to_automate=True
                )
            ],
            incident_types=["api_error_rate", "service_degradation"]
        ))
        
    def add_runbook(self, runbook: Runbook):
        """Add or update a runbook."""
        self.runbooks[runbook.name] = runbook
        logger.info("📖 Runbook added", 
                   name=runbook.name, 
                   version=runbook.version,
                   steps=len(runbook.steps),
                   auto_execute=runbook.auto_execute)
                   
    async def trigger_runbook(self, runbook_name: str, incident_id: str, 
                            trigger_context: Dict[str, Any],
                            triggered_by: str = "system") -> str:
        """Trigger runbook execution for an incident."""
        try:
            if runbook_name not in self.runbooks:
                raise ValueError(f"Runbook '{runbook_name}' not found")
                
            if len(self.active_executions) >= self.max_concurrent_executions:
                raise RuntimeError("Maximum concurrent executions reached")
                
            runbook = self.runbooks[runbook_name]
            
            execution_id = f"exec_{int(time.time())}_{runbook_name}"
            
            execution = RunbookExecution(
                id=execution_id,
                runbook_name=runbook_name,
                incident_id=incident_id,
                triggered_by=triggered_by,
                started_at=datetime.utcnow(),
                current_step=runbook.steps[0].id if runbook.steps else None
            )
            
            self.active_executions[execution_id] = execution
            self.automation_stats["total_executions"] += 1
            
            # Log execution start
            self._log_execution_event(execution, f"Runbook execution started by {triggered_by}")
            
            # Start execution
            asyncio.create_task(self._execute_runbook(execution, runbook, trigger_context))
            
            logger.info("🚀 Runbook execution started",
                       execution_id=execution_id,
                       runbook=runbook_name,
                       incident_id=incident_id,
                       triggered_by=triggered_by)
                       
            return execution_id
            
        except Exception as e:
            logger.error("❌ Failed to trigger runbook", 
                        runbook=runbook_name,
                        incident_id=incident_id,
                        error=str(e))
            raise
            
    async def _execute_runbook(self, execution: RunbookExecution, runbook: Runbook,
                             context: Dict[str, Any]):
        """Execute a runbook with all its steps."""
        try:
            current_step_id = execution.current_step
            
            while current_step_id:
                step = self._find_step(runbook, current_step_id)
                if not step:
                    break
                    
                # Execute the step
                step_result = await self._execute_step(execution, step, context)
                
                # Determine next step based on result
                if step_result["success"]:
                    current_step_id = step.on_success
                else:
                    current_step_id = step.on_failure
                    
                execution.current_step = current_step_id
                
                # Break if we've reached a terminal state
                if not current_step_id:
                    break
                    
            # Complete execution
            execution.status = "completed"
            execution.completed_at = datetime.utcnow()
            
            # Update statistics
            self.automation_stats["successful_executions"] += 1
            self._update_execution_stats(execution)
            
            # Move to history
            self.execution_history.append(execution)
            del self.active_executions[execution.id]
            
            self._log_execution_event(execution, "Runbook execution completed successfully")
            
            logger.info("✅ Runbook execution completed",
                       execution_id=execution.id,
                       runbook=execution.runbook_name,
                       duration_minutes=(execution.completed_at - execution.started_at).total_seconds() / 60)
                       
        except Exception as e:
            # Handle execution failure
            execution.status = "failed"
            execution.completed_at = datetime.utcnow()
            
            self.automation_stats["failed_executions"] += 1
            self._update_execution_stats(execution)
            
            # Move to history
            self.execution_history.append(execution)
            if execution.id in self.active_executions:
                del self.active_executions[execution.id]
                
            self._log_execution_event(execution, f"Runbook execution failed: {str(e)}")
            
            logger.error("❌ Runbook execution failed",
                        execution_id=execution.id,
                        runbook=execution.runbook_name,
                        error=str(e))
                        
    async def _execute_step(self, execution: RunbookExecution, step: RunbookStep,
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single runbook step."""
        try:
            step_start_time = datetime.utcnow()
            
            # Initialize step execution record
            step_execution = {
                "status": RunbookStepStatus.RUNNING.value,
                "started_at": step_start_time.isoformat(),
                "attempts": 0,
                "output": None,
                "error": None
            }
            
            execution.step_executions[step.id] = step_execution
            
            self._log_execution_event(execution, f"Starting step: {step.name}")
            
            # Check if step requires approval and is not safe to automate
            if step.required_approval and not self._has_approval(execution, step.id):
                if self.safe_mode and not step.safe_to_automate:
                    return await self._request_approval(execution, step)
                    
            # Execute step based on type
            result = None
            for attempt in range(step.retry_count):
                step_execution["attempts"] = attempt + 1
                
                try:
                    if step.step_type == RunbookStepType.COMMAND:
                        result = await self._execute_command_step(step, context)
                    elif step.step_type == RunbookStepType.API_CALL:
                        result = await self._execute_api_step(step, context)
                    elif step.step_type == RunbookStepType.DATABASE_QUERY:
                        result = await self._execute_database_step(step, context)
                    elif step.step_type == RunbookStepType.HEALTH_CHECK:
                        result = await self._execute_health_check_step(step, context)
                    elif step.step_type == RunbookStepType.NOTIFICATION:
                        result = await self._execute_notification_step(step, context)
                    elif step.step_type == RunbookStepType.WAIT:
                        result = await self._execute_wait_step(step, context)
                    elif step.step_type == RunbookStepType.CONDITIONAL:
                        result = await self._execute_conditional_step(step, context)
                    else:
                        raise ValueError(f"Unknown step type: {step.step_type}")
                        
                    # Success - break retry loop
                    break
                    
                except Exception as e:
                    if attempt < step.retry_count - 1:
                        self._log_execution_event(
                            execution, 
                            f"Step {step.name} failed (attempt {attempt + 1}), retrying: {str(e)}"
                        )
                        await asyncio.sleep(step.retry_delay_seconds)
                    else:
                        # Final attempt failed
                        raise
                        
            # Update step execution record
            step_execution["status"] = RunbookStepStatus.COMPLETED.value
            step_execution["completed_at"] = datetime.utcnow().isoformat()
            step_execution["output"] = result
            
            self._log_execution_event(execution, f"Completed step: {step.name}")
            
            return {"success": True, "result": result}
            
        except Exception as e:
            # Update step execution record
            step_execution["status"] = RunbookStepStatus.FAILED.value
            step_execution["completed_at"] = datetime.utcnow().isoformat()
            step_execution["error"] = str(e)
            
            self._log_execution_event(execution, f"Failed step: {step.name} - {str(e)}")
            
            return {"success": False, "error": str(e)}
            
    async def _execute_command_step(self, step: RunbookStep, context: Dict[str, Any]) -> str:
        """Execute a command step."""
        try:
            # Format command with context parameters
            command = step.command.format(**context, **step.parameters)
            
            # Safety check for dangerous commands in safe mode
            if self.safe_mode and not step.safe_to_automate:
                return f"Command skipped due to safe mode: {command}"
                
            # Execute command with timeout
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=step.timeout_seconds
                )
                
                if process.returncode == 0:
                    return stdout.decode('utf-8').strip()
                else:
                    raise RuntimeError(f"Command failed with return code {process.returncode}: {stderr.decode('utf-8')}")
                    
            except asyncio.TimeoutError:
                process.kill()
                raise RuntimeError(f"Command timed out after {step.timeout_seconds} seconds")
                
        except Exception as e:
            logger.error("❌ Command step execution failed", command=step.command, error=str(e))
            raise
            
    async def _execute_api_step(self, step: RunbookStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an API call step."""
        try:
            # Mock API call - in production would make actual HTTP requests
            logger.info("📡 Executing API step", command=step.command)
            
            # Simulate API call
            await asyncio.sleep(0.1)
            
            return {
                "status": "success",
                "data": {"message": f"API call {step.command} executed successfully"},
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("❌ API step execution failed", command=step.command, error=str(e))
            raise
            
    async def _execute_database_step(self, step: RunbookStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a database query step."""
        try:
            # Mock database query - in production would execute actual queries
            logger.info("🗄️  Executing database step", command=step.command)
            
            # Simulate database query
            await asyncio.sleep(0.1)
            
            return {
                "rows_affected": 1,
                "result": [{"column": "value"}],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("❌ Database step execution failed", command=step.command, error=str(e))
            raise
            
    async def _execute_health_check_step(self, step: RunbookStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a health check step."""
        try:
            logger.info("🏥 Executing health check step", command=step.command)
            
            # Mock health check - in production would perform actual health checks
            await asyncio.sleep(0.1)
            
            return {
                "healthy": True,
                "checks": {
                    "api": "healthy",
                    "database": "healthy",
                    "redis": "healthy"
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("❌ Health check step execution failed", command=step.command, error=str(e))
            raise
            
    async def _execute_notification_step(self, step: RunbookStep, context: Dict[str, Any]) -> str:
        """Execute a notification step."""
        try:
            logger.info("📢 Executing notification step", command=step.command)
            
            # Mock notification - in production would send actual notifications
            await asyncio.sleep(0.1)
            
            return f"Notification sent: {step.command}"
            
        except Exception as e:
            logger.error("❌ Notification step execution failed", command=step.command, error=str(e))
            raise
            
    async def _execute_wait_step(self, step: RunbookStep, context: Dict[str, Any]) -> str:
        """Execute a wait step."""
        try:
            wait_seconds = step.parameters.get("wait_seconds", 30)
            logger.info("⏰ Executing wait step", wait_seconds=wait_seconds)
            
            await asyncio.sleep(wait_seconds)
            
            return f"Waited for {wait_seconds} seconds"
            
        except Exception as e:
            logger.error("❌ Wait step execution failed", error=str(e))
            raise
            
    async def _execute_conditional_step(self, step: RunbookStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a conditional step."""
        try:
            logger.info("🔀 Executing conditional step", command=step.command)
            
            # Mock conditional logic - in production would evaluate actual conditions
            await asyncio.sleep(0.1)
            
            condition_result = step.conditions.get("default_result", True)
            
            return {
                "condition_met": condition_result,
                "evaluation": f"Condition {step.command} evaluated to {condition_result}",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("❌ Conditional step execution failed", command=step.command, error=str(e))
            raise
            
    async def _request_approval(self, execution: RunbookExecution, step: RunbookStep) -> Dict[str, Any]:
        """Request human approval for a step."""
        try:
            approval_request = {
                "id": f"approval_{int(time.time())}_{step.id}",
                "execution_id": execution.id,
                "step_id": step.id,
                "step_name": step.name,
                "step_description": step.description,
                "command": step.command,
                "requested_at": datetime.utcnow().isoformat(),
                "timeout_minutes": step.approval_timeout_minutes,
                "status": "pending"
            }
            
            execution.approval_requests.append(approval_request)
            
            self._log_execution_event(execution, f"Approval requested for step: {step.name}")
            
            # In production, would send approval request to appropriate personnel
            logger.warning("📋 Approval required",
                          execution_id=execution.id,
                          step=step.name,
                          timeout_minutes=step.approval_timeout_minutes)
                          
            # Mock approval process - in production would wait for actual approval
            await asyncio.sleep(1)  # Simulate approval delay
            
            return {"success": False, "waiting_approval": True}
            
        except Exception as e:
            logger.error("❌ Failed to request approval", step_id=step.id, error=str(e))
            raise
            
    def _find_step(self, runbook: Runbook, step_id: str) -> Optional[RunbookStep]:
        """Find a step by ID in the runbook."""
        for step in runbook.steps:
            if step.id == step_id:
                return step
        return None
        
    def _has_approval(self, execution: RunbookExecution, step_id: str) -> bool:
        """Check if a step has received approval."""
        for approval in execution.approval_requests:
            if (approval["step_id"] == step_id and 
                approval["status"] == "approved"):
                return True
        return False
        
    def _log_execution_event(self, execution: RunbookExecution, message: str):
        """Log an event in the execution log."""
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] {message}"
        execution.execution_log.append(log_entry)
        
    def _update_execution_stats(self, execution: RunbookExecution):
        """Update execution statistics."""
        if execution.completed_at and execution.started_at:
            duration_minutes = (execution.completed_at - execution.started_at).total_seconds() / 60
            
            # Update average execution time
            current_avg = self.automation_stats["avg_execution_time_minutes"]
            total_executions = self.automation_stats["total_executions"]
            
            self.automation_stats["avg_execution_time_minutes"] = (
                (current_avg * (total_executions - 1) + duration_minutes) / total_executions
            )
            
            # Update success rate
            success_rate = (
                self.automation_stats["successful_executions"] / 
                self.automation_stats["total_executions"] * 100
            )
            self.automation_stats["automation_success_rate"] = success_rate
            
    async def get_automation_summary(self) -> Dict[str, Any]:
        """Get comprehensive automation summary."""
        try:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "automation_enabled": self.automation_enabled,
                "safe_mode": self.safe_mode,
                "statistics": self.automation_stats,
                "active_executions": len(self.active_executions),
                "runbooks_configured": len(self.runbooks),
                "recent_executions": [
                    {
                        "id": exec.id,
                        "runbook": exec.runbook_name,
                        "incident_id": exec.incident_id,
                        "status": exec.status,
                        "started_at": exec.started_at.isoformat(),
                        "completed_at": exec.completed_at.isoformat() if exec.completed_at else None,
                        "duration_minutes": (
                            (exec.completed_at - exec.started_at).total_seconds() / 60 
                            if exec.completed_at else None
                        )
                    }
                    for exec in sorted(
                        self.execution_history[-10:], 
                        key=lambda x: x.started_at, 
                        reverse=True
                    )
                ],
                "runbook_catalog": [
                    {
                        "name": rb.name,
                        "description": rb.description,
                        "version": rb.version,
                        "auto_execute": rb.auto_execute,
                        "steps_count": len(rb.steps),
                        "incident_types": rb.incident_types
                    }
                    for rb in self.runbooks.values()
                ]
            }
            
        except Exception as e:
            logger.error("❌ Failed to get automation summary", error=str(e))
            return {"error": str(e)}


# Global runbook automation instance
runbook_automation = RunbookAutomationSystem()


async def init_runbook_automation():
    """Initialize the runbook automation system."""
    logger.info("🤖 Initializing Runbook Automation System for Epic 7 Phase 3")
    

if __name__ == "__main__":
    # Test the runbook automation system
    async def test_automation():
        await init_runbook_automation()
        
        # Trigger a runbook execution
        execution_id = await runbook_automation.trigger_runbook(
            "high_cpu_response",
            "incident_123",
            {"cpu_usage_percent": 85.0, "service_name": "leanvibe-api"},
            "alert_system"
        )
        
        # Wait for execution to complete
        await asyncio.sleep(2)
        
        # Get automation summary
        summary = await runbook_automation.get_automation_summary()
        print(json.dumps(summary, indent=2))
        
    asyncio.run(test_automation())