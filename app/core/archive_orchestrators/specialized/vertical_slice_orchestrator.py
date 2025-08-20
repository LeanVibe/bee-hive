"""
Vertical Slice Orchestrator for LeanVibe Agent Hive 2.0

Integrates all agent lifecycle components into a cohesive vertical slice:
- Agent Registration/Deregistration
- Task Assignment and Execution
- Redis Streams Messaging
- Hook System Integration
- Database Persistence

This orchestrator demonstrates the complete 80/20 core capabilities.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import json

import structlog
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_async_session
from .redis import get_redis
from .agent_lifecycle_manager import AgentLifecycleManager, AgentRegistrationResult, TaskAssignmentResult
from .task_execution_engine import TaskExecutionEngine, ExecutionOutcome, ExecutionPhase
from .agent_messaging_service import AgentMessagingService, MessageType, MessagePriority
from .agent_lifecycle_hooks import AgentLifecycleHooks, HookExecutionResult, SecurityAction
from .hook_lifecycle_system import HookLifecycleSystem
from .agent_persona_system import AgentPersonaSystem, get_agent_persona_system
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.task import Task, TaskStatus, TaskType, TaskPriority

logger = structlog.get_logger()


@dataclass
class VerticalSliceMetrics:
    """Metrics for the vertical slice operation."""
    agents_registered: int = 0
    tasks_assigned: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    hooks_executed: int = 0
    messages_sent: int = 0
    average_assignment_time_ms: float = 0.0
    average_execution_time_ms: float = 0.0
    security_violations: int = 0
    system_uptime_seconds: float = 0.0


class VerticalSliceOrchestrator:
    """
    Orchestrates the complete agent lifecycle vertical slice.
    
    This class demonstrates the full 80/20 capabilities by coordinating
    all the lifecycle components in a production-ready implementation.
    """
    
    def __init__(self):
        self.redis = get_redis()
        
        # Initialize core components
        self.messaging_service = AgentMessagingService(self.redis)
        self.hook_system = HookLifecycleSystem()  # Will be integrated with existing system
        self.lifecycle_hooks = AgentLifecycleHooks(
            redis_client=self.redis,
            messaging_service=self.messaging_service,
            hook_system=self.hook_system
        )
        self.persona_system = get_agent_persona_system()
        
        # Initialize managers
        self.lifecycle_manager = AgentLifecycleManager(
            redis_client=self.redis,
            persona_system=self.persona_system,
            hook_system=self.hook_system
        )
        self.execution_engine = TaskExecutionEngine(
            redis_client=self.redis,
            hook_system=self.hook_system
        )
        
        # System state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.metrics = VerticalSliceMetrics()
        
        # Register message handlers
        self._register_message_handlers()
        
        # Register hook handlers
        self._register_hook_handlers()
        
        logger.info("🎯 Vertical Slice Orchestrator initialized")
    
    async def start_system(self) -> bool:
        """
        Start the complete vertical slice system.
        
        Returns:
            True if system started successfully
        """
        try:
            self.start_time = datetime.utcnow()
            self.is_running = True
            
            logger.info("🚀 Starting Vertical Slice system...")
            
            # Start messaging consumers
            await self.messaging_service.start_agent_consumer("orchestrator")
            
            # Initialize database schema if needed
            await self._ensure_database_ready()
            
            logger.info(f"✅ Vertical Slice system started at {self.start_time}")
            return True
            
        except Exception as e:
            logger.error("❌ Failed to start Vertical Slice system", error=str(e))
            self.is_running = False
            return False
    
    async def stop_system(self) -> bool:
        """
        Stop the vertical slice system gracefully.
        
        Returns:
            True if system stopped successfully
        """
        try:
            logger.info("🛑 Stopping Vertical Slice system...")
            
            # Send shutdown messages to all active agents
            await self.messaging_service.send_system_shutdown("System shutdown initiated")
            
            # Stop messaging consumers
            await self.messaging_service.stop_agent_consumer("orchestrator")
            
            # Wait for active executions to complete (with timeout)
            await self._wait_for_active_executions(timeout_seconds=30)
            
            self.is_running = False
            
            if self.start_time:
                uptime = (datetime.utcnow() - self.start_time).total_seconds()
                self.metrics.system_uptime_seconds = uptime
                logger.info(f"✅ System stopped after {uptime:.2f} seconds uptime")
            
            return True
            
        except Exception as e:
            logger.error("❌ Failed to stop system gracefully", error=str(e))
            return False
    
    async def demonstrate_complete_lifecycle(self) -> Dict[str, Any]:
        """
        Demonstrate the complete agent lifecycle flow.
        
        This method showcases all 80/20 capabilities in a single workflow:
        1. Agent registration with persona assignment
        2. Task creation and assignment
        3. Task execution with hooks
        4. Completion tracking and metrics
        
        Returns:
            Dictionary with demonstration results and metrics
        """
        demo_start_time = datetime.utcnow()
        demo_results = {
            "demo_id": str(uuid.uuid4()),
            "started_at": demo_start_time.isoformat(),
            "steps_completed": [],
            "metrics": {},
            "errors": []
        }
        
        try:
            logger.info("🎬 Starting complete lifecycle demonstration")
            
            # Step 1: Register multiple agents with different personas
            logger.info("📋 Step 1: Registering agents with personas")
            agents = []
            
            for i, (role, capabilities) in enumerate([
                ("backend_developer", [
                    {"name": "python_development", "confidence_level": 0.9, "specialization_areas": ["FastAPI", "SQLAlchemy", "async"]},
                    {"name": "database_design", "confidence_level": 0.8, "specialization_areas": ["PostgreSQL", "migrations"]}
                ]),
                ("frontend_developer", [
                    {"name": "ui_development", "confidence_level": 0.85, "specialization_areas": ["Vue.js", "TypeScript", "CSS"]},
                    {"name": "responsive_design", "confidence_level": 0.9, "specialization_areas": ["mobile", "desktop"]}
                ]),
                ("qa_engineer", [
                    {"name": "test_automation", "confidence_level": 0.95, "specialization_areas": ["pytest", "integration_tests"]},
                    {"name": "quality_assurance", "confidence_level": 0.9, "specialization_areas": ["code_review", "standards"]}
                ])
            ], 1):
                
                registration_result = await self.lifecycle_manager.register_agent(
                    name=f"demo_agent_{i}",
                    agent_type=AgentType.CLAUDE,
                    role=role,
                    capabilities=capabilities,
                    system_prompt=f"You are a {role} specialized in the given capabilities."
                )
                
                if registration_result.success:
                    agents.append(registration_result.agent_id)
                    self.metrics.agents_registered += 1
                    demo_results["steps_completed"].append(f"Registered {role} agent")
                else:
                    demo_results["errors"].append(f"Failed to register {role} agent: {registration_result.error_message}")
            
            await asyncio.sleep(1)  # Allow registration to propagate
            
            # Step 2: Create and assign tasks
            logger.info("📋 Step 2: Creating and assigning tasks")
            tasks = []
            
            task_definitions = [
                {
                    "title": "Implement user authentication API",
                    "description": "Create FastAPI endpoints for user login and registration",
                    "task_type": TaskType.FEATURE_DEVELOPMENT,
                    "priority": TaskPriority.HIGH,
                    "required_capabilities": ["python_development", "database_design"],
                    "estimated_effort": 120  # minutes
                },
                {
                    "title": "Design responsive login form",
                    "description": "Create a mobile-friendly login form with validation",
                    "task_type": TaskType.FEATURE_DEVELOPMENT,
                    "priority": TaskPriority.MEDIUM,
                    "required_capabilities": ["ui_development", "responsive_design"],
                    "estimated_effort": 90
                },
                {
                    "title": "Write integration tests for auth system",
                    "description": "Create comprehensive tests for authentication endpoints",
                    "task_type": TaskType.TESTING,
                    "priority": TaskPriority.HIGH,
                    "required_capabilities": ["test_automation", "quality_assurance"],
                    "estimated_effort": 60
                }
            ]
            
            # Create tasks in database
            async with get_async_session() as db:
                for task_def in task_definitions:
                    task = Task(**task_def)
                    db.add(task)
                    await db.commit()
                    await db.refresh(task)
                    tasks.append(task.id)
            
            # Assign tasks to agents
            assignment_times = []
            for task_id in tasks:
                assignment_start = datetime.utcnow()
                
                assignment_result = await self.lifecycle_manager.assign_task_to_agent(
                    task_id=task_id,
                    max_assignment_time_ms=500.0
                )
                
                assignment_time_ms = (datetime.utcnow() - assignment_start).total_seconds() * 1000
                assignment_times.append(assignment_time_ms)
                
                if assignment_result.success:
                    self.metrics.tasks_assigned += 1
                    demo_results["steps_completed"].append(f"Assigned task {task_id} to agent {assignment_result.agent_id}")
                else:
                    demo_results["errors"].append(f"Failed to assign task {task_id}: {assignment_result.error_message}")
            
            # Calculate average assignment time
            if assignment_times:
                self.metrics.average_assignment_time_ms = sum(assignment_times) / len(assignment_times)
            
            await asyncio.sleep(1)  # Allow assignments to propagate
            
            # Step 3: Simulate task executions with hooks
            logger.info("📋 Step 3: Executing tasks with hooks")
            execution_times = []
            
            for task_id in tasks[:2]:  # Execute first 2 tasks for demonstration
                execution_start = datetime.utcnow()
                
                # Get task and assigned agent
                async with get_async_session() as db:
                    task_result = await db.execute(select(Task).where(Task.id == task_id))
                    task = task_result.scalar_one_or_none()
                    
                    if task and task.assigned_agent_id:
                        # Start task execution
                        await self.execution_engine.start_task_execution(
                            task_id=task_id,
                            agent_id=task.assigned_agent_id,
                            execution_context={"demo_mode": True}
                        )
                        
                        # Simulate some work with progress updates
                        await asyncio.sleep(0.5)
                        await self.execution_engine.update_execution_progress(
                            task_id=task_id,
                            phase=ExecutionPhase.EXECUTION,
                            progress_percentage=50.0,
                            metadata={"current_step": "Implementing core logic"}
                        )
                        
                        await asyncio.sleep(0.5)
                        await self.execution_engine.update_execution_progress(
                            task_id=task_id,
                            phase=ExecutionPhase.VALIDATION,
                            progress_percentage=90.0,
                            metadata={"current_step": "Running tests"}
                        )
                        
                        # Complete task
                        execution_result = await self.execution_engine.complete_task_execution(
                            task_id=task_id,
                            outcome=ExecutionOutcome.SUCCESS,
                            result_data={
                                "implementation": "Task completed successfully",
                                "files_modified": ["auth.py", "models.py", "tests.py"],
                                "lines_of_code": 150
                            }
                        )
                        
                        execution_time_ms = (datetime.utcnow() - execution_start).total_seconds() * 1000
                        execution_times.append(execution_time_ms)
                        
                        if execution_result.outcome == ExecutionOutcome.SUCCESS:
                            self.metrics.tasks_completed += 1
                            demo_results["steps_completed"].append(f"Completed task {task_id}")
                        else:
                            self.metrics.tasks_failed += 1
                            demo_results["errors"].append(f"Task {task_id} failed")
            
            # Calculate average execution time
            if execution_times:
                self.metrics.average_execution_time_ms = sum(execution_times) / len(execution_times)
            
            # Step 4: Demonstrate hook system with security validation
            logger.info("📋 Step 4: Demonstrating hook system")
            if agents:
                # Test PreToolUse hook with safe command
                pre_hook_result = await self.lifecycle_hooks.execute_pre_tool_hooks(
                    agent_id=agents[0],
                    session_id=uuid.uuid4(),
                    tool_name="python_interpreter",
                    parameters={"code": "print('Hello, World!')"},
                    metadata={"demo_mode": True}
                )
                
                if pre_hook_result.success:
                    self.metrics.hooks_executed += 1
                    demo_results["steps_completed"].append("Executed PreToolUse hook successfully")
                
                # Test PostToolUse hook
                post_hook_result = await self.lifecycle_hooks.execute_post_tool_hooks(
                    agent_id=agents[0],
                    session_id=uuid.uuid4(),
                    tool_name="python_interpreter",
                    parameters={"code": "print('Hello, World!')"},
                    result={"output": "Hello, World!", "exit_code": 0},
                    success=True,
                    execution_time_ms=150.0,
                    metadata={"demo_mode": True}
                )
                
                if post_hook_result.success:
                    self.metrics.hooks_executed += 1
                    demo_results["steps_completed"].append("Executed PostToolUse hook successfully")
                
                # Test security validation with dangerous command
                dangerous_hook_result = await self.lifecycle_hooks.execute_pre_tool_hooks(
                    agent_id=agents[0],
                    session_id=uuid.uuid4(),
                    tool_name="bash",
                    parameters={"command": "rm -rf /tmp/test"},
                    metadata={"demo_mode": True, "security_test": True}
                )
                
                if dangerous_hook_result.security_action == SecurityAction.BLOCK:
                    self.metrics.security_violations += 1
                    demo_results["steps_completed"].append("Security system blocked dangerous command")
            
            # Step 5: Collect final metrics
            logger.info("📋 Step 5: Collecting final metrics")
            
            # Get system metrics
            system_metrics = await self.lifecycle_manager.get_system_metrics()
            hook_metrics = await self.lifecycle_hooks.get_hook_metrics()
            messaging_metrics = await self.messaging_service.get_messaging_metrics()
            execution_metrics = await self.execution_engine.get_performance_metrics()
            
            demo_results["metrics"] = {
                "demonstration": self.metrics.__dict__,
                "system": system_metrics,
                "hooks": hook_metrics,
                "messaging": messaging_metrics,
                "execution": execution_metrics
            }
            
            # Calculate demo duration
            demo_duration = (datetime.utcnow() - demo_start_time).total_seconds()
            demo_results["completed_at"] = datetime.utcnow().isoformat()
            demo_results["duration_seconds"] = demo_duration
            demo_results["success"] = len(demo_results["errors"]) == 0
            
            logger.info(
                "🎉 Complete lifecycle demonstration finished",
                duration_seconds=demo_duration,
                agents_registered=self.metrics.agents_registered,
                tasks_assigned=self.metrics.tasks_assigned,
                tasks_completed=self.metrics.tasks_completed,
                hooks_executed=self.metrics.hooks_executed,
                security_violations=self.metrics.security_violations,
                avg_assignment_time_ms=self.metrics.average_assignment_time_ms,
                success=demo_results["success"]
            )
            
            return demo_results
            
        except Exception as e:
            demo_results["errors"].append(f"Demonstration failed: {str(e)}")
            demo_results["success"] = False
            demo_results["completed_at"] = datetime.utcnow().isoformat()
            
            logger.error("❌ Lifecycle demonstration failed", error=str(e))
            return demo_results
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status across all components."""
        try:
            status = {
                "system": {
                    "is_running": self.is_running,
                    "start_time": self.start_time.isoformat() if self.start_time else None,
                    "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
                },
                "components": {},
                "metrics": self.metrics.__dict__,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Get component statuses
            if self.is_running:
                status["components"]["lifecycle_manager"] = await self.lifecycle_manager.get_system_metrics()
                status["components"]["execution_engine"] = await self.execution_engine.get_performance_metrics()
                status["components"]["messaging_service"] = await self.messaging_service.get_messaging_metrics()
                status["components"]["hooks"] = await self.lifecycle_hooks.get_hook_metrics()
            
            return status
            
        except Exception as e:
            logger.error("Failed to get comprehensive status", error=str(e))
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _register_message_handlers(self) -> None:
        """Register message handlers for lifecycle events."""
        async def handle_heartbeat_response(message):
            """Handle agent heartbeat responses."""
            agent_id = uuid.UUID(message.payload.get("agent_id"))
            status_data = message.payload.get("status", {})
            await self.lifecycle_manager.process_agent_heartbeat(agent_id, status_data)
        
        async def handle_task_completion(message):
            """Handle task completion messages."""
            task_id = uuid.UUID(message.payload.get("task_id"))
            agent_id = uuid.UUID(message.payload.get("agent_id"))
            result = message.payload.get("result", {})
            success = message.payload.get("success", False)
            
            await self.lifecycle_manager.complete_task(task_id, agent_id, result, success)
        
        # Register handlers (would be done through the messaging service)
        # This is a simplified example of how handlers would be registered
    
    def _register_hook_handlers(self) -> None:
        """Register custom hook handlers for demonstration."""
        
        async def demo_pre_tool_hook(context):
            """Demonstration PreToolUse hook."""
            logger.info(
                f"🪝 PreToolUse: Agent {context.agent_id} using tool {context.tool_name}",
                tool_name=context.tool_name,
                parameters=context.parameters
            )
            return {"hook_processed": True, "timestamp": datetime.utcnow().isoformat()}
        
        async def demo_post_tool_hook(context):
            """Demonstration PostToolUse hook."""
            tool_success = context.metadata.get("tool_success", False)
            logger.info(
                f"🪝 PostToolUse: Agent {context.agent_id} completed {context.tool_name}",
                tool_name=context.tool_name,
                success=tool_success,
                execution_time_ms=context.metadata.get("tool_execution_time_ms", 0)
            )
            return {"hook_processed": True, "timestamp": datetime.utcnow().isoformat()}
        
        def demo_error_hook(context, error):
            """Demonstration error hook."""
            logger.error(
                f"🪝 Error: Agent {context.agent_id} error in {context.tool_name}",
                tool_name=context.tool_name,
                error=str(error)
            )
        
        # Register hooks
        self.lifecycle_hooks.register_pre_tool_hook(demo_pre_tool_hook)
        self.lifecycle_hooks.register_post_tool_hook(demo_post_tool_hook)
        self.lifecycle_hooks.register_error_hook(demo_error_hook)
    
    async def _ensure_database_ready(self) -> None:
        """Ensure database is ready for operations."""
        try:
            async with get_async_session() as db:
                # Test database connectivity
                result = await db.execute(select(func.count(Agent.id)))
                agent_count = result.scalar()
                logger.info(f"✅ Database ready, {agent_count} agents in system")
        except Exception as e:
            logger.error("❌ Database not ready", error=str(e))
            raise
    
    async def _wait_for_active_executions(self, timeout_seconds: int = 30) -> None:
        """Wait for active executions to complete."""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout_seconds:
            active_count = len(self.execution_engine.active_executions)
            if active_count == 0:
                logger.info("✅ All executions completed")
                return
            
            logger.info(f"⏳ Waiting for {active_count} active executions...")
            await asyncio.sleep(1)
        
        logger.warning(f"⚠️ Timeout waiting for executions, {len(self.execution_engine.active_executions)} still active")