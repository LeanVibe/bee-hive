"""
Subagent Coordination Framework

Manages and coordinates multiple subagents working on different aspects
of the LeanVibe Agent Hive system. Provides health monitoring, task
coordination, and communication between specialized agents.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

import structlog
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

logger = structlog.get_logger(__name__)
console = Console()


class SubagentRole(Enum):
    """Specialized subagent roles for coordination."""
    QA_TEST_GUARDIAN = "qa-test-guardian"
    PROJECT_ORCHESTRATOR = "project-orchestrator"
    BACKEND_ENGINEER = "backend-engineer" 
    FRONTEND_BUILDER = "frontend-builder"
    DEVOPS_DEPLOYER = "devops-deployer"
    GENERAL_PURPOSE = "general-purpose"


class SubagentStatus(Enum):
    """Current status of a subagent."""
    ACTIVE = "active"
    IDLE = "idle"
    STUCK = "stuck"
    ERROR = "error"
    OFFLINE = "offline"
    INITIALIZING = "initializing"


class TaskPriority(Enum):
    """Priority levels for subagent tasks."""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SubagentInfo:
    """Information about a registered subagent."""
    agent_id: str
    role: SubagentRole
    session_name: str
    workspace_path: str
    status: SubagentStatus
    last_heartbeat: datetime
    assigned_tasks: List[str]
    capabilities: List[str]
    performance_metrics: Dict[str, Any]
    created_at: datetime
    error_count: int = 0
    restart_count: int = 0


@dataclass 
class CoordinationTask:
    """Task to be coordinated among subagents."""
    task_id: str
    title: str
    description: str
    priority: TaskPriority
    required_roles: List[SubagentRole]
    assigned_agents: List[str]
    dependencies: List[str]
    estimated_duration: Optional[timedelta]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    status: str = "pending"


class SubagentCoordinator:
    """
    Central coordination system for managing multiple subagents.
    
    Provides:
    - Health monitoring and recovery
    - Task assignment and load balancing
    - Inter-agent communication
    - Performance tracking and optimization
    """
    
    def __init__(self):
        self.agents: Dict[str, SubagentInfo] = {}
        self.tasks: Dict[str, CoordinationTask] = {}
        self.coordination_log: List[Dict[str, Any]] = []
        self.health_check_interval = 30  # seconds
        self.stuck_detection_threshold = 300  # seconds
        self._running = False
        self.logger = logger.bind(component="SubagentCoordinator")
        
        # Performance metrics
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "agent_restarts": 0,
            "coordination_events": 0,
            "average_task_duration": 0.0
        }
    
    async def start(self):
        """Start the coordination system."""
        self._running = True
        self.logger.info("Subagent coordination system starting")
        
        # Start background tasks
        asyncio.create_task(self._health_monitor())
        asyncio.create_task(self._task_scheduler())
        asyncio.create_task(self._performance_tracker())
        
        console.print("🤖 [green]Subagent Coordination System Started[/green]")
    
    async def stop(self):
        """Stop the coordination system."""
        self._running = False
        self.logger.info("Subagent coordination system stopping")
        console.print("🛑 [yellow]Subagent Coordination System Stopped[/yellow]")
    
    async def register_agent(self, 
                           agent_id: str,
                           role: SubagentRole,
                           session_name: str,
                           workspace_path: str,
                           capabilities: List[str] = None) -> bool:
        """Register a new subagent with the coordinator."""
        try:
            agent_info = SubagentInfo(
                agent_id=agent_id,
                role=role,
                session_name=session_name,
                workspace_path=workspace_path,
                status=SubagentStatus.INITIALIZING,
                last_heartbeat=datetime.utcnow(),
                assigned_tasks=[],
                capabilities=capabilities or [],
                performance_metrics={},
                created_at=datetime.utcnow()
            )
            
            self.agents[agent_id] = agent_info
            self._log_event("agent_registered", {
                "agent_id": agent_id,
                "role": role.value,
                "session": session_name
            })
            
            console.print(f"✅ [green]Registered {role.value} agent: {agent_id}[/green]")
            return True
            
        except Exception as e:
            self.logger.error("Failed to register agent", 
                            agent_id=agent_id, error=str(e))
            return False
    
    async def create_task(self,
                         title: str,
                         description: str,
                         priority: TaskPriority,
                         required_roles: List[SubagentRole],
                         dependencies: List[str] = None,
                         estimated_duration: Optional[timedelta] = None) -> str:
        """Create a new coordination task."""
        task_id = f"task-{str(uuid.uuid4())[:8]}"
        
        task = CoordinationTask(
            task_id=task_id,
            title=title,
            description=description,
            priority=priority,
            required_roles=required_roles,
            assigned_agents=[],
            dependencies=dependencies or [],
            estimated_duration=estimated_duration,
            created_at=datetime.utcnow()
        )
        
        self.tasks[task_id] = task
        self._log_event("task_created", {
            "task_id": task_id,
            "title": title,
            "priority": priority.value,
            "roles_required": [r.value for r in required_roles]
        })
        
        console.print(f"📋 [cyan]Created task: {title} ({task_id})[/cyan]")
        return task_id
    
    async def assign_task(self, task_id: str, agent_ids: List[str] = None) -> bool:
        """Assign a task to specific agents or auto-assign to best available."""
        if task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        
        if agent_ids:
            # Manual assignment
            available_agents = [aid for aid in agent_ids if aid in self.agents]
        else:
            # Auto-assignment based on role requirements and availability
            available_agents = await self._find_best_agents(task)
        
        if not available_agents:
            self.logger.warning("No available agents for task", task_id=task_id)
            return False
        
        # Assign task to agents
        task.assigned_agents = available_agents
        task.status = "assigned"
        task.started_at = datetime.utcnow()
        
        for agent_id in available_agents:
            self.agents[agent_id].assigned_tasks.append(task_id)
        
        self._log_event("task_assigned", {
            "task_id": task_id,
            "assigned_agents": available_agents
        })
        
        console.print(f"🎯 [green]Assigned task {task_id} to agents: {', '.join(available_agents)}[/green]")
        return True
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        active_agents = [a for a in self.agents.values() if a.status == SubagentStatus.ACTIVE]
        stuck_agents = [a for a in self.agents.values() if a.status == SubagentStatus.STUCK]
        pending_tasks = [t for t in self.tasks.values() if t.status == "pending"]
        active_tasks = [t for t in self.tasks.values() if t.status == "assigned"]
        
        return {
            "coordination_active": self._running,
            "total_agents": len(self.agents),
            "active_agents": len(active_agents),
            "stuck_agents": len(stuck_agents),
            "total_tasks": len(self.tasks),
            "pending_tasks": len(pending_tasks),
            "active_tasks": len(active_tasks),
            "metrics": self.metrics,
            "agent_details": [asdict(a) for a in self.agents.values()],
            "task_details": [asdict(t) for t in self.tasks.values()]
        }
    
    async def display_status(self):
        """Display coordination system status in rich format."""
        status = await self.get_system_status()
        
        # Agent Status Table
        agent_table = Table(title="🤖 Subagent Status")
        agent_table.add_column("Agent ID", style="cyan", no_wrap=True)
        agent_table.add_column("Role", style="green")
        agent_table.add_column("Status", style="yellow")
        agent_table.add_column("Tasks", justify="center")
        agent_table.add_column("Last Heartbeat", style="dim")
        
        for agent in self.agents.values():
            status_icon = {
                SubagentStatus.ACTIVE: "🟢",
                SubagentStatus.IDLE: "🟡", 
                SubagentStatus.STUCK: "🔴",
                SubagentStatus.ERROR: "⚠️",
                SubagentStatus.OFFLINE: "⚫"
            }.get(agent.status, "❓")
            
            time_diff = datetime.utcnow() - agent.last_heartbeat
            heartbeat_str = f"{time_diff.seconds}s ago" if time_diff.seconds < 3600 else f"{time_diff.seconds//3600}h ago"
            
            agent_table.add_row(
                agent.agent_id[:8],
                agent.role.value,
                f"{status_icon} {agent.status.value}",
                str(len(agent.assigned_tasks)),
                heartbeat_str
            )
        
        # Task Status Table  
        task_table = Table(title="📋 Task Coordination")
        task_table.add_column("Task ID", style="cyan", no_wrap=True)
        task_table.add_column("Title", style="white")
        task_table.add_column("Priority", style="red")
        task_table.add_column("Status", style="green") 
        task_table.add_column("Assigned Agents", style="yellow")
        
        for task in list(self.tasks.values())[-10:]:  # Show last 10 tasks
            priority_icon = {
                TaskPriority.CRITICAL: "🚨",
                TaskPriority.HIGH: "🔥", 
                TaskPriority.MEDIUM: "📋",
                TaskPriority.LOW: "📝"
            }.get(task.priority, "❓")
            
            task_table.add_row(
                task.task_id[:8],
                task.title[:30] + ("..." if len(task.title) > 30 else ""),
                f"{priority_icon} {task.priority.value}",
                task.status,
                ", ".join([aid[:8] for aid in task.assigned_agents])
            )
        
        console.print(agent_table)
        console.print(task_table)
        
        # System Metrics
        metrics_panel = Panel(
            f"Tasks Completed: {self.metrics['tasks_completed']} | "
            f"Tasks Failed: {self.metrics['tasks_failed']} | "
            f"Agent Restarts: {self.metrics['agent_restarts']} | "
            f"Coordination Events: {self.metrics['coordination_events']}",
            title="📊 System Metrics",
            border_style="blue"
        )
        console.print(metrics_panel)
    
    async def _health_monitor(self):
        """Background health monitoring for all agents."""
        while self._running:
            try:
                current_time = datetime.utcnow()
                
                for agent_id, agent in self.agents.items():
                    time_since_heartbeat = current_time - agent.last_heartbeat
                    
                    # Check if agent is stuck
                    if time_since_heartbeat.seconds > self.stuck_detection_threshold:
                        if agent.status != SubagentStatus.STUCK:
                            agent.status = SubagentStatus.STUCK
                            self._log_event("agent_stuck", {"agent_id": agent_id})
                            await self._attempt_agent_recovery(agent_id)
                    
                    # Update heartbeat via tmux session check
                    await self._update_agent_heartbeat(agent_id)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error("Health monitor error", error=str(e))
                await asyncio.sleep(5)
    
    async def _task_scheduler(self):
        """Background task scheduling and assignment."""
        while self._running:
            try:
                # Find pending tasks that can be assigned
                pending_tasks = [t for t in self.tasks.values() 
                               if t.status == "pending" and self._dependencies_met(t)]
                
                for task in pending_tasks:
                    await self.assign_task(task.task_id)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error("Task scheduler error", error=str(e))
                await asyncio.sleep(5)
    
    async def _performance_tracker(self):
        """Track and update performance metrics."""
        while self._running:
            try:
                # Update completion metrics
                completed_tasks = [t for t in self.tasks.values() if t.completed_at]
                self.metrics["tasks_completed"] = len(completed_tasks)
                
                # Calculate average task duration
                if completed_tasks:
                    durations = [(t.completed_at - t.started_at).seconds 
                               for t in completed_tasks if t.started_at]
                    if durations:
                        self.metrics["average_task_duration"] = sum(durations) / len(durations)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error("Performance tracker error", error=str(e))
                await asyncio.sleep(10)
    
    async def _find_best_agents(self, task: CoordinationTask) -> List[str]:
        """Find the best available agents for a task."""
        available_agents = []
        
        for role in task.required_roles:
            # Find agents with matching role that are available
            role_agents = [
                agent_id for agent_id, agent in self.agents.items()
                if agent.role == role and agent.status in [SubagentStatus.ACTIVE, SubagentStatus.IDLE]
                and len(agent.assigned_tasks) < 3  # Max 3 concurrent tasks per agent
            ]
            
            if role_agents:
                # Sort by performance and availability
                best_agent = min(role_agents, key=lambda aid: len(self.agents[aid].assigned_tasks))
                available_agents.append(best_agent)
        
        return available_agents
    
    async def _update_agent_heartbeat(self, agent_id: str):
        """Update agent heartbeat by checking tmux session."""
        try:
            agent = self.agents[agent_id]
            
            # Check if tmux session is active
            import subprocess
            result = subprocess.run(
                ["tmux", "list-sessions", "-F", "#{session_name}"],
                capture_output=True, text=True
            )
            
            if agent.session_name in result.stdout:
                agent.last_heartbeat = datetime.utcnow()
                if agent.status == SubagentStatus.OFFLINE:
                    agent.status = SubagentStatus.ACTIVE
                    
        except Exception as e:
            self.logger.error("Failed to update heartbeat", 
                            agent_id=agent_id, error=str(e))
    
    async def _attempt_agent_recovery(self, agent_id: str):
        """Attempt to recover a stuck agent."""
        agent = self.agents[agent_id]
        self.logger.info("Attempting agent recovery", agent_id=agent_id)
        
        try:
            # Send interrupt to tmux session to break out of stuck commands
            import subprocess
            subprocess.run([
                "tmux", "send-keys", "-t", agent.session_name, "C-c"
            ], capture_output=True)
            
            # Wait a bit and check status
            await asyncio.sleep(2)
            
            # Send a simple command to test responsiveness
            subprocess.run([
                "tmux", "send-keys", "-t", agent.session_name, 
                "echo 'Agent recovery check'", "Enter"
            ], capture_output=True)
            
            agent.restart_count += 1
            agent.status = SubagentStatus.ACTIVE
            self.metrics["agent_restarts"] += 1
            
            self._log_event("agent_recovered", {
                "agent_id": agent_id,
                "restart_count": agent.restart_count
            })
            
            console.print(f"🔧 [yellow]Recovered stuck agent: {agent_id}[/yellow]")
            
        except Exception as e:
            self.logger.error("Agent recovery failed", 
                            agent_id=agent_id, error=str(e))
            agent.status = SubagentStatus.ERROR
            agent.error_count += 1
    
    def _dependencies_met(self, task: CoordinationTask) -> bool:
        """Check if all task dependencies are completed."""
        if not task.dependencies:
            return True
            
        for dep_id in task.dependencies:
            if dep_id not in self.tasks or self.tasks[dep_id].status != "completed":
                return False
        
        return True
    
    def _log_event(self, event_type: str, details: Dict[str, Any]):
        """Log coordination events."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        self.coordination_log.append(event)
        self.metrics["coordination_events"] += 1
        
        # Keep only last 1000 events
        if len(self.coordination_log) > 1000:
            self.coordination_log = self.coordination_log[-1000:]


# Global coordinator instance
_coordinator_instance: Optional[SubagentCoordinator] = None

def get_coordinator() -> SubagentCoordinator:
    """Get the global coordinator instance."""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = SubagentCoordinator()
    return _coordinator_instance


async def initialize_coordination_system():
    """Initialize and start the coordination system."""
    coordinator = get_coordinator()
    await coordinator.start()
    return coordinator