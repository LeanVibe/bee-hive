"""
Project Task Execution Bridge for LeanVibe Agent Hive 2.0

Bridges the gap between project management tasks and SimpleOrchestrator agent execution.
Provides seamless integration between hierarchical project planning and actual agent execution.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import structlog
from sqlalchemy.orm import Session

from .simple_orchestrator import SimpleOrchestrator, AgentRole
from .project_management_orchestrator_integration import ProjectManagementOrchestratorIntegration
from .enhanced_agent_launcher import AgentLauncherType, AgentLaunchConfig, Priority
from .kanban_state_machine import KanbanState
from ..models.project_management import ProjectTask, TaskType, TaskPriority

logger = structlog.get_logger(__name__)

@dataclass
class TaskExecutionResult:
    """Result of project task execution through SimpleOrchestrator."""
    success: bool
    agent_id: Optional[str] = None
    session_name: Optional[str] = None
    orchestrator_task_id: Optional[str] = None
    error_message: Optional[str] = None
    execution_details: Dict[str, Any] = None

class ProjectTaskExecutionBridge:
    """
    Bridge between project tasks and SimpleOrchestrator execution.
    
    Handles:
    - Converting project tasks to orchestrator-executable format
    - Agent auto-spawning based on task requirements
    - Real-time execution monitoring and state synchronization
    - Results propagation back to project management system
    """
    
    def __init__(self, 
                 orchestrator: SimpleOrchestrator,
                 pm_integration: ProjectManagementOrchestratorIntegration,
                 db_session: Session):
        """
        Initialize the execution bridge.
        
        Args:
            orchestrator: SimpleOrchestrator instance
            pm_integration: Project management integration
            db_session: Database session
        """
        self.orchestrator = orchestrator
        self.pm_integration = pm_integration
        self.db_session = db_session
        self.logger = logger.bind(component="ProjectTaskExecutionBridge")
        
        # Track active executions
        self.active_executions: Dict[str, TaskExecutionResult] = {}
    
    async def execute_project_task(self, 
                                 task_id: uuid.UUID,
                                 auto_spawn_agent: bool = True,
                                 preferred_agent_type: Optional[AgentLauncherType] = None) -> TaskExecutionResult:
        """
        Execute a project task through SimpleOrchestrator.
        
        Args:
            task_id: Project task UUID
            auto_spawn_agent: Whether to auto-spawn agent if none available
            preferred_agent_type: Preferred agent type for execution
            
        Returns:
            TaskExecutionResult with execution details
        """
        self.logger.info("Executing project task", task_id=str(task_id))
        
        try:
            # Get project task
            project_task = self.db_session.query(ProjectTask).filter(
                ProjectTask.id == task_id
            ).first()
            
            if not project_task:
                return TaskExecutionResult(
                    success=False,
                    error_message=f"Project task not found: {task_id}"
                )
            
            # Convert to orchestrator task format
            orchestrator_task = self._convert_to_orchestrator_task(project_task)
            
            # Determine agent requirements
            agent_requirements = self._determine_agent_requirements(
                project_task, preferred_agent_type
            )
            
            # Execute through orchestrator
            if auto_spawn_agent:
                result = await self._execute_with_auto_spawn(
                    project_task, orchestrator_task, agent_requirements
                )
            else:
                result = await self._execute_with_existing_agents(
                    project_task, orchestrator_task, agent_requirements
                )
            
            # Update project task state
            if result.success:
                await self._update_project_task_state(
                    project_task, result, KanbanState.IN_PROGRESS
                )
            
            # Track execution
            self.active_executions[str(task_id)] = result
            
            return result
            
        except Exception as e:
            self.logger.error("Failed to execute project task", 
                            task_id=str(task_id), error=str(e))
            return TaskExecutionResult(
                success=False,
                error_message=f"Execution failed: {str(e)}"
            )
    
    async def monitor_task_execution(self, task_id: uuid.UUID) -> Dict[str, Any]:
        """
        Monitor the execution status of a project task.
        
        Args:
            task_id: Project task UUID
            
        Returns:
            Current execution status and metrics
        """
        task_key = str(task_id)
        
        if task_key not in self.active_executions:
            return {"status": "not_found", "message": "Task execution not tracked"}
        
        execution = self.active_executions[task_key]
        
        if not execution.agent_id:
            return {"status": "no_agent", "execution": execution}
        
        # Get agent status from orchestrator
        agent_status = await self.orchestrator.get_agent_status(execution.agent_id)
        
        return {
            "status": "active",
            "execution": execution,
            "agent_status": agent_status,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def complete_task_execution(self, 
                                    task_id: uuid.UUID,
                                    success: bool,
                                    results: Optional[Dict[str, Any]] = None) -> bool:
        """
        Mark a project task execution as completed and sync results.
        
        Args:
            task_id: Project task UUID
            success: Whether execution was successful
            results: Execution results to record
            
        Returns:
            Whether completion was successful
        """
        try:
            # Get project task
            project_task = self.db_session.query(ProjectTask).filter(
                ProjectTask.id == task_id
            ).first()
            
            if not project_task:
                return False
            
            # Update task state based on success
            new_state = KanbanState.DONE if success else KanbanState.BACKLOG
            
            await self._update_project_task_state(
                project_task, 
                self.active_executions.get(str(task_id)),
                new_state
            )
            
            # Record results if provided
            if results:
                project_task.completion_notes = results.get('completion_notes', '')
                self.db_session.commit()
            
            # Clean up tracking
            if str(task_id) in self.active_executions:
                del self.active_executions[str(task_id)]
            
            self.logger.info("Task execution completed", 
                           task_id=str(task_id), success=success)
            return True
            
        except Exception as e:
            self.logger.error("Failed to complete task execution", 
                            task_id=str(task_id), error=str(e))
            return False
    
    def _convert_to_orchestrator_task(self, project_task: ProjectTask) -> Dict[str, Any]:
        """
        Convert project task to SimpleOrchestrator task format.
        
        Args:
            project_task: ProjectTask instance
            
        Returns:
            Task in orchestrator format
        """
        # Build task context from hierarchy
        context = {
            "project_task_id": str(project_task.id),
            "task_title": project_task.title,
            "task_description": project_task.description or "",
            "task_type": project_task.task_type.value,
            "priority": project_task.priority.name,
        }
        
        # Add PRD context if available
        if project_task.prd:
            context.update({
                "prd_title": project_task.prd.title,
                "prd_description": project_task.prd.description or "",
                "epic_name": project_task.prd.epic.name if project_task.prd.epic else "",
                "project_name": project_task.prd.epic.project.name if project_task.prd.epic and project_task.prd.epic.project else ""
            })
        
        # Build orchestrator task
        orchestrator_task = {
            "id": str(uuid.uuid4()),
            "title": project_task.title,
            "description": self._build_task_description(project_task, context),
            "priority": self._convert_priority(project_task.priority),
            "estimated_duration": project_task.estimated_effort_minutes,
            "context": context,
            "workspace": f"task-{project_task.get_display_id()}" if hasattr(project_task, 'get_display_id') else f"task-{str(project_task.id)[:8]}"
        }
        
        return orchestrator_task
    
    def _determine_agent_requirements(self, 
                                    project_task: ProjectTask,
                                    preferred_type: Optional[AgentLauncherType] = None) -> Dict[str, Any]:
        """
        Determine agent requirements for task execution.
        
        Args:
            project_task: ProjectTask instance
            preferred_type: Preferred agent type
            
        Returns:
            Agent requirements specification
        """
        # Map task types to agent roles
        task_type_to_agent = {
            TaskType.FEATURE_DEVELOPMENT: AgentLauncherType.CLAUDE_CODE,
            TaskType.BUG_FIX: AgentLauncherType.CLAUDE_CODE,
            TaskType.TESTING: AgentLauncherType.CLAUDE_CODE,  # Could be specialized QA agent
            TaskType.DOCUMENTATION: AgentLauncherType.CLAUDE_CODE,
            TaskType.RESEARCH: AgentLauncherType.CLAUDE_CODE,
            TaskType.CODE_REVIEW: AgentLauncherType.CLAUDE_CODE,
            TaskType.DEPLOYMENT: AgentLauncherType.TMUX_SESSION,  # For DevOps tasks
            TaskType.MAINTENANCE: AgentLauncherType.CLAUDE_CODE,
        }
        
        # Determine agent type
        agent_type = preferred_type or task_type_to_agent.get(
            project_task.task_type, AgentLauncherType.CLAUDE_CODE
        )
        
        # Map priority to agent launch priority
        priority_mapping = {
            TaskPriority.CRITICAL: Priority.HIGH,
            TaskPriority.HIGH: Priority.HIGH,
            TaskPriority.MEDIUM: Priority.MEDIUM,
            TaskPriority.LOW: Priority.LOW
        }
        
        return {
            "agent_type": agent_type,
            "priority": priority_mapping.get(project_task.priority, Priority.MEDIUM),
            "estimated_effort": project_task.estimated_effort_minutes,
            "required_capabilities": self._extract_required_capabilities(project_task)
        }
    
    def _extract_required_capabilities(self, project_task: ProjectTask) -> List[str]:
        """Extract required capabilities from task details."""
        capabilities = []
        
        # Based on task type
        task_capabilities = {
            TaskType.FEATURE_DEVELOPMENT: ["development", "coding", "testing"],
            TaskType.BUG_FIX: ["debugging", "testing", "code_analysis"],
            TaskType.TESTING: ["testing", "quality_assurance", "automation"],
            TaskType.DOCUMENTATION: ["documentation", "writing", "technical_writing"],
            TaskType.RESEARCH: ["research", "analysis", "documentation"],
            TaskType.CODE_REVIEW: ["code_analysis", "review", "quality_assurance"],
            TaskType.DEPLOYMENT: ["deployment", "devops", "automation"],
            TaskType.MAINTENANCE: ["maintenance", "monitoring", "analysis"],
        }
        
        capabilities.extend(task_capabilities.get(project_task.task_type, []))
        
        # Add language-specific capabilities based on description
        if project_task.description:
            desc_lower = project_task.description.lower()
            language_keywords = {
                "python": ["python", "django", "flask", "fastapi"],
                "javascript": ["javascript", "node", "react", "vue", "angular"],
                "typescript": ["typescript", "ts"],
                "rust": ["rust"],
                "go": ["golang", "go"],
                "sql": ["database", "sql", "postgresql", "mysql"]
            }
            
            for lang, keywords in language_keywords.items():
                if any(keyword in desc_lower for keyword in keywords):
                    capabilities.append(f"language_{lang}")
        
        return capabilities
    
    async def _execute_with_auto_spawn(self, 
                                     project_task: ProjectTask,
                                     orchestrator_task: Dict[str, Any],
                                     agent_requirements: Dict[str, Any]) -> TaskExecutionResult:
        """
        Execute task with automatic agent spawning.
        
        Args:
            project_task: ProjectTask instance
            orchestrator_task: Task in orchestrator format
            agent_requirements: Agent requirements
            
        Returns:
            TaskExecutionResult
        """
        try:
            # Create agent launch config
            config = AgentLaunchConfig(
                agent_type=agent_requirements["agent_type"],
                task_id=orchestrator_task["id"],
                workspace_name=orchestrator_task["workspace"],
                priority=agent_requirements["priority"],
                environment_vars={
                    "PROJECT_TASK_ID": str(project_task.id),
                    "TASK_TITLE": project_task.title,
                    "TASK_TYPE": project_task.task_type.value
                }
            )
            
            # Spawn agent through orchestrator
            agent_result = await self.orchestrator.spawn_agent_advanced(config)
            
            if not agent_result or not agent_result.get("agent_id"):
                return TaskExecutionResult(
                    success=False,
                    error_message="Failed to spawn agent"
                )
            
            # Delegate task to spawned agent
            delegation_result = await self.orchestrator.delegate_task(
                orchestrator_task, 
                agent_result["agent_id"]
            )
            
            return TaskExecutionResult(
                success=True,
                agent_id=agent_result["agent_id"],
                session_name=agent_result.get("session_name"),
                orchestrator_task_id=orchestrator_task["id"],
                execution_details={
                    "spawn_result": agent_result,
                    "delegation_result": delegation_result,
                    "agent_requirements": agent_requirements
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error_message=f"Auto-spawn execution failed: {str(e)}"
            )
    
    async def _execute_with_existing_agents(self, 
                                          project_task: ProjectTask,
                                          orchestrator_task: Dict[str, Any],
                                          agent_requirements: Dict[str, Any]) -> TaskExecutionResult:
        """
        Execute task with existing agents.
        
        Args:
            project_task: ProjectTask instance
            orchestrator_task: Task in orchestrator format
            agent_requirements: Agent requirements
            
        Returns:
            TaskExecutionResult
        """
        try:
            # Find suitable agent
            suitable_agent = await self.orchestrator.find_suitable_agent(
                agent_requirements["required_capabilities"]
            )
            
            if not suitable_agent:
                return TaskExecutionResult(
                    success=False,
                    error_message="No suitable agents available"
                )
            
            # Delegate task
            delegation_result = await self.orchestrator.delegate_task(
                orchestrator_task,
                suitable_agent["agent_id"]
            )
            
            return TaskExecutionResult(
                success=True,
                agent_id=suitable_agent["agent_id"],
                orchestrator_task_id=orchestrator_task["id"],
                execution_details={
                    "agent_selection": suitable_agent,
                    "delegation_result": delegation_result,
                    "agent_requirements": agent_requirements
                }
            )
            
        except Exception as e:
            return TaskExecutionResult(
                success=False,
                error_message=f"Existing agent execution failed: {str(e)}"
            )
    
    async def _update_project_task_state(self, 
                                       project_task: ProjectTask,
                                       execution_result: Optional[TaskExecutionResult],
                                       new_state: KanbanState) -> bool:
        """
        Update project task state and assignment.
        
        Args:
            project_task: ProjectTask instance
            execution_result: Execution result
            new_state: New Kanban state
            
        Returns:
            Whether update was successful
        """
        try:
            # Update state through PM integration
            transition_result = await self.pm_integration.transition_task_state(
                project_task, new_state
            )
            
            # Update agent assignment if execution was successful
            if execution_result and execution_result.success and execution_result.agent_id:
                # Convert agent ID format for PM system
                agent_uuid = self._convert_agent_id_to_uuid(execution_result.agent_id)
                project_task.assigned_agent_id = agent_uuid
                
                # Add execution metadata
                if not hasattr(project_task, 'execution_metadata'):
                    project_task.execution_metadata = {}
                
                project_task.execution_metadata.update({
                    "orchestrator_agent_id": execution_result.agent_id,
                    "session_name": execution_result.session_name,
                    "orchestrator_task_id": execution_result.orchestrator_task_id,
                    "execution_started": datetime.utcnow().isoformat()
                })
            
            self.db_session.commit()
            return True
            
        except Exception as e:
            self.logger.error("Failed to update project task state", 
                            project_task_id=str(project_task.id), error=str(e))
            self.db_session.rollback()
            return False
    
    def _build_task_description(self, project_task: ProjectTask, context: Dict[str, Any]) -> str:
        """Build comprehensive task description for agent execution."""
        description_parts = []
        
        # Add hierarchy context
        if context.get("project_name"):
            description_parts.append(f"**Project**: {context['project_name']}")
        if context.get("epic_name"):
            description_parts.append(f"**Epic**: {context['epic_name']}")
        if context.get("prd_title"):
            description_parts.append(f"**PRD**: {context['prd_title']}")
        
        # Add task details
        description_parts.append(f"**Task**: {project_task.title}")
        
        if project_task.description:
            description_parts.append(f"**Description**: {project_task.description}")
        
        # Add task type and priority
        description_parts.append(f"**Type**: {project_task.task_type.value}")
        description_parts.append(f"**Priority**: {project_task.priority.name}")
        
        # Add effort estimate
        if project_task.estimated_effort_minutes:
            hours = project_task.estimated_effort_minutes // 60
            minutes = project_task.estimated_effort_minutes % 60
            if hours > 0:
                description_parts.append(f"**Estimated Effort**: {hours}h {minutes}m")
            else:
                description_parts.append(f"**Estimated Effort**: {minutes}m")
        
        # Add context from PRD if available
        if context.get("prd_description"):
            description_parts.append(f"**PRD Context**: {context['prd_description']}")
        
        return "\n\n".join(description_parts)
    
    def _convert_priority(self, task_priority: TaskPriority) -> str:
        """Convert task priority to orchestrator format."""
        priority_mapping = {
            TaskPriority.CRITICAL: "critical",
            TaskPriority.HIGH: "high", 
            TaskPriority.MEDIUM: "medium",
            TaskPriority.LOW: "low"
        }
        return priority_mapping.get(task_priority, "medium")
    
    def _convert_agent_id_to_uuid(self, orchestrator_agent_id: str) -> uuid.UUID:
        """
        Convert orchestrator agent ID to UUID format for PM system.
        
        For now, generate a deterministic UUID based on the agent ID.
        In the future, this could map to actual agent UUIDs.
        """
        import hashlib
        # Generate deterministic UUID from agent ID
        hash_obj = hashlib.md5(orchestrator_agent_id.encode())
        hex_dig = hash_obj.hexdigest()
        return uuid.UUID(hex_dig)
    
    async def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all active task executions."""
        summary = {
            "total_active_executions": len(self.active_executions),
            "successful_executions": len([e for e in self.active_executions.values() if e.success]),
            "failed_executions": len([e for e in self.active_executions.values() if not e.success]),
            "executions": {}
        }
        
        for task_id, execution in self.active_executions.items():
            summary["executions"][task_id] = {
                "success": execution.success,
                "agent_id": execution.agent_id,
                "session_name": execution.session_name,
                "has_error": execution.error_message is not None
            }
        
        return summary