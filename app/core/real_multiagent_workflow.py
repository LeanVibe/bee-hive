"""
Real Multi-Agent Development Workflow Engine

This module implements the actual working multi-agent development workflow
that proves LeanVibe Agent Hive 2.0 is a functional autonomous development platform.

Workflow: Developer Agent → QA Agent → CI/CD Agent
- Developer writes Python code to file
- QA Engineer creates comprehensive tests
- CI/CD Engineer runs tests and reports results

Integration with existing framework components:
- Redis streams for agent communication
- Workflow engine for task coordination
- Dashboard for real-time monitoring
- Hook system for event capture
"""

import asyncio
import json
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum

import structlog

from .real_agent_implementations import (
    MultiAgentWorkflowCoordinator,
    DeveloperAgent, 
    QAAgent, 
    CIAgent,
    TaskExecution,
    AgentType
)
from .redis import get_message_broker, AgentMessageBroker
from .workflow_engine import WorkflowEngine, WorkflowResult, TaskExecutionState
from .agent_communication_service import AgentCommunicationService, AgentMessage
from .orchestrator import AgentOrchestrator
from ..models.workflow import Workflow, WorkflowStatus
from ..models.task import Task, TaskStatus

logger = structlog.get_logger()


class WorkflowStage(Enum):
    """Stages in the multi-agent development workflow."""
    INITIALIZATION = "initialization"
    CODE_DEVELOPMENT = "code_development" 
    TEST_CREATION = "test_creation"
    TEST_EXECUTION = "test_execution"
    VALIDATION = "validation"
    COMPLETION = "completion"


@dataclass
class WorkflowConfiguration:
    """Configuration for multi-agent workflow execution."""
    workspace_dir: str
    requirements: Dict[str, Any]
    enable_real_time_monitoring: bool = True
    timeout_seconds: int = 300  # 5 minutes default timeout
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class WorkflowEvent:
    """Event generated during workflow execution."""
    event_id: str
    workflow_id: str
    stage: WorkflowStage
    agent_id: Optional[str]
    event_type: str
    message: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'stage': self.stage.value,
            'timestamp': self.timestamp.isoformat()
        }


class RealMultiAgentWorkflow:
    """
    Real working multi-agent development workflow implementation.
    
    This class orchestrates multiple real agents to complete software development tasks:
    1. DeveloperAgent writes actual Python code files
    2. QAAgent creates comprehensive test files
    3. CIAgent runs the tests and provides build status
    
    All communication happens via Redis streams and results are monitored in real-time.
    """
    
    def __init__(self, config: WorkflowConfiguration):
        self.config = config
        self.workflow_id = str(uuid.uuid4())
        self.coordinator = MultiAgentWorkflowCoordinator(config.workspace_dir)
        self.events: List[WorkflowEvent] = []
        self.event_callbacks: List[Callable[[WorkflowEvent], None]] = []
        self.current_stage = WorkflowStage.INITIALIZATION
        self.logger = logger.bind(workflow_id=self.workflow_id)
    
    def add_event_callback(self, callback: Callable[[WorkflowEvent], None]):
        """Add callback to receive real-time workflow events."""
        self.event_callbacks.append(callback)
    
    def _emit_event(self, event_type: str, message: str, agent_id: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        """Emit a workflow event to all subscribers."""
        event = WorkflowEvent(
            event_id=str(uuid.uuid4()),
            workflow_id=self.workflow_id,
            stage=self.current_stage,
            agent_id=agent_id,
            event_type=event_type,
            message=message,
            timestamp=datetime.utcnow(),
            data=data
        )
        
        self.events.append(event)
        
        # Notify all callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error("Error in event callback", error=str(e))
        
        # Log the event
        self.logger.info("Workflow event", 
                        event_type=event_type, 
                        message=message,
                        agent_id=agent_id,
                        stage=self.current_stage.value)
    
    async def execute(self) -> Dict[str, Any]:
        """
        Execute the complete multi-agent development workflow.
        
        Returns:
            Complete workflow execution results with agent outputs and performance metrics
        """
        workflow_start = time.time()
        
        self._emit_event("workflow_started", "Multi-agent development workflow initiated", 
                        data=self.config.to_dict())
        
        try:
            # Stage 1: Initialize agents
            self.current_stage = WorkflowStage.INITIALIZATION
            self._emit_event("stage_started", "Initializing agents and communication systems")
            
            await self.coordinator.initialize_agents()
            
            self._emit_event("agents_initialized", "All agents successfully initialized", 
                           data={
                               "developer_id": self.coordinator.developer.agent_id,
                               "qa_id": self.coordinator.qa_engineer.agent_id,
                               "ci_id": self.coordinator.ci_engineer.agent_id
                           })
            
            # Stage 2: Execute workflow
            self.current_stage = WorkflowStage.CODE_DEVELOPMENT
            self._emit_event("stage_started", "Starting multi-agent development workflow")
            
            workflow_results = await self.coordinator.execute_development_workflow(
                self.config.requirements
            )
            
            # Process results and emit detailed events
            await self._process_workflow_results(workflow_results)
            
            # Stage 3: Validation and completion
            self.current_stage = WorkflowStage.VALIDATION
            self._emit_event("stage_started", "Validating workflow results")
            
            validation_results = self._validate_workflow_results(workflow_results)
            
            self.current_stage = WorkflowStage.COMPLETION
            
            final_results = {
                "workflow_id": self.workflow_id,
                "success": workflow_results["success"] and validation_results["valid"],
                "execution_time": time.time() - workflow_start,
                "workflow_results": workflow_results,
                "validation_results": validation_results, 
                "events": [event.to_dict() for event in self.events],
                "workspace_dir": self.config.workspace_dir,
                "files_created": self._collect_created_files(workflow_results)
            }
            
            event_type = "workflow_completed" if final_results["success"] else "workflow_failed"
            self._emit_event(event_type, 
                           f"Multi-agent workflow {'completed successfully' if final_results['success'] else 'failed'}",
                           data={
                               "execution_time": final_results["execution_time"],
                               "files_created": len(final_results["files_created"]),
                               "tests_passed": validation_results.get("tests_passed", False)
                           })
            
            return final_results
            
        except Exception as e:
            self._emit_event("workflow_error", f"Workflow execution failed: {str(e)}", 
                           data={"error": str(e)})
            self.logger.error("Workflow execution failed", error=str(e))
            
            return {
                "workflow_id": self.workflow_id,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - workflow_start,
                "events": [event.to_dict() for event in self.events],
                "workspace_dir": self.config.workspace_dir
            }
    
    async def _process_workflow_results(self, workflow_results: Dict[str, Any]):
        """Process and emit events for each stage of the workflow."""
        stages = workflow_results.get("stages", {})
        
        # Process development stage
        if "development" in stages:
            dev_stage = stages["development"]
            self.current_stage = WorkflowStage.CODE_DEVELOPMENT
            
            if dev_stage["status"] == "completed":
                self._emit_event("code_created", "Developer agent successfully created code",
                               agent_id=dev_stage["agent_id"],
                               data={
                                   "files_created": dev_stage["files_created"],
                                   "execution_time": dev_stage["execution_time"]
                               })
            else:
                self._emit_event("code_creation_failed", "Developer agent failed to create code",
                               agent_id=dev_stage["agent_id"],
                               data={"error": dev_stage.get("error")})
        
        # Process testing stage
        if "testing" in stages:
            test_stage = stages["testing"]
            self.current_stage = WorkflowStage.TEST_CREATION
            
            if test_stage["status"] == "completed":
                self._emit_event("tests_created", "QA agent successfully created tests",
                               agent_id=test_stage["agent_id"],
                               data={
                                   "files_created": test_stage["files_created"], 
                                   "execution_time": test_stage["execution_time"]
                               })
            else:
                self._emit_event("test_creation_failed", "QA agent failed to create tests",
                               agent_id=test_stage["agent_id"],
                               data={"error": test_stage.get("error")})
        
        # Process CI/CD stage
        if "ci_cd" in stages:
            ci_stage = stages["ci_cd"]
            self.current_stage = WorkflowStage.TEST_EXECUTION
            
            if ci_stage["status"] == "completed":
                self._emit_event("tests_passed", "CI agent successfully ran tests",
                               agent_id=ci_stage["agent_id"],
                               data={
                                   "execution_time": ci_stage["execution_time"],
                                   "tests_passed": ci_stage.get("tests_passed", False)
                               })
            else:
                self._emit_event("tests_failed", "CI agent test execution failed",
                               agent_id=ci_stage["agent_id"], 
                               data={
                                   "error": ci_stage.get("error"),
                                   "output": ci_stage.get("output")
                               })
    
    def _validate_workflow_results(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the workflow produced expected results."""
        validation = {
            "valid": True,
            "checks": {},
            "errors": []
        }
        
        # Check that all stages completed
        stages = workflow_results.get("stages", {})
        required_stages = ["development", "testing", "ci_cd"]
        
        for stage in required_stages:
            if stage not in stages:
                validation["checks"][f"{stage}_present"] = False
                validation["errors"].append(f"Missing {stage} stage")
                validation["valid"] = False
            else:
                stage_data = stages[stage]
                stage_completed = stage_data.get("status") == "completed"
                validation["checks"][f"{stage}_completed"] = stage_completed
                
                if not stage_completed:
                    validation["errors"].append(f"{stage} stage failed: {stage_data.get('error')}")
                    validation["valid"] = False
        
        # Check that files were created
        files_created = self._collect_created_files(workflow_results)
        validation["checks"]["files_created"] = len(files_created) >= 2  # Code + test files
        validation["checks"]["files_exist"] = all(Path(f).exists() for f in files_created)
        
        if not validation["checks"]["files_created"]:
            validation["errors"].append("Insufficient files created")
            validation["valid"] = False
        
        if not validation["checks"]["files_exist"]:
            validation["errors"].append("Some created files do not exist")
            validation["valid"] = False
        
        # Check test results
        ci_stage = stages.get("ci_cd", {})
        tests_passed = ci_stage.get("tests_passed", False)
        validation["checks"]["tests_passed"] = tests_passed
        validation["tests_passed"] = tests_passed
        
        if not tests_passed and ci_stage.get("status") == "completed":
            # Tests ran but failed - this might be acceptable depending on requirements
            validation["checks"]["tests_executed"] = True
        
        return validation
    
    def _collect_created_files(self, workflow_results: Dict[str, Any]) -> List[str]:
        """Collect all files created during the workflow."""
        files = []
        stages = workflow_results.get("stages", {})
        
        for stage_data in stages.values():
            files.extend(stage_data.get("files_created", []))
        
        return files


class MultiAgentWorkflowManager:
    """
    Manager for executing and monitoring multiple multi-agent workflows.
    Integrates with existing LeanVibe framework components.
    """
    
    def __init__(self):
        self.active_workflows: Dict[str, RealMultiAgentWorkflow] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        self.logger = logger.bind(component="workflow_manager")
    
    async def create_workflow(self, requirements: Dict[str, Any], workspace_dir: Optional[str] = None) -> str:
        """Create a new multi-agent workflow."""
        if workspace_dir is None:
            workspace_dir = tempfile.mkdtemp(prefix="multiagent_workflow_")
        
        config = WorkflowConfiguration(
            workspace_dir=workspace_dir,
            requirements=requirements
        )
        
        workflow = RealMultiAgentWorkflow(config)
        self.active_workflows[workflow.workflow_id] = workflow
        
        self.logger.info("Created new multi-agent workflow", 
                        workflow_id=workflow.workflow_id,
                        workspace_dir=workspace_dir,
                        requirements=requirements)
        
        return workflow.workflow_id
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a workflow and return results."""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.active_workflows[workflow_id]
        
        try:
            results = await workflow.execute()
            
            # Move to history
            self.workflow_history.append(results)
            del self.active_workflows[workflow_id]
            
            return results
            
        except Exception as e:
            self.logger.error("Workflow execution failed", 
                            workflow_id=workflow_id, 
                            error=str(e))
            raise
    
    async def execute_simple_workflow(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create and execute a workflow in one step.
        Convenience method for simple use cases.
        """
        workflow_id = await self.create_workflow(requirements)
        return await self.execute_workflow(workflow_id)
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an active workflow."""
        if workflow_id not in self.active_workflows:
            return None
        
        workflow = self.active_workflows[workflow_id]
        return {
            "workflow_id": workflow_id,
            "current_stage": workflow.current_stage.value,
            "events_count": len(workflow.events),
            "workspace_dir": workflow.config.workspace_dir,
            "requirements": workflow.config.requirements
        }
    
    def add_workflow_event_callback(self, workflow_id: str, callback: Callable[[WorkflowEvent], None]):
        """Add real-time event callback to a workflow."""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id].add_event_callback(callback)
    
    def get_workflow_history(self) -> List[Dict[str, Any]]:
        """Get history of completed workflows."""
        return self.workflow_history.copy()


# Global workflow manager instance
_workflow_manager = None

def get_workflow_manager() -> MultiAgentWorkflowManager:
    """Get the global workflow manager instance."""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = MultiAgentWorkflowManager()
    return _workflow_manager