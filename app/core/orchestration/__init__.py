"""
Enhanced Orchestrator for Multi-CLI Agent Coordination

This module provides the core orchestration capabilities for coordinating
multiple CLI agents (Claude Code, Cursor, Gemini CLI, etc.) in complex
workflows with intelligent task routing and execution monitoring.

Architecture Components:
- UniversalOrchestrator: Core coordination engine
- TaskRouter: Intelligent capability-based task routing  
- WorkflowCoordinator: Multi-step workflow execution
- ExecutionMonitor: Cross-agent execution tracking
- FailureRecovery: Automatic error recovery and task reassignment

Key Features:
- Heterogeneous CLI agent coordination
- Intelligent task routing based on agent capabilities
- Multi-step workflow execution with dependency management
- Real-time execution monitoring and progress tracking
- Automatic failure recovery and task reassignment
- Resource optimization and load balancing
"""

from .universal_orchestrator import UniversalOrchestrator
from .task_router import TaskRouter
from .workflow_coordinator import WorkflowCoordinator
from .execution_monitor import ExecutionMonitor
from .orchestration_models import (
    OrchestrationRequest,
    OrchestrationResult, 
    WorkflowDefinition,
    WorkflowResult,
    ExecutionStatus,
    TaskAssignment,
    AgentPool
)

__all__ = [
    "UniversalOrchestrator",
    "TaskRouter", 
    "WorkflowCoordinator",
    "ExecutionMonitor",
    "OrchestrationRequest",
    "OrchestrationResult",
    "WorkflowDefinition", 
    "WorkflowResult",
    "ExecutionStatus",
    "TaskAssignment",
    "AgentPool"
]

# Version and metadata
__version__ = "2.0.0"
__author__ = "LeanVibe Agent Hive"
__description__ = "Enhanced Multi-CLI Agent Orchestration System"