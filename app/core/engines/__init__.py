"""
Consolidated Engine Architecture for LeanVibe Agent Hive 2.0

8 Specialized Engines consolidating 35+ legacy implementations:
- TaskExecutionEngine: Unified task execution and scheduling
- WorkflowEngine: DAG-based workflow orchestration  
- DataProcessingEngine: Semantic search and context processing
- SecurityEngine: RBAC, authorization, and threat detection
- CommunicationEngine: Inter-agent messaging and event processing
- MonitoringEngine: Performance analytics and observability
- IntegrationEngine: External system integrations
- OptimizationEngine: Performance and resource optimization
"""

from .base_engine import (
    BaseEngine,
    EngineConfig,
    EngineRequest,
    EngineResponse,
    HealthStatus,
    EngineMetrics,
    EnginePlugin
)

from .task_execution_engine import TaskExecutionEngine
from .workflow_engine import WorkflowEngine
from .data_processing_engine import DataProcessingEngine
from .security_engine import SecurityEngine
from .communication_engine import CommunicationEngine
from .monitoring_engine import MonitoringEngine
from .integration_engine import IntegrationEngine
from .optimization_engine import OptimizationEngine

__all__ = [
    'BaseEngine',
    'EngineConfig',
    'EngineRequest', 
    'EngineResponse',
    'HealthStatus',
    'EngineMetrics',
    'EnginePlugin',
    'TaskExecutionEngine',
    'WorkflowEngine', 
    'DataProcessingEngine',
    'SecurityEngine',
    'CommunicationEngine',
    'MonitoringEngine',
    'IntegrationEngine',
    'OptimizationEngine'
]