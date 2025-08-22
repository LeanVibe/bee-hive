"""
Consolidated Manager System for LeanVibe Agent Hive 2.0

This package consolidates 69+ manager files into 6 unified managers:
- AgentLifecycleManager: Agent spawning, monitoring, shutdown
- TaskCoordinationManager: Task assignment, progress, completion  
- IntegrationManager: API, WebSocket, database, Redis interfaces
- PluginManager: Plugin loading, management, security
- PerformanceManager: Monitoring, optimization, benchmarking
- ProductionManager: Production scaling, alerts, SLA monitoring

Architecture consolidation reduces 149 files to 8 while preserving 
all functionality and performance optimizations.
"""

from .agent_lifecycle_manager import AgentLifecycleManager, AgentRole, AgentInstance
from .task_coordination_manager import TaskCoordinationManager, TaskAssignment, TaskPriority
from .integration_manager import IntegrationManager, ConnectionManager
from .plugin_manager import PluginManager, PluginSystem
from .performance_manager import PerformanceManager, PerformanceMetrics
from .production_manager import ProductionManager, ProductionMetrics, SystemHealth

__all__ = [
    "AgentLifecycleManager",
    "TaskCoordinationManager", 
    "IntegrationManager",
    "PluginManager",
    "PerformanceManager",
    "ProductionManager",
    "AgentRole",
    "AgentInstance",
    "TaskAssignment", 
    "TaskPriority",
    "ConnectionManager",
    "PluginSystem",
    "PerformanceMetrics",
    "ProductionMetrics",
    "SystemHealth"
]