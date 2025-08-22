"""
Operations Specialist Agent - Module Initialization
Epic G: Production Readiness

Enterprise-grade operational infrastructure for the LeanVibe Agent Hive 2.0 platform.
Provides production monitoring, deployment automation, operational excellence,
and enterprise integration capabilities.
"""

from .production_monitoring import (
    ProductionMonitoringOrchestrator,
    DistributedTracingSystem,
    ProductionMetricsCollector,
    RealTimeHealthDashboard,
    IntelligentAlertingSystem,
    MonitoringComponent,
    AlertSeverity,
    get_production_monitoring,
    start_production_monitoring,
    stop_production_monitoring
)

from .deployment_automation import (
    ContainerImageBuilder,
    KubernetesDeploymentManager,
    CICDPipelineOrchestrator,
    BuildConfiguration,
    DeploymentConfiguration,
    DeploymentResult,
    DeploymentEnvironment,
    DeploymentStrategy,
    DeploymentStatus,
    get_deployment_automation
)

from .operational_excellence import (
    AutomatedBackupSystem,
    LogAggregationSystem,
    ResourceOptimizationSystem,
    OperationalExcellenceOrchestrator,
    BackupConfiguration,
    BackupType,
    BackupStatus,
    ResourceMetrics,
    CapacityPrediction,
    get_operational_excellence,
    start_operational_excellence,
    stop_operational_excellence
)

from .enterprise_integration import (
    EnterpriseSSO,
    ComplianceMonitoringSystem,
    AuditTrailSystem,
    SupportEscalationSystem,
    EnterpriseIntegrationOrchestrator,
    SSOConfiguration,
    SSOProvider,
    ComplianceFramework,
    AuditEventType,
    SupportTicketPriority,
    EnterpriseUser,
    AuditEvent,
    SupportTicket,
    get_enterprise_integration,
    start_enterprise_integration,
    stop_enterprise_integration
)

__all__ = [
    # Production Monitoring
    'ProductionMonitoringOrchestrator',
    'DistributedTracingSystem',
    'ProductionMetricsCollector',
    'RealTimeHealthDashboard',
    'IntelligentAlertingSystem',
    'MonitoringComponent',
    'AlertSeverity',
    'get_production_monitoring',
    'start_production_monitoring',
    'stop_production_monitoring',
    
    # Deployment Automation
    'ContainerImageBuilder',
    'KubernetesDeploymentManager',
    'CICDPipelineOrchestrator',
    'BuildConfiguration',
    'DeploymentConfiguration',
    'DeploymentResult',
    'DeploymentEnvironment',
    'DeploymentStrategy',
    'DeploymentStatus',
    'get_deployment_automation',
    
    # Operational Excellence
    'AutomatedBackupSystem',
    'LogAggregationSystem',
    'ResourceOptimizationSystem',
    'OperationalExcellenceOrchestrator',
    'BackupConfiguration',
    'BackupType',
    'BackupStatus',
    'ResourceMetrics',
    'CapacityPrediction',
    'get_operational_excellence',
    'start_operational_excellence',
    'stop_operational_excellence',
    
    # Enterprise Integration
    'EnterpriseSSO',
    'ComplianceMonitoringSystem',
    'AuditTrailSystem',
    'SupportEscalationSystem',
    'EnterpriseIntegrationOrchestrator',
    'SSOConfiguration',
    'SSOProvider',
    'ComplianceFramework',
    'AuditEventType',
    'SupportTicketPriority',
    'EnterpriseUser',
    'AuditEvent',
    'SupportTicket',
    'get_enterprise_integration',
    'start_enterprise_integration',
    'stop_enterprise_integration',
]

# Version information
__version__ = "1.0.0"
__author__ = "Operations Specialist Agent"
__description__ = "Enterprise-grade operational infrastructure for LeanVibe Agent Hive 2.0"