#!/usr/bin/env python3
"""
Epic 4 Phase 1: Unified API Architecture Design
LeanVibe Agent Hive 2.0 - OpenAPI 3.0 Compliant Consolidated API Design

Based on comprehensive analysis of 129 API files, this design consolidates
system monitoring (30 files), agent management (18 files), task execution (12 files),
and other domains into unified, production-ready API modules.

CONSOLIDATION TARGET: 129 files → 8 unified modules (93.8% reduction)
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class APIModule:
    """Unified API module specification."""
    name: str
    business_domain: str
    description: str
    prefix: str
    source_files: List[str]
    endpoints_count: int
    consolidation_priority: int
    security_requirements: List[str]
    dependencies: List[str]
    response_models: List[str]
    middleware: List[str]
    openapi_tags: List[str]

@dataclass
class APIEndpointSpec:
    """OpenAPI 3.0 compliant endpoint specification."""
    method: str
    path: str
    summary: str
    description: str
    tags: List[str]
    parameters: List[Dict[str, Any]]
    request_body: Optional[Dict[str, Any]]
    responses: Dict[str, Dict[str, Any]]
    security: List[Dict[str, Any]]
    deprecated: bool = False

@dataclass
class UnifiedAPIArchitecture:
    """Complete unified API architecture specification."""
    version: str
    title: str
    description: str
    modules: List[APIModule]
    endpoints: List[APIEndpointSpec]
    security_schemes: Dict[str, Dict[str, Any]]
    servers: List[Dict[str, Any]]
    consolidation_metrics: Dict[str, Any]

class UnifiedAPIArchitectureDesigner:
    """Designer for unified API architecture following OpenAPI 3.0 standards."""
    
    def __init__(self):
        self.architecture = UnifiedAPIArchitecture(
            version="2.0.0",
            title="LeanVibe Agent Hive 2.0 - Unified API",
            description="Consolidated multi-agent orchestration API with comprehensive business domain coverage",
            modules=[],
            endpoints=[],
            security_schemes=self._define_security_schemes(),
            servers=self._define_servers(),
            consolidation_metrics={}
        )
        
        # Load audit results for informed design
        self.audit_data = self._load_audit_data()
    
    def _load_audit_data(self) -> Dict[str, Any]:
        """Load audit data from previous analysis."""
        try:
            with open('/Users/bogdan/work/leanvibe-dev/bee-hive/epic4_phase1_api_audit_report.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️  Audit data not found, using design defaults")
            return {
                'business_domain_analysis': {
                    'system_monitoring': {'file_count': 30},
                    'agent_management': {'file_count': 18},
                    'task_execution': {'file_count': 12},
                    'authentication_security': {'file_count': 10},
                    'project_management': {'file_count': 12},
                    'enterprise_features': {'file_count': 3},
                    'communication_integration': {'file_count': 8},
                    'development_tooling': {'file_count': 5}
                }
            }
    
    def design_unified_architecture(self) -> UnifiedAPIArchitecture:
        """Design complete unified API architecture."""
        print("🏗️  Designing Epic 4 Unified API Architecture...")
        
        # Design core business modules
        self._design_system_monitoring_module()
        self._design_agent_management_module()
        self._design_task_execution_module()
        self._design_authentication_security_module()
        self._design_project_management_module()
        self._design_enterprise_features_module()
        self._design_communication_integration_module()
        self._design_development_tooling_module()
        
        # Design unified endpoints
        self._design_unified_endpoints()
        
        # Calculate consolidation metrics
        self._calculate_consolidation_metrics()
        
        return self.architecture
    
    def _design_system_monitoring_module(self):
        """Design unified system monitoring API module."""
        module = APIModule(
            name="SystemMonitoringAPI",
            business_domain="system_monitoring",
            description="Unified observability, metrics, dashboards, and performance monitoring",
            prefix="/api/v2/monitoring",
            source_files=[
                "dashboard_monitoring.py", "observability.py", "performance_intelligence.py",
                "monitoring_reporting.py", "business_analytics.py", "dashboard_prometheus.py",
                "strategic_monitoring.py", "mobile_monitoring.py", "observability_hooks.py"
            ],
            endpoints_count=45,  # Estimated from 30 files
            consolidation_priority=10,
            security_requirements=["api_key", "rbac"],
            dependencies=["redis", "prometheus", "grafana"],
            response_models=["MetricsResponse", "DashboardData", "AlertConfiguration"],
            middleware=["rate_limiting", "caching", "compression"],
            openapi_tags=["monitoring", "observability", "analytics", "dashboards"]
        )
        self.architecture.modules.append(module)
    
    def _design_agent_management_module(self):
        """Design unified agent management API module."""
        module = APIModule(
            name="AgentManagementAPI",
            business_domain="agent_management",
            description="Comprehensive agent lifecycle, coordination, and activation management",
            prefix="/api/v2/agents",
            source_files=[
                "agent_coordination.py", "agent_activation.py", "coordination_endpoints.py",
                "v1/agents.py", "v1/coordination.py", "endpoints/agents.py"
            ],
            endpoints_count=35,  # Estimated from 18 files
            consolidation_priority=10,
            security_requirements=["oauth2", "rbac"],
            dependencies=["database", "redis", "consolidated_engine"],
            response_models=["Agent", "AgentStatus", "CoordinationResponse"],
            middleware=["authentication", "validation", "logging"],
            openapi_tags=["agents", "coordination", "lifecycle"]
        )
        self.architecture.modules.append(module)
    
    def _design_task_execution_module(self):
        """Design unified task execution API module."""
        module = APIModule(
            name="TaskExecutionAPI",
            business_domain="task_execution",
            description="Unified workflow orchestration, task delegation, and execution management",
            prefix="/api/v2/tasks",
            source_files=[
                "intelligent_scheduling.py", "v1/workflows.py", "v1/orchestrator_core.py",
                "v1/team_coordination.py", "endpoints/tasks.py", "v2/tasks.py"
            ],
            endpoints_count=28,  # Estimated from 12 files
            consolidation_priority=10,
            security_requirements=["oauth2", "task_permissions"],
            dependencies=["database", "workflow_engine", "consolidated_orchestrator"],
            response_models=["Task", "WorkflowExecution", "ScheduleResponse"],
            middleware=["authentication", "validation", "rate_limiting"],
            openapi_tags=["tasks", "workflows", "scheduling", "orchestration"]
        )
        self.architecture.modules.append(module)
    
    def _design_authentication_security_module(self):
        """Design unified authentication and security API module."""
        module = APIModule(
            name="AuthenticationSecurityAPI",
            business_domain="authentication_security",
            description="Comprehensive authentication, authorization, and security management",
            prefix="/api/v2/auth",
            source_files=[
                "auth_endpoints.py", "oauth2_endpoints.py", "security_endpoints.py",
                "rbac.py", "enterprise_security.py", "v1/security.py"
            ],
            endpoints_count=22,  # Estimated from 10 files
            consolidation_priority=8,
            security_requirements=["oauth2", "jwt", "rbac", "mfa"],
            dependencies=["database", "redis", "oauth_provider"],
            response_models=["AuthResponse", "TokenResponse", "UserProfile", "PermissionSet"],
            middleware=["cors", "security_headers", "rate_limiting"],
            openapi_tags=["authentication", "security", "authorization", "rbac"]
        )
        self.architecture.modules.append(module)
    
    def _design_project_management_module(self):
        """Design unified project management API module."""
        module = APIModule(
            name="ProjectManagementAPI", 
            business_domain="project_management",
            description="Unified project indexing, context management, and workspace operations",
            prefix="/api/v2/projects",
            source_files=[
                "project_index.py", "project_index_optimization.py", "context_optimization.py",
                "memory_operations.py", "v1/contexts.py", "v1/workspaces.py"
            ],
            endpoints_count=20,  # Estimated from 12 files
            consolidation_priority=6,
            security_requirements=["oauth2", "project_permissions"],
            dependencies=["database", "vector_store", "context_engine"],
            response_models=["Project", "Context", "Workspace", "IndexResponse"],
            middleware=["authentication", "validation", "caching"],
            openapi_tags=["projects", "context", "indexing", "workspaces"]
        )
        self.architecture.modules.append(module)
    
    def _design_enterprise_features_module(self):
        """Design unified enterprise features API module."""
        module = APIModule(
            name="EnterpriseAPI",
            business_domain="enterprise_features",
            description="Enterprise sales, pilots, and strategic business features",
            prefix="/api/v2/enterprise",
            source_files=[
                "enterprise_pilots.py", "enterprise_sales.py", "v2/enterprise.py"
            ],
            endpoints_count=12,  # Estimated from 3 files
            consolidation_priority=4,
            security_requirements=["oauth2", "enterprise_permissions"],
            dependencies=["database", "crm_integration"],
            response_models=["PilotProgram", "SalesData", "EnterpriseMetrics"],
            middleware=["authentication", "enterprise_validation"],
            openapi_tags=["enterprise", "sales", "pilots"]
        )
        self.architecture.modules.append(module)
    
    def _design_communication_integration_module(self):
        """Design unified communication integration API module."""
        module = APIModule(
            name="CommunicationAPI",
            business_domain="communication_integration", 
            description="WebSocket, GitHub, Claude, and external system integrations",
            prefix="/api/v2/integrations",
            source_files=[
                "dashboard_websockets.py", "v1/websocket.py", "claude_integration.py",
                "v1/github_integration.py", "pwa_backend.py"
            ],
            endpoints_count=18,  # Estimated from 8 files
            consolidation_priority=6,
            security_requirements=["oauth2", "webhook_validation"],
            dependencies=["websocket", "github_api", "claude_api"],
            response_models=["WebSocketResponse", "IntegrationStatus", "WebhookEvent"],
            middleware=["websocket_auth", "webhook_validation"],
            openapi_tags=["websockets", "integrations", "github", "claude"]
        )
        self.architecture.modules.append(module)
    
    def _design_development_tooling_module(self):
        """Design unified development tooling API module."""
        module = APIModule(
            name="DevelopmentToolingAPI",
            business_domain="development_tooling",
            description="Developer experience, debugging, and technical debt management tools",
            prefix="/api/v2/dev",
            source_files=[
                "dx_debugging.py", "technical_debt.py", "self_modification_endpoints.py"
            ],
            endpoints_count=15,  # Estimated from 5 files
            consolidation_priority=4,
            security_requirements=["oauth2", "developer_permissions"],
            dependencies=["code_analysis", "git_integration"],
            response_models=["DebuggingInfo", "TechnicalDebtReport", "ModificationResponse"],
            middleware=["authentication", "developer_validation"],
            openapi_tags=["development", "debugging", "technical-debt"]
        )
        self.architecture.modules.append(module)
    
    def _design_unified_endpoints(self):
        """Design unified API endpoints following OpenAPI 3.0 standards."""
        
        # System Monitoring endpoints
        self.architecture.endpoints.extend([
            APIEndpointSpec(
                method="GET",
                path="/api/v2/monitoring/health",
                summary="System Health Check",
                description="Comprehensive system health and status information",
                tags=["monitoring", "health"],
                parameters=[],
                request_body=None,
                responses={
                    "200": {
                        "description": "System health information",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/HealthResponse"}
                            }
                        }
                    }
                },
                security=[{"oauth2": ["read:monitoring"]}]
            ),
            APIEndpointSpec(
                method="GET", 
                path="/api/v2/monitoring/metrics",
                summary="System Metrics",
                description="Real-time system performance and operational metrics",
                tags=["monitoring", "metrics"],
                parameters=[
                    {
                        "name": "timeRange",
                        "in": "query",
                        "description": "Time range for metrics (1h, 24h, 7d, 30d)",
                        "schema": {"type": "string", "enum": ["1h", "24h", "7d", "30d"]}
                    }
                ],
                request_body=None,
                responses={
                    "200": {
                        "description": "System metrics data",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MetricsResponse"}
                            }
                        }
                    }
                },
                security=[{"oauth2": ["read:monitoring"]}]
            )
        ])
        
        # Agent Management endpoints
        self.architecture.endpoints.extend([
            APIEndpointSpec(
                method="GET",
                path="/api/v2/agents",
                summary="List Agents",
                description="Retrieve list of all agents with status and configuration",
                tags=["agents"],
                parameters=[
                    {
                        "name": "status",
                        "in": "query",
                        "description": "Filter agents by status",
                        "schema": {"type": "string", "enum": ["active", "idle", "error", "offline"]}
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Maximum number of agents to return",
                        "schema": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}
                    }
                ],
                request_body=None,
                responses={
                    "200": {
                        "description": "List of agents",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AgentListResponse"}
                            }
                        }
                    }
                },
                security=[{"oauth2": ["read:agents"]}]
            ),
            APIEndpointSpec(
                method="POST",
                path="/api/v2/agents",
                summary="Create Agent",
                description="Create new agent with specified configuration and capabilities",
                tags=["agents"],
                parameters=[],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/CreateAgentRequest"}
                        }
                    }
                },
                responses={
                    "201": {
                        "description": "Agent created successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Agent"}
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid agent configuration",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    }
                },
                security=[{"oauth2": ["write:agents"]}]
            )
        ])
        
        # Task Execution endpoints
        self.architecture.endpoints.extend([
            APIEndpointSpec(
                method="POST",
                path="/api/v2/tasks/execute",
                summary="Execute Task",
                description="Execute task with specified parameters and routing",
                tags=["tasks", "execution"],
                parameters=[],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/TaskExecutionRequest"}
                        }
                    }
                },
                responses={
                    "202": {
                        "description": "Task accepted for execution",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/TaskExecutionResponse"}
                            }
                        }
                    }
                },
                security=[{"oauth2": ["execute:tasks"]}]
            )
        ])
    
    def _calculate_consolidation_metrics(self):
        """Calculate consolidation impact metrics."""
        original_files = self.audit_data.get('audit_summary', {}).get('total_api_files', 129)
        unified_modules = len(self.architecture.modules)
        
        self.architecture.consolidation_metrics = {
            'original_file_count': original_files,
            'unified_module_count': unified_modules,
            'consolidation_ratio': round(unified_modules / original_files, 3),
            'complexity_reduction_percentage': round((1 - unified_modules / original_files) * 100, 1),
            'estimated_endpoints_total': sum(module.endpoints_count for module in self.architecture.modules),
            'business_domains_covered': len(set(module.business_domain for module in self.architecture.modules)),
            'migration_complexity_score': self._calculate_migration_complexity(),
            'maintenance_improvement_factor': round(original_files / unified_modules, 1)
        }
    
    def _calculate_migration_complexity(self) -> str:
        """Calculate overall migration complexity score."""
        high_priority_modules = sum(1 for m in self.architecture.modules if m.consolidation_priority >= 8)
        total_modules = len(self.architecture.modules)
        
        if high_priority_modules > total_modules * 0.6:
            return "HIGH - Multiple critical systems requiring careful migration"
        elif high_priority_modules > total_modules * 0.3:
            return "MEDIUM - Balanced mix of critical and standard systems"
        else:
            return "LOW - Mostly straightforward consolidation opportunities"
    
    def _define_security_schemes(self) -> Dict[str, Dict[str, Any]]:
        """Define OpenAPI security schemes."""
        return {
            "oauth2": {
                "type": "oauth2",
                "flows": {
                    "authorizationCode": {
                        "authorizationUrl": "/api/v2/auth/oauth/authorize",
                        "tokenUrl": "/api/v2/auth/oauth/token",
                        "scopes": {
                            "read:agents": "Read agent information",
                            "write:agents": "Create and modify agents",
                            "execute:tasks": "Execute tasks and workflows",
                            "read:monitoring": "Access monitoring data",
                            "admin": "Administrative access"
                        }
                    }
                }
            },
            "apiKey": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key"
            },
            "jwt": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        }
    
    def _define_servers(self) -> List[Dict[str, Any]]:
        """Define API servers."""
        return [
            {
                "url": "https://api.leanvibe.dev",
                "description": "Production server"
            },
            {
                "url": "https://staging-api.leanvibe.dev",
                "description": "Staging server"
            },
            {
                "url": "http://localhost:8000",
                "description": "Local development server"
            }
        ]

def main():
    """Generate unified API architecture specification."""
    print("="*80)
    print("🏗️  EPIC 4 PHASE 1: UNIFIED API ARCHITECTURE DESIGN")
    print("="*80)
    
    designer = UnifiedAPIArchitectureDesigner()
    architecture = designer.design_unified_architecture()
    
    # Generate comprehensive architecture report
    architecture_dict = asdict(architecture)
    
    # Save architecture specification
    spec_path = Path("/Users/bogdan/work/leanvibe-dev/bee-hive/epic4_unified_api_architecture_spec.json")
    with open(spec_path, 'w', encoding='utf-8') as f:
        json.dump(architecture_dict, f, indent=2, default=str)
    
    # Generate OpenAPI 3.0 specification
    openapi_spec = generate_openapi_spec(architecture)
    openapi_path = Path("/Users/bogdan/work/leanvibe-dev/bee-hive/epic4_openapi_3_0_specification.json")
    with open(openapi_path, 'w', encoding='utf-8') as f:
        json.dump(openapi_spec, f, indent=2, default=str)
    
    # Print architecture summary
    print(f"\n📊 UNIFIED API ARCHITECTURE SUMMARY:")
    print("="*60)
    print(f"📦 Unified modules designed: {len(architecture.modules)}")
    print(f"🔗 API endpoints specified: {len(architecture.endpoints)}")
    print(f"📉 File consolidation: {architecture.consolidation_metrics['original_file_count']} → {architecture.consolidation_metrics['unified_module_count']}")
    print(f"📈 Complexity reduction: {architecture.consolidation_metrics['complexity_reduction_percentage']}%")
    print(f"🎯 Business domains covered: {architecture.consolidation_metrics['business_domains_covered']}")
    
    print(f"\n🏗️  UNIFIED API MODULES:")
    print("="*60)
    for i, module in enumerate(architecture.modules, 1):
        print(f"{i}. {module.name}")
        print(f"   Domain: {module.business_domain}")
        print(f"   Prefix: {module.prefix}")
        print(f"   Endpoints: ~{module.endpoints_count}")
        print(f"   Priority: {module.consolidation_priority}/10")
        print()
    
    print(f"💾 Architecture specification saved to: {spec_path}")
    print(f"📋 OpenAPI 3.0 spec saved to: {openapi_path}")
    print("\n✅ UNIFIED API ARCHITECTURE DESIGN COMPLETE")
    
    return architecture

def generate_openapi_spec(architecture: UnifiedAPIArchitecture) -> Dict[str, Any]:
    """Generate OpenAPI 3.0 specification from architecture."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": architecture.title,
            "description": architecture.description,
            "version": architecture.version,
            "contact": {
                "name": "LeanVibe Support",
                "email": "support@leanvibe.dev",
                "url": "https://leanvibe.dev/support"
            },
            "license": {
                "name": "Proprietary",
                "url": "https://leanvibe.dev/license"
            }
        },
        "servers": architecture.servers,
        "security": [
            {"oauth2": []},
            {"apiKey": []}
        ],
        "components": {
            "securitySchemes": architecture.security_schemes,
            "schemas": {
                # Placeholder schemas - would be generated from actual models
                "HealthResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["healthy", "degraded", "unhealthy"]},
                        "timestamp": {"type": "string", "format": "date-time"},
                        "services": {"type": "object"}
                    }
                },
                "Agent": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "format": "uuid"},
                        "name": {"type": "string"},
                        "status": {"type": "string", "enum": ["active", "idle", "error", "offline"]},
                        "capabilities": {"type": "array", "items": {"type": "string"}},
                        "created_at": {"type": "string", "format": "date-time"}
                    }
                },
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "error": {"type": "string"},
                        "message": {"type": "string"},
                        "code": {"type": "integer"},
                        "details": {"type": "object"}
                    }
                }
            }
        },
        "paths": {
            endpoint.path: {
                endpoint.method.lower(): {
                    "summary": endpoint.summary,
                    "description": endpoint.description,
                    "tags": endpoint.tags,
                    "parameters": endpoint.parameters,
                    "requestBody": endpoint.request_body,
                    "responses": endpoint.responses,
                    "security": endpoint.security
                }
            } for endpoint in architecture.endpoints
        },
        "tags": [
            {"name": "monitoring", "description": "System monitoring and observability"},
            {"name": "agents", "description": "Agent management and coordination"},
            {"name": "tasks", "description": "Task execution and workflows"},
            {"name": "auth", "description": "Authentication and authorization"},
            {"name": "projects", "description": "Project and context management"},
            {"name": "enterprise", "description": "Enterprise features"},
            {"name": "integrations", "description": "External integrations"},
            {"name": "development", "description": "Development tooling"}
        ]
    }

if __name__ == '__main__':
    main()