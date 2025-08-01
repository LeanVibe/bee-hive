"""
Advanced Command Templates for LeanVibe Agent Hive 2.0 - Phase 6.1

Intelligent workflow templates and project-specific command configurations for
enterprise-grade autonomous development. Provides customizable templates for
different project types, team sizes, and complexity levels.
"""

import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

import structlog

from ..schemas.custom_commands import (
    CommandDefinition, WorkflowStep, AgentRequirement, SecurityPolicy, AgentRole
)

logger = structlog.get_logger()


class ProjectType(str, Enum):
    """Supported project types for template customization."""
    WEB_APPLICATION = "web_application"
    API_SERVICE = "api_service"
    MOBILE_APP = "mobile_app"
    MICROSERVICE = "microservice"
    DATA_PIPELINE = "data_pipeline"
    ML_PROJECT = "ml_project"
    DEVOPS_INFRASTRUCTURE = "devops_infrastructure"
    LIBRARY_SDK = "library_sdk"


class TeamSize(str, Enum):
    """Team size categories for workflow optimization."""
    SOLO = "solo"
    SMALL = "small"        # 2-5 developers
    MEDIUM = "medium"      # 6-15 developers
    LARGE = "large"        # 16-50 developers
    ENTERPRISE = "enterprise"  # 50+ developers


class TechnologyStack(str, Enum):
    """Technology stacks for specialized workflows."""
    PYTHON_DJANGO = "python_django"
    PYTHON_FASTAPI = "python_fastapi"
    NODEJS_EXPRESS = "nodejs_express"
    NODEJS_NEXTJS = "nodejs_nextjs"
    JAVA_SPRING = "java_spring"
    DOTNET_CORE = "dotnet_core"
    GO_GIN = "go_gin"
    RUST_ACTIX = "rust_actix"
    REACT_FRONTEND = "react_frontend"
    VUE_FRONTEND = "vue_frontend"
    MOBILE_REACT_NATIVE = "mobile_react_native"
    MOBILE_FLUTTER = "mobile_flutter"


@dataclass
class ProjectConfiguration:
    """Project-specific configuration for command templates."""
    project_type: ProjectType
    team_size: TeamSize
    tech_stack: TechnologyStack
    complexity_level: str = "moderate"
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    security_requirements: Dict[str, Any] = field(default_factory=dict)
    compliance_standards: List[str] = field(default_factory=list)
    deployment_environments: List[str] = field(default_factory=list)
    quality_gates: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemplateCustomization:
    """Template customization options."""
    enable_ai_optimization: bool = True
    include_security_scans: bool = True
    enable_performance_testing: bool = True
    code_coverage_threshold: float = 90.0
    max_workflow_duration_minutes: int = 480
    parallel_execution_enabled: bool = True
    automated_rollback_enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)


class CommandTemplateEngine:
    """
    Advanced command template engine for intelligent workflow generation.
    
    Features:
    - Project-specific template customization
    - Technology stack optimizations
    - Team size and complexity adaptations
    - Industry best practices integration
    - Compliance and security standard adherence
    """
    
    def __init__(self):
        self.templates_cache: Dict[str, CommandDefinition] = {}
        self.project_configurations: Dict[str, ProjectConfiguration] = {}
        
        # Load base templates
        self.base_templates = {
            ProjectType.WEB_APPLICATION: self._create_web_app_templates(),
            ProjectType.API_SERVICE: self._create_api_service_templates(),
            ProjectType.MICROSERVICE: self._create_microservice_templates(),
            ProjectType.DATA_PIPELINE: self._create_data_pipeline_templates(),
            ProjectType.ML_PROJECT: self._create_ml_project_templates(),
            ProjectType.DEVOPS_INFRASTRUCTURE: self._create_devops_templates(),
            ProjectType.LIBRARY_SDK: self._create_library_templates()
        }
        
        logger.info(
            "CommandTemplateEngine initialized",
            base_templates=len(self.base_templates)
        )
    
    async def generate_customized_command(
        self,
        command_type: str,
        project_config: ProjectConfiguration,
        customization: TemplateCustomization,
        additional_requirements: Dict[str, Any] = None
    ) -> CommandDefinition:
        """
        Generate customized command based on project configuration.
        
        Args:
            command_type: Type of command to generate
            project_config: Project configuration
            customization: Template customization options
            additional_requirements: Additional specific requirements
            
        Returns:
            Customized command definition
        """
        try:
            # Get base template for project type
            base_templates = self.base_templates.get(project_config.project_type, {})
            base_template = base_templates.get(command_type)
            
            if not base_template:
                raise ValueError(f"No base template found for {command_type} in {project_config.project_type}")
            
            # Customize template based on configuration
            customized_template = await self._customize_template(
                base_template, project_config, customization, additional_requirements or {}
            )
            
            # Optimize for team size
            optimized_template = await self._optimize_for_team_size(
                customized_template, project_config.team_size
            )
            
            # Apply technology stack optimizations
            tech_optimized_template = await self._apply_tech_stack_optimizations(
                optimized_template, project_config.tech_stack
            )
            
            # Apply compliance and security requirements
            final_template = await self._apply_compliance_requirements(
                tech_optimized_template, project_config
            )
            
            logger.info(
                "Customized command generated",
                command_type=command_type,
                project_type=project_config.project_type.value,
                team_size=project_config.team_size.value,
                tech_stack=project_config.tech_stack.value
            )
            
            return final_template
            
        except Exception as e:
            logger.error("Failed to generate customized command", error=str(e))
            raise
    
    async def create_project_workflow_suite(
        self,
        project_config: ProjectConfiguration,
        customization: TemplateCustomization
    ) -> Dict[str, CommandDefinition]:
        """
        Create complete workflow suite for a project.
        
        Args:
            project_config: Project configuration
            customization: Template customization options
            
        Returns:
            Dictionary of command definitions for the project
        """
        try:
            workflow_suite = {}
            
            # Core development commands
            core_commands = [
                "feature_development",
                "bug_fix",
                "code_review",
                "testing",
                "performance_optimization"
            ]
            
            for command_type in core_commands:
                try:
                    command = await self.generate_customized_command(
                        command_type, project_config, customization
                    )
                    workflow_suite[command_type] = command
                except Exception as e:
                    logger.warning(f"Failed to generate {command_type} command", error=str(e))
            
            # DevOps and deployment commands
            if project_config.deployment_environments:
                devops_commands = ["deployment", "monitoring", "rollback"]
                for command_type in devops_commands:
                    try:
                        command = await self.generate_customized_command(
                            command_type, project_config, customization
                        )
                        workflow_suite[command_type] = command
                    except Exception as e:
                        logger.warning(f"Failed to generate {command_type} command", error=str(e))
            
            # Security and compliance commands
            if customization.include_security_scans or project_config.compliance_standards:
                security_commands = ["security_audit", "compliance_check"]
                for command_type in security_commands:
                    try:
                        command = await self.generate_customized_command(
                            command_type, project_config, customization
                        )
                        workflow_suite[command_type] = command
                    except Exception as e:
                        logger.warning(f"Failed to generate {command_type} command", error=str(e))
            
            logger.info(
                "Project workflow suite created",
                project_type=project_config.project_type.value,
                commands_generated=len(workflow_suite)
            )
            
            return workflow_suite
            
        except Exception as e:
            logger.error("Failed to create project workflow suite", error=str(e))
            raise
    
    async def save_project_configuration(
        self,
        project_id: str,
        project_config: ProjectConfiguration,
        workflow_suite: Dict[str, CommandDefinition]
    ) -> None:
        """Save project configuration and workflow suite."""
        try:
            # Save project configuration
            self.project_configurations[project_id] = project_config
            
            # Save workflow templates
            project_templates_dir = Path(f"./templates/projects/{project_id}")
            project_templates_dir.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            config_file = project_templates_dir / "project_config.yaml"
            with open(config_file, 'w') as f:
                config_data = {
                    "project_type": project_config.project_type.value,
                    "team_size": project_config.team_size.value,
                    "tech_stack": project_config.tech_stack.value,
                    "complexity_level": project_config.complexity_level,
                    "performance_requirements": project_config.performance_requirements,
                    "security_requirements": project_config.security_requirements,
                    "compliance_standards": project_config.compliance_standards,
                    "deployment_environments": project_config.deployment_environments,
                    "quality_gates": project_config.quality_gates
                }
                yaml.dump(config_data, f, default_flow_style=False)
            
            # Save workflow commands
            commands_dir = project_templates_dir / "commands"
            commands_dir.mkdir(exist_ok=True)
            
            for command_name, command_def in workflow_suite.items():
                command_file = commands_dir / f"{command_name}.yaml"
                with open(command_file, 'w') as f:
                    yaml.dump(command_def.model_dump(), f, default_flow_style=False)
            
            logger.info(
                "Project configuration saved",
                project_id=project_id,
                commands_saved=len(workflow_suite)
            )
            
        except Exception as e:
            logger.error("Failed to save project configuration", error=str(e))
            raise
    
    # Private template creation methods
    
    def _create_web_app_templates(self) -> Dict[str, CommandDefinition]:
        """Create web application templates."""
        return {
            "feature_development": CommandDefinition(
                name="web_app_feature_development",
                version="1.0.0",
                description="Web application feature development with frontend/backend coordination",
                category="development",
                tags=["web", "frontend", "backend", "feature"],
                agents=[
                    AgentRequirement(
                        role=AgentRole.FRONTEND_BUILDER,
                        required_capabilities=["react", "typescript", "ui_components"]
                    ),
                    AgentRequirement(
                        role=AgentRole.BACKEND_ENGINEER,
                        required_capabilities=["api_development", "database", "testing"]
                    )
                ],
                workflow=[
                    WorkflowStep(
                        step="requirements_analysis",
                        agent=AgentRole.BACKEND_ENGINEER,
                        task="Analyze feature requirements and create technical specification",
                        timeout_minutes=30
                    ),
                    WorkflowStep(
                        step="api_design",
                        agent=AgentRole.BACKEND_ENGINEER,
                        task="Design API endpoints and data models",
                        depends_on=["requirements_analysis"],
                        timeout_minutes=45
                    ),
                    WorkflowStep(
                        step="ui_design",
                        agent=AgentRole.FRONTEND_BUILDER,
                        task="Design user interface components and user experience",
                        depends_on=["requirements_analysis"],
                        timeout_minutes=45
                    ),
                    WorkflowStep(
                        step="backend_implementation",
                        agent=AgentRole.BACKEND_ENGINEER,
                        task="Implement API endpoints and business logic",
                        depends_on=["api_design"],
                        timeout_minutes=120
                    ),
                    WorkflowStep(
                        step="frontend_implementation",
                        agent=AgentRole.FRONTEND_BUILDER,
                        task="Implement user interface components and integration",
                        depends_on=["ui_design", "backend_implementation"],
                        timeout_minutes=120
                    ),
                    WorkflowStep(
                        step="integration_testing",
                        agent=AgentRole.QA_TEST_GUARDIAN,
                        task="Test frontend-backend integration and user workflows",
                        depends_on=["frontend_implementation"],
                        timeout_minutes=60
                    )
                ],
                security_policy=SecurityPolicy(
                    allowed_operations=["file_read", "file_write", "api_testing", "ui_testing"],
                    network_access=True
                )
            ),
            
            "testing": CommandDefinition(
                name="web_app_testing",
                version="1.0.0",
                description="Comprehensive web application testing strategy",
                category="testing",
                tags=["web", "testing", "frontend", "backend", "e2e"],
                agents=[
                    AgentRequirement(
                        role=AgentRole.QA_TEST_GUARDIAN,
                        required_capabilities=["unit_testing", "integration_testing", "e2e_testing"]
                    )
                ],
                workflow=[
                    WorkflowStep(
                        step="unit_tests",
                        agent=AgentRole.QA_TEST_GUARDIAN,
                        task="Execute frontend and backend unit tests",
                        timeout_minutes=30
                    ),
                    WorkflowStep(
                        step="api_integration_tests",
                        agent=AgentRole.QA_TEST_GUARDIAN,
                        task="Test API endpoints and data integration",
                        depends_on=["unit_tests"],
                        timeout_minutes=45
                    ),
                    WorkflowStep(
                        step="ui_component_tests",
                        agent=AgentRole.QA_TEST_GUARDIAN,
                        task="Test UI components and user interactions",
                        depends_on=["unit_tests"],
                        timeout_minutes=45
                    ),
                    WorkflowStep(
                        step="e2e_user_workflows",
                        agent=AgentRole.QA_TEST_GUARDIAN,
                        task="Execute end-to-end user workflow testing",
                        depends_on=["api_integration_tests", "ui_component_tests"],
                        timeout_minutes=60
                    ),
                    WorkflowStep(
                        step="performance_testing",
                        agent=AgentRole.QA_TEST_GUARDIAN,
                        task="Execute web application performance testing",
                        timeout_minutes=45
                    )
                ],
                security_policy=SecurityPolicy(
                    allowed_operations=["test_execution", "performance_testing"],
                    network_access=True
                )
            )
        }
    
    def _create_api_service_templates(self) -> Dict[str, CommandDefinition]:
        """Create API service templates."""
        return {
            "feature_development": CommandDefinition(
                name="api_service_feature_development",
                version="1.0.0",
                description="API service feature development with comprehensive testing",
                category="development",
                tags=["api", "service", "backend", "feature"],
                agents=[
                    AgentRequirement(
                        role=AgentRole.BACKEND_ENGINEER,
                        required_capabilities=["api_development", "database", "testing", "documentation"]
                    )
                ],
                workflow=[
                    WorkflowStep(
                        step="api_specification",
                        agent=AgentRole.BACKEND_ENGINEER,
                        task="Create detailed API specification and documentation",
                        timeout_minutes=45
                    ),
                    WorkflowStep(
                        step="data_model_design",
                        agent=AgentRole.BACKEND_ENGINEER,
                        task="Design data models and database schema",
                        depends_on=["api_specification"],
                        timeout_minutes=30
                    ),
                    WorkflowStep(
                        step="api_implementation",
                        agent=AgentRole.BACKEND_ENGINEER,
                        task="Implement API endpoints with validation and error handling",
                        depends_on=["data_model_design"],
                        timeout_minutes=120
                    ),
                    WorkflowStep(
                        step="api_testing",
                        agent=AgentRole.QA_TEST_GUARDIAN,
                        task="Create and execute comprehensive API test suite",
                        depends_on=["api_implementation"],
                        timeout_minutes=60
                    ),
                    WorkflowStep(
                        step="api_documentation",
                        agent=AgentRole.TECHNICAL_WRITER,
                        task="Generate interactive API documentation and examples",
                        depends_on=["api_testing"],
                        timeout_minutes=30
                    )
                ],
                security_policy=SecurityPolicy(
                    allowed_operations=["api_development", "database_access", "documentation"],
                    network_access=True,
                    audit_level="comprehensive"
                )
            )
        }
    
    def _create_microservice_templates(self) -> Dict[str, CommandDefinition]:
        """Create microservice templates."""
        return {
            "deployment": CommandDefinition(
                name="microservice_deployment",
                version="1.0.0",
                description="Microservice deployment with container orchestration",
                category="deployment",
                tags=["microservice", "deployment", "containers", "kubernetes"],
                agents=[
                    AgentRequirement(
                        role=AgentRole.DEVOPS_SPECIALIST,
                        required_capabilities=["docker", "kubernetes", "monitoring", "deployment"]
                    )
                ],
                workflow=[
                    WorkflowStep(
                        step="container_build",
                        agent=AgentRole.DEVOPS_SPECIALIST,
                        task="Build optimized container image with security scanning",
                        timeout_minutes=20
                    ),
                    WorkflowStep(
                        step="service_configuration",
                        agent=AgentRole.DEVOPS_SPECIALIST,
                        task="Configure service deployment and resource limits",
                        depends_on=["container_build"],
                        timeout_minutes=15
                    ),
                    WorkflowStep(
                        step="deploy_to_staging",
                        agent=AgentRole.DEVOPS_SPECIALIST,
                        task="Deploy to staging environment with health checks",
                        depends_on=["service_configuration"],
                        timeout_minutes=30
                    ),
                    WorkflowStep(
                        step="integration_validation",
                        agent=AgentRole.QA_TEST_GUARDIAN,
                        task="Validate service integration and communication",
                        depends_on=["deploy_to_staging"],
                        timeout_minutes=30
                    ),
                    WorkflowStep(
                        step="production_deployment",
                        agent=AgentRole.DEVOPS_SPECIALIST,
                        task="Deploy to production with blue-green strategy",
                        depends_on=["integration_validation"],
                        timeout_minutes=45
                    )
                ],
                security_policy=SecurityPolicy(
                    allowed_operations=["container_operations", "deployment", "monitoring"],
                    network_access=True,
                    requires_approval=True
                )
            )
        }
    
    def _create_data_pipeline_templates(self) -> Dict[str, CommandDefinition]:
        """Create data pipeline templates."""
        return {
            "feature_development": CommandDefinition(
                name="data_pipeline_development",
                version="1.0.0",
                description="Data pipeline development with quality validation",
                category="development",
                tags=["data", "pipeline", "etl", "quality"],
                agents=[
                    AgentRequirement(
                        role=AgentRole.DATA_ANALYST,
                        required_capabilities=["data_processing", "pipeline_design", "quality_validation"]
                    )
                ],
                workflow=[
                    WorkflowStep(
                        step="data_source_analysis",
                        agent=AgentRole.DATA_ANALYST,
                        task="Analyze data sources and define pipeline requirements",
                        timeout_minutes=60
                    ),
                    WorkflowStep(
                        step="pipeline_design",
                        agent=AgentRole.DATA_ANALYST,
                        task="Design data transformation and processing pipeline",
                        depends_on=["data_source_analysis"],
                        timeout_minutes=45
                    ),
                    WorkflowStep(
                        step="pipeline_implementation",
                        agent=AgentRole.DATA_ANALYST,
                        task="Implement data pipeline with error handling and monitoring",
                        depends_on=["pipeline_design"],
                        timeout_minutes=120
                    ),
                    WorkflowStep(
                        step="data_quality_validation",
                        agent=AgentRole.QA_TEST_GUARDIAN,
                        task="Validate data quality and pipeline correctness",
                        depends_on=["pipeline_implementation"],
                        timeout_minutes=60
                    )
                ],
                security_policy=SecurityPolicy(
                    allowed_operations=["data_processing", "database_access", "file_operations"],
                    network_access=True
                )
            )
        }
    
    def _create_ml_project_templates(self) -> Dict[str, CommandDefinition]:
        """Create ML project templates."""
        return {
            "feature_development": CommandDefinition(
                name="ml_model_development",
                version="1.0.0",
                description="Machine learning model development and validation",
                category="development",
                tags=["ml", "model", "training", "validation"],
                agents=[
                    AgentRequirement(
                        role=AgentRole.DATA_ANALYST,
                        required_capabilities=["ml_modeling", "data_analysis", "model_validation"]
                    )
                ],
                workflow=[
                    WorkflowStep(
                        step="data_exploration",
                        agent=AgentRole.DATA_ANALYST,
                        task="Explore and analyze training data",
                        timeout_minutes=90
                    ),
                    WorkflowStep(
                        step="feature_engineering",
                        agent=AgentRole.DATA_ANALYST,
                        task="Design and implement feature engineering pipeline",
                        depends_on=["data_exploration"],
                        timeout_minutes=120
                    ),
                    WorkflowStep(
                        step="model_training",
                        agent=AgentRole.DATA_ANALYST,
                        task="Train and tune machine learning model",
                        depends_on=["feature_engineering"],
                        timeout_minutes=180
                    ),
                    WorkflowStep(
                        step="model_validation",
                        agent=AgentRole.QA_TEST_GUARDIAN,
                        task="Validate model performance and accuracy",
                        depends_on=["model_training"],
                        timeout_minutes=60
                    ),
                    WorkflowStep(
                        step="model_deployment_prep",
                        agent=AgentRole.DEVOPS_SPECIALIST,
                        task="Prepare model for deployment with monitoring",
                        depends_on=["model_validation"],
                        timeout_minutes=45
                    )
                ],
                security_policy=SecurityPolicy(
                    allowed_operations=["data_processing", "model_training", "file_operations"],
                    network_access=True,
                    resource_limits={"max_memory_mb": 8192}
                )
            )
        }
    
    def _create_devops_templates(self) -> Dict[str, CommandDefinition]:
        """Create DevOps infrastructure templates."""
        return {
            "infrastructure_deployment": CommandDefinition(
                name="infrastructure_deployment",
                version="1.0.0",
                description="Infrastructure as Code deployment and management",
                category="infrastructure",
                tags=["infrastructure", "iac", "deployment", "monitoring"],
                agents=[
                    AgentRequirement(
                        role=AgentRole.DEVOPS_SPECIALIST,
                        required_capabilities=["infrastructure_as_code", "monitoring", "security"]
                    )
                ],
                workflow=[
                    WorkflowStep(
                        step="infrastructure_planning",
                        agent=AgentRole.DEVOPS_SPECIALIST,
                        task="Plan infrastructure requirements and architecture",
                        timeout_minutes=60
                    ),
                    WorkflowStep(
                        step="iac_development",
                        agent=AgentRole.DEVOPS_SPECIALIST,
                        task="Develop Infrastructure as Code templates",
                        depends_on=["infrastructure_planning"],
                        timeout_minutes=120
                    ),
                    WorkflowStep(
                        step="security_validation",
                        agent=AgentRole.SECURITY_AUDITOR,
                        task="Validate infrastructure security and compliance",
                        depends_on=["iac_development"],
                        timeout_minutes=45
                    ),
                    WorkflowStep(
                        step="infrastructure_deployment",
                        agent=AgentRole.DEVOPS_SPECIALIST,
                        task="Deploy infrastructure with monitoring and alerting",
                        depends_on=["security_validation"],
                        timeout_minutes=90
                    )
                ],
                security_policy=SecurityPolicy(
                    allowed_operations=["infrastructure_operations", "deployment", "monitoring"],
                    network_access=True,
                    requires_approval=True,
                    audit_level="comprehensive"
                )
            )
        }
    
    def _create_library_templates(self) -> Dict[str, CommandDefinition]:
        """Create library/SDK templates."""
        return {
            "feature_development": CommandDefinition(
                name="library_feature_development",
                version="1.0.0",
                description="Library feature development with comprehensive testing and documentation",
                category="development",
                tags=["library", "sdk", "api", "documentation"],
                agents=[
                    AgentRequirement(
                        role=AgentRole.BACKEND_ENGINEER,
                        required_capabilities=["library_development", "api_design", "testing"]
                    ),
                    AgentRequirement(
                        role=AgentRole.TECHNICAL_WRITER,
                        required_capabilities=["technical_writing", "documentation", "examples"]
                    )
                ],
                workflow=[
                    WorkflowStep(
                        step="api_design",
                        agent=AgentRole.BACKEND_ENGINEER,
                        task="Design public API interface and contracts",
                        timeout_minutes=60
                    ),
                    WorkflowStep(
                        step="implementation",
                        agent=AgentRole.BACKEND_ENGINEER,
                        task="Implement library functionality with comprehensive testing",
                        depends_on=["api_design"],
                        timeout_minutes=120
                    ),
                    WorkflowStep(
                        step="documentation_creation",
                        agent=AgentRole.TECHNICAL_WRITER,
                        task="Create comprehensive documentation and usage examples",
                        depends_on=["implementation"],
                        timeout_minutes=90
                    ),
                    WorkflowStep(
                        step="integration_examples",
                        agent=AgentRole.BACKEND_ENGINEER,
                        task="Create integration examples and sample applications",
                        depends_on=["documentation_creation"],
                        timeout_minutes=60
                    )
                ],
                security_policy=SecurityPolicy(
                    allowed_operations=["library_development", "testing", "documentation"],
                    network_access=False
                )
            )
        }
    
    # Private customization methods
    
    async def _customize_template(
        self,
        base_template: CommandDefinition,
        project_config: ProjectConfiguration,
        customization: TemplateCustomization,
        additional_requirements: Dict[str, Any]
    ) -> CommandDefinition:
        """Customize template based on project configuration."""
        # Create a copy of the base template
        customized = CommandDefinition(**base_template.model_dump())
        
        # Apply performance requirements
        if project_config.performance_requirements:
            await self._apply_performance_requirements(customized, project_config.performance_requirements)
        
        # Apply security requirements
        if project_config.security_requirements:
            await self._apply_security_requirements(customized, project_config.security_requirements)
        
        # Apply customization options
        if customization.code_coverage_threshold != 90.0:
            await self._adjust_coverage_requirements(customized, customization.code_coverage_threshold)
        
        if customization.max_workflow_duration_minutes != 480:
            await self._adjust_workflow_timeouts(customized, customization.max_workflow_duration_minutes)
        
        return customized
    
    async def _optimize_for_team_size(
        self,
        template: CommandDefinition,
        team_size: TeamSize
    ) -> CommandDefinition:
        """Optimize template for team size."""
        if team_size == TeamSize.SOLO:
            # Reduce parallelism and combine steps for solo developers
            await self._reduce_parallelism(template)
        elif team_size in [TeamSize.LARGE, TeamSize.ENTERPRISE]:
            # Increase parallelism and add coordination steps
            await self._increase_parallelism(template)
            await self._add_coordination_steps(template)
        
        return template
    
    async def _apply_tech_stack_optimizations(
        self,
        template: CommandDefinition,
        tech_stack: TechnologyStack
    ) -> CommandDefinition:
        """Apply technology stack specific optimizations."""
        # Add technology-specific steps and tools
        if tech_stack in [TechnologyStack.PYTHON_DJANGO, TechnologyStack.PYTHON_FASTAPI]:
            await self._add_python_specific_steps(template)
        elif tech_stack in [TechnologyStack.NODEJS_EXPRESS, TechnologyStack.NODEJS_NEXTJS]:
            await self._add_nodejs_specific_steps(template)
        elif tech_stack in [TechnologyStack.REACT_FRONTEND, TechnologyStack.VUE_FRONTEND]:
            await self._add_frontend_specific_steps(template)
        
        return template
    
    async def _apply_compliance_requirements(
        self,
        template: CommandDefinition,
        project_config: ProjectConfiguration
    ) -> CommandDefinition:
        """Apply compliance and regulatory requirements."""
        if project_config.compliance_standards:
            for standard in project_config.compliance_standards:
                if standard.upper() in ["SOX", "HIPAA", "GDPR", "PCI-DSS"]:
                    await self._add_compliance_steps(template, standard)
        
        return template
    
    # Helper methods for template customization
    
    async def _apply_performance_requirements(
        self,
        template: CommandDefinition,
        performance_requirements: Dict[str, Any]
    ) -> None:
        """Apply performance requirements to template."""
        # Add performance testing steps if required
        if performance_requirements.get("load_testing_required", False):
            await self._add_performance_testing_step(template)
    
    async def _apply_security_requirements(
        self,
        template: CommandDefinition,
        security_requirements: Dict[str, Any]
    ) -> None:
        """Apply security requirements to template."""
        # Enhance security policy
        if security_requirements.get("strict_security", False):
            template.security_policy.audit_level = "comprehensive"
            template.security_policy.requires_approval = True
    
    async def _adjust_coverage_requirements(
        self,
        template: CommandDefinition,
        coverage_threshold: float
    ) -> None:
        """Adjust test coverage requirements."""
        # Find testing steps and update requirements
        for step in template.workflow:
            if "test" in step.task.lower():
                step.task += f" with {coverage_threshold}% coverage requirement"
    
    async def _adjust_workflow_timeouts(
        self,
        template: CommandDefinition,
        max_duration_minutes: int
    ) -> None:
        """Adjust workflow timeouts based on maximum duration."""
        total_timeout = sum(step.timeout_minutes or 60 for step in template.workflow)
        if total_timeout > max_duration_minutes:
            scale_factor = max_duration_minutes / total_timeout
            for step in template.workflow:
                if step.timeout_minutes:
                    step.timeout_minutes = int(step.timeout_minutes * scale_factor)
    
    async def _reduce_parallelism(self, template: CommandDefinition) -> None:
        """Reduce parallelism for solo developers."""
        # Remove parallel execution from steps
        for step in template.workflow:
            step.parallel = None
    
    async def _increase_parallelism(self, template: CommandDefinition) -> None:
        """Increase parallelism for large teams."""
        # Add parallel execution where possible
        independent_steps = [step for step in template.workflow if not step.depends_on]
        if len(independent_steps) > 1:
            # Group first independent steps for parallel execution
            main_step = independent_steps[0]
            parallel_steps = independent_steps[1:3]  # Limit to 2 additional parallel steps
            main_step.parallel = parallel_steps
    
    async def _add_coordination_steps(self, template: CommandDefinition) -> None:
        """Add team coordination steps."""
        # Add coordination checkpoint at workflow end
        coordination_step = WorkflowStep(
            step="team_coordination_checkpoint",
            agent=AgentRole.PRODUCT_MANAGER,
            task="Coordinate team progress and resolve any blockers",
            depends_on=[step.step for step in template.workflow[-2:]],
            timeout_minutes=30
        )
        template.workflow.append(coordination_step)
    
    async def _add_python_specific_steps(self, template: CommandDefinition) -> None:
        """Add Python-specific workflow steps."""
        # Add Python linting and formatting step
        python_quality_step = WorkflowStep(
            step="python_code_quality",
            agent=AgentRole.BACKEND_ENGINEER,
            task="Run Python linting (flake8, black) and type checking (mypy)",
            timeout_minutes=15
        )
        # Insert after implementation steps
        template.workflow.insert(-1, python_quality_step)
    
    async def _add_nodejs_specific_steps(self, template: CommandDefinition) -> None:
        """Add Node.js-specific workflow steps."""
        # Add Node.js linting and security audit
        nodejs_quality_step = WorkflowStep(
            step="nodejs_code_quality",
            agent=AgentRole.BACKEND_ENGINEER,
            task="Run ESLint, Prettier, and npm security audit",
            timeout_minutes=15
        )
        template.workflow.insert(-1, nodejs_quality_step)
    
    async def _add_frontend_specific_steps(self, template: CommandDefinition) -> None:
        """Add frontend-specific workflow steps."""
        # Add frontend build optimization step
        frontend_optimization_step = WorkflowStep(
            step="frontend_optimization",
            agent=AgentRole.FRONTEND_BUILDER,
            task="Optimize bundle size and perform lighthouse audit",
            timeout_minutes=20
        )
        template.workflow.insert(-1, frontend_optimization_step)
    
    async def _add_compliance_steps(self, template: CommandDefinition, standard: str) -> None:
        """Add compliance-specific steps."""
        compliance_step = WorkflowStep(
            step=f"{standard.lower()}_compliance_validation",
            agent=AgentRole.SECURITY_AUDITOR,
            task=f"Validate {standard} compliance requirements and generate audit report",
            timeout_minutes=45
        )
        template.workflow.append(compliance_step)
    
    async def _add_performance_testing_step(self, template: CommandDefinition) -> None:
        """Add performance testing step."""
        performance_step = WorkflowStep(
            step="performance_load_testing",
            agent=AgentRole.QA_TEST_GUARDIAN,
            task="Execute comprehensive performance and load testing",
            timeout_minutes=60
        )
        template.workflow.append(performance_step)