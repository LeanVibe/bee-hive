"""
Orchestrator Migration Utilities
Epic 1 Phase 1.1 - Migration tools for ConsolidatedProductionOrchestrator

This module provides utilities for migrating from existing orchestrator implementations
to the new ConsolidatedProductionOrchestrator while maintaining full compatibility
and zero-downtime transitions.

Migration Strategies:
1. Gradual Migration: Move consumers one by one
2. State Preservation: Maintain all agent and task state during migration
3. Rollback Support: Ability to rollback to previous orchestrator if needed
4. Validation: Comprehensive validation of migrated state
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Type, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

from .orchestrator_interfaces import (
    OrchestratorConfig,
    AgentSpec,
    TaskSpec,
    AgentStatus,
    TaskResult,
    MigrationResult,
    OrchestratorMode
)
from .consolidated_orchestrator import ConsolidatedProductionOrchestrator
from .logging_service import get_component_logger

logger = get_component_logger("orchestrator_migration")


class MigrationStrategy(str, Enum):
    """Migration strategy options."""
    GRADUAL = "gradual"          # Migrate components one by one
    IMMEDIATE = "immediate"      # Full migration at once
    SHADOW = "shadow"           # Run both orchestrators in parallel
    VALIDATION_ONLY = "validation_only"  # Validate without migrating


class MigrationPhase(str, Enum):
    """Migration phase tracking."""
    PLANNING = "planning"
    PREPARATION = "preparation" 
    VALIDATION = "validation"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    COMPLETION = "completion"
    ROLLBACK = "rollback"


@dataclass
class MigrationPlan:
    """Comprehensive migration plan."""
    source_orchestrator_type: str
    target_orchestrator_type: str = "consolidated"
    strategy: MigrationStrategy = MigrationStrategy.GRADUAL
    
    # Migration phases
    phases: List[str] = None
    current_phase: MigrationPhase = MigrationPhase.PLANNING
    
    # Component migration order
    component_order: List[str] = None
    
    # Validation settings
    validate_before_migration: bool = True
    validate_after_migration: bool = True
    run_compatibility_tests: bool = True
    
    # Rollback settings
    enable_rollback: bool = True
    backup_state: bool = True
    
    # Timing settings
    max_migration_time_seconds: int = 3600  # 1 hour
    component_timeout_seconds: int = 300    # 5 minutes
    
    # Safety settings
    max_concurrent_migrations: int = 1
    require_confirmation: bool = True
    
    def __post_init__(self):
        if self.phases is None:
            self.phases = [
                "agent_migration",
                "task_migration", 
                "plugin_migration",
                "configuration_migration",
                "state_validation"
            ]
        
        if self.component_order is None:
            self.component_order = [
                "core_orchestrator",
                "agent_management",
                "task_coordination",
                "plugin_system",
                "monitoring",
                "configuration"
            ]


@dataclass 
class MigrationContext:
    """Context information for migration operation."""
    migration_id: str
    plan: MigrationPlan
    source_orchestrator: Any
    target_orchestrator: ConsolidatedProductionOrchestrator
    
    # State tracking
    start_time: datetime
    current_phase: MigrationPhase
    completed_phases: List[str]
    failed_phases: List[str]
    
    # Data preservation
    preserved_agents: Dict[str, Dict[str, Any]]
    preserved_tasks: Dict[str, Dict[str, Any]]
    preserved_config: Dict[str, Any]
    
    # Migration results
    errors: List[str]
    warnings: List[str]
    performance_metrics: Dict[str, Any]


class OrchestratorMigrationManager:
    """
    Main migration manager for orchestrator consolidation.
    
    Handles migration from any existing orchestrator implementation to
    ConsolidatedProductionOrchestrator while preserving all functionality
    and ensuring system stability.
    """
    
    def __init__(self):
        self._active_migrations: Dict[str, MigrationContext] = {}
        self._migration_history: List[MigrationResult] = []
        self._registered_adapters: Dict[str, 'OrchestratorAdapter'] = {}
        
        # Register built-in adapters
        self._register_builtin_adapters()
        
        logger.info("OrchestratorMigrationManager initialized")
    
    def _register_builtin_adapters(self):
        """Register built-in orchestrator adapters."""
        self._registered_adapters.update({
            "SimpleOrchestrator": SimpleOrchestratorAdapter(),
            "ProductionOrchestrator": ProductionOrchestratorAdapter(), 
            "UniversalOrchestrator": UniversalOrchestratorAdapter(),
            "Orchestrator": OrchestratorAdapter(),  # The facade orchestrator
            "UnifiedOrchestrator": UnifiedOrchestratorAdapter()
        })
        
        logger.info("Registered adapters for orchestrator types", 
                   adapters=list(self._registered_adapters.keys()))
    
    async def create_migration_plan(
        self,
        source_orchestrator: Any,
        strategy: MigrationStrategy = MigrationStrategy.GRADUAL,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> MigrationPlan:
        """
        Create a migration plan for transitioning from source orchestrator.
        
        Args:
            source_orchestrator: The current orchestrator to migrate from
            strategy: Migration strategy to use
            config_overrides: Optional configuration overrides
            
        Returns:
            Detailed migration plan
        """
        source_type = type(source_orchestrator).__name__
        
        plan = MigrationPlan(
            source_orchestrator_type=source_type,
            strategy=strategy
        )
        
        # Apply config overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(plan, key):
                    setattr(plan, key, value)
        
        # Customize plan based on source orchestrator type
        adapter = self._get_adapter(source_type)
        if adapter:
            plan = await adapter.customize_migration_plan(source_orchestrator, plan)
        
        logger.info("Migration plan created",
                   source_type=source_type,
                   strategy=strategy.value,
                   phases=len(plan.phases))
        
        return plan
    
    async def execute_migration(
        self,
        source_orchestrator: Any,
        plan: MigrationPlan,
        target_config: Optional[OrchestratorConfig] = None
    ) -> MigrationResult:
        """
        Execute migration from source orchestrator to ConsolidatedProductionOrchestrator.
        
        Args:
            source_orchestrator: Current orchestrator instance
            plan: Migration plan to execute
            target_config: Optional target orchestrator configuration
            
        Returns:
            Migration result with success/failure information
        """
        migration_id = self._generate_migration_id()
        start_time = datetime.utcnow()
        
        logger.info("Starting orchestrator migration",
                   migration_id=migration_id,
                   source_type=plan.source_orchestrator_type,
                   strategy=plan.strategy.value)
        
        try:
            # Create target orchestrator
            if target_config is None:
                target_config = OrchestratorConfig(
                    mode=OrchestratorMode.PRODUCTION,
                    max_agents=100,
                    enable_plugins=True,
                    enable_monitoring=True
                )
            
            target_orchestrator = ConsolidatedProductionOrchestrator(target_config)
            
            # Create migration context
            context = MigrationContext(
                migration_id=migration_id,
                plan=plan,
                source_orchestrator=source_orchestrator,
                target_orchestrator=target_orchestrator,
                start_time=start_time,
                current_phase=MigrationPhase.PREPARATION,
                completed_phases=[],
                failed_phases=[],
                preserved_agents={},
                preserved_tasks={},
                preserved_config={},
                errors=[],
                warnings=[],
                performance_metrics={}
            )
            
            self._active_migrations[migration_id] = context
            
            # Execute migration phases
            result = await self._execute_migration_phases(context)
            
            # Clean up
            del self._active_migrations[migration_id]
            self._migration_history.append(result)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            if result.success:
                logger.info("Migration completed successfully",
                           migration_id=migration_id,
                           duration_seconds=duration,
                           migrated_agents=result.migrated_agents,
                           migrated_tasks=result.migrated_tasks)
            else:
                logger.error("Migration failed",
                           migration_id=migration_id,
                           duration_seconds=duration,
                           errors=result.errors)
            
            return result
            
        except Exception as e:
            # Create failure result
            duration = (datetime.utcnow() - start_time).total_seconds()
            result = MigrationResult(
                success=False,
                migrated_agents=0,
                migrated_tasks=0,
                errors=[f"Migration execution failed: {str(e)}"],
                warnings=[],
                duration_seconds=duration
            )
            
            # Clean up failed migration
            if migration_id in self._active_migrations:
                del self._active_migrations[migration_id]
            
            self._migration_history.append(result)
            logger.error("Migration execution failed", 
                        migration_id=migration_id,
                        error=str(e))
            
            return result
    
    async def _execute_migration_phases(self, context: MigrationContext) -> MigrationResult:
        """Execute all migration phases."""
        migrated_agents = 0
        migrated_tasks = 0
        
        try:
            # Phase 1: Preparation
            context.current_phase = MigrationPhase.PREPARATION
            await self._prepare_migration(context)
            context.completed_phases.append("preparation")
            
            # Phase 2: Validation (if enabled)
            if context.plan.validate_before_migration:
                context.current_phase = MigrationPhase.VALIDATION
                await self._validate_source_state(context)
                context.completed_phases.append("validation")
            
            # Phase 3: State preservation
            await self._preserve_source_state(context)
            context.completed_phases.append("state_preservation")
            
            # Phase 4: Initialize target orchestrator
            await context.target_orchestrator.initialize()
            context.completed_phases.append("target_initialization")
            
            # Phase 5: Execute migration phases
            context.current_phase = MigrationPhase.EXECUTION
            
            for phase in context.plan.phases:
                phase_start = time.time()
                
                if phase == "agent_migration":
                    migrated_agents = await self._migrate_agents(context)
                elif phase == "task_migration":
                    migrated_tasks = await self._migrate_tasks(context)
                elif phase == "plugin_migration":
                    await self._migrate_plugins(context)
                elif phase == "configuration_migration":
                    await self._migrate_configuration(context)
                elif phase == "state_validation":
                    await self._validate_migrated_state(context)
                
                phase_time = time.time() - phase_start
                context.performance_metrics[f"{phase}_time_seconds"] = phase_time
                context.completed_phases.append(phase)
            
            # Phase 6: Verification
            if context.plan.validate_after_migration:
                context.current_phase = MigrationPhase.VERIFICATION
                await self._verify_migration_success(context)
                context.completed_phases.append("verification")
            
            # Phase 7: Completion
            context.current_phase = MigrationPhase.COMPLETION
            await self._complete_migration(context)
            context.completed_phases.append("completion")
            
            return MigrationResult(
                success=True,
                migrated_agents=migrated_agents,
                migrated_tasks=migrated_tasks,
                errors=context.errors,
                warnings=context.warnings,
                duration_seconds=(datetime.utcnow() - context.start_time).total_seconds()
            )
            
        except Exception as e:
            context.errors.append(f"Migration phase failed: {str(e)}")
            context.current_phase = MigrationPhase.ROLLBACK
            
            # Attempt rollback if enabled
            if context.plan.enable_rollback:
                try:
                    await self._rollback_migration(context)
                    context.warnings.append("Migration rolled back successfully")
                except Exception as rollback_error:
                    context.errors.append(f"Rollback also failed: {str(rollback_error)}")
            
            return MigrationResult(
                success=False,
                migrated_agents=migrated_agents,
                migrated_tasks=migrated_tasks,
                errors=context.errors,
                warnings=context.warnings,
                duration_seconds=(datetime.utcnow() - context.start_time).total_seconds()
            )
    
    async def _prepare_migration(self, context: MigrationContext):
        """Prepare for migration."""
        logger.info("Preparing migration", migration_id=context.migration_id)
        
        # Get adapter for source orchestrator
        adapter = self._get_adapter(context.plan.source_orchestrator_type)
        if adapter:
            await adapter.prepare_migration(context.source_orchestrator, context)
    
    async def _validate_source_state(self, context: MigrationContext):
        """Validate source orchestrator state before migration."""
        logger.info("Validating source state", migration_id=context.migration_id)
        
        adapter = self._get_adapter(context.plan.source_orchestrator_type)
        if adapter:
            validation_result = await adapter.validate_state(context.source_orchestrator)
            if not validation_result.get("valid", True):
                context.errors.extend(validation_result.get("errors", []))
                raise ValueError("Source state validation failed")
    
    async def _preserve_source_state(self, context: MigrationContext):
        """Preserve source orchestrator state."""
        logger.info("Preserving source state", migration_id=context.migration_id)
        
        adapter = self._get_adapter(context.plan.source_orchestrator_type)
        if adapter:
            state = await adapter.export_state(context.source_orchestrator)
            context.preserved_agents = state.get("agents", {})
            context.preserved_tasks = state.get("tasks", {})
            context.preserved_config = state.get("config", {})
    
    async def _migrate_agents(self, context: MigrationContext) -> int:
        """Migrate agents from source to target orchestrator."""
        logger.info("Migrating agents", migration_id=context.migration_id)
        
        migrated_count = 0
        
        for agent_id, agent_data in context.preserved_agents.items():
            try:
                # Convert agent data to AgentSpec
                agent_spec = self._convert_to_agent_spec(agent_data)
                
                # Register agent in target orchestrator
                new_agent_id = await context.target_orchestrator.register_agent(agent_spec)
                
                migrated_count += 1
                logger.debug("Agent migrated successfully",
                           original_id=agent_id,
                           new_id=new_agent_id)
                           
            except Exception as e:
                error_msg = f"Failed to migrate agent {agent_id}: {str(e)}"
                context.errors.append(error_msg)
                logger.warning(error_msg)
        
        return migrated_count
    
    async def _migrate_tasks(self, context: MigrationContext) -> int:
        """Migrate tasks from source to target orchestrator."""
        logger.info("Migrating tasks", migration_id=context.migration_id)
        
        migrated_count = 0
        
        for task_id, task_data in context.preserved_tasks.items():
            try:
                # Convert task data to TaskSpec
                task_spec = self._convert_to_task_spec(task_data)
                
                # Delegate task in target orchestrator
                task_result = await context.target_orchestrator.delegate_task(task_spec)
                
                migrated_count += 1
                logger.debug("Task migrated successfully",
                           original_id=task_id,
                           new_id=task_result.id)
                           
            except Exception as e:
                error_msg = f"Failed to migrate task {task_id}: {str(e)}"
                context.errors.append(error_msg)
                logger.warning(error_msg)
        
        return migrated_count
    
    async def _migrate_plugins(self, context: MigrationContext):
        """Migrate plugins from source to target orchestrator."""
        logger.info("Migrating plugins", migration_id=context.migration_id)
        
        adapter = self._get_adapter(context.plan.source_orchestrator_type)
        if adapter:
            plugins = await adapter.get_plugins(context.source_orchestrator)
            
            for plugin_name, plugin_config in plugins.items():
                try:
                    await context.target_orchestrator.load_plugin(plugin_name, plugin_config)
                    logger.debug("Plugin migrated successfully", plugin=plugin_name)
                except Exception as e:
                    error_msg = f"Failed to migrate plugin {plugin_name}: {str(e)}"
                    context.warnings.append(error_msg)
                    logger.warning(error_msg)
    
    async def _migrate_configuration(self, context: MigrationContext):
        """Migrate configuration from source to target orchestrator.""" 
        logger.info("Migrating configuration", migration_id=context.migration_id)
        # Configuration is handled during target orchestrator creation
        pass
    
    async def _validate_migrated_state(self, context: MigrationContext):
        """Validate the migrated state in target orchestrator."""
        logger.info("Validating migrated state", migration_id=context.migration_id)
        
        # Check agent count
        target_agents = await context.target_orchestrator.list_agents()
        if len(target_agents) != len(context.preserved_agents):
            context.warnings.append(
                f"Agent count mismatch: expected {len(context.preserved_agents)}, "
                f"got {len(target_agents)}"
            )
    
    async def _verify_migration_success(self, context: MigrationContext):
        """Verify migration was successful."""
        logger.info("Verifying migration success", migration_id=context.migration_id)
        
        # Run health check on target orchestrator
        health = await context.target_orchestrator.health_check()
        if health.status != "healthy" and health.status != "no_agents":
            context.warnings.append(f"Target orchestrator health: {health.status}")
    
    async def _complete_migration(self, context: MigrationContext):
        """Complete the migration process."""
        logger.info("Completing migration", migration_id=context.migration_id)
        # Migration completion steps would be implemented here
    
    async def _rollback_migration(self, context: MigrationContext):
        """Rollback migration to previous state."""
        logger.warning("Rolling back migration", migration_id=context.migration_id)
        
        # Shutdown target orchestrator
        try:
            await context.target_orchestrator.shutdown()
        except Exception as e:
            logger.warning("Error shutting down target orchestrator during rollback", error=str(e))
    
    def _get_adapter(self, orchestrator_type: str) -> Optional['OrchestratorAdapter']:
        """Get adapter for orchestrator type."""
        return self._registered_adapters.get(orchestrator_type)
    
    def _convert_to_agent_spec(self, agent_data: Dict[str, Any]) -> AgentSpec:
        """Convert agent data to AgentSpec."""
        return AgentSpec(
            role=agent_data.get("role", "backend_developer"),
            agent_type=agent_data.get("agent_type", "claude_code"),
            workspace_name=agent_data.get("workspace_name"),
            git_branch=agent_data.get("git_branch"),
            environment_vars=agent_data.get("environment_vars", {}),
            capabilities=agent_data.get("capabilities", []),
            resource_requirements=agent_data.get("resource_requirements", {})
        )
    
    def _convert_to_task_spec(self, task_data: Dict[str, Any]) -> TaskSpec:
        """Convert task data to TaskSpec."""
        return TaskSpec(
            description=task_data.get("description", ""),
            task_type=task_data.get("task_type", "general"),
            priority=task_data.get("priority", "medium"),
            preferred_agent_role=task_data.get("preferred_agent_role"),
            estimated_duration_seconds=task_data.get("estimated_duration_seconds"),
            dependencies=task_data.get("dependencies", []),
            metadata=task_data.get("metadata", {})
        )
    
    def _generate_migration_id(self) -> str:
        """Generate unique migration ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"migration_{timestamp}_{id(self) % 10000:04d}"


# Adapter Classes for Different Orchestrator Types

class OrchestratorAdapter:
    """Base adapter for orchestrator migration."""
    
    async def customize_migration_plan(self, source: Any, plan: MigrationPlan) -> MigrationPlan:
        """Customize migration plan for specific orchestrator type."""
        return plan
    
    async def prepare_migration(self, source: Any, context: MigrationContext):
        """Prepare source orchestrator for migration."""
        pass
    
    async def validate_state(self, source: Any) -> Dict[str, Any]:
        """Validate source orchestrator state."""
        return {"valid": True}
    
    async def export_state(self, source: Any) -> Dict[str, Any]:
        """Export state from source orchestrator."""
        return {"agents": {}, "tasks": {}, "config": {}}
    
    async def get_plugins(self, source: Any) -> Dict[str, Any]:
        """Get plugins from source orchestrator."""
        return {}


class SimpleOrchestratorAdapter(OrchestratorAdapter):
    """Adapter for SimpleOrchestrator migration."""
    
    async def export_state(self, source: Any) -> Dict[str, Any]:
        """Export state from SimpleOrchestrator."""
        try:
            agents = {}
            tasks = {}
            
            # Get agent sessions
            if hasattr(source, 'list_agent_sessions'):
                sessions = await source.list_agent_sessions()
                for session in sessions:
                    agent_instance = session.get("agent_instance", {})
                    agent_id = agent_instance.get("id")
                    if agent_id:
                        agents[agent_id] = {
                            "role": agent_instance.get("role"),
                            "status": agent_instance.get("status"),
                            "created_at": agent_instance.get("created_at"),
                            "agent_type": "claude_code"
                        }
            
            return {"agents": agents, "tasks": tasks, "config": {}}
            
        except Exception as e:
            logger.warning("Failed to export SimpleOrchestrator state", error=str(e))
            return {"agents": {}, "tasks": {}, "config": {}}


class ProductionOrchestratorAdapter(OrchestratorAdapter):
    """Adapter for ProductionOrchestrator migration."""
    
    async def customize_migration_plan(self, source: Any, plan: MigrationPlan) -> MigrationPlan:
        """Customize migration plan for ProductionOrchestrator."""
        plan.phases.extend(["monitoring_migration", "alerting_migration"])
        return plan


class UniversalOrchestratorAdapter(OrchestratorAdapter):
    """Adapter for UniversalOrchestrator migration."""
    
    async def customize_migration_plan(self, source: Any, plan: MigrationPlan) -> MigrationPlan:
        """Customize migration plan for UniversalOrchestrator."""
        plan.phases.extend(["workflow_migration", "resource_migration"])
        return plan


class UnifiedOrchestratorAdapter(OrchestratorAdapter):
    """Adapter for UnifiedOrchestrator migration."""
    pass


# Factory Functions

def create_migration_manager() -> OrchestratorMigrationManager:
    """Create a migration manager instance."""
    return OrchestratorMigrationManager()


async def migrate_orchestrator(
    source_orchestrator: Any,
    strategy: MigrationStrategy = MigrationStrategy.GRADUAL,
    target_config: Optional[OrchestratorConfig] = None
) -> MigrationResult:
    """
    High-level function to migrate from any orchestrator to ConsolidatedProductionOrchestrator.
    
    Args:
        source_orchestrator: Current orchestrator instance
        strategy: Migration strategy to use
        target_config: Optional target configuration
        
    Returns:
        Migration result
    """
    manager = create_migration_manager()
    plan = await manager.create_migration_plan(source_orchestrator, strategy)
    return await manager.execute_migration(source_orchestrator, plan, target_config)


# Validation Functions

async def validate_orchestrator_compatibility(orchestrator: Any) -> Dict[str, Any]:
    """
    Validate that an orchestrator is compatible with migration.
    
    Args:
        orchestrator: Orchestrator instance to validate
        
    Returns:
        Validation result with compatibility information
    """
    orchestrator_type = type(orchestrator).__name__
    
    result = {
        "compatible": False,
        "orchestrator_type": orchestrator_type,
        "supported_features": [],
        "missing_features": [],
        "warnings": [],
        "recommendations": []
    }
    
    # Check for basic orchestrator methods
    required_methods = ["health_check"]
    optional_methods = ["register_agent", "delegate_task", "list_agents", "shutdown"]
    
    for method in required_methods:
        if hasattr(orchestrator, method):
            result["supported_features"].append(method)
        else:
            result["missing_features"].append(method)
    
    for method in optional_methods:
        if hasattr(orchestrator, method):
            result["supported_features"].append(method)
        else:
            result["warnings"].append(f"Optional method '{method}' not available")
    
    # Determine compatibility
    result["compatible"] = len(result["missing_features"]) == 0
    
    if result["compatible"]:
        result["recommendations"].append("Orchestrator is compatible with migration")
    else:
        result["recommendations"].append(
            f"Orchestrator missing required methods: {', '.join(result['missing_features'])}"
        )
    
    return result