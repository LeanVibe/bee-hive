"""
Migration Utilities for Engine Consolidation

Provides utilities to migrate from existing engine implementations
to the consolidated engine system while maintaining backward compatibility.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Type
from datetime import datetime
from dataclasses import dataclass, asdict
import json

from .consolidated_engine import (
    EngineCoordinationLayer,
    ConsolidatedWorkflowEngine, 
    ConsolidatedTaskExecutionEngine,
    ConsolidatedCommunicationEngine,
    WorkflowExecutionContext,
    TaskExecutionContext,
    CommunicationMessage
)

logger = logging.getLogger(__name__)


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    success: bool
    source_engine: str
    target_engine: str
    migrated_items: int
    failed_items: int
    error_messages: List[str]
    migration_time_ms: float
    compatibility_warnings: List[str]


class EngineConfigMigrator:
    """Migrates engine configurations to consolidated format."""
    
    LEGACY_WORKFLOW_MAPPINGS = {
        # Legacy workflow type -> New workflow type
        "orchestration_workflow": "agent_coordination",
        "task_workflow": "task_pipeline", 
        "data_workflow": "data_processing",
        "monitor_workflow": "monitoring_workflow",
        "deploy_workflow": "deployment_workflow",
        "test_workflow": "testing_workflow",
        "integration_workflow": "integration_workflow",
        "notify_workflow": "notification_workflow"
    }
    
    LEGACY_TASK_MAPPINGS = {
        # Legacy task type -> New task type
        "code_gen": "code_generation",
        "code_check": "code_review",
        "test_run": "testing",
        "doc_gen": "documentation", 
        "deploy": "deployment",
        "monitor": "monitoring",
        "validate": "validation",
        "optimize": "optimization",
        "scan": "security_scan",
        "perf_test": "performance_analysis",
        "integration": "integration_test",
        "unit": "unit_test"
    }
    
    @classmethod
    def migrate_workflow_config(cls, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate legacy workflow configuration to consolidated format."""
        migrated = {}
        warnings = []
        
        # Map workflow ID
        if "id" in legacy_config:
            migrated["workflow_id"] = legacy_config["id"]
        elif "workflow_id" in legacy_config:
            migrated["workflow_id"] = legacy_config["workflow_id"]
        else:
            migrated["workflow_id"] = f"migrated_workflow_{datetime.utcnow().timestamp()}"
            warnings.append("No workflow ID found, generated new ID")
        
        # Map workflow type
        legacy_type = legacy_config.get("type", legacy_config.get("workflow_type", "unknown"))
        migrated["workflow_type"] = cls.LEGACY_WORKFLOW_MAPPINGS.get(legacy_type, legacy_type)
        if legacy_type in cls.LEGACY_WORKFLOW_MAPPINGS:
            warnings.append(f"Mapped legacy type '{legacy_type}' to '{migrated['workflow_type']}'")
        
        # Map steps
        legacy_steps = legacy_config.get("steps", legacy_config.get("definition", {}).get("steps", []))
        migrated_steps = []
        
        for i, step in enumerate(legacy_steps):
            migrated_step = {}
            
            # Map step ID
            migrated_step["step_id"] = step.get("id", step.get("name", f"step_{i}"))
            
            # Map step type
            step_type = step.get("type", step.get("action", "processing"))
            migrated_step["type"] = step_type
            
            # Map step parameters
            if "params" in step:
                migrated_step.update(step["params"])
            elif "parameters" in step:
                migrated_step.update(step["parameters"])
                
            # Copy other step properties
            for key in ["agent_role", "validation_type", "timeout", "dependencies"]:
                if key in step:
                    migrated_step[key] = step[key]
            
            migrated_steps.append(migrated_step)
        
        migrated["steps"] = migrated_steps
        
        # Map metadata
        metadata = {}
        for key in ["priority", "timeout", "tags", "description", "owner"]:
            if key in legacy_config:
                metadata[key] = legacy_config[key]
        
        if metadata:
            migrated["metadata"] = metadata
        
        migrated["_migration_warnings"] = warnings
        return migrated
    
    @classmethod
    def migrate_task_config(cls, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate legacy task configuration to consolidated format."""
        migrated = {}
        warnings = []
        
        # Map task ID
        if "id" in legacy_config:
            migrated["task_id"] = legacy_config["id"]
        elif "task_id" in legacy_config:
            migrated["task_id"] = legacy_config["task_id"]
        else:
            migrated["task_id"] = f"migrated_task_{datetime.utcnow().timestamp()}"
            warnings.append("No task ID found, generated new ID")
        
        # Map task type
        legacy_type = legacy_config.get("type", legacy_config.get("task_type", "unknown"))
        migrated["task_type"] = cls.LEGACY_TASK_MAPPINGS.get(legacy_type, legacy_type)
        if legacy_type in cls.LEGACY_TASK_MAPPINGS:
            warnings.append(f"Mapped legacy type '{legacy_type}' to '{migrated['task_type']}'")
        
        # Map priority
        migrated["priority"] = legacy_config.get("priority", "normal")
        
        # Map requirements
        requirements = {}
        if "params" in legacy_config:
            requirements.update(legacy_config["params"])
        elif "parameters" in legacy_config:
            requirements.update(legacy_config["parameters"])
        elif "requirements" in legacy_config:
            requirements.update(legacy_config["requirements"])
        
        # Handle common legacy parameter mappings
        legacy_mappings = {
            "lang": "language",
            "fw": "framework",
            "complexity": "complexity",
            "timeout": "max_execution_time"
        }
        
        for old_key, new_key in legacy_mappings.items():
            if old_key in requirements:
                requirements[new_key] = requirements.pop(old_key)
                warnings.append(f"Mapped parameter '{old_key}' to '{new_key}'")
        
        if requirements:
            migrated["requirements"] = requirements
        
        # Map constraints
        constraints = {}
        if "constraints" in legacy_config:
            constraints.update(legacy_config["constraints"])
        elif "limits" in legacy_config:
            constraints.update(legacy_config["limits"])
        
        if constraints:
            migrated["constraints"] = constraints
        
        migrated["_migration_warnings"] = warnings
        return migrated
    
    @classmethod
    def migrate_communication_config(cls, legacy_config: Dict[str, Any]) -> CommunicationMessage:
        """Migrate legacy communication configuration to consolidated format."""
        warnings = []
        
        # Extract message data
        message_id = legacy_config.get("id", legacy_config.get("message_id", f"migrated_msg_{datetime.utcnow().timestamp()}"))
        source = legacy_config.get("from", legacy_config.get("source", "unknown"))
        destination = legacy_config.get("to", legacy_config.get("destination", "unknown"))
        message_type = legacy_config.get("type", legacy_config.get("message_type", "unknown"))
        
        # Handle legacy content formats
        content = legacy_config.get("content", legacy_config.get("data", legacy_config.get("payload", {})))
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                content = {"message": content}
        
        # Map priority
        priority = legacy_config.get("priority", "normal")
        if priority not in ["low", "normal", "high", "urgent"]:
            priority = "normal"
            warnings.append(f"Unknown priority '{legacy_config.get('priority')}', defaulted to 'normal'")
        
        # Create message
        message = CommunicationMessage(
            message_id=message_id,
            source=source,
            destination=destination,
            message_type=message_type,
            content=content,
            priority=priority,
            timestamp=datetime.utcnow()
        )
        
        if warnings:
            message.metadata = message.metadata or {}
            message.metadata["_migration_warnings"] = warnings
        
        return message


class EngineMigrationManager:
    """Manages migration from legacy engines to consolidated system."""
    
    def __init__(self, target_coordinator: EngineCoordinationLayer):
        """Initialize migration manager with target coordinator."""
        self.target_coordinator = target_coordinator
        self.migration_stats = {
            "workflows_migrated": 0,
            "tasks_migrated": 0,
            "messages_migrated": 0,
            "failed_migrations": 0
        }
    
    async def migrate_workflow_engine(self, legacy_workflows: List[Dict[str, Any]]) -> MigrationResult:
        """Migrate workflows from legacy workflow engine."""
        start_time = datetime.utcnow()
        migrated_items = 0
        failed_items = 0
        error_messages = []
        compatibility_warnings = []
        
        for workflow_config in legacy_workflows:
            try:
                # Migrate configuration
                migrated_config = EngineConfigMigrator.migrate_workflow_config(workflow_config)
                
                # Extract warnings
                if "_migration_warnings" in migrated_config:
                    compatibility_warnings.extend(migrated_config.pop("_migration_warnings"))
                
                # Execute migrated workflow
                result = await self.target_coordinator.execute_workflow(
                    migrated_config["workflow_id"],
                    migrated_config
                )
                
                if result.success:
                    migrated_items += 1
                    self.migration_stats["workflows_migrated"] += 1
                else:
                    failed_items += 1
                    error_messages.append(f"Workflow {migrated_config['workflow_id']}: {result.error_message}")
                    
            except Exception as e:
                failed_items += 1
                self.migration_stats["failed_migrations"] += 1
                error_messages.append(f"Migration error: {str(e)}")
        
        migration_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return MigrationResult(
            success=(failed_items == 0),
            source_engine="legacy_workflow_engine",
            target_engine="consolidated_workflow_engine",
            migrated_items=migrated_items,
            failed_items=failed_items,
            error_messages=error_messages,
            migration_time_ms=migration_time,
            compatibility_warnings=compatibility_warnings
        )
    
    async def migrate_task_engine(self, legacy_tasks: List[Dict[str, Any]]) -> MigrationResult:
        """Migrate tasks from legacy task engine."""
        start_time = datetime.utcnow()
        migrated_items = 0
        failed_items = 0
        error_messages = []
        compatibility_warnings = []
        
        for task_config in legacy_tasks:
            try:
                # Migrate configuration
                migrated_config = EngineConfigMigrator.migrate_task_config(task_config)
                
                # Extract warnings
                if "_migration_warnings" in migrated_config:
                    compatibility_warnings.extend(migrated_config.pop("_migration_warnings"))
                
                # Execute migrated task
                result = await self.target_coordinator.execute_task(
                    migrated_config["task_id"],
                    migrated_config
                )
                
                if result.success:
                    migrated_items += 1
                    self.migration_stats["tasks_migrated"] += 1
                else:
                    failed_items += 1
                    error_messages.append(f"Task {migrated_config['task_id']}: {result.error_message}")
                    
            except Exception as e:
                failed_items += 1
                self.migration_stats["failed_migrations"] += 1
                error_messages.append(f"Migration error: {str(e)}")
        
        migration_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return MigrationResult(
            success=(failed_items == 0),
            source_engine="legacy_task_engine",
            target_engine="consolidated_task_engine",
            migrated_items=migrated_items,
            failed_items=failed_items,
            error_messages=error_messages,
            migration_time_ms=migration_time,
            compatibility_warnings=compatibility_warnings
        )
    
    async def migrate_communication_engine(self, legacy_messages: List[Dict[str, Any]]) -> MigrationResult:
        """Migrate messages from legacy communication engine."""
        start_time = datetime.utcnow()
        migrated_items = 0
        failed_items = 0
        error_messages = []
        compatibility_warnings = []
        
        for message_config in legacy_messages:
            try:
                # Migrate configuration
                migrated_message = EngineConfigMigrator.migrate_communication_config(message_config)
                
                # Extract warnings
                if migrated_message.metadata and "_migration_warnings" in migrated_message.metadata:
                    compatibility_warnings.extend(migrated_message.metadata.pop("_migration_warnings"))
                
                # Send migrated message
                result = await self.target_coordinator.send_message(migrated_message)
                
                if result.success:
                    migrated_items += 1
                    self.migration_stats["messages_migrated"] += 1
                else:
                    failed_items += 1
                    error_messages.append(f"Message {migrated_message.message_id}: {result.error_message}")
                    
            except Exception as e:
                failed_items += 1
                self.migration_stats["failed_migrations"] += 1
                error_messages.append(f"Migration error: {str(e)}")
        
        migration_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return MigrationResult(
            success=(failed_items == 0),
            source_engine="legacy_communication_engine",
            target_engine="consolidated_communication_engine",
            migrated_items=migrated_items,
            failed_items=failed_items,
            error_messages=error_messages,
            migration_time_ms=migration_time,
            compatibility_warnings=compatibility_warnings
        )
    
    async def migrate_all_engines(
        self, 
        legacy_workflows: List[Dict[str, Any]] = None,
        legacy_tasks: List[Dict[str, Any]] = None,
        legacy_messages: List[Dict[str, Any]] = None
    ) -> Dict[str, MigrationResult]:
        """Migrate all engines in parallel."""
        results = {}
        
        migration_tasks = []
        
        if legacy_workflows:
            migration_tasks.append(("workflows", self.migrate_workflow_engine(legacy_workflows)))
        
        if legacy_tasks:
            migration_tasks.append(("tasks", self.migrate_task_engine(legacy_tasks)))
        
        if legacy_messages:
            migration_tasks.append(("messages", self.migrate_communication_engine(legacy_messages)))
        
        # Execute migrations in parallel
        import asyncio
        completed_tasks = await asyncio.gather(*[task for name, task in migration_tasks], return_exceptions=True)
        
        # Collect results
        for i, (name, task) in enumerate(migration_tasks):
            result = completed_tasks[i]
            if isinstance(result, Exception):
                results[name] = MigrationResult(
                    success=False,
                    source_engine=f"legacy_{name}_engine",
                    target_engine=f"consolidated_{name}_engine",
                    migrated_items=0,
                    failed_items=1,
                    error_messages=[str(result)],
                    migration_time_ms=0,
                    compatibility_warnings=[]
                )
            else:
                results[name] = result
        
        return results
    
    def get_migration_summary(self) -> Dict[str, Any]:
        """Get summary of all migration operations."""
        return {
            "total_workflows_migrated": self.migration_stats["workflows_migrated"],
            "total_tasks_migrated": self.migration_stats["tasks_migrated"],
            "total_messages_migrated": self.migration_stats["messages_migrated"],
            "total_failed_migrations": self.migration_stats["failed_migrations"],
            "overall_success_rate": self._calculate_success_rate(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall migration success rate."""
        total_attempted = sum([
            self.migration_stats["workflows_migrated"],
            self.migration_stats["tasks_migrated"], 
            self.migration_stats["messages_migrated"],
            self.migration_stats["failed_migrations"]
        ])
        
        if total_attempted == 0:
            return 0.0
        
        successful = total_attempted - self.migration_stats["failed_migrations"]
        return (successful / total_attempted) * 100


class BackwardCompatibilityLayer:
    """Provides backward compatibility for legacy engine interfaces."""
    
    def __init__(self, coordinator: EngineCoordinationLayer):
        """Initialize compatibility layer with coordinator."""
        self.coordinator = coordinator
    
    async def execute_legacy_workflow(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow using legacy interface format."""
        try:
            # Migrate and execute
            migrated_config = EngineConfigMigrator.migrate_workflow_config(workflow_data)
            result = await self.coordinator.execute_workflow(
                migrated_config["workflow_id"],
                migrated_config
            )
            
            # Convert result to legacy format
            return {
                "id": result.workflow_id,
                "success": result.success,
                "execution_time": result.execution_time_ms,
                "steps_completed": result.steps_completed,
                "error": result.error_message if not result.success else None,
                "warnings": migrated_config.get("_migration_warnings", [])
            }
            
        except Exception as e:
            return {
                "id": workflow_data.get("id", "unknown"),
                "success": False,
                "error": str(e),
                "execution_time": 0,
                "steps_completed": 0
            }
    
    async def execute_legacy_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using legacy interface format."""
        try:
            # Migrate and execute
            migrated_config = EngineConfigMigrator.migrate_task_config(task_data)
            result = await self.coordinator.execute_task(
                migrated_config["task_id"],
                migrated_config
            )
            
            # Convert result to legacy format
            return {
                "id": result.task_id,
                "success": result.success,
                "execution_time": result.execution_time_ms,
                "output": result.task_output,
                "error": result.error_message if not result.success else None,
                "warnings": migrated_config.get("_migration_warnings", [])
            }
            
        except Exception as e:
            return {
                "id": task_data.get("id", "unknown"),
                "success": False,
                "error": str(e),
                "execution_time": 0,
                "output": None
            }
    
    async def send_legacy_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send message using legacy interface format."""
        try:
            # Migrate and send
            migrated_message = EngineConfigMigrator.migrate_communication_config(message_data)
            result = await self.coordinator.send_message(migrated_message)
            
            # Convert result to legacy format
            return {
                "id": result.message_id,
                "success": result.success,
                "delivery_time": result.delivery_time_ms,
                "error": result.error_message if not result.success else None,
                "warnings": migrated_message.metadata.get("_migration_warnings", []) if migrated_message.metadata else []
            }
            
        except Exception as e:
            return {
                "id": message_data.get("id", "unknown"),
                "success": False,
                "error": str(e),
                "delivery_time": 0
            }