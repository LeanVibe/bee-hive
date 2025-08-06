"""
Command Registry for LeanVibe Agent Hive 2.0 - Phase 6.1

Manages registration, discovery, and validation of custom multi-agent workflow commands.
Provides secure command storage with version management and capability matching.
"""

import asyncio
import uuid
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import hashlib

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.dialects.postgresql import JSONB

from .database import get_session
from .advanced_security_validator import AdvancedSecurityValidator, CommandContext
from .agent_registry import AgentRegistry
from ..schemas.custom_commands import (
    CommandDefinition, CommandValidationResult, CommandMetrics,
    AgentRole, AgentRequirement, WorkflowStep, SecurityPolicy
)

logger = structlog.get_logger()


class CommandRegistryError(Exception):
    """Base exception for command registry operations."""
    pass


class CommandValidationError(CommandRegistryError):
    """Exception raised when command validation fails."""
    pass


class CommandConflictError(CommandRegistryError):
    """Exception raised when command conflicts occur."""
    pass


class CommandRegistry:
    """
    Enhanced command registry for multi-agent workflow commands.
    
    Features:
    - Secure command registration with validation
    - Version management and compatibility checks
    - Agent capability matching and availability validation
    - Command discovery with filtering and search
    - Performance metrics and usage tracking
    """
    
    def __init__(
        self,
        agent_registry: Optional[AgentRegistry] = None,
        security_validator: Optional[AdvancedSecurityValidator] = None,
        command_storage_path: Optional[Path] = None
    ):
        self.agent_registry = agent_registry
        self.security_validator = security_validator
        self.command_storage_path = command_storage_path or Path("./commands")
        
        # In-memory command cache for performance
        self._command_cache: Dict[str, Dict[str, CommandDefinition]] = {}
        self._cache_last_updated: Dict[str, datetime] = {}
        self._cache_ttl_seconds = 300  # 5 minutes
        
        # Command execution tracking
        self._execution_metrics: Dict[str, CommandMetrics] = {}
        
        # Security settings
        self.max_command_size_mb = 10
        self.max_workflow_steps = 100
        self.max_parallel_agents = 20
        
        # Validation settings
        self.validate_agent_availability = True
        self.require_security_approval = True
        self.enable_command_signing = True
        
        logger.info(
            "CommandRegistry initialized",
            storage_path=str(self.command_storage_path),
            cache_ttl=self._cache_ttl_seconds
        )
    
    async def register_command(
        self,
        definition: CommandDefinition,
        author_id: Optional[str] = None,
        validate_agents: bool = True,
        dry_run: bool = False
    ) -> Tuple[bool, CommandValidationResult]:
        """
        Register a new command with comprehensive validation.
        
        Args:
            definition: Command definition to register
            author_id: ID of the user registering the command
            validate_agents: Whether to validate agent availability
            dry_run: Perform validation only without registration
            
        Returns:
            Tuple of (success, validation_result)
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate command definition
            validation_result = await self._validate_command_definition(
                definition, validate_agents
            )
            
            if not validation_result.is_valid:
                logger.warning(
                    "Command validation failed",
                    command_name=definition.name,
                    errors=validation_result.errors
                )
                return False, validation_result
            
            if dry_run:
                logger.info(
                    "Command dry run validation completed",
                    command_name=definition.name,
                    duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
                )
                return True, validation_result
            
            # Check for existing command with same name
            existing_command = await self._get_command_from_db(definition.name)
            if existing_command:
                if existing_command.get("version") == definition.version:
                    raise CommandConflictError(
                        f"Command {definition.name} version {definition.version} already exists"
                    )
            
            # Generate command signature for integrity
            command_signature = self._generate_command_signature(definition)
            
            # Store command in database
            command_id = await self._store_command_in_db(
                definition, author_id, command_signature
            )
            
            # Update cache
            self._update_command_cache(definition)
            
            # Initialize metrics
            self._execution_metrics[definition.name] = CommandMetrics(
                command_name=definition.name,
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                average_execution_time=0.0,
                success_rate=0.0
            )
            
            # Store command definition file
            await self._store_command_file(definition)
            
            registration_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.info(
                "Command registered successfully",
                command_id=str(command_id),
                command_name=definition.name,
                version=definition.version,
                registration_time_ms=registration_time,
                workflow_steps=len(definition.workflow),
                required_agents=len(definition.agents)
            )
            
            return True, validation_result
            
        except Exception as e:
            logger.error(
                "Command registration failed",
                command_name=definition.name,
                error=str(e)
            )
            raise CommandRegistryError(f"Failed to register command: {str(e)}")
    
    async def get_command(
        self,
        command_name: str,
        version: Optional[str] = None
    ) -> Optional[CommandDefinition]:
        """
        Get command definition by name and optional version.
        
        Args:
            command_name: Name of the command
            version: Specific version (latest if not specified)
            
        Returns:
            Command definition or None if not found
        """
        try:
            # Check cache first
            if command_name in self._command_cache:
                cached_commands = self._command_cache[command_name]
                cache_time = self._cache_last_updated.get(command_name, datetime.min)
                
                if (datetime.utcnow() - cache_time).total_seconds() < self._cache_ttl_seconds:
                    if version:
                        return cached_commands.get(version)
                    else:
                        # Return latest version
                        if cached_commands:
                            latest_version = max(cached_commands.keys(), key=lambda v: tuple(map(int, v.split('.'))))
                            return cached_commands[latest_version]
            
            # Load from database
            command_data = await self._get_command_from_db(command_name, version)
            if not command_data:
                return None
            
            # Parse and validate command definition
            definition = CommandDefinition(**command_data["definition"])
            
            # Update cache
            self._update_command_cache(definition)
            
            return definition
            
        except Exception as e:
            logger.error(
                "Failed to get command",
                command_name=command_name,
                version=version,
                error=str(e)
            )
            return None
    
    async def list_commands(
        self,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        agent_role: Optional[AgentRole] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        List available commands with filtering options.
        
        Args:
            category: Filter by command category
            tag: Filter by command tag
            agent_role: Filter by required agent role
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            Tuple of (command_list, total_count)
        """
        try:
            async with get_session() as db:
                # Build query
                query = select(
                    func.distinct("command_registry.name"),
                    "command_registry.name",
                    "command_registry.version",
                    "command_registry.definition",
                    "command_registry.created_at",
                    "command_registry.author_id",
                    "command_registry.execution_count",
                    "command_registry.success_rate"
                ).select_from("command_registry")
                
                # Apply filters
                conditions = []
                if category:
                    conditions.append("command_registry.definition->>'category' = :category")
                if tag:
                    conditions.append("command_registry.definition->'tags' ? :tag")
                if agent_role:
                    conditions.append(
                        "EXISTS (SELECT 1 FROM jsonb_array_elements(command_registry.definition->'agents') AS agent "
                        "WHERE agent->>'role' = :agent_role)"
                    )
                
                if conditions:
                    query = query.where(" AND ".join(conditions))
                
                # Get total count
                count_query = select(func.count(func.distinct("command_registry.name"))).select_from("command_registry")
                if conditions:
                    count_query = count_query.where(" AND ".join(conditions))
                
                # Execute queries
                params = {}
                if category:
                    params["category"] = category
                if tag:
                    params["tag"] = tag
                if agent_role:
                    params["agent_role"] = agent_role.value
                
                total_result = await db.execute(count_query.params(**params))
                total_count = total_result.scalar() or 0
                
                # Get paginated results
                query = query.order_by("command_registry.created_at DESC").offset(offset).limit(limit)
                result = await db.execute(query.params(**params))
                rows = result.fetchall()
                
                # Format results
                commands = []
                for row in rows:
                    definition = row[2]  # definition column
                    command_info = {
                        "name": row[1],
                        "version": row[1],
                        "description": definition.get("description", ""),
                        "category": definition.get("category", "general"),
                        "tags": definition.get("tags", []),
                        "required_agents": len(definition.get("agents", [])),
                        "workflow_steps": len(definition.get("workflow", [])),
                        "created_at": row[4].isoformat() if row[4] else None,
                        "author_id": row[5],
                        "execution_count": row[6] or 0,
                        "success_rate": row[7] or 0.0
                    }
                    commands.append(command_info)
                
                return commands, total_count
                
        except Exception as e:
            logger.error("Failed to list commands", error=str(e))
            return [], 0
    
    async def validate_command(
        self,
        definition: CommandDefinition,
        validate_agents: bool = True
    ) -> CommandValidationResult:
        """
        Validate command definition without registration.
        
        Args:
            definition: Command definition to validate
            validate_agents: Whether to validate agent availability
            
        Returns:
            Validation result with errors and warnings
        """
        return await self._validate_command_definition(definition, validate_agents)
    
    async def delete_command(
        self,
        command_name: str,
        version: Optional[str] = None,
        author_id: Optional[str] = None
    ) -> bool:
        """
        Delete command(s) from registry.
        
        Args:
            command_name: Name of command to delete
            version: Specific version (all versions if not specified)
            author_id: Author ID for authorization check
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with get_session() as db:
                # Build delete query
                conditions = ["name = :command_name"]
                params = {"command_name": command_name}
                
                if version:
                    conditions.append("version = :version")
                    params["version"] = version
                
                if author_id:
                    conditions.append("author_id = :author_id")
                    params["author_id"] = author_id
                
                delete_query = f"""
                    DELETE FROM command_registry 
                    WHERE {' AND '.join(conditions)}
                """
                
                result = await db.execute(delete_query, params)
                await db.commit()
                
                deleted_count = result.rowcount
                
                if deleted_count > 0:
                    # Clear cache
                    if command_name in self._command_cache:
                        if version:
                            self._command_cache[command_name].pop(version, None)
                            if not self._command_cache[command_name]:
                                del self._command_cache[command_name]
                        else:
                            del self._command_cache[command_name]
                    
                    # Remove metrics
                    self._execution_metrics.pop(command_name, None)
                    
                    # Delete command files
                    await self._delete_command_files(command_name, version)
                    
                    logger.info(
                        "Command deleted successfully",
                        command_name=command_name,
                        version=version,
                        deleted_count=deleted_count
                    )
                    return True
                else:
                    logger.warning(
                        "No commands found to delete",
                        command_name=command_name,
                        version=version
                    )
                    return False
                
        except Exception as e:
            logger.error(
                "Failed to delete command",
                command_name=command_name,
                version=version,
                error=str(e)
            )
            return False
    
    async def get_command_metrics(self, command_name: str) -> Optional[CommandMetrics]:
        """Get execution metrics for a command."""
        return self._execution_metrics.get(command_name)
    
    async def update_execution_metrics(
        self,
        command_name: str,
        execution_time_seconds: float,
        success: bool
    ) -> None:
        """Update command execution metrics."""
        try:
            metrics = self._execution_metrics.get(command_name)
            if not metrics:
                metrics = CommandMetrics(
                    command_name=command_name,
                    total_executions=0,
                    successful_executions=0,
                    failed_executions=0,
                    average_execution_time=0.0,
                    success_rate=0.0
                )
                self._execution_metrics[command_name] = metrics
            
            # Update metrics
            metrics.total_executions += 1
            if success:
                metrics.successful_executions += 1
            else:
                metrics.failed_executions += 1
            
            # Update average execution time
            current_avg = metrics.average_execution_time
            total_executions = metrics.total_executions
            metrics.average_execution_time = (
                (current_avg * (total_executions - 1) + execution_time_seconds) / total_executions
            )
            
            # Update success rate
            metrics.success_rate = (metrics.successful_executions / metrics.total_executions) * 100
            metrics.last_executed = datetime.utcnow()
            
            # Update database metrics
            await self._update_command_metrics_in_db(command_name, metrics)
            
        except Exception as e:
            logger.error(
                "Failed to update execution metrics",
                command_name=command_name,
                error=str(e)
            )
    
    # Private helper methods
    
    async def _validate_command_definition(
        self,
        definition: CommandDefinition,
        validate_agents: bool = True
    ) -> CommandValidationResult:
        """Perform comprehensive command validation."""
        errors = []
        warnings = []
        agent_availability = {}
        
        try:
            # Basic schema validation (handled by Pydantic)
            
            # Validate workflow structure
            workflow_errors = self._validate_workflow_structure(definition.workflow)
            errors.extend(workflow_errors)
            
            # Validate agent requirements
            if validate_agents and self.agent_registry:
                agent_errors, availability = await self._validate_agent_requirements(definition.agents)
                errors.extend(agent_errors)
                agent_availability.update(availability)
            
            # Security validation using AdvancedSecurityValidator
            if self.security_validator:
                security_errors = await self._validate_command_security(definition, validate_agents)
                errors.extend(security_errors)
            
            # Performance validation
            perf_warnings = self._validate_performance_requirements(definition)
            warnings.extend(perf_warnings)
            
            # Estimate execution duration
            estimated_duration = self._estimate_execution_duration(definition)
            
            return CommandValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                agent_availability=agent_availability,
                estimated_duration=estimated_duration
            )
            
        except Exception as e:
            logger.error("Command validation error", error=str(e))
            return CommandValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=warnings,
                agent_availability=agent_availability
            )
    
    def _validate_workflow_structure(self, workflow: List[WorkflowStep]) -> List[str]:
        """Validate workflow step structure and dependencies."""
        errors = []
        
        if len(workflow) > self.max_workflow_steps:
            errors.append(f"Workflow exceeds maximum steps ({self.max_workflow_steps})")
        
        step_names = set()
        for step in workflow:
            if step.step in step_names:
                errors.append(f"Duplicate step name: {step.step}")
            step_names.add(step.step)
            
            # Validate dependencies
            for dep in step.depends_on:
                if dep not in step_names and dep not in [s.step for s in workflow[:workflow.index(step)]]:
                    errors.append(f"Step {step.step} depends on undefined step: {dep}")
        
        return errors
    
    async def _validate_agent_requirements(
        self,
        agent_requirements: List[AgentRequirement]
    ) -> Tuple[List[str], Dict[str, bool]]:
        """Validate agent availability and capabilities."""
        errors = []
        availability = {}
        
        if not self.agent_registry:
            return errors, availability
        
        try:
            # Get active agents
            active_agents = await self.agent_registry.get_active_agents()
            
            for req in agent_requirements:
                # Check if we have agents with required role
                matching_agents = [
                    agent for agent in active_agents
                    if agent.role == req.role.value
                ]
                
                availability[req.role.value] = len(matching_agents) > 0
                
                if not matching_agents:
                    errors.append(f"No active agents found for role: {req.role.value}")
                    continue
                
                # Check capabilities
                for capability in req.required_capabilities:
                    has_capability = any(
                        any(cap.get("name") == capability for cap in agent.capabilities)
                        for agent in matching_agents
                    )
                    if not has_capability:
                        errors.append(
                            f"No agents with capability '{capability}' for role {req.role.value}"
                        )
            
            return errors, availability
            
        except Exception as e:
            logger.error("Agent validation error", error=str(e))
            return [f"Agent validation failed: {str(e)}"], availability
    
    async def _validate_command_security(
        self,
        definition: CommandDefinition,
        validate_agents: bool = True
    ) -> List[str]:
        """Validate command definition for security risks."""
        errors = []
        
        if not self.security_validator:
            return errors
        
        try:
            # Create command context for security validation
            context = CommandContext(
                agent_id=uuid.uuid4(),  # Placeholder for validation
                command=f"Command: {definition.name}",
                agent_type="validator",
                trust_level=0.5  # Neutral trust for validation
            )
            
            # Validate each workflow step
            for step in definition.workflow:
                # Check step actions for dangerous commands
                step_commands = []
                if hasattr(step, 'action') and step.action:
                    step_commands.append(step.action)
                if hasattr(step, 'commands') and step.commands:
                    step_commands.extend(step.commands)
                
                for command in step_commands:
                    result = await self.security_validator.validate_command_advanced(
                        command, context
                    )
                    
                    if not result.is_safe:
                        errors.append(f"Dangerous command in step '{step.step}': {command}")
                        errors.extend([f"Risk: {factor}" for factor in result.risk_factors])
                    
                    if result.threat_categories:
                        threat_names = [cat.value for cat in result.threat_categories]
                        errors.append(f"Threat categories detected in '{step.step}': {', '.join(threat_names)}")
                    
                    if result.behavioral_anomalies:
                        errors.append(f"Behavioral anomalies in '{step.step}': {', '.join(result.behavioral_anomalies)}")
            
            # Check for command complexity and resource requirements
            total_steps = len(definition.workflow)
            if total_steps > 50:
                errors.append(f"Workflow has {total_steps} steps, which may pose management risks")
            
            # Check for dangerous agent role combinations
            admin_roles = [req for req in definition.agents if req.role.value in ['admin', 'system_administrator']]
            if len(admin_roles) > 1:
                errors.append("Multiple admin-level agents in single workflow may pose privilege escalation risks")
            
            # Validate timeout and resource constraints
            for step in definition.workflow:
                if hasattr(step, 'timeout_minutes') and step.timeout_minutes:
                    if step.timeout_minutes > 720:  # 12 hours
                        errors.append(f"Step '{step.step}' has excessive timeout ({step.timeout_minutes} minutes)")
                
                if hasattr(step, 'resources') and step.resources:
                    # Check for resource abuse patterns
                    if step.resources.get('cpu_limit', 0) > 8:
                        errors.append(f"Step '{step.step}' requests high CPU limit, potential resource abuse")
                    if step.resources.get('memory_limit', 0) > 16384:  # 16GB
                        errors.append(f"Step '{step.step}' requests high memory limit, potential resource abuse")
            
            return errors
            
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            return [f"Security validation failed: {str(e)}"]
    
    def _validate_performance_requirements(self, definition: CommandDefinition) -> List[str]:
        """Validate performance and resource requirements."""
        warnings = []
        
        # Check parallel task limits
        total_parallel_tasks = 0
        for step in definition.workflow:
            if step.parallel:
                total_parallel_tasks += len(step.parallel)
        
        if total_parallel_tasks > self.max_parallel_agents:
            warnings.append(
                f"High parallel task count ({total_parallel_tasks}) may impact performance"
            )
        
        # Check timeout values
        total_timeout = sum(step.timeout_minutes or 60 for step in definition.workflow)
        if total_timeout > 720:  # 12 hours
            warnings.append(f"Total workflow timeout ({total_timeout} minutes) is very high")
        
        return warnings
    
    def _estimate_execution_duration(self, definition: CommandDefinition) -> int:
        """Estimate command execution duration in minutes."""
        sequential_time = 0
        parallel_time = 0
        
        for step in definition.workflow:
            step_time = step.timeout_minutes or 60
            
            if step.parallel:
                # For parallel steps, take the maximum time
                max_parallel_time = max(
                    sub_step.timeout_minutes or 60 for sub_step in step.parallel
                )
                parallel_time = max(parallel_time, max_parallel_time)
            else:
                sequential_time += step_time
        
        return sequential_time + parallel_time
    
    def _generate_command_signature(self, definition: CommandDefinition) -> str:
        """Generate cryptographic signature for command integrity."""
        command_json = definition.model_dump_json(sort_keys=True)
        return hashlib.sha256(command_json.encode()).hexdigest()
    
    async def _store_command_in_db(
        self,
        definition: CommandDefinition,
        author_id: Optional[str],
        signature: str
    ) -> uuid.UUID:
        """Store command definition in database."""
        async with get_session() as db:
            command_id = uuid.uuid4()
            
            insert_query = """
                INSERT INTO command_registry (
                    id, name, version, definition, author_id, signature,
                    created_at, updated_at, execution_count, success_rate
                ) VALUES (
                    :id, :name, :version, :definition, :author_id, :signature,
                    :created_at, :updated_at, 0, 0.0
                )
            """
            
            await db.execute(insert_query, {
                "id": command_id,
                "name": definition.name,
                "version": definition.version,
                "definition": definition.model_dump(),
                "author_id": author_id,
                "signature": signature,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            
            await db.commit()
            return command_id
    
    async def _get_command_from_db(
        self,
        command_name: str,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get command from database."""
        async with get_session() as db:
            if version:
                query = """
                    SELECT * FROM command_registry 
                    WHERE name = :name AND version = :version
                """
                params = {"name": command_name, "version": version}
            else:
                query = """
                    SELECT * FROM command_registry 
                    WHERE name = :name 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """
                params = {"name": command_name}
            
            result = await db.execute(query, params)
            row = result.fetchone()
            
            if row:
                return {
                    "id": row[0],
                    "name": row[1],
                    "version": row[2],
                    "definition": row[3],
                    "author_id": row[4],
                    "signature": row[5],
                    "created_at": row[6],
                    "updated_at": row[7],
                    "execution_count": row[8],
                    "success_rate": row[9]
                }
            
            return None
    
    def _update_command_cache(self, definition: CommandDefinition) -> None:
        """Update in-memory command cache."""
        if definition.name not in self._command_cache:
            self._command_cache[definition.name] = {}
        
        self._command_cache[definition.name][definition.version] = definition
        self._cache_last_updated[definition.name] = datetime.utcnow()
    
    async def _store_command_file(self, definition: CommandDefinition) -> None:
        """Store command definition as YAML file."""
        try:
            command_dir = self.command_storage_path / definition.name
            command_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = command_dir / f"{definition.version}.yaml"
            
            with open(file_path, 'w') as f:
                # Convert to dict and write as YAML
                command_dict = definition.model_dump()
                yaml.dump(command_dict, f, default_flow_style=False, sort_keys=False)
            
            logger.debug(
                "Command file stored",
                command_name=definition.name,
                version=definition.version,
                file_path=str(file_path)
            )
            
        except Exception as e:
            logger.warning(
                "Failed to store command file",
                command_name=definition.name,
                error=str(e)
            )
    
    async def _delete_command_files(
        self,
        command_name: str,
        version: Optional[str] = None
    ) -> None:
        """Delete command definition files."""
        try:
            command_dir = self.command_storage_path / command_name
            
            if not command_dir.exists():
                return
            
            if version:
                file_path = command_dir / f"{version}.yaml"
                if file_path.exists():
                    file_path.unlink()
            else:
                # Delete entire command directory
                import shutil
                shutil.rmtree(command_dir)
            
        except Exception as e:
            logger.warning(
                "Failed to delete command files",
                command_name=command_name,
                error=str(e)
            )
    
    async def _update_command_metrics_in_db(
        self,
        command_name: str,
        metrics: CommandMetrics
    ) -> None:
        """Update command metrics in database."""
        try:
            async with get_session() as db:
                update_query = """
                    UPDATE command_registry 
                    SET execution_count = :execution_count,
                        success_rate = :success_rate,
                        updated_at = :updated_at
                    WHERE name = :name
                """
                
                await db.execute(update_query, {
                    "execution_count": metrics.total_executions,
                    "success_rate": metrics.success_rate,
                    "updated_at": datetime.utcnow(),
                    "name": command_name
                })
                
                await db.commit()
                
        except Exception as e:
            logger.warning(
                "Failed to update command metrics in database",
                command_name=command_name,
                error=str(e)
            )