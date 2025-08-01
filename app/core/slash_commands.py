"""
Claude Code-style Slash Commands for LeanVibe Agent Hive 2.0

Implements intuitive slash command interface for agent control and workflow management,
following Claude Code patterns while integrating with LeanVibe's orchestration system.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog
from pydantic import BaseModel, Field

from app.core.orchestrator import AgentOrchestrator
from app.core.communication import MessageBroker
from app.core.claude_code_hooks import get_claude_code_hooks_engine
from app.core.leanvibe_hooks_system import get_leanvibe_hooks_engine

logger = structlog.get_logger()


class SlashCommandArguments(BaseModel):
    """Arguments for slash command execution."""
    command: str = Field(..., description="Command name without /")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    raw_args: str = Field(default="", description="Raw argument string")


class SlashCommandResult(BaseModel):
    """Result of slash command execution."""
    success: bool = Field(..., description="Whether command succeeded")
    output: str = Field(default="", description="Command output")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time_ms: float = Field(..., description="Execution time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SlashCommandDefinition(BaseModel):
    """Definition of a slash command."""
    name: str = Field(..., description="Command name")
    description: str = Field(..., description="Command description")
    argument_hint: Optional[str] = Field(None, description="Argument hint for autocomplete")
    allowed_tools: List[str] = Field(default_factory=list, description="Tools command can use")
    file_path: Optional[Path] = Field(None, description="Path to command file")
    content: str = Field(default="", description="Command content/prompt")
    source: str = Field(default="built-in", description="Command source (built-in, project, user)")


class SlashCommandsEngine:
    """
    Claude Code-style slash commands engine for LeanVibe Agent Hive 2.0.
    
    Provides intuitive command interface for:
    - Agent management and control
    - Workflow creation and execution
    - System status and monitoring
    - Quality gates and validation
    - Development automation
    """
    
    def __init__(
        self,
        project_root: Optional[Path] = None,
        orchestrator: Optional[AgentOrchestrator] = None,
        communication_bus: Optional[MessageBroker] = None
    ):
        """
        Initialize slash commands engine.
        
        Args:
            project_root: Project root directory
            orchestrator: Agent orchestrator instance
            communication_bus: Communication bus for agent interaction
        """
        self.project_root = project_root or Path.cwd()
        self.orchestrator = orchestrator
        self.communication_bus = communication_bus
        
        # Command directories (Claude Code style)
        self.user_commands_dir = Path.home() / ".leanvibe" / "commands"
        self.project_commands_dir = self.project_root / ".leanvibe" / "commands"
        
        # Built-in commands
        self.built_in_commands = self._initialize_built_in_commands()
        
        # Loaded commands cache
        self.commands_cache: Dict[str, SlashCommandDefinition] = {}
        self.cache_timestamp = 0
        
        logger.info(
            "⚡ Slash Commands Engine initialized",
            project_root=str(self.project_root),
            user_commands=str(self.user_commands_dir),
            project_commands=str(self.project_commands_dir)
        )
    
    def _initialize_built_in_commands(self) -> Dict[str, SlashCommandDefinition]:
        """Initialize built-in slash commands."""
        return {
            "agents": SlashCommandDefinition(
                name="agents",
                description="List and manage available agents",
                content="Show all available agents, their status, and capabilities. Include active agents, their current tasks, and performance metrics."
            ),
            
            "workflow": SlashCommandDefinition(
                name="workflow",
                description="Create and manage workflows",
                argument_hint="[create|list|status|stop] [workflow_name]",
                content="Manage agent workflows. Create new workflows from specifications, list active workflows, check status, or stop running workflows."
            ),
            
            "status": SlashCommandDefinition(
                name="status",
                description="Show current system and agent status",
                content="Display comprehensive system status including agent health, resource utilization, performance metrics, and any active issues."
            ),
            
            "feature": SlashCommandDefinition(
                name="feature",
                description="Complete feature development from requirements to implementation",
                argument_hint="<description> [--scope=frontend|backend|fullstack] [--priority=low|medium|high]",
                allowed_tools=["Read", "Write", "Edit", "Bash", "GitHub", "Grep", "Glob"],
                content="""Execute sophisticated feature development workflow with AI-powered optimization.

Feature Description: $ARGUMENTS

Advanced Workflow Steps:
1. Requirements Analysis & Architecture Design
   - Parse feature requirements and identify technical specifications
   - Design optimal architecture patterns and data flow
   - Identify integration points and dependencies
   - Create implementation roadmap with priority ordering

2. Intelligent Code Generation & Implementation
   - Generate code scaffolding using best practices and patterns
   - Implement core business logic with error handling
   - Apply security best practices and input validation
   - Optimize for performance and maintainability

3. Comprehensive Testing Strategy
   - Create unit tests with high coverage (>90%)
   - Implement integration tests for all interfaces
   - Add performance tests for critical paths
   - Create end-to-end validation scenarios

4. Quality Assurance & Code Review
   - Run static analysis and security scanning
   - Perform automated code quality checks
   - Validate compliance with coding standards
   - Generate code review checklist

5. Documentation & Deployment
   - Create comprehensive API documentation
   - Update user guides and technical documentation
   - Prepare deployment scripts and configuration
   - Validate deployment readiness

This command uses AI-powered workflow optimization to adapt the development process based on project complexity, team size, available resources, and historical performance patterns."""
            ),

            "fix": SlashCommandDefinition(
                name="fix",
                description="Intelligent bug diagnosis and automated resolution",
                argument_hint="<issue_description> [--severity=low|medium|high|critical] [--component=<name>]",
                allowed_tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob"],
                content="""Execute advanced bug diagnosis and resolution workflow.

Issue Description: $ARGUMENTS

Intelligent Bug Resolution Process:
1. Issue Reproduction & Analysis
   - Reproduce the reported issue with detailed logging
   - Collect comprehensive diagnostic information
   - Analyze error patterns and failure modes
   - Identify affected components and systems

2. Root Cause Investigation
   - Perform deep code analysis using static and dynamic methods
   - Trace execution paths and data flow
   - Identify contributing factors and edge cases
   - Analyze recent changes and deployment history

3. Solution Design & Implementation
   - Design minimal, targeted fix with comprehensive testing
   - Implement solution with proper error handling
   - Add preventive measures and monitoring
   - Create validation tests for the specific issue

4. Verification & Quality Gates
   - Verify fix resolves the issue without side effects
   - Run comprehensive regression testing
   - Validate performance impact and resource usage
   - Test rollback procedures and recovery mechanisms

Advanced features: AI-powered pattern recognition, automatic test generation, and intelligent suggestion of related areas needing attention."""
            ),

            "deploy": SlashCommandDefinition(
                name="deploy",
                description="Enterprise-grade deployment with automated validation and rollback",
                argument_hint="[environment] [--strategy=blue-green|rolling|canary] [--validate]",
                allowed_tools=["Bash", "GitHub", "Docker", "Read"],
                content="""Execute enterprise-grade deployment workflow with comprehensive validation.

Deployment Target: $ARGUMENTS

Advanced Deployment Process:
1. Pre-Deployment Validation
   - Validate all quality gates and test results
   - Check deployment prerequisites and dependencies
   - Verify infrastructure readiness and capacity
   - Validate configuration and environment settings

2. Deployment Strategy Selection
   - Choose optimal deployment strategy based on risk and requirements
   - Configure blue-green, rolling, or canary deployment parameters
   - Set up monitoring and validation checkpoints
   - Prepare rollback mechanisms and procedures

3. Secure Deployment Execution
   - Execute deployment with comprehensive logging
   - Monitor deployment progress and system health
   - Validate each deployment step and checkpoint
   - Implement security scanning and compliance checks

4. Post-Deployment Validation
   - Run comprehensive smoke tests and health checks
   - Validate application functionality and performance
   - Check integration points and external dependencies
   - Monitor error rates and system metrics

Enterprise Features: Automated compliance checking, integration with enterprise monitoring systems, and advanced rollback strategies."""
            ),
            
            "test": SlashCommandDefinition(
                name="test",
                description="Comprehensive automated testing with intelligent coverage analysis",
                argument_hint="[scope] [--type=unit|integration|e2e] [--coverage] [--watch]",
                allowed_tools=["Bash", "Read", "Write", "Edit"],
                content="""Execute comprehensive testing workflow with AI-powered optimization.

Test Scope: $ARGUMENTS

Advanced Testing Strategy:
1. Test Strategy & Planning
   - Analyze codebase to identify testing requirements
   - Create comprehensive test plan with coverage targets
   - Identify critical paths and edge cases for testing
   - Plan test data generation and environment setup

2. Intelligent Test Generation
   - Generate unit tests with AI-powered analysis
   - Create integration tests for all component interfaces  
   - Develop end-to-end scenarios based on user journeys
   - Generate edge case and boundary condition tests

3. Test Automation & Execution
   - Set up automated test execution pipelines
   - Configure parallel test execution for efficiency
   - Implement smart test selection based on code changes
   - Create comprehensive test reporting and analytics

4. Coverage Analysis & Quality Gates
   - Analyze test coverage with detailed reporting (>90% target)
   - Identify untested code paths and missing scenarios
   - Validate test quality and effectiveness metrics
   - Implement quality gates for continuous integration

Advanced capabilities: AI-powered test case generation, intelligent test failure analysis, and automatic test data generation."""
            ),

            "optimize": SlashCommandDefinition(
                name="optimize",
                description="Comprehensive performance optimization and system enhancement",
                argument_hint="<target> [--type=performance|memory|cpu|network] [--scope=component|system]",
                allowed_tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob"],
                content="""Execute comprehensive performance optimization workflow.

Optimization Target: $ARGUMENTS

Advanced Performance Optimization Process:
1. Performance Baseline & Profiling
   - Establish comprehensive performance baselines
   - Profile application using advanced tools and techniques
   - Identify performance bottlenecks and resource constraints
   - Analyze user experience and system responsiveness

2. Bottleneck Analysis & Prioritization
   - Deep-dive analysis of identified performance issues
   - Quantify impact of each bottleneck on overall performance
   - Prioritize optimizations by ROI and implementation complexity
   - Create optimization roadmap with measurable targets

3. Code & Architecture Optimization
   - Optimize algorithms and data structures
   - Implement caching strategies and data access patterns
   - Refactor inefficient code paths and database queries
   - Apply design patterns for better resource utilization

4. Monitoring & Continuous Improvement
   - Implement comprehensive performance monitoring
   - Set up alerting for performance degradation
   - Create automated performance regression testing
   - Establish continuous optimization processes

Intelligent features: AI-powered identification of optimization opportunities and machine learning-based prediction of optimization impact."""
            ),
            
            "review": SlashCommandDefinition(
                name="review",
                description="AI-powered comprehensive code review and quality analysis",
                argument_hint="[--scope=recent|branch|full] [--depth=basic|comprehensive]",
                allowed_tools=["Read", "Grep", "Bash", "Edit"],
                content="""Execute comprehensive code review workflow with AI assistance.

Review Scope: $ARGUMENTS

Advanced Code Review Process:
1. Code Analysis & Static Review
   - Perform comprehensive static code analysis
   - Check coding standards and style compliance
   - Identify potential bugs and code smells
   - Analyze code complexity and maintainability metrics

2. Security & Vulnerability Assessment
   - Scan for security vulnerabilities and weaknesses
   - Validate input sanitization and access controls
   - Check for exposed secrets and configuration issues
   - Analyze dependency security and license compliance

3. Architecture & Design Review
   - Evaluate architectural patterns and design decisions
   - Check adherence to SOLID principles and best practices
   - Analyze component coupling and cohesion
   - Review API design and interface contracts

4. Performance & Efficiency Analysis
   - Identify performance bottlenecks and optimization opportunities
   - Analyze algorithmic complexity and resource usage
   - Review database queries and data access patterns
   - Check for memory leaks and resource management issues

5. Test Coverage & Quality Assessment
   - Analyze test coverage and identify gaps
   - Review test quality and effectiveness
   - Check for proper error handling and edge cases
   - Validate integration and end-to-end test scenarios

AI-Enhanced Features: Machine learning-powered bug detection, intelligent suggestion of improvements, and automated generation of review summaries."""
            ),
            
            "security": SlashCommandDefinition(
                name="security",
                description="Advanced security audit and vulnerability assessment",
                argument_hint="[--scope=code|dependencies|config] [--depth=basic|comprehensive]",
                allowed_tools=["Read", "Bash", "Grep", "Edit"],
                content="""Execute advanced security audit and vulnerability assessment.

Security Scope: $ARGUMENTS

Comprehensive Security Analysis:
1. Static Security Analysis
   - Perform comprehensive static code analysis for vulnerabilities
   - Check for common security anti-patterns and weaknesses
   - Analyze input validation and sanitization mechanisms
   - Review authentication and authorization implementations

2. Dependency Security Assessment
   - Scan all dependencies for known security vulnerabilities
   - Analyze transitive dependencies and supply chain risks
   - Check for outdated packages with security patches
   - Validate license compliance and security policies

3. Configuration Security Review
   - Review security configurations and access controls
   - Analyze secrets management and credential handling
   - Check for exposed sensitive information and debug data
   - Validate encryption and data protection mechanisms

4. Infrastructure Security Analysis
   - Review deployment configurations and security settings
   - Analyze network security and access controls
   - Check for security hardening and best practices
   - Validate monitoring and incident response capabilities

Advanced Security Features: AI-powered vulnerability detection, intelligent threat modeling, and automated security remediation suggestions."""
            ),
            
            "docs": SlashCommandDefinition(
                name="docs",
                description="Intelligent documentation generation and maintenance",
                argument_hint="[scope] [--format=markdown|rst] [--audience=developer|user|api]",
                allowed_tools=["Read", "Write", "Grep", "Edit"],
                content="""Execute intelligent documentation generation and maintenance workflow.

Documentation Scope: $ARGUMENTS

Advanced Documentation Process:
1. Documentation Strategy & Planning
   - Analyze codebase and identify documentation requirements
   - Assess current documentation quality and coverage gaps
   - Create comprehensive documentation plan with priorities
   - Define target audiences and documentation formats

2. Automated API Documentation
   - Generate comprehensive API documentation from code annotations  
   - Create interactive API reference with examples and usage patterns
   - Generate SDK documentation and integration guides
   - Validate API documentation completeness and accuracy

3. User Documentation & Guides
   - Create user-friendly guides and tutorials
   - Generate getting started and onboarding documentation
   - Develop troubleshooting guides and FAQ sections
   - Create video tutorials and interactive documentation

4. Technical Documentation
   - Generate architecture documentation and system diagrams
   - Create deployment guides and operational runbooks
   - Document configuration options and environment setup
   - Generate performance tuning and optimization guides

5. Documentation Maintenance & Quality
   - Implement automated documentation testing and validation
   - Set up documentation versioning and release management
   - Create documentation review and approval workflows
   - Establish metrics for documentation effectiveness

AI-Powered Features: Intelligent content generation, automated code-to-documentation synchronization, and documentation quality scoring."""
            ),
            
            "metrics": SlashCommandDefinition(
                name="metrics",
                description="Show performance and quality metrics",
                content="""Display comprehensive performance and quality metrics.

Metrics include:
- Agent performance and utilization
- Workflow execution statistics
- Code quality trends
- Test coverage metrics
- System resource usage
- Error rates and reliability"""
            ),
            
            "hooks": SlashCommandDefinition(
                name="hooks",
                description="Manage and configure hooks",
                argument_hint="[list|enable|disable|test] [hook_name]",
                content="""Manage Claude Code-style hooks configuration.

Hook management: $ARGUMENTS

Available operations:
- list: Show all configured hooks
- enable: Enable specific hook
- disable: Disable specific hook  
- test: Test hook execution
- configure: Interactive hook configuration"""
            ),
            
            "config": SlashCommandDefinition(
                name="config",
                description="View and modify system configuration",
                argument_hint="[get|set|list] [key] [value]",
                content="""View and modify LeanVibe system configuration.

Configuration operation: $ARGUMENTS

Operations:
- list: Show all configuration options
- get <key>: Get specific configuration value
- set <key> <value>: Set configuration value
- reset: Reset to default configuration"""
            ),
            
            "compact": SlashCommandDefinition(
                name="compact",
                description="Compact conversation context",
                argument_hint="[instructions]",
                content="""Compact conversation context to optimize memory usage.

Compaction instructions: $ARGUMENTS

Steps:
1. Analyze current context usage
2. Identify redundant or outdated information
3. Compress context while preserving important details
4. Update agent memory with compacted context
5. Report memory savings achieved"""
            )
        }
    
    async def load_commands(self, force_reload: bool = False) -> None:
        """Load all available commands from various sources."""
        current_time = datetime.now().timestamp()
        
        # Check if cache is still valid (5 minute cache)
        if not force_reload and (current_time - self.cache_timestamp) < 300:
            return
        
        self.commands_cache = {}
        
        try:
            # Load built-in commands
            for name, cmd in self.built_in_commands.items():
                self.commands_cache[name] = cmd
            
            # Load user commands
            if self.user_commands_dir.exists():
                user_commands = await self._load_commands_from_directory(
                    self.user_commands_dir, "user"
                )
                self.commands_cache.update(user_commands)
            
            # Load project commands (override user commands)
            if self.project_commands_dir.exists():
                project_commands = await self._load_commands_from_directory(
                    self.project_commands_dir, "project"
                )
                self.commands_cache.update(project_commands)
            
            self.cache_timestamp = current_time
            
            logger.info(
                "📚 Commands loaded",
                total_commands=len(self.commands_cache),
                built_in=len(self.built_in_commands),
                user=len([c for c in self.commands_cache.values() if c.source == "user"]),
                project=len([c for c in self.commands_cache.values() if c.source == "project"])
            )
            
        except Exception as e:
            logger.error(
                "❌ Failed to load commands",
                error=str(e),
                exc_info=True
            )
    
    async def _load_commands_from_directory(
        self, 
        directory: Path, 
        source: str
    ) -> Dict[str, SlashCommandDefinition]:
        """Load commands from a directory."""
        commands = {}
        
        try:
            for file_path in directory.rglob("*.md"):
                try:
                    # Parse command file
                    command = await self._parse_command_file(file_path, source)
                    if command:
                        commands[command.name] = command
                        
                except Exception as e:
                    logger.warning(
                        "⚠️ Failed to parse command file",
                        file_path=str(file_path),
                        error=str(e)
                    )
        
        except Exception as e:
            logger.error(
                "❌ Failed to load commands from directory",
                directory=str(directory),
                error=str(e)
            )
        
        return commands
    
    async def _parse_command_file(
        self, 
        file_path: Path, 
        source: str
    ) -> Optional[SlashCommandDefinition]:
        """Parse a command file (Markdown with YAML frontmatter)."""
        try:
            content = file_path.read_text()
            
            # Parse YAML frontmatter
            frontmatter = {}
            if content.startswith("---\n"):
                parts = content.split("---\n", 2)
                if len(parts) >= 3:
                    import yaml
                    frontmatter = yaml.safe_load(parts[1]) or {}
                    content = parts[2]
            
            # Determine command name from file path
            relative_path = file_path.relative_to(file_path.parent.parent)
            name_parts = relative_path.with_suffix("").parts
            
            if len(name_parts) > 1:
                # Namespaced command (e.g., frontend/component.md -> frontend:component)
                command_name = ":".join(name_parts)
            else:
                command_name = name_parts[0]
            
            return SlashCommandDefinition(
                name=command_name,
                description=frontmatter.get("description", f"Custom {source} command"),
                argument_hint=frontmatter.get("argument-hint"),
                allowed_tools=frontmatter.get("allowed-tools", []),
                file_path=file_path,
                content=content.strip(),
                source=source
            )
            
        except Exception as e:
            logger.error(
                "❌ Failed to parse command file",
                file_path=str(file_path),
                error=str(e)
            )
            return None
    
    async def execute_command(
        self,
        command_str: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> SlashCommandResult:
        """
        Execute a slash command.
        
        Args:
            command_str: Command string (e.g., "/agents list" or "/deploy production")
            agent_id: Agent executing the command
            session_id: Current session ID
            
        Returns:
            SlashCommandResult with execution details
        """
        start_time = datetime.now()
        
        try:
            # Parse command
            args = self._parse_command_arguments(command_str)
            
            # Load commands if needed
            await self.load_commands()
            
            # Find command definition
            command_def = self.commands_cache.get(args.command)
            if not command_def:
                return SlashCommandResult(
                    success=False,
                    error=f"Unknown command: /{args.command}",
                    execution_time_ms=0
                )
            
            # Execute command based on type
            if command_def.source == "built-in":
                result = await self._execute_built_in_command(args, command_def, agent_id, session_id)
            else:
                result = await self._execute_custom_command(args, command_def, agent_id, session_id)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            
            # Log execution
            logger.info(
                f"⚡ Command executed: /{args.command}",
                command=args.command,
                args=args.raw_args,
                success=result.success,
                execution_time_ms=execution_time
            )
            
            # Execute hooks if available (both Claude Code and LeanVibe hooks)
            hooks_engine = get_claude_code_hooks_engine()
            if hooks_engine:
                await hooks_engine.execute_workflow_hooks(
                    event="SlashCommand",
                    workflow_id=f"command_{args.command}",
                    workflow_data={
                        "command": args.command,
                        "args": args.args,
                        "success": result.success,
                        "output": result.output
                    },
                    agent_id=agent_id or "system",
                    session_id=session_id or str(uuid.uuid4())
                )
            
            # Execute LeanVibe hooks for enhanced automation
            leanvibe_hooks = get_leanvibe_hooks_engine()
            if leanvibe_hooks:
                from app.core.leanvibe_hooks_system import HookEventType
                await leanvibe_hooks.execute_workflow_hooks(
                    event=HookEventType.POST_AGENT_TASK,
                    workflow_id=f"slash_command_{args.command}",
                    workflow_data={
                        "command": args.command,
                        "arguments": args.args,
                        "success": result.success,
                        "output": result.output,
                        "task_type": "slash_command",
                        "agent_name": "slash_commands_engine"
                    },
                    agent_id=agent_id or "system",
                    session_id=session_id or str(uuid.uuid4())
                )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.error(
                "❌ Command execution failed",
                command=command_str,
                error=str(e),
                exc_info=True
            )
            
            return SlashCommandResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    def _parse_command_arguments(self, command_str: str) -> SlashCommandArguments:
        """Parse command string into arguments."""
        # Remove leading slash
        if command_str.startswith("/"):
            command_str = command_str[1:]
        
        # Split command and arguments
        parts = command_str.split(None, 1)
        command = parts[0] if parts else ""
        raw_args = parts[1] if len(parts) > 1 else ""
        
        # Parse arguments
        args = raw_args.split() if raw_args else []
        
        return SlashCommandArguments(
            command=command,
            args=args,
            raw_args=raw_args
        )
    
    async def _execute_built_in_command(
        self,
        args: SlashCommandArguments,
        command_def: SlashCommandDefinition,
        agent_id: Optional[str],
        session_id: Optional[str]
    ) -> SlashCommandResult:
        """Execute a built-in command."""
        try:
            if args.command == "agents":
                return await self._cmd_agents(args)
            elif args.command == "workflow":
                return await self._cmd_workflow(args)
            elif args.command == "status":
                return await self._cmd_status(args)
            elif args.command == "metrics":
                return await self._cmd_metrics(args)
            elif args.command == "hooks":
                return await self._cmd_hooks(args)
            elif args.command == "config":
                return await self._cmd_config(args)
            elif args.command == "compact":
                return await self._cmd_compact(args)
            else:
                # For commands that need agent execution (deploy, test, review, etc.)
                return await self._execute_agent_command(args, command_def, agent_id, session_id)
        
        except Exception as e:
            return SlashCommandResult(
                success=False,
                error=str(e),
                execution_time_ms=0
            )
    
    async def _execute_custom_command(
        self,
        args: SlashCommandArguments,
        command_def: SlashCommandDefinition,
        agent_id: Optional[str],
        session_id: Optional[str]
    ) -> SlashCommandResult:
        """Execute a custom command by delegating to an agent."""
        return await self._execute_agent_command(args, command_def, agent_id, session_id)
    
    async def _execute_agent_command(
        self,
        args: SlashCommandArguments,
        command_def: SlashCommandDefinition,
        agent_id: Optional[str],
        session_id: Optional[str]
    ) -> SlashCommandResult:
        """Execute command by delegating to an agent."""
        if not self.orchestrator:
            return SlashCommandResult(
                success=False,
                error="No orchestrator available for agent delegation",
                execution_time_ms=0
            )
        
        try:
            # Prepare command prompt
            prompt = command_def.content
            
            # Replace $ARGUMENTS placeholder
            if "$ARGUMENTS" in prompt:
                prompt = prompt.replace("$ARGUMENTS", args.raw_args)
            
            # Create task for agent execution
            task_data = {
                "type": "slash_command",
                "command": args.command,
                "prompt": prompt,
                "arguments": args.args,
                "allowed_tools": command_def.allowed_tools,
                "session_id": session_id,
                "correlation_id": f"cmd_{args.command}_{uuid.uuid4().hex[:8]}"
            }
            
            # Execute via orchestrator
            if agent_id:
                # Use specific agent
                result = await self.orchestrator.execute_agent_task(agent_id, task_data)
            else:
                # Auto-assign to best agent
                result = await self.orchestrator.assign_and_execute_task(task_data)
            
            return SlashCommandResult(
                success=result.get("success", True),
                output=result.get("output", "Command executed successfully"),
                metadata={
                    "agent_id": result.get("agent_id"),
                    "task_id": result.get("task_id"),
                    "execution_details": result
                }
            )
            
        except Exception as e:
            return SlashCommandResult(
                success=False,
                error=f"Agent execution failed: {str(e)}",
                execution_time_ms=0
            )
    
    async def _cmd_agents(self, args: SlashCommandArguments) -> SlashCommandResult:
        """Handle /agents command."""
        if not self.orchestrator:
            return SlashCommandResult(
                success=False,
                error="No orchestrator available",
                execution_time_ms=0
            )
        
        try:
            if not args.args or args.args[0] == "list":
                # List all agents
                agents = await self.orchestrator.get_all_agents()
                
                output = "🤖 **Available Agents**\\n\\n"
                for agent in agents:
                    status_emoji = "🟢" if agent.get("status") == "active" else "🔴"
                    output += f"{status_emoji} **{agent.get('name', 'Unknown')}** ({agent.get('role', 'Unknown')})\\n"
                    output += f"   Capabilities: {', '.join(agent.get('capabilities', []))}\\n"
                    output += f"   Tasks: {agent.get('active_tasks', 0)} active\\n\\n"
                
                return SlashCommandResult(
                    success=True,
                    output=output,
                    metadata={"agents_count": len(agents)}
                )
            
            elif args.args[0] == "create":
                # Create new agent
                if len(args.args) < 2:
                    return SlashCommandResult(
                        success=False,
                        error="Usage: /agents create <agent_name> [role] [capabilities...]"
                    )
                
                agent_name = args.args[1]
                role = args.args[2] if len(args.args) > 2 else "general"
                capabilities = args.args[3:] if len(args.args) > 3 else []
                
                agent_data = {
                    "name": agent_name,
                    "role": role,
                    "capabilities": capabilities
                }
                
                result = await self.orchestrator.create_agent(agent_data)
                
                return SlashCommandResult(
                    success=True,
                    output=f"✅ Agent '{agent_name}' created successfully",
                    metadata={"agent_id": result.get("agent_id")}
                )
            
            else:
                return SlashCommandResult(
                    success=False,
                    error=f"Unknown agents subcommand: {args.args[0]}"
                )
        
        except Exception as e:
            return SlashCommandResult(
                success=False,
                error=str(e)
            )
    
    async def _cmd_workflow(self, args: SlashCommandArguments) -> SlashCommandResult:
        """Handle /workflow command."""
        if not self.orchestrator:
            return SlashCommandResult(
                success=False,
                error="No orchestrator available"
            )
        
        try:
            if not args.args or args.args[0] == "list":
                # List active workflows
                workflows = await self.orchestrator.get_active_workflows()
                
                output = "🔄 **Active Workflows**\\n\\n"
                if not workflows:
                    output += "No active workflows"
                else:
                    for workflow in workflows:
                        status_emoji = "🟢" if workflow.get("status") == "running" else "⏸️"
                        output += f"{status_emoji} **{workflow.get('name', 'Unknown')}**\\n"
                        output += f"   Status: {workflow.get('status', 'Unknown')}\\n"
                        output += f"   Progress: {workflow.get('progress', 0)}%\\n"
                        output += f"   Agents: {len(workflow.get('agents', []))}\\n\\n"
                
                return SlashCommandResult(
                    success=True,
                    output=output,
                    metadata={"workflows_count": len(workflows)}
                )
            
            elif args.args[0] == "create":
                return SlashCommandResult(
                    success=False,
                    error="Workflow creation requires specification. Use specification ingestion system."
                )
            
            elif args.args[0] == "status":
                if len(args.args) < 2:
                    return SlashCommandResult(
                        success=False,
                        error="Usage: /workflow status <workflow_id>"
                    )
                
                workflow_id = args.args[1]
                status = await self.orchestrator.get_workflow_status(workflow_id)
                
                return SlashCommandResult(
                    success=True,
                    output=f"Workflow {workflow_id} status: {status}",
                    metadata={"workflow_status": status}
                )
            
            else:
                return SlashCommandResult(
                    success=False,
                    error=f"Unknown workflow subcommand: {args.args[0]}"
                )
        
        except Exception as e:
            return SlashCommandResult(
                success=False,
                error=str(e)
            )
    
    async def _cmd_status(self, args: SlashCommandArguments) -> SlashCommandResult:
        """Handle /status command."""
        try:
            # Gather system status
            status_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "system": "healthy",
                "components": {}
            }
            
            # Orchestrator status
            if self.orchestrator:
                orchestrator_status = await self.orchestrator.get_health_status()
                status_data["components"]["orchestrator"] = orchestrator_status
            
            # Communication bus status
            if self.communication_bus:
                comm_status = await self.communication_bus.get_health_status()
                status_data["components"]["communication"] = comm_status
            
            # Hooks engine status
            hooks_engine = get_claude_code_hooks_engine()
            if hooks_engine:
                hooks_stats = await hooks_engine.get_performance_stats()
                status_data["components"]["hooks"] = hooks_stats
            
            # Generate output
            output = "📊 **System Status**\\n\\n"
            
            overall_status = "🟢 Healthy"
            if any(comp.get("status") != "healthy" for comp in status_data["components"].values()):
                overall_status = "🟡 Degraded"
            
            output += f"**Overall Status:** {overall_status}\\n\\n"
            
            for component, comp_status in status_data["components"].items():
                status_emoji = "🟢" if comp_status.get("status") == "healthy" else "🔴"
                output += f"{status_emoji} **{component.title()}**: {comp_status.get('status', 'unknown')}\\n"
            
            return SlashCommandResult(
                success=True,
                output=output,
                metadata=status_data
            )
        
        except Exception as e:
            return SlashCommandResult(
                success=False,
                error=str(e)
            )
    
    async def _cmd_metrics(self, args: SlashCommandArguments) -> SlashCommandResult:
        """Handle /metrics command."""
        try:
            # Gather performance metrics
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "system_metrics": {},
                "agent_metrics": {},
                "workflow_metrics": {}
            }
            
            if self.orchestrator:
                agent_metrics = await self.orchestrator.get_performance_metrics()
                metrics["agent_metrics"] = agent_metrics
            
            # Generate output
            output = "📈 **Performance Metrics**\\n\\n"
            
            if metrics["agent_metrics"]:
                output += "**Agent Performance:**\\n"
                agent_data = metrics["agent_metrics"]
                output += f"- Active Agents: {agent_data.get('active_agents', 0)}\\n"
                output += f"- Tasks Completed: {agent_data.get('tasks_completed', 0)}\\n"
                output += f"- Average Response Time: {agent_data.get('avg_response_time_ms', 0):.1f}ms\\n"
                output += f"- Success Rate: {agent_data.get('success_rate', 0):.1f}%\\n\\n"
            
            return SlashCommandResult(
                success=True,
                output=output,
                metadata=metrics
            )
        
        except Exception as e:
            return SlashCommandResult(
                success=False,
                error=str(e)
            )
    
    async def _cmd_hooks(self, args: SlashCommandArguments) -> SlashCommandResult:
        """Handle /hooks command."""
        hooks_engine = get_claude_code_hooks_engine()
        if not hooks_engine:
            return SlashCommandResult(
                success=False,
                error="Hooks engine not available"
            )
        
        try:
            if not args.args or args.args[0] == "list":
                # List configured hooks
                stats = await hooks_engine.get_performance_stats()
                
                output = "🎣 **Hooks Configuration**\\n\\n"
                output += f"**Hooks Directory:** {stats['project_hooks_dir']}\\n"
                output += f"**Directory Exists:** {'✅' if stats['project_hooks_dir_exists'] else '❌'}\\n"
                output += f"**Config Events:** {stats['config_events']}\\n\\n"
                
                output += "**Execution Statistics:**\\n"
                exec_stats = stats['execution_stats']
                output += f"- Hooks Executed: {exec_stats['hooks_executed']}\\n"
                output += f"- Hooks Failed: {exec_stats['hooks_failed']}\\n"
                output += f"- Actions Blocked: {exec_stats['blocked_actions']}\\n"
                output += f"- Total Execution Time: {exec_stats['total_execution_time_ms']:.1f}ms\\n"
                
                return SlashCommandResult(
                    success=True,
                    output=output,
                    metadata=stats
                )
            
            else:
                return SlashCommandResult(
                    success=False,
                    error=f"Hooks subcommand '{args.args[0]}' not implemented yet"
                )
        
        except Exception as e:
            return SlashCommandResult(
                success=False,
                error=str(e)
            )
    
    async def _cmd_config(self, args: SlashCommandArguments) -> SlashCommandResult:
        """Handle /config command."""
        try:
            if not args.args or args.args[0] == "list":
                # List configuration
                config_data = {
                    "project_root": str(self.project_root),
                    "commands_loaded": len(self.commands_cache),
                    "user_commands_dir": str(self.user_commands_dir),
                    "project_commands_dir": str(self.project_commands_dir),
                    "orchestrator_available": self.orchestrator is not None,
                    "communication_bus_available": self.communication_bus is not None
                }
                
                output = "⚙️ **System Configuration**\\n\\n"
                for key, value in config_data.items():
                    output += f"**{key.replace('_', ' ').title()}:** {value}\\n"
                
                return SlashCommandResult(
                    success=True,
                    output=output,
                    metadata=config_data
                )
            
            else:
                return SlashCommandResult(
                    success=False,
                    error="Configuration modification not implemented yet"
                )
        
        except Exception as e:
            return SlashCommandResult(
                success=False,
                error=str(e)
            )
    
    async def _cmd_compact(self, args: SlashCommandArguments) -> SlashCommandResult:
        """Handle /compact command."""
        try:
            instructions = args.raw_args or "Compact conversation context to optimize memory usage"
            
            # This would integrate with LeanVibe's context compression system
            output = f"🗜️ **Context Compaction**\\n\\nCompacting context with instructions: {instructions}\\n\\n"
            output += "⚠️ Context compaction integration with LeanVibe's context engine pending implementation."
            
            return SlashCommandResult(
                success=True,
                output=output,
                metadata={"instructions": instructions}
            )
        
        except Exception as e:
            return SlashCommandResult(
                success=False,
                error=str(e)
            )
    
    async def get_available_commands(self) -> List[SlashCommandDefinition]:
        """Get list of all available commands."""
        await self.load_commands()
        return list(self.commands_cache.values())
    
    async def get_command_suggestions(self, partial_command: str) -> List[str]:
        """Get command suggestions for autocomplete."""
        await self.load_commands()
        
        if partial_command.startswith("/"):
            partial_command = partial_command[1:]
        
        suggestions = []
        for name in self.commands_cache.keys():
            if name.startswith(partial_command):
                suggestions.append(f"/{name}")
        
        return sorted(suggestions)
    
    async def create_project_commands_directory(self) -> None:
        """Create project commands directory with example commands."""
        commands_dir = self.project_commands_dir
        commands_dir.mkdir(parents=True, exist_ok=True)
        
        # Create example commands
        examples = {
            "quick-test.md": '''---
description: Quick test runner for current changes
argument-hint: [scope]
allowed-tools: Bash, Read
---

Run tests for recently modified files.

Test scope: $ARGUMENTS

Steps:
1. Identify recently modified files
2. Find corresponding test files
3. Run relevant tests
4. Report results
''',
            
            "format-code.md": '''---
description: Format code using project standards
allowed-tools: Bash, Edit
---

Format all code in the project using configured formatters.

Steps:
1. Detect file types needing formatting
2. Apply appropriate formatters (black, prettier, etc.)
3. Report formatting changes
''',
            
            "check-deps.md": '''---
description: Check for dependency updates
allowed-tools: Bash, Read
---

Check for outdated dependencies and security vulnerabilities.

Steps:
1. Check for outdated packages
2. Scan for security vulnerabilities
3. Generate update recommendations
4. Check for breaking changes
'''
        }
        
        for filename, content in examples.items():
            command_file = commands_dir / filename
            if not command_file.exists():
                command_file.write_text(content)
        
        logger.info(
            "📁 Project commands directory created",
            commands_dir=str(commands_dir),
            examples_created=len(examples)
        )


# Global slash commands engine instance
_slash_commands_engine: Optional[SlashCommandsEngine] = None


def get_slash_commands_engine() -> Optional[SlashCommandsEngine]:
    """Get the global slash commands engine instance."""
    return _slash_commands_engine


def set_slash_commands_engine(engine: SlashCommandsEngine) -> None:
    """Set the global slash commands engine instance."""
    global _slash_commands_engine
    _slash_commands_engine = engine
    logger.info("🔗 Global slash commands engine set")


async def initialize_slash_commands_engine(
    project_root: Optional[Path] = None,
    orchestrator: Optional[AgentOrchestrator] = None,
    communication_bus: Optional[MessageBroker] = None
) -> SlashCommandsEngine:
    """
    Initialize and set the global slash commands engine.
    
    Args:
        project_root: Project root directory
        orchestrator: Agent orchestrator instance
        communication_bus: Communication bus instance
        
    Returns:
        SlashCommandsEngine instance
    """
    engine = SlashCommandsEngine(
        project_root=project_root,
        orchestrator=orchestrator,
        communication_bus=communication_bus
    )
    
    # Load commands
    await engine.load_commands()
    
    # Create project commands directory
    await engine.create_project_commands_directory()
    
    set_slash_commands_engine(engine)
    
    logger.info("✅ Slash commands engine initialized")
    return engine