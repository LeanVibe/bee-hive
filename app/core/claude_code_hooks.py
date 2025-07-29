"""
Claude Code Hooks Integration for LeanVibe Agent Hive 2.0

Implements Claude Code-style hooks system with deterministic automation
for workflow events, agent actions, and quality gates.

This integrates Claude Code hook patterns with LeanVibe's existing
hook_processor.py and observability infrastructure.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog
from pydantic import BaseModel, Field

from app.core.hook_processor import HookEventProcessor, get_hook_event_processor
from app.models.observability import EventType

logger = structlog.get_logger()


class HookMatcher(BaseModel):
    """Claude Code-style hook matcher for tool patterns."""
    pattern: str = Field(..., description="Pattern to match tools/events")
    hooks: List["HookCommand"] = Field(default_factory=list, description="Commands to execute")


class HookCommand(BaseModel):
    """Claude Code-style hook command definition."""
    type: str = Field(default="command", description="Hook type (currently only 'command')")
    command: str = Field(..., description="Shell command to execute")
    timeout: Optional[int] = Field(default=60, description="Timeout in seconds")
    description: Optional[str] = Field(None, description="Human-readable description")


class HookResult(BaseModel):
    """Result of hook execution."""
    success: bool = Field(..., description="Whether hook executed successfully")
    exit_code: int = Field(..., description="Command exit code")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    blocked: bool = Field(default=False, description="Whether hook blocked the action")
    continue_execution: bool = Field(default=True, description="Whether to continue workflow")


class ClaudeCodeHooksConfig(BaseModel):
    """Claude Code hooks configuration."""
    PreToolUse: List[HookMatcher] = Field(default_factory=list)
    PostToolUse: List[HookMatcher] = Field(default_factory=list)
    AgentStart: List[HookMatcher] = Field(default_factory=list)
    AgentStop: List[HookMatcher] = Field(default_factory=list)
    WorkflowStart: List[HookMatcher] = Field(default_factory=list)
    WorkflowComplete: List[HookMatcher] = Field(default_factory=list)
    QualityGate: List[HookMatcher] = Field(default_factory=list)
    Notification: List[HookMatcher] = Field(default_factory=list)


class ClaudeCodeHooksEngine:
    """
    Claude Code-style hooks engine for LeanVibe Agent Hive 2.0.
    
    Provides deterministic automation for:
    - Quality gates and validation
    - Code formatting and standards
    - Security checks and compliance  
    - Notifications and alerts
    - Workflow automation
    """
    
    def __init__(
        self,
        project_root: Optional[Path] = None,
        hook_event_processor: Optional[HookEventProcessor] = None
    ):
        """
        Initialize Claude Code hooks engine.
        
        Args:
            project_root: Project root directory for .leanvibe/hooks/
            hook_event_processor: Existing LeanVibe hook event processor
        """
        self.project_root = project_root or Path.cwd()
        self.hook_event_processor = hook_event_processor or get_hook_event_processor()
        
        # Hook configuration paths (Claude Code style)
        self.user_config_path = Path.home() / ".leanvibe" / "settings.json"
        self.project_config_path = self.project_root / ".leanvibe" / "settings.json"
        self.project_hooks_dir = self.project_root / ".leanvibe" / "hooks"
        
        # Loaded configuration
        self.config: ClaudeCodeHooksConfig = ClaudeCodeHooksConfig()
        
        # Performance tracking
        self.execution_stats = {
            "hooks_executed": 0,
            "hooks_failed": 0,
            "total_execution_time_ms": 0,
            "blocked_actions": 0
        }
        
        logger.info(
            "🎣 Claude Code Hooks Engine initialized",
            project_root=str(self.project_root),
            user_config=str(self.user_config_path),
            project_config=str(self.project_config_path)
        )
    
    async def load_configuration(self) -> None:
        """Load hooks configuration from user and project settings."""
        try:
            # Load user configuration
            user_hooks = {}
            if self.user_config_path.exists():
                with open(self.user_config_path) as f:
                    user_config = json.load(f)
                    user_hooks = user_config.get("hooks", {})
            
            # Load project configuration (overrides user)
            project_hooks = {}
            if self.project_config_path.exists():
                with open(self.project_config_path) as f:
                    project_config = json.load(f)
                    project_hooks = project_config.get("hooks", {})
            
            # Merge configurations (project takes precedence)
            merged_hooks = {**user_hooks, **project_hooks}
            
            # Parse into ClaudeCodeHooksConfig
            self.config = ClaudeCodeHooksConfig(**merged_hooks)
            
            logger.info(
                "📋 Hooks configuration loaded",
                user_hooks_events=list(user_hooks.keys()),
                project_hooks_events=list(project_hooks.keys()),
                total_events=len(merged_hooks)
            )
            
        except Exception as e:
            logger.error(
                "❌ Failed to load hooks configuration",
                error=str(e),
                exc_info=True
            )
            # Use empty configuration on error
            self.config = ClaudeCodeHooksConfig()
    
    async def execute_pre_tool_use_hooks(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        agent_id: str,
        session_id: str,
        correlation_id: Optional[str] = None
    ) -> HookResult:
        """
        Execute PreToolUse hooks with blocking capability.
        
        Args:
            tool_name: Name of the tool being executed
            tool_input: Tool input parameters
            agent_id: Agent executing the tool
            session_id: Current session ID
            correlation_id: Correlation ID for tracking
            
        Returns:
            HookResult indicating whether to proceed or block
        """
        start_time = time.time()
        
        try:
            # Find matching hooks
            matching_hooks = self._find_matching_hooks(self.config.PreToolUse, tool_name)
            
            if not matching_hooks:
                return HookResult(
                    success=True,
                    exit_code=0,
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Prepare hook input data (Claude Code format)
            hook_input = {
                "session_id": session_id,
                "agent_id": agent_id,
                "hook_event_name": "PreToolUse",
                "tool_name": tool_name,
                "tool_input": tool_input,
                "correlation_id": correlation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "cwd": str(self.project_root),
                "project_dir": str(self.project_root)
            }
            
            # Execute hooks in parallel
            hook_results = await self._execute_hooks_parallel(matching_hooks, hook_input)
            
            # Analyze results for blocking
            overall_result = self._analyze_hook_results(hook_results, "PreToolUse")
            
            # Log execution to LeanVibe observability
            if self.hook_event_processor:
                await self.hook_event_processor.process_pre_tool_use({
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "tool_name": tool_name,
                    "parameters": tool_input,
                    "hook_results": [r.dict() for r in hook_results],
                    "blocked": overall_result.blocked,
                    "correlation_id": correlation_id
                })
            
            # Update stats
            self.execution_stats["hooks_executed"] += len(hook_results)
            self.execution_stats["total_execution_time_ms"] += overall_result.execution_time_ms
            if overall_result.blocked:
                self.execution_stats["blocked_actions"] += 1
            
            logger.info(
                "🎣 PreToolUse hooks executed",
                tool_name=tool_name,
                hooks_count=len(hook_results),
                blocked=overall_result.blocked,
                execution_time_ms=overall_result.execution_time_ms
            )
            
            return overall_result
            
        except Exception as e:
            logger.error(
                "❌ Failed to execute PreToolUse hooks",
                tool_name=tool_name,
                error=str(e),
                exc_info=True
            )
            
            self.execution_stats["hooks_failed"] += 1
            
            return HookResult(
                success=False,
                exit_code=1,
                stderr=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def execute_post_tool_use_hooks(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_response: Dict[str, Any],
        success: bool,
        agent_id: str,
        session_id: str,
        execution_time_ms: float,
        correlation_id: Optional[str] = None
    ) -> HookResult:
        """
        Execute PostToolUse hooks for automation and quality checks.
        
        Args:
            tool_name: Name of the executed tool
            tool_input: Tool input parameters
            tool_response: Tool execution result
            success: Whether tool execution succeeded
            agent_id: Agent that executed the tool
            session_id: Current session ID
            execution_time_ms: Tool execution time
            correlation_id: Correlation ID for tracking
            
        Returns:
            HookResult with automation results
        """
        start_time = time.time()
        
        try:
            # Find matching hooks
            matching_hooks = self._find_matching_hooks(self.config.PostToolUse, tool_name)
            
            if not matching_hooks:
                return HookResult(
                    success=True,
                    exit_code=0,
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Prepare hook input data (Claude Code format)
            hook_input = {
                "session_id": session_id,
                "agent_id": agent_id,
                "hook_event_name": "PostToolUse",
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_response": tool_response,
                "success": success,
                "execution_time_ms": execution_time_ms,
                "correlation_id": correlation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "cwd": str(self.project_root),
                "project_dir": str(self.project_root)
            }
            
            # Execute hooks in parallel
            hook_results = await self._execute_hooks_parallel(matching_hooks, hook_input)
            
            # Analyze results
            overall_result = self._analyze_hook_results(hook_results, "PostToolUse")
            
            # Log execution to LeanVibe observability
            if self.hook_event_processor:
                await self.hook_event_processor.process_post_tool_use({
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "tool_name": tool_name,
                    "success": success,
                    "result": tool_response,
                    "execution_time_ms": execution_time_ms,
                    "hook_results": [r.dict() for r in hook_results],
                    "correlation_id": correlation_id
                })
            
            # Update stats
            self.execution_stats["hooks_executed"] += len(hook_results)
            self.execution_stats["total_execution_time_ms"] += overall_result.execution_time_ms
            
            logger.info(
                "🎣 PostToolUse hooks executed",
                tool_name=tool_name,
                hooks_count=len(hook_results),
                success=success,
                execution_time_ms=overall_result.execution_time_ms
            )
            
            return overall_result
            
        except Exception as e:
            logger.error(
                "❌ Failed to execute PostToolUse hooks",
                tool_name=tool_name,
                error=str(e),
                exc_info=True
            )
            
            self.execution_stats["hooks_failed"] += 1
            
            return HookResult(
                success=False,
                exit_code=1,
                stderr=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def execute_workflow_hooks(
        self,
        event: str,  # "WorkflowStart" or "WorkflowComplete"
        workflow_id: str,
        workflow_data: Dict[str, Any],
        agent_id: str,
        session_id: str
    ) -> HookResult:
        """
        Execute workflow lifecycle hooks.
        
        Args:
            event: Workflow event type
            workflow_id: Workflow identifier
            workflow_data: Workflow context data
            agent_id: Primary agent ID
            session_id: Current session ID
            
        Returns:
            HookResult with workflow automation results
        """
        start_time = time.time()
        
        try:
            # Get hooks for the event
            hooks_config = getattr(self.config, event, [])
            matching_hooks = self._find_matching_hooks(hooks_config, workflow_data.get("workflow_type", "default"))
            
            if not matching_hooks:
                return HookResult(
                    success=True,
                    exit_code=0,
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Prepare hook input data
            hook_input = {
                "session_id": session_id,
                "agent_id": agent_id,
                "hook_event_name": event,
                "workflow_id": workflow_id,
                "workflow_data": workflow_data,
                "timestamp": datetime.utcnow().isoformat(),
                "cwd": str(self.project_root),
                "project_dir": str(self.project_root)
            }
            
            # Execute hooks
            hook_results = await self._execute_hooks_parallel(matching_hooks, hook_input)
            overall_result = self._analyze_hook_results(hook_results, event)
            
            # Update stats
            self.execution_stats["hooks_executed"] += len(hook_results)
            self.execution_stats["total_execution_time_ms"] += overall_result.execution_time_ms
            
            logger.info(
                f"🎣 {event} hooks executed",
                workflow_id=workflow_id,
                hooks_count=len(hook_results),
                execution_time_ms=overall_result.execution_time_ms
            )
            
            return overall_result
            
        except Exception as e:
            logger.error(
                f"❌ Failed to execute {event} hooks",
                workflow_id=workflow_id,
                error=str(e),
                exc_info=True
            )
            
            self.execution_stats["hooks_failed"] += 1
            
            return HookResult(
                success=False,
                exit_code=1,
                stderr=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def execute_quality_gate_hooks(
        self,
        gate_type: str,
        validation_data: Dict[str, Any],
        agent_id: str,
        session_id: str
    ) -> HookResult:
        """
        Execute quality gate hooks for automated validation.
        
        Args:
            gate_type: Type of quality gate (e.g., "code_quality", "security", "performance")
            validation_data: Data to validate
            agent_id: Agent requesting validation
            session_id: Current session ID
            
        Returns:
            HookResult indicating validation success/failure
        """
        start_time = time.time()
        
        try:
            # Find matching quality gate hooks
            matching_hooks = self._find_matching_hooks(self.config.QualityGate, gate_type)
            
            if not matching_hooks:
                # No hooks configured - pass by default
                return HookResult(
                    success=True,
                    exit_code=0,
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Prepare hook input data
            hook_input = {
                "session_id": session_id,
                "agent_id": agent_id,
                "hook_event_name": "QualityGate",
                "gate_type": gate_type,
                "validation_data": validation_data,
                "timestamp": datetime.utcnow().isoformat(),
                "cwd": str(self.project_root),
                "project_dir": str(self.project_root)
            }
            
            # Execute hooks
            hook_results = await self._execute_hooks_parallel(matching_hooks, hook_input)
            overall_result = self._analyze_hook_results(hook_results, "QualityGate")
            
            # Quality gates block on any failure
            if any(not r.success or r.exit_code != 0 for r in hook_results):
                overall_result.blocked = True
                overall_result.continue_execution = False
            
            # Update stats
            self.execution_stats["hooks_executed"] += len(hook_results)
            self.execution_stats["total_execution_time_ms"] += overall_result.execution_time_ms
            if overall_result.blocked:
                self.execution_stats["blocked_actions"] += 1
            
            logger.info(
                "🛡️ Quality gate hooks executed",
                gate_type=gate_type,
                hooks_count=len(hook_results),
                passed=not overall_result.blocked,
                execution_time_ms=overall_result.execution_time_ms
            )
            
            return overall_result
            
        except Exception as e:
            logger.error(
                "❌ Failed to execute quality gate hooks",
                gate_type=gate_type,
                error=str(e),
                exc_info=True
            )
            
            self.execution_stats["hooks_failed"] += 1
            
            # Quality gates fail closed
            return HookResult(
                success=False,
                exit_code=1,
                stderr=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
                blocked=True,
                continue_execution=False
            )
    
    def _find_matching_hooks(self, hooks_config: List[HookMatcher], target: str) -> List[HookCommand]:
        """Find hooks that match the target pattern."""
        matching_hooks = []
        
        for matcher in hooks_config:
            if self._pattern_matches(matcher.pattern, target):
                matching_hooks.extend(matcher.hooks)
        
        return matching_hooks
    
    def _pattern_matches(self, pattern: str, target: str) -> bool:
        """Check if pattern matches target (Claude Code style)."""
        # Handle empty pattern or wildcard
        if not pattern or pattern == "*" or pattern == "":
            return True
        
        # Handle exact match
        if pattern == target:
            return True
        
        # Handle regex-style patterns (simple implementation)
        if "|" in pattern:
            # Handle OR patterns like "Edit|Write|MultiEdit"
            patterns = [p.strip() for p in pattern.split("|")]
            return any(self._pattern_matches(p, target) for p in patterns)
        
        # Handle wildcard patterns
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return target.startswith(prefix)
        
        if pattern.startswith("*"):
            suffix = pattern[1:]
            return target.endswith(suffix)
        
        return False
    
    async def _execute_hooks_parallel(
        self, 
        hooks: List[HookCommand], 
        hook_input: Dict[str, Any]
    ) -> List[HookResult]:
        """Execute hooks in parallel with timeout handling."""
        if not hooks:
            return []
        
        # Create tasks for parallel execution
        tasks = []
        for hook in hooks:
            task = asyncio.create_task(
                self._execute_single_hook(hook, hook_input)
            )
            tasks.append(task)
        
        # Wait for all hooks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        hook_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                hook_results.append(HookResult(
                    success=False,
                    exit_code=1,
                    stderr=str(result),
                    execution_time_ms=0
                ))
            else:
                hook_results.append(result)
        
        return hook_results
    
    async def _execute_single_hook(
        self, 
        hook: HookCommand, 
        hook_input: Dict[str, Any]
    ) -> HookResult:
        """Execute a single hook command."""
        start_time = time.time()
        
        try:
            # Prepare command with environment variables
            env = os.environ.copy()
            env["LEANVIBE_PROJECT_DIR"] = str(self.project_root)
            env["CLAUDE_PROJECT_DIR"] = str(self.project_root)  # Claude Code compatibility
            
            # Create temporary input file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(hook_input, f, default=str)
                input_file = f.name
            
            try:
                # Execute command with stdin from input file
                process = await asyncio.create_subprocess_shell(
                    hook.command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=str(self.project_root)
                )
                
                # Read input file and send to stdin
                with open(input_file, 'r') as f:
                    input_data = f.read()
                
                # Wait for completion with timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input_data.encode()),
                    timeout=hook.timeout or 60
                )
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Analyze output for blocking (Claude Code style)
                stdout_str = stdout.decode() if stdout else ""
                stderr_str = stderr.decode() if stderr else ""
                
                # Check for JSON output (advanced control)
                blocked = False
                continue_execution = True
                
                if process.returncode == 2:
                    # Exit code 2 means blocking error (Claude Code style)
                    blocked = True
                    continue_execution = False
                
                # Try to parse JSON output for advanced control
                try:
                    if stdout_str.strip():
                        output_data = json.loads(stdout_str)
                        if isinstance(output_data, dict):
                            blocked = not output_data.get("continue", True)
                            continue_execution = output_data.get("continue", True)
                except json.JSONDecodeError:
                    pass  # Not JSON output, use exit code logic
                
                return HookResult(
                    success=process.returncode == 0,
                    exit_code=process.returncode,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    execution_time_ms=execution_time_ms,
                    blocked=blocked,
                    continue_execution=continue_execution
                )
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(input_file)
                except OSError:
                    pass
                
        except asyncio.TimeoutError:
            execution_time_ms = (time.time() - start_time) * 1000
            return HookResult(
                success=False,
                exit_code=124,  # Timeout exit code
                stderr=f"Hook timed out after {hook.timeout}s",
                execution_time_ms=execution_time_ms
            )
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return HookResult(
                success=False,
                exit_code=1,
                stderr=str(e),
                execution_time_ms=execution_time_ms
            )
    
    def _analyze_hook_results(
        self, 
        hook_results: List[HookResult], 
        event_type: str
    ) -> HookResult:
        """Analyze multiple hook results into overall result."""
        if not hook_results:
            return HookResult(
                success=True,
                exit_code=0,
                execution_time_ms=0
            )
        
        # Aggregate results
        total_time = sum(r.execution_time_ms for r in hook_results)
        all_stdout = "\n".join(r.stdout for r in hook_results if r.stdout)
        all_stderr = "\n".join(r.stderr for r in hook_results if r.stderr)
        
        # Determine overall success
        overall_success = all(r.success for r in hook_results)
        overall_exit_code = 0 if overall_success else max(r.exit_code for r in hook_results)
        
        # Determine blocking behavior
        blocked = any(r.blocked for r in hook_results)
        continue_execution = all(r.continue_execution for r in hook_results)
        
        return HookResult(
            success=overall_success,
            exit_code=overall_exit_code,
            stdout=all_stdout,
            stderr=all_stderr,
            execution_time_ms=total_time,
            blocked=blocked,
            continue_execution=continue_execution
        )
    
    async def create_project_hooks_directory(self) -> None:
        """Create project hooks directory with example hooks."""
        hooks_dir = self.project_hooks_dir
        hooks_dir.mkdir(parents=True, exist_ok=True)
        
        # Create example hooks
        examples = {
            "security_check.py": self._get_security_hook_example(),
            "code_format.py": self._get_formatting_hook_example(),
            "test_runner.py": self._get_testing_hook_example(),
            "quality_gate.py": self._get_quality_gate_example()
        }
        
        for filename, content in examples.items():
            hook_file = hooks_dir / filename
            if not hook_file.exists():
                hook_file.write_text(content)
        
        # Create example configuration
        config_example = {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Edit|Write|MultiEdit",
                        "hooks": [
                            {
                                "type": "command",
                                "command": f"python {hooks_dir}/security_check.py",
                                "description": "Check for security issues before file modifications"
                            }
                        ]
                    }
                ],
                "PostToolUse": [
                    {
                        "matcher": "Edit|Write|MultiEdit",
                        "hooks": [
                            {
                                "type": "command", 
                                "command": f"python {hooks_dir}/code_format.py",
                                "description": "Auto-format code after modifications"
                            }
                        ]
                    }
                ],
                "QualityGate": [
                    {
                        "matcher": "*",
                        "hooks": [
                            {
                                "type": "command",
                                "command": f"python {hooks_dir}/quality_gate.py",
                                "description": "Comprehensive quality validation"
                            }
                        ]
                    }
                ]
            }
        }
        
        config_file = self.project_root / ".leanvibe" / "settings.example.json"
        if not config_file.exists():
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config_example, f, indent=2)
        
        logger.info(
            "📁 Project hooks directory created",
            hooks_dir=str(hooks_dir),
            examples_created=len(examples)
        )
    
    def _get_security_hook_example(self) -> str:
        """Get example security check hook."""
        return '''#!/usr/bin/env python3
"""
Example security check hook for LeanVibe Agent Hive.
Validates file modifications for security issues.
"""

import json
import sys
import re

def check_security_issues(file_path, content):
    """Check for common security issues."""
    issues = []
    
    # Check for exposed secrets
    secret_patterns = [
        (r'(?i)(password|secret|key|token)\\s*[:=]\\s*["\']?([^"\\'\\s,}]+)', 'Potential secret in code'),
        (r'(?i)api[_-]?key\\s*[:=]\\s*["\']?([a-zA-Z0-9]{20,})', 'API key detected'),
        (r'\\b[A-Za-z0-9]{32,}\\b', 'Potential token or key'),
    ]
    
    for pattern, message in secret_patterns:
        if re.search(pattern, content):
            issues.append(f"{message}: {file_path}")
    
    # Check for unsafe operations
    if 'eval(' in content:
        issues.append(f"Unsafe eval() usage in {file_path}")
    
    if 'exec(' in content:
        issues.append(f"Unsafe exec() usage in {file_path}")
    
    return issues

def main():
    try:
        # Read hook input
        input_data = json.load(sys.stdin)
        
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})
        
        # Only check file modification tools
        if tool_name not in ["Edit", "Write", "MultiEdit"]:
            sys.exit(0)
        
        # Get file path and content
        file_path = tool_input.get("file_path", "")
        content = tool_input.get("content", "") or tool_input.get("new_string", "")
        
        if not file_path or not content:
            sys.exit(0)
        
        # Skip certain file types
        if any(file_path.endswith(ext) for ext in ['.md', '.txt', '.json', '.yml', '.yaml']):
            sys.exit(0)
        
        # Check for security issues
        issues = check_security_issues(file_path, content)
        
        if issues:
            # Report issues and block
            for issue in issues:
                print(f"🔒 Security issue: {issue}", file=sys.stderr)
            
            # Exit code 2 blocks the action (Claude Code style)
            sys.exit(2)
        
        # No issues found
        print("✅ Security check passed")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Security check failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    def _get_formatting_hook_example(self) -> str:
        """Get example code formatting hook."""
        return '''#!/usr/bin/env python3
"""
Example code formatting hook for LeanVibe Agent Hive.
Auto-formats code after modifications.
"""

import json
import sys
import subprocess
import os

def format_file(file_path):
    """Format file based on extension."""
    try:
        if file_path.endswith('.py'):
            # Format Python with black
            result = subprocess.run(['black', file_path], capture_output=True, text=True)
            if result.returncode == 0:
                return True, "Formatted with black"
            else:
                return False, f"Black failed: {result.stderr}"
        
        elif file_path.endswith(('.ts', '.tsx', '.js', '.jsx')):
            # Format TypeScript/JavaScript with prettier
            result = subprocess.run(['npx', 'prettier', '--write', file_path], capture_output=True, text=True)
            if result.returncode == 0:
                return True, "Formatted with prettier"
            else:
                return False, f"Prettier failed: {result.stderr}"
        
        elif file_path.endswith('.go'):
            # Format Go with gofmt
            result = subprocess.run(['gofmt', '-w', file_path], capture_output=True, text=True)
            if result.returncode == 0:
                return True, "Formatted with gofmt"
            else:
                return False, f"gofmt failed: {result.stderr}"
        
        return True, "No formatter configured for this file type"
        
    except FileNotFoundError as e:
        return False, f"Formatter not found: {e}"
    except Exception as e:
        return False, f"Formatting error: {e}"

def main():
    try:
        # Read hook input
        input_data = json.load(sys.stdin)
        
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})
        
        # Only format after file modifications
        if tool_name not in ["Edit", "Write", "MultiEdit"]:
            sys.exit(0)
        
        file_path = tool_input.get("file_path", "")
        
        if not file_path or not os.path.exists(file_path):
            sys.exit(0)
        
        # Format the file
        success, message = format_file(file_path)
        
        if success:
            print(f"🎨 {message}: {file_path}")
            sys.exit(0)
        else:
            print(f"⚠️ {message}", file=sys.stderr)
            # Don't block on formatting failures
            sys.exit(0)
        
    except Exception as e:
        print(f"❌ Formatting hook failed: {e}", file=sys.stderr)
        sys.exit(0)  # Don't block on hook errors

if __name__ == "__main__":
    main()
'''
    
    def _get_testing_hook_example(self) -> str:
        """Get example testing hook."""
        return '''#!/usr/bin/env python3
"""
Example testing hook for LeanVibe Agent Hive.
Runs relevant tests after code modifications.
"""

import json
import sys
import subprocess
import os

def run_tests(file_path):
    """Run tests related to the modified file."""
    try:
        # Determine test command based on file type and project structure
        if file_path.endswith('.py'):
            # Python tests
            test_file = file_path.replace('.py', '_test.py').replace('/app/', '/tests/')
            if os.path.exists(test_file):
                result = subprocess.run(['pytest', test_file, '-v'], capture_output=True, text=True)
                return result.returncode == 0, result.stdout, result.stderr
            else:
                # Run all tests if specific test file not found
                result = subprocess.run(['pytest', '-x', '--tb=short'], capture_output=True, text=True)
                return result.returncode == 0, result.stdout, result.stderr
        
        elif file_path.endswith(('.ts', '.tsx', '.js', '.jsx')):
            # JavaScript/TypeScript tests
            result = subprocess.run(['npm', 'test', '--', '--watchAll=false'], capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        
        return True, "No tests configured for this file type", ""
        
    except Exception as e:
        return False, "", str(e)

def main():
    try:
        # Read hook input
        input_data = json.load(sys.stdin)
        
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})
        
        # Only run tests after file modifications
        if tool_name not in ["Edit", "Write", "MultiEdit"]:
            sys.exit(0)
        
        file_path = tool_input.get("file_path", "")
        
        if not file_path:
            sys.exit(0)
        
        # Skip test files themselves
        if 'test' in file_path.lower():
            sys.exit(0)
        
        # Run tests
        success, stdout, stderr = run_tests(file_path)
        
        if success:
            print(f"✅ Tests passed for {file_path}")
            if stdout:
                print(stdout)
            sys.exit(0)
        else:
            print(f"❌ Tests failed for {file_path}", file=sys.stderr)
            if stderr:
                print(stderr, file=sys.stderr)
            # Exit code 2 to provide feedback to agent
            sys.exit(2)
        
    except Exception as e:
        print(f"❌ Test hook failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    def _get_quality_gate_example(self) -> str:
        """Get example quality gate hook."""
        return '''#!/usr/bin/env python3
"""
Example quality gate hook for LeanVibe Agent Hive.
Comprehensive quality validation.
"""

import json
import sys
import subprocess
import os

def check_code_quality():
    """Run comprehensive code quality checks."""
    checks = []
    
    try:
        # Python code quality
        if any(f.endswith('.py') for f in os.listdir('.') if os.path.isfile(f)):
            # Run flake8
            result = subprocess.run(['flake8', '.'], capture_output=True, text=True)
            checks.append({
                'name': 'Python Style (flake8)',
                'passed': result.returncode == 0,
                'output': result.stdout or result.stderr
            })
            
            # Run mypy
            result = subprocess.run(['mypy', '.'], capture_output=True, text=True)
            checks.append({
                'name': 'Python Types (mypy)',
                'passed': result.returncode == 0,
                'output': result.stdout or result.stderr
            })
    
    except FileNotFoundError:
        pass  # Tool not installed
    
    try:
        # TypeScript/JavaScript quality
        if os.path.exists('package.json'):
            # Run ESLint
            result = subprocess.run(['npx', 'eslint', '.'], capture_output=True, text=True)
            checks.append({
                'name': 'JavaScript/TypeScript Lint (ESLint)',
                'passed': result.returncode == 0,
                'output': result.stdout or result.stderr
            })
    
    except FileNotFoundError:
        pass
    
    return checks

def check_test_coverage():
    """Check test coverage."""
    try:
        # Python coverage
        result = subprocess.run(['coverage', 'report', '--show-missing'], capture_output=True, text=True)
        if result.returncode == 0:
            # Parse coverage percentage
            lines = result.stdout.split('\\n')
            total_line = [l for l in lines if 'TOTAL' in l]
            if total_line:
                coverage_pct = total_line[0].split()[-1].replace('%', '')
                return float(coverage_pct) >= 90.0, f"Coverage: {coverage_pct}%"
        
        return True, "Coverage check skipped"
    
    except (FileNotFoundError, ValueError):
        return True, "Coverage tool not available"

def main():
    try:
        # Read hook input
        input_data = json.load(sys.stdin)
        
        gate_type = input_data.get("gate_type", "comprehensive")
        
        results = {
            'quality_checks': [],
            'coverage_check': None,
            'overall_passed': True
        }
        
        # Run quality checks
        if gate_type in ["comprehensive", "code_quality"]:
            results['quality_checks'] = check_code_quality()
        
        # Run coverage check
        if gate_type in ["comprehensive", "test_coverage"]:
            coverage_passed, coverage_msg = check_test_coverage()
            results['coverage_check'] = {
                'passed': coverage_passed,
                'message': coverage_msg
            }
        
        # Determine overall result
        quality_passed = all(check['passed'] for check in results['quality_checks'])
        coverage_passed = results['coverage_check']['passed'] if results['coverage_check'] else True
        
        results['overall_passed'] = quality_passed and coverage_passed
        
        # Output results
        print(json.dumps(results, indent=2))
        
        if results['overall_passed']:
            print("✅ All quality gates passed", file=sys.stderr)
            sys.exit(0)
        else:
            print("❌ Quality gates failed", file=sys.stderr)
            sys.exit(2)  # Block action
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'overall_passed': False
        }
        print(json.dumps(error_result, indent=2))
        print(f"❌ Quality gate error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the hooks engine."""
        return {
            "execution_stats": self.execution_stats,
            "config_loaded": bool(self.config),
            "config_events": len([
                event for event in ["PreToolUse", "PostToolUse", "AgentStart", "AgentStop", 
                                  "WorkflowStart", "WorkflowComplete", "QualityGate", "Notification"]
                if getattr(self.config, event, [])
            ]),
            "project_hooks_dir": str(self.project_hooks_dir),
            "project_hooks_dir_exists": self.project_hooks_dir.exists()
        }


# Global hooks engine instance
_claude_code_hooks_engine: Optional[ClaudeCodeHooksEngine] = None


def get_claude_code_hooks_engine() -> Optional[ClaudeCodeHooksEngine]:
    """Get the global Claude Code hooks engine instance."""
    return _claude_code_hooks_engine


def set_claude_code_hooks_engine(engine: ClaudeCodeHooksEngine) -> None:
    """Set the global Claude Code hooks engine instance."""
    global _claude_code_hooks_engine
    _claude_code_hooks_engine = engine
    logger.info("🔗 Global Claude Code hooks engine set")


async def initialize_claude_code_hooks_engine(
    project_root: Optional[Path] = None
) -> ClaudeCodeHooksEngine:
    """
    Initialize and set the global Claude Code hooks engine.
    
    Args:
        project_root: Project root directory
        
    Returns:
        ClaudeCodeHooksEngine instance
    """
    engine = ClaudeCodeHooksEngine(project_root=project_root)
    
    # Load configuration
    await engine.load_configuration()
    
    # Create project hooks directory if it doesn't exist
    await engine.create_project_hooks_directory()
    
    set_claude_code_hooks_engine(engine)
    
    logger.info("✅ Claude Code hooks engine initialized")
    return engine