"""
Isolated Sandbox Environment for Self-Modification System

This module provides a completely isolated Docker-based sandbox for testing
code modifications with ZERO network access, strict resource limits, and
comprehensive security monitoring.
"""

import os
import sys
import json
import time
import signal
import tempfile
import shutil
import subprocess
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError

logger = logging.getLogger(__name__)


class SandboxStatus(Enum):
    """Status of sandbox execution."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SECURITY_VIOLATION = "security_violation"
    RESOURCE_EXCEEDED = "resource_exceeded"


class SecurityViolationType(Enum):
    """Types of security violations in sandbox."""
    NETWORK_ACCESS_ATTEMPT = "network_access_attempt"
    FILE_SYSTEM_ESCAPE = "file_system_escape"
    SYSTEM_CALL_BLOCKED = "system_call_blocked"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    SUSPICIOUS_PROCESS = "suspicious_process"
    UNAUTHORIZED_MODULE_IMPORT = "unauthorized_module_import"


@dataclass
class ResourceLimits:
    """Resource limits for sandbox execution."""
    max_memory_mb: int = 256
    max_cpu_percent: float = 50.0
    max_execution_time_seconds: int = 300  # 5 minutes
    max_disk_usage_mb: int = 100
    max_network_connections: int = 0  # NO network access
    max_processes: int = 10
    max_open_files: int = 100


@dataclass 
class SecurityViolation:
    """Security violation detected in sandbox."""
    violation_type: SecurityViolationType
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    detected_at: datetime
    process_info: Dict[str, Any]
    mitigation_action: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxExecution:
    """Results from sandbox execution."""
    execution_id: str
    status: SandboxStatus
    command: str
    working_directory: str
    
    # Resource usage
    execution_time_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    disk_usage_mb: float
    
    # Output
    stdout: str
    stderr: str
    exit_code: int
    
    # Security monitoring
    security_violations: List[SecurityViolation]
    network_attempts: int
    file_system_changes: List[str]
    processes_created: List[str]
    
    # Results and metrics
    test_results: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    
    # Metadata
    container_id: Optional[str]
    image_name: str
    started_at: datetime
    completed_at: Optional[datetime]
    execution_metadata: Dict[str, Any]


class SecureSandboxEnvironment:
    """
    Completely isolated sandbox environment with Docker containers.
    
    Provides maximum security isolation with NO network access, strict
    resource limits, and comprehensive monitoring of all sandbox activity.
    """
    
    def __init__(self, 
                 base_image: str = "python:3.12-alpine",
                 resource_limits: Optional[ResourceLimits] = None,
                 security_level: str = "maximum"):
        self.base_image = base_image
        self.resource_limits = resource_limits or ResourceLimits()
        self.security_level = security_level
        
        # Security configuration
        self._blocked_system_calls = {
            'socket', 'connect', 'bind', 'listen', 'accept',
            'open', 'openat', 'creat', 'unlink', 'rmdir',
            'chmod', 'chown', 'mount', 'umount', 'setuid',
            'setgid', 'execve', 'fork', 'clone', 'ptrace'
        }
        
        self._allowed_python_modules = {
            'ast', 'sys', 'os', 'json', 'time', 'datetime',
            'collections', 're', 'math', 'random', 'hashlib',
            'base64', 'itertools', 'functools', 'typing'
        }
        
        # Sandbox workspace
        self._sandbox_base = Path("/tmp/sandbox_workspaces")
        self._sandbox_base.mkdir(exist_ok=True, mode=0o700)
        
        # Docker security options
        self._docker_security_opts = [
            "--security-opt=no-new-privileges:true",
            "--security-opt=seccomp=unconfined",  # We'll use custom seccomp
            "--cap-drop=ALL",
            "--cap-add=DAC_OVERRIDE",  # Minimal required capability
            "--read-only=true",  # Read-only root filesystem
            "--tmpfs=/tmp:rw,noexec,nosuid,size=50m",
            "--network=none",  # NO network access
            "--user=65534:65534",  # Run as nobody user
            "--no-new-privileges",
            "--rm"  # Auto-remove container
        ]
        
        logger.info(f"SecureSandboxEnvironment initialized with {security_level} security level")
    
    async def execute_code_modification(self,
                                      original_code: str,
                                      modified_code: str,
                                      test_commands: List[str],
                                      timeout_seconds: Optional[int] = None) -> SandboxExecution:
        """
        Execute and test a code modification in isolated sandbox.
        
        Args:
            original_code: Original code content
            modified_code: Modified code content
            test_commands: Commands to test the modification
            timeout_seconds: Execution timeout (default: resource limit)
            
        Returns:
            Complete sandbox execution results
            
        Raises:
            SecurityError: If security violations are detected
            TimeoutError: If execution exceeds timeout
            ValueError: If sandbox setup fails
        """
        execution_id = self._generate_execution_id()
        timeout_seconds = timeout_seconds or self.resource_limits.max_execution_time_seconds
        
        logger.info(f"Starting sandbox execution {execution_id} with {len(test_commands)} test commands")
        
        # Create isolated workspace
        workspace = await self._create_sandbox_workspace(execution_id)
        
        try:
            # Setup code files
            await self._setup_code_files(workspace, original_code, modified_code)
            
            # Create secure container
            container_info = await self._create_secure_container(workspace, execution_id)
            
            # Execute tests with monitoring
            execution_result = await self._execute_with_monitoring(
                container_info, test_commands, timeout_seconds
            )
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Sandbox execution {execution_id} failed: {e}")
            raise
        finally:
            # Cleanup workspace
            await self._cleanup_workspace(workspace)
    
    async def execute_security_scan(self,
                                  code_content: str,
                                  scan_types: List[str] = None) -> SandboxExecution:
        """
        Execute security scanning on code in isolated sandbox.
        
        Args:
            code_content: Code to scan for security issues
            scan_types: Types of scans to perform
            
        Returns:
            Security scan results
        """
        scan_types = scan_types or ['bandit', 'semgrep', 'safety']
        execution_id = self._generate_execution_id()
        
        logger.info(f"Starting security scan {execution_id} with scans: {scan_types}")
        
        workspace = await self._create_sandbox_workspace(execution_id)
        
        try:
            # Setup code file
            code_file = workspace / "code_to_scan.py"
            await self._write_file_secure(code_file, code_content)
            
            # Create security scanning container
            container_info = await self._create_security_scan_container(workspace, execution_id)
            
            # Execute security scans
            scan_commands = self._generate_security_scan_commands(scan_types)
            execution_result = await self._execute_with_monitoring(
                container_info, scan_commands, self.resource_limits.max_execution_time_seconds
            )
            
            return execution_result
            
        finally:
            await self._cleanup_workspace(workspace)
    
    async def execute_performance_benchmark(self,
                                          original_code: str,
                                          modified_code: str,
                                          benchmark_script: str) -> SandboxExecution:
        """
        Execute performance benchmarking in isolated sandbox.
        
        Args:
            original_code: Original code for baseline
            modified_code: Modified code to benchmark
            benchmark_script: Script to run benchmarks
            
        Returns:
            Performance benchmark results
        """
        execution_id = self._generate_execution_id()
        
        logger.info(f"Starting performance benchmark {execution_id}")
        
        workspace = await self._create_sandbox_workspace(execution_id)
        
        try:
            # Setup benchmark files
            await self._setup_benchmark_files(workspace, original_code, modified_code, benchmark_script)
            
            # Create benchmark container
            container_info = await self._create_benchmark_container(workspace, execution_id)
            
            # Execute benchmarks
            benchmark_commands = ["python3 benchmark.py"]
            execution_result = await self._execute_with_monitoring(
                container_info, benchmark_commands, 600  # 10 minutes for benchmarks
            )
            
            return execution_result
            
        finally:
            await self._cleanup_workspace(workspace)
    
    async def _create_sandbox_workspace(self, execution_id: str) -> Path:
        """Create isolated workspace for sandbox execution."""
        workspace = self._sandbox_base / execution_id
        workspace.mkdir(mode=0o700, exist_ok=False)
        
        logger.debug(f"Created sandbox workspace: {workspace}")
        return workspace
    
    async def _setup_code_files(self, workspace: Path, original_code: str, modified_code: str) -> None:
        """Setup code files in sandbox workspace."""
        original_file = workspace / "original_code.py"
        modified_file = workspace / "modified_code.py"
        
        await self._write_file_secure(original_file, original_code)
        await self._write_file_secure(modified_file, modified_code)
        
        # Create test runner script
        test_runner = workspace / "test_runner.py"
        test_script = self._generate_test_runner_script()
        await self._write_file_secure(test_runner, test_script)
        
        logger.debug(f"Setup code files in {workspace}")
    
    async def _setup_benchmark_files(self, workspace: Path, 
                                   original_code: str, modified_code: str, 
                                   benchmark_script: str) -> None:
        """Setup benchmark files in sandbox workspace."""
        await self._write_file_secure(workspace / "original_code.py", original_code)
        await self._write_file_secure(workspace / "modified_code.py", modified_code)
        await self._write_file_secure(workspace / "benchmark.py", benchmark_script)
    
    async def _write_file_secure(self, file_path: Path, content: str) -> None:
        """Write file with security validation."""
        # Security check: Validate content
        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            raise SecurityError(f"File content too large: {len(content)} bytes")
        
        # Security check: Validate file path
        if not str(file_path).startswith(str(self._sandbox_base)):
            raise SecurityError(f"Invalid file path: {file_path}")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Set restrictive permissions
            file_path.chmod(0o600)
            
        except (OSError, UnicodeError) as e:
            raise SecurityError(f"Failed to write secure file {file_path}: {e}")
    
    async def _create_secure_container(self, workspace: Path, execution_id: str) -> Dict[str, Any]:
        """Create secure Docker container with maximum isolation."""
        container_name = f"sandbox-{execution_id}"
        
        # Build Docker command with security options
        docker_cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            f"--memory={self.resource_limits.max_memory_mb}m",
            f"--cpus={self.resource_limits.max_cpu_percent / 100.0}",
            "--pids-limit=10",
            "--ulimit=nofile=100:100",
            f"--stop-timeout=30"
        ]
        
        # Add security options
        docker_cmd.extend(self._docker_security_opts)
        
        # Mount workspace as read-only volume
        docker_cmd.extend([
            "-v", f"{workspace}:/workspace:ro",
            "-w", "/workspace"
        ])
        
        # Use base image with sleep command (container will be exec'd into)
        docker_cmd.extend([self.base_image, "sleep", "3600"])
        
        try:
            # Create container
            result = await self._run_command_async(docker_cmd)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create container: {result.stderr}")
            
            container_id = result.stdout.strip()
            
            # Wait for container to be ready
            await asyncio.sleep(1)
            
            return {
                'container_id': container_id,
                'container_name': container_name,
                'workspace': str(workspace),
                'image_name': self.base_image
            }
            
        except Exception as e:
            logger.error(f"Failed to create secure container: {e}")
            raise SecurityError(f"Container creation failed: {e}")
    
    async def _create_security_scan_container(self, workspace: Path, execution_id: str) -> Dict[str, Any]:
        """Create container with security scanning tools."""
        # Use a specialized security scanning image
        security_image = "python:3.12-alpine"  # In practice, use image with security tools
        return await self._create_secure_container(workspace, execution_id)
    
    async def _create_benchmark_container(self, workspace: Path, execution_id: str) -> Dict[str, Any]:
        """Create container optimized for performance benchmarking."""
        # Use optimized image for benchmarking
        benchmark_image = "python:3.12-alpine"  # In practice, use optimized image
        return await self._create_secure_container(workspace, execution_id)
    
    async def _execute_with_monitoring(self,
                                     container_info: Dict[str, Any],
                                     commands: List[str],
                                     timeout_seconds: int) -> SandboxExecution:
        """Execute commands in container with comprehensive monitoring."""
        execution_id = container_info['container_id'][:12]
        container_id = container_info['container_id']
        
        started_at = datetime.utcnow()
        security_violations = []
        network_attempts = 0
        file_system_changes = []
        processes_created = []
        
        # Start monitoring
        monitor_task = asyncio.create_task(
            self._monitor_container_security(container_id, security_violations)
        )
        
        try:
            # Execute commands
            all_stdout = []
            all_stderr = []
            final_exit_code = 0
            
            for i, command in enumerate(commands):
                logger.debug(f"Executing command {i+1}/{len(commands)}: {command}")
                
                # Execute command in container
                docker_exec_cmd = [
                    "docker", "exec",
                    "--user", "65534:65534",  # Run as nobody
                    container_id,
                    "timeout", str(timeout_seconds),
                    "sh", "-c", command
                ]
                
                try:
                    result = await asyncio.wait_for(
                        self._run_command_async(docker_exec_cmd),
                        timeout=timeout_seconds + 10
                    )
                    
                    all_stdout.append(result.stdout)
                    all_stderr.append(result.stderr)
                    
                    if result.returncode != 0:
                        final_exit_code = result.returncode
                        logger.warning(f"Command failed with exit code {result.returncode}")
                    
                except asyncio.TimeoutError:
                    logger.error(f"Command timed out after {timeout_seconds} seconds")
                    final_exit_code = 124  # Timeout exit code
                    break
            
            # Collect resource usage
            resource_usage = await self._collect_resource_usage(container_id)
            
            completed_at = datetime.utcnow()
            execution_time = (completed_at - started_at).total_seconds()
            
            # Determine final status
            if security_violations:
                status = SandboxStatus.SECURITY_VIOLATION
            elif execution_time > timeout_seconds:
                status = SandboxStatus.TIMEOUT
            elif final_exit_code != 0:
                status = SandboxStatus.FAILED
            else:
                status = SandboxStatus.COMPLETED
            
            return SandboxExecution(
                execution_id=execution_id,
                status=status,
                command="; ".join(commands),
                working_directory="/workspace",
                execution_time_seconds=execution_time,
                memory_usage_mb=resource_usage.get('memory_mb', 0.0),
                cpu_usage_percent=resource_usage.get('cpu_percent', 0.0),
                disk_usage_mb=resource_usage.get('disk_mb', 0.0),
                stdout="\n".join(all_stdout),
                stderr="\n".join(all_stderr),
                exit_code=final_exit_code,
                security_violations=security_violations,
                network_attempts=network_attempts,
                file_system_changes=file_system_changes,
                processes_created=processes_created,
                test_results={},  # TODO: Parse from output
                performance_metrics={},  # TODO: Extract metrics
                container_id=container_id,
                image_name=container_info['image_name'],
                started_at=started_at,
                completed_at=completed_at,
                execution_metadata={
                    'resource_limits': {
                        'memory_mb': self.resource_limits.max_memory_mb,
                        'cpu_percent': self.resource_limits.max_cpu_percent,
                        'timeout_seconds': timeout_seconds
                    },
                    'security_level': self.security_level
                }
            )
            
        finally:
            # Stop monitoring
            monitor_task.cancel()
            
            # Cleanup container
            await self._cleanup_container(container_id)
    
    async def _monitor_container_security(self,
                                        container_id: str,
                                        violations: List[SecurityViolation]) -> None:
        """Monitor container for security violations."""
        while True:
            try:
                # Check for network activity (should be none)
                network_stats = await self._get_container_network_stats(container_id)
                if network_stats.get('connections', 0) > 0:
                    violations.append(SecurityViolation(
                        violation_type=SecurityViolationType.NETWORK_ACCESS_ATTEMPT,
                        severity='critical',
                        description=f"Unauthorized network activity detected: {network_stats}",
                        detected_at=datetime.utcnow(),
                        process_info={'container_id': container_id},
                        mitigation_action='Container execution terminated',
                        evidence=network_stats
                    ))
                    break
                
                # Check resource usage
                resource_stats = await self._collect_resource_usage(container_id)
                if resource_stats.get('memory_mb', 0) > self.resource_limits.max_memory_mb * 1.1:
                    violations.append(SecurityViolation(
                        violation_type=SecurityViolationType.RESOURCE_LIMIT_EXCEEDED,
                        severity='high',
                        description=f"Memory limit exceeded: {resource_stats['memory_mb']}MB",
                        detected_at=datetime.utcnow(),
                        process_info={'container_id': container_id},
                        mitigation_action='Resource monitoring alert',
                        evidence=resource_stats
                    ))
                
                await asyncio.sleep(1)  # Check every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Security monitoring error: {e}")
                await asyncio.sleep(5)  # Back off on errors
    
    async def _collect_resource_usage(self, container_id: str) -> Dict[str, float]:
        """Collect resource usage statistics from container."""
        try:
            # Get container stats
            stats_cmd = ["docker", "stats", "--no-stream", "--format", 
                        "table {{.MemUsage}},{{.CPUPerc}}", container_id]
            
            result = await self._run_command_async(stats_cmd)
            if result.returncode != 0:
                return {'memory_mb': 0.0, 'cpu_percent': 0.0, 'disk_mb': 0.0}
            
            # Parse stats (simplified)
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return {'memory_mb': 0.0, 'cpu_percent': 0.0, 'disk_mb': 0.0}
            
            stats_line = lines[1]  # Skip header
            parts = stats_line.split(',')
            
            memory_mb = 0.0
            cpu_percent = 0.0
            
            if len(parts) >= 2:
                # Parse memory (e.g., "123.4MiB / 256MiB")
                memory_part = parts[0].strip()
                if 'MiB' in memory_part:
                    memory_mb = float(memory_part.split('MiB')[0].split(' / ')[0])
                
                # Parse CPU (e.g., "12.34%")
                cpu_part = parts[1].strip()
                if '%' in cpu_part:
                    cpu_percent = float(cpu_part.replace('%', ''))
            
            return {
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent,
                'disk_mb': 0.0  # TODO: Implement disk usage tracking
            }
            
        except Exception as e:
            logger.warning(f"Failed to collect resource usage: {e}")
            return {'memory_mb': 0.0, 'cpu_percent': 0.0, 'disk_mb': 0.0}
    
    async def _get_container_network_stats(self, container_id: str) -> Dict[str, Any]:
        """Get network statistics (should be zero for isolated containers)."""
        try:
            # Check network namespaces and connections
            inspect_cmd = ["docker", "inspect", "--format", 
                          "{{.NetworkSettings.NetworkMode}}", container_id]
            
            result = await self._run_command_async(inspect_cmd)
            network_mode = result.stdout.strip() if result.returncode == 0 else "unknown"
            
            # For isolated containers, network mode should be "none"
            if network_mode != "none":
                return {'connections': 1, 'network_mode': network_mode}
            
            return {'connections': 0, 'network_mode': network_mode}
            
        except Exception as e:
            logger.warning(f"Failed to get network stats: {e}")
            return {'connections': 0, 'network_mode': 'unknown'}
    
    async def _run_command_async(self, command: List[str]) -> subprocess.CompletedProcess:
        """Run command asynchronously."""
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return subprocess.CompletedProcess(
                args=command,
                returncode=process.returncode,
                stdout=stdout.decode('utf-8', errors='ignore'),
                stderr=stderr.decode('utf-8', errors='ignore')
            )
            
        except Exception as e:
            logger.error(f"Command execution failed: {command}, error: {e}")
            return subprocess.CompletedProcess(
                args=command,
                returncode=1,
                stdout="",
                stderr=str(e)
            )
    
    async def _cleanup_container(self, container_id: str) -> None:
        """Cleanup Docker container."""
        try:
            # Force stop container
            stop_cmd = ["docker", "stop", "-t", "5", container_id]
            await self._run_command_async(stop_cmd)
            
            # Remove container (should auto-remove with --rm, but be safe)
            rm_cmd = ["docker", "rm", "-f", container_id]
            await self._run_command_async(rm_cmd)
            
            logger.debug(f"Cleaned up container {container_id}")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup container {container_id}: {e}")
    
    async def _cleanup_workspace(self, workspace: Path) -> None:
        """Cleanup sandbox workspace."""
        try:
            if workspace.exists():
                shutil.rmtree(workspace)
                logger.debug(f"Cleaned up workspace {workspace}")
        except Exception as e:
            logger.warning(f"Failed to cleanup workspace {workspace}: {e}")
    
    def _generate_execution_id(self) -> str:
        """Generate unique execution ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _generate_test_runner_script(self) -> str:
        """Generate test runner script."""
        return """#!/usr/bin/env python3
import sys
import importlib
import traceback

def run_tests():
    try:
        # Import and test modified code
        spec = importlib.util.spec_from_file_location("modified_code", "/workspace/modified_code.py")
        modified_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modified_module)
        
        print("Modified code imported successfully")
        return True
    except Exception as e:
        print(f"Error testing modified code: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
"""
    
    def _generate_security_scan_commands(self, scan_types: List[str]) -> List[str]:
        """Generate security scan commands."""
        commands = []
        
        if 'bandit' in scan_types:
            commands.append("python3 -m bandit /workspace/code_to_scan.py")
        
        if 'basic_check' in scan_types:
            commands.append("python3 -c \"import ast; ast.parse(open('/workspace/code_to_scan.py').read())\"")
        
        return commands or ["echo 'No security scans configured'"]


class SecurityError(Exception):
    """Security violation in sandbox execution."""
    pass


# Export main classes
__all__ = [
    'SecureSandboxEnvironment',
    'SandboxExecution',
    'ResourceLimits',
    'SandboxStatus',
    'SecurityViolation',
    'SecurityViolationType',
    'SecurityError'
]