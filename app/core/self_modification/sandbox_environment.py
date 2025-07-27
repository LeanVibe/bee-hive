"""
Sandbox Environment

Docker-based isolated execution environment for safely testing code modifications.
Provides secure, isolated containers with resource limits, network isolation,
and comprehensive monitoring to prevent sandbox escapes.
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import docker
import structlog
from docker.models.containers import Container
from docker.errors import ContainerError, ImageNotFound, APIError

logger = structlog.get_logger()


@dataclass
class ResourceLimits:
    """Resource limits for sandbox containers."""
    
    memory_mb: int = 256  # Memory limit in MB
    cpu_percent: float = 50.0  # CPU percentage (0-100)
    disk_mb: int = 100  # Disk space limit in MB
    execution_timeout: int = 300  # Execution timeout in seconds
    network_disabled: bool = True  # Disable network access
    max_processes: int = 10  # Maximum number of processes


@dataclass
class SecurityPolicy:
    """Security policies for sandbox execution."""
    
    read_only_filesystem: bool = True
    no_new_privileges: bool = True
    drop_capabilities: List[str] = field(default_factory=lambda: [
        "NET_ADMIN", "SYS_ADMIN", "SYS_TIME", "AUDIT_CONTROL",
        "BLOCK_SUSPEND", "DAC_OVERRIDE", "IPC_LOCK", "MAC_ADMIN",
        "MAC_OVERRIDE", "MKNOD", "SETFCAP", "SETPCAP", "SYS_MODULE",
        "SYS_NICE", "SYS_PACCT", "SYS_PTRACE", "SYS_RAWIO",
        "SYS_RESOURCE", "SYS_BOOT", "WAKE_ALARM"
    ])
    allowed_syscalls: Optional[List[str]] = None  # Seccomp profile
    user_namespace: bool = True


@dataclass
class SandboxResult:
    """Result of sandbox execution."""
    
    execution_id: str
    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    
    # Resource usage
    execution_time_ms: int = 0
    memory_usage_mb: int = 0
    cpu_usage_percent: float = 0.0
    disk_usage_mb: int = 0
    
    # Security monitoring
    network_attempts: int = 0
    security_violations: List[Dict[str, Any]] = field(default_factory=list)
    file_system_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Test results (if applicable)
    test_results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    container_id: Optional[str] = None
    image_name: str = ""
    command: str = ""
    working_directory: str = ""
    environment: Dict[str, str] = field(default_factory=dict)
    
    @property
    def execution_time_seconds(self) -> float:
        """Get execution time in seconds."""
        return self.execution_time_ms / 1000.0
    
    @property
    def has_security_violations(self) -> bool:
        """Check if execution had security violations."""
        return len(self.security_violations) > 0 or self.network_attempts > 0


class SandboxEnvironment:
    """Docker-based sandbox environment for safe code execution."""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.base_images = {
            "python": "python:3.11-slim",
            "node": "node:18-alpine", 
            "generic": "ubuntu:22.04"
        }
        self.active_containers: Dict[str, Container] = {}
        
        # Ensure base images are available
        self._ensure_base_images()
    
    async def execute_code(
        self,
        code_content: str,
        language: str = "python",
        command: Optional[str] = None,
        working_dir: str = "/workspace",
        environment: Optional[Dict[str, str]] = None,
        resource_limits: Optional[ResourceLimits] = None,
        security_policy: Optional[SecurityPolicy] = None,
        file_mappings: Optional[Dict[str, str]] = None
    ) -> SandboxResult:
        """Execute code in a secure sandbox environment."""
        
        execution_id = str(uuid4())
        resource_limits = resource_limits or ResourceLimits()
        security_policy = security_policy or SecurityPolicy()
        environment = environment or {}
        
        logger.info(
            "Starting sandbox execution",
            execution_id=execution_id,
            language=language,
            resource_limits=resource_limits.__dict__
        )
        
        # Create temporary workspace
        workspace_path = None
        container = None
        
        try:
            # Create workspace with code and files
            workspace_path = await self._create_workspace(
                code_content, file_mappings or {}, language
            )
            
            # Create and configure container
            container = await self._create_container(
                language, workspace_path, working_dir, environment,
                resource_limits, security_policy, execution_id
            )
            
            # Execute code and monitor
            result = await self._execute_and_monitor(
                container, command or self._get_default_command(language),
                resource_limits, execution_id
            )
            
            # Analyze results
            await self._analyze_execution_results(result, workspace_path, container)
            
            return result
            
        except Exception as e:
            logger.error(
                "Sandbox execution failed",
                execution_id=execution_id,
                error=str(e)
            )
            
            return SandboxResult(
                execution_id=execution_id,
                success=False,
                exit_code=-1,
                stderr=f"Sandbox execution error: {str(e)}",
                security_violations=[{
                    "type": "execution_error",
                    "description": str(e),
                    "severity": "high"
                }]
            )
            
        finally:
            # Cleanup
            await self._cleanup_execution(container, workspace_path, execution_id)
    
    async def execute_tests(
        self,
        test_files: Dict[str, str],
        source_files: Dict[str, str],
        test_framework: str = "pytest",
        language: str = "python"
    ) -> SandboxResult:
        """Execute tests in sandbox environment."""
        
        # Combine test and source files
        all_files = {**source_files, **test_files}
        
        # Determine test command
        test_commands = {
            "pytest": "python -m pytest -v --tb=short",
            "unittest": "python -m unittest discover -v",
            "nose": "python -m nose2 -v"
        }
        
        command = test_commands.get(test_framework, "python -m pytest -v")
        
        # Execute with test-specific resource limits
        resource_limits = ResourceLimits(
            memory_mb=512,  # More memory for tests
            execution_timeout=600,  # Longer timeout for tests
            cpu_percent=75.0
        )
        
        return await self.execute_code(
            code_content="",  # No single code file, using file_mappings
            language=language,
            command=command,
            file_mappings=all_files,
            resource_limits=resource_limits
        )
    
    async def execute_security_scan(
        self,
        code_files: Dict[str, str],
        scan_tools: List[str] = None
    ) -> SandboxResult:
        """Execute security scanning tools in sandbox."""
        
        scan_tools = scan_tools or ["bandit", "safety", "semgrep"]
        
        # Create security scan script
        scan_script = self._create_security_scan_script(scan_tools)
        
        # Add scan script to files
        files_with_scanner = {**code_files, "security_scan.sh": scan_script}
        
        return await self.execute_code(
            code_content="",
            language="python",
            command="bash security_scan.sh",
            file_mappings=files_with_scanner,
            resource_limits=ResourceLimits(
                memory_mb=512,
                execution_timeout=900,  # Longer timeout for scans
                network_disabled=False  # May need network for tool updates
            )
        )
    
    async def execute_performance_benchmark(
        self,
        code_content: str,
        benchmark_script: str,
        iterations: int = 10
    ) -> SandboxResult:
        """Execute performance benchmarking in sandbox."""
        
        # Create benchmark runner
        benchmark_runner = f"""
import time
import psutil
import json
import sys

def run_benchmark():
    results = []
    
    for i in range({iterations}):
        # Memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute benchmark
        start_time = time.perf_counter()
        {benchmark_script}
        end_time = time.perf_counter()
        
        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        results.append({{
            'iteration': i + 1,
            'execution_time_ms': (end_time - start_time) * 1000,
            'memory_usage_mb': memory_after - memory_before,
            'peak_memory_mb': memory_after
        }})
    
    # Calculate statistics
    times = [r['execution_time_ms'] for r in results]
    memory_usage = [r['memory_usage_mb'] for r in results]
    
    stats = {{
        'avg_time_ms': sum(times) / len(times),
        'min_time_ms': min(times),
        'max_time_ms': max(times),
        'avg_memory_mb': sum(memory_usage) / len(memory_usage),
        'max_memory_mb': max(memory_usage),
        'iterations': {iterations},
        'raw_results': results
    }}
    
    print("BENCHMARK_RESULTS_START")
    print(json.dumps(stats, indent=2))
    print("BENCHMARK_RESULTS_END")

if __name__ == "__main__":
    run_benchmark()
"""
        
        files = {
            "main.py": code_content,
            "benchmark.py": benchmark_runner
        }
        
        return await self.execute_code(
            code_content="",
            command="python benchmark.py",
            file_mappings=files,
            resource_limits=ResourceLimits(
                memory_mb=1024,  # More memory for benchmarking
                execution_timeout=1800,  # Longer timeout
                cpu_percent=90.0  # More CPU for accurate benchmarks
            )
        )
    
    async def _create_workspace(
        self,
        code_content: str,
        file_mappings: Dict[str, str],
        language: str
    ) -> str:
        """Create temporary workspace with code and files."""
        
        workspace = tempfile.mkdtemp(prefix="sandbox_workspace_")
        workspace_path = Path(workspace)
        
        try:
            # Write main code file
            if code_content:
                main_file = workspace_path / self._get_main_filename(language)
                main_file.write_text(code_content, encoding="utf-8")
            
            # Write additional files
            for file_path, content in file_mappings.items():
                target_file = workspace_path / file_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(content, encoding="utf-8")
            
            # Create requirements.txt for Python if needed
            if language == "python" and any("import " in content for content in file_mappings.values()):
                self._create_requirements_file(workspace_path, file_mappings)
            
            return workspace
            
        except Exception as e:
            # Cleanup on failure
            shutil.rmtree(workspace, ignore_errors=True)
            raise e
    
    async def _create_container(
        self,
        language: str,
        workspace_path: str,
        working_dir: str,
        environment: Dict[str, str],
        resource_limits: ResourceLimits,
        security_policy: SecurityPolicy,
        execution_id: str
    ) -> Container:
        """Create and configure secure container."""
        
        image = self.base_images.get(language, self.base_images["generic"])
        
        # Container configuration
        container_config = {
            "image": image,
            "volumes": {workspace_path: {"bind": working_dir, "mode": "rw"}},
            "working_dir": working_dir,
            "environment": environment,
            "name": f"sandbox_{execution_id}",
            "labels": {"sandbox_execution": execution_id},
            "detach": True,
            "stdin_open": False,
            "tty": False,
            "remove": False,  # We'll remove manually after analysis
            
            # Resource limits
            "mem_limit": f"{resource_limits.memory_mb}m",
            "memswap_limit": f"{resource_limits.memory_mb}m",  # Disable swap
            "cpu_percent": resource_limits.cpu_percent,
            "pids_limit": resource_limits.max_processes,
            
            # Security settings
            "read_only": security_policy.read_only_filesystem,
            "user": "1000:1000",  # Non-root user
            "cap_drop": security_policy.drop_capabilities,
            "security_opt": ["no-new-privileges"] if security_policy.no_new_privileges else [],
        }
        
        # Network isolation
        if resource_limits.network_disabled:
            container_config["network_mode"] = "none"
        
        # Create tmp directory as writable
        if security_policy.read_only_filesystem:
            container_config["tmpfs"] = {"/tmp": "size=10m,exec"}
        
        try:
            container = self.docker_client.containers.create(**container_config)
            self.active_containers[execution_id] = container
            return container
            
        except Exception as e:
            logger.error("Failed to create container", error=str(e), config=container_config)
            raise
    
    async def _execute_and_monitor(
        self,
        container: Container,
        command: str,
        resource_limits: ResourceLimits,
        execution_id: str
    ) -> SandboxResult:
        """Execute command in container and monitor execution."""
        
        start_time = time.time()
        
        try:
            # Start container
            container.start()
            
            # Execute command
            exec_result = container.exec_run(
                command,
                stdout=True,
                stderr=True,
                stream=False,
                demux=True
            )
            
            stdout_bytes, stderr_bytes = exec_result.output
            stdout = stdout_bytes.decode("utf-8") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8") if stderr_bytes else ""
            
            # Wait for completion or timeout
            try:
                container.wait(timeout=resource_limits.execution_timeout)
            except Exception:
                # Container might have already exited
                pass
            
            # Get final container state
            container.reload()
            exit_code = container.attrs.get("State", {}).get("ExitCode", exec_result.exit_code)
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Create result
            result = SandboxResult(
                execution_id=execution_id,
                success=(exit_code == 0),
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time_ms=execution_time_ms,
                container_id=container.id,
                image_name=container.image.tags[0] if container.image.tags else "unknown",
                command=command,
                working_directory=container.attrs.get("Config", {}).get("WorkingDir", "")
            )
            
            # Monitor resource usage
            await self._monitor_resource_usage(container, result)
            
            return result
            
        except ContainerError as e:
            logger.error("Container execution error", execution_id=execution_id, error=str(e))
            return SandboxResult(
                execution_id=execution_id,
                success=False,
                exit_code=e.exit_status,
                stderr=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
        
        except Exception as e:
            logger.error("Execution monitoring failed", execution_id=execution_id, error=str(e))
            return SandboxResult(
                execution_id=execution_id,
                success=False,
                exit_code=-1,
                stderr=f"Monitoring error: {str(e)}",
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _monitor_resource_usage(self, container: Container, result: SandboxResult) -> None:
        """Monitor and record resource usage."""
        
        try:
            # Get container stats
            stats = container.stats(stream=False)
            
            # Memory usage
            memory_usage = stats.get("memory_usage", {})
            if "usage" in memory_usage:
                result.memory_usage_mb = memory_usage["usage"] / 1024 / 1024
            
            # CPU usage (simplified calculation)
            cpu_stats = stats.get("cpu_stats", {})
            precpu_stats = stats.get("precpu_stats", {})
            
            if cpu_stats and precpu_stats:
                cpu_delta = cpu_stats.get("cpu_usage", {}).get("total_usage", 0) - \
                           precpu_stats.get("cpu_usage", {}).get("total_usage", 0)
                system_delta = cpu_stats.get("system_cpu_usage", 0) - \
                              precpu_stats.get("system_cpu_usage", 0)
                
                if system_delta > 0:
                    num_cpus = len(cpu_stats.get("cpu_usage", {}).get("percpu_usage", [1]))
                    result.cpu_usage_percent = (cpu_delta / system_delta) * num_cpus * 100.0
            
            # Network monitoring (check for unexpected network activity)
            network_stats = stats.get("networks", {})
            total_bytes = sum(
                interface.get("rx_bytes", 0) + interface.get("tx_bytes", 0)
                for interface in network_stats.values()
            )
            
            if total_bytes > 0:
                result.network_attempts = 1
                result.security_violations.append({
                    "type": "network_activity",
                    "description": f"Unexpected network activity detected: {total_bytes} bytes",
                    "severity": "medium",
                    "bytes_transferred": total_bytes
                })
                
        except Exception as e:
            logger.warning("Failed to collect resource stats", error=str(e))
    
    async def _analyze_execution_results(
        self,
        result: SandboxResult,
        workspace_path: str,
        container: Container
    ) -> None:
        """Analyze execution results for security violations and performance metrics."""
        
        # Check for file system changes
        await self._check_filesystem_changes(result, workspace_path)
        
        # Parse performance metrics from output
        await self._parse_performance_metrics(result)
        
        # Parse test results if applicable
        await self._parse_test_results(result)
        
        # Check container logs for security issues
        await self._check_security_violations(result, container)
    
    async def _check_filesystem_changes(self, result: SandboxResult, workspace_path: str) -> None:
        """Check for unexpected filesystem changes."""
        
        try:
            workspace = Path(workspace_path)
            for file_path in workspace.rglob("*"):
                if file_path.is_file():
                    # Check for suspicious files
                    if file_path.name.startswith(".") and file_path.name not in [".gitignore", ".env"]:
                        result.security_violations.append({
                            "type": "suspicious_file",
                            "description": f"Suspicious hidden file created: {file_path.name}",
                            "severity": "low",
                            "file_path": str(file_path.relative_to(workspace))
                        })
                    
                    # Check for executable files
                    if os.access(file_path, os.X_OK) and not file_path.suffix in [".py", ".sh"]:
                        result.security_violations.append({
                            "type": "executable_file",
                            "description": f"Executable file created: {file_path.name}",
                            "severity": "medium",
                            "file_path": str(file_path.relative_to(workspace))
                        })
                        
        except Exception as e:
            logger.warning("Failed to check filesystem changes", error=str(e))
    
    async def _parse_performance_metrics(self, result: SandboxResult) -> None:
        """Parse performance metrics from execution output."""
        
        try:
            # Look for benchmark results in output
            if "BENCHMARK_RESULTS_START" in result.stdout:
                start_marker = result.stdout.find("BENCHMARK_RESULTS_START")
                end_marker = result.stdout.find("BENCHMARK_RESULTS_END")
                
                if start_marker != -1 and end_marker != -1:
                    benchmark_json = result.stdout[start_marker + len("BENCHMARK_RESULTS_START"):end_marker].strip()
                    result.performance_metrics = json.loads(benchmark_json)
                    
        except Exception as e:
            logger.warning("Failed to parse performance metrics", error=str(e))
    
    async def _parse_test_results(self, result: SandboxResult) -> None:
        """Parse test results from execution output."""
        
        try:
            # Parse pytest output
            if "pytest" in result.command.lower():
                test_summary = self._parse_pytest_output(result.stdout)
                if test_summary:
                    result.test_results = test_summary
                    
        except Exception as e:
            logger.warning("Failed to parse test results", error=str(e))
    
    def _parse_pytest_output(self, output: str) -> Optional[Dict[str, Any]]:
        """Parse pytest output for test results."""
        
        # Simple pytest output parsing
        lines = output.split("\n")
        
        results = {
            "framework": "pytest",
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "failures": []
        }
        
        for line in lines:
            # Look for test summary line
            if " passed" in line or " failed" in line:
                # Example: "5 passed, 1 failed, 2 skipped in 0.23s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        results["passed"] = int(parts[i-1])
                    elif part == "failed" and i > 0:
                        results["failed"] = int(parts[i-1])
                    elif part == "skipped" and i > 0:
                        results["skipped"] = int(parts[i-1])
                    elif part == "error" and i > 0:
                        results["errors"] = int(parts[i-1])
            
            # Collect failure details
            if "FAILED" in line:
                results["failures"].append(line)
        
        results["total_tests"] = results["passed"] + results["failed"] + results["skipped"] + results["errors"]
        
        return results if results["total_tests"] > 0 else None
    
    async def _check_security_violations(self, result: SandboxResult, container: Container) -> None:
        """Check container logs and state for security violations."""
        
        try:
            # Check container logs for suspicious activity
            logs = container.logs().decode("utf-8")
            
            # Look for common security violation patterns
            violation_patterns = [
                (r"permission denied", "permission_violation", "low"),
                (r"access denied", "access_violation", "low"),
                (r"segmentation fault", "memory_violation", "medium"),
                (r"killed", "resource_violation", "medium"),
                (r"timeout", "timeout_violation", "low")
            ]
            
            for pattern, violation_type, severity in violation_patterns:
                if re.search(pattern, logs, re.IGNORECASE):
                    result.security_violations.append({
                        "type": violation_type,
                        "description": f"Detected {violation_type.replace('_', ' ')} in logs",
                        "severity": severity,
                        "pattern": pattern
                    })
                    
        except Exception as e:
            logger.warning("Failed to check security violations", error=str(e))
    
    async def _cleanup_execution(
        self,
        container: Optional[Container],
        workspace_path: Optional[str],
        execution_id: str
    ) -> None:
        """Clean up container and workspace after execution."""
        
        # Remove container
        if container:
            try:
                if container.status != "exited":
                    container.kill()
                container.remove(force=True)
                
                if execution_id in self.active_containers:
                    del self.active_containers[execution_id]
                    
            except Exception as e:
                logger.warning("Failed to remove container", execution_id=execution_id, error=str(e))
        
        # Remove workspace
        if workspace_path and os.path.exists(workspace_path):
            try:
                shutil.rmtree(workspace_path)
            except Exception as e:
                logger.warning("Failed to remove workspace", workspace_path=workspace_path, error=str(e))
    
    def _ensure_base_images(self) -> None:
        """Ensure base Docker images are available."""
        
        for language, image in self.base_images.items():
            try:
                self.docker_client.images.get(image)
                logger.debug("Base image available", language=language, image=image)
            except ImageNotFound:
                logger.info("Pulling base image", language=language, image=image)
                try:
                    self.docker_client.images.pull(image)
                except Exception as e:
                    logger.error("Failed to pull base image", image=image, error=str(e))
    
    def _get_main_filename(self, language: str) -> str:
        """Get main filename for language."""
        filenames = {
            "python": "main.py",
            "node": "main.js",
            "generic": "main.txt"
        }
        return filenames.get(language, "main.txt")
    
    def _get_default_command(self, language: str) -> str:
        """Get default execution command for language."""
        commands = {
            "python": "python main.py",
            "node": "node main.js",
            "generic": "cat main.txt"
        }
        return commands.get(language, "echo 'No default command'")
    
    def _create_requirements_file(self, workspace_path: Path, file_mappings: Dict[str, str]) -> None:
        """Create requirements.txt file based on imports."""
        
        # Simple import detection (could be improved)
        imports = set()
        
        for content in file_mappings.values():
            # Find import statements
            import_lines = [line.strip() for line in content.split("\n") if line.strip().startswith("import ") or line.strip().startswith("from ")]
            
            for line in import_lines:
                if line.startswith("import "):
                    module = line.split()[1].split(".")[0]
                    imports.add(module)
                elif line.startswith("from "):
                    module = line.split()[1].split(".")[0]
                    imports.add(module)
        
        # Filter out standard library modules (simplified)
        stdlib_modules = {"os", "sys", "json", "time", "datetime", "re", "math", "random", "collections"}
        external_imports = imports - stdlib_modules
        
        if external_imports:
            requirements_content = "\n".join(sorted(external_imports))
            (workspace_path / "requirements.txt").write_text(requirements_content)
    
    def _create_security_scan_script(self, tools: List[str]) -> str:
        """Create security scanning script."""
        
        script_parts = ["#!/bin/bash", "set -e", ""]
        
        if "bandit" in tools:
            script_parts.extend([
                "echo 'Running Bandit security scan...'",
                "pip install bandit > /dev/null 2>&1",
                "bandit -r . -f json -o bandit_results.json || true",
                "cat bandit_results.json",
                ""
            ])
        
        if "safety" in tools:
            script_parts.extend([
                "echo 'Running Safety dependency scan...'", 
                "pip install safety > /dev/null 2>&1",
                "safety check --json || true",
                ""
            ])
        
        if "semgrep" in tools:
            script_parts.extend([
                "echo 'Running Semgrep static analysis...'",
                "pip install semgrep > /dev/null 2>&1", 
                "semgrep --config=auto --json . || true",
                ""
            ])
        
        return "\n".join(script_parts)
    
    async def cleanup_all(self) -> None:
        """Clean up all active containers."""
        
        for execution_id, container in list(self.active_containers.items()):
            await self._cleanup_execution(container, None, execution_id)


# Export main class and result
__all__ = ["SandboxEnvironment", "SandboxResult", "ResourceLimits", "SecurityPolicy"]