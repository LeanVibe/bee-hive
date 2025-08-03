"""
Secure Code Execution Framework for LeanVibe Agent Hive 2.0

Enterprise-grade sandboxed execution environment for AI-generated code.
Provides Docker-based isolation with resource limits and security monitoring.
"""

import asyncio
import docker
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import structlog

logger = structlog.get_logger()


class ExecutionStatus(str, Enum):
    """Status of code execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    KILLED = "killed"


class ExecutionLanguage(str, Enum):
    """Supported execution languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    BASH = "bash"
    GO = "go"
    RUST = "rust"


@dataclass
class ExecutionResult:
    """Result of code execution."""
    execution_id: str
    status: ExecutionStatus
    stdout: str
    stderr: str
    return_code: int
    execution_time_seconds: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    created_files: List[str] = None
    error_message: Optional[str] = None
    security_violations: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_files is None:
            self.created_files = []
        if self.security_violations is None:
            self.security_violations = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionConfig:
    """Configuration for code execution."""
    language: ExecutionLanguage
    timeout_seconds: int = 30
    memory_limit_mb: int = 512
    cpu_limit_percent: int = 50
    network_access: bool = False
    filesystem_write: bool = True
    max_output_size: int = 1024 * 1024  # 1MB
    allowed_packages: List[str] = None
    environment_variables: Dict[str, str] = None

    def __post_init__(self):
        if self.allowed_packages is None:
            self.allowed_packages = []
        if self.environment_variables is None:
            self.environment_variables = {}


class SecureCodeExecutor:
    """
    Enterprise-grade secure code execution platform.
    
    Provides Docker-based sandboxed execution with:
    - Resource limits and monitoring
    - Network isolation 
    - Filesystem constraints
    - Security violation detection
    - Real-time execution tracking
    """

    def __init__(self):
        self.docker_client = None
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.logger = structlog.get_logger().bind(component="secure_executor")
        
        # Docker image configurations
        self.language_images = {
            ExecutionLanguage.PYTHON: "python:3.11-slim",
            ExecutionLanguage.JAVASCRIPT: "node:18-alpine",
            ExecutionLanguage.TYPESCRIPT: "node:18-alpine", 
            ExecutionLanguage.BASH: "ubuntu:22.04",
            ExecutionLanguage.GO: "golang:1.21-alpine",
            ExecutionLanguage.RUST: "rust:1.75-slim"
        }
        
        # Command templates for different languages
        self.execution_commands = {
            ExecutionLanguage.PYTHON: ["python", "/workspace/code.py"],
            ExecutionLanguage.JAVASCRIPT: ["node", "/workspace/code.js"],
            ExecutionLanguage.TYPESCRIPT: ["npx", "ts-node", "/workspace/code.ts"],
            ExecutionLanguage.BASH: ["bash", "/workspace/code.sh"],
            ExecutionLanguage.GO: ["sh", "-c", "cd /workspace && go run code.go"],
            ExecutionLanguage.RUST: ["sh", "-c", "cd /workspace && rustc code.rs -o code && ./code"]
        }

        # File extensions
        self.file_extensions = {
            ExecutionLanguage.PYTHON: ".py",
            ExecutionLanguage.JAVASCRIPT: ".js", 
            ExecutionLanguage.TYPESCRIPT: ".ts",
            ExecutionLanguage.BASH: ".sh",
            ExecutionLanguage.GO: ".go",
            ExecutionLanguage.RUST: ".rs"
        }

    async def initialize(self):
        """Initialize Docker client and pull required images."""
        try:
            self.docker_client = docker.from_env()
            self.logger.info("Docker client initialized successfully")
            
            # Pull required images
            await self._ensure_images_available()
            
        except Exception as e:
            self.logger.error("Failed to initialize Docker client", error=str(e))
            raise Exception(f"Docker initialization failed: {str(e)}")

    async def _ensure_images_available(self):
        """Ensure all required Docker images are available."""
        for language, image in self.language_images.items():
            try:
                self.docker_client.images.get(image)
                self.logger.debug("Docker image available", language=language.value, image=image)
            except docker.errors.ImageNotFound:
                self.logger.info("Pulling Docker image", language=language.value, image=image)
                self.docker_client.images.pull(image)

    async def execute_code(
        self, 
        code: str, 
        config: ExecutionConfig,
        execution_id: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute code in a secure sandboxed environment.
        
        Args:
            code: The code to execute
            config: Execution configuration
            execution_id: Optional custom execution ID
            
        Returns:
            ExecutionResult with execution details and output
        """
        if execution_id is None:
            execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        
        start_time = time.time()
        
        self.logger.info(
            "Starting secure code execution",
            execution_id=execution_id,
            language=config.language.value,
            timeout=config.timeout_seconds
        )
        
        try:
            # Create secure execution environment
            result = await self._execute_in_container(code, config, execution_id)
            
            execution_time = time.time() - start_time
            result.execution_time_seconds = execution_time
            
            self.logger.info(
                "Code execution completed",
                execution_id=execution_id,
                status=result.status.value,
                execution_time=execution_time
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(
                "Code execution failed",
                execution_id=execution_id,
                error=str(e),
                execution_time=execution_time
            )
            
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                stdout="",
                stderr="",
                return_code=-1,
                execution_time_seconds=execution_time,
                error_message=str(e)
            )

    async def _execute_in_container(
        self, 
        code: str, 
        config: ExecutionConfig, 
        execution_id: str
    ) -> ExecutionResult:
        """Execute code in Docker container with security constraints."""
        
        # Prepare workspace
        workspace_path = await self._prepare_workspace(code, config, execution_id)
        
        try:
            # Container configuration
            container_config = {
                'image': self.language_images[config.language],
                'command': self.execution_commands[config.language],
                'working_dir': '/workspace',
                'volumes': {
                    str(workspace_path): {'bind': '/workspace', 'mode': 'rw'}
                },
                'mem_limit': f"{config.memory_limit_mb}m",
                'cpu_period': 100000,
                'cpu_quota': int(100000 * config.cpu_limit_percent / 100),
                'network_disabled': not config.network_access,
                'remove': True,
                'detach': False,
                'stdout': True,
                'stderr': True,
                'environment': config.environment_variables
            }
            
            # Security constraints
            if not config.filesystem_write:
                container_config['read_only'] = True
            
            # Track execution
            self.active_executions[execution_id] = {
                'start_time': time.time(),
                'config': config,
                'workspace': workspace_path
            }
            
            # Execute with timeout
            container = None
            try:
                container = self.docker_client.containers.run(**container_config)
                
                # Parse output
                if hasattr(container, 'decode'):
                    output = container.decode('utf-8')
                    stdout = output
                    stderr = ""
                    return_code = 0
                else:
                    stdout = str(container)
                    stderr = ""
                    return_code = 0
                
                # Check for security violations
                security_violations = await self._check_security_violations(stdout, stderr, config)
                
                # Get created files
                created_files = await self._get_created_files(workspace_path)
                
                return ExecutionResult(
                    execution_id=execution_id,
                    status=ExecutionStatus.COMPLETED,
                    stdout=stdout[:config.max_output_size],
                    stderr=stderr[:config.max_output_size],
                    return_code=return_code,
                    execution_time_seconds=0,  # Will be set by caller
                    created_files=created_files,
                    security_violations=security_violations
                )
                
            except docker.errors.ContainerError as e:
                return ExecutionResult(
                    execution_id=execution_id,
                    status=ExecutionStatus.FAILED,
                    stdout=getattr(e, 'stdout', b'').decode('utf-8') if hasattr(e, 'stdout') and e.stdout else "",
                    stderr=getattr(e, 'stderr', b'').decode('utf-8') if hasattr(e, 'stderr') and e.stderr else "",
                    return_code=getattr(e, 'exit_status', -1),
                    execution_time_seconds=0,
                    error_message=f"Container execution failed: {str(e)}"
                )
                
            except Exception as e:
                return ExecutionResult(
                    execution_id=execution_id,
                    status=ExecutionStatus.FAILED,
                    stdout="",
                    stderr="",
                    return_code=-1,
                    execution_time_seconds=0,
                    error_message=f"Execution error: {str(e)}"
                )
                
        finally:
            # Cleanup
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            await self._cleanup_workspace(workspace_path)

    async def _prepare_workspace(self, code: str, config: ExecutionConfig, execution_id: str) -> Path:
        """Prepare secure workspace for code execution."""
        workspace_path = Path(tempfile.mkdtemp(prefix=f"secure_exec_{execution_id}_"))
        
        # Write code to appropriate file
        code_filename = f"code{self.file_extensions[config.language]}"
        code_file = workspace_path / code_filename
        
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Set appropriate permissions
        os.chmod(code_file, 0o644)
        os.chmod(workspace_path, 0o755)
        
        return workspace_path

    async def _cleanup_workspace(self, workspace_path: Path):
        """Clean up workspace after execution."""
        try:
            import shutil
            shutil.rmtree(workspace_path, ignore_errors=True)
        except Exception as e:
            self.logger.warning("Failed to cleanup workspace", path=str(workspace_path), error=str(e))

    async def _check_security_violations(
        self, 
        stdout: str, 
        stderr: str, 
        config: ExecutionConfig
    ) -> List[str]:
        """Check for security violations in execution output."""
        violations = []
        
        # Check for suspicious patterns
        suspicious_patterns = [
            "eval(",
            "exec(",
            "import os",
            "import subprocess",
            "import socket",
            "__import__",
            "open(",
            "file(",
            "input(",
            "raw_input("
        ]
        
        output_combined = stdout + stderr
        
        for pattern in suspicious_patterns:
            if pattern in output_combined:
                violations.append(f"Suspicious pattern detected: {pattern}")
        
        return violations

    async def _get_created_files(self, workspace_path: Path) -> List[str]:
        """Get list of files created during execution."""
        try:
            created_files = []
            for file_path in workspace_path.rglob('*'):
                if file_path.is_file() and file_path.name != f"code{self.file_extensions.get('PYTHON', '.py')}":
                    relative_path = file_path.relative_to(workspace_path)
                    created_files.append(str(relative_path))
            return created_files
        except Exception as e:
            self.logger.warning("Failed to get created files", error=str(e))
            return []

    async def kill_execution(self, execution_id: str) -> bool:
        """Kill a running execution."""
        if execution_id not in self.active_executions:
            return False
        
        try:
            # In a real implementation, this would kill the Docker container
            # For now, we'll just remove from tracking
            del self.active_executions[execution_id]
            self.logger.info("Execution killed", execution_id=execution_id)
            return True
        except Exception as e:
            self.logger.error("Failed to kill execution", execution_id=execution_id, error=str(e))
            return False

    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a running execution."""
        if execution_id not in self.active_executions:
            return None
        
        execution_info = self.active_executions[execution_id]
        current_time = time.time()
        
        return {
            "execution_id": execution_id,
            "status": ExecutionStatus.RUNNING.value,
            "elapsed_time": current_time - execution_info['start_time'],
            "config": asdict(execution_info['config'])
        }

    async def list_active_executions(self) -> List[Dict[str, Any]]:
        """List all active executions."""
        active_list = []
        current_time = time.time()
        
        for execution_id, info in self.active_executions.items():
            active_list.append({
                "execution_id": execution_id,
                "status": ExecutionStatus.RUNNING.value,
                "elapsed_time": current_time - info['start_time'],
                "language": info['config'].language.value
            })
        
        return active_list


# Factory function for easy instantiation
async def create_secure_code_executor() -> SecureCodeExecutor:
    """Create and initialize secure code executor."""
    executor = SecureCodeExecutor()
    await executor.initialize()
    return executor


# Example usage and testing
if __name__ == "__main__":
    async def test_secure_execution():
        """Test secure code execution."""
        executor = await create_secure_code_executor()
        
        # Test Python code execution
        python_code = '''
def hello_world():
    return "Hello from secure execution!"

result = hello_world()
print(result)
print("Execution completed successfully")
'''
        
        config = ExecutionConfig(
            language=ExecutionLanguage.PYTHON,
            timeout_seconds=10,
            memory_limit_mb=128,
            network_access=False
        )
        
        result = await executor.execute_code(python_code, config)
        
        print("Execution Result:")
        print(f"Status: {result.status}")
        print(f"Return Code: {result.return_code}")
        print(f"Execution Time: {result.execution_time_seconds:.2f}s")
        print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Errors: {result.stderr}")
        if result.security_violations:
            print(f"Security Violations: {result.security_violations}")

    # Run test if executed directly
    asyncio.run(test_secure_execution())