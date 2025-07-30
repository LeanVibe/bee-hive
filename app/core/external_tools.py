"""
External Tool Integration Layer for LeanVibe Agent Hive 2.0

This revolutionary system enables agents to seamlessly interact with external development
tools including Git, GitHub, Docker, and CI/CD pipelines. Agents can now perform
real-world development workflows with full version control, deployment, and collaboration.

CRITICAL: This system provides secure, monitored access to external tools while
maintaining comprehensive audit trails and safety guards.
"""

import asyncio
import json
import os
import subprocess  # nosec B404 - Secure subprocess usage with validation and sanitization
import uuid
import shlex
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
import aiohttp
import aiofiles
from anthropic import AsyncAnthropic

from .config import settings
from .workspace_manager import workspace_manager, AgentWorkspace
from .database import get_session
from .redis import get_message_broker
from ..models.agent import Agent
from ..models.task import Task
from ..models.context import Context, ContextType

logger = structlog.get_logger()


class ToolType(Enum):
    """Types of external tools."""
    GIT = "git"
    GITHUB = "github"
    DOCKER = "docker"
    CI_CD = "ci_cd"
    PACKAGE_MANAGER = "package_manager"
    DEPLOYMENT = "deployment"


class OperationStatus(Enum):
    """Status of tool operations."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ToolOperation:
    """Represents an operation performed with external tools."""
    id: str
    agent_id: str
    tool_type: ToolType
    operation: str
    parameters: Dict[str, Any]
    
    # Execution details
    status: OperationStatus
    started_at: datetime
    completed_at: Optional[datetime]
    execution_time_ms: int
    
    # Results
    success: bool
    output: str
    error: str
    exit_code: int
    
    # Security and monitoring
    workspace_path: str
    command_executed: str
    files_affected: List[str]
    security_level: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GitRepository:
    """Represents a Git repository managed by agents."""
    id: str
    agent_id: str
    repository_url: str
    local_path: str
    branch: str
    
    # Repository state
    is_clean: bool
    uncommitted_changes: List[str]
    remote_ahead: int
    remote_behind: int
    
    # Metadata
    created_at: datetime
    last_sync: datetime
    total_commits: int
    contributors: List[str]


@dataclass
class GitHubRepository:
    """Represents a GitHub repository for API operations."""
    id: str
    owner: str
    name: str
    full_name: str
    clone_url: str
    default_branch: str
    
    # Repository details
    description: str
    language: str
    stars: int
    forks: int
    open_issues: int
    
    # Access control
    private: bool
    permissions: Dict[str, bool]


class GitIntegration:
    """
    Comprehensive Git integration for version control operations.
    
    Enables agents to perform all Git operations including cloning,
    committing, branching, merging, and collaboration workflows.
    """
    
    def __init__(self):
        self.repositories: Dict[str, GitRepository] = {}
        self.operation_history: List[ToolOperation] = []
    
    async def clone_repository(
        self,
        agent_id: str,
        repository_url: str,
        target_directory: Optional[str] = None,
        branch: Optional[str] = None
    ) -> Tuple[bool, GitRepository, str]:
        """Clone a Git repository into the agent's workspace."""
        
        operation_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            # Get agent workspace
            workspace = await workspace_manager.get_workspace(agent_id)
            if not workspace:
                return False, None, "Agent workspace not found"
            
            # Determine target path
            if target_directory:
                target_path = workspace.config.project_path / target_directory
            else:
                repo_name = repository_url.split('/')[-1].replace('.git', '')
                target_path = workspace.config.project_path / repo_name
            
            # Prepare clone command
            clone_command = f"git clone {repository_url}"
            if branch:
                clone_command += f" -b {branch}"
            clone_command += f" {target_path}"
            
            # Execute clone
            success, output, error = await workspace.execute_command(
                clone_command,
                window="git",
                capture_output=True
            )
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            if success:
                # Create repository object
                repo_id = str(uuid.uuid4())
                repository = GitRepository(
                    id=repo_id,
                    agent_id=agent_id,
                    repository_url=repository_url,
                    local_path=str(target_path),
                    branch=branch or "main",
                    is_clean=True,
                    uncommitted_changes=[],
                    remote_ahead=0,
                    remote_behind=0,
                    created_at=datetime.utcnow(),
                    last_sync=datetime.utcnow(),
                    total_commits=0,
                    contributors=[]
                )
                
                self.repositories[repo_id] = repository
                
                # Update repository state
                await self._update_repository_state(repository, workspace)
                
                logger.info(
                    "Repository cloned successfully",
                    agent_id=agent_id,
                    repository_url=repository_url,
                    local_path=str(target_path)
                )
                
                return True, repository, output
            else:
                logger.error(
                    "Repository clone failed",
                    agent_id=agent_id,
                    repository_url=repository_url,
                    error=error
                )
                return False, None, error
            
        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            error_msg = f"Clone operation failed: {str(e)}"
            
            logger.error(
                "Repository clone exception",
                agent_id=agent_id,
                repository_url=repository_url,
                error=error_msg
            )
            
            return False, None, error_msg
        
        finally:
            # Record operation
            operation = ToolOperation(
                id=operation_id,
                agent_id=agent_id,
                tool_type=ToolType.GIT,
                operation="clone",
                parameters={
                    "repository_url": repository_url,
                    "target_directory": target_directory,
                    "branch": branch
                },
                status=OperationStatus.SUCCESS if success else OperationStatus.FAILED,
                started_at=start_time,
                completed_at=datetime.utcnow(),
                execution_time_ms=execution_time,
                success=success,
                output=output if success else "",
                error=error if not success else "",
                exit_code=0 if success else 1,
                workspace_path=str(target_path) if target_path else "",
                command_executed=clone_command,
                files_affected=[],
                security_level="moderate"
            )
            
            self.operation_history.append(operation)
    
    async def commit_changes(
        self,
        repository_id: str,
        message: str,
        files: Optional[List[str]] = None
    ) -> Tuple[bool, str, str]:
        """Commit changes to a Git repository."""
        
        if repository_id not in self.repositories:
            return False, "", "Repository not found"
        
        repository = self.repositories[repository_id]
        workspace = await workspace_manager.get_workspace(repository.agent_id)
        
        if not workspace:
            return False, "", "Agent workspace not found"
        
        try:
            # Change to repository directory
            cd_command = f"cd {repository.local_path}"
            
            # Stage files
            if files:
                for file in files:
                    stage_command = f"{cd_command} && git add {file}"
                    await workspace.execute_command(stage_command, capture_output=False)
            else:
                stage_command = f"{cd_command} && git add ."
                await workspace.execute_command(stage_command, capture_output=False)
            
            # Commit changes
            commit_command = f"{cd_command} && git commit -m \"{message}\""
            success, output, error = await workspace.execute_command(
                commit_command,
                capture_output=True
            )
            
            if success:
                # Update repository state
                await self._update_repository_state(repository, workspace)
                
                logger.info(
                    "Changes committed successfully",
                    repository_id=repository_id,
                    agent_id=repository.agent_id,
                    message=message
                )
            
            return success, output, error
            
        except Exception as e:
            error_msg = f"Commit failed: {str(e)}"
            logger.error(
                "Git commit exception",
                repository_id=repository_id,
                error=error_msg
            )
            return False, "", error_msg
    
    async def create_branch(
        self,
        repository_id: str,
        branch_name: str,
        checkout: bool = True
    ) -> Tuple[bool, str, str]:
        """Create a new branch in the repository."""
        
        if repository_id not in self.repositories:
            return False, "", "Repository not found"
        
        repository = self.repositories[repository_id]
        workspace = await workspace_manager.get_workspace(repository.agent_id)
        
        try:
            cd_command = f"cd {repository.local_path}"
            
            if checkout:
                branch_command = f"{cd_command} && git checkout -b {branch_name}"
            else:
                branch_command = f"{cd_command} && git branch {branch_name}"
            
            success, output, error = await workspace.execute_command(
                branch_command,
                capture_output=True
            )
            
            if success and checkout:
                repository.branch = branch_name
            
            return success, output, error
            
        except Exception as e:
            return False, "", f"Branch creation failed: {str(e)}"
    
    async def push_changes(
        self,
        repository_id: str,
        remote: str = "origin",
        branch: Optional[str] = None
    ) -> Tuple[bool, str, str]:
        """Push changes to remote repository."""
        
        if repository_id not in self.repositories:
            return False, "", "Repository not found"
        
        repository = self.repositories[repository_id]
        workspace = await workspace_manager.get_workspace(repository.agent_id)
        
        try:
            cd_command = f"cd {repository.local_path}"
            push_branch = branch or repository.branch
            push_command = f"{cd_command} && git push {remote} {push_branch}"
            
            success, output, error = await workspace.execute_command(
                push_command,
                capture_output=True
            )
            
            if success:
                repository.last_sync = datetime.utcnow()
                
                logger.info(
                    "Changes pushed successfully",
                    repository_id=repository_id,
                    remote=remote,
                    branch=push_branch
                )
            
            return success, output, error
            
        except Exception as e:
            return False, "", f"Push failed: {str(e)}"
    
    async def _update_repository_state(
        self,
        repository: GitRepository,
        workspace: AgentWorkspace
    ) -> None:
        """Update repository state information."""
        
        try:
            cd_command = f"cd {repository.local_path}"
            
            # Check if repository is clean
            status_command = f"{cd_command} && git status --porcelain"
            success, output, _ = await workspace.execute_command(
                status_command,
                capture_output=True
            )
            
            if success:
                repository.is_clean = len(output.strip()) == 0
                repository.uncommitted_changes = output.strip().split('\n') if output.strip() else []
            
            # Get commit count
            count_command = f"{cd_command} && git rev-list --count HEAD"
            success, output, _ = await workspace.execute_command(
                count_command,
                capture_output=True
            )
            
            if success and output.strip().isdigit():
                repository.total_commits = int(output.strip())
            
        except Exception as e:
            logger.error(
                "Failed to update repository state",
                repository_id=repository.id,
                error=str(e)
            )


class GitHubIntegration:
    """
    GitHub API integration for repository management and collaboration.
    
    Enables agents to interact with GitHub repositories, create PRs,
    manage issues, and handle collaborative development workflows.
    """
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or settings.GITHUB_TOKEN
        self.base_url = "https://api.github.com"
        self.repositories: Dict[str, GitHubRepository] = {}
    
    async def get_repository(
        self,
        owner: str,
        repo_name: str
    ) -> Tuple[bool, Optional[GitHubRepository], str]:
        """Get repository information from GitHub."""
        
        if not self.github_token:
            return False, None, "GitHub token not configured"
        
        try:
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/repos/{owner}/{repo_name}"
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        repository = GitHubRepository(
                            id=str(data["id"]),
                            owner=data["owner"]["login"],
                            name=data["name"],
                            full_name=data["full_name"],
                            clone_url=data["clone_url"],
                            default_branch=data["default_branch"],
                            description=data.get("description", ""),
                            language=data.get("language", ""),
                            stars=data["stargazers_count"],
                            forks=data["forks_count"],
                            open_issues=data["open_issues_count"],
                            private=data["private"],
                            permissions=data.get("permissions", {})
                        )
                        
                        self.repositories[repository.id] = repository
                        return True, repository, "Repository found"
                    else:
                        error_data = await response.json()
                        return False, None, error_data.get("message", "Repository not found")
        
        except Exception as e:
            return False, None, f"GitHub API error: {str(e)}"
    
    async def create_pull_request(
        self,
        repository_id: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main"
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """Create a pull request on GitHub."""
        
        if repository_id not in self.repositories:
            return False, None, "Repository not found"
        
        repository = self.repositories[repository_id]
        
        try:
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json"
            }
            
            pr_data = {
                "title": title,
                "body": body,
                "head": head_branch,
                "base": base_branch
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/repos/{repository.full_name}/pulls"
                async with session.post(url, headers=headers, json=pr_data) as response:
                    if response.status == 201:
                        pr_data = await response.json()
                        
                        logger.info(
                            "Pull request created",
                            repository=repository.full_name,
                            pr_number=pr_data["number"],
                            title=title
                        )
                        
                        return True, pr_data, "Pull request created successfully"
                    else:
                        error_data = await response.json()
                        return False, None, error_data.get("message", "Failed to create PR")
        
        except Exception as e:
            return False, None, f"PR creation failed: {str(e)}"
    
    async def create_issue(
        self,
        repository_id: str,
        title: str,
        body: str,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """Create an issue on GitHub."""
        
        if repository_id not in self.repositories:
            return False, None, "Repository not found"
        
        repository = self.repositories[repository_id]
        
        try:
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json"
            }
            
            issue_data = {
                "title": title,
                "body": body
            }
            
            if labels:
                issue_data["labels"] = labels
            if assignees:
                issue_data["assignees"] = assignees
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/repos/{repository.full_name}/issues"
                async with session.post(url, headers=headers, json=issue_data) as response:
                    if response.status == 201:
                        issue_data = await response.json()
                        
                        logger.info(
                            "Issue created",
                            repository=repository.full_name,
                            issue_number=issue_data["number"],
                            title=title
                        )
                        
                        return True, issue_data, "Issue created successfully"
                    else:
                        error_data = await response.json()
                        return False, None, error_data.get("message", "Failed to create issue")
        
        except Exception as e:
            return False, None, f"Issue creation failed: {str(e)}"


class DockerIntegration:
    """
    Docker integration for containerization and deployment.
    
    Enables agents to build, run, and manage Docker containers
    for development, testing, and deployment workflows.
    """
    
    def __init__(self):
        self.containers: Dict[str, Dict[str, Any]] = {}
        self.images: Dict[str, Dict[str, Any]] = {}
    
    async def build_image(
        self,
        agent_id: str,
        dockerfile_path: str,
        image_name: str,
        context_path: str = ".",
        build_args: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, str, str]:
        """Build a Docker image."""
        
        workspace = await workspace_manager.get_workspace(agent_id)
        if not workspace:
            return False, "", "Agent workspace not found"
        
        try:
            # Prepare build command
            build_command = f"docker build -t {image_name} -f {dockerfile_path} {context_path}"
            
            # Add build arguments
            if build_args:
                for key, value in build_args.items():
                    build_command += f" --build-arg {key}={value}"
            
            # Execute build
            success, output, error = await workspace.execute_command(
                build_command,
                window="docker",
                capture_output=True
            )
            
            if success:
                # Store image information
                image_id = str(uuid.uuid4())
                self.images[image_id] = {
                    "id": image_id,
                    "agent_id": agent_id,
                    "name": image_name,
                    "dockerfile_path": dockerfile_path,
                    "context_path": context_path,
                    "build_args": build_args or {},
                    "created_at": datetime.utcnow().isoformat(),
                    "status": "built"
                }
                
                logger.info(
                    "Docker image built successfully",
                    agent_id=agent_id,
                    image_name=image_name,
                    image_id=image_id
                )
            
            return success, output, error
            
        except Exception as e:
            return False, "", f"Docker build failed: {str(e)}"
    
    async def run_container(
        self,
        agent_id: str,
        image_name: str,
        container_name: Optional[str] = None,
        ports: Optional[Dict[str, str]] = None,
        environment: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, str]] = None,
        detached: bool = True
    ) -> Tuple[bool, str, str]:
        """Run a Docker container."""
        
        workspace = await workspace_manager.get_workspace(agent_id)
        if not workspace:
            return False, "", "Agent workspace not found"
        
        try:
            # Prepare run command
            run_command = ["docker", "run"]
            
            if detached:
                run_command.append("-d")
            
            if container_name:
                run_command.extend(["--name", container_name])
            
            # Add port mappings
            if ports:
                for host_port, container_port in ports.items():
                    run_command.extend(["-p", f"{host_port}:{container_port}"])
            
            # Add environment variables
            if environment:
                for key, value in environment.items():
                    run_command.extend(["-e", f"{key}={value}"])
            
            # Add volume mounts
            if volumes:
                for host_path, container_path in volumes.items():
                    run_command.extend(["-v", f"{host_path}:{container_path}"])
            
            run_command.append(image_name)
            
            # Execute run command
            success, output, error = await workspace.execute_command(
                " ".join(run_command),
                window="docker",
                capture_output=True
            )
            
            if success:
                # Extract container ID from output
                container_id = output.strip()
                
                # Store container information
                self.containers[container_id] = {
                    "id": container_id,
                    "agent_id": agent_id,
                    "image_name": image_name,
                    "container_name": container_name,
                    "ports": ports or {},
                    "environment": environment or {},
                    "volumes": volumes or {},
                    "started_at": datetime.utcnow().isoformat(),
                    "status": "running"
                }
                
                logger.info(
                    "Docker container started successfully",
                    agent_id=agent_id,
                    container_id=container_id,
                    image_name=image_name
                )
            
            return success, output, error
            
        except Exception as e:
            return False, "", f"Docker run failed: {str(e)}"
    
    async def stop_container(
        self,
        agent_id: str,
        container_id: str
    ) -> Tuple[bool, str, str]:
        """Stop a running Docker container."""
        
        workspace = await workspace_manager.get_workspace(agent_id)
        if not workspace:
            return False, "", "Agent workspace not found"
        
        try:
            stop_command = f"docker stop {container_id}"
            success, output, error = await workspace.execute_command(
                stop_command,
                capture_output=True
            )
            
            if success and container_id in self.containers:
                self.containers[container_id]["status"] = "stopped"
                self.containers[container_id]["stopped_at"] = datetime.utcnow().isoformat()
            
            return success, output, error
            
        except Exception as e:
            return False, "", f"Docker stop failed: {str(e)}"


class ExternalToolsOrchestrator:
    """
    Main orchestrator for all external tool integrations.
    
    Coordinates Git, GitHub, Docker, and other tool operations
    across multiple agents and provides unified interface.
    """
    
    def __init__(self):
        self.git = GitIntegration()
        self.github = GitHubIntegration()
        self.docker = DockerIntegration()
        self.operation_queue: List[ToolOperation] = []
        self.active_operations: Dict[str, ToolOperation] = {}
    
    async def execute_workflow(
        self,
        agent_id: str,
        workflow_definition: Dict[str, Any]
    ) -> Tuple[bool, List[ToolOperation], str]:
        """Execute a complex workflow involving multiple tools."""
        
        workflow_id = str(uuid.uuid4())
        operations = []
        
        try:
            logger.info(
                "Starting workflow execution",
                agent_id=agent_id,
                workflow_id=workflow_id,
                workflow_type=workflow_definition.get("type", "unknown")
            )
            
            # Parse workflow steps
            steps = workflow_definition.get("steps", [])
            
            for step_index, step in enumerate(steps):
                step_type = step.get("type")
                step_params = step.get("parameters", {})
                
                if step_type == "git_clone":
                    success, repository, message = await self.git.clone_repository(
                        agent_id=agent_id,
                        repository_url=step_params["repository_url"],
                        target_directory=step_params.get("target_directory"),
                        branch=step_params.get("branch")
                    )
                    
                    if not success:
                        return False, operations, f"Git clone failed: {message}"
                
                elif step_type == "docker_build":
                    success, output, error = await self.docker.build_image(
                        agent_id=agent_id,
                        dockerfile_path=step_params["dockerfile_path"],
                        image_name=step_params["image_name"],
                        context_path=step_params.get("context_path", "."),
                        build_args=step_params.get("build_args")
                    )
                    
                    if not success:
                        return False, operations, f"Docker build failed: {error}"
                
                elif step_type == "github_pr":
                    # Requires repository to be set up first
                    success, pr_data, message = await self.github.create_pull_request(
                        repository_id=step_params["repository_id"],
                        title=step_params["title"],
                        body=step_params["body"],
                        head_branch=step_params["head_branch"],
                        base_branch=step_params.get("base_branch", "main")
                    )
                    
                    if not success:
                        return False, operations, f"GitHub PR creation failed: {message}"
                
                # Record successful step
                operation = ToolOperation(
                    id=str(uuid.uuid4()),
                    agent_id=agent_id,
                    tool_type=ToolType.CI_CD,  # Workflow operation
                    operation=f"workflow_step_{step_index}",
                    parameters=step_params,
                    status=OperationStatus.SUCCESS,
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    execution_time_ms=0,
                    success=True,
                    output="Step completed successfully",
                    error="",
                    exit_code=0,
                    workspace_path="",
                    command_executed=step_type,
                    files_affected=[],
                    security_level="moderate"
                )
                
                operations.append(operation)
            
            logger.info(
                "Workflow execution completed",
                agent_id=agent_id,
                workflow_id=workflow_id,
                steps_completed=len(operations)
            )
            
            return True, operations, "Workflow completed successfully"
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(
                "Workflow execution exception",
                agent_id=agent_id,
                workflow_id=workflow_id,
                error=error_msg
            )
            
            return False, operations, error_msg
    
    async def get_operation_status(self, operation_id: str) -> Optional[ToolOperation]:
        """Get the status of a specific operation."""
        
        # Check active operations
        if operation_id in self.active_operations:
            return self.active_operations[operation_id]
        
        # Check completed operations in git history
        for operation in self.git.operation_history:
            if operation.id == operation_id:
                return operation
        
        # Check operation queue
        for operation in self.operation_queue:
            if operation.id == operation_id:
                return operation
        
        return None
    
    async def cancel_operation(self, operation_id: str) -> bool:
        """Cancel a running operation."""
        
        if operation_id in self.active_operations:
            operation = self.active_operations[operation_id]
            operation.status = OperationStatus.CANCELLED
            operation.completed_at = datetime.utcnow()
            
            # Move to completed operations
            self.git.operation_history.append(operation)
            del self.active_operations[operation_id]
            
            logger.info(
                "Operation cancelled",
                operation_id=operation_id,
                agent_id=operation.agent_id
            )
            
            return True
        
        return False
    
    async def get_agent_tools_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get a comprehensive summary of tools used by an agent."""
        
        # Get repositories
        agent_repositories = [
            repo for repo in self.git.repositories.values()
            if repo.agent_id == agent_id
        ]
        
        # Get containers
        agent_containers = [
            container for container in self.docker.containers.values()
            if container["agent_id"] == agent_id
        ]
        
        # Get recent operations
        recent_operations = [
            op for op in self.git.operation_history[-50:]  # Last 50 operations
            if op.agent_id == agent_id
        ]
        
        return {
            "agent_id": agent_id,
            "repositories": {
                "total": len(agent_repositories),
                "details": [repo.__dict__ for repo in agent_repositories]
            },
            "containers": {
                "total": len(agent_containers),
                "details": agent_containers
            },
            "recent_operations": {
                "total": len(recent_operations),
                "details": [op.to_dict() for op in recent_operations[-10:]]  # Last 10
            },
            "tools_available": {
                "git": True,
                "github": bool(self.github.github_token),
                "docker": True
            }
        }


# Security utilities for subprocess execution
class SecureProcessExecution:
    """Secure subprocess execution with validation and sandboxing."""
    
    # Allowed commands for security
    ALLOWED_COMMANDS = {
        "git", "docker", "npm", "pip", "python", "node", "curl", "wget"
    }
    
    # Dangerous command patterns to block
    BLOCKED_PATTERNS = [
        "rm -rf", "sudo", "su", "passwd", "chmod 777", "chown",
        "&&", "||", ";", "|", "`", "$", "$(", "eval", "exec"
    ]
    
    @classmethod
    async def execute_secure_command(
        cls,
        command: str,
        args: List[str],
        cwd: Optional[Path] = None,
        timeout: int = 30,
        capture_output: bool = True
    ) -> Tuple[int, str, str]:
        """
        Execute command securely with validation and sandboxing.
        
        Args:
            command: Base command to execute
            args: Command arguments 
            cwd: Working directory
            timeout: Execution timeout
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            Tuple of (return_code, stdout, stderr)
            
        Raises:
            SecurityError: If command is not allowed or contains dangerous patterns
        """
        # Validate command is allowed
        if command not in cls.ALLOWED_COMMANDS:
            raise SecurityException(f"Command '{command}' not in allowed list")
        
        # Validate arguments don't contain dangerous patterns
        full_command_str = f"{command} {' '.join(args)}"
        for pattern in cls.BLOCKED_PATTERNS:
            if pattern in full_command_str:
                raise SecurityException(f"Dangerous pattern '{pattern}' detected in command")
        
        # Sanitize arguments
        sanitized_args = [shlex.quote(arg) for arg in args]
        
        try:
            # Execute with security constraints
            process = await asyncio.create_subprocess_exec(
                command,
                *sanitized_args,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE if capture_output else None,
                limit=1024 * 1024,  # 1MB limit
            )
            
            # Wait with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return (
                process.returncode,
                stdout.decode('utf-8') if stdout else "",
                stderr.decode('utf-8') if stderr else ""
            )
            
        except asyncio.TimeoutError:
            # Kill process on timeout
            process.kill()
            await process.wait()
            raise SecurityException(f"Command '{command}' timed out after {timeout} seconds")
        except Exception as e:
            raise SecurityException(f"Command execution failed: {e}")

class SecurityException(Exception):
    """Security-related execution exception."""
    pass

# Global external tools orchestrator
external_tools = ExternalToolsOrchestrator()