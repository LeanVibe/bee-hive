"""
External Tools API endpoints for LeanVibe Agent Hive 2.0

Provides HTTP endpoints for Git, GitHub, Docker, and CI/CD integrations,
enabling agents to interact with real-world development tools and workflows.
"""

import uuid
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

import structlog

from ...core.external_tools import external_tools, ToolType, OperationStatus
from ...core.database import get_session_dependency
from ...models.agent import Agent

logger = structlog.get_logger()
router = APIRouter()


# Request Models
class GitCloneRequest(BaseModel):
    """Request to clone a Git repository."""
    agent_id: str = Field(..., description="Agent ID")
    repository_url: str = Field(..., description="Git repository URL")
    target_directory: Optional[str] = Field(None, description="Target directory name")
    branch: Optional[str] = Field(None, description="Branch to clone")


class GitCommitRequest(BaseModel):
    """Request to commit changes."""
    repository_id: str = Field(..., description="Repository ID")
    message: str = Field(..., description="Commit message")
    files: Optional[List[str]] = Field(None, description="Specific files to commit")


class GitBranchRequest(BaseModel):
    """Request to create a branch."""
    repository_id: str = Field(..., description="Repository ID")
    branch_name: str = Field(..., description="Branch name")
    checkout: bool = Field(default=True, description="Checkout after creation")


class GitPushRequest(BaseModel):
    """Request to push changes."""
    repository_id: str = Field(..., description="Repository ID")
    remote: str = Field(default="origin", description="Remote name")
    branch: Optional[str] = Field(None, description="Branch to push")


class GitHubRepositoryRequest(BaseModel):
    """Request to get GitHub repository info."""
    owner: str = Field(..., description="Repository owner")
    repo_name: str = Field(..., description="Repository name")


class GitHubPullRequestRequest(BaseModel):
    """Request to create a pull request."""
    repository_id: str = Field(..., description="GitHub repository ID")
    title: str = Field(..., description="PR title")
    body: str = Field(..., description="PR description")
    head_branch: str = Field(..., description="Head branch")
    base_branch: str = Field(default="main", description="Base branch")


class GitHubIssueRequest(BaseModel):
    """Request to create an issue."""
    repository_id: str = Field(..., description="GitHub repository ID")
    title: str = Field(..., description="Issue title")
    body: str = Field(..., description="Issue description")
    labels: Optional[List[str]] = Field(None, description="Issue labels")
    assignees: Optional[List[str]] = Field(None, description="Issue assignees")


class DockerBuildRequest(BaseModel):
    """Request to build a Docker image."""
    agent_id: str = Field(..., description="Agent ID")
    dockerfile_path: str = Field(..., description="Path to Dockerfile")
    image_name: str = Field(..., description="Image name")
    context_path: str = Field(default=".", description="Build context path")
    build_args: Optional[Dict[str, str]] = Field(None, description="Build arguments")


class DockerRunRequest(BaseModel):
    """Request to run a Docker container."""
    agent_id: str = Field(..., description="Agent ID")
    image_name: str = Field(..., description="Docker image name")
    container_name: Optional[str] = Field(None, description="Container name")
    ports: Optional[Dict[str, str]] = Field(None, description="Port mappings")
    environment: Optional[Dict[str, str]] = Field(None, description="Environment variables")
    volumes: Optional[Dict[str, str]] = Field(None, description="Volume mounts")
    detached: bool = Field(default=True, description="Run in detached mode")


class WorkflowExecuteRequest(BaseModel):
    """Request to execute a workflow."""
    agent_id: str = Field(..., description="Agent ID")
    workflow_definition: Dict[str, Any] = Field(..., description="Workflow definition")


# Response Models
class GitRepositoryResponse(BaseModel):
    """Response with Git repository information."""
    repository_id: str
    agent_id: str
    repository_url: str
    local_path: str
    branch: str
    is_clean: bool
    uncommitted_changes: List[str]
    total_commits: int


class GitHubRepositoryResponse(BaseModel):
    """Response with GitHub repository information."""
    repository_id: str
    owner: str
    name: str
    full_name: str
    clone_url: str
    default_branch: str
    description: str
    language: str
    stars: int
    forks: int
    open_issues: int
    private: bool


class OperationResponse(BaseModel):
    """Response with operation status."""
    success: bool
    output: str
    error: str
    operation_id: Optional[str] = None


class WorkflowResponse(BaseModel):
    """Response from workflow execution."""
    success: bool
    workflow_id: str
    steps_completed: int
    operations: List[Dict[str, Any]]
    message: str


# Git Endpoints
@router.post("/git/clone", response_model=GitRepositoryResponse)
async def clone_repository(
    request: GitCloneRequest,
    db: AsyncSession = Depends(get_session_dependency)
) -> GitRepositoryResponse:
    """Clone a Git repository into agent's workspace."""
    
    try:
        # Verify agent exists
        agent = await db.get(Agent, request.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Clone repository
        success, repository, message = await external_tools.git.clone_repository(
            agent_id=request.agent_id,
            repository_url=request.repository_url,
            target_directory=request.target_directory,
            branch=request.branch
        )
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Clone failed: {message}")
        
        logger.info(
            "Repository cloned via API",
            agent_id=request.agent_id,
            repository_url=request.repository_url,
            repository_id=repository.id
        )
        
        return GitRepositoryResponse(
            repository_id=repository.id,
            agent_id=repository.agent_id,
            repository_url=repository.repository_url,
            local_path=repository.local_path,
            branch=repository.branch,
            is_clean=repository.is_clean,
            uncommitted_changes=repository.uncommitted_changes,
            total_commits=repository.total_commits
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Git clone failed via API", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/git/commit", response_model=OperationResponse)
async def commit_changes(request: GitCommitRequest) -> OperationResponse:
    """Commit changes to a Git repository."""
    
    try:
        success, output, error = await external_tools.git.commit_changes(
            repository_id=request.repository_id,
            message=request.message,
            files=request.files
        )
        
        logger.info(
            "Git commit via API",
            repository_id=request.repository_id,
            success=success,
            message=request.message[:100]
        )
        
        return OperationResponse(
            success=success,
            output=output,
            error=error
        )
        
    except Exception as e:
        logger.error("Git commit failed via API", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/git/branch", response_model=OperationResponse)
async def create_branch(request: GitBranchRequest) -> OperationResponse:
    """Create a new Git branch."""
    
    try:
        success, output, error = await external_tools.git.create_branch(
            repository_id=request.repository_id,
            branch_name=request.branch_name,
            checkout=request.checkout
        )
        
        return OperationResponse(
            success=success,
            output=output,
            error=error
        )
        
    except Exception as e:
        logger.error("Git branch creation failed via API", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/git/push", response_model=OperationResponse)
async def push_changes(request: GitPushRequest) -> OperationResponse:
    """Push changes to remote repository."""
    
    try:
        success, output, error = await external_tools.git.push_changes(
            repository_id=request.repository_id,
            remote=request.remote,
            branch=request.branch
        )
        
        return OperationResponse(
            success=success,
            output=output,
            error=error
        )
        
    except Exception as e:
        logger.error("Git push failed via API", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# GitHub Endpoints
@router.post("/github/repository", response_model=GitHubRepositoryResponse)
async def get_github_repository(request: GitHubRepositoryRequest) -> GitHubRepositoryResponse:
    """Get GitHub repository information."""
    
    try:
        success, repository, message = await external_tools.github.get_repository(
            owner=request.owner,
            repo_name=request.repo_name
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Repository not found: {message}")
        
        return GitHubRepositoryResponse(
            repository_id=repository.id,
            owner=repository.owner,
            name=repository.name,
            full_name=repository.full_name,
            clone_url=repository.clone_url,
            default_branch=repository.default_branch,
            description=repository.description,
            language=repository.language,
            stars=repository.stars,
            forks=repository.forks,
            open_issues=repository.open_issues,
            private=repository.private
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("GitHub repository fetch failed via API", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/github/pull-request")
async def create_pull_request(request: GitHubPullRequestRequest) -> Dict[str, Any]:
    """Create a GitHub pull request."""
    
    try:
        success, pr_data, message = await external_tools.github.create_pull_request(
            repository_id=request.repository_id,
            title=request.title,
            body=request.body,
            head_branch=request.head_branch,
            base_branch=request.base_branch
        )
        
        if not success:
            raise HTTPException(status_code=500, detail=f"PR creation failed: {message}")
        
        logger.info(
            "Pull request created via API",
            repository_id=request.repository_id,
            pr_number=pr_data["number"],
            title=request.title
        )
        
        return {
            "success": True,
            "pr_number": pr_data["number"],
            "pr_url": pr_data["html_url"],
            "message": message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("GitHub PR creation failed via API", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/github/issue")
async def create_issue(request: GitHubIssueRequest) -> Dict[str, Any]:
    """Create a GitHub issue."""
    
    try:
        success, issue_data, message = await external_tools.github.create_issue(
            repository_id=request.repository_id,
            title=request.title,
            body=request.body,
            labels=request.labels,
            assignees=request.assignees
        )
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Issue creation failed: {message}")
        
        return {
            "success": True,
            "issue_number": issue_data["number"],
            "issue_url": issue_data["html_url"],
            "message": message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("GitHub issue creation failed via API", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# Docker Endpoints
@router.post("/docker/build", response_model=OperationResponse)
async def build_docker_image(
    request: DockerBuildRequest,
    db: AsyncSession = Depends(get_session_dependency)
) -> OperationResponse:
    """Build a Docker image."""
    
    try:
        # Verify agent exists
        agent = await db.get(Agent, request.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        success, output, error = await external_tools.docker.build_image(
            agent_id=request.agent_id,
            dockerfile_path=request.dockerfile_path,
            image_name=request.image_name,
            context_path=request.context_path,
            build_args=request.build_args
        )
        
        logger.info(
            "Docker image built via API",
            agent_id=request.agent_id,
            image_name=request.image_name,
            success=success
        )
        
        return OperationResponse(
            success=success,
            output=output,
            error=error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Docker build failed via API", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/docker/run", response_model=OperationResponse)
async def run_docker_container(
    request: DockerRunRequest,
    db: AsyncSession = Depends(get_session_dependency)
) -> OperationResponse:
    """Run a Docker container."""
    
    try:
        # Verify agent exists
        agent = await db.get(Agent, request.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        success, output, error = await external_tools.docker.run_container(
            agent_id=request.agent_id,
            image_name=request.image_name,
            container_name=request.container_name,
            ports=request.ports,
            environment=request.environment,
            volumes=request.volumes,
            detached=request.detached
        )
        
        logger.info(
            "Docker container started via API",
            agent_id=request.agent_id,
            image_name=request.image_name,
            success=success
        )
        
        return OperationResponse(
            success=success,
            output=output,
            error=error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Docker run failed via API", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/docker/stop/{agent_id}/{container_id}", response_model=OperationResponse)
async def stop_docker_container(
    agent_id: str,
    container_id: str,
    db: AsyncSession = Depends(get_session_dependency)
) -> OperationResponse:
    """Stop a Docker container."""
    
    try:
        # Verify agent exists
        agent = await db.get(Agent, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        success, output, error = await external_tools.docker.stop_container(
            agent_id=agent_id,
            container_id=container_id
        )
        
        return OperationResponse(
            success=success,
            output=output,
            error=error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Docker stop failed via API", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# Workflow Endpoints
@router.post("/workflow/execute", response_model=WorkflowResponse)
async def execute_workflow(
    request: WorkflowExecuteRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session_dependency)
) -> WorkflowResponse:
    """Execute a complex multi-tool workflow."""
    
    try:
        # Verify agent exists
        agent = await db.get(Agent, request.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        success, operations, message = await external_tools.execute_workflow(
            agent_id=request.agent_id,
            workflow_definition=request.workflow_definition
        )
        
        workflow_id = str(uuid.uuid4())
        
        logger.info(
            "Workflow executed via API",
            agent_id=request.agent_id,
            workflow_id=workflow_id,
            success=success,
            steps_completed=len(operations)
        )
        
        return WorkflowResponse(
            success=success,
            workflow_id=workflow_id,
            steps_completed=len(operations),
            operations=[op.to_dict() for op in operations],
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Workflow execution failed via API", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# Monitoring and Status Endpoints
@router.get("/operations/{operation_id}")
async def get_operation_status(operation_id: str) -> Dict[str, Any]:
    """Get the status of a specific operation."""
    
    try:
        operation = await external_tools.get_operation_status(operation_id)
        
        if not operation:
            raise HTTPException(status_code=404, detail="Operation not found")
        
        return operation.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Operation status fetch failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/operations/{operation_id}")
async def cancel_operation(operation_id: str) -> Dict[str, str]:
    """Cancel a running operation."""
    
    try:
        success = await external_tools.cancel_operation(operation_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Operation not found or not cancellable")
        
        return {"operation_id": operation_id, "status": "cancelled"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Operation cancellation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/agents/{agent_id}/tools-summary")
async def get_agent_tools_summary(agent_id: str) -> Dict[str, Any]:
    """Get comprehensive summary of tools used by an agent."""
    
    try:
        summary = await external_tools.get_agent_tools_summary(agent_id)
        return summary
        
    except Exception as e:
        logger.error("Tools summary fetch failed", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# Predefined Workflow Templates
@router.get("/workflows/templates")
async def get_workflow_templates() -> Dict[str, Any]:
    """Get predefined workflow templates."""
    
    templates = {
        "full_stack_deployment": {
            "type": "full_stack_deployment",
            "description": "Clone repository, build Docker image, and deploy",
            "steps": [
                {
                    "type": "git_clone",
                    "parameters": {
                        "repository_url": "{repository_url}",
                        "target_directory": "{project_name}",
                        "branch": "main"
                    }
                },
                {
                    "type": "docker_build",
                    "parameters": {
                        "dockerfile_path": "Dockerfile",
                        "image_name": "{project_name}:latest",
                        "context_path": "."
                    }
                },
                {
                    "type": "docker_run",
                    "parameters": {
                        "image_name": "{project_name}:latest",
                        "container_name": "{project_name}_container",
                        "ports": {"8000": "8000"},
                        "detached": True
                    }
                }
            ]
        },
        "feature_branch_workflow": {
            "type": "feature_branch_workflow", 
            "description": "Create feature branch, commit changes, and create PR",
            "steps": [
                {
                    "type": "git_branch",
                    "parameters": {
                        "branch_name": "feature/{feature_name}",
                        "checkout": True
                    }
                },
                {
                    "type": "git_commit",
                    "parameters": {
                        "message": "feat: {feature_description}",
                        "files": []
                    }
                },
                {
                    "type": "git_push",
                    "parameters": {
                        "remote": "origin",
                        "branch": "feature/{feature_name}"
                    }
                },
                {
                    "type": "github_pr",
                    "parameters": {
                        "title": "Feature: {feature_name}",
                        "body": "{feature_description}",
                        "head_branch": "feature/{feature_name}",
                        "base_branch": "main"
                    }
                }
            ]
        }
    }
    
    return {
        "templates": templates,
        "usage": "Replace variables in {} with actual values when executing workflows"
    }


@router.get("/health")
async def get_tools_health() -> Dict[str, Any]:
    """Get health status of all external tools."""
    
    return {
        "git": {
            "available": True,
            "repositories_managed": len(external_tools.git.repositories),
            "recent_operations": len(external_tools.git.operation_history[-10:])
        },
        "github": {
            "available": bool(external_tools.github.github_token),
            "repositories_accessed": len(external_tools.github.repositories)
        },
        "docker": {
            "available": True,
            "containers_managed": len(external_tools.docker.containers),
            "images_built": len(external_tools.docker.images)
        },
        "system": {
            "active_operations": len(external_tools.active_operations),
            "queued_operations": len(external_tools.operation_queue)
        }
    }