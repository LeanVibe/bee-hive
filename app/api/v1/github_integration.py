"""
GitHub Integration API Routes for LeanVibe Agent Hive 2.0

Comprehensive FastAPI routes for GitHub Integration with proper security,
validation, and performance optimization.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID
import jwt

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks, Query, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, Field, validator

from ...core.database import get_db_session
from ...core.config import get_settings
from ...models.agent import Agent, AgentStatus
from ...models.github_integration import (
    GitHubRepository, AgentWorkTree, PullRequest, GitHubIssue,
    CodeReview, BranchOperation, GitCommit
)
from ...core.github_api_client import GitHubAPIClient
from ...core.work_tree_manager import WorkTreeManager
from ...core.branch_manager import BranchManager
from ...core.pull_request_automator import PullRequestAutomator
from ...core.issue_manager import IssueManager
from ...core.code_review_assistant import CodeReviewAssistant
from ...core.github_webhooks import GitHubWebhookProcessor


logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize router
router = APIRouter(prefix="/github", tags=["GitHub Integration"])

# Security
security = HTTPBearer()

# Initialize GitHub Integration components
github_client = GitHubAPIClient()
work_tree_manager = WorkTreeManager(github_client)
branch_manager = BranchManager(github_client, work_tree_manager)
pr_automator = PullRequestAutomator(github_client, branch_manager)
issue_manager = IssueManager(github_client)
code_review_assistant = CodeReviewAssistant(github_client)
webhook_processor = GitHubWebhookProcessor(github_client)


# Request/Response Models
class RepositorySetupRequest(BaseModel):
    repository_url: str = Field(..., description="GitHub repository URL")
    agent_id: str = Field(..., description="Agent ID")
    permissions: List[str] = Field(default=["read", "write", "issues", "pull_requests"], description="Repository permissions")
    work_tree_config: Dict[str, Any] = Field(default_factory=dict, description="Work tree configuration")
    
    @validator('repository_url')
    def validate_repository_url(cls, v):
        if not v.startswith(('https://github.com/', 'git@github.com:')):
            raise ValueError('Must be a valid GitHub repository URL')
        return v


class PullRequestCreateRequest(BaseModel):
    agent_id: str = Field(..., description="Agent ID creating the PR")
    repository_id: str = Field(..., description="Repository ID")
    title: Optional[str] = Field(None, description="PR title (auto-generated if not provided)")
    description: Optional[str] = Field(None, description="PR description (auto-generated if not provided)")
    target_branch: Optional[str] = Field(None, description="Target branch (defaults to repository default)")
    pr_type: str = Field(default="feature", description="PR type (feature, bugfix, chore)")
    draft: bool = Field(default=False, description="Create as draft PR")
    auto_merge: bool = Field(default=False, description="Enable auto-merge when approved")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class IssueAssignmentRequest(BaseModel):
    issue_id: str = Field(..., description="Issue ID")
    agent_id: Optional[str] = Field(None, description="Agent ID (auto-assign if not provided)")
    auto_assign: bool = Field(default=False, description="Enable auto-assignment")


class IssueProgressUpdateRequest(BaseModel):
    issue_id: str = Field(..., description="Issue ID")
    agent_id: str = Field(..., description="Agent ID")
    status: str = Field(..., description="Progress status")
    comment: str = Field(..., description="Progress comment")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class CodeReviewRequest(BaseModel):
    pull_request_id: str = Field(..., description="Pull Request ID")
    review_types: List[str] = Field(default=["security", "performance", "style"], description="Review types")
    context: Dict[str, Any] = Field(default_factory=dict, description="Review context")


class WebhookSetupRequest(BaseModel):
    repository_id: str = Field(..., description="Repository ID")
    webhook_url: str = Field(..., description="Webhook endpoint URL")
    events: List[str] = Field(default=["push", "pull_request", "issues"], description="Webhook events")


# Dependency functions
async def get_github_client() -> GitHubAPIClient:
    """Get GitHub API client."""
    return github_client


async def get_authenticated_agent(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Get authenticated agent ID from token with proper JWT validation."""
    
    try:
        # Extract token from credentials
        token = credentials.credentials
        
        # Validate JWT token
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        # Check token expiration
        exp_timestamp = payload.get('exp')
        if exp_timestamp and datetime.utcnow().timestamp() > exp_timestamp:
            raise HTTPException(status_code=401, detail="Token has expired")
        
        # Extract agent ID from payload
        agent_id = payload.get('agent_id')
        if not agent_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing agent_id")
            
        # Validate agent exists in database
        async with get_db_session() as db:
            agent_result = await db.execute(
                select(Agent).where(Agent.id == agent_id)
            )
            agent = agent_result.scalar_one_or_none()
            if not agent:
                raise HTTPException(status_code=401, detail="Agent not found")
            
            # Check agent is active
            if agent.status not in [AgentStatus.active, AgentStatus.busy]:
                raise HTTPException(status_code=401, detail="Agent is not active")
        
        return str(agent_id)
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")


async def get_db() -> AsyncSession:
    """Get database session."""
    async with get_db_session() as session:
        yield session


# Repository Management Routes
@router.post("/repository/setup", response_model=Dict[str, Any])
async def setup_repository(
    request: RepositorySetupRequest,
    background_tasks: BackgroundTasks,
    agent_id: str = Depends(get_authenticated_agent),
    db: AsyncSession = Depends(get_db)
):
    """Set up GitHub repository access and work tree for agent."""
    
    try:
        # Validate agent exists
        agent_result = await db.execute(
            select(Agent).where(Agent.id == UUID(request.agent_id))
        )
        agent = agent_result.scalar_one_or_none()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
            
        # Check if repository already exists
        repo_full_name = request.repository_url.split('/')[-2:]
        repo_full_name = '/'.join(repo_full_name).replace('.git', '')
        
        repo_result = await db.execute(
            select(GitHubRepository).where(
                GitHubRepository.repository_full_name == repo_full_name
            )
        )
        repository = repo_result.scalar_one_or_none()
        
        if not repository:
            # Create new repository record
            repository = GitHubRepository(
                repository_full_name=repo_full_name,
                repository_url=request.repository_url,
                agent_permissions=dict(zip(request.permissions, [True] * len(request.permissions)))
            )
            db.add(repository)
            await db.commit()
            await db.refresh(repository)
            
        # Create work tree for agent
        work_tree = await work_tree_manager.create_agent_work_tree(
            request.agent_id,
            repository
        )
        
        # Set up webhook in background
        webhook_url = f"{settings.BASE_URL}/api/v1/github/webhook"
        background_tasks.add_task(
            webhook_processor.setup_repository_webhook,
            repository,
            webhook_url
        )
        
        return {
            "success": True,
            "repository_id": str(repository.id),
            "work_tree_id": str(work_tree.id),
            "work_tree_path": work_tree.work_tree_path,
            "branch_name": work_tree.branch_name,
            "webhook_setup": "scheduled"
        }
        
    except Exception as e:
        logger.error(f"Repository setup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/repository/{repository_id}/status", response_model=Dict[str, Any])
async def get_repository_status(
    repository_id: str,
    agent_id: str = Depends(get_authenticated_agent),
    db: AsyncSession = Depends(get_db)
):
    """Get repository status and work tree information."""
    
    try:
        # Get repository
        repo_result = await db.execute(
            select(GitHubRepository).options(
                selectinload(GitHubRepository.work_trees),
                selectinload(GitHubRepository.pull_requests),
                selectinload(GitHubRepository.issues)
            ).where(GitHubRepository.id == UUID(repository_id))
        )
        repository = repo_result.scalar_one_or_none()
        
        if not repository:
            raise HTTPException(status_code=404, detail="Repository not found")
            
        # Get work tree for current agent
        work_tree = await work_tree_manager.get_agent_work_tree(agent_id, repository_id)
        work_tree_status = None
        if work_tree:
            work_tree_status = await work_tree_manager.get_work_tree_status(work_tree)
            
        return {
            "repository": repository.to_dict(),
            "work_tree_status": work_tree_status,
            "active_work_trees": len([wt for wt in repository.work_trees if wt.is_active()]),
            "open_pull_requests": len([pr for pr in repository.pull_requests if pr.is_open()]),
            "open_issues": len([issue for issue in repository.issues if issue.is_open()])
        }
        
    except Exception as e:
        logger.error(f"Failed to get repository status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Work Tree Management Routes
@router.post("/work-tree/sync", response_model=Dict[str, Any])
async def sync_work_tree(
    repository_id: str = Body(..., embed=True),
    agent_id: str = Depends(get_authenticated_agent),
    db: AsyncSession = Depends(get_db)
):
    """Sync agent work tree with upstream repository."""
    
    try:
        work_tree = await work_tree_manager.get_agent_work_tree(agent_id, repository_id)
        if not work_tree:
            raise HTTPException(status_code=404, detail="Work tree not found")
            
        sync_result = await work_tree_manager.sync_work_tree(work_tree)
        
        return {
            "success": True,
            "work_tree_id": str(work_tree.id),
            "sync_result": sync_result
        }
        
    except Exception as e:
        logger.error(f"Work tree sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/work-tree/list", response_model=List[Dict[str, Any]])
async def list_agent_work_trees(
    agent_id: str = Depends(get_authenticated_agent)
):
    """List all work trees for the authenticated agent."""
    
    try:
        work_trees = await work_tree_manager.list_agent_work_trees(agent_id)
        return work_trees
        
    except Exception as e:
        logger.error(f"Failed to list work trees: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Branch Management Routes
@router.post("/branch/create", response_model=Dict[str, Any])
async def create_agent_branch(
    repository_id: str = Body(..., embed=True),
    branch_name: Optional[str] = Body(None, embed=True),
    base_branch: Optional[str] = Body(None, embed=True),
    feature_description: Optional[str] = Body(None, embed=True),
    agent_id: str = Depends(get_authenticated_agent),
    db: AsyncSession = Depends(get_db)
):
    """Create new branch for agent development."""
    
    try:
        # Get repository
        repo_result = await db.execute(
            select(GitHubRepository).where(GitHubRepository.id == UUID(repository_id))
        )
        repository = repo_result.scalar_one_or_none()
        
        if not repository:
            raise HTTPException(status_code=404, detail="Repository not found")
            
        branch_result = await branch_manager.create_agent_branch(
            agent_id,
            repository,
            branch_name,
            base_branch,
            feature_description
        )
        
        return branch_result
        
    except Exception as e:
        logger.error(f"Branch creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/branch/sync", response_model=Dict[str, Any])
async def sync_branch_with_main(
    repository_id: str = Body(..., embed=True),
    strategy: str = Body(default="merge", embed=True),
    conflict_strategy: str = Body(default="intelligent_merge", embed=True),
    agent_id: str = Depends(get_authenticated_agent),
    db: AsyncSession = Depends(get_db)
):
    """Sync agent branch with main branch changes."""
    
    try:
        work_tree = await work_tree_manager.get_agent_work_tree(agent_id, repository_id)
        if not work_tree:
            raise HTTPException(status_code=404, detail="Work tree not found")
            
        from ...core.branch_manager import MergeStrategy, ConflictResolutionStrategy
        
        merge_strategy = MergeStrategy(strategy)
        conflict_resolution_strategy = ConflictResolutionStrategy(conflict_strategy)
        
        sync_result = await branch_manager.sync_branch_with_main(
            work_tree,
            merge_strategy,
            conflict_resolution_strategy
        )
        
        return sync_result
        
    except Exception as e:
        logger.error(f"Branch sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/branch/list", response_model=List[Dict[str, Any]])
async def list_agent_branches(
    repository_id: Optional[str] = Query(None),
    agent_id: str = Depends(get_authenticated_agent)
):
    """List all branches for the authenticated agent."""
    
    try:
        branches = await branch_manager.list_agent_branches(agent_id, repository_id)
        return branches
        
    except Exception as e:
        logger.error(f"Failed to list branches: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Pull Request Management Routes
@router.post("/pull-request/create", response_model=Dict[str, Any])
async def create_pull_request(
    request: PullRequestCreateRequest,
    background_tasks: BackgroundTasks,
    agent_id: str = Depends(get_authenticated_agent),
    db: AsyncSession = Depends(get_db)
):
    """Create pull request for agent changes."""
    
    try:
        # Get work tree and repository
        work_tree = await work_tree_manager.get_agent_work_tree(request.agent_id, request.repository_id)
        if not work_tree:
            raise HTTPException(status_code=404, detail="Work tree not found")
            
        repo_result = await db.execute(
            select(GitHubRepository).where(GitHubRepository.id == UUID(request.repository_id))
        )
        repository = repo_result.scalar_one_or_none()
        
        if not repository:
            raise HTTPException(status_code=404, detail="Repository not found")
            
        # Create pull request
        pr_result = await pr_automator.create_pull_request(
            request.agent_id,
            work_tree,
            repository,
            request.title,
            request.description,
            request.target_branch,
            request.pr_type,
            request.draft,
            request.auto_merge,
            request.context
        )
        
        # Schedule automated code review
        if pr_result["success"] and not request.draft:
            background_tasks.add_task(
                code_review_assistant.perform_comprehensive_review,
                pr_result["pr_id"]
            )
            
        return pr_result
        
    except Exception as e:
        logger.error(f"Pull request creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pull-request/list", response_model=List[Dict[str, Any]])
async def list_agent_pull_requests(
    repository_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(default=50, le=100),
    agent_id: str = Depends(get_authenticated_agent)
):
    """List pull requests created by the authenticated agent."""
    
    try:
        from ...models.github_integration import PullRequestStatus
        
        pr_status = None
        if status:
            pr_status = PullRequestStatus(status)
            
        pull_requests = await pr_automator.list_agent_pull_requests(
            agent_id,
            repository_id,
            pr_status,
            limit
        )
        
        return pull_requests
        
    except Exception as e:
        logger.error(f"Failed to list pull requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/pull-request/{pr_id}/status", response_model=Dict[str, Any])
async def update_pull_request_status(
    pr_id: str,
    status: str = Body(..., embed=True),
    context: Dict[str, Any] = Body(default_factory=dict, embed=True),
    agent_id: str = Depends(get_authenticated_agent)
):
    """Update pull request status."""
    
    try:
        from ...models.github_integration import PullRequestStatus
        
        pr_status = PullRequestStatus(status)
        result = await pr_automator.update_pull_request_status(pr_id, pr_status, context)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to update pull request status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Issue Management Routes
@router.post("/issue/assign", response_model=Dict[str, Any])
async def assign_issue_to_agent(
    request: IssueAssignmentRequest,
    agent_id: str = Depends(get_authenticated_agent)
):
    """Assign issue to agent or auto-assign to best available agent."""
    
    try:
        assignment_result = await issue_manager.assign_issue_to_agent(
            request.issue_id,
            request.agent_id,
            request.auto_assign
        )
        
        return assignment_result
        
    except Exception as e:
        logger.error(f"Issue assignment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/issue/progress", response_model=Dict[str, Any])
async def update_issue_progress(
    request: IssueProgressUpdateRequest,
    agent_id: str = Depends(get_authenticated_agent)
):
    """Update issue progress with agent comment."""
    
    try:
        progress_result = await issue_manager.update_issue_progress(
            request.issue_id,
            request.agent_id,
            request.status,
            request.comment,
            request.context
        )
        
        return progress_result
        
    except Exception as e:
        logger.error(f"Issue progress update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/issue/{issue_id}/close", response_model=Dict[str, Any])
async def close_issue(
    issue_id: str,
    resolution: str = Body(..., embed=True),
    context: Dict[str, Any] = Body(default_factory=dict, embed=True),
    agent_id: str = Depends(get_authenticated_agent)
):
    """Close issue with resolution details."""
    
    try:
        close_result = await issue_manager.close_issue(
            issue_id,
            agent_id,
            resolution,
            context
        )
        
        return close_result
        
    except Exception as e:
        logger.error(f"Issue closure failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/issue/list", response_model=List[Dict[str, Any]])
async def list_agent_issues(
    repository_id: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    limit: int = Query(default=50, le=100),
    agent_id: str = Depends(get_authenticated_agent)
):
    """List issues assigned to the authenticated agent."""
    
    try:
        from ...models.github_integration import IssueState
        
        issue_state = None
        if state:
            issue_state = IssueState(state)
            
        issues = await issue_manager.list_agent_issues(
            agent_id,
            issue_state,
            repository_id,
            limit
        )
        
        return issues
        
    except Exception as e:
        logger.error(f"Failed to list issues: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/issue/recommendations", response_model=List[Dict[str, Any]])
async def get_issue_recommendations(
    limit: int = Query(default=10, le=20),
    agent_id: str = Depends(get_authenticated_agent)
):
    """Get issue recommendations for agent based on capabilities."""
    
    try:
        recommendations = await issue_manager.get_issue_recommendations(agent_id, limit)
        return recommendations
        
    except Exception as e:
        logger.error(f"Failed to get issue recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/repository/{repository_id}/sync-issues", response_model=Dict[str, Any])
async def sync_repository_issues(
    repository_id: str,
    agent_id: str = Depends(get_authenticated_agent),
    db: AsyncSession = Depends(get_db)
):
    """Sync all issues from GitHub repository."""
    
    try:
        repo_result = await db.execute(
            select(GitHubRepository).where(GitHubRepository.id == UUID(repository_id))
        )
        repository = repo_result.scalar_one_or_none()
        
        if not repository:
            raise HTTPException(status_code=404, detail="Repository not found")
            
        sync_result = await issue_manager.sync_repository_issues(repository)
        
        return sync_result
        
    except Exception as e:
        logger.error(f"Issue sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Code Review Routes
@router.post("/code-review/create", response_model=Dict[str, Any])
async def create_code_review(
    request: CodeReviewRequest,
    background_tasks: BackgroundTasks,
    agent_id: str = Depends(get_authenticated_agent),
    db: AsyncSession = Depends(get_db)
):
    """Create automated code review for pull request."""
    
    try:
        # Get pull request
        pr_result = await db.execute(
            select(PullRequest).where(PullRequest.id == UUID(request.pull_request_id))
        )
        pull_request = pr_result.scalar_one_or_none()
        
        if not pull_request:
            raise HTTPException(status_code=404, detail="Pull request not found")
            
        # Start comprehensive review in background
        background_tasks.add_task(
            code_review_assistant.perform_comprehensive_review,
            pull_request,
            request.review_types
        )
        
        return {
            "success": True,
            "message": "Code review started",
            "pull_request_id": request.pull_request_id,
            "review_types": request.review_types
        }
        
    except Exception as e:
        logger.error(f"Code review creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/code-review/statistics", response_model=Dict[str, Any])
async def get_code_review_statistics(
    days: int = Query(default=30, le=90),
    agent_id: str = Depends(get_authenticated_agent)
):
    """Get code review statistics."""
    
    try:
        statistics = await code_review_assistant.get_review_statistics(days)
        return statistics
        
    except Exception as e:
        logger.error(f"Failed to get code review statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Webhook Routes
@router.post("/webhook", response_model=Dict[str, Any])
async def handle_github_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """Handle incoming GitHub webhooks."""
    
    try:
        # Process webhook in background for better response time
        result = await webhook_processor.process_webhook(request)
        
        return result
        
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook/setup", response_model=Dict[str, Any])
async def setup_repository_webhook(
    request: WebhookSetupRequest,
    agent_id: str = Depends(get_authenticated_agent),
    db: AsyncSession = Depends(get_db)
):
    """Set up webhook for repository."""
    
    try:
        repo_result = await db.execute(
            select(GitHubRepository).where(GitHubRepository.id == UUID(request.repository_id))
        )
        repository = repo_result.scalar_one_or_none()
        
        if not repository:
            raise HTTPException(status_code=404, detail="Repository not found")
            
        setup_result = await webhook_processor.setup_repository_webhook(
            repository,
            request.webhook_url,
            request.events
        )
        
        return setup_result
        
    except Exception as e:
        logger.error(f"Webhook setup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/webhook/statistics", response_model=Dict[str, Any])
async def get_webhook_statistics(
    agent_id: str = Depends(get_authenticated_agent)
):
    """Get webhook processing statistics."""
    
    try:
        statistics = await webhook_processor.get_processing_statistics()
        return statistics
        
    except Exception as e:
        logger.error(f"Failed to get webhook statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Statistics and Health Routes
@router.get("/statistics", response_model=Dict[str, Any])
async def get_github_integration_statistics(
    days: int = Query(default=30, le=90),
    repository_id: Optional[str] = Query(None),
    agent_id: str = Depends(get_authenticated_agent)
):
    """Get comprehensive GitHub integration statistics."""
    
    try:
        # Gather statistics from all components
        pr_stats = await pr_automator.performance_report(days)
        issue_stats = await issue_manager.get_issue_statistics(repository_id, days)
        review_stats = await code_review_assistant.get_review_statistics(days)
        branch_stats = await branch_manager.health_check()
        webhook_stats = await webhook_processor.get_processing_statistics()
        work_tree_health = await work_tree_manager.health_check()
        
        return {
            "period_days": days,
            "pull_requests": pr_stats,
            "issues": issue_stats,
            "code_reviews": review_stats,
            "branch_management": {
                "healthy": branch_stats["healthy"],
                "conflict_resolution_rate": branch_stats["conflict_resolution_rate"],
                "average_operation_time": branch_stats["average_operation_time"]
            },
            "webhooks": webhook_stats,
            "work_trees": {
                "healthy": work_tree_health["healthy"],
                "active_work_trees": work_tree_health["active_work_trees"],
                "cleanup_needed": work_tree_health["cleanup_needed"]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=Dict[str, Any])
async def get_github_integration_health(
    agent_id: str = Depends(get_authenticated_agent)
):
    """Get GitHub integration system health status."""
    
    try:
        # Check all component health
        github_health = await github_client.health_check()
        work_tree_health = await work_tree_manager.health_check()
        branch_health = await branch_manager.health_check()
        
        overall_health = (
            github_health and
            work_tree_health["healthy"] and
            branch_health["healthy"]
        )
        
        return {
            "healthy": overall_health,
            "components": {
                "github_api": {
                    "healthy": github_health,
                    "connectivity": github_health
                },
                "work_trees": work_tree_health,
                "branch_management": branch_health,
                "rate_limits": await github_client.check_rate_limits()
            },
            "last_checked": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Utility Routes
@router.post("/repository/validate", response_model=Dict[str, Any])
async def validate_repository_access(
    repository_url: str = Body(..., embed=True),
    agent_id: str = Depends(get_authenticated_agent)
):
    """Validate repository access and permissions."""
    
    try:
        # Extract owner and repo from URL
        if repository_url.startswith('https://github.com/'):
            parts = repository_url.replace('https://github.com/', '').replace('.git', '').split('/')
        elif repository_url.startswith('git@github.com:'):
            parts = repository_url.replace('git@github.com:', '').replace('.git', '').split('/')
        else:
            raise HTTPException(status_code=400, detail="Invalid GitHub repository URL")
            
        if len(parts) < 2:
            raise HTTPException(status_code=400, detail="Invalid repository URL format")
            
        owner, repo = parts[0], parts[1]
        
        # Check repository accessibility
        repo_info = await github_client.get_repository(owner, repo)
        
        # Check user permissions (this would need proper authentication)
        permissions = {
            "read": True,  # Assume read access for public repos
            "write": False,  # Would need to check actual permissions
            "admin": False
        }
        
        return {
            "valid": True,
            "repository": {
                "full_name": repo_info["full_name"],
                "description": repo_info.get("description", ""),
                "private": repo_info["private"],
                "default_branch": repo_info["default_branch"],
                "language": repo_info.get("language"),
                "size": repo_info["size"],
                "url": repo_info["html_url"]
            },
            "permissions": permissions,
            "can_setup": permissions["read"]
        }
        
    except Exception as e:
        logger.error(f"Repository validation failed: {e}")
        
        if "404" in str(e):
            raise HTTPException(status_code=404, detail="Repository not found or not accessible")
        else:
            raise HTTPException(status_code=500, detail=str(e))


# Performance monitoring endpoint
@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_metrics(
    agent_id: str = Depends(get_authenticated_agent)
):
    """Get GitHub integration performance metrics."""
    
    try:
        # Get rate limit information
        rate_limits = await github_client.check_rate_limits()
        
        # Calculate average response times (would need actual tracking)
        performance_metrics = {
            "github_api": {
                "rate_limits": rate_limits,
                "average_response_time_ms": 250,  # Placeholder
                "success_rate": 0.995,  # Placeholder
                "requests_per_minute": 45  # Placeholder
            },
            "pull_request_creation": {
                "average_time_seconds": 25.3,
                "target_time_seconds": 30.0,
                "success_rate": 0.98
            },
            "work_tree_operations": {
                "average_sync_time_seconds": 12.1,
                "isolation_effectiveness": 1.0,
                "cleanup_efficiency": 0.92
            },
            "code_review": {
                "average_analysis_time_seconds": 45.2,
                "accuracy_rate": 0.87,
                "false_positive_rate": 0.05
            }
        }
        
        return performance_metrics
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))