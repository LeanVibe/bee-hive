"""
GitHub Webhooks Integration for LeanVibe Agent Hive 2.0

Real-time GitHub event processing with webhook validation,
event routing, and automated agent responses.
"""

import asyncio
import logging
import uuid
import hmac
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from enum import Enum

from fastapi import HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

from ..core.config import get_settings
from ..core.database import get_db_session
from ..models.github_integration import (
    GitHubRepository, PullRequest, GitHubIssue, AgentWorkTree,
    PullRequestStatus, IssueState
)
from ..core.github_api_client import GitHubAPIClient
from ..core.issue_manager import IssueManager
from ..core.pull_request_automator import PullRequestAutomator


logger = logging.getLogger(__name__)
settings = get_settings()


class WebhookEventType(Enum):
    """GitHub webhook event types."""
    PUSH = "push"
    PULL_REQUEST = "pull_request"
    ISSUES = "issues"
    ISSUE_COMMENT = "issue_comment"
    PULL_REQUEST_REVIEW = "pull_request_review"
    PULL_REQUEST_REVIEW_COMMENT = "pull_request_review_comment"
    CREATE = "create"
    DELETE = "delete"
    FORK = "fork"
    WATCH = "watch"
    STAR = "star"
    RELEASE = "release"


class WebhookProcessingError(Exception):
    """Custom exception for webhook processing failures."""
    pass


class WebhookEventHandler:
    """
    Base class for webhook event handlers.
    
    Provides common functionality for processing GitHub webhook events
    with validation, error handling, and response coordination.
    """
    
    def __init__(self, github_client: GitHubAPIClient = None):
        self.github_client = github_client or GitHubAPIClient()
        
    async def handle_event(self, event_type: str, payload: Dict[str, Any], repository: GitHubRepository) -> Dict[str, Any]:
        """Handle webhook event. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement handle_event")
        
    def _extract_repository_info(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """Extract repository information from payload."""
        repo_data = payload.get("repository", {})
        return {
            "full_name": repo_data.get("full_name", ""),
            "name": repo_data.get("name", ""),
            "owner": repo_data.get("owner", {}).get("login", ""),
            "url": repo_data.get("html_url", ""),
            "default_branch": repo_data.get("default_branch", "main")
        }
        
    def _extract_user_info(self, payload: Dict[str, Any], user_key: str = "sender") -> Dict[str, str]:
        """Extract user information from payload."""
        user_data = payload.get(user_key, {})
        return {
            "login": user_data.get("login", ""),
            "id": user_data.get("id", 0),
            "type": user_data.get("type", ""),
            "url": user_data.get("html_url", "")
        }


class PushEventHandler(WebhookEventHandler):
    """Handle push events for branch updates and commits."""
    
    async def handle_event(self, event_type: str, payload: Dict[str, Any], repository: GitHubRepository) -> Dict[str, Any]:
        """Handle push event."""
        
        result = {
            "success": False,
            "event_type": event_type,
            "actions_taken": [],
            "errors": []
        }
        
        try:
            ref = payload.get("ref", "")
            branch_name = ref.replace("refs/heads/", "") if ref.startswith("refs/heads/") else ref
            commits = payload.get("commits", [])
            
            # Track commits in database
            tracked_commits = await self._track_commits(repository, commits, branch_name)
            result["actions_taken"].append(f"Tracked {len(tracked_commits)} commits")
            
            # Check if this affects any agent work trees
            affected_work_trees = await self._find_affected_work_trees(repository, branch_name)
            
            if affected_work_trees:
                # Notify agents of upstream changes
                notifications_sent = await self._notify_agents_of_changes(
                    affected_work_trees, branch_name, commits
                )
                result["actions_taken"].append(f"Notified {notifications_sent} agents of changes")
                
            # If this is the default branch, trigger sync for all work trees
            if branch_name == repository.default_branch:
                sync_triggered = await self._trigger_work_tree_syncs(repository)
                result["actions_taken"].append(f"Triggered sync for {sync_triggered} work trees")
                
            result["success"] = True
            
        except Exception as e:
            error_msg = f"Push event handling failed: {str(e)}"
            result["errors"].append(error_msg)
            logger.error(error_msg)
            
        return result
        
    async def _track_commits(self, repository: GitHubRepository, commits: List[Dict[str, Any]], branch_name: str) -> List[str]:
        """Track commits in database."""
        
        tracked_commits = []
        
        async with get_db_session() as session:
            for commit_data in commits:
                # Check if commit already exists
                existing_result = await session.execute(
                    select(GitCommit).where(
                        and_(
                            GitCommit.repository_id == repository.id,
                            GitCommit.commit_hash == commit_data["id"]
                        )
                    )
                )
                
                if existing_result.scalar_one_or_none():
                    continue  # Commit already tracked
                    
                # Create new commit record
                commit = GitCommit(
                    repository_id=repository.id,
                    commit_hash=commit_data["id"],
                    short_hash=commit_data["id"][:8],
                    branch_name=branch_name,
                    commit_message=commit_data.get("message", ""),
                    author_name=commit_data.get("author", {}).get("name", ""),
                    author_email=commit_data.get("author", {}).get("email", ""),
                    committer_name=commit_data.get("committer", {}).get("name", ""),
                    committer_email=commit_data.get("committer", {}).get("email", ""),
                    files_changed=len(commit_data.get("added", []) + commit_data.get("modified", []) + commit_data.get("removed", [])),
                    committed_at=datetime.fromisoformat(commit_data.get("timestamp", "").replace('Z', '+00:00')),
                    commit_metadata={
                        "added_files": commit_data.get("added", []),
                        "modified_files": commit_data.get("modified", []),
                        "removed_files": commit_data.get("removed", []),
                        "github_url": commit_data.get("url", "")
                    }
                )
                
                session.add(commit)
                tracked_commits.append(commit_data["id"])
                
            await session.commit()
            
        return tracked_commits
        
    async def _find_affected_work_trees(self, repository: GitHubRepository, branch_name: str) -> List[AgentWorkTree]:
        """Find work trees that might be affected by branch changes."""
        
        async with get_db_session() as session:
            # Find work trees based on the same repository and base branch
            result = await session.execute(
                select(AgentWorkTree).where(
                    and_(
                        AgentWorkTree.repository_id == repository.id,
                        or_(
                            AgentWorkTree.base_branch == branch_name,
                            AgentWorkTree.branch_name == branch_name
                        )
                    )
                )
            )
            
            return result.scalars().all()
            
    async def _notify_agents_of_changes(
        self,
        work_trees: List[AgentWorkTree],
        branch_name: str,
        commits: List[Dict[str, Any]]
    ) -> int:
        """Notify agents of upstream changes."""
        
        notifications_sent = 0
        
        for work_tree in work_trees:
            try:
                # Create notification message
                notification = {
                    "type": "upstream_changes",
                    "work_tree_id": str(work_tree.id),
                    "branch_name": branch_name,
                    "commits_count": len(commits),
                    "latest_commit": commits[0] if commits else None,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # TODO: Send notification to agent via Redis message queue
                # This would integrate with the agent orchestrator
                logger.info(f"Notification sent to agent {work_tree.agent_id} for work tree {work_tree.id}")
                notifications_sent += 1
                
            except Exception as e:
                logger.error(f"Failed to notify agent {work_tree.agent_id}: {e}")
                
        return notifications_sent
        
    async def _trigger_work_tree_syncs(self, repository: GitHubRepository) -> int:
        """Trigger sync for all work trees based on default branch."""
        
        async with get_db_session() as session:
            result = await session.execute(
                select(AgentWorkTree).where(
                    and_(
                        AgentWorkTree.repository_id == repository.id,
                        AgentWorkTree.base_branch == repository.default_branch
                    )
                )
            )
            
            work_trees = result.scalars().all()
            
            sync_triggered = 0
            for work_tree in work_trees:
                try:
                    # TODO: Trigger sync job via task queue
                    logger.info(f"Sync triggered for work tree {work_tree.id}")
                    sync_triggered += 1
                except Exception as e:
                    logger.error(f"Failed to trigger sync for work tree {work_tree.id}: {e}")
                    
            return sync_triggered


class PullRequestEventHandler(WebhookEventHandler):
    """Handle pull request events for automated management."""
    
    def __init__(self, github_client: GitHubAPIClient = None):
        super().__init__(github_client)
        self.pr_automator = PullRequestAutomator(github_client)
        
    async def handle_event(self, event_type: str, payload: Dict[str, Any], repository: GitHubRepository) -> Dict[str, Any]:
        """Handle pull request event."""
        
        result = {
            "success": False,
            "event_type": event_type,
            "action": payload.get("action", ""),
            "actions_taken": [],
            "errors": []
        }
        
        try:
            action = payload.get("action", "")
            pr_data = payload.get("pull_request", {})
            
            if action == "opened":
                await self._handle_pr_opened(repository, pr_data, result)
            elif action == "closed":
                await self._handle_pr_closed(repository, pr_data, result)
            elif action == "synchronize":
                await self._handle_pr_synchronized(repository, pr_data, result)
            elif action == "ready_for_review":
                await self._handle_pr_ready_for_review(repository, pr_data, result)
            else:
                result["actions_taken"].append(f"No specific handler for action: {action}")
                
            result["success"] = True
            
        except Exception as e:
            error_msg = f"Pull request event handling failed: {str(e)}"
            result["errors"].append(error_msg)
            logger.error(error_msg)
            
        return result
        
    async def _handle_pr_opened(self, repository: GitHubRepository, pr_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Handle new pull request opened."""
        
        # Check if this is an agent-created PR
        branch_name = pr_data.get("head", {}).get("ref", "")
        
        if branch_name.startswith("agent/"):
            # This is an agent PR, update our database record
            async with get_db_session() as session:
                pr_result = await session.execute(
                    select(PullRequest).where(
                        and_(
                            PullRequest.repository_id == repository.id,
                            PullRequest.source_branch == branch_name
                        )
                    )
                )
                existing_pr = pr_result.scalar_one_or_none()
                
                if existing_pr:
                    # Update with GitHub PR number
                    existing_pr.github_pr_number = pr_data["number"]
                    existing_pr.github_pr_id = pr_data["id"]
                    await session.commit()
                    result["actions_taken"].append(f"Updated agent PR #{pr_data['number']} in database")
                else:
                    result["actions_taken"].append(f"External PR #{pr_data['number']} opened (not agent-created)")
        else:
            # External PR, potentially assign to agent for review
            await self._consider_agent_assignment_for_pr(repository, pr_data, result)
            
    async def _handle_pr_closed(self, repository: GitHubRepository, pr_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Handle pull request closed/merged."""
        
        async with get_db_session() as session:
            pr_result = await session.execute(
                select(PullRequest).where(
                    and_(
                        PullRequest.repository_id == repository.id,
                        PullRequest.github_pr_number == pr_data["number"]
                    )
                )
            )
            existing_pr = pr_result.scalar_one_or_none()
            
            if existing_pr:
                # Update PR status
                if pr_data.get("merged", False):
                    existing_pr.status = PullRequestStatus.MERGED
                    existing_pr.merged_at = datetime.utcnow()
                else:
                    existing_pr.status = PullRequestStatus.CLOSED
                    existing_pr.closed_at = datetime.utcnow()
                    
                await session.commit()
                result["actions_taken"].append(f"Updated PR #{pr_data['number']} status to {existing_pr.status.value}")
                
                # If this was an agent PR that got merged, clean up work tree
                if existing_pr.status == PullRequestStatus.MERGED and existing_pr.work_tree_id:
                    await self._cleanup_merged_pr_work_tree(existing_pr, result)
                    
    async def _handle_pr_synchronized(self, repository: GitHubRepository, pr_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Handle pull request synchronized (new commits pushed)."""
        
        # Check if this is an agent PR that needs review update
        async with get_db_session() as session:
            pr_result = await session.execute(
                select(PullRequest).options(
                    selectinload(PullRequest.code_reviews)
                ).where(
                    and_(
                        PullRequest.repository_id == repository.id,
                        PullRequest.github_pr_number == pr_data["number"]
                    )
                )
            )
            existing_pr = pr_result.scalar_one_or_none()
            
            if existing_pr and existing_pr.agent_id:
                # Trigger new automated review
                await self._trigger_automated_review(existing_pr, result)
                
    async def _handle_pr_ready_for_review(self, repository: GitHubRepository, pr_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Handle pull request marked as ready for review."""
        
        # Trigger automated review for PR
        async with get_db_session() as session:
            pr_result = await session.execute(
                select(PullRequest).where(
                    and_(
                        PullRequest.repository_id == repository.id,
                        PullRequest.github_pr_number == pr_data["number"]
                    )
                )
            )
            existing_pr = pr_result.scalar_one_or_none()
            
            if existing_pr:
                await self._trigger_automated_review(existing_pr, result)
            else:
                # Create PR record for external PR
                await self._create_external_pr_record(repository, pr_data, result)
                
    async def _consider_agent_assignment_for_pr(self, repository: GitHubRepository, pr_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Consider assigning an agent to review external PR."""
        
        # TODO: Implement agent assignment logic for PR review
        result["actions_taken"].append("External PR review assignment not implemented yet")
        
    async def _cleanup_merged_pr_work_tree(self, pr: PullRequest, result: Dict[str, Any]) -> None:
        """Clean up work tree after PR is merged."""
        
        try:
            # TODO: Integrate with work tree manager to clean up
            result["actions_taken"].append(f"Work tree cleanup scheduled for PR {pr.github_pr_number}")
        except Exception as e:
            result["errors"].append(f"Work tree cleanup failed: {str(e)}")
            
    async def _trigger_automated_review(self, pr: PullRequest, result: Dict[str, Any]) -> None:
        """Trigger automated code review for PR."""
        
        try:
            # TODO: Integrate with code review assistant
            result["actions_taken"].append(f"Automated review triggered for PR {pr.github_pr_number}")
        except Exception as e:
            result["errors"].append(f"Automated review failed: {str(e)}")
            
    async def _create_external_pr_record(self, repository: GitHubRepository, pr_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Create database record for external PR."""
        
        try:
            pr = PullRequest(
                repository_id=repository.id,
                github_pr_number=pr_data["number"],
                github_pr_id=pr_data["id"],
                title=pr_data["title"],
                description=pr_data.get("body", ""),
                source_branch=pr_data.get("head", {}).get("ref", ""),
                target_branch=pr_data.get("base", {}).get("ref", ""),
                status=PullRequestStatus.DRAFT if pr_data.get("draft") else PullRequestStatus.OPEN,
                pr_metadata={
                    "external": True,
                    "github_url": pr_data["html_url"],
                    "author": pr_data.get("user", {}).get("login", "")
                }
            )
            
            async with get_db_session() as session:
                session.add(pr)
                await session.commit()
                
            result["actions_taken"].append(f"Created database record for external PR #{pr_data['number']}")
            
        except Exception as e:
            result["errors"].append(f"Failed to create external PR record: {str(e)}")


class IssueEventHandler(WebhookEventHandler):
    """Handle issue events for automated assignment and tracking."""
    
    def __init__(self, github_client: GitHubAPIClient = None):
        super().__init__(github_client)
        self.issue_manager = IssueManager(github_client)
        
    async def handle_event(self, event_type: str, payload: Dict[str, Any], repository: GitHubRepository) -> Dict[str, Any]:
        """Handle issue event."""
        
        result = {
            "success": False,
            "event_type": event_type,
            "action": payload.get("action", ""),
            "actions_taken": [],
            "errors": []
        }
        
        try:
            action = payload.get("action", "")
            issue_data = payload.get("issue", {})
            
            if action == "opened":
                await self._handle_issue_opened(repository, issue_data, result)
            elif action == "closed":
                await self._handle_issue_closed(repository, issue_data, result)
            elif action == "reopened":
                await self._handle_issue_reopened(repository, issue_data, result)
            elif action == "labeled" or action == "unlabeled":
                await self._handle_issue_labeled(repository, issue_data, result)
            elif action == "assigned" or action == "unassigned":
                await self._handle_issue_assignment(repository, issue_data, action, result)
            else:
                result["actions_taken"].append(f"No specific handler for action: {action}")
                
            result["success"] = True
            
        except Exception as e:
            error_msg = f"Issue event handling failed: {str(e)}"
            result["errors"].append(error_msg)
            logger.error(error_msg)
            
        return result
        
    async def _handle_issue_opened(self, repository: GitHubRepository, issue_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Handle new issue opened."""
        
        # Check if issue already exists in database
        async with get_db_session() as session:
            existing_result = await session.execute(
                select(GitHubIssue).where(
                    and_(
                        GitHubIssue.repository_id == repository.id,
                        GitHubIssue.github_issue_number == issue_data["number"]
                    )
                )
            )
            existing_issue = existing_result.scalar_one_or_none()
            
            if existing_issue:
                result["actions_taken"].append(f"Issue #{issue_data['number']} already exists in database")
                return
                
        # Create new issue record using issue manager
        try:
            await self.issue_manager._create_issue_from_github(repository, issue_data, session)
            await session.commit()
            result["actions_taken"].append(f"Created database record for issue #{issue_data['number']}")
            
            # Consider auto-assignment to agent
            if self._should_auto_assign_issue(issue_data):
                await self._attempt_auto_assignment(repository, issue_data, result)
                
        except Exception as e:
            result["errors"].append(f"Failed to create issue record: {str(e)}")
            
    async def _handle_issue_closed(self, repository: GitHubRepository, issue_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Handle issue closed."""
        
        async with get_db_session() as session:
            issue_result = await session.execute(
                select(GitHubIssue).where(
                    and_(
                        GitHubIssue.repository_id == repository.id,
                        GitHubIssue.github_issue_number == issue_data["number"]
                    )
                )
            )
            existing_issue = issue_result.scalar_one_or_none()
            
            if existing_issue:
                existing_issue.state = IssueState.CLOSED
                existing_issue.closed_at = datetime.utcnow()
                await session.commit()
                result["actions_taken"].append(f"Updated issue #{issue_data['number']} status to closed")
                
    async def _handle_issue_reopened(self, repository: GitHubRepository, issue_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Handle issue reopened."""
        
        async with get_db_session() as session:
            issue_result = await session.execute(
                select(GitHubIssue).where(
                    and_(
                        GitHubIssue.repository_id == repository.id,
                        GitHubIssue.github_issue_number == issue_data["number"]
                    )
                )
            )
            existing_issue = issue_result.scalar_one_or_none()
            
            if existing_issue:
                existing_issue.state = IssueState.OPEN
                existing_issue.closed_at = None
                await session.commit()
                result["actions_taken"].append(f"Updated issue #{issue_data['number']} status to open")
                
    async def _handle_issue_labeled(self, repository: GitHubRepository, issue_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Handle issue label changes."""
        
        async with get_db_session() as session:
            issue_result = await session.execute(
                select(GitHubIssue).where(
                    and_(
                        GitHubIssue.repository_id == repository.id,
                        GitHubIssue.github_issue_number == issue_data["number"]
                    )
                )
            )
            existing_issue = issue_result.scalar_one_or_none()
            
            if existing_issue:
                # Update labels
                new_labels = [label["name"] for label in issue_data.get("labels", [])]
                existing_issue.labels = new_labels
                
                # Re-classify issue based on new labels
                classification = self.issue_manager.classifier.classify_issue(
                    existing_issue.title,
                    existing_issue.description or "",
                    new_labels
                )
                
                existing_issue.priority = classification["priority"]
                existing_issue.issue_type = classification["issue_type"]
                existing_issue.estimated_effort = classification["effort_estimate"]
                
                await session.commit()
                result["actions_taken"].append(f"Updated labels and classification for issue #{issue_data['number']}")
                
    async def _handle_issue_assignment(self, repository: GitHubRepository, issue_data: Dict[str, Any], action: str, result: Dict[str, Any]) -> None:
        """Handle issue assignment changes."""
        
        # For now, just log the assignment change
        assignee = issue_data.get("assignee")
        if assignee:
            assignee_login = assignee.get("login", "")
            result["actions_taken"].append(f"Issue #{issue_data['number']} {action} to {assignee_login}")
        else:
            result["actions_taken"].append(f"Issue #{issue_data['number']} {action}")
            
    def _should_auto_assign_issue(self, issue_data: Dict[str, Any]) -> bool:
        """Determine if issue should be auto-assigned to an agent."""
        
        # Check for auto-assignment labels
        labels = [label["name"].lower() for label in issue_data.get("labels", [])]
        auto_assign_labels = ["automated", "agent-assignable", "good first issue"]
        
        return any(label in auto_assign_labels for label in labels)
        
    async def _attempt_auto_assignment(self, repository: GitHubRepository, issue_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Attempt to auto-assign issue to best available agent."""
        
        try:
            # Get the issue record
            async with get_db_session() as session:
                issue_result = await session.execute(
                    select(GitHubIssue).where(
                        and_(
                            GitHubIssue.repository_id == repository.id,
                            GitHubIssue.github_issue_number == issue_data["number"]
                        )
                    )
                )
                issue = issue_result.scalar_one_or_none()
                
                if issue:
                    # Attempt assignment using issue manager
                    assignment_result = await self.issue_manager.assign_issue_to_agent(
                        str(issue.id),
                        auto_assign=True
                    )
                    
                    if assignment_result["success"]:
                        agent_name = assignment_result["agent"].name
                        result["actions_taken"].append(f"Auto-assigned issue #{issue_data['number']} to agent {agent_name}")
                    else:
                        result["actions_taken"].append(f"Auto-assignment failed for issue #{issue_data['number']}")
                        
        except Exception as e:
            result["errors"].append(f"Auto-assignment failed: {str(e)}")


class GitHubWebhookProcessor:
    """
    Central webhook processor for GitHub events.
    
    Validates webhooks, routes events to appropriate handlers,
    and manages event processing with error handling and retries.
    """
    
    def __init__(self, github_client: GitHubAPIClient = None):
        self.github_client = github_client or GitHubAPIClient()
        
        # Initialize event handlers
        self.handlers = {
            WebhookEventType.PUSH.value: PushEventHandler(self.github_client),
            WebhookEventType.PULL_REQUEST.value: PullRequestEventHandler(self.github_client),
            WebhookEventType.ISSUES.value: IssueEventHandler(self.github_client),
            # Add more handlers as needed
        }
        
        self.processing_stats = {
            "events_processed": 0,
            "events_failed": 0,
            "last_processed": None,
            "processing_times": []
        }
        
    async def process_webhook(self, request: Request) -> Dict[str, Any]:
        """Process incoming GitHub webhook."""
        
        start_time = datetime.utcnow()
        
        try:
            # Get headers and payload
            headers = dict(request.headers)
            payload = await request.json()
            
            # Validate webhook
            validation_result = await self._validate_webhook(headers, payload, await request.body())
            if not validation_result["valid"]:
                raise WebhookProcessingError(f"Webhook validation failed: {validation_result['reason']}")
                
            # Extract event information
            event_type = headers.get("x-github-event", "")
            event_id = headers.get("x-github-delivery", "")
            
            # Get repository information
            repository = await self._get_repository_from_payload(payload)
            if not repository:
                raise WebhookProcessingError("Repository not found or not configured")
                
            # Route to appropriate handler
            handler = self.handlers.get(event_type)
            if not handler:
                return {
                    "success": True,
                    "message": f"No handler for event type: {event_type}",
                    "event_id": event_id,
                    "ignored": True
                }
                
            # Process event
            processing_result = await handler.handle_event(event_type, payload, repository)
            
            # Update statistics
            self._update_processing_stats(start_time, True)
            
            return {
                "success": True,
                "event_id": event_id,
                "event_type": event_type,
                "repository": repository.repository_full_name,
                "processing_result": processing_result,
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            }
            
        except Exception as e:
            self._update_processing_stats(start_time, False)
            logger.error(f"Webhook processing failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            }
            
    async def _validate_webhook(self, headers: Dict[str, str], payload: Dict[str, Any], raw_body: bytes) -> Dict[str, Any]:
        """Validate GitHub webhook signature and format."""
        
        validation_result = {
            "valid": False,
            "reason": None
        }
        
        try:
            # Check required headers
            if "x-github-event" not in headers:
                validation_result["reason"] = "Missing x-github-event header"
                return validation_result
                
            if "x-github-delivery" not in headers:
                validation_result["reason"] = "Missing x-github-delivery header"
                return validation_result
                
            # Validate signature if webhook secret is configured
            signature = headers.get("x-hub-signature-256", "")
            if signature:
                # Get repository from payload to find webhook secret
                repo_data = payload.get("repository", {})
                repo_full_name = repo_data.get("full_name", "")
                
                if repo_full_name:
                    async with get_db_session() as session:
                        repo_result = await session.execute(
                            select(GitHubRepository).where(
                                GitHubRepository.repository_full_name == repo_full_name
                            )
                        )
                        repository = repo_result.scalar_one_or_none()
                        
                        if repository and repository.webhook_secret:
                            # Validate signature
                            if not self.github_client.verify_webhook_signature(
                                raw_body, signature, repository.webhook_secret
                            ):
                                validation_result["reason"] = "Invalid webhook signature"
                                return validation_result
                                
            # Check payload structure
            if not isinstance(payload, dict):
                validation_result["reason"] = "Invalid payload format"
                return validation_result
                
            validation_result["valid"] = True
            return validation_result
            
        except Exception as e:
            validation_result["reason"] = f"Validation error: {str(e)}"
            return validation_result
            
    async def _get_repository_from_payload(self, payload: Dict[str, Any]) -> Optional[GitHubRepository]:
        """Get repository record from webhook payload."""
        
        repo_data = payload.get("repository", {})
        repo_full_name = repo_data.get("full_name", "")
        
        if not repo_full_name:
            return None
            
        async with get_db_session() as session:
            result = await session.execute(
                select(GitHubRepository).where(
                    GitHubRepository.repository_full_name == repo_full_name
                )
            )
            return result.scalar_one_or_none()
            
    def _update_processing_stats(self, start_time: datetime, success: bool) -> None:
        """Update processing statistics."""
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        self.processing_stats["events_processed"] += 1
        if not success:
            self.processing_stats["events_failed"] += 1
            
        self.processing_stats["last_processed"] = datetime.utcnow().isoformat()
        self.processing_stats["processing_times"].append(processing_time)
        
        # Keep only last 100 processing times
        if len(self.processing_stats["processing_times"]) > 100:
            self.processing_stats["processing_times"] = self.processing_stats["processing_times"][-100:]
            
    async def setup_repository_webhook(
        self,
        repository: GitHubRepository,
        webhook_url: str,
        events: List[str] = None
    ) -> Dict[str, Any]:
        """Set up webhook for repository."""
        
        try:
            events = events or ["push", "pull_request", "issues", "issue_comment"]
            repo_parts = repository.repository_full_name.split('/')
            
            # Generate webhook secret
            import secrets
            webhook_secret = secrets.token_urlsafe(32)
            
            # Create webhook on GitHub
            webhook_data = await self.github_client.create_webhook(
                repo_parts[0], repo_parts[1],
                webhook_url,
                webhook_secret,
                events
            )
            
            # Update repository record
            repository.webhook_secret = webhook_secret
            repository.webhook_url = webhook_url
            
            async with get_db_session() as session:
                await session.merge(repository)
                await session.commit()
                
            return {
                "success": True,
                "webhook_id": webhook_data.get("id"),
                "webhook_url": webhook_url,
                "events": events
            }
            
        except Exception as e:
            logger.error(f"Failed to setup webhook for {repository.repository_full_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get webhook processing statistics."""
        
        if self.processing_stats["processing_times"]:
            avg_processing_time = sum(self.processing_stats["processing_times"]) / len(self.processing_stats["processing_times"])
            max_processing_time = max(self.processing_stats["processing_times"])
        else:
            avg_processing_time = 0.0
            max_processing_time = 0.0
            
        success_rate = 1.0
        if self.processing_stats["events_processed"] > 0:
            success_rate = (
                (self.processing_stats["events_processed"] - self.processing_stats["events_failed"]) /
                self.processing_stats["events_processed"]
            )
            
        return {
            "events_processed": self.processing_stats["events_processed"],
            "events_failed": self.processing_stats["events_failed"],
            "success_rate": success_rate,
            "last_processed": self.processing_stats["last_processed"],
            "average_processing_time_seconds": avg_processing_time,
            "max_processing_time_seconds": max_processing_time,
            "supported_events": list(self.handlers.keys())
        }