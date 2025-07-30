"""
GitHub Integration models for LeanVibe Agent Hive 2.0

Comprehensive GitHub integration supporting multi-agent development workflows,
version control, and automated code review processes.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from sqlalchemy import Column, String, Text, DateTime, JSON, Enum as SQLEnum, Boolean, Integer, BigInteger, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey, UniqueConstraint, Index

from ..core.database import Base
from ..core.database_types import DatabaseAgnosticUUID, UUIDArray, StringArray


class WorkTreeStatus(Enum):
    """Work tree status for agent isolation."""
    ACTIVE = "active"
    CLEANING = "cleaning"
    ARCHIVED = "archived"
    ERROR = "error"


class PullRequestStatus(Enum):
    """Pull request status."""
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"
    DRAFT = "draft"


class IssueState(Enum):
    """GitHub issue state."""
    OPEN = "open"
    CLOSED = "closed"


class ReviewStatus(Enum):
    """Code review status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class BranchOperationType(Enum):
    """Branch operation types."""
    MERGE = "merge"
    REBASE = "rebase"
    CHERRY_PICK = "cherry_pick"
    SYNC = "sync"
    CREATE = "create"
    DELETE = "delete"


class GitHubRepository(Base):
    """
    GitHub repository configuration and access management.
    
    Stores repository metadata, permissions, and access configuration
    for multi-agent development workflows.
    """
    
    __tablename__ = "github_repositories"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    repository_full_name = Column(String(255), nullable=False, unique=True, index=True)  # "owner/repo"
    repository_url = Column(String(500), nullable=False)
    clone_url = Column(String(500), nullable=True)  # SSH/HTTPS clone URL
    
    # Repository configuration
    default_branch = Column(String(100), nullable=False, default="main")
    agent_permissions = Column(JSON, nullable=True, default=dict)  # {"read": true, "write": true, "issues": true}
    repository_config = Column(JSON, nullable=True, default=dict)
    
    # Webhook and security
    webhook_secret = Column(String(255), nullable=True)
    webhook_url = Column(String(500), nullable=True)
    access_token_hash = Column(String(255), nullable=True)  # Encrypted token
    
    # Sync status
    last_sync = Column(DateTime(timezone=True), nullable=True)
    sync_status = Column(String(50), nullable=False, default="pending")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    work_trees = relationship("AgentWorkTree", back_populates="repository", cascade="all, delete-orphan")
    pull_requests = relationship("PullRequest", back_populates="repository", cascade="all, delete-orphan")
    issues = relationship("GitHubIssue", back_populates="repository", cascade="all, delete-orphan")
    commits = relationship("GitCommit", back_populates="repository", cascade="all, delete-orphan")
    branch_operations = relationship("BranchOperation", back_populates="repository", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<GitHubRepository(id={self.id}, name='{self.repository_full_name}', status='{self.sync_status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert repository to dictionary for serialization."""
        return {
            "id": str(self.id),
            "repository_full_name": self.repository_full_name,
            "repository_url": self.repository_url,
            "clone_url": self.clone_url,
            "default_branch": self.default_branch,
            "agent_permissions": self.agent_permissions,
            "repository_config": self.repository_config,
            "sync_status": self.sync_status,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def has_permission(self, permission: str) -> bool:
        """Check if repository allows specific permission."""
        return self.agent_permissions.get(permission, False) if self.agent_permissions else False
    
    def get_clone_url_for_agent(self, with_token: bool = False) -> str:
        """Get appropriate clone URL for agent access."""
        if with_token and self.access_token_hash:
            # Return authenticated HTTPS URL (token would be decrypted)
            return self.clone_url or self.repository_url
        return self.clone_url or self.repository_url


class AgentWorkTree(Base):
    """
    Isolated development environment for each agent.
    
    Provides 100% isolation between agents working on the same repository
    through separate work trees and branch management.
    """
    
    __tablename__ = "agent_work_trees"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    agent_id = Column(DatabaseAgnosticUUID(), ForeignKey('agents.id', ondelete='CASCADE'), nullable=False, index=True)
    repository_id = Column(DatabaseAgnosticUUID(), ForeignKey('github_repositories.id', ondelete='CASCADE'), 
                          nullable=False, index=True)
    
    # Work tree configuration
    work_tree_path = Column(String(500), nullable=False, unique=True)
    branch_name = Column(String(255), nullable=False, index=True)
    base_branch = Column(String(255), nullable=True)  # Branch this work tree is based on
    upstream_branch = Column(String(255), nullable=True)  # Remote tracking branch
    
    # Status and isolation
    status = Column(SQLEnum(WorkTreeStatus), nullable=False, default=WorkTreeStatus.ACTIVE, index=True)
    isolation_config = Column(JSON, nullable=True, default=dict)
    
    # Git state tracking
    last_commit_hash = Column(String(40), nullable=True)
    uncommitted_changes = Column(Boolean, nullable=False, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used = Column(DateTime(timezone=True), server_default=func.now())
    cleaned_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    agent = relationship("Agent", backref="work_trees")
    repository = relationship("GitHubRepository", back_populates="work_trees")
    pull_requests = relationship("PullRequest", back_populates="work_tree")
    commits = relationship("GitCommit", back_populates="work_tree")
    branch_operations = relationship("BranchOperation", back_populates="work_tree")
    
    # Unique constraint: one work tree per agent per repository
    __table_args__ = (
        UniqueConstraint('agent_id', 'repository_id', name='uq_agent_work_tree_per_repo'),
        Index('idx_work_tree_status_used', 'status', 'last_used'),
    )
    
    def __repr__(self) -> str:
        return f"<AgentWorkTree(id={self.id}, agent_id={self.agent_id}, branch='{self.branch_name}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert work tree to dictionary for serialization."""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "repository_id": str(self.repository_id),
            "work_tree_path": self.work_tree_path,
            "branch_name": self.branch_name,
            "base_branch": self.base_branch,
            "upstream_branch": self.upstream_branch,
            "status": self.status.value,
            "isolation_config": self.isolation_config,
            "last_commit_hash": self.last_commit_hash,
            "uncommitted_changes": self.uncommitted_changes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "cleaned_at": self.cleaned_at.isoformat() if self.cleaned_at else None,
        }
    
    def is_active(self) -> bool:
        """Check if work tree is active and available."""
        return self.status == WorkTreeStatus.ACTIVE
    
    def needs_cleanup(self) -> bool:
        """Check if work tree needs cleanup based on inactivity."""
        if not self.last_used:
            return False
        
        # Consider cleanup if unused for more than 7 days
        inactive_days = (datetime.utcnow() - self.last_used).days
        return inactive_days > 7
    
    def update_activity(self) -> None:
        """Update last used timestamp."""
        self.last_used = datetime.utcnow()


class PullRequest(Base):
    """
    Pull request management and automation.
    
    Tracks pull request lifecycle, review status, and integration
    with CI/CD pipelines for automated development workflows.
    """
    
    __tablename__ = "pull_requests"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    repository_id = Column(DatabaseAgnosticUUID(), ForeignKey('github_repositories.id', ondelete='CASCADE'), 
                          nullable=False, index=True)
    agent_id = Column(DatabaseAgnosticUUID(), ForeignKey('agents.id', ondelete='CASCADE'), 
                     nullable=False, index=True)
    work_tree_id = Column(DatabaseAgnosticUUID(), ForeignKey('agent_work_trees.id', ondelete='SET NULL'), 
                         nullable=True, index=True)
    
    # GitHub PR identification
    github_pr_number = Column(Integer, nullable=False, index=True)
    github_pr_id = Column(BigInteger, nullable=True)  # GitHub's internal PR ID
    
    # PR content
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    source_branch = Column(String(255), nullable=False)
    target_branch = Column(String(255), nullable=False)
    
    # PR status and state
    status = Column(SQLEnum(PullRequestStatus), nullable=False, default=PullRequestStatus.OPEN, index=True)
    mergeable = Column(Boolean, nullable=True)
    conflicts = Column(JSON, nullable=True, default=list)
    
    # Review and CI status
    review_status = Column(String(50), nullable=True)  # 'pending', 'approved', 'changes_requested'
    ci_status = Column(String(50), nullable=True)  # 'pending', 'success', 'failure'
    
    # Metadata
    labels = Column(JSON, nullable=True, default=list)
    reviewers = Column(JSON, nullable=True, default=list)
    pr_metadata = Column(JSON, nullable=True, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    merged_at = Column(DateTime(timezone=True), nullable=True)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    repository = relationship("GitHubRepository", back_populates="pull_requests")
    agent = relationship("Agent", backref="pull_requests")
    work_tree = relationship("AgentWorkTree", back_populates="pull_requests")
    code_reviews = relationship("CodeReview", back_populates="pull_request", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<PullRequest(id={self.id}, pr_number={self.github_pr_number}, status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pull request to dictionary for serialization."""
        return {
            "id": str(self.id),
            "repository_id": str(self.repository_id),
            "agent_id": str(self.agent_id),
            "work_tree_id": str(self.work_tree_id) if self.work_tree_id else None,
            "github_pr_number": self.github_pr_number,
            "github_pr_id": self.github_pr_id,
            "title": self.title,
            "description": self.description,
            "source_branch": self.source_branch,
            "target_branch": self.target_branch,
            "status": self.status.value,
            "mergeable": self.mergeable,
            "conflicts": self.conflicts,
            "review_status": self.review_status,
            "ci_status": self.ci_status,
            "labels": self.labels,
            "reviewers": self.reviewers,
            "pr_metadata": self.pr_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "merged_at": self.merged_at.isoformat() if self.merged_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
        }
    
    def is_open(self) -> bool:
        """Check if pull request is open."""
        return self.status == PullRequestStatus.OPEN
    
    def is_mergeable(self) -> bool:
        """Check if pull request can be merged."""
        return (
            self.is_open() and 
            self.mergeable and 
            self.review_status == 'approved' and 
            self.ci_status == 'success'
        )
    
    def has_conflicts(self) -> bool:
        """Check if pull request has merge conflicts."""
        return bool(self.conflicts and len(self.conflicts) > 0)


class GitHubIssue(Base):
    """
    GitHub issue tracking and agent assignment.
    
    Manages bi-directional issue synchronization and intelligent
    assignment to agents based on capabilities and workload.
    """
    
    __tablename__ = "github_issues"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    repository_id = Column(DatabaseAgnosticUUID(), ForeignKey('github_repositories.id', ondelete='CASCADE'), 
                          nullable=False, index=True)
    
    # GitHub issue identification
    github_issue_number = Column(Integer, nullable=False, index=True)
    github_issue_id = Column(BigInteger, nullable=True)
    
    # Issue content
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    labels = Column(JSON, nullable=True, default=list)  # ["bug", "enhancement", "priority:high"]
    
    # Assignment
    assignee_agent_id = Column(DatabaseAgnosticUUID(), ForeignKey('agents.id', ondelete='SET NULL'), 
                              nullable=True, index=True)
    assignee_github_username = Column(String(255), nullable=True)
    
    # Issue state and classification
    state = Column(SQLEnum(IssueState), nullable=False, default=IssueState.OPEN, index=True)
    priority = Column(String(20), nullable=True, index=True)
    issue_type = Column(String(50), nullable=True, index=True)  # 'bug', 'feature', 'enhancement'
    
    # Effort tracking
    estimated_effort = Column(Integer, nullable=True)  # Story points or hours
    actual_effort = Column(Integer, nullable=True)
    milestone = Column(String(255), nullable=True)
    
    # Metadata and progress
    issue_metadata = Column(JSON, nullable=True, default=dict)
    progress_updates = Column(JSON, nullable=True, default=list)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    assigned_at = Column(DateTime(timezone=True), nullable=True)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    repository = relationship("GitHubRepository", back_populates="issues")
    assignee_agent = relationship("Agent", backref="assigned_issues")
    
    def __repr__(self) -> str:
        return f"<GitHubIssue(id={self.id}, issue_number={self.github_issue_number}, state='{self.state}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary for serialization."""
        return {
            "id": str(self.id),
            "repository_id": str(self.repository_id),
            "github_issue_number": self.github_issue_number,
            "github_issue_id": self.github_issue_id,
            "title": self.title,
            "description": self.description,
            "labels": self.labels,
            "assignee_agent_id": str(self.assignee_agent_id) if self.assignee_agent_id else None,
            "assignee_github_username": self.assignee_github_username,
            "state": self.state.value,
            "priority": self.priority,
            "issue_type": self.issue_type,
            "estimated_effort": self.estimated_effort,
            "actual_effort": self.actual_effort,
            "milestone": self.milestone,
            "issue_metadata": self.issue_metadata,
            "progress_updates": self.progress_updates,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
        }
    
    def is_open(self) -> bool:
        """Check if issue is open."""
        return self.state == IssueState.OPEN
    
    def is_assigned(self) -> bool:
        """Check if issue is assigned to an agent."""
        return self.assignee_agent_id is not None
    
    def add_progress_update(self, status: str, comment: str, agent_id: str) -> None:
        """Add progress update to issue."""
        if not self.progress_updates:
            self.progress_updates = []
        
        update = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            "comment": comment,
            "agent_id": agent_id
        }
        self.progress_updates.append(update)
    
    def get_priority_score(self) -> int:
        """Get numerical priority score for assignment algorithms."""
        priority_scores = {
            "critical": 5,
            "high": 4,
            "medium": 3,
            "low": 2,
            "very_low": 1
        }
        return priority_scores.get(self.priority, 3)


class CodeReview(Base):
    """
    Automated code review and analysis.
    
    Provides comprehensive code review including security, performance,
    and style analysis with detailed findings and suggestions.
    """
    
    __tablename__ = "code_reviews"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    pull_request_id = Column(DatabaseAgnosticUUID(), ForeignKey('pull_requests.id', ondelete='CASCADE'), 
                            nullable=False, index=True)
    reviewer_agent_id = Column(DatabaseAgnosticUUID(), ForeignKey('agents.id', ondelete='SET NULL'), 
                              nullable=True, index=True)
    
    # Review configuration
    reviewer_type = Column(String(50), nullable=False)  # 'agent', 'human', 'automated'
    review_type = Column(String(50), nullable=False)  # 'security', 'performance', 'style', 'comprehensive'
    review_status = Column(SQLEnum(ReviewStatus), nullable=False, default=ReviewStatus.PENDING)
    
    # Review findings
    findings = Column(JSON, nullable=True, default=list)  # Detailed review findings and suggestions
    security_issues = Column(JSON, nullable=True, default=list)
    performance_issues = Column(JSON, nullable=True, default=list)
    style_issues = Column(JSON, nullable=True, default=list)
    suggestions = Column(JSON, nullable=True, default=list)
    
    # Review outcome
    overall_score = Column(Float, nullable=True)  # 0.0 to 1.0
    approved = Column(Boolean, nullable=False, default=False)
    changes_requested = Column(Boolean, nullable=False, default=False)
    review_metadata = Column(JSON, nullable=True, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    pull_request = relationship("PullRequest", back_populates="code_reviews")
    reviewer_agent = relationship("Agent", backref="code_reviews")
    
    def __repr__(self) -> str:
        return f"<CodeReview(id={self.id}, type='{self.review_type}', status='{self.review_status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert code review to dictionary for serialization."""
        return {
            "id": str(self.id),
            "pull_request_id": str(self.pull_request_id),
            "reviewer_agent_id": str(self.reviewer_agent_id) if self.reviewer_agent_id else None,
            "reviewer_type": self.reviewer_type,
            "review_type": self.review_type,
            "review_status": self.review_status.value,
            "findings": self.findings,
            "security_issues": self.security_issues,
            "performance_issues": self.performance_issues,
            "style_issues": self.style_issues,
            "suggestions": self.suggestions,
            "overall_score": self.overall_score,
            "approved": self.approved,
            "changes_requested": self.changes_requested,
            "review_metadata": self.review_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
    
    def is_completed(self) -> bool:
        """Check if review is completed."""
        return self.review_status == ReviewStatus.COMPLETED
    
    def get_issue_count(self) -> int:
        """Get total count of issues found."""
        return (
            len(self.security_issues or []) +
            len(self.performance_issues or []) +
            len(self.style_issues or [])
        )
    
    def start_review(self) -> None:
        """Mark review as started."""
        self.review_status = ReviewStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()
    
    def complete_review(self, approved: bool = False) -> None:
        """Mark review as completed."""
        self.review_status = ReviewStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.approved = approved
        self.changes_requested = not approved and self.get_issue_count() > 0


class GitCommit(Base):
    """
    Git commit tracking and attribution.
    
    Tracks all commits made by agents with detailed metadata
    for audit trails and contribution analysis.
    """
    
    __tablename__ = "git_commits"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    repository_id = Column(DatabaseAgnosticUUID(), ForeignKey('github_repositories.id', ondelete='CASCADE'), 
                          nullable=False, index=True)
    agent_id = Column(DatabaseAgnosticUUID(), ForeignKey('agents.id', ondelete='SET NULL'), 
                     nullable=True, index=True)
    work_tree_id = Column(DatabaseAgnosticUUID(), ForeignKey('agent_work_trees.id', ondelete='SET NULL'), 
                         nullable=True, index=True)
    
    # Git commit identification
    commit_hash = Column(String(40), nullable=False, index=True)
    short_hash = Column(String(10), nullable=True)
    branch_name = Column(String(255), nullable=True, index=True)
    
    # Commit content
    commit_message = Column(Text, nullable=True)
    commit_message_body = Column(Text, nullable=True)
    
    # Author and committer information
    author_name = Column(String(255), nullable=True)
    author_email = Column(String(255), nullable=True)
    committer_name = Column(String(255), nullable=True)
    committer_email = Column(String(255), nullable=True)
    
    # Commit statistics
    files_changed = Column(Integer, nullable=True)
    lines_added = Column(Integer, nullable=True)
    lines_deleted = Column(Integer, nullable=True)
    
    # Git metadata
    parent_hashes = Column(JSON, nullable=True, default=list)
    is_merge = Column(Boolean, nullable=False, default=False)
    commit_metadata = Column(JSON, nullable=True, default=dict)
    
    # Timestamps
    committed_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    repository = relationship("GitHubRepository", back_populates="commits")
    agent = relationship("Agent", backref="commits")
    work_tree = relationship("AgentWorkTree", back_populates="commits")
    
    # Unique constraint: one commit per repository
    __table_args__ = (
        UniqueConstraint('repository_id', 'commit_hash', name='uq_commit_per_repo'),
        Index('idx_commit_agent_date', 'agent_id', 'committed_at'),
    )
    
    def __repr__(self) -> str:
        return f"<GitCommit(id={self.id}, hash='{self.short_hash}', agent_id={self.agent_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert commit to dictionary for serialization."""
        return {
            "id": str(self.id),
            "repository_id": str(self.repository_id),
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "work_tree_id": str(self.work_tree_id) if self.work_tree_id else None,
            "commit_hash": self.commit_hash,
            "short_hash": self.short_hash,
            "branch_name": self.branch_name,
            "commit_message": self.commit_message,
            "commit_message_body": self.commit_message_body,
            "author_name": self.author_name,
            "author_email": self.author_email,
            "committer_name": self.committer_name,
            "committer_email": self.committer_email,
            "files_changed": self.files_changed,
            "lines_added": self.lines_added,
            "lines_deleted": self.lines_deleted,
            "parent_hashes": self.parent_hashes,
            "is_merge": self.is_merge,
            "commit_metadata": self.commit_metadata,
            "committed_at": self.committed_at.isoformat() if self.committed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    def get_change_summary(self) -> str:
        """Get human-readable change summary."""
        files = self.files_changed or 0
        added = self.lines_added or 0
        deleted = self.lines_deleted or 0
        
        return f"{files} files changed, {added} insertions(+), {deleted} deletions(-)"


class BranchOperation(Base):
    """
    Branch management and conflict resolution tracking.
    
    Tracks branch operations, merge conflicts, and resolution
    strategies for automated branch management.
    """
    
    __tablename__ = "branch_operations"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    repository_id = Column(DatabaseAgnosticUUID(), ForeignKey('github_repositories.id', ondelete='CASCADE'), 
                          nullable=False, index=True)
    agent_id = Column(DatabaseAgnosticUUID(), ForeignKey('agents.id', ondelete='SET NULL'), 
                     nullable=True, index=True)
    work_tree_id = Column(DatabaseAgnosticUUID(), ForeignKey('agent_work_trees.id', ondelete='SET NULL'), 
                         nullable=True, index=True)
    
    # Operation details
    operation_type = Column(SQLEnum(BranchOperationType), nullable=False, index=True)
    source_branch = Column(String(255), nullable=True)
    target_branch = Column(String(255), nullable=True)
    status = Column(String(50), nullable=False, default="pending")  # 'pending', 'in_progress', 'completed', 'failed'
    
    # Conflict tracking
    conflicts_detected = Column(Integer, nullable=False, default=0)
    conflicts_resolved = Column(Integer, nullable=False, default=0)
    conflict_details = Column(JSON, nullable=True, default=list)
    resolution_strategy = Column(String(100), nullable=True)
    
    # Operation result
    operation_result = Column(JSON, nullable=True, default=dict)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    repository = relationship("GitHubRepository", back_populates="branch_operations")
    agent = relationship("Agent", backref="branch_operations")
    work_tree = relationship("AgentWorkTree", back_populates="branch_operations")
    
    def __repr__(self) -> str:
        return f"<BranchOperation(id={self.id}, type='{self.operation_type}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert branch operation to dictionary for serialization."""
        return {
            "id": str(self.id),
            "repository_id": str(self.repository_id),
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "work_tree_id": str(self.work_tree_id) if self.work_tree_id else None,
            "operation_type": self.operation_type.value,
            "source_branch": self.source_branch,
            "target_branch": self.target_branch,
            "status": self.status,
            "conflicts_detected": self.conflicts_detected,
            "conflicts_resolved": self.conflicts_resolved,
            "conflict_details": self.conflict_details,
            "resolution_strategy": self.resolution_strategy,
            "operation_result": self.operation_result,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
    
    def is_completed(self) -> bool:
        """Check if operation is completed."""
        return self.status == "completed"
    
    def has_conflicts(self) -> bool:
        """Check if operation detected conflicts."""
        return self.conflicts_detected > 0
    
    def is_successful(self) -> bool:
        """Check if operation completed successfully."""
        return self.is_completed() and not self.error_message
    
    def start_operation(self) -> None:
        """Mark operation as started."""
        self.status = "in_progress"
        self.started_at = datetime.utcnow()
    
    def complete_operation(self, success: bool = True, error_message: str = None) -> None:
        """Mark operation as completed."""
        self.status = "completed" if success else "failed"
        self.completed_at = datetime.utcnow()
        if error_message:
            self.error_message = error_message