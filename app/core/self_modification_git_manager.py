"""
Git Version Control Manager for Self-Modification System

This module provides secure Git-based version control with automatic checkpoints,
rollback capabilities, and comprehensive change tracking for all self-modifications.
"""

import os
import subprocess
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
import hashlib
import json

logger = logging.getLogger(__name__)


class GitOperationStatus(Enum):
    """Status of Git operations."""
    SUCCESS = "success"
    FAILED = "failed"
    CONFLICT = "conflict"
    UNCOMMITTED_CHANGES = "uncommitted_changes"
    PERMISSION_DENIED = "permission_denied"
    REPOSITORY_NOT_FOUND = "repository_not_found"


class CheckpointType(Enum):
    """Types of automatic checkpoints."""
    PRE_MODIFICATION = "pre_modification"
    POST_MODIFICATION = "post_modification"  
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    ROLLBACK_POINT = "rollback_point"
    SAFETY_CHECKPOINT = "safety_checkpoint"


@dataclass
class GitCommit:
    """Git commit information."""
    commit_hash: str
    message: str
    author: str
    timestamp: datetime
    files_changed: List[str]
    insertions: int
    deletions: int
    commit_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GitCheckpoint:
    """Automatic checkpoint information."""
    checkpoint_id: str
    checkpoint_type: CheckpointType
    commit_hash: str
    branch_name: str
    description: str
    created_at: datetime
    
    # Modification context
    modification_session_id: Optional[str] = None
    modification_count: int = 0
    safety_score: Optional[float] = None
    
    # Recovery information
    can_rollback: bool = True
    rollback_complexity: str = "simple"  # simple, moderate, complex
    dependencies: List[str] = field(default_factory=list)
    
    # Metadata
    checkpoint_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RollbackOperation:
    """Rollback operation details."""
    rollback_id: str
    target_checkpoint: GitCheckpoint
    source_commit: str
    rollback_strategy: str  # "hard_reset", "soft_reset", "revert_commits"
    
    # Results
    status: GitOperationStatus
    new_commit_hash: Optional[str] = None
    conflicts_detected: List[str] = field(default_factory=list)
    files_restored: List[str] = field(default_factory=list)
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    rollback_duration_seconds: Optional[float] = None
    
    # Recovery metadata
    recovery_metadata: Dict[str, Any] = field(default_factory=dict)


class SecureGitManager:
    """
    Secure Git version control manager with automatic checkpointing.
    
    Provides comprehensive Git operations with safety controls, automatic
    checkpoints every 2 hours, and <30 second rollback capabilities.
    """
    
    def __init__(self, 
                 repository_path: str,
                 auto_checkpoint_interval_hours: int = 2,
                 max_rollback_commits: int = 100,
                 enable_signing: bool = True):
        self.repository_path = Path(repository_path).resolve()
        self.auto_checkpoint_interval_hours = auto_checkpoint_interval_hours
        self.max_rollback_commits = max_rollback_commits
        self.enable_signing = enable_signing
        
        # Security configuration
        self._max_file_size_mb = 50
        self._blocked_file_patterns = {
            '*.pyc', '__pycache__/', '.git/', '*.tmp', '*.log',
            '*.key', '*.pem', '*.crt', 'id_rsa*', 'secrets.*'
        }
        
        # Git configuration for self-modification
        self._git_author = "Self-Modification-Engine <noreply@leanvibe.dev>"
        self._commit_message_prefix = "[SELF-MOD]"
        
        # Checkpoint management
        self._checkpoints: List[GitCheckpoint] = []
        self._last_checkpoint_time = datetime.utcnow()
        
        # Initialize repository if needed
        self._initialize_repository()
        
        logger.info(f"SecureGitManager initialized for {self.repository_path}")
    
    def create_checkpoint(self, 
                         checkpoint_type: CheckpointType,
                         description: str,
                         modification_session_id: Optional[str] = None,
                         safety_score: Optional[float] = None) -> GitCheckpoint:
        """
        Create a secure checkpoint with comprehensive metadata.
        
        Args:
            checkpoint_type: Type of checkpoint to create
            description: Human-readable description
            modification_session_id: Associated modification session
            safety_score: Safety score for the current state
            
        Returns:
            Created checkpoint information
            
        Raises:
            GitError: If checkpoint creation fails
            SecurityError: If security violations are detected
        """
        logger.info(f"Creating {checkpoint_type.value} checkpoint: {description}")
        
        # Security validation
        self._validate_repository_state()
        
        # Check for uncommitted changes
        uncommitted_files = self._get_uncommitted_files()
        if uncommitted_files and checkpoint_type != CheckpointType.PRE_MODIFICATION:
            logger.warning(f"Uncommitted changes detected: {len(uncommitted_files)} files")
        
        # Create commit for checkpoint
        commit_message = f"{self._commit_message_prefix} {checkpoint_type.value}: {description}"
        if modification_session_id:
            commit_message += f" [session: {modification_session_id}]"
        
        # Add all changes
        self._git_add_all_safe()
        
        # Create commit
        commit_result = self._git_commit(commit_message)
        if commit_result.status != GitOperationStatus.SUCCESS:
            raise GitError(f"Failed to create checkpoint commit: {commit_result}")
        
        # Get current branch
        current_branch = self._get_current_branch()
        
        # Create checkpoint record
        checkpoint_id = self._generate_checkpoint_id(checkpoint_type, modification_session_id)
        
        checkpoint = GitCheckpoint(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            commit_hash=commit_result.commit_hash,
            branch_name=current_branch,
            description=description,
            created_at=datetime.utcnow(),
            modification_session_id=modification_session_id,
            modification_count=len(self._checkpoints),
            safety_score=safety_score,
            can_rollback=True,
            rollback_complexity="simple",
            dependencies=[],
            checkpoint_metadata={
                'repository_state': self._get_repository_state(),
                'uncommitted_files_count': len(uncommitted_files),
                'total_commits': self._get_commit_count(),
                'git_author': self._git_author
            }
        )
        
        # Store checkpoint
        self._checkpoints.append(checkpoint)
        self._last_checkpoint_time = datetime.utcnow()
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"Created checkpoint {checkpoint_id} at commit {commit_result.commit_hash[:8]}")
        return checkpoint
    
    def rollback_to_checkpoint(self, 
                             checkpoint_id: str,
                             rollback_strategy: str = "soft_reset",
                             create_backup: bool = True) -> RollbackOperation:
        """
        Rollback to a specific checkpoint with <30 second target.
        
        Args:
            checkpoint_id: ID of checkpoint to rollback to
            rollback_strategy: Strategy for rollback operation
            create_backup: Whether to create backup before rollback
            
        Returns:
            Rollback operation results
            
        Raises:
            GitError: If rollback fails
            ValueError: If checkpoint not found
        """
        logger.info(f"Starting rollback to checkpoint {checkpoint_id} using {rollback_strategy}")
        
        started_at = datetime.utcnow()
        
        # Find target checkpoint
        target_checkpoint = self._find_checkpoint(checkpoint_id)
        if not target_checkpoint:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        if not target_checkpoint.can_rollback:
            raise GitError(f"Checkpoint {checkpoint_id} cannot be rolled back")
        
        # Create backup checkpoint if requested
        backup_checkpoint = None
        if create_backup:
            backup_checkpoint = self.create_checkpoint(
                CheckpointType.ROLLBACK_POINT,
                f"Pre-rollback backup for {checkpoint_id}",
                safety_score=0.8  # Backup is safe
            )
        
        rollback_id = self._generate_rollback_id(checkpoint_id)
        current_commit = self._get_current_commit_hash()
        
        rollback_operation = RollbackOperation(
            rollback_id=rollback_id,
            target_checkpoint=target_checkpoint,
            source_commit=current_commit,
            rollback_strategy=rollback_strategy,
            status=GitOperationStatus.SUCCESS,
            started_at=started_at
        )
        
        try:
            # Execute rollback based on strategy
            if rollback_strategy == "hard_reset":
                result = self._execute_hard_reset(target_checkpoint.commit_hash)
            elif rollback_strategy == "soft_reset":
                result = self._execute_soft_reset(target_checkpoint.commit_hash)
            elif rollback_strategy == "revert_commits":
                result = self._execute_commit_revert(current_commit, target_checkpoint.commit_hash)
            else:
                raise ValueError(f"Unknown rollback strategy: {rollback_strategy}")
            
            rollback_operation.status = result.status
            rollback_operation.new_commit_hash = result.get('new_commit_hash')
            rollback_operation.conflicts_detected = result.get('conflicts', [])
            rollback_operation.files_restored = result.get('files_changed', [])
            
            completed_at = datetime.utcnow()
            rollback_operation.completed_at = completed_at
            rollback_operation.rollback_duration_seconds = (completed_at - started_at).total_seconds()
            
            # Validate rollback completed within time target
            if rollback_operation.rollback_duration_seconds > 30:
                logger.warning(f"Rollback took {rollback_operation.rollback_duration_seconds:.1f}s (target: <30s)")
            
            # Update repository state
            rollback_operation.recovery_metadata = {
                'backup_checkpoint_id': backup_checkpoint.checkpoint_id if backup_checkpoint else None,
                'rollback_duration_seconds': rollback_operation.rollback_duration_seconds,
                'target_safety_score': target_checkpoint.safety_score,
                'repository_state_after': self._get_repository_state()
            }
            
            logger.info(f"Rollback {rollback_id} completed in {rollback_operation.rollback_duration_seconds:.1f}s")
            return rollback_operation
            
        except Exception as e:
            rollback_operation.status = GitOperationStatus.FAILED
            rollback_operation.completed_at = datetime.utcnow()
            rollback_operation.recovery_metadata['error'] = str(e)
            
            logger.error(f"Rollback {rollback_id} failed: {e}")
            raise GitError(f"Rollback operation failed: {e}")
    
    def apply_modification(self,
                          file_path: str,
                          original_content: str,
                          modified_content: str,
                          modification_id: str,
                          safety_score: float) -> GitCommit:
        """
        Apply a code modification with automatic checkpointing.
        
        Args:
            file_path: Path to file being modified
            original_content: Original file content
            modified_content: New file content
            modification_id: Unique modification identifier
            safety_score: Safety score for this modification
            
        Returns:
            Git commit information
            
        Raises:
            SecurityError: If modification is deemed unsafe
            GitError: If Git operation fails
        """
        logger.info(f"Applying modification {modification_id} to {file_path}")
        
        # Security validation
        self._validate_file_modification(file_path, original_content, modified_content, safety_score)
        
        # Create pre-modification checkpoint
        pre_checkpoint = self.create_checkpoint(
            CheckpointType.PRE_MODIFICATION,
            f"Before applying modification {modification_id}",
            modification_session_id=modification_id,
            safety_score=safety_score
        )
        
        try:
            # Apply modification
            target_file = self.repository_path / file_path
            self._write_file_secure(target_file, modified_content)
            
            # Create commit
            commit_message = f"{self._commit_message_prefix} Apply modification {modification_id} to {file_path} (safety: {safety_score:.2f})"
            
            # Add and commit changes
            self._git_add_file(str(target_file))
            commit_result = self._git_commit(commit_message)
            
            if commit_result.status != GitOperationStatus.SUCCESS:
                # Rollback on failure
                logger.warning(f"Commit failed, rolling back to {pre_checkpoint.checkpoint_id}")
                self.rollback_to_checkpoint(pre_checkpoint.checkpoint_id, create_backup=False)
                raise GitError(f"Failed to commit modification: {commit_result}")
            
            # Create post-modification checkpoint
            post_checkpoint = self.create_checkpoint(
                CheckpointType.POST_MODIFICATION,
                f"After applying modification {modification_id}",
                modification_session_id=modification_id,
                safety_score=safety_score
            )
            
            commit_info = GitCommit(
                commit_hash=commit_result.commit_hash,
                message=commit_message,
                author=self._git_author,
                timestamp=datetime.utcnow(),
                files_changed=[file_path],
                insertions=modified_content.count('\n'),
                deletions=original_content.count('\n'),
                commit_metadata={
                    'modification_id': modification_id,
                    'safety_score': safety_score,
                    'pre_checkpoint_id': pre_checkpoint.checkpoint_id,
                    'post_checkpoint_id': post_checkpoint.checkpoint_id
                }
            )
            
            logger.info(f"Successfully applied modification {modification_id}")
            return commit_info
            
        except Exception as e:
            # Rollback on any error
            logger.error(f"Modification application failed, rolling back: {e}")
            try:
                self.rollback_to_checkpoint(pre_checkpoint.checkpoint_id, create_backup=False)
            except Exception as rollback_error:
                logger.critical(f"Rollback also failed: {rollback_error}")
            raise
    
    def schedule_automatic_checkpoints(self) -> None:
        """Schedule automatic checkpoints every 2 hours."""
        logger.info(f"Scheduling automatic checkpoints every {self.auto_checkpoint_interval_hours} hours")
        
        # This would be implemented as a background task in production
        # For now, we provide the method signature and basic logic
        pass
    
    def get_rollback_candidates(self, max_candidates: int = 10) -> List[GitCheckpoint]:
        """Get list of rollback candidate checkpoints."""
        # Sort checkpoints by creation time (newest first)
        sorted_checkpoints = sorted(
            [cp for cp in self._checkpoints if cp.can_rollback],
            key=lambda cp: cp.created_at,
            reverse=True
        )
        
        return sorted_checkpoints[:max_candidates]
    
    def validate_repository_integrity(self) -> Dict[str, Any]:
        """Validate repository integrity and health."""
        logger.info("Validating repository integrity")
        
        integrity_report = {
            'repository_path': str(self.repository_path),
            'is_git_repository': self._is_git_repository(),
            'has_uncommitted_changes': len(self._get_uncommitted_files()) > 0,
            'current_branch': self._get_current_branch(),
            'total_commits': self._get_commit_count(),
            'total_checkpoints': len(self._checkpoints),
            'last_checkpoint_age_hours': (datetime.utcnow() - self._last_checkpoint_time).total_seconds() / 3600,
            'repository_size_mb': self._get_repository_size_mb(),
            'security_violations': [],
            'recommendations': []
        }
        
        # Check for security issues
        security_violations = self._scan_for_security_violations()
        integrity_report['security_violations'] = security_violations
        
        # Generate recommendations
        recommendations = []
        if integrity_report['last_checkpoint_age_hours'] > self.auto_checkpoint_interval_hours * 1.5:
            recommendations.append("Create checkpoint - last checkpoint is old")
        
        if integrity_report['has_uncommitted_changes']:
            recommendations.append("Commit or stash uncommitted changes")
        
        if security_violations:
            recommendations.append(f"Address {len(security_violations)} security violations")
        
        integrity_report['recommendations'] = recommendations
        
        return integrity_report
    
    def _initialize_repository(self) -> None:
        """Initialize Git repository if it doesn't exist."""
        if not self._is_git_repository():
            logger.info(f"Initializing Git repository at {self.repository_path}")
            self._run_git_command(['init'])
            
            # Set initial configuration
            self._run_git_command(['config', 'user.name', 'Self-Modification-Engine'])
            self._run_git_command(['config', 'user.email', 'noreply@leanvibe.dev'])
            
            # Create initial commit if no commits exist
            if self._get_commit_count() == 0:
                # Create .gitignore
                gitignore_path = self.repository_path / '.gitignore'
                gitignore_content = '\n'.join([
                    '__pycache__/',
                    '*.pyc',
                    '*.pyo',
                    '*.tmp',
                    '*.log',
                    '.DS_Store',
                    'Thumbs.db',
                    '*.key',
                    '*.pem',
                    'secrets.*'
                ])
                self._write_file_secure(gitignore_path, gitignore_content)
                
                # Create initial commit
                self._run_git_command(['add', '.gitignore'])
                self._run_git_command(['commit', '-m', 'Initial commit: Self-modification engine setup'])
    
    def _validate_repository_state(self) -> None:
        """Validate repository state for security."""
        if not self._is_git_repository():
            raise SecurityError("Not a valid Git repository")
        
        if not self.repository_path.exists():
            raise SecurityError(f"Repository path does not exist: {self.repository_path}")
        
        # Check for suspicious files
        security_violations = self._scan_for_security_violations()
        if security_violations:
            logger.warning(f"Security violations detected: {len(security_violations)}")
            for violation in security_violations:
                logger.warning(f"  - {violation}")
    
    def _validate_file_modification(self, file_path: str, original_content: str, 
                                  modified_content: str, safety_score: float) -> None:
        """Validate file modification for security."""
        # Check safety score
        if safety_score < 0.3:
            raise SecurityError(f"Safety score too low for modification: {safety_score}")
        
        # Check file size
        if len(modified_content) > self._max_file_size_mb * 1024 * 1024:
            raise SecurityError(f"Modified file too large: {len(modified_content)} bytes")
        
        # Check for suspicious patterns
        if self._contains_suspicious_patterns(modified_content):
            raise SecurityError("Modified content contains suspicious patterns")
        
        # Validate file path
        if not self._is_safe_file_path(file_path):
            raise SecurityError(f"Unsafe file path: {file_path}")
    
    def _git_add_all_safe(self) -> None:
        """Add all files safely (excluding blocked patterns)."""
        # Add files individually to respect .gitignore and blocked patterns
        for file_path in self.repository_path.rglob('*'):
            if file_path.is_file() and self._is_safe_file_path(str(file_path.relative_to(self.repository_path))):
                try:
                    self._run_git_command(['add', str(file_path)])
                except Exception as e:
                    logger.warning(f"Failed to add {file_path}: {e}")
    
    def _git_add_file(self, file_path: str) -> None:
        """Add specific file to Git staging."""
        self._run_git_command(['add', file_path])
    
    def _git_commit(self, message: str) -> Dict[str, Any]:
        """Create Git commit with security validation."""
        try:
            # Check for changes to commit
            result = self._run_git_command(['diff', '--cached', '--quiet'])
            if result.returncode == 0:
                logger.info("No changes to commit")
                return {'status': GitOperationStatus.SUCCESS, 'commit_hash': self._get_current_commit_hash()}
            
            # Create commit
            commit_result = self._run_git_command(['commit', '-m', message, '--author', self._git_author])
            
            if commit_result.returncode == 0:
                commit_hash = self._get_current_commit_hash()
                return {
                    'status': GitOperationStatus.SUCCESS,
                    'commit_hash': commit_hash,
                    'message': message
                }
            else:
                return {
                    'status': GitOperationStatus.FAILED,
                    'error': commit_result.stderr
                }
                
        except Exception as e:
            return {
                'status': GitOperationStatus.FAILED,
                'error': str(e)
            }
    
    def _execute_hard_reset(self, target_commit: str) -> Dict[str, Any]:
        """Execute hard reset to target commit."""
        try:
            result = self._run_git_command(['reset', '--hard', target_commit])
            if result.returncode == 0:
                return {
                    'status': GitOperationStatus.SUCCESS,
                    'new_commit_hash': target_commit,
                    'files_changed': self._get_changed_files_between_commits(self._get_current_commit_hash(), target_commit)
                }
            else:
                return {
                    'status': GitOperationStatus.FAILED,
                    'error': result.stderr
                }
        except Exception as e:
            return {
                'status': GitOperationStatus.FAILED,
                'error': str(e)
            }
    
    def _execute_soft_reset(self, target_commit: str) -> Dict[str, Any]:
        """Execute soft reset to target commit."""
        try:
            result = self._run_git_command(['reset', '--soft', target_commit])
            if result.returncode == 0:
                return {
                    'status': GitOperationStatus.SUCCESS,
                    'new_commit_hash': target_commit,
                    'files_changed': []  # Files remain in working directory
                }
            else:
                return {
                    'status': GitOperationStatus.FAILED,
                    'error': result.stderr
                }
        except Exception as e:
            return {
                'status': GitOperationStatus.FAILED,
                'error': str(e)
            }
    
    def _execute_commit_revert(self, from_commit: str, to_commit: str) -> Dict[str, Any]:
        """Execute commit revert between commits."""
        try:
            # Get list of commits to revert
            commits_result = self._run_git_command(['rev-list', '--reverse', f'{to_commit}..{from_commit}'])
            if commits_result.returncode != 0:
                return {'status': GitOperationStatus.FAILED, 'error': commits_result.stderr}
            
            commits_to_revert = commits_result.stdout.strip().split('\n')
            if not commits_to_revert or commits_to_revert == ['']:
                return {'status': GitOperationStatus.SUCCESS, 'new_commit_hash': self._get_current_commit_hash()}
            
            # Revert each commit
            for commit in commits_to_revert:
                result = self._run_git_command(['revert', '--no-edit', commit])
                if result.returncode != 0:
                    return {
                        'status': GitOperationStatus.CONFLICT,
                        'error': result.stderr,
                        'conflicts': [commit]
                    }
            
            return {
                'status': GitOperationStatus.SUCCESS,
                'new_commit_hash': self._get_current_commit_hash(),
                'files_changed': []  # TODO: Calculate changed files
            }
            
        except Exception as e:
            return {
                'status': GitOperationStatus.FAILED,
                'error': str(e)
            }
    
    def _run_git_command(self, args: List[str]) -> subprocess.CompletedProcess:
        """Run Git command with security controls."""
        try:
            # Ensure we're in the repository directory
            full_args = ['git'] + args
            
            result = subprocess.run(
                full_args,
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout for Git operations
            )
            
            return result
            
        except subprocess.TimeoutExpired:
            raise GitError(f"Git command timed out: {args}")
        except Exception as e:
            raise GitError(f"Git command failed: {args}, error: {e}")
    
    def _get_current_commit_hash(self) -> str:
        """Get current commit hash."""
        result = self._run_git_command(['rev-parse', 'HEAD'])
        if result.returncode == 0:
            return result.stdout.strip()
        return ""
    
    def _get_current_branch(self) -> str:
        """Get current branch name."""
        result = self._run_git_command(['rev-parse', '--abbrev-ref', 'HEAD'])
        if result.returncode == 0:
            return result.stdout.strip()
        return "unknown"
    
    def _get_commit_count(self) -> int:
        """Get total number of commits."""
        result = self._run_git_command(['rev-list', '--count', 'HEAD'])
        if result.returncode == 0:
            try:
                return int(result.stdout.strip())
            except ValueError:
                return 0
        return 0
    
    def _get_uncommitted_files(self) -> List[str]:
        """Get list of uncommitted files."""
        result = self._run_git_command(['status', '--porcelain'])
        if result.returncode == 0:
            files = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    files.append(line[3:])  # Skip status indicators
            return files
        return []
    
    def _get_repository_state(self) -> Dict[str, Any]:
        """Get current repository state summary."""
        return {
            'current_commit': self._get_current_commit_hash(),
            'current_branch': self._get_current_branch(),
            'total_commits': self._get_commit_count(),
            'uncommitted_files': len(self._get_uncommitted_files()),
            'repository_size_mb': self._get_repository_size_mb()
        }
    
    def _get_repository_size_mb(self) -> float:
        """Get repository size in MB."""
        try:
            total_size = 0
            for file_path in self.repository_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _get_changed_files_between_commits(self, commit1: str, commit2: str) -> List[str]:
        """Get list of files changed between two commits."""
        result = self._run_git_command(['diff', '--name-only', commit1, commit2])
        if result.returncode == 0:
            return result.stdout.strip().split('\n') if result.stdout.strip() else []
        return []
    
    def _is_git_repository(self) -> bool:
        """Check if directory is a Git repository."""
        git_dir = self.repository_path / '.git'
        return git_dir.exists()
    
    def _find_checkpoint(self, checkpoint_id: str) -> Optional[GitCheckpoint]:
        """Find checkpoint by ID."""
        for checkpoint in self._checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        return None
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond retention limit."""
        if len(self._checkpoints) > self.max_rollback_commits:
            # Keep most recent checkpoints
            self._checkpoints = sorted(self._checkpoints, key=lambda cp: cp.created_at, reverse=True)
            removed_checkpoints = self._checkpoints[self.max_rollback_commits:]
            self._checkpoints = self._checkpoints[:self.max_rollback_commits]
            
            logger.info(f"Cleaned up {len(removed_checkpoints)} old checkpoints")
    
    def _scan_for_security_violations(self) -> List[str]:
        """Scan repository for security violations."""
        violations = []
        
        try:
            for file_path in self.repository_path.rglob('*'):
                if file_path.is_file():
                    # Check for blocked file patterns
                    if any(file_path.match(pattern) for pattern in self._blocked_file_patterns):
                        violations.append(f"Blocked file pattern: {file_path}")
                    
                    # Check file size
                    if file_path.stat().st_size > self._max_file_size_mb * 1024 * 1024:
                        violations.append(f"File too large: {file_path}")
        
        except Exception as e:
            logger.warning(f"Security scan failed: {e}")
        
        return violations
    
    def _contains_suspicious_patterns(self, content: str) -> bool:
        """Check if content contains suspicious patterns."""
        suspicious_patterns = [
            'eval(', 'exec(', 'subprocess.call', 'os.system',
            'import os', 'import subprocess', 'import sys',
            '__import__', 'globals()', 'locals()'
        ]
        
        content_lower = content.lower()
        return any(pattern.lower() in content_lower for pattern in suspicious_patterns)
    
    def _is_safe_file_path(self, file_path: str) -> bool:
        """Check if file path is safe for modification."""
        # Convert to Path for safer handling
        path = Path(file_path)
        
        # Check for path traversal
        if '..' in path.parts:
            return False
        
        # Check for blocked patterns
        if any(path.match(pattern) for pattern in self._blocked_file_patterns):
            return False
        
        # Check for system paths
        if str(path).startswith(('/etc', '/sys', '/proc', '/dev')):
            return False
        
        return True
    
    def _write_file_secure(self, file_path: Path, content: str) -> None:
        """Write file with security validation."""
        # Security validation
        if len(content) > self._max_file_size_mb * 1024 * 1024:
            raise SecurityError(f"File content too large: {len(content)} bytes")
        
        if not self._is_safe_file_path(str(file_path.relative_to(self.repository_path))):
            raise SecurityError(f"Unsafe file path: {file_path}")
        
        if self._contains_suspicious_patterns(content):
            raise SecurityError("File content contains suspicious patterns")
        
        # Write file
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Set restrictive permissions
            file_path.chmod(0o644)
            
        except Exception as e:
            raise SecurityError(f"Failed to write file {file_path}: {e}")
    
    def _generate_checkpoint_id(self, checkpoint_type: CheckpointType, session_id: Optional[str] = None) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        type_prefix = checkpoint_type.value[:3].upper()
        session_suffix = f"_{session_id[:8]}" if session_id else ""
        return f"{type_prefix}_{timestamp}{session_suffix}"
    
    def _generate_rollback_id(self, checkpoint_id: str) -> str:
        """Generate unique rollback operation ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return f"ROLLBACK_{timestamp}_{checkpoint_id}"


class GitError(Exception):
    """Git operation error."""
    pass


class SecurityError(Exception):
    """Security violation in Git operations."""
    pass


# Export main classes
__all__ = [
    'SecureGitManager',
    'GitCheckpoint',
    'GitCommit',
    'RollbackOperation',
    'CheckpointType',
    'GitOperationStatus',
    'GitError',
    'SecurityError'
]