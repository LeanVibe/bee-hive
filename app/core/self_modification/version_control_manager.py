"""
Version Control Manager

Git integration for change tracking, branching, and rollback capabilities.
Provides secure, isolated git operations with comprehensive history tracking
and automated rollback mechanisms for the self-modification engine.
"""

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import git
import structlog
from git import Repo, InvalidGitRepositoryError, GitCommandError

logger = structlog.get_logger()


@dataclass
class CommitInfo:
    """Information about a git commit."""
    
    hash: str
    short_hash: str
    message: str
    author_name: str
    author_email: str
    timestamp: datetime
    files_changed: List[str] = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0
    parent_hashes: List[str] = field(default_factory=list)
    is_merge: bool = False


@dataclass
class BranchInfo:
    """Information about a git branch."""
    
    name: str
    commit_hash: str
    is_active: bool = False
    is_remote: bool = False
    upstream: Optional[str] = None
    last_commit_date: Optional[datetime] = None


@dataclass
class ModificationCommit:
    """Represents a self-modification commit with metadata."""
    
    commit_hash: str
    modification_session_id: str
    modification_ids: List[str]
    safety_level: str
    agent_id: str
    created_at: datetime
    rollback_point: Optional[str] = None  # Commit hash to rollback to
    
    @property
    def is_self_modification(self) -> bool:
        """Check if this is a self-modification commit."""
        return bool(self.modification_session_id)


class VersionControlManager:
    """Manages git operations for the self-modification engine."""
    
    def __init__(self, repository_path: str):
        self.repository_path = Path(repository_path).resolve()
        self.repo: Optional[Repo] = None
        self.modification_branch_prefix = "self-mod"
        self.rollback_branch_prefix = "rollback"
        
        self._initialize_repository()
    
    def _initialize_repository(self) -> None:
        """Initialize or connect to git repository."""
        
        try:
            # Try to open existing repository
            self.repo = Repo(self.repository_path)
            logger.info("Connected to existing git repository", path=str(self.repository_path))
            
        except InvalidGitRepositoryError:
            # Initialize new repository if none exists
            logger.info("Initializing new git repository", path=str(self.repository_path))
            self.repo = Repo.init(self.repository_path)
            
            # Set up initial configuration
            self._configure_repository()
            
        except Exception as e:
            logger.error("Failed to initialize git repository", path=str(self.repository_path), error=str(e))
            raise
    
    def _configure_repository(self) -> None:
        """Configure repository for self-modification."""
        
        if not self.repo:
            raise RuntimeError("Repository not initialized")
        
        # Configure user if not set
        try:
            self.repo.config_reader().get_value("user", "name")
            self.repo.config_reader().get_value("user", "email")
        except Exception:
            # Set default configuration for self-modification
            with self.repo.config_writer() as config:
                config.set_value("user", "name", "LeanVibe Self-Modification Engine")
                config.set_value("user", "email", "self-modification@leanvibe.com")
        
        # Create .gitignore if it doesn't exist
        gitignore_path = self.repository_path / ".gitignore"
        if not gitignore_path.exists():
            gitignore_content = """
# Self-modification temporary files
.self_mod_*
*.self_mod_backup
.modification_workspace/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
            gitignore_path.write_text(gitignore_content.strip())
    
    def create_modification_branch(
        self,
        session_id: str,
        base_branch: str = "main"
    ) -> str:
        """Create a new branch for modifications."""
        
        if not self.repo:
            raise RuntimeError("Repository not initialized")
        
        branch_name = f"{self.modification_branch_prefix}/{session_id}"
        
        try:
            # Ensure we're on the base branch and it's up to date
            self._checkout_branch(base_branch)
            
            # Create new branch
            new_branch = self.repo.create_head(branch_name)
            new_branch.checkout()
            
            logger.info(
                "Created modification branch",
                branch_name=branch_name,
                base_branch=base_branch,
                session_id=session_id
            )
            
            return branch_name
            
        except Exception as e:
            logger.error(
                "Failed to create modification branch",
                session_id=session_id,
                base_branch=base_branch,
                error=str(e)
            )
            raise
    
    def apply_modifications(
        self,
        modifications: Dict[str, str],  # file_path -> new_content
        session_id: str,
        modification_ids: List[str],
        safety_level: str,
        agent_id: str,
        commit_message: Optional[str] = None
    ) -> CommitInfo:
        """Apply modifications and create commit."""
        
        if not self.repo:
            raise RuntimeError("Repository not initialized")
        
        try:
            # Ensure we're on the correct modification branch
            branch_name = f"{self.modification_branch_prefix}/{session_id}"
            self._checkout_branch(branch_name)
            
            # Create rollback point before modifications
            rollback_commit = self.repo.head.commit.hexsha
            
            # Apply file modifications
            modified_files = []
            for file_path, new_content in modifications.items():
                full_path = self.repository_path / file_path
                
                # Create backup
                self._create_file_backup(full_path, session_id)
                
                # Write new content
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(new_content, encoding="utf-8")
                
                modified_files.append(file_path)
            
            # Stage changes
            self.repo.index.add(modified_files)
            
            # Create commit
            if not commit_message:
                commit_message = self._generate_commit_message(
                    session_id, modification_ids, safety_level, len(modified_files)
                )
            
            commit = self.repo.index.commit(commit_message)
            
            # Create commit info
            commit_info = self._create_commit_info(commit)
            
            logger.info(
                "Applied modifications and created commit",
                session_id=session_id,
                commit_hash=commit_info.hash,
                files_modified=len(modified_files)
            )
            
            return commit_info
            
        except Exception as e:
            logger.error(
                "Failed to apply modifications",
                session_id=session_id,
                error=str(e)
            )
            # Attempt to rollback
            self._rollback_failed_modifications(session_id)
            raise
    
    def rollback_modifications(
        self,
        commit_hash: str,
        rollback_reason: str,
        force: bool = False
    ) -> CommitInfo:
        """Rollback modifications to a previous state."""
        
        if not self.repo:
            raise RuntimeError("Repository not initialized")
        
        try:
            # Find the commit to rollback to
            target_commit = self.repo.commit(commit_hash)
            
            # Determine rollback strategy
            current_commit = self.repo.head.commit
            
            if force or self._is_safe_rollback(current_commit, target_commit):
                # Create rollback branch for safety
                rollback_branch = f"{self.rollback_branch_prefix}/{str(uuid4())[:8]}"
                rollback_ref = self.repo.create_head(rollback_branch)
                
                # Reset to target commit
                self.repo.head.reset(target_commit, index=True, working_tree=True)
                
                # Create rollback commit
                rollback_commit_message = f"""Rollback modifications

Reason: {rollback_reason}
Rolled back from: {current_commit.hexsha[:8]}
Rolled back to: {target_commit.hexsha[:8]}

🤖 Generated by Self-Modification Engine
Rollback performed at: {datetime.utcnow().isoformat()}
"""
                
                commit = self.repo.index.commit(rollback_commit_message)
                commit_info = self._create_commit_info(commit)
                
                logger.info(
                    "Successfully rolled back modifications",
                    from_commit=current_commit.hexsha[:8],
                    to_commit=target_commit.hexsha[:8],
                    rollback_commit=commit_info.hash
                )
                
                return commit_info
                
            else:
                raise ValueError("Rollback would cause conflicts. Use force=True to override.")
                
        except Exception as e:
            logger.error(
                "Failed to rollback modifications",
                commit_hash=commit_hash,
                error=str(e)
            )
            raise
    
    def get_modification_history(
        self,
        limit: int = 50,
        since_date: Optional[datetime] = None
    ) -> List[ModificationCommit]:
        """Get history of self-modification commits."""
        
        if not self.repo:
            raise RuntimeError("Repository not initialized")
        
        modification_commits = []
        
        try:
            # Get commits with self-modification markers
            commits = list(self.repo.iter_commits(max_count=limit))
            
            for commit in commits:
                # Check if this is a self-modification commit
                if self._is_self_modification_commit(commit):
                    mod_commit = self._parse_modification_commit(commit)
                    if since_date is None or mod_commit.created_at >= since_date:
                        modification_commits.append(mod_commit)
            
            return modification_commits
            
        except Exception as e:
            logger.error("Failed to get modification history", error=str(e))
            return []
    
    def get_branch_info(self, branch_name: Optional[str] = None) -> BranchInfo:
        """Get information about a branch."""
        
        if not self.repo:
            raise RuntimeError("Repository not initialized")
        
        try:
            if branch_name:
                branch = self.repo.heads[branch_name]
            else:
                branch = self.repo.active_branch
            
            return BranchInfo(
                name=branch.name,
                commit_hash=branch.commit.hexsha,
                is_active=(branch == self.repo.active_branch),
                last_commit_date=datetime.fromtimestamp(branch.commit.committed_date)
            )
            
        except Exception as e:
            logger.error("Failed to get branch info", branch_name=branch_name, error=str(e))
            raise
    
    def cleanup_modification_branches(
        self,
        older_than_days: int = 7,
        keep_recent: int = 10
    ) -> List[str]:
        """Clean up old modification branches."""
        
        if not self.repo:
            raise RuntimeError("Repository not initialized")
        
        deleted_branches = []
        cutoff_date = datetime.now().timestamp() - (older_than_days * 24 * 3600)
        
        try:
            # Get all modification branches
            mod_branches = [
                branch for branch in self.repo.heads
                if branch.name.startswith(self.modification_branch_prefix)
            ]
            
            # Sort by last commit date
            mod_branches.sort(key=lambda b: b.commit.committed_date, reverse=True)
            
            # Keep recent branches, delete old ones
            for branch in mod_branches[keep_recent:]:
                if branch.commit.committed_date < cutoff_date:
                    branch_name = branch.name
                    self.repo.delete_head(branch, force=True)
                    deleted_branches.append(branch_name)
            
            logger.info(
                "Cleaned up modification branches",
                deleted_count=len(deleted_branches),
                branches_deleted=deleted_branches
            )
            
            return deleted_branches
            
        except Exception as e:
            logger.error("Failed to cleanup modification branches", error=str(e))
            return []
    
    def create_safety_checkpoint(self, checkpoint_name: str) -> str:
        """Create a safety checkpoint before risky operations."""
        
        if not self.repo:
            raise RuntimeError("Repository not initialized")
        
        try:
            # Create annotated tag as checkpoint
            tag_name = f"checkpoint-{checkpoint_name}-{int(datetime.now().timestamp())}"
            
            tag = self.repo.create_tag(
                tag_name,
                message=f"Safety checkpoint: {checkpoint_name}"
            )
            
            logger.info("Created safety checkpoint", tag_name=tag_name, commit=tag.commit.hexsha[:8])
            
            return tag_name
            
        except Exception as e:
            logger.error("Failed to create safety checkpoint", checkpoint_name=checkpoint_name, error=str(e))
            raise
    
    def _checkout_branch(self, branch_name: str) -> None:
        """Safely checkout a branch."""
        
        if not self.repo:
            raise RuntimeError("Repository not initialized")
        
        try:
            if branch_name in [head.name for head in self.repo.heads]:
                # Local branch exists
                branch = self.repo.heads[branch_name]
                branch.checkout()
            else:
                # Try to create from remote
                origin = self.repo.remotes.origin if self.repo.remotes else None
                if origin and f"origin/{branch_name}" in [ref.name for ref in origin.refs]:
                    branch = self.repo.create_head(branch_name, origin.refs[branch_name])
                    branch.checkout()
                else:
                    raise ValueError(f"Branch '{branch_name}' not found")
                    
        except Exception as e:
            logger.error("Failed to checkout branch", branch_name=branch_name, error=str(e))
            raise
    
    def _create_file_backup(self, file_path: Path, session_id: str) -> None:
        """Create backup of file before modification."""
        
        if file_path.exists():
            backup_dir = self.repository_path / ".modification_workspace" / session_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create relative path for backup
            rel_path = file_path.relative_to(self.repository_path)
            backup_path = backup_dir / f"{rel_path}.backup"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(file_path, backup_path)
    
    def _generate_commit_message(
        self,
        session_id: str,
        modification_ids: List[str],
        safety_level: str,
        files_count: int
    ) -> str:
        """Generate standardized commit message for modifications."""
        
        return f"""Self-modification: {files_count} files updated

Session ID: {session_id}
Modification IDs: {', '.join(modification_ids[:3])}{'...' if len(modification_ids) > 3 else ''}
Safety Level: {safety_level}
Files Modified: {files_count}

🤖 Generated by Self-Modification Engine
Generated at: {datetime.utcnow().isoformat()}

Co-Authored-By: LeanVibe Self-Modification Engine <self-modification@leanvibe.com>
"""
    
    def _create_commit_info(self, commit: git.Commit) -> CommitInfo:
        """Create CommitInfo from git commit."""
        
        # Get file statistics
        stats = commit.stats
        files_changed = list(stats.files.keys()) if stats.files else []
        
        return CommitInfo(
            hash=commit.hexsha,
            short_hash=commit.hexsha[:8],
            message=commit.message,
            author_name=commit.author.name,
            author_email=commit.author.email,
            timestamp=datetime.fromtimestamp(commit.committed_date),
            files_changed=files_changed,
            lines_added=stats.total.get("insertions", 0),
            lines_removed=stats.total.get("deletions", 0),
            parent_hashes=[parent.hexsha for parent in commit.parents],
            is_merge=len(commit.parents) > 1
        )
    
    def _is_self_modification_commit(self, commit: git.Commit) -> bool:
        """Check if commit is from self-modification engine."""
        
        return (
            "Self-modification:" in commit.message or
            "🤖 Generated by Self-Modification Engine" in commit.message or
            commit.author.email == "self-modification@leanvibe.com"
        )
    
    def _parse_modification_commit(self, commit: git.Commit) -> ModificationCommit:
        """Parse self-modification commit for metadata."""
        
        # Extract metadata from commit message
        message = commit.message
        session_id = ""
        modification_ids = []
        safety_level = "unknown"
        
        for line in message.split("\n"):
            if line.startswith("Session ID:"):
                session_id = line.split(":", 1)[1].strip()
            elif line.startswith("Modification IDs:"):
                mod_ids_str = line.split(":", 1)[1].strip()
                modification_ids = [mid.strip() for mid in mod_ids_str.split(",")]
            elif line.startswith("Safety Level:"):
                safety_level = line.split(":", 1)[1].strip()
        
        return ModificationCommit(
            commit_hash=commit.hexsha,
            modification_session_id=session_id,
            modification_ids=modification_ids,
            safety_level=safety_level,
            agent_id="unknown",  # Could be extracted from metadata
            created_at=datetime.fromtimestamp(commit.committed_date),
            rollback_point=commit.parents[0].hexsha if commit.parents else None
        )
    
    def _is_safe_rollback(self, current_commit: git.Commit, target_commit: git.Commit) -> bool:
        """Check if rollback is safe (no conflicts)."""
        
        try:
            # Simple check: ensure target commit is an ancestor of current
            merge_base = self.repo.merge_base(current_commit, target_commit)
            return merge_base and merge_base[0] == target_commit
            
        except Exception:
            return False
    
    def _rollback_failed_modifications(self, session_id: str) -> None:
        """Rollback failed modifications using backups."""
        
        try:
            backup_dir = self.repository_path / ".modification_workspace" / session_id
            if backup_dir.exists():
                # Restore files from backup
                for backup_file in backup_dir.rglob("*.backup"):
                    original_path = self.repository_path / backup_file.relative_to(backup_dir).with_suffix("")
                    if original_path.parent.exists():
                        shutil.copy2(backup_file, original_path)
                
                # Clean up workspace
                shutil.rmtree(backup_dir, ignore_errors=True)
                
                logger.info("Restored files from backup after failed modification", session_id=session_id)
                
        except Exception as e:
            logger.error("Failed to restore from backup", session_id=session_id, error=str(e))


# Export main class and data structures  
__all__ = [
    "VersionControlManager",
    "CommitInfo", 
    "BranchInfo",
    "ModificationCommit"
]