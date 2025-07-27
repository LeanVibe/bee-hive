"""
Work Tree Manager for LeanVibe Agent Hive 2.0

Provides 100% isolated development environments per agent using Git worktrees
with comprehensive security, cleanup, and resource management.
"""

import os
import shutil
import asyncio
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import tempfile
import stat

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
import psutil

from ..core.config import get_settings
from ..core.database import get_db_session
from ..models.agent import Agent
from ..models.github_integration import (
    GitHubRepository, AgentWorkTree, WorkTreeStatus, GitCommit
)
from ..core.github_api_client import GitHubAPIClient


logger = logging.getLogger(__name__)
settings = get_settings()


class WorkTreeConfig:
    """Configuration class for work tree management."""
    
    def __init__(
        self,
        base_path: str = None,
        file_permissions: int = 0o750,
        directory_permissions: int = 0o750,
        max_disk_usage_mb: int = 2048,
        max_file_count: int = 10000,
        cleanup_after_days: int = 7
    ):
        self.base_path = base_path or "/tmp/agent-workspaces"
        self.file_permissions = file_permissions
        self.directory_permissions = directory_permissions
        self.max_disk_usage_mb = max_disk_usage_mb
        self.max_file_count = max_file_count
        self.cleanup_after_days = cleanup_after_days


class WorkTreeError(Exception):
    """Custom exception for work tree operations."""
    pass


class WorkTreeIsolationManager:
    """
    Manages filesystem-level isolation between agent work trees.
    
    Ensures complete separation of agent workspaces with security controls
    and resource monitoring.
    """
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path or settings.WORK_TREES_BASE_PATH or "/tmp/agent-workspaces")
        self.isolation_config = {
            "file_permissions": 0o750,  # rwxr-x---
            "directory_permissions": 0o750,
            "max_disk_usage_mb": 2048,  # 2GB per work tree
            "max_file_count": 10000,
            "restricted_paths": ["/etc", "/var", "/usr", "/root", "/home"],
            "allowed_extensions": [".py", ".js", ".ts", ".md", ".json", ".yaml", ".yml", ".txt", ".sh"]
        }
        
    async def ensure_base_directory(self) -> None:
        """Ensure base work tree directory exists with proper permissions."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        os.chmod(self.base_path, self.isolation_config["directory_permissions"])
        
    def get_work_tree_path(self, agent_id: str, repository_name: str) -> Path:
        """Generate isolated work tree path for agent and repository."""
        safe_repo_name = "".join(c for c in repository_name if c.isalnum() or c in "-_.")
        return self.base_path / f"agent-{agent_id}" / safe_repo_name
        
    async def create_isolated_directory(self, work_tree_path: Path, agent_id: str) -> None:
        """Create isolated directory with proper permissions and ownership."""
        work_tree_path.mkdir(parents=True, exist_ok=True)
        
        # Set restrictive permissions
        os.chmod(work_tree_path, self.isolation_config["directory_permissions"])
        
        # Create .gitignore to prevent accidental commits of sensitive files
        gitignore_path = work_tree_path / ".gitignore"
        gitignore_content = """
# Agent workspace isolation
.env
.env.*
*.key
*.pem
*.p12
*.pfx
secrets/
.secrets/
private/
.private/

# Temporary files
*.tmp
*.temp
.cache/
.temp/

# OS specific
.DS_Store
Thumbs.db
"""
        with open(gitignore_path, "w") as f:
            f.write(gitignore_content.strip())
            
        logger.info(f"Created isolated work tree directory: {work_tree_path}")
        
    def check_path_security(self, path: Path) -> bool:
        """Check if path is safe for agent operations."""
        try:
            resolved_path = path.resolve()
            
            # Check if path tries to escape base directory
            if not str(resolved_path).startswith(str(self.base_path.resolve())):
                logger.warning(f"Path escape attempt detected: {path}")
                return False
                
            # Check against restricted paths
            for restricted in self.isolation_config["restricted_paths"]:
                if str(resolved_path).startswith(restricted):
                    logger.warning(f"Restricted path access attempt: {path}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Path security check failed: {e}")
            return False
            
    async def check_resource_limits(self, work_tree_path: Path) -> Dict[str, Any]:
        """Check if work tree exceeds resource limits."""
        if not work_tree_path.exists():
            return {"valid": True, "disk_usage_mb": 0, "file_count": 0}
            
        disk_usage = 0
        file_count = 0
        
        try:
            for root, dirs, files in os.walk(work_tree_path):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.exists():
                        disk_usage += file_path.stat().st_size
                        file_count += 1
                        
        except Exception as e:
            logger.error(f"Resource check failed for {work_tree_path}: {e}")
            return {"valid": False, "error": str(e)}
            
        disk_usage_mb = disk_usage / (1024 * 1024)
        
        return {
            "valid": (
                disk_usage_mb <= self.isolation_config["max_disk_usage_mb"] and
                file_count <= self.isolation_config["max_file_count"]
            ),
            "disk_usage_mb": disk_usage_mb,
            "file_count": file_count,
            "max_disk_usage_mb": self.isolation_config["max_disk_usage_mb"],
            "max_file_count": self.isolation_config["max_file_count"]
        }
        
    async def cleanup_work_tree(self, work_tree_path: Path, force: bool = False) -> bool:
        """Safely cleanup work tree directory."""
        if not work_tree_path.exists():
            return True
            
        try:
            # Security check
            if not self.check_path_security(work_tree_path):
                raise WorkTreeError(f"Security check failed for cleanup: {work_tree_path}")
                
            # Remove read-only permissions recursively
            def remove_readonly(func, path, _):
                os.chmod(path, stat.S_IWRITE)
                func(path)
                
            shutil.rmtree(work_tree_path, onerror=remove_readonly)
            logger.info(f"Cleaned up work tree: {work_tree_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup work tree {work_tree_path}: {e}")
            if force:
                # Last resort: use sudo rm -rf (only in controlled environments)
                try:
                    import subprocess
                    subprocess.run(["rm", "-rf", str(work_tree_path)], check=True)
                    return True
                except subprocess.CalledProcessError:
                    pass
            return False


class GitWorkTreeManager:
    """
    Git-specific work tree management with advanced isolation and branch handling.
    
    Manages Git worktrees for agent development with automatic branch management,
    conflict detection, and repository synchronization.
    """
    
    def __init__(self, github_client: GitHubAPIClient = None):
        self.github_client = github_client or GitHubAPIClient()
        self.isolation_manager = WorkTreeIsolationManager()
        self.git_config = {
            "user.name": "LeanVibe Agent",
            "user.email": "agents@leanvibe.com",
            "core.autocrlf": "false",
            "core.filemode": "false",
            "init.defaultBranch": "main"
        }
        
    async def execute_git_command(
        self,
        command: List[str],
        cwd: Path,
        timeout: int = 300,
        capture_output: bool = True
    ) -> Tuple[int, str, str]:
        """Execute Git command with security controls and timeout."""
        
        # Security validation
        if not self.isolation_manager.check_path_security(cwd):
            raise WorkTreeError(f"Insecure path for Git operation: {cwd}")
            
        # Validate Git command
        if not command or command[0] != "git":
            raise WorkTreeError(f"Invalid Git command: {command}")
            
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE if capture_output else None,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # Disable interactive prompts
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return_code = process.returncode
            stdout_str = stdout.decode('utf-8') if stdout else ""
            stderr_str = stderr.decode('utf-8') if stderr else ""
            
            if return_code != 0:
                logger.warning(f"Git command failed: {command}, stderr: {stderr_str}")
                
            return return_code, stdout_str, stderr_str
            
        except asyncio.TimeoutError:
            logger.error(f"Git command timed out: {command}")
            raise WorkTreeError(f"Git command timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Git command execution failed: {command}, error: {e}")
            raise WorkTreeError(f"Git command execution failed: {str(e)}")
            
    async def clone_repository(
        self,
        repository_url: str,
        work_tree_path: Path,
        branch: str = None,
        depth: int = 1
    ) -> bool:
        """Clone repository to work tree with specified branch."""
        
        await self.isolation_manager.ensure_base_directory()
        await self.isolation_manager.create_isolated_directory(work_tree_path, "agent")
        
        # Prepare clone command
        clone_cmd = ["git", "clone"]
        
        if depth:
            clone_cmd.extend(["--depth", str(depth)])
            
        if branch:
            clone_cmd.extend(["--branch", branch])
            
        clone_cmd.extend([repository_url, str(work_tree_path)])
        
        # Execute clone
        return_code, stdout, stderr = await self.execute_git_command(
            clone_cmd,
            work_tree_path.parent,
            timeout=600  # 10 minutes for large repositories
        )
        
        if return_code != 0:
            raise WorkTreeError(f"Failed to clone repository: {stderr}")
            
        # Configure Git settings
        await self.configure_git_repository(work_tree_path)
        
        logger.info(f"Successfully cloned repository to {work_tree_path}")
        return True
        
    async def configure_git_repository(self, work_tree_path: Path) -> None:
        """Configure Git repository with agent-specific settings."""
        for key, value in self.git_config.items():
            cmd = ["git", "config", key, value]
            await self.execute_git_command(cmd, work_tree_path)
            
    async def create_agent_branch(
        self,
        work_tree_path: Path,
        branch_name: str,
        base_branch: str = "main"
    ) -> bool:
        """Create and checkout agent-specific branch."""
        
        # Fetch latest changes
        await self.execute_git_command(["git", "fetch", "origin"], work_tree_path)
        
        # Create and checkout new branch
        cmd = ["git", "checkout", "-b", branch_name, f"origin/{base_branch}"]
        return_code, stdout, stderr = await self.execute_git_command(cmd, work_tree_path)
        
        if return_code != 0:
            raise WorkTreeError(f"Failed to create branch {branch_name}: {stderr}")
            
        # Set upstream tracking
        upstream_cmd = ["git", "push", "--set-upstream", "origin", branch_name]
        await self.execute_git_command(upstream_cmd, work_tree_path)
        
        logger.info(f"Created agent branch: {branch_name}")
        return True
        
    async def sync_with_upstream(self, work_tree_path: Path, base_branch: str = "main") -> Dict[str, Any]:
        """Sync work tree with upstream changes."""
        
        sync_result = {
            "success": False,
            "conflicts": [],
            "changes_applied": False,
            "error": None
        }
        
        try:
            # Fetch latest changes
            await self.execute_git_command(["git", "fetch", "origin"], work_tree_path)
            
            # Check for uncommitted changes
            status_code, status_output, _ = await self.execute_git_command(
                ["git", "status", "--porcelain"],
                work_tree_path
            )
            
            has_uncommitted = bool(status_output.strip())
            
            if has_uncommitted:
                # Stash uncommitted changes
                await self.execute_git_command(["git", "stash", "push", "-m", "Auto-stash before sync"], work_tree_path)
                
            # Attempt to merge upstream changes
            merge_cmd = ["git", "merge", f"origin/{base_branch}"]
            merge_code, merge_output, merge_error = await self.execute_git_command(merge_cmd, work_tree_path)
            
            if merge_code == 0:
                sync_result["success"] = True
                sync_result["changes_applied"] = True
                
                # Restore stashed changes if any
                if has_uncommitted:
                    stash_code, _, _ = await self.execute_git_command(["git", "stash", "pop"], work_tree_path)
                    if stash_code != 0:
                        logger.warning("Failed to restore stashed changes after sync")
                        
            else:
                # Handle merge conflicts
                if "CONFLICT" in merge_error or "CONFLICT" in merge_output:
                    conflicts = await self.detect_conflicts(work_tree_path)
                    sync_result["conflicts"] = conflicts
                    
                    # Abort merge to return to clean state
                    await self.execute_git_command(["git", "merge", "--abort"], work_tree_path)
                    
                    # Restore stashed changes
                    if has_uncommitted:
                        await self.execute_git_command(["git", "stash", "pop"], work_tree_path)
                        
                sync_result["error"] = merge_error
                
        except Exception as e:
            sync_result["error"] = str(e)
            logger.error(f"Sync with upstream failed: {e}")
            
        return sync_result
        
    async def detect_conflicts(self, work_tree_path: Path) -> List[Dict[str, Any]]:
        """Detect and analyze merge conflicts."""
        conflicts = []
        
        try:
            # Get conflicted files
            status_code, status_output, _ = await self.execute_git_command(
                ["git", "status", "--porcelain"],
                work_tree_path
            )
            
            for line in status_output.strip().split('\n'):
                if line.startswith('UU ') or line.startswith('AA ') or line.startswith('DD '):
                    file_path = line[3:].strip()
                    conflict_info = await self.analyze_conflict(work_tree_path, file_path)
                    conflicts.append(conflict_info)
                    
        except Exception as e:
            logger.error(f"Failed to detect conflicts: {e}")
            
        return conflicts
        
    async def analyze_conflict(self, work_tree_path: Path, file_path: str) -> Dict[str, Any]:
        """Analyze specific file conflict."""
        conflict_info = {
            "file": file_path,
            "type": "merge_conflict",
            "markers": [],
            "complexity": "low"
        }
        
        try:
            full_path = work_tree_path / file_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Count conflict markers
                markers = {
                    "conflict_start": content.count('<<<<<<<'),
                    "conflict_middle": content.count('======='),
                    "conflict_end": content.count('>>>>>>>')
                }
                
                conflict_info["markers"] = markers
                
                # Determine complexity
                total_markers = sum(markers.values())
                if total_markers > 10:
                    conflict_info["complexity"] = "high"
                elif total_markers > 3:
                    conflict_info["complexity"] = "medium"
                    
        except Exception as e:
            logger.error(f"Failed to analyze conflict in {file_path}: {e}")
            
        return conflict_info
        
    async def get_repository_status(self, work_tree_path: Path) -> Dict[str, Any]:
        """Get comprehensive repository status."""
        status = {
            "clean": False,
            "branch": None,
            "ahead": 0,
            "behind": 0,
            "modified": [],
            "untracked": [],
            "staged": []
        }
        
        try:
            # Get current branch
            branch_code, branch_output, _ = await self.execute_git_command(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                work_tree_path
            )
            
            if branch_code == 0:
                status["branch"] = branch_output.strip()
                
            # Get status information
            status_code, status_output, _ = await self.execute_git_command(
                ["git", "status", "--porcelain", "-b"],
                work_tree_path
            )
            
            if status_code == 0:
                lines = status_output.strip().split('\n')
                
                for line in lines:
                    if line.startswith('##'):
                        # Parse branch tracking info
                        if '[ahead' in line:
                            ahead_part = line.split('[ahead ')[1].split(']')[0]
                            if ',' in ahead_part:
                                status["ahead"] = int(ahead_part.split(',')[0])
                            else:
                                status["ahead"] = int(ahead_part)
                        if '[behind' in line:
                            behind_part = line.split('[behind ')[1].split(']')[0]
                            status["behind"] = int(behind_part)
                    else:
                        # Parse file status
                        if len(line) >= 3:
                            index_status = line[0]
                            worktree_status = line[1]
                            filename = line[3:]
                            
                            if index_status != ' ':
                                status["staged"].append(filename)
                            if worktree_status == 'M':
                                status["modified"].append(filename)
                            elif worktree_status == '?':
                                status["untracked"].append(filename)
                                
                status["clean"] = not (status["modified"] or status["untracked"] or status["staged"])
                
        except Exception as e:
            logger.error(f"Failed to get repository status: {e}")
            
        return status


class WorkTreeManager:
    """
    High-level work tree management orchestrating all work tree operations.
    
    Provides the main interface for creating, managing, and cleaning up
    agent work trees with complete isolation and security.
    """
    
    def __init__(self, github_client: GitHubAPIClient = None):
        self.github_client = github_client or GitHubAPIClient()
        self.git_manager = GitWorkTreeManager(self.github_client)
        self.isolation_manager = WorkTreeIsolationManager()
        
    async def create_agent_work_tree(
        self,
        agent_id: str,
        repository: GitHubRepository,
        branch_name: str = None,
        base_branch: str = None
    ) -> AgentWorkTree:
        """Create isolated work tree for agent."""
        
        try:
            # Generate work tree configuration
            work_tree_path = self.isolation_manager.get_work_tree_path(
                agent_id, 
                repository.repository_full_name.replace('/', '_')
            )
            
            if not branch_name:
                branch_name = f"agent/{agent_id}/{uuid.uuid4().hex[:8]}"
                
            base_branch = base_branch or repository.default_branch
            
            # Clone repository
            clone_url = repository.get_clone_url_for_agent(with_token=True)
            await self.git_manager.clone_repository(
                clone_url,
                work_tree_path,
                branch=base_branch
            )
            
            # Create agent branch
            await self.git_manager.create_agent_branch(
                work_tree_path,
                branch_name,
                base_branch
            )
            
            # Create database record
            work_tree = AgentWorkTree(
                agent_id=uuid.UUID(agent_id),
                repository_id=repository.id,
                work_tree_path=str(work_tree_path),
                branch_name=branch_name,
                base_branch=base_branch,
                upstream_branch=f"origin/{branch_name}",
                status=WorkTreeStatus.ACTIVE,
                isolation_config=self.isolation_manager.isolation_config.copy()
            )
            
            async with get_db_session() as session:
                session.add(work_tree)
                await session.commit()
                await session.refresh(work_tree)
                
            logger.info(f"Created work tree for agent {agent_id}: {work_tree_path}")
            return work_tree
            
        except Exception as e:
            logger.error(f"Failed to create work tree for agent {agent_id}: {e}")
            # Cleanup on failure
            if 'work_tree_path' in locals():
                await self.isolation_manager.cleanup_work_tree(work_tree_path, force=True)
            raise WorkTreeError(f"Work tree creation failed: {str(e)}")
            
    async def get_agent_work_tree(
        self,
        agent_id: str,
        repository_id: str
    ) -> Optional[AgentWorkTree]:
        """Get existing work tree for agent and repository."""
        
        async with get_db_session() as session:
            result = await session.execute(
                select(AgentWorkTree).where(
                    and_(
                        AgentWorkTree.agent_id == uuid.UUID(agent_id),
                        AgentWorkTree.repository_id == uuid.UUID(repository_id),
                        AgentWorkTree.status == WorkTreeStatus.ACTIVE
                    )
                )
            )
            return result.scalar_one_or_none()
            
    async def sync_work_tree(self, work_tree: AgentWorkTree) -> Dict[str, Any]:
        """Sync work tree with upstream repository."""
        
        work_tree_path = Path(work_tree.work_tree_path)
        
        if not work_tree_path.exists():
            raise WorkTreeError(f"Work tree path does not exist: {work_tree_path}")
            
        # Update activity timestamp
        work_tree.update_activity()
        
        # Perform sync
        sync_result = await self.git_manager.sync_with_upstream(
            work_tree_path,
            work_tree.base_branch
        )
        
        # Update database with sync result
        async with get_db_session() as session:
            await session.merge(work_tree)
            await session.commit()
            
        return sync_result
        
    async def cleanup_work_tree(self, work_tree: AgentWorkTree, force: bool = False) -> bool:
        """Clean up work tree and mark as archived."""
        
        work_tree_path = Path(work_tree.work_tree_path)
        
        # Perform filesystem cleanup
        cleanup_success = await self.isolation_manager.cleanup_work_tree(work_tree_path, force)
        
        if cleanup_success:
            # Update database status
            work_tree.status = WorkTreeStatus.ARCHIVED
            work_tree.cleaned_at = datetime.utcnow()
            
            async with get_db_session() as session:
                await session.merge(work_tree)
                await session.commit()
                
            logger.info(f"Cleaned up work tree: {work_tree_path}")
            
        return cleanup_success
        
    async def cleanup_inactive_work_trees(self, max_age_days: int = 7) -> List[str]:
        """Clean up work trees that haven't been used recently."""
        
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        cleaned_work_trees = []
        
        async with get_db_session() as session:
            result = await session.execute(
                select(AgentWorkTree).where(
                    and_(
                        AgentWorkTree.status == WorkTreeStatus.ACTIVE,
                        AgentWorkTree.last_used < cutoff_date
                    )
                )
            )
            
            inactive_work_trees = result.scalars().all()
            
            for work_tree in inactive_work_trees:
                try:
                    if await self.cleanup_work_tree(work_tree):
                        cleaned_work_trees.append(work_tree.work_tree_path)
                except Exception as e:
                    logger.error(f"Failed to cleanup work tree {work_tree.work_tree_path}: {e}")
                    
        return cleaned_work_trees
        
    async def get_work_tree_status(self, work_tree: AgentWorkTree) -> Dict[str, Any]:
        """Get comprehensive work tree status including Git and resource info."""
        
        work_tree_path = Path(work_tree.work_tree_path)
        
        if not work_tree_path.exists():
            return {"exists": False, "error": "Work tree path does not exist"}
            
        # Get Git status
        git_status = await self.git_manager.get_repository_status(work_tree_path)
        
        # Get resource usage
        resource_status = await self.isolation_manager.check_resource_limits(work_tree_path)
        
        # Get security status
        security_status = {
            "path_secure": self.isolation_manager.check_path_security(work_tree_path),
            "permissions_valid": oct(work_tree_path.stat().st_mode)[-3:] == "750"
        }
        
        return {
            "exists": True,
            "work_tree_id": str(work_tree.id),
            "agent_id": str(work_tree.agent_id),
            "repository_id": str(work_tree.repository_id),
            "path": str(work_tree_path),
            "branch": work_tree.branch_name,
            "status": work_tree.status.value,
            "last_used": work_tree.last_used.isoformat() if work_tree.last_used else None,
            "git_status": git_status,
            "resource_status": resource_status,
            "security_status": security_status,
            "isolation_effective": (
                security_status["path_secure"] and 
                security_status["permissions_valid"] and 
                resource_status["valid"]
            )
        }
        
    async def list_agent_work_trees(self, agent_id: str) -> List[Dict[str, Any]]:
        """List all work trees for specific agent."""
        
        async with get_db_session() as session:
            result = await session.execute(
                select(AgentWorkTree).where(
                    AgentWorkTree.agent_id == uuid.UUID(agent_id)
                ).order_by(AgentWorkTree.last_used.desc())
            )
            
            work_trees = result.scalars().all()
            
            work_tree_list = []
            for work_tree in work_trees:
                status = await self.get_work_tree_status(work_tree)
                work_tree_list.append(status)
                
            return work_tree_list
            
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of work tree system."""
        
        health_status = {
            "healthy": True,
            "base_directory_exists": False,
            "base_directory_writable": False,
            "disk_space_available": False,
            "active_work_trees": 0,
            "total_work_trees": 0,
            "cleanup_needed": 0,
            "errors": []
        }
        
        try:
            # Check base directory
            await self.isolation_manager.ensure_base_directory()
            health_status["base_directory_exists"] = self.isolation_manager.base_path.exists()
            
            # Check write permissions
            test_file = self.isolation_manager.base_path / ".health_check"
            try:
                test_file.touch()
                test_file.unlink()
                health_status["base_directory_writable"] = True
            except:
                health_status["base_directory_writable"] = False
                health_status["errors"].append("Base directory not writable")
                
            # Check disk space
            disk_usage = psutil.disk_usage(str(self.isolation_manager.base_path))
            available_gb = disk_usage.free / (1024**3)
            health_status["disk_space_available"] = available_gb > 1.0  # At least 1GB free
            
            if not health_status["disk_space_available"]:
                health_status["errors"].append(f"Low disk space: {available_gb:.2f}GB available")
                
            # Check work tree counts
            async with get_db_session() as session:
                active_result = await session.execute(
                    select(AgentWorkTree).where(AgentWorkTree.status == WorkTreeStatus.ACTIVE)
                )
                total_result = await session.execute(select(AgentWorkTree))
                
                active_work_trees = active_result.scalars().all()
                total_work_trees = total_result.scalars().all()
                
                health_status["active_work_trees"] = len(active_work_trees)
                health_status["total_work_trees"] = len(total_work_trees)
                
                # Count work trees needing cleanup
                cutoff_date = datetime.utcnow() - timedelta(days=7)
                cleanup_needed = sum(
                    1 for wt in active_work_trees 
                    if wt.last_used and wt.last_used < cutoff_date
                )
                health_status["cleanup_needed"] = cleanup_needed
                
        except Exception as e:
            health_status["healthy"] = False
            health_status["errors"].append(f"Health check failed: {str(e)}")
            
        # Overall health determination
        health_status["healthy"] = (
            health_status["base_directory_exists"] and
            health_status["base_directory_writable"] and
            health_status["disk_space_available"] and
            len(health_status["errors"]) == 0
        )
        
        return health_status