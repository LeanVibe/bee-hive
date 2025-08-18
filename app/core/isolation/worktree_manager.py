"""
Git Worktree Manager for Multi-Agent Isolation

This module provides secure git worktree management enabling isolated execution
environments for each agent. It ensures path security, resource monitoring,
and proper cleanup to prevent cross-contamination between agent workspaces.
"""

import asyncio
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .path_validator import PathValidator
from .security_enforcer import SecurityEnforcer

logger = logging.getLogger(__name__)

# ================================================================================
# Data Models
# ================================================================================

@dataclass
class WorktreeContext:
    """Context information for an isolated git worktree."""
    
    # Core identification
    worktree_id: str
    agent_id: str
    branch_name: str
    
    # File system paths
    worktree_path: str
    base_repository_path: str
    
    # Security and isolation
    allowed_paths: List[str] = field(default_factory=list)
    restricted_patterns: List[str] = field(default_factory=list)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    # Lifecycle tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Status and metrics
    is_active: bool = True
    file_count: int = 0
    disk_usage_mb: float = 0.0
    access_violations: int = 0

@dataclass 
class WorktreeCreationRequest:
    """Request to create a new worktree."""
    
    agent_id: str
    branch_name: str
    base_path: str
    lifetime_minutes: int = 60
    max_disk_mb: float = 100.0
    max_files: int = 1000
    allowed_extensions: List[str] = field(default_factory=lambda: [
        '.py', '.js', '.ts', '.md', '.txt', '.json', '.yaml', '.yml'
    ])

# ================================================================================
# Git Worktree Manager
# ================================================================================

class WorktreeManager:
    """
    Manages isolated git worktrees for secure multi-agent coordination.
    
    Provides secure worktree creation, lifecycle management, and cleanup
    with comprehensive security validation and resource monitoring.
    
    Key Features:
    - Secure worktree creation with git isolation
    - Path traversal attack prevention
    - Resource usage monitoring and limits
    - Automatic cleanup and garbage collection
    - Agent workspace isolation
    """
    
    def __init__(self, base_worktree_dir: Optional[str] = None):
        """
        Initialize worktree manager.
        
        Args:
            base_worktree_dir: Base directory for all worktrees (default: temp dir)
        """
        # Configuration
        self._base_worktree_dir = base_worktree_dir or tempfile.mkdtemp(prefix="agent_worktrees_")
        self._max_concurrent_worktrees = 50
        self._default_lifetime_minutes = 60
        self._cleanup_interval_seconds = 300  # 5 minutes
        
        # State tracking
        self._active_worktrees: Dict[str, WorktreeContext] = {}
        self._agent_worktrees: Dict[str, Set[str]] = {}  # agent_id -> set of worktree_ids
        self._last_cleanup = datetime.utcnow()
        
        # Security components
        self._path_validator = PathValidator()
        self._security_enforcer = SecurityEnforcer()
        
        # Ensure base directory exists
        os.makedirs(self._base_worktree_dir, exist_ok=True)
        
        logger.info(f"WorktreeManager initialized with base directory: {self._base_worktree_dir}")
    
    # ================================================================================
    # Core Worktree Operations
    # ================================================================================
    
    async def create_worktree(
        self, 
        agent_id: str, 
        branch_name: str, 
        base_path: str,
        **kwargs
    ) -> WorktreeContext:
        """
        Create a new isolated git worktree for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            branch_name: Git branch to checkout in the worktree
            base_path: Base repository path
            **kwargs: Additional options (lifetime_minutes, max_disk_mb, etc.)
            
        Returns:
            WorktreeContext: Created worktree context with metadata
            
        Raises:
            SecurityError: If security validation fails
            ResourceLimitError: If resource limits are exceeded
            RuntimeError: If worktree creation fails
        """
        start_time = time.time()
        
        try:
            # 1. Validate inputs and check limits
            await self._validate_creation_request(agent_id, branch_name, base_path)
            
            # 2. Generate unique worktree ID and paths
            worktree_id = f"{agent_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            worktree_path = os.path.join(self._base_worktree_dir, worktree_id)
            
            # 3. Create creation request with defaults
            request = WorktreeCreationRequest(
                agent_id=agent_id,
                branch_name=branch_name,
                base_path=base_path,
                **kwargs
            )
            
            # 4. Validate base repository path
            if not await self._validate_base_repository(base_path):
                raise RuntimeError(f"Invalid base repository: {base_path}")
            
            # 5. Execute git worktree add command
            await self._execute_git_worktree_add(base_path, worktree_path, branch_name)
            
            # 6. Set up security constraints and permissions
            await self._setup_worktree_security(worktree_path, request)
            
            # 7. Create worktree context
            expires_at = datetime.utcnow() + timedelta(minutes=request.lifetime_minutes)
            context = WorktreeContext(
                worktree_id=worktree_id,
                agent_id=agent_id,
                branch_name=branch_name,
                worktree_path=worktree_path,
                base_repository_path=base_path,
                expires_at=expires_at,
                resource_limits={
                    "max_disk_mb": request.max_disk_mb,
                    "max_files": request.max_files,
                    "allowed_extensions": request.allowed_extensions
                }
            )
            
            # 8. Register worktree
            self._active_worktrees[worktree_id] = context
            if agent_id not in self._agent_worktrees:
                self._agent_worktrees[agent_id] = set()
            self._agent_worktrees[agent_id].add(worktree_id)
            
            # 9. Update metrics
            context.file_count = await self._count_files(worktree_path)
            context.disk_usage_mb = await self._calculate_disk_usage(worktree_path)
            
            execution_time = time.time() - start_time
            logger.info(
                f"Worktree created successfully: {worktree_id} "
                f"for agent {agent_id} in {execution_time:.2f}s"
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Worktree creation failed for agent {agent_id}: {e}")
            # Cleanup any partial state
            try:
                if 'worktree_path' in locals() and os.path.exists(worktree_path):
                    shutil.rmtree(worktree_path, ignore_errors=True)
            except Exception:
                pass
            raise
    
    async def cleanup_worktree(self, worktree_id: str) -> bool:
        """
        Clean up and remove a worktree.
        
        Args:
            worktree_id: ID of worktree to clean up
            
        Returns:
            bool: True if cleanup successful
        """
        start_time = time.time()
        
        try:
            # 1. Get worktree context
            context = self._active_worktrees.get(worktree_id)
            if not context:
                logger.warning(f"Worktree {worktree_id} not found for cleanup")
                return False
            
            logger.info(f"Starting cleanup for worktree {worktree_id}")
            
            # 2. Mark as inactive
            context.is_active = False
            
            # 3. Execute git worktree remove
            success = await self._execute_git_worktree_remove(
                context.base_repository_path, 
                context.worktree_path
            )
            
            # 4. Force remove directory if git command failed
            if not success and os.path.exists(context.worktree_path):
                logger.warning(f"Git worktree remove failed, forcing directory removal")
                shutil.rmtree(context.worktree_path, ignore_errors=True)
                success = True
            
            # 5. Remove from tracking
            self._active_worktrees.pop(worktree_id, None)
            if context.agent_id in self._agent_worktrees:
                self._agent_worktrees[context.agent_id].discard(worktree_id)
                if not self._agent_worktrees[context.agent_id]:
                    del self._agent_worktrees[context.agent_id]
            
            execution_time = time.time() - start_time
            logger.info(f"Worktree cleanup completed: {worktree_id} in {execution_time:.2f}s")
            
            return success
            
        except Exception as e:
            logger.error(f"Worktree cleanup failed for {worktree_id}: {e}")
            return False
    
    async def validate_path_access(self, worktree_id: str, file_path: str) -> bool:
        """
        Validate that a file path is safe for access within the worktree.
        
        Args:
            worktree_id: ID of the worktree
            file_path: File path to validate
            
        Returns:
            bool: True if access is allowed
        """
        try:
            # 1. Get worktree context
            context = self._active_worktrees.get(worktree_id)
            if not context or not context.is_active:
                return False
            
            # 2. Use path validator for security checks
            return self._path_validator.validate_file_access(
                context.worktree_path, 
                file_path
            )
            
        except Exception as e:
            logger.error(f"Path validation failed for {worktree_id}, {file_path}: {e}")
            return False
    
    # ================================================================================
    # Git Operations
    # ================================================================================
    
    async def _execute_git_worktree_add(
        self, 
        base_path: str, 
        worktree_path: str, 
        branch_name: str
    ) -> None:
        """Execute git worktree add command."""
        try:
            # Build git command
            cmd = [
                'git', '-C', base_path, 
                'worktree', 'add', 
                worktree_path, branch_name
            ]
            
            # Execute command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=30
            )
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='replace')
                raise RuntimeError(f"Git worktree add failed: {error_msg}")
            
            logger.debug(f"Git worktree add successful: {worktree_path}")
            
        except asyncio.TimeoutError:
            raise RuntimeError("Git worktree add timed out")
        except Exception as e:
            raise RuntimeError(f"Git worktree add failed: {e}")
    
    async def _execute_git_worktree_remove(
        self, 
        base_path: str, 
        worktree_path: str
    ) -> bool:
        """Execute git worktree remove command."""
        try:
            # Build git command
            cmd = [
                'git', '-C', base_path,
                'worktree', 'remove', 
                '--force', worktree_path
            ]
            
            # Execute command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30
            )
            
            success = process.returncode == 0
            if not success:
                error_msg = stderr.decode('utf-8', errors='replace')
                logger.warning(f"Git worktree remove failed: {error_msg}")
            
            return success
            
        except Exception as e:
            logger.error(f"Git worktree remove error: {e}")
            return False
    
    async def _validate_base_repository(self, base_path: str) -> bool:
        """Validate that base path is a valid git repository."""
        try:
            # Check if directory exists
            if not os.path.isdir(base_path):
                return False
            
            # Check if it's a git repository
            cmd = ['git', '-C', base_path, 'rev-parse', '--git-dir']
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=5
            )
            
            return process.returncode == 0
            
        except Exception:
            return False
    
    # ================================================================================
    # Security and Validation
    # ================================================================================
    
    async def _validate_creation_request(
        self, 
        agent_id: str, 
        branch_name: str, 
        base_path: str
    ) -> None:
        """Validate worktree creation request."""
        # 1. Check agent ID
        if not agent_id or not agent_id.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Invalid agent ID: {agent_id}")
        
        # 2. Check branch name
        if not branch_name or '..' in branch_name or '/' in branch_name[:1]:
            raise ValueError(f"Invalid branch name: {branch_name}")
        
        # 3. Check base path security
        if not self._path_validator.check_security_constraints(base_path):
            raise SecurityError(f"Base path violates security constraints: {base_path}")
        
        # 4. Check concurrent worktree limits
        if len(self._active_worktrees) >= self._max_concurrent_worktrees:
            raise ResourceLimitError("Maximum concurrent worktrees exceeded")
        
        # 5. Check per-agent limits
        agent_worktree_count = len(self._agent_worktrees.get(agent_id, set()))
        if agent_worktree_count >= 5:  # Max 5 worktrees per agent
            raise ResourceLimitError(f"Agent {agent_id} has too many active worktrees")
    
    async def _setup_worktree_security(
        self, 
        worktree_path: str, 
        request: WorktreeCreationRequest
    ) -> None:
        """Set up security constraints for the worktree."""
        try:
            # 1. Set directory permissions (readable/writable by owner only)
            os.chmod(worktree_path, 0o700)
            
            # 2. Create .gitignore for sensitive files
            gitignore_path = os.path.join(worktree_path, '.gitignore')
            gitignore_content = """
# Security - never commit these
*.key
*.pem
*.p12
*.password
*.secret
.env
.env.*
config/secrets/
"""
            with open(gitignore_path, 'w') as f:
                f.write(gitignore_content.strip())
            
            # 3. Set up resource monitoring
            await self._security_enforcer.setup_monitoring(
                worktree_path, 
                request.max_disk_mb,
                request.max_files
            )
            
            logger.debug(f"Security setup completed for worktree: {worktree_path}")
            
        except Exception as e:
            logger.error(f"Security setup failed for {worktree_path}: {e}")
            raise
    
    # ================================================================================
    # Resource Monitoring
    # ================================================================================
    
    async def _count_files(self, worktree_path: str) -> int:
        """Count files in worktree."""
        try:
            count = 0
            for root, dirs, files in os.walk(worktree_path):
                count += len(files)
                # Skip .git directory
                if '.git' in dirs:
                    dirs.remove('.git')
            return count
        except Exception:
            return 0
    
    async def _calculate_disk_usage(self, worktree_path: str) -> float:
        """Calculate disk usage in MB."""
        try:
            total_size = 0
            for root, dirs, files in os.walk(worktree_path):
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
                    except Exception:
                        continue
                # Skip .git directory
                if '.git' in dirs:
                    dirs.remove('.git')
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0
    
    # ================================================================================
    # Lifecycle Management
    # ================================================================================
    
    async def cleanup_expired_worktrees(self) -> int:
        """Clean up expired worktrees. Returns count of cleaned up worktrees."""
        cleaned_count = 0
        current_time = datetime.utcnow()
        
        # Find expired worktrees
        expired_ids = []
        for worktree_id, context in self._active_worktrees.items():
            if context.expires_at and current_time > context.expires_at:
                expired_ids.append(worktree_id)
        
        # Clean up expired worktrees
        for worktree_id in expired_ids:
            if await self.cleanup_worktree(worktree_id):
                cleaned_count += 1
        
        self._last_cleanup = current_time
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired worktrees")
        
        return cleaned_count
    
    async def get_worktree_status(self, worktree_id: str) -> Optional[Dict[str, Any]]:
        """Get status information for a worktree."""
        context = self._active_worktrees.get(worktree_id)
        if not context:
            return None
        
        # Update metrics
        context.file_count = await self._count_files(context.worktree_path)
        context.disk_usage_mb = await self._calculate_disk_usage(context.worktree_path)
        context.last_accessed = datetime.utcnow()
        
        return {
            "worktree_id": context.worktree_id,
            "agent_id": context.agent_id,
            "branch_name": context.branch_name,
            "worktree_path": context.worktree_path,
            "is_active": context.is_active,
            "created_at": context.created_at.isoformat(),
            "last_accessed": context.last_accessed.isoformat(),
            "expires_at": context.expires_at.isoformat() if context.expires_at else None,
            "file_count": context.file_count,
            "disk_usage_mb": context.disk_usage_mb,
            "resource_limits": context.resource_limits
        }
    
    def get_agent_worktrees(self, agent_id: str) -> List[str]:
        """Get list of active worktree IDs for an agent."""
        return list(self._agent_worktrees.get(agent_id, set()))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall worktree manager statistics."""
        total_disk_usage = sum(
            context.disk_usage_mb for context in self._active_worktrees.values()
        )
        
        return {
            "active_worktrees": len(self._active_worktrees),
            "unique_agents": len(self._agent_worktrees),
            "total_disk_usage_mb": total_disk_usage,
            "base_directory": self._base_worktree_dir,
            "last_cleanup": self._last_cleanup.isoformat(),
            "max_concurrent": self._max_concurrent_worktrees
        }

# ================================================================================
# Exception Classes
# ================================================================================

class SecurityError(Exception):
    """Raised when security constraints are violated."""
    pass

class ResourceLimitError(Exception):
    """Raised when resource limits are exceeded."""
    pass