"""
Git Checkpoint Optimizer - Advanced Git-based state versioning and cleanup.

Provides optimized Git-based checkpoint operations including:
- Intelligent branching strategies for agent and system checkpoints
- Automated cleanup with history preservation policies  
- Performance optimization for large checkpoint repositories
- Advanced Git operations with compression and garbage collection
- Branch management and merge optimization
- Historical analysis and space utilization tracking
"""

import asyncio
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID
import git
from git.exc import GitCommandError, InvalidGitRepositoryError

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc

from ..models.sleep_wake import Checkpoint, CheckpointType
from ..core.database import get_async_session
from ..core.config import get_settings


logger = logging.getLogger(__name__)


class GitRepositoryStats:
    """Statistics for Git repository analysis."""
    
    def __init__(self):
        self.total_commits: int = 0
        self.total_branches: int = 0
        self.repository_size_mb: float = 0.0
        self.oldest_commit_date: Optional[datetime] = None
        self.newest_commit_date: Optional[datetime] = None
        self.branches_by_type: Dict[str, int] = {}
        self.commits_by_month: Dict[str, int] = {}
        self.large_files: List[Dict[str, Any]] = []
        self.optimization_opportunities: List[str] = []


class BranchManagementStrategy:
    """Strategy for managing Git branches in checkpoint repositories."""
    
    def __init__(self):
        # Branch naming conventions
        self.agent_branch_prefix = "agent"
        self.system_branch_prefix = "system"
        self.archive_branch_prefix = "archive"
        self.temp_branch_prefix = "temp"
        
        # Retention policies
        self.max_agent_branches = 50
        self.max_system_branches = 20
        self.archive_after_days = 90
        self.cleanup_temp_branches_after_hours = 24
        
        # Performance settings
        self.max_commits_per_branch = 100
        self.enable_branch_compression = True
        self.enable_automatic_gc = True


class GitCheckpointOptimizer:
    """
    Advanced Git checkpoint optimization with intelligent cleanup and performance tuning.
    
    Features:
    - Intelligent branch management with agent-specific strategies
    - Automated cleanup with configurable retention policies
    - Repository optimization and garbage collection
    - Performance monitoring and space utilization analysis
    - Advanced Git operations with error recovery
    - Historical analysis and reporting
    """
    
    def __init__(self, repository_path: Path):
        self.settings = get_settings()
        self.repository_path = repository_path
        self.repo: Optional[git.Repo] = None
        
        # Strategy and configuration
        self.branch_strategy = BranchManagementStrategy()
        
        # Performance tracking
        self.optimization_history: List[Dict[str, Any]] = []
        self.last_optimization: Optional[datetime] = None
        
        # Cleanup settings
        self.enable_aggressive_cleanup = False
        self.preserve_critical_branches = True
        self.backup_before_cleanup = True
        
        # Repository statistics
        self._repo_stats: Optional[GitRepositoryStats] = None
        self._stats_cache_time: Optional[datetime] = None
        self._stats_cache_duration = timedelta(hours=1)
    
    async def initialize_repository(self) -> bool:
        """Initialize or validate Git repository for checkpoint optimization."""
        try:
            if self.repository_path.exists() and (self.repository_path / ".git").exists():
                # Open existing repository
                self.repo = git.Repo(self.repository_path)
                logger.info(f"Opened existing checkpoint repository at {self.repository_path}")
            else:
                # Initialize new repository
                self.repository_path.mkdir(parents=True, exist_ok=True)
                self.repo = git.Repo.init(self.repository_path)
                await self._setup_initial_repository_structure()
                logger.info(f"Initialized new checkpoint repository at {self.repository_path}")
            
            # Configure repository for optimization
            await self._configure_repository_for_performance()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Git repository: {e}")
            return False
    
    async def optimize_repository(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive repository optimization.
        
        Args:
            aggressive: Enable aggressive optimization (may take longer)
            
        Returns:
            Optimization results and statistics
        """
        start_time = time.time()
        optimization_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "aggressive_mode": aggressive,
            "operations_performed": [],
            "space_saved_mb": 0.0,
            "performance_improvements": {},
            "errors": []
        }
        
        try:
            logger.info(f"Starting Git repository optimization (aggressive: {aggressive})")
            
            # Get initial repository statistics
            initial_stats = await self.get_repository_statistics()
            initial_size = initial_stats.repository_size_mb
            
            # Phase 1: Cleanup old and unnecessary branches
            cleanup_result = await self._cleanup_branches()
            optimization_results["operations_performed"].extend(cleanup_result["operations"])
            
            # Phase 2: Garbage collection and pruning
            gc_result = await self._perform_garbage_collection(aggressive)
            optimization_results["operations_performed"].extend(gc_result["operations"])
            
            # Phase 3: Compress Git objects
            compression_result = await self._compress_git_objects()
            optimization_results["operations_performed"].extend(compression_result["operations"])
            
            # Phase 4: Optimize pack files
            if aggressive:
                pack_result = await self._optimize_pack_files()
                optimization_results["operations_performed"].extend(pack_result["operations"])
            
            # Phase 5: Cleanup loose objects
            loose_cleanup_result = await self._cleanup_loose_objects()
            optimization_results["operations_performed"].extend(loose_cleanup_result["operations"])
            
            # Calculate space savings
            final_stats = await self.get_repository_statistics(force_refresh=True)
            final_size = final_stats.repository_size_mb
            space_saved = max(0, initial_size - final_size)
            optimization_results["space_saved_mb"] = space_saved
            
            # Performance improvements
            optimization_time = time.time() - start_time
            optimization_results["performance_improvements"] = {
                "optimization_time_seconds": optimization_time,
                "space_reduction_percentage": (space_saved / initial_size * 100) if initial_size > 0 else 0,
                "final_repository_size_mb": final_size,
                "total_commits": final_stats.total_commits,
                "total_branches": final_stats.total_branches
            }
            
            # Record optimization
            self.last_optimization = datetime.utcnow()
            self.optimization_history.append(optimization_results.copy())
            
            logger.info(
                f"Repository optimization completed in {optimization_time:.2f}s: "
                f"{space_saved:.1f}MB saved ({optimization_results['performance_improvements']['space_reduction_percentage']:.1f}% reduction)"
            )
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error during repository optimization: {e}")
            optimization_results["errors"].append(str(e))
            return optimization_results
    
    async def get_repository_statistics(self, force_refresh: bool = False) -> GitRepositoryStats:
        """Get comprehensive repository statistics with caching."""
        try:
            # Check cache validity
            if (not force_refresh and self._repo_stats and self._stats_cache_time and 
                datetime.utcnow() - self._stats_cache_time < self._stats_cache_duration):
                return self._repo_stats
            
            if not self.repo:
                raise ValueError("Repository not initialized")
            
            stats = GitRepositoryStats()
            
            # Basic repository information
            stats.total_commits = len(list(self.repo.iter_commits('--all')))
            stats.total_branches = len(list(self.repo.branches))
            
            # Repository size
            stats.repository_size_mb = await self._calculate_repository_size()
            
            # Commit date range
            if stats.total_commits > 0:
                commits = list(self.repo.iter_commits('--all', max_count=1))
                if commits:
                    stats.newest_commit_date = datetime.fromtimestamp(commits[0].committed_date)
                
                # Get oldest commit (expensive operation, limit to 1000 commits)
                oldest_commits = list(self.repo.iter_commits('--all', reverse=True, max_count=1))
                if oldest_commits:
                    stats.oldest_commit_date = datetime.fromtimestamp(oldest_commits[0].committed_date)
            
            # Branch analysis
            stats.branches_by_type = await self._analyze_branches()
            
            # Commit frequency analysis
            stats.commits_by_month = await self._analyze_commit_frequency()
            
            # Large file analysis
            stats.large_files = await self._find_large_files()
            
            # Optimization opportunities
            stats.optimization_opportunities = await self._identify_optimization_opportunities(stats)
            
            # Cache results
            self._repo_stats = stats
            self._stats_cache_time = datetime.utcnow()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting repository statistics: {e}")
            return GitRepositoryStats()  # Return empty stats on error
    
    async def create_optimized_branch(self, branch_name: str, branch_type: str = "agent") -> bool:
        """Create a new branch with optimized settings."""
        try:
            if not self.repo:
                return False
            
            # Check if branch already exists
            if branch_name in [branch.name for branch in self.repo.branches]:
                logger.warning(f"Branch {branch_name} already exists")
                return True
            
            # Create branch from current HEAD
            new_branch = self.repo.create_head(branch_name)
            
            # Configure branch-specific settings
            await self._configure_branch(new_branch, branch_type)
            
            logger.info(f"Created optimized {branch_type} branch: {branch_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating optimized branch {branch_name}: {e}")
            return False
    
    async def archive_old_branches(self, age_threshold: timedelta = None) -> Dict[str, Any]:
        """Archive old branches to reduce repository size."""
        if age_threshold is None:
            age_threshold = timedelta(days=self.branch_strategy.archive_after_days)
        
        archive_results = {
            "archived_branches": [],
            "space_saved_mb": 0.0,
            "errors": []
        }
        
        try:
            if not self.repo:
                return archive_results
            
            cutoff_date = datetime.utcnow() - age_threshold
            
            for branch in self.repo.branches:
                try:
                    # Get last commit date for branch
                    last_commit = branch.commit
                    last_commit_date = datetime.fromtimestamp(last_commit.committed_date)
                    
                    if last_commit_date < cutoff_date:
                        # Check if branch should be preserved
                        if await self._should_preserve_branch(branch):
                            continue
                        
                        # Create archive tag before deleting branch
                        archive_tag_name = f"archive/{branch.name}/{int(time.time())}"
                        self.repo.create_tag(archive_tag_name, branch.commit, 
                                           message=f"Archived branch {branch.name}")
                        
                        # Calculate space usage before deletion
                        branch_size = await self._estimate_branch_size(branch)
                        
                        # Delete the branch
                        self.repo.delete_head(branch, force=True)
                        
                        archive_results["archived_branches"].append({
                            "branch_name": branch.name,
                            "last_commit_date": last_commit_date.isoformat(),
                            "archive_tag": archive_tag_name,
                            "estimated_size_mb": branch_size
                        })
                        archive_results["space_saved_mb"] += branch_size
                        
                        logger.info(f"Archived branch {branch.name} as tag {archive_tag_name}")
                
                except Exception as e:
                    logger.error(f"Error archiving branch {branch.name}: {e}")
                    archive_results["errors"].append(f"Branch {branch.name}: {str(e)}")
            
            return archive_results
            
        except Exception as e:
            logger.error(f"Error during branch archival: {e}")
            archive_results["errors"].append(str(e))
            return archive_results
    
    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for repository optimization."""
        recommendations = []
        
        try:
            stats = await self.get_repository_statistics()
            
            # Repository size recommendations
            if stats.repository_size_mb > 1000:  # 1GB
                recommendations.append({
                    "type": "size_optimization",
                    "priority": "high",
                    "title": "Large Repository Size",
                    "description": f"Repository is {stats.repository_size_mb:.1f}MB. Consider aggressive cleanup.",
                    "action": "run_aggressive_optimization",
                    "estimated_savings_mb": stats.repository_size_mb * 0.3
                })
            
            # Branch count recommendations
            if stats.total_branches > self.branch_strategy.max_agent_branches:
                recommendations.append({
                    "type": "branch_cleanup",
                    "priority": "medium",
                    "title": "Too Many Branches",
                    "description": f"{stats.total_branches} branches found. Consider archiving old branches.",
                    "action": "archive_old_branches",
                    "estimated_savings_mb": stats.repository_size_mb * 0.2
                })
            
            # Large file recommendations
            if stats.large_files:
                total_large_size = sum(f["size_mb"] for f in stats.large_files)
                recommendations.append({
                    "type": "large_files",
                    "priority": "medium",
                    "title": "Large Files Detected",
                    "description": f"{len(stats.large_files)} large files totaling {total_large_size:.1f}MB",
                    "action": "compress_large_files",
                    "estimated_savings_mb": total_large_size * 0.5
                })
            
            # Optimization frequency recommendations
            if self.last_optimization:
                days_since_optimization = (datetime.utcnow() - self.last_optimization).days
                if days_since_optimization > 7:
                    recommendations.append({
                        "type": "maintenance",
                        "priority": "low",
                        "title": "Regular Maintenance Due",
                        "description": f"Last optimization was {days_since_optimization} days ago",
                        "action": "run_regular_optimization",
                        "estimated_savings_mb": 50
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {e}")
            return []
    
    async def _setup_initial_repository_structure(self) -> None:
        """Set up initial repository structure and configuration."""
        try:
            if not self.repo:
                return
            
            # Create directory structure
            directories = ["agents", "system", "archive", "metadata"]
            for directory in directories:
                (self.repository_path / directory).mkdir(exist_ok=True)
            
            # Create initial files
            gitignore_content = """
# Temporary files
*.tmp
*.temp
.DS_Store
Thumbs.db

# Large binary files
*.bin
*.data

# Backup files
*.bak
*.backup

# System files
.system/
"""
            gitignore_path = self.repository_path / ".gitignore"
            gitignore_path.write_text(gitignore_content.strip())
            
            readme_content = """# LeanVibe Agent Hive Checkpoints

Optimized Git repository for agent checkpoint storage and versioning.

## Structure

- `agents/`: Agent-specific checkpoint branches
- `system/`: System-wide checkpoint branches  
- `archive/`: Archived checkpoint references
- `metadata/`: Repository metadata and indexes

## Optimization Features

- Automatic branch cleanup and archival
- Intelligent garbage collection
- Large file compression
- Performance monitoring
"""
            readme_path = self.repository_path / "README.md"
            readme_path.write_text(readme_content.strip())
            
            # Initial commit
            self.repo.index.add([".gitignore", "README.md"])
            self.repo.index.commit("Initial optimized checkpoint repository setup")
            
        except Exception as e:
            logger.error(f"Error setting up repository structure: {e}")
    
    async def _configure_repository_for_performance(self) -> None:
        """Configure Git repository for optimal performance."""
        try:
            if not self.repo:
                return
            
            with self.repo.config_writer() as config:
                # Performance optimizations
                config.set_value("core", "preloadindex", "true")
                config.set_value("core", "compression", "9")  # Maximum compression
                config.set_value("gc", "auto", "1")
                config.set_value("gc", "autoPackLimit", "50")
                config.set_value("gc", "autoDetach", "false")
                
                # Optimize for checkpoint operations
                config.set_value("pack", "window", "250")
                config.set_value("pack", "depth", "50")
                config.set_value("pack", "compression", "9")
                
                # Enable advanced features
                config.set_value("feature", "manyFiles", "true")
                config.set_value("index", "version", "4")
                
                # Configure user (if not set)
                try:
                    config.get_value("user", "name")
                except:
                    config.set_value("user", "name", "LeanVibe Checkpoint Optimizer")
                    config.set_value("user", "email", "checkpoints@leanvibe.com")
            
            logger.debug("Configured repository for optimal performance")
            
        except Exception as e:
            logger.error(f"Error configuring repository performance: {e}")
    
    async def _cleanup_branches(self) -> Dict[str, Any]:
        """Clean up old and unnecessary branches."""
        cleanup_result = {
            "operations": [],
            "branches_removed": 0,
            "space_saved_mb": 0.0
        }
        
        try:
            # Archive old branches
            archive_result = await self.archive_old_branches()
            cleanup_result["branches_removed"] = len(archive_result["archived_branches"])
            cleanup_result["space_saved_mb"] = archive_result["space_saved_mb"]
            
            if cleanup_result["branches_removed"] > 0:
                cleanup_result["operations"].append(
                    f"Archived {cleanup_result['branches_removed']} old branches"
                )
            
            # Clean up temporary branches
            temp_branches_cleaned = await self._cleanup_temporary_branches()
            if temp_branches_cleaned > 0:
                cleanup_result["operations"].append(
                    f"Cleaned up {temp_branches_cleaned} temporary branches"
                )
            
            return cleanup_result
            
        except Exception as e:
            logger.error(f"Error during branch cleanup: {e}")
            return cleanup_result
    
    async def _perform_garbage_collection(self, aggressive: bool = False) -> Dict[str, Any]:
        """Perform Git garbage collection."""
        gc_result = {
            "operations": [],
            "success": False
        }
        
        try:
            if not self.repo:
                return gc_result
            
            # Standard garbage collection
            self.repo.git.gc()
            gc_result["operations"].append("Performed standard garbage collection")
            
            if aggressive:
                # Aggressive garbage collection
                self.repo.git.gc('--aggressive', '--prune=now')
                gc_result["operations"].append("Performed aggressive garbage collection")
            
            gc_result["success"] = True
            return gc_result
            
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")
            return gc_result
    
    async def _compress_git_objects(self) -> Dict[str, Any]:
        """Compress Git objects for space efficiency."""
        compression_result = {
            "operations": [],
            "success": False
        }
        
        try:
            if not self.repo:
                return compression_result
            
            # Repack objects with compression
            self.repo.git.repack('-a', '-d', '-f')
            compression_result["operations"].append("Repacked objects with compression")
            
            compression_result["success"] = True
            return compression_result
            
        except Exception as e:
            logger.error(f"Error during object compression: {e}")
            return compression_result
    
    async def _optimize_pack_files(self) -> Dict[str, Any]:
        """Optimize Git pack files for better performance."""
        pack_result = {
            "operations": [],
            "success": False
        }
        
        try:
            if not self.repo:
                return pack_result
            
            # Optimize pack files
            self.repo.git.repack('-a', '-d', '--depth=250', '--window=250')
            pack_result["operations"].append("Optimized pack files with deep delta compression")
            
            pack_result["success"] = True
            return pack_result
            
        except Exception as e:
            logger.error(f"Error during pack file optimization: {e}")
            return pack_result
    
    async def _cleanup_loose_objects(self) -> Dict[str, Any]:
        """Clean up loose Git objects."""
        cleanup_result = {
            "operations": [],
            "success": False
        }
        
        try:
            if not self.repo:
                return cleanup_result
            
            # Prune loose objects older than 2 weeks
            self.repo.git.prune()
            cleanup_result["operations"].append("Pruned old loose objects")
            
            cleanup_result["success"] = True
            return cleanup_result
            
        except Exception as e:
            logger.error(f"Error during loose object cleanup: {e}")
            return cleanup_result
    
    async def _calculate_repository_size(self) -> float:
        """Calculate repository size in megabytes."""
        try:
            if not self.repository_path.exists():
                return 0.0
            
            total_size = 0
            for path in self.repository_path.rglob("*"):
                if path.is_file():
                    total_size += path.stat().st_size
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.error(f"Error calculating repository size: {e}")
            return 0.0
    
    async def _analyze_branches(self) -> Dict[str, int]:
        """Analyze branches by type."""
        try:
            if not self.repo:
                return {}
            
            branch_types = {
                "agent": 0,
                "system": 0,
                "archive": 0,
                "temp": 0,
                "other": 0
            }
            
            for branch in self.repo.branches:
                if branch.name.startswith(self.branch_strategy.agent_branch_prefix):
                    branch_types["agent"] += 1
                elif branch.name.startswith(self.branch_strategy.system_branch_prefix):
                    branch_types["system"] += 1
                elif branch.name.startswith(self.branch_strategy.archive_branch_prefix):
                    branch_types["archive"] += 1
                elif branch.name.startswith(self.branch_strategy.temp_branch_prefix):
                    branch_types["temp"] += 1
                else:
                    branch_types["other"] += 1
            
            return branch_types
            
        except Exception as e:
            logger.error(f"Error analyzing branches: {e}")
            return {}
    
    async def _analyze_commit_frequency(self) -> Dict[str, int]:
        """Analyze commit frequency by month."""
        try:
            if not self.repo:
                return {}
            
            commits_by_month = {}
            
            # Analyze last 12 months of commits
            for commit in self.repo.iter_commits('--all', max_count=1000):
                commit_date = datetime.fromtimestamp(commit.committed_date)
                month_key = commit_date.strftime("%Y-%m")
                commits_by_month[month_key] = commits_by_month.get(month_key, 0) + 1
            
            return commits_by_month
            
        except Exception as e:
            logger.error(f"Error analyzing commit frequency: {e}")
            return {}
    
    async def _find_large_files(self) -> List[Dict[str, Any]]:
        """Find large files in the repository."""
        try:
            large_files = []
            size_threshold_mb = 10  # 10MB threshold
            
            if not self.repository_path.exists():
                return large_files
            
            for path in self.repository_path.rglob("*"):
                if path.is_file():
                    size_mb = path.stat().st_size / (1024 * 1024)
                    if size_mb > size_threshold_mb:
                        large_files.append({
                            "path": str(path.relative_to(self.repository_path)),
                            "size_mb": size_mb,
                            "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                        })
            
            # Sort by size (largest first)
            large_files.sort(key=lambda x: x["size_mb"], reverse=True)
            
            return large_files[:20]  # Return top 20 largest files
            
        except Exception as e:
            logger.error(f"Error finding large files: {e}")
            return []
    
    async def _identify_optimization_opportunities(self, stats: GitRepositoryStats) -> List[str]:
        """Identify optimization opportunities based on repository statistics."""
        opportunities = []
        
        try:
            # Large repository size
            if stats.repository_size_mb > 500:
                opportunities.append(f"Large repository size ({stats.repository_size_mb:.1f}MB) - consider aggressive cleanup")
            
            # Too many branches
            if stats.total_branches > 30:
                opportunities.append(f"High branch count ({stats.total_branches}) - consider branch archival")
            
            # Old repository without recent optimization
            if self.last_optimization is None:
                opportunities.append("No previous optimization detected - run initial optimization")
            elif (datetime.utcnow() - self.last_optimization).days > 30:
                opportunities.append("Repository hasn't been optimized in over 30 days")
            
            # Large files present
            if stats.large_files:
                total_large_size = sum(f["size_mb"] for f in stats.large_files)
                opportunities.append(f"Large files detected ({total_large_size:.1f}MB total) - consider compression")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying optimization opportunities: {e}")
            return []
    
    async def _configure_branch(self, branch: git.Head, branch_type: str) -> None:
        """Configure branch-specific settings."""
        try:
            # Branch-specific configuration would go here
            # For now, just log the configuration
            logger.debug(f"Configured {branch_type} branch: {branch.name}")
            
        except Exception as e:
            logger.error(f"Error configuring branch {branch.name}: {e}")
    
    async def _should_preserve_branch(self, branch: git.Head) -> bool:
        """Determine if a branch should be preserved during cleanup."""
        try:
            # Preserve main/master branches
            if branch.name in ["main", "master"]:
                return True
            
            # Preserve recent branches (last 7 days)
            last_commit_date = datetime.fromtimestamp(branch.commit.committed_date)
            if (datetime.utcnow() - last_commit_date).days < 7:
                return True
            
            # Preserve branches with recent database checkpoints
            if await self._has_recent_checkpoints(branch.name):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if branch {branch.name} should be preserved: {e}")
            return True  # Err on the side of caution
    
    async def _has_recent_checkpoints(self, branch_name: str) -> bool:
        """Check if branch has recent checkpoints in database."""
        try:
            # Extract agent ID or system identifier from branch name
            if "/" in branch_name:
                branch_parts = branch_name.split("/")
                if len(branch_parts) >= 2:
                    identifier = branch_parts[1]
                    
                    # Check for recent checkpoints
                    async with get_async_session() as session:
                        recent_checkpoints = await session.execute(
                            select(func.count(Checkpoint.id)).where(
                                and_(
                                    Checkpoint.checkpoint_metadata.contains({"git_commit_hash"}),
                                    Checkpoint.created_at >= datetime.utcnow() - timedelta(days=30)
                                )
                            )
                        )
                        
                        count = recent_checkpoints.scalar() or 0
                        return count > 0
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking recent checkpoints for branch {branch_name}: {e}")
            return False
    
    async def _estimate_branch_size(self, branch: git.Head) -> float:
        """Estimate the size contribution of a branch in MB."""
        try:
            # This is a rough estimate - actual implementation would need
            # more sophisticated Git object analysis
            commit_count = len(list(self.repo.iter_commits(branch.name, max_count=1000)))
            estimated_size_mb = commit_count * 0.1  # Rough estimate: 0.1MB per commit
            
            return estimated_size_mb
            
        except Exception as e:
            logger.error(f"Error estimating size for branch {branch.name}: {e}")
            return 0.0
    
    async def _cleanup_temporary_branches(self) -> int:
        """Clean up temporary branches older than threshold."""
        try:
            if not self.repo:
                return 0
            
            cleaned_count = 0
            cutoff_time = datetime.utcnow() - timedelta(hours=self.branch_strategy.cleanup_temp_branches_after_hours)
            
            for branch in self.repo.branches:
                if branch.name.startswith(self.branch_strategy.temp_branch_prefix):
                    last_commit_date = datetime.fromtimestamp(branch.commit.committed_date)
                    if last_commit_date < cutoff_time:
                        self.repo.delete_head(branch, force=True)
                        cleaned_count += 1
                        logger.debug(f"Cleaned up temporary branch: {branch.name}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up temporary branches: {e}")
            return 0


# Global Git checkpoint optimizer instance
_git_optimizer_instance: Optional[GitCheckpointOptimizer] = None


def get_git_checkpoint_optimizer(repository_path: Path) -> GitCheckpointOptimizer:
    """Get Git checkpoint optimizer instance."""
    global _git_optimizer_instance
    if _git_optimizer_instance is None or _git_optimizer_instance.repository_path != repository_path:
        _git_optimizer_instance = GitCheckpointOptimizer(repository_path)
    return _git_optimizer_instance