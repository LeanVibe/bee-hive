"""
Branch Manager for LeanVibe Agent Hive 2.0

Advanced branch management with automated conflict resolution, intelligent merging,
and comprehensive branch lifecycle management for multi-agent development.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc
from sqlalchemy.orm import selectinload

from ..core.config import get_settings
from ..core.database import get_db_session
from ..models.agent import Agent
from ..models.github_integration import (
    GitHubRepository, AgentWorkTree, BranchOperation, BranchOperationType,
    GitCommit, PullRequest, PullRequestStatus
)
from ..core.github_api_client import GitHubAPIClient
from ..core.work_tree_manager import WorkTreeManager


logger = logging.getLogger(__name__)
settings = get_settings()


class ConflictResolutionStrategy(Enum):
    """Conflict resolution strategies."""
    MANUAL = "manual"
    PREFER_OURS = "prefer_ours"
    PREFER_THEIRS = "prefer_theirs"
    PREFER_LARGER_TIMEOUT = "prefer_larger_timeout"
    PREFER_NEWER_TIMESTAMP = "prefer_newer_timestamp"
    INTELLIGENT_MERGE = "intelligent_merge"
    ABORT_ON_CONFLICT = "abort_on_conflict"


class MergeStrategy(Enum):
    """Git merge strategies."""
    MERGE = "merge"
    REBASE = "rebase"
    SQUASH = "squash"
    FAST_FORWARD = "fast_forward"


class BranchManagerError(Exception):
    """Custom exception for branch management operations."""
    pass


class ConflictResolver:
    """
    Intelligent conflict resolution with multiple strategies.
    
    Provides automated conflict resolution for common scenarios
    while escalating complex conflicts to human review.
    """
    
    def __init__(self):
        self.resolution_patterns = {
            "timeout_conflict": r"TIMEOUT\s*=\s*(\d+)",
            "version_conflict": r"version\s*=\s*[\"']([^\"']+)[\"']",
            "import_conflict": r"^(import|from)\s+.*",
            "config_conflict": r"^(export|set)\s+\w+\s*=",
        }
        
    async def analyze_conflict(self, file_path: Path, conflict_content: str) -> Dict[str, Any]:
        """Analyze conflict to determine resolution strategy."""
        
        analysis = {
            "file_path": str(file_path),
            "conflict_type": "unknown",
            "complexity": "medium",
            "auto_resolvable": False,
            "suggested_strategy": ConflictResolutionStrategy.MANUAL,
            "confidence": 0.0,
            "sections": []
        }
        
        # Split conflict sections
        sections = self._parse_conflict_sections(conflict_content)
        analysis["sections"] = sections
        
        # Determine conflict type and complexity
        if len(sections) == 1:
            analysis["complexity"] = "low"
            analysis["auto_resolvable"] = True
            
        # Analyze specific conflict patterns
        for section in sections:
            conflict_type = self._identify_conflict_pattern(section)
            if conflict_type:
                analysis["conflict_type"] = conflict_type
                analysis["suggested_strategy"] = self._get_strategy_for_type(conflict_type)
                analysis["confidence"] = 0.8
                analysis["auto_resolvable"] = True
                break
                
        return analysis
        
    def _parse_conflict_sections(self, content: str) -> List[Dict[str, Any]]:
        """Parse conflict markers into structured sections."""
        sections = []
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            if line.startswith('<<<<<<<'):
                current_section = {
                    "start_marker": line,
                    "ours": [],
                    "theirs": [],
                    "current_side": "ours"
                }
            elif line.startswith('=======') and current_section:
                current_section["current_side"] = "theirs"
            elif line.startswith('>>>>>>>') and current_section:
                current_section["end_marker"] = line
                sections.append(current_section)
                current_section = None
            elif current_section:
                current_section[current_section["current_side"]].append(line)
                
        return sections
        
    def _identify_conflict_pattern(self, section: Dict[str, Any]) -> Optional[str]:
        """Identify conflict pattern for automated resolution."""
        
        ours_text = '\n'.join(section["ours"])
        theirs_text = '\n'.join(section["theirs"])
        
        # Check for timeout conflicts
        if "TIMEOUT" in ours_text.upper() and "TIMEOUT" in theirs_text.upper():
            return "timeout_conflict"
            
        # Check for version conflicts
        if "version" in ours_text.lower() and "version" in theirs_text.lower():
            return "version_conflict"
            
        # Check for import conflicts
        if any(line.strip().startswith(('import', 'from')) for line in section["ours"]):
            return "import_conflict"
            
        # Check for simple content conflicts
        if len(section["ours"]) == 1 and len(section["theirs"]) == 1:
            return "simple_line_conflict"
            
        return None
        
    def _get_strategy_for_type(self, conflict_type: str) -> ConflictResolutionStrategy:
        """Get recommended resolution strategy for conflict type."""
        strategy_map = {
            "timeout_conflict": ConflictResolutionStrategy.PREFER_LARGER_TIMEOUT,
            "version_conflict": ConflictResolutionStrategy.PREFER_NEWER_TIMESTAMP,
            "import_conflict": ConflictResolutionStrategy.INTELLIGENT_MERGE,
            "simple_line_conflict": ConflictResolutionStrategy.PREFER_THEIRS,
        }
        return strategy_map.get(conflict_type, ConflictResolutionStrategy.MANUAL)
        
    async def resolve_conflict(
        self,
        file_path: Path,
        strategy: ConflictResolutionStrategy,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Resolve conflict using specified strategy."""
        
        result = {
            "success": False,
            "strategy_used": strategy.value,
            "resolution_applied": False,
            "backup_created": False,
            "error": None
        }
        
        try:
            # Read conflict file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Create backup
            backup_path = file_path.with_suffix(f"{file_path.suffix}.conflict_backup")
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            result["backup_created"] = True
            
            # Apply resolution strategy
            resolved_content = await self._apply_resolution_strategy(
                content, strategy, context or {}
            )
            
            if resolved_content != content:
                # Write resolved content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(resolved_content)
                result["resolution_applied"] = True
                result["success"] = True
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Failed to resolve conflict in {file_path}: {e}")
            
        return result
        
    async def _apply_resolution_strategy(
        self,
        content: str,
        strategy: ConflictResolutionStrategy,
        context: Dict[str, Any]
    ) -> str:
        """Apply specific resolution strategy to content."""
        
        if strategy == ConflictResolutionStrategy.PREFER_OURS:
            return self._resolve_prefer_ours(content)
        elif strategy == ConflictResolutionStrategy.PREFER_THEIRS:
            return self._resolve_prefer_theirs(content)
        elif strategy == ConflictResolutionStrategy.PREFER_LARGER_TIMEOUT:
            return self._resolve_prefer_larger_timeout(content)
        elif strategy == ConflictResolutionStrategy.PREFER_NEWER_TIMESTAMP:
            return self._resolve_prefer_newer_timestamp(content, context)
        elif strategy == ConflictResolutionStrategy.INTELLIGENT_MERGE:
            return self._resolve_intelligent_merge(content)
        else:
            return content  # No automatic resolution
            
    def _resolve_prefer_ours(self, content: str) -> str:
        """Resolve by keeping 'ours' version."""
        result_lines = []
        in_conflict = False
        skip_until_end = False
        
        for line in content.split('\n'):
            if line.startswith('<<<<<<<'):
                in_conflict = True
                skip_until_end = False
            elif line.startswith('=======') and in_conflict:
                skip_until_end = True
            elif line.startswith('>>>>>>>') and in_conflict:
                in_conflict = False
                skip_until_end = False
            elif not skip_until_end and not line.startswith(('<<<<<<<', '=======')):
                result_lines.append(line)
                
        return '\n'.join(result_lines)
        
    def _resolve_prefer_theirs(self, content: str) -> str:
        """Resolve by keeping 'theirs' version."""
        result_lines = []
        in_conflict = False
        in_theirs_section = False
        
        for line in content.split('\n'):
            if line.startswith('<<<<<<<'):
                in_conflict = True
                in_theirs_section = False
            elif line.startswith('=======') and in_conflict:
                in_theirs_section = True
            elif line.startswith('>>>>>>>') and in_conflict:
                in_conflict = False
                in_theirs_section = False
            elif in_conflict and in_theirs_section:
                result_lines.append(line)
            elif not in_conflict:
                result_lines.append(line)
                
        return '\n'.join(result_lines)
        
    def _resolve_prefer_larger_timeout(self, content: str) -> str:
        """Resolve timeout conflicts by choosing larger value."""
        import re
        
        sections = self._parse_conflict_sections(content)
        resolved_content = content
        
        for section in sections:
            ours_text = '\n'.join(section["ours"])
            theirs_text = '\n'.join(section["theirs"])
            
            # Extract timeout values
            ours_match = re.search(self.resolution_patterns["timeout_conflict"], ours_text)
            theirs_match = re.search(self.resolution_patterns["timeout_conflict"], theirs_text)
            
            if ours_match and theirs_match:
                ours_timeout = int(ours_match.group(1))
                theirs_timeout = int(theirs_match.group(1))
                
                chosen_text = theirs_text if theirs_timeout > ours_timeout else ours_text
                
                # Replace conflict section with chosen text
                conflict_block = f"<<<<<<< {section['start_marker'][7:]}\n{ours_text}\n=======\n{theirs_text}\n{section['end_marker']}"
                resolved_content = resolved_content.replace(conflict_block, chosen_text)
                
        return resolved_content
        
    def _resolve_prefer_newer_timestamp(self, content: str, context: Dict[str, Any]) -> str:
        """Resolve by preferring content from more recent commit."""
        # This would require commit timestamp context
        # For now, default to prefer_theirs (assuming theirs is newer)
        return self._resolve_prefer_theirs(content)
        
    def _resolve_intelligent_merge(self, content: str) -> str:
        """Attempt intelligent merge for compatible changes."""
        sections = self._parse_conflict_sections(content)
        resolved_content = content
        
        for section in sections:
            ours_lines = section["ours"]
            theirs_lines = section["theirs"]
            
            # For imports, try to merge both sets
            if self._is_import_section(ours_lines) and self._is_import_section(theirs_lines):
                merged_imports = self._merge_imports(ours_lines, theirs_lines)
                conflict_block = f"<<<<<<< {section['start_marker'][7:]}\n" + \
                               '\n'.join(ours_lines) + "\n=======\n" + \
                               '\n'.join(theirs_lines) + f"\n{section['end_marker']}"
                resolved_content = resolved_content.replace(conflict_block, '\n'.join(merged_imports))
                
        return resolved_content
        
    def _is_import_section(self, lines: List[str]) -> bool:
        """Check if section contains import statements."""
        return any(line.strip().startswith(('import', 'from')) for line in lines)
        
    def _merge_imports(self, ours_lines: List[str], theirs_lines: List[str]) -> List[str]:
        """Merge import statements intelligently."""
        all_imports = set()
        
        for line in ours_lines + theirs_lines:
            clean_line = line.strip()
            if clean_line and not clean_line.startswith('#'):
                all_imports.add(clean_line)
                
        return sorted(all_imports)


class BranchManager:
    """
    Comprehensive branch management for multi-agent development.
    
    Handles branch creation, synchronization, merging, and conflict resolution
    with intelligent automation and comprehensive audit trails.
    """
    
    def __init__(self, github_client: GitHubAPIClient = None, work_tree_manager: WorkTreeManager = None):
        self.github_client = github_client or GitHubAPIClient()
        self.work_tree_manager = work_tree_manager or WorkTreeManager(self.github_client)
        self.conflict_resolver = ConflictResolver()
        
        self.default_strategies = {
            "merge_strategy": MergeStrategy.MERGE,
            "conflict_strategy": ConflictResolutionStrategy.INTELLIGENT_MERGE,
            "auto_resolve_threshold": 0.8,  # Confidence threshold for auto-resolution
            "max_conflict_files": 5,  # Max files with conflicts to auto-resolve
        }
        
    async def create_agent_branch(
        self,
        agent_id: str,
        repository: GitHubRepository,
        branch_name: str = None,
        base_branch: str = None,
        feature_description: str = None
    ) -> Dict[str, Any]:
        """Create new branch for agent development."""
        
        operation = BranchOperation(
            repository_id=repository.id,
            agent_id=uuid.UUID(agent_id),
            operation_type=BranchOperationType.CREATE,
            target_branch=branch_name,
            source_branch=base_branch or repository.default_branch,
            status="pending"
        )
        
        try:
            # Generate branch name if not provided
            if not branch_name:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                feature_suffix = feature_description.replace(' ', '-').lower()[:20] if feature_description else "feature"
                branch_name = f"agent/{agent_id[:8]}/{feature_suffix}_{timestamp}"
                operation.target_branch = branch_name
                
            operation.start_operation()
            
            # Get or create work tree
            work_tree = await self.work_tree_manager.get_agent_work_tree(agent_id, str(repository.id))
            if not work_tree:
                work_tree = await self.work_tree_manager.create_agent_work_tree(
                    agent_id, repository, branch_name, base_branch or repository.default_branch
                )
            else:
                # Create new branch in existing work tree
                work_tree_path = Path(work_tree.work_tree_path)
                await self._create_branch_in_work_tree(work_tree_path, branch_name, base_branch or repository.default_branch)
                work_tree.branch_name = branch_name
                work_tree.base_branch = base_branch or repository.default_branch
                
            operation.work_tree_id = work_tree.id
            
            # Create branch on GitHub
            repo_parts = repository.repository_full_name.split('/')
            base_branch_info = await self.github_client.get_branch(repo_parts[0], repo_parts[1], base_branch or repository.default_branch)
            
            await self.github_client.create_branch(
                repo_parts[0],
                repo_parts[1], 
                branch_name,
                base_branch_info["commit"]["sha"]
            )
            
            operation.complete_operation(success=True)
            
            # Save to database
            async with get_db_session() as session:
                session.add(operation)
                await session.merge(work_tree)
                await session.commit()
                
            logger.info(f"Created branch {branch_name} for agent {agent_id}")
            
            return {
                "success": True,
                "branch_name": branch_name,
                "work_tree_id": str(work_tree.id),
                "operation_id": str(operation.id),
                "base_branch": base_branch or repository.default_branch
            }
            
        except Exception as e:
            operation.complete_operation(success=False, error_message=str(e))
            async with get_db_session() as session:
                session.add(operation)
                await session.commit()
            raise BranchManagerError(f"Failed to create branch: {str(e)}")
            
    async def _create_branch_in_work_tree(self, work_tree_path: Path, branch_name: str, base_branch: str) -> None:
        """Create new branch in existing work tree."""
        git_manager = self.work_tree_manager.git_manager
        
        # Fetch latest changes
        await git_manager.execute_git_command(["git", "fetch", "origin"], work_tree_path)
        
        # Create and checkout new branch
        await git_manager.execute_git_command(
            ["git", "checkout", "-b", branch_name, f"origin/{base_branch}"],
            work_tree_path
        )
        
        # Set upstream tracking
        await git_manager.execute_git_command(
            ["git", "push", "--set-upstream", "origin", branch_name],
            work_tree_path
        )
        
    async def sync_branch_with_main(
        self,
        work_tree: AgentWorkTree,
        strategy: MergeStrategy = MergeStrategy.MERGE,
        conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.INTELLIGENT_MERGE
    ) -> Dict[str, Any]:
        """Sync agent branch with main branch changes."""
        
        operation = BranchOperation(
            repository_id=work_tree.repository_id,
            agent_id=work_tree.agent_id,
            work_tree_id=work_tree.id,
            operation_type=BranchOperationType.SYNC,
            source_branch=work_tree.base_branch,
            target_branch=work_tree.branch_name,
            status="pending"
        )
        
        try:
            operation.start_operation()
            work_tree_path = Path(work_tree.work_tree_path)
            
            # Perform sync based on strategy
            if strategy == MergeStrategy.MERGE:
                sync_result = await self._merge_sync(work_tree_path, work_tree.base_branch, conflict_strategy, operation)
            elif strategy == MergeStrategy.REBASE:
                sync_result = await self._rebase_sync(work_tree_path, work_tree.base_branch, conflict_strategy, operation)
            else:
                raise BranchManagerError(f"Unsupported sync strategy: {strategy}")
                
            operation.operation_result = sync_result
            operation.complete_operation(success=sync_result["success"])
            
            # Update work tree activity
            work_tree.update_activity()
            
            # Save to database
            async with get_db_session() as session:
                session.add(operation)
                await session.merge(work_tree)
                await session.commit()
                
            return sync_result
            
        except Exception as e:
            operation.complete_operation(success=False, error_message=str(e))
            async with get_db_session() as session:
                session.add(operation)
                await session.commit()
            raise BranchManagerError(f"Branch sync failed: {str(e)}")
            
    async def _merge_sync(
        self,
        work_tree_path: Path,
        base_branch: str,
        conflict_strategy: ConflictResolutionStrategy,
        operation: BranchOperation
    ) -> Dict[str, Any]:
        """Perform merge-based sync with conflict resolution."""
        
        git_manager = self.work_tree_manager.git_manager
        result = {
            "success": False,
            "strategy": "merge",
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "files_changed": 0,
            "auto_resolved": False,
            "manual_intervention_required": False
        }
        
        # Fetch latest changes
        await git_manager.execute_git_command(["git", "fetch", "origin"], work_tree_path)
        
        # Attempt merge
        merge_code, merge_output, merge_error = await git_manager.execute_git_command(
            ["git", "merge", f"origin/{base_branch}"],
            work_tree_path
        )
        
        if merge_code == 0:
            # Successful merge
            result["success"] = True
            result["auto_resolved"] = True
            
            # Count changed files
            diff_code, diff_output, _ = await git_manager.execute_git_command(
                ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
                work_tree_path
            )
            if diff_code == 0:
                result["files_changed"] = len([f for f in diff_output.strip().split('\n') if f])
                
        else:
            # Handle conflicts
            conflicts = await git_manager.detect_conflicts(work_tree_path)
            result["conflicts_detected"] = len(conflicts)
            operation.conflicts_detected = len(conflicts)
            operation.conflict_details = conflicts
            
            if len(conflicts) <= self.default_strategies["max_conflict_files"]:
                # Attempt automatic resolution
                resolution_result = await self._auto_resolve_conflicts(
                    work_tree_path, conflicts, conflict_strategy
                )
                
                result["conflicts_resolved"] = resolution_result["resolved_count"]
                operation.conflicts_resolved = resolution_result["resolved_count"]
                operation.resolution_strategy = conflict_strategy.value
                
                if resolution_result["all_resolved"]:
                    # Complete the merge
                    commit_code, _, _ = await git_manager.execute_git_command(
                        ["git", "commit", "-m", f"Auto-merge with {base_branch} - conflicts resolved automatically"],
                        work_tree_path
                    )
                    
                    if commit_code == 0:
                        result["success"] = True
                        result["auto_resolved"] = True
                    else:
                        result["manual_intervention_required"] = True
                else:
                    # Abort merge and require manual intervention
                    await git_manager.execute_git_command(["git", "merge", "--abort"], work_tree_path)
                    result["manual_intervention_required"] = True
            else:
                # Too many conflicts - abort and require manual intervention
                await git_manager.execute_git_command(["git", "merge", "--abort"], work_tree_path)
                result["manual_intervention_required"] = True
                
        return result
        
    async def _rebase_sync(
        self,
        work_tree_path: Path,
        base_branch: str,
        conflict_strategy: ConflictResolutionStrategy,
        operation: BranchOperation
    ) -> Dict[str, Any]:
        """Perform rebase-based sync with conflict resolution."""
        
        git_manager = self.work_tree_manager.git_manager
        result = {
            "success": False,
            "strategy": "rebase",
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "commits_rebased": 0,
            "auto_resolved": False,
            "manual_intervention_required": False
        }
        
        # Fetch latest changes
        await git_manager.execute_git_command(["git", "fetch", "origin"], work_tree_path)
        
        # Count commits to be rebased
        count_code, count_output, _ = await git_manager.execute_git_command(
            ["git", "rev-list", "--count", f"origin/{base_branch}..HEAD"],
            work_tree_path
        )
        if count_code == 0:
            result["commits_rebased"] = int(count_output.strip() or 0)
            
        # Attempt rebase
        rebase_code, rebase_output, rebase_error = await git_manager.execute_git_command(
            ["git", "rebase", f"origin/{base_branch}"],
            work_tree_path
        )
        
        if rebase_code == 0:
            result["success"] = True
            result["auto_resolved"] = True
        else:
            # Handle rebase conflicts
            conflicts = await git_manager.detect_conflicts(work_tree_path)
            result["conflicts_detected"] = len(conflicts)
            
            if len(conflicts) <= self.default_strategies["max_conflict_files"]:
                # Attempt automatic resolution
                resolution_result = await self._auto_resolve_conflicts(
                    work_tree_path, conflicts, conflict_strategy
                )
                
                result["conflicts_resolved"] = resolution_result["resolved_count"]
                
                if resolution_result["all_resolved"]:
                    # Continue rebase
                    continue_code, _, _ = await git_manager.execute_git_command(
                        ["git", "rebase", "--continue"],
                        work_tree_path
                    )
                    
                    if continue_code == 0:
                        result["success"] = True
                        result["auto_resolved"] = True
                    else:
                        result["manual_intervention_required"] = True
                else:
                    # Abort rebase and require manual intervention
                    await git_manager.execute_git_command(["git", "rebase", "--abort"], work_tree_path)
                    result["manual_intervention_required"] = True
            else:
                # Too many conflicts - abort rebase
                await git_manager.execute_git_command(["git", "rebase", "--abort"], work_tree_path)
                result["manual_intervention_required"] = True
                
        return result
        
    async def _auto_resolve_conflicts(
        self,
        work_tree_path: Path,
        conflicts: List[Dict[str, Any]],
        strategy: ConflictResolutionStrategy
    ) -> Dict[str, Any]:
        """Attempt automatic conflict resolution."""
        
        resolution_result = {
            "resolved_count": 0,
            "failed_count": 0,
            "all_resolved": False,
            "resolutions": []
        }
        
        for conflict in conflicts:
            file_path = work_tree_path / conflict["file"]
            
            try:
                # Read conflict file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Analyze conflict
                analysis = await self.conflict_resolver.analyze_conflict(file_path, content)
                
                # Apply resolution if confidence is high enough
                if (analysis["auto_resolvable"] and 
                    analysis["confidence"] >= self.default_strategies["auto_resolve_threshold"]):
                    
                    resolution = await self.conflict_resolver.resolve_conflict(
                        file_path, 
                        analysis["suggested_strategy"]
                    )
                    
                    if resolution["success"]:
                        resolution_result["resolved_count"] += 1
                        
                        # Stage resolved file
                        git_manager = self.work_tree_manager.git_manager
                        await git_manager.execute_git_command(
                            ["git", "add", conflict["file"]],
                            work_tree_path
                        )
                    else:
                        resolution_result["failed_count"] += 1
                        
                    resolution_result["resolutions"].append({
                        "file": conflict["file"],
                        "analysis": analysis,
                        "resolution": resolution
                    })
                else:
                    resolution_result["failed_count"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to resolve conflict in {file_path}: {e}")
                resolution_result["failed_count"] += 1
                
        resolution_result["all_resolved"] = (
            resolution_result["resolved_count"] == len(conflicts) and
            resolution_result["failed_count"] == 0
        )
        
        return resolution_result
        
    async def delete_agent_branch(
        self,
        agent_id: str,
        repository: GitHubRepository,
        branch_name: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """Delete agent branch after successful merge."""
        
        operation = BranchOperation(
            repository_id=repository.id,
            agent_id=uuid.UUID(agent_id),
            operation_type=BranchOperationType.DELETE,
            target_branch=branch_name,
            status="pending"
        )
        
        try:
            operation.start_operation()
            
            # Check if branch is safe to delete
            if not force:
                safety_check = await self._check_branch_deletion_safety(repository, branch_name)
                if not safety_check["safe"]:
                    raise BranchManagerError(f"Branch deletion not safe: {safety_check['reason']}")
                    
            # Delete branch on GitHub
            repo_parts = repository.repository_full_name.split('/')
            deleted = await self.github_client.delete_branch(repo_parts[0], repo_parts[1], branch_name)
            
            if not deleted and not force:
                raise BranchManagerError("Failed to delete branch on GitHub")
                
            operation.complete_operation(success=True)
            
            # Clean up work tree if it exists
            work_tree = await self.work_tree_manager.get_agent_work_tree(agent_id, str(repository.id))
            if work_tree and work_tree.branch_name == branch_name:
                await self.work_tree_manager.cleanup_work_tree(work_tree)
                
            # Save to database
            async with get_db_session() as session:
                session.add(operation)
                await session.commit()
                
            logger.info(f"Deleted branch {branch_name} for agent {agent_id}")
            
            return {
                "success": True,
                "branch_name": branch_name,
                "deleted_from_github": deleted,
                "work_tree_cleaned": work_tree is not None
            }
            
        except Exception as e:
            operation.complete_operation(success=False, error_message=str(e))
            async with get_db_session() as session:
                session.add(operation)
                await session.commit()
            raise BranchManagerError(f"Failed to delete branch: {str(e)}")
            
    async def _check_branch_deletion_safety(self, repository: GitHubRepository, branch_name: str) -> Dict[str, Any]:
        """Check if branch is safe to delete."""
        
        safety_check = {
            "safe": False,
            "reason": None,
            "checks": {}
        }
        
        try:
            repo_parts = repository.repository_full_name.split('/')
            
            # Check if branch exists
            try:
                branch_info = await self.github_client.get_branch(repo_parts[0], repo_parts[1], branch_name)
                safety_check["checks"]["branch_exists"] = True
            except:
                safety_check["checks"]["branch_exists"] = False
                safety_check["reason"] = "Branch does not exist"
                return safety_check
                
            # Check for open PRs
            open_prs = await self.github_client.list_pull_requests(
                repo_parts[0], repo_parts[1], state="open", head=f"{repo_parts[0]}:{branch_name}"
            )
            safety_check["checks"]["has_open_prs"] = len(open_prs) > 0
            
            if len(open_prs) > 0:
                safety_check["reason"] = f"Branch has {len(open_prs)} open pull request(s)"
                return safety_check
                
            # Check if branch is protected
            if branch_name in [repository.default_branch, "main", "master", "develop"]:
                safety_check["checks"]["is_protected"] = True
                safety_check["reason"] = "Cannot delete protected branch"
                return safety_check
            else:
                safety_check["checks"]["is_protected"] = False
                
            # Check if branch is merged
            # This would require more complex logic to check merge status
            safety_check["checks"]["is_merged"] = True  # Assume merged for now
            
            safety_check["safe"] = True
            
        except Exception as e:
            safety_check["reason"] = f"Safety check failed: {str(e)}"
            
        return safety_check
        
    async def list_agent_branches(self, agent_id: str, repository_id: str = None) -> List[Dict[str, Any]]:
        """List all branches for specific agent."""
        
        async with get_db_session() as session:
            query = select(BranchOperation).where(
                BranchOperation.agent_id == uuid.UUID(agent_id)
            ).options(
                selectinload(BranchOperation.repository),
                selectinload(BranchOperation.work_tree)
            ).order_by(desc(BranchOperation.created_at))
            
            if repository_id:
                query = query.where(BranchOperation.repository_id == uuid.UUID(repository_id))
                
            result = await session.execute(query)
            operations = result.scalars().all()
            
            branches = []
            for operation in operations:
                if operation.operation_type == BranchOperationType.CREATE and operation.is_successful():
                    branch_info = {
                        "branch_name": operation.target_branch,
                        "repository": operation.repository.repository_full_name,
                        "created_at": operation.created_at.isoformat(),
                        "base_branch": operation.source_branch,
                        "work_tree_id": str(operation.work_tree_id) if operation.work_tree_id else None,
                        "status": "active"
                    }
                    
                    # Check if branch still exists
                    try:
                        repo_parts = operation.repository.repository_full_name.split('/')
                        await self.github_client.get_branch(repo_parts[0], repo_parts[1], operation.target_branch)
                        branch_info["exists_on_github"] = True
                    except:
                        branch_info["exists_on_github"] = False
                        branch_info["status"] = "deleted"
                        
                    branches.append(branch_info)
                    
            return branches
            
    async def get_branch_operations_history(
        self,
        agent_id: str = None,
        repository_id: str = None,
        operation_type: BranchOperationType = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get history of branch operations with filtering."""
        
        async with get_db_session() as session:
            query = select(BranchOperation).options(
                selectinload(BranchOperation.repository),
                selectinload(BranchOperation.agent),
                selectinload(BranchOperation.work_tree)
            ).order_by(desc(BranchOperation.created_at))
            
            if agent_id:
                query = query.where(BranchOperation.agent_id == uuid.UUID(agent_id))
            if repository_id:
                query = query.where(BranchOperation.repository_id == uuid.UUID(repository_id))
            if operation_type:
                query = query.where(BranchOperation.operation_type == operation_type)
                
            query = query.limit(limit)
            
            result = await session.execute(query)
            operations = result.scalars().all()
            
            return [operation.to_dict() for operation in operations]
            
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of branch management system."""
        
        health_status = {
            "healthy": True,
            "github_connectivity": False,
            "active_operations": 0,
            "failed_operations_24h": 0,
            "conflict_resolution_rate": 0.0,
            "average_operation_time": 0.0,
            "errors": []
        }
        
        try:
            # Check GitHub connectivity
            health_status["github_connectivity"] = await self.github_client.health_check()
            
            # Get operation statistics
            async with get_db_session() as session:
                # Count active operations
                active_result = await session.execute(
                    select(BranchOperation).where(
                        BranchOperation.status.in_(["pending", "in_progress"])
                    )
                )
                health_status["active_operations"] = len(active_result.scalars().all())
                
                # Count failed operations in last 24 hours
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                failed_result = await session.execute(
                    select(BranchOperation).where(
                        and_(
                            BranchOperation.status == "failed",
                            BranchOperation.created_at >= cutoff_time
                        )
                    )
                )
                health_status["failed_operations_24h"] = len(failed_result.scalars().all())
                
                # Calculate conflict resolution rate
                conflict_ops_result = await session.execute(
                    select(BranchOperation).where(
                        and_(
                            BranchOperation.conflicts_detected > 0,
                            BranchOperation.created_at >= cutoff_time
                        )
                    )
                )
                conflict_ops = conflict_ops_result.scalars().all()
                
                if conflict_ops:
                    total_conflicts = sum(op.conflicts_detected for op in conflict_ops)
                    total_resolved = sum(op.conflicts_resolved for op in conflict_ops)
                    health_status["conflict_resolution_rate"] = total_resolved / total_conflicts if total_conflicts > 0 else 0.0
                    
                # Calculate average operation time
                completed_ops_result = await session.execute(
                    select(BranchOperation).where(
                        and_(
                            BranchOperation.status == "completed",
                            BranchOperation.started_at.isnot(None),
                            BranchOperation.completed_at.isnot(None),
                            BranchOperation.created_at >= cutoff_time
                        )
                    )
                )
                completed_ops = completed_ops_result.scalars().all()
                
                if completed_ops:
                    total_time = sum(
                        (op.completed_at - op.started_at).total_seconds()
                        for op in completed_ops
                    )
                    health_status["average_operation_time"] = total_time / len(completed_ops)
                    
        except Exception as e:
            health_status["healthy"] = False
            health_status["errors"].append(f"Health check failed: {str(e)}")
            
        # Overall health determination
        health_status["healthy"] = (
            health_status["github_connectivity"] and
            health_status["failed_operations_24h"] < 10 and  # Less than 10 failures per day
            health_status["conflict_resolution_rate"] > 0.5 and  # At least 50% conflict resolution
            len(health_status["errors"]) == 0
        )
        
        return health_status