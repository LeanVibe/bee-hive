"""
Advanced Repository Management for LeanVibe Agent Hive 2.0

Intelligent merge conflict resolution, automated dependency analysis,
branch management with cleanup automation, and advanced repository operations.
"""

import asyncio
import json
import logging
import re
import subprocess
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from enum import Enum
from dataclasses import dataclass
import hashlib

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func
from sqlalchemy.orm import selectinload
import structlog
import git
from git import Repo
try:
    import semver
except ImportError:
    # Fallback for semver functionality
    semver = None

from ..core.config import get_settings
from ..core.database import get_db_session
from ..models.github_integration import (
    GitHubRepository, AgentWorkTree, BranchOperation, 
    BranchOperationType, PullRequest
)
from ..core.github_api_client import GitHubAPIClient
from ..core.redis import get_redis

logger = structlog.get_logger()
settings = get_settings()


class ConflictResolutionStrategy(Enum):
    """Merge conflict resolution strategies."""
    AUTOMATIC = "automatic"
    PREFER_OURS = "prefer_ours"
    PREFER_THEIRS = "prefer_theirs"
    INTELLIGENT_MERGE = "intelligent_merge"
    MANUAL_REQUIRED = "manual_required"
    SEMANTIC_MERGE = "semantic_merge"


class DependencyUpdateType(Enum):
    """Dependency update types."""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    SECURITY = "security"
    BREAKING = "breaking"


class RepositoryHealthStatus(Enum):
    """Repository health status."""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ConflictFile:
    """Represents a conflicted file."""
    file_path: str
    conflict_markers: List[Dict[str, Any]]
    conflict_type: str
    resolution_confidence: float
    suggested_resolution: Optional[str] = None
    

@dataclass
class MergeConflict:
    """Represents a merge conflict."""
    conflict_id: str
    source_branch: str
    target_branch: str
    conflicted_files: List[ConflictFile]
    conflict_summary: str
    resolution_strategy: ConflictResolutionStrategy
    auto_resolvable: bool
    

@dataclass
class DependencyInfo:
    """Dependency information."""
    name: str
    current_version: str
    latest_version: str
    update_type: DependencyUpdateType
    security_vulnerabilities: List[Dict[str, Any]]
    breaking_changes: List[str]
    update_priority: int  # 1-10, 10 being highest


@dataclass
class BranchAnalysis:
    """Branch analysis results."""
    branch_name: str
    commit_count: int
    commits_behind_main: int
    last_commit_date: datetime
    author_count: int
    file_change_count: int
    lines_added: int
    lines_deleted: int
    merge_conflicts: Optional[MergeConflict] = None


class AdvancedRepositoryManagementError(Exception):
    """Custom exception for advanced repository management operations."""
    pass


class IntelligentConflictResolver:
    """
    Intelligent merge conflict resolution system.
    
    Uses heuristics, semantic analysis, and pattern recognition
    to automatically resolve merge conflicts with high confidence.
    """
    
    def __init__(self):
        self.resolution_patterns = {
            "import_statements": {
                "pattern": r"^(import|from)\s+",
                "strategy": "merge_both",
                "confidence": 0.9
            },
            "version_conflicts": {
                "pattern": r"version\s*[=:]\s*[\"'][\d\.]+[\"']",
                "strategy": "prefer_higher_version", 
                "confidence": 0.8
            },
            "dependency_conflicts": {
                "pattern": r"(requirements\.txt|package\.json|pom\.xml)",
                "strategy": "intelligent_merge",
                "confidence": 0.7
            },
            "documentation": {
                "pattern": r"\.(md|txt|rst|doc)$",
                "strategy": "merge_both",
                "confidence": 0.8
            },
            "config_files": {
                "pattern": r"\.(yml|yaml|json|toml|ini|cfg)$",
                "strategy": "semantic_merge",
                "confidence": 0.6
            }
        }
        
        self.conflict_markers = [
            "<<<<<<< HEAD",
            "=======",
            ">>>>>>> "
        ]
    
    async def analyze_conflicts(
        self,
        repo_path: str,
        source_branch: str,
        target_branch: str
    ) -> MergeConflict:
        """Analyze merge conflicts between branches."""
        
        try:
            repo = Repo(repo_path)
            
            # Attempt merge to identify conflicts
            conflict_id = str(uuid.uuid4())
            
            # Create temporary branch for conflict analysis
            temp_branch = f"conflict-analysis-{conflict_id[:8]}"
            
            # Switch to target branch and create temp branch
            repo.git.checkout(target_branch)
            temp_branch_ref = repo.create_head(temp_branch)
            repo.git.checkout(temp_branch)
            
            try:
                # Attempt merge
                repo.git.merge(source_branch, "--no-commit", "--no-ff")
                
                # No conflicts - clean merge
                conflicted_files = []
                auto_resolvable = True
                
            except git.exc.GitCommandError as e:
                # Merge conflicts occurred
                conflicted_files = await self._analyze_conflicted_files(repo_path)
                auto_resolvable = all(
                    self._can_auto_resolve(cf) for cf in conflicted_files
                )
            
            # Cleanup temp branch
            repo.git.checkout(target_branch)
            repo.delete_head(temp_branch_ref, force=True)
            
            # Generate conflict summary
            conflict_summary = self._generate_conflict_summary(conflicted_files)
            
            # Determine resolution strategy
            resolution_strategy = self._determine_resolution_strategy(conflicted_files)
            
            return MergeConflict(
                conflict_id=conflict_id,
                source_branch=source_branch,
                target_branch=target_branch,
                conflicted_files=conflicted_files,
                conflict_summary=conflict_summary,
                resolution_strategy=resolution_strategy,
                auto_resolvable=auto_resolvable
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze conflicts: {e}")
            raise AdvancedRepositoryManagementError(f"Conflict analysis failed: {str(e)}")
    
    async def _analyze_conflicted_files(self, repo_path: str) -> List[ConflictFile]:
        """Analyze individual conflicted files."""
        
        repo = Repo(repo_path)
        conflicted_files = []
        
        # Get files with conflicts
        unmerged_files = [item[0] for item in repo.index.unmerged_blobs().keys()]
        
        for file_path in unmerged_files:
            full_path = Path(repo_path) / file_path
            
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Parse conflict markers
                conflict_markers = self._parse_conflict_markers(content)
                
                # Determine conflict type
                conflict_type = self._classify_conflict_type(file_path, content)
                
                # Calculate resolution confidence
                resolution_confidence = self._calculate_resolution_confidence(
                    file_path, conflict_markers, conflict_type
                )
                
                # Generate suggested resolution
                suggested_resolution = await self._generate_suggested_resolution(
                    file_path, content, conflict_markers, conflict_type
                )
                
                conflicted_file = ConflictFile(
                    file_path=file_path,
                    conflict_markers=conflict_markers,
                    conflict_type=conflict_type,
                    resolution_confidence=resolution_confidence,
                    suggested_resolution=suggested_resolution
                )
                
                conflicted_files.append(conflicted_file)
        
        return conflicted_files
    
    def _parse_conflict_markers(self, content: str) -> List[Dict[str, Any]]:
        """Parse conflict markers in file content."""
        
        markers = []
        lines = content.split('\n')
        
        current_conflict = None
        for i, line in enumerate(lines):
            if line.startswith('<<<<<<< '):
                current_conflict = {
                    "start_line": i + 1,
                    "our_branch": line[8:].strip(),
                    "our_content": [],
                    "their_content": [],
                    "separator_line": None,
                    "end_line": None,
                    "their_branch": None
                }
            elif line.startswith('=======') and current_conflict:
                current_conflict["separator_line"] = i + 1
            elif line.startswith('>>>>>>> ') and current_conflict:
                current_conflict["end_line"] = i + 1
                current_conflict["their_branch"] = line[8:].strip()
                markers.append(current_conflict)
                current_conflict = None
            elif current_conflict:
                if current_conflict["separator_line"] is None:
                    current_conflict["our_content"].append(line)
                else:
                    current_conflict["their_content"].append(line)
        
        return markers
    
    def _classify_conflict_type(self, file_path: str, content: str) -> str:
        """Classify the type of conflict."""
        
        file_path_lower = file_path.lower()
        
        if any(pattern in file_path_lower for pattern in ['.py', '.js', '.java', '.cpp', '.c']):
            if 'import' in content or 'from' in content:
                return "import_conflict"
            elif 'def ' in content or 'function ' in content or 'class ' in content:
                return "code_conflict"
            else:
                return "logic_conflict"
        
        elif any(pattern in file_path_lower for pattern in ['requirements', 'package.json', 'pom.xml']):
            return "dependency_conflict"
        
        elif any(pattern in file_path_lower for pattern in ['.md', '.txt', '.rst']):
            return "documentation_conflict"
        
        elif any(pattern in file_path_lower for pattern in ['.yml', '.yaml', '.json', '.toml']):
            return "configuration_conflict"
        
        elif 'version' in content.lower():
            return "version_conflict"
        
        else:
            return "generic_conflict"
    
    def _calculate_resolution_confidence(
        self,
        file_path: str,
        conflict_markers: List[Dict[str, Any]],
        conflict_type: str
    ) -> float:
        """Calculate confidence in automatic resolution."""
        
        base_confidence = {
            "import_conflict": 0.9,
            "dependency_conflict": 0.7,
            "documentation_conflict": 0.8,
            "configuration_conflict": 0.6,
            "version_conflict": 0.8,
            "code_conflict": 0.4,
            "logic_conflict": 0.3,
            "generic_conflict": 0.2
        }.get(conflict_type, 0.3)
        
        # Adjust based on conflict complexity
        avg_conflict_size = sum(
            len(marker["our_content"]) + len(marker["their_content"])
            for marker in conflict_markers
        ) / len(conflict_markers) if conflict_markers else 0
        
        # Reduce confidence for large conflicts
        size_penalty = min(0.3, avg_conflict_size / 20.0)
        
        # Reduce confidence for multiple conflicts
        multiple_conflicts_penalty = min(0.2, len(conflict_markers) / 10.0)
        
        final_confidence = max(0.1, base_confidence - size_penalty - multiple_conflicts_penalty)
        
        return final_confidence
    
    async def _generate_suggested_resolution(
        self,
        file_path: str,
        content: str,
        conflict_markers: List[Dict[str, Any]],
        conflict_type: str
    ) -> Optional[str]:
        """Generate suggested resolution for conflict."""
        
        try:
            if conflict_type == "import_conflict":
                return self._resolve_import_conflict(content, conflict_markers)
            elif conflict_type == "dependency_conflict":
                return await self._resolve_dependency_conflict(file_path, content, conflict_markers)
            elif conflict_type == "documentation_conflict":
                return self._resolve_documentation_conflict(content, conflict_markers)
            elif conflict_type == "version_conflict":
                return self._resolve_version_conflict(content, conflict_markers)
            elif conflict_type == "configuration_conflict":
                return await self._resolve_configuration_conflict(file_path, content, conflict_markers)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate resolution for {file_path}: {e}")
            return None
    
    def _resolve_import_conflict(self, content: str, conflict_markers: List[Dict[str, Any]]) -> str:
        """Resolve import statement conflicts by merging both."""
        
        lines = content.split('\n')
        resolved_lines = []
        
        for i, line in enumerate(lines):
            # Skip conflict markers and merge imports
            if any(line.startswith(marker) for marker in self.conflict_markers):
                continue
                
            # Find the conflict this line belongs to
            in_conflict = False
            for marker in conflict_markers:
                if marker["start_line"] <= i + 1 <= marker["end_line"]:
                    in_conflict = True
                    
                    if i + 1 == marker["start_line"] + 1:  # First line after start marker
                        # Merge both import sections
                        our_imports = set(marker["our_content"])
                        their_imports = set(marker["their_content"])
                        all_imports = sorted(our_imports | their_imports)
                        resolved_lines.extend(all_imports)
                    break
            
            if not in_conflict:
                resolved_lines.append(line)
        
        return '\n'.join(resolved_lines)
    
    async def _resolve_dependency_conflict(
        self,
        file_path: str,
        content: str,
        conflict_markers: List[Dict[str, Any]]
    ) -> str:
        """Resolve dependency conflicts by choosing newer versions."""
        
        if 'requirements.txt' in file_path:
            return self._resolve_requirements_conflict(content, conflict_markers)
        elif 'package.json' in file_path:
            return self._resolve_package_json_conflict(content, conflict_markers)
        else:
            # Generic dependency resolution
            return self._merge_with_preference(content, conflict_markers, "higher_version")
    
    def _resolve_requirements_conflict(self, content: str, conflict_markers: List[Dict[str, Any]]) -> str:
        """Resolve Python requirements.txt conflicts."""
        
        lines = content.split('\n')
        resolved_lines = []
        
        for i, line in enumerate(lines):
            if any(line.startswith(marker) for marker in self.conflict_markers):
                continue
                
            in_conflict = False
            for marker in conflict_markers:
                if marker["start_line"] <= i + 1 <= marker["end_line"]:
                    in_conflict = True
                    
                    if i + 1 == marker["start_line"] + 1:
                        # Parse and merge requirements
                        our_reqs = self._parse_requirements(marker["our_content"])
                        their_reqs = self._parse_requirements(marker["their_content"])
                        merged_reqs = self._merge_requirements(our_reqs, their_reqs)
                        resolved_lines.extend([f"{name}{version}" for name, version in merged_reqs.items()])
                    break
            
            if not in_conflict:
                resolved_lines.append(line)
        
        return '\n'.join(resolved_lines)
    
    def _parse_requirements(self, req_lines: List[str]) -> Dict[str, str]:
        """Parse requirements.txt format lines."""
        
        requirements = {}
        for line in req_lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Simple parsing for package==version format
                if '==' in line:
                    name, version = line.split('==', 1)
                    requirements[name.strip()] = f"=={version.strip()}"
                elif '>=' in line:
                    name, version = line.split('>=', 1)
                    requirements[name.strip()] = f">={version.strip()}"
                else:
                    requirements[line] = ""
        
        return requirements
    
    def _merge_requirements(self, our_reqs: Dict[str, str], their_reqs: Dict[str, str]) -> Dict[str, str]:
        """Merge requirement dictionaries, preferring newer versions."""
        
        merged = {}
        all_packages = set(our_reqs.keys()) | set(their_reqs.keys())
        
        for package in all_packages:
            our_version = our_reqs.get(package, "")
            their_version = their_reqs.get(package, "")
            
            if our_version and their_version:
                # Choose higher version
                merged[package] = self._choose_higher_version(our_version, their_version)
            else:
                merged[package] = our_version or their_version
        
        return merged
    
    def _choose_higher_version(self, version1: str, version2: str) -> str:
        """Choose the higher version between two version strings."""
        
        try:
            # Extract version numbers (remove operators like ==, >=)
            v1_clean = re.sub(r'^[><=!]+', '', version1)
            v2_clean = re.sub(r'^[><=!]+', '', version2)
            
            if semver.compare(v1_clean, v2_clean) > 0:
                return version1
            else:
                return version2
        except:
            # Fallback to string comparison if semver fails
            return version2 if version2 > version1 else version1
    
    def _resolve_documentation_conflict(self, content: str, conflict_markers: List[Dict[str, Any]]) -> str:
        """Resolve documentation conflicts by merging content."""
        
        lines = content.split('\n')
        resolved_lines = []
        
        for i, line in enumerate(lines):
            if any(line.startswith(marker) for marker in self.conflict_markers):
                continue
                
            in_conflict = False
            for marker in conflict_markers:
                if marker["start_line"] <= i + 1 <= marker["end_line"]:
                    in_conflict = True
                    
                    if i + 1 == marker["start_line"] + 1:
                        # Merge documentation sections
                        resolved_lines.extend(marker["our_content"])
                        if marker["their_content"]:
                            resolved_lines.append("")  # Add separator
                            resolved_lines.extend(marker["their_content"])
                    break
            
            if not in_conflict:
                resolved_lines.append(line)
        
        return '\n'.join(resolved_lines)
    
    def _resolve_version_conflict(self, content: str, conflict_markers: List[Dict[str, Any]]) -> str:
        """Resolve version conflicts by choosing higher version."""
        
        return self._merge_with_preference(content, conflict_markers, "higher_version")
    
    async def _resolve_configuration_conflict(
        self,
        file_path: str,
        content: str,
        conflict_markers: List[Dict[str, Any]]
    ) -> str:
        """Resolve configuration file conflicts intelligently."""
        
        if file_path.endswith('.json'):
            return self._resolve_json_conflict(content, conflict_markers)
        elif file_path.endswith(('.yml', '.yaml')):
            return self._resolve_yaml_conflict(content, conflict_markers)
        else:
            return self._merge_with_preference(content, conflict_markers, "prefer_ours")
    
    def _resolve_json_conflict(self, content: str, conflict_markers: List[Dict[str, Any]]) -> str:
        """Resolve JSON configuration conflicts by merging objects."""
        
        try:
            lines = content.split('\n')
            resolved_lines = []
            
            for i, line in enumerate(lines):
                if any(line.startswith(marker) for marker in self.conflict_markers):
                    continue
                    
                in_conflict = False
                for marker in conflict_markers:
                    if marker["start_line"] <= i + 1 <= marker["end_line"]:
                        in_conflict = True
                        
                        if i + 1 == marker["start_line"] + 1:
                            # Try to parse and merge JSON objects
                            our_json_str = '\n'.join(marker["our_content"])
                            their_json_str = '\n'.join(marker["their_content"])
                            
                            try:
                                our_json = json.loads(our_json_str)
                                their_json = json.loads(their_json_str)
                                
                                # Simple merge (their values override ours)
                                merged_json = {**our_json, **their_json}
                                merged_str = json.dumps(merged_json, indent=2)
                                resolved_lines.extend(merged_str.split('\n'))
                            except:
                                # Fallback to line merge
                                resolved_lines.extend(marker["our_content"])
                                resolved_lines.extend(marker["their_content"])
                        break
                
                if not in_conflict:
                    resolved_lines.append(line)
            
            return '\n'.join(resolved_lines)
            
        except Exception:
            # Fallback to simple merge
            return self._merge_with_preference(content, conflict_markers, "prefer_theirs")
    
    def _merge_with_preference(
        self,
        content: str,
        conflict_markers: List[Dict[str, Any]],
        preference: str
    ) -> str:
        """Merge conflicts with a specific preference strategy."""
        
        lines = content.split('\n')
        resolved_lines = []
        
        for i, line in enumerate(lines):
            if any(line.startswith(marker) for marker in self.conflict_markers):
                continue
                
            in_conflict = False
            for marker in conflict_markers:
                if marker["start_line"] <= i + 1 <= marker["end_line"]:
                    in_conflict = True
                    
                    if i + 1 == marker["start_line"] + 1:
                        if preference == "prefer_ours":
                            resolved_lines.extend(marker["our_content"])
                        elif preference == "prefer_theirs":
                            resolved_lines.extend(marker["their_content"])
                        elif preference == "merge_both":
                            resolved_lines.extend(marker["our_content"])
                            resolved_lines.extend(marker["their_content"])
                        elif preference == "higher_version":
                            # Try to determine which has higher version
                            our_text = '\n'.join(marker["our_content"])
                            their_text = '\n'.join(marker["their_content"])
                            
                            our_versions = re.findall(r'[\d\.]+', our_text)
                            their_versions = re.findall(r'[\d\.]+', their_text)
                            
                            if our_versions and their_versions:
                                try:
                                    if semver.compare(our_versions[0], their_versions[0]) > 0:
                                        resolved_lines.extend(marker["our_content"])
                                    else:
                                        resolved_lines.extend(marker["their_content"])
                                except:
                                    resolved_lines.extend(marker["their_content"])
                            else:
                                resolved_lines.extend(marker["their_content"])
                    break
            
            if not in_conflict:
                resolved_lines.append(line)
        
        return '\n'.join(resolved_lines)
    
    def _can_auto_resolve(self, conflict_file: ConflictFile) -> bool:
        """Determine if a conflict can be automatically resolved."""
        
        return (
            conflict_file.resolution_confidence > 0.7 and
            conflict_file.conflict_type in [
                "import_conflict",
                "dependency_conflict", 
                "documentation_conflict",
                "version_conflict"
            ]
        )
    
    def _generate_conflict_summary(self, conflicted_files: List[ConflictFile]) -> str:
        """Generate a human-readable conflict summary."""
        
        if not conflicted_files:
            return "No conflicts detected"
        
        conflict_types = {}
        for cf in conflicted_files:
            if cf.conflict_type not in conflict_types:
                conflict_types[cf.conflict_type] = 0
            conflict_types[cf.conflict_type] += 1
        
        summary_parts = []
        summary_parts.append(f"{len(conflicted_files)} conflicted files:")
        
        for conflict_type, count in conflict_types.items():
            friendly_name = conflict_type.replace('_', ' ').title()
            summary_parts.append(f"  - {count} {friendly_name}")
        
        auto_resolvable_count = sum(1 for cf in conflicted_files if self._can_auto_resolve(cf))
        summary_parts.append(f"{auto_resolvable_count} conflicts can be automatically resolved")
        
        return '\n'.join(summary_parts)
    
    def _determine_resolution_strategy(self, conflicted_files: List[ConflictFile]) -> ConflictResolutionStrategy:
        """Determine the best resolution strategy for conflicts."""
        
        if not conflicted_files:
            return ConflictResolutionStrategy.AUTOMATIC
        
        auto_resolvable_count = sum(1 for cf in conflicted_files if self._can_auto_resolve(cf))
        total_conflicts = len(conflicted_files)
        
        if auto_resolvable_count == total_conflicts:
            return ConflictResolutionStrategy.AUTOMATIC
        elif auto_resolvable_count > total_conflicts * 0.8:
            return ConflictResolutionStrategy.INTELLIGENT_MERGE
        elif any(cf.conflict_type == "code_conflict" for cf in conflicted_files):
            return ConflictResolutionStrategy.MANUAL_REQUIRED
        else:
            return ConflictResolutionStrategy.SEMANTIC_MERGE


class DependencyAnalyzer:
    """
    Automated dependency analysis and update management.
    
    Analyzes project dependencies, identifies security vulnerabilities,
    suggests updates, and manages dependency conflicts.
    """
    
    def __init__(self):
        self.dependency_parsers = {
            "requirements.txt": self._parse_python_requirements,
            "package.json": self._parse_node_dependencies,
            "pom.xml": self._parse_maven_dependencies,
            "Cargo.toml": self._parse_rust_dependencies,
            "go.mod": self._parse_go_dependencies
        }
        
        # Mock vulnerability database - in production, integrate with actual security databases
        self.vulnerability_db = {}
    
    async def analyze_dependencies(self, repo_path: str) -> Dict[str, Any]:
        """Analyze all dependencies in repository."""
        
        try:
            repo_path = Path(repo_path)
            analysis = {
                "total_dependencies": 0,
                "outdated_dependencies": 0,
                "security_vulnerabilities": 0,
                "breaking_updates": 0,
                "dependency_files": {},
                "recommendations": [],
                "update_strategy": {}
            }
            
            # Find dependency files
            dependency_files = []
            for pattern in self.dependency_parsers.keys():
                matches = list(repo_path.rglob(pattern))
                dependency_files.extend(matches)
            
            # Analyze each dependency file
            for dep_file in dependency_files:
                relative_path = str(dep_file.relative_to(repo_path))
                
                if dep_file.name in self.dependency_parsers:
                    parser = self.dependency_parsers[dep_file.name]
                    
                    with open(dep_file, 'r') as f:
                        content = f.read()
                    
                    dependencies = await parser(content)
                    
                    # Analyze each dependency
                    analyzed_deps = []
                    for dep in dependencies:
                        dep_analysis = await self._analyze_single_dependency(dep)
                        analyzed_deps.append(dep_analysis)
                        
                        # Update counters
                        analysis["total_dependencies"] += 1
                        if dep_analysis.update_type in [DependencyUpdateType.MAJOR, DependencyUpdateType.MINOR, DependencyUpdateType.PATCH]:
                            analysis["outdated_dependencies"] += 1
                        if dep_analysis.security_vulnerabilities:
                            analysis["security_vulnerabilities"] += len(dep_analysis.security_vulnerabilities)
                        if dep_analysis.update_type == DependencyUpdateType.BREAKING:
                            analysis["breaking_updates"] += 1
                    
                    analysis["dependency_files"][relative_path] = {
                        "dependencies": analyzed_deps,
                        "file_type": self._get_dependency_file_type(dep_file.name),
                        "total_count": len(analyzed_deps)
                    }
            
            # Generate recommendations
            analysis["recommendations"] = await self._generate_dependency_recommendations(analysis)
            
            # Generate update strategy
            analysis["update_strategy"] = self._generate_update_strategy(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze dependencies: {e}")
            raise AdvancedRepositoryManagementError(f"Dependency analysis failed: {str(e)}")
    
    async def _parse_python_requirements(self, content: str) -> List[Dict[str, str]]:
        """Parse Python requirements.txt file."""
        
        dependencies = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                if '==' in line:
                    name, version = line.split('==', 1)
                    dependencies.append({
                        "name": name.strip(),
                        "version": version.strip(),
                        "type": "python"
                    })
                elif '>=' in line:
                    name, version = line.split('>=', 1)
                    dependencies.append({
                        "name": name.strip(),
                        "version": f">={version.strip()}",
                        "type": "python"
                    })
                else:
                    dependencies.append({
                        "name": line,
                        "version": "latest",
                        "type": "python"
                    })
        
        return dependencies
    
    async def _parse_node_dependencies(self, content: str) -> List[Dict[str, str]]:
        """Parse Node.js package.json file."""
        
        try:
            data = json.loads(content)
            dependencies = []
            
            for dep_type in ["dependencies", "devDependencies"]:
                deps = data.get(dep_type, {})
                for name, version in deps.items():
                    dependencies.append({
                        "name": name,
                        "version": version,
                        "type": "node",
                        "dev_dependency": dep_type == "devDependencies"
                    })
            
            return dependencies
            
        except json.JSONDecodeError:
            return []
    
    async def _parse_maven_dependencies(self, content: str) -> List[Dict[str, str]]:
        """Parse Maven pom.xml file."""
        
        # Simplified XML parsing for dependencies
        dependencies = []
        
        # This would use proper XML parsing in production
        import xml.etree.ElementTree as ET
        
        try:
            root = ET.fromstring(content)
            
            # Find dependency elements
            for dep in root.findall(".//{http://maven.apache.org/POM/4.0.0}dependency"):
                group_id = dep.find("{http://maven.apache.org/POM/4.0.0}groupId")
                artifact_id = dep.find("{http://maven.apache.org/POM/4.0.0}artifactId")
                version = dep.find("{http://maven.apache.org/POM/4.0.0}version")
                
                if group_id is not None and artifact_id is not None:
                    dependencies.append({
                        "name": f"{group_id.text}:{artifact_id.text}",
                        "version": version.text if version is not None else "latest",
                        "type": "maven"
                    })
            
            return dependencies
            
        except ET.ParseError:
            return []
    
    async def _parse_rust_dependencies(self, content: str) -> List[Dict[str, str]]:
        """Parse Rust Cargo.toml file."""
        
        # Simplified TOML parsing
        dependencies = []
        
        try:
            import toml
            data = toml.loads(content)
            
            for dep_type in ["dependencies", "dev-dependencies"]:
                deps = data.get(dep_type, {})
                for name, version_info in deps.items():
                    if isinstance(version_info, str):
                        version = version_info
                    elif isinstance(version_info, dict):
                        version = version_info.get("version", "latest")
                    else:
                        version = "latest"
                    
                    dependencies.append({
                        "name": name,
                        "version": version,
                        "type": "rust",
                        "dev_dependency": dep_type == "dev-dependencies"
                    })
            
            return dependencies
            
        except:
            return []
    
    async def _parse_go_dependencies(self, content: str) -> List[Dict[str, str]]:
        """Parse Go go.mod file."""
        
        dependencies = []
        
        lines = content.split('\n')
        in_require_block = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("require ("):
                in_require_block = True
                continue
            elif line == ")" and in_require_block:
                in_require_block = False
                continue
            elif line.startswith("require ") and not in_require_block:
                parts = line[8:].strip().split()
                if len(parts) >= 2:
                    dependencies.append({
                        "name": parts[0],
                        "version": parts[1],
                        "type": "go"
                    })
            elif in_require_block and line:
                parts = line.split()
                if len(parts) >= 2:
                    dependencies.append({
                        "name": parts[0],
                        "version": parts[1],
                        "type": "go"
                    })
        
        return dependencies
    
    async def _analyze_single_dependency(self, dependency: Dict[str, str]) -> DependencyInfo:
        """Analyze a single dependency for updates and vulnerabilities."""
        
        name = dependency["name"]
        current_version = dependency["version"]
        dep_type = dependency["type"]
        
        # Mock analysis - in production, integrate with package registries
        latest_version = await self._get_latest_version(name, dep_type)
        update_type = self._determine_update_type(current_version, latest_version)
        vulnerabilities = await self._check_security_vulnerabilities(name, current_version, dep_type)
        breaking_changes = await self._check_breaking_changes(name, current_version, latest_version, dep_type)
        
        # Calculate update priority
        priority = self._calculate_update_priority(update_type, vulnerabilities, breaking_changes)
        
        return DependencyInfo(
            name=name,
            current_version=current_version,
            latest_version=latest_version,
            update_type=update_type,
            security_vulnerabilities=vulnerabilities,
            breaking_changes=breaking_changes,
            update_priority=priority
        )
    
    async def _get_latest_version(self, name: str, dep_type: str) -> str:
        """Get latest version from package registry."""
        
        # Mock implementation - integrate with actual package registries
        version_increments = {
            "python": "0.1.0",
            "node": "0.2.0", 
            "maven": "0.0.1",
            "rust": "0.3.0",
            "go": "0.1.0"
        }
        
        # Simulate version increment
        increment = version_increments.get(dep_type, "0.1.0")
        return f"latest+{increment}"
    
    def _determine_update_type(self, current: str, latest: str) -> DependencyUpdateType:
        """Determine the type of update (major, minor, patch)."""
        
        try:
            # Clean version strings
            current_clean = re.sub(r'^[^0-9]*', '', current)
            latest_clean = re.sub(r'^[^0-9]*', '', latest)
            
            if 'latest' in latest_clean:
                return DependencyUpdateType.MINOR  # Default assumption
            
            current_parts = current_clean.split('.')
            latest_parts = latest_clean.split('.')
            
            if len(current_parts) >= 1 and len(latest_parts) >= 1:
                if int(latest_parts[0]) > int(current_parts[0]):
                    return DependencyUpdateType.MAJOR
                elif len(current_parts) >= 2 and len(latest_parts) >= 2:
                    if int(latest_parts[1]) > int(current_parts[1]):
                        return DependencyUpdateType.MINOR
                    elif len(current_parts) >= 3 and len(latest_parts) >= 3:
                        if int(latest_parts[2]) > int(current_parts[2]):
                            return DependencyUpdateType.PATCH
            
            return DependencyUpdateType.PATCH
            
        except:
            return DependencyUpdateType.MINOR
    
    async def _check_security_vulnerabilities(
        self,
        name: str,
        version: str,
        dep_type: str
    ) -> List[Dict[str, Any]]:
        """Check for security vulnerabilities."""
        
        # Mock vulnerability check - integrate with security databases
        mock_vulnerabilities = []
        
        # Simulate some packages having vulnerabilities
        if "test" in name.lower() or "demo" in name.lower():
            mock_vulnerabilities.append({
                "id": "CVE-2024-0001",
                "severity": "medium",
                "description": f"Mock vulnerability in {name}",
                "fixed_version": "latest",
                "cvss_score": 5.3
            })
        
        return mock_vulnerabilities
    
    async def _check_breaking_changes(
        self,
        name: str,
        current_version: str,
        latest_version: str,
        dep_type: str
    ) -> List[str]:
        """Check for breaking changes between versions."""
        
        # Mock breaking changes check
        breaking_changes = []
        
        update_type = self._determine_update_type(current_version, latest_version)
        
        if update_type == DependencyUpdateType.MAJOR:
            breaking_changes.append(f"Major version update may contain breaking API changes")
            breaking_changes.append(f"Review migration guide for {name}")
        
        return breaking_changes
    
    def _calculate_update_priority(
        self,
        update_type: DependencyUpdateType,
        vulnerabilities: List[Dict[str, Any]],
        breaking_changes: List[str]
    ) -> int:
        """Calculate update priority (1-10, 10 being highest)."""
        
        priority = 1
        
        # Base priority from update type
        type_priorities = {
            DependencyUpdateType.SECURITY: 10,
            DependencyUpdateType.PATCH: 7,
            DependencyUpdateType.MINOR: 5,
            DependencyUpdateType.MAJOR: 3,
            DependencyUpdateType.BREAKING: 2
        }
        
        priority = type_priorities.get(update_type, 3)
        
        # Boost priority for security vulnerabilities
        if vulnerabilities:
            max_severity = max(
                vuln.get("cvss_score", 0) for vuln in vulnerabilities
            )
            if max_severity >= 9.0:  # Critical
                priority = 10
            elif max_severity >= 7.0:  # High
                priority = max(priority, 9)
            elif max_severity >= 4.0:  # Medium
                priority = max(priority, 7)
        
        # Reduce priority for breaking changes
        if breaking_changes:
            priority = max(1, priority - 2)
        
        return priority
    
    def _get_dependency_file_type(self, filename: str) -> str:
        """Get dependency file type from filename."""
        
        type_map = {
            "requirements.txt": "python",
            "package.json": "node",
            "pom.xml": "maven",
            "Cargo.toml": "rust",
            "go.mod": "go"
        }
        
        return type_map.get(filename, "unknown")
    
    async def _generate_dependency_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate dependency update recommendations."""
        
        recommendations = []
        
        # High priority security updates
        security_count = analysis.get("security_vulnerabilities", 0)
        if security_count > 0:
            recommendations.append({
                "type": "security",
                "priority": "critical",
                "title": "Security Vulnerabilities Found",
                "description": f"{security_count} security vulnerabilities detected",
                "recommendation": "Update affected packages immediately",
                "action": "Run security-focused dependency updates first"
            })
        
        # Outdated dependencies
        outdated_count = analysis.get("outdated_dependencies", 0)
        total_count = analysis.get("total_dependencies", 0)
        
        if outdated_count > 0:
            outdated_percentage = (outdated_count / total_count) * 100 if total_count > 0 else 0
            
            if outdated_percentage > 50:
                recommendations.append({
                    "type": "maintenance",
                    "priority": "high",
                    "title": "Many Outdated Dependencies",
                    "description": f"{outdated_count} out of {total_count} dependencies are outdated",
                    "recommendation": "Schedule comprehensive dependency update sprint",
                    "action": "Update non-breaking dependencies in batches"
                })
            elif outdated_percentage > 25:
                recommendations.append({
                    "type": "maintenance",
                    "priority": "medium", 
                    "title": "Some Outdated Dependencies",
                    "description": f"{outdated_count} dependencies need updates",
                    "recommendation": "Update patch and minor version dependencies",
                    "action": "Focus on low-risk updates first"
                })
        
        # Breaking changes
        breaking_count = analysis.get("breaking_updates", 0)
        if breaking_count > 0:
            recommendations.append({
                "type": "breaking_changes",
                "priority": "low",
                "title": "Breaking Changes Available",
                "description": f"{breaking_count} major version updates available",
                "recommendation": "Plan migration strategy for major updates",
                "action": "Schedule separate sprint for breaking changes"
            })
        
        return recommendations
    
    def _generate_update_strategy(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommended update strategy."""
        
        security_count = analysis.get("security_vulnerabilities", 0)
        outdated_count = analysis.get("outdated_dependencies", 0)
        breaking_count = analysis.get("breaking_updates", 0)
        
        strategy = {
            "phase_1_security": [],
            "phase_2_patches": [],
            "phase_3_minor": [],
            "phase_4_major": [],
            "estimated_effort_hours": 0
        }
        
        # Collect high-priority updates by phase
        for file_info in analysis.get("dependency_files", {}).values():
            for dep in file_info.get("dependencies", []):
                if isinstance(dep, DependencyInfo):
                    if dep.security_vulnerabilities:
                        strategy["phase_1_security"].append(dep.name)
                    elif dep.update_type == DependencyUpdateType.PATCH:
                        strategy["phase_2_patches"].append(dep.name)
                    elif dep.update_type == DependencyUpdateType.MINOR:
                        strategy["phase_3_minor"].append(dep.name)
                    elif dep.update_type in [DependencyUpdateType.MAJOR, DependencyUpdateType.BREAKING]:
                        strategy["phase_4_major"].append(dep.name)
        
        # Estimate effort
        effort = (
            len(strategy["phase_1_security"]) * 0.5 +
            len(strategy["phase_2_patches"]) * 0.25 +
            len(strategy["phase_3_minor"]) * 0.75 +
            len(strategy["phase_4_major"]) * 2.0
        )
        
        strategy["estimated_effort_hours"] = max(1, int(effort))
        
        return strategy


class AdvancedRepositoryManagement:
    """
    Advanced repository management system combining intelligent conflict resolution,
    dependency analysis, and automated repository maintenance.
    """
    
    def __init__(self, github_client: GitHubAPIClient = None):
        self.github_client = github_client or GitHubAPIClient()
        self.conflict_resolver = IntelligentConflictResolver()
        self.dependency_analyzer = DependencyAnalyzer()
        self.redis = get_redis()
    
    async def perform_intelligent_merge(
        self,
        pull_request: PullRequest,
        resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.AUTOMATIC
    ) -> Dict[str, Any]:
        """Perform intelligent merge with conflict resolution."""
        
        try:
            logger.info(
                "Performing intelligent merge",
                pr_number=pull_request.github_pr_number,
                strategy=resolution_strategy.value
            )
            
            # Get work tree for merge operation
            work_tree = await self._get_or_create_work_tree(pull_request)
            
            # Analyze conflicts
            conflict_analysis = await self.conflict_resolver.analyze_conflicts(
                work_tree.work_tree_path,
                pull_request.source_branch,
                pull_request.target_branch
            )
            
            merge_result = {
                "success": False,
                "conflict_analysis": {
                    "conflicts_detected": len(conflict_analysis.conflicted_files),
                    "auto_resolvable": conflict_analysis.auto_resolvable,
                    "resolution_strategy": conflict_analysis.resolution_strategy.value,
                    "conflict_summary": conflict_analysis.conflict_summary
                },
                "resolution_applied": False,
                "files_modified": [],
                "merge_commit": None
            }
            
            # Attempt resolution if conflicts exist
            if conflict_analysis.conflicted_files:
                if conflict_analysis.auto_resolvable and resolution_strategy in [
                    ConflictResolutionStrategy.AUTOMATIC,
                    ConflictResolutionStrategy.INTELLIGENT_MERGE
                ]:
                    # Apply automatic resolution
                    resolution_success = await self._apply_conflict_resolutions(
                        work_tree.work_tree_path,
                        conflict_analysis.conflicted_files
                    )
                    
                    if resolution_success:
                        merge_result["resolution_applied"] = True
                        merge_result["files_modified"] = [cf.file_path for cf in conflict_analysis.conflicted_files]
                        
                        # Complete merge
                        merge_commit = await self._complete_merge(
                            work_tree.work_tree_path,
                            f"Intelligent merge: {pull_request.title}"
                        )
                        
                        merge_result["merge_commit"] = merge_commit
                        merge_result["success"] = True
                
                else:
                    merge_result["message"] = "Manual resolution required - conflicts too complex for automatic resolution"
            
            else:
                # No conflicts - clean merge
                merge_commit = await self._complete_merge(
                    work_tree.work_tree_path,
                    f"Clean merge: {pull_request.title}"
                )
                
                merge_result["merge_commit"] = merge_commit
                merge_result["success"] = True
            
            # Record merge operation
            await self._record_merge_operation(pull_request, merge_result, conflict_analysis)
            
            return merge_result
            
        except Exception as e:
            logger.error(f"Intelligent merge failed: {e}")
            raise AdvancedRepositoryManagementError(f"Merge failed: {str(e)}")
    
    async def analyze_repository_health(
        self,
        repository: GitHubRepository,
        include_dependencies: bool = True
    ) -> Dict[str, Any]:
        """Perform comprehensive repository health analysis."""
        
        try:
            logger.info(
                "Analyzing repository health",
                repository=repository.repository_full_name
            )
            
            health_analysis = {
                "overall_status": RepositoryHealthStatus.UNKNOWN,
                "score": 0.0,
                "branch_analysis": {},
                "dependency_analysis": {},
                "repository_metrics": {},
                "recommendations": [],
                "alerts": []
            }
            
            # Get repository work tree
            work_tree = await self._get_repository_work_tree(repository)
            
            # Analyze branches
            branch_analysis = await self._analyze_branches(work_tree.work_tree_path)
            health_analysis["branch_analysis"] = branch_analysis
            
            # Analyze dependencies if requested
            if include_dependencies:
                dependency_analysis = await self.dependency_analyzer.analyze_dependencies(
                    work_tree.work_tree_path
                )
                health_analysis["dependency_analysis"] = dependency_analysis
            
            # Calculate repository metrics
            repo_metrics = await self._calculate_repository_metrics(
                work_tree.work_tree_path,
                branch_analysis,
                dependency_analysis if include_dependencies else {}
            )
            health_analysis["repository_metrics"] = repo_metrics
            
            # Calculate overall health score
            overall_score = self._calculate_health_score(
                branch_analysis, dependency_analysis if include_dependencies else {}, repo_metrics
            )
            health_analysis["score"] = overall_score
            health_analysis["overall_status"] = self._determine_health_status(overall_score)
            
            # Generate recommendations
            recommendations = await self._generate_health_recommendations(
                branch_analysis, dependency_analysis if include_dependencies else {}, repo_metrics
            )
            health_analysis["recommendations"] = recommendations
            
            # Generate alerts for critical issues
            alerts = self._generate_health_alerts(
                branch_analysis, dependency_analysis if include_dependencies else {}
            )
            health_analysis["alerts"] = alerts
            
            return health_analysis
            
        except Exception as e:
            logger.error(f"Repository health analysis failed: {e}")
            raise AdvancedRepositoryManagementError(f"Health analysis failed: {str(e)}")
    
    async def _get_or_create_work_tree(self, pull_request: PullRequest) -> AgentWorkTree:
        """Get or create work tree for pull request."""
        
        async with get_db_session() as session:
            # Try to find existing work tree
            result = await session.execute(
                select(AgentWorkTree).where(
                    and_(
                        AgentWorkTree.agent_id == pull_request.agent_id,
                        AgentWorkTree.repository_id == pull_request.repository_id
                    )
                )
            )
            
            work_tree = result.scalar_one_or_none()
            
            if not work_tree:
                # Create new work tree
                work_tree_path = self._generate_work_tree_path(pull_request)
                
                work_tree = AgentWorkTree(
                    agent_id=pull_request.agent_id,
                    repository_id=pull_request.repository_id,
                    work_tree_path=work_tree_path,
                    branch_name=pull_request.source_branch,
                    base_branch=pull_request.target_branch
                )
                
                session.add(work_tree)
                await session.commit()
                await session.refresh(work_tree)
                
                # Initialize work tree
                await self._initialize_work_tree(work_tree)
            
            return work_tree
    
    async def _get_repository_work_tree(self, repository: GitHubRepository) -> AgentWorkTree:
        """Get or create work tree for repository analysis."""
        
        # Create temporary work tree for analysis
        work_tree_path = f"/tmp/repo-analysis-{uuid.uuid4()}"
        
        # Clone repository
        repo_url = repository.get_clone_url_for_agent(with_token=True)
        
        repo = Repo.clone_from(repo_url, work_tree_path)
        
        # Create temporary work tree object
        work_tree = AgentWorkTree(
            id=uuid.uuid4(),
            agent_id=uuid.uuid4(),  # Temporary agent ID
            repository_id=repository.id,
            work_tree_path=work_tree_path,
            branch_name=repository.default_branch,
            base_branch=repository.default_branch
        )
        
        return work_tree
    
    def _generate_work_tree_path(self, pull_request: PullRequest) -> str:
        """Generate work tree path for pull request."""
        
        base_path = getattr(settings, 'WORK_TREE_BASE_PATH', '/tmp/work-trees')
        return f"{base_path}/pr-{pull_request.id}-{uuid.uuid4()}"
    
    async def _initialize_work_tree(self, work_tree: AgentWorkTree) -> None:
        """Initialize work tree with repository clone."""
        
        try:
            # Get repository info
            async with get_db_session() as session:
                result = await session.execute(
                    select(GitHubRepository).where(
                        GitHubRepository.id == work_tree.repository_id
                    )
                )
                repository = result.scalar_one()
            
            # Create work tree directory
            Path(work_tree.work_tree_path).mkdir(parents=True, exist_ok=True)
            
            # Clone repository
            repo_url = repository.get_clone_url_for_agent(with_token=True)
            repo = Repo.clone_from(repo_url, work_tree.work_tree_path)
            
            # Checkout source branch
            if work_tree.branch_name != repository.default_branch:
                try:
                    repo.git.checkout(work_tree.branch_name)
                except:
                    # Branch doesn't exist locally, create from remote
                    repo.git.checkout("-b", work_tree.branch_name, f"origin/{work_tree.branch_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize work tree: {e}")
            raise AdvancedRepositoryManagementError(f"Work tree initialization failed: {str(e)}")
    
    async def _apply_conflict_resolutions(
        self,
        work_tree_path: str,
        conflicted_files: List[ConflictFile]
    ) -> bool:
        """Apply automatic resolutions to conflicted files."""
        
        try:
            for conflict_file in conflicted_files:
                if conflict_file.suggested_resolution and self.conflict_resolver._can_auto_resolve(conflict_file):
                    file_path = Path(work_tree_path) / conflict_file.file_path
                    
                    # Write resolved content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(conflict_file.suggested_resolution)
                    
                    # Stage the resolved file
                    repo = Repo(work_tree_path)
                    repo.index.add([conflict_file.file_path])
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply conflict resolutions: {e}")
            return False
    
    async def _complete_merge(self, work_tree_path: str, commit_message: str) -> str:
        """Complete merge operation with commit."""
        
        try:
            repo = Repo(work_tree_path)
            
            # Commit the merge
            commit = repo.index.commit(commit_message)
            
            return commit.hexsha
            
        except Exception as e:
            logger.error(f"Failed to complete merge: {e}")
            raise AdvancedRepositoryManagementError(f"Merge completion failed: {str(e)}")
    
    async def _record_merge_operation(
        self,
        pull_request: PullRequest,
        merge_result: Dict[str, Any],
        conflict_analysis: MergeConflict
    ) -> None:
        """Record merge operation in database."""
        
        try:
            async with get_db_session() as session:
                # Get work tree
                work_tree = await self._get_or_create_work_tree(pull_request)
                
                operation = BranchOperation(
                    repository_id=pull_request.repository_id,
                    agent_id=pull_request.agent_id,
                    work_tree_id=work_tree.id,
                    operation_type=BranchOperationType.MERGE,
                    source_branch=pull_request.source_branch,
                    target_branch=pull_request.target_branch,
                    status="completed" if merge_result["success"] else "failed",
                    conflicts_detected=len(conflict_analysis.conflicted_files),
                    conflicts_resolved=len([cf for cf in conflict_analysis.conflicted_files if self.conflict_resolver._can_auto_resolve(cf)]),
                    conflict_details=[{
                        "file_path": cf.file_path,
                        "conflict_type": cf.conflict_type,
                        "resolution_confidence": cf.resolution_confidence
                    } for cf in conflict_analysis.conflicted_files],
                    resolution_strategy=conflict_analysis.resolution_strategy.value,
                    operation_result=merge_result
                )
                
                operation.start_operation()
                operation.complete_operation(merge_result["success"])
                
                session.add(operation)
                await session.commit()
            
        except Exception as e:
            logger.error(f"Failed to record merge operation: {e}")
    
    async def _analyze_branches(self, repo_path: str) -> Dict[str, Any]:
        """Analyze all branches in repository."""
        
        try:
            repo = Repo(repo_path)
            branch_analysis = {}
            
            for branch in repo.branches:
                branch_name = branch.name
                
                # Get branch commits
                commits = list(repo.iter_commits(branch_name, max_count=100))
                
                # Calculate metrics
                commit_count = len(commits)
                
                # Commits behind main
                try:
                    main_commits = set(commit.hexsha for commit in repo.iter_commits('main', max_count=200))
                    branch_commits = set(commit.hexsha for commit in commits)
                    commits_behind_main = len(main_commits - branch_commits)
                except:
                    commits_behind_main = 0
                
                # Last commit date
                last_commit_date = commits[0].committed_datetime if commits else datetime.utcnow()
                
                # Author count
                authors = set(commit.author.email for commit in commits)
                author_count = len(authors)
                
                # File changes (simplified analysis)
                file_changes = 0
                lines_added = 0
                lines_deleted = 0
                
                for commit in commits[:10]:  # Analyze recent commits
                    stats = commit.stats
                    file_changes += len(stats.files)
                    lines_added += stats.total['insertions']
                    lines_deleted += stats.total['deletions']
                
                branch_info = BranchAnalysis(
                    branch_name=branch_name,
                    commit_count=commit_count,
                    commits_behind_main=commits_behind_main,
                    last_commit_date=last_commit_date,
                    author_count=author_count,
                    file_change_count=file_changes,
                    lines_added=lines_added,
                    lines_deleted=lines_deleted
                )
                
                branch_analysis[branch_name] = {
                    "commit_count": commit_count,
                    "commits_behind_main": commits_behind_main,
                    "last_commit_date": last_commit_date.isoformat(),
                    "author_count": author_count,
                    "file_change_count": file_changes,
                    "lines_added": lines_added,
                    "lines_deleted": lines_deleted,
                    "is_stale": (datetime.utcnow() - last_commit_date).days > 30,
                    "is_active": (datetime.utcnow() - last_commit_date).days <= 7
                }
            
            return branch_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze branches: {e}")
            return {}
    
    async def _calculate_repository_metrics(
        self,
        repo_path: str,
        branch_analysis: Dict[str, Any],
        dependency_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive repository metrics."""
        
        try:
            repo = Repo(repo_path)
            
            metrics = {
                "total_commits": 0,
                "total_contributors": 0,
                "total_branches": len(branch_analysis),
                "stale_branches": 0,
                "active_branches": 0,
                "total_files": 0,
                "code_files": 0,
                "test_files": 0,
                "documentation_files": 0,
                "last_activity_days": 0
            }
            
            # Branch metrics
            for branch_info in branch_analysis.values():
                if branch_info["is_stale"]:
                    metrics["stale_branches"] += 1
                if branch_info["is_active"]:
                    metrics["active_branches"] += 1
            
            # Repository-wide metrics
            all_commits = list(repo.iter_commits('main', max_count=1000))
            metrics["total_commits"] = len(all_commits)
            
            contributors = set(commit.author.email for commit in all_commits)
            metrics["total_contributors"] = len(contributors)
            
            # File metrics
            for root, dirs, files in Path(repo_path).rglob('*'):
                if '.git' not in str(root):
                    for file in files:
                        file_path = Path(root) / file
                        metrics["total_files"] += 1
                        
                        if file_path.suffix in ['.py', '.js', '.java', '.cpp', '.c', '.go', '.rs']:
                            metrics["code_files"] += 1
                        elif 'test' in str(file_path).lower() or file_path.suffix in ['.test.js', '.spec.py']:
                            metrics["test_files"] += 1
                        elif file_path.suffix in ['.md', '.txt', '.rst']:
                            metrics["documentation_files"] += 1
            
            # Last activity
            if all_commits:
                last_commit_date = all_commits[0].committed_datetime
                metrics["last_activity_days"] = (datetime.utcnow() - last_commit_date).days
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate repository metrics: {e}")
            return {}
    
    def _calculate_health_score(
        self,
        branch_analysis: Dict[str, Any],
        dependency_analysis: Dict[str, Any],
        repo_metrics: Dict[str, Any]
    ) -> float:
        """Calculate overall repository health score (0-100)."""
        
        score = 100.0
        
        # Branch health (30% weight)
        stale_branches = repo_metrics.get("stale_branches", 0)
        total_branches = repo_metrics.get("total_branches", 1)
        stale_ratio = stale_branches / total_branches
        
        branch_penalty = stale_ratio * 30
        score -= branch_penalty
        
        # Dependency health (40% weight)
        if dependency_analysis:
            total_deps = dependency_analysis.get("total_dependencies", 1)
            outdated_deps = dependency_analysis.get("outdated_dependencies", 0)
            security_vulns = dependency_analysis.get("security_vulnerabilities", 0)
            
            outdated_ratio = outdated_deps / total_deps if total_deps > 0 else 0
            dependency_penalty = outdated_ratio * 25 + min(security_vulns * 5, 15)
            score -= dependency_penalty
        
        # Activity health (20% weight)
        last_activity = repo_metrics.get("last_activity_days", 0)
        if last_activity > 30:
            activity_penalty = min((last_activity - 30) / 30 * 20, 20)
            score -= activity_penalty
        
        # Documentation and testing (10% weight)
        total_files = repo_metrics.get("total_files", 1)
        code_files = repo_metrics.get("code_files", 0)
        test_files = repo_metrics.get("test_files", 0)
        doc_files = repo_metrics.get("documentation_files", 0)
        
        test_ratio = test_files / code_files if code_files > 0 else 0
        doc_ratio = doc_files / total_files if total_files > 0 else 0
        
        if test_ratio < 0.2:  # Less than 20% test coverage
            score -= 5
        if doc_ratio < 0.1:  # Less than 10% documentation
            score -= 5
        
        return max(0.0, min(100.0, score))
    
    def _determine_health_status(self, score: float) -> RepositoryHealthStatus:
        """Determine health status from score."""
        
        if score >= 90:
            return RepositoryHealthStatus.EXCELLENT
        elif score >= 75:
            return RepositoryHealthStatus.GOOD
        elif score >= 50:
            return RepositoryHealthStatus.WARNING
        else:
            return RepositoryHealthStatus.CRITICAL
    
    async def _generate_health_recommendations(
        self,
        branch_analysis: Dict[str, Any],
        dependency_analysis: Dict[str, Any],
        repo_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate repository health recommendations."""
        
        recommendations = []
        
        # Stale branch recommendations
        stale_branches = repo_metrics.get("stale_branches", 0)
        if stale_branches > 0:
            recommendations.append({
                "type": "branch_management",
                "priority": "medium",
                "title": "Clean Up Stale Branches",
                "description": f"{stale_branches} stale branches detected",
                "recommendation": "Review and delete unused branches to improve repository hygiene",
                "action": "Run branch cleanup automation"
            })
        
        # Dependency recommendations
        if dependency_analysis:
            dep_recommendations = dependency_analysis.get("recommendations", [])
            recommendations.extend(dep_recommendations)
        
        # Activity recommendations
        last_activity = repo_metrics.get("last_activity_days", 0)
        if last_activity > 14:
            recommendations.append({
                "type": "activity",
                "priority": "low",
                "title": "Low Repository Activity",
                "description": f"No commits in {last_activity} days",
                "recommendation": "Consider archiving if no longer active, or schedule maintenance",
                "action": "Review repository status"
            })
        
        # Testing recommendations
        code_files = repo_metrics.get("code_files", 0)
        test_files = repo_metrics.get("test_files", 0)
        
        if code_files > 0:
            test_ratio = test_files / code_files
            if test_ratio < 0.2:
                recommendations.append({
                    "type": "testing",
                    "priority": "high",
                    "title": "Low Test Coverage",
                    "description": f"Only {test_files} test files for {code_files} code files",
                    "recommendation": "Increase test coverage to improve code quality",
                    "action": "Add unit tests for core functionality"
                })
        
        return recommendations
    
    def _generate_health_alerts(
        self,
        branch_analysis: Dict[str, Any],
        dependency_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate critical health alerts."""
        
        alerts = []
        
        # Security vulnerability alerts
        if dependency_analysis:
            security_count = dependency_analysis.get("security_vulnerabilities", 0)
            if security_count > 0:
                alerts.append({
                    "type": "security",
                    "severity": "critical",
                    "title": "Security Vulnerabilities",
                    "message": f"{security_count} security vulnerabilities found in dependencies",
                    "action_required": "Update affected packages immediately"
                })
        
        # Too many stale branches
        stale_count = sum(1 for info in branch_analysis.values() if info.get("is_stale", False))
        if stale_count > 10:
            alerts.append({
                "type": "maintenance",
                "severity": "warning",
                "title": "Too Many Stale Branches",
                "message": f"{stale_count} branches haven't been updated in over 30 days",
                "action_required": "Review and clean up unused branches"
            })
        
        return alerts


# Factory function
async def create_advanced_repository_management() -> AdvancedRepositoryManagement:
    """Create and initialize advanced repository management."""
    
    github_client = GitHubAPIClient()
    return AdvancedRepositoryManagement(github_client)


# Export main classes
__all__ = [
    "AdvancedRepositoryManagement",
    "IntelligentConflictResolver",
    "DependencyAnalyzer",
    "MergeConflict",
    "ConflictFile",
    "DependencyInfo",
    "BranchAnalysis",
    "ConflictResolutionStrategy",
    "RepositoryHealthStatus",
    "AdvancedRepositoryManagementError",
    "create_advanced_repository_management"
]