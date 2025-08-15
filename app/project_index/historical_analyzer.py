"""
Historical Analyzer for LeanVibe Agent Hive 2.0

Advanced Git history analysis engine that provides insights into code evolution,
change patterns, bug history, and team collaboration metrics. Helps assess
file relevance based on historical development patterns.
"""

import asyncio
import json
import subprocess
import time
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import structlog

logger = structlog.get_logger()


class ChangeType(Enum):
    """Types of file changes."""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    COPIED = "copied"


class AnalysisScope(Enum):
    """Scope of historical analysis."""
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_QUARTER = "last_quarter"
    LAST_YEAR = "last_year"
    ALL_TIME = "all_time"


@dataclass
class GitCommit:
    """Git commit information."""
    commit_hash: str
    author: str
    author_email: str
    date: datetime
    message: str
    files_changed: List[str]
    additions: int
    deletions: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileHistory:
    """Historical information for a file."""
    file_path: str
    creation_date: Optional[datetime]
    last_modified: Optional[datetime]
    total_commits: int
    total_authors: int
    total_additions: int
    total_deletions: int
    change_frequency: float
    stability_score: float
    bug_fix_count: int
    feature_commit_count: int
    refactor_commit_count: int
    authors: List[str]
    recent_commits: List[GitCommit]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChangePattern:
    """Pattern of changes over time."""
    pattern_type: str
    description: str
    confidence: float
    time_period: str
    evidence: List[str]
    files_involved: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TeamCollaborationMetrics:
    """Team collaboration metrics for files."""
    file_path: str
    primary_author: str
    primary_author_percentage: float
    author_count: int
    collaboration_score: float
    knowledge_distribution: Dict[str, float]
    bus_factor: int
    expertise_level: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoricalAnalysisResult:
    """Complete historical analysis result."""
    file_path: str
    file_history: FileHistory
    change_patterns: List[ChangePattern]
    collaboration_metrics: TeamCollaborationMetrics
    relevance_indicators: Dict[str, float]
    risk_assessment: Dict[str, float]
    recommendations: List[str]
    analysis_timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class HistoricalAnalyzer:
    """
    Advanced Git history analyzer for code intelligence.
    
    Provides comprehensive analysis of:
    - File change patterns and frequency
    - Bug history and stability metrics
    - Team collaboration and knowledge distribution
    - Development velocity and trends
    - Risk assessment based on historical data
    """
    
    def __init__(self, cache_results: bool = True):
        """Initialize HistoricalAnalyzer."""
        self.cache_results = cache_results
        self.analysis_cache: Dict[str, HistoricalAnalysisResult] = {}
        
        # Analysis statistics
        self.analysis_stats = {
            "files_analyzed": 0,
            "git_commands_executed": 0,
            "cache_hits": 0,
            "analysis_time_total": 0.0
        }
        
        # Configuration
        self.config = {
            "max_commits_per_file": 1000,
            "analysis_timeout": 30,  # seconds
            "bug_keywords": [
                "fix", "bug", "issue", "error", "crash", "problem",
                "broken", "failing", "regression", "hotfix"
            ],
            "feature_keywords": [
                "add", "new", "feature", "implement", "create",
                "enhance", "improve", "support"
            ],
            "refactor_keywords": [
                "refactor", "cleanup", "reorganize", "restructure",
                "simplify", "optimize", "clean"
            ]
        }
    
    async def analyze_file_history(
        self,
        file_path: str,
        project_path: str,
        scope: AnalysisScope = AnalysisScope.LAST_YEAR
    ) -> HistoricalAnalysisResult:
        """
        Analyze complete history for a specific file.
        
        Args:
            file_path: Path to the file relative to project root
            project_path: Path to the project root
            scope: Time scope for analysis
            
        Returns:
            Complete historical analysis result
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{file_path}_{scope.value}"
            if self.cache_results and cache_key in self.analysis_cache:
                self.analysis_stats["cache_hits"] += 1
                return self.analysis_cache[cache_key]
            
            logger.debug("Starting file history analysis",
                        file_path=file_path,
                        scope=scope.value)
            
            # Get file history
            file_history = await self._analyze_file_history(
                file_path, project_path, scope
            )
            
            # Identify change patterns
            change_patterns = await self._identify_change_patterns(
                file_path, project_path, file_history, scope
            )
            
            # Analyze team collaboration
            collaboration_metrics = await self._analyze_team_collaboration(
                file_path, project_path, file_history
            )
            
            # Calculate relevance indicators
            relevance_indicators = await self._calculate_relevance_indicators(
                file_history, change_patterns, collaboration_metrics
            )
            
            # Assess risks
            risk_assessment = await self._assess_historical_risks(
                file_history, change_patterns, collaboration_metrics
            )
            
            # Generate recommendations
            recommendations = await self._generate_historical_recommendations(
                file_history, change_patterns, collaboration_metrics, relevance_indicators
            )
            
            # Create result
            result = HistoricalAnalysisResult(
                file_path=file_path,
                file_history=file_history,
                change_patterns=change_patterns,
                collaboration_metrics=collaboration_metrics,
                relevance_indicators=relevance_indicators,
                risk_assessment=risk_assessment,
                recommendations=recommendations,
                analysis_timestamp=datetime.utcnow(),
                metadata={
                    "scope": scope.value,
                    "analysis_time": time.time() - start_time,
                    "project_path": project_path
                }
            )
            
            # Cache result
            if self.cache_results:
                self.analysis_cache[cache_key] = result
            
            # Update statistics
            self.analysis_stats["files_analyzed"] += 1
            self.analysis_stats["analysis_time_total"] += time.time() - start_time
            
            logger.debug("File history analysis completed",
                        file_path=file_path,
                        analysis_time=time.time() - start_time)
            
            return result
            
        except Exception as e:
            logger.error("File history analysis failed",
                        file_path=file_path,
                        error=str(e))
            
            # Return default result
            return self._create_default_analysis_result(file_path)
    
    async def get_change_frequency(
        self,
        file_path: str,
        project_path: str,
        days: int = 90
    ) -> float:
        """
        Get change frequency for a file over specified days.
        
        Args:
            file_path: Path to the file
            project_path: Project root path
            days: Number of days to analyze
            
        Returns:
            Change frequency (changes per day)
        """
        try:
            # Get commits for the file in the specified period
            commits = await self._get_file_commits(
                file_path, project_path, days=days
            )
            
            if not commits:
                return 0.0
            
            # Calculate frequency
            frequency = len(commits) / max(1, days)
            return frequency
            
        except Exception as e:
            logger.warning("Change frequency calculation failed",
                          file_path=file_path,
                          error=str(e))
            return 0.0
    
    async def get_recent_changes(
        self,
        file_path: str,
        project_path: str,
        days: int = 30
    ) -> int:
        """
        Get number of recent changes to a file.
        
        Args:
            file_path: Path to the file
            project_path: Project root path
            days: Number of days to look back
            
        Returns:
            Number of recent changes
        """
        try:
            commits = await self._get_file_commits(
                file_path, project_path, days=days
            )
            return len(commits)
            
        except Exception as e:
            logger.warning("Recent changes calculation failed",
                          file_path=file_path,
                          error=str(e))
            return 0
    
    async def get_bug_history(
        self,
        file_path: str,
        project_path: str
    ) -> int:
        """
        Get number of bug-related commits for a file.
        
        Args:
            file_path: Path to the file
            project_path: Project root path
            
        Returns:
            Number of bug-related commits
        """
        try:
            commits = await self._get_file_commits(file_path, project_path)
            
            bug_commits = 0
            for commit in commits:
                if self._is_bug_fix_commit(commit.message):
                    bug_commits += 1
            
            return bug_commits
            
        except Exception as e:
            logger.warning("Bug history calculation failed",
                          file_path=file_path,
                          error=str(e))
            return 0
    
    async def get_contributor_count(
        self,
        file_path: str,
        project_path: str
    ) -> int:
        """
        Get number of unique contributors to a file.
        
        Args:
            file_path: Path to the file
            project_path: Project root path
            
        Returns:
            Number of unique contributors
        """
        try:
            commits = await self._get_file_commits(file_path, project_path)
            
            contributors = set()
            for commit in commits:
                contributors.add(commit.author_email)
            
            return len(contributors)
            
        except Exception as e:
            logger.warning("Contributor count calculation failed",
                          file_path=file_path,
                          error=str(e))
            return 0
    
    async def identify_hotspots(
        self,
        project_path: str,
        threshold: float = 0.8
    ) -> List[Tuple[str, float]]:
        """
        Identify code hotspots (frequently changed files).
        
        Args:
            project_path: Project root path
            threshold: Threshold for hotspot identification
            
        Returns:
            List of (file_path, hotspot_score) tuples
        """
        try:
            # Get all files with changes
            all_commits = await self._get_all_commits(project_path)
            
            file_change_counts = defaultdict(int)
            for commit in all_commits:
                for file_path in commit.files_changed:
                    file_change_counts[file_path] += 1
            
            if not file_change_counts:
                return []
            
            # Calculate hotspot scores
            max_changes = max(file_change_counts.values())
            hotspots = []
            
            for file_path, changes in file_change_counts.items():
                hotspot_score = changes / max_changes
                if hotspot_score >= threshold:
                    hotspots.append((file_path, hotspot_score))
            
            # Sort by hotspot score (descending)
            hotspots.sort(key=lambda x: x[1], reverse=True)
            
            return hotspots
            
        except Exception as e:
            logger.error("Hotspot identification failed",
                        project_path=project_path,
                        error=str(e))
            return []
    
    async def analyze_development_velocity(
        self,
        project_path: str,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze development velocity metrics.
        
        Args:
            project_path: Project root path
            period_days: Period for velocity analysis
            
        Returns:
            Development velocity metrics
        """
        try:
            commits = await self._get_all_commits(project_path, days=period_days)
            
            if not commits:
                return {
                    "commits_per_day": 0.0,
                    "lines_added_per_day": 0.0,
                    "lines_deleted_per_day": 0.0,
                    "files_changed_per_day": 0.0,
                    "active_contributors": 0,
                    "commit_frequency": "low"
                }
            
            # Calculate metrics
            total_commits = len(commits)
            total_additions = sum(commit.additions for commit in commits)
            total_deletions = sum(commit.deletions for commit in commits)
            total_files = len(set(file_path for commit in commits for file_path in commit.files_changed))
            contributors = set(commit.author_email for commit in commits)
            
            velocity_metrics = {
                "commits_per_day": total_commits / period_days,
                "lines_added_per_day": total_additions / period_days,
                "lines_deleted_per_day": total_deletions / period_days,
                "files_changed_per_day": total_files / period_days,
                "active_contributors": len(contributors),
                "avg_additions_per_commit": total_additions / total_commits if total_commits > 0 else 0,
                "avg_deletions_per_commit": total_deletions / total_commits if total_commits > 0 else 0,
                "net_lines_per_day": (total_additions - total_deletions) / period_days
            }
            
            # Classify commit frequency
            commits_per_day = velocity_metrics["commits_per_day"]
            if commits_per_day >= 5:
                velocity_metrics["commit_frequency"] = "very_high"
            elif commits_per_day >= 2:
                velocity_metrics["commit_frequency"] = "high"
            elif commits_per_day >= 0.5:
                velocity_metrics["commit_frequency"] = "medium"
            elif commits_per_day >= 0.1:
                velocity_metrics["commit_frequency"] = "low"
            else:
                velocity_metrics["commit_frequency"] = "very_low"
            
            return velocity_metrics
            
        except Exception as e:
            logger.error("Development velocity analysis failed",
                        project_path=project_path,
                        error=str(e))
            return {}
    
    # Private helper methods
    
    async def _analyze_file_history(
        self,
        file_path: str,
        project_path: str,
        scope: AnalysisScope
    ) -> FileHistory:
        """Analyze detailed history for a file."""
        try:
            # Get commits for the file
            days = self._scope_to_days(scope)
            commits = await self._get_file_commits(file_path, project_path, days)
            
            if not commits:
                return self._create_empty_file_history(file_path)
            
            # Calculate basic metrics
            total_commits = len(commits)
            authors = list(set(commit.author_email for commit in commits))
            total_authors = len(authors)
            
            total_additions = sum(commit.additions for commit in commits)
            total_deletions = sum(commit.deletions for commit in commits)
            
            # Find creation and last modified dates
            creation_date = commits[-1].date if commits else None
            last_modified = commits[0].date if commits else None
            
            # Calculate change frequency
            if creation_date and last_modified:
                time_span = (last_modified - creation_date).days
                change_frequency = total_commits / max(1, time_span)
            else:
                change_frequency = 0.0
            
            # Calculate stability score
            stability_score = await self._calculate_stability_score(commits)
            
            # Count different types of commits
            bug_fix_count = sum(1 for commit in commits if self._is_bug_fix_commit(commit.message))
            feature_commit_count = sum(1 for commit in commits if self._is_feature_commit(commit.message))
            refactor_commit_count = sum(1 for commit in commits if self._is_refactor_commit(commit.message))
            
            # Get recent commits (last 10)
            recent_commits = commits[:10]
            
            return FileHistory(
                file_path=file_path,
                creation_date=creation_date,
                last_modified=last_modified,
                total_commits=total_commits,
                total_authors=total_authors,
                total_additions=total_additions,
                total_deletions=total_deletions,
                change_frequency=change_frequency,
                stability_score=stability_score,
                bug_fix_count=bug_fix_count,
                feature_commit_count=feature_commit_count,
                refactor_commit_count=refactor_commit_count,
                authors=authors,
                recent_commits=recent_commits,
                metadata={
                    "analysis_scope": scope.value,
                    "commits_analyzed": len(commits)
                }
            )
            
        except Exception as e:
            logger.warning("File history analysis failed",
                          file_path=file_path,
                          error=str(e))
            return self._create_empty_file_history(file_path)
    
    async def _identify_change_patterns(
        self,
        file_path: str,
        project_path: str,
        file_history: FileHistory,
        scope: AnalysisScope
    ) -> List[ChangePattern]:
        """Identify patterns in file changes."""
        patterns = []
        
        try:
            commits = file_history.recent_commits
            
            if len(commits) < 3:
                return patterns
            
            # Pattern 1: Frequent recent changes
            recent_commits = [c for c in commits if 
                            (datetime.utcnow() - c.date).days <= 7]
            
            if len(recent_commits) >= 3:
                patterns.append(ChangePattern(
                    pattern_type="frequent_recent_changes",
                    description="File has been changed frequently in the last week",
                    confidence=0.8,
                    time_period="last_week",
                    evidence=[f"Commit on {c.date.strftime('%Y-%m-%d')}: {c.message[:50]}" 
                             for c in recent_commits[:3]],
                    files_involved=[file_path],
                    metadata={"recent_commit_count": len(recent_commits)}
                ))
            
            # Pattern 2: Bug fix pattern
            bug_commits = [c for c in commits if self._is_bug_fix_commit(c.message)]
            if len(bug_commits) >= 2:
                bug_ratio = len(bug_commits) / len(commits)
                confidence = min(0.9, bug_ratio * 2)
                
                patterns.append(ChangePattern(
                    pattern_type="bug_fix_pattern",
                    description="File has a pattern of bug fixes",
                    confidence=confidence,
                    time_period=scope.value,
                    evidence=[f"Bug fix: {c.message[:50]}" for c in bug_commits[:3]],
                    files_involved=[file_path],
                    metadata={"bug_commit_ratio": bug_ratio}
                ))
            
            # Pattern 3: Feature development pattern
            feature_commits = [c for c in commits if self._is_feature_commit(c.message)]
            if len(feature_commits) >= 2:
                feature_ratio = len(feature_commits) / len(commits)
                confidence = min(0.9, feature_ratio * 1.5)
                
                patterns.append(ChangePattern(
                    pattern_type="feature_development_pattern",
                    description="File is actively used for feature development",
                    confidence=confidence,
                    time_period=scope.value,
                    evidence=[f"Feature: {c.message[:50]}" for c in feature_commits[:3]],
                    files_involved=[file_path],
                    metadata={"feature_commit_ratio": feature_ratio}
                ))
            
            # Pattern 4: Refactoring pattern
            refactor_commits = [c for c in commits if self._is_refactor_commit(c.message)]
            if len(refactor_commits) >= 2:
                refactor_ratio = len(refactor_commits) / len(commits)
                confidence = min(0.9, refactor_ratio * 2)
                
                patterns.append(ChangePattern(
                    pattern_type="refactoring_pattern",
                    description="File undergoes regular refactoring",
                    confidence=confidence,
                    time_period=scope.value,
                    evidence=[f"Refactor: {c.message[:50]}" for c in refactor_commits[:3]],
                    files_involved=[file_path],
                    metadata={"refactor_commit_ratio": refactor_ratio}
                ))
            
            # Pattern 5: Stability pattern
            if file_history.stability_score > 0.8:
                patterns.append(ChangePattern(
                    pattern_type="stability_pattern",
                    description="File shows stable development with consistent changes",
                    confidence=file_history.stability_score,
                    time_period=scope.value,
                    evidence=["Consistent commit intervals", "Regular development activity"],
                    files_involved=[file_path],
                    metadata={"stability_score": file_history.stability_score}
                ))
            
        except Exception as e:
            logger.warning("Change pattern identification failed",
                          file_path=file_path,
                          error=str(e))
        
        return patterns
    
    async def _analyze_team_collaboration(
        self,
        file_path: str,
        project_path: str,
        file_history: FileHistory
    ) -> TeamCollaborationMetrics:
        """Analyze team collaboration metrics for a file."""
        try:
            if not file_history.authors:
                return self._create_empty_collaboration_metrics(file_path)
            
            # Calculate author contributions
            author_commits = defaultdict(int)
            for commit in file_history.recent_commits:
                author_commits[commit.author_email] += 1
            
            # Find primary author
            primary_author = max(author_commits, key=author_commits.get)
            primary_author_commits = author_commits[primary_author]
            primary_author_percentage = primary_author_commits / file_history.total_commits
            
            # Calculate collaboration score
            # Higher score means better distributed knowledge
            if len(author_commits) == 1:
                collaboration_score = 0.0  # Single author = no collaboration
            else:
                # Use Gini coefficient for knowledge distribution
                collaboration_score = self._calculate_collaboration_score(author_commits)
            
            # Calculate knowledge distribution
            knowledge_distribution = {
                author: commits / file_history.total_commits
                for author, commits in author_commits.items()
            }
            
            # Calculate bus factor (number of people who need to be hit by a bus)
            # to lose 50% of the knowledge
            cumulative_knowledge = 0.0
            bus_factor = 0
            sorted_authors = sorted(knowledge_distribution.items(), 
                                  key=lambda x: x[1], reverse=True)
            
            for author, percentage in sorted_authors:
                cumulative_knowledge += percentage
                bus_factor += 1
                if cumulative_knowledge >= 0.5:
                    break
            
            # Determine expertise level
            if primary_author_percentage >= 0.8:
                expertise_level = "single_expert"
            elif primary_author_percentage >= 0.6:
                expertise_level = "primary_expert"
            elif primary_author_percentage >= 0.4:
                expertise_level = "shared_expertise"
            else:
                expertise_level = "distributed_expertise"
            
            return TeamCollaborationMetrics(
                file_path=file_path,
                primary_author=primary_author,
                primary_author_percentage=primary_author_percentage,
                author_count=len(author_commits),
                collaboration_score=collaboration_score,
                knowledge_distribution=knowledge_distribution,
                bus_factor=bus_factor,
                expertise_level=expertise_level,
                metadata={
                    "total_commits_analyzed": file_history.total_commits,
                    "recent_commits_analyzed": len(file_history.recent_commits)
                }
            )
            
        except Exception as e:
            logger.warning("Team collaboration analysis failed",
                          file_path=file_path,
                          error=str(e))
            return self._create_empty_collaboration_metrics(file_path)
    
    async def _calculate_relevance_indicators(
        self,
        file_history: FileHistory,
        change_patterns: List[ChangePattern],
        collaboration_metrics: TeamCollaborationMetrics
    ) -> Dict[str, float]:
        """Calculate relevance indicators based on historical data."""
        indicators = {}
        
        # Change frequency indicator
        if file_history.change_frequency > 0.1:  # More than 1 change per 10 days
            indicators["high_change_frequency"] = min(1.0, file_history.change_frequency * 10)
        else:
            indicators["high_change_frequency"] = 0.0
        
        # Recent activity indicator
        recent_commits = len([c for c in file_history.recent_commits 
                            if (datetime.utcnow() - c.date).days <= 30])
        indicators["recent_activity"] = min(1.0, recent_commits / 5.0)
        
        # Bug proneness indicator
        if file_history.total_commits > 0:
            bug_ratio = file_history.bug_fix_count / file_history.total_commits
            indicators["bug_proneness"] = min(1.0, bug_ratio * 3)
        else:
            indicators["bug_proneness"] = 0.0
        
        # Feature development indicator
        if file_history.total_commits > 0:
            feature_ratio = file_history.feature_commit_count / file_history.total_commits
            indicators["feature_development"] = min(1.0, feature_ratio * 2)
        else:
            indicators["feature_development"] = 0.0
        
        # Stability indicator
        indicators["stability"] = file_history.stability_score
        
        # Team knowledge indicator
        indicators["team_knowledge"] = collaboration_metrics.collaboration_score
        
        # Risk indicator (inverse of stability and collaboration)
        risk_score = (
            (1.0 - file_history.stability_score) * 0.4 +
            (1.0 - collaboration_metrics.collaboration_score) * 0.3 +
            indicators["bug_proneness"] * 0.3
        )
        indicators["risk_level"] = risk_score
        
        return indicators
    
    async def _assess_historical_risks(
        self,
        file_history: FileHistory,
        change_patterns: List[ChangePattern],
        collaboration_metrics: TeamCollaborationMetrics
    ) -> Dict[str, float]:
        """Assess risks based on historical patterns."""
        risks = {}
        
        # Knowledge concentration risk
        if collaboration_metrics.bus_factor <= 1:
            risks["knowledge_concentration"] = 0.9
        elif collaboration_metrics.bus_factor <= 2:
            risks["knowledge_concentration"] = 0.6
        else:
            risks["knowledge_concentration"] = 0.2
        
        # Change instability risk
        if file_history.change_frequency > 0.5:  # Very frequent changes
            risks["change_instability"] = 0.8
        elif file_history.change_frequency > 0.2:
            risks["change_instability"] = 0.5
        else:
            risks["change_instability"] = 0.1
        
        # Bug proneness risk
        if file_history.total_commits > 0:
            bug_ratio = file_history.bug_fix_count / file_history.total_commits
            if bug_ratio > 0.3:
                risks["bug_proneness"] = 0.9
            elif bug_ratio > 0.1:
                risks["bug_proneness"] = 0.6
            else:
                risks["bug_proneness"] = 0.2
        else:
            risks["bug_proneness"] = 0.5
        
        # Maintenance burden risk
        maintenance_indicators = [
            file_history.change_frequency > 0.3,
            file_history.bug_fix_count > file_history.feature_commit_count,
            collaboration_metrics.primary_author_percentage > 0.8
        ]
        
        maintenance_risk = sum(maintenance_indicators) / len(maintenance_indicators)
        risks["maintenance_burden"] = maintenance_risk
        
        return risks
    
    async def _generate_historical_recommendations(
        self,
        file_history: FileHistory,
        change_patterns: List[ChangePattern],
        collaboration_metrics: TeamCollaborationMetrics,
        relevance_indicators: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on historical analysis."""
        recommendations = []
        
        # Knowledge sharing recommendations
        if collaboration_metrics.bus_factor <= 1:
            recommendations.append(
                "Critical: Single point of failure - encourage knowledge sharing"
            )
        elif collaboration_metrics.bus_factor <= 2:
            recommendations.append(
                "Consider involving more team members in this file's development"
            )
        
        # Stability recommendations
        if file_history.stability_score < 0.5:
            recommendations.append(
                "File shows unstable development patterns - consider refactoring"
            )
        
        # Bug proneness recommendations
        if relevance_indicators.get("bug_proneness", 0) > 0.6:
            recommendations.append(
                "High bug frequency detected - increase testing and code review"
            )
        
        # Change frequency recommendations
        if relevance_indicators.get("high_change_frequency", 0) > 0.7:
            recommendations.append(
                "Frequent changes detected - ensure proper documentation"
            )
        elif relevance_indicators.get("recent_activity", 0) < 0.2:
            recommendations.append(
                "Low recent activity - verify if file is still relevant"
            )
        
        # Pattern-based recommendations
        for pattern in change_patterns:
            if pattern.pattern_type == "bug_fix_pattern" and pattern.confidence > 0.7:
                recommendations.append(
                    "Strong bug fix pattern - prioritize for quality improvements"
                )
            elif pattern.pattern_type == "refactoring_pattern" and pattern.confidence > 0.7:
                recommendations.append(
                    "Active refactoring detected - good maintenance practices"
                )
        
        return recommendations
    
    # Git interaction methods
    
    async def _get_file_commits(
        self,
        file_path: str,
        project_path: str,
        days: Optional[int] = None
    ) -> List[GitCommit]:
        """Get Git commits for a specific file."""
        try:
            self.analysis_stats["git_commands_executed"] += 1
            
            # Build git log command
            cmd = [
                "git", "log",
                "--follow",  # Follow file renames
                "--pretty=format:%H|%an|%ae|%ad|%s",
                "--date=iso",
                "--numstat"
            ]
            
            if days:
                since_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
                cmd.extend(["--since", since_date])
            
            cmd.append("--")
            cmd.append(file_path)
            
            # Execute git command
            result = await self._execute_git_command(cmd, project_path)
            
            if not result:
                return []
            
            # Parse output
            commits = self._parse_git_log_output(result, file_path)
            return commits
            
        except Exception as e:
            logger.warning("Git commit retrieval failed",
                          file_path=file_path,
                          error=str(e))
            return []
    
    async def _get_all_commits(
        self,
        project_path: str,
        days: Optional[int] = None
    ) -> List[GitCommit]:
        """Get all Git commits in the project."""
        try:
            self.analysis_stats["git_commands_executed"] += 1
            
            # Build git log command
            cmd = [
                "git", "log",
                "--pretty=format:%H|%an|%ae|%ad|%s",
                "--date=iso",
                "--numstat"
            ]
            
            if days:
                since_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
                cmd.extend(["--since", since_date])
            
            # Execute git command
            result = await self._execute_git_command(cmd, project_path)
            
            if not result:
                return []
            
            # Parse output
            commits = self._parse_git_log_output(result)
            return commits
            
        except Exception as e:
            logger.warning("Git commit retrieval failed",
                          project_path=project_path,
                          error=str(e))
            return []
    
    async def _execute_git_command(
        self,
        cmd: List[str],
        project_path: str,
        timeout: int = None
    ) -> str:
        """Execute a Git command safely."""
        try:
            timeout = timeout or self.config["analysis_timeout"]
            
            # Change to project directory and execute command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                
                if process.returncode != 0:
                    logger.warning("Git command failed",
                                  command=" ".join(cmd),
                                  stderr=stderr.decode())
                    return ""
                
                return stdout.decode()
                
            except asyncio.TimeoutError:
                process.kill()
                logger.warning("Git command timed out",
                              command=" ".join(cmd),
                              timeout=timeout)
                return ""
            
        except Exception as e:
            logger.warning("Git command execution failed",
                          command=" ".join(cmd),
                          error=str(e))
            return ""
    
    def _parse_git_log_output(
        self,
        output: str,
        target_file: Optional[str] = None
    ) -> List[GitCommit]:
        """Parse git log output into GitCommit objects."""
        commits = []
        
        try:
            lines = output.strip().split('\n')
            current_commit = None
            
            for line in lines:
                if not line.strip():
                    continue
                
                # Check if this is a commit header line
                if '|' in line and len(line.split('|')) == 5:
                    # Save previous commit if exists
                    if current_commit:
                        commits.append(current_commit)
                    
                    # Parse commit header
                    parts = line.split('|')
                    commit_hash = parts[0]
                    author = parts[1]
                    author_email = parts[2]
                    date_str = parts[3]
                    message = parts[4]
                    
                    # Parse date
                    try:
                        commit_date = datetime.fromisoformat(date_str.replace(' ', 'T'))
                    except:
                        commit_date = datetime.utcnow()
                    
                    current_commit = GitCommit(
                        commit_hash=commit_hash,
                        author=author,
                        author_email=author_email,
                        date=commit_date,
                        message=message,
                        files_changed=[],
                        additions=0,
                        deletions=0
                    )
                
                # Check if this is a file change line (numstat format)
                elif current_commit and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        try:
                            additions = int(parts[0]) if parts[0] != '-' else 0
                            deletions = int(parts[1]) if parts[1] != '-' else 0
                            file_path = parts[2]
                            
                            current_commit.files_changed.append(file_path)
                            current_commit.additions += additions
                            current_commit.deletions += deletions
                            
                        except ValueError:
                            # Skip malformed lines
                            pass
            
            # Don't forget the last commit
            if current_commit:
                commits.append(current_commit)
            
            return commits
            
        except Exception as e:
            logger.warning("Git log parsing failed", error=str(e))
            return []
    
    # Utility methods
    
    def _scope_to_days(self, scope: AnalysisScope) -> Optional[int]:
        """Convert analysis scope to number of days."""
        scope_map = {
            AnalysisScope.LAST_WEEK: 7,
            AnalysisScope.LAST_MONTH: 30,
            AnalysisScope.LAST_QUARTER: 90,
            AnalysisScope.LAST_YEAR: 365,
            AnalysisScope.ALL_TIME: None
        }
        return scope_map.get(scope)
    
    def _is_bug_fix_commit(self, message: str) -> bool:
        """Check if commit message indicates a bug fix."""
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.config["bug_keywords"])
    
    def _is_feature_commit(self, message: str) -> bool:
        """Check if commit message indicates a feature."""
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.config["feature_keywords"])
    
    def _is_refactor_commit(self, message: str) -> bool:
        """Check if commit message indicates refactoring."""
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.config["refactor_keywords"])
    
    async def _calculate_stability_score(self, commits: List[GitCommit]) -> float:
        """Calculate stability score based on commit patterns."""
        if len(commits) < 2:
            return 0.5
        
        # Calculate time intervals between commits
        intervals = []
        sorted_commits = sorted(commits, key=lambda c: c.date)
        
        for i in range(1, len(sorted_commits)):
            interval = (sorted_commits[i].date - sorted_commits[i-1].date).days
            intervals.append(interval)
        
        if not intervals:
            return 0.5
        
        # Calculate coefficient of variation (lower = more stable)
        mean_interval = sum(intervals) / len(intervals)
        if mean_interval == 0:
            return 0.5
        
        variance = sum((interval - mean_interval) ** 2 for interval in intervals) / len(intervals)
        std_dev = variance ** 0.5
        cv = std_dev / mean_interval
        
        # Convert to stability score (lower CV = higher stability)
        stability_score = max(0.0, 1.0 - min(1.0, cv / 2.0))
        
        return stability_score
    
    def _calculate_collaboration_score(self, author_commits: Dict[str, int]) -> float:
        """Calculate collaboration score using Gini coefficient."""
        if len(author_commits) <= 1:
            return 0.0
        
        # Calculate Gini coefficient
        commits = list(author_commits.values())
        commits.sort()
        n = len(commits)
        
        if n == 0 or sum(commits) == 0:
            return 0.0
        
        cumulative_sum = 0
        for i, value in enumerate(commits):
            cumulative_sum += value * (n - i)
        
        gini = (2 * cumulative_sum) / (n * sum(commits)) - (n + 1) / n
        
        # Convert Gini coefficient to collaboration score
        # Lower Gini = more equal distribution = better collaboration
        collaboration_score = 1.0 - gini
        
        return max(0.0, min(1.0, collaboration_score))
    
    def _create_empty_file_history(self, file_path: str) -> FileHistory:
        """Create empty file history for files with no Git data."""
        return FileHistory(
            file_path=file_path,
            creation_date=None,
            last_modified=None,
            total_commits=0,
            total_authors=0,
            total_additions=0,
            total_deletions=0,
            change_frequency=0.0,
            stability_score=0.5,
            bug_fix_count=0,
            feature_commit_count=0,
            refactor_commit_count=0,
            authors=[],
            recent_commits=[],
            metadata={"empty_history": True}
        )
    
    def _create_empty_collaboration_metrics(self, file_path: str) -> TeamCollaborationMetrics:
        """Create empty collaboration metrics."""
        return TeamCollaborationMetrics(
            file_path=file_path,
            primary_author="unknown",
            primary_author_percentage=0.0,
            author_count=0,
            collaboration_score=0.0,
            knowledge_distribution={},
            bus_factor=0,
            expertise_level="unknown",
            metadata={"empty_metrics": True}
        )
    
    def _create_default_analysis_result(self, file_path: str) -> HistoricalAnalysisResult:
        """Create default analysis result for error cases."""
        return HistoricalAnalysisResult(
            file_path=file_path,
            file_history=self._create_empty_file_history(file_path),
            change_patterns=[],
            collaboration_metrics=self._create_empty_collaboration_metrics(file_path),
            relevance_indicators={"risk_level": 0.5},
            risk_assessment={"unknown_risk": 0.5},
            recommendations=["Historical analysis unavailable - manual review recommended"],
            analysis_timestamp=datetime.utcnow(),
            metadata={"analysis_failed": True}
        )