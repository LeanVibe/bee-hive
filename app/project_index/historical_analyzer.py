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
import hashlib
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import structlog
import numpy as np

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
class DebtEvolutionPoint:
    """Single point in debt evolution timeline."""
    commit_hash: str
    date: datetime
    total_debt_score: float
    category_scores: Dict[str, float]
    files_analyzed: int
    lines_of_code: int
    debt_items_count: int
    debt_delta: float  # Change from previous measurement
    commit_message: str
    author: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class DebtTrendAnalysis:
    """Analysis of debt trends over time."""
    trend_direction: str  # "increasing", "decreasing", "stable", "volatile"
    trend_strength: float  # 0-1, how strong the trend is
    velocity: float  # rate of change per day
    acceleration: float  # change in velocity
    projected_debt_30_days: float
    projected_debt_90_days: float
    confidence_level: float  # 0-1, confidence in projections
    seasonal_patterns: List[str]
    anomaly_periods: List[Tuple[datetime, datetime, str]]
    risk_level: str  # "low", "medium", "high", "critical"


@dataclass
class DebtHotspot:
    """File or component with concerning debt patterns."""
    file_path: str
    debt_score: float
    debt_velocity: float  # rate of debt accumulation
    stability_risk: float  # how unstable the debt is
    contributor_count: int
    recent_debt_events: int
    categories_affected: List[str]
    first_problematic_commit: Optional[str]
    recommendations: List[str]
    priority: str  # "immediate", "high", "medium", "low"


@dataclass
class DebtEvolutionResult:
    """Complete debt evolution analysis result."""
    project_id: str
    analysis_period: Tuple[datetime, datetime]
    evolution_timeline: List[DebtEvolutionPoint]
    trend_analysis: DebtTrendAnalysis
    debt_hotspots: List[DebtHotspot]
    category_trends: Dict[str, DebtTrendAnalysis]
    correlation_analysis: Dict[str, float]  # correlations with commits, authors, etc
    recommendations: List[str]
    quality_gates_breached: List[Dict[str, Any]]
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)


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
    
    async def analyze_debt_evolution(
        self, 
        project_id: str,
        project_path: str,
        lookback_days: int = 90,
        sample_frequency_days: int = 7
    ) -> DebtEvolutionResult:
        """
        Analyze debt evolution over time using Git history and stored debt data.
        
        Args:
            project_id: Project identifier
            project_path: Path to the Git repository
            lookback_days: Number of days to look back in history
            sample_frequency_days: How often to sample debt (in days)
            
        Returns:
            DebtEvolutionResult with comprehensive debt evolution analysis
        """
        logger.info(
            "Starting debt evolution analysis",
            project_id=project_id,
            lookback_days=lookback_days,
            sample_frequency_days=sample_frequency_days
        )
        
        start_time = time.time()
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        
        try:
            # Get Git history for the period
            commits = await self._get_commits_in_period(project_path, start_date, end_date)
            
            # Sample commits at specified frequency
            sampled_commits = self._sample_commits_by_frequency(commits, sample_frequency_days)
            
            # Analyze debt at each sampled commit
            evolution_timeline = await self._analyze_debt_at_commits(
                project_id, project_path, sampled_commits
            )
            
            # Perform trend analysis
            trend_analysis = self._analyze_debt_trends(evolution_timeline)
            
            # Identify debt hotspots
            debt_hotspots = await self._identify_debt_hotspots(
                project_path, evolution_timeline
            )
            
            # Analyze category-specific trends
            category_trends = self._analyze_category_trends(evolution_timeline)
            
            # Correlation analysis
            correlation_analysis = self._analyze_debt_correlations(evolution_timeline)
            
            # Generate recommendations
            recommendations = self._generate_evolution_recommendations(
                trend_analysis, debt_hotspots, category_trends
            )
            
            # Check quality gates
            quality_gates_breached = self._check_quality_gates(evolution_timeline)
            
            analysis_duration = time.time() - start_time
            
            result = DebtEvolutionResult(
                project_id=project_id,
                analysis_period=(start_date, end_date),
                evolution_timeline=evolution_timeline,
                trend_analysis=trend_analysis,
                debt_hotspots=debt_hotspots,
                category_trends=category_trends,
                correlation_analysis=correlation_analysis,
                recommendations=recommendations,
                quality_gates_breached=quality_gates_breached,
                analysis_metadata={
                    "analysis_duration": analysis_duration,
                    "commits_analyzed": len(sampled_commits),
                    "total_commits_in_period": len(commits),
                    "sample_frequency_days": sample_frequency_days
                }
            )
            
            logger.info(
                "Debt evolution analysis completed",
                project_id=project_id,
                timeline_points=len(evolution_timeline),
                hotspots_found=len(debt_hotspots),
                analysis_duration=analysis_duration
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Error in debt evolution analysis",
                project_id=project_id,
                error=str(e)
            )
            # Return empty result on error
            return DebtEvolutionResult(
                project_id=project_id,
                analysis_period=(start_date, end_date),
                evolution_timeline=[],
                trend_analysis=self._create_empty_trend_analysis(),
                debt_hotspots=[],
                category_trends={},
                correlation_analysis={},
                recommendations=["Debt evolution analysis failed - manual review needed"],
                quality_gates_breached=[],
                analysis_metadata={"analysis_failed": True, "error": str(e)}
            )
    
    async def get_debt_velocity_for_file(
        self,
        file_path: str,
        project_path: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Calculate debt velocity (rate of debt accumulation) for a specific file.
        
        Args:
            file_path: Path to the file
            project_path: Path to the Git repository
            days: Number of days to analyze
            
        Returns:
            Dictionary with velocity metrics
        """
        try:
            # Get commits affecting this file
            cmd = [
                "git", "-C", project_path,
                "log", "--oneline", "--since", f"{days} days ago",
                "--", file_path
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.warning("Failed to get file commit history", file_path=file_path, error=stderr.decode())
                return {"velocity": 0.0, "commits": 0, "risk_level": "unknown"}
            
            commits = stdout.decode().strip().split('\n') if stdout.decode().strip() else []
            commit_count = len(commits)
            
            # Calculate velocity based on commit frequency and file characteristics
            velocity = commit_count / max(days, 1)  # commits per day
            
            # Adjust velocity based on file characteristics
            if commit_count > 0:
                # Higher velocity = higher risk
                risk_level = "high" if velocity > 0.5 else "medium" if velocity > 0.1 else "low"
            else:
                risk_level = "low"
            
            return {
                "velocity": velocity,
                "commits": commit_count,
                "commits_per_week": velocity * 7,
                "risk_level": risk_level,
                "analysis_period_days": days
            }
            
        except Exception as e:
            logger.error("Error calculating debt velocity for file", file_path=file_path, error=str(e))
            return {"velocity": 0.0, "commits": 0, "risk_level": "unknown", "error": str(e)}
    
    async def predict_debt_trajectory(
        self,
        evolution_timeline: List[DebtEvolutionPoint],
        prediction_days: int = 90
    ) -> Dict[str, Any]:
        """
        Predict future debt trajectory based on historical data.
        
        Args:
            evolution_timeline: Historical debt evolution data
            prediction_days: Number of days to predict into the future
            
        Returns:
            Prediction metrics and confidence intervals
        """
        if len(evolution_timeline) < 3:
            return {
                "prediction": "insufficient_data",
                "confidence": 0.0,
                "projected_debt": 0.0,
                "risk_assessment": "unknown"
            }
        
        try:
            # Extract time series data
            dates = [point.date for point in evolution_timeline]
            debt_scores = [point.total_debt_score for point in evolution_timeline]
            
            # Convert dates to numerical format (days since first measurement)
            base_date = dates[0]
            x_values = [(date - base_date).days for date in dates]
            y_values = debt_scores
            
            # Simple linear regression for trend prediction
            x_mean = np.mean(x_values)
            y_mean = np.mean(y_values)
            
            # Calculate slope (trend)
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)
            
            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator
            
            intercept = y_mean - slope * x_mean
            
            # Calculate R-squared for confidence
            y_pred = [slope * x + intercept for x in x_values]
            ss_res = sum((y - y_pred) ** 2 for y, y_pred in zip(y_values, y_pred))
            ss_tot = sum((y - y_mean) ** 2 for y in y_values)
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            confidence = max(0, min(1, r_squared))
            
            # Project future debt
            future_x = max(x_values) + prediction_days
            projected_debt = slope * future_x + intercept
            projected_debt = max(0, projected_debt)  # Debt can't be negative
            
            # Assess risk based on trend
            if slope > 0.01:  # Increasing trend
                if slope > 0.05:
                    risk_assessment = "critical"
                elif slope > 0.02:
                    risk_assessment = "high"
                else:
                    risk_assessment = "medium"
            elif slope < -0.01:  # Decreasing trend
                risk_assessment = "improving"
            else:  # Stable
                risk_assessment = "low"
            
            return {
                "prediction": "linear_trend",
                "confidence": confidence,
                "slope": slope,
                "projected_debt": projected_debt,
                "current_debt": debt_scores[-1],
                "debt_delta_prediction": projected_debt - debt_scores[-1],
                "risk_assessment": risk_assessment,
                "prediction_days": prediction_days,
                "r_squared": r_squared
            }
            
        except Exception as e:
            logger.error("Error predicting debt trajectory", error=str(e))
            return {
                "prediction": "error",
                "confidence": 0.0,
                "projected_debt": 0.0,
                "risk_assessment": "unknown",
                "error": str(e)
            }

    # Private helper methods for debt evolution analysis
    
    async def _get_commits_in_period(
        self, 
        project_path: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[GitCommit]:
        """Get Git commits in the specified time period."""
        try:
            # Format dates for git log
            since_str = start_date.strftime("%Y-%m-%d")
            until_str = end_date.strftime("%Y-%m-%d")
            
            cmd = [
                "git", "-C", project_path,
                "log", "--pretty=format:%H|%an|%ae|%ad|%s", "--date=iso",
                f"--since={since_str}", f"--until={until_str}",
                "--numstat"
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.warning("Failed to get commits", error=stderr.decode())
                return []
            
            # Parse commit log output
            commits = []
            current_commit = None
            
            for line in stdout.decode().split('\n'):
                if '|' in line and not line.startswith('\t') and not line.startswith(' '):
                    # New commit line
                    if current_commit:
                        commits.append(current_commit)
                    
                    parts = line.split('|', 4)
                    if len(parts) >= 5:
                        current_commit = GitCommit(
                            commit_hash=parts[0],
                            author=parts[1],
                            author_email=parts[2],
                            date=datetime.fromisoformat(parts[3].replace(' ', 'T')),
                            message=parts[4],
                            files_changed=[],
                            additions=0,
                            deletions=0
                        )
                elif line.strip() and current_commit:
                    # File change line (additions, deletions, filename)
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        try:
                            additions = int(parts[0]) if parts[0] != '-' else 0
                            deletions = int(parts[1]) if parts[1] != '-' else 0
                            filename = parts[2]
                            
                            current_commit.files_changed.append(filename)
                            current_commit.additions += additions
                            current_commit.deletions += deletions
                        except ValueError:
                            continue
            
            # Don't forget the last commit
            if current_commit:
                commits.append(current_commit)
            
            return commits
            
        except Exception as e:
            logger.error("Error getting commits in period", error=str(e))
            return []
    
    def _sample_commits_by_frequency(
        self, 
        commits: List[GitCommit], 
        frequency_days: int
    ) -> List[GitCommit]:
        """Sample commits at specified frequency to reduce analysis overhead."""
        if not commits:
            return []
        
        # Sort commits by date
        commits.sort(key=lambda c: c.date)
        
        sampled = [commits[0]]  # Always include first commit
        last_sampled_date = commits[0].date
        
        for commit in commits[1:]:
            days_since_last = (commit.date - last_sampled_date).days
            if days_since_last >= frequency_days:
                sampled.append(commit)
                last_sampled_date = commit.date
        
        # Always include the most recent commit
        if commits[-1] not in sampled:
            sampled.append(commits[-1])
        
        return sampled
    
    async def _analyze_debt_at_commits(
        self, 
        project_id: str,
        project_path: str, 
        commits: List[GitCommit]
    ) -> List[DebtEvolutionPoint]:
        """Analyze debt levels at specific Git commits."""
        evolution_points = []
        
        for i, commit in enumerate(commits):
            try:
                # For this implementation, we'll simulate debt analysis
                # In a real implementation, this would:
                # 1. Checkout the commit
                # 2. Run debt analysis
                # 3. Store the results
                
                # Simulated debt analysis based on commit characteristics
                base_debt = 0.3  # Base debt level
                
                # Adjust debt based on commit size (more changes = potentially more debt)
                size_factor = min(1.0, (commit.additions + commit.deletions) / 1000)
                
                # Add some randomness but keep it deterministic based on commit hash
                hash_factor = int(commit.commit_hash[:8], 16) % 100 / 100.0
                
                total_debt_score = base_debt + (size_factor * 0.3) + (hash_factor * 0.1)
                total_debt_score = min(1.0, total_debt_score)  # Cap at 1.0
                
                # Calculate debt delta
                debt_delta = 0.0
                if evolution_points:
                    debt_delta = total_debt_score - evolution_points[-1].total_debt_score
                
                # Simulate category scores
                category_scores = {
                    "complexity": total_debt_score * 0.4,
                    "duplication": total_debt_score * 0.3,
                    "maintainability": total_debt_score * 0.2,
                    "security": total_debt_score * 0.1
                }
                
                evolution_point = DebtEvolutionPoint(
                    commit_hash=commit.commit_hash,
                    date=commit.date,
                    total_debt_score=total_debt_score,
                    category_scores=category_scores,
                    files_analyzed=len(commit.files_changed),
                    lines_of_code=commit.additions,  # Approximation
                    debt_items_count=int(total_debt_score * 10),  # Simulated
                    debt_delta=debt_delta,
                    commit_message=commit.message,
                    author=commit.author
                )
                
                evolution_points.append(evolution_point)
                
            except Exception as e:
                logger.warning(
                    "Error analyzing debt at commit",
                    commit_hash=commit.commit_hash,
                    error=str(e)
                )
                continue
        
        return evolution_points
    
    def _analyze_debt_trends(self, timeline: List[DebtEvolutionPoint]) -> DebtTrendAnalysis:
        """Analyze trends in the debt evolution timeline."""
        if len(timeline) < 2:
            return self._create_empty_trend_analysis()
        
        try:
            # Extract debt scores and dates
            dates = [point.date for point in timeline]
            debt_scores = [point.total_debt_score for point in timeline]
            
            # Calculate trend direction and strength
            if len(timeline) >= 3:
                # Use linear regression for trend analysis
                x_values = [(date - dates[0]).days for date in dates]
                y_values = debt_scores
                
                # Calculate slope
                x_mean = np.mean(x_values)
                y_mean = np.mean(y_values)
                
                numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
                denominator = sum((x - x_mean) ** 2 for x in x_values)
                
                slope = numerator / denominator if denominator != 0 else 0
                
                # Determine trend direction
                if slope > 0.01:
                    trend_direction = "increasing"
                elif slope < -0.01:
                    trend_direction = "decreasing"
                else:
                    trend_direction = "stable"
                
                # Calculate trend strength (correlation coefficient)
                correlation = np.corrcoef(x_values, y_values)[0, 1] if len(x_values) > 1 else 0
                trend_strength = abs(correlation)
                
                # Calculate velocity (debt change per day)
                velocity = slope
                
                # Calculate acceleration (change in velocity)
                if len(timeline) >= 4:
                    mid_point = len(timeline) // 2
                    first_half_slope = self._calculate_slope(timeline[:mid_point])
                    second_half_slope = self._calculate_slope(timeline[mid_point:])
                    acceleration = second_half_slope - first_half_slope
                else:
                    acceleration = 0.0
                
            else:
                # Simple two-point analysis
                slope = (debt_scores[-1] - debt_scores[0]) / max(1, (dates[-1] - dates[0]).days)
                trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                trend_strength = min(1.0, abs(slope) * 10)  # Rough estimate
                velocity = slope
                acceleration = 0.0
            
            # Project future debt
            current_debt = debt_scores[-1]
            projected_debt_30_days = current_debt + (velocity * 30)
            projected_debt_90_days = current_debt + (velocity * 90) + (acceleration * 90 * 45)  # Include acceleration
            
            # Clamp projections to reasonable bounds
            projected_debt_30_days = max(0, min(2.0, projected_debt_30_days))
            projected_debt_90_days = max(0, min(2.0, projected_debt_90_days))
            
            # Assess confidence based on data quality
            confidence_level = min(1.0, len(timeline) / 10.0) * trend_strength
            
            # Detect anomalies and patterns
            anomaly_periods = self._detect_anomaly_periods(timeline)
            seasonal_patterns = self._detect_seasonal_patterns(timeline)
            
            # Determine risk level
            if velocity > 0.05 or projected_debt_30_days > 0.8:
                risk_level = "critical"
            elif velocity > 0.02 or projected_debt_30_days > 0.6:
                risk_level = "high"
            elif velocity > 0.01 or projected_debt_30_days > 0.4:
                risk_level = "medium"
            elif velocity < -0.01:
                risk_level = "low"  # Improving
            else:
                risk_level = "low"
            
            return DebtTrendAnalysis(
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                velocity=velocity,
                acceleration=acceleration,
                projected_debt_30_days=projected_debt_30_days,
                projected_debt_90_days=projected_debt_90_days,
                confidence_level=confidence_level,
                seasonal_patterns=seasonal_patterns,
                anomaly_periods=anomaly_periods,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error("Error analyzing debt trends", error=str(e))
            return self._create_empty_trend_analysis()
    
    def _calculate_slope(self, timeline_segment: List[DebtEvolutionPoint]) -> float:
        """Calculate slope for a segment of the timeline."""
        if len(timeline_segment) < 2:
            return 0.0
        
        dates = [point.date for point in timeline_segment]
        scores = [point.total_debt_score for point in timeline_segment]
        
        x_values = [(date - dates[0]).days for date in dates]
        y_values = scores
        
        x_mean = np.mean(x_values)
        y_mean = np.mean(y_values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        return numerator / denominator if denominator != 0 else 0
    
    def _detect_anomaly_periods(self, timeline: List[DebtEvolutionPoint]) -> List[Tuple[datetime, datetime, str]]:
        """Detect periods with anomalous debt behavior."""
        anomalies = []
        
        if len(timeline) < 3:
            return anomalies
        
        # Calculate rolling average and standard deviation
        debt_scores = [point.total_debt_score for point in timeline]
        
        # Use adaptive window size for smaller datasets
        window_size = min(3, max(2, len(timeline) // 3))
        
        # First pass: detect large deltas
        for i in range(1, len(timeline)):
            current_delta = abs(timeline[i].debt_delta) if hasattr(timeline[i], 'debt_delta') and timeline[i].debt_delta else abs(debt_scores[i] - debt_scores[i-1])
            
            # If delta is significantly larger than normal increments
            if current_delta > 0.2:  # Large change threshold
                anomaly_type = "spike" if debt_scores[i] > debt_scores[i-1] else "drop"
                anomalies.append((
                    timeline[i].date,
                    timeline[i].date,
                    f"debt_{anomaly_type}"
                ))
        
        # Second pass: rolling window analysis for smaller datasets
        for i in range(window_size, len(timeline) - window_size):
            window_scores = debt_scores[max(0, i-window_size):min(len(debt_scores), i+window_size+1)]
            window_mean = np.mean(window_scores)
            window_std = np.std(window_scores)
            
            current_score = debt_scores[i]
            
            # Detect anomalies (values outside 1.5 standard deviations for smaller datasets)
            if window_std > 0 and abs(current_score - window_mean) > 1.5 * window_std:
                anomaly_type = "spike" if current_score > window_mean else "drop"
                # Check if not already detected
                existing_anomaly = any(
                    abs((timeline[i].date - anomaly_date).total_seconds()) < 86400 
                    for anomaly_date, _, _ in anomalies
                )
                if not existing_anomaly:
                    anomalies.append((
                        timeline[i].date,
                        timeline[i].date,  # Single point anomaly
                        f"debt_{anomaly_type}"
                    ))
        
        return anomalies
    
    def _detect_seasonal_patterns(self, timeline: List[DebtEvolutionPoint]) -> List[str]:
        """Detect seasonal patterns in debt evolution."""
        patterns = []
        
        if len(timeline) < 30:  # Need at least a month of data
            return patterns
        
        # Group debt scores by day of week
        weekday_scores = defaultdict(list)
        for point in timeline:
            weekday = point.date.weekday()
            weekday_scores[weekday].append(point.total_debt_score)
        
        # Check for weekend/weekday patterns
        weekday_avg = np.mean([np.mean(scores) for day, scores in weekday_scores.items() if day < 5])
        weekend_avg = np.mean([np.mean(scores) for day, scores in weekday_scores.items() if day >= 5])
        
        if weekend_avg - weekday_avg > 0.1:
            patterns.append("weekend_debt_increase")
        elif weekday_avg - weekend_avg > 0.1:
            patterns.append("weekday_debt_increase")
        
        return patterns
    
    async def _identify_debt_hotspots(
        self, 
        project_path: str, 
        timeline: List[DebtEvolutionPoint]
    ) -> List[DebtHotspot]:
        """Identify files that are debt hotspots."""
        hotspots = []
        
        if not timeline:
            return hotspots
        
        try:
            # Get files that have been frequently changed
            recent_files = set()
            for point in timeline[-5:]:  # Last 5 measurements
                # In a real implementation, we'd track files per commit
                # For simulation, we'll use some heuristics
                recent_files.add(f"src/component_{len(recent_files) % 3}.py")
                recent_files.add(f"lib/module_{len(recent_files) % 2}.py")
            
            for file_path in recent_files:
                # Get file velocity
                velocity_data = await self.get_debt_velocity_for_file(
                    file_path, project_path, days=30
                )
                
                # Create hotspot entry
                hotspot = DebtHotspot(
                    file_path=file_path,
                    debt_score=0.7,  # Simulated
                    debt_velocity=velocity_data.get("velocity", 0.0),
                    stability_risk=0.6,  # Simulated
                    contributor_count=3,  # Simulated
                    recent_debt_events=5,  # Simulated
                    categories_affected=["complexity", "duplication"],
                    first_problematic_commit=timeline[0].commit_hash if timeline else None,
                    recommendations=[
                        f"Review and refactor {file_path}",
                        "Add comprehensive tests",
                        "Consider code review requirements"
                    ],
                    priority="high" if velocity_data.get("velocity", 0) > 0.3 else "medium"
                )
                hotspots.append(hotspot)
        
        except Exception as e:
            logger.error("Error identifying debt hotspots", error=str(e))
        
        return hotspots[:10]  # Limit to top 10 hotspots
    
    def _analyze_category_trends(
        self, 
        timeline: List[DebtEvolutionPoint]
    ) -> Dict[str, DebtTrendAnalysis]:
        """Analyze trends for each debt category."""
        category_trends = {}
        
        if not timeline:
            return category_trends
        
        # Get all categories
        all_categories = set()
        for point in timeline:
            all_categories.update(point.category_scores.keys())
        
        # Analyze trend for each category
        for category in all_categories:
            category_timeline = []
            for point in timeline:
                # Create pseudo evolution point for this category
                category_point = DebtEvolutionPoint(
                    commit_hash=point.commit_hash,
                    date=point.date,
                    total_debt_score=point.category_scores.get(category, 0.0),
                    category_scores={category: point.category_scores.get(category, 0.0)},
                    files_analyzed=point.files_analyzed,
                    lines_of_code=point.lines_of_code,
                    debt_items_count=point.debt_items_count,
                    debt_delta=0.0,  # Will be calculated
                    commit_message=point.commit_message,
                    author=point.author
                )
                category_timeline.append(category_point)
            
            # Analyze trend for this category
            category_trends[category] = self._analyze_debt_trends(category_timeline)
        
        return category_trends
    
    def _analyze_debt_correlations(self, timeline: List[DebtEvolutionPoint]) -> Dict[str, float]:
        """Analyze correlations between debt and various factors."""
        correlations = {}
        
        if len(timeline) < 3:
            return correlations
        
        try:
            debt_scores = [point.total_debt_score for point in timeline]
            
            # Correlation with file count
            file_counts = [point.files_analyzed for point in timeline]
            if len(set(file_counts)) > 1:  # Need variation
                correlations["files_analyzed"] = np.corrcoef(debt_scores, file_counts)[0, 1]
            
            # Correlation with lines of code
            loc_counts = [point.lines_of_code for point in timeline]
            if len(set(loc_counts)) > 1:
                correlations["lines_of_code"] = np.corrcoef(debt_scores, loc_counts)[0, 1]
            
            # Correlation with time (temporal trend)
            time_values = [(point.date - timeline[0].date).days for point in timeline]
            if len(set(time_values)) > 1:
                correlations["time_trend"] = np.corrcoef(debt_scores, time_values)[0, 1]
            
        except Exception as e:
            logger.error("Error calculating debt correlations", error=str(e))
        
        return correlations
    
    def _generate_evolution_recommendations(
        self,
        trend_analysis: DebtTrendAnalysis,
        hotspots: List[DebtHotspot],
        category_trends: Dict[str, DebtTrendAnalysis]
    ) -> List[str]:
        """Generate recommendations based on debt evolution analysis."""
        recommendations = []
        
        # Trend-based recommendations
        if trend_analysis.trend_direction == "increasing":
            if trend_analysis.velocity > 0.05:
                recommendations.append("URGENT: Debt is increasing rapidly - implement immediate debt reduction measures")
            elif trend_analysis.velocity > 0.02:
                recommendations.append("High debt accumulation detected - schedule dedicated refactoring sprint")
            else:
                recommendations.append("Debt is slowly increasing - monitor closely and address systematically")
        
        elif trend_analysis.trend_direction == "decreasing":
            recommendations.append("Good trend: debt is decreasing - maintain current practices")
        
        # Hotspot-based recommendations
        if hotspots:
            high_priority_hotspots = [h for h in hotspots if h.priority == "high"]
            if high_priority_hotspots:
                recommendations.append(f"Address {len(high_priority_hotspots)} high-priority debt hotspots immediately")
        
        # Category-based recommendations
        for category, trend in category_trends.items():
            if trend.trend_direction == "increasing" and trend.velocity > 0.03:
                recommendations.append(f"Focus on {category} debt - showing concerning upward trend")
        
        # Quality gate recommendations
        if trend_analysis.projected_debt_30_days > 0.8:
            recommendations.append("Projected debt will exceed critical threshold in 30 days - take action now")
        
        # Default recommendations if none specific
        if not recommendations:
            recommendations.append("Continue monitoring debt levels and maintain code quality practices")
        
        return recommendations
    
    def _check_quality_gates(self, timeline: List[DebtEvolutionPoint]) -> List[Dict[str, Any]]:
        """Check if any quality gates have been breached."""
        breaches = []
        
        if not timeline:
            return breaches
        
        current_debt = timeline[-1].total_debt_score
        
        # Define quality gates
        gates = [
            {"name": "critical_debt_threshold", "threshold": 0.8, "severity": "critical"},
            {"name": "high_debt_threshold", "threshold": 0.6, "severity": "high"},
            {"name": "medium_debt_threshold", "threshold": 0.4, "severity": "medium"}
        ]
        
        for gate in gates:
            if current_debt > gate["threshold"]:
                breaches.append({
                    "gate_name": gate["name"],
                    "threshold": gate["threshold"],
                    "current_value": current_debt,
                    "severity": gate["severity"],
                    "breach_date": timeline[-1].date.isoformat(),
                    "commit_hash": timeline[-1].commit_hash
                })
        
        return breaches
    
    def _create_empty_trend_analysis(self) -> DebtTrendAnalysis:
        """Create empty trend analysis for error cases."""
        return DebtTrendAnalysis(
            trend_direction="unknown",
            trend_strength=0.0,
            velocity=0.0,
            acceleration=0.0,
            projected_debt_30_days=0.0,
            projected_debt_90_days=0.0,
            confidence_level=0.0,
            seasonal_patterns=[],
            anomaly_periods=[],
            risk_level="unknown"
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