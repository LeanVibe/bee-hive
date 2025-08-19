"""
Intelligent Debt Remediation Engine for LeanVibe Agent Hive 2.0

Advanced recommendation system that analyzes technical debt patterns and provides
actionable remediation strategies based on project context, historical patterns,
and cost-benefit analysis.
"""

import asyncio
import json
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from enum import Enum
import re

import structlog
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .debt_analyzer import DebtAnalysisResult, DebtItem, DebtCategory, DebtSeverity
from .advanced_debt_detector import AdvancedDebtPattern, AdvancedDebtDetector
from .historical_analyzer import HistoricalAnalyzer, DebtEvolutionResult, DebtTrendAnalysis
from ..models.project_index import ProjectIndex, FileEntry
from ..core.database import get_session

logger = structlog.get_logger()


class RemediationStrategy(Enum):
    """Types of debt remediation strategies."""
    REFACTOR = "refactor"
    EXTRACT_METHOD = "extract_method"
    EXTRACT_CLASS = "extract_class"
    SIMPLIFY_CONDITION = "simplify_condition"
    REMOVE_DUPLICATION = "remove_duplication"
    IMPROVE_NAMING = "improve_naming"
    ADD_DOCUMENTATION = "add_documentation"
    SPLIT_FILE = "split_file"
    MERGE_FILES = "merge_files"
    OPTIMIZE_IMPORTS = "optimize_imports"
    REMOVE_DEAD_CODE = "remove_dead_code"
    EXTRACT_CONFIGURATION = "extract_configuration"
    APPLY_DESIGN_PATTERN = "apply_design_pattern"
    ARCHITECTURAL_CHANGE = "architectural_change"


class RemediationPriority(Enum):
    """Priority levels for remediation recommendations."""
    IMMEDIATE = "immediate"      # Critical, must fix now
    HIGH = "high"               # Should fix in current sprint
    MEDIUM = "medium"           # Should fix in next 2-3 sprints
    LOW = "low"                 # Nice to have, technical improvement
    DEFERRED = "deferred"       # Can be postponed indefinitely


class RemediationImpact(Enum):
    """Expected impact levels of remediation."""
    CRITICAL = "critical"       # Major architectural improvement
    SIGNIFICANT = "significant" # Notable code quality improvement
    MODERATE = "moderate"       # Good improvement with reasonable effort
    MINOR = "minor"            # Small improvement, easy to implement


@dataclass
class RemediationRecommendation:
    """A specific debt remediation recommendation."""
    id: str
    strategy: RemediationStrategy
    priority: RemediationPriority
    impact: RemediationImpact
    title: str
    description: str
    rationale: str
    
    # Target information
    file_path: str
    line_ranges: List[Tuple[int, int]]  # [(start, end), ...]
    affected_functions: List[str]
    affected_classes: List[str]
    
    # Metrics
    debt_reduction_score: float      # 0.0-1.0, expected debt reduction
    implementation_effort: float     # 0.0-1.0, estimated effort required
    risk_level: float               # 0.0-1.0, risk of breaking changes
    cost_benefit_ratio: float       # Higher is better
    
    # Implementation details
    suggested_approach: str
    code_examples: List[str]
    related_patterns: List[str]
    dependencies: List[str]         # Other recommendations this depends on
    
    # Context
    debt_categories: List[DebtCategory]
    historical_context: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RemediationPlan:
    """Comprehensive remediation plan for a project or file."""
    project_id: str
    scope: str                      # "project", "file", "directory"
    target_path: str
    
    recommendations: List[RemediationRecommendation]
    execution_phases: List[List[str]]  # Phases with recommendation IDs
    
    # Plan metrics
    total_debt_reduction: float
    total_effort_estimate: float
    total_risk_score: float
    estimated_duration_days: int
    
    # Prioritization
    immediate_actions: List[str]    # Recommendation IDs
    quick_wins: List[str]          # High impact, low effort
    long_term_goals: List[str]     # Strategic improvements
    
    # Context and validation
    plan_rationale: str
    success_criteria: List[str]
    potential_blockers: List[str]
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RemediationTemplate:
    """Template for common remediation patterns."""
    name: str
    strategy: RemediationStrategy
    pattern_regex: str
    applicability_check: str       # Python code to evaluate applicability
    template_code: str
    explanation: str
    effort_multiplier: float = 1.0


class DebtRemediationEngine:
    """
    Intelligent technical debt remediation engine.
    
    Analyzes debt patterns, project context, and historical data to generate
    actionable remediation recommendations with cost-benefit analysis.
    """
    
    def __init__(
        self,
        debt_analyzer: Optional['TechnicalDebtAnalyzer'] = None,
        advanced_detector: Optional[AdvancedDebtDetector] = None,
        historical_analyzer: Optional[HistoricalAnalyzer] = None
    ):
        """Initialize debt remediation engine."""
        self.debt_analyzer = debt_analyzer
        self.advanced_detector = advanced_detector
        self.historical_analyzer = historical_analyzer
        
        # Load remediation templates and patterns
        self.remediation_templates = self._load_remediation_templates()
        self.pattern_matchers = self._initialize_pattern_matchers()
        
        # Configuration
        self.config = {
            'max_recommendations_per_file': 10,
            'min_debt_threshold': 0.1,
            'effort_estimation_factors': {
                'lines_of_code': 0.001,
                'complexity': 0.1,
                'dependencies': 0.05,
                'test_coverage': -0.02  # Better coverage reduces effort
            },
            'priority_weights': {
                'debt_score': 0.4,
                'historical_trend': 0.3,
                'code_churn': 0.2,
                'business_impact': 0.1
            }
        }
    
    async def generate_remediation_plan(
        self,
        project: ProjectIndex,
        scope: str = "project",
        target_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RemediationPlan:
        """Generate comprehensive remediation plan for project or specific scope."""
        logger.info(
            "Generating debt remediation plan",
            project_id=str(project.id),
            scope=scope,
            target_path=target_path
        )
        
        start_time = datetime.utcnow()
        
        try:
            # Analyze current debt state
            debt_analysis = await self._analyze_current_debt(project, target_path)
            
            # Get historical context
            historical_analysis = await self._get_historical_context(project)
            
            # Generate individual recommendations
            recommendations = await self._generate_recommendations(
                project, debt_analysis, historical_analysis, target_path, context
            )
            
            # Prioritize and organize recommendations
            prioritized_recommendations = await self._prioritize_recommendations(
                recommendations, historical_analysis
            )
            
            # Create execution phases
            execution_phases = await self._create_execution_phases(prioritized_recommendations)
            
            # Calculate plan metrics
            plan_metrics = await self._calculate_plan_metrics(prioritized_recommendations)
            
            # Identify key recommendation categories
            immediate_actions = [
                r.id for r in prioritized_recommendations 
                if r.priority == RemediationPriority.IMMEDIATE
            ]
            
            quick_wins = [
                r.id for r in prioritized_recommendations
                if r.cost_benefit_ratio > 2.0 and r.implementation_effort < 0.3
            ]
            
            long_term_goals = [
                r.id for r in prioritized_recommendations
                if r.impact in [RemediationImpact.CRITICAL, RemediationImpact.SIGNIFICANT]
                and r.implementation_effort > 0.7
            ]
            
            # Generate plan rationale
            plan_rationale = await self._generate_plan_rationale(
                debt_analysis, historical_analysis, prioritized_recommendations
            )
            
            # Create remediation plan
            plan = RemediationPlan(
                project_id=str(project.id),
                scope=scope,
                target_path=target_path or project.root_path,
                recommendations=prioritized_recommendations,
                execution_phases=execution_phases,
                total_debt_reduction=plan_metrics['total_debt_reduction'],
                total_effort_estimate=plan_metrics['total_effort'],
                total_risk_score=plan_metrics['total_risk'],
                estimated_duration_days=plan_metrics['estimated_days'],
                immediate_actions=immediate_actions,
                quick_wins=quick_wins,
                long_term_goals=long_term_goals,
                plan_rationale=plan_rationale,
                success_criteria=await self._generate_success_criteria(prioritized_recommendations),
                potential_blockers=await self._identify_potential_blockers(prioritized_recommendations),
                metadata={
                    'generation_time_seconds': (datetime.utcnow() - start_time).total_seconds(),
                    'debt_analysis_summary': debt_analysis,
                    'historical_trends': historical_analysis.trend_analysis.__dict__ if historical_analysis else None
                }
            )
            
            logger.info(
                "Remediation plan generated successfully",
                project_id=str(project.id),
                recommendations_count=len(prioritized_recommendations),
                immediate_actions=len(immediate_actions),
                quick_wins=len(quick_wins),
                generation_time=plan.metadata['generation_time_seconds']
            )
            
            return plan
            
        except Exception as e:
            logger.error(
                "Error generating remediation plan",
                project_id=str(project.id),
                error=str(e)
            )
            raise
    
    async def get_file_specific_recommendations(
        self,
        project: ProjectIndex,
        file_path: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[RemediationRecommendation]:
        """Get specific recommendations for a single file."""
        logger.debug("Generating file-specific recommendations", file_path=file_path)
        
        # Find file entry
        file_entry = next(
            (fe for fe in project.file_entries if fe.file_path == file_path),
            None
        )
        
        if not file_entry:
            raise ValueError(f"File {file_path} not found in project")
        
        # Analyze file debt
        async with get_session() as session:
            if self.debt_analyzer:
                debt_items = await self.debt_analyzer._analyze_file_debt(file_entry, session)
            else:
                debt_items = []
        
        # Get file historical data
        historical_data = None
        if self.historical_analyzer:
            try:
                historical_data = await self.historical_analyzer.get_debt_velocity_for_file(
                    file_path=file_path,
                    project_path=project.root_path,
                    days=90
                )
            except Exception as e:
                logger.warning("Failed to get historical data for file", file_path=file_path, error=str(e))
        
        # Generate recommendations for this file
        recommendations = []
        
        for debt_item in debt_items:
            file_recommendations = await self._generate_item_recommendations(
                debt_item, file_entry, historical_data, context
            )
            recommendations.extend(file_recommendations)
        
        # Apply file-specific patterns
        pattern_recommendations = await self._apply_file_patterns(
            file_entry, debt_items, historical_data
        )
        recommendations.extend(pattern_recommendations)
        
        # Limit and prioritize
        prioritized = sorted(
            recommendations,
            key=lambda r: (r.priority.value, -r.cost_benefit_ratio, -r.debt_reduction_score)
        )
        
        return prioritized[:self.config['max_recommendations_per_file']]
    
    # Private implementation methods
    
    async def _analyze_current_debt(
        self, 
        project: ProjectIndex, 
        target_path: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze current debt state for the project or target path."""
        if not self.debt_analyzer:
            return {'total_debt_score': 0.0, 'debt_items': [], 'category_breakdown': {}}
        
        try:
            async with get_session() as session:
                # Get debt analysis for the project
                from sqlalchemy import select
                from sqlalchemy.orm import selectinload
                
                stmt = select(ProjectIndex).where(ProjectIndex.id == project.id).options(
                    selectinload(ProjectIndex.file_entries)
                )
                result = await session.execute(stmt)
                full_project = result.scalar_one_or_none()
                
                if not full_project:
                    return {'total_debt_score': 0.0, 'debt_items': [], 'category_breakdown': {}}
                
                # Filter files by target path if specified
                target_files = full_project.file_entries
                if target_path:
                    target_files = [
                        fe for fe in target_files 
                        if fe.file_path.startswith(target_path)
                    ]
                
                # Analyze debt for target files
                all_debt_items = []
                for file_entry in target_files:
                    if not file_entry.is_binary:
                        file_debt = await self.debt_analyzer._analyze_file_debt(file_entry, session)
                        all_debt_items.extend(file_debt)
                
                # Calculate summary statistics
                total_debt_score = sum(item.debt_score for item in all_debt_items)
                category_breakdown = defaultdict(float)
                severity_breakdown = defaultdict(int)
                
                for item in all_debt_items:
                    category_breakdown[item.category.value] += item.debt_score
                    severity_breakdown[item.severity.value] += 1
                
                return {
                    'total_debt_score': total_debt_score,
                    'debt_items': all_debt_items,
                    'category_breakdown': dict(category_breakdown),
                    'severity_breakdown': dict(severity_breakdown),
                    'files_analyzed': len(target_files),
                    'debt_per_file': total_debt_score / max(len(target_files), 1)
                }
                
        except Exception as e:
            logger.error("Error analyzing current debt", error=str(e))
            return {'total_debt_score': 0.0, 'debt_items': [], 'category_breakdown': {}}
    
    async def _get_historical_context(self, project: ProjectIndex) -> Optional[DebtEvolutionResult]:
        """Get historical debt evolution context."""
        if not self.historical_analyzer:
            return None
        
        try:
            return await self.historical_analyzer.analyze_debt_evolution(
                project_id=str(project.id),
                project_path=project.root_path,
                lookback_days=90
            )
        except Exception as e:
            logger.warning("Failed to get historical context", error=str(e))
            return None
    
    async def _generate_recommendations(
        self,
        project: ProjectIndex,
        debt_analysis: Dict[str, Any],
        historical_analysis: Optional[DebtEvolutionResult],
        target_path: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> List[RemediationRecommendation]:
        """Generate individual remediation recommendations."""
        recommendations = []
        debt_items = debt_analysis.get('debt_items', [])
        
        # Group debt items by file for better analysis
        items_by_file = defaultdict(list)
        for item in debt_items:
            file_path = item.location.get("file_path", "") if isinstance(item.location, dict) else str(item.location)
            items_by_file[file_path].append(item)
        
        # Generate recommendations for each file
        for file_path, file_debt_items in items_by_file.items():
            file_entry = next(
                (fe for fe in project.file_entries if fe.file_path == file_path),
                None
            )
            
            if file_entry:
                file_recommendations = await self._generate_file_recommendations(
                    file_entry, file_debt_items, historical_analysis, context
                )
                recommendations.extend(file_recommendations)
        
        return recommendations
    
    async def _generate_file_recommendations(
        self,
        file_entry: FileEntry,
        debt_items: List[DebtItem],
        historical_analysis: Optional[DebtEvolutionResult],
        context: Optional[Dict[str, Any]]
    ) -> List[RemediationRecommendation]:
        """Generate recommendations for a specific file."""
        recommendations = []
        
        # Analyze debt patterns in this file
        debt_by_category = defaultdict(list)
        for item in debt_items:
            debt_by_category[item.category].append(item)
        
        # Generate category-specific recommendations
        for category, category_items in debt_by_category.items():
            category_recommendations = await self._generate_category_recommendations(
                file_entry, category, category_items, historical_analysis
            )
            recommendations.extend(category_recommendations)
        
        # Apply cross-cutting patterns
        pattern_recommendations = await self._apply_file_patterns(
            file_entry, debt_items, historical_analysis
        )
        recommendations.extend(pattern_recommendations)
        
        return recommendations
    
    async def _generate_category_recommendations(
        self,
        file_entry: FileEntry,
        category: DebtCategory,
        items: List[DebtItem],
        historical_analysis: Optional[DebtEvolutionResult]
    ) -> List[RemediationRecommendation]:
        """Generate recommendations for a specific debt category."""
        recommendations = []
        
        # Calculate category severity and impact
        total_debt_score = sum(item.debt_score for item in items)
        max_severity = max((item.severity for item in items), default=DebtSeverity.LOW)
        
        # Generate recommendations based on category
        if category == DebtCategory.COMPLEXITY:
            recommendations.extend(
                await self._generate_complexity_recommendations(
                    file_entry, items, total_debt_score, max_severity
                )
            )
        elif category == DebtCategory.DUPLICATION:
            recommendations.extend(
                await self._generate_duplication_recommendations(
                    file_entry, items, total_debt_score, max_severity
                )
            )
        elif category == DebtCategory.CODE_SMELLS:
            recommendations.extend(
                await self._generate_code_smell_recommendations(
                    file_entry, items, total_debt_score, max_severity
                )
            )
        elif category == DebtCategory.ARCHITECTURE:
            recommendations.extend(
                await self._generate_architecture_recommendations(
                    file_entry, items, total_debt_score, max_severity
                )
            )
        elif category == DebtCategory.DOCUMENTATION:
            recommendations.extend(
                await self._generate_documentation_recommendations(
                    file_entry, items, total_debt_score, max_severity
                )
            )
        
        return recommendations
    
    async def _generate_complexity_recommendations(
        self,
        file_entry: FileEntry,
        items: List[DebtItem],
        total_debt_score: float,
        max_severity: DebtSeverity
    ) -> List[RemediationRecommendation]:
        """Generate recommendations for complexity debt."""
        recommendations = []
        
        # High complexity functions - extract method
        if total_debt_score > 0.3:
            rec_id = f"complexity_extract_method_{file_entry.id}"
            recommendations.append(RemediationRecommendation(
                id=rec_id,
                strategy=RemediationStrategy.EXTRACT_METHOD,
                priority=RemediationPriority.HIGH if max_severity == DebtSeverity.CRITICAL else RemediationPriority.MEDIUM,
                impact=RemediationImpact.SIGNIFICANT,
                title="Extract methods from complex functions",
                description=f"Break down complex functions in {file_entry.file_name} into smaller, more manageable methods",
                rationale=f"High complexity score ({total_debt_score:.2f}) indicates functions that are difficult to understand and maintain",
                file_path=file_entry.file_path,
                line_ranges=[(item.location.get("line_number", 1), item.location.get("line_number", 1) + 10) for item in items if isinstance(item.location, dict) and item.location.get("line_number")],
                affected_functions=[item.description.split()[0] for item in items if 'function' in item.description.lower()],
                affected_classes=[],
                debt_reduction_score=min(0.8, total_debt_score * 0.6),
                implementation_effort=0.4 + (total_debt_score * 0.2),
                risk_level=0.3,
                cost_benefit_ratio=2.5 if total_debt_score > 0.5 else 1.8,
                suggested_approach="Identify logical blocks within complex functions and extract them into separate methods with clear names and purposes",
                code_examples=[
                    "# Before: Complex function with high cyclomatic complexity",
                    "# After: Main function with extracted helper methods"
                ],
                related_patterns=["Single Responsibility Principle", "Extract Method Refactoring"],
                dependencies=[],
                debt_categories=[DebtCategory.COMPLEXITY],
                historical_context={}
            ))
        
        return recommendations
    
    async def _generate_duplication_recommendations(
        self,
        file_entry: FileEntry,
        items: List[DebtItem],
        total_debt_score: float,
        max_severity: DebtSeverity
    ) -> List[RemediationRecommendation]:
        """Generate recommendations for code duplication."""
        recommendations = []
        
        if total_debt_score > 0.2:
            rec_id = f"duplication_extract_common_{file_entry.id}"
            recommendations.append(RemediationRecommendation(
                id=rec_id,
                strategy=RemediationStrategy.REMOVE_DUPLICATION,
                priority=RemediationPriority.MEDIUM,
                impact=RemediationImpact.MODERATE,
                title="Remove code duplication",
                description=f"Extract common code patterns in {file_entry.file_name} into shared utilities",
                rationale=f"Duplication score ({total_debt_score:.2f}) indicates repeated code that should be consolidated",
                file_path=file_entry.file_path,
                line_ranges=[(item.location.get("line_number", 1), item.location.get("line_number", 1) + 5) for item in items if isinstance(item.location, dict) and item.location.get("line_number")],
                affected_functions=[],
                affected_classes=[],
                debt_reduction_score=total_debt_score * 0.7,
                implementation_effort=0.3 + (total_debt_score * 0.15),
                risk_level=0.2,
                cost_benefit_ratio=3.0,
                suggested_approach="Identify duplicated code blocks and extract them into common functions or utilities",
                code_examples=[
                    "# Extract duplicated validation logic into a shared function",
                    "# Replace repeated code with function calls"
                ],
                related_patterns=["DRY Principle", "Extract Function"],
                dependencies=[],
                debt_categories=[DebtCategory.CODE_DUPLICATION],
                historical_context={}
            ))
        
        return recommendations
    
    async def _generate_code_smell_recommendations(
        self,
        file_entry: FileEntry,
        items: List[DebtItem],
        total_debt_score: float,
        max_severity: DebtSeverity
    ) -> List[RemediationRecommendation]:
        """Generate recommendations for code smells."""
        recommendations = []
        
        # Analyze specific smells
        smell_descriptions = [item.description.lower() for item in items]
        
        # Long method smell
        if any('long method' in desc or 'long function' in desc for desc in smell_descriptions):
            rec_id = f"smell_long_method_{file_entry.id}"
            recommendations.append(RemediationRecommendation(
                id=rec_id,
                strategy=RemediationStrategy.EXTRACT_METHOD,
                priority=RemediationPriority.MEDIUM,
                impact=RemediationImpact.MODERATE,
                title="Shorten long methods",
                description=f"Break down long methods in {file_entry.file_name} for better readability",
                rationale="Long methods are harder to understand, test, and maintain",
                file_path=file_entry.file_path,
                line_ranges=[(item.location.get("line_number", 1), item.location.get("line_number", 1) + 20) for item in items if isinstance(item.location, dict) and item.location.get("line_number")],
                affected_functions=[],
                affected_classes=[],
                debt_reduction_score=0.5,
                implementation_effort=0.4,
                risk_level=0.25,
                cost_benefit_ratio=2.0,
                suggested_approach="Extract logical blocks into separate methods with descriptive names",
                code_examples=[],
                related_patterns=["Extract Method", "Compose Method"],
                dependencies=[],
                debt_categories=[DebtCategory.CODE_SMELLS],
                historical_context={}
            ))
        
        return recommendations
    
    async def _generate_architecture_recommendations(
        self,
        file_entry: FileEntry,
        items: List[DebtItem],
        total_debt_score: float,
        max_severity: DebtSeverity
    ) -> List[RemediationRecommendation]:
        """Generate recommendations for architectural debt."""
        recommendations = []
        
        if total_debt_score > 0.4:
            rec_id = f"architecture_refactor_{file_entry.id}"
            recommendations.append(RemediationRecommendation(
                id=rec_id,
                strategy=RemediationStrategy.ARCHITECTURAL_CHANGE,
                priority=RemediationPriority.LOW,
                impact=RemediationImpact.CRITICAL,
                title="Address architectural concerns",
                description=f"Refactor architectural issues in {file_entry.file_name}",
                rationale=f"High architectural debt ({total_debt_score:.2f}) may impact system maintainability",
                file_path=file_entry.file_path,
                line_ranges=[],
                affected_functions=[],
                affected_classes=[],
                debt_reduction_score=total_debt_score * 0.8,
                implementation_effort=0.8,
                risk_level=0.7,
                cost_benefit_ratio=1.5,
                suggested_approach="Review and refactor architectural dependencies and coupling",
                code_examples=[],
                related_patterns=["Dependency Injection", "Interface Segregation"],
                dependencies=[],
                debt_categories=[DebtCategory.ARCHITECTURE],
                historical_context={}
            ))
        
        return recommendations
    
    async def _generate_documentation_recommendations(
        self,
        file_entry: FileEntry,
        items: List[DebtItem],
        total_debt_score: float,
        max_severity: DebtSeverity
    ) -> List[RemediationRecommendation]:
        """Generate recommendations for documentation debt."""
        recommendations = []
        
        if total_debt_score > 0.1:
            rec_id = f"documentation_improve_{file_entry.id}"
            recommendations.append(RemediationRecommendation(
                id=rec_id,
                strategy=RemediationStrategy.ADD_DOCUMENTATION,
                priority=RemediationPriority.LOW,
                impact=RemediationImpact.MINOR,
                title="Improve documentation",
                description=f"Add missing documentation to {file_entry.file_name}",
                rationale="Proper documentation improves code maintainability and team collaboration",
                file_path=file_entry.file_path,
                line_ranges=[],
                affected_functions=[],
                affected_classes=[],
                debt_reduction_score=total_debt_score * 0.9,
                implementation_effort=0.2,
                risk_level=0.1,
                cost_benefit_ratio=3.5,
                suggested_approach="Add docstrings, comments, and type hints where missing",
                code_examples=[],
                related_patterns=["Self-Documenting Code"],
                dependencies=[],
                debt_categories=[DebtCategory.DOCUMENTATION],
                historical_context={}
            ))
        
        return recommendations
    
    async def _generate_item_recommendations(
        self,
        debt_item: DebtItem,
        file_entry: FileEntry,
        historical_data: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> List[RemediationRecommendation]:
        """Generate recommendations for a specific debt item."""
        # This method would be called from get_file_specific_recommendations
        # For now, delegate to category-specific methods
        return await self._generate_category_recommendations(
            file_entry, debt_item.category, [debt_item], None
        )
    
    async def _apply_file_patterns(
        self,
        file_entry: FileEntry,
        debt_items: List[DebtItem],
        historical_data: Any
    ) -> List[RemediationRecommendation]:
        """Apply file-level patterns for recommendations."""
        recommendations = []
        
        # File size pattern
        if file_entry.line_count and file_entry.line_count > 500:
            recommendations.append(RemediationRecommendation(
                id=f"pattern_large_file_{file_entry.id}",
                strategy=RemediationStrategy.SPLIT_FILE,
                priority=RemediationPriority.MEDIUM,
                impact=RemediationImpact.MODERATE,
                title="Split large file",
                description=f"Consider splitting {file_entry.file_name} ({file_entry.line_count} lines) into smaller modules",
                rationale="Large files are harder to navigate and maintain",
                file_path=file_entry.file_path,
                line_ranges=[],
                affected_functions=[],
                affected_classes=[],
                debt_reduction_score=0.3,
                implementation_effort=0.6,
                risk_level=0.4,
                cost_benefit_ratio=1.2,
                suggested_approach="Identify logical groupings and extract them into separate files",
                code_examples=[],
                related_patterns=["Single Responsibility Principle"],
                dependencies=[],
                debt_categories=[DebtCategory.ARCHITECTURE],
                historical_context={}
            ))
        
        return recommendations
    
    async def _prioritize_recommendations(
        self,
        recommendations: List[RemediationRecommendation],
        historical_analysis: Optional[DebtEvolutionResult]
    ) -> List[RemediationRecommendation]:
        """Prioritize recommendations based on various factors."""
        # Apply historical context to adjust priorities
        if historical_analysis:
            for rec in recommendations:
                # Increase priority for files with increasing debt trends
                trend = historical_analysis.trend_analysis
                if trend and trend.trend_direction == "increasing" and trend.trend_strength > 0.5:
                    if rec.priority == RemediationPriority.MEDIUM:
                        rec.priority = RemediationPriority.HIGH
                    elif rec.priority == RemediationPriority.LOW:
                        rec.priority = RemediationPriority.MEDIUM
        
        # Sort by priority, then by cost-benefit ratio
        return sorted(
            recommendations,
            key=lambda r: (
                r.priority.value,
                -r.cost_benefit_ratio,
                -r.debt_reduction_score,
                r.implementation_effort
            )
        )
    
    async def _create_execution_phases(
        self,
        recommendations: List[RemediationRecommendation]
    ) -> List[List[str]]:
        """Create execution phases for recommendations."""
        phases = []
        
        # Phase 1: Immediate actions
        immediate = [r.id for r in recommendations if r.priority == RemediationPriority.IMMEDIATE]
        if immediate:
            phases.append(immediate)
        
        # Phase 2: High priority items
        high_priority = [r.id for r in recommendations if r.priority == RemediationPriority.HIGH]
        if high_priority:
            phases.append(high_priority)
        
        # Phase 3: Medium priority items (chunked)
        medium_priority = [r.id for r in recommendations if r.priority == RemediationPriority.MEDIUM]
        if medium_priority:
            # Split medium items into chunks of 5
            for i in range(0, len(medium_priority), 5):
                phases.append(medium_priority[i:i+5])
        
        # Phase 4: Low priority items
        low_priority = [r.id for r in recommendations if r.priority == RemediationPriority.LOW]
        if low_priority:
            phases.append(low_priority)
        
        return phases
    
    async def _calculate_plan_metrics(
        self,
        recommendations: List[RemediationRecommendation]
    ) -> Dict[str, float]:
        """Calculate overall metrics for the remediation plan."""
        if not recommendations:
            return {
                'total_debt_reduction': 0.0,
                'total_effort': 0.0,
                'total_risk': 0.0,
                'estimated_days': 0
            }
        
        total_debt_reduction = sum(r.debt_reduction_score for r in recommendations)
        total_effort = sum(r.implementation_effort for r in recommendations)
        avg_risk = sum(r.risk_level for r in recommendations) / len(recommendations)
        
        # Estimate days based on effort (assuming 1.0 effort = 1 day)
        estimated_days = int(total_effort * 1.2)  # Add 20% buffer
        
        return {
            'total_debt_reduction': total_debt_reduction,
            'total_effort': total_effort,
            'total_risk': avg_risk,
            'estimated_days': estimated_days
        }
    
    async def _generate_plan_rationale(
        self,
        debt_analysis: Dict[str, Any],
        historical_analysis: Optional[DebtEvolutionResult],
        recommendations: List[RemediationRecommendation]
    ) -> str:
        """Generate rationale for the remediation plan."""
        rationale_parts = []
        
        # Current state
        total_debt = debt_analysis.get('total_debt_score', 0)
        files_analyzed = debt_analysis.get('files_analyzed', 0)
        
        rationale_parts.append(
            f"Current technical debt analysis reveals a total debt score of {total_debt:.2f} "
            f"across {files_analyzed} files."
        )
        
        # Historical context
        if historical_analysis and historical_analysis.trend_analysis:
            trend = historical_analysis.trend_analysis
            rationale_parts.append(
                f"Historical analysis shows debt is {trend.trend_direction} with "
                f"{trend.trend_strength:.1%} confidence over the past 90 days."
            )
        
        # Recommendation summary
        priority_counts = Counter(r.priority for r in recommendations)
        rationale_parts.append(
            f"Generated {len(recommendations)} recommendations: "
            f"{priority_counts.get(RemediationPriority.IMMEDIATE, 0)} immediate, "
            f"{priority_counts.get(RemediationPriority.HIGH, 0)} high priority, "
            f"{priority_counts.get(RemediationPriority.MEDIUM, 0)} medium priority, "
            f"and {priority_counts.get(RemediationPriority.LOW, 0)} low priority items."
        )
        
        return " ".join(rationale_parts)
    
    async def _generate_success_criteria(
        self,
        recommendations: List[RemediationRecommendation]
    ) -> List[str]:
        """Generate success criteria for the remediation plan."""
        criteria = []
        
        # Debt reduction targets
        total_debt_reduction = sum(r.debt_reduction_score for r in recommendations)
        if total_debt_reduction > 0:
            criteria.append(f"Reduce total technical debt score by at least {total_debt_reduction:.1f} points")
        
        # Category-specific criteria
        categories = set()
        for rec in recommendations:
            categories.update(rec.debt_categories)
        
        if DebtCategory.COMPLEXITY in categories:
            criteria.append("Reduce cyclomatic complexity in identified functions")
        
        if DebtCategory.CODE_DUPLICATION in categories:
            criteria.append("Eliminate code duplication through extraction and consolidation")
        
        # General quality criteria
        criteria.extend([
            "Maintain or improve test coverage during refactoring",
            "Complete implementation without breaking existing functionality",
            "Achieve measurable improvement in code maintainability metrics"
        ])
        
        return criteria
    
    async def _identify_potential_blockers(
        self,
        recommendations: List[RemediationRecommendation]
    ) -> List[str]:
        """Identify potential blockers for the remediation plan."""
        blockers = []
        
        # Check for high-risk recommendations
        high_risk_count = len([r for r in recommendations if r.risk_level > 0.6])
        if high_risk_count > 0:
            blockers.append(f"{high_risk_count} high-risk changes that may require extensive testing")
        
        # Check for architectural changes
        arch_changes = len([r for r in recommendations if r.strategy == RemediationStrategy.ARCHITECTURAL_CHANGE])
        if arch_changes > 0:
            blockers.append("Architectural changes may require team alignment and design review")
        
        # Check for dependencies
        dependent_recs = [r for r in recommendations if r.dependencies]
        if dependent_recs:
            blockers.append("Some recommendations have dependencies that must be completed first")
        
        # General blockers
        blockers.extend([
            "Limited development time and competing priorities",
            "Need for thorough testing to prevent regressions",
            "Potential merge conflicts if multiple developers work on same files"
        ])
        
        return blockers
    
    def _load_remediation_templates(self) -> List[RemediationTemplate]:
        """Load remediation templates for common patterns."""
        templates = [
            RemediationTemplate(
                name="Extract Long Method",
                strategy=RemediationStrategy.EXTRACT_METHOD,
                pattern_regex=r"def\s+\w+\(.*\):\s*\n(\s+.*\n){20,}",  # Methods with 20+ lines
                applicability_check="lines_of_code > 20 and cyclomatic_complexity > 10",
                template_code="""
def original_method(self, params):
    # Extract logical blocks into separate methods
    result1 = self._extracted_method_1(params)
    result2 = self._extracted_method_2(result1)
    return self._finalize_result(result2)

def _extracted_method_1(self, params):
    # First logical block
    pass

def _extracted_method_2(self, intermediate):
    # Second logical block  
    pass

def _finalize_result(self, data):
    # Final processing
    pass
                """,
                explanation="Break down complex methods into smaller, focused methods",
                effort_multiplier=1.0
            ),
            
            RemediationTemplate(
                name="Remove Code Duplication",
                strategy=RemediationStrategy.REMOVE_DUPLICATION,
                pattern_regex=r"(.+\n){3,}.*\1.*\n.*\1.*\n",  # Repeated lines pattern
                applicability_check="duplicate_lines > 5",
                template_code="""
def extracted_common_logic(param1, param2):
    # Common logic extracted here
    return processed_result

# Usage in multiple places:
result = extracted_common_logic(data1, data2)
                """,
                explanation="Extract common code into reusable functions",
                effort_multiplier=0.8
            )
        ]
        
        return templates
    
    def _initialize_pattern_matchers(self) -> Dict[str, Any]:
        """Initialize pattern matching utilities."""
        return {
            'complexity_patterns': [],
            'duplication_patterns': [],
            'smell_patterns': []
        }