"""
Enhanced Agent Capabilities for LeanVibe Agent Hive 2.0

Specialized agent implementations with deep project understanding:
- Code Intelligence Agent: Deep code analysis and architectural insights
- Context-Aware QA Agent: Intelligent testing and quality assurance
- Project Health Agent: Continuous monitoring and health assessment
- Documentation Agent: Intelligent documentation generation and maintenance
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
import ast
import re

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from ..core.database import get_session
from ..core.redis import get_redis_client, RedisClient
from ..models.agent import Agent, AgentStatus
from ..models.project_index import ProjectIndex, FileEntry, DependencyRelationship
from .context_integration import AgentContextIntegration, ContextRequest, AgentTaskType, ContextScope

logger = structlog.get_logger()


class AnalysisType(Enum):
    """Types of code analysis."""
    ARCHITECTURE = "architecture"
    PATTERNS = "patterns"
    QUALITY = "quality"
    COMPLEXITY = "complexity"
    DEPENDENCIES = "dependencies"
    SECURITY = "security"
    PERFORMANCE = "performance"


class TestingStrategy(Enum):
    """Testing strategies for QA analysis."""
    UNIT_TESTING = "unit_testing"
    INTEGRATION_TESTING = "integration_testing"
    END_TO_END_TESTING = "end_to_end_testing"
    PERFORMANCE_TESTING = "performance_testing"
    SECURITY_TESTING = "security_testing"
    ACCESSIBILITY_TESTING = "accessibility_testing"


class HealthMetric(Enum):
    """Project health metrics."""
    CODE_QUALITY = "code_quality"
    TEST_COVERAGE = "test_coverage"
    DEPENDENCY_HEALTH = "dependency_health"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    TECHNICAL_DEBT = "technical_debt"


@dataclass
class CodeInsight:
    """Code analysis insight."""
    insight_id: str
    analysis_type: AnalysisType
    file_path: str
    line_number: Optional[int]
    severity: str
    title: str
    description: str
    recommendation: str
    confidence: float
    impact: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestingRecommendation:
    """Testing recommendation from QA analysis."""
    recommendation_id: str
    strategy: TestingStrategy
    target_files: List[str]
    priority: str
    description: str
    implementation_guide: str
    estimated_effort: str
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthAssessment:
    """Project health assessment result."""
    assessment_id: str
    project_id: str
    overall_score: float
    metric_scores: Dict[str, float]
    critical_issues: List[str]
    recommendations: List[str]
    trend_analysis: Dict[str, str]
    assessed_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "assessed_at": self.assessed_at.isoformat()
        }


class CodeIntelligenceAgent:
    """
    Deep code analysis agent with architectural understanding.
    
    Provides comprehensive code analysis, pattern recognition,
    architectural insights, and refactoring recommendations.
    """
    
    def __init__(
        self,
        session: AsyncSession,
        redis_client: RedisClient,
        context_integration: AgentContextIntegration
    ):
        self.session = session
        self.redis = redis_client
        self.context_integration = context_integration
        
        # Analysis configuration
        self.max_analysis_files = 100
        self.complexity_threshold = 10
        self.pattern_confidence_threshold = 0.7
        
        # Cache TTL
        self.analysis_cache_ttl = 3600  # 1 hour
    
    async def analyze_codebase_architecture(
        self,
        project_id: str,
        analysis_scope: str = "full"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive architectural analysis of codebase.
        
        Args:
            project_id: Project to analyze
            analysis_scope: Scope of analysis (full, recent, specific)
            
        Returns:
            Architectural analysis results
        """
        analysis_id = str(uuid.uuid4())
        
        logger.info(
            "Starting codebase architecture analysis",
            analysis_id=analysis_id,
            project_id=project_id,
            scope=analysis_scope
        )
        
        try:
            # Get project context
            context_request = ContextRequest(
                agent_id="code_intelligence_agent",
                project_id=project_id,
                task_type=AgentTaskType.ARCHITECTURE_REVIEW,
                task_description="Comprehensive architectural analysis",
                scope=ContextScope.FULL_PROJECT if analysis_scope == "full" else ContextScope.RELEVANT_FILES,
                max_files=self.max_analysis_files,
                include_dependencies=True
            )
            
            context = await self.context_integration.request_context(context_request)
            
            # Perform architectural analysis
            architecture_insights = await self._analyze_architecture_patterns(
                context.context_data
            )
            
            # Analyze dependency structure
            dependency_analysis = await self._analyze_dependency_architecture(
                context.context_data
            )
            
            # Identify design patterns
            pattern_analysis = await self._identify_design_patterns(
                context.context_data
            )
            
            # Calculate architectural metrics
            architectural_metrics = await self._calculate_architectural_metrics(
                context.context_data
            )
            
            # Generate refactoring recommendations
            refactoring_recommendations = await self._generate_refactoring_recommendations(
                architecture_insights, dependency_analysis, pattern_analysis
            )
            
            # Compile results
            analysis_result = {
                "analysis_id": analysis_id,
                "project_id": project_id,
                "analysis_scope": analysis_scope,
                "architecture_insights": architecture_insights,
                "dependency_analysis": dependency_analysis,
                "pattern_analysis": pattern_analysis,
                "architectural_metrics": architectural_metrics,
                "refactoring_recommendations": refactoring_recommendations,
                "analyzed_files_count": len(context.context_data.get("files", [])),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "confidence_score": self._calculate_analysis_confidence(
                    architecture_insights, dependency_analysis, pattern_analysis
                )
            }
            
            # Cache results
            await self._cache_analysis_results(analysis_id, analysis_result)
            
            logger.info(
                "Codebase architecture analysis completed",
                analysis_id=analysis_id,
                files_analyzed=analysis_result["analyzed_files_count"],
                insights_found=len(architecture_insights),
                confidence=analysis_result["confidence_score"]
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(
                "Codebase architecture analysis failed",
                analysis_id=analysis_id,
                project_id=project_id,
                error=str(e)
            )
            raise
    
    async def identify_code_smells(
        self,
        project_id: str,
        file_paths: Optional[List[str]] = None
    ) -> List[CodeInsight]:
        """
        Identify code smells and quality issues.
        
        Args:
            project_id: Project to analyze
            file_paths: Optional specific files to analyze
            
        Returns:
            List of identified code smells
        """
        try:
            # Get project files
            if file_paths:
                # Analyze specific files
                files = await self._get_specific_files(project_id, file_paths)
            else:
                # Analyze all relevant source files
                files = await self._get_source_files(project_id)
            
            code_smells = []
            
            for file_entry in files:
                if file_entry.is_binary or not file_entry.content_preview:
                    continue
                
                # Analyze file for code smells
                file_smells = await self._analyze_file_for_smells(file_entry)
                code_smells.extend(file_smells)
            
            # Sort by severity and confidence
            code_smells.sort(
                key=lambda x: (self._severity_to_priority(x.severity), x.confidence),
                reverse=True
            )
            
            logger.info(
                "Code smell analysis completed",
                project_id=project_id,
                files_analyzed=len(files),
                smells_found=len(code_smells)
            )
            
            return code_smells
            
        except Exception as e:
            logger.error(
                "Code smell analysis failed",
                project_id=project_id,
                error=str(e)
            )
            raise
    
    async def suggest_refactoring_opportunities(
        self,
        project_id: str,
        complexity_threshold: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest refactoring opportunities based on code analysis.
        
        Args:
            project_id: Project to analyze
            complexity_threshold: Custom complexity threshold
            
        Returns:
            List of refactoring opportunities
        """
        threshold = complexity_threshold or self.complexity_threshold
        
        try:
            # Get architectural analysis
            arch_analysis = await self.analyze_codebase_architecture(project_id, "full")
            
            # Identify refactoring opportunities
            opportunities = []
            
            # Extract complex methods/functions
            for insight in arch_analysis.get("architecture_insights", []):
                if insight.get("type") == "complexity" and insight.get("score", 0) > threshold:
                    opportunities.append({
                        "type": "complexity_reduction",
                        "file_path": insight.get("file_path"),
                        "location": insight.get("location"),
                        "current_complexity": insight.get("score"),
                        "recommended_approach": "Break down into smaller functions",
                        "estimated_effort": "medium",
                        "priority": "high" if insight.get("score", 0) > threshold * 1.5 else "medium"
                    })
            
            # Look for duplicate code patterns
            duplicate_patterns = await self._identify_duplicate_code(project_id)
            for pattern in duplicate_patterns:
                opportunities.append({
                    "type": "duplicate_elimination",
                    "affected_files": pattern.get("files", []),
                    "pattern_description": pattern.get("description"),
                    "recommended_approach": "Extract common functionality",
                    "estimated_effort": "medium",
                    "priority": "medium"
                })
            
            # Identify architectural improvements
            arch_improvements = await self._identify_architectural_improvements(arch_analysis)
            opportunities.extend(arch_improvements)
            
            return opportunities[:20]  # Limit to top 20 opportunities
            
        except Exception as e:
            logger.error(
                "Refactoring analysis failed",
                project_id=project_id,
                error=str(e)
            )
            raise
    
    # ================== PRIVATE METHODS ==================
    
    async def _analyze_architecture_patterns(self, context_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze architectural patterns in the codebase."""
        insights = []
        files = context_data.get("files", [])
        
        # Analyze file organization patterns
        file_structure = self._analyze_file_structure(files)
        insights.append({
            "type": "file_organization",
            "pattern": file_structure.get("pattern", "unknown"),
            "confidence": file_structure.get("confidence", 0.5),
            "description": file_structure.get("description", ""),
            "recommendations": file_structure.get("recommendations", [])
        })
        
        # Analyze naming conventions
        naming_analysis = self._analyze_naming_conventions(files)
        insights.append({
            "type": "naming_conventions",
            "consistency_score": naming_analysis.get("consistency", 0.5),
            "patterns_found": naming_analysis.get("patterns", []),
            "recommendations": naming_analysis.get("recommendations", [])
        })
        
        return insights
    
    async def _analyze_dependency_architecture(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dependency architecture and relationships."""
        dependencies = context_data.get("dependencies", [])
        
        # Calculate dependency metrics
        total_deps = len(dependencies)
        external_deps = len([d for d in dependencies if d.get("is_external", False)])
        internal_deps = total_deps - external_deps
        
        # Analyze dependency patterns
        dependency_patterns = self._analyze_dependency_patterns(dependencies)
        
        # Identify potential issues
        issues = []
        if external_deps > internal_deps * 2:
            issues.append("High external dependency ratio")
        
        circular_deps = self._detect_circular_dependencies(dependencies)
        if circular_deps:
            issues.append(f"Circular dependencies detected: {len(circular_deps)}")
        
        return {
            "total_dependencies": total_deps,
            "external_dependencies": external_deps,
            "internal_dependencies": internal_deps,
            "dependency_ratio": external_deps / max(internal_deps, 1),
            "patterns": dependency_patterns,
            "issues": issues,
            "circular_dependencies": circular_deps
        }
    
    async def _identify_design_patterns(self, context_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify design patterns in the codebase."""
        patterns = []
        files = context_data.get("files", [])
        
        # Look for common design patterns
        for file_data in files:
            content = file_data.get("content_preview", "")
            if not content:
                continue
            
            # Singleton pattern detection
            if self._detect_singleton_pattern(content):
                patterns.append({
                    "pattern": "Singleton",
                    "file_path": file_data.get("relative_path"),
                    "confidence": 0.8,
                    "description": "Singleton pattern implementation detected"
                })
            
            # Factory pattern detection
            if self._detect_factory_pattern(content):
                patterns.append({
                    "pattern": "Factory",
                    "file_path": file_data.get("relative_path"),
                    "confidence": 0.7,
                    "description": "Factory pattern implementation detected"
                })
            
            # Observer pattern detection
            if self._detect_observer_pattern(content):
                patterns.append({
                    "pattern": "Observer",
                    "file_path": file_data.get("relative_path"),
                    "confidence": 0.6,
                    "description": "Observer pattern implementation detected"
                })
        
        return patterns
    
    async def _calculate_architectural_metrics(self, context_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate architectural quality metrics."""
        files = context_data.get("files", [])
        dependencies = context_data.get("dependencies", [])
        
        # Calculate cohesion metric
        cohesion = self._calculate_cohesion(files, dependencies)
        
        # Calculate coupling metric
        coupling = self._calculate_coupling(dependencies)
        
        # Calculate complexity metric
        complexity = self._calculate_overall_complexity(files)
        
        # Calculate maintainability index
        maintainability = (cohesion * 0.4 + (1 - coupling) * 0.3 + (1 - complexity) * 0.3)
        
        return {
            "cohesion": cohesion,
            "coupling": coupling,
            "complexity": complexity,
            "maintainability": maintainability,
            "modularity": self._calculate_modularity(files),
            "testability": self._estimate_testability(files)
        }
    
    async def _generate_refactoring_recommendations(
        self,
        arch_insights: List[Dict[str, Any]],
        dep_analysis: Dict[str, Any],
        pattern_analysis: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate refactoring recommendations based on analysis."""
        recommendations = []
        
        # Recommendations based on dependency analysis
        if dep_analysis.get("dependency_ratio", 0) > 2.0:
            recommendations.append({
                "type": "dependency_reduction",
                "priority": "high",
                "description": "Consider reducing external dependencies",
                "rationale": "High external dependency ratio detected",
                "suggested_actions": [
                    "Evaluate necessity of each external dependency",
                    "Consider implementing internal alternatives for simple dependencies",
                    "Bundle related dependencies"
                ]
            })
        
        # Recommendations based on circular dependencies
        if dep_analysis.get("circular_dependencies"):
            recommendations.append({
                "type": "circular_dependency_resolution",
                "priority": "critical",
                "description": "Resolve circular dependencies",
                "affected_components": dep_analysis["circular_dependencies"],
                "suggested_actions": [
                    "Introduce dependency injection",
                    "Extract common interfaces",
                    "Refactor to eliminate circular references"
                ]
            })
        
        # Recommendations based on architectural insights
        for insight in arch_insights:
            if insight.get("type") == "naming_conventions":
                consistency = insight.get("consistency_score", 1.0)
                if consistency < 0.7:
                    recommendations.append({
                        "type": "naming_standardization",
                        "priority": "medium",
                        "description": "Standardize naming conventions",
                        "current_consistency": consistency,
                        "suggested_actions": [
                            "Establish naming convention guidelines",
                            "Refactor inconsistent names",
                            "Add linting rules for naming"
                        ]
                    })
        
        return recommendations
    
    def _calculate_analysis_confidence(
        self,
        arch_insights: List[Dict[str, Any]],
        dep_analysis: Dict[str, Any],
        pattern_analysis: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence in the analysis."""
        confidence_scores = []
        
        # Add insight confidences
        for insight in arch_insights:
            if "confidence" in insight:
                confidence_scores.append(insight["confidence"])
        
        # Add pattern confidences
        for pattern in pattern_analysis:
            if "confidence" in pattern:
                confidence_scores.append(pattern["confidence"])
        
        # Base confidence on dependency analysis completeness
        if dep_analysis.get("total_dependencies", 0) > 0:
            confidence_scores.append(0.8)
        
        return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
    
    def _analyze_file_structure(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze file organization structure."""
        directories = set()
        file_types = {}
        
        for file_data in files:
            file_path = file_data.get("relative_path", "")
            directory = "/".join(file_path.split("/")[:-1])
            if directory:
                directories.add(directory)
            
            file_type = file_data.get("file_type", "unknown")
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        # Determine organization pattern
        if "src" in directories or "lib" in directories:
            pattern = "standard_source_organization"
            confidence = 0.8
        elif "components" in directories and "services" in directories:
            pattern = "component_service_organization"
            confidence = 0.7
        else:
            pattern = "custom_organization"
            confidence = 0.5
        
        return {
            "pattern": pattern,
            "confidence": confidence,
            "directories_count": len(directories),
            "file_types": file_types,
            "description": f"Project follows {pattern.replace('_', ' ')} pattern",
            "recommendations": self._get_organization_recommendations(pattern, file_types)
        }
    
    def _analyze_naming_conventions(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze naming convention consistency."""
        naming_patterns = {
            "snake_case": 0,
            "camelCase": 0,
            "PascalCase": 0,
            "kebab-case": 0,
            "inconsistent": 0
        }
        
        for file_data in files:
            file_name = file_data.get("file_name", "")
            if not file_name:
                continue
            
            # Remove file extension for analysis
            name_without_ext = file_name.split(".")[0]
            
            if "_" in name_without_ext and name_without_ext.islower():
                naming_patterns["snake_case"] += 1
            elif "-" in name_without_ext:
                naming_patterns["kebab-case"] += 1
            elif name_without_ext[0].isupper() and any(c.isupper() for c in name_without_ext[1:]):
                naming_patterns["PascalCase"] += 1
            elif name_without_ext[0].islower() and any(c.isupper() for c in name_without_ext[1:]):
                naming_patterns["camelCase"] += 1
            else:
                naming_patterns["inconsistent"] += 1
        
        # Calculate consistency score
        total_files = sum(naming_patterns.values())
        if total_files == 0:
            return {"consistency": 1.0, "patterns": [], "recommendations": []}
        
        dominant_pattern = max(naming_patterns, key=naming_patterns.get)
        consistency = naming_patterns[dominant_pattern] / total_files
        
        recommendations = []
        if consistency < 0.8:
            recommendations.append(f"Standardize on {dominant_pattern} naming convention")
            recommendations.append("Refactor inconsistent file names")
        
        return {
            "consistency": consistency,
            "patterns": naming_patterns,
            "dominant_pattern": dominant_pattern,
            "recommendations": recommendations
        }
    
    def _analyze_dependency_patterns(self, dependencies: List[Dict[str, Any]]) -> List[str]:
        """Analyze patterns in dependency usage."""
        patterns = []
        
        # Analyze dependency types
        dep_types = {}
        for dep in dependencies:
            dep_type = dep.get("dependency_type", "unknown")
            dep_types[dep_type] = dep_types.get(dep_type, 0) + 1
        
        # Identify patterns
        if dep_types.get("import", 0) > dep_types.get("require", 0) * 2:
            patterns.append("ES6_module_preference")
        
        if dep_types.get("calls", 0) > len(dependencies) * 0.3:
            patterns.append("high_function_coupling")
        
        return patterns
    
    def _detect_circular_dependencies(self, dependencies: List[Dict[str, Any]]) -> List[str]:
        """Detect circular dependencies in the codebase."""
        # Simplified circular dependency detection
        # In practice, would need more sophisticated graph analysis
        file_dependencies = {}
        
        for dep in dependencies:
            source = dep.get("source_file_id")
            target = dep.get("target_file_id")
            if source and target:
                if source not in file_dependencies:
                    file_dependencies[source] = set()
                file_dependencies[source].add(target)
        
        # Look for simple circular references
        circular_deps = []
        for source, targets in file_dependencies.items():
            for target in targets:
                if target in file_dependencies and source in file_dependencies[target]:
                    circular_deps.append(f"{source} <-> {target}")
        
        return circular_deps
    
    def _detect_singleton_pattern(self, content: str) -> bool:
        """Detect singleton pattern in code content."""
        singleton_indicators = [
            "class.*Singleton",
            "getInstance",
            "private.*constructor",
            "static.*instance"
        ]
        
        for indicator in singleton_indicators:
            if re.search(indicator, content, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_factory_pattern(self, content: str) -> bool:
        """Detect factory pattern in code content."""
        factory_indicators = [
            "class.*Factory",
            "createInstance",
            "factory.*method",
            "create.*Object"
        ]
        
        for indicator in factory_indicators:
            if re.search(indicator, content, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_observer_pattern(self, content: str) -> bool:
        """Detect observer pattern in code content."""
        observer_indicators = [
            "addObserver",
            "removeObserver",
            "notifyObservers",
            "addEventListener",
            "emit.*event"
        ]
        
        for indicator in observer_indicators:
            if re.search(indicator, content, re.IGNORECASE):
                return True
        
        return False
    
    def _calculate_cohesion(self, files: List[Dict[str, Any]], dependencies: List[Dict[str, Any]]) -> float:
        """Calculate module cohesion metric."""
        # Simplified cohesion calculation
        # In practice, would analyze actual code structure
        if not files:
            return 0.0
        
        # Count internal vs external references
        internal_refs = sum(1 for dep in dependencies if not dep.get("is_external", True))
        total_refs = len(dependencies)
        
        return internal_refs / max(total_refs, 1)
    
    def _calculate_coupling(self, dependencies: List[Dict[str, Any]]) -> float:
        """Calculate coupling metric."""
        if not dependencies:
            return 0.0
        
        # Count unique file relationships
        file_pairs = set()
        for dep in dependencies:
            source = dep.get("source_file_id")
            target = dep.get("target_file_id")
            if source and target and source != target:
                file_pairs.add((source, target))
        
        # Normalize coupling score
        total_deps = len(dependencies)
        unique_pairs = len(file_pairs)
        
        return unique_pairs / max(total_deps, 1)
    
    def _calculate_overall_complexity(self, files: List[Dict[str, Any]]) -> float:
        """Calculate overall complexity metric."""
        complexity_scores = []
        
        for file_data in files:
            # Estimate complexity based on file size and type
            line_count = file_data.get("line_count", 0)
            file_type = file_data.get("file_type", "")
            
            if file_type == "source" and line_count > 0:
                # Simple complexity estimation
                complexity = min(line_count / 1000.0, 1.0)  # Normalize to 0-1
                complexity_scores.append(complexity)
        
        return sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0.0
    
    def _calculate_modularity(self, files: List[Dict[str, Any]]) -> float:
        """Calculate modularity metric."""
        # Count distinct modules/directories
        directories = set()
        for file_data in files:
            file_path = file_data.get("relative_path", "")
            directory = "/".join(file_path.split("/")[:-1])
            if directory:
                directories.add(directory)
        
        total_files = len(files)
        total_dirs = len(directories)
        
        # Good modularity has reasonable files per directory
        if total_dirs == 0:
            return 0.0
        
        files_per_dir = total_files / total_dirs
        optimal_files_per_dir = 10  # Assumed optimal
        
        return max(0.0, 1.0 - abs(files_per_dir - optimal_files_per_dir) / optimal_files_per_dir)
    
    def _estimate_testability(self, files: List[Dict[str, Any]]) -> float:
        """Estimate testability of the codebase."""
        source_files = [f for f in files if f.get("file_type") == "source"]
        test_files = [f for f in files if "test" in f.get("relative_path", "").lower()]
        
        if not source_files:
            return 0.0
        
        test_ratio = len(test_files) / len(source_files)
        return min(test_ratio, 1.0)
    
    def _get_organization_recommendations(self, pattern: str, file_types: Dict[str, int]) -> List[str]:
        """Get recommendations for file organization."""
        recommendations = []
        
        if pattern == "custom_organization":
            recommendations.append("Consider adopting standard source organization pattern")
            recommendations.append("Separate source code from configuration files")
        
        if file_types.get("config", 0) > file_types.get("source", 0):
            recommendations.append("Move configuration files to dedicated config directory")
        
        return recommendations
    
    async def _get_specific_files(self, project_id: str, file_paths: List[str]) -> List[FileEntry]:
        """Get specific files by paths."""
        stmt = select(FileEntry).where(
            and_(
                FileEntry.project_id == project_id,
                FileEntry.relative_path.in_(file_paths)
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def _get_source_files(self, project_id: str) -> List[FileEntry]:
        """Get all source files for a project."""
        stmt = select(FileEntry).where(
            and_(
                FileEntry.project_id == project_id,
                FileEntry.file_type == "source",
                FileEntry.is_binary == False
            )
        ).limit(self.max_analysis_files)
        
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def _analyze_file_for_smells(self, file_entry: FileEntry) -> List[CodeInsight]:
        """Analyze a single file for code smells."""
        smells = []
        content = file_entry.content_preview or ""
        
        # Long method detection
        lines = content.split('\n')
        current_function = None
        function_start = 0
        
        for i, line in enumerate(lines):
            # Simple function detection (language-agnostic)
            if any(keyword in line for keyword in ['def ', 'function ', 'func ', 'method ']):
                if current_function and (i - function_start) > 50:
                    # Long method smell
                    smells.append(CodeInsight(
                        insight_id=str(uuid.uuid4()),
                        analysis_type=AnalysisType.QUALITY,
                        file_path=file_entry.relative_path,
                        line_number=function_start,
                        severity="medium",
                        title="Long Method",
                        description=f"Method '{current_function}' is {i - function_start} lines long",
                        recommendation="Consider breaking down into smaller methods",
                        confidence=0.8,
                        impact="maintainability"
                    ))
                
                current_function = line.strip()
                function_start = i
        
        # Large class detection
        if len(lines) > 1000:
            smells.append(CodeInsight(
                insight_id=str(uuid.uuid4()),
                analysis_type=AnalysisType.QUALITY,
                file_path=file_entry.relative_path,
                line_number=1,
                severity="high",
                title="Large Class",
                description=f"File has {len(lines)} lines, indicating a potentially large class",
                recommendation="Consider splitting into multiple classes",
                confidence=0.7,
                impact="maintainability"
            ))
        
        # Duplicate code detection (simplified)
        duplicate_patterns = self._find_duplicate_patterns(lines)
        for pattern in duplicate_patterns:
            smells.append(CodeInsight(
                insight_id=str(uuid.uuid4()),
                analysis_type=AnalysisType.QUALITY,
                file_path=file_entry.relative_path,
                line_number=pattern["line"],
                severity="medium",
                title="Duplicate Code",
                description=f"Duplicate code pattern found: {pattern['pattern'][:50]}...",
                recommendation="Extract common functionality",
                confidence=0.6,
                impact="maintainability"
            ))
        
        return smells
    
    def _find_duplicate_patterns(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Find duplicate code patterns in lines."""
        patterns = []
        
        # Simple duplicate detection - look for repeated line sequences
        min_sequence_length = 3
        
        for i in range(len(lines) - min_sequence_length):
            sequence = lines[i:i + min_sequence_length]
            
            # Look for this sequence later in the file
            for j in range(i + min_sequence_length, len(lines) - min_sequence_length):
                if lines[j:j + min_sequence_length] == sequence:
                    patterns.append({
                        "line": i + 1,
                        "pattern": "\n".join(sequence),
                        "duplicate_at": j + 1
                    })
                    break
        
        return patterns[:5]  # Limit to 5 patterns
    
    def _severity_to_priority(self, severity: str) -> int:
        """Convert severity to numeric priority."""
        return {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }.get(severity.lower(), 1)
    
    async def _identify_duplicate_code(self, project_id: str) -> List[Dict[str, Any]]:
        """Identify duplicate code across the project."""
        # Simplified implementation
        return []
    
    async def _identify_architectural_improvements(self, arch_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify architectural improvement opportunities."""
        improvements = []
        
        metrics = arch_analysis.get("architectural_metrics", {})
        
        # Low cohesion improvement
        if metrics.get("cohesion", 1.0) < 0.6:
            improvements.append({
                "type": "cohesion_improvement",
                "description": "Improve module cohesion",
                "current_score": metrics["cohesion"],
                "recommended_approach": "Group related functionality together",
                "estimated_effort": "high",
                "priority": "medium"
            })
        
        # High coupling improvement
        if metrics.get("coupling", 0.0) > 0.7:
            improvements.append({
                "type": "coupling_reduction",
                "description": "Reduce inter-module coupling",
                "current_score": metrics["coupling"],
                "recommended_approach": "Introduce interfaces and dependency injection",
                "estimated_effort": "high",
                "priority": "medium"
            })
        
        return improvements
    
    async def _cache_analysis_results(self, analysis_id: str, results: Dict[str, Any]) -> None:
        """Cache analysis results for performance."""
        cache_key = f"code_analysis:{analysis_id}"
        await self.redis.setex(
            cache_key,
            self.analysis_cache_ttl,
            json.dumps(results)
        )


class ContextAwareQAAgent:
    """
    Intelligent testing and quality assurance agent.
    
    Provides context-aware test generation, coverage analysis,
    quality assessments, and testing strategy recommendations.
    """
    
    def __init__(
        self,
        session: AsyncSession,
        redis_client: RedisClient,
        context_integration: AgentContextIntegration
    ):
        self.session = session
        self.redis = redis_client
        self.context_integration = context_integration
        
        # QA configuration
        self.coverage_threshold = 0.8
        self.quality_threshold = 0.7
        self.test_cache_ttl = 1800  # 30 minutes
    
    async def generate_test_recommendations(
        self,
        project_id: str,
        focus_areas: Optional[List[str]] = None
    ) -> List[TestingRecommendation]:
        """
        Generate intelligent test recommendations based on project analysis.
        
        Args:
            project_id: Project to analyze
            focus_areas: Optional focus areas for testing
            
        Returns:
            List of testing recommendations
        """
        try:
            # Get project context for testing
            context_request = ContextRequest(
                agent_id="qa_agent",
                project_id=project_id,
                task_type=AgentTaskType.TESTING,
                task_description="Generate test recommendations",
                scope=ContextScope.RELEVANT_FILES,
                focus_areas=focus_areas
            )
            
            context = await self.context_integration.request_context(context_request)
            
            recommendations = []
            
            # Analyze test coverage gaps
            coverage_recommendations = await self._analyze_test_coverage_gaps(context.context_data)
            recommendations.extend(coverage_recommendations)
            
            # Identify untested critical paths
            critical_path_tests = await self._identify_critical_path_tests(context.context_data)
            recommendations.extend(critical_path_tests)
            
            # Suggest integration tests
            integration_tests = await self._suggest_integration_tests(context.context_data)
            recommendations.extend(integration_tests)
            
            # Performance testing opportunities
            performance_tests = await self._identify_performance_testing_opportunities(context.context_data)
            recommendations.extend(performance_tests)
            
            # Security testing recommendations
            security_tests = await self._identify_security_testing_needs(context.context_data)
            recommendations.extend(security_tests)
            
            # Sort by priority
            recommendations.sort(key=lambda x: self._priority_to_score(x.priority), reverse=True)
            
            logger.info(
                "Test recommendations generated",
                project_id=project_id,
                recommendations_count=len(recommendations),
                focus_areas=focus_areas
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(
                "Test recommendation generation failed",
                project_id=project_id,
                error=str(e)
            )
            raise
    
    async def assess_test_quality(
        self,
        project_id: str,
        test_file_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Assess the quality of existing tests.
        
        Args:
            project_id: Project to assess
            test_file_paths: Optional specific test files
            
        Returns:
            Test quality assessment
        """
        try:
            # Get test files
            if test_file_paths:
                test_files = await self._get_specific_files(project_id, test_file_paths)
            else:
                test_files = await self._get_test_files(project_id)
            
            if not test_files:
                return {
                    "overall_quality": 0.0,
                    "test_count": 0,
                    "issues": ["No test files found"],
                    "recommendations": ["Start by adding basic unit tests"]
                }
            
            # Analyze test quality metrics
            quality_metrics = {}
            
            # Test coverage estimation
            coverage_estimate = await self._estimate_test_coverage(project_id, test_files)
            quality_metrics["coverage"] = coverage_estimate
            
            # Test completeness
            completeness = await self._assess_test_completeness(test_files)
            quality_metrics["completeness"] = completeness
            
            # Test maintainability
            maintainability = await self._assess_test_maintainability(test_files)
            quality_metrics["maintainability"] = maintainability
            
            # Test isolation
            isolation = await self._assess_test_isolation(test_files)
            quality_metrics["isolation"] = isolation
            
            # Calculate overall quality
            overall_quality = sum(quality_metrics.values()) / len(quality_metrics)
            
            # Generate quality report
            issues = []
            recommendations = []
            
            if coverage_estimate < self.coverage_threshold:
                issues.append(f"Low test coverage: {coverage_estimate:.1%}")
                recommendations.append("Increase test coverage by adding more unit tests")
            
            if completeness < 0.7:
                issues.append("Incomplete test scenarios")
                recommendations.append("Add tests for edge cases and error conditions")
            
            if maintainability < 0.6:
                issues.append("Test maintainability concerns")
                recommendations.append("Refactor tests to improve readability and reduce duplication")
            
            return {
                "overall_quality": overall_quality,
                "quality_metrics": quality_metrics,
                "test_count": len(test_files),
                "issues": issues,
                "recommendations": recommendations,
                "assessment_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(
                "Test quality assessment failed",
                project_id=project_id,
                error=str(e)
            )
            raise
    
    # ================== PRIVATE METHODS ==================
    
    async def _analyze_test_coverage_gaps(self, context_data: Dict[str, Any]) -> List[TestingRecommendation]:
        """Analyze test coverage gaps and suggest tests."""
        recommendations = []
        files = context_data.get("files", [])
        
        # Find source files without corresponding tests
        source_files = [f for f in files if f.get("file_type") == "source"]
        test_files = [f for f in files if "test" in f.get("relative_path", "").lower()]
        
        tested_modules = set()
        for test_file in test_files:
            # Extract module name from test file
            test_path = test_file.get("relative_path", "")
            module_name = test_path.replace("test_", "").replace("_test", "").replace(".test", "")
            tested_modules.add(module_name)
        
        # Find untested source files
        for source_file in source_files:
            source_path = source_file.get("relative_path", "")
            if not any(module in source_path for module in tested_modules):
                recommendations.append(TestingRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    strategy=TestingStrategy.UNIT_TESTING,
                    target_files=[source_path],
                    priority="high",
                    description=f"Add unit tests for {source_file.get('file_name')}",
                    implementation_guide="Create unit tests covering main functions and edge cases",
                    estimated_effort="medium"
                ))
        
        return recommendations[:10]  # Limit to top 10
    
    async def _identify_critical_path_tests(self, context_data: Dict[str, Any]) -> List[TestingRecommendation]:
        """Identify critical paths that need testing."""
        recommendations = []
        
        # Look for files with high dependency counts (critical paths)
        dependencies = context_data.get("dependencies", [])
        file_dependency_counts = {}
        
        for dep in dependencies:
            target_file = dep.get("target_file_id")
            if target_file:
                file_dependency_counts[target_file] = file_dependency_counts.get(target_file, 0) + 1
        
        # Identify highly depended-upon files
        critical_files = sorted(
            file_dependency_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        for file_id, dependency_count in critical_files:
            # Find the actual file
            files = context_data.get("files", [])
            target_file = next((f for f in files if f.get("id") == file_id), None)
            
            if target_file:
                recommendations.append(TestingRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    strategy=TestingStrategy.INTEGRATION_TESTING,
                    target_files=[target_file.get("relative_path")],
                    priority="critical",
                    description=f"Add integration tests for critical component: {target_file.get('file_name')}",
                    implementation_guide="Focus on testing interactions with dependent components",
                    estimated_effort="high",
                    dependencies=[f"file_{file_id}"]
                ))
        
        return recommendations
    
    async def _suggest_integration_tests(self, context_data: Dict[str, Any]) -> List[TestingRecommendation]:
        """Suggest integration tests based on component interactions."""
        recommendations = []
        
        # Analyze dependencies to find integration points
        dependencies = context_data.get("dependencies", [])
        
        # Group dependencies by source file to find complex integrations
        source_dependencies = {}
        for dep in dependencies:
            source_file = dep.get("source_file_id")
            if source_file:
                if source_file not in source_dependencies:
                    source_dependencies[source_file] = []
                source_dependencies[source_file].append(dep)
        
        # Find files with many external dependencies (integration candidates)
        for source_file, deps in source_dependencies.items():
            external_deps = [d for d in deps if d.get("is_external", False)]
            
            if len(external_deps) >= 3:  # Files with 3+ external dependencies
                files = context_data.get("files", [])
                target_file = next((f for f in files if f.get("id") == source_file), None)
                
                if target_file:
                    recommendations.append(TestingRecommendation(
                        recommendation_id=str(uuid.uuid4()),
                        strategy=TestingStrategy.INTEGRATION_TESTING,
                        target_files=[target_file.get("relative_path")],
                        priority="medium",
                        description=f"Add integration tests for {target_file.get('file_name')} (multiple external dependencies)",
                        implementation_guide="Test interactions with external dependencies and error handling",
                        estimated_effort="medium"
                    ))
        
        return recommendations
    
    async def _identify_performance_testing_opportunities(self, context_data: Dict[str, Any]) -> List[TestingRecommendation]:
        """Identify opportunities for performance testing."""
        recommendations = []
        files = context_data.get("files", [])
        
        # Look for files that might have performance implications
        performance_keywords = ["loop", "iteration", "batch", "process", "compute", "algorithm"]
        
        for file_data in files:
            content = file_data.get("content_preview", "").lower()
            file_name = file_data.get("file_name", "").lower()
            
            # Check for performance-related content
            if any(keyword in content or keyword in file_name for keyword in performance_keywords):
                recommendations.append(TestingRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    strategy=TestingStrategy.PERFORMANCE_TESTING,
                    target_files=[file_data.get("relative_path")],
                    priority="medium",
                    description=f"Add performance tests for {file_data.get('file_name')}",
                    implementation_guide="Test performance under load and measure execution time",
                    estimated_effort="medium"
                ))
        
        return recommendations[:3]  # Limit to 3 performance test recommendations
    
    async def _identify_security_testing_needs(self, context_data: Dict[str, Any]) -> List[TestingRecommendation]:
        """Identify security testing needs."""
        recommendations = []
        files = context_data.get("files", [])
        
        # Look for files with security implications
        security_keywords = ["auth", "password", "token", "encrypt", "decrypt", "security", "session"]
        
        for file_data in files:
            content = file_data.get("content_preview", "").lower()
            file_name = file_data.get("file_name", "").lower()
            
            if any(keyword in content or keyword in file_name for keyword in security_keywords):
                recommendations.append(TestingRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    strategy=TestingStrategy.SECURITY_TESTING,
                    target_files=[file_data.get("relative_path")],
                    priority="high",
                    description=f"Add security tests for {file_data.get('file_name')}",
                    implementation_guide="Test for common vulnerabilities and security edge cases",
                    estimated_effort="high"
                ))
        
        return recommendations[:5]  # Limit to 5 security test recommendations
    
    def _priority_to_score(self, priority: str) -> int:
        """Convert priority to numeric score for sorting."""
        return {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }.get(priority.lower(), 1)
    
    async def _get_specific_files(self, project_id: str, file_paths: List[str]) -> List[FileEntry]:
        """Get specific files by paths."""
        stmt = select(FileEntry).where(
            and_(
                FileEntry.project_id == project_id,
                FileEntry.relative_path.in_(file_paths)
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def _get_test_files(self, project_id: str) -> List[FileEntry]:
        """Get all test files for a project."""
        stmt = select(FileEntry).where(
            and_(
                FileEntry.project_id == project_id,
                or_(
                    FileEntry.relative_path.like("%test%"),
                    FileEntry.relative_path.like("%spec%"),
                    FileEntry.file_type == "test"
                )
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def _estimate_test_coverage(self, project_id: str, test_files: List[FileEntry]) -> float:
        """Estimate test coverage based on test files vs source files."""
        # Get source files count
        source_stmt = select(func.count(FileEntry.id)).where(
            and_(
                FileEntry.project_id == project_id,
                FileEntry.file_type == "source"
            )
        )
        source_result = await self.session.execute(source_stmt)
        source_count = source_result.scalar() or 0
        
        if source_count == 0:
            return 0.0
        
        # Estimate coverage based on test-to-source ratio
        test_count = len(test_files)
        coverage_ratio = test_count / source_count
        
        # Cap at 100% and apply some heuristics
        return min(coverage_ratio * 0.8, 1.0)  # Assume 80% max coverage estimation
    
    async def _assess_test_completeness(self, test_files: List[FileEntry]) -> float:
        """Assess completeness of test scenarios."""
        if not test_files:
            return 0.0
        
        completeness_scores = []
        
        for test_file in test_files:
            content = test_file.content_preview or ""
            
            # Look for different types of test scenarios
            scenario_indicators = {
                "positive_tests": ["test_success", "test_valid", "should_pass"],
                "negative_tests": ["test_error", "test_invalid", "should_fail", "test_exception"],
                "edge_cases": ["test_edge", "test_boundary", "test_limit"],
                "integration": ["test_integration", "test_end_to_end"]
            }
            
            found_scenarios = 0
            for scenario_type, indicators in scenario_indicators.items():
                if any(indicator in content.lower() for indicator in indicators):
                    found_scenarios += 1
            
            # Score based on scenario diversity
            completeness_scores.append(found_scenarios / len(scenario_indicators))
        
        return sum(completeness_scores) / len(completeness_scores)
    
    async def _assess_test_maintainability(self, test_files: List[FileEntry]) -> float:
        """Assess maintainability of test code."""
        if not test_files:
            return 0.0
        
        maintainability_scores = []
        
        for test_file in test_files:
            score = 1.0
            
            # Penalize for very long test files
            line_count = test_file.line_count or 0
            if line_count > 1000:
                score -= 0.3
            elif line_count > 500:
                score -= 0.1
            
            # Check for good test structure indicators
            content = test_file.content_preview or ""
            if "setup" in content.lower() or "teardown" in content.lower():
                score += 0.1
            
            if "describe" in content.lower() or "context" in content.lower():
                score += 0.1
            
            maintainability_scores.append(max(0.0, min(1.0, score)))
        
        return sum(maintainability_scores) / len(maintainability_scores)
    
    async def _assess_test_isolation(self, test_files: List[FileEntry]) -> float:
        """Assess isolation of test cases."""
        if not test_files:
            return 0.0
        
        isolation_scores = []
        
        for test_file in test_files:
            content = test_file.content_preview or ""
            
            # Look for isolation indicators
            isolation_score = 0.5  # Default score
            
            # Good indicators
            if "mock" in content.lower() or "stub" in content.lower():
                isolation_score += 0.2
            
            if "beforeeach" in content.lower() or "aftereach" in content.lower():
                isolation_score += 0.2
            
            # Bad indicators
            if "sleep" in content.lower() or "wait" in content.lower():
                isolation_score -= 0.2
            
            if "database" in content.lower() and "mock" not in content.lower():
                isolation_score -= 0.1
            
            isolation_scores.append(max(0.0, min(1.0, isolation_score)))
        
        return sum(isolation_scores) / len(isolation_scores)


class ProjectHealthAgent:
    """
    Continuous project monitoring and health assessment agent.
    
    Monitors project health metrics, tracks trends, identifies issues,
    and provides recommendations for maintaining project quality.
    """
    
    def __init__(
        self,
        session: AsyncSession,
        redis_client: RedisClient,
        context_integration: AgentContextIntegration
    ):
        self.session = session
        self.redis = redis_client
        self.context_integration = context_integration
        
        # Health monitoring configuration
        self.health_check_interval = 300  # 5 minutes
        self.trend_analysis_days = 7
        self.health_cache_ttl = 600  # 10 minutes
    
    async def assess_project_health(
        self,
        project_id: str,
        include_trends: bool = True
    ) -> HealthAssessment:
        """
        Perform comprehensive project health assessment.
        
        Args:
            project_id: Project to assess
            include_trends: Whether to include trend analysis
            
        Returns:
            HealthAssessment with scores and recommendations
        """
        assessment_id = str(uuid.uuid4())
        
        logger.info(
            "Starting project health assessment",
            assessment_id=assessment_id,
            project_id=project_id,
            include_trends=include_trends
        )
        
        try:
            # Get project context
            context_request = ContextRequest(
                agent_id="project_health_agent",
                project_id=project_id,
                task_type=AgentTaskType.CODE_ANALYSIS,
                task_description="Project health assessment",
                scope=ContextScope.FULL_PROJECT,
                include_dependencies=True
            )
            
            context = await self.context_integration.request_context(context_request)
            
            # Calculate health metrics
            metric_scores = {}
            
            # Code quality metric
            metric_scores[HealthMetric.CODE_QUALITY.value] = await self._assess_code_quality(context.context_data)
            
            # Test coverage metric
            metric_scores[HealthMetric.TEST_COVERAGE.value] = await self._assess_test_coverage(context.context_data)
            
            # Dependency health metric
            metric_scores[HealthMetric.DEPENDENCY_HEALTH.value] = await self._assess_dependency_health(context.context_data)
            
            # Performance metric
            metric_scores[HealthMetric.PERFORMANCE.value] = await self._assess_performance(context.context_data)
            
            # Security metric
            metric_scores[HealthMetric.SECURITY.value] = await self._assess_security(context.context_data)
            
            # Maintainability metric
            metric_scores[HealthMetric.MAINTAINABILITY.value] = await self._assess_maintainability(context.context_data)
            
            # Technical debt metric
            metric_scores[HealthMetric.TECHNICAL_DEBT.value] = await self._assess_technical_debt(context.context_data)
            
            # Calculate overall score
            overall_score = sum(metric_scores.values()) / len(metric_scores)
            
            # Identify critical issues
            critical_issues = await self._identify_critical_issues(metric_scores, context.context_data)
            
            # Generate recommendations
            recommendations = await self._generate_health_recommendations(metric_scores, critical_issues)
            
            # Perform trend analysis if requested
            trend_analysis = {}
            if include_trends:
                trend_analysis = await self._analyze_health_trends(project_id, metric_scores)
            
            # Create assessment
            assessment = HealthAssessment(
                assessment_id=assessment_id,
                project_id=project_id,
                overall_score=overall_score,
                metric_scores=metric_scores,
                critical_issues=critical_issues,
                recommendations=recommendations,
                trend_analysis=trend_analysis,
                assessed_at=datetime.utcnow()
            )
            
            # Store assessment for trend tracking
            await self._store_health_assessment(assessment)
            
            logger.info(
                "Project health assessment completed",
                assessment_id=assessment_id,
                overall_score=overall_score,
                critical_issues_count=len(critical_issues)
            )
            
            return assessment
            
        except Exception as e:
            logger.error(
                "Project health assessment failed",
                assessment_id=assessment_id,
                project_id=project_id,
                error=str(e)
            )
            raise
    
    async def monitor_project_continuously(
        self,
        project_id: str,
        monitoring_duration_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Start continuous monitoring of project health.
        
        Args:
            project_id: Project to monitor
            monitoring_duration_hours: Duration of monitoring
            
        Returns:
            Monitoring session information
        """
        session_id = str(uuid.uuid4())
        
        try:
            # Start monitoring session
            monitoring_data = {
                "session_id": session_id,
                "project_id": project_id,
                "started_at": datetime.utcnow().isoformat(),
                "duration_hours": monitoring_duration_hours,
                "check_interval_seconds": self.health_check_interval,
                "status": "active"
            }
            
            # Store monitoring session
            session_key = f"health_monitoring:{session_id}"
            await self.redis.setex(
                session_key,
                monitoring_duration_hours * 3600,
                json.dumps(monitoring_data)
            )
            
            # Schedule health checks
            await self._schedule_health_checks(session_id, project_id, monitoring_duration_hours)
            
            logger.info(
                "Continuous health monitoring started",
                session_id=session_id,
                project_id=project_id,
                duration_hours=monitoring_duration_hours
            )
            
            return {
                "monitoring_session_id": session_id,
                "project_id": project_id,
                "status": "started",
                "duration_hours": monitoring_duration_hours,
                "check_interval_seconds": self.health_check_interval
            }
            
        except Exception as e:
            logger.error(
                "Failed to start continuous monitoring",
                project_id=project_id,
                error=str(e)
            )
            raise
    
    # ================== PRIVATE METHODS ==================
    
    async def _assess_code_quality(self, context_data: Dict[str, Any]) -> float:
        """Assess code quality metric."""
        files = context_data.get("files", [])
        
        if not files:
            return 0.0
        
        quality_scores = []
        
        for file_data in files:
            if file_data.get("file_type") != "source":
                continue
            
            score = 1.0
            
            # File size penalty
            line_count = file_data.get("line_count", 0)
            if line_count > 1000:
                score -= 0.3
            elif line_count > 500:
                score -= 0.1
            
            # Naming convention bonus
            file_name = file_data.get("file_name", "")
            if self._follows_naming_conventions(file_name):
                score += 0.1
            
            # Documentation presence
            content = file_data.get("content_preview", "")
            if any(indicator in content.lower() for indicator in ["/**", "//", "#", "'''", '"""']):
                score += 0.1
            
            quality_scores.append(max(0.0, min(1.0, score)))
        
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    async def _assess_test_coverage(self, context_data: Dict[str, Any]) -> float:
        """Assess test coverage metric."""
        files = context_data.get("files", [])
        
        source_files = [f for f in files if f.get("file_type") == "source"]
        test_files = [f for f in files if "test" in f.get("relative_path", "").lower()]
        
        if not source_files:
            return 0.0
        
        # Simple coverage estimation based on test-to-source ratio
        coverage_ratio = len(test_files) / len(source_files)
        return min(coverage_ratio, 1.0)
    
    async def _assess_dependency_health(self, context_data: Dict[str, Any]) -> float:
        """Assess dependency health metric."""
        dependencies = context_data.get("dependencies", [])
        
        if not dependencies:
            return 1.0  # No dependencies is perfectly healthy
        
        # Calculate health based on dependency characteristics
        external_deps = len([d for d in dependencies if d.get("is_external", False)])
        total_deps = len(dependencies)
        
        # Prefer internal dependencies
        internal_ratio = 1.0 - (external_deps / total_deps)
        
        # Check for circular dependencies (simplified)
        circular_deps = self._detect_simple_circular_deps(dependencies)
        circular_penalty = len(circular_deps) * 0.1
        
        health_score = internal_ratio - circular_penalty
        return max(0.0, min(1.0, health_score))
    
    async def _assess_performance(self, context_data: Dict[str, Any]) -> float:
        """Assess performance metric."""
        files = context_data.get("files", [])
        
        # Look for performance indicators
        performance_score = 0.8  # Default good score
        
        # Check for potential performance issues
        for file_data in files:
            content = file_data.get("content_preview", "").lower()
            
            # Performance red flags
            if any(flag in content for flag in ["n*n", "nested loop", "recursive", "while true"]):
                performance_score -= 0.1
            
            # Performance good practices
            if any(practice in content for practice in ["cache", "optimize", "efficient", "index"]):
                performance_score += 0.05
        
        return max(0.0, min(1.0, performance_score))
    
    async def _assess_security(self, context_data: Dict[str, Any]) -> float:
        """Assess security metric."""
        files = context_data.get("files", [])
        
        security_score = 0.8  # Default good score
        
        for file_data in files:
            content = file_data.get("content_preview", "").lower()
            
            # Security red flags
            security_issues = [
                "password" in content and "plain" in content,
                "token" in content and "hardcoded" in content,
                "sql" in content and "concatenation" in content,
                "eval(" in content,
                "exec(" in content
            ]
            
            security_score -= sum(security_issues) * 0.2
            
            # Security good practices
            if any(practice in content for practice in ["encrypt", "hash", "validate", "sanitize"]):
                security_score += 0.05
        
        return max(0.0, min(1.0, security_score))
    
    async def _assess_maintainability(self, context_data: Dict[str, Any]) -> float:
        """Assess maintainability metric."""
        files = context_data.get("files", [])
        
        maintainability_scores = []
        
        for file_data in files:
            if file_data.get("file_type") != "source":
                continue
            
            score = 0.8  # Default score
            
            # File size impact
            line_count = file_data.get("line_count", 0)
            if line_count > 1000:
                score -= 0.4
            elif line_count > 500:
                score -= 0.2
            elif line_count < 50:
                score += 0.1
            
            # Documentation presence
            content = file_data.get("content_preview", "")
            if any(doc in content for doc in ["/**", "//", "#", "'''", '"""']):
                score += 0.2
            
            maintainability_scores.append(max(0.0, min(1.0, score)))
        
        return sum(maintainability_scores) / len(maintainability_scores) if maintainability_scores else 0.0
    
    async def _assess_technical_debt(self, context_data: Dict[str, Any]) -> float:
        """Assess technical debt metric (lower debt = higher score)."""
        files = context_data.get("files", [])
        
        debt_indicators = 0
        total_checks = 0
        
        for file_data in files:
            content = file_data.get("content_preview", "").lower()
            
            # Technical debt indicators
            debt_keywords = ["todo", "fixme", "hack", "workaround", "temporary", "quick fix"]
            debt_indicators += sum(1 for keyword in debt_keywords if keyword in content)
            total_checks += len(debt_keywords)
            
            # Code smell indicators
            if file_data.get("line_count", 0) > 1000:
                debt_indicators += 1
                total_checks += 1
        
        if total_checks == 0:
            return 1.0
        
        debt_ratio = debt_indicators / total_checks
        return max(0.0, 1.0 - debt_ratio)
    
    async def _identify_critical_issues(
        self,
        metric_scores: Dict[str, float],
        context_data: Dict[str, Any]
    ) -> List[str]:
        """Identify critical issues based on metric scores."""
        issues = []
        
        # Check each metric for critical thresholds
        for metric, score in metric_scores.items():
            if score < 0.3:  # Critical threshold
                issues.append(f"Critical {metric.replace('_', ' ')} issues detected (score: {score:.2f})")
            elif score < 0.5:  # Warning threshold
                issues.append(f"Low {metric.replace('_', ' ')} score (score: {score:.2f})")
        
        # Specific issue detection
        dependencies = context_data.get("dependencies", [])
        if dependencies:
            external_ratio = len([d for d in dependencies if d.get("is_external", False)]) / len(dependencies)
            if external_ratio > 0.8:
                issues.append("High external dependency ratio")
        
        files = context_data.get("files", [])
        large_files = [f for f in files if f.get("line_count", 0) > 1000]
        if len(large_files) > len(files) * 0.2:
            issues.append("Many large files detected")
        
        return issues
    
    async def _generate_health_recommendations(
        self,
        metric_scores: Dict[str, float],
        critical_issues: List[str]
    ) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        # Recommendations based on metric scores
        if metric_scores.get(HealthMetric.CODE_QUALITY.value, 1.0) < 0.6:
            recommendations.append("Improve code quality by refactoring large files and adding documentation")
        
        if metric_scores.get(HealthMetric.TEST_COVERAGE.value, 1.0) < 0.5:
            recommendations.append("Increase test coverage by adding unit and integration tests")
        
        if metric_scores.get(HealthMetric.DEPENDENCY_HEALTH.value, 1.0) < 0.6:
            recommendations.append("Review and reduce external dependencies where possible")
        
        if metric_scores.get(HealthMetric.SECURITY.value, 1.0) < 0.7:
            recommendations.append("Address security concerns by implementing proper validation and encryption")
        
        if metric_scores.get(HealthMetric.TECHNICAL_DEBT.value, 1.0) < 0.6:
            recommendations.append("Address technical debt by resolving TODOs and refactoring workarounds")
        
        # Critical issue recommendations
        if critical_issues:
            recommendations.append("Prioritize addressing critical issues identified in the assessment")
        
        return recommendations
    
    async def _analyze_health_trends(
        self,
        project_id: str,
        current_scores: Dict[str, float]
    ) -> Dict[str, str]:
        """Analyze health trends over time."""
        try:
            # Get historical assessments
            historical_assessments = await self._get_historical_assessments(project_id)
            
            if len(historical_assessments) < 2:
                return {"trend": "insufficient_data"}
            
            trends = {}
            
            # Analyze trend for each metric
            for metric in current_scores.keys():
                historical_scores = [
                    assessment.get("metric_scores", {}).get(metric, 0.0)
                    for assessment in historical_assessments
                ]
                
                if len(historical_scores) >= 2:
                    recent_avg = sum(historical_scores[-3:]) / min(3, len(historical_scores))
                    older_avg = sum(historical_scores[:-3]) / max(1, len(historical_scores) - 3)
                    
                    if recent_avg > older_avg * 1.1:
                        trends[metric] = "improving"
                    elif recent_avg < older_avg * 0.9:
                        trends[metric] = "declining"
                    else:
                        trends[metric] = "stable"
            
            return trends
            
        except Exception as e:
            logger.warning(
                "Failed to analyze health trends",
                project_id=project_id,
                error=str(e)
            )
            return {"trend": "analysis_failed"}
    
    def _follows_naming_conventions(self, file_name: str) -> bool:
        """Check if file follows naming conventions."""
        # Simple check for common naming patterns
        name_without_ext = file_name.split(".")[0]
        
        # Check for consistent patterns
        if ("_" in name_without_ext and name_without_ext.islower()) or \
           (name_without_ext[0].isupper() and any(c.isupper() for c in name_without_ext[1:])):
            return True
        
        return False
    
    def _detect_simple_circular_deps(self, dependencies: List[Dict[str, Any]]) -> List[str]:
        """Detect simple circular dependencies."""
        # Simplified circular dependency detection
        file_deps = {}
        
        for dep in dependencies:
            source = dep.get("source_file_id")
            target = dep.get("target_file_id")
            if source and target:
                if source not in file_deps:
                    file_deps[source] = set()
                file_deps[source].add(target)
        
        circular = []
        for source, targets in file_deps.items():
            for target in targets:
                if target in file_deps and source in file_deps[target]:
                    circular.append(f"{source}-{target}")
        
        return circular
    
    async def _store_health_assessment(self, assessment: HealthAssessment) -> None:
        """Store health assessment for trend tracking."""
        try:
            # Store in Redis with timestamp
            assessment_key = f"health_assessment:{assessment.project_id}:{int(assessment.assessed_at.timestamp())}"
            await self.redis.setex(
                assessment_key,
                86400 * 30,  # 30 days
                json.dumps(assessment.to_dict())
            )
            
            # Add to project assessment index
            index_key = f"health_assessments:{assessment.project_id}"
            await self.redis.zadd(
                index_key,
                {assessment_key: assessment.assessed_at.timestamp()}
            )
            await self.redis.expire(index_key, 86400 * 30)
            
        except Exception as e:
            logger.warning(
                "Failed to store health assessment",
                assessment_id=assessment.assessment_id,
                error=str(e)
            )
    
    async def _get_historical_assessments(self, project_id: str) -> List[Dict[str, Any]]:
        """Get historical health assessments for trend analysis."""
        try:
            # Get assessment keys from index
            index_key = f"health_assessments:{project_id}"
            cutoff = datetime.utcnow() - timedelta(days=self.trend_analysis_days)
            
            assessment_keys = await self.redis.zrangebyscore(
                index_key,
                cutoff.timestamp(),
                "+inf"
            )
            
            # Retrieve assessments
            assessments = []
            for key in assessment_keys:
                assessment_data = await self.redis.get(key)
                if assessment_data:
                    try:
                        assessments.append(json.loads(assessment_data))
                    except json.JSONDecodeError:
                        continue
            
            return assessments
            
        except Exception as e:
            logger.warning(
                "Failed to get historical assessments",
                project_id=project_id,
                error=str(e)
            )
            return []
    
    async def _schedule_health_checks(
        self,
        session_id: str,
        project_id: str,
        duration_hours: int
    ) -> None:
        """Schedule periodic health checks for monitoring session."""
        # This would integrate with a task scheduler
        # For now, just log the scheduling
        logger.info(
            "Health checks scheduled",
            session_id=session_id,
            project_id=project_id,
            duration_hours=duration_hours,
            check_interval=self.health_check_interval
        )


class DocumentationAgent:
    """
    Intelligent documentation generation and maintenance agent.
    
    Provides automated documentation generation, outdated documentation detection,
    API documentation maintenance, and architectural documentation updates.
    """
    
    def __init__(
        self,
        session: AsyncSession,
        redis_client: RedisClient,
        context_integration: AgentContextIntegration
    ):
        self.session = session
        self.redis = redis_client
        self.context_integration = context_integration
        
        # Documentation configuration
        self.doc_cache_ttl = 3600  # 1 hour
        self.staleness_threshold_days = 30
    
    async def generate_project_documentation(
        self,
        project_id: str,
        doc_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive project documentation.
        
        Args:
            project_id: Project to document
            doc_types: Optional list of documentation types to generate
            
        Returns:
            Generated documentation content
        """
        if doc_types is None:
            doc_types = ["overview", "api", "architecture", "setup"]
        
        try:
            # Get project context
            context_request = ContextRequest(
                agent_id="documentation_agent",
                project_id=project_id,
                task_type=AgentTaskType.DOCUMENTATION,
                task_description="Generate project documentation",
                scope=ContextScope.FULL_PROJECT,
                include_dependencies=True
            )
            
            context = await self.context_integration.request_context(context_request)
            
            documentation = {}
            
            # Generate different types of documentation
            if "overview" in doc_types:
                documentation["overview"] = await self._generate_project_overview(context.context_data)
            
            if "api" in doc_types:
                documentation["api"] = await self._generate_api_documentation(context.context_data)
            
            if "architecture" in doc_types:
                documentation["architecture"] = await self._generate_architecture_documentation(context.context_data)
            
            if "setup" in doc_types:
                documentation["setup"] = await self._generate_setup_documentation(context.context_data)
            
            # Add metadata
            documentation["metadata"] = {
                "project_id": project_id,
                "generated_at": datetime.utcnow().isoformat(),
                "doc_types": doc_types,
                "files_analyzed": len(context.context_data.get("files", [])),
                "agent": "documentation_agent"
            }
            
            logger.info(
                "Project documentation generated",
                project_id=project_id,
                doc_types=doc_types,
                sections_generated=len([k for k in documentation.keys() if k != "metadata"])
            )
            
            return documentation
            
        except Exception as e:
            logger.error(
                "Documentation generation failed",
                project_id=project_id,
                error=str(e)
            )
            raise
    
    async def detect_outdated_documentation(
        self,
        project_id: str
    ) -> List[Dict[str, Any]]:
        """
        Detect outdated documentation that needs updates.
        
        Args:
            project_id: Project to analyze
            
        Returns:
            List of outdated documentation items
        """
        try:
            # Get documentation files
            doc_files = await self._get_documentation_files(project_id)
            
            outdated_items = []
            cutoff_date = datetime.utcnow() - timedelta(days=self.staleness_threshold_days)
            
            for doc_file in doc_files:
                last_modified = doc_file.last_modified or doc_file.created_at
                
                if last_modified < cutoff_date:
                    # Check if related source files have been modified more recently
                    related_changes = await self._find_related_source_changes(
                        project_id, doc_file, last_modified
                    )
                    
                    if related_changes:
                        outdated_items.append({
                            "doc_file": doc_file.relative_path,
                            "last_updated": last_modified.isoformat(),
                            "days_outdated": (datetime.utcnow() - last_modified).days,
                            "related_changes": related_changes,
                            "severity": self._calculate_staleness_severity(
                                (datetime.utcnow() - last_modified).days
                            ),
                            "update_recommendations": self._generate_update_recommendations(
                                doc_file, related_changes
                            )
                        })
            
            # Sort by severity
            outdated_items.sort(
                key=lambda x: {"critical": 4, "high": 3, "medium": 2, "low": 1}[x["severity"]],
                reverse=True
            )
            
            logger.info(
                "Outdated documentation detection completed",
                project_id=project_id,
                outdated_items_found=len(outdated_items)
            )
            
            return outdated_items
            
        except Exception as e:
            logger.error(
                "Outdated documentation detection failed",
                project_id=project_id,
                error=str(e)
            )
            raise
    
    # ================== PRIVATE METHODS ==================
    
    async def _generate_project_overview(self, context_data: Dict[str, Any]) -> str:
        """Generate project overview documentation."""
        project_info = context_data.get("project", {})
        files = context_data.get("files", [])
        
        overview = f"""# {project_info.get('name', 'Project')} Overview

## Description
{project_info.get('description', 'No description available.')}

## Project Statistics
- Total Files: {len(files)}
- Source Files: {len([f for f in files if f.get('file_type') == 'source'])}
- Test Files: {len([f for f in files if 'test' in f.get('relative_path', '').lower()])}
- Configuration Files: {len([f for f in files if f.get('file_type') == 'config'])}

## File Structure
"""
        
        # Add file structure
        directories = set()
        for file_data in files[:20]:  # Limit to first 20 files
            file_path = file_data.get("relative_path", "")
            directory = "/".join(file_path.split("/")[:-1])
            if directory:
                directories.add(directory)
        
        for directory in sorted(directories):
            overview += f"- `{directory}/`\n"
        
        return overview
    
    async def _generate_api_documentation(self, context_data: Dict[str, Any]) -> str:
        """Generate API documentation."""
        files = context_data.get("files", [])
        
        api_doc = "# API Documentation\n\n"
        
        # Look for API-related files
        api_files = [
            f for f in files
            if any(indicator in f.get("relative_path", "").lower()
                  for indicator in ["api", "endpoint", "route", "controller"])
        ]
        
        if not api_files:
            api_doc += "No API files detected in the project.\n"
            return api_doc
        
        api_doc += "## API Endpoints\n\n"
        
        for api_file in api_files:
            api_doc += f"### {api_file.get('file_name')}\n"
            api_doc += f"Location: `{api_file.get('relative_path')}`\n\n"
            
            # Extract API patterns from content
            content = api_file.get("content_preview", "")
            endpoints = self._extract_api_endpoints(content)
            
            for endpoint in endpoints:
                api_doc += f"- **{endpoint['method']}** `{endpoint['path']}` - {endpoint['description']}\n"
            
            api_doc += "\n"
        
        return api_doc
    
    async def _generate_architecture_documentation(self, context_data: Dict[str, Any]) -> str:
        """Generate architecture documentation."""
        files = context_data.get("files", [])
        dependencies = context_data.get("dependencies", [])
        
        arch_doc = "# Architecture Documentation\n\n"
        
        # Analyze architecture patterns
        file_types = {}
        for file_data in files:
            file_type = file_data.get("file_type", "unknown")
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        arch_doc += "## Project Structure\n\n"
        for file_type, count in file_types.items():
            arch_doc += f"- {file_type.title()} Files: {count}\n"
        
        arch_doc += "\n## Dependencies\n\n"
        if dependencies:
            external_deps = [d for d in dependencies if d.get("is_external", False)]
            internal_deps = [d for d in dependencies if not d.get("is_external", False)]
            
            arch_doc += f"- External Dependencies: {len(external_deps)}\n"
            arch_doc += f"- Internal Dependencies: {len(internal_deps)}\n"
            
            # List some key external dependencies
            arch_doc += "\n### Key External Dependencies\n"
            for dep in external_deps[:10]:  # Top 10
                arch_doc += f"- {dep.get('target_name', 'Unknown')}\n"
        else:
            arch_doc += "No dependencies detected.\n"
        
        return arch_doc
    
    async def _generate_setup_documentation(self, context_data: Dict[str, Any]) -> str:
        """Generate setup documentation."""
        files = context_data.get("files", [])
        
        setup_doc = "# Setup Documentation\n\n"
        
        # Look for setup-related files
        setup_files = [
            f for f in files
            if any(name in f.get("file_name", "").lower()
                  for name in ["readme", "setup", "install", "requirements", "package", "makefile"])
        ]
        
        if setup_files:
            setup_doc += "## Setup Files\n\n"
            for setup_file in setup_files:
                setup_doc += f"- `{setup_file.get('relative_path')}` - {setup_file.get('file_name')}\n"
        
        # Generate basic setup instructions
        setup_doc += "\n## Basic Setup\n\n"
        
        # Detect project type and add appropriate instructions
        if any("package.json" in f.get("file_name", "") for f in files):
            setup_doc += """### Node.js Project
```bash
npm install
npm start
```
"""
        
        if any("requirements.txt" in f.get("file_name", "") for f in files):
            setup_doc += """### Python Project
```bash
pip install -r requirements.txt
python main.py
```
"""
        
        if any("Dockerfile" in f.get("file_name", "") for f in files):
            setup_doc += """### Docker Setup
```bash
docker build -t project-name .
docker run -p 8080:8080 project-name
```
"""
        
        return setup_doc
    
    def _extract_api_endpoints(self, content: str) -> List[Dict[str, str]]:
        """Extract API endpoints from file content."""
        endpoints = []
        
        # Simple patterns for common API frameworks
        patterns = [
            (r'@app\.route\(["\']([^"\']+)["\'].*methods=\[["\']([^"\']+)["\']', "Flask"),
            (r'router\.([a-z]+)\(["\']([^"\']+)["\']', "Express"),
            (r'@([A-Z]+).*["\']([^"\']+)["\']', "Spring/FastAPI")
        ]
        
        for pattern, framework in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                if framework == "Express":
                    method = match.group(1).upper()
                    path = match.group(2)
                elif framework == "Flask":
                    path = match.group(1)
                    method = match.group(2).upper()
                else:
                    method = match.group(1)
                    path = match.group(2)
                
                endpoints.append({
                    "method": method,
                    "path": path,
                    "description": f"API endpoint ({framework})"
                })
        
        return endpoints[:10]  # Limit to 10 endpoints
    
    async def _get_documentation_files(self, project_id: str) -> List[FileEntry]:
        """Get documentation files for a project."""
        stmt = select(FileEntry).where(
            and_(
                FileEntry.project_id == project_id,
                or_(
                    FileEntry.file_type == "documentation",
                    FileEntry.file_extension.in_([".md", ".rst", ".txt"]),
                    FileEntry.file_name.like("%readme%"),
                    FileEntry.file_name.like("%doc%")
                )
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def _find_related_source_changes(
        self,
        project_id: str,
        doc_file: FileEntry,
        cutoff_date: datetime
    ) -> List[str]:
        """Find source files that changed after documentation was last updated."""
        # Get source files that were modified after the documentation
        stmt = select(FileEntry).where(
            and_(
                FileEntry.project_id == project_id,
                FileEntry.file_type == "source",
                FileEntry.last_modified > cutoff_date
            )
        )
        result = await self.session.execute(stmt)
        changed_files = result.scalars().all()
        
        return [f.relative_path for f in changed_files]
    
    def _calculate_staleness_severity(self, days_outdated: int) -> str:
        """Calculate severity of documentation staleness."""
        if days_outdated > 90:
            return "critical"
        elif days_outdated > 60:
            return "high"
        elif days_outdated > 30:
            return "medium"
        else:
            return "low"
    
    def _generate_update_recommendations(
        self,
        doc_file: FileEntry,
        related_changes: List[str]
    ) -> List[str]:
        """Generate recommendations for updating documentation."""
        recommendations = []
        
        recommendations.append(f"Review and update {doc_file.file_name}")
        
        if related_changes:
            recommendations.append(f"Check changes in: {', '.join(related_changes[:3])}")
            if len(related_changes) > 3:
                recommendations.append(f"And {len(related_changes) - 3} other files")
        
        if "api" in doc_file.file_name.lower():
            recommendations.append("Verify API endpoint documentation is current")
        
        if "readme" in doc_file.file_name.lower():
            recommendations.append("Update setup instructions if needed")
        
        return recommendations


# Factory functions for dependency injection
async def get_code_intelligence_agent(
    session: AsyncSession = None,
    redis_client: RedisClient = None,
    context_integration: AgentContextIntegration = None
) -> CodeIntelligenceAgent:
    """Factory function to create CodeIntelligenceAgent instance."""
    if session is None:
        session = await get_session()
    if redis_client is None:
        redis_client = await get_redis_client()
    if context_integration is None:
        from .context_integration import get_agent_context_integration
        context_integration = await get_agent_context_integration(session, redis_client)
    
    return CodeIntelligenceAgent(session, redis_client, context_integration)


async def get_context_aware_qa_agent(
    session: AsyncSession = None,
    redis_client: RedisClient = None,
    context_integration: AgentContextIntegration = None
) -> ContextAwareQAAgent:
    """Factory function to create ContextAwareQAAgent instance."""
    if session is None:
        session = await get_session()
    if redis_client is None:
        redis_client = await get_redis_client()
    if context_integration is None:
        from .context_integration import get_agent_context_integration
        context_integration = await get_agent_context_integration(session, redis_client)
    
    return ContextAwareQAAgent(session, redis_client, context_integration)


async def get_project_health_agent(
    session: AsyncSession = None,
    redis_client: RedisClient = None,
    context_integration: AgentContextIntegration = None
) -> ProjectHealthAgent:
    """Factory function to create ProjectHealthAgent instance."""
    if session is None:
        session = await get_session()
    if redis_client is None:
        redis_client = await get_redis_client()
    if context_integration is None:
        from .context_integration import get_agent_context_integration
        context_integration = await get_agent_context_integration(session, redis_client)
    
    return ProjectHealthAgent(session, redis_client, context_integration)


async def get_documentation_agent(
    session: AsyncSession = None,
    redis_client: RedisClient = None,
    context_integration: AgentContextIntegration = None
) -> DocumentationAgent:
    """Factory function to create DocumentationAgent instance."""
    if session is None:
        session = await get_session()
    if redis_client is None:
        redis_client = await get_redis_client()
    if context_integration is None:
        from .context_integration import get_agent_context_integration
        context_integration = await get_agent_context_integration(session, redis_client)
    
    return DocumentationAgent(session, redis_client, context_integration)