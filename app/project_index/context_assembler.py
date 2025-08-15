"""
Context Assembly Engine for LeanVibe Agent Hive 2.0

Advanced context assembly system that organizes and formats optimized file selections
into coherent contexts for AI agents. Supports multiple assembly strategies including
hierarchical organization, dependency-driven assembly, and task-focused optimization.
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import structlog

from .context_optimizer import RelevanceScore, ContextRequest, TaskType
from .models import FileAnalysisResult
from .graph import DependencyGraph
from .utils import PathUtils, FileUtils

logger = structlog.get_logger()


class AssemblyStrategy(Enum):
    """Context assembly strategies."""
    HIERARCHICAL = "hierarchical"
    DEPENDENCY_FIRST = "dependency_first"
    TASK_FOCUSED = "task_focused"
    BALANCED = "balanced"
    STREAMING = "streaming"
    LAYERED = "layered"


class ContextFormat(Enum):
    """Context output formats."""
    STRUCTURED = "structured"
    NARRATIVE = "narrative"
    GRAPH = "graph"
    MINIMAL = "minimal"
    MARKDOWN = "markdown"


@dataclass
class ContextLayer:
    """A layer in hierarchical context organization."""
    layer_name: str
    layer_type: str
    files: List[RelevanceScore]
    description: str
    priority: int
    estimated_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssembledContext:
    """Complete assembled context ready for AI consumption."""
    strategy: AssemblyStrategy
    format: ContextFormat
    layers: List[ContextLayer]
    dependency_map: Dict[str, List[str]]
    navigation_structure: Dict[str, Any]
    context_summary: str
    total_tokens: int
    assembly_metadata: Dict[str, Any]
    recommendations: List[str]


@dataclass
class AssemblyConfiguration:
    """Configuration for context assembly."""
    strategy: AssemblyStrategy = AssemblyStrategy.BALANCED
    format: ContextFormat = ContextFormat.STRUCTURED
    max_tokens: int = 32000
    max_layers: int = 5
    include_navigation: bool = True
    include_summaries: bool = True
    token_buffer: int = 2000
    prioritize_core_files: bool = True
    assembly_options: Dict[str, Any] = field(default_factory=dict)


class ContextAssembler:
    """
    Advanced context assembly engine with multiple strategies.
    
    Provides intelligent organization of selected files into coherent
    contexts optimized for AI agent understanding and task execution.
    """
    
    def __init__(self):
        """Initialize ContextAssembler."""
        self.assembly_stats = {
            "contexts_assembled": 0,
            "avg_assembly_time": 0.0,
            "strategy_usage": defaultdict(int),
            "format_usage": defaultdict(int)
        }
        
        # Token estimation coefficients
        self.token_coefficients = {
            "base_overhead": 50,
            "per_file_overhead": 25,
            "summary_factor": 0.1,
            "navigation_factor": 0.05,
            "dependency_factor": 0.03
        }
    
    async def assemble_context(
        self,
        relevance_scores: List[RelevanceScore],
        context_request: ContextRequest,
        dependency_graph: DependencyGraph,
        config: AssemblyConfiguration,
        file_results: List[FileAnalysisResult] = None
    ) -> AssembledContext:
        """
        Assemble optimized context from relevance scores.
        
        Args:
            relevance_scores: List of file relevance scores
            context_request: Original context request
            dependency_graph: Project dependency graph
            config: Assembly configuration
            file_results: Optional file analysis results for content
            
        Returns:
            AssembledContext ready for AI consumption
        """
        start_time = time.time()
        
        logger.info("Starting context assembly",
                   strategy=config.strategy.value,
                   format=config.format.value,
                   file_count=len(relevance_scores))
        
        try:
            # Apply token budget constraints
            constrained_scores = await self._apply_token_constraints(
                relevance_scores, config
            )
            
            # Apply assembly strategy
            layers = await self._apply_assembly_strategy(
                constrained_scores, context_request, dependency_graph, config
            )
            
            # Create dependency map
            dependency_map = await self._create_dependency_map(
                constrained_scores, dependency_graph
            )
            
            # Build navigation structure
            navigation_structure = await self._build_navigation_structure(
                layers, dependency_map, config
            )
            
            # Generate context summary
            context_summary = await self._generate_context_summary(
                layers, context_request, config
            )
            
            # Calculate total tokens
            total_tokens = self._calculate_total_tokens(layers, config)
            
            # Create assembly metadata
            assembly_time = time.time() - start_time
            assembly_metadata = await self._create_assembly_metadata(
                config, constrained_scores, assembly_time
            )
            
            # Generate recommendations
            recommendations = await self._generate_assembly_recommendations(
                layers, context_request, config
            )
            
            # Create assembled context
            assembled_context = AssembledContext(
                strategy=config.strategy,
                format=config.format,
                layers=layers,
                dependency_map=dependency_map,
                navigation_structure=navigation_structure,
                context_summary=context_summary,
                total_tokens=total_tokens,
                assembly_metadata=assembly_metadata,
                recommendations=recommendations
            )
            
            # Update statistics
            self._update_assembly_stats(config, assembly_time)
            
            logger.info("Context assembly completed",
                       strategy=config.strategy.value,
                       layers=len(layers),
                       total_tokens=total_tokens,
                       assembly_time=assembly_time)
            
            return assembled_context
            
        except Exception as e:
            logger.error("Context assembly failed",
                        strategy=config.strategy.value,
                        error=str(e))
            raise
    
    async def _apply_token_constraints(
        self,
        relevance_scores: List[RelevanceScore],
        config: AssemblyConfiguration
    ) -> List[RelevanceScore]:
        """Apply token budget constraints to file selection."""
        # Reserve tokens for overhead
        available_tokens = config.max_tokens - config.token_buffer
        
        # Sort by relevance score
        sorted_scores = sorted(
            relevance_scores, 
            key=lambda x: x.relevance_score, 
            reverse=True
        )
        
        # Select files within token budget
        constrained_scores = []
        total_tokens = 0
        
        for score in sorted_scores:
            # Estimate overhead tokens for this file
            overhead = self.token_coefficients["per_file_overhead"]
            file_tokens = score.estimated_tokens + overhead
            
            if total_tokens + file_tokens <= available_tokens:
                constrained_scores.append(score)
                total_tokens += file_tokens
            else:
                # Check if we can include a smaller portion
                remaining_tokens = available_tokens - total_tokens
                if remaining_tokens > overhead + 100:  # Minimum viable content
                    # Create truncated version
                    truncated_score = self._create_truncated_score(
                        score, remaining_tokens - overhead
                    )
                    constrained_scores.append(truncated_score)
                break
        
        logger.debug("Applied token constraints",
                    original_files=len(relevance_scores),
                    constrained_files=len(constrained_scores),
                    estimated_tokens=total_tokens)
        
        return constrained_scores
    
    async def _apply_assembly_strategy(
        self,
        relevance_scores: List[RelevanceScore],
        context_request: ContextRequest,
        dependency_graph: DependencyGraph,
        config: AssemblyConfiguration
    ) -> List[ContextLayer]:
        """Apply the specified assembly strategy."""
        strategy_map = {
            AssemblyStrategy.HIERARCHICAL: self._hierarchical_assembly,
            AssemblyStrategy.DEPENDENCY_FIRST: self._dependency_first_assembly,
            AssemblyStrategy.TASK_FOCUSED: self._task_focused_assembly,
            AssemblyStrategy.BALANCED: self._balanced_assembly,
            AssemblyStrategy.STREAMING: self._streaming_assembly,
            AssemblyStrategy.LAYERED: self._layered_assembly
        }
        
        assembly_func = strategy_map.get(
            config.strategy, 
            self._balanced_assembly
        )
        
        return await assembly_func(
            relevance_scores, context_request, dependency_graph, config
        )
    
    async def _hierarchical_assembly(
        self,
        relevance_scores: List[RelevanceScore],
        context_request: ContextRequest,
        dependency_graph: DependencyGraph,
        config: AssemblyConfiguration
    ) -> List[ContextLayer]:
        """Organize context hierarchically by importance and structure."""
        layers = []
        
        # Layer 1: Core Entry Points
        entry_points = self._identify_entry_points(relevance_scores)
        if entry_points:
            layers.append(ContextLayer(
                layer_name="Entry Points",
                layer_type="core",
                files=entry_points,
                description="Main entry points and orchestrator files",
                priority=1,
                estimated_tokens=sum(f.estimated_tokens for f in entry_points),
                metadata={"file_count": len(entry_points)}
            ))
        
        # Layer 2: High-Relevance Core Files
        core_files = [
            score for score in relevance_scores 
            if score.relevance_score >= 0.7 and score not in entry_points
        ]
        if core_files:
            layers.append(ContextLayer(
                layer_name="Core Implementation",
                layer_type="core",
                files=core_files,
                description="High-relevance implementation files",
                priority=2,
                estimated_tokens=sum(f.estimated_tokens for f in core_files),
                metadata={"file_count": len(core_files)}
            ))
        
        # Layer 3: Supporting Dependencies
        supporting_files = [
            score for score in relevance_scores
            if 0.4 <= score.relevance_score < 0.7
        ]
        if supporting_files:
            # Group by directory structure
            directory_groups = self._group_by_directory(supporting_files)
            
            for directory, files in directory_groups.items():
                layers.append(ContextLayer(
                    layer_name=f"Supporting: {directory or 'Root'}",
                    layer_type="supporting",
                    files=files,
                    description=f"Supporting files in {directory or 'root'} directory",
                    priority=3,
                    estimated_tokens=sum(f.estimated_tokens for f in files),
                    metadata={"directory": directory, "file_count": len(files)}
                ))
        
        # Layer 4: Configuration and Utilities
        config_files = [
            score for score in relevance_scores
            if self._is_config_or_utility_file(score.file_path)
        ]
        if config_files:
            layers.append(ContextLayer(
                layer_name="Configuration & Utilities",
                layer_type="utility",
                files=config_files,
                description="Configuration files and utility functions",
                priority=4,
                estimated_tokens=sum(f.estimated_tokens for f in config_files),
                metadata={"file_count": len(config_files)}
            ))
        
        # Layer 5: Tests and Documentation
        test_doc_files = [
            score for score in relevance_scores
            if self._is_test_or_doc_file(score.file_path)
        ]
        if test_doc_files and context_request.context_preferences.get("include_tests", False):
            layers.append(ContextLayer(
                layer_name="Tests & Documentation",
                layer_type="auxiliary",
                files=test_doc_files,
                description="Test files and documentation",
                priority=5,
                estimated_tokens=sum(f.estimated_tokens for f in test_doc_files),
                metadata={"file_count": len(test_doc_files)}
            ))
        
        return layers
    
    async def _dependency_first_assembly(
        self,
        relevance_scores: List[RelevanceScore],
        context_request: ContextRequest,
        dependency_graph: DependencyGraph,
        config: AssemblyConfiguration
    ) -> List[ContextLayer]:
        """Organize context starting with dependencies and building outward."""
        layers = []
        
        # Build dependency levels using topological sort
        dependency_levels = self._calculate_dependency_levels(
            relevance_scores, dependency_graph
        )
        
        for level, files in enumerate(dependency_levels):
            if not files:
                continue
            
            layer_name = f"Dependency Level {level + 1}"
            layer_type = "dependency"
            description = f"Files at dependency level {level + 1}"
            
            if level == 0:
                layer_name = "Foundation Dependencies"
                description = "Core dependencies with no internal dependencies"
            elif level == len(dependency_levels) - 1:
                layer_name = "Top-Level Components"
                description = "High-level components that depend on others"
            
            layers.append(ContextLayer(
                layer_name=layer_name,
                layer_type=layer_type,
                files=files,
                description=description,
                priority=level + 1,
                estimated_tokens=sum(f.estimated_tokens for f in files),
                metadata={"dependency_level": level, "file_count": len(files)}
            ))
        
        return layers
    
    async def _task_focused_assembly(
        self,
        relevance_scores: List[RelevanceScore],
        context_request: ContextRequest,
        dependency_graph: DependencyGraph,
        config: AssemblyConfiguration
    ) -> List[ContextLayer]:
        """Organize context based on task-specific priorities."""
        layers = []
        
        # Task-specific organization
        if context_request.task_type == TaskType.FEATURE:
            layers = await self._feature_focused_layers(
                relevance_scores, context_request, dependency_graph
            )
        elif context_request.task_type == TaskType.BUGFIX:
            layers = await self._bugfix_focused_layers(
                relevance_scores, context_request, dependency_graph
            )
        elif context_request.task_type == TaskType.REFACTORING:
            layers = await self._refactoring_focused_layers(
                relevance_scores, context_request, dependency_graph
            )
        elif context_request.task_type == TaskType.ANALYSIS:
            layers = await self._analysis_focused_layers(
                relevance_scores, context_request, dependency_graph
            )
        else:
            # Default to balanced approach
            layers = await self._balanced_assembly(
                relevance_scores, context_request, dependency_graph, config
            )
        
        return layers
    
    async def _balanced_assembly(
        self,
        relevance_scores: List[RelevanceScore],
        context_request: ContextRequest,
        dependency_graph: DependencyGraph,
        config: AssemblyConfiguration
    ) -> List[ContextLayer]:
        """Balanced assembly combining multiple organizational principles."""
        layers = []
        
        # Sort files by relevance
        sorted_scores = sorted(
            relevance_scores, 
            key=lambda x: x.relevance_score, 
            reverse=True
        )
        
        # Layer 1: Highest Relevance (Top 20% or minimum 3 files)
        top_count = max(3, len(sorted_scores) // 5)
        top_files = sorted_scores[:top_count]
        
        layers.append(ContextLayer(
            layer_name="Primary Focus",
            layer_type="primary",
            files=top_files,
            description="Highest relevance files for immediate attention",
            priority=1,
            estimated_tokens=sum(f.estimated_tokens for f in top_files),
            metadata={"file_count": len(top_files), "avg_relevance": sum(f.relevance_score for f in top_files) / len(top_files)}
        ))
        
        # Layer 2: Functional Groups
        remaining_files = sorted_scores[top_count:]
        functional_groups = self._group_by_functionality(remaining_files)
        
        for group_name, files in functional_groups.items():
            if files:
                layers.append(ContextLayer(
                    layer_name=f"{group_name.title()} Components",
                    layer_type="functional",
                    files=files,
                    description=f"Files related to {group_name} functionality",
                    priority=2,
                    estimated_tokens=sum(f.estimated_tokens for f in files),
                    metadata={"functionality": group_name, "file_count": len(files)}
                ))
        
        # Layer 3: Directory Structure (remaining files)
        ungrouped_files = [
            f for f in remaining_files 
            if not any(f in group_files for group_files in functional_groups.values())
        ]
        
        if ungrouped_files:
            directory_groups = self._group_by_directory(ungrouped_files)
            
            for directory, files in directory_groups.items():
                layers.append(ContextLayer(
                    layer_name=f"Directory: {directory or 'Root'}",
                    layer_type="structural",
                    files=files,
                    description=f"Files in {directory or 'root'} directory",
                    priority=3,
                    estimated_tokens=sum(f.estimated_tokens for f in files),
                    metadata={"directory": directory, "file_count": len(files)}
                ))
        
        return layers
    
    async def _streaming_assembly(
        self,
        relevance_scores: List[RelevanceScore],
        context_request: ContextRequest,
        dependency_graph: DependencyGraph,
        config: AssemblyConfiguration
    ) -> List[ContextLayer]:
        """Streaming assembly for progressive context building."""
        layers = []
        
        # Sort by relevance for streaming order
        sorted_scores = sorted(
            relevance_scores, 
            key=lambda x: x.relevance_score, 
            reverse=True
        )
        
        # Create streaming chunks
        chunk_size = max(3, len(sorted_scores) // 4)
        
        for i, chunk_start in enumerate(range(0, len(sorted_scores), chunk_size)):
            chunk_files = sorted_scores[chunk_start:chunk_start + chunk_size]
            
            if not chunk_files:
                continue
            
            layers.append(ContextLayer(
                layer_name=f"Context Chunk {i + 1}",
                layer_type="stream",
                files=chunk_files,
                description=f"Streaming context chunk {i + 1} - relevance {chunk_files[0].relevance_score:.2f} to {chunk_files[-1].relevance_score:.2f}",
                priority=i + 1,
                estimated_tokens=sum(f.estimated_tokens for f in chunk_files),
                metadata={
                    "chunk_index": i,
                    "chunk_size": len(chunk_files),
                    "min_relevance": chunk_files[-1].relevance_score,
                    "max_relevance": chunk_files[0].relevance_score
                }
            ))
        
        return layers
    
    async def _layered_assembly(
        self,
        relevance_scores: List[RelevanceScore],
        context_request: ContextRequest,
        dependency_graph: DependencyGraph,
        config: AssemblyConfiguration
    ) -> List[ContextLayer]:
        """Layered assembly with architectural considerations."""
        layers = []
        
        # Identify architectural layers
        arch_layers = self._identify_architectural_layers(relevance_scores)
        
        layer_order = [
            ("data", "Data Layer"),
            ("service", "Service Layer"), 
            ("controller", "Controller Layer"),
            ("presentation", "Presentation Layer"),
            ("config", "Configuration Layer"),
            ("test", "Test Layer")
        ]
        
        for layer_key, layer_name in layer_order:
            layer_files = arch_layers.get(layer_key, [])
            
            if layer_files:
                layers.append(ContextLayer(
                    layer_name=layer_name,
                    layer_type="architectural",
                    files=layer_files,
                    description=f"Files in the {layer_name.lower()}",
                    priority=len(layers) + 1,
                    estimated_tokens=sum(f.estimated_tokens for f in layer_files),
                    metadata={"architectural_layer": layer_key, "file_count": len(layer_files)}
                ))
        
        # Add remaining files as "Other"
        all_layered_files = set()
        for layer_files in arch_layers.values():
            all_layered_files.update(f.file_path for f in layer_files)
        
        other_files = [
            score for score in relevance_scores
            if score.file_path not in all_layered_files
        ]
        
        if other_files:
            layers.append(ContextLayer(
                layer_name="Other Components",
                layer_type="other",
                files=other_files,
                description="Files not fitting standard architectural layers",
                priority=len(layers) + 1,
                estimated_tokens=sum(f.estimated_tokens for f in other_files),
                metadata={"file_count": len(other_files)}
            ))
        
        return layers
    
    # Task-specific assembly methods
    
    async def _feature_focused_layers(
        self,
        relevance_scores: List[RelevanceScore],
        context_request: ContextRequest,
        dependency_graph: DependencyGraph
    ) -> List[ContextLayer]:
        """Create layers focused on feature implementation."""
        layers = []
        
        # Layer 1: Similar existing features
        similar_features = self._find_similar_features(relevance_scores, context_request)
        if similar_features:
            layers.append(ContextLayer(
                layer_name="Similar Features",
                layer_type="reference",
                files=similar_features,
                description="Existing features with similar patterns",
                priority=1,
                estimated_tokens=sum(f.estimated_tokens for f in similar_features)
            ))
        
        # Layer 2: Core implementation files
        core_files = [f for f in relevance_scores if f.relevance_score >= 0.6]
        if core_files:
            layers.append(ContextLayer(
                layer_name="Implementation Core",
                layer_type="implementation",
                files=core_files,
                description="Core files for feature implementation",
                priority=2,
                estimated_tokens=sum(f.estimated_tokens for f in core_files)
            ))
        
        # Layer 3: Integration points
        integration_files = self._find_integration_points(relevance_scores, dependency_graph)
        if integration_files:
            layers.append(ContextLayer(
                layer_name="Integration Points",
                layer_type="integration",
                files=integration_files,
                description="Files requiring integration with new feature",
                priority=3,
                estimated_tokens=sum(f.estimated_tokens for f in integration_files)
            ))
        
        return layers
    
    async def _bugfix_focused_layers(
        self,
        relevance_scores: List[RelevanceScore],
        context_request: ContextRequest,
        dependency_graph: DependencyGraph
    ) -> List[ContextLayer]:
        """Create layers focused on bug investigation and fixing."""
        layers = []
        
        # Layer 1: Error-prone files
        error_prone = self._identify_error_prone_files(relevance_scores)
        if error_prone:
            layers.append(ContextLayer(
                layer_name="Error-Prone Areas",
                layer_type="investigation",
                files=error_prone,
                description="Files with history of bugs or high complexity",
                priority=1,
                estimated_tokens=sum(f.estimated_tokens for f in error_prone)
            ))
        
        # Layer 2: Recent changes
        recent_changes = self._identify_recent_changes(relevance_scores)
        if recent_changes:
            layers.append(ContextLayer(
                layer_name="Recent Changes",
                layer_type="investigation",
                files=recent_changes,
                description="Recently modified files that might contain bugs",
                priority=2,
                estimated_tokens=sum(f.estimated_tokens for f in recent_changes)
            ))
        
        # Layer 3: Related test files
        test_files = [f for f in relevance_scores if self._is_test_or_doc_file(f.file_path)]
        if test_files:
            layers.append(ContextLayer(
                layer_name="Related Tests",
                layer_type="validation",
                files=test_files,
                description="Test files for validation and reproduction",
                priority=3,
                estimated_tokens=sum(f.estimated_tokens for f in test_files)
            ))
        
        return layers
    
    async def _refactoring_focused_layers(
        self,
        relevance_scores: List[RelevanceScore],
        context_request: ContextRequest,
        dependency_graph: DependencyGraph
    ) -> List[ContextLayer]:
        """Create layers focused on refactoring analysis."""
        layers = []
        
        # Layer 1: High coupling files
        high_coupling = self._identify_high_coupling_files(relevance_scores, dependency_graph)
        if high_coupling:
            layers.append(ContextLayer(
                layer_name="High Coupling",
                layer_type="refactoring_target",
                files=high_coupling,
                description="Files with high coupling requiring refactoring",
                priority=1,
                estimated_tokens=sum(f.estimated_tokens for f in high_coupling)
            ))
        
        # Layer 2: Impact analysis
        impact_files = self._identify_refactoring_impact(relevance_scores, dependency_graph)
        if impact_files:
            layers.append(ContextLayer(
                layer_name="Impact Analysis",
                layer_type="impact",
                files=impact_files,
                description="Files that will be affected by refactoring",
                priority=2,
                estimated_tokens=sum(f.estimated_tokens for f in impact_files)
            ))
        
        # Layer 3: Architectural boundaries
        boundary_files = self._identify_architectural_boundaries(relevance_scores)
        if boundary_files:
            layers.append(ContextLayer(
                layer_name="Architectural Boundaries",
                layer_type="architecture",
                files=boundary_files,
                description="Files defining architectural boundaries",
                priority=3,
                estimated_tokens=sum(f.estimated_tokens for f in boundary_files)
            ))
        
        return layers
    
    async def _analysis_focused_layers(
        self,
        relevance_scores: List[RelevanceScore],
        context_request: ContextRequest,
        dependency_graph: DependencyGraph
    ) -> List[ContextLayer]:
        """Create layers focused on comprehensive analysis."""
        layers = []
        
        # Layer 1: System overview
        overview_files = self._identify_overview_files(relevance_scores)
        if overview_files:
            layers.append(ContextLayer(
                layer_name="System Overview",
                layer_type="overview",
                files=overview_files,
                description="Files providing system overview and entry points",
                priority=1,
                estimated_tokens=sum(f.estimated_tokens for f in overview_files)
            ))
        
        # Layer 2: Core components
        core_components = [f for f in relevance_scores if f.relevance_score >= 0.5]
        if core_components:
            layers.append(ContextLayer(
                layer_name="Core Components",
                layer_type="core",
                files=core_components,
                description="Core system components for analysis",
                priority=2,
                estimated_tokens=sum(f.estimated_tokens for f in core_components)
            ))
        
        # Layer 3: Supporting infrastructure
        supporting_files = [f for f in relevance_scores if f.relevance_score < 0.5]
        if supporting_files:
            layers.append(ContextLayer(
                layer_name="Supporting Infrastructure",
                layer_type="supporting",
                files=supporting_files,
                description="Supporting files and infrastructure",
                priority=3,
                estimated_tokens=sum(f.estimated_tokens for f in supporting_files)
            ))
        
        return layers
    
    # Helper methods
    
    def _create_truncated_score(self, score: RelevanceScore, max_tokens: int) -> RelevanceScore:
        """Create truncated version of relevance score within token limit."""
        # Create a copy with reduced token estimate
        truncated_score = RelevanceScore(
            file_path=score.file_path,
            relevance_score=score.relevance_score,
            confidence_score=score.confidence_score * 0.8,  # Reduce confidence for truncated
            relevance_reasons=score.relevance_reasons + ["Content truncated to fit token budget"],
            content_summary=score.content_summary + " [TRUNCATED]",
            key_functions=score.key_functions[:5],  # Limit key functions
            key_classes=score.key_classes[:5],  # Limit key classes
            import_relationships=score.import_relationships[:10],  # Limit imports
            estimated_tokens=max_tokens,
            metadata={**score.metadata, "truncated": True, "original_tokens": score.estimated_tokens}
        )
        
        return truncated_score
    
    def _identify_entry_points(self, relevance_scores: List[RelevanceScore]) -> List[RelevanceScore]:
        """Identify entry point files."""
        entry_points = []
        
        for score in relevance_scores:
            filename = Path(score.file_path).name.lower()
            
            # Common entry point patterns
            if any(pattern in filename for pattern in [
                'main', 'index', 'app', 'server', 'run', 'start', 'cli', '__main__'
            ]):
                entry_points.append(score)
            
            # Files with main functions (from analysis)
            if 'main' in score.key_functions:
                entry_points.append(score)
        
        # Remove duplicates and sort by relevance
        unique_entry_points = list({score.file_path: score for score in entry_points}.values())
        return sorted(unique_entry_points, key=lambda x: x.relevance_score, reverse=True)
    
    def _group_by_directory(self, relevance_scores: List[RelevanceScore]) -> Dict[str, List[RelevanceScore]]:
        """Group files by directory structure."""
        directory_groups = defaultdict(list)
        
        for score in relevance_scores:
            directory = '/'.join(Path(score.file_path).parts[:-1])
            directory_groups[directory].append(score)
        
        return dict(directory_groups)
    
    def _group_by_functionality(self, relevance_scores: List[RelevanceScore]) -> Dict[str, List[RelevanceScore]]:
        """Group files by functionality based on naming patterns."""
        functionality_groups = defaultdict(list)
        
        functionality_keywords = {
            'api': ['api', 'endpoint', 'route', 'controller', 'handler'],
            'database': ['db', 'database', 'model', 'schema', 'migration', 'sql'],
            'auth': ['auth', 'login', 'password', 'token', 'session', 'user'],
            'utils': ['util', 'helper', 'common', 'shared', 'tool'],
            'config': ['config', 'setting', 'env', 'environment'],
            'test': ['test', 'spec', 'mock', 'fixture'],
            'ui': ['ui', 'view', 'component', 'render', 'display']
        }
        
        for score in relevance_scores:
            file_path_lower = score.file_path.lower()
            
            # Check which functionality this file belongs to
            matched_functionality = None
            for functionality, keywords in functionality_keywords.items():
                if any(keyword in file_path_lower for keyword in keywords):
                    matched_functionality = functionality
                    break
            
            if matched_functionality:
                functionality_groups[matched_functionality].append(score)
            else:
                functionality_groups['other'].append(score)
        
        # Remove empty groups
        return {k: v for k, v in functionality_groups.items() if v}
    
    def _is_config_or_utility_file(self, file_path: str) -> bool:
        """Check if file is configuration or utility."""
        path_lower = file_path.lower()
        return any(pattern in path_lower for pattern in [
            'config', 'setting', 'env', 'util', 'helper', 'common', 'shared'
        ])
    
    def _is_test_or_doc_file(self, file_path: str) -> bool:
        """Check if file is test or documentation."""
        path_lower = file_path.lower()
        return any(pattern in path_lower for pattern in [
            'test', 'spec', 'doc', 'readme', 'md', '__test__', '__spec__'
        ])
    
    def _calculate_dependency_levels(
        self, 
        relevance_scores: List[RelevanceScore], 
        dependency_graph: DependencyGraph
    ) -> List[List[RelevanceScore]]:
        """Calculate dependency levels using topological sort."""
        # Create file lookup
        file_lookup = {score.file_path: score for score in relevance_scores}
        
        # Calculate in-degrees
        in_degrees = {}
        for score in relevance_scores:
            deps = dependency_graph.get_dependencies(score.file_path)
            # Only count dependencies that are in our relevance_scores
            internal_deps = [dep for dep in deps if dep in file_lookup]
            in_degrees[score.file_path] = len(internal_deps)
        
        # Topological sort to create levels
        levels = []
        remaining_files = set(score.file_path for score in relevance_scores)
        
        while remaining_files:
            # Find files with no dependencies in remaining set
            current_level = []
            for file_path in list(remaining_files):
                if in_degrees[file_path] == 0:
                    current_level.append(file_lookup[file_path])
                    remaining_files.remove(file_path)
            
            if not current_level:
                # Circular dependency - add all remaining files
                current_level = [file_lookup[fp] for fp in remaining_files]
                remaining_files.clear()
            
            levels.append(current_level)
            
            # Update in-degrees for next iteration
            for score in current_level:
                dependents = dependency_graph.get_dependents(score.file_path)
                for dependent in dependents:
                    if dependent in in_degrees:
                        in_degrees[dependent] -= 1
        
        return levels
    
    def _identify_architectural_layers(self, relevance_scores: List[RelevanceScore]) -> Dict[str, List[RelevanceScore]]:
        """Identify files belonging to different architectural layers."""
        layers = defaultdict(list)
        
        layer_patterns = {
            'data': ['model', 'entity', 'repository', 'dao', 'data', 'schema'],
            'service': ['service', 'business', 'logic', 'manager', 'processor'],
            'controller': ['controller', 'handler', 'endpoint', 'route', 'api'],
            'presentation': ['view', 'ui', 'component', 'template', 'render'],
            'config': ['config', 'setting', 'env', 'properties'],
            'test': ['test', 'spec', 'mock', '__test__']
        }
        
        for score in relevance_scores:
            path_lower = score.file_path.lower()
            
            # Check which layer this file belongs to
            for layer, patterns in layer_patterns.items():
                if any(pattern in path_lower for pattern in patterns):
                    layers[layer].append(score)
                    break
        
        return dict(layers)
    
    # Task-specific helper methods
    
    def _find_similar_features(
        self, 
        relevance_scores: List[RelevanceScore], 
        context_request: ContextRequest
    ) -> List[RelevanceScore]:
        """Find files implementing similar features."""
        # Simple heuristic based on file naming and task description
        task_keywords = set(context_request.task_description.lower().split())
        similar_features = []
        
        for score in relevance_scores:
            file_keywords = set(score.file_path.lower().replace('/', ' ').replace('_', ' ').split())
            
            # Check for keyword overlap
            if len(task_keywords.intersection(file_keywords)) >= 2:
                similar_features.append(score)
        
        return sorted(similar_features, key=lambda x: x.relevance_score, reverse=True)[:5]
    
    def _find_integration_points(
        self, 
        relevance_scores: List[RelevanceScore], 
        dependency_graph: DependencyGraph
    ) -> List[RelevanceScore]:
        """Find files that are integration points."""
        integration_files = []
        
        for score in relevance_scores:
            # Files with many dependencies or dependents are integration points
            deps = len(dependency_graph.get_dependencies(score.file_path))
            dependents = len(dependency_graph.get_dependents(score.file_path))
            
            if deps + dependents > 5:  # Arbitrary threshold
                integration_files.append(score)
        
        return sorted(integration_files, key=lambda x: x.relevance_score, reverse=True)
    
    def _identify_error_prone_files(self, relevance_scores: List[RelevanceScore]) -> List[RelevanceScore]:
        """Identify files that are likely to contain errors."""
        error_prone = []
        
        for score in relevance_scores:
            # Heuristics for error-prone files
            is_error_prone = False
            
            # Large files
            if score.estimated_tokens > 2000:
                is_error_prone = True
            
            # Files with many functions/classes
            if len(score.key_functions) > 10 or len(score.key_classes) > 5:
                is_error_prone = True
            
            # Files with complex imports
            if len(score.import_relationships) > 15:
                is_error_prone = True
            
            if is_error_prone:
                error_prone.append(score)
        
        return error_prone
    
    def _identify_recent_changes(self, relevance_scores: List[RelevanceScore]) -> List[RelevanceScore]:
        """Identify recently changed files (simulated)."""
        # In real implementation, integrate with Git history
        # For now, use heuristics
        recent_changes = []
        
        for score in relevance_scores:
            # Simulate: files with certain patterns are "recently changed"
            if any(pattern in score.file_path.lower() for pattern in ['new', 'recent', 'temp', 'wip']):
                recent_changes.append(score)
        
        return recent_changes
    
    def _identify_high_coupling_files(
        self, 
        relevance_scores: List[RelevanceScore], 
        dependency_graph: DependencyGraph
    ) -> List[RelevanceScore]:
        """Identify files with high coupling."""
        high_coupling = []
        
        for score in relevance_scores:
            # Count total connections
            deps = len(dependency_graph.get_dependencies(score.file_path))
            dependents = len(dependency_graph.get_dependents(score.file_path))
            imports = len(score.import_relationships)
            
            coupling_score = deps + dependents + imports * 0.5
            
            if coupling_score > 10:  # Arbitrary threshold
                high_coupling.append(score)
        
        return sorted(high_coupling, key=lambda x: x.relevance_score, reverse=True)
    
    def _identify_refactoring_impact(
        self, 
        relevance_scores: List[RelevanceScore], 
        dependency_graph: DependencyGraph
    ) -> List[RelevanceScore]:
        """Identify files that will be impacted by refactoring."""
        impact_files = []
        
        # Files that depend on high-relevance files
        high_relevance_files = [s.file_path for s in relevance_scores if s.relevance_score >= 0.7]
        
        for score in relevance_scores:
            deps = dependency_graph.get_dependencies(score.file_path)
            
            # If this file depends on high-relevance files, it might be impacted
            if any(dep in high_relevance_files for dep in deps):
                impact_files.append(score)
        
        return impact_files
    
    def _identify_architectural_boundaries(self, relevance_scores: List[RelevanceScore]) -> List[RelevanceScore]:
        """Identify files that define architectural boundaries."""
        boundary_files = []
        
        for score in relevance_scores:
            # Interface files, abstract classes, base classes
            if any(pattern in score.file_path.lower() for pattern in [
                'interface', 'abstract', 'base', 'facade', 'adapter'
            ]):
                boundary_files.append(score)
            
            # Files with many classes (potential boundary definitions)
            if len(score.key_classes) > 3:
                boundary_files.append(score)
        
        return boundary_files
    
    def _identify_overview_files(self, relevance_scores: List[RelevanceScore]) -> List[RelevanceScore]:
        """Identify files that provide system overview."""
        overview_files = []
        
        for score in relevance_scores:
            # Documentation files
            if self._is_test_or_doc_file(score.file_path):
                overview_files.append(score)
            
            # Main/index files
            filename = Path(score.file_path).name.lower()
            if any(pattern in filename for pattern in ['main', 'index', 'app', 'init']):
                overview_files.append(score)
        
        return overview_files
    
    async def _create_dependency_map(
        self, 
        relevance_scores: List[RelevanceScore], 
        dependency_graph: DependencyGraph
    ) -> Dict[str, List[str]]:
        """Create dependency map for selected files."""
        dependency_map = {}
        file_paths = {score.file_path for score in relevance_scores}
        
        for score in relevance_scores:
            # Only include dependencies that are also in our selection
            deps = dependency_graph.get_dependencies(score.file_path)
            internal_deps = [dep for dep in deps if dep in file_paths]
            dependency_map[score.file_path] = internal_deps
        
        return dependency_map
    
    async def _build_navigation_structure(
        self, 
        layers: List[ContextLayer], 
        dependency_map: Dict[str, List[str]], 
        config: AssemblyConfiguration
    ) -> Dict[str, Any]:
        """Build navigation structure for the context."""
        if not config.include_navigation:
            return {}
        
        navigation = {
            "layers": [
                {
                    "name": layer.layer_name,
                    "type": layer.layer_type,
                    "file_count": len(layer.files),
                    "priority": layer.priority,
                    "files": [
                        {
                            "path": score.file_path,
                            "relevance": score.relevance_score,
                            "summary": score.content_summary
                        }
                        for score in layer.files
                    ]
                }
                for layer in layers
            ],
            "quick_access": {
                "entry_points": [
                    score.file_path for layer in layers 
                    for score in layer.files 
                    if layer.layer_type == "core" and score.relevance_score >= 0.8
                ],
                "high_relevance": [
                    score.file_path for layer in layers 
                    for score in layer.files 
                    if score.relevance_score >= 0.7
                ],
                "dependencies": dependency_map
            },
            "statistics": {
                "total_layers": len(layers),
                "total_files": sum(len(layer.files) for layer in layers),
                "avg_relevance": sum(
                    score.relevance_score for layer in layers for score in layer.files
                ) / sum(len(layer.files) for layer in layers) if layers else 0
            }
        }
        
        return navigation
    
    async def _generate_context_summary(
        self, 
        layers: List[ContextLayer], 
        context_request: ContextRequest, 
        config: AssemblyConfiguration
    ) -> str:
        """Generate context summary."""
        if not config.include_summaries:
            return ""
        
        total_files = sum(len(layer.files) for layer in layers)
        avg_relevance = sum(
            score.relevance_score for layer in layers for score in layer.files
        ) / total_files if total_files > 0 else 0
        
        summary_parts = [
            f"Context assembled for {context_request.task_type.value} task",
            f"Total files: {total_files} across {len(layers)} layers",
            f"Average relevance: {avg_relevance:.2f}",
            f"Assembly strategy: {config.strategy.value}"
        ]
        
        # Add layer descriptions
        if layers:
            summary_parts.append("Layers:")
            for layer in layers[:3]:  # Top 3 layers
                summary_parts.append(f"  - {layer.layer_name}: {len(layer.files)} files")
        
        return "; ".join(summary_parts)
    
    def _calculate_total_tokens(self, layers: List[ContextLayer], config: AssemblyConfiguration) -> int:
        """Calculate total estimated tokens for the context."""
        base_tokens = self.token_coefficients["base_overhead"]
        file_tokens = sum(layer.estimated_tokens for layer in layers)
        
        # Add overhead for each file
        file_count = sum(len(layer.files) for layer in layers)
        file_overhead = file_count * self.token_coefficients["per_file_overhead"]
        
        # Add overhead for navigation and summaries
        navigation_overhead = 0
        if config.include_navigation:
            navigation_overhead = int(file_tokens * self.token_coefficients["navigation_factor"])
        
        summary_overhead = 0
        if config.include_summaries:
            summary_overhead = int(file_tokens * self.token_coefficients["summary_factor"])
        
        total_tokens = (
            base_tokens + 
            file_tokens + 
            file_overhead + 
            navigation_overhead + 
            summary_overhead
        )
        
        return total_tokens
    
    async def _create_assembly_metadata(
        self, 
        config: AssemblyConfiguration, 
        relevance_scores: List[RelevanceScore], 
        assembly_time: float
    ) -> Dict[str, Any]:
        """Create assembly metadata."""
        return {
            "strategy": config.strategy.value,
            "format": config.format.value,
            "assembly_time_seconds": assembly_time,
            "files_processed": len(relevance_scores),
            "token_budget": config.max_tokens,
            "token_buffer": config.token_buffer,
            "relevance_stats": {
                "min_relevance": min(s.relevance_score for s in relevance_scores) if relevance_scores else 0,
                "max_relevance": max(s.relevance_score for s in relevance_scores) if relevance_scores else 0,
                "avg_relevance": sum(s.relevance_score for s in relevance_scores) / len(relevance_scores) if relevance_scores else 0
            },
            "assembly_options": config.assembly_options
        }
    
    async def _generate_assembly_recommendations(
        self, 
        layers: List[ContextLayer], 
        context_request: ContextRequest, 
        config: AssemblyConfiguration
    ) -> List[str]:
        """Generate recommendations for context usage."""
        recommendations = []
        
        # Strategy-specific recommendations
        if config.strategy == AssemblyStrategy.HIERARCHICAL:
            recommendations.append("Start with entry points and work through layers systematically")
        elif config.strategy == AssemblyStrategy.DEPENDENCY_FIRST:
            recommendations.append("Begin with foundation dependencies to understand the base architecture")
        elif config.strategy == AssemblyStrategy.TASK_FOCUSED:
            recommendations.append(f"Focus on {context_request.task_type.value}-specific layers first")
        
        # Layer-specific recommendations
        if len(layers) > 3:
            recommendations.append("Consider processing layers incrementally to manage complexity")
        
        # Token usage recommendations
        total_tokens = sum(layer.estimated_tokens for layer in layers)
        if total_tokens > config.max_tokens * 0.8:
            recommendations.append("Context is near token limit - consider focusing on highest relevance files")
        
        # File count recommendations
        total_files = sum(len(layer.files) for layer in layers)
        if total_files > 20:
            recommendations.append("Large number of files - consider using layer priorities for focused analysis")
        
        return recommendations
    
    def _update_assembly_stats(self, config: AssemblyConfiguration, assembly_time: float):
        """Update assembly statistics."""
        self.assembly_stats["contexts_assembled"] += 1
        self.assembly_stats["strategy_usage"][config.strategy.value] += 1
        self.assembly_stats["format_usage"][config.format.value] += 1
        
        # Update average assembly time
        total_contexts = self.assembly_stats["contexts_assembled"]
        current_avg = self.assembly_stats["avg_assembly_time"]
        new_avg = ((current_avg * (total_contexts - 1)) + assembly_time) / total_contexts
        self.assembly_stats["avg_assembly_time"] = new_avg