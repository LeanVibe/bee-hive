"""
Agent Delegation System for Project Index

Provides intelligent task decomposition, agent coordination, and context optimization
to prevent context rot and enable efficient multi-agent workflows.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4

import asyncpg
from pathlib import Path


class TaskComplexity(Enum):
    """Task complexity levels for agent assignment"""
    TRIVIAL = "trivial"          # <1000 tokens, single file
    SIMPLE = "simple"            # 1K-5K tokens, 2-3 files  
    MODERATE = "moderate"        # 5K-15K tokens, 3-8 files
    COMPLEX = "complex"          # 15K-50K tokens, 8-20 files
    LARGE = "large"              # 50K+ tokens, 20+ files, needs decomposition


class AgentSpecialization(Enum):
    """Agent specialization types"""
    BACKEND_ENGINEER = "backend-engineer"
    FRONTEND_ENGINEER = "frontend-engineer"
    DATABASE_SPECIALIST = "database-specialist"
    SECURITY_SPECIALIST = "security-specialist"
    DEVOPS_ENGINEER = "devops-engineer"
    TESTING_SPECIALIST = "testing-specialist"
    DOCUMENTATION_WRITER = "documentation-writer"
    GENERAL_PURPOSE = "general-purpose"


class TaskType(Enum):
    """Types of development tasks"""
    FEATURE_IMPLEMENTATION = "feature-implementation"
    BUG_FIX = "bug-fix"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    OPTIMIZATION = "optimization"
    SECURITY_AUDIT = "security-audit"
    DATABASE_MIGRATION = "database-migration"
    API_DEVELOPMENT = "api-development"
    UI_IMPLEMENTATION = "ui-implementation"


@dataclass
class ContextRequirements:
    """Context requirements for a task"""
    estimated_tokens: int
    max_files: int
    primary_languages: List[str]
    file_types: List[str]
    dependency_depth: int = 2
    include_tests: bool = True
    include_docs: bool = False


@dataclass
class AgentTask:
    """Represents a task that can be assigned to an agent"""
    id: str
    title: str
    description: str
    task_type: TaskType
    complexity: TaskComplexity
    context_requirements: ContextRequirements
    preferred_specialization: AgentSpecialization
    
    # File context
    primary_files: List[str] = field(default_factory=list)
    related_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    
    # Task relationships
    parent_task_id: Optional[str] = None
    subtask_ids: List[str] = field(default_factory=list)
    dependency_task_ids: List[str] = field(default_factory=list)
    
    # Execution metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    estimated_duration_minutes: int = 60
    priority: int = 5  # 1-10 scale
    
    # Agent assignment
    assigned_agent_id: Optional[str] = None
    status: str = "pending"  # pending, assigned, in_progress, completed, failed
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskDecompositionResult:
    """Result of task decomposition"""
    original_task: AgentTask
    subtasks: List[AgentTask]
    coordination_plan: Dict[str, Any]
    estimated_total_duration: int
    decomposition_strategy: str
    success: bool
    reason: str


class TaskDecomposer:
    """
    Intelligent task decomposition system that breaks large development tasks
    into agent-sized chunks while maintaining coherence and dependencies.
    """
    
    def __init__(self, project_id: UUID, db_pool: asyncpg.Pool):
        self.project_id = project_id
        self.db_pool = db_pool
        
        # Configuration
        self.max_context_tokens = 100000  # Maximum context per agent
        self.optimal_context_tokens = 50000  # Optimal context size
        self.max_files_per_task = 15  # Maximum files per subtask
        
    async def decompose_task(self, task_description: str, task_type: TaskType) -> TaskDecompositionResult:
        """
        Main task decomposition entry point.
        Analyzes the task and breaks it into manageable subtasks.
        """
        # Step 1: Analyze task complexity and context requirements
        complexity_analysis = await self._analyze_task_complexity(task_description, task_type)
        
        # Step 2: Gather relevant context from project index
        context_analysis = await self._gather_task_context(task_description, complexity_analysis)
        
        # Step 3: Create initial task object
        main_task = await self._create_main_task(task_description, task_type, complexity_analysis, context_analysis)
        
        # Step 4: Decide if decomposition is needed
        if main_task.complexity in [TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE]:
            return TaskDecompositionResult(
                original_task=main_task,
                subtasks=[main_task],
                coordination_plan={"strategy": "single_agent", "parallel": False},
                estimated_total_duration=main_task.estimated_duration_minutes,
                decomposition_strategy="no_decomposition_needed",
                success=True,
                reason="Task is small enough for single agent"
            )
        
        # Step 5: Perform intelligent decomposition
        if main_task.complexity == TaskComplexity.MODERATE:
            return await self._decompose_moderate_task(main_task, context_analysis)
        elif main_task.complexity in [TaskComplexity.COMPLEX, TaskComplexity.LARGE]:
            return await self._decompose_complex_task(main_task, context_analysis)
        
        return TaskDecompositionResult(
            original_task=main_task,
            subtasks=[],
            coordination_plan={},
            estimated_total_duration=0,
            decomposition_strategy="unknown",
            success=False,
            reason="Unknown complexity level"
        )
    
    async def _analyze_task_complexity(self, task_description: str, task_type: TaskType) -> Dict[str, Any]:
        """Analyze task complexity based on description and type"""
        
        # Extract keywords and scope indicators
        keywords = task_description.lower().split()
        scope_indicators = {
            'new': 2, 'create': 2, 'implement': 2, 'build': 2,
            'refactor': 3, 'migrate': 3, 'optimize': 2,
            'system': 3, 'architecture': 4, 'framework': 4,
            'database': 2, 'api': 2, 'frontend': 2, 'backend': 2,
            'security': 3, 'performance': 2, 'integration': 3,
            'comprehensive': 4, 'complete': 3, 'full': 3
        }
        
        # Calculate base complexity score
        complexity_score = 1
        for keyword in keywords:
            if keyword in scope_indicators:
                complexity_score += scope_indicators[keyword]
        
        # Task type modifiers
        type_modifiers = {
            TaskType.FEATURE_IMPLEMENTATION: 1.5,
            TaskType.REFACTORING: 2.0,
            TaskType.SECURITY_AUDIT: 1.8,
            TaskType.DATABASE_MIGRATION: 1.7,
            TaskType.API_DEVELOPMENT: 1.3,
            TaskType.BUG_FIX: 0.8,
            TaskType.TESTING: 1.0,
            TaskType.DOCUMENTATION: 0.7
        }
        
        complexity_score *= type_modifiers.get(task_type, 1.0)
        
        # Determine complexity level
        if complexity_score <= 3:
            complexity = TaskComplexity.TRIVIAL
            estimated_tokens = 2000
            estimated_files = 1
        elif complexity_score <= 6:
            complexity = TaskComplexity.SIMPLE
            estimated_tokens = 8000
            estimated_files = 3
        elif complexity_score <= 10:
            complexity = TaskComplexity.MODERATE
            estimated_tokens = 25000
            estimated_files = 8
        elif complexity_score <= 15:
            complexity = TaskComplexity.COMPLEX
            estimated_tokens = 80000
            estimated_files = 15
        else:
            complexity = TaskComplexity.LARGE
            estimated_tokens = 150000
            estimated_files = 30
        
        return {
            "complexity": complexity,
            "complexity_score": complexity_score,
            "estimated_tokens": estimated_tokens,
            "estimated_files": estimated_files,
            "estimated_duration_minutes": min(estimated_tokens // 500, 480),  # Cap at 8 hours
            "keywords": keywords,
            "scope_indicators_found": [k for k in keywords if k in scope_indicators]
        }
    
    async def _gather_task_context(self, task_description: str, complexity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Gather relevant context files and dependencies for the task"""
        
        async with self.db_pool.acquire() as conn:
            # Use the context assembly API logic to find relevant files
            keywords = task_description.lower().split()
            search_terms = [word for word in keywords if len(word) > 3][:5]
            
            if not search_terms:
                return {"relevant_files": [], "total_context_tokens": 0}
            
            # Search for relevant files
            search_term = f"%{search_terms[0]}%"
            
            query = """
                SELECT fe.id, fe.relative_path, fe.file_name, fe.language, 
                       fe.content_preview, fe.file_size, fe.line_count,
                       CASE 
                           WHEN fe.relative_path ILIKE $2 THEN 5
                           WHEN fe.file_name ILIKE $2 THEN 4
                           WHEN fe.content_preview ILIKE $2 THEN 3
                           ELSE 1
                       END as relevance_score,
                       COUNT(dr.id) as dependency_count
                FROM file_entries fe
                LEFT JOIN dependency_relationships dr ON fe.id = dr.source_file_id
                WHERE fe.project_id = $1 
                AND (fe.relative_path ILIKE $2 OR fe.file_name ILIKE $2 OR fe.content_preview ILIKE $2)
                GROUP BY fe.id, fe.relative_path, fe.file_name, fe.language, 
                         fe.content_preview, fe.file_size, fe.line_count
                ORDER BY relevance_score DESC, fe.file_size ASC
                LIMIT $3
            """
            
            max_files = complexity_analysis["estimated_files"]
            rows = await conn.fetch(query, self.project_id, search_term, max_files)
            
            relevant_files = []
            total_tokens = 0
            
            for row in rows:
                file_info = {
                    "id": str(row['id']),
                    "relative_path": row['relative_path'],
                    "file_name": row['file_name'],
                    "language": row['language'],
                    "relevance_score": row['relevance_score'],
                    "dependency_count": row['dependency_count'],
                    "estimated_tokens": (row['line_count'] or 0) * 15
                }
                relevant_files.append(file_info)
                total_tokens += file_info["estimated_tokens"]
            
            # Get related files via dependencies for top relevant files
            related_files = []
            for file_info in relevant_files[:3]:  # Top 3 files
                related_query = """
                    SELECT DISTINCT fe2.relative_path, fe2.language, fe2.line_count
                    FROM dependency_relationships dr
                    JOIN file_entries fe2 ON dr.source_file_id = fe2.id
                    WHERE dr.target_name IN (
                        SELECT dr2.target_name 
                        FROM dependency_relationships dr2 
                        WHERE dr2.source_file_id = $1
                        AND dr2.is_external = false
                        LIMIT 2
                    )
                    AND fe2.id != $1
                    LIMIT 3
                """
                related_rows = await conn.fetch(related_query, UUID(file_info["id"]))
                
                for related_row in related_rows:
                    related_info = {
                        "relative_path": related_row['relative_path'],
                        "language": related_row['language'],
                        "estimated_tokens": (related_row['line_count'] or 0) * 15,
                        "relation_type": "dependency"
                    }
                    related_files.append(related_info)
                    total_tokens += related_info["estimated_tokens"]
        
        return {
            "relevant_files": relevant_files,
            "related_files": related_files,
            "total_context_tokens": total_tokens,
            "search_terms": search_terms,
            "primary_languages": list(set(f["language"] for f in relevant_files if f["language"])),
            "file_types": list(set(f["relative_path"].split(".")[-1] for f in relevant_files))
        }
    
    async def _create_main_task(
        self, 
        task_description: str, 
        task_type: TaskType, 
        complexity_analysis: Dict[str, Any],
        context_analysis: Dict[str, Any]
    ) -> AgentTask:
        """Create the main task object from analysis results"""
        
        # Determine preferred agent specialization
        specialization_map = {
            TaskType.API_DEVELOPMENT: AgentSpecialization.BACKEND_ENGINEER,
            TaskType.UI_IMPLEMENTATION: AgentSpecialization.FRONTEND_ENGINEER,
            TaskType.DATABASE_MIGRATION: AgentSpecialization.DATABASE_SPECIALIST,
            TaskType.SECURITY_AUDIT: AgentSpecialization.SECURITY_SPECIALIST,
            TaskType.TESTING: AgentSpecialization.TESTING_SPECIALIST,
            TaskType.DOCUMENTATION: AgentSpecialization.DOCUMENTATION_WRITER,
        }
        
        preferred_specialization = specialization_map.get(task_type, AgentSpecialization.GENERAL_PURPOSE)
        
        # Create context requirements
        context_requirements = ContextRequirements(
            estimated_tokens=context_analysis["total_context_tokens"],
            max_files=len(context_analysis["relevant_files"]) + len(context_analysis["related_files"]),
            primary_languages=context_analysis["primary_languages"],
            file_types=context_analysis["file_types"],
            dependency_depth=2,
            include_tests=task_type in [TaskType.FEATURE_IMPLEMENTATION, TaskType.BUG_FIX],
            include_docs=task_type in [TaskType.DOCUMENTATION, TaskType.FEATURE_IMPLEMENTATION]
        )
        
        # Extract file paths
        primary_files = [f["relative_path"] for f in context_analysis["relevant_files"]]
        related_files = [f["relative_path"] for f in context_analysis["related_files"]]
        
        return AgentTask(
            id=str(uuid4()),
            title=f"{task_type.value.replace('_', ' ').title()}: {task_description[:50]}",
            description=task_description,
            task_type=task_type,
            complexity=complexity_analysis["complexity"],
            context_requirements=context_requirements,
            preferred_specialization=preferred_specialization,
            primary_files=primary_files,
            related_files=related_files,
            estimated_duration_minutes=complexity_analysis["estimated_duration_minutes"],
            metadata={
                "complexity_score": complexity_analysis["complexity_score"],
                "keywords": complexity_analysis["keywords"],
                "search_terms": context_analysis["search_terms"]
            }
        )
    
    async def _decompose_moderate_task(self, main_task: AgentTask, context_analysis: Dict[str, Any]) -> TaskDecompositionResult:
        """Decompose moderate complexity tasks into 2-3 subtasks"""
        
        subtasks = []
        
        # Strategy 1: Decompose by file groups
        if len(main_task.primary_files) > 5:
            # Group files by directory or functionality
            file_groups = self._group_files_by_functionality(main_task.primary_files)
            
            for i, (group_name, files) in enumerate(file_groups.items()):
                subtask = AgentTask(
                    id=str(uuid4()),
                    title=f"{main_task.title} - {group_name}",
                    description=f"Implement {group_name} component: {main_task.description}",
                    task_type=main_task.task_type,
                    complexity=TaskComplexity.SIMPLE,
                    context_requirements=ContextRequirements(
                        estimated_tokens=min(main_task.context_requirements.estimated_tokens // len(file_groups), 30000),
                        max_files=len(files) + 2,
                        primary_languages=main_task.context_requirements.primary_languages,
                        file_types=main_task.context_requirements.file_types
                    ),
                    preferred_specialization=main_task.preferred_specialization,
                    primary_files=files,
                    related_files=[f for f in main_task.related_files if any(rf in f for rf in files)],
                    parent_task_id=main_task.id,
                    estimated_duration_minutes=main_task.estimated_duration_minutes // len(file_groups),
                    priority=main_task.priority,
                    metadata={"decomposition_strategy": "file_groups", "group_name": group_name}
                )
                subtasks.append(subtask)
        
        # Strategy 2: Decompose by task phases
        else:
            phase_strategies = {
                TaskType.FEATURE_IMPLEMENTATION: ["design_and_setup", "core_implementation", "testing_and_validation"],
                TaskType.REFACTORING: ["analysis_and_planning", "implementation", "testing_and_cleanup"],
                TaskType.API_DEVELOPMENT: ["api_design", "implementation", "testing_and_documentation"]
            }
            
            phases = phase_strategies.get(main_task.task_type, ["planning", "implementation", "validation"])
            
            for i, phase in enumerate(phases):
                subtask = AgentTask(
                    id=str(uuid4()),
                    title=f"{main_task.title} - Phase {i+1}: {phase.replace('_', ' ').title()}",
                    description=f"{phase.replace('_', ' ').title()} phase: {main_task.description}",
                    task_type=main_task.task_type,
                    complexity=TaskComplexity.SIMPLE,
                    context_requirements=ContextRequirements(
                        estimated_tokens=main_task.context_requirements.estimated_tokens // len(phases),
                        max_files=max(main_task.context_requirements.max_files // len(phases), 3),
                        primary_languages=main_task.context_requirements.primary_languages,
                        file_types=main_task.context_requirements.file_types
                    ),
                    preferred_specialization=main_task.preferred_specialization,
                    primary_files=main_task.primary_files,
                    related_files=main_task.related_files,
                    parent_task_id=main_task.id,
                    estimated_duration_minutes=main_task.estimated_duration_minutes // len(phases),
                    priority=main_task.priority,
                    metadata={"decomposition_strategy": "phases", "phase": phase}
                )
                
                # Add dependency relationships between phases
                if i > 0:
                    subtask.dependency_task_ids = [subtasks[i-1].id]
                
                subtasks.append(subtask)
        
        # Update main task with subtask references
        main_task.subtask_ids = [task.id for task in subtasks]
        
        return TaskDecompositionResult(
            original_task=main_task,
            subtasks=subtasks,
            coordination_plan={
                "strategy": "sequential" if main_task.task_type == TaskType.FEATURE_IMPLEMENTATION else "parallel",
                "parallel": main_task.task_type in [TaskType.REFACTORING, TaskType.TESTING],
                "coordination_required": True,
                "estimated_parallel_duration": max(task.estimated_duration_minutes for task in subtasks),
                "estimated_sequential_duration": sum(task.estimated_duration_minutes for task in subtasks)
            },
            estimated_total_duration=sum(task.estimated_duration_minutes for task in subtasks),
            decomposition_strategy="moderate_task_breakdown",
            success=True,
            reason=f"Task decomposed into {len(subtasks)} manageable subtasks"
        )
    
    async def _decompose_complex_task(self, main_task: AgentTask, context_analysis: Dict[str, Any]) -> TaskDecompositionResult:
        """Decompose complex tasks into 4-8 specialized subtasks"""
        
        subtasks = []
        
        # Complex tasks require multi-dimensional decomposition
        # 1. By architectural layers
        # 2. By specialized concerns
        # 3. By file groups
        
        # Strategy: Architectural layer decomposition
        architectural_layers = {
            "data_layer": {"keywords": ["model", "database", "schema", "migration"], "specialization": AgentSpecialization.DATABASE_SPECIALIST},
            "business_logic": {"keywords": ["service", "logic", "core", "engine"], "specialization": AgentSpecialization.BACKEND_ENGINEER},
            "api_layer": {"keywords": ["api", "endpoint", "route", "controller"], "specialization": AgentSpecialization.BACKEND_ENGINEER},
            "frontend": {"keywords": ["ui", "component", "view", "frontend"], "specialization": AgentSpecialization.FRONTEND_ENGINEER},
            "testing": {"keywords": ["test", "spec", "validation"], "specialization": AgentSpecialization.TESTING_SPECIALIST},
            "security": {"keywords": ["auth", "security", "permission", "access"], "specialization": AgentSpecialization.SECURITY_SPECIALIST}
        }
        
        # Categorize files by architectural layer
        file_categorization = {}
        for file_path in main_task.primary_files:
            file_lower = file_path.lower()
            for layer, config in architectural_layers.items():
                if any(keyword in file_lower for keyword in config["keywords"]):
                    if layer not in file_categorization:
                        file_categorization[layer] = []
                    file_categorization[layer].append(file_path)
                    break
            else:
                # Uncategorized files go to business logic
                if "business_logic" not in file_categorization:
                    file_categorization["business_logic"] = []
                file_categorization["business_logic"].append(file_path)
        
        # Create subtasks for each layer that has files
        for layer, files in file_categorization.items():
            if not files:
                continue
                
            layer_config = architectural_layers[layer]
            
            subtask = AgentTask(
                id=str(uuid4()),
                title=f"{main_task.title} - {layer.replace('_', ' ').title()} Layer",
                description=f"Implement {layer.replace('_', ' ')} components: {main_task.description}",
                task_type=main_task.task_type,
                complexity=TaskComplexity.MODERATE if len(files) > 8 else TaskComplexity.SIMPLE,
                context_requirements=ContextRequirements(
                    estimated_tokens=min(main_task.context_requirements.estimated_tokens // len(file_categorization), 40000),
                    max_files=len(files) + 3,
                    primary_languages=main_task.context_requirements.primary_languages,
                    file_types=main_task.context_requirements.file_types
                ),
                preferred_specialization=layer_config["specialization"],
                primary_files=files,
                related_files=[f for f in main_task.related_files if any(rf in f for rf in files)],
                parent_task_id=main_task.id,
                estimated_duration_minutes=main_task.estimated_duration_minutes // len(file_categorization),
                priority=main_task.priority,
                metadata={
                    "decomposition_strategy": "architectural_layers", 
                    "layer": layer,
                    "specialization_required": layer_config["specialization"].value
                }
            )
            subtasks.append(subtask)
        
        # Add coordination task if multiple specialized agents are involved
        if len(set(task.preferred_specialization for task in subtasks)) > 1:
            coordination_task = AgentTask(
                id=str(uuid4()),
                title=f"{main_task.title} - Integration & Coordination",
                description=f"Coordinate and integrate components: {main_task.description}",
                task_type=TaskType.FEATURE_IMPLEMENTATION,
                complexity=TaskComplexity.SIMPLE,
                context_requirements=ContextRequirements(
                    estimated_tokens=15000,
                    max_files=5,
                    primary_languages=main_task.context_requirements.primary_languages,
                    file_types=["py", "js", "ts"]
                ),
                preferred_specialization=AgentSpecialization.GENERAL_PURPOSE,
                primary_files=[],
                related_files=main_task.primary_files[:5],  # Reference key files
                parent_task_id=main_task.id,
                dependency_task_ids=[task.id for task in subtasks],
                estimated_duration_minutes=60,
                priority=main_task.priority + 1,  # Higher priority for coordination
                metadata={"decomposition_strategy": "coordination", "role": "integration"}
            )
            subtasks.append(coordination_task)
        
        # Update main task
        main_task.subtask_ids = [task.id for task in subtasks]
        
        return TaskDecompositionResult(
            original_task=main_task,
            subtasks=subtasks,
            coordination_plan={
                "strategy": "specialized_parallel",
                "parallel": True,
                "coordination_required": True,
                "coordination_agent_required": len(set(task.preferred_specialization for task in subtasks)) > 2,
                "estimated_parallel_duration": max(task.estimated_duration_minutes for task in subtasks[:-1]) + (subtasks[-1].estimated_duration_minutes if subtasks else 0),
                "specializations_involved": list(set(task.preferred_specialization.value for task in subtasks))
            },
            estimated_total_duration=max(task.estimated_duration_minutes for task in subtasks[:-1]) + (subtasks[-1].estimated_duration_minutes if subtasks else 0),
            decomposition_strategy="complex_architectural_decomposition",
            success=True,
            reason=f"Complex task decomposed into {len(subtasks)} specialized subtasks across {len(set(task.preferred_specialization for task in subtasks))} specializations"
        )
    
    def _group_files_by_functionality(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """Group files by functionality based on path patterns"""
        groups = {}
        
        for file_path in file_paths:
            path_parts = file_path.split('/')
            
            # Group by top-level directory
            if len(path_parts) > 1:
                group_key = path_parts[0]
                if len(path_parts) > 2 and path_parts[1] in ['core', 'models', 'services', 'utils', 'tests']:
                    group_key = f"{path_parts[0]}/{path_parts[1]}"
            else:
                group_key = "root"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(file_path)
        
        return groups


class AgentCoordinator:
    """
    Coordinates multiple agents working on decomposed tasks with context optimization
    and conflict prevention.
    """
    
    def __init__(self, project_id: UUID, db_pool: asyncpg.Pool):
        self.project_id = project_id
        self.db_pool = db_pool
        
        # Active agent tracking
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        
        # Context management
        self.agent_context_usage: Dict[str, int] = {}
        self.context_refresh_threshold = 85000  # 85% of max context
        
    async def assign_agents_to_tasks(self, decomposition_result: TaskDecompositionResult) -> Dict[str, Any]:
        """
        Assign specialized agents to decomposed tasks with optimal scheduling.
        """
        assignments = []
        
        # Sort tasks by priority and dependencies
        sorted_tasks = self._sort_tasks_for_assignment(decomposition_result.subtasks)
        
        for task in sorted_tasks:
            # Find or create suitable agent
            agent_assignment = await self._find_optimal_agent(task)
            
            # Check for context conflicts
            context_conflicts = await self._check_context_conflicts(task, agent_assignment["agent_id"])
            
            if context_conflicts:
                # Trigger context refresh or assign different agent
                await self._resolve_context_conflicts(task, agent_assignment, context_conflicts)
            
            # Make assignment
            assignment = {
                "task_id": task.id,
                "agent_id": agent_assignment["agent_id"],
                "assignment_time": datetime.utcnow(),
                "estimated_start_time": agent_assignment["estimated_start_time"],
                "context_requirements": task.context_requirements,
                "specialization_match": agent_assignment["specialization_match"],
                "priority": task.priority
            }
            
            assignments.append(assignment)
            self.task_assignments[task.id] = agent_assignment["agent_id"]
            
            # Update agent tracking
            if agent_assignment["agent_id"] not in self.active_agents:
                self.active_agents[agent_assignment["agent_id"]] = {
                    "specialization": task.preferred_specialization.value,
                    "assigned_tasks": [],
                    "context_usage": 0,
                    "last_context_refresh": datetime.utcnow()
                }
            
            self.active_agents[agent_assignment["agent_id"]]["assigned_tasks"].append(task.id)
            self.active_agents[agent_assignment["agent_id"]]["context_usage"] += task.context_requirements.estimated_tokens
        
        # Generate coordination plan
        coordination_plan = self._generate_coordination_plan(decomposition_result, assignments)
        
        return {
            "assignments": assignments,
            "coordination_plan": coordination_plan,
            "total_agents": len(set(a["agent_id"] for a in assignments)),
            "parallel_execution": decomposition_result.coordination_plan.get("parallel", False),
            "estimated_completion": self._estimate_completion_time(assignments, decomposition_result.coordination_plan)
        }
    
    def _sort_tasks_for_assignment(self, tasks: List[AgentTask]) -> List[AgentTask]:
        """Sort tasks for optimal assignment order"""
        # Priority: dependency order > priority > complexity
        return sorted(tasks, key=lambda t: (
            len(t.dependency_task_ids),  # Tasks with dependencies come later
            -t.priority,                 # Higher priority first
            -t.context_requirements.estimated_tokens  # Larger tasks first
        ))
    
    async def _find_optimal_agent(self, task: AgentTask) -> Dict[str, Any]:
        """Find the optimal agent for a task"""
        
        # Check for existing agents with matching specialization and available capacity
        suitable_agents = []
        for agent_id, agent_info in self.active_agents.items():
            if (agent_info["specialization"] == task.preferred_specialization.value and 
                agent_info["context_usage"] + task.context_requirements.estimated_tokens < self.context_refresh_threshold):
                
                suitable_agents.append({
                    "agent_id": agent_id,
                    "current_load": len(agent_info["assigned_tasks"]),
                    "context_usage": agent_info["context_usage"],
                    "specialization_match": True,
                    "estimated_start_time": datetime.utcnow()
                })
        
        # If no suitable existing agent, create new assignment
        if not suitable_agents:
            agent_id = f"agent_{task.preferred_specialization.value}_{len(self.active_agents) + 1}"
            return {
                "agent_id": agent_id,
                "current_load": 0,
                "context_usage": 0,
                "specialization_match": True,
                "estimated_start_time": datetime.utcnow(),
                "new_agent": True
            }
        
        # Return agent with lowest current load
        return min(suitable_agents, key=lambda a: a["current_load"])
    
    async def _check_context_conflicts(self, task: AgentTask, agent_id: str) -> List[Dict[str, Any]]:
        """Check for context conflicts with other tasks assigned to the same agent"""
        conflicts = []
        
        if agent_id not in self.active_agents:
            return conflicts
        
        agent_info = self.active_agents[agent_id]
        
        # Check context size limits
        total_context = agent_info["context_usage"] + task.context_requirements.estimated_tokens
        if total_context > self.context_refresh_threshold:
            conflicts.append({
                "type": "context_overflow",
                "current_usage": agent_info["context_usage"],
                "additional_requirement": task.context_requirements.estimated_tokens,
                "total_projected": total_context,
                "threshold": self.context_refresh_threshold
            })
        
        # Check for file conflicts (same files being modified)
        for assigned_task_id in agent_info["assigned_tasks"]:
            # In a real implementation, we would check for file overlap
            # For now, we'll flag this as a potential conflict
            conflicts.append({
                "type": "potential_file_conflict",
                "conflicting_task": assigned_task_id,
                "files": task.primary_files
            })
        
        return conflicts
    
    async def _resolve_context_conflicts(self, task: AgentTask, agent_assignment: Dict[str, Any], conflicts: List[Dict[str, Any]]) -> None:
        """Resolve context conflicts through refresh or reassignment"""
        
        for conflict in conflicts:
            if conflict["type"] == "context_overflow":
                # Trigger context refresh for the agent
                agent_id = agent_assignment["agent_id"]
                await self._trigger_context_refresh(agent_id)
                
                # Update agent context usage
                if agent_id in self.active_agents:
                    self.active_agents[agent_id]["context_usage"] = 0
                    self.active_agents[agent_id]["last_context_refresh"] = datetime.utcnow()
            
            elif conflict["type"] == "potential_file_conflict":
                # For now, we'll allow this but flag it for coordination
                pass
    
    async def _trigger_context_refresh(self, agent_id: str) -> None:
        """Trigger context refresh (sleep/wake cycle) for an agent"""
        # In a real implementation, this would trigger the agent's sleep/wake cycle
        print(f"Triggering context refresh for agent {agent_id}")
    
    def _generate_coordination_plan(self, decomposition_result: TaskDecompositionResult, assignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate coordination plan for multi-agent execution"""
        
        # Group assignments by agent
        agent_groups = {}
        for assignment in assignments:
            agent_id = assignment["agent_id"]
            if agent_id not in agent_groups:
                agent_groups[agent_id] = []
            agent_groups[agent_id].append(assignment)
        
        # Determine execution strategy
        execution_strategy = "parallel" if decomposition_result.coordination_plan.get("parallel", False) else "sequential"
        
        # Create synchronization points
        sync_points = []
        if len(agent_groups) > 1:
            # Add sync point after each major phase
            sync_points.append({
                "type": "milestone_sync",
                "description": "All agents complete initial implementation",
                "required_tasks": [a["task_id"] for a in assignments if "coordination" not in a.get("metadata", {}).get("role", "")]
            })
            
            # Add final integration sync
            integration_tasks = [a["task_id"] for a in assignments if a.get("metadata", {}).get("role") == "integration"]
            if integration_tasks:
                sync_points.append({
                    "type": "integration_sync", 
                    "description": "Integration and coordination completion",
                    "required_tasks": integration_tasks
                })
        
        return {
            "execution_strategy": execution_strategy,
            "total_agents": len(agent_groups),
            "agent_assignments": agent_groups,
            "synchronization_points": sync_points,
            "coordination_required": decomposition_result.coordination_plan.get("coordination_required", False),
            "estimated_duration": decomposition_result.estimated_total_duration
        }
    
    def _estimate_completion_time(self, assignments: List[Dict[str, Any]], coordination_plan: Dict[str, Any]) -> datetime:
        """Estimate when all tasks will be completed"""
        
        if coordination_plan.get("parallel", False):
            # Parallel execution - completion time is the longest task
            max_duration = max((a.get("estimated_duration", 60) for a in assignments), default=60)
        else:
            # Sequential execution - sum of all durations
            max_duration = sum(a.get("estimated_duration", 60) for a in assignments)
        
        return datetime.utcnow() + timedelta(minutes=max_duration)


class ContextRotPrevention:
    """
    Monitors agent context usage and prevents context rot through intelligent
    refresh cycles and context optimization.
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        
        # Context monitoring thresholds
        self.context_warning_threshold = 75000   # 75% of typical max context
        self.context_critical_threshold = 90000  # 90% of typical max context  
        self.context_max_threshold = 100000      # Hard limit
        
        # Monitoring state
        self.agent_metrics: Dict[str, Dict[str, Any]] = {}
    
    async def monitor_agent_context(self, agent_id: str, current_context_size: int) -> Dict[str, Any]:
        """Monitor an agent's context usage and return recommendations"""
        
        # Update metrics
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = {
                "context_history": [],
                "refresh_count": 0,
                "last_refresh": None,
                "efficiency_score": 1.0
            }
        
        metrics = self.agent_metrics[agent_id]
        metrics["context_history"].append({
            "timestamp": datetime.utcnow(),
            "context_size": current_context_size
        })
        
        # Keep only recent history
        cutoff_time = datetime.utcnow() - timedelta(hours=2)
        metrics["context_history"] = [
            entry for entry in metrics["context_history"] 
            if entry["timestamp"] > cutoff_time
        ]
        
        # Analyze context usage pattern
        context_trend = self._analyze_context_trend(metrics["context_history"])
        
        # Generate recommendations
        recommendations = []
        
        if current_context_size >= self.context_critical_threshold:
            recommendations.append({
                "type": "immediate_refresh",
                "priority": "critical",
                "message": f"Context at {current_context_size} tokens - immediate refresh required",
                "action": "trigger_sleep_wake_cycle"
            })
        elif current_context_size >= self.context_warning_threshold:
            recommendations.append({
                "type": "planned_refresh", 
                "priority": "warning",
                "message": f"Context at {current_context_size} tokens - plan refresh soon",
                "action": "schedule_context_optimization"
            })
        
        if context_trend["growth_rate"] > 5000:  # tokens per hour
            recommendations.append({
                "type": "growth_concern",
                "priority": "warning", 
                "message": f"Context growing rapidly at {context_trend['growth_rate']} tokens/hour",
                "action": "optimize_context_usage"
            })
        
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(metrics, current_context_size)
        metrics["efficiency_score"] = efficiency_score
        
        return {
            "agent_id": agent_id,
            "current_context_size": current_context_size,
            "threshold_status": self._get_threshold_status(current_context_size),
            "recommendations": recommendations,
            "context_trend": context_trend,
            "efficiency_score": efficiency_score,
            "next_refresh_estimate": self._estimate_next_refresh(current_context_size, context_trend)
        }
    
    def _analyze_context_trend(self, context_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze context usage trends"""
        if len(context_history) < 2:
            return {"growth_rate": 0, "trend": "stable", "volatility": 0}
        
        # Calculate growth rate (tokens per hour)
        recent_entries = context_history[-5:]  # Last 5 entries
        if len(recent_entries) < 2:
            return {"growth_rate": 0, "trend": "stable", "volatility": 0}
        
        time_diff = (recent_entries[-1]["timestamp"] - recent_entries[0]["timestamp"]).total_seconds() / 3600
        if time_diff == 0:
            return {"growth_rate": 0, "trend": "stable", "volatility": 0}
        
        size_diff = recent_entries[-1]["context_size"] - recent_entries[0]["context_size"]
        growth_rate = size_diff / time_diff
        
        # Determine trend
        if growth_rate > 1000:
            trend = "growing_fast"
        elif growth_rate > 500:
            trend = "growing"
        elif growth_rate < -500:
            trend = "shrinking"
        else:
            trend = "stable"
        
        # Calculate volatility
        sizes = [entry["context_size"] for entry in recent_entries]
        avg_size = sum(sizes) / len(sizes)
        volatility = sum(abs(size - avg_size) for size in sizes) / len(sizes)
        
        return {
            "growth_rate": growth_rate,
            "trend": trend,
            "volatility": volatility,
            "time_span_hours": time_diff
        }
    
    def _get_threshold_status(self, context_size: int) -> str:
        """Get current threshold status"""
        if context_size >= self.context_critical_threshold:
            return "critical"
        elif context_size >= self.context_warning_threshold:
            return "warning"
        else:
            return "normal"
    
    def _calculate_efficiency_score(self, metrics: Dict[str, Any], current_context_size: int) -> float:
        """Calculate agent efficiency score based on context usage patterns"""
        
        # Base score
        score = 1.0
        
        # Penalize for high context usage
        if current_context_size > self.context_warning_threshold:
            score -= 0.2
        if current_context_size > self.context_critical_threshold:
            score -= 0.3
        
        # Reward for stable context usage
        if len(metrics["context_history"]) > 3:
            recent_sizes = [entry["context_size"] for entry in metrics["context_history"][-5:]]
            stability = 1.0 - (max(recent_sizes) - min(recent_sizes)) / max(recent_sizes, 1)
            score += stability * 0.2
        
        # Penalize for frequent refreshes
        if metrics["refresh_count"] > 3:
            score -= 0.1 * (metrics["refresh_count"] - 3)
        
        return max(0.0, min(1.0, score))
    
    def _estimate_next_refresh(self, current_context_size: int, context_trend: Dict[str, Any]) -> Optional[datetime]:
        """Estimate when the next context refresh will be needed"""
        
        growth_rate = context_trend["growth_rate"]
        
        if growth_rate <= 0:
            return None  # No refresh needed if not growing
        
        # Calculate time until critical threshold
        tokens_until_critical = self.context_critical_threshold - current_context_size
        
        if tokens_until_critical <= 0:
            return datetime.utcnow()  # Immediate refresh needed
        
        hours_until_critical = tokens_until_critical / growth_rate
        
        return datetime.utcnow() + timedelta(hours=hours_until_critical)
    
    async def trigger_context_refresh(self, agent_id: str, refresh_type: str = "full") -> Dict[str, Any]:
        """Trigger a context refresh for an agent"""
        
        if agent_id in self.agent_metrics:
            self.agent_metrics[agent_id]["refresh_count"] += 1
            self.agent_metrics[agent_id]["last_refresh"] = datetime.utcnow()
        
        # In a real implementation, this would trigger the actual sleep/wake cycle
        # For now, we'll return the refresh plan
        
        refresh_plan = {
            "agent_id": agent_id,
            "refresh_type": refresh_type,
            "timestamp": datetime.utcnow(),
            "steps": []
        }
        
        if refresh_type == "full":
            refresh_plan["steps"] = [
                {"step": "save_current_state", "description": "Save current progress and context"},
                {"step": "trigger_sleep", "description": "Trigger agent sleep cycle"},
                {"step": "context_consolidation", "description": "Consolidate and optimize context"},
                {"step": "wake_with_optimized_context", "description": "Wake agent with optimized context"},
                {"step": "resume_tasks", "description": "Resume assigned tasks"}
            ]
        elif refresh_type == "light":
            refresh_plan["steps"] = [
                {"step": "context_optimization", "description": "Optimize current context without full sleep"},
                {"step": "memory_cleanup", "description": "Clean up unnecessary context"},
                {"step": "continue_tasks", "description": "Continue with optimized context"}
            ]
        
        return refresh_plan