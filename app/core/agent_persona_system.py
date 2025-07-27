"""
Agent Persona System for LeanVibe Agent Hive 2.0

Provides role-based cognitive specialization with:
- Dynamic persona assignment based on task requirements
- Context-aware behavior adaptation
- Persona performance analytics and optimization
- Integration with agent lifecycle and task coordination
- Hierarchical persona inheritance and capability mapping
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_

from ..models.agent import Agent
from ..models.task import Task, TaskType, TaskStatus
from ..models.context import Context
from ..core.database import get_async_session
from ..core.redis import get_redis
from ..core.config import get_settings
from .enhanced_lifecycle_hooks import (
    EnhancedLifecycleHookProcessor, 
    EnhancedEventType, 
    get_enhanced_lifecycle_hook_processor
)

logger = structlog.get_logger()


class PersonaType(str, Enum):
    """Core persona types for agent specialization."""
    
    # Technical Specialists
    BACKEND_ENGINEER = "backend_engineer"
    FRONTEND_DEVELOPER = "frontend_developer"
    DEVOPS_SPECIALIST = "devops_specialist"
    QA_ENGINEER = "qa_engineer"
    SECURITY_SPECIALIST = "security_specialist"
    DATA_ENGINEER = "data_engineer"
    
    # Domain Experts
    PROJECT_MANAGER = "project_manager"
    PRODUCT_OWNER = "product_owner"
    ARCHITECT = "architect"
    TECH_LEAD = "tech_lead"
    
    # General Purpose
    GENERALIST = "generalist"
    RESEARCHER = "researcher"
    COORDINATOR = "coordinator"
    OPTIMIZER = "optimizer"
    
    # Specialized Roles
    CODE_REVIEWER = "code_reviewer"
    DEPLOYMENT_SPECIALIST = "deployment_specialist"
    PERFORMANCE_ANALYST = "performance_analyst"
    DOCUMENTATION_SPECIALIST = "documentation_specialist"


class PersonaCapabilityLevel(str, Enum):
    """Capability levels for persona skills."""
    NOVICE = "novice"        # 0-20% proficiency
    INTERMEDIATE = "intermediate"  # 21-60% proficiency  
    ADVANCED = "advanced"    # 61-85% proficiency
    EXPERT = "expert"        # 86-100% proficiency


class PersonaAdaptationMode(str, Enum):
    """How persona adapts to new situations."""
    STATIC = "static"           # Fixed behavior patterns
    ADAPTIVE = "adaptive"       # Learns from context
    DYNAMIC = "dynamic"         # Real-time adaptation
    COLLABORATIVE = "collaborative"  # Adapts based on team interaction


@dataclass
class PersonaCapability:
    """Individual capability with proficiency tracking."""
    name: str
    level: PersonaCapabilityLevel
    proficiency_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0 - how confident the persona is in this capability
    last_used: Optional[datetime] = None
    success_rate: float = 0.0
    usage_count: int = 0
    
    def update_proficiency(self, success: bool, complexity: float = 0.5):
        """Update proficiency based on task success and complexity."""
        self.usage_count += 1
        
        # Update success rate
        old_success_rate = self.success_rate
        self.success_rate = ((old_success_rate * (self.usage_count - 1)) + (1.0 if success else 0.0)) / self.usage_count
        
        # Adjust proficiency based on performance
        adjustment = 0.02 * complexity if success else -0.01 * complexity
        self.proficiency_score = max(0.0, min(1.0, self.proficiency_score + adjustment))
        
        # Update confidence based on recent performance
        self.confidence = min(1.0, self.success_rate * 0.8 + self.proficiency_score * 0.2)
        
        # Update capability level based on proficiency score
        if self.proficiency_score >= 0.86:
            self.level = PersonaCapabilityLevel.EXPERT
        elif self.proficiency_score >= 0.61:
            self.level = PersonaCapabilityLevel.ADVANCED
        elif self.proficiency_score >= 0.21:
            self.level = PersonaCapabilityLevel.INTERMEDIATE
        else:
            self.level = PersonaCapabilityLevel.NOVICE
        
        self.last_used = datetime.utcnow()


@dataclass 
class PersonaDefinition:
    """Complete definition of an agent persona."""
    
    id: str
    name: str
    description: str
    persona_type: PersonaType
    adaptation_mode: PersonaAdaptationMode
    
    # Core capabilities and strengths
    capabilities: Dict[str, PersonaCapability]
    preferred_task_types: List[TaskType]
    expertise_domains: List[str]
    
    # Behavioral characteristics
    communication_style: Dict[str, Any]  # formal, casual, technical, etc.
    decision_making_style: Dict[str, Any]  # analytical, intuitive, collaborative
    problem_solving_approach: Dict[str, Any]  # systematic, creative, pragmatic
    
    # Collaboration preferences
    preferred_team_size: Tuple[int, int]  # min, max
    collaboration_patterns: List[str]  # pair_programming, code_review, etc.
    mentoring_capability: bool
    
    # Performance characteristics
    typical_response_time: float  # seconds
    accuracy_vs_speed_preference: float  # 0.0 (speed) to 1.0 (accuracy)
    risk_tolerance: float  # 0.0 (conservative) to 1.0 (aggressive)
    
    # Metadata
    created_at: datetime
    last_updated: datetime
    version: str = "1.0.0"
    active: bool = True
    
    def get_capability_score(self, capability_name: str) -> float:
        """Get proficiency score for a specific capability."""
        capability = self.capabilities.get(capability_name)
        return capability.proficiency_score if capability else 0.0
    
    def get_task_affinity(self, task_type: TaskType) -> float:
        """Calculate affinity score for a task type."""
        if task_type in self.preferred_task_types:
            return 1.0
        
        # Calculate based on related capabilities
        task_capability_map = {
            TaskType.CODE_GENERATION: ["programming", "software_architecture", "code_quality"],
            TaskType.CODE_REVIEW: ["code_quality", "security_analysis", "best_practices"],
            TaskType.TESTING: ["quality_assurance", "test_automation", "debugging"],
            TaskType.DEPLOYMENT: ["devops", "cloud_platforms", "automation"],
            TaskType.DOCUMENTATION: ["technical_writing", "knowledge_management"]
        }
        
        related_capabilities = task_capability_map.get(task_type, [])
        if not related_capabilities:
            return 0.3  # Base affinity for generalists
        
        # Check if we have any of the related capabilities
        capability_scores = [self.get_capability_score(cap) for cap in related_capabilities]
        available_scores = [score for score in capability_scores if score > 0]
        
        if not available_scores:
            return 0.3  # Base affinity if no related capabilities
        
        avg_score = sum(available_scores) / len(available_scores)
        return avg_score * 0.8  # Reduce for non-preferred tasks
    
    def adapt_to_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt persona behavior based on current context."""
        adaptations = {}
        
        if self.adaptation_mode == PersonaAdaptationMode.STATIC:
            return adaptations
        
        # Adapt based on team size
        team_size = context.get("team_size", 1)
        if team_size < self.preferred_team_size[0]:
            adaptations["increase_autonomy"] = True
        elif team_size > self.preferred_team_size[1]:
            adaptations["focus_specialization"] = True
        
        # Adapt based on project complexity
        complexity = context.get("project_complexity", 0.5)
        if complexity > 0.8:
            adaptations["request_collaboration"] = True
            adaptations["increase_planning_time"] = True
        
        # Adapt based on time pressure
        urgency = context.get("urgency", 0.5)
        if urgency > 0.7:
            # Shift toward speed over accuracy if time-pressured
            speed_bias = min(0.3, urgency - 0.7)
            current_preference = self.accuracy_vs_speed_preference
            adaptations["temp_speed_preference"] = max(0.0, current_preference - speed_bias)
        
        return adaptations


@dataclass
class PersonaAssignment:
    """Assignment of persona to specific agent."""
    
    agent_id: uuid.UUID
    persona_id: str
    session_id: str
    
    assigned_at: datetime
    assignment_reason: str
    confidence_score: float  # How confident we are in this assignment
    
    # Performance tracking for this assignment
    tasks_completed: int = 0
    success_rate: float = 0.0
    avg_completion_time: float = 0.0
    
    # Context adaptations applied
    active_adaptations: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.active_adaptations is None:
            self.active_adaptations = {}
    
    def update_performance(self, task_success: bool, completion_time: float):
        """Update performance metrics for this assignment."""
        self.tasks_completed += 1
        
        # Update success rate
        old_rate = self.success_rate
        self.success_rate = ((old_rate * (self.tasks_completed - 1)) + (1.0 if task_success else 0.0)) / self.tasks_completed
        
        # Update average completion time
        old_avg = self.avg_completion_time
        self.avg_completion_time = ((old_avg * (self.tasks_completed - 1)) + completion_time) / self.tasks_completed


class PersonaRecommendationEngine:
    """Intelligent persona recommendation based on task and context analysis."""
    
    def __init__(self, persona_system: 'AgentPersonaSystem'):
        self.persona_system = persona_system
        self.recommendation_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def recommend_persona(
        self,
        task: Task,
        agent_id: uuid.UUID,
        context: Dict[str, Any],
        available_personas: List[PersonaDefinition]
    ) -> Tuple[PersonaDefinition, float, Dict[str, Any]]:
        """
        Recommend the best persona for a given task and context.
        
        Returns:
            (recommended_persona, confidence_score, reasoning)
        """
        
        # Check cache first
        cache_key = f"{task.id}:{agent_id}:{hash(str(sorted(context.items())))}"
        if cache_key in self.recommendation_cache:
            cached_result, timestamp = self.recommendation_cache[cache_key]
            if (datetime.utcnow() - timestamp).seconds < self.cache_ttl:
                return cached_result
        
        scores = []
        
        for persona in available_personas:
            score, reasoning = await self._calculate_persona_score(persona, task, agent_id, context)
            scores.append((persona, score, reasoning))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if not scores:
            # Fallback to generalist
            generalist = await self._get_fallback_persona()
            result = (generalist, 0.5, {"reason": "fallback_to_generalist"})
        else:
            result = scores[0]
        
        # Cache result
        self.recommendation_cache[cache_key] = (result, datetime.utcnow())
        return result
    
    async def _calculate_persona_score(
        self,
        persona: PersonaDefinition,
        task: Task,
        agent_id: uuid.UUID,
        context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate comprehensive score for persona-task-context match."""
        
        reasoning = {}
        total_score = 0.0
        
        # 1. Task type affinity (30% weight)
        task_affinity = persona.get_task_affinity(task.task_type)
        total_score += task_affinity * 0.3
        reasoning["task_affinity"] = task_affinity
        
        # 2. Required capabilities match (25% weight)
        required_capabilities = task.metadata.get("required_capabilities", [])
        if required_capabilities:
            capability_match = sum(
                persona.get_capability_score(cap) for cap in required_capabilities
            ) / len(required_capabilities)
        else:
            capability_match = 0.5  # Neutral score for no specific requirements
        
        total_score += capability_match * 0.25
        reasoning["capability_match"] = capability_match
        
        # 3. Historical performance (20% weight)
        historical_score = await self._get_historical_performance_score(persona.id, agent_id, task.task_type)
        total_score += historical_score * 0.2
        reasoning["historical_performance"] = historical_score
        
        # 4. Context compatibility (15% weight)
        context_score = self._calculate_context_compatibility(persona, context)
        total_score += context_score * 0.15
        reasoning["context_compatibility"] = context_score
        
        # 5. Current workload and availability (10% weight)
        availability_score = await self._calculate_availability_score(persona.id, agent_id)
        total_score += availability_score * 0.1
        reasoning["availability"] = availability_score
        
        return total_score, reasoning
    
    async def _get_historical_performance_score(
        self, 
        persona_id: str, 
        agent_id: uuid.UUID, 
        task_type: TaskType
    ) -> float:
        """Get historical performance score for persona on similar tasks."""
        
        # Query recent assignments for this persona and task type
        async with get_async_session() as session:
            # This would need proper database queries
            # For now, return a placeholder based on persona type
            return 0.7  # Placeholder
    
    def _calculate_context_compatibility(self, persona: PersonaDefinition, context: Dict[str, Any]) -> float:
        """Calculate how well persona fits the current context."""
        
        score = 0.0
        
        # Team size compatibility
        team_size = context.get("team_size", 1)
        if persona.preferred_team_size[0] <= team_size <= persona.preferred_team_size[1]:
            score += 0.3
        else:
            # Penalty for being outside preferred range
            distance = min(
                abs(team_size - persona.preferred_team_size[0]),
                abs(team_size - persona.preferred_team_size[1])
            )
            score += max(0.0, 0.3 - (distance * 0.1))
        
        # Urgency compatibility
        urgency = context.get("urgency", 0.5)
        speed_preference = 1.0 - persona.accuracy_vs_speed_preference
        urgency_match = 1.0 - abs(urgency - speed_preference)
        score += urgency_match * 0.4
        
        # Risk tolerance compatibility
        project_risk = context.get("risk_level", 0.5)
        risk_match = 1.0 - abs(project_risk - persona.risk_tolerance)
        score += risk_match * 0.3
        
        return score
    
    async def _calculate_availability_score(self, persona_id: str, agent_id: uuid.UUID) -> float:
        """Calculate persona availability score based on current workload."""
        
        # Check if persona is already assigned to this agent
        current_assignment = await self.persona_system.get_agent_current_persona(agent_id)
        if current_assignment and current_assignment.persona_id == persona_id:
            return 1.0  # Already using this persona
        
        # Check global usage of this persona (simplified)
        active_assignments = len(self.persona_system.active_assignments)
        
        # Prefer less heavily used personas for load balancing
        if active_assignments < 5:
            return 1.0
        elif active_assignments < 10:
            return 0.8
        else:
            return 0.6
    
    async def _get_fallback_persona(self) -> PersonaDefinition:
        """Get fallback generalist persona."""
        return await self.persona_system.get_persona("generalist_default")


class AgentPersonaSystem:
    """
    Main system for managing agent personas and role-based specialization.
    
    Provides dynamic persona assignment, performance tracking, and
    context-aware behavior adaptation for improved agent coordination.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.lifecycle_hooks = get_enhanced_lifecycle_hook_processor()
        
        # Core persona management
        self.personas: Dict[str, PersonaDefinition] = {}
        self.active_assignments: Dict[uuid.UUID, PersonaAssignment] = {}
        
        # Recommendation and analytics
        self.recommendation_engine = PersonaRecommendationEngine(self)
        self.performance_analytics = {}
        
        # Cache for frequently accessed data
        self._persona_cache = {}
        self._cache_ttl = 600  # 10 minutes
        
        logger.info("🎭 Agent Persona System initialized")
    
    async def initialize_default_personas(self):
        """Initialize the system with default persona definitions."""
        
        default_personas = [
            await self._create_backend_engineer_persona(),
            await self._create_frontend_developer_persona(), 
            await self._create_devops_specialist_persona(),
            await self._create_qa_engineer_persona(),
            await self._create_project_manager_persona(),
            await self._create_generalist_persona()
        ]
        
        for persona in default_personas:
            await self.register_persona(persona)
        
        logger.info(f"✅ Initialized {len(default_personas)} default personas")
    
    async def register_persona(self, persona: PersonaDefinition) -> bool:
        """Register a new persona definition."""
        try:
            # Validate persona definition
            await self._validate_persona_definition(persona)
            
            # Store in memory and optionally persist
            self.personas[persona.id] = persona
            
            # Clear cache
            self._persona_cache.clear()
            
            logger.info(
                f"🎭 Persona registered",
                persona_id=persona.id,
                persona_type=persona.persona_type.value,
                capabilities_count=len(persona.capabilities)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register persona {persona.id}: {e}")
            return False
    
    async def assign_persona_to_agent(
        self,
        agent_id: uuid.UUID,
        task: Optional[Task] = None,
        context: Optional[Dict[str, Any]] = None,
        preferred_persona_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> PersonaAssignment:
        """
        Assign optimal persona to agent based on task and context.
        
        Args:
            agent_id: Target agent ID
            task: Current task for context
            context: Additional context for decision making
            preferred_persona_id: Specific persona ID if known
            session_id: Session ID for tracking
            
        Returns:
            PersonaAssignment with assignment details
        """
        
        try:
            context = context or {}
            session_id = session_id or str(uuid.uuid4())
            
            # Get available personas
            available_personas = list(self.personas.values())
            
            if preferred_persona_id:
                # Use specific persona if requested
                persona = self.personas.get(preferred_persona_id)
                if not persona:
                    raise ValueError(f"Persona {preferred_persona_id} not found")
                
                confidence = 0.9  # High confidence for explicit assignment
                reasoning = {"reason": "explicit_assignment"}
                
            elif task:
                # Use recommendation engine for task-based assignment
                persona, confidence, reasoning = await self.recommendation_engine.recommend_persona(
                    task, agent_id, context, available_personas
                )
            else:
                # Default to generalist
                persona = await self._get_default_persona()
                confidence = 0.5
                reasoning = {"reason": "default_assignment"}
            
            # Create assignment
            assignment = PersonaAssignment(
                agent_id=agent_id,
                persona_id=persona.id,
                session_id=session_id,
                assigned_at=datetime.utcnow(),
                assignment_reason=reasoning.get("reason", "recommendation"),
                confidence_score=confidence
            )
            
            # Apply context adaptations
            if context:
                adaptations = persona.adapt_to_context(context)
                assignment.active_adaptations = adaptations
            
            # Store assignment
            self.active_assignments[agent_id] = assignment
            
            # Log assignment event
            await self.lifecycle_hooks.process_enhanced_event(
                session_id=uuid.UUID(session_id),
                agent_id=agent_id,
                event_type=EnhancedEventType.AGENT_COORDINATION,
                payload={
                    "event_subtype": "persona_assignment",
                    "persona_id": persona.id,
                    "persona_type": persona.persona_type.value,
                    "confidence_score": confidence,
                    "assignment_reasoning": reasoning,
                    "adaptations": assignment.active_adaptations
                }
            )
            
            logger.info(
                f"🎭 Persona assigned to agent",
                agent_id=str(agent_id),
                persona_id=persona.id,
                persona_type=persona.persona_type.value,
                confidence=confidence
            )
            
            return assignment
            
        except Exception as e:
            logger.error(f"❌ Failed to assign persona to agent {agent_id}: {e}")
            
            # Fallback to generalist
            fallback_persona = await self._get_default_persona()
            return PersonaAssignment(
                agent_id=agent_id,
                persona_id=fallback_persona.id,
                session_id=session_id or str(uuid.uuid4()),
                assigned_at=datetime.utcnow(),
                assignment_reason="fallback_assignment",
                confidence_score=0.3
            )
    
    async def get_agent_current_persona(self, agent_id: uuid.UUID) -> Optional[PersonaAssignment]:
        """Get current persona assignment for agent."""
        return self.active_assignments.get(agent_id)
    
    async def get_persona(self, persona_id: str) -> Optional[PersonaDefinition]:
        """Get persona definition by ID."""
        return self.personas.get(persona_id)
    
    async def update_persona_performance(
        self,
        agent_id: uuid.UUID,
        task: Task,
        success: bool,
        completion_time: float,
        complexity: float = 0.5
    ):
        """Update persona performance based on task completion."""
        
        assignment = self.active_assignments.get(agent_id)
        if not assignment:
            return
        
        persona = self.personas.get(assignment.persona_id)
        if not persona:
            return
        
        try:
            # Update assignment performance
            assignment.update_performance(success, completion_time)
            
            # Update persona capabilities based on task
            required_capabilities = task.metadata.get("required_capabilities", [])
            for capability_name in required_capabilities:
                if capability_name in persona.capabilities:
                    persona.capabilities[capability_name].update_proficiency(success, complexity)
            
            # Log performance update
            await self.lifecycle_hooks.process_enhanced_event(
                session_id=uuid.UUID(assignment.session_id),
                agent_id=agent_id,
                event_type=EnhancedEventType.PERFORMANCE_METRIC,
                payload={
                    "event_subtype": "persona_performance_update",
                    "persona_id": persona.id,
                    "task_id": str(task.id),
                    "success": success,
                    "completion_time": completion_time,
                    "updated_capabilities": required_capabilities,
                    "assignment_success_rate": assignment.success_rate
                }
            )
            
            logger.debug(
                f"📊 Persona performance updated",
                agent_id=str(agent_id),
                persona_id=persona.id,
                success=success,
                completion_time=completion_time
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to update persona performance: {e}")
    
    async def get_persona_analytics(
        self,
        persona_id: Optional[str] = None,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """Get comprehensive analytics for persona performance."""
        
        analytics = {
            "summary": {},
            "performance_metrics": {},
            "capability_trends": {},
            "assignment_patterns": {},
            "recommendations": []
        }
        
        try:
            if persona_id:
                # Single persona analytics
                persona = self.personas.get(persona_id)
                if persona:
                    analytics = await self._analyze_single_persona(persona, time_range_hours)
            else:
                # System-wide analytics
                analytics = await self._analyze_all_personas(time_range_hours)
            
            return analytics
            
        except Exception as e:
            logger.error(f"❌ Failed to generate persona analytics: {e}")
            return {"error": str(e)}
    
    async def list_available_personas(
        self,
        task_type: Optional[TaskType] = None,
        required_capabilities: Optional[List[str]] = None
    ) -> List[PersonaDefinition]:
        """List available personas, optionally filtered by requirements."""
        
        personas = list(self.personas.values())
        
        if task_type:
            personas = [p for p in personas if task_type in p.preferred_task_types]
        
        if required_capabilities:
            personas = [
                p for p in personas 
                if all(cap in p.capabilities for cap in required_capabilities)
            ]
        
        return personas
    
    async def remove_persona_assignment(self, agent_id: uuid.UUID) -> bool:
        """Remove persona assignment from agent."""
        try:
            assignment = self.active_assignments.pop(agent_id, None)
            if assignment:
                # Log removal event
                await self.lifecycle_hooks.process_enhanced_event(
                    session_id=uuid.UUID(assignment.session_id),
                    agent_id=agent_id,
                    event_type=EnhancedEventType.AGENT_COORDINATION,
                    payload={
                        "event_subtype": "persona_unassignment",
                        "persona_id": assignment.persona_id,
                        "final_success_rate": assignment.success_rate,
                        "tasks_completed": assignment.tasks_completed
                    }
                )
                
                logger.info(f"🎭 Persona assignment removed for agent {agent_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to remove persona assignment: {e}")
            return False
    
    # Private helper methods
    
    async def _validate_persona_definition(self, persona: PersonaDefinition):
        """Validate persona definition for correctness."""
        if not persona.id or not persona.name:
            raise ValueError("Persona must have ID and name")
        
        if not persona.capabilities:
            raise ValueError("Persona must have at least one capability")
        
        if persona.preferred_team_size[0] > persona.preferred_team_size[1]:
            raise ValueError("Invalid team size range")
    
    async def _get_default_persona(self) -> PersonaDefinition:
        """Get default generalist persona."""
        return self.personas.get("generalist_default") or await self._create_generalist_persona()
    
    async def _analyze_single_persona(self, persona: PersonaDefinition, time_range_hours: int) -> Dict[str, Any]:
        """Analyze performance metrics for a single persona."""
        
        # Count active assignments
        active_assignments = [a for a in self.active_assignments.values() if a.persona_id == persona.id]
        
        # Calculate performance metrics
        if active_assignments:
            avg_success_rate = sum(a.success_rate for a in active_assignments) / len(active_assignments)
            avg_completion_time = sum(a.avg_completion_time for a in active_assignments) / len(active_assignments)
            total_tasks = sum(a.tasks_completed for a in active_assignments)
        else:
            avg_success_rate = 0.0
            avg_completion_time = 0.0
            total_tasks = 0
        
        return {
            "summary": {
                "persona_id": persona.id,
                "persona_type": persona.persona_type.value,
                "active_assignments": len(active_assignments),
                "total_tasks_completed": total_tasks,
                "average_success_rate": avg_success_rate,
                "average_completion_time": avg_completion_time
            },
            "capabilities": {
                name: {
                    "level": cap.level.value,
                    "proficiency": cap.proficiency_score,
                    "confidence": cap.confidence,
                    "usage_count": cap.usage_count,
                    "success_rate": cap.success_rate
                }
                for name, cap in persona.capabilities.items()
            }
        }
    
    async def _analyze_all_personas(self, time_range_hours: int) -> Dict[str, Any]:
        """Analyze performance across all personas."""
        
        persona_summaries = {}
        for persona_id, persona in self.personas.items():
            persona_summaries[persona_id] = await self._analyze_single_persona(persona, time_range_hours)
        
        # System-wide metrics
        total_assignments = len(self.active_assignments)
        total_personas = len(self.personas)
        
        return {
            "summary": {
                "total_personas": total_personas,
                "active_assignments": total_assignments,
                "assignment_rate": total_assignments / max(1, total_personas)
            },
            "persona_details": persona_summaries
        }
    
    # Default persona creation methods
    
    async def _create_backend_engineer_persona(self) -> PersonaDefinition:
        """Create backend engineer persona."""
        capabilities = {
            "api_development": PersonaCapability("api_development", PersonaCapabilityLevel.EXPERT, 0.9, 0.85),
            "database_design": PersonaCapability("database_design", PersonaCapabilityLevel.ADVANCED, 0.8, 0.8),
            "system_architecture": PersonaCapability("system_architecture", PersonaCapabilityLevel.ADVANCED, 0.75, 0.7),
            "performance_optimization": PersonaCapability("performance_optimization", PersonaCapabilityLevel.INTERMEDIATE, 0.6, 0.65),
            "security_implementation": PersonaCapability("security_implementation", PersonaCapabilityLevel.INTERMEDIATE, 0.55, 0.6)
        }
        
        return PersonaDefinition(
            id="backend_engineer_default",
            name="Backend Engineer",
            description="Specialized in server-side development, APIs, and system architecture",
            persona_type=PersonaType.BACKEND_ENGINEER,
            adaptation_mode=PersonaAdaptationMode.ADAPTIVE,
            capabilities=capabilities,
            preferred_task_types=[TaskType.CODE_GENERATION, TaskType.CODE_REVIEW],
            expertise_domains=["api_development", "database_management", "microservices"],
            communication_style={"formality": "technical", "detail_level": "high"},
            decision_making_style={"approach": "analytical", "risk_preference": "moderate"},
            problem_solving_approach={"style": "systematic", "collaboration": "moderate"},
            preferred_team_size=(2, 6),
            collaboration_patterns=["code_review", "pair_programming", "architecture_discussion"],
            mentoring_capability=True,
            typical_response_time=120.0,
            accuracy_vs_speed_preference=0.75,
            risk_tolerance=0.4,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
    
    async def _create_frontend_developer_persona(self) -> PersonaDefinition:
        """Create frontend developer persona."""
        capabilities = {
            "ui_development": PersonaCapability("ui_development", PersonaCapabilityLevel.EXPERT, 0.9, 0.85),
            "user_experience": PersonaCapability("user_experience", PersonaCapabilityLevel.ADVANCED, 0.8, 0.75),
            "responsive_design": PersonaCapability("responsive_design", PersonaCapabilityLevel.ADVANCED, 0.75, 0.8),
            "javascript_frameworks": PersonaCapability("javascript_frameworks", PersonaCapabilityLevel.EXPERT, 0.85, 0.8),
            "performance_optimization": PersonaCapability("performance_optimization", PersonaCapabilityLevel.INTERMEDIATE, 0.6, 0.65)
        }
        
        return PersonaDefinition(
            id="frontend_developer_default",
            name="Frontend Developer",
            description="Specialized in user interfaces, user experience, and client-side development",
            persona_type=PersonaType.FRONTEND_DEVELOPER,
            adaptation_mode=PersonaAdaptationMode.DYNAMIC,
            capabilities=capabilities,
            preferred_task_types=[TaskType.CODE_GENERATION, TaskType.CODE_REVIEW],
            expertise_domains=["ui_development", "user_experience", "javascript", "css"],
            communication_style={"formality": "casual", "visual_preference": "high"},
            decision_making_style={"approach": "creative", "user_focused": True},
            problem_solving_approach={"style": "iterative", "prototyping": True},
            preferred_team_size=(1, 4),
            collaboration_patterns=["design_review", "user_testing", "pair_programming"],
            mentoring_capability=True,
            typical_response_time=90.0,
            accuracy_vs_speed_preference=0.6,
            risk_tolerance=0.6,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
    
    async def _create_devops_specialist_persona(self) -> PersonaDefinition:
        """Create DevOps specialist persona."""
        capabilities = {
            "deployment_automation": PersonaCapability("deployment_automation", PersonaCapabilityLevel.EXPERT, 0.9, 0.85),
            "infrastructure_management": PersonaCapability("infrastructure_management", PersonaCapabilityLevel.EXPERT, 0.85, 0.8),
            "monitoring_systems": PersonaCapability("monitoring_systems", PersonaCapabilityLevel.ADVANCED, 0.75, 0.75),
            "container_orchestration": PersonaCapability("container_orchestration", PersonaCapabilityLevel.ADVANCED, 0.8, 0.7),
            "security_operations": PersonaCapability("security_operations", PersonaCapabilityLevel.INTERMEDIATE, 0.65, 0.7)
        }
        
        return PersonaDefinition(
            id="devops_specialist_default",
            name="DevOps Specialist",
            description="Specialized in deployment, infrastructure, and operational excellence",
            persona_type=PersonaType.DEVOPS_SPECIALIST,
            adaptation_mode=PersonaAdaptationMode.ADAPTIVE,
            capabilities=capabilities,
            preferred_task_types=[TaskType.DEPLOYMENT, TaskType.TESTING],
            expertise_domains=["deployment", "infrastructure", "monitoring", "automation"],
            communication_style={"formality": "technical", "automation_focused": True},
            decision_making_style={"approach": "pragmatic", "reliability_focused": True},
            problem_solving_approach={"style": "systematic", "automation_first": True},
            preferred_team_size=(1, 3),
            collaboration_patterns=["incident_response", "capacity_planning", "automation_review"],
            mentoring_capability=True,
            typical_response_time=60.0,
            accuracy_vs_speed_preference=0.8,
            risk_tolerance=0.3,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
    
    async def _create_qa_engineer_persona(self) -> PersonaDefinition:
        """Create QA engineer persona."""
        capabilities = {
            "test_automation": PersonaCapability("test_automation", PersonaCapabilityLevel.EXPERT, 0.9, 0.85),
            "quality_assurance": PersonaCapability("quality_assurance", PersonaCapabilityLevel.EXPERT, 0.85, 0.9),
            "bug_detection": PersonaCapability("bug_detection", PersonaCapabilityLevel.ADVANCED, 0.8, 0.85),
            "performance_testing": PersonaCapability("performance_testing", PersonaCapabilityLevel.INTERMEDIATE, 0.65, 0.7),
            "security_testing": PersonaCapability("security_testing", PersonaCapabilityLevel.INTERMEDIATE, 0.6, 0.65)
        }
        
        return PersonaDefinition(
            id="qa_engineer_default",
            name="QA Engineer",
            description="Specialized in quality assurance, testing, and defect prevention",
            persona_type=PersonaType.QA_ENGINEER,
            adaptation_mode=PersonaAdaptationMode.ADAPTIVE,
            capabilities=capabilities,
            preferred_task_types=[TaskType.TESTING, TaskType.CODE_REVIEW],
            expertise_domains=["testing", "quality_assurance", "automation", "defect_analysis"],
            communication_style={"formality": "professional", "detail_oriented": True},
            decision_making_style={"approach": "thorough", "quality_focused": True},
            problem_solving_approach={"style": "methodical", "prevention_focused": True},
            preferred_team_size=(1, 5),
            collaboration_patterns=["test_planning", "defect_triage", "quality_review"],
            mentoring_capability=True,
            typical_response_time=150.0,
            accuracy_vs_speed_preference=0.9,
            risk_tolerance=0.2,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
    
    async def _create_project_manager_persona(self) -> PersonaDefinition:
        """Create project manager persona."""
        capabilities = {
            "project_planning": PersonaCapability("project_planning", PersonaCapabilityLevel.EXPERT, 0.9, 0.85),
            "team_coordination": PersonaCapability("team_coordination", PersonaCapabilityLevel.EXPERT, 0.85, 0.9),
            "stakeholder_management": PersonaCapability("stakeholder_management", PersonaCapabilityLevel.ADVANCED, 0.75, 0.8),
            "risk_management": PersonaCapability("risk_management", PersonaCapabilityLevel.ADVANCED, 0.7, 0.75),
            "resource_optimization": PersonaCapability("resource_optimization", PersonaCapabilityLevel.INTERMEDIATE, 0.65, 0.7)
        }
        
        return PersonaDefinition(
            id="project_manager_default",
            name="Project Manager",
            description="Specialized in project coordination, planning, and team management",
            persona_type=PersonaType.PROJECT_MANAGER,
            adaptation_mode=PersonaAdaptationMode.COLLABORATIVE,
            capabilities=capabilities,
            preferred_task_types=[TaskType.COORDINATION, TaskType.PLANNING],
            expertise_domains=["project_management", "team_coordination", "planning", "communication"],
            communication_style={"formality": "professional", "clarity_focused": True},
            decision_making_style={"approach": "collaborative", "consensus_building": True},
            problem_solving_approach={"style": "holistic", "stakeholder_inclusive": True},
            preferred_team_size=(3, 12),
            collaboration_patterns=["daily_standups", "sprint_planning", "retrospectives"],
            mentoring_capability=True,
            typical_response_time=45.0,
            accuracy_vs_speed_preference=0.65,
            risk_tolerance=0.5,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
    
    async def _create_generalist_persona(self) -> PersonaDefinition:
        """Create generalist persona."""
        capabilities = {
            "general_programming": PersonaCapability("general_programming", PersonaCapabilityLevel.INTERMEDIATE, 0.65, 0.7),
            "problem_solving": PersonaCapability("problem_solving", PersonaCapabilityLevel.ADVANCED, 0.75, 0.8),
            "learning_adaptation": PersonaCapability("learning_adaptation", PersonaCapabilityLevel.EXPERT, 0.85, 0.8),
            "communication": PersonaCapability("communication", PersonaCapabilityLevel.ADVANCED, 0.8, 0.75),
            "collaboration": PersonaCapability("collaboration", PersonaCapabilityLevel.ADVANCED, 0.75, 0.8)
        }
        
        return PersonaDefinition(
            id="generalist_default",
            name="Generalist",
            description="Versatile agent capable of handling diverse tasks across domains",
            persona_type=PersonaType.GENERALIST,
            adaptation_mode=PersonaAdaptationMode.DYNAMIC,
            capabilities=capabilities,
            preferred_task_types=list(TaskType),  # All task types
            expertise_domains=["general_purpose", "learning", "adaptation", "communication"],
            communication_style={"formality": "adaptive", "context_aware": True},
            decision_making_style={"approach": "balanced", "context_driven": True},
            problem_solving_approach={"style": "flexible", "learning_oriented": True},
            preferred_team_size=(1, 8),
            collaboration_patterns=["flexible_collaboration", "cross_functional_support"],
            mentoring_capability=False,
            typical_response_time=90.0,
            accuracy_vs_speed_preference=0.65,
            risk_tolerance=0.5,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )


# Factory function and convenience methods

def get_agent_persona_system() -> AgentPersonaSystem:
    """Get the global agent persona system instance."""
    if not hasattr(get_agent_persona_system, '_instance'):
        get_agent_persona_system._instance = AgentPersonaSystem()
    return get_agent_persona_system._instance


async def assign_optimal_persona(
    agent_id: uuid.UUID,
    task: Optional[Task] = None,
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> PersonaAssignment:
    """Convenience function to assign optimal persona to agent."""
    persona_system = get_agent_persona_system()
    return await persona_system.assign_persona_to_agent(agent_id, task, context, None, session_id)


async def get_agent_persona(agent_id: uuid.UUID) -> Optional[PersonaDefinition]:
    """Convenience function to get agent's current persona."""
    persona_system = get_agent_persona_system()
    assignment = await persona_system.get_agent_current_persona(agent_id)
    if assignment:
        return await persona_system.get_persona(assignment.persona_id)
    return None


async def update_agent_persona_performance(
    agent_id: uuid.UUID,
    task: Task,
    success: bool,
    completion_time: float,
    complexity: float = 0.5
):
    """Convenience function to update persona performance."""
    persona_system = get_agent_persona_system()
    await persona_system.update_persona_performance(agent_id, task, success, completion_time, complexity)