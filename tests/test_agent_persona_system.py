"""
Comprehensive tests for Agent Persona System.

Tests include:
- Persona definition creation and management
- Dynamic persona assignment and recommendation engine
- Performance tracking and capability evolution
- Context-aware behavior adaptation
- Analytics and optimization recommendations
- Integration with existing agent and task systems
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from app.core.agent_persona_system import (
    AgentPersonaSystem,
    PersonaDefinition,
    PersonaAssignment,
    PersonaCapability,
    PersonaType,
    PersonaAdaptationMode,
    PersonaCapabilityLevel,
    PersonaRecommendationEngine,
    get_agent_persona_system,
    assign_optimal_persona,
    get_agent_persona,
    update_agent_persona_performance
)
from app.models.task import Task, TaskType, TaskStatus
from app.models.agent import Agent, AgentStatus


class TestPersonaCapability:
    """Test persona capability tracking and evolution."""
    
    def test_capability_creation(self):
        """Test capability creation with all parameters."""
        capability = PersonaCapability(
            name="api_development",
            level=PersonaCapabilityLevel.ADVANCED,
            proficiency_score=0.75,
            confidence=0.8
        )
        
        assert capability.name == "api_development"
        assert capability.level == PersonaCapabilityLevel.ADVANCED
        assert capability.proficiency_score == 0.75
        assert capability.confidence == 0.8
        assert capability.usage_count == 0
        assert capability.success_rate == 0.0
        assert capability.last_used is None
    
    def test_capability_proficiency_update_success(self):
        """Test capability proficiency improvement on success."""
        capability = PersonaCapability(
            name="testing",
            level=PersonaCapabilityLevel.INTERMEDIATE,
            proficiency_score=0.5,
            confidence=0.6
        )
        
        initial_score = capability.proficiency_score
        capability.update_proficiency(success=True, complexity=0.7)
        
        assert capability.proficiency_score > initial_score
        assert capability.usage_count == 1
        assert capability.success_rate == 1.0
        assert capability.confidence > 0.6
        assert capability.last_used is not None
    
    def test_capability_proficiency_update_failure(self):
        """Test capability proficiency decrease on failure."""
        capability = PersonaCapability(
            name="debugging",
            level=PersonaCapabilityLevel.ADVANCED,
            proficiency_score=0.8,
            confidence=0.75
        )
        
        initial_score = capability.proficiency_score
        capability.update_proficiency(success=False, complexity=0.5)
        
        assert capability.proficiency_score < initial_score
        assert capability.usage_count == 1
        assert capability.success_rate == 0.0
        assert capability.confidence < 0.75
    
    def test_capability_level_promotion(self):
        """Test automatic level promotion based on proficiency."""
        capability = PersonaCapability(
            name="deployment",
            level=PersonaCapabilityLevel.INTERMEDIATE,
            proficiency_score=0.59,
            confidence=0.6
        )
        
        # Multiple successful tasks to increase proficiency
        for _ in range(20):
            capability.update_proficiency(success=True, complexity=0.8)
        
        # Should promote to advanced or expert
        assert capability.level in [PersonaCapabilityLevel.ADVANCED, PersonaCapabilityLevel.EXPERT]
        assert capability.proficiency_score > 0.6
    
    def test_capability_mixed_performance(self):
        """Test capability evolution with mixed success/failure."""
        capability = PersonaCapability(
            name="code_review",
            level=PersonaCapabilityLevel.INTERMEDIATE,
            proficiency_score=0.5,
            confidence=0.5
        )
        
        # Simulate mixed performance
        for i in range(10):
            success = i % 3 != 0  # 2/3 success rate
            capability.update_proficiency(success=success, complexity=0.6)
        
        assert capability.usage_count == 10
        assert 0.6 <= capability.success_rate <= 0.7
        assert 0.4 <= capability.confidence <= 0.8


class TestPersonaDefinition:
    """Test persona definition and behavior."""
    
    @pytest.fixture
    def sample_capabilities(self):
        """Sample capabilities for testing."""
        return {
            "programming": PersonaCapability("programming", PersonaCapabilityLevel.EXPERT, 0.9, 0.85),
            "debugging": PersonaCapability("debugging", PersonaCapabilityLevel.ADVANCED, 0.75, 0.8),
            "testing": PersonaCapability("testing", PersonaCapabilityLevel.INTERMEDIATE, 0.6, 0.65)
        }
    
    @pytest.fixture
    def sample_persona(self, sample_capabilities):
        """Sample persona definition for testing."""
        return PersonaDefinition(
            id="test_backend_engineer",
            name="Test Backend Engineer",
            description="Backend engineering specialist for testing",
            persona_type=PersonaType.BACKEND_ENGINEER,
            adaptation_mode=PersonaAdaptationMode.ADAPTIVE,
            capabilities=sample_capabilities,
            preferred_task_types=[TaskType.CODE_GENERATION, TaskType.CODE_REVIEW],
            expertise_domains=["api_development", "database_design"],
            communication_style={"formality": "technical"},
            decision_making_style={"approach": "analytical"},
            problem_solving_approach={"style": "systematic"},
            preferred_team_size=(2, 6),
            collaboration_patterns=["code_review", "pair_programming"],
            mentoring_capability=True,
            typical_response_time=120.0,
            accuracy_vs_speed_preference=0.75,
            risk_tolerance=0.4,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
    
    def test_persona_creation(self, sample_persona):
        """Test persona definition creation."""
        assert sample_persona.id == "test_backend_engineer"
        assert sample_persona.persona_type == PersonaType.BACKEND_ENGINEER
        assert len(sample_persona.capabilities) == 3
        assert len(sample_persona.preferred_task_types) == 2
        assert sample_persona.mentoring_capability is True
    
    def test_capability_score_retrieval(self, sample_persona):
        """Test getting capability scores."""
        assert sample_persona.get_capability_score("programming") == 0.9
        assert sample_persona.get_capability_score("debugging") == 0.75
        assert sample_persona.get_capability_score("nonexistent") == 0.0
    
    def test_task_affinity_calculation(self, sample_persona):
        """Test task affinity scoring."""
        # Preferred task type
        affinity_preferred = sample_persona.get_task_affinity(TaskType.CODE_GENERATION)
        assert affinity_preferred == 1.0
        
        # Related task type
        affinity_related = sample_persona.get_task_affinity(TaskType.TESTING)
        assert 0.0 < affinity_related < 1.0
        
        # Unrelated task type
        affinity_unrelated = sample_persona.get_task_affinity(TaskType.DOCUMENTATION)
        assert affinity_unrelated == 0.3  # Base affinity
    
    def test_context_adaptation_static(self, sample_capabilities):
        """Test static adaptation mode (no adaptations)."""
        persona = PersonaDefinition(
            id="static_persona",
            name="Static Persona",
            description="Non-adaptive persona",
            persona_type=PersonaType.GENERALIST,
            adaptation_mode=PersonaAdaptationMode.STATIC,
            capabilities=sample_capabilities,
            preferred_task_types=[TaskType.CODE_GENERATION],
            expertise_domains=["general"],
            communication_style={},
            decision_making_style={},
            problem_solving_approach={},
            preferred_team_size=(1, 5),
            collaboration_patterns=[],
            mentoring_capability=False,
            typical_response_time=90.0,
            accuracy_vs_speed_preference=0.5,
            risk_tolerance=0.5,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        adaptations = persona.adapt_to_context({"team_size": 10, "urgency": 0.9})
        assert len(adaptations) == 0
    
    def test_context_adaptation_team_size(self, sample_persona):
        """Test adaptation based on team size."""
        # Too small team
        small_team_context = {"team_size": 1}
        adaptations = sample_persona.adapt_to_context(small_team_context)
        assert "increase_autonomy" in adaptations
        
        # Too large team
        large_team_context = {"team_size": 10}
        adaptations = sample_persona.adapt_to_context(large_team_context)
        assert "focus_specialization" in adaptations
    
    def test_context_adaptation_complexity(self, sample_persona):
        """Test adaptation based on project complexity."""
        high_complexity_context = {"project_complexity": 0.9}
        adaptations = sample_persona.adapt_to_context(high_complexity_context)
        
        assert "request_collaboration" in adaptations
        assert "increase_planning_time" in adaptations
    
    def test_context_adaptation_urgency(self, sample_persona):
        """Test adaptation based on time pressure."""
        urgent_context = {"urgency": 0.8}
        adaptations = sample_persona.adapt_to_context(urgent_context)
        
        assert "temp_speed_preference" in adaptations
        # Should be lower than original preference (more speed-focused)
        assert adaptations["temp_speed_preference"] < sample_persona.accuracy_vs_speed_preference


class TestPersonaAssignment:
    """Test persona assignment functionality."""
    
    @pytest.fixture
    def sample_assignment(self):
        """Sample persona assignment for testing."""
        agent_id = uuid.uuid4()
        return PersonaAssignment(
            agent_id=agent_id,
            persona_id="backend_engineer_default",
            session_id="test_session_123",
            assigned_at=datetime.utcnow(),
            assignment_reason="recommendation",
            confidence_score=0.85
        )
    
    def test_assignment_creation(self, sample_assignment):
        """Test assignment creation with defaults."""
        assert isinstance(sample_assignment.agent_id, uuid.UUID)
        assert sample_assignment.persona_id == "backend_engineer_default"
        assert sample_assignment.confidence_score == 0.85
        assert sample_assignment.tasks_completed == 0
        assert sample_assignment.success_rate == 0.0
        assert sample_assignment.active_adaptations == {}
    
    def test_performance_update_success(self, sample_assignment):
        """Test performance update with successful task."""
        initial_tasks = sample_assignment.tasks_completed
        
        sample_assignment.update_performance(task_success=True, completion_time=150.0)
        
        assert sample_assignment.tasks_completed == initial_tasks + 1
        assert sample_assignment.success_rate == 1.0
        assert sample_assignment.avg_completion_time == 150.0
    
    def test_performance_update_failure(self, sample_assignment):
        """Test performance update with failed task."""
        sample_assignment.update_performance(task_success=False, completion_time=300.0)
        
        assert sample_assignment.tasks_completed == 1
        assert sample_assignment.success_rate == 0.0
        assert sample_assignment.avg_completion_time == 300.0
    
    def test_performance_update_mixed(self, sample_assignment):
        """Test performance tracking with mixed results."""
        # Three tasks: success, failure, success
        sample_assignment.update_performance(task_success=True, completion_time=100.0)
        sample_assignment.update_performance(task_success=False, completion_time=200.0)
        sample_assignment.update_performance(task_success=True, completion_time=150.0)
        
        assert sample_assignment.tasks_completed == 3
        assert abs(sample_assignment.success_rate - 2/3) < 0.01
        assert abs(sample_assignment.avg_completion_time - 150.0) < 1.0


class TestPersonaRecommendationEngine:
    """Test persona recommendation engine."""
    
    @pytest.fixture
    def mock_persona_system(self):
        """Mock persona system for testing."""
        mock_system = Mock()
        mock_system.personas = {}
        mock_system.active_assignments = {}
        return mock_system
    
    @pytest.fixture
    def recommendation_engine(self, mock_persona_system):
        """Recommendation engine for testing."""
        return PersonaRecommendationEngine(mock_persona_system)
    
    @pytest.fixture
    def sample_personas(self):
        """Sample personas for recommendation testing."""
        backend_capabilities = {
            "api_development": PersonaCapability("api_development", PersonaCapabilityLevel.EXPERT, 0.9, 0.85),
            "database_design": PersonaCapability("database_design", PersonaCapabilityLevel.ADVANCED, 0.8, 0.75)
        }
        
        frontend_capabilities = {
            "ui_development": PersonaCapability("ui_development", PersonaCapabilityLevel.EXPERT, 0.9, 0.85),
            "user_experience": PersonaCapability("user_experience", PersonaCapabilityLevel.ADVANCED, 0.8, 0.8)
        }
        
        backend_persona = PersonaDefinition(
            id="backend_specialist",
            name="Backend Specialist",
            description="Backend development expert",
            persona_type=PersonaType.BACKEND_ENGINEER,
            adaptation_mode=PersonaAdaptationMode.ADAPTIVE,
            capabilities=backend_capabilities,
            preferred_task_types=[TaskType.CODE_GENERATION, TaskType.CODE_REVIEW],
            expertise_domains=["api_development", "database_design"],
            communication_style={},
            decision_making_style={},
            problem_solving_approach={},
            preferred_team_size=(2, 6),
            collaboration_patterns=[],
            mentoring_capability=True,
            typical_response_time=120.0,
            accuracy_vs_speed_preference=0.8,
            risk_tolerance=0.3,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        frontend_persona = PersonaDefinition(
            id="frontend_specialist",
            name="Frontend Specialist",
            description="Frontend development expert",
            persona_type=PersonaType.FRONTEND_DEVELOPER,
            adaptation_mode=PersonaAdaptationMode.DYNAMIC,
            capabilities=frontend_capabilities,
            preferred_task_types=[TaskType.CODE_GENERATION],
            expertise_domains=["ui_development", "user_experience"],
            communication_style={},
            decision_making_style={},
            problem_solving_approach={},
            preferred_team_size=(1, 4),
            collaboration_patterns=[],
            mentoring_capability=False,
            typical_response_time=90.0,
            accuracy_vs_speed_preference=0.6,
            risk_tolerance=0.6,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        return [backend_persona, frontend_persona]
    
    @pytest.fixture
    def sample_task(self):
        """Sample task for recommendation testing."""
        task = Mock()
        task.id = uuid.uuid4()
        task.task_type = TaskType.CODE_GENERATION
        task.metadata = {"required_capabilities": ["api_development"]}
        return task
    
    @pytest.mark.asyncio
    async def test_recommend_persona_for_backend_task(self, recommendation_engine, sample_personas, sample_task):
        """Test persona recommendation for backend task."""
        agent_id = uuid.uuid4()
        context = {"team_size": 3, "urgency": 0.5}
        
        with patch.object(recommendation_engine, '_get_historical_performance_score', return_value=0.7), \
             patch.object(recommendation_engine, '_calculate_availability_score', return_value=0.9), \
             patch.object(recommendation_engine, '_get_fallback_persona', return_value=sample_personas[0]):
            
            persona, confidence, reasoning = await recommendation_engine.recommend_persona(
                sample_task, agent_id, context, sample_personas
            )
            
            # Should recommend backend specialist for backend task
            assert persona.id == "backend_specialist"
            assert confidence > 0.5
            assert "task_affinity" in reasoning
            assert "capability_match" in reasoning
    
    @pytest.mark.asyncio
    async def test_recommend_persona_context_compatibility(self, recommendation_engine, sample_personas, sample_task):
        """Test context compatibility in recommendations."""
        agent_id = uuid.uuid4()
        
        # Context favoring frontend persona (smaller team, higher urgency)
        frontend_context = {"team_size": 2, "urgency": 0.8}
        
        with patch.object(recommendation_engine, '_get_historical_performance_score', return_value=0.5), \
             patch.object(recommendation_engine, '_calculate_availability_score', return_value=0.8):
            
            persona, confidence, reasoning = await recommendation_engine.recommend_persona(
                sample_task, agent_id, frontend_context, sample_personas
            )
            
            assert "context_compatibility" in reasoning
            # Context compatibility should influence the decision
            assert reasoning["context_compatibility"] > 0.0
    
    def test_context_compatibility_calculation(self, recommendation_engine, sample_personas):
        """Test context compatibility scoring."""
        backend_persona = sample_personas[0]  # Preferred team size (2, 6)
        
        # Perfect match context
        perfect_context = {
            "team_size": 4,  # Within preferred range
            "urgency": 0.2,  # Matches accuracy preference (low urgency, high accuracy)
            "risk_level": 0.3  # Matches risk tolerance
        }
        
        perfect_score = recommendation_engine._calculate_context_compatibility(backend_persona, perfect_context)
        
        # Poor match context
        poor_context = {
            "team_size": 15,  # Outside preferred range
            "urgency": 0.9,   # High urgency vs accuracy preference
            "risk_level": 0.9  # High risk vs low tolerance
        }
        
        poor_score = recommendation_engine._calculate_context_compatibility(backend_persona, poor_context)
        
        assert perfect_score > poor_score
        assert 0.0 <= poor_score <= 1.0
        assert 0.0 <= perfect_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_recommendation_caching(self, recommendation_engine, sample_personas, sample_task):
        """Test recommendation result caching."""
        agent_id = uuid.uuid4()
        context = {"team_size": 3}
        
        with patch.object(recommendation_engine, '_calculate_persona_score', return_value=(0.8, {})) as mock_calculate:
            # First call
            persona1, confidence1, reasoning1 = await recommendation_engine.recommend_persona(
                sample_task, agent_id, context, sample_personas
            )
            
            # Second call (should use cache)
            persona2, confidence2, reasoning2 = await recommendation_engine.recommend_persona(
                sample_task, agent_id, context, sample_personas
            )
            
            # Should return same results
            assert persona1.id == persona2.id
            assert confidence1 == confidence2
            
            # Should only calculate scores once (due to caching)
            assert mock_calculate.call_count == len(sample_personas)


class TestAgentPersonaSystem:
    """Test main agent persona system functionality."""
    
    @pytest.fixture
    def persona_system(self):
        """Fresh persona system for each test."""
        return AgentPersonaSystem()
    
    @pytest.fixture
    def sample_persona_definition(self):
        """Sample persona definition for testing."""
        capabilities = {
            "testing": PersonaCapability("testing", PersonaCapabilityLevel.EXPERT, 0.9, 0.85)
        }
        
        return PersonaDefinition(
            id="test_qa_engineer",
            name="Test QA Engineer",
            description="QA engineering specialist",
            persona_type=PersonaType.QA_ENGINEER,
            adaptation_mode=PersonaAdaptationMode.ADAPTIVE,
            capabilities=capabilities,
            preferred_task_types=[TaskType.TESTING],
            expertise_domains=["quality_assurance"],
            communication_style={},
            decision_making_style={},
            problem_solving_approach={},
            preferred_team_size=(1, 5),
            collaboration_patterns=[],
            mentoring_capability=True,
            typical_response_time=150.0,
            accuracy_vs_speed_preference=0.9,
            risk_tolerance=0.2,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
    
    @pytest.mark.asyncio
    async def test_persona_system_initialization(self, persona_system):
        """Test persona system initializes correctly."""
        assert len(persona_system.personas) == 0
        assert len(persona_system.active_assignments) == 0
        assert isinstance(persona_system.recommendation_engine, PersonaRecommendationEngine)
    
    @pytest.mark.asyncio
    async def test_register_persona(self, persona_system, sample_persona_definition):
        """Test persona registration."""
        success = await persona_system.register_persona(sample_persona_definition)
        
        assert success is True
        assert sample_persona_definition.id in persona_system.personas
        assert persona_system.personas[sample_persona_definition.id] == sample_persona_definition
    
    @pytest.mark.asyncio
    async def test_register_persona_validation_failure(self, persona_system):
        """Test persona registration with invalid definition."""
        # Invalid persona (missing required fields)
        invalid_persona = PersonaDefinition(
            id="",  # Empty ID should fail validation
            name="Invalid Persona",
            description="This should fail",
            persona_type=PersonaType.GENERALIST,
            adaptation_mode=PersonaAdaptationMode.STATIC,
            capabilities={},  # Empty capabilities should fail
            preferred_task_types=[],
            expertise_domains=[],
            communication_style={},
            decision_making_style={},
            problem_solving_approach={},
            preferred_team_size=(3, 2),  # Invalid range should fail
            collaboration_patterns=[],
            mentoring_capability=False,
            typical_response_time=90.0,
            accuracy_vs_speed_preference=0.5,
            risk_tolerance=0.5,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        success = await persona_system.register_persona(invalid_persona)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_assign_persona_to_agent(self, persona_system, sample_persona_definition):
        """Test persona assignment to agent."""
        await persona_system.register_persona(sample_persona_definition)
        agent_id = uuid.uuid4()
        
        with patch.object(persona_system.lifecycle_hooks, 'process_enhanced_event') as mock_hook:
            assignment = await persona_system.assign_persona_to_agent(
                agent_id=agent_id,
                preferred_persona_id=sample_persona_definition.id,
                session_id="test_session"
            )
            
            assert assignment.agent_id == agent_id
            assert assignment.persona_id == sample_persona_definition.id
            assert assignment.confidence_score == 0.9  # High confidence for explicit assignment
            assert agent_id in persona_system.active_assignments
            mock_hook.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_assign_persona_with_task_context(self, persona_system, sample_persona_definition):
        """Test persona assignment with task context."""
        await persona_system.register_persona(sample_persona_definition)
        agent_id = uuid.uuid4()
        
        # Create mock task
        task = Mock()
        task.id = uuid.uuid4()
        task.task_type = TaskType.TESTING
        task.metadata = {"required_capabilities": ["testing"]}
        
        context = {"team_size": 3, "urgency": 0.5}
        
        assignment = await persona_system.assign_persona_to_agent(
            agent_id=agent_id,
            task=task,
            context=context,
            session_id="test_session"
        )
        
        assert assignment.agent_id == agent_id
        assert assignment.persona_id == sample_persona_definition.id
        assert assignment.confidence_score > 0.5
    
    @pytest.mark.asyncio
    async def test_get_agent_current_persona(self, persona_system, sample_persona_definition):
        """Test getting current persona assignment."""
        await persona_system.register_persona(sample_persona_definition)
        agent_id = uuid.uuid4()
        
        # No assignment initially
        assignment = await persona_system.get_agent_current_persona(agent_id)
        assert assignment is None
        
        # After assignment
        await persona_system.assign_persona_to_agent(
            agent_id=agent_id,
            preferred_persona_id=sample_persona_definition.id
        )
        
        assignment = await persona_system.get_agent_current_persona(agent_id)
        assert assignment is not None
        assert assignment.agent_id == agent_id
        assert assignment.persona_id == sample_persona_definition.id
    
    @pytest.mark.asyncio
    async def test_update_persona_performance(self, persona_system, sample_persona_definition):
        """Test persona performance updating."""
        await persona_system.register_persona(sample_persona_definition)
        agent_id = uuid.uuid4()
        
        # Assign persona first
        await persona_system.assign_persona_to_agent(
            agent_id=agent_id,
            preferred_persona_id=sample_persona_definition.id
        )
        
        # Create mock task
        task = Mock()
        task.id = uuid.uuid4()
        task.metadata = {"required_capabilities": ["testing"]}
        
        initial_proficiency = sample_persona_definition.capabilities["testing"].proficiency_score
        
        with patch.object(persona_system.lifecycle_hooks, 'process_enhanced_event'):
            await persona_system.update_persona_performance(
                agent_id=agent_id,
                task=task,
                success=True,
                completion_time=120.0,
                complexity=0.7
            )
            
            # Performance should improve
            current_proficiency = sample_persona_definition.capabilities["testing"].proficiency_score
            assert current_proficiency >= initial_proficiency
            
            # Assignment metrics should update
            assignment = await persona_system.get_agent_current_persona(agent_id)
            assert assignment.tasks_completed == 1
            assert assignment.success_rate == 1.0
    
    @pytest.mark.asyncio
    async def test_list_available_personas(self, persona_system, sample_persona_definition):
        """Test listing available personas with filters."""
        await persona_system.register_persona(sample_persona_definition)
        
        # List all personas
        all_personas = await persona_system.list_available_personas()
        assert len(all_personas) == 1
        assert all_personas[0].id == sample_persona_definition.id
        
        # Filter by task type
        testing_personas = await persona_system.list_available_personas(task_type=TaskType.TESTING)
        assert len(testing_personas) == 1
        
        code_personas = await persona_system.list_available_personas(task_type=TaskType.CODE_GENERATION)
        assert len(code_personas) == 0
        
        # Filter by required capabilities
        qa_personas = await persona_system.list_available_personas(required_capabilities=["testing"])
        assert len(qa_personas) == 1
        
        dev_personas = await persona_system.list_available_personas(required_capabilities=["programming"])
        assert len(dev_personas) == 0
    
    @pytest.mark.asyncio
    async def test_remove_persona_assignment(self, persona_system, sample_persona_definition):
        """Test removing persona assignment."""
        await persona_system.register_persona(sample_persona_definition)
        agent_id = uuid.uuid4()
        
        # Assign persona
        await persona_system.assign_persona_to_agent(
            agent_id=agent_id,
            preferred_persona_id=sample_persona_definition.id
        )
        
        assert agent_id in persona_system.active_assignments
        
        # Remove assignment
        with patch.object(persona_system.lifecycle_hooks, 'process_enhanced_event'):
            success = await persona_system.remove_persona_assignment(agent_id)
            
            assert success is True
            assert agent_id not in persona_system.active_assignments
    
    @pytest.mark.asyncio
    async def test_get_persona_analytics(self, persona_system, sample_persona_definition):
        """Test persona analytics generation."""
        await persona_system.register_persona(sample_persona_definition)
        agent_id = uuid.uuid4()
        
        # Assign persona and add some performance data
        await persona_system.assign_persona_to_agent(
            agent_id=agent_id,
            preferred_persona_id=sample_persona_definition.id
        )
        
        analytics = await persona_system.get_persona_analytics(
            persona_id=sample_persona_definition.id,
            time_range_hours=24
        )
        
        assert "summary" in analytics
        assert "capabilities" in analytics
        assert analytics["summary"]["persona_id"] == sample_persona_definition.id
    
    @pytest.mark.asyncio
    async def test_initialize_default_personas(self, persona_system):
        """Test initialization of default personas."""
        await persona_system.initialize_default_personas()
        
        assert len(persona_system.personas) >= 6  # Should have at least 6 default personas
        
        # Check for specific default personas
        persona_ids = list(persona_system.personas.keys())
        assert "backend_engineer_default" in persona_ids
        assert "frontend_developer_default" in persona_ids
        assert "devops_specialist_default" in persona_ids
        assert "qa_engineer_default" in persona_ids
        assert "generalist_default" in persona_ids


class TestPersonaIntegration:
    """Test persona system integration with other components."""
    
    @pytest.mark.asyncio
    async def test_assign_optimal_persona_convenience_function(self):
        """Test convenience function for optimal persona assignment."""
        agent_id = uuid.uuid4()
        
        with patch('app.core.agent_persona_system.get_agent_persona_system') as mock_get_system:
            mock_system = AsyncMock()
            mock_assignment = Mock()
            mock_assignment.agent_id = agent_id
            mock_assignment.persona_id = "test_persona"
            mock_system.assign_persona_to_agent.return_value = mock_assignment
            mock_get_system.return_value = mock_system
            
            assignment = await assign_optimal_persona(agent_id)
            
            assert assignment.agent_id == agent_id
            mock_system.assign_persona_to_agent.assert_called_once_with(
                agent_id, None, None, None, None
            )
    
    @pytest.mark.asyncio
    async def test_get_agent_persona_convenience_function(self):
        """Test convenience function for getting agent persona."""
        agent_id = uuid.uuid4()
        
        with patch('app.core.agent_persona_system.get_agent_persona_system') as mock_get_system:
            mock_system = AsyncMock()
            mock_assignment = Mock()
            mock_assignment.persona_id = "test_persona"
            mock_persona = Mock()
            mock_persona.id = "test_persona"
            
            mock_system.get_agent_current_persona.return_value = mock_assignment
            mock_system.get_persona.return_value = mock_persona
            mock_get_system.return_value = mock_system
            
            persona = await get_agent_persona(agent_id)
            
            assert persona.id == "test_persona"
            mock_system.get_agent_current_persona.assert_called_once_with(agent_id)
            mock_system.get_persona.assert_called_once_with("test_persona")
    
    @pytest.mark.asyncio
    async def test_update_agent_persona_performance_convenience_function(self):
        """Test convenience function for updating performance."""
        agent_id = uuid.uuid4()
        task = Mock()
        
        with patch('app.core.agent_persona_system.get_agent_persona_system') as mock_get_system:
            mock_system = AsyncMock()
            mock_get_system.return_value = mock_system
            
            await update_agent_persona_performance(
                agent_id=agent_id,
                task=task,
                success=True,
                completion_time=120.0,
                complexity=0.6
            )
            
            mock_system.update_persona_performance.assert_called_once_with(
                agent_id, task, True, 120.0, 0.6
            )


class TestPersonaPerformanceAndAnalytics:
    """Test persona performance tracking and analytics."""
    
    @pytest.fixture
    def persona_with_history(self):
        """Persona with some usage history."""
        capabilities = {
            "programming": PersonaCapability("programming", PersonaCapabilityLevel.ADVANCED, 0.75, 0.8),
            "debugging": PersonaCapability("debugging", PersonaCapabilityLevel.INTERMEDIATE, 0.6, 0.65)
        }
        
        # Simulate some usage
        capabilities["programming"].usage_count = 10
        capabilities["programming"].success_rate = 0.8
        capabilities["debugging"].usage_count = 5
        capabilities["debugging"].success_rate = 0.6
        
        return PersonaDefinition(
            id="experienced_persona",
            name="Experienced Developer",
            description="Developer with history",
            persona_type=PersonaType.BACKEND_ENGINEER,
            adaptation_mode=PersonaAdaptationMode.ADAPTIVE,
            capabilities=capabilities,
            preferred_task_types=[TaskType.CODE_GENERATION],
            expertise_domains=["programming"],
            communication_style={},
            decision_making_style={},
            problem_solving_approach={},
            preferred_team_size=(2, 5),
            collaboration_patterns=[],
            mentoring_capability=True,
            typical_response_time=100.0,
            accuracy_vs_speed_preference=0.7,
            risk_tolerance=0.4,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
    
    @pytest.mark.asyncio
    async def test_capability_evolution_tracking(self, persona_with_history):
        """Test tracking of capability evolution over time."""
        programming_cap = persona_with_history.capabilities["programming"]
        
        initial_proficiency = programming_cap.proficiency_score
        initial_confidence = programming_cap.confidence
        
        # Simulate successful task
        programming_cap.update_proficiency(success=True, complexity=0.8)
        
        assert programming_cap.proficiency_score >= initial_proficiency
        assert programming_cap.usage_count == 11
        assert programming_cap.success_rate > 0.8  # Should improve slightly
    
    @pytest.mark.asyncio
    async def test_persona_performance_analytics(self):
        """Test generation of persona performance analytics."""
        persona_system = AgentPersonaSystem()
        
        # Add some test data
        agent_id = uuid.uuid4()
        test_assignment = PersonaAssignment(
            agent_id=agent_id,
            persona_id="test_persona",
            session_id="test_session",
            assigned_at=datetime.utcnow(),
            assignment_reason="test",
            confidence_score=0.8
        )
        
        # Simulate some task completions
        test_assignment.update_performance(True, 120.0)
        test_assignment.update_performance(True, 100.0)
        test_assignment.update_performance(False, 180.0)
        
        persona_system.active_assignments[agent_id] = test_assignment
        
        analytics = await persona_system.get_persona_analytics()
        
        assert "summary" in analytics
        assert analytics["summary"]["active_assignments"] == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_persona_assignments(self):
        """Test handling multiple concurrent persona assignments."""
        persona_system = AgentPersonaSystem()
        
        # Initialize with default personas
        await persona_system.initialize_default_personas()
        
        # Assign personas to multiple agents concurrently
        agent_ids = [uuid.uuid4() for _ in range(5)]
        
        assignments = await asyncio.gather(*[
            persona_system.assign_persona_to_agent(agent_id)
            for agent_id in agent_ids
        ])
        
        assert len(assignments) == 5
        assert all(assignment.agent_id in agent_ids for assignment in assignments)
        assert len(persona_system.active_assignments) == 5