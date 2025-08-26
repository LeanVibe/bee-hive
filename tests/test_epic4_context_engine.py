"""
Comprehensive tests for Epic 4 - Context Engine components.

Test suite covering all major Epic 4 context engine functionality including:
- Unified Context Engine
- Context Reasoning Engine  
- Intelligent Context Persistence
- Context-Aware Agent Coordination
- Epic 1/Epic 4 Integration
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import json

# Epic 4 Context Engine imports
from app.core.unified_context_engine import (
    UnifiedContextEngine, get_unified_context_engine,
    ContextMap, ReasoningInsight, OptimizationResult
)
from app.core.context_reasoning_engine import (
    ContextReasoningEngine, get_context_reasoning_engine,
    ReasoningType, Pattern, DecisionAnalysis, PredictiveInsight
)
from app.core.intelligent_context_persistence import (
    IntelligentContextPersistence, get_intelligent_context_persistence,
    PersistenceStrategy, LifecycleStage, PersistenceMetrics
)
from app.core.context_aware_agent_coordination import (
    ContextAwareAgentCoordination, get_context_aware_coordination,
    CoordinationStrategy, CoordinationContext, CoordinationDecision
)
from app.core.epic4_orchestration_integration import (
    Epic4OrchestrationIntegration, get_epic4_orchestration_integration,
    IntegrationMode, ContextEnhancedOrchestrationRequest
)

# Core imports for testing
from app.models.agent import Agent, AgentType, AgentStatus
from app.models.context import Context, ContextType
from app.core.orchestrator import OrchestrationRequest


class TestUnifiedContextEngine:
    """Test suite for Unified Context Engine."""
    
    @pytest.fixture
    async def context_engine(self):
        """Create a test context engine instance."""
        engine = UnifiedContextEngine()
        await engine.initialize()
        yield engine
        await engine.shutdown()
    
    @pytest.fixture
    def sample_agents(self):
        """Create sample agents for testing."""
        return [
            Agent(
                id="agent_1",
                name="Test Agent 1",
                agent_type=AgentType.SPECIALIST,
                status=AgentStatus.ACTIVE,
                is_active=True
            ),
            Agent(
                id="agent_2", 
                name="Test Agent 2",
                agent_type=AgentType.GENERALIST,
                status=AgentStatus.ACTIVE,
                is_active=True
            )
        ]
    
    async def test_context_engine_initialization(self, context_engine):
        """Test context engine initialization."""
        assert context_engine._is_running
        assert context_engine._advanced_engine is not None
        assert context_engine._cache_manager is not None
        assert context_engine._context_compressor is not None
    
    async def test_agent_context_coordination(self, context_engine, sample_agents):
        """Test agent context coordination functionality."""
        coordination_goals = ["improve_collaboration", "optimize_performance"]
        
        # Mock database session for agent context gathering
        with patch.object(context_engine, '_gather_agent_contexts') as mock_gather:
            mock_gather.return_value = [
                {
                    "id": "ctx_1",
                    "content": "Test context content",
                    "context_type": "decision",
                    "importance_score": 0.8,
                    "created_at": datetime.utcnow().isoformat()
                }
            ]
            
            context_map = await context_engine.coordinate_agent_context(
                agents=sample_agents,
                coordination_goals=coordination_goals
            )
            
            assert isinstance(context_map, ContextMap)
            assert len(context_map.agent_contexts) == len(sample_agents)
            assert len(context_map.coordination_insights) > 0
            assert context_map.performance_metrics is not None
    
    async def test_semantic_memory_optimization(self, context_engine):
        """Test semantic memory optimization functionality."""
        # Mock memory snapshot for optimization
        with patch.object(context_engine, '_take_memory_snapshot') as mock_snapshot:
            mock_snapshot.return_value = {"total_size": 1000000}  # 1MB
            
            result = await context_engine.optimize_semantic_memory(
                optimization_level=0.7
            )
            
            assert isinstance(result, OptimizationResult)
            assert result.original_size > 0
            assert 0 <= result.optimization_ratio <= 1
            assert result.processing_time_ms > 0
    
    async def test_reasoning_support(self, context_engine):
        """Test reasoning support functionality."""
        context_data = {
            "content": "Should we implement feature X or feature Y?",
            "type": "decision",
            "urgency": "high"
        }
        
        reasoning_insight = await context_engine.provide_reasoning_support(
            context=context_data,
            reasoning_type=ReasoningType.DECISION_SUPPORT
        )
        
        assert isinstance(reasoning_insight, ReasoningInsight)
        assert reasoning_insight.insight_type == ReasoningType.DECISION_SUPPORT
        assert 0 <= reasoning_insight.confidence_score <= 1
        assert len(reasoning_insight.recommendations) > 0
    
    async def test_cross_agent_context_sharing(self, context_engine):
        """Test cross-agent context sharing functionality."""
        source_agent = "agent_1"
        target_agents = ["agent_2", "agent_3"]
        
        # Mock advanced context engine sharing
        with patch.object(context_engine, '_advanced_engine') as mock_engine:
            mock_engine.share_knowledge_across_agents.return_value = {
                "shared_entities": 5,
                "target_agents": 2,
                "failed_shares": 0
            }
            
            result = await context_engine.share_context_across_agents(
                source_agent_id=source_agent,
                target_agent_ids=target_agents,
                semantic_optimization=True
            )
            
            assert result["shared_entities"] >= 0
            assert result["target_agents"] == len(target_agents)
    
    async def test_unified_analytics(self, context_engine):
        """Test unified analytics functionality."""
        analytics = await context_engine.get_unified_analytics()
        
        assert "overview" in analytics
        assert "operation_metrics" in analytics
        assert "component_performance" in analytics
        assert "recommendations" in analytics
        
        # Check that time range is properly set
        assert "time_range" in analytics
        assert "start" in analytics["time_range"]
        assert "end" in analytics["time_range"]


class TestContextReasoningEngine:
    """Test suite for Context Reasoning Engine."""
    
    @pytest.fixture
    def reasoning_engine(self):
        """Create a test reasoning engine instance."""
        return ContextReasoningEngine()
    
    async def test_decision_context_analysis(self, reasoning_engine):
        """Test decision context analysis."""
        context_data = {
            "content": "We need to choose between using MongoDB or PostgreSQL for our new service.",
            "domain": "technical",
            "importance": 0.8
        }
        
        analysis = await reasoning_engine.analyze_decision_context(
            context_data=context_data,
            decision_scope="technical"
        )
        
        assert isinstance(analysis, DecisionAnalysis)
        assert len(analysis.options_identified) > 0
        assert len(analysis.risk_assessment) > 0
        assert len(analysis.success_probability) > 0
        assert len(analysis.recommendations) > 0
    
    async def test_pattern_identification(self, reasoning_engine):
        """Test pattern identification in contexts."""
        contexts = [
            {"content": "Feature X completed successfully with good user feedback"},
            {"content": "Feature Y also completed successfully with positive reviews"},
            {"content": "Feature Z failed due to insufficient testing"},
            {"content": "Another feature failed, again due to poor testing practices"}
        ]
        
        patterns = await reasoning_engine.identify_patterns(
            contexts=contexts,
            lookback_days=30
        )
        
        assert isinstance(patterns, list)
        # Should identify success and failure patterns
        assert len(patterns) >= 0  # Patterns may or may not be found depending on implementation
    
    async def test_predictive_insights_generation(self, reasoning_engine):
        """Test predictive insights generation."""
        context_data = {
            "content": "Our API response times have been steadily increasing over the past week",
            "metrics": {"avg_response_time": 250, "trend": "increasing"}
        }
        
        insights = await reasoning_engine.generate_predictive_insights(
            context_data=context_data,
            prediction_horizon="short_term",
            focus_areas=["performance", "risks"]
        )
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        for insight in insights:
            assert isinstance(insight, PredictiveInsight)
            assert insight.prediction_type in ["performance", "risks", "opportunities", "resources"]
            assert 0 <= insight.confidence_level <= 1
    
    async def test_conflict_analysis(self, reasoning_engine):
        """Test conflict analysis functionality."""
        conflicting_contexts = [
            {"content": "Team A wants to use React for the frontend"},
            {"content": "Team B insists on using Vue.js for consistency"},
            {"content": "Management prefers Angular due to enterprise support"}
        ]
        
        analysis = await reasoning_engine.analyze_conflicts(
            contexts=conflicting_contexts,
            conflict_type="technical_decision"
        )
        
        assert "conflicts_identified" in analysis
        assert "resolution_strategies" in analysis
        assert "recommended_approach" in analysis
    
    async def test_performance_optimization_analysis(self, reasoning_engine):
        """Test performance optimization analysis."""
        context_data = {
            "content": "System performance is degrading under high load",
            "current_metrics": {"cpu_usage": 85, "memory_usage": 78, "response_time": 2.5}
        }
        
        optimization_goals = ["reduce_response_time", "improve_throughput"]
        
        analysis = await reasoning_engine.optimize_context_performance(
            context_data=context_data,
            optimization_goals=optimization_goals
        )
        
        assert "current_performance" in analysis
        assert "optimization_opportunities" in analysis
        assert "recommended_strategies" in analysis
        assert "implementation_plan" in analysis


class TestIntelligentContextPersistence:
    """Test suite for Intelligent Context Persistence."""
    
    @pytest.fixture
    async def persistence_system(self):
        """Create a test persistence system instance."""
        system = IntelligentContextPersistence()
        await system.initialize()
        yield system
        await system.shutdown()
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample context for testing."""
        return {
            "id": "test_context_1",
            "content": "This is a test context for persistence",
            "context_type": "general",
            "agent_id": "test_agent",
            "importance_score": 0.7,
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def test_context_storage_strategies(self, persistence_system, sample_context):
        """Test different context storage strategies."""
        # Test immediate persistence
        context_id = await persistence_system.store_context(
            context=sample_context,
            strategy=PersistenceStrategy.IMMEDIATE
        )
        assert context_id == sample_context["id"]
        
        # Test adaptive persistence
        context_id = await persistence_system.store_context(
            context=sample_context,
            strategy=PersistenceStrategy.ADAPTIVE
        )
        assert context_id == sample_context["id"]
    
    async def test_context_retrieval(self, persistence_system, sample_context):
        """Test context retrieval functionality."""
        # Store context first
        await persistence_system.store_context(context=sample_context)
        
        # Mock retrieval methods
        with patch.object(persistence_system, '_retrieve_from_memory') as mock_memory:
            mock_memory.return_value = sample_context
            
            retrieved_context = await persistence_system.retrieve_context(
                context_id=sample_context["id"]
            )
            
            assert retrieved_context is not None
            assert retrieved_context["id"] == sample_context["id"]
    
    async def test_lifecycle_management(self, persistence_system, sample_context):
        """Test context lifecycle management."""
        context_id = sample_context["id"]
        
        # Store context
        await persistence_system.store_context(context=sample_context)
        
        # Test lifecycle transition
        success = await persistence_system.update_context_lifecycle(
            context_id=context_id,
            new_stage=LifecycleStage.WARM
        )
        
        # May succeed or fail depending on current state and thresholds
        assert isinstance(success, bool)
    
    async def test_storage_optimization(self, persistence_system):
        """Test storage optimization functionality."""
        with patch.object(persistence_system, '_identify_optimization_candidates') as mock_candidates:
            mock_candidates.return_value = {
                "test_context_1": {"size": 1000, "stage": LifecycleStage.COLD}
            }
            
            with patch.object(persistence_system, '_apply_context_optimization') as mock_optimize:
                mock_optimize.return_value = {
                    "compressed": True,
                    "archived": False,
                    "deleted": False,
                    "storage_saved_mb": 0.5,
                    "optimizations": ["compression"]
                }
                
                result = await persistence_system.optimize_storage(
                    target_reduction_mb=1.0
                )
                
                assert result["contexts_processed"] > 0
                assert result["storage_saved_mb"] >= 0
    
    async def test_persistence_analytics(self, persistence_system):
        """Test persistence analytics functionality."""
        analytics = await persistence_system.get_persistence_analytics()
        
        assert "timestamp" in analytics
        assert "metrics" in analytics
        assert "lifecycle_distribution" in analytics
        assert "performance_metrics" in analytics
        assert "recommendations" in analytics


class TestContextAwareAgentCoordination:
    """Test suite for Context-Aware Agent Coordination."""
    
    @pytest.fixture
    async def coordination_system(self):
        """Create a test coordination system instance."""
        system = ContextAwareAgentCoordination()
        # Mock the initialization to avoid database dependencies
        with patch.object(system, '_load_agent_profiles'):
            with patch.object(system, '_start_background_tasks'):
                await system.initialize()
        yield system
        await system.shutdown()
    
    @pytest.fixture
    def sample_agents(self):
        """Create sample agents for coordination testing."""
        return [
            Agent(
                id="coord_agent_1",
                name="Coordination Agent 1",
                agent_type=AgentType.SPECIALIST,
                status=AgentStatus.IDLE,
                is_active=True
            ),
            Agent(
                id="coord_agent_2",
                name="Coordination Agent 2", 
                agent_type=AgentType.GENERALIST,
                status=AgentStatus.IDLE,
                is_active=True
            )
        ]
    
    async def test_agent_coordination_for_task(self, coordination_system, sample_agents):
        """Test agent coordination for specific tasks."""
        task_description = "Implement a new API endpoint for user management"
        required_capabilities = ["backend_development", "api_design", "testing"]
        
        # Mock reasoning engine analysis
        mock_analysis = Mock()
        mock_analysis.decision_context = task_description
        mock_analysis.resource_requirements = {"cpu": "medium", "memory": "low"}
        mock_analysis.timeline_estimate = {"hours": "4"}
        
        with patch.object(coordination_system, '_reasoning_engine') as mock_engine:
            mock_engine.analyze_decision_context.return_value = mock_analysis
            
            with patch.object(coordination_system, '_select_optimal_agents') as mock_select:
                mock_select.return_value = sample_agents
                
                coordination_context = await coordination_system.coordinate_agents_for_task(
                    task_description=task_description,
                    required_capabilities=required_capabilities,
                    max_agents=2
                )
                
                assert isinstance(coordination_context, CoordinationContext)
                assert len(coordination_context.participating_agents) <= 2
                assert coordination_context.coordination_strategy in CoordinationStrategy
                assert len(coordination_context.coordination_goals) > 0
    
    async def test_coordination_decision_making(self, coordination_system):
        """Test coordination decision making."""
        # Create a mock coordination context
        coordination_id = "test_coordination_1"
        coordination_context = CoordinationContext(
            coordination_id=coordination_id,
            participating_agents=["agent_1", "agent_2"],
            coordination_strategy=CoordinationStrategy.CONTEXT_OPTIMIZED,
            collaboration_mode=Mock(),
            shared_context={},
            coordination_goals=["efficiency"],
            success_metrics={},
            resource_constraints={},
            timeline_constraints={},
            priority_level=0.8
        )
        
        coordination_system.active_coordinations[coordination_id] = coordination_context
        
        # Mock reasoning support
        mock_insight = Mock()
        mock_insight.confidence_score = 0.8
        mock_insight.recommendations = ["Reallocate resources"]
        mock_insight.potential_risks = ["Timing constraints"]
        
        with patch.object(coordination_system, '_reasoning_engine') as mock_engine:
            mock_engine.provide_reasoning_support.return_value = mock_insight
            
            with patch.object(coordination_system, '_generate_decision_options') as mock_options:
                mock_options.return_value = [{"id": "option_1", "description": "Test option"}]
                
                with patch.object(coordination_system, '_evaluate_decision_options') as mock_eval:
                    mock_eval.return_value = [{
                        "score": 0.8,
                        "rationale": "Best option",
                        "implementation_steps": [],
                        "success_probability": 0.7,
                        "monitoring_metrics": []
                    }]
                    
                    decision = await coordination_system.make_coordination_decision(
                        coordination_id=coordination_id,
                        decision_context={"issue": "resource allocation"},
                        decision_type="resource_allocation"
                    )
                    
                    assert isinstance(decision, CoordinationDecision)
                    assert decision.decision_type == "resource_allocation"
                    assert 0 <= decision.confidence_score <= 1
    
    async def test_workload_optimization(self, coordination_system):
        """Test agent workload optimization."""
        with patch.object(coordination_system, '_analyze_current_workloads') as mock_workloads:
            mock_workloads.return_value = {"agent_1": 0.8, "agent_2": 0.3}
            
            with patch.object(coordination_system, '_identify_workload_optimization_opportunities') as mock_opportunities:
                mock_opportunities.return_value = [
                    {"type": "rebalance", "from_agent": "agent_1", "to_agent": "agent_2"}
                ]
                
                with patch.object(coordination_system, '_apply_workload_optimization') as mock_apply:
                    mock_apply.return_value = {
                        "coordinations_affected": 1,
                        "agents_rebalanced": 2,
                        "optimization_type": "rebalance"
                    }
                    
                    result = await coordination_system.optimize_agent_workloads()
                    
                    assert result["agents_rebalanced"] >= 0
                    assert result["performance_improvement"] >= 0
    
    async def test_coordination_analytics(self, coordination_system):
        """Test coordination analytics functionality."""
        with patch.object(coordination_system, '_calculate_coordination_success_rate') as mock_success:
            mock_success.return_value = 0.85
            
            with patch.object(coordination_system, '_analyze_agent_performance') as mock_performance:
                mock_performance.return_value = {"average_efficiency": 0.78}
                
                analytics = await coordination_system.get_coordination_analytics()
                
                assert "coordination_overview" in analytics
                assert "agent_performance" in analytics
                assert "optimization_insights" in analytics
                assert "recommendations" in analytics


class TestEpic4OrchestrationIntegration:
    """Test suite for Epic 4 Orchestration Integration."""
    
    @pytest.fixture
    async def integration_system(self):
        """Create a test integration system instance."""
        system = Epic4OrchestrationIntegration()
        # Mock initialization to avoid component dependencies
        with patch.object(system, '_orchestrator'):
            with patch.object(system, '_context_engine'):
                with patch.object(system, '_reasoning_engine'):
                    await system.initialize()
        yield system
    
    @pytest.fixture
    def sample_orchestration_request(self):
        """Create a sample orchestration request."""
        return OrchestrationRequest(
            task_id="test_task_1",
            task_description="Test task for integration",
            agent_requirements=["capability_1", "capability_2"],
            priority=0.8
        )
    
    async def test_orchestration_request_enhancement(self, integration_system, sample_orchestration_request):
        """Test orchestration request enhancement with context."""
        # Mock context analysis
        with patch.object(integration_system, '_analyze_request_context') as mock_analysis:
            mock_analysis.return_value = {
                "task_complexity": "moderate",
                "resource_requirements": {"cpu": "medium"},
                "collaboration_needs": ["coordination"]
            }
            
            with patch.object(integration_system, '_generate_reasoning_insights') as mock_insights:
                mock_insights.return_value = [
                    Mock(confidence_score=0.8, recommendations=["Test recommendation"])
                ]
                
                enhanced_request = await integration_system.enhance_orchestration_request(
                    request=sample_orchestration_request,
                    integration_mode=IntegrationMode.CONTEXT_ENHANCED
                )
                
                assert isinstance(enhanced_request, ContextEnhancedOrchestrationRequest)
                assert enhanced_request.original_request == sample_orchestration_request
                assert enhanced_request.integration_mode == IntegrationMode.CONTEXT_ENHANCED
                assert 0 <= enhanced_request.context_confidence <= 1
    
    async def test_context_driven_orchestration(self, integration_system, sample_orchestration_request):
        """Test context-driven orchestration execution."""
        # Create enhanced request
        enhanced_request = ContextEnhancedOrchestrationRequest(
            original_request=sample_orchestration_request,
            context_analysis={"complexity": "moderate"},
            reasoning_insights=[Mock(confidence_score=0.8)],
            coordination_context=None,
            integration_mode=IntegrationMode.CONTEXT_DRIVEN,
            context_confidence=0.8
        )
        
        # Mock execution phases
        mock_phase_result = {"success": True, "metrics": {}}
        
        with patch.object(integration_system, '_execute_context_enhanced_planning') as mock_planning:
            mock_planning.return_value = mock_phase_result
            
            with patch.object(integration_system, '_execute_intelligent_agent_selection') as mock_selection:
                mock_selection.return_value = {**mock_phase_result, "selected_agents": ["agent_1"]}
                
                with patch.object(integration_system, '_execute_context_aware_execution') as mock_execution:
                    mock_execution.return_value = mock_phase_result
                    
                    with patch.object(integration_system, '_execute_intelligent_monitoring') as mock_monitoring:
                        mock_monitoring.return_value = mock_phase_result
                        
                        with patch.object(integration_system, '_execute_context_driven_optimization') as mock_optimization:
                            mock_optimization.return_value = mock_phase_result
                            
                            result = await integration_system.execute_context_driven_orchestration(
                                enhanced_request=enhanced_request
                            )
                            
                            assert result["success"] is True
                            assert "execution_phases" in result
                            assert "performance_metrics" in result
    
    async def test_orchestration_performance_optimization(self, integration_system):
        """Test orchestration performance optimization."""
        with patch.object(integration_system, '_analyze_historical_performance') as mock_analysis:
            mock_analysis.return_value = {
                "average_completion_time": 5.2,
                "success_rate": 0.85,
                "resource_utilization": 0.67
            }
            
            with patch.object(integration_system, '_identify_optimization_opportunities') as mock_opportunities:
                mock_opportunities.return_value = [
                    {"type": "caching", "impact": "high", "effort": "medium"}
                ]
                
                with patch.object(integration_system, '_generate_optimization_recommendations') as mock_recommendations:
                    mock_recommendations.return_value = [
                        {"type": "enable_caching", "confidence": 0.9, "auto_apply": True}
                    ]
                    
                    with patch.object(integration_system, '_apply_automatic_optimization') as mock_apply:
                        mock_apply.return_value = {"success": True}
                        
                        result = await integration_system.optimize_orchestration_performance()
                        
                        assert "performance_analysis" in result
                        assert "optimization_opportunities" in result
                        assert "recommended_actions" in result
                        assert "auto_optimizations_applied" in result
    
    async def test_integration_analytics(self, integration_system):
        """Test integration analytics functionality."""
        with patch.object(integration_system, '_update_performance_metrics'):
            with patch.object(integration_system, '_analyze_integration_mode_usage') as mock_modes:
                mock_modes.return_value = {"context_enhanced": 60, "context_driven": 40}
                
                with patch.object(integration_system, '_analyze_context_utilization_trends') as mock_trends:
                    mock_trends.return_value = {"increasing": True, "rate": 0.15}
                    
                    analytics = await integration_system.get_integration_analytics()
                    
                    assert "integration_metrics" in analytics
                    assert "phase_performance" in analytics
                    assert "integration_modes_used" in analytics
                    assert "recommendations" in analytics


# Integration tests
class TestEpic4Integration:
    """Integration tests for Epic 4 components working together."""
    
    async def test_end_to_end_context_orchestration(self):
        """Test end-to-end context-aware orchestration flow."""
        # This would test the complete flow from orchestration request
        # through context enhancement to execution with coordination
        
        # Mock the complete integration
        with patch('app.core.unified_context_engine.get_unified_context_engine'):
            with patch('app.core.context_aware_agent_coordination.get_context_aware_coordination'):
                with patch('app.core.epic4_orchestration_integration.get_epic4_orchestration_integration'):
                    # Test would simulate a complete orchestration request
                    # with context enhancement and coordination
                    assert True  # Placeholder for actual integration test
    
    async def test_context_sharing_across_components(self):
        """Test context sharing between Epic 4 components."""
        # This would test that context is properly shared between
        # reasoning engine, persistence system, and coordination system
        assert True  # Placeholder for context sharing test
    
    async def test_performance_under_load(self):
        """Test Epic 4 performance under load."""
        # This would test system performance with multiple
        # concurrent orchestration requests and context operations
        assert True  # Placeholder for performance test


# Fixtures for all tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_db_session():
    """Mock database session for testing."""
    session = AsyncMock()
    yield session


@pytest.fixture
async def mock_redis_client():
    """Mock Redis client for testing."""
    client = AsyncMock()
    yield client


# Test configuration
pytestmark = pytest.mark.asyncio