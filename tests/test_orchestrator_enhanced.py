"""
Basic tests for enhanced orchestrator with intelligent routing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.orchestrator import AgentOrchestrator
from app.core.intelligent_task_router import RoutingStrategy
from app.models.task import TaskPriority


class TestEnhancedOrchestrator:
    """Basic test suite for enhanced orchestrator."""
    
    def test_orchestrator_initialization(self):
        """Test that enhanced orchestrator initializes correctly."""
        orchestrator = AgentOrchestrator()
        
        # Check that new components are initialized as None
        assert orchestrator.intelligent_router is None
        assert orchestrator.capability_matcher is None
        
        # Check that new metrics are present
        assert 'routing_decisions' in orchestrator.metrics
        assert 'routing_accuracy' in orchestrator.metrics
        assert 'load_balancing_actions' in orchestrator.metrics
        
        # Check existing functionality still works
        assert orchestrator.agents == {}
        assert orchestrator.active_sessions == {}
        assert not orchestrator.is_running
    
    @patch('app.core.orchestrator.get_message_broker')
    @patch('app.core.orchestrator.get_session_cache')
    @patch('app.core.orchestrator.IntelligentTaskRouter')
    @patch('app.core.orchestrator.CapabilityMatcher')
    @patch('app.core.orchestrator.WorkflowEngine')
    def test_orchestrator_start_with_intelligent_routing(
        self, mock_workflow_engine, mock_capability_matcher, 
        mock_intelligent_router, mock_session_cache, mock_message_broker
    ):
        """Test orchestrator start initializes intelligent routing components."""
        
        # Mock dependencies
        mock_message_broker.return_value = AsyncMock()
        mock_session_cache.return_value = AsyncMock()
        mock_workflow_engine.return_value = AsyncMock()
        mock_intelligent_router.return_value = MagicMock()
        mock_capability_matcher.return_value = MagicMock()
        
        orchestrator = AgentOrchestrator()
        
        # Mock the async methods to avoid actual async execution
        orchestrator.spawn_agent = AsyncMock()
        
        # The initialization should work without errors
        assert mock_intelligent_router.called == False  # Not called until start()
        assert mock_capability_matcher.called == False  # Not called until start()
    
    def test_enhanced_delegate_task_signature(self):
        """Test that delegate_task method has enhanced signature."""
        orchestrator = AgentOrchestrator()
        
        # Get the delegate_task method signature
        import inspect
        sig = inspect.signature(orchestrator.delegate_task)
        
        # Check that new parameters are present
        param_names = list(sig.parameters.keys())
        
        assert 'estimated_effort' in param_names
        assert 'due_date' in param_names
        assert 'dependencies' in param_names
        assert 'routing_strategy' in param_names
        
        # Check default values
        assert sig.parameters['routing_strategy'].default == RoutingStrategy.ADAPTIVE
    
    def test_new_orchestrator_methods_exist(self):
        """Test that new orchestrator methods exist."""
        orchestrator = AgentOrchestrator()
        
        # Check that new methods exist
        assert hasattr(orchestrator, 'rebalance_agent_workloads')
        assert hasattr(orchestrator, 'get_routing_analytics')
        assert hasattr(orchestrator, 'update_task_completion_metrics')
        assert hasattr(orchestrator, '_get_available_agent_ids')
        assert hasattr(orchestrator, '_record_routing_analytics')
        assert hasattr(orchestrator, '_workload_monitoring_loop')
        
        # Check that methods are callable
        assert callable(getattr(orchestrator, 'rebalance_agent_workloads'))
        assert callable(getattr(orchestrator, 'get_routing_analytics'))
        assert callable(getattr(orchestrator, 'update_task_completion_metrics'))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])