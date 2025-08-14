"""
Pydantic Model Validation Testing for LeanVibe Agent Hive 2.0

Tests Agent model validation, relationships, serialization, and business logic
to increase coverage from 50% to 80%.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class TestAgentEnums:
    """Test Agent-related enums and their values."""
    
    def test_agent_status_enum_values(self):
        """Test AgentStatus enum has correct values."""
        from app.models.agent import AgentStatus
        
        # Test primary lowercase values
        assert AgentStatus.inactive.value == "inactive"
        assert AgentStatus.active.value == "active"
        assert AgentStatus.busy.value == "busy"
        assert AgentStatus.error.value == "error"
        assert AgentStatus.maintenance.value == "maintenance"
        assert AgentStatus.shutting_down.value == "shutting_down"
        
        # Test uppercase aliases for backward compatibility
        assert AgentStatus.INACTIVE.value == "inactive"
        assert AgentStatus.ACTIVE.value == "active"
        assert AgentStatus.BUSY.value == "busy"
        assert AgentStatus.ERROR.value == "error"
        assert AgentStatus.MAINTENANCE.value == "maintenance"
        assert AgentStatus.SHUTTING_DOWN.value == "shutting_down"
    
    def test_agent_type_enum_values(self):
        """Test AgentType enum has correct values."""
        from app.models.agent import AgentType
        
        assert AgentType.CLAUDE.value == "claude"
        assert AgentType.GPT.value == "gpt"
        assert AgentType.GEMINI.value == "gemini"
        assert AgentType.CUSTOM.value == "custom"
    
    def test_enum_equality(self):
        """Test enum equality and comparison."""
        from app.models.agent import AgentStatus
        
        # Test primary and alias equality
        assert AgentStatus.active == AgentStatus.ACTIVE
        assert AgentStatus.inactive == AgentStatus.INACTIVE
        assert AgentStatus.busy == AgentStatus.BUSY


class TestAgentModelCreation:
    """Test Agent model creation and basic properties."""
    
    def test_agent_model_creation_minimal(self):
        """Test creating an Agent with minimal required fields."""
        from app.models.agent import Agent, AgentStatus, AgentType
        
        agent = Agent(name="test-agent")
        
        # Check default values
        assert agent.name == "test-agent"
        assert agent.type == AgentType.CLAUDE  # Default type
        assert agent.status == AgentStatus.inactive  # Default status
        assert agent.capabilities == []  # Default empty list
        assert agent.config == {}  # Default empty dict
        assert agent.total_tasks_completed == "0"
        assert agent.total_tasks_failed == "0"
        assert agent.average_response_time == "0.0"
        assert agent.context_window_usage == "0.0"
        assert agent.current_sleep_state == 'AWAKE'
    
    def test_agent_model_creation_full(self):
        """Test creating an Agent with all fields."""
        from app.models.agent import Agent, AgentStatus, AgentType
        
        agent_id = uuid.uuid4()
        now = datetime.utcnow()
        
        agent = Agent(
            id=agent_id,
            name="full-test-agent",
            type=AgentType.GPT,
            role="developer",
            capabilities=[{"name": "coding", "confidence": 0.9}],
            system_prompt="You are a developer agent",
            status=AgentStatus.active,
            config={"max_tokens": 1000},
            tmux_session="session-123",
            total_tasks_completed="10",
            total_tasks_failed="2",
            average_response_time="1.5",
            context_window_usage="0.7",
            created_at=now,
            updated_at=now,
            last_heartbeat=now,
            last_active=now,
            current_sleep_state="SLEEPING",
            current_cycle_id=uuid.uuid4(),
            last_sleep_time=now,
            last_wake_time=now - timedelta(hours=1)
        )
        
        assert agent.id == agent_id
        assert agent.name == "full-test-agent"
        assert agent.type == AgentType.GPT
        assert agent.role == "developer"
        assert len(agent.capabilities) == 1
        assert agent.system_prompt == "You are a developer agent"
        assert agent.status == AgentStatus.active
        assert agent.config["max_tokens"] == 1000
        assert agent.tmux_session == "session-123"
        assert agent.total_tasks_completed == "10"
        assert agent.total_tasks_failed == "2"
        assert agent.average_response_time == "1.5"
        assert agent.context_window_usage == "0.7"
        assert agent.current_sleep_state == "SLEEPING"
    
    def test_agent_repr(self):
        """Test Agent string representation."""
        from app.models.agent import Agent, AgentStatus
        
        agent = Agent(
            id=uuid.uuid4(),
            name="test-agent",
            role="developer",
            status=AgentStatus.active
        )
        
        repr_str = repr(agent)
        assert "Agent(" in repr_str
        assert "test-agent" in repr_str
        assert "developer" in repr_str
        assert "active" in repr_str


class TestAgentSerialization:
    """Test Agent model serialization and dict conversion."""
    
    def test_agent_to_dict_minimal(self):
        """Test to_dict with minimal agent."""
        from app.models.agent import Agent
        
        agent = Agent(name="test-agent")
        agent.id = uuid.uuid4()  # Set ID for consistent testing
        
        data = agent.to_dict()
        
        assert data["name"] == "test-agent"
        assert data["type"] == "claude"
        assert data["status"] == "inactive"
        assert data["capabilities"] == []
        assert data["config"] == {}
        assert data["total_tasks_completed"] == 0
        assert data["total_tasks_failed"] == 0
        assert data["average_response_time"] == 0.0
        assert data["context_window_usage"] == 0.0
        assert data["current_sleep_state"] == 'AWAKE'
        assert data["id"] == str(agent.id)
    
    def test_agent_to_dict_full(self):
        """Test to_dict with fully populated agent."""
        from app.models.agent import Agent, AgentStatus, AgentType
        
        agent_id = uuid.uuid4()
        cycle_id = uuid.uuid4()
        now = datetime.utcnow()
        
        agent = Agent(
            id=agent_id,
            name="full-agent",
            type=AgentType.GPT,
            role="developer",
            capabilities=[{"name": "coding", "confidence": 0.9}],
            status=AgentStatus.active,
            config={"max_tokens": 1000},
            tmux_session="session-123",
            total_tasks_completed="10",
            total_tasks_failed="2",
            average_response_time="1.5",
            context_window_usage="0.7",
            created_at=now,
            updated_at=now,
            last_heartbeat=now,
            last_active=now,
            current_sleep_state="SLEEPING",
            current_cycle_id=cycle_id,
            last_sleep_time=now,
            last_wake_time=now - timedelta(hours=1)
        )
        
        data = agent.to_dict()
        
        assert data["id"] == str(agent_id)
        assert data["name"] == "full-agent"
        assert data["type"] == "gpt"
        assert data["role"] == "developer"
        assert data["capabilities"] == [{"name": "coding", "confidence": 0.9}]
        assert data["status"] == "active"
        assert data["config"] == {"max_tokens": 1000}
        assert data["tmux_session"] == "session-123"
        assert data["total_tasks_completed"] == 10
        assert data["total_tasks_failed"] == 2
        assert data["average_response_time"] == 1.5
        assert data["context_window_usage"] == 0.7
        assert data["current_sleep_state"] == "SLEEPING"
        assert data["current_cycle_id"] == str(cycle_id)
        assert data["created_at"] == now.isoformat()
        assert data["updated_at"] == now.isoformat()
        assert data["last_heartbeat"] == now.isoformat()
        assert data["last_active"] == now.isoformat()
        assert data["last_sleep_time"] == now.isoformat()
        assert data["last_wake_time"] == (now - timedelta(hours=1)).isoformat()
    
    def test_agent_to_dict_with_none_values(self):
        """Test to_dict handling of None values."""
        from app.models.agent import Agent
        
        agent = Agent(name="test-agent")
        agent.id = uuid.uuid4()
        agent.role = None
        agent.capabilities = None
        agent.config = None
        agent.created_at = None
        agent.updated_at = None
        agent.last_heartbeat = None
        agent.last_active = None
        agent.current_cycle_id = None
        agent.last_sleep_time = None
        agent.last_wake_time = None
        
        data = agent.to_dict()
        
        assert data["role"] is None
        assert data["capabilities"] is None
        assert data["config"] is None
        assert data["created_at"] is None
        assert data["updated_at"] is None
        assert data["last_heartbeat"] is None
        assert data["last_active"] is None
        assert data["current_cycle_id"] is None
        assert data["last_sleep_time"] is None
        assert data["last_wake_time"] is None
    
    def test_agent_to_dict_string_conversion_edge_cases(self):
        """Test to_dict string conversion edge cases."""
        from app.models.agent import Agent
        
        agent = Agent(name="test-agent")
        agent.id = uuid.uuid4()
        
        # Test with None string values
        agent.total_tasks_completed = None
        agent.total_tasks_failed = None
        agent.average_response_time = None
        agent.context_window_usage = None
        
        data = agent.to_dict()
        
        # Should convert None to 0 for numeric fields
        assert data["total_tasks_completed"] == 0
        assert data["total_tasks_failed"] == 0
        assert data["average_response_time"] == 0.0
        assert data["context_window_usage"] == 0.0


class TestAgentHeartbeatManagement:
    """Test Agent heartbeat and activity tracking."""
    
    def test_update_heartbeat_basic(self):
        """Test basic heartbeat update."""
        from app.models.agent import Agent, AgentStatus
        
        agent = Agent(name="test-agent", status=AgentStatus.inactive)
        original_heartbeat = agent.last_heartbeat
        original_active = agent.last_active
        
        # Mock datetime to control timestamp
        with patch('app.models.agent.datetime') as mock_datetime:
            mock_now = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            agent.update_heartbeat()
            
            assert agent.last_heartbeat == mock_now
            # Should not update last_active for inactive agent
            assert agent.last_active == original_active
    
    def test_update_heartbeat_active_agent(self):
        """Test heartbeat update for active agent."""
        from app.models.agent import Agent, AgentStatus
        
        agent = Agent(name="test-agent", status=AgentStatus.active)
        
        with patch('app.models.agent.datetime') as mock_datetime:
            mock_now = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            agent.update_heartbeat()
            
            assert agent.last_heartbeat == mock_now
            assert agent.last_active == mock_now  # Should update for active agent


class TestAgentCapabilityManagement:
    """Test Agent capability management methods."""
    
    def test_add_capability_to_empty_list(self):
        """Test adding capability to agent with no existing capabilities."""
        from app.models.agent import Agent
        
        agent = Agent(name="test-agent")
        agent.capabilities = None  # Start with None
        
        agent.add_capability("coding", "Python development", 0.9, ["web", "backend"])
        
        assert agent.capabilities is not None
        assert len(agent.capabilities) == 1
        
        capability = agent.capabilities[0]
        assert capability["name"] == "coding"
        assert capability["description"] == "Python development"
        assert capability["confidence_level"] == 0.9
        assert capability["specialization_areas"] == ["web", "backend"]
    
    def test_add_capability_to_existing_list(self):
        """Test adding capability to agent with existing capabilities."""
        from app.models.agent import Agent
        
        agent = Agent(name="test-agent")
        agent.capabilities = [{"name": "testing", "confidence_level": 0.8}]
        
        agent.add_capability("coding", "Python development", 0.9, ["web"])
        
        assert len(agent.capabilities) == 2
        assert agent.capabilities[1]["name"] == "coding"
        assert agent.capabilities[1]["confidence_level"] == 0.9
    
    def test_has_capability_existing(self):
        """Test checking for existing capability."""
        from app.models.agent import Agent
        
        agent = Agent(name="test-agent")
        agent.capabilities = [
            {"name": "coding", "confidence_level": 0.9},
            {"name": "testing", "confidence_level": 0.8}
        ]
        
        assert agent.has_capability("coding") is True
        assert agent.has_capability("testing") is True
        assert agent.has_capability("deployment") is False
    
    def test_has_capability_no_capabilities(self):
        """Test checking capability when agent has no capabilities."""
        from app.models.agent import Agent
        
        agent = Agent(name="test-agent")
        agent.capabilities = None
        
        assert agent.has_capability("coding") is False
        
        agent.capabilities = []
        assert agent.has_capability("coding") is False
    
    def test_get_capability_confidence_existing(self):
        """Test getting confidence for existing capability."""
        from app.models.agent import Agent
        
        agent = Agent(name="test-agent")
        agent.capabilities = [
            {"name": "coding", "confidence_level": 0.9},
            {"name": "testing", "confidence_level": 0.8}
        ]
        
        assert agent.get_capability_confidence("coding") == 0.9
        assert agent.get_capability_confidence("testing") == 0.8
        assert agent.get_capability_confidence("deployment") == 0.0
    
    def test_get_capability_confidence_no_capabilities(self):
        """Test getting confidence when agent has no capabilities."""
        from app.models.agent import Agent
        
        agent = Agent(name="test-agent")
        agent.capabilities = None
        
        assert agent.get_capability_confidence("coding") == 0.0
        
        agent.capabilities = []
        assert agent.get_capability_confidence("coding") == 0.0
    
    def test_get_capability_confidence_missing_key(self):
        """Test getting confidence when capability exists but lacks confidence_level."""
        from app.models.agent import Agent
        
        agent = Agent(name="test-agent")
        agent.capabilities = [
            {"name": "coding"},  # Missing confidence_level
            {"name": "testing", "confidence_level": 0.8}
        ]
        
        assert agent.get_capability_confidence("coding") == 0.0  # Default for missing key
        assert agent.get_capability_confidence("testing") == 0.8


class TestAgentTaskSuitability:
    """Test Agent task availability and suitability assessment."""
    
    def test_is_available_for_task_active_low_usage(self):
        """Test agent availability when active with low context usage."""
        from app.models.agent import Agent, AgentStatus
        
        agent = Agent(name="test-agent", status=AgentStatus.active)
        agent.context_window_usage = "0.5"  # 50% usage
        
        assert agent.is_available_for_task() is True
    
    def test_is_available_for_task_active_high_usage(self):
        """Test agent availability when active with high context usage."""
        from app.models.agent import Agent, AgentStatus
        
        agent = Agent(name="test-agent", status=AgentStatus.active)
        agent.context_window_usage = "0.9"  # 90% usage (over 80% threshold)
        
        assert agent.is_available_for_task() is False
    
    def test_is_available_for_task_inactive(self):
        """Test agent availability when inactive."""
        from app.models.agent import Agent, AgentStatus
        
        agent = Agent(name="test-agent", status=AgentStatus.inactive)
        agent.context_window_usage = "0.5"
        
        assert agent.is_available_for_task() is False
    
    def test_is_available_for_task_busy(self):
        """Test agent availability when busy."""
        from app.models.agent import Agent, AgentStatus
        
        agent = Agent(name="test-agent", status=AgentStatus.busy)
        agent.context_window_usage = "0.5"
        
        assert agent.is_available_for_task() is False
    
    def test_is_available_for_task_no_context_usage(self):
        """Test agent availability with no context usage data."""
        from app.models.agent import Agent, AgentStatus
        
        agent = Agent(name="test-agent", status=AgentStatus.active)
        agent.context_window_usage = None
        
        # Should handle None gracefully (likely return False due to float() conversion)
        result = agent.is_available_for_task()
        assert isinstance(result, bool)
    
    def test_calculate_task_suitability_unavailable_agent(self):
        """Test task suitability for unavailable agent."""
        from app.models.agent import Agent, AgentStatus
        
        agent = Agent(name="test-agent", status=AgentStatus.inactive)
        
        suitability = agent.calculate_task_suitability("coding", ["python", "web"])
        
        assert suitability == 0.0
    
    def test_calculate_task_suitability_no_capabilities(self):
        """Test task suitability when agent has no capabilities."""
        from app.models.agent import Agent, AgentStatus
        
        agent = Agent(name="test-agent", status=AgentStatus.active)
        agent.context_window_usage = "0.5"
        agent.capabilities = None
        
        suitability = agent.calculate_task_suitability("coding", ["python", "web"])
        
        assert suitability == 0.5  # Neutral score
    
    def test_calculate_task_suitability_no_required_capabilities(self):
        """Test task suitability when no required capabilities specified."""
        from app.models.agent import Agent, AgentStatus
        
        agent = Agent(name="test-agent", status=AgentStatus.active)
        agent.context_window_usage = "0.5"
        agent.capabilities = [{"name": "coding", "confidence_level": 0.9}]
        
        suitability = agent.calculate_task_suitability("coding", [])
        
        assert suitability == 0.5  # Neutral score
    
    def test_calculate_task_suitability_exact_match(self):
        """Test task suitability with exact capability match."""
        from app.models.agent import Agent, AgentStatus
        
        agent = Agent(name="test-agent", status=AgentStatus.active)
        agent.context_window_usage = "0.5"
        agent.capabilities = [
            {"name": "python coding", "confidence_level": 0.9},
            {"name": "web development", "confidence_level": 0.8}
        ]
        
        suitability = agent.calculate_task_suitability("coding", ["python", "web"])
        
        # Should get high score for exact matches
        assert suitability > 0.5
        assert suitability <= 1.0
    
    def test_calculate_task_suitability_specialization_match(self):
        """Test task suitability with specialization area match."""
        from app.models.agent import Agent, AgentStatus
        
        agent = Agent(name="test-agent", status=AgentStatus.active)
        agent.context_window_usage = "0.5"
        agent.capabilities = [
            {
                "name": "backend development",
                "confidence_level": 0.9,
                "specialization_areas": ["python", "django", "fastapi"]
            }
        ]
        
        suitability = agent.calculate_task_suitability("coding", ["python"])
        
        # Should get good score for specialization match (80% of confidence)
        assert suitability > 0.5
        assert suitability <= 1.0
    
    def test_calculate_task_suitability_partial_match(self):
        """Test task suitability with partial capability match."""
        from app.models.agent import Agent, AgentStatus
        
        agent = Agent(name="test-agent", status=AgentStatus.active)
        agent.context_window_usage = "0.5"
        agent.capabilities = [
            {"name": "python coding", "confidence_level": 0.9}
        ]
        
        # Request capabilities where only one matches
        suitability = agent.calculate_task_suitability("coding", ["python", "javascript"])
        
        # Should get partial score (0.9 for python match, 0 for javascript)
        expected = 0.9 / 2  # 1 match out of 2 required
        assert abs(suitability - expected) < 0.01
    
    def test_calculate_task_suitability_no_matches(self):
        """Test task suitability with no capability matches."""
        from app.models.agent import Agent, AgentStatus
        
        agent = Agent(name="test-agent", status=AgentStatus.active)
        agent.context_window_usage = "0.5"
        agent.capabilities = [
            {"name": "testing", "confidence_level": 0.8}
        ]
        
        suitability = agent.calculate_task_suitability("coding", ["python", "javascript"])
        
        assert suitability == 0.0  # No matches


class TestAgentDataValidation:
    """Test Agent model data validation and constraints."""
    
    def test_agent_enum_validation(self):
        """Test that enum values are properly validated."""
        from app.models.agent import Agent, AgentStatus, AgentType
        
        # Valid enum values should work
        agent = Agent(name="test-agent", status=AgentStatus.active, type=AgentType.GPT)
        assert agent.status == AgentStatus.active
        assert agent.type == AgentType.GPT
        
        # Test enum equality with string values
        assert agent.status.value == "active"
        assert agent.type.value == "gpt"
    
    def test_agent_json_fields_validation(self):
        """Test JSON field handling for capabilities and config."""
        from app.models.agent import Agent
        
        # Test with valid JSON-serializable data
        capabilities = [
            {"name": "coding", "confidence": 0.9},
            {"name": "testing", "confidence": 0.8}
        ]
        config = {"max_tokens": 1000, "temperature": 0.7}
        
        agent = Agent(name="test-agent", capabilities=capabilities, config=config)
        
        assert agent.capabilities == capabilities
        assert agent.config == config
        assert len(agent.capabilities) == 2
        assert agent.config["max_tokens"] == 1000
    
    def test_agent_string_field_lengths(self):
        """Test string field constraints and lengths."""
        from app.models.agent import Agent
        
        # Test normal length strings
        agent = Agent(name="test-agent", role="developer")
        assert agent.name == "test-agent"
        assert agent.role == "developer"
        
        # Test longer strings (within reasonable limits)
        long_name = "a" * 255  # Max name length
        long_role = "b" * 100   # Max role length
        
        agent = Agent(name=long_name, role=long_role)
        assert len(agent.name) == 255
        assert len(agent.role) == 100
    
    def test_agent_numeric_string_fields(self):
        """Test numeric fields stored as strings."""
        from app.models.agent import Agent
        
        agent = Agent(
            name="test-agent",
            total_tasks_completed="100",
            total_tasks_failed="5",
            average_response_time="2.5",
            context_window_usage="0.75"
        )
        
        # Values should be stored as strings
        assert agent.total_tasks_completed == "100"
        assert agent.total_tasks_failed == "5"
        assert agent.average_response_time == "2.5"
        assert agent.context_window_usage == "0.75"
        
        # to_dict should convert to appropriate numeric types
        data = agent.to_dict()
        assert data["total_tasks_completed"] == 100
        assert data["total_tasks_failed"] == 5
        assert data["average_response_time"] == 2.5
        assert data["context_window_usage"] == 0.75


class TestAgentRelationships:
    """Test Agent model relationships with other models."""
    
    def test_agent_relationship_attributes(self):
        """Test that relationship attributes exist on Agent model."""
        from app.models.agent import Agent
        
        agent = Agent(name="test-agent")
        
        # Check that relationship attributes exist
        assert hasattr(agent, 'sleep_windows')
        assert hasattr(agent, 'checkpoints')
        assert hasattr(agent, 'sleep_wake_cycles')
        assert hasattr(agent, 'sleep_wake_analytics')
    
    def test_agent_sleep_wake_state_fields(self):
        """Test sleep-wake system integration fields."""
        from app.models.agent import Agent
        
        agent_id = uuid.uuid4()
        cycle_id = uuid.uuid4()
        now = datetime.utcnow()
        
        agent = Agent(
            name="test-agent",
            current_sleep_state="SLEEPING",
            current_cycle_id=cycle_id,
            last_sleep_time=now,
            last_wake_time=now - timedelta(hours=1)
        )
        
        assert agent.current_sleep_state == "SLEEPING"
        assert agent.current_cycle_id == cycle_id
        assert agent.last_sleep_time == now
        assert agent.last_wake_time == now - timedelta(hours=1)


class TestAgentEdgeCases:
    """Test Agent model edge cases and error handling."""
    
    def test_agent_with_empty_strings(self):
        """Test agent creation with empty string values."""
        from app.models.agent import Agent
        
        agent = Agent(
            name="test-agent",
            role="",
            tmux_session="",
            total_tasks_completed="",
            total_tasks_failed="",
            average_response_time="",
            context_window_usage=""
        )
        
        assert agent.role == ""
        assert agent.tmux_session == ""
        
        # to_dict should handle empty strings gracefully
        data = agent.to_dict()
        # Empty strings should convert to 0 for numeric fields
        assert data["total_tasks_completed"] == 0
        assert data["total_tasks_failed"] == 0
        assert data["average_response_time"] == 0.0
        assert data["context_window_usage"] == 0.0
    
    def test_agent_capability_edge_cases(self):
        """Test capability methods with edge case data."""
        from app.models.agent import Agent
        
        agent = Agent(name="test-agent")
        
        # Test with capabilities that have missing or malformed data
        agent.capabilities = [
            {"name": "coding"},  # Missing confidence_level
            {"confidence_level": 0.8},  # Missing name
            {"name": "", "confidence_level": 0.9},  # Empty name
            {"name": "testing", "confidence_level": "invalid"},  # Invalid confidence
            {"name": "deployment", "confidence_level": 0.7, "specialization_areas": None}  # None specialization
        ]
        
        # Methods should handle malformed data gracefully
        assert agent.has_capability("coding") is True
        assert agent.has_capability("") is True  # Empty string name exists
        assert agent.get_capability_confidence("coding") == 0.0  # Missing confidence
        assert agent.get_capability_confidence("testing") == "invalid"  # Returns as-is
        
        # Task suitability should handle malformed data
        suitability = agent.calculate_task_suitability("test", ["deployment"])
        assert isinstance(suitability, float)
        assert 0.0 <= suitability <= 1.0
    
    def test_agent_context_usage_edge_cases(self):
        """Test context usage handling with edge case values."""
        from app.models.agent import Agent, AgentStatus
        
        agent = Agent(name="test-agent", status=AgentStatus.active)
        
        # Test with various edge case values
        test_values = ["", "0", "1", "1.0", "invalid", "999", "-1"]
        
        for value in test_values:
            agent.context_window_usage = value
            
            # is_available_for_task should handle all values gracefully
            try:
                result = agent.is_available_for_task()
                assert isinstance(result, bool)
            except (ValueError, TypeError):
                # It's acceptable for invalid values to raise exceptions
                pass