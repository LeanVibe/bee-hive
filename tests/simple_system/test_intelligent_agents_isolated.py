"""
Intelligent Agents - Component Isolation Tests
==============================================

Tests the intelligent agent system components in complete isolation.
This validates agent lifecycle, capability matching, task execution,
and inter-agent communication without external dependencies.

Testing Strategy:
- Mock all external dependencies (Anthropic API, database, Redis)
- Test agent behavior and decision-making logic
- Validate capability assessment and task matching
- Ensure proper agent collaboration patterns
- Test adaptive learning and persona evolution
"""

import asyncio
import uuid
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch, call

from app.core.real_agent_implementations import ClaudeAgent, AgentPersona
from app.core.agent_persona_system import AgentPersonaSystem
from app.core.capability_matcher import CapabilityMatcher
from app.core.intelligent_task_router import IntelligentTaskRouter
from app.core.agent_communication_service import AgentCommunicationService
from app.models.agent import Agent, AgentStatus, AgentType
from app.models.task import Task, TaskStatus, TaskPriority, TaskType


@pytest.mark.isolation
@pytest.mark.unit
class TestClaudeAgentIsolated:
    """Test Claude agent implementation in isolation."""
    
    @pytest.fixture
    async def isolated_claude_agent(
        self,
        mock_anthropic_client,
        mock_database_session,
        mock_redis_streams,
        isolated_agent_config,
        assert_isolated
    ):
        """Create isolated Claude agent with all dependencies mocked."""
        
        agent_config = isolated_agent_config(
            name="test-claude-agent",
            role="backend-engineer",
            capabilities=["python", "fastapi", "testing"]
        )
        
        with patch('app.core.real_agent_implementations.get_anthropic_client', return_value=mock_anthropic_client), \
             patch('app.core.real_agent_implementations.get_database_session', return_value=mock_database_session), \
             patch('app.core.real_agent_implementations.get_redis_client', return_value=mock_redis_streams):
            
            agent = ClaudeAgent(**agent_config)
            await agent.initialize()
            
            # Assert complete isolation
            assert_isolated(agent, {
                "anthropic": mock_anthropic_client,
                "database": mock_database_session,
                "redis": mock_redis_streams
            })
            
            yield agent
            
            await agent.shutdown()
    
    async def test_agent_initialization_isolated(self, isolated_claude_agent):
        """Test agent initialization and configuration in isolation."""
        agent = isolated_claude_agent
        
        # Verify agent state
        assert agent.id is not None
        assert agent.name == "test-claude-agent"
        assert agent.role == "backend-engineer"
        assert agent.status == AgentStatus.ACTIVE
        
        # Verify capabilities
        assert len(agent.capabilities) == 3
        capability_names = [cap["name"] for cap in agent.capabilities]
        assert "python" in capability_names
        assert "fastapi" in capability_names
        assert "testing" in capability_names
        
        # Verify agent has proper persona
        assert agent.persona is not None
        assert agent.persona.role == "backend-engineer"
        
        # Verify no real external connections
        assert not hasattr(agent, "_real_anthropic_connection")
        assert not hasattr(agent, "_real_database_connection")
    
    async def test_task_execution_isolated(
        self,
        isolated_claude_agent,
        isolated_task_config,
        mock_anthropic_client,
        capture_component_calls
    ):
        """Test task execution logic in isolation."""
        agent = isolated_claude_agent
        
        # Setup mock Claude response
        mock_anthropic_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Task completed successfully. Here's the implementation:\n\n```python\ndef hello_world():\n    return 'Hello, World!'\n```")],
            id="msg_test_123",
            model="claude-3-sonnet-20240229",
            role="assistant"
        )
        
        # Capture method calls
        calls, _ = capture_component_calls(agent, [
            "execute_task", "prepare_context", "generate_response"
        ])
        
        # Create task
        task_config = isolated_task_config(
            title="Implement Hello World Function",
            description="Create a simple Python function that returns 'Hello, World!'",
            task_type="feature_development",
            required_capabilities=["python"]
        )
        
        # Execute task
        result = await agent.execute_task(**task_config)
        
        # Verify execution result
        assert result["success"] is True
        assert "output" in result
        assert "execution_time" in result
        assert "Hello, World!" in result["output"]
        assert "python" in result["output"].lower()
        
        # Verify Claude API was called
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args
        assert "Hello World" in str(call_args)
        
        # Verify method calls
        assert len(calls) >= 2  # At least execute_task and generate_response
        assert any(call["method"] == "execute_task" for call in calls)
    
    async def test_capability_assessment_isolated(
        self,
        isolated_claude_agent,
        isolated_task_config
    ):
        """Test agent capability assessment in isolation."""
        agent = isolated_claude_agent
        
        # Test tasks with different capability requirements
        test_cases = [
            {
                "task": isolated_task_config(
                    title="Python API Development",
                    required_capabilities=["python", "fastapi"],
                    description="Build REST API with FastAPI"
                ),
                "expected_confidence": 0.9  # High confidence - agent has both capabilities
            },
            {
                "task": isolated_task_config(
                    title="Frontend React Component",
                    required_capabilities=["react", "typescript"],
                    description="Build React component with TypeScript"
                ),
                "expected_confidence": 0.1  # Low confidence - agent lacks these capabilities
            },
            {
                "task": isolated_task_config(
                    title="Write Unit Tests",
                    required_capabilities=["testing", "python"],
                    description="Create pytest test suite"
                ),
                "expected_confidence": 0.85  # High confidence - testing is a strength
            }
        ]
        
        for case in test_cases:
            confidence = await agent.assess_task_capability(**case["task"])
            
            assert 0 <= confidence <= 1, f"Confidence must be between 0 and 1, got {confidence}"
            
            # Allow some variance in confidence assessment
            if case["expected_confidence"] > 0.7:
                assert confidence > 0.6, f"Expected high confidence for {case['task']['title']}"
            elif case["expected_confidence"] < 0.3:
                assert confidence < 0.4, f"Expected low confidence for {case['task']['title']}"
    
    async def test_adaptive_learning_isolated(
        self,
        isolated_claude_agent,
        isolated_task_config
    ):
        """Test agent adaptive learning without external storage."""
        agent = isolated_claude_agent
        
        # Execute several similar tasks to enable learning
        learning_tasks = [
            {
                "title": "Test Case 1: Basic API Endpoint",
                "required_capabilities": ["python", "fastapi"],
                "success": True,
                "execution_time": 45.0,
                "feedback": "Well implemented, good error handling"
            },
            {
                "title": "Test Case 2: Advanced API with Validation",
                "required_capabilities": ["python", "fastapi"],
                "success": True,
                "execution_time": 38.0,
                "feedback": "Excellent use of Pydantic models"
            },
            {
                "title": "Test Case 3: API with Database Integration",
                "required_capabilities": ["python", "fastapi"],
                "success": False,
                "execution_time": 60.0,
                "feedback": "Needs improvement in database error handling"
            }
        ]
        
        initial_confidence = await agent.assess_task_capability(
            **isolated_task_config(required_capabilities=["python", "fastapi"])
        )
        
        # Execute learning tasks
        for task_data in learning_tasks:
            task_config = isolated_task_config(**{k: v for k, v in task_data.items() if k in ["title", "required_capabilities"]})
            
            # Mock execution result
            with patch.object(agent, 'execute_task', return_value={
                "success": task_data["success"],
                "execution_time": task_data["execution_time"],
                "output": f"Completed {task_data['title']}"
            }):
                await agent.execute_task(**task_config)
            
            # Provide feedback for learning
            await agent.receive_feedback(
                task_id=str(uuid.uuid4()),
                feedback=task_data["feedback"],
                success=task_data["success"]
            )
        
        # Check if confidence has been updated based on learning
        updated_confidence = await agent.assess_task_capability(
            **isolated_task_config(required_capabilities=["python", "fastapi"])
        )
        
        # Confidence should be influenced by experience (mix of success and failure)
        assert abs(updated_confidence - initial_confidence) > 0.01, "Agent should learn from experience"
        
        # Verify learning metrics
        learning_metrics = await agent.get_learning_metrics()
        assert learning_metrics["total_tasks_executed"] == 3
        assert learning_metrics["success_rate"] == 2/3  # 2 successes out of 3
        assert "capability_improvements" in learning_metrics
    
    async def test_context_management_isolated(
        self,
        isolated_claude_agent,
        isolated_task_config
    ):
        """Test agent context management in isolation."""
        agent = isolated_claude_agent
        
        # Add context information
        context_data = {
            "project_background": "Building a microservices architecture for e-commerce",
            "recent_decisions": [
                "Chose FastAPI over Flask for performance",
                "Decided on PostgreSQL for primary database"
            ],
            "coding_standards": {
                "language": "python",
                "style_guide": "PEP 8",
                "testing_framework": "pytest"
            },
            "current_focus": "API authentication and authorization"
        }
        
        await agent.update_context(context_data)
        
        # Execute task with context
        task_config = isolated_task_config(
            title="Implement User Authentication API",
            description="Create secure authentication endpoint",
            required_capabilities=["python", "fastapi"]
        )
        
        # Mock Claude response that should use context
        with patch.object(agent, '_generate_contextualized_prompt') as mock_prompt:
            mock_prompt.return_value = "Context-aware prompt with project background"
            
            await agent.execute_task(**task_config)
            
            # Verify context was used in prompt generation
            mock_prompt.assert_called_once()
            call_args = mock_prompt.call_args[1]
            assert "microservices" in str(call_args) or "context" in str(call_args)
        
        # Test context retrieval
        retrieved_context = await agent.get_context()
        assert "project_background" in retrieved_context
        assert "microservices" in retrieved_context["project_background"]
        
        # Test context compression (for memory efficiency)
        compressed_context = await agent.compress_context(target_size=0.5)
        assert len(str(compressed_context)) < len(str(context_data))
        assert "microservices" in str(compressed_context)  # Important info preserved


@pytest.mark.isolation
@pytest.mark.unit
class TestCapabilityMatcherIsolated:
    """Test capability matching system in isolation."""
    
    @pytest.fixture
    async def isolated_capability_matcher(
        self,
        mock_vector_search,
        isolated_test_environment
    ):
        """Create isolated capability matcher."""
        
        with patch('app.core.capability_matcher.get_embedding_service', return_value=mock_vector_search):
            matcher = CapabilityMatcher()
            await matcher.initialize()
            
            yield matcher
            
            await matcher.shutdown()
    
    async def test_capability_matching_algorithms_isolated(
        self,
        isolated_capability_matcher,
        isolated_agent_config,
        isolated_task_config
    ):
        """Test capability matching algorithms in isolation."""
        matcher = isolated_capability_matcher
        
        # Create agents with different capabilities
        agents = [
            isolated_agent_config(
                name="backend-specialist",
                role="backend-engineer", 
                capabilities=["python", "fastapi", "postgresql", "redis"]
            ),
            isolated_agent_config(
                name="frontend-specialist",
                role="frontend-developer",
                capabilities=["react", "typescript", "css", "webpack"]
            ),
            isolated_agent_config(
                name="fullstack-generalist",
                role="fullstack-developer",
                capabilities=["python", "react", "postgresql", "typescript"]
            ),
            isolated_agent_config(
                name="qa-specialist",
                role="qa-engineer",
                capabilities=["testing", "pytest", "playwright", "selenium"]
            )
        ]
        
        # Test tasks requiring different capability combinations
        test_cases = [
            {
                "task": isolated_task_config(
                    title="Backend API Development",
                    required_capabilities=["python", "fastapi", "postgresql"],
                    priority="high"
                ),
                "expected_best_match": "backend-specialist"
            },
            {
                "task": isolated_task_config(
                    title="Frontend Component Development", 
                    required_capabilities=["react", "typescript"],
                    priority="medium"
                ),
                "expected_best_match": "frontend-specialist"
            },
            {
                "task": isolated_task_config(
                    title="End-to-End Feature Development",
                    required_capabilities=["python", "react", "postgresql"],
                    priority="high"
                ),
                "expected_best_match": "fullstack-generalist"
            },
            {
                "task": isolated_task_config(
                    title="Test Automation",
                    required_capabilities=["testing", "pytest"],
                    priority="low"
                ),
                "expected_best_match": "qa-specialist"
            }
        ]
        
        for case in test_cases:
            matches = await matcher.find_best_matches(
                task=case["task"],
                available_agents=agents,
                max_matches=3
            )
            
            assert len(matches) > 0, f"No matches found for {case['task']['title']}"
            
            # Best match should be first in sorted results
            best_match = matches[0]
            assert best_match["agent"]["name"] == case["expected_best_match"], \
                f"Expected {case['expected_best_match']}, got {best_match['agent']['name']}"
            
            # Verify match score is reasonable
            assert 0.5 <= best_match["match_score"] <= 1.0, \
                f"Match score {best_match['match_score']} out of expected range"
            
            # Verify capability coverage
            assert best_match["capability_coverage"] >= 0.8, \
                "Best match should cover most required capabilities"
    
    async def test_semantic_capability_matching_isolated(
        self,
        isolated_capability_matcher,
        isolated_agent_config,
        isolated_task_config
    ):
        """Test semantic capability matching beyond exact keyword matching."""
        matcher = isolated_capability_matcher
        
        # Agent with semantic capabilities
        agent = isolated_agent_config(
            name="ml-engineer",
            capabilities=["machine_learning", "tensorflow", "data_analysis", "statistics"]
        )
        
        # Task using related but not exact terms
        task = isolated_task_config(
            title="AI Model Training",
            description="Train a neural network for image classification",
            required_capabilities=["artificial_intelligence", "deep_learning", "model_training"]
        )
        
        # Test semantic matching
        matches = await matcher.find_semantic_matches(
            task=task,
            available_agents=[agent],
            semantic_threshold=0.6
        )
        
        assert len(matches) > 0, "Should find semantic matches"
        
        match = matches[0]
        assert match["match_type"] == "semantic"
        assert match["semantic_similarity"] >= 0.6
        
        # Verify semantic reasoning
        reasoning = match["matching_reasoning"]
        assert "machine_learning" in reasoning or "AI" in reasoning
        assert "related" in reasoning.lower() or "similar" in reasoning.lower()
    
    async def test_dynamic_capability_scoring_isolated(
        self,
        isolated_capability_matcher,
        isolated_agent_config
    ):
        """Test dynamic capability scoring based on agent performance."""
        matcher = isolated_capability_matcher
        
        agent = isolated_agent_config(
            name="adaptive-agent",
            capabilities=["python", "fastapi", "testing"]
        )
        
        # Simulate performance history
        performance_history = [
            {"capability": "python", "success_rate": 0.95, "avg_time": 30.0, "complexity_handled": "high"},
            {"capability": "fastapi", "success_rate": 0.88, "avg_time": 45.0, "complexity_handled": "medium"},
            {"capability": "testing", "success_rate": 0.92, "avg_time": 25.0, "complexity_handled": "high"}
        ]
        
        # Update capability scores based on performance
        updated_scores = await matcher.update_capability_scores(
            agent_id=agent["id"],
            performance_history=performance_history
        )
        
        assert "python" in updated_scores
        assert "fastapi" in updated_scores
        assert "testing" in updated_scores
        
        # Python should have highest score due to best performance
        python_score = updated_scores["python"]
        fastapi_score = updated_scores["fastapi"]
        
        assert python_score > fastapi_score, "Python should score higher due to better performance"
        assert python_score >= 0.9, "High-performing capability should have high score"
        
        # Test capability score decay over time (simulating skill degradation)
        decayed_scores = await matcher.apply_temporal_decay(
            capability_scores=updated_scores,
            days_since_last_use={"python": 1, "fastapi": 30, "testing": 7}
        )
        
        # Capabilities used recently should maintain higher scores
        assert decayed_scores["python"] >= updated_scores["python"] * 0.95  # Minimal decay
        assert decayed_scores["fastapi"] <= updated_scores["fastapi"] * 0.8  # Significant decay
        assert decayed_scores["testing"] >= updated_scores["testing"] * 0.9  # Moderate decay


@pytest.mark.isolation
@pytest.mark.unit
class TestAgentCommunicationServiceIsolated:
    """Test agent communication service in isolation."""
    
    @pytest.fixture
    async def isolated_communication_service(
        self,
        mock_redis_streams,
        isolated_test_environment,
        assert_isolated
    ):
        """Create isolated communication service."""
        
        with patch('app.core.agent_communication_service.get_redis_client', return_value=mock_redis_streams):
            service = AgentCommunicationService()
            await service.initialize()
            
            assert_isolated(service, {"redis": mock_redis_streams})
            
            yield service
            
            await service.shutdown()
    
    async def test_inter_agent_messaging_isolated(
        self,
        isolated_communication_service,
        isolated_agent_config,
        mock_redis_streams
    ):
        """Test inter-agent messaging in isolation."""
        comm_service = isolated_communication_service
        
        # Setup mock Redis stream responses
        mock_redis_streams.xadd.return_value = "1234567890-0"
        mock_redis_streams.xreadgroup.return_value = [
            ("agent_messages", [
                ("1234567890-1", {
                    b"from_agent": b"agent_1",
                    b"to_agent": b"agent_2", 
                    b"message_type": b"task_request",
                    b"content": b'{"task": "help with API design"}'
                })
            ])
        ]
        
        # Create test agents
        agent_1 = isolated_agent_config(name="sender-agent")
        agent_2 = isolated_agent_config(name="receiver-agent")
        
        # Send message from agent 1 to agent 2
        message_data = {
            "from_agent_id": agent_1["id"],
            "to_agent_id": agent_2["id"],
            "message_type": "task_request",
            "content": {
                "task": "help with API design",
                "context": "building authentication service",
                "urgency": "medium"
            },
            "metadata": {"correlation_id": str(uuid.uuid4())}
        }
        
        result = await comm_service.send_message(**message_data)
        
        assert result["success"] is True
        assert "message_id" in result
        
        # Verify Redis stream was used
        mock_redis_streams.xadd.assert_called_once()
        call_args = mock_redis_streams.xadd.call_args
        assert "agent_messages" in str(call_args)
        
        # Test message retrieval
        messages = await comm_service.get_messages_for_agent(agent_2["id"])
        
        assert len(messages) > 0
        received_message = messages[0]
        assert received_message["from_agent_id"] == agent_1["id"]
        assert received_message["message_type"] == "task_request"
        assert "API design" in received_message["content"]["task"]
    
    async def test_broadcast_messaging_isolated(
        self,
        isolated_communication_service,
        isolated_agent_config,
        mock_redis_streams
    ):
        """Test broadcast messaging to multiple agents."""
        comm_service = isolated_communication_service
        
        # Setup multiple agents
        agents = [
            isolated_agent_config(name=f"agent-{i}", role="team-member")
            for i in range(5)
        ]
        
        # Broadcast announcement
        broadcast_data = {
            "from_agent_id": "system",
            "message_type": "announcement",
            "content": {
                "announcement": "New sprint starting tomorrow",
                "action_required": "Please review sprint backlog",
                "deadline": "2024-01-15T09:00:00Z"
            },
            "target_filter": {"role": "team-member"}
        }
        
        result = await comm_service.broadcast_message(**broadcast_data)
        
        assert result["success"] is True
        assert result["recipients_count"] == 5
        
        # Verify all agents received the message
        for agent in agents:
            messages = await comm_service.get_messages_for_agent(agent["id"])
            announcement_messages = [
                msg for msg in messages 
                if msg["message_type"] == "announcement"
            ]
            assert len(announcement_messages) > 0
    
    async def test_message_routing_patterns_isolated(
        self,
        isolated_communication_service,
        isolated_agent_config
    ):
        """Test different message routing patterns in isolation."""
        comm_service = isolated_communication_service
        
        # Create agents with different roles
        backend_agent = isolated_agent_config(role="backend-engineer")
        frontend_agent = isolated_agent_config(role="frontend-developer")
        qa_agent = isolated_agent_config(role="qa-engineer")
        
        # Test role-based routing
        role_message = {
            "from_agent_id": backend_agent["id"],
            "message_type": "help_request",
            "content": {"help_needed": "CSS styling issue"},
            "routing": {"target_role": "frontend-developer"}
        }
        
        routed_agents = await comm_service.route_message(**role_message)
        assert len(routed_agents) == 1
        assert routed_agents[0]["id"] == frontend_agent["id"]
        
        # Test capability-based routing
        capability_message = {
            "from_agent_id": frontend_agent["id"],
            "message_type": "collaboration_request",
            "content": {"collaboration_needed": "test automation setup"},
            "routing": {"required_capabilities": ["testing", "automation"]}
        }
        
        # Mock capability lookup
        with patch.object(comm_service, '_find_agents_by_capabilities', return_value=[qa_agent]):
            routed_agents = await comm_service.route_message(**capability_message)
            assert len(routed_agents) == 1
            assert routed_agents[0]["id"] == qa_agent["id"]
        
        # Test priority-based routing
        urgent_message = {
            "from_agent_id": qa_agent["id"],
            "message_type": "urgent_alert",
            "content": {"alert": "Production issue detected"},
            "routing": {"priority": "critical", "max_recipients": 2}
        }
        
        with patch.object(comm_service, '_find_available_agents', return_value=[backend_agent, frontend_agent]):
            routed_agents = await comm_service.route_message(**urgent_message)
            assert len(routed_agents) <= 2  # Respects max_recipients
    
    async def test_message_persistence_and_history_isolated(
        self,
        isolated_communication_service,
        isolated_agent_config,
        mock_redis_streams
    ):
        """Test message persistence and conversation history in isolation."""
        comm_service = isolated_communication_service
        
        agent_1 = isolated_agent_config(name="agent-alpha")
        agent_2 = isolated_agent_config(name="agent-beta")
        
        # Send conversation messages
        conversation_messages = [
            {
                "from_agent_id": agent_1["id"],
                "to_agent_id": agent_2["id"],
                "message_type": "question",
                "content": {"question": "How should we implement user authentication?"}
            },
            {
                "from_agent_id": agent_2["id"],
                "to_agent_id": agent_1["id"],
                "message_type": "response",
                "content": {"response": "I recommend using JWT tokens with refresh rotation"}
            },
            {
                "from_agent_id": agent_1["id"],
                "to_agent_id": agent_2["id"],
                "message_type": "follow_up",
                "content": {"follow_up": "What about session storage?"}
            }
        ]
        
        conversation_id = str(uuid.uuid4())
        
        for msg in conversation_messages:
            msg["conversation_id"] = conversation_id
            await comm_service.send_message(**msg)
        
        # Retrieve conversation history
        history = await comm_service.get_conversation_history(
            conversation_id=conversation_id,
            limit=10
        )
        
        assert len(history) == 3
        assert history[0]["message_type"] == "question"
        assert history[1]["message_type"] == "response" 
        assert history[2]["message_type"] == "follow_up"
        
        # Test conversation threading
        thread = await comm_service.get_conversation_thread(
            participant_1=agent_1["id"],
            participant_2=agent_2["id"]
        )
        
        assert len(thread) >= 3
        assert "authentication" in str(thread)
        assert "JWT" in str(thread)