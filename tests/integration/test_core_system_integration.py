"""
Comprehensive integration tests for core system workflows.

Tests cover:
- End-to-end agent orchestration workflows
- Multi-agent coordination scenarios
- System startup and shutdown procedures
- Cross-component integration
- Real-world usage patterns
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

from app.core.orchestrator import AgentOrchestrator, AgentRole, AgentInstance
from app.core.database import init_database, close_database, get_async_session
from app.core.redis import init_redis, close_redis, get_message_broker, get_session_cache
from app.core.workflow_engine import WorkflowEngine, TaskExecutionState
from app.models.agent import Agent, AgentStatus, AgentType
from app.models.task import Task, TaskStatus, TaskPriority
from app.models.workflow import Workflow, WorkflowStatus
from app.api.agent_activation import activate_agent_system
from app.main import app


class TestSystemInitializationWorkflow:
    """Test complete system initialization and startup workflow."""
    
    @pytest.mark.asyncio
    async def test_full_system_startup_sequence(self):
        """Test complete system startup from cold start to operational."""
        with patch('app.core.database.create_async_engine') as mock_engine, \
             patch('app.core.redis.redis.from_url') as mock_redis, \
             patch('app.core.orchestrator.get_container_orchestrator') as mock_container:
            
            # Mock successful database initialization
            mock_db_engine = AsyncMock()
            mock_db_conn = AsyncMock()
            mock_db_engine.begin.return_value.__aenter__.return_value = mock_db_conn
            mock_engine.return_value = mock_db_engine
            
            # Mock successful Redis initialization
            mock_redis_client = AsyncMock()
            mock_redis_client.ping.return_value = True
            mock_redis.return_value = mock_redis_client
            
            # Mock container orchestrator
            mock_container_orch = AsyncMock()
            mock_container.return_value = mock_container_orch
            
            # Step 1: Initialize database
            await init_database()
            mock_db_conn.execute.assert_called()
            
            # Step 2: Initialize Redis
            await init_redis()
            mock_redis_client.ping.assert_called()
            
            # Step 3: Create orchestrator
            orchestrator = AgentOrchestrator()
            await orchestrator._initialize()
            
            # Verify orchestrator is ready
            assert orchestrator.message_broker is not None
            assert orchestrator.session_cache is not None
            assert not orchestrator._shutdown_event.is_set()
            
            # Step 4: Cleanup
            await orchestrator.shutdown()
            await close_database()
            await close_redis()
    
    @pytest.mark.asyncio
    async def test_system_startup_with_database_failure(self):
        """Test system startup handling database connection failure."""
        with patch('app.core.database.create_async_engine') as mock_engine:
            # Mock database connection failure
            mock_engine.side_effect = Exception("Database connection failed")
            
            with pytest.raises(Exception, match="Database connection failed"):
                await init_database()
    
    @pytest.mark.asyncio
    async def test_system_startup_with_redis_failure(self):
        """Test system startup handling Redis connection failure."""
        with patch('app.core.redis.redis.from_url') as mock_redis:
            # Mock Redis connection failure
            mock_redis_client = AsyncMock()
            mock_redis_client.ping.side_effect = Exception("Redis connection failed")
            mock_redis.return_value = mock_redis_client
            
            with pytest.raises(Exception):
                await init_redis()


class TestAgentOrchestrationWorkflow:
    """Test complete agent orchestration workflows."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator with mocked dependencies."""
        with patch('app.core.orchestrator.get_message_broker'), \
             patch('app.core.orchestrator.get_session_cache'), \
             patch('app.core.orchestrator.get_session'), \
             patch('app.core.orchestrator.get_container_orchestrator'), \
             patch('app.core.orchestrator.get_agent_persona_system'):
            
            orchestrator = AgentOrchestrator()
            
            # Mock dependencies
            orchestrator.message_broker = AsyncMock()
            orchestrator.session_cache = AsyncMock()
            orchestrator.container_orchestrator = AsyncMock()
            orchestrator.persona_system = AsyncMock()
            orchestrator.workflow_engine = AsyncMock()
            orchestrator.task_router = AsyncMock()
            
            # Initialize
            orchestrator.agent_instances = {}
            orchestrator.agent_tasks = {}
            orchestrator.task_queue = asyncio.Queue()
            orchestrator._shutdown_event = asyncio.Event()
            
            await orchestrator._initialize()
            return orchestrator
    
    @pytest.fixture
    def sample_development_tasks(self):
        """Create sample development tasks for testing."""
        return [
            Task(
                id="task-1",
                title="Design API architecture",
                description="Design REST API for user management",
                priority=TaskPriority.HIGH,
                status=TaskStatus.PENDING,
                requirements=["architecture", "api_design"]
            ),
            Task(
                id="task-2",
                title="Implement backend endpoints",
                description="Implement user CRUD endpoints",
                priority=TaskPriority.HIGH,
                status=TaskStatus.PENDING,
                requirements=["python", "fastapi", "database"]
            ),
            Task(
                id="task-3",
                title="Create frontend components",
                description="Build user management UI components",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING,
                requirements=["frontend", "react", "typescript"]
            ),
            Task(
                id="task-4",
                title="Write comprehensive tests",
                description="Unit and integration tests for user management",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING,
                requirements=["testing", "pytest", "jest"]
            )
        ]
    
    @pytest.mark.asyncio
    async def test_complete_development_team_workflow(self, orchestrator, sample_development_tasks):
        """Test complete multi-agent development workflow."""
        # Step 1: Spawn development team
        team_agents = []
        for role in [AgentRole.ARCHITECT, AgentRole.BACKEND_DEVELOPER, AgentRole.FRONTEND_DEVELOPER, AgentRole.QA_ENGINEER]:
            # Mock successful agent spawning
            orchestrator.container_orchestrator.spawn_agent.return_value = {
                'container_id': f'container-{role.value}',
                'tmux_session': f'tmux-{role.value}',
                'status': 'running'
            }
            
            orchestrator.persona_system.assign_persona.return_value = MagicMock(
                role=role,
                capabilities=[]
            )
            
            agent_id = await orchestrator.spawn_agent(role=role, capabilities=[])
            team_agents.append(agent_id)
        
        assert len(team_agents) == 4
        assert len(orchestrator.agent_instances) == 4
        
        # Step 2: Assign tasks to appropriate agents
        task_assignments = []
        
        # Mock task router to assign tasks to appropriate agents
        def mock_task_routing(task, context):
            task_to_agent = {
                "task-1": team_agents[0],  # Architecture -> Architect
                "task-2": team_agents[1],  # Backend -> Backend Developer  
                "task-3": team_agents[2],  # Frontend -> Frontend Developer
                "task-4": team_agents[3],  # Testing -> QA Engineer
            }
            agent_id = task_to_agent.get(task.id)
            if agent_id:
                from app.core.intelligent_task_router import AgentSuitabilityScore
                return AgentSuitabilityScore(
                    agent_id=agent_id,
                    score=0.9,
                    reasoning=f"Perfect match for {task.title}",
                    confidence=0.95
                )
            return None
        
        orchestrator.task_router.find_best_agent.side_effect = mock_task_routing
        
        # Assign all tasks
        for task in sample_development_tasks:
            assigned_agent = await orchestrator.assign_task(task)
            task_assignments.append((task.id, assigned_agent))
        
        # Verify all tasks were assigned
        assert len(task_assignments) == 4
        assert all(agent_id is not None for _, agent_id in task_assignments)
        
        # Step 3: Simulate task execution and coordination
        for task_id, agent_id in task_assignments:
            agent = orchestrator.agent_instances[agent_id]
            assert agent.current_task == task_id
            assert task_id in orchestrator.agent_tasks[agent_id]
        
        # Step 4: Simulate task completion and workflow progression
        completed_tasks = []
        for task_id, agent_id in task_assignments:
            # Mock task completion
            orchestrator.workflow_engine.complete_task.return_value = TaskExecutionState.COMPLETED
            
            await orchestrator._handle_task_completion(task_id, agent_id)
            completed_tasks.append(task_id)
            
            # Agent should be freed for next task
            agent = orchestrator.agent_instances[agent_id]
            assert agent.current_task is None
        
        assert len(completed_tasks) == 4
        
        # Step 5: Shutdown team
        for agent_id in team_agents:
            orchestrator.container_orchestrator.shutdown_agent.return_value = True
            success = await orchestrator.shutdown_agent(agent_id)
            assert success is True
        
        assert len(orchestrator.agent_instances) == 0
    
    @pytest.mark.asyncio
    async def test_agent_failure_and_recovery_workflow(self, orchestrator):
        """Test agent failure recovery in multi-agent workflow."""
        # Step 1: Spawn agent
        orchestrator.container_orchestrator.spawn_agent.return_value = {
            'container_id': 'container-1',
            'tmux_session': 'tmux-1',
            'status': 'running'
        }
        
        orchestrator.persona_system.assign_persona.return_value = MagicMock(
            role=AgentRole.BACKEND_DEVELOPER,
            capabilities=[]
        )
        
        agent_id = await orchestrator.spawn_agent(
            role=AgentRole.BACKEND_DEVELOPER,
            capabilities=[]
        )
        
        # Step 2: Assign task to agent
        task = Task(
            id="critical-task",
            title="Critical backend implementation",
            priority=TaskPriority.CRITICAL,
            status=TaskStatus.PENDING
        )
        
        from app.core.intelligent_task_router import AgentSuitabilityScore
        orchestrator.task_router.find_best_agent.return_value = AgentSuitabilityScore(
            agent_id=agent_id,
            score=0.9,
            reasoning="Only available agent",
            confidence=0.8
        )
        
        assigned_agent = await orchestrator.assign_task(task)
        assert assigned_agent == agent_id
        
        # Step 3: Simulate agent failure
        agent = orchestrator.agent_instances[agent_id]
        agent.status = AgentStatus.ERROR
        orchestrator.container_orchestrator.check_agent_health.return_value = False
        
        # Step 4: Trigger failure recovery
        orchestrator.container_orchestrator.restart_agent.return_value = True
        orchestrator.task_router.find_best_agent.return_value = AgentSuitabilityScore(
            agent_id=agent_id,
            score=0.8,
            reasoning="Recovered agent",
            confidence=0.7
        )
        
        await orchestrator._handle_agent_failure(agent_id)
        
        # Verify recovery actions were taken
        orchestrator.container_orchestrator.restart_agent.assert_called_once()
        
        # Task should still be assigned after recovery
        assert task.id in orchestrator.agent_tasks[agent_id]
    
    @pytest.mark.asyncio
    async def test_load_balancing_workflow(self, orchestrator):
        """Test load balancing across multiple agents."""
        # Step 1: Spawn multiple similar agents
        agent_ids = []
        for i in range(3):
            orchestrator.container_orchestrator.spawn_agent.return_value = {
                'container_id': f'container-{i}',
                'tmux_session': f'tmux-{i}',
                'status': 'running'
            }
            
            orchestrator.persona_system.assign_persona.return_value = MagicMock(
                role=AgentRole.BACKEND_DEVELOPER,
                capabilities=[]
            )
            
            agent_id = await orchestrator.spawn_agent(
                role=AgentRole.BACKEND_DEVELOPER,
                capabilities=[]
            )
            agent_ids.append(agent_id)
            
            # Set different initial loads
            orchestrator.agent_tasks[agent_id] = [f"existing-{j}" for j in range(i)]
        
        # Step 2: Create multiple tasks
        tasks = [
            Task(id=f"task-{i}", title=f"Task {i}", priority=TaskPriority.MEDIUM, status=TaskStatus.PENDING)
            for i in range(6)
        ]
        
        # Step 3: Mock load-balancing task router
        assignment_counter = 0
        def mock_load_balancing_router(task, context):
            nonlocal assignment_counter
            # Assign to agent with least current load
            agent_loads = {
                agent_id: len(orchestrator.agent_tasks[agent_id])
                for agent_id in agent_ids
            }
            best_agent = min(agent_loads.keys(), key=lambda x: agent_loads[x])
            
            from app.core.intelligent_task_router import AgentSuitabilityScore
            assignment_counter += 1
            return AgentSuitabilityScore(
                agent_id=best_agent,
                score=0.8,
                reasoning=f"Load balanced assignment {assignment_counter}",
                confidence=0.8
            )
        
        orchestrator.task_router.find_best_agent.side_effect = mock_load_balancing_router
        
        # Step 4: Assign all tasks
        assignments = []
        for task in tasks:
            assigned_agent = await orchestrator.assign_task(task)
            assignments.append(assigned_agent)
        
        # Step 5: Verify load distribution
        final_loads = {
            agent_id: len(orchestrator.agent_tasks[agent_id])
            for agent_id in agent_ids
        }
        
        # Loads should be relatively balanced
        min_load = min(final_loads.values())
        max_load = max(final_loads.values())
        assert max_load - min_load <= 2  # Reasonable load distribution


class TestMultiAgentCoordination:
    """Test multi-agent coordination scenarios."""
    
    @pytest.fixture
    async def coordination_setup(self):
        """Set up coordination test environment."""
        with patch('app.core.redis.redis.from_url') as mock_redis:
            # Mock Redis for message broker
            mock_redis_client = AsyncMock()
            mock_redis_client.ping.return_value = True
            mock_redis.return_value = mock_redis_client
            
            await init_redis()
            
            # Get message broker
            message_broker = get_message_broker()
            
            return message_broker, mock_redis_client
    
    @pytest.mark.asyncio
    async def test_agent_communication_workflow(self, coordination_setup):
        """Test agent-to-agent communication workflow."""
        message_broker, mock_redis_client = coordination_setup
        
        # Mock Redis stream operations
        mock_redis_client.xadd.return_value = "1234567890-0"
        mock_redis_client.xread.return_value = {
            b'agent_messages:agent-2': [
                (b'1234567890-0', {
                    b'message_id': b'msg-123',
                    b'from_agent': b'agent-1',
                    b'to_agent': b'agent-2',
                    b'type': b'task_handoff',
                    b'payload': b'{"task_id": "task-456", "status": "ready_for_review"}',
                    b'correlation_id': b'corr-789'
                })
            ]
        }
        
        # Step 1: Agent 1 sends task handoff to Agent 2
        message_id = await message_broker.send_message(
            from_agent='agent-1',
            to_agent='agent-2',
            message_type='task_handoff',
            payload={
                'task_id': 'task-456',
                'status': 'ready_for_review',
                'artifacts': ['code.py', 'tests.py'],
                'notes': 'Please review the implementation'
            }
        )
        
        assert isinstance(message_id, str)
        mock_redis_client.xadd.assert_called_once()
        
        # Step 2: Agent 2 receives the message
        messages = await message_broker.receive_messages('agent-2', count=1)
        
        assert len(messages) == 1
        message = messages[0]
        assert message.from_agent == 'agent-1'
        assert message.to_agent == 'agent-2'
        assert message.message_type == 'task_handoff'
        assert message.payload['task_id'] == 'task-456'
        assert message.payload['status'] == 'ready_for_review'
        
        # Step 3: Agent 2 responds with review complete
        mock_redis_client.xadd.return_value = "1234567891-0"
        
        response_id = await message_broker.send_message(
            from_agent='agent-2',
            to_agent='agent-1',
            message_type='review_complete',
            payload={
                'task_id': 'task-456',
                'review_status': 'approved',
                'feedback': 'Implementation looks good, ready for deployment'
            },
            correlation_id=message.correlation_id
        )
        
        assert isinstance(response_id, str)
        assert mock_redis_client.xadd.call_count == 2
    
    @pytest.mark.asyncio
    async def test_broadcast_coordination_workflow(self, coordination_setup):
        """Test broadcast communication for team coordination."""
        message_broker, mock_redis_client = coordination_setup
        
        # Mock broadcast message
        mock_redis_client.xadd.return_value = "1234567890-0"
        
        # Step 1: System broadcasts sprint planning meeting
        broadcast_id = await message_broker.send_broadcast(
            from_agent='system',
            message_type='sprint_planning',
            payload={
                'meeting_time': '2025-01-15T10:00:00Z',
                'agenda': [
                    'Review previous sprint',
                    'Plan upcoming tasks',
                    'Assign responsibilities'
                ],
                'required_attendees': ['agent-1', 'agent-2', 'agent-3']
            }
        )
        
        assert isinstance(broadcast_id, str)
        mock_redis_client.xadd.assert_called_once()
        
        # Verify broadcast went to correct stream
        call_args = mock_redis_client.xadd.call_args
        assert call_args[0][0] == "agent_messages:broadcast"
        
        # Step 2: Agents register for coordination events
        await message_broker.register_agent('agent-1')
        await message_broker.register_agent('agent-2') 
        await message_broker.register_agent('agent-3')
        
        assert len(message_broker.active_agents) == 3
        
        # Step 3: Agents send acknowledgments
        for agent_id in ['agent-1', 'agent-2', 'agent-3']:
            mock_redis_client.xadd.return_value = f"123456789{agent_id[-1]}-0"
            
            await message_broker.send_message(
                from_agent=agent_id,
                to_agent='system',
                message_type='meeting_ack',
                payload={
                    'agent_id': agent_id,
                    'availability': 'confirmed',
                    'preparation_status': 'ready'
                }
            )
        
        # All agents should have acknowledged
        assert mock_redis_client.xadd.call_count == 4  # 1 broadcast + 3 acks
    
    @pytest.mark.asyncio
    async def test_heartbeat_coordination_workflow(self, coordination_setup):
        """Test agent heartbeat and health monitoring workflow."""
        message_broker, mock_redis_client = coordination_setup
        
        # Mock Redis hash operations for heartbeats
        mock_redis_client.hset.return_value = True
        mock_redis_client.hget.return_value = json.dumps({
            'status': 'active',
            'last_heartbeat': datetime.utcnow().isoformat(),
            'current_task': 'task-123',
            'memory_usage': 0.45
        }).encode()
        
        # Step 1: Register agents
        agents = ['agent-1', 'agent-2', 'agent-3']
        for agent_id in agents:
            await message_broker.register_agent(agent_id)
        
        # Step 2: Send heartbeats
        for agent_id in agents:
            await message_broker.send_heartbeat(agent_id, {
                'current_task': f'task-{agent_id.split("-")[1]}',
                'memory_usage': 0.3 + int(agent_id.split("-")[1]) * 0.1,
                'context_usage': 0.4
            })
        
        # Verify heartbeats were stored
        assert mock_redis_client.hset.call_count == len(agents)
        
        # Step 3: Check agent statuses
        for agent_id in agents:
            status = await message_broker.get_agent_status(agent_id)
            assert status['status'] == 'active'
            assert 'last_heartbeat' in status
            assert 'current_task' in status


class TestWorkflowEngineIntegration:
    """Test workflow engine integration with orchestrator."""
    
    @pytest.fixture
    def workflow_engine(self):
        """Create workflow engine instance."""
        with patch('app.core.database.get_async_session'):
            engine = WorkflowEngine()
            return engine
    
    @pytest.fixture
    def sample_workflow(self):
        """Create sample development workflow."""
        return Workflow(
            id="workflow-123",
            name="User Management Feature Development", 
            description="Complete user management feature implementation",
            status=WorkflowStatus.ACTIVE,
            steps=[
                {"id": "step-1", "name": "Architecture Design", "dependencies": []},
                {"id": "step-2", "name": "Backend Implementation", "dependencies": ["step-1"]},
                {"id": "step-3", "name": "Frontend Development", "dependencies": ["step-1"]},
                {"id": "step-4", "name": "Testing", "dependencies": ["step-2", "step-3"]},
                {"id": "step-5", "name": "Deployment", "dependencies": ["step-4"]}
            ]
        )
    
    @pytest.mark.asyncio
    async def test_workflow_execution_with_orchestrator(self, workflow_engine, sample_workflow):
        """Test complete workflow execution through orchestrator."""
        with patch('app.core.orchestrator.get_message_broker'), \
             patch('app.core.orchestrator.get_session_cache'), \
             patch('app.core.orchestrator.get_session'), \
             patch('app.core.orchestrator.get_container_orchestrator'):
            
            orchestrator = AgentOrchestrator()
            orchestrator.workflow_engine = workflow_engine
            
            # Mock workflow engine methods
            workflow_engine.start_workflow = AsyncMock(return_value=True)
            workflow_engine.get_ready_tasks = AsyncMock()
            workflow_engine.complete_task = AsyncMock(return_value=TaskExecutionState.COMPLETED)
            workflow_engine.is_workflow_complete = AsyncMock(return_value=False)
            
            # Step 1: Start workflow
            started = await orchestrator.start_workflow(sample_workflow)
            assert started is True
            workflow_engine.start_workflow.assert_called_once_with(sample_workflow)
            
            # Step 2: Get ready tasks (initially just step-1)
            workflow_engine.get_ready_tasks.return_value = [
                Task(id="task-step-1", title="Architecture Design", status=TaskStatus.PENDING)
            ]
            
            ready_tasks = await orchestrator.get_workflow_ready_tasks(sample_workflow.id)
            assert len(ready_tasks) == 1
            assert ready_tasks[0].id == "task-step-1"
            
            # Step 3: Complete step-1, which should unlock step-2 and step-3
            await orchestrator.complete_workflow_task(sample_workflow.id, "task-step-1")
            workflow_engine.complete_task.assert_called_once()
            
            # Step 4: Verify workflow progression
            workflow_engine.get_ready_tasks.return_value = [
                Task(id="task-step-2", title="Backend Implementation", status=TaskStatus.PENDING),
                Task(id="task-step-3", title="Frontend Development", status=TaskStatus.PENDING)
            ]
            
            ready_tasks = await orchestrator.get_workflow_ready_tasks(sample_workflow.id)
            assert len(ready_tasks) == 2
            task_ids = [task.id for task in ready_tasks]
            assert "task-step-2" in task_ids
            assert "task-step-3" in task_ids


class TestSystemShutdownWorkflow:
    """Test graceful system shutdown procedures."""
    
    @pytest.mark.asyncio
    async def test_graceful_orchestrator_shutdown(self):
        """Test graceful shutdown of orchestrator with active agents."""
        with patch('app.core.orchestrator.get_message_broker'), \
             patch('app.core.orchestrator.get_session_cache'), \
             patch('app.core.orchestrator.get_session'), \
             patch('app.core.orchestrator.get_container_orchestrator'):
            
            orchestrator = AgentOrchestrator()
            
            # Mock dependencies
            orchestrator.container_orchestrator = AsyncMock()
            orchestrator.message_broker = AsyncMock()
            orchestrator.session_cache = AsyncMock()
            
            # Initialize with some active agents
            orchestrator.agent_instances = {
                'agent-1': MagicMock(id='agent-1', status=AgentStatus.ACTIVE),
                'agent-2': MagicMock(id='agent-2', status=AgentStatus.ACTIVE)
            }
            
            orchestrator.agent_tasks = {
                'agent-1': ['task-1', 'task-2'],
                'agent-2': ['task-3']
            }
            
            # Mock successful agent shutdowns
            orchestrator.container_orchestrator.shutdown_agent.return_value = True
            
            # Perform graceful shutdown
            await orchestrator.shutdown()
            
            # Verify all agents were shut down
            assert orchestrator.container_orchestrator.shutdown_agent.call_count == 2
            
            # Verify cleanup
            assert len(orchestrator.agent_instances) == 0
            assert len(orchestrator.agent_tasks) == 0
            assert orchestrator._shutdown_event.is_set()
    
    @pytest.mark.asyncio
    async def test_system_shutdown_with_stuck_agents(self):
        """Test system shutdown handling agents that don't respond."""
        with patch('app.core.orchestrator.get_message_broker'), \
             patch('app.core.orchestrator.get_session_cache'), \
             patch('app.core.orchestrator.get_session'), \
             patch('app.core.orchestrator.get_container_orchestrator'):
            
            orchestrator = AgentOrchestrator()
            
            # Mock dependencies
            orchestrator.container_orchestrator = AsyncMock()
            
            # Initialize with agents
            orchestrator.agent_instances = {
                'agent-1': MagicMock(id='agent-1', status=AgentStatus.ACTIVE),
                'agent-2': MagicMock(id='agent-2', status=AgentStatus.ACTIVE)
            }
            
            # Mock one agent shutting down successfully, one getting stuck
            def mock_shutdown(tmux_session):
                if 'agent-1' in str(tmux_session):
                    return True
                else:
                    # Simulate stuck agent
                    raise asyncio.TimeoutError("Agent shutdown timed out")
            
            orchestrator.container_orchestrator.shutdown_agent.side_effect = mock_shutdown
            
            # Force shutdown should complete despite stuck agent
            await orchestrator.shutdown(force=True, timeout=1.0)
            
            # Verify shutdown was attempted for both agents
            assert orchestrator.container_orchestrator.shutdown_agent.call_count == 2
            
            # System should still complete shutdown
            assert orchestrator._shutdown_event.is_set()


class TestRealWorldScenarios:
    """Test real-world usage patterns and edge cases."""
    
    @pytest.mark.asyncio
    async def test_high_load_scenario(self):
        """Test system behavior under high load."""
        with patch('app.core.orchestrator.get_message_broker'), \
             patch('app.core.orchestrator.get_session_cache'), \
             patch('app.core.orchestrator.get_session'), \
             patch('app.core.orchestrator.get_container_orchestrator'):
            
            orchestrator = AgentOrchestrator()
            
            # Mock dependencies
            orchestrator.container_orchestrator = AsyncMock()
            orchestrator.task_router = AsyncMock()
            orchestrator.workflow_engine = AsyncMock()
            
            # Create many agents
            agent_ids = []
            for i in range(20):  # 20 agents
                orchestrator.container_orchestrator.spawn_agent.return_value = {
                    'container_id': f'container-{i}',
                    'tmux_session': f'tmux-{i}',
                    'status': 'running'
                }
                
                agent_id = await orchestrator.spawn_agent(
                    role=AgentRole.BACKEND_DEVELOPER,
                    capabilities=[]
                )
                agent_ids.append(agent_id)
            
            # Create many tasks
            tasks = [
                Task(id=f"task-{i}", title=f"Task {i}", priority=TaskPriority.MEDIUM, status=TaskStatus.PENDING)
                for i in range(100)  # 100 tasks
            ]
            
            # Mock task routing to distribute tasks
            assignment_index = 0
            def mock_high_load_routing(task, context):
                nonlocal assignment_index
                agent_id = agent_ids[assignment_index % len(agent_ids)]
                assignment_index += 1
                
                from app.core.intelligent_task_router import AgentSuitabilityScore
                return AgentSuitabilityScore(
                    agent_id=agent_id,
                    score=0.7,
                    reasoning="Round-robin assignment",
                    confidence=0.6
                )
            
            orchestrator.task_router.find_best_agent.side_effect = mock_high_load_routing
            
            # Assign all tasks concurrently
            async def assign_task(task):
                return await orchestrator.assign_task(task)
            
            assignments = await asyncio.gather(
                *[assign_task(task) for task in tasks],
                return_exceptions=True
            )
            
            # Verify all tasks were assigned successfully
            successful_assignments = [a for a in assignments if not isinstance(a, Exception)]
            assert len(successful_assignments) >= 95  # Allow for some failures under high load
            
            # Verify load distribution
            task_counts = {}
            for agent_id in agent_ids:
                task_counts[agent_id] = len(orchestrator.agent_tasks.get(agent_id, []))
            
            avg_tasks = sum(task_counts.values()) / len(agent_ids)
            # Load should be relatively balanced
            assert all(abs(count - avg_tasks) <= 3 for count in task_counts.values())
    
    @pytest.mark.asyncio
    async def test_mixed_priority_workflow(self):
        """Test handling of mixed priority tasks and emergency scenarios."""
        with patch('app.core.orchestrator.get_message_broker'), \
             patch('app.core.orchestrator.get_session_cache'), \
             patch('app.core.orchestrator.get_session'), \
             patch('app.core.orchestrator.get_container_orchestrator'):
            
            orchestrator = AgentOrchestrator()
            orchestrator.task_router = AsyncMock()
            
            # Create agents
            agent_id = await orchestrator.spawn_agent(
                role=AgentRole.BACKEND_DEVELOPER,
                capabilities=[]
            )
            
            # Create mixed priority tasks
            normal_task = Task(
                id="normal-task",
                title="Regular feature",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING
            )
            
            critical_task = Task(
                id="critical-task", 
                title="Production bug fix",
                priority=TaskPriority.CRITICAL,
                status=TaskStatus.PENDING
            )
            
            # Assign normal task first
            from app.core.intelligent_task_router import AgentSuitabilityScore
            orchestrator.task_router.find_best_agent.return_value = AgentSuitabilityScore(
                agent_id=agent_id,
                score=0.8,
                reasoning="Available agent",
                confidence=0.8
            )
            
            await orchestrator.assign_task(normal_task)
            assert orchestrator.agent_instances[agent_id].current_task == normal_task.id
            
            # Critical task arrives - should preempt normal task
            await orchestrator.assign_critical_task(critical_task)
            
            # Normal task should be paused/queued
            assert not orchestrator.task_queue.empty()
            queued_task = await orchestrator.task_queue.get()
            assert queued_task.id == normal_task.id
            
            # Critical task should be assigned
            assert orchestrator.agent_instances[agent_id].current_task == critical_task.id