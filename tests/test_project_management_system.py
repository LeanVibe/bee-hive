"""
Tests for Project Management System

Comprehensive test suite for the project management hierarchy,
kanban workflow, and orchestrator integration.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.project_management import (
    Project, Epic, PRD, Task, KanbanState, ProjectStatus, 
    EpicStatus, PRDStatus, TaskType, TaskPriority
)
from app.models.agent import Agent, AgentStatus, AgentType
from app.core.kanban_state_machine import (
    KanbanStateMachine, StateTransitionResult, WorkflowMetrics
)
from app.services.project_management_service import (
    ProjectManagementService, ProjectHierarchyStats, WorkloadAnalysis
)
from app.core.project_management_orchestrator_integration import (
    ProjectManagementOrchestratorIntegration, TaskMigrationMapping, 
    WorkflowTransitionEvent
)


class TestDatabaseSetup:
    """Test database setup utilities."""
    
    @pytest.fixture
    def db_session(self):
        """Create in-memory test database."""
        # Create in-memory SQLite database
        engine = create_engine(
            "sqlite:///:memory:", 
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        
        # Create all tables
        Base.metadata.create_all(engine)
        
        # Create session
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        yield session
        
        session.close()
    
    @pytest.fixture
    def sample_agent(self, db_session):
        """Create a sample agent for testing."""
        agent = Agent(
            name="Test Agent",
            status=AgentStatus.ACTIVE,
            agent_type=AgentType.CLAUDE,
            capabilities=["python", "testing", "development"],
            current_context_tokens=0,
            max_context_tokens=100000
        )
        db_session.add(agent)
        db_session.commit()
        return agent


class TestProjectManagementModels:
    """Test project management data models."""
    
    def test_project_creation(self, db_session, sample_agent):
        """Test project creation with proper defaults."""
        project = Project(
            name="Test Project",
            description="A test project",
            owner_agent_id=sample_agent.id
        )
        project.ensure_short_id(db_session)
        
        db_session.add(project)
        db_session.commit()
        
        assert project.name == "Test Project"
        assert project.status == ProjectStatus.PLANNING
        assert project.kanban_state == KanbanState.BACKLOG
        assert project.short_id is not None
        assert project.short_id.startswith("PRJ-")
        assert len(project.objectives) == 0
        assert len(project.success_criteria) == 0
    
    def test_epic_creation(self, db_session, sample_agent):
        """Test epic creation within project."""
        project = Project(name="Test Project")
        project.ensure_short_id(db_session)
        db_session.add(project)
        db_session.flush()
        
        epic = Epic(
            name="Test Epic",
            description="A test epic",
            project_id=project.id,
            priority=TaskPriority.HIGH,
            owner_agent_id=sample_agent.id
        )
        epic.ensure_short_id(db_session)
        
        db_session.add(epic)
        db_session.commit()
        
        assert epic.name == "Test Epic"
        assert epic.status == EpicStatus.DRAFT
        assert epic.kanban_state == KanbanState.BACKLOG
        assert epic.priority == TaskPriority.HIGH
        assert epic.short_id.startswith("EPC-")
        assert epic.project_id == project.id
    
    def test_prd_creation(self, db_session, sample_agent):
        """Test PRD creation within epic."""
        project = Project(name="Test Project")
        project.ensure_short_id(db_session)
        db_session.add(project)
        db_session.flush()
        
        epic = Epic(name="Test Epic", project_id=project.id)
        epic.ensure_short_id(db_session)
        db_session.add(epic)
        db_session.flush()
        
        prd = PRD(
            title="Test PRD",
            description="A test PRD",
            epic_id=epic.id,
            complexity_score=5,
            owner_agent_id=sample_agent.id
        )
        prd.ensure_short_id(db_session)
        
        db_session.add(prd)
        db_session.commit()
        
        assert prd.title == "Test PRD"
        assert prd.status == PRDStatus.DRAFT
        assert prd.kanban_state == KanbanState.BACKLOG
        assert prd.complexity_score == 5
        assert prd.short_id.startswith("PRD-")
        assert prd.epic_id == epic.id
    
    def test_task_creation(self, db_session, sample_agent):
        """Test task creation within PRD."""
        project = Project(name="Test Project")
        project.ensure_short_id(db_session)
        db_session.add(project)
        db_session.flush()
        
        epic = Epic(name="Test Epic", project_id=project.id)
        epic.ensure_short_id(db_session)
        db_session.add(epic)
        db_session.flush()
        
        prd = PRD(title="Test PRD", epic_id=epic.id)
        prd.ensure_short_id(db_session)
        db_session.add(prd)
        db_session.flush()
        
        task = Task(
            title="Test Task",
            description="A test task",
            prd_id=prd.id,
            task_type=TaskType.FEATURE_DEVELOPMENT,
            priority=TaskPriority.MEDIUM,
            estimated_effort_minutes=120,
            assigned_agent_id=sample_agent.id
        )
        task.ensure_short_id(db_session)
        
        db_session.add(task)
        db_session.commit()
        
        assert task.title == "Test Task"
        assert task.kanban_state == KanbanState.BACKLOG
        assert task.task_type == TaskType.FEATURE_DEVELOPMENT
        assert task.priority == TaskPriority.MEDIUM
        assert task.estimated_effort_minutes == 120
        assert task.short_id.startswith("TSK-")
        assert task.prd_id == prd.id
        assert task.assigned_agent_id == sample_agent.id
    
    def test_hierarchy_relationships(self, db_session, sample_agent):
        """Test hierarchical relationships between entities."""
        # Create full hierarchy
        project = Project(name="Test Project")
        project.ensure_short_id(db_session)
        db_session.add(project)
        db_session.flush()
        
        epic = Epic(name="Test Epic", project_id=project.id)
        epic.ensure_short_id(db_session)
        db_session.add(epic)
        db_session.flush()
        
        prd = PRD(title="Test PRD", epic_id=epic.id)
        prd.ensure_short_id(db_session)
        db_session.add(prd)
        db_session.flush()
        
        task = Task(
            title="Test Task",
            prd_id=prd.id,
            assigned_agent_id=sample_agent.id
        )
        task.ensure_short_id(db_session)
        db_session.add(task)
        db_session.commit()
        
        # Test relationships
        db_session.refresh(project)
        db_session.refresh(epic)
        db_session.refresh(prd)
        db_session.refresh(task)
        
        assert len(project.epics) == 1
        assert project.epics[0].id == epic.id
        assert len(epic.prds) == 1
        assert epic.prds[0].id == prd.id
        assert len(prd.tasks) == 1
        assert prd.tasks[0].id == task.id
        
        # Test reverse navigation
        assert task.get_project().id == project.id
        assert task.get_epic().id == epic.id


class TestKanbanStateMachine:
    """Test Kanban state machine functionality."""
    
    @pytest.fixture
    def kanban_machine(self, db_session):
        """Create Kanban state machine instance."""
        return KanbanStateMachine(db_session)
    
    def test_valid_task_transitions(self, db_session, kanban_machine, sample_agent):
        """Test valid task state transitions."""
        # Create task hierarchy
        project = Project(name="Test Project")
        project.ensure_short_id(db_session)
        db_session.add(project)
        db_session.flush()
        
        epic = Epic(name="Test Epic", project_id=project.id)
        epic.ensure_short_id(db_session)
        db_session.add(epic)
        db_session.flush()
        
        prd = PRD(title="Test PRD", epic_id=epic.id)
        prd.ensure_short_id(db_session)
        db_session.add(prd)
        db_session.flush()
        
        task = Task(
            title="Test Task",
            prd_id=prd.id,
            assigned_agent_id=sample_agent.id
        )
        task.ensure_short_id(db_session)
        db_session.add(task)
        db_session.commit()
        
        # Test valid transitions
        assert task.kanban_state == KanbanState.BACKLOG
        
        # BACKLOG -> READY
        result = kanban_machine.transition_entity_state(
            task, KanbanState.READY, sample_agent.id, "Moving to ready"
        )
        assert result.success
        assert task.kanban_state == KanbanState.READY
        
        # READY -> IN_PROGRESS
        result = kanban_machine.transition_entity_state(
            task, KanbanState.IN_PROGRESS, sample_agent.id, "Starting work"
        )
        assert result.success
        assert task.kanban_state == KanbanState.IN_PROGRESS
        assert task.actual_start is not None
        
        # IN_PROGRESS -> REVIEW
        result = kanban_machine.transition_entity_state(
            task, KanbanState.REVIEW, sample_agent.id, "Ready for review"
        )
        assert result.success
        assert task.kanban_state == KanbanState.REVIEW
        
        # REVIEW -> DONE
        result = kanban_machine.transition_entity_state(
            task, KanbanState.DONE, sample_agent.id, "Completed"
        )
        assert result.success
        assert task.kanban_state == KanbanState.DONE
        assert task.actual_completion is not None
    
    def test_invalid_task_transitions(self, db_session, kanban_machine, sample_agent):
        """Test invalid task state transitions are rejected."""
        project = Project(name="Test Project")
        project.ensure_short_id(db_session)
        db_session.add(project)
        db_session.flush()
        
        epic = Epic(name="Test Epic", project_id=project.id)
        epic.ensure_short_id(db_session)
        db_session.add(epic)
        db_session.flush()
        
        prd = PRD(title="Test PRD", epic_id=epic.id)
        prd.ensure_short_id(db_session)
        db_session.add(prd)
        db_session.flush()
        
        task = Task(title="Test Task", prd_id=prd.id)
        task.ensure_short_id(db_session)
        db_session.add(task)
        db_session.commit()
        
        # Try invalid transition: BACKLOG -> DONE
        result = kanban_machine.transition_entity_state(
            task, KanbanState.DONE, sample_agent.id, "Invalid transition"
        )
        assert not result.success
        assert len(result.errors) > 0
        assert task.kanban_state == KanbanState.BACKLOG  # Should remain unchanged
    
    def test_workflow_metrics(self, db_session, kanban_machine, sample_agent):
        """Test workflow metrics calculation."""
        # Create multiple tasks in different states
        project = Project(name="Test Project")
        project.ensure_short_id(db_session)
        db_session.add(project)
        db_session.flush()
        
        epic = Epic(name="Test Epic", project_id=project.id)
        epic.ensure_short_id(db_session)
        db_session.add(epic)
        db_session.flush()
        
        prd = PRD(title="Test PRD", epic_id=epic.id)
        prd.ensure_short_id(db_session)
        db_session.add(prd)
        db_session.flush()
        
        # Create tasks in different states
        states = [KanbanState.BACKLOG, KanbanState.READY, KanbanState.IN_PROGRESS, KanbanState.DONE]
        for i, state in enumerate(states):
            task = Task(
                title=f"Test Task {i}",
                prd_id=prd.id,
                kanban_state=state
            )
            task.ensure_short_id(db_session)
            
            # Set timestamps for completed tasks
            if state == KanbanState.DONE:
                task.actual_start = datetime.utcnow() - timedelta(hours=2)
                task.actual_completion = datetime.utcnow()
            
            db_session.add(task)
        
        db_session.commit()
        
        # Get metrics
        metrics = kanban_machine.get_workflow_metrics("Task", 7)
        
        assert metrics.entity_type == "Task"
        assert metrics.total_entities == 4
        assert metrics.state_counts[KanbanState.BACKLOG] == 1
        assert metrics.state_counts[KanbanState.READY] == 1
        assert metrics.state_counts[KanbanState.IN_PROGRESS] == 1
        assert metrics.state_counts[KanbanState.DONE] == 1


class TestProjectManagementService:
    """Test project management service functionality."""
    
    @pytest.fixture
    def project_service(self, db_session):
        """Create project management service instance."""
        return ProjectManagementService(db_session)
    
    def test_create_project_with_template(self, db_session, project_service, sample_agent):
        """Test creating project with initial structure."""
        project, initial_epics = project_service.create_project_with_initial_structure(
            name="Test Web App",
            description="A test web application",
            template_type="web_application",
            owner_agent_id=sample_agent.id
        )
        
        assert project.name == "Test Web App"
        assert project.owner_agent_id == sample_agent.id
        assert len(initial_epics) > 0
        assert any("Authentication" in epic.name for epic in initial_epics)
        assert any("Core Features" in epic.name for epic in initial_epics)
    
    def test_auto_generate_implementation_tasks(self, db_session, project_service, sample_agent):
        """Test automatic task generation for PRDs."""
        # Create hierarchy
        project = Project(name="Test Project")
        project.ensure_short_id(db_session)
        db_session.add(project)
        db_session.flush()
        
        epic = Epic(name="Test Epic", project_id=project.id)
        epic.ensure_short_id(db_session)
        db_session.add(epic)
        db_session.flush()
        
        prd = PRD(
            title="User Authentication PRD",
            epic_id=epic.id,
            complexity_score=7,
            estimated_effort_days=5
        )
        prd.ensure_short_id(db_session)
        db_session.add(prd)
        db_session.commit()
        
        # Generate tasks
        generated_tasks = project_service.auto_generate_implementation_tasks(
            prd.id, "standard"
        )
        
        assert len(generated_tasks) > 0
        assert any("Implementation" in task.title for task in generated_tasks)
        assert any("Testing" in task.title for task in generated_tasks)
        
        # Verify tasks are saved
        db_session.refresh(prd)
        assert len(prd.tasks) == len(generated_tasks)
    
    def test_smart_task_assignment(self, db_session, project_service, sample_agent):
        """Test smart task assignment to agents."""
        # Create additional agents
        agent2 = Agent(
            name="Agent 2",
            status=AgentStatus.ACTIVE,
            agent_type=AgentType.CLAUDE,
            capabilities=["python", "backend"]
        )
        db_session.add(agent2)
        db_session.flush()
        
        # Create tasks
        project = Project(name="Test Project")
        project.ensure_short_id(db_session)
        db_session.add(project)
        db_session.flush()
        
        epic = Epic(name="Test Epic", project_id=project.id)
        epic.ensure_short_id(db_session)
        db_session.add(epic)
        db_session.flush()
        
        prd = PRD(title="Test PRD", epic_id=epic.id)
        prd.ensure_short_id(db_session)
        db_session.add(prd)
        db_session.flush()
        
        tasks = []
        for i in range(3):
            task = Task(
                title=f"Test Task {i}",
                prd_id=prd.id,
                task_type=TaskType.FEATURE_DEVELOPMENT,
                kanban_state=KanbanState.READY
            )
            task.ensure_short_id(db_session)
            db_session.add(task)
            tasks.append(task)
        
        db_session.commit()
        
        # Perform smart assignment
        task_ids = [task.id for task in tasks]
        assignments = project_service.smart_task_assignment(task_ids, "balanced")
        
        assert len(assignments) == len(tasks)
        for task_id, agent_id in assignments.items():
            assert agent_id in [sample_agent.id, agent2.id]
    
    def test_get_project_hierarchy_stats(self, db_session, project_service, sample_agent):
        """Test project hierarchy statistics calculation."""
        # Create full hierarchy with completed items
        project = Project(name="Test Project")
        project.ensure_short_id(db_session)
        db_session.add(project)
        db_session.flush()
        
        epic = Epic(name="Test Epic", project_id=project.id, kanban_state=KanbanState.DONE)
        epic.ensure_short_id(db_session)
        db_session.add(epic)
        db_session.flush()
        
        prd = PRD(title="Test PRD", epic_id=epic.id, kanban_state=KanbanState.DONE)
        prd.ensure_short_id(db_session)
        db_session.add(prd)
        db_session.flush()
        
        # Create mix of completed and pending tasks
        for i in range(5):
            state = KanbanState.DONE if i < 3 else KanbanState.IN_PROGRESS
            task = Task(
                title=f"Test Task {i}",
                prd_id=prd.id,
                kanban_state=state
            )
            task.ensure_short_id(db_session)
            
            if state == KanbanState.DONE:
                task.actual_start = datetime.utcnow() - timedelta(hours=2)
                task.actual_completion = datetime.utcnow()
            
            db_session.add(task)
        
        db_session.commit()
        
        # Get stats
        stats = project_service.get_project_hierarchy_stats(project.id)
        
        assert stats.project_count == 1
        assert stats.epic_count == 1
        assert stats.prd_count == 1
        assert stats.task_count == 5
        assert stats.completed_epics == 1
        assert stats.completed_prds == 1
        assert stats.completed_tasks == 3
        assert stats.state_distribution[KanbanState.DONE] == 3
        assert stats.state_distribution[KanbanState.IN_PROGRESS] == 2


class TestOrchestratorIntegration:
    """Test orchestrator integration functionality."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock SimpleOrchestrator."""
        orchestrator = MagicMock()
        orchestrator._handle_task_assignment_change = MagicMock()
        orchestrator._update_agent_workload = MagicMock()
        orchestrator._trigger_workflow_evaluation = MagicMock()
        return orchestrator
    
    @pytest.fixture
    def integration(self, db_session, mock_orchestrator):
        """Create orchestrator integration instance."""
        from app.core.project_management_orchestrator_integration import (
            ProjectManagementOrchestratorIntegration
        )
        integration = ProjectManagementOrchestratorIntegration(db_session, mock_orchestrator)
        integration.initialize_integration()
        return integration
    
    def test_workflow_transition_handling(self, db_session, integration, sample_agent):
        """Test workflow transition event handling."""
        # Create task
        project = Project(name="Test Project")
        project.ensure_short_id(db_session)
        db_session.add(project)
        db_session.flush()
        
        epic = Epic(name="Test Epic", project_id=project.id)
        epic.ensure_short_id(db_session)
        db_session.add(epic)
        db_session.flush()
        
        prd = PRD(title="Test PRD", epic_id=epic.id)
        prd.ensure_short_id(db_session)
        db_session.add(prd)
        db_session.flush()
        
        task = Task(
            title="Test Task",
            prd_id=prd.id,
            kanban_state=KanbanState.READY
        )
        task.ensure_short_id(db_session)
        db_session.add(task)
        db_session.commit()
        
        # Handle workflow transition
        old_state = task.kanban_state
        new_state = KanbanState.IN_PROGRESS
        
        actions = integration.handle_workflow_transition(
            task, old_state, new_state, sample_agent.id
        )
        
        assert isinstance(actions, list)
        # Should auto-assign task since it's moving to IN_PROGRESS without assignee
        assert any("Auto-assigned" in action for action in actions)
    
    def test_agent_workload_analysis(self, db_session, integration, sample_agent):
        """Test agent workload analysis."""
        # Create tasks assigned to agent
        project = Project(name="Test Project")
        project.ensure_short_id(db_session)
        db_session.add(project)
        db_session.flush()
        
        epic = Epic(name="Test Epic", project_id=project.id)
        epic.ensure_short_id(db_session)
        db_session.add(epic)
        db_session.flush()
        
        prd = PRD(title="Test PRD", epic_id=epic.id)
        prd.ensure_short_id(db_session)
        db_session.add(prd)
        db_session.flush()
        
        # Create tasks with different states
        states = [KanbanState.READY, KanbanState.IN_PROGRESS, KanbanState.DONE]
        for i, state in enumerate(states):
            task = Task(
                title=f"Test Task {i}",
                prd_id=prd.id,
                kanban_state=state,
                assigned_agent_id=sample_agent.id,
                estimated_effort_minutes=120
            )
            task.ensure_short_id(db_session)
            db_session.add(task)
        
        db_session.commit()
        
        # Analyze workload
        workload = integration.get_agent_project_workload(sample_agent.id)
        
        assert workload["agent_id"] == str(sample_agent.id)
        assert workload["project_tasks"]["total"] == 3
        assert workload["project_tasks"]["by_state"]["ready"] == 1
        assert workload["project_tasks"]["by_state"]["in_progress"] == 1
        assert workload["project_tasks"]["by_state"]["done"] == 1
        assert workload["project_tasks"]["estimated_hours"] == 6.0  # 3 tasks * 2 hours each
    
    def test_integration_health_check(self, db_session, integration):
        """Test integration health monitoring."""
        health = integration.get_integration_health()
        
        assert health["initialized"] is True
        assert "components" in health
        assert health["components"]["kanban_machine"] is True
        assert health["components"]["project_service"] is True
        assert health["components"]["task_router"] is True
        assert "entity_counts" in health


# Integration Tests

class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    def test_complete_project_workflow(self, db_session, sample_agent):
        """Test complete project workflow from creation to completion."""
        # Create project with service
        service = ProjectManagementService(db_session)
        
        project, epics = service.create_project_with_initial_structure(
            name="Complete Test Project",
            description="End-to-end test",
            template_type="web_application",
            owner_agent_id=sample_agent.id
        )
        
        assert len(epics) > 0
        
        # Create PRD in first epic
        first_epic = epics[0]
        prd = PRD(
            title="Authentication PRD",
            epic_id=first_epic.id,
            complexity_score=5
        )
        prd.ensure_short_id(db_session)
        db_session.add(prd)
        db_session.commit()
        
        # Generate implementation tasks
        tasks = service.auto_generate_implementation_tasks(prd.id, "standard")
        assert len(tasks) > 0
        
        # Assign tasks
        task_ids = [task.id for task in tasks]
        assignments = service.smart_task_assignment(task_ids, "balanced")
        assert len(assignments) == len(tasks)
        
        # Simulate workflow progression
        kanban = KanbanStateMachine(db_session)
        
        for task in tasks[:2]:  # Complete first 2 tasks
            # Move through workflow
            result = kanban.transition_entity_state(task, KanbanState.READY, sample_agent.id)
            assert result.success
            
            result = kanban.transition_entity_state(task, KanbanState.IN_PROGRESS, sample_agent.id)
            assert result.success
            
            result = kanban.transition_entity_state(task, KanbanState.REVIEW, sample_agent.id)
            assert result.success
            
            result = kanban.transition_entity_state(task, KanbanState.DONE, sample_agent.id)
            assert result.success
        
        # Check project stats
        stats = service.get_project_hierarchy_stats(project.id)
        assert stats.completed_tasks == 2
        assert stats.task_count == len(tasks)
        
        # Verify metrics
        metrics = kanban.get_workflow_metrics("Task", 1)
        assert metrics.total_entities >= 2
        assert metrics.state_counts[KanbanState.DONE] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])