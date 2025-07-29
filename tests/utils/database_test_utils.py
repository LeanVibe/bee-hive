"""
Database testing utilities for LeanVibe Agent Hive 2.0.

Provides utilities for:
- Database schema validation
- Test data factories
- Transaction management
- Performance testing
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Type
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, inspect

from app.models.agent import Agent, AgentType, AgentStatus
from app.models.session import Session, SessionType, SessionStatus
from app.models.task import Task, TaskType, TaskStatus, TaskPriority
from app.models.workflow import Workflow, WorkflowStatus, WorkflowPriority
from app.models.context import Context, ContextType


class DatabaseTestUtils:
    """Utilities for database testing."""
    
    @staticmethod
    async def create_test_agent(
        session: AsyncSession,
        name: str = "Test Agent",
        agent_type: AgentType = AgentType.CLAUDE,
        **kwargs
    ) -> Agent:
        """Create a test agent with sensible defaults."""
        agent_data = {
            "name": name,
            "type": agent_type,
            "role": kwargs.get("role", "test_role"),
            "capabilities": kwargs.get("capabilities", [
                {
                    "name": "test_capability",
                    "description": "Test capability",
                    "confidence_level": 0.8,
                    "specialization_areas": ["testing"]
                }
            ]),
            "status": kwargs.get("status", AgentStatus.ACTIVE),
            "config": kwargs.get("config", {"test": True}),
            **kwargs
        }
        
        agent = Agent(**agent_data)
        session.add(agent)
        await session.flush()
        return agent
    
    @staticmethod
    async def create_test_session(
        db_session: AsyncSession,
        name: str = "Test Session",
        lead_agent: Optional[Agent] = None,
        **kwargs
    ) -> Session:
        """Create a test session with sensible defaults."""
        if lead_agent is None:
            lead_agent = await DatabaseTestUtils.create_test_agent(db_session)
        
        session_data = {
            "name": name,
            "description": kwargs.get("description", "Test session"),
            "session_type": kwargs.get("session_type", SessionType.FEATURE_DEVELOPMENT),
            "status": kwargs.get("status", SessionStatus.ACTIVE),
            "participant_agents": kwargs.get("participant_agents", [lead_agent.id]),
            "lead_agent_id": lead_agent.id,
            "objectives": kwargs.get("objectives", ["Test objective"]),
            **kwargs
        }
        
        session = Session(**session_data)
        db_session.add(session)
        await db_session.flush()
        return session
    
    @staticmethod
    async def create_test_task(
        session: AsyncSession,
        title: str = "Test Task",
        assigned_agent: Optional[Agent] = None,
        **kwargs
    ) -> Task:
        """Create a test task with sensible defaults."""
        if assigned_agent is None:
            assigned_agent = await DatabaseTestUtils.create_test_agent(session)
        
        task_data = {
            "title": title,
            "description": kwargs.get("description", "Test task description"),
            "task_type": kwargs.get("task_type", TaskType.FEATURE_DEVELOPMENT),
            "status": kwargs.get("status", TaskStatus.PENDING),
            "priority": kwargs.get("priority", TaskPriority.MEDIUM),
            "assigned_agent_id": assigned_agent.id,
            "required_capabilities": kwargs.get("required_capabilities", ["test_capability"]),
            "estimated_effort": kwargs.get("estimated_effort", 60),
            "context": kwargs.get("context", {"test": True}),
            **kwargs
        }
        
        task = Task(**task_data)
        session.add(task)
        await session.flush()
        return task
    
    @staticmethod
    async def create_test_workflow(
        session: AsyncSession,
        name: str = "Test Workflow",
        **kwargs
    ) -> Workflow:
        """Create a test workflow with sensible defaults."""
        workflow_data = {
            "name": name,
            "description": kwargs.get("description", "Test workflow"),
            "status": kwargs.get("status", WorkflowStatus.CREATED),
            "priority": kwargs.get("priority", WorkflowPriority.MEDIUM),
            "definition": kwargs.get("definition", {"type": "sequential", "steps": []}),
            "context": kwargs.get("context", {"test": True}),
            "variables": kwargs.get("variables", {}),
            "estimated_duration": kwargs.get("estimated_duration", 120),
            **kwargs
        }
        
        workflow = Workflow(**workflow_data)
        session.add(workflow)
        await session.flush()
        return workflow
    
    @staticmethod
    async def verify_database_schema(session: AsyncSession) -> Dict[str, Any]:
        """Verify database schema integrity."""
        inspector = inspect(session.bind)
        
        # Get table names
        tables = await session.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            if "postgresql" in str(session.bind.url)
            else "SELECT name FROM sqlite_master WHERE type='table'"
        ))
        
        table_names = [row[0] for row in tables.fetchall()]
        
        schema_info = {
            "tables": table_names,
            "table_count": len(table_names),
            "expected_tables": [
                "agents", "sessions", "tasks", "workflows", "contexts",
                "agent_events", "system_checkpoints", "conversations"
            ]
        }
        
        # Check for missing tables
        missing_tables = set(schema_info["expected_tables"]) - set(table_names)
        schema_info["missing_tables"] = list(missing_tables)
        schema_info["schema_valid"] = len(missing_tables) == 0
        
        return schema_info
    
    @staticmethod
    async def cleanup_test_data(session: AsyncSession, created_objects: List[Any]) -> None:
        """Clean up test data."""
        for obj in reversed(created_objects):  # Reverse order for FK constraints
            try:
                await session.delete(obj)
            except Exception as e:
                print(f"Warning: Failed to delete {obj}: {e}")
        
        try:
            await session.flush()
        except Exception as e:
            print(f"Warning: Failed to flush deletions: {e}")


class PerformanceTestUtils:
    """Utilities for database performance testing."""
    
    @staticmethod
    async def measure_query_time(session: AsyncSession, query_func) -> float:
        """Measure query execution time in milliseconds."""
        start_time = asyncio.get_event_loop().time()
        await query_func(session)
        end_time = asyncio.get_event_loop().time()
        return (end_time - start_time) * 1000  # Convert to ms
    
    @staticmethod
    async def create_bulk_test_data(
        session: AsyncSession, 
        count: int = 1000
    ) -> Dict[str, List[Any]]:
        """Create bulk test data for performance testing."""
        agents = []
        sessions = []
        tasks = []
        
        # Create agents
        for i in range(count // 10):  # 10% agents
            agent = Agent(
                name=f"Perf Agent {i}",
                type=AgentType.CLAUDE,
                role=f"test_role_{i}",
                capabilities=[{
                    "name": f"capability_{i}",
                    "confidence_level": 0.8
                }],
                status=AgentStatus.ACTIVE
            )
            agents.append(agent)
            session.add(agent)
        
        await session.flush()  # Get agent IDs
        
        # Create sessions
        for i in range(count // 5):  # 20% sessions
            test_session = Session(
                name=f"Perf Session {i}",
                session_type=SessionType.FEATURE_DEVELOPMENT,
                status=SessionStatus.ACTIVE,
                lead_agent_id=agents[i % len(agents)].id,
                participant_agents=[agents[i % len(agents)].id]
            )
            sessions.append(test_session)
            session.add(test_session)
        
        await session.flush()
        
        # Create tasks
        for i in range(count):  # 100% tasks
            task = Task(
                title=f"Perf Task {i}",
                description=f"Performance test task {i}",
                task_type=TaskType.FEATURE_DEVELOPMENT,
                status=TaskStatus.PENDING,
                priority=TaskPriority.MEDIUM,
                assigned_agent_id=agents[i % len(agents)].id,
                estimated_effort=60
            )
            tasks.append(task)
            session.add(task)
        
        await session.flush()
        
        return {
            "agents": agents,
            "sessions": sessions,
            "tasks": tasks
        }
