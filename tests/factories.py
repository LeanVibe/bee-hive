"""
Test Data Factories for LeanVibe Agent Hive 2.0

Provides consistent, realistic test data generation for all test scenarios.
Includes factories for agents, sessions, tasks, contexts, and performance metrics.
"""

import uuid
import random
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from app.models.agent import Agent, AgentStatus, AgentType
from app.models.session import Session, SessionStatus, SessionType
from app.models.task import Task, TaskStatus, TaskPriority, TaskType
from app.models.context import Context, ContextType
from app.schemas.agent import AgentCreate


@dataclass
class TestDataConfig:
    """Configuration for test data generation."""
    seed: int = 42
    realistic_delays: bool = True
    include_edge_cases: bool = True
    performance_mode: bool = False  # Faster generation for performance tests


class AgentFactory:
    """Factory for creating test agents with realistic data."""
    
    NAMES = [
        "Claude Senior", "GPT Assistant", "Gemini Pro", "Custom Bot",
        "QA Specialist", "Performance Tester", "Integration Worker", 
        "Context Manager", "Sleep Coordinator", "Recovery Agent"
    ]
    
    ROLES = [
        "orchestrator", "worker", "specialist", "coordinator", "monitor",
        "validator", "optimizer", "analyzer", "reporter", "cleaner"
    ]
    
    CAPABILITY_TEMPLATES = [
        {
            "name": "python_development",
            "description": "Python programming and development",
            "confidence_level": 0.9,
            "specialization_areas": ["backend", "api", "automation"]
        },
        {
            "name": "quality_assurance", 
            "description": "Testing and quality validation",
            "confidence_level": 0.85,
            "specialization_areas": ["testing", "validation", "performance"]
        },
        {
            "name": "data_processing",
            "description": "Data analysis and processing",
            "confidence_level": 0.8,
            "specialization_areas": ["analytics", "etl", "ml"]
        },
        {
            "name": "system_monitoring",
            "description": "System health and performance monitoring",
            "confidence_level": 0.95,
            "specialization_areas": ["observability", "metrics", "alerting"]
        },
        {
            "name": "context_management",
            "description": "Context storage and retrieval optimization",
            "confidence_level": 0.88,
            "specialization_areas": ["memory", "compression", "search"]
        }
    ]
    
    @classmethod
    def create_agent(
        cls, 
        name: Optional[str] = None,
        agent_type: Optional[AgentType] = None,
        status: Optional[AgentStatus] = None,
        capabilities: Optional[List[Dict]] = None,
        config: Optional[Dict] = None
    ) -> Agent:
        """Create a test agent with realistic data."""
        
        agent = Agent(
            name=name or random.choice(cls.NAMES),
            type=agent_type or random.choice(list(AgentType)),
            role=random.choice(cls.ROLES),
            capabilities=capabilities or random.sample(cls.CAPABILITY_TEMPLATES, k=random.randint(1, 3)),
            status=status or random.choice(list(AgentStatus)),
            config=config or {
                "max_context_tokens": random.randint(4000, 8000),
                "response_timeout": random.randint(30, 120),
                "retry_attempts": random.randint(2, 5)
            },
            context_window_usage=str(random.uniform(0.1, 0.9)),
            total_tasks_completed=str(random.randint(0, 1000)),
            total_tasks_failed=str(random.randint(0, 50)),
            average_response_time=str(random.uniform(0.5, 5.0))
        )
        
        # Set realistic timestamps
        created_time = datetime.utcnow() - timedelta(days=random.randint(1, 30))
        agent.created_at = created_time
        agent.updated_at = created_time + timedelta(hours=random.randint(1, 24))
        
        if status == AgentStatus.active:
            agent.last_heartbeat = datetime.utcnow() - timedelta(minutes=random.randint(1, 10))
            agent.last_active = datetime.utcnow() - timedelta(minutes=random.randint(1, 30))
        
        return agent
    
    @classmethod
    def create_agent_schema(
        cls,
        name: Optional[str] = None,
        agent_type: Optional[AgentType] = None,
        capabilities: Optional[List[Dict]] = None
    ) -> AgentCreate:
        """Create an AgentCreate schema for API testing."""
        
        return AgentCreate(
            name=name or random.choice(cls.NAMES),
            type=agent_type or random.choice(list(AgentType)),
            role=random.choice(cls.ROLES),
            capabilities=capabilities or random.sample(cls.CAPABILITY_TEMPLATES, k=random.randint(1, 2)),
            system_prompt=f"You are a {random.choice(cls.ROLES)} agent specializing in test scenarios.",
            config={
                "test_mode": True,
                "max_context_tokens": random.randint(4000, 8000)
            }
        )
    
    @classmethod
    def create_high_performance_agent(cls) -> Agent:
        """Create an agent optimized for performance testing."""
        
        return Agent(
            name="Performance Test Agent",
            type=AgentType.CLAUDE,
            role="performance_tester",
            capabilities=[{
                "name": "performance_validation",
                "description": "High-speed performance testing",
                "confidence_level": 0.95,
                "specialization_areas": ["speed", "throughput", "latency"]
            }],
            status=AgentStatus.active,
            context_window_usage="0.5",
            config={"performance_optimized": True}
        )
    
    @classmethod
    def create_error_prone_agent(cls) -> Agent:
        """Create an agent designed for error testing."""
        
        return Agent(
            name="Error Test Agent",
            type=AgentType.CUSTOM,
            role="error_generator",
            capabilities=[{
                "name": "error_simulation",
                "description": "Simulates various error conditions",
                "confidence_level": 0.3,  # Low confidence for error testing
                "specialization_areas": ["failures", "timeouts", "exceptions"]
            }],
            status=AgentStatus.ERROR,
            context_window_usage="0.95",  # High usage for edge case testing
            config={"error_simulation": True, "failure_rate": 0.3}
        )


class SessionFactory:
    """Factory for creating test sessions."""
    
    SESSION_NAMES = [
        "Performance Validation Session",
        "Error Handling Test Session", 
        "Integration Test Workflow",
        "Load Testing Session",
        "Quality Assurance Session"
    ]
    
    @classmethod
    def create_session(
        cls,
        name: Optional[str] = None,
        session_type: Optional[SessionType] = None,
        status: Optional[SessionStatus] = None,
        participant_count: int = 3
    ) -> Session:
        """Create a test session with realistic data."""
        
        # Create participant agent IDs
        participant_agents = [str(uuid.uuid4()) for _ in range(participant_count)]
        
        session = Session(
            name=name or random.choice(cls.SESSION_NAMES),
            description=f"Test session for validation with {participant_count} agents",
            session_type=session_type or random.choice(list(SessionType)),
            status=status or random.choice(list(SessionStatus)),
            participant_agents=participant_agents,
            lead_agent_id=participant_agents[0] if participant_agents else None,
            objectives=[
                f"Objective {i+1}: Test scenario validation"
                for i in range(random.randint(2, 5))
            ],
            context={"test_session": True, "validation_mode": True},
            max_participants=participant_count + random.randint(1, 3)
        )
        
        # Set realistic timestamps
        created_time = datetime.utcnow() - timedelta(hours=random.randint(1, 48))
        session.created_at = created_time
        session.updated_at = created_time + timedelta(minutes=random.randint(10, 120))
        
        if status == SessionStatus.ACTIVE:
            session.last_activity = datetime.utcnow() - timedelta(minutes=random.randint(1, 30))
        
        return session


class TaskFactory:
    """Factory for creating test tasks."""
    
    TASK_TITLES = [
        "Validate Performance Metrics",
        "Execute Integration Tests", 
        "Process Context Compression",
        "Monitor System Health",
        "Generate Test Report",
        "Optimize Database Queries",
        "Validate Error Handling",
        "Compress Historical Data"
    ]
    
    @classmethod
    def create_task(
        cls,
        title: Optional[str] = None,
        task_type: Optional[TaskType] = None,
        status: Optional[TaskStatus] = None,
        priority: Optional[TaskPriority] = None,
        assigned_agent_id: Optional[str] = None
    ) -> Task:
        """Create a test task with realistic data."""
        
        task = Task(
            title=title or random.choice(cls.TASK_TITLES),
            description=f"Test task for validation - {random.choice(['high', 'medium', 'low'])} complexity",
            task_type=task_type or random.choice(list(TaskType)),
            status=status or random.choice(list(TaskStatus)),
            priority=priority or random.choice(list(TaskPriority)),
            assigned_agent_id=assigned_agent_id or str(uuid.uuid4()),
            required_capabilities=random.sample([
                "python_development", "quality_assurance", "data_processing",
                "system_monitoring", "context_management"
            ], k=random.randint(1, 3)),
            estimated_effort=random.randint(15, 240),  # 15 minutes to 4 hours
            context={
                "test_task": True,
                "complexity": random.choice(["low", "medium", "high"]),
                "validation_required": True
            },
            dependencies=[]
        )
        
        # Set realistic timestamps based on status
        created_time = datetime.utcnow() - timedelta(hours=random.randint(1, 72))
        task.created_at = created_time
        task.updated_at = created_time + timedelta(minutes=random.randint(5, 60))
        
        if status in [TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED]:
            task.started_at = created_time + timedelta(minutes=random.randint(5, 30))
            
        if status == TaskStatus.COMPLETED:
            task.completed_at = task.started_at + timedelta(minutes=random.randint(10, 120))
        
        return task
    
    @classmethod
    def create_performance_task(cls) -> Task:
        """Create a task optimized for performance testing."""
        
        return Task(
            title="Performance Benchmark Task",
            description="High-speed task for performance validation",
            task_type=TaskType.SYSTEM_TASK,
            status=TaskStatus.PENDING,
            priority=TaskPriority.HIGH,
            required_capabilities=["performance_validation"],
            estimated_effort=5,  # Very quick for performance testing
            context={"performance_test": True, "benchmark": True}
        )
    
    @classmethod
    def create_complex_task_chain(cls, chain_length: int = 5) -> List[Task]:
        """Create a chain of dependent tasks for integration testing."""
        
        tasks = []
        for i in range(chain_length):
            task = cls.create_task(
                title=f"Chain Task {i+1}",
                task_type=TaskType.FEATURE_DEVELOPMENT,
                status=TaskStatus.PENDING if i == 0 else TaskStatus.BLOCKED
            )
            
            # Set up dependencies
            if i > 0:
                task.dependencies = [tasks[i-1].id] if hasattr(tasks[i-1], 'id') else []
            
            tasks.append(task)
        
        return tasks


class ContextFactory:
    """Factory for creating test contexts."""
    
    CONTEXT_CONTENTS = [
        "Performance test execution completed successfully with 95% accuracy",
        "Error handling validation shows robust recovery mechanisms in place",
        "Integration test suite covers all critical system workflows",
        "Database optimization resulted in 40% query speed improvement", 
        "Context compression achieved 75% token reduction while maintaining accuracy",
        "System monitoring detects no performance degradation under load",
        "Quality assurance metrics indicate production readiness",
        "Sleep-wake cycle optimization improved memory efficiency by 60%"
    ]
    
    @classmethod
    def create_context(
        cls,
        content: Optional[str] = None,
        context_type: Optional[ContextType] = None,
        metadata: Optional[Dict] = None
    ) -> Context:
        """Create a test context with realistic data."""
        
        context = Context(
            content=content or random.choice(cls.CONTEXT_CONTENTS),
            context_type=context_type or random.choice(list(ContextType)),
            embedding_vector=[random.uniform(-1, 1) for _ in range(1536)],  # OpenAI embedding size
            metadata=metadata or {
                "source": "test_generator",
                "confidence": random.uniform(0.7, 0.95),
                "tags": random.sample(["test", "validation", "performance", "quality"], k=2)
            },
            token_count=random.randint(50, 500),
            compression_ratio=random.uniform(0.6, 0.8)
        )
        
        # Set realistic timestamps
        created_time = datetime.utcnow() - timedelta(days=random.randint(1, 14))
        context.created_at = created_time
        context.updated_at = created_time + timedelta(hours=random.randint(1, 12))
        context.last_accessed = datetime.utcnow() - timedelta(hours=random.randint(1, 48))
        
        return context
    
    @classmethod
    def create_performance_contexts(cls, count: int = 100) -> List[Context]:
        """Create multiple contexts for performance testing."""
        
        contexts = []
        for i in range(count):
            context = Context(
                content=f"Performance test context {i+1} with standardized content for benchmarking",
                context_type=ContextType.CONVERSATION,
                embedding_vector=[0.1] * 1536,  # Simplified for performance
                metadata={"performance_test": True, "index": i},
                token_count=100  # Standardized size
            )
            contexts.append(context)
        
        return contexts


class MessageFactory:
    """Factory for creating test messages for Redis streams."""
    
    MESSAGE_TYPES = [
        "task_assignment", "task_completion", "heartbeat", "error_report",
        "performance_update", "context_request", "sleep_notification", "wake_notification"
    ]
    
    @classmethod
    def create_message_data(
        cls,
        from_agent: Optional[str] = None,
        to_agent: Optional[str] = None,
        message_type: Optional[str] = None,
        payload: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create message data for testing Redis streams."""
        
        return {
            "message_id": str(uuid.uuid4()),
            "from_agent": from_agent or f"agent_{random.randint(1, 100)}",
            "to_agent": to_agent or f"agent_{random.randint(1, 100)}",
            "type": message_type or random.choice(cls.MESSAGE_TYPES),
            "payload": payload or {
                "test_message": True,
                "timestamp": time.time(),
                "sequence": random.randint(1, 1000),
                "data": f"Test payload data {random.randint(1, 100)}"
            },
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @classmethod
    def create_performance_messages(cls, count: int = 1000) -> List[Dict[str, Any]]:
        """Create standardized messages for performance testing."""
        
        messages = []
        for i in range(count):
            message = {
                "message_id": f"perf_msg_{i}",
                "from_agent": f"sender_{i % 10}",
                "to_agent": f"receiver_{i % 5}",
                "type": "performance_test",
                "payload": {"sequence": i, "test_data": f"data_{i}"},
                "correlation_id": f"perf_corr_{i}",
                "timestamp": datetime.utcnow().isoformat()
            }
            messages.append(message)
        
        return messages


class PerformanceMetricFactory:
    """Factory for creating realistic performance metrics."""
    
    @classmethod
    def create_baseline_metrics(cls) -> Dict[str, Any]:
        """Create baseline performance metrics for comparison."""
        
        return {
            "context_retrieval_ms": random.uniform(30, 50),
            "message_delivery_success_rate": random.uniform(99.9, 100.0),
            "message_p95_latency_ms": random.uniform(150, 200),
            "agent_operation_ms": random.uniform(300, 500),
            "sleep_wake_recovery_ms": random.uniform(45000, 60000),
            "api_response_ms": random.uniform(100, 200),
            "database_query_ms": random.uniform(50, 100),
            "concurrent_agents_supported": random.randint(50, 100),
            "message_throughput_per_sec": random.randint(1000, 2000),
            "memory_usage_mb": random.uniform(200, 500),
            "cpu_utilization_percent": random.uniform(20, 60),
            "error_rate_percent": random.uniform(0.1, 1.0)
        }
    
    @classmethod
    def create_performance_degraded_metrics(cls) -> Dict[str, Any]:
        """Create metrics showing performance degradation for testing."""
        
        return {
            "context_retrieval_ms": random.uniform(80, 120),  # Above target
            "message_delivery_success_rate": random.uniform(98.0, 99.5),  # Below target
            "message_p95_latency_ms": random.uniform(250, 400),  # Above target
            "agent_operation_ms": random.uniform(600, 1000),  # Above target
            "sleep_wake_recovery_ms": random.uniform(70000, 90000),  # Above target
            "api_response_ms": random.uniform(300, 500),  # Above target
            "database_query_ms": random.uniform(150, 250),  # Above target
            "concurrent_agents_supported": random.randint(30, 45),  # Below target
            "message_throughput_per_sec": random.randint(500, 800),  # Below target
            "memory_usage_mb": random.uniform(600, 1000),  # Higher usage
            "cpu_utilization_percent": random.uniform(80, 95),  # High utilization
            "error_rate_percent": random.uniform(2.0, 5.0)  # High error rate
        }


class TestScenarioFactory:
    """Factory for creating complete test scenarios."""
    
    @classmethod
    def create_integration_scenario(cls) -> Dict[str, Any]:
        """Create a complete integration test scenario."""
        
        # Create agents
        orchestrator = AgentFactory.create_agent(
            name="Orchestrator Agent",
            agent_type=AgentType.CLAUDE,
            status=AgentStatus.active
        )
        
        workers = [
            AgentFactory.create_agent(
                name=f"Worker Agent {i+1}",
                status=AgentStatus.active
            ) for i in range(3)
        ]
        
        # Create session
        session = SessionFactory.create_session(
            name="Integration Test Session",
            session_type=SessionType.COORDINATION,
            status=SessionStatus.ACTIVE,
            participant_count=4
        )
        
        # Create task chain
        tasks = TaskFactory.create_complex_task_chain(chain_length=5)
        
        # Create contexts
        contexts = ContextFactory.create_performance_contexts(count=20)
        
        # Create messages
        messages = MessageFactory.create_performance_messages(count=50)
        
        return {
            "orchestrator": orchestrator,
            "workers": workers,
            "session": session,
            "tasks": tasks,
            "contexts": contexts,
            "messages": messages,
            "expected_outcomes": {
                "all_tasks_completed": True,
                "message_success_rate": 100.0,
                "context_retrieval_accuracy": 95.0,
                "total_duration_max_minutes": 30
            }
        }
    
    @classmethod
    def create_performance_scenario(cls) -> Dict[str, Any]:
        """Create a performance testing scenario."""
        
        return {
            "agents": [AgentFactory.create_high_performance_agent() for _ in range(50)],
            "tasks": [TaskFactory.create_performance_task() for _ in range(100)],
            "contexts": ContextFactory.create_performance_contexts(count=1000),
            "messages": MessageFactory.create_performance_messages(count=5000),
            "performance_targets": PerformanceMetricFactory.create_baseline_metrics(),
            "test_duration_seconds": 300,  # 5 minutes
            "expected_throughput": 1000,
            "max_error_rate": 0.1
        }
    
    @classmethod
    def create_error_scenario(cls) -> Dict[str, Any]:
        """Create an error handling test scenario."""
        
        return {
            "stable_agents": [AgentFactory.create_agent(status=AgentStatus.active) for _ in range(5)],
            "error_agents": [AgentFactory.create_error_prone_agent() for _ in range(2)],
            "error_conditions": [
                "database_timeout",
                "redis_connection_failure", 
                "memory_pressure",
                "network_partition",
                "service_overload"
            ],
            "recovery_targets": {
                "max_recovery_time_seconds": 60,
                "min_service_availability_percent": 95,
                "max_data_loss_percent": 0
            },
            "test_scenarios": [
                "gradual_degradation",
                "sudden_failure",
                "cascading_failures",
                "recovery_validation"
            ]
        }


# Utility functions for test data management
def seed_random_data(seed: int = 42) -> None:
    """Seed random generators for reproducible test data."""
    random.seed(seed)


def reset_test_data() -> None:
    """Reset all factories to initial state."""
    seed_random_data()


def get_test_config(performance_mode: bool = False) -> TestDataConfig:
    """Get test configuration for different testing modes."""
    return TestDataConfig(
        performance_mode=performance_mode,
        realistic_delays=not performance_mode,
        include_edge_cases=not performance_mode
    )


# Example usage and validation
if __name__ == "__main__":
    # Demonstrate factory usage
    print("LeanVibe Agent Hive 2.0 - Test Data Factories")
    print("=" * 50)
    
    # Create sample test data
    agent = AgentFactory.create_agent()
    print(f"Sample Agent: {agent.name} ({agent.type.value}) - {agent.status.value}")
    
    session = SessionFactory.create_session()
    print(f"Sample Session: {session.name} - {len(session.participant_agents)} participants")
    
    task = TaskFactory.create_task()
    print(f"Sample Task: {task.title} ({task.priority.value})")
    
    context = ContextFactory.create_context()
    print(f"Sample Context: {context.content[:50]}... ({context.token_count} tokens)")
    
    message = MessageFactory.create_message_data()
    print(f"Sample Message: {message['type']} from {message['from_agent']} to {message['to_agent']}")
    
    metrics = PerformanceMetricFactory.create_baseline_metrics()
    print(f"Sample Metrics: Context retrieval {metrics['context_retrieval_ms']:.1f}ms")
    
    print("\n✅ All factories working correctly!")