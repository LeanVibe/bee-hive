"""
Mock Observability Events Stream Service for LeanVibe Agent Hive 2.0

Generates realistic multi-agent workflow events for testing and development.
Provides comprehensive event streams simulating intelligent agent coordination patterns.
"""

import asyncio
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generator
from enum import Enum
import structlog

from app.schemas.observability import (
    WorkflowStartedEvent,
    WorkflowEndedEvent,
    NodeExecutingEvent,
    NodeCompletedEvent,
    AgentStateChangedEvent,
    AgentCapabilityUtilizedEvent,
    PreToolUseEvent,
    PostToolUseEvent,
    SemanticQueryEvent,
    SemanticUpdateEvent,
    MessagePublishedEvent,
    MessageReceivedEvent,
    FailureDetectedEvent,
    RecoveryInitiatedEvent,
    SystemHealthCheckEvent,
    EventCategory,
    PerformanceMetrics,
    EventMetadata,
)
from app.core.event_serialization import serialize_for_stream
from app.core.redis import get_redis

logger = structlog.get_logger()


class WorkflowScenario(str, Enum):
    """Predefined workflow scenarios for realistic simulation."""
    CODE_REVIEW = "code_review"
    FEATURE_DEVELOPMENT = "feature_development"
    BUG_INVESTIGATION = "bug_investigation"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SECURITY_AUDIT = "security_audit"
    DOCUMENTATION_UPDATE = "documentation_update"
    DEPLOYMENT_PIPELINE = "deployment_pipeline"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"


class AgentPersona(str, Enum):
    """Agent personas with different capabilities and behaviors."""
    SENIOR_DEVELOPER = "senior_developer"
    PERFORMANCE_SPECIALIST = "performance_specialist"
    SECURITY_ANALYST = "security_analyst"
    DOCUMENTATION_WRITER = "documentation_writer"
    QA_ENGINEER = "qa_engineer"
    ORCHESTRATOR = "orchestrator"
    ANALYTICS_AGENT = "analytics_agent"
    MEMORY_MANAGER = "memory_manager"


class MockEventGenerator:
    """Generates realistic observability events for multi-agent workflows."""
    
    def __init__(self):
        """Initialize the mock event generator."""
        self.active_workflows: Dict[uuid.UUID, Dict[str, Any]] = {}
        self.active_agents: Dict[uuid.UUID, Dict[str, Any]] = {}
        self.session_context: Dict[uuid.UUID, Dict[str, Any]] = {}
        
        # Realistic tool names and parameters
        self.common_tools = [
            "Read", "Write", "Edit", "MultiEdit", "Bash", "Glob", "Grep",
            "LS", "WebFetch", "WebSearch", "TodoWrite", "NotebookRead",
            "NotebookEdit", "ExitPlanMode"
        ]
        
        # Semantic query templates
        self.query_templates = [
            "How to implement authentication in Python FastAPI",
            "Best practices for Redis Streams performance optimization",
            "Error handling patterns in multi-agent systems",
            "Database migration strategies with zero downtime",
            "Testing approaches for asynchronous workflows",
            "Memory management in high-throughput applications",
            "Security considerations for AI agent communication",
            "Monitoring and observability in distributed systems"
        ]
        
        logger.info("Mock event generator initialized")
    
    def generate_workflow_scenario(self, scenario: WorkflowScenario) -> Generator[Dict[str, Any], None, None]:
        """Generate events for a complete workflow scenario."""
        workflow_id = uuid.uuid4()
        session_id = uuid.uuid4()
        
        # Create workflow definition based on scenario
        workflow_def = self._create_workflow_definition(scenario)
        
        # Generate workflow started event
        yield self._create_workflow_started_event(
            workflow_id=workflow_id,
            session_id=session_id,
            scenario=scenario,
            definition=workflow_def
        )
        
        # Generate agent events
        agents = self._create_scenario_agents(scenario)
        for agent in agents:
            agent_id = agent['id']
            self.active_agents[agent_id] = agent
            
            # Agent state changes
            yield self._create_agent_state_changed_event(
                agent_id=agent_id,
                session_id=session_id,
                previous_state="idle",
                new_state="active",
                reason=f"Assigned to workflow {scenario.value}"
            )
        
        # Generate task execution events
        tasks = workflow_def.get('tasks', [])
        completed_tasks = []
        
        for task in tasks:
            task_id = task['id']
            assigned_agent = random.choice(agents)
            agent_id = assigned_agent['id']
            
            # Node executing event
            yield self._create_node_executing_event(
                workflow_id=workflow_id,
                session_id=session_id,
                agent_id=agent_id,
                node_id=task_id,
                node_type=task['type'],
                node_name=task['name']
            )
            
            # Generate tool usage events for this task
            tools_used = self._select_tools_for_task(task['type'])
            for tool_name in tools_used:
                # Pre-tool use
                correlation_id = uuid.uuid4()
                yield self._create_pre_tool_use_event(
                    agent_id=agent_id,
                    session_id=session_id,
                    tool_name=tool_name,
                    correlation_id=correlation_id
                )
                
                # Simulate tool execution delay
                execution_time = random.randint(100, 2000)
                
                # Post-tool use
                yield self._create_post_tool_use_event(
                    agent_id=agent_id,
                    session_id=session_id,
                    tool_name=tool_name,
                    correlation_id=correlation_id,
                    execution_time_ms=execution_time,
                    success=random.random() > 0.05  # 95% success rate
                )
            
            # Generate semantic memory events
            if random.random() > 0.5:  # 50% chance of memory operations
                yield self._create_semantic_query_event(
                    agent_id=agent_id,
                    session_id=session_id,
                    context_id=uuid.uuid4()
                )
                
                if random.random() > 0.3:  # 70% chance of memory update after query
                    yield self._create_semantic_update_event(
                        agent_id=agent_id,
                        session_id=session_id,
                        context_id=uuid.uuid4()
                    )
            
            # Generate communication events
            if len(agents) > 1 and random.random() > 0.4:  # 60% chance of inter-agent communication
                target_agent = random.choice([a for a in agents if a['id'] != agent_id])
                message_id = uuid.uuid4()
                
                # Message published
                yield self._create_message_published_event(
                    from_agent=str(agent_id),
                    to_agent=str(target_agent['id']),
                    message_id=message_id,
                    session_id=session_id
                )
                
                # Message received (with slight delay simulation)
                yield self._create_message_received_event(
                    from_agent=str(agent_id),
                    message_id=message_id,
                    session_id=session_id
                )
            
            # Task completion with possible failure
            task_success = random.random() > 0.1  # 90% success rate
            if not task_success and random.random() > 0.5:  # 50% chance of failure detection/recovery
                # Generate failure event
                yield self._create_failure_detected_event(
                    agent_id=agent_id,
                    session_id=session_id,
                    affected_component=f"Task {task_id}"
                )
                
                # Generate recovery event
                yield self._create_recovery_initiated_event(
                    agent_id=agent_id,
                    session_id=session_id,
                    trigger_failure=f"Task {task_id} execution failure"
                )
            
            # Node completed event
            yield self._create_node_completed_event(
                workflow_id=workflow_id,
                session_id=session_id,
                node_id=task_id,
                success=task_success,
                agent_id=agent_id
            )
            
            if task_success:
                completed_tasks.append(task_id)
        
        # Generate system health checks periodically
        if random.random() > 0.7:  # 30% chance of health check
            yield self._create_system_health_check_event(session_id=session_id)
        
        # Workflow completion
        workflow_success = len(completed_tasks) >= len(tasks) * 0.8  # 80% task completion threshold
        yield self._create_workflow_ended_event(
            workflow_id=workflow_id,
            session_id=session_id,
            status="completed" if workflow_success else "failed",
            total_tasks=len(tasks),
            completed_tasks=len(completed_tasks)
        )
        
        # Agent cleanup
        for agent in agents:
            yield self._create_agent_state_changed_event(
                agent_id=agent['id'],
                session_id=session_id,
                previous_state="active",
                new_state="idle",
                reason="Workflow completed"
            )
    
    def _create_workflow_definition(self, scenario: WorkflowScenario) -> Dict[str, Any]:
        """Create realistic workflow definition for scenario."""
        base_tasks = {
            WorkflowScenario.CODE_REVIEW: [
                {"id": "fetch_pr", "name": "Fetch Pull Request", "type": "git_operation"},
                {"id": "analyze_changes", "name": "Analyze Code Changes", "type": "analysis"},
                {"id": "run_tests", "name": "Run Test Suite", "type": "testing"},
                {"id": "security_scan", "name": "Security Vulnerability Scan", "type": "security"},
                {"id": "generate_review", "name": "Generate Review Comments", "type": "generation"},
                {"id": "submit_review", "name": "Submit Review", "type": "git_operation"}
            ],
            WorkflowScenario.FEATURE_DEVELOPMENT: [
                {"id": "requirement_analysis", "name": "Analyze Requirements", "type": "analysis"},
                {"id": "design_planning", "name": "Create Technical Design", "type": "planning"},
                {"id": "implementation", "name": "Implement Feature", "type": "coding"},
                {"id": "unit_testing", "name": "Write Unit Tests", "type": "testing"},
                {"id": "integration_testing", "name": "Integration Testing", "type": "testing"},
                {"id": "documentation", "name": "Update Documentation", "type": "documentation"}
            ],
            WorkflowScenario.BUG_INVESTIGATION: [
                {"id": "reproduce_bug", "name": "Reproduce Bug", "type": "debugging"},
                {"id": "log_analysis", "name": "Analyze System Logs", "type": "analysis"},
                {"id": "root_cause_analysis", "name": "Root Cause Analysis", "type": "investigation"},
                {"id": "fix_implementation", "name": "Implement Fix", "type": "coding"},
                {"id": "regression_testing", "name": "Regression Testing", "type": "testing"},
                {"id": "deployment", "name": "Deploy Fix", "type": "deployment"}
            ]
        }
        
        tasks = base_tasks.get(scenario, base_tasks[WorkflowScenario.FEATURE_DEVELOPMENT])
        
        return {
            "scenario": scenario.value,
            "tasks": tasks,
            "estimated_duration_minutes": len(tasks) * random.randint(15, 45),
            "priority": random.choice(["low", "medium", "high", "critical"]),
            "complexity": random.choice(["simple", "moderate", "complex", "advanced"])
        }
    
    def _create_scenario_agents(self, scenario: WorkflowScenario) -> List[Dict[str, Any]]:
        """Create agents appropriate for the scenario."""
        agent_assignments = {
            WorkflowScenario.CODE_REVIEW: [
                AgentPersona.SENIOR_DEVELOPER,
                AgentPersona.SECURITY_ANALYST,
                AgentPersona.QA_ENGINEER
            ],
            WorkflowScenario.FEATURE_DEVELOPMENT: [
                AgentPersona.SENIOR_DEVELOPER,
                AgentPersona.ORCHESTRATOR,
                AgentPersona.DOCUMENTATION_WRITER
            ],
            WorkflowScenario.BUG_INVESTIGATION: [
                AgentPersona.SENIOR_DEVELOPER,
                AgentPersona.PERFORMANCE_SPECIALIST,
                AgentPersona.ANALYTICS_AGENT
            ]
        }
        
        personas = agent_assignments.get(scenario, [AgentPersona.SENIOR_DEVELOPER, AgentPersona.ORCHESTRATOR])
        
        agents = []
        for persona in personas:
            agent = {
                "id": uuid.uuid4(),
                "persona": persona.value,
                "capabilities": self._get_agent_capabilities(persona),
                "state": "idle",
                "created_at": datetime.utcnow()
            }
            agents.append(agent)
        
        return agents
    
    def _get_agent_capabilities(self, persona: AgentPersona) -> List[str]:
        """Get capabilities for agent persona."""
        capability_map = {
            AgentPersona.SENIOR_DEVELOPER: ["coding", "code_review", "debugging", "architecture"],
            AgentPersona.PERFORMANCE_SPECIALIST: ["performance_analysis", "optimization", "profiling"],
            AgentPersona.SECURITY_ANALYST: ["security_scanning", "vulnerability_assessment", "compliance"],
            AgentPersona.DOCUMENTATION_WRITER: ["documentation", "technical_writing", "knowledge_management"],
            AgentPersona.QA_ENGINEER: ["testing", "quality_assurance", "test_automation"],
            AgentPersona.ORCHESTRATOR: ["workflow_management", "coordination", "scheduling"],
            AgentPersona.ANALYTICS_AGENT: ["data_analysis", "pattern_recognition", "reporting"],
            AgentPersona.MEMORY_MANAGER: ["semantic_search", "knowledge_extraction", "context_management"]
        }
        
        return capability_map.get(persona, ["general"])
    
    def _select_tools_for_task(self, task_type: str) -> List[str]:
        """Select appropriate tools for task type."""
        tool_mappings = {
            "git_operation": ["Bash", "Read", "Write"],
            "analysis": ["Read", "Grep", "Glob", "WebSearch"],
            "testing": ["Bash", "Read", "LS"],
            "security": ["Bash", "Grep", "WebFetch"],
            "generation": ["Write", "Edit", "MultiEdit"],
            "coding": ["Read", "Write", "Edit", "MultiEdit", "Bash"],
            "planning": ["TodoWrite", "Write", "WebSearch"],
            "debugging": ["Read", "Bash", "Grep", "LS"],
            "investigation": ["Grep", "Glob", "Read", "WebSearch"],
            "deployment": ["Bash", "Read", "LS"],
            "documentation": ["Write", "Edit", "Read", "WebFetch"]
        }
        
        available_tools = tool_mappings.get(task_type, ["Read", "Write", "Bash"])
        return random.sample(available_tools, min(len(available_tools), random.randint(1, 4)))
    
    def _create_workflow_started_event(
        self,
        workflow_id: uuid.UUID,
        session_id: uuid.UUID,
        scenario: WorkflowScenario,
        definition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create workflow started event."""
        return WorkflowStartedEvent(
            workflow_id=workflow_id,
            session_id=session_id,
            workflow_name=f"{scenario.value.replace('_', ' ').title()} Workflow",
            workflow_definition=definition,
            initial_context={"scenario": scenario.value},
            estimated_duration_ms=definition["estimated_duration_minutes"] * 60 * 1000,
            priority=definition["priority"],
            performance_metrics=PerformanceMetrics(
                execution_time_ms=random.uniform(1.0, 3.0),
                memory_usage_mb=random.uniform(10.0, 50.0),
                cpu_usage_percent=random.uniform(5.0, 15.0)
            ),
            metadata=EventMetadata(
                correlation_id=uuid.uuid4(),
                source_service="mock_generator"
            )
        ).dict()
    
    def _create_workflow_ended_event(
        self,
        workflow_id: uuid.UUID,
        session_id: uuid.UUID,
        status: str,
        total_tasks: int,
        completed_tasks: int
    ) -> Dict[str, Any]:
        """Create workflow ended event."""
        return WorkflowEndedEvent(
            workflow_id=workflow_id,
            session_id=session_id,
            status=status,
            completion_reason=f"Workflow {status} with {completed_tasks}/{total_tasks} tasks completed",
            final_result={"success": status == "completed"},
            total_tasks_executed=total_tasks,
            failed_tasks=total_tasks - completed_tasks,
            performance_metrics=PerformanceMetrics(
                execution_time_ms=random.uniform(2.0, 5.0),
                memory_usage_mb=random.uniform(15.0, 75.0),
                cpu_usage_percent=random.uniform(10.0, 25.0)
            ),
            metadata=EventMetadata(
                correlation_id=uuid.uuid4(),
                source_service="mock_generator"
            )
        ).dict()
    
    def _create_node_executing_event(
        self,
        workflow_id: uuid.UUID,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        node_id: str,
        node_type: str,
        node_name: str
    ) -> Dict[str, Any]:
        """Create node executing event."""
        return NodeExecutingEvent(
            workflow_id=workflow_id,
            session_id=session_id,
            agent_id=agent_id,
            node_id=node_id,
            node_type=node_type,
            node_name=node_name,
            input_data={"task_context": f"Executing {node_name}"},
            assigned_agent=agent_id,
            estimated_execution_time_ms=random.uniform(5000, 30000),
            performance_metrics=PerformanceMetrics(
                execution_time_ms=random.uniform(0.5, 2.0),
                memory_usage_mb=random.uniform(5.0, 25.0),
                cpu_usage_percent=random.uniform(3.0, 12.0)
            ),
            metadata=EventMetadata(
                correlation_id=uuid.uuid4(),
                source_service="mock_generator"
            )
        ).dict()
    
    def _create_node_completed_event(
        self,
        workflow_id: uuid.UUID,
        session_id: uuid.UUID,
        node_id: str,
        success: bool,
        agent_id: uuid.UUID
    ) -> Dict[str, Any]:
        """Create node completed event."""
        return NodeCompletedEvent(
            workflow_id=workflow_id,
            session_id=session_id,
            agent_id=agent_id,
            node_id=node_id,
            success=success,
            output_data={"result": "completed" if success else "failed"},
            error_details=None if success else {"error": "Task execution failed"},
            retry_count=0 if success else random.randint(1, 3),
            performance_metrics=PerformanceMetrics(
                execution_time_ms=random.uniform(1.0, 4.0),
                memory_usage_mb=random.uniform(8.0, 35.0),
                cpu_usage_percent=random.uniform(5.0, 18.0)
            ),
            metadata=EventMetadata(
                correlation_id=uuid.uuid4(),
                source_service="mock_generator"
            )
        ).dict()
    
    def _create_agent_state_changed_event(
        self,
        agent_id: uuid.UUID,
        session_id: uuid.UUID,
        previous_state: str,
        new_state: str,
        reason: str
    ) -> Dict[str, Any]:
        """Create agent state changed event."""
        agent = self.active_agents.get(agent_id, {})
        return AgentStateChangedEvent(
            agent_id=agent_id,
            session_id=session_id,
            previous_state=previous_state,
            new_state=new_state,
            state_transition_reason=reason,
            capabilities=agent.get("capabilities", []),
            resource_allocation={"cpu_percent": random.uniform(10, 80), "memory_mb": random.uniform(100, 500)},
            persona_data={"persona": agent.get("persona", "unknown")},
            performance_metrics=PerformanceMetrics(
                execution_time_ms=random.uniform(0.5, 2.0),
                memory_usage_mb=random.uniform(3.0, 15.0),
                cpu_usage_percent=random.uniform(2.0, 8.0)
            ),
            metadata=EventMetadata(
                correlation_id=uuid.uuid4(),
                source_service="mock_generator"
            )
        ).dict()
    
    def _create_pre_tool_use_event(
        self,
        agent_id: uuid.UUID,
        session_id: uuid.UUID,
        tool_name: str,
        correlation_id: uuid.UUID
    ) -> Dict[str, Any]:
        """Create pre-tool use event."""
        parameters = self._generate_tool_parameters(tool_name)
        
        return PreToolUseEvent(
            agent_id=agent_id,
            session_id=session_id,
            tool_name=tool_name,
            parameters=parameters,
            tool_version="1.0.0",
            expected_output_type="text",
            timeout_ms=30000,
            performance_metrics=PerformanceMetrics(
                execution_time_ms=random.uniform(0.2, 1.0),
                memory_usage_mb=random.uniform(1.0, 8.0),
                cpu_usage_percent=random.uniform(1.0, 5.0)
            ),
            metadata=EventMetadata(
                correlation_id=correlation_id,
                source_service="mock_generator"
            )
        ).dict()
    
    def _create_post_tool_use_event(
        self,
        agent_id: uuid.UUID,
        session_id: uuid.UUID,
        tool_name: str,
        correlation_id: uuid.UUID,
        execution_time_ms: int,
        success: bool
    ) -> Dict[str, Any]:
        """Create post-tool use event."""
        result = self._generate_tool_result(tool_name, success) if success else None
        error = None if success else f"{tool_name} execution failed: {random.choice(['timeout', 'permission_denied', 'not_found', 'invalid_input'])}"
        
        return PostToolUseEvent(
            agent_id=agent_id,
            session_id=session_id,
            tool_name=tool_name,
            success=success,
            result=result,
            error=error,
            error_type="execution_error" if not success else None,
            retry_count=0 if success else random.randint(1, 2),
            result_truncated=False,
            performance_metrics=PerformanceMetrics(
                execution_time_ms=execution_time_ms,
                memory_usage_mb=random.uniform(2.0, 20.0),
                cpu_usage_percent=random.uniform(5.0, 25.0)
            ),
            metadata=EventMetadata(
                correlation_id=correlation_id,
                source_service="mock_generator"
            )
        ).dict()
    
    def _create_semantic_query_event(
        self,
        agent_id: uuid.UUID,
        session_id: uuid.UUID,
        context_id: uuid.UUID
    ) -> Dict[str, Any]:
        """Create semantic query event."""
        query_text = random.choice(self.query_templates)
        query_embedding = [random.uniform(-1, 1) for _ in range(1536)]  # Mock embedding
        
        return SemanticQueryEvent(
            agent_id=agent_id,
            session_id=session_id,
            context_id=context_id,
            query_text=query_text,
            query_embedding=query_embedding,
            similarity_threshold=0.7,
            max_results=10,
            results_count=random.randint(0, 10),
            search_strategy="semantic_similarity",
            performance_metrics=PerformanceMetrics(
                execution_time_ms=random.uniform(50.0, 200.0),
                memory_usage_mb=random.uniform(25.0, 100.0),
                cpu_usage_percent=random.uniform(15.0, 40.0)
            ),
            metadata=EventMetadata(
                correlation_id=uuid.uuid4(),
                source_service="mock_generator"
            )
        ).dict()
    
    def _create_semantic_update_event(
        self,
        agent_id: uuid.UUID,
        session_id: uuid.UUID,
        context_id: uuid.UUID
    ) -> Dict[str, Any]:
        """Create semantic update event."""
        return SemanticUpdateEvent(
            agent_id=agent_id,
            session_id=session_id,
            context_id=context_id,
            operation_type=random.choice(["insert", "update", "delete"]),
            content={"type": "knowledge", "data": "Updated semantic knowledge"},
            content_embedding=[random.uniform(-1, 1) for _ in range(1536)],
            content_id=str(uuid.uuid4()),
            content_type="knowledge_fragment",
            affected_records=random.randint(1, 5),
            performance_metrics=PerformanceMetrics(
                execution_time_ms=random.uniform(30.0, 150.0),
                memory_usage_mb=random.uniform(15.0, 75.0),
                cpu_usage_percent=random.uniform(10.0, 30.0)
            ),
            metadata=EventMetadata(
                correlation_id=uuid.uuid4(),
                source_service="mock_generator"
            )
        ).dict()
    
    def _create_message_published_event(
        self,
        from_agent: str,
        to_agent: str,
        message_id: uuid.UUID,
        session_id: uuid.UUID
    ) -> Dict[str, Any]:
        """Create message published event."""
        return MessagePublishedEvent(
            session_id=session_id,
            message_id=message_id,
            from_agent=from_agent,
            to_agent=to_agent,
            message_type="coordination",
            message_content={"action": "status_update", "data": "Task progress update"},
            priority="medium",
            delivery_method="redis_stream",
            expected_response=random.choice([True, False]),
            performance_metrics=PerformanceMetrics(
                execution_time_ms=random.uniform(1.0, 5.0),
                memory_usage_mb=random.uniform(0.5, 3.0),
                cpu_usage_percent=random.uniform(1.0, 4.0)
            ),
            metadata=EventMetadata(
                correlation_id=uuid.uuid4(),
                source_service="mock_generator"
            )
        ).dict()
    
    def _create_message_received_event(
        self,
        from_agent: str,
        message_id: uuid.UUID,
        session_id: uuid.UUID
    ) -> Dict[str, Any]:
        """Create message received event."""
        return MessageReceivedEvent(
            session_id=session_id,
            message_id=message_id,
            from_agent=from_agent,
            processing_status=random.choice(["accepted", "rejected", "deferred"]),
            processing_reason="Message processed successfully",
            response_generated=random.choice([True, False]),
            delivery_latency_ms=random.uniform(10.0, 100.0),
            performance_metrics=PerformanceMetrics(
                execution_time_ms=random.uniform(0.5, 3.0),
                memory_usage_mb=random.uniform(0.3, 2.0),
                cpu_usage_percent=random.uniform(0.5, 3.0)
            ),
            metadata=EventMetadata(
                correlation_id=uuid.uuid4(),
                source_service="mock_generator"
            )
        ).dict()
    
    def _create_failure_detected_event(
        self,
        agent_id: uuid.UUID,
        session_id: uuid.UUID,
        affected_component: str
    ) -> Dict[str, Any]:
        """Create failure detected event."""
        return FailureDetectedEvent(
            agent_id=agent_id,
            session_id=session_id,
            failure_type=random.choice(["timeout", "resource_exhaustion", "dependency_failure", "validation_error"]),
            failure_description=f"Failure detected in {affected_component}",
            affected_component=affected_component,
            severity=random.choice(["low", "medium", "high", "critical"]),
            error_details={"component": affected_component, "timestamp": datetime.utcnow().isoformat()},
            detection_method="automated_monitoring",
            performance_metrics=PerformanceMetrics(
                execution_time_ms=random.uniform(1.0, 5.0),
                memory_usage_mb=random.uniform(2.0, 10.0),
                cpu_usage_percent=random.uniform(2.0, 8.0)
            ),
            metadata=EventMetadata(
                correlation_id=uuid.uuid4(),
                source_service="mock_generator"
            )
        ).dict()
    
    def _create_recovery_initiated_event(
        self,
        agent_id: uuid.UUID,
        session_id: uuid.UUID,
        trigger_failure: str
    ) -> Dict[str, Any]:
        """Create recovery initiated event."""
        return RecoveryInitiatedEvent(
            agent_id=agent_id,
            session_id=session_id,
            recovery_strategy=random.choice(["retry", "fallback", "rollback", "escalation"]),
            trigger_failure=trigger_failure,
            recovery_steps=["analyze_failure", "apply_recovery", "verify_system"],
            estimated_recovery_time_ms=random.uniform(5000.0, 30000.0),
            backup_systems_activated=["backup_processor"],
            performance_metrics=PerformanceMetrics(
                execution_time_ms=random.uniform(2.0, 8.0),
                memory_usage_mb=random.uniform(5.0, 20.0),
                cpu_usage_percent=random.uniform(5.0, 15.0)
            ),
            metadata=EventMetadata(
                correlation_id=uuid.uuid4(),
                source_service="mock_generator"
            )
        ).dict()
    
    def _create_system_health_check_event(self, session_id: uuid.UUID) -> Dict[str, Any]:
        """Create system health check event."""
        return SystemHealthCheckEvent(
            session_id=session_id,
            health_status=random.choice(["healthy", "degraded", "unhealthy"]),
            check_type="periodic_health_check",
            component_statuses={
                "redis": "healthy",
                "postgres": "healthy",
                "agents": random.choice(["healthy", "degraded"]),
                "workflows": "healthy"
            },
            performance_indicators={
                "memory_usage_percent": random.uniform(30.0, 85.0),
                "cpu_usage_percent": random.uniform(15.0, 70.0),
                "disk_usage_percent": random.uniform(20.0, 60.0),
                "network_latency_ms": random.uniform(10.0, 100.0)
            },
            alerts_triggered=[],
            recommended_actions=["monitor_memory_usage"] if random.random() > 0.7 else [],
            performance_metrics=PerformanceMetrics(
                execution_time_ms=random.uniform(10.0, 50.0),
                memory_usage_mb=random.uniform(5.0, 25.0),
                cpu_usage_percent=random.uniform(8.0, 20.0)
            ),
            metadata=EventMetadata(
                correlation_id=uuid.uuid4(),
                source_service="mock_generator"
            )
        ).dict()
    
    def _generate_tool_parameters(self, tool_name: str) -> Dict[str, Any]:
        """Generate realistic parameters for tool usage."""
        param_templates = {
            "Read": {"file_path": "/Users/bogdan/work/project/src/main.py"},
            "Write": {"file_path": "/Users/bogdan/work/project/src/new_file.py", "content": "# New implementation"},
            "Edit": {"file_path": "/Users/bogdan/work/project/src/main.py", "old_string": "old_code", "new_string": "new_code"},
            "Bash": {"command": "python -m pytest", "description": "Run test suite"},
            "Grep": {"pattern": "TODO", "path": "/Users/bogdan/work/project", "output_mode": "files_with_matches"},
            "WebSearch": {"query": "Python async best practices", "max_results": 10},
            "TodoWrite": {"todos": [{"content": "Fix bug in authentication", "status": "pending", "priority": "high"}]}
        }
        
        return param_templates.get(tool_name, {"generic_param": "value"})
    
    def _generate_tool_result(self, tool_name: str, success: bool) -> Any:
        """Generate realistic tool execution results."""
        if not success:
            return None
        
        result_templates = {
            "Read": "File content:\nclass MyClass:\n    def __init__(self):\n        pass",
            "Write": "File written successfully",
            "Edit": "File edited successfully",
            "Bash": "Tests passed: 25/25",
            "Grep": ["file1.py", "file2.py", "file3.py"],
            "WebSearch": {"results": [{"title": "Python Async Guide", "url": "https://example.com"}]},
            "TodoWrite": {"status": "success", "todos_updated": 1}
        }
        
        return result_templates.get(tool_name, f"{tool_name} executed successfully")


class MockEventStreamService:
    """Mock event stream service for testing observability systems."""
    
    def __init__(
        self,
        redis_stream_name: str = "observability_events",
        event_rate_per_second: float = 10.0,
        max_concurrent_workflows: int = 3
    ):
        """
        Initialize mock event stream service.
        
        Args:
            redis_stream_name: Redis stream name for events
            event_rate_per_second: Target event generation rate
            max_concurrent_workflows: Maximum concurrent workflows
        """
        self.redis_stream_name = redis_stream_name
        self.event_rate_per_second = event_rate_per_second
        self.max_concurrent_workflows = max_concurrent_workflows
        self.generator = MockEventGenerator()
        self.running = False
        
        logger.info(
            "Mock event stream service initialized",
            stream_name=redis_stream_name,
            event_rate=event_rate_per_second,
            max_workflows=max_concurrent_workflows
        )
    
    async def start_streaming(self) -> None:
        """Start streaming mock events to Redis."""
        self.running = True
        redis_client = get_redis()
        
        logger.info("Starting mock event streaming")
        
        try:
            while self.running:
                # Generate a workflow scenario
                scenario = random.choice(list(WorkflowScenario))
                
                logger.info(f"Generating workflow scenario: {scenario.value}")
                
                # Generate events for the scenario
                event_count = 0
                for event_dict in self.generator.generate_workflow_scenario(scenario):
                    if not self.running:
                        break
                    
                    try:
                        # Serialize event
                        serialized_data, metadata = serialize_for_stream(event_dict)
                        
                        # Add to Redis stream
                        stream_id = await redis_client.xadd(
                            self.redis_stream_name,
                            {
                                'event_data': serialized_data,
                                'metadata': str(metadata),
                                'generated_at': datetime.utcnow().isoformat()
                            }
                        )
                        
                        event_count += 1
                        
                        # Rate limiting
                        await asyncio.sleep(1.0 / self.event_rate_per_second)
                        
                    except Exception as e:
                        logger.error(f"Failed to stream event: {e}", exc_info=True)
                
                logger.info(
                    f"Completed workflow scenario: {scenario.value}",
                    events_generated=event_count
                )
                
                # Pause between workflows
                await asyncio.sleep(random.uniform(5.0, 15.0))
                
        except Exception as e:
            logger.error(f"Event streaming failed: {e}", exc_info=True)
        finally:
            logger.info("Mock event streaming stopped")
    
    async def stop_streaming(self) -> None:
        """Stop streaming mock events."""
        self.running = False
        logger.info("Stopping mock event streaming")
    
    async def generate_single_scenario(self, scenario: WorkflowScenario) -> List[Dict[str, Any]]:
        """Generate events for a single scenario without streaming."""
        events = []
        for event_dict in self.generator.generate_workflow_scenario(scenario):
            events.append(event_dict)
        
        logger.info(
            f"Generated single scenario: {scenario.value}",
            event_count=len(events)
        )
        
        return events


# Convenience functions for testing
async def start_mock_stream(
    duration_seconds: int = 300,
    event_rate: float = 5.0
) -> None:
    """Start mock event stream for specified duration."""
    service = MockEventStreamService(event_rate_per_second=event_rate)
    
    # Start streaming in background
    streaming_task = asyncio.create_task(service.start_streaming())
    
    # Wait for specified duration
    await asyncio.sleep(duration_seconds)
    
    # Stop streaming
    await service.stop_streaming()
    streaming_task.cancel()
    
    try:
        await streaming_task
    except asyncio.CancelledError:
        pass


def generate_sample_events(count: int = 100) -> List[Dict[str, Any]]:
    """Generate sample events for testing."""
    generator = MockEventGenerator()
    events = []
    
    scenarios = list(WorkflowScenario)
    for i in range(count // len(scenarios) + 1):
        if len(events) >= count:
            break
        
        scenario = scenarios[i % len(scenarios)]
        scenario_events = list(generator.generate_workflow_scenario(scenario))
        events.extend(scenario_events[:count - len(events)])
    
    return events[:count]


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        logger.info("Starting mock event stream for 60 seconds")
        await start_mock_stream(duration_seconds=60, event_rate=8.0)
    
    asyncio.run(main())