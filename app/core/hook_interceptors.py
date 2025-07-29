"""
Hook Interceptors for LeanVibe Agent Hive 2.0 - VS 6.1

Advanced hook interceptors that integrate with existing system components:
- Workflow engine integration with DAG execution hooks
- Agent orchestrator lifecycle hooks
- Tool execution interception with <5ms overhead
- Semantic memory operation hooks
- Communication system hooks
- Recovery and failure detection hooks
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, TypeVar, Union
from functools import wraps
import inspect
import structlog

from .observability_hooks import ObservabilityHooks, get_hooks
from ..models.workflow import WorkflowStatus
from ..models.task import TaskStatus

logger = structlog.get_logger()

T = TypeVar('T')


class HookTiming:
    """Context manager for measuring hook execution time."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_ms: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000


def workflow_lifecycle_hook(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for workflow lifecycle methods to automatically emit hooks.
    
    Intercepts workflow start, end, pause, resume events.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract workflow information from method call
        self_instance = args[0] if args else None
        workflow_id = None
        agent_id = None
        session_id = None
        
        # Try to extract identifiers from arguments
        if hasattr(self_instance, 'workflow_id'):
            workflow_id = getattr(self_instance, 'workflow_id')
        if hasattr(self_instance, 'agent_id'):
            agent_id = getattr(self_instance, 'agent_id')
        if hasattr(self_instance, 'session_id'):
            session_id = getattr(self_instance, 'session_id')
        
        # Check kwargs for identifiers
        workflow_id = workflow_id or kwargs.get('workflow_id')
        agent_id = agent_id or kwargs.get('agent_id')
        session_id = session_id or kwargs.get('session_id')
        
        hooks = get_hooks()
        method_name = func.__name__
        
        with HookTiming() as timing:
            try:
                # Emit pre-execution hook based on method name
                if method_name in ['execute_workflow', 'start_workflow']:
                    await hooks.workflow_started(
                        workflow_id=workflow_id or uuid.uuid4(),
                        workflow_name=kwargs.get('workflow_name', 'Unknown'),
                        workflow_definition=kwargs.get('workflow_definition', {}),
                        agent_id=agent_id,
                        session_id=session_id,
                        initial_context=kwargs.get('initial_context'),
                        estimated_duration_ms=kwargs.get('estimated_duration_ms'),
                        priority=kwargs.get('priority')
                    )
                
                # Execute the original function
                result = await func(*args, **kwargs)
                
                # Emit post-execution hook based on method name and result
                if method_name in ['execute_workflow', 'complete_workflow']:
                    status = "completed"
                    if hasattr(result, 'status'):
                        status = result.status.value if hasattr(result.status, 'value') else str(result.status)
                    
                    await hooks.workflow_ended(
                        workflow_id=workflow_id or uuid.uuid4(),
                        status=status,
                        completion_reason=kwargs.get('completion_reason', 'Normal completion'),
                        agent_id=agent_id,
                        session_id=session_id,
                        final_result=getattr(result, 'result', None) if hasattr(result, 'result') else None,
                        total_tasks_executed=getattr(result, 'completed_tasks', None) if hasattr(result, 'completed_tasks') else None,
                        failed_tasks=getattr(result, 'failed_tasks', None) if hasattr(result, 'failed_tasks') else None,
                        actual_duration_ms=timing.duration_ms
                    )
                
                return result
                
            except Exception as e:
                # Emit failure hook
                await hooks.failure_detected(
                    failure_type="workflow_execution_failure",
                    failure_description=str(e),
                    affected_component=f"{self_instance.__class__.__name__}.{method_name}",
                    severity="high",
                    error_details={
                        "method": method_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "execution_time_ms": timing.duration_ms
                    },
                    agent_id=agent_id,
                    session_id=session_id,
                    workflow_id=workflow_id
                )
                raise
    
    return wrapper


def task_execution_hook(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for task execution methods to automatically emit node hooks.
    
    Intercepts task node execution, completion, and failure events.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract task information
        self_instance = args[0] if args else None
        task_id = kwargs.get('task_id') or (args[1] if len(args) > 1 else None)
        workflow_id = kwargs.get('workflow_id')
        agent_id = kwargs.get('agent_id')
        session_id = kwargs.get('session_id')
        
        # Try to extract from instance
        if hasattr(self_instance, 'workflow_id'):
            workflow_id = workflow_id or getattr(self_instance, 'workflow_id')
        if hasattr(self_instance, 'agent_id'):
            agent_id = agent_id or getattr(self_instance, 'agent_id')
        
        hooks = get_hooks()
        method_name = func.__name__
        
        with HookTiming() as timing:
            try:
                # Emit node executing hook
                await hooks.node_executing(
                    workflow_id=workflow_id or uuid.uuid4(),
                    node_id=str(task_id) if task_id else f"task-{uuid.uuid4().hex[:8]}",
                    node_type=kwargs.get('task_type', 'generic'),
                    agent_id=agent_id,
                    session_id=session_id,
                    node_name=kwargs.get('task_name'),
                    input_data=kwargs.get('input_data'),
                    dependencies_satisfied=kwargs.get('dependencies_satisfied'),
                    assigned_agent=agent_id,
                    estimated_execution_time_ms=kwargs.get('estimated_execution_time_ms')
                )
                
                # Execute the original function
                result = await func(*args, **kwargs)
                
                # Emit node completed hook
                success = True
                error_details = None
                
                if hasattr(result, 'status'):
                    success = result.status in [TaskStatus.COMPLETED, 'completed', 'success']
                    if not success and hasattr(result, 'error'):
                        error_details = {"error": str(result.error)}
                
                await hooks.node_completed(
                    workflow_id=workflow_id or uuid.uuid4(),
                    node_id=str(task_id) if task_id else f"task-{uuid.uuid4().hex[:8]}",
                    success=success,
                    agent_id=agent_id,
                    session_id=session_id,
                    output_data=getattr(result, 'result', None) if hasattr(result, 'result') else None,
                    error_details=error_details,
                    retry_count=kwargs.get('retry_count', 0),
                    downstream_nodes=kwargs.get('downstream_nodes'),
                    execution_time_ms=timing.duration_ms
                )
                
                return result
                
            except Exception as e:
                # Emit node failed hook
                await hooks.node_completed(
                    workflow_id=workflow_id or uuid.uuid4(),
                    node_id=str(task_id) if task_id else f"task-{uuid.uuid4().hex[:8]}",
                    success=False,
                    agent_id=agent_id,
                    session_id=session_id,
                    error_details={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "execution_time_ms": timing.duration_ms
                    },
                    execution_time_ms=timing.duration_ms
                )
                raise
    
    return wrapper


def tool_execution_hook(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for tool execution methods to automatically emit tool hooks.
    
    This is the MOST CRITICAL hook for debugging tool execution.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract tool information
        self_instance = args[0] if args else None
        tool_name = kwargs.get('tool_name', func.__name__)
        agent_id = kwargs.get('agent_id')
        session_id = kwargs.get('session_id')
        
        # Try to extract from instance
        if hasattr(self_instance, 'agent_id'):
            agent_id = agent_id or getattr(self_instance, 'agent_id')
        if hasattr(self_instance, 'session_id'):
            session_id = session_id or getattr(self_instance, 'session_id')
        
        # Extract tool parameters
        parameters = kwargs.copy()
        # Remove non-parameter keys
        non_param_keys = {'agent_id', 'session_id', 'tool_name', 'self'}
        parameters = {k: v for k, v in parameters.items() if k not in non_param_keys}
        
        hooks = get_hooks()
        
        with HookTiming() as timing:
            # Emit pre-tool use hook
            await hooks.pre_tool_use(
                agent_id=agent_id or uuid.uuid4(),
                tool_name=tool_name,
                parameters=parameters,
                session_id=session_id,
                tool_version=kwargs.get('tool_version'),
                expected_output_type=kwargs.get('expected_output_type'),
                timeout_ms=kwargs.get('timeout_ms'),
                retry_policy=kwargs.get('retry_policy')
            )
            
            try:
                # Execute the original function
                result = await func(*args, **kwargs)
                
                # Emit post-tool use hook (success)
                await hooks.post_tool_use(
                    agent_id=agent_id or uuid.uuid4(),
                    tool_name=tool_name,
                    success=True,
                    session_id=session_id,
                    result=result,
                    execution_time_ms=timing.duration_ms
                )
                
                return result
                
            except Exception as e:
                # Emit post-tool use hook (failure)
                await hooks.post_tool_use(
                    agent_id=agent_id or uuid.uuid4(),
                    tool_name=tool_name,
                    success=False,
                    session_id=session_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    execution_time_ms=timing.duration_ms
                )
                raise
    
    return wrapper


def semantic_memory_hook(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for semantic memory operations to emit memory hooks.
    
    Intercepts semantic queries and updates for intelligence analytics.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        self_instance = args[0] if args else None
        agent_id = kwargs.get('agent_id')
        session_id = kwargs.get('session_id')
        
        # Try to extract from instance
        if hasattr(self_instance, 'agent_id'):
            agent_id = agent_id or getattr(self_instance, 'agent_id')
        if hasattr(self_instance, 'session_id'):
            session_id = session_id or getattr(self_instance, 'session_id')
        
        hooks = get_hooks()
        method_name = func.__name__
        
        with HookTiming() as timing:
            try:
                # Emit appropriate memory hook based on method name
                if method_name in ['search', 'query', 'similarity_search']:
                    query_text = kwargs.get('query', kwargs.get('query_text', ''))
                    query_embedding = kwargs.get('query_embedding', kwargs.get('embedding', []))
                    
                    # Execute the function first to get results count
                    result = await func(*args, **kwargs)
                    
                    results_count = len(result) if isinstance(result, list) else None
                    
                    await hooks.semantic_query(
                        query_text=query_text,
                        query_embedding=query_embedding,
                        agent_id=agent_id,
                        session_id=session_id,
                        similarity_threshold=kwargs.get('threshold', kwargs.get('similarity_threshold')),
                        max_results=kwargs.get('limit', kwargs.get('max_results')),
                        filter_criteria=kwargs.get('filter_criteria'),
                        results_count=results_count,
                        search_strategy=kwargs.get('search_strategy'),
                        execution_time_ms=timing.duration_ms
                    )
                    
                    return result
                
                elif method_name in ['insert', 'update', 'delete', 'upsert']:
                    # Execute the function first to get affected records
                    result = await func(*args, **kwargs)
                    
                    affected_records = getattr(result, 'affected_rows', None) if hasattr(result, 'affected_rows') else None
                    
                    await hooks.semantic_update(
                        operation_type=method_name,
                        content=kwargs.get('content', kwargs.get('data', {})),
                        agent_id=agent_id,
                        session_id=session_id,
                        content_embedding=kwargs.get('embedding'),
                        content_id=kwargs.get('content_id', kwargs.get('id')),
                        content_type=kwargs.get('content_type'),
                        content_metadata=kwargs.get('metadata'),
                        affected_records=affected_records,
                        execution_time_ms=timing.duration_ms
                    )
                    
                    return result
                
                else:
                    # Generic memory operation
                    return await func(*args, **kwargs)
                
            except Exception as e:
                # Emit failure hook for memory operations
                await hooks.failure_detected(
                    failure_type="semantic_memory_failure",
                    failure_description=str(e),
                    affected_component=f"SemanticMemory.{method_name}",
                    severity="medium",
                    error_details={
                        "method": method_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "execution_time_ms": timing.duration_ms
                    },
                    agent_id=agent_id,
                    session_id=session_id
                )
                raise
    
    return wrapper


def agent_communication_hook(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for agent communication methods to emit communication hooks.
    
    Intercepts message publishing and receiving events.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        self_instance = args[0] if args else None
        agent_id = kwargs.get('agent_id')
        session_id = kwargs.get('session_id')
        
        # Try to extract from instance
        if hasattr(self_instance, 'agent_id'):
            agent_id = agent_id or getattr(self_instance, 'agent_id')
        
        hooks = get_hooks()
        method_name = func.__name__
        
        with HookTiming() as timing:
            try:
                result = await func(*args, **kwargs)
                
                # Emit communication hooks based on method name
                if method_name in ['publish_message', 'send_message', 'broadcast']:
                    message_id = kwargs.get('message_id', uuid.uuid4())
                    
                    await hooks.message_published(
                        message_id=message_id,
                        from_agent=kwargs.get('from_agent', str(agent_id) if agent_id else 'unknown'),
                        to_agent=kwargs.get('to_agent', kwargs.get('recipient', 'broadcast')),
                        message_type=kwargs.get('message_type', 'generic'),
                        message_content=kwargs.get('message', kwargs.get('content', {})),
                        agent_id=agent_id,
                        session_id=session_id,
                        priority=kwargs.get('priority'),
                        delivery_method=kwargs.get('delivery_method'),
                        expected_response=kwargs.get('expected_response', False)
                    )
                
                elif method_name in ['receive_message', 'handle_message', 'process_message']:
                    message_id = kwargs.get('message_id', uuid.uuid4())
                    
                    processing_status = "accepted"
                    if hasattr(result, 'status'):
                        processing_status = result.status
                    elif not result:
                        processing_status = "rejected"
                    
                    await hooks.message_received(
                        message_id=message_id,
                        from_agent=kwargs.get('from_agent', 'unknown'),
                        processing_status=processing_status,
                        agent_id=agent_id,
                        session_id=session_id,
                        processing_reason=kwargs.get('processing_reason'),
                        response_generated=kwargs.get('response_generated', False),
                        delivery_latency_ms=timing.duration_ms
                    )
                
                return result
                
            except Exception as e:
                # Emit failure hook for communication
                await hooks.failure_detected(
                    failure_type="communication_failure",
                    failure_description=str(e),
                    affected_component=f"Communication.{method_name}",
                    severity="medium",
                    error_details={
                        "method": method_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "execution_time_ms": timing.duration_ms
                    },
                    agent_id=agent_id,
                    session_id=session_id
                )
                raise
    
    return wrapper


def agent_state_hook(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for agent state transitions to emit agent state hooks.
    
    Intercepts agent state changes and capability utilization.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        self_instance = args[0] if args else None
        agent_id = kwargs.get('agent_id')
        session_id = kwargs.get('session_id')
        
        # Try to extract from instance
        if hasattr(self_instance, 'agent_id'):
            agent_id = agent_id or getattr(self_instance, 'agent_id')
        if hasattr(self_instance, 'id'):
            agent_id = agent_id or getattr(self_instance, 'id')
        
        hooks = get_hooks()
        method_name = func.__name__
        
        # Get previous state if available
        previous_state = getattr(self_instance, 'state', 'unknown') if self_instance else 'unknown'
        
        with HookTiming() as timing:
            try:
                result = await func(*args, **kwargs)
                
                # Emit agent state changed hook for state transitions
                if method_name in ['set_state', 'transition_to', 'change_state', 'update_state']:
                    new_state = kwargs.get('new_state', kwargs.get('state', 'unknown'))
                    
                    await hooks.agent_state_changed(
                        agent_id=agent_id or uuid.uuid4(),
                        previous_state=str(previous_state),
                        new_state=str(new_state),
                        state_transition_reason=kwargs.get('reason', f"Method call: {method_name}"),
                        session_id=session_id,
                        capabilities=kwargs.get('capabilities'),
                        resource_allocation=kwargs.get('resource_allocation'),
                        persona_data=kwargs.get('persona_data')
                    )
                
                # Emit capability utilization hook for capability methods
                elif method_name.startswith('use_') or method_name.endswith('_capability'):
                    capability_name = method_name.replace('use_', '').replace('_capability', '')
                    
                    await hooks.agent_capability_utilized(
                        agent_id=agent_id or uuid.uuid4(),
                        capability_name=capability_name,
                        utilization_context=kwargs.get('context', f"Direct invocation of {method_name}"),
                        session_id=session_id,
                        input_parameters=kwargs.copy(),
                        capability_result=result if hasattr(result, '__dict__') else {"result": result},
                        efficiency_score=kwargs.get('efficiency_score'),
                        execution_time_ms=timing.duration_ms
                    )
                
                return result
                
            except Exception as e:
                # Emit failure hook for agent operations
                await hooks.failure_detected(
                    failure_type="agent_operation_failure",
                    failure_description=str(e),
                    affected_component=f"Agent.{method_name}",
                    severity="medium",
                    error_details={
                        "method": method_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "agent_id": str(agent_id) if agent_id else None,
                        "execution_time_ms": timing.duration_ms
                    },
                    agent_id=agent_id,
                    session_id=session_id
                )
                raise
    
    return wrapper


class SystemHookInterceptor:
    """
    System-level hook interceptor for comprehensive event capture.
    
    Provides centralized hook management and automatic integration.
    """
    
    def __init__(self, hooks: Optional[ObservabilityHooks] = None):
        """Initialize system hook interceptor."""
        self.hooks = hooks or get_hooks()
        self._intercepted_classes = set()
        
        logger.info("🎣 SystemHookInterceptor initialized")
    
    def intercept_class(self, cls: type, hook_patterns: Dict[str, Callable]) -> None:
        """
        Intercept all methods of a class with appropriate hooks.
        
        Args:
            cls: Class to intercept
            hook_patterns: Dict mapping method patterns to hook decorators
        """
        class_name = cls.__name__
        
        if class_name in self._intercepted_classes:
            logger.warning(f"Class {class_name} already intercepted")
            return
        
        intercepted_methods = 0
        
        for method_name in dir(cls):
            if method_name.startswith('_'):
                continue
            
            method = getattr(cls, method_name)
            if not callable(method):
                continue
            
            # Find matching hook pattern
            for pattern, hook_decorator in hook_patterns.items():
                if pattern in method_name.lower() or method_name.startswith(pattern):
                    # Apply hook decorator
                    setattr(cls, method_name, hook_decorator(method))
                    intercepted_methods += 1
                    break
        
        self._intercepted_classes.add(class_name)
        
        logger.info(
            f"🎯 Intercepted class {class_name}",
            intercepted_methods=intercepted_methods
        )
    
    def auto_intercept_workflow_engine(self, workflow_engine_class: type) -> None:
        """Auto-intercept workflow engine with appropriate hooks."""
        hook_patterns = {
            'execute': workflow_lifecycle_hook,
            'start': workflow_lifecycle_hook,
            'complete': workflow_lifecycle_hook,
            'task': task_execution_hook,
            'node': task_execution_hook
        }
        
        self.intercept_class(workflow_engine_class, hook_patterns)
    
    def auto_intercept_agent_registry(self, agent_registry_class: type) -> None:
        """Auto-intercept agent registry with state hooks."""
        hook_patterns = {
            'register': agent_state_hook,
            'unregister': agent_state_hook,
            'update': agent_state_hook,
            'set_state': agent_state_hook
        }
        
        self.intercept_class(agent_registry_class, hook_patterns)
    
    def auto_intercept_semantic_memory(self, semantic_memory_class: type) -> None:
        """Auto-intercept semantic memory with memory hooks."""
        hook_patterns = {
            'search': semantic_memory_hook,
            'query': semantic_memory_hook,
            'insert': semantic_memory_hook,
            'update': semantic_memory_hook,
            'delete': semantic_memory_hook,
            'upsert': semantic_memory_hook
        }
        
        self.intercept_class(semantic_memory_class, hook_patterns)
    
    def auto_intercept_communication(self, communication_class: type) -> None:
        """Auto-intercept communication system with communication hooks."""
        hook_patterns = {
            'publish': agent_communication_hook,
            'send': agent_communication_hook,
            'receive': agent_communication_hook,
            'broadcast': agent_communication_hook,
            'handle': agent_communication_hook
        }
        
        self.intercept_class(communication_class, hook_patterns)
    
    async def emit_system_health_check(self) -> None:
        """Emit periodic system health check event."""
        try:
            # Collect system health information
            component_statuses = {
                "hooks": "healthy" if self.hooks else "unhealthy",
                "interceptor": "healthy",
                "intercepted_classes": str(len(self._intercepted_classes))
            }
            
            performance_indicators = self.hooks.get_performance_metrics() if self.hooks else {}
            
            health_status = "healthy"
            if not self.hooks or not self.hooks.performance_tracker.is_within_target():
                health_status = "degraded"
            
            await self.hooks.system_health_check(
                health_status=health_status,
                check_type="periodic_health_check",
                component_statuses=component_statuses,
                performance_indicators=performance_indicators,
                alerts_triggered=None,
                recommended_actions=None
            )
            
        except Exception as e:
            logger.error("❌ Failed to emit system health check", error=str(e))


# Global system interceptor
_system_interceptor: Optional[SystemHookInterceptor] = None


def get_system_interceptor() -> Optional[SystemHookInterceptor]:
    """Get the global system interceptor."""
    return _system_interceptor


def initialize_system_interceptor(hooks: Optional[ObservabilityHooks] = None) -> SystemHookInterceptor:
    """Initialize and set the global system interceptor."""
    global _system_interceptor
    
    _system_interceptor = SystemHookInterceptor(hooks)
    
    logger.info("✅ Global system interceptor initialized")
    return _system_interceptor