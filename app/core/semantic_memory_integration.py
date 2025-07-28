"""
Semantic Memory Service Integration with Redis Streams and DAG Workflows

This module provides integration components that bridge the Semantic Memory Service
with the existing LeanVibe Agent Hive infrastructure, including Redis Streams
messaging and DAG workflow execution.

Features:
- Redis Streams integration for semantic memory tasks
- DAG workflow nodes for memory operations
- Consumer group management for memory service
- Event-driven memory updates
- Workflow context preservation
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

import httpx
from redis.asyncio import Redis
from pydantic import BaseModel, Field

# Import existing infrastructure components
from .redis import get_redis
from .workflow_engine import WorkflowNode, WorkflowContext, WorkflowResult
from .enhanced_redis_streams_manager import RedisStreamsManager
from .consumer_group_coordinator import ConsumerGroupCoordinator

# Import semantic memory schemas
from ..schemas.semantic_memory import (
    DocumentIngestRequest, DocumentIngestResponse, BatchIngestRequest,
    SemanticSearchRequest, SemanticSearchResponse, ContextCompressionRequest,
    ContextCompressionResponse, AgentKnowledgeResponse, CompressionMethod,
    ProcessingPriority, RelationshipType
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# SEMANTIC MEMORY TASK DEFINITIONS
# =============================================================================

class SemanticMemoryTaskType(str, Enum):
    """Types of semantic memory tasks."""
    INGEST_DOCUMENT = "ingest_document"
    BATCH_INGEST = "batch_ingest"
    SEARCH_SEMANTIC = "search_semantic"
    FIND_SIMILAR = "find_similar"
    GET_RELATED = "get_related"
    COMPRESS_CONTEXT = "compress_context"
    CONTEXTUALIZE = "contextualize"
    GET_AGENT_KNOWLEDGE = "get_agent_knowledge"
    HEALTH_CHECK = "health_check"


@dataclass
class SemanticMemoryTask:
    """Represents a semantic memory task for Redis Streams."""
    task_id: str
    task_type: SemanticMemoryTaskType
    agent_id: str
    workflow_id: Optional[str] = None
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0
    max_retries: int = 3
    
    def to_redis_message(self) -> Dict[str, str]:
        """Convert task to Redis Stream message format."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "agent_id": self.agent_id,
            "workflow_id": self.workflow_id or "",
            "priority": self.priority.value,
            "payload": json.dumps(self.payload),
            "metadata": json.dumps(self.metadata),
            "created_at": self.created_at.isoformat(),
            "retry_count": str(self.retry_count),
            "max_retries": str(self.max_retries)
        }
    
    @classmethod
    def from_redis_message(cls, message_data: Dict[str, bytes]) -> 'SemanticMemoryTask':
        """Create task from Redis Stream message."""
        return cls(
            task_id=message_data[b"task_id"].decode(),
            task_type=SemanticMemoryTaskType(message_data[b"task_type"].decode()),
            agent_id=message_data[b"agent_id"].decode(),
            workflow_id=message_data[b"workflow_id"].decode() or None,
            priority=ProcessingPriority(message_data[b"priority"].decode()),
            payload=json.loads(message_data[b"payload"].decode()),
            metadata=json.loads(message_data[b"metadata"].decode()),
            created_at=datetime.fromisoformat(message_data[b"created_at"].decode()),
            retry_count=int(message_data[b"retry_count"].decode()),
            max_retries=int(message_data[b"max_retries"].decode())
        )


class SemanticMemoryResult(BaseModel):
    """Result of a semantic memory task."""
    task_id: str
    success: bool
    result_data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# REDIS STREAMS INTEGRATION
# =============================================================================

class SemanticMemoryStreamsManager:
    """Manages Redis Streams for semantic memory operations."""
    
    # Stream names
    MEMORY_TASKS_STREAM = "semantic_memory_tasks"
    MEMORY_RESULTS_STREAM = "semantic_memory_results"
    MEMORY_EVENTS_STREAM = "semantic_memory_events"
    
    # Consumer groups
    MEMORY_PROCESSORS_GROUP = "memory_processors"
    MEMORY_LISTENERS_GROUP = "memory_listeners"
    
    def __init__(self, redis_client: Redis, memory_service_url: str):
        self.redis = redis_client
        self.memory_service_url = memory_service_url
        self.http_client = httpx.AsyncClient(base_url=memory_service_url, timeout=30.0)
        self.streams_manager = RedisStreamsManager(redis_client)
        self.consumer_coordinator = ConsumerGroupCoordinator(redis_client)
        self.running = False
        self._consumers = {}
    
    async def initialize(self):
        """Initialize streams and consumer groups."""
        logger.info("Initializing Semantic Memory Streams")
        
        # Create streams if they don't exist
        streams_to_create = [
            self.MEMORY_TASKS_STREAM,
            self.MEMORY_RESULTS_STREAM,
            self.MEMORY_EVENTS_STREAM
        ]
        
        for stream_name in streams_to_create:
            try:
                await self.redis.xgroup_create(
                    stream_name, 
                    self.MEMORY_PROCESSORS_GROUP, 
                    id="0", 
                    mkstream=True
                )
                logger.info(f"Created consumer group {self.MEMORY_PROCESSORS_GROUP} for {stream_name}")
            except Exception as e:
                if "BUSYGROUP" not in str(e):
                    logger.error(f"Failed to create consumer group for {stream_name}: {e}")
        
        # Initialize consumer coordinator
        await self.consumer_coordinator.initialize()
        
        logger.info("Semantic Memory Streams initialized successfully")
    
    async def submit_task(self, task: SemanticMemoryTask) -> str:
        """Submit a semantic memory task to the processing queue."""
        message_id = await self.redis.xadd(
            self.MEMORY_TASKS_STREAM,
            task.to_redis_message(),
            maxlen=10000  # Limit stream size
        )
        
        # Publish task event
        await self.publish_event("task_submitted", {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "agent_id": task.agent_id,
            "workflow_id": task.workflow_id,
            "priority": task.priority.value
        })
        
        logger.info(f"Submitted semantic memory task {task.task_id} to stream")
        return message_id.decode()
    
    async def publish_result(self, result: SemanticMemoryResult) -> str:
        """Publish a task result to the results stream."""
        message_data = {
            "task_id": result.task_id,
            "success": str(result.success),
            "result_data": json.dumps(result.result_data),
            "error_message": result.error_message or "",
            "processing_time_ms": str(result.processing_time_ms),
            "timestamp": result.timestamp.isoformat()
        }
        
        message_id = await self.redis.xadd(
            self.MEMORY_RESULTS_STREAM,
            message_data,
            maxlen=5000
        )
        
        # Publish result event
        await self.publish_event("task_completed", {
            "task_id": result.task_id,
            "success": result.success,
            "processing_time_ms": result.processing_time_ms
        })
        
        return message_id.decode()
    
    async def publish_event(self, event_type: str, event_data: Dict[str, Any]):
        """Publish an event to the events stream."""
        message_data = {
            "event_type": event_type,
            "event_data": json.dumps(event_data),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.redis.xadd(
            self.MEMORY_EVENTS_STREAM,
            message_data,
            maxlen=1000
        )
    
    async def start_task_processor(self, consumer_name: str, 
                                 processor_callback: Optional[Callable] = None):
        """Start a task processor consumer."""
        if processor_callback is None:
            processor_callback = self._default_task_processor
        
        consumer_id = f"{consumer_name}_{uuid.uuid4().hex[:8]}"
        
        async def process_tasks():
            """Process tasks from the stream."""
            while self.running:
                try:
                    # Read pending messages first
                    pending_messages = await self.redis.xreadgroup(
                        self.MEMORY_PROCESSORS_GROUP,
                        consumer_id,
                        {self.MEMORY_TASKS_STREAM: "0"},
                        count=1,
                        block=100
                    )
                    
                    if not pending_messages:
                        # No pending messages, read new ones
                        messages = await self.redis.xreadgroup(
                            self.MEMORY_PROCESSORS_GROUP,
                            consumer_id,
                            {self.MEMORY_TASKS_STREAM: ">"},
                            count=1,
                            block=1000
                        )
                    else:
                        messages = pending_messages
                    
                    for stream_name, stream_messages in messages:
                        for message_id, message_data in stream_messages:
                            try:
                                # Process the task
                                task = SemanticMemoryTask.from_redis_message(message_data)
                                result = await processor_callback(task)
                                
                                # Publish result
                                await self.publish_result(result)
                                
                                # Acknowledge message
                                await self.redis.xack(
                                    self.MEMORY_TASKS_STREAM,
                                    self.MEMORY_PROCESSORS_GROUP,
                                    message_id
                                )
                                
                                logger.debug(f"Processed task {task.task_id} successfully")
                                
                            except Exception as e:
                                logger.error(f"Error processing task: {e}")
                                
                                # Handle retry logic
                                task = SemanticMemoryTask.from_redis_message(message_data)
                                if task.retry_count < task.max_retries:
                                    task.retry_count += 1
                                    await self.submit_task(task)
                                else:
                                    # Max retries exceeded, publish failure
                                    failure_result = SemanticMemoryResult(
                                        task_id=task.task_id,
                                        success=False,
                                        error_message=str(e),
                                        processing_time_ms=0
                                    )
                                    await self.publish_result(failure_result)
                                
                                # Acknowledge failed message
                                await self.redis.xack(
                                    self.MEMORY_TASKS_STREAM,
                                    self.MEMORY_PROCESSORS_GROUP,
                                    message_id
                                )
                
                except Exception as e:
                    logger.error(f"Error in task processor loop: {e}")
                    await asyncio.sleep(1)
        
        # Start the processor
        self._consumers[consumer_id] = asyncio.create_task(process_tasks())
        logger.info(f"Started semantic memory task processor: {consumer_id}")
        
        return consumer_id
    
    async def _default_task_processor(self, task: SemanticMemoryTask) -> SemanticMemoryResult:
        """Default task processor that calls the semantic memory service."""
        start_time = time.time()
        
        try:
            if task.task_type == SemanticMemoryTaskType.INGEST_DOCUMENT:
                response = await self.http_client.post("/memory/ingest", json=task.payload)
                result_data = response.json()
                
            elif task.task_type == SemanticMemoryTaskType.BATCH_INGEST:
                response = await self.http_client.post("/memory/batch-ingest", json=task.payload)
                result_data = response.json()
                
            elif task.task_type == SemanticMemoryTaskType.SEARCH_SEMANTIC:
                response = await self.http_client.post("/memory/search", json=task.payload)
                result_data = response.json()
                
            elif task.task_type == SemanticMemoryTaskType.FIND_SIMILAR:
                response = await self.http_client.post("/memory/similarity", json=task.payload)
                result_data = response.json()
                
            elif task.task_type == SemanticMemoryTaskType.GET_RELATED:
                document_id = task.payload.get("document_id")
                params = {k: v for k, v in task.payload.items() if k != "document_id"}
                response = await self.http_client.get(f"/memory/related/{document_id}", params=params)
                result_data = response.json()
                
            elif task.task_type == SemanticMemoryTaskType.COMPRESS_CONTEXT:
                response = await self.http_client.post("/memory/compress", json=task.payload)
                result_data = response.json()
                
            elif task.task_type == SemanticMemoryTaskType.CONTEXTUALIZE:
                response = await self.http_client.post("/memory/contextualize", json=task.payload)
                result_data = response.json()
                
            elif task.task_type == SemanticMemoryTaskType.GET_AGENT_KNOWLEDGE:
                agent_id = task.payload.get("agent_id", task.agent_id)
                params = {k: v for k, v in task.payload.items() if k != "agent_id"}
                response = await self.http_client.get(f"/memory/agent-knowledge/{agent_id}", params=params)
                result_data = response.json()
                
            elif task.task_type == SemanticMemoryTaskType.HEALTH_CHECK:
                response = await self.http_client.get("/memory/health")
                result_data = response.json()
                
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            response.raise_for_status()
            processing_time = (time.time() - start_time) * 1000
            
            return SemanticMemoryResult(
                task_id=task.task_id,
                success=True,
                result_data=result_data,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            return SemanticMemoryResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time
            )
    
    async def start(self):
        """Start the streams manager."""
        self.running = True
        await self.initialize()
        
        # Start default task processor
        await self.start_task_processor("default_processor")
        
        logger.info("Semantic Memory Streams Manager started")
    
    async def stop(self):
        """Stop the streams manager."""
        self.running = False
        
        # Cancel all consumers
        for consumer_id, consumer_task in self._consumers.items():
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass
            logger.info(f"Stopped consumer: {consumer_id}")
        
        self._consumers.clear()
        
        # Close HTTP client
        await self.http_client.aclose()
        
        logger.info("Semantic Memory Streams Manager stopped")


# =============================================================================
# DAG WORKFLOW INTEGRATION
# =============================================================================

class SemanticMemoryWorkflowNode(WorkflowNode):
    """Base class for semantic memory workflow nodes."""
    
    def __init__(self, node_id: str, task_type: SemanticMemoryTaskType,
                 streams_manager: SemanticMemoryStreamsManager):
        super().__init__(node_id)
        self.task_type = task_type
        self.streams_manager = streams_manager
    
    async def execute(self, context: WorkflowContext) -> WorkflowResult:
        """Execute the semantic memory operation via Redis Streams."""
        try:
            # Create task from workflow context
            task = SemanticMemoryTask(
                task_id=f"{context.workflow_id}_{self.node_id}_{uuid.uuid4().hex[:8]}",
                task_type=self.task_type,
                agent_id=context.agent_id,
                workflow_id=context.workflow_id,
                priority=ProcessingPriority.NORMAL,
                payload=await self.prepare_task_payload(context),
                metadata={
                    "node_id": self.node_id,
                    "workflow_step": context.current_step,
                    "dag_execution": True
                }
            )
            
            # Submit task and wait for result
            await self.streams_manager.submit_task(task)
            result = await self.wait_for_result(task.task_id)
            
            if result.success:
                return WorkflowResult(
                    success=True,
                    output_data=result.result_data,
                    metadata={
                        "processing_time_ms": result.processing_time_ms,
                        "task_id": task.task_id
                    }
                )
            else:
                return WorkflowResult(
                    success=False,
                    error_message=result.error_message,
                    metadata={"task_id": task.task_id}
                )
                
        except Exception as e:
            logger.error(f"Error in semantic memory workflow node {self.node_id}: {e}")
            return WorkflowResult(
                success=False,
                error_message=str(e)
            )
    
    async def prepare_task_payload(self, context: WorkflowContext) -> Dict[str, Any]:
        """Prepare task payload from workflow context. Override in subclasses."""
        return context.input_data
    
    async def wait_for_result(self, task_id: str, timeout: float = 30.0) -> SemanticMemoryResult:
        """Wait for task result from the results stream."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Read from results stream
            messages = await self.streams_manager.redis.xread(
                {self.streams_manager.MEMORY_RESULTS_STREAM: "$"},
                count=10,
                block=1000
            )
            
            for stream_name, stream_messages in messages:
                for message_id, message_data in stream_messages:
                    result_task_id = message_data[b"task_id"].decode()
                    
                    if result_task_id == task_id:
                        # Found our result
                        return SemanticMemoryResult(
                            task_id=result_task_id,
                            success=message_data[b"success"].decode() == "True",
                            result_data=json.loads(message_data[b"result_data"].decode()),
                            error_message=message_data[b"error_message"].decode() or None,
                            processing_time_ms=float(message_data[b"processing_time_ms"].decode()),
                            timestamp=datetime.fromisoformat(message_data[b"timestamp"].decode())
                        )
        
        # Timeout
        raise TimeoutError(f"Timeout waiting for result of task {task_id}")


class DocumentIngestWorkflowNode(SemanticMemoryWorkflowNode):
    """Workflow node for document ingestion."""
    
    def __init__(self, node_id: str, streams_manager: SemanticMemoryStreamsManager):
        super().__init__(node_id, SemanticMemoryTaskType.INGEST_DOCUMENT, streams_manager)
    
    async def prepare_task_payload(self, context: WorkflowContext) -> Dict[str, Any]:
        """Prepare document ingestion payload."""
        return {
            "content": context.input_data.get("content", ""),
            "agent_id": context.agent_id,
            "workflow_id": context.workflow_id,
            "tags": context.input_data.get("tags", []) + ["dag_generated"],
            "metadata": {
                "node_id": self.node_id,
                "workflow_step": context.current_step,
                "importance": context.input_data.get("importance", 0.5)
            },
            "processing_options": {
                "generate_summary": context.input_data.get("generate_summary", False),
                "priority": ProcessingPriority.NORMAL
            }
        }


class SemanticSearchWorkflowNode(SemanticMemoryWorkflowNode):
    """Workflow node for semantic search."""
    
    def __init__(self, node_id: str, streams_manager: SemanticMemoryStreamsManager):
        super().__init__(node_id, SemanticMemoryTaskType.SEARCH_SEMANTIC, streams_manager)
    
    async def prepare_task_payload(self, context: WorkflowContext) -> Dict[str, Any]:
        """Prepare semantic search payload."""
        return {
            "query": context.input_data.get("query", ""),
            "limit": context.input_data.get("limit", 5),
            "similarity_threshold": context.input_data.get("similarity_threshold", 0.7),
            "agent_id": context.agent_id,
            "workflow_id": context.workflow_id,
            "filters": context.input_data.get("filters", {}),
            "search_options": {
                "rerank": True,
                "include_metadata": True,
                "explain_relevance": context.input_data.get("explain_relevance", False)
            }
        }


class ContextCompressionWorkflowNode(SemanticMemoryWorkflowNode):
    """Workflow node for context compression."""
    
    def __init__(self, node_id: str, streams_manager: SemanticMemoryStreamsManager):
        super().__init__(node_id, SemanticMemoryTaskType.COMPRESS_CONTEXT, streams_manager)
    
    async def prepare_task_payload(self, context: WorkflowContext) -> Dict[str, Any]:
        """Prepare context compression payload."""
        context_id = context.input_data.get("context_id", f"workflow_{context.workflow_id}")
        
        return {
            "context_id": context_id,
            "compression_method": context.input_data.get("compression_method", CompressionMethod.SEMANTIC_CLUSTERING),
            "target_reduction": context.input_data.get("target_reduction", 0.6),
            "preserve_importance_threshold": context.input_data.get("preserve_importance_threshold", 0.8),
            "agent_id": context.agent_id,
            "compression_options": {
                "preserve_recent": True,
                "maintain_relationships": True,
                "generate_summary": True
            }
        }


class AgentKnowledgeWorkflowNode(SemanticMemoryWorkflowNode):
    """Workflow node for retrieving agent knowledge."""
    
    def __init__(self, node_id: str, streams_manager: SemanticMemoryStreamsManager):
        super().__init__(node_id, SemanticMemoryTaskType.GET_AGENT_KNOWLEDGE, streams_manager)
    
    async def prepare_task_payload(self, context: WorkflowContext) -> Dict[str, Any]:
        """Prepare agent knowledge retrieval payload."""
        return {
            "agent_id": context.input_data.get("target_agent_id", context.agent_id),
            "knowledge_type": context.input_data.get("knowledge_type", "all"),
            "time_range": context.input_data.get("time_range", "7d")
        }


# =============================================================================
# WORKFLOW CONTEXT PRESERVATION
# =============================================================================

class WorkflowMemoryManager:
    """Manages workflow context preservation in semantic memory."""
    
    def __init__(self, streams_manager: SemanticMemoryStreamsManager):
        self.streams_manager = streams_manager
        self.workflow_contexts = {}
    
    async def capture_workflow_step(self, workflow_id: str, step_name: str,
                                  step_output: Dict[str, Any], agent_id: str):
        """Capture context from a workflow step."""
        # Create content describing the step
        content = f"Workflow '{workflow_id}' Step '{step_name}': {json.dumps(step_output, indent=2)}"
        
        # Create ingestion task
        task = SemanticMemoryTask(
            task_id=f"workflow_capture_{workflow_id}_{step_name}_{uuid.uuid4().hex[:8]}",
            task_type=SemanticMemoryTaskType.INGEST_DOCUMENT,
            agent_id=agent_id,
            workflow_id=workflow_id,
            priority=ProcessingPriority.NORMAL,
            payload={
                "content": content,
                "agent_id": agent_id,
                "workflow_id": workflow_id,
                "tags": ["workflow", "step_output", step_name, workflow_id],
                "metadata": {
                    "step_name": step_name,
                    "workflow_id": workflow_id,
                    "capture_type": "step_output",
                    "importance": 0.7
                }
            }
        )
        
        # Submit task
        await self.streams_manager.submit_task(task)
        
        # Track workflow context
        if workflow_id not in self.workflow_contexts:
            self.workflow_contexts[workflow_id] = {
                "steps": [],
                "agents": set(),
                "created_at": datetime.utcnow()
            }
        
        self.workflow_contexts[workflow_id]["steps"].append({
            "step_name": step_name,
            "agent_id": agent_id,
            "captured_at": datetime.utcnow(),
            "task_id": task.task_id
        })
        self.workflow_contexts[workflow_id]["agents"].add(agent_id)
        
        logger.info(f"Captured workflow step: {workflow_id}/{step_name}")
    
    async def compress_workflow_context(self, workflow_id: str) -> Optional[str]:
        """Compress the entire workflow context."""
        if workflow_id not in self.workflow_contexts:
            logger.warning(f"No context found for workflow {workflow_id}")
            return None
        
        # Create compression task
        task = SemanticMemoryTask(
            task_id=f"workflow_compress_{workflow_id}_{uuid.uuid4().hex[:8]}",
            task_type=SemanticMemoryTaskType.COMPRESS_CONTEXT,
            agent_id="workflow_manager",
            workflow_id=workflow_id,
            priority=ProcessingPriority.HIGH,
            payload={
                "context_id": f"workflow_context_{workflow_id}",
                "compression_method": CompressionMethod.HYBRID,
                "target_reduction": 0.6,
                "preserve_importance_threshold": 0.8,
                "agent_id": "workflow_manager"
            }
        )
        
        await self.streams_manager.submit_task(task)
        logger.info(f"Started context compression for workflow {workflow_id}")
        
        return task.task_id
    
    async def get_workflow_summary(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of the workflow from semantic memory."""
        if workflow_id not in self.workflow_contexts:
            return None
        
        context_info = self.workflow_contexts[workflow_id]
        
        # Search for workflow-related content
        search_task = SemanticMemoryTask(
            task_id=f"workflow_summary_{workflow_id}_{uuid.uuid4().hex[:8]}",
            task_type=SemanticMemoryTaskType.SEARCH_SEMANTIC,
            agent_id="workflow_manager",
            workflow_id=workflow_id,
            payload={
                "query": f"workflow {workflow_id} summary results",
                "limit": 10,
                "similarity_threshold": 0.6,
                "workflow_id": workflow_id,
                "filters": {
                    "tags": ["workflow", workflow_id]
                }
            }
        )
        
        await self.streams_manager.submit_task(search_task)
        
        return {
            "workflow_id": workflow_id,
            "total_steps": len(context_info["steps"]),
            "involved_agents": list(context_info["agents"]),
            "created_at": context_info["created_at"].isoformat(),
            "search_task_id": search_task.task_id
        }


# =============================================================================
# INTEGRATION SERVICE
# =============================================================================

class SemanticMemoryIntegrationService:
    """Main service for semantic memory integration."""
    
    def __init__(self, redis_client: Redis, memory_service_url: str):
        self.redis = redis_client
        self.memory_service_url = memory_service_url
        self.streams_manager = SemanticMemoryStreamsManager(redis_client, memory_service_url)
        self.workflow_memory_manager = WorkflowMemoryManager(self.streams_manager)
        self.running = False
    
    async def start(self):
        """Start the integration service."""
        logger.info("Starting Semantic Memory Integration Service")
        
        await self.streams_manager.start()
        self.running = True
        
        logger.info("Semantic Memory Integration Service started successfully")
    
    async def stop(self):
        """Stop the integration service."""
        logger.info("Stopping Semantic Memory Integration Service")
        
        self.running = False
        await self.streams_manager.stop()
        
        logger.info("Semantic Memory Integration Service stopped")
    
    def create_ingest_node(self, node_id: str) -> DocumentIngestWorkflowNode:
        """Create a document ingestion workflow node."""
        return DocumentIngestWorkflowNode(node_id, self.streams_manager)
    
    def create_search_node(self, node_id: str) -> SemanticSearchWorkflowNode:
        """Create a semantic search workflow node."""
        return SemanticSearchWorkflowNode(node_id, self.streams_manager)
    
    def create_compression_node(self, node_id: str) -> ContextCompressionWorkflowNode:
        """Create a context compression workflow node."""
        return ContextCompressionWorkflowNode(node_id, self.streams_manager)
    
    def create_knowledge_node(self, node_id: str) -> AgentKnowledgeWorkflowNode:
        """Create an agent knowledge workflow node."""
        return AgentKnowledgeWorkflowNode(node_id, self.streams_manager)
    
    async def submit_memory_task(self, task: SemanticMemoryTask) -> str:
        """Submit a semantic memory task."""
        return await self.streams_manager.submit_task(task)
    
    async def capture_workflow_step(self, workflow_id: str, step_name: str,
                                  step_output: Dict[str, Any], agent_id: str):
        """Capture workflow step context."""
        await self.workflow_memory_manager.capture_workflow_step(
            workflow_id, step_name, step_output, agent_id
        )
    
    async def get_workflow_summary(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow summary from semantic memory."""
        return await self.workflow_memory_manager.get_workflow_summary(workflow_id)


# =============================================================================
# GLOBAL INTEGRATION INSTANCE
# =============================================================================

_semantic_memory_integration: Optional[SemanticMemoryIntegrationService] = None


async def get_semantic_memory_integration() -> SemanticMemoryIntegrationService:
    """Get the global semantic memory integration service instance."""
    global _semantic_memory_integration
    
    if _semantic_memory_integration is None:
        redis_client = get_redis()
        # Use mock server URL for development, production URL when available
        memory_service_url = "http://localhost:8001/api/v1"  # Mock server
        
        _semantic_memory_integration = SemanticMemoryIntegrationService(
            redis_client, 
            memory_service_url
        )
        
        await _semantic_memory_integration.start()
    
    return _semantic_memory_integration


async def shutdown_semantic_memory_integration():
    """Shutdown the global semantic memory integration service."""
    global _semantic_memory_integration
    
    if _semantic_memory_integration:
        await _semantic_memory_integration.stop()
        _semantic_memory_integration = None


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

async def example_workflow_with_semantic_memory():
    """Example of using semantic memory in a DAG workflow."""
    integration = await get_semantic_memory_integration()
    
    # Create workflow nodes
    ingest_node = integration.create_ingest_node("ingest_analysis")
    search_node = integration.create_search_node("search_patterns")
    compress_node = integration.create_compression_node("compress_context")
    
    # Simulate workflow execution
    workflow_id = str(uuid.uuid4())
    agent_id = "example_orchestrator"
    
    # Step 1: Ingest analysis results
    ingest_context = WorkflowContext(
        workflow_id=workflow_id,
        agent_id=agent_id,
        current_step=1,
        input_data={
            "content": "Analysis completed: Found 5 coordination patterns in agent behavior",
            "tags": ["analysis", "patterns", "coordination"],
            "importance": 0.9,
            "generate_summary": True
        }
    )
    
    ingest_result = await ingest_node.execute(ingest_context)
    print(f"Ingest Result: {ingest_result.success}")
    
    # Capture workflow step
    await integration.capture_workflow_step(
        workflow_id, "analysis_ingestion", 
        {"document_id": ingest_result.output_data.get("document_id")},
        agent_id
    )
    
    # Step 2: Search for related patterns
    search_context = WorkflowContext(
        workflow_id=workflow_id,
        agent_id=agent_id,
        current_step=2,
        input_data={
            "query": "coordination patterns in multi-agent systems",
            "limit": 5,
            "similarity_threshold": 0.7
        }
    )
    
    search_result = await search_node.execute(search_context)
    print(f"Search Result: {len(search_result.output_data.get('results', []))} results found")
    
    # Step 3: Compress workflow context
    compress_context = WorkflowContext(
        workflow_id=workflow_id,
        agent_id=agent_id,
        current_step=3,
        input_data={
            "context_id": f"analysis_workflow_{workflow_id}",
            "target_reduction": 0.6
        }
    )
    
    compress_result = await compress_node.execute(compress_context)
    print(f"Compression Result: {compress_result.output_data.get('compression_ratio', 0):.2f} ratio")
    
    # Get workflow summary
    workflow_summary = await integration.get_workflow_summary(workflow_id)
    print(f"Workflow Summary: {workflow_summary}")


if __name__ == "__main__":
    """Example usage of the semantic memory integration."""
    import asyncio
    
    async def main():
        await example_workflow_with_semantic_memory()
        await shutdown_semantic_memory_integration()
    
    asyncio.run(main())