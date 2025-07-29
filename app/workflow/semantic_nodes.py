"""
Semantic Memory DAG Node Types for LeanVibe Agent Hive 2.0

Implements intelligent workflow node types that integrate semantic memory operations
with the existing DAG workflow system, enabling context-aware task execution.

Node Types:
- SemanticSearchNode: Search semantic memory for relevant context
- ContextualizeNode: Inject context into task payload
- IngestMemoryNode: Store task results in semantic memory  
- CrossAgentKnowledgeNode: Access shared agent knowledge
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import structlog
import httpx

from ..core.database import get_session
from ..models.workflow import Workflow, WorkflowStatus
from ..models.task import Task, TaskStatus
from ..schemas.semantic_memory import (
    DocumentIngestRequest, SemanticSearchRequest, ContextCompressionRequest,
    AgentKnowledgeResponse, ProcessingPriority
)

logger = structlog.get_logger()


# =============================================================================
# BASE WORKFLOW NODE TYPES
# =============================================================================

@dataclass
class WorkflowContext:
    """Context information for workflow node execution."""
    workflow_id: str
    agent_id: str
    current_step: int
    total_steps: int
    input_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass 
class WorkflowResult:
    """Result of workflow node execution."""
    success: bool
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_injected: bool = False
    context_size_tokens: int = 0


class WorkflowNode(ABC):
    """Abstract base class for workflow nodes."""
    
    def __init__(self, node_id: str, node_type: str = "generic"):
        self.node_id = node_id
        self.node_type = node_type
        self.created_at = datetime.utcnow()
        self.execution_count = 0
        
    @abstractmethod
    async def execute(self, context: WorkflowContext) -> WorkflowResult:
        """Execute the workflow node with given context."""
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.node_id}, type={self.node_type})>"


# =============================================================================
# SEMANTIC MEMORY NODE TYPES
# =============================================================================

class SemanticNodeType(str, Enum):
    """Types of semantic memory workflow nodes."""
    SEMANTIC_SEARCH = "semantic_search"
    CONTEXTUALIZE = "contextualize"  
    INGEST_MEMORY = "ingest_memory"
    CROSS_AGENT_KNOWLEDGE = "cross_agent_knowledge"


@dataclass
class SemanticMemoryConfig:
    """Configuration for semantic memory service connection."""
    service_url: str = "http://semantic-memory-service:8001/api/v1"
    timeout_seconds: int = 30
    max_retries: int = 3
    enable_fallback: bool = True
    performance_targets: Dict[str, float] = field(default_factory=lambda: {
        "context_retrieval_ms": 50.0,
        "memory_task_processing_ms": 100.0, 
        "workflow_overhead_ms": 10.0
    })


class SemanticWorkflowNode(WorkflowNode):
    """Base class for semantic memory workflow nodes."""
    
    def __init__(self, node_id: str, node_type: SemanticNodeType, 
                 memory_config: SemanticMemoryConfig):
        super().__init__(node_id, node_type.value)
        self.semantic_type = node_type
        self.memory_config = memory_config
        self.http_client = httpx.AsyncClient(
            base_url=memory_config.service_url,
            timeout=memory_config.timeout_seconds
        )
        
        # Performance monitoring
        self.execution_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time_ms": 0.0,
            "context_injection_count": 0,
            "context_compression_ratio": 0.0
        }
        
    async def execute(self, context: WorkflowContext) -> WorkflowResult:
        """Execute semantic memory operation with monitoring."""
        start_time = time.time()
        self.execution_count += 1
        self.execution_metrics["total_executions"] += 1
        
        try:
            # Validate context and prepare request
            await self._validate_context(context)
            request_payload = await self._prepare_request_payload(context)
            
            # Execute semantic memory operation
            response_data = await self._execute_memory_operation(request_payload)
            
            # Process response and inject context if needed
            result = await self._process_response(response_data, context)
            
            # Update metrics
            execution_time_ms = (time.time() - start_time) * 1000
            result.execution_time_ms = execution_time_ms
            
            self._update_success_metrics(execution_time_ms)
            
            # Validate performance targets
            await self._validate_performance_targets(execution_time_ms)
            
            logger.info(
                f"✅ Semantic node {self.node_id} executed successfully",
                node_type=self.semantic_type.value,
                execution_time_ms=execution_time_ms,
                context_injected=result.context_injected,
                context_size_tokens=result.context_size_tokens
            )
            
            return result
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.execution_metrics["failed_executions"] += 1
            
            logger.error(
                f"❌ Semantic node {self.node_id} execution failed",
                node_type=self.semantic_type.value,
                error=str(e),
                execution_time_ms=execution_time_ms
            )
            
            return WorkflowResult(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time_ms,
                metadata={"node_id": self.node_id, "node_type": self.semantic_type.value}
            )
    
    @abstractmethod
    async def _prepare_request_payload(self, context: WorkflowContext) -> Dict[str, Any]:
        """Prepare request payload for semantic memory service."""
        pass
    
    @abstractmethod
    async def _execute_memory_operation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the specific semantic memory operation."""
        pass
    
    @abstractmethod 
    async def _process_response(self, response_data: Dict[str, Any], 
                              context: WorkflowContext) -> WorkflowResult:
        """Process semantic memory service response."""
        pass
    
    async def _validate_context(self, context: WorkflowContext) -> None:
        """Validate workflow context for semantic operations."""
        if not context.workflow_id:
            raise ValueError("workflow_id is required for semantic operations")
        if not context.agent_id:
            raise ValueError("agent_id is required for semantic operations")
    
    async def _validate_performance_targets(self, execution_time_ms: float) -> None:
        """Validate execution against performance targets."""
        target_key = {
            SemanticNodeType.SEMANTIC_SEARCH: "context_retrieval_ms",
            SemanticNodeType.CONTEXTUALIZE: "workflow_overhead_ms", 
            SemanticNodeType.INGEST_MEMORY: "memory_task_processing_ms",
            SemanticNodeType.CROSS_AGENT_KNOWLEDGE: "context_retrieval_ms"
        }.get(self.semantic_type, "memory_task_processing_ms")
        
        target_ms = self.memory_config.performance_targets.get(target_key, 100.0)
        
        if execution_time_ms > target_ms:
            logger.warning(
                f"Performance target exceeded",
                node_id=self.node_id,
                target_ms=target_ms,
                actual_ms=execution_time_ms,
                overage_percent=((execution_time_ms - target_ms) / target_ms) * 100
            )
    
    def _update_success_metrics(self, execution_time_ms: float) -> None:
        """Update success metrics."""
        self.execution_metrics["successful_executions"] += 1
        
        # Update average execution time
        current_avg = self.execution_metrics["average_execution_time_ms"]
        total_executions = self.execution_metrics["total_executions"]
        
        new_avg = ((current_avg * (total_executions - 1)) + execution_time_ms) / total_executions
        self.execution_metrics["average_execution_time_ms"] = new_avg
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.http_client.aclose()


# =============================================================================
# SEMANTIC SEARCH NODE
# =============================================================================

class SemanticSearchNode(SemanticWorkflowNode):
    """Search semantic memory for relevant context."""
    
    def __init__(self, node_id: str, memory_config: SemanticMemoryConfig,
                 default_limit: int = 5, default_threshold: float = 0.7):
        super().__init__(node_id, SemanticNodeType.SEMANTIC_SEARCH, memory_config)
        self.default_limit = default_limit
        self.default_threshold = default_threshold
    
    async def _prepare_request_payload(self, context: WorkflowContext) -> Dict[str, Any]:
        """Prepare semantic search request payload."""
        input_data = context.input_data
        
        # Extract search query from context
        query = input_data.get("query")
        if not query:
            # Generate query from workflow context
            query = self._generate_contextual_query(context)
        
        return {
            "query": query,
            "limit": input_data.get("limit", self.default_limit),
            "similarity_threshold": input_data.get("similarity_threshold", self.default_threshold),
            "agent_id": context.agent_id,
            "workflow_id": context.workflow_id,
            "filters": input_data.get("filters", {}),
            "search_options": {
                "rerank": input_data.get("enable_rerank", True),
                "include_metadata": True,
                "explain_relevance": input_data.get("explain_relevance", True)
            }
        }
    
    async def _execute_memory_operation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute semantic search operation."""
        response = await self.http_client.post("/memory/search", json=payload)
        response.raise_for_status()
        return response.json()
    
    async def _process_response(self, response_data: Dict[str, Any], 
                              context: WorkflowContext) -> WorkflowResult:
        """Process semantic search response."""
        results = response_data.get("results", [])
        
        # Calculate context size in tokens (rough estimate)
        context_size_tokens = sum(len(result.get("content", "").split()) for result in results)
        
        return WorkflowResult(
            success=True,
            output_data={
                "search_results": results,
                "total_found": response_data.get("total_found", 0),
                "search_time_ms": response_data.get("search_time_ms", 0),
                "context_extracted": True
            },
            context_injected=True,
            context_size_tokens=context_size_tokens,
            metadata={
                "node_id": self.node_id,
                "search_query": response_data.get("query", ""),
                "results_count": len(results)
            }
        )
    
    def _generate_contextual_query(self, context: WorkflowContext) -> str:
        """Generate search query from workflow context."""
        # Use workflow metadata and previous step outputs to generate query
        workflow_id = context.workflow_id
        agent_id = context.agent_id
        step = context.current_step
        
        # Extract key terms from input data
        input_terms = []
        if isinstance(context.input_data, dict):
            for key, value in context.input_data.items():
                if isinstance(value, str) and len(value) < 100:  # Short strings likely keywords
                    input_terms.append(value)
        
        if input_terms:
            query = f"workflow {workflow_id} agent {agent_id} " + " ".join(input_terms[:5])
        else:
            query = f"workflow {workflow_id} step {step} agent {agent_id} context"
        
        return query


# =============================================================================
# CONTEXTUALIZE NODE  
# =============================================================================

class ContextualizeNode(SemanticWorkflowNode):
    """Inject relevant context into task payload."""
    
    def __init__(self, node_id: str, memory_config: SemanticMemoryConfig,
                 max_context_tokens: int = 2000, compression_threshold: float = 0.7):
        super().__init__(node_id, SemanticNodeType.CONTEXTUALIZE, memory_config)
        self.max_context_tokens = max_context_tokens
        self.compression_threshold = compression_threshold
    
    async def _prepare_request_payload(self, context: WorkflowContext) -> Dict[str, Any]:
        """Prepare contextualization request payload.""" 
        input_data = context.input_data
        
        # Get base task content
        base_content = input_data.get("base_task", input_data.get("content", ""))
        
        # Get context documents to use
        context_documents = input_data.get("context_documents", [])
        if not context_documents:
            # Search for relevant context automatically
            search_query = input_data.get("context_query", base_content[:200])
            context_documents = await self._find_relevant_context(search_query, context)
        
        return {
            "content": base_content,
            "context_documents": context_documents,
            "contextualization_method": input_data.get("method", "attention_based"),
            "agent_id": context.agent_id
        }
    
    async def _execute_memory_operation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute contextualization operation."""
        response = await self.http_client.post("/memory/contextualize", json=payload)
        response.raise_for_status()
        return response.json()
    
    async def _process_response(self, response_data: Dict[str, Any], 
                              context: WorkflowContext) -> WorkflowResult:
        """Process contextualization response."""
        contextual_embedding = response_data.get("contextual_embedding", [])
        context_influence = response_data.get("context_influence_scores", {})
        
        # Estimate context size  
        context_size_tokens = len(contextual_embedding) if contextual_embedding else 0
        
        # Apply token compression if needed
        compressed_context = contextual_embedding
        compression_ratio = 1.0
        
        if context_size_tokens > self.max_context_tokens:
            compressed_context, compression_ratio = await self._compress_context(
                contextual_embedding, context
            )
            context_size_tokens = len(compressed_context)
            self.execution_metrics["context_compression_ratio"] = compression_ratio
        
        # Inject context into original input data
        enhanced_input_data = context.input_data.copy()
        enhanced_input_data["_injected_context"] = {
            "contextual_embedding": compressed_context,
            "context_influence_scores": context_influence,
            "context_summary": response_data.get("context_summary", ""),
            "compression_ratio": compression_ratio
        }
        
        self.execution_metrics["context_injection_count"] += 1
        
        return WorkflowResult(
            success=True,
            output_data={
                "enhanced_task_data": enhanced_input_data,
                "context_embedding": compressed_context,
                "context_influence": context_influence,
                "compression_applied": compression_ratio < 1.0,
                "compression_ratio": compression_ratio
            },
            context_injected=True,
            context_size_tokens=context_size_tokens,
            metadata={
                "node_id": self.node_id,
                "original_context_size": len(contextual_embedding) if contextual_embedding else 0,
                "compressed_context_size": context_size_tokens
            }
        )
    
    async def _find_relevant_context(self, query: str, context: WorkflowContext) -> List[str]:
        """Find relevant context documents automatically."""
        try:
            search_payload = {
                "query": query,
                "limit": 5,
                "similarity_threshold": 0.6,
                "agent_id": context.agent_id,
                "workflow_id": context.workflow_id
            }
            
            response = await self.http_client.post("/memory/search", json=search_payload)
            response.raise_for_status()
            
            search_results = response.json()
            return [result.get("document_id") for result in search_results.get("results", [])]
            
        except Exception as e:
            logger.warning(f"Failed to find relevant context automatically: {e}")
            return []
    
    async def _compress_context(self, context_embedding: List[float], 
                              context: WorkflowContext) -> tuple[List[float], float]:
        """Compress context using semantic memory service."""
        try:
            compression_payload = {
                "context_id": f"dynamic_context_{context.workflow_id}_{uuid.uuid4().hex[:8]}",
                "compression_method": "semantic_clustering",
                "target_reduction": self.compression_threshold,
                "preserve_importance_threshold": 0.8,
                "agent_id": context.agent_id
            }
            
            response = await self.http_client.post("/memory/compress", json=compression_payload)
            response.raise_for_status()
            
            compression_result = response.json()
            compression_ratio = compression_result.get("compression_ratio", 1.0)
            
            # Apply compression ratio to embedding (simplified)
            compressed_size = int(len(context_embedding) * compression_ratio)
            compressed_embedding = context_embedding[:compressed_size]
            
            return compressed_embedding, compression_ratio
            
        except Exception as e:
            logger.warning(f"Context compression failed, using original: {e}")
            return context_embedding, 1.0


# =============================================================================
# INGEST MEMORY NODE
# =============================================================================

class IngestMemoryNode(SemanticWorkflowNode):
    """Store task results in semantic memory."""
    
    def __init__(self, node_id: str, memory_config: SemanticMemoryConfig,
                 auto_tag: bool = True, importance_threshold: float = 0.5):
        super().__init__(node_id, SemanticNodeType.INGEST_MEMORY, memory_config)
        self.auto_tag = auto_tag
        self.importance_threshold = importance_threshold
    
    async def _prepare_request_payload(self, context: WorkflowContext) -> Dict[str, Any]:
        """Prepare memory ingestion request payload."""
        input_data = context.input_data
        
        # Extract content to store
        content = input_data.get("content")
        if not content:
            # Generate content from task results
            content = self._generate_content_from_context(context)
        
        # Generate tags
        tags = input_data.get("tags", [])
        if self.auto_tag:
            auto_tags = self._generate_auto_tags(context)
            tags.extend(auto_tags)
        
        # Determine importance
        importance = input_data.get("importance", self._calculate_importance(context))
        
        return {
            "content": content,
            "metadata": {
                "title": input_data.get("title", f"Workflow {context.workflow_id} Step {context.current_step}"),
                "source": f"workflow_step_{context.current_step}",
                "importance": importance,
                "workflow_id": context.workflow_id,
                "execution_id": context.execution_id,
                "step_number": context.current_step,
                **input_data.get("metadata", {})
            },
            "agent_id": context.agent_id,
            "workflow_id": context.workflow_id,
            "tags": list(set(tags)),  # Remove duplicates
            "processing_options": {
                "generate_summary": input_data.get("generate_summary", True),
                "extract_entities": input_data.get("extract_entities", False),
                "priority": ProcessingPriority.NORMAL
            }
        }
    
    async def _execute_memory_operation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute memory ingestion operation."""
        response = await self.http_client.post("/memory/ingest", json=payload)
        response.raise_for_status()
        return response.json()
    
    async def _process_response(self, response_data: Dict[str, Any], 
                              context: WorkflowContext) -> WorkflowResult:
        """Process memory ingestion response."""
        document_id = response_data.get("document_id")
        embedding_id = response_data.get("embedding_id")
        processing_time_ms = response_data.get("processing_time_ms", 0)
        
        return WorkflowResult(
            success=True,
            output_data={
                "document_id": document_id,
                "embedding_id": embedding_id,
                "ingestion_time_ms": processing_time_ms,
                "index_updated": response_data.get("index_updated", False),
                "vector_dimensions": response_data.get("vector_dimensions", 0)
            },
            metadata={
                "node_id": self.node_id,
                "document_stored": True,
                "summary_generated": bool(response_data.get("summary"))
            }
        )
    
    def _generate_content_from_context(self, context: WorkflowContext) -> str:
        """Generate content from workflow context."""
        content_parts = [
            f"Workflow: {context.workflow_id}",
            f"Agent: {context.agent_id}",  
            f"Step: {context.current_step}/{context.total_steps}",
            f"Timestamp: {context.timestamp.isoformat()}"
        ]
        
        if context.input_data:
            content_parts.append("Input Data:")
            content_parts.append(json.dumps(context.input_data, indent=2))
        
        if context.metadata:
            content_parts.append("Metadata:")
            content_parts.append(json.dumps(context.metadata, indent=2))
        
        return "\n\n".join(content_parts)
    
    def _generate_auto_tags(self, context: WorkflowContext) -> List[str]:
        """Generate automatic tags from context."""
        tags = [
            "workflow",
            f"workflow_{context.workflow_id}",
            f"agent_{context.agent_id}",
            f"step_{context.current_step}",
            "dag_generated"
        ]
        
        # Add tags based on input data
        if context.input_data:
            for key, value in context.input_data.items():
                if isinstance(value, str) and len(value) < 50:
                    # Short strings are likely tags or keywords
                    tags.append(f"{key}_{value}".lower().replace(" ", "_"))
        
        return tags
    
    def _calculate_importance(self, context: WorkflowContext) -> float:
        """Calculate content importance from context."""
        importance = self.importance_threshold
        
        # Increase importance for later steps
        step_factor = min(context.current_step / context.total_steps, 1.0)
        importance += step_factor * 0.3
        
        # Increase importance if execution_id suggests this is a critical execution
        if context.execution_id and "critical" in context.execution_id.lower():
            importance += 0.2
        
        # Increase importance based on input data size (more data = more important)
        if context.input_data:
            data_size = len(json.dumps(context.input_data))
            if data_size > 1000:  # Large payloads are more important
                importance += 0.1
        
        return min(importance, 1.0)  # Cap at 1.0


# =============================================================================
# CROSS AGENT KNOWLEDGE NODE
# =============================================================================

class CrossAgentKnowledgeNode(SemanticWorkflowNode):
    """Access shared agent knowledge across the system."""
    
    def __init__(self, node_id: str, memory_config: SemanticMemoryConfig,
                 knowledge_types: List[str] = None, time_ranges: List[str] = None):
        super().__init__(node_id, SemanticNodeType.CROSS_AGENT_KNOWLEDGE, memory_config)
        self.default_knowledge_types = knowledge_types or ["patterns", "interactions", "consolidated"]
        self.default_time_ranges = time_ranges or ["1h", "24h", "7d"]
    
    async def _prepare_request_payload(self, context: WorkflowContext) -> Dict[str, Any]:
        """Prepare cross-agent knowledge request payload."""
        input_data = context.input_data
        
        # Determine target agents
        target_agents = input_data.get("target_agents", [])
        if not target_agents:
            target_agents = [context.agent_id]  # Default to current agent
        
        # Prepare requests for multiple agents
        agent_requests = []
        for agent_id in target_agents:
            agent_requests.append({
                "agent_id": agent_id,
                "knowledge_type": input_data.get("knowledge_type", "all"),
                "time_range": input_data.get("time_range", "7d")
            })
        
        return {
            "agent_requests": agent_requests,
            "consolidate_results": input_data.get("consolidate_results", True),
            "include_cross_references": input_data.get("include_cross_references", True)
        }
    
    async def _execute_memory_operation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cross-agent knowledge retrieval."""
        agent_requests = payload["agent_requests"]
        
        # Execute requests for all agents in parallel
        tasks = []
        for request in agent_requests:
            agent_id = request["agent_id"] 
            params = {k: v for k, v in request.items() if k != "agent_id"}
            
            task = self.http_client.get(f"/memory/agent-knowledge/{agent_id}", params=params)
            tasks.append(task)
        
        # Wait for all requests to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process responses
        agent_knowledge = {}
        for i, response in enumerate(responses):
            agent_id = agent_requests[i]["agent_id"]
            
            if isinstance(response, Exception):
                logger.warning(f"Failed to get knowledge for agent {agent_id}: {response}")
                agent_knowledge[agent_id] = {"error": str(response)}
            else:
                try:
                    response.raise_for_status()
                    agent_knowledge[agent_id] = response.json()
                except Exception as e:
                    agent_knowledge[agent_id] = {"error": str(e)}
        
        # Consolidate results if requested
        consolidated = None
        if payload.get("consolidate_results", True):
            consolidated = self._consolidate_agent_knowledge(agent_knowledge)
        
        return {
            "agent_knowledge": agent_knowledge,
            "consolidated_knowledge": consolidated,
            "agents_queried": len(agent_requests),
            "successful_queries": sum(1 for knowledge in agent_knowledge.values() if "error" not in knowledge)
        }
    
    async def _process_response(self, response_data: Dict[str, Any], 
                              context: WorkflowContext) -> WorkflowResult:
        """Process cross-agent knowledge response."""
        agent_knowledge = response_data.get("agent_knowledge", {})
        consolidated = response_data.get("consolidated_knowledge", {})
        
        # Calculate context size
        context_size_tokens = sum(
            len(json.dumps(knowledge).split()) 
            for knowledge in agent_knowledge.values()
        )
        
        # Extract key insights
        key_insights = []
        cross_agent_patterns = []
        
        if consolidated:
            key_insights = consolidated.get("key_insights", [])
            cross_agent_patterns = consolidated.get("cross_agent_patterns", [])
        
        return WorkflowResult(
            success=True,
            output_data={
                "agent_knowledge": agent_knowledge,
                "consolidated_knowledge": consolidated,
                "key_insights": key_insights,
                "cross_agent_patterns": cross_agent_patterns,
                "agents_queried": response_data.get("agents_queried", 0),
                "successful_queries": response_data.get("successful_queries", 0)
            },
            context_injected=True,
            context_size_tokens=context_size_tokens,
            metadata={
                "node_id": self.node_id,
                "knowledge_sharing_enabled": True,
                "consolidation_applied": bool(consolidated)
            }
        )
    
    def _consolidate_agent_knowledge(self, agent_knowledge: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate knowledge from multiple agents."""
        consolidated = {
            "key_insights": [],
            "expertise_areas": set(),
            "cross_agent_patterns": [],
            "common_interactions": [],
            "knowledge_confidence": 0.0
        }
        
        valid_knowledge = {
            agent_id: knowledge for agent_id, knowledge in agent_knowledge.items()
            if "error" not in knowledge
        }
        
        if not valid_knowledge:
            return consolidated
        
        # Extract insights from all agents
        for agent_id, knowledge in valid_knowledge.items():
            knowledge_base = knowledge.get("knowledge_base", {})
            
            # Collect insights
            patterns = knowledge_base.get("patterns", [])
            for pattern in patterns:
                insight = f"Agent {agent_id}: {pattern.get('description', '')}"
                consolidated["key_insights"].append(insight)
            
            # Collect expertise areas  
            expertise = knowledge_base.get("consolidated_knowledge", {}).get("expertise_areas", [])
            consolidated["expertise_areas"].update(expertise)
            
            # Collect interactions
            interactions = knowledge_base.get("interactions", [])
            consolidated["common_interactions"].extend(interactions)
        
        # Find cross-agent patterns
        agent_patterns = {}
        for agent_id, knowledge in valid_knowledge.items():
            patterns = knowledge.get("knowledge_base", {}).get("patterns", [])
            agent_patterns[agent_id] = [p.get("description", "") for p in patterns]
        
        # Look for common patterns across agents
        for agent1, patterns1 in agent_patterns.items():
            for agent2, patterns2 in agent_patterns.items():
                if agent1 >= agent2:  # Avoid duplicates
                    continue
                
                common_patterns = set(patterns1) & set(patterns2)
                for pattern in common_patterns:
                    consolidated["cross_agent_patterns"].append({
                        "pattern": pattern,
                        "agents": [agent1, agent2],
                        "confidence": 0.8  # Simple confidence score
                    })
        
        # Calculate overall confidence
        total_confidence = sum(
            knowledge.get("knowledge_stats", {}).get("knowledge_confidence", 0.5)
            for knowledge in valid_knowledge.values()
        )
        consolidated["knowledge_confidence"] = total_confidence / len(valid_knowledge)
        
        # Convert set to list for JSON serialization
        consolidated["expertise_areas"] = list(consolidated["expertise_areas"])
        
        return consolidated


# =============================================================================
# NODE FACTORY
# =============================================================================

class SemanticNodeFactory:
    """Factory for creating semantic memory workflow nodes."""
    
    def __init__(self, memory_config: SemanticMemoryConfig):
        self.memory_config = memory_config
        self.created_nodes = {}
    
    def create_semantic_search_node(self, node_id: str, **kwargs) -> SemanticSearchNode:
        """Create a semantic search node."""
        node = SemanticSearchNode(node_id, self.memory_config, **kwargs)
        self.created_nodes[node_id] = node
        return node
    
    def create_contextualize_node(self, node_id: str, **kwargs) -> ContextualizeNode:
        """Create a contextualize node."""
        node = ContextualizeNode(node_id, self.memory_config, **kwargs)
        self.created_nodes[node_id] = node
        return node
    
    def create_ingest_memory_node(self, node_id: str, **kwargs) -> IngestMemoryNode:
        """Create an ingest memory node."""
        node = IngestMemoryNode(node_id, self.memory_config, **kwargs)
        self.created_nodes[node_id] = node
        return node
    
    def create_cross_agent_knowledge_node(self, node_id: str, **kwargs) -> CrossAgentKnowledgeNode:
        """Create a cross-agent knowledge node."""
        node = CrossAgentKnowledgeNode(node_id, self.memory_config, **kwargs)
        self.created_nodes[node_id] = node
        return node
    
    def get_node(self, node_id: str) -> Optional[SemanticWorkflowNode]:
        """Get a created node by ID."""
        return self.created_nodes.get(node_id)
    
    def get_all_nodes(self) -> Dict[str, SemanticWorkflowNode]:
        """Get all created nodes."""
        return self.created_nodes.copy()
    
    async def cleanup_all_nodes(self) -> None:
        """Cleanup all created nodes."""
        for node in self.created_nodes.values():
            await node.cleanup()
        self.created_nodes.clear()


# =============================================================================
# USAGE EXAMPLES AND WORKFLOW INTEGRATION
# =============================================================================

async def create_intelligent_development_workflow() -> List[SemanticWorkflowNode]:
    """Example: Create an intelligent development workflow with semantic memory."""
    
    # Initialize memory configuration
    memory_config = SemanticMemoryConfig(
        service_url="http://semantic-memory-service:8001/api/v1",
        timeout_seconds=30,
        performance_targets={
            "context_retrieval_ms": 50.0,
            "memory_task_processing_ms": 100.0,
            "workflow_overhead_ms": 10.0
        }
    )
    
    # Create node factory
    factory = SemanticNodeFactory(memory_config)
    
    # Create workflow nodes
    nodes = [
        # Search for similar requirements analysis
        factory.create_semantic_search_node(
            "analyze-requirements",
            default_limit=5,
            default_threshold=0.7
        ),
        
        # Inject context into system design task
        factory.create_contextualize_node(
            "design-system",
            max_context_tokens=2000,
            compression_threshold=0.7
        ),
        
        # Store design document in memory
        factory.create_ingest_memory_node(
            "store-design",
            auto_tag=True,
            importance_threshold=0.8
        ),
        
        # Access shared knowledge from other agents
        factory.create_cross_agent_knowledge_node(
            "cross-agent-knowledge",
            knowledge_types=["patterns", "best_practices"],
            time_ranges=["7d", "30d"]
        )
    ]
    
    logger.info(f"Created intelligent development workflow with {len(nodes)} semantic nodes")
    return nodes


if __name__ == "__main__":
    """Example usage of semantic DAG nodes."""
    async def main():
        nodes = await create_intelligent_development_workflow()
        
        # Example workflow context
        context = WorkflowContext(
            workflow_id="intelligent-dev-workflow-001",
            agent_id="senior-architect",
            current_step=1,
            total_steps=4,
            input_data={
                "query": "microservices architecture patterns for high scalability",
                "content": "Design a microservices architecture for a high-traffic application",
                "requirements": ["scalability", "fault-tolerance", "monitoring"]
            }
        )
        
        # Execute first node (semantic search)
        search_node = nodes[0]
        result = await search_node.execute(context)
        
        if result.success:
            print(f"Search found {len(result.output_data.get('search_results', []))} relevant documents")
            print(f"Context size: {result.context_size_tokens} tokens")
        
        # Cleanup
        factory = SemanticNodeFactory(SemanticMemoryConfig())
        await factory.cleanup_all_nodes()
    
    import asyncio
    asyncio.run(main())