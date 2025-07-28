# Semantic Memory Service API Documentation

## Overview

The LeanVibe Semantic Memory Service provides advanced semantic search, context management, and memory consolidation capabilities for multi-agent orchestration systems. This document provides comprehensive integration patterns, code examples, and best practices for subagent development.

## Table of Contents

- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Integration Patterns](#integration-patterns)
- [DAG Workflow Integration](#dag-workflow-integration)
- [Performance Optimization](#performance-optimization)
- [Error Handling](#error-handling)
- [Code Examples](#code-examples)
- [Best Practices](#best-practices)

## Quick Start

### Mock Server Setup

For parallel development, use the mock server:

```bash
# Start mock server on port 8001
cd /Users/bogdan/work/leanvibe-dev/bee-hive
python -m mock_servers.semantic_memory_mock

# Verify mock server is running
curl http://localhost:8001/api/v1/memory/health
```

### Basic Usage Example

```python
import httpx
import asyncio
from uuid import uuid4

# Initialize client
client = httpx.AsyncClient(base_url="http://localhost:8001/api/v1")

async def basic_example():
    # Ingest a document
    ingest_response = await client.post("/memory/ingest", json={
        "content": "Agent coordination patterns in distributed systems",
        "agent_id": "orchestrator-001",
        "tags": ["coordination", "patterns"],
        "metadata": {
            "importance": 0.8,
            "type": "technical_knowledge"
        }
    })
    
    document_id = ingest_response.json()["document_id"]
    print(f"Document ingested: {document_id}")
    
    # Search for related content
    search_response = await client.post("/memory/search", json={
        "query": "how do agents coordinate?",
        "limit": 5,
        "similarity_threshold": 0.7
    })
    
    results = search_response.json()["results"]
    print(f"Found {len(results)} related documents")

# Run example
asyncio.run(basic_example())
```

## API Endpoints

### Document Management

#### POST /memory/ingest
Ingest a single document into semantic memory.

**Request:**
```json
{
  "content": "Agent coordination requires careful message ordering",
  "agent_id": "orchestrator-001",
  "workflow_id": "12345678-1234-5678-9012-123456789abc",
  "tags": ["coordination", "messaging"],
  "metadata": {
    "importance": 0.8,
    "source": "technical_documentation"
  },
  "processing_options": {
    "generate_summary": true,
    "extract_entities": false,
    "priority": "high"
  }
}
```

**Response:**
```json
{
  "document_id": "doc_12345678-1234-5678-9012-123456789abc",
  "embedding_id": "emb_87654321-4321-8765-2109-876543210fed",
  "processing_time_ms": 45.3,
  "vector_dimensions": 1536,
  "index_updated": true,
  "summary": "Document about agent coordination and message ordering"
}
```

#### POST /memory/batch-ingest
Efficiently ingest multiple documents in a single operation.

**Request:**
```json
{
  "documents": [
    {
      "content": "Redis streams provide reliable message ordering",
      "agent_id": "messaging-service",
      "tags": ["redis", "messaging"]
    },
    {
      "content": "Context compression reduces memory usage by 70%",
      "agent_id": "context-optimizer", 
      "tags": ["compression", "optimization"]
    }
  ],
  "batch_options": {
    "parallel_processing": true,
    "generate_summary": true,
    "fail_on_error": false
  }
}
```

### Semantic Search

#### POST /memory/search
Perform advanced semantic search with filtering and reranking.

**Request:**
```json
{
  "query": "How do agents coordinate in distributed workflows?",
  "limit": 10,
  "similarity_threshold": 0.7,
  "agent_id": "orchestrator-001",
  "filters": {
    "tags": ["coordination", "workflows"],
    "importance_min": 0.5,
    "date_from": "2024-01-01T00:00:00Z"
  },
  "search_options": {
    "rerank": true,
    "include_metadata": true,
    "explain_relevance": true
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "document_id": "doc_12345",
      "content": "Agent coordination patterns require...",
      "similarity_score": 0.87,
      "metadata": {"type": "pattern", "importance": 0.8},
      "agent_id": "orchestrator-001",
      "tags": ["coordination"],
      "relevance_explanation": "High semantic similarity in coordination concepts"
    }
  ],
  "total_found": 25,
  "search_time_ms": 143.7,
  "query_embedding_time_ms": 12.4,
  "reranking_applied": true
}
```

### Context Management

#### POST /memory/compress
Compress and consolidate context to reduce memory usage.

**Request:**
```json
{
  "context_id": "ctx_workflow_analysis_2024",
  "compression_method": "semantic_clustering",
  "target_reduction": 0.7,
  "preserve_importance_threshold": 0.8,
  "agent_id": "context-optimizer",
  "compression_options": {
    "preserve_recent": true,
    "maintain_relationships": true,
    "generate_summary": true
  }
}
```

**Response:**
```json
{
  "compressed_context_id": "ctx_compressed_workflow_analysis_2024",
  "original_size": 15420,
  "compressed_size": 4626,
  "compression_ratio": 0.7,
  "semantic_preservation_score": 0.94,
  "processing_time_ms": 234.5,
  "compression_summary": "Applied semantic clustering, reduced size by 70%"
}
```

## Integration Patterns

### Pattern 1: Agent-Scoped Knowledge Management

```python
class AgentMemoryManager:
    """Manages semantic memory for a specific agent."""
    
    def __init__(self, agent_id: str, memory_client: httpx.AsyncClient):
        self.agent_id = agent_id
        self.client = memory_client
    
    async def store_knowledge(self, content: str, tags: List[str], 
                            importance: float = 0.5) -> str:
        """Store knowledge with agent-specific context."""
        response = await self.client.post("/memory/ingest", json={
            "content": content,
            "agent_id": self.agent_id,
            "tags": tags,
            "metadata": {
                "importance": importance,
                "stored_by": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        })
        return response.json()["document_id"]
    
    async def retrieve_relevant_knowledge(self, query: str, 
                                        limit: int = 5) -> List[Dict]:
        """Retrieve agent-specific relevant knowledge."""
        response = await self.client.post("/memory/search", json={
            "query": query,
            "limit": limit,
            "agent_id": self.agent_id,  # Scope to this agent
            "similarity_threshold": 0.6,
            "search_options": {
                "rerank": True,
                "include_metadata": True
            }
        })
        return response.json()["results"]
    
    async def compress_agent_context(self, context_id: str) -> Dict:
        """Compress agent's accumulated context."""
        response = await self.client.post("/memory/compress", json={
            "context_id": context_id,
            "compression_method": "semantic_clustering",
            "target_reduction": 0.6,
            "preserve_importance_threshold": 0.7,
            "agent_id": self.agent_id
        })
        return response.json()

# Usage example
async def agent_workflow_example():
    client = httpx.AsyncClient(base_url="http://localhost:8001/api/v1")
    memory_manager = AgentMemoryManager("orchestrator-001", client)
    
    # Store learned patterns
    doc_id = await memory_manager.store_knowledge(
        content="Agents should validate message ordering before processing",
        tags=["validation", "messaging", "patterns"],
        importance=0.9
    )
    
    # Later, retrieve relevant knowledge for decision making
    relevant_knowledge = await memory_manager.retrieve_relevant_knowledge(
        query="how to handle message validation?"
    )
    
    for knowledge in relevant_knowledge:
        print(f"Relevant: {knowledge['content']} (score: {knowledge['similarity_score']})")
```

### Pattern 2: Workflow-Integrated Context Management

```python
class WorkflowContextManager:
    """Integrates semantic memory with DAG workflows."""
    
    def __init__(self, workflow_id: str, memory_client: httpx.AsyncClient):
        self.workflow_id = workflow_id
        self.client = memory_client
        self.workflow_context = []
    
    async def capture_workflow_step(self, step_name: str, 
                                  step_output: str, agent_id: str):
        """Capture context from each workflow step."""
        doc_id = await self.client.post("/memory/ingest", json={
            "content": f"Step '{step_name}': {step_output}",
            "agent_id": agent_id,
            "workflow_id": self.workflow_id,
            "tags": ["workflow", "step_output", step_name],
            "metadata": {
                "step_name": step_name,
                "workflow_id": self.workflow_id,
                "capture_time": datetime.utcnow().isoformat()
            }
        })
        
        self.workflow_context.append({
            "step_name": step_name,
            "document_id": doc_id.json()["document_id"],
            "agent_id": agent_id
        })
        
        return doc_id.json()["document_id"]
    
    async def get_workflow_summary(self) -> str:
        """Generate contextual summary of entire workflow."""
        if not self.workflow_context:
            return "No workflow context captured"
        
        # Use contextualization to create workflow summary
        context_doc_ids = [step["document_id"] for step in self.workflow_context]
        
        response = await self.client.post("/memory/contextualize", json={
            "content": f"Summary of workflow {self.workflow_id}",
            "context_documents": context_doc_ids,
            "contextualization_method": "attention_based"
        })
        
        return response.json().get("context_summary", "Workflow summary generated")
    
    async def find_similar_workflows(self) -> List[Dict]:
        """Find workflows with similar patterns."""
        if not self.workflow_context:
            return []
        
        # Search for similar workflow patterns
        workflow_description = f"Workflow with steps: {[s['step_name'] for s in self.workflow_context]}"
        
        response = await self.client.post("/memory/search", json={
            "query": workflow_description,
            "limit": 5,
            "filters": {
                "tags": ["workflow"],
                "metadata_filters": {
                    "workflow_id": {"$ne": self.workflow_id}  # Exclude current workflow
                }
            }
        })
        
        return response.json()["results"]

# Usage in DAG workflow
async def dag_integration_example():
    client = httpx.AsyncClient(base_url="http://localhost:8001/api/v1")
    workflow_context = WorkflowContextManager("workflow_123", client)
    
    # Simulate workflow steps
    await workflow_context.capture_workflow_step(
        "data_analysis", 
        "Analyzed 1000 records, found 3 anomalies",
        "data-analyzer-agent"
    )
    
    await workflow_context.capture_workflow_step(
        "anomaly_processing",
        "Processed anomalies with 95% confidence",
        "anomaly-processor-agent"
    )
    
    # Get contextual workflow summary
    summary = await workflow_context.get_workflow_summary()
    print(f"Workflow Summary: {summary}")
    
    # Find similar workflow patterns
    similar_workflows = await workflow_context.find_similar_workflows()
    print(f"Found {len(similar_workflows)} similar workflows")
```

### Pattern 3: Real-time Knowledge Base Updates

```python
class RealTimeKnowledgeSync:
    """Synchronizes real-time agent learning with semantic memory."""
    
    def __init__(self, agent_id: str, memory_client: httpx.AsyncClient,
                 redis_client):
        self.agent_id = agent_id
        self.memory_client = memory_client
        self.redis_client = redis_client
        self.knowledge_buffer = []
    
    async def buffer_knowledge(self, content: str, tags: List[str], 
                             importance: float):
        """Buffer knowledge for batch processing."""
        self.knowledge_buffer.append({
            "content": content,
            "agent_id": self.agent_id,
            "tags": tags,
            "metadata": {
                "importance": importance,
                "buffered_at": datetime.utcnow().isoformat()
            }
        })
        
        # Process buffer when it reaches threshold
        if len(self.knowledge_buffer) >= 10:
            await self.flush_knowledge_buffer()
    
    async def flush_knowledge_buffer(self):
        """Flush buffered knowledge to semantic memory."""
        if not self.knowledge_buffer:
            return
        
        response = await self.memory_client.post("/memory/batch-ingest", json={
            "documents": self.knowledge_buffer,
            "batch_options": {
                "parallel_processing": True,
                "generate_summary": False,
                "fail_on_error": False
            }
        })
        
        result = response.json()
        print(f"Flushed {result['successful_ingestions']} knowledge items")
        
        # Publish update to Redis for other agents
        await self.redis_client.publish(
            f"agent_knowledge_updates:{self.agent_id}",
            json.dumps({
                "agent_id": self.agent_id,
                "documents_added": result["successful_ingestions"],
                "timestamp": datetime.utcnow().isoformat()
            })
        )
        
        self.knowledge_buffer.clear()
    
    async def subscribe_to_knowledge_updates(self):
        """Subscribe to knowledge updates from other agents."""
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe("agent_knowledge_updates:*")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                update_data = json.loads(message["data"])
                if update_data["agent_id"] != self.agent_id:
                    await self.handle_external_knowledge_update(update_data)
    
    async def handle_external_knowledge_update(self, update_data: Dict):
        """Handle knowledge updates from other agents."""
        # Search for potentially relevant knowledge from other agents
        response = await self.memory_client.post("/memory/search", json={
            "query": f"knowledge from {update_data['agent_id']}",
            "agent_id": update_data["agent_id"],
            "limit": 3,
            "similarity_threshold": 0.8
        })
        
        relevant_knowledge = response.json()["results"]
        if relevant_knowledge:
            print(f"Discovered {len(relevant_knowledge)} relevant items from {update_data['agent_id']}")
```

## DAG Workflow Integration

### Semantic Memory Node Types

```python
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class SemanticMemoryNode:
    """Base class for semantic memory DAG nodes."""
    node_id: str
    memory_client: httpx.AsyncClient
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the semantic memory operation."""
        raise NotImplementedError

class IngestNode(SemanticMemoryNode):
    """DAG node for document ingestion."""
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest content into semantic memory."""
        content = input_data.get("content", "")
        agent_id = input_data.get("agent_id", "unknown")
        tags = input_data.get("tags", [])
        
        response = await self.memory_client.post("/memory/ingest", json={
            "content": content,
            "agent_id": agent_id,
            "tags": tags + ["dag_generated"],
            "workflow_id": input_data.get("workflow_id"),
            "metadata": {
                "node_id": self.node_id,
                "dag_execution": True
            }
        })
        
        return {
            "document_id": response.json()["document_id"],
            "processing_time_ms": response.json()["processing_time_ms"],
            "content": content
        }

class SearchNode(SemanticMemoryNode):
    """DAG node for semantic search."""
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic search."""
        query = input_data.get("query", "")
        limit = input_data.get("limit", 5)
        
        response = await self.memory_client.post("/memory/search", json={
            "query": query,
            "limit": limit,
            "similarity_threshold": input_data.get("similarity_threshold", 0.7),
            "agent_id": input_data.get("agent_id"),
            "search_options": {
                "rerank": True,
                "include_metadata": True
            }
        })
        
        results = response.json()["results"]
        
        return {
            "search_results": results,
            "total_found": len(results),
            "search_time_ms": response.json()["search_time_ms"],
            "query": query
        }

class CompressionNode(SemanticMemoryNode):
    """DAG node for context compression."""
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress workflow context."""
        context_id = input_data.get("context_id", f"workflow_{input_data.get('workflow_id', 'unknown')}")
        
        response = await self.memory_client.post("/memory/compress", json={
            "context_id": context_id,
            "compression_method": input_data.get("compression_method", "semantic_clustering"),
            "target_reduction": input_data.get("target_reduction", 0.6),
            "preserve_importance_threshold": 0.8,
            "agent_id": input_data.get("agent_id", "dag_executor")
        })
        
        result = response.json()
        
        return {
            "compressed_context_id": result["compressed_context_id"],
            "compression_ratio": result["compression_ratio"],
            "semantic_preservation_score": result["semantic_preservation_score"],
            "original_size": result["original_size"],
            "compressed_size": result["compressed_size"]
        }

# DAG workflow definition example
async def create_semantic_memory_dag():
    """Create a DAG workflow with semantic memory operations."""
    
    memory_client = httpx.AsyncClient(base_url="http://localhost:8001/api/v1")
    workflow_id = str(uuid4())
    
    # Define nodes
    ingest_node = IngestNode("ingest_analysis", memory_client)
    search_node = SearchNode("search_related", memory_client)
    compress_node = CompressionNode("compress_context", memory_client)
    
    # Execute workflow
    workflow_data = {
        "workflow_id": workflow_id,
        "agent_id": "dag_orchestrator"
    }
    
    # Step 1: Ingest analysis results
    ingest_result = await ingest_node.execute({
        **workflow_data,
        "content": "Analysis completed: Found 5 patterns in distributed coordination",
        "tags": ["analysis", "patterns", "coordination"]
    })
    
    # Step 2: Search for related knowledge
    search_result = await search_node.execute({
        **workflow_data,
        "query": "distributed coordination patterns",
        "limit": 3
    })
    
    # Step 3: Compress accumulated context
    compression_result = await compress_node.execute({
        **workflow_data,
        "context_id": f"analysis_workflow_{workflow_id}",
        "compression_method": "hybrid",
        "target_reduction": 0.7
    })
    
    return {
        "workflow_id": workflow_id,
        "ingest_result": ingest_result,
        "search_result": search_result,
        "compression_result": compression_result
    }
```

## Performance Optimization

### Batch Processing Strategies

```python
class OptimizedMemoryClient:
    """Optimized client for high-performance semantic memory operations."""
    
    def __init__(self, base_url: str, max_batch_size: int = 50):
        self.client = httpx.AsyncClient(base_url=base_url)
        self.max_batch_size = max_batch_size
        self.ingest_queue = []
        self.search_cache = {}
    
    async def optimized_ingest(self, documents: List[Dict], 
                             auto_flush: bool = True) -> List[str]:
        """Optimized document ingestion with batching."""
        self.ingest_queue.extend(documents)
        
        if auto_flush and len(self.ingest_queue) >= self.max_batch_size:
            return await self.flush_ingest_queue()
        
        return []
    
    async def flush_ingest_queue(self) -> List[str]:
        """Flush the ingestion queue."""
        if not self.ingest_queue:
            return []
        
        # Process in batches
        document_ids = []
        
        for i in range(0, len(self.ingest_queue), self.max_batch_size):
            batch = self.ingest_queue[i:i + self.max_batch_size]
            
            response = await self.client.post("/memory/batch-ingest", json={
                "documents": batch,
                "batch_options": {
                    "parallel_processing": True,
                    "fail_on_error": False
                }
            })
            
            result = response.json()
            batch_ids = [r["document_id"] for r in result["results"] 
                        if r["status"] == "success"]
            document_ids.extend(batch_ids)
        
        self.ingest_queue.clear()
        return document_ids
    
    async def cached_search(self, query: str, cache_ttl: int = 300) -> List[Dict]:
        """Search with caching for repeated queries."""
        cache_key = f"search:{hash(query)}"
        
        # Check cache
        if cache_key in self.search_cache:
            cached_result, timestamp = self.search_cache[cache_key]
            if time.time() - timestamp < cache_ttl:
                return cached_result
        
        # Perform search
        response = await self.client.post("/memory/search", json={
            "query": query,
            "limit": 10,
            "search_options": {
                "rerank": True,
                "include_metadata": True
            }
        })
        
        results = response.json()["results"]
        
        # Cache results
        self.search_cache[cache_key] = (results, time.time())
        
        return results
    
    async def parallel_searches(self, queries: List[str]) -> Dict[str, List[Dict]]:
        """Execute multiple searches in parallel."""
        tasks = []
        
        for query in queries:
            task = self.client.post("/memory/search", json={
                "query": query,
                "limit": 5,
                "similarity_threshold": 0.7
            })
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        results = {}
        for i, response in enumerate(responses):
            results[queries[i]] = response.json()["results"]
        
        return results

# Usage example
async def performance_optimization_example():
    """Demonstrate performance optimization techniques."""
    
    optimized_client = OptimizedMemoryClient("http://localhost:8001/api/v1")
    
    # Batch ingestion
    documents = [
        {
            "content": f"Document {i} about agent coordination",
            "agent_id": "performance_tester",
            "tags": ["test", "coordination"]
        }
        for i in range(100)
    ]
    
    # This will automatically batch the documents
    document_ids = await optimized_client.optimized_ingest(documents)
    print(f"Ingested {len(document_ids)} documents efficiently")
    
    # Parallel searches
    queries = [
        "agent coordination patterns",
        "distributed system design",
        "message ordering strategies",
        "context compression techniques"
    ]
    
    search_results = await optimized_client.parallel_searches(queries)
    
    for query, results in search_results.items():
        print(f"Query '{query}': {len(results)} results")
    
    # Cached search (subsequent calls will be faster)
    results1 = await optimized_client.cached_search("coordination patterns")
    results2 = await optimized_client.cached_search("coordination patterns")  # From cache
```

## Error Handling

### Robust Error Handling Patterns

```python
class SemanticMemoryError(Exception):
    """Base exception for semantic memory operations."""
    pass

class IngestionError(SemanticMemoryError):
    """Error during document ingestion."""
    pass

class SearchError(SemanticMemoryError):
    """Error during search operations."""
    pass

class CompressionError(SemanticMemoryError):
    """Error during context compression."""
    pass

class ResilientMemoryClient:
    """Memory client with comprehensive error handling and retry logic."""
    
    def __init__(self, base_url: str, max_retries: int = 3):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
        self.max_retries = max_retries
    
    async def resilient_request(self, method: str, endpoint: str, 
                              data: Dict = None, retries: int = 0) -> Dict:
        """Make a resilient HTTP request with retry logic."""
        try:
            if method.upper() == "POST":
                response = await self.client.post(endpoint, json=data)
            elif method.upper() == "GET":
                response = await self.client.get(endpoint)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500 and retries < self.max_retries:
                # Retry on server errors
                await asyncio.sleep(2 ** retries)  # Exponential backoff
                return await self.resilient_request(method, endpoint, data, retries + 1)
            
            # Handle specific error codes
            if e.response.status_code == 404:
                raise SemanticMemoryError(f"Resource not found: {endpoint}")
            elif e.response.status_code == 429:
                raise SemanticMemoryError("Rate limit exceeded, please retry later")
            else:
                error_detail = e.response.json().get("message", "Unknown error")
                raise SemanticMemoryError(f"API error: {error_detail}")
        
        except httpx.TimeoutException:
            if retries < self.max_retries:
                await asyncio.sleep(2 ** retries)
                return await self.resilient_request(method, endpoint, data, retries + 1)
            raise SemanticMemoryError("Request timeout after retries")
        
        except httpx.RequestError as e:
            if retries < self.max_retries:
                await asyncio.sleep(2 ** retries)
                return await self.resilient_request(method, endpoint, data, retries + 1)
            raise SemanticMemoryError(f"Network error: {str(e)}")
    
    async def safe_ingest(self, content: str, agent_id: str, 
                         tags: List[str] = None) -> Optional[str]:
        """Safely ingest a document with comprehensive error handling."""
        try:
            result = await self.resilient_request("POST", "/memory/ingest", {
                "content": content,
                "agent_id": agent_id,
                "tags": tags or [],
                "processing_options": {
                    "priority": "normal"
                }
            })
            
            return result["document_id"]
            
        except SemanticMemoryError as e:
            logger.error(f"Ingestion failed for agent {agent_id}: {e}")
            
            # Try fallback storage mechanism
            return await self.fallback_storage(content, agent_id, tags)
        
        except Exception as e:
            logger.error(f"Unexpected error during ingestion: {e}")
            return None
    
    async def fallback_storage(self, content: str, agent_id: str, 
                             tags: List[str] = None) -> Optional[str]:
        """Fallback storage mechanism when main ingestion fails."""
        # Could store in Redis, local file, or queue for later processing
        fallback_id = f"fallback_{uuid4()}"
        
        # Store in Redis as fallback (assuming Redis is available)
        try:
            import redis.asyncio as redis
            redis_client = redis.Redis(host='localhost', port=6379, db=0)
            
            await redis_client.hset(
                f"semantic_memory_fallback:{fallback_id}",
                mapping={
                    "content": content,
                    "agent_id": agent_id,
                    "tags": json.dumps(tags or []),
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "pending_retry"
                }
            )
            
            logger.info(f"Stored document in fallback storage: {fallback_id}")
            return fallback_id
            
        except Exception as e:
            logger.error(f"Fallback storage also failed: {e}")
            return None
    
    async def safe_search(self, query: str, agent_id: str = None, 
                         limit: int = 5) -> List[Dict]:
        """Safely perform semantic search with error handling."""
        try:
            result = await self.resilient_request("POST", "/memory/search", {
                "query": query,
                "limit": limit,
                "agent_id": agent_id,
                "similarity_threshold": 0.6,
                "search_options": {
                    "rerank": True,
                    "include_metadata": True
                }
            })
            
            return result["results"]
            
        except SemanticMemoryError as e:
            logger.error(f"Search failed for query '{query}': {e}")
            
            # Return empty results instead of failing
            return []
        
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return []

# Usage example with error handling
async def error_handling_example():
    """Demonstrate robust error handling."""
    
    resilient_client = ResilientMemoryClient("http://localhost:8001/api/v1")
    
    # Safe ingestion
    document_id = await resilient_client.safe_ingest(
        content="Test document with error handling",
        agent_id="error_test_agent",
        tags=["testing", "error_handling"]
    )
    
    if document_id:
        print(f"Successfully ingested document: {document_id}")
    else:
        print("Ingestion failed, but application continues")
    
    # Safe search
    results = await resilient_client.safe_search(
        query="error handling patterns",
        agent_id="error_test_agent"
    )
    
    print(f"Search returned {len(results)} results (may be 0 if service unavailable)")
```

## Best Practices

### 1. Content Quality and Tagging

```python
class ContentProcessor:
    """Best practices for content processing and tagging."""
    
    @staticmethod
    def optimize_content_for_search(content: str) -> str:
        """Optimize content for better search results."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Ensure minimum content length
        if len(content) < 50:
            content = f"Context: {content}. This content describes important system behavior."
        
        return content
    
    @staticmethod
    def generate_semantic_tags(content: str, agent_id: str) -> List[str]:
        """Generate semantic tags from content."""
        tags = []
        
        # Add agent-based tag
        tags.append(f"agent_{agent_id}")
        
        # Extract key concepts (simplified)
        if "coordination" in content.lower():
            tags.append("coordination")
        if "error" in content.lower() or "failure" in content.lower():
            tags.append("error_handling")
        if "performance" in content.lower() or "optimization" in content.lower():
            tags.append("performance")
        if "workflow" in content.lower() or "dag" in content.lower():
            tags.append("workflow")
        
        # Add timestamp-based tag for temporal queries
        tags.append(f"period_{datetime.utcnow().strftime('%Y-%m')}")
        
        return tags
    
    @staticmethod
    def calculate_importance_score(content: str, agent_id: str, 
                                 context: Dict[str, Any] = None) -> float:
        """Calculate content importance score."""
        score = 0.5  # Base score
        
        # Length-based scoring
        if len(content) > 200:
            score += 0.1
        
        # Keyword-based scoring
        important_keywords = ["critical", "important", "key", "essential", "vital"]
        for keyword in important_keywords:
            if keyword in content.lower():
                score += 0.1
                break
        
        # Context-based scoring
        if context:
            if context.get("workflow_critical", False):
                score += 0.2
            if context.get("error_related", False):
                score += 0.15
        
        return min(1.0, score)

# Usage example
async def content_best_practices_example():
    """Demonstrate content processing best practices."""
    
    client = httpx.AsyncClient(base_url="http://localhost:8001/api/v1")
    processor = ContentProcessor()
    
    raw_content = "   Agent   coordination    failed   due to   network   issues   "
    agent_id = "network_monitor"
    
    # Process content
    optimized_content = processor.optimize_content_for_search(raw_content)
    semantic_tags = processor.generate_semantic_tags(optimized_content, agent_id)
    importance = processor.calculate_importance_score(
        optimized_content, 
        agent_id, 
        {"error_related": True, "workflow_critical": True}
    )
    
    # Ingest with optimized parameters
    response = await client.post("/memory/ingest", json={
        "content": optimized_content,
        "agent_id": agent_id,
        "tags": semantic_tags,
        "metadata": {
            "importance": importance,
            "processed": True,
            "content_length": len(optimized_content)
        },
        "processing_options": {
            "generate_summary": True,
            "priority": "high" if importance > 0.8 else "normal"
        }
    })
    
    print(f"Ingested optimized content with importance {importance}")
```

### 2. Performance Monitoring

```python
class PerformanceMonitor:
    """Monitor semantic memory service performance."""
    
    def __init__(self, memory_client: httpx.AsyncClient):
        self.client = memory_client
        self.metrics = {}
    
    async def benchmark_operations(self) -> Dict[str, float]:
        """Benchmark key operations."""
        operations = {}
        
        # Benchmark ingestion
        start_time = time.time()
        await self.client.post("/memory/ingest", json={
            "content": "Benchmark test document for performance measurement",
            "agent_id": "benchmark_agent",
            "tags": ["benchmark", "performance"]
        })
        operations["ingest_time_ms"] = (time.time() - start_time) * 1000
        
        # Benchmark search
        start_time = time.time()
        await self.client.post("/memory/search", json={
            "query": "benchmark test performance",
            "limit": 5
        })
        operations["search_time_ms"] = (time.time() - start_time) * 1000
        
        return operations
    
    async def check_service_health(self) -> Dict[str, Any]:
        """Check service health and performance metrics."""
        try:
            response = await self.client.get("/memory/health")
            return response.json()
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        try:
            response = await self.client.get("/memory/metrics", params={
                "format": "json",
                "time_range": "1h"
            })
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}

# Performance monitoring example
async def performance_monitoring_example():
    """Demonstrate performance monitoring."""
    
    client = httpx.AsyncClient(base_url="http://localhost:8001/api/v1")
    monitor = PerformanceMonitor(client)
    
    # Benchmark operations
    benchmark_results = await monitor.benchmark_operations()
    print(f"Performance Benchmark Results:")
    for operation, time_ms in benchmark_results.items():
        print(f"  {operation}: {time_ms:.2f}ms")
    
    # Check service health
    health = await monitor.check_service_health()
    print(f"Service Status: {health.get('status', 'unknown')}")
    
    # Get detailed metrics
    metrics = await monitor.get_performance_metrics()
    if metrics:
        perf_metrics = metrics.get("performance_metrics", {})
        search_ops = perf_metrics.get("search_operations", {})
        print(f"Average Search Time: {search_ops.get('avg_duration_ms', 'N/A')}ms")
        print(f"P95 Search Time: {search_ops.get('p95_duration_ms', 'N/A')}ms")
```

## Conclusion

This documentation provides comprehensive patterns and examples for integrating with the Semantic Memory Service API. The mock server enables immediate parallel development while maintaining realistic performance characteristics and error scenarios.

Key takeaways:
- Use agent-scoped memory management for isolation
- Integrate with DAG workflows using specialized nodes
- Implement robust error handling with fallback mechanisms
- Optimize performance through batching and caching
- Monitor service health and performance metrics
- Follow content optimization best practices

For questions or additional examples, refer to the OpenAPI specification at `/api_contracts/semantic_memory_service.yaml`.