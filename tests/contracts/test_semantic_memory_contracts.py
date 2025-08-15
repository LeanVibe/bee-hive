"""
Semantic Memory and Context Engine Contracts Tests
==================================================

Tests all contracts for the semantic memory and context management system:
- pgvector integration contracts for vector storage and retrieval
- Embedding service contracts for text vectorization
- Knowledge graph contracts for relationship modeling
- Context compression contracts for memory optimization
- Cross-agent knowledge sharing contracts

This ensures consistent interfaces for memory operations across the system.
"""

import asyncio
import uuid
import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from tests.contracts.contract_testing_framework import (
    ContractTestSuite, ContractValidator, ContractDefinition, ContractType
)


@pytest.mark.contracts
@pytest.mark.unit
class TestSemanticMemoryContracts:
    """Test semantic memory service interface contracts."""
    
    @pytest.fixture
    def memory_contracts_suite(self):
        """Setup semantic memory contracts."""
        suite = ContractTestSuite()
        suite.setup_context_engine_contracts()
        return suite
    
    async def test_memory_storage_contract_validation(self, memory_contracts_suite):
        """Test memory storage interface contract."""
        suite = memory_contracts_suite
        
        # Valid memory storage request
        valid_memory = {
            "memory_id": str(uuid.uuid4()),
            "content": "The user prefers Python over JavaScript for backend development because of its simplicity and extensive libraries",
            "memory_type": "PREFERENCE",
            "agent_id": str(uuid.uuid4()),
            "context": {
                "conversation_id": str(uuid.uuid4()),
                "domain": "programming_languages",
                "confidence_level": 0.9,
                "source": "direct_conversation",
                "related_topics": ["backend_development", "python", "javascript"],
                "emotional_context": "positive_preference"
            },
            "tags": ["programming", "languages", "backend", "preference"],
            "importance_score": 0.8,
            "access_count": 0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat()
        }
        
        result = suite.validator.validate_data(
            contract_name="memory_storage",
            contract_version="1.0",
            data=valid_memory
        )
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        # Test all valid memory types
        valid_memory_types = ["FACT", "PREFERENCE", "SKILL", "CONTEXT", "RELATIONSHIP"]
        
        for memory_type in valid_memory_types:
            type_variant = valid_memory.copy()
            type_variant["memory_type"] = memory_type
            type_variant["memory_id"] = str(uuid.uuid4())
            type_variant["content"] = f"Test {memory_type.lower()} memory content"
            
            type_result = suite.validator.validate_data(
                contract_name="memory_storage",
                contract_version="1.0",
                data=type_variant
            )
            
            assert type_result["valid"] is True, f"Memory type {memory_type} should be valid"
        
        # Test importance score boundaries
        importance_scores = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for score in importance_scores:
            score_variant = valid_memory.copy()
            score_variant["importance_score"] = score
            score_variant["memory_id"] = str(uuid.uuid4())
            
            score_result = suite.validator.validate_data(
                contract_name="memory_storage",
                contract_version="1.0",
                data=score_variant
            )
            
            assert score_result["valid"] is True, f"Importance score {score} should be valid"
        
        # Test invalid importance scores
        invalid_scores = [-0.1, 1.1, 2.0]
        
        for score in invalid_scores:
            invalid_variant = valid_memory.copy()
            invalid_variant["importance_score"] = score
            invalid_variant["memory_id"] = str(uuid.uuid4())
            
            invalid_result = suite.validator.validate_data(
                contract_name="memory_storage",
                contract_version="1.0",
                data=invalid_variant
            )
            
            assert invalid_result["valid"] is False, f"Importance score {score} should be invalid"
    
    async def test_vector_search_contract_validation(self, memory_contracts_suite):
        """Test vector search interface contract."""
        suite = memory_contracts_suite
        
        # Valid vector search with query vector
        valid_vector_search = {
            "query_vector": np.random.random(1536).tolist(),  # Standard embedding dimension
            "similarity_threshold": 0.7,
            "max_results": 10,
            "filters": {
                "agent_id": str(uuid.uuid4()),
                "memory_type": ["FACT", "PREFERENCE"],
                "importance_score": {"min": 0.5, "max": 1.0},
                "tags": {"include": ["programming"], "exclude": ["deprecated"]},
                "date_range": {
                    "start": (datetime.utcnow() - timedelta(days=30)).isoformat(),
                    "end": datetime.utcnow().isoformat()
                }
            },
            "include_metadata": True
        }
        
        result = suite.validator.validate_data(
            contract_name="vector_search",
            contract_version="1.0",
            data=valid_vector_search
        )
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        # Valid text search (alternative to vector search)
        valid_text_search = {
            "query_text": "Python programming best practices for web development",
            "similarity_threshold": 0.6,
            "max_results": 20,
            "filters": {
                "memory_type": ["SKILL", "FACT"],
                "domain": "programming"
            },
            "include_metadata": False
        }
        
        text_result = suite.validator.validate_data(
            contract_name="vector_search",
            contract_version="1.0",
            data=valid_text_search
        )
        
        assert text_result["valid"] is True
        
        # Test max_results boundaries
        valid_max_results = [1, 10, 50, 100, 1000]
        
        for max_results in valid_max_results:
            max_variant = valid_text_search.copy()
            max_variant["max_results"] = max_results
            
            max_result = suite.validator.validate_data(
                contract_name="vector_search",
                contract_version="1.0",
                data=max_variant
            )
            
            assert max_result["valid"] is True, f"max_results {max_results} should be valid"
        
        # Test invalid max_results
        invalid_max_results = [0, -1, 1001, 10000]
        
        for max_results in invalid_max_results:
            invalid_variant = valid_text_search.copy()
            invalid_variant["max_results"] = max_results
            
            invalid_result = suite.validator.validate_data(
                contract_name="vector_search",
                contract_version="1.0",
                data=invalid_variant
            )
            
            assert invalid_result["valid"] is False, f"max_results {max_results} should be invalid"
        
        # Test similarity threshold boundaries
        valid_thresholds = [0.0, 0.3, 0.5, 0.8, 1.0]
        
        for threshold in valid_thresholds:
            threshold_variant = valid_text_search.copy()
            threshold_variant["similarity_threshold"] = threshold
            
            threshold_result = suite.validator.validate_data(
                contract_name="vector_search",
                contract_version="1.0",
                data=threshold_variant
            )
            
            assert threshold_result["valid"] is True, f"Similarity threshold {threshold} should be valid"
    
    async def test_memory_contract_missing_required_fields(self, memory_contracts_suite):
        """Test memory contracts with missing required fields."""
        suite = memory_contracts_suite
        
        # Test memory storage without required fields
        incomplete_memory = {
            "content": "Some memory content",
            "memory_type": "FACT",
            # Missing: memory_id, agent_id, importance_score
        }
        
        incomplete_result = suite.validator.validate_data(
            contract_name="memory_storage",
            contract_version="1.0",
            data=incomplete_memory
        )
        
        assert incomplete_result["valid"] is False
        assert len(incomplete_result["errors"]) > 0
        
        # Verify specific required fields are reported as missing
        error_messages = str(incomplete_result["errors"])
        assert "memory_id" in error_messages
        assert "agent_id" in error_messages
        assert "importance_score" in error_messages
        
        # Test vector search without query vector or text
        incomplete_search = {
            "similarity_threshold": 0.7,
            "max_results": 10
            # Missing: query_vector OR query_text
        }
        
        search_result = suite.validator.validate_data(
            contract_name="vector_search",
            contract_version="1.0",
            data=incomplete_search
        )
        
        assert search_result["valid"] is False
        
        # Test vector search without max_results
        no_max_search = {
            "query_text": "test query"
            # Missing: max_results (required)
        }
        
        no_max_result = suite.validator.validate_data(
            contract_name="vector_search",
            contract_version="1.0",
            data=no_max_search
        )
        
        assert no_max_result["valid"] is False


@pytest.mark.contracts
@pytest.mark.unit
class TestContextEngineExtendedContracts:
    """Test extended context engine contracts."""
    
    @pytest.fixture
    def extended_context_contracts(self):
        """Setup extended context engine contracts."""
        validator = ContractValidator()
        
        # Knowledge graph contract
        knowledge_graph_schema = {
            "type": "object",
            "properties": {
                "graph_id": {"type": "string", "format": "uuid"},
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "node_id": {"type": "string"},
                            "label": {"type": "string"},
                            "node_type": {"type": "string", "enum": ["CONCEPT", "ENTITY", "SKILL", "RELATIONSHIP"]},
                            "properties": {"type": "object"},
                            "embeddings": {"type": "array", "items": {"type": "number"}},
                            "importance": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["node_id", "label", "node_type"]
                    }
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "edge_id": {"type": "string"},
                            "source_node": {"type": "string"},
                            "target_node": {"type": "string"},
                            "relationship_type": {"type": "string", "enum": ["RELATES_TO", "DEPENDS_ON", "PART_OF", "ENABLES", "CONFLICTS_WITH"]},
                            "weight": {"type": "number", "minimum": 0, "maximum": 1},
                            "properties": {"type": "object"}
                        },
                        "required": ["edge_id", "source_node", "target_node", "relationship_type", "weight"]
                    }
                },
                "metadata": {"type": "object"},
                "created_at": {"type": "string", "format": "date-time"},
                "updated_at": {"type": "string", "format": "date-time"}
            },
            "required": ["graph_id", "nodes", "edges"]
        }
        
        knowledge_graph_contract = ContractDefinition(
            name="knowledge_graph",
            version="1.0",
            contract_type=ContractType.API_RESPONSE,
            schema=knowledge_graph_schema,
            description="Contract for knowledge graph structure",
            producer="context_engine",
            consumer="knowledge_graph_builder"
        )
        
        validator.register_contract(knowledge_graph_contract)
        
        # Context compression contract
        context_compression_schema = {
            "type": "object",
            "properties": {
                "compression_id": {"type": "string", "format": "uuid"},
                "original_context": {"type": "object"},
                "compressed_context": {"type": "object"},
                "compression_strategy": {"type": "string", "enum": ["SEMANTIC", "TEMPORAL", "IMPORTANCE", "HYBRID"]},
                "compression_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                "preserved_elements": {"type": "array", "items": {"type": "string"}},
                "lost_information": {"type": "array", "items": {"type": "string"}},
                "quality_metrics": {
                    "type": "object",
                    "properties": {
                        "semantic_preservation": {"type": "number", "minimum": 0, "maximum": 1},
                        "information_density": {"type": "number", "minimum": 0},
                        "reconstruction_fidelity": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                },
                "target_size": {"type": "integer", "minimum": 1},
                "actual_size": {"type": "integer", "minimum": 1},
                "processing_time_ms": {"type": "number", "minimum": 0}
            },
            "required": ["compression_id", "compressed_context", "compression_strategy", "compression_ratio"]
        }
        
        compression_contract = ContractDefinition(
            name="context_compression",
            version="1.0",
            contract_type=ContractType.API_RESPONSE,
            schema=context_compression_schema,
            description="Contract for context compression operations",
            producer="context_compression_engine",
            consumer="context_manager"
        )
        
        validator.register_contract(compression_contract)
        
        # Cross-agent knowledge sharing contract
        knowledge_sharing_schema = {
            "type": "object",
            "properties": {
                "sharing_request_id": {"type": "string", "format": "uuid"},
                "requesting_agent_id": {"type": "string", "format": "uuid"},
                "source_agent_id": {"type": "string", "format": "uuid"},
                "knowledge_query": {"type": "string", "minLength": 1},
                "knowledge_domain": {"type": "string"},
                "sharing_scope": {"type": "string", "enum": ["PUBLIC", "TEAM", "PROJECT", "PRIVATE"]},
                "knowledge_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "knowledge_id": {"type": "string"},
                            "content": {"type": "string"},
                            "relevance_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "adapted_content": {"type": "string"},
                            "adaptation_notes": {"type": "string"},
                            "usage_constraints": {"type": "array", "items": {"type": "string"}},
                            "sharing_metadata": {"type": "object"}
                        },
                        "required": ["knowledge_id", "content", "relevance_score"]
                    }
                },
                "sharing_status": {"type": "string", "enum": ["REQUESTED", "APPROVED", "SHARED", "DENIED"]},
                "approval_required": {"type": "boolean"},
                "expires_at": {"type": "string", "format": "date-time"}
            },
            "required": ["sharing_request_id", "requesting_agent_id", "knowledge_query", "sharing_scope", "sharing_status"]
        }
        
        sharing_contract = ContractDefinition(
            name="knowledge_sharing",
            version="1.0",
            contract_type=ContractType.API_REQUEST,
            schema=knowledge_sharing_schema,
            description="Contract for cross-agent knowledge sharing",
            producer="semantic_memory_service",
            consumer="agent_communication_service"
        )
        
        validator.register_contract(sharing_contract)
        
        return validator
    
    async def test_knowledge_graph_contract_validation(self, extended_context_contracts):
        """Test knowledge graph contract validation."""
        validator = extended_context_contracts
        
        # Valid knowledge graph
        valid_graph = {
            "graph_id": str(uuid.uuid4()),
            "nodes": [
                {
                    "node_id": "python_programming",
                    "label": "Python Programming",
                    "node_type": "SKILL",
                    "properties": {
                        "difficulty": "intermediate",
                        "category": "programming_language",
                        "popularity": 0.95
                    },
                    "embeddings": np.random.random(768).tolist(),
                    "importance": 0.9
                },
                {
                    "node_id": "web_development",
                    "label": "Web Development",
                    "node_type": "CONCEPT",
                    "properties": {
                        "domain": "software_engineering",
                        "complexity": "high"
                    },
                    "importance": 0.8
                },
                {
                    "node_id": "fastapi_framework",
                    "label": "FastAPI Framework",
                    "node_type": "ENTITY",
                    "properties": {
                        "type": "web_framework",
                        "language": "python",
                        "performance": "high"
                    },
                    "importance": 0.75
                }
            ],
            "edges": [
                {
                    "edge_id": "python_enables_web",
                    "source_node": "python_programming",
                    "target_node": "web_development",
                    "relationship_type": "ENABLES",
                    "weight": 0.8,
                    "properties": {
                        "strength": "strong",
                        "context": "backend_development"
                    }
                },
                {
                    "edge_id": "fastapi_part_of_python",
                    "source_node": "fastapi_framework",
                    "target_node": "python_programming",
                    "relationship_type": "PART_OF",
                    "weight": 0.9,
                    "properties": {
                        "relationship": "framework_to_language"
                    }
                }
            ],
            "metadata": {
                "domain": "programming",
                "agent_id": str(uuid.uuid4()),
                "node_count": 3,
                "edge_count": 2,
                "complexity_score": 0.6
            },
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = validator.validate_data(
            contract_name="knowledge_graph",
            contract_version="1.0",
            data=valid_graph
        )
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        # Test all valid node types
        valid_node_types = ["CONCEPT", "ENTITY", "SKILL", "RELATIONSHIP"]
        
        for node_type in valid_node_types:
            node_variant = valid_graph.copy()
            node_variant["nodes"] = [{
                "node_id": f"test_{node_type.lower()}",
                "label": f"Test {node_type}",
                "node_type": node_type
            }]
            node_variant["edges"] = []  # Simplified for type testing
            node_variant["graph_id"] = str(uuid.uuid4())
            
            type_result = validator.validate_data(
                contract_name="knowledge_graph",
                contract_version="1.0",
                data=node_variant
            )
            
            assert type_result["valid"] is True, f"Node type {node_type} should be valid"
        
        # Test all valid relationship types
        valid_relationship_types = ["RELATES_TO", "DEPENDS_ON", "PART_OF", "ENABLES", "CONFLICTS_WITH"]
        
        for rel_type in valid_relationship_types:
            rel_variant = valid_graph.copy()
            rel_variant["edges"] = [{
                "edge_id": f"test_{rel_type.lower()}",
                "source_node": "node1",
                "target_node": "node2",
                "relationship_type": rel_type,
                "weight": 0.5
            }]
            rel_variant["graph_id"] = str(uuid.uuid4())
            
            rel_result = validator.validate_data(
                contract_name="knowledge_graph",
                contract_version="1.0",
                data=rel_variant
            )
            
            assert rel_result["valid"] is True, f"Relationship type {rel_type} should be valid"
    
    async def test_context_compression_contract_validation(self, extended_context_contracts):
        """Test context compression contract validation."""
        validator = extended_context_contracts
        
        # Valid context compression result
        valid_compression = {
            "compression_id": str(uuid.uuid4()),
            "original_context": {
                "conversation_history": [
                    {"role": "user", "content": "How do I implement authentication in FastAPI?"},
                    {"role": "assistant", "content": "Here's a comprehensive guide to implementing JWT authentication..."}
                ],
                "code_snippets": [
                    {"language": "python", "code": "from fastapi import FastAPI, Depends..."},
                    {"language": "python", "code": "def get_current_user(token: str = Depends(...))"}
                ],
                "documentation": [
                    {"title": "FastAPI Security", "content": "FastAPI provides several utilities..."}
                ]
            },
            "compressed_context": {
                "key_points": [
                    "User asked about FastAPI authentication",
                    "JWT implementation discussed",
                    "Security utilities mentioned"
                ],
                "code_summary": "Authentication implementation with JWT tokens and dependency injection",
                "main_topics": ["fastapi", "jwt", "authentication", "security"]
            },
            "compression_strategy": "SEMANTIC",
            "compression_ratio": 0.35,
            "preserved_elements": [
                "authentication_concept",
                "jwt_implementation",
                "fastapi_security"
            ],
            "lost_information": [
                "verbose_explanations",
                "example_variations",
                "background_context"
            ],
            "quality_metrics": {
                "semantic_preservation": 0.92,
                "information_density": 2.8,
                "reconstruction_fidelity": 0.87
            },
            "target_size": 500,
            "actual_size": 475,
            "processing_time_ms": 250.5
        }
        
        result = validator.validate_data(
            contract_name="context_compression",
            contract_version="1.0",
            data=valid_compression
        )
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        # Test all valid compression strategies
        valid_strategies = ["SEMANTIC", "TEMPORAL", "IMPORTANCE", "HYBRID"]
        
        for strategy in valid_strategies:
            strategy_variant = valid_compression.copy()
            strategy_variant["compression_strategy"] = strategy
            strategy_variant["compression_id"] = str(uuid.uuid4())
            
            strategy_result = validator.validate_data(
                contract_name="context_compression",
                contract_version="1.0",
                data=strategy_variant
            )
            
            assert strategy_result["valid"] is True, f"Compression strategy {strategy} should be valid"
        
        # Test compression ratio boundaries
        valid_ratios = [0.0, 0.1, 0.5, 0.8, 1.0]
        
        for ratio in valid_ratios:
            ratio_variant = valid_compression.copy()
            ratio_variant["compression_ratio"] = ratio
            ratio_variant["compression_id"] = str(uuid.uuid4())
            
            ratio_result = validator.validate_data(
                contract_name="context_compression",
                contract_version="1.0",
                data=ratio_variant
            )
            
            assert ratio_result["valid"] is True, f"Compression ratio {ratio} should be valid"
    
    async def test_knowledge_sharing_contract_validation(self, extended_context_contracts):
        """Test cross-agent knowledge sharing contract validation."""
        validator = extended_context_contracts
        
        # Valid knowledge sharing request
        valid_sharing = {
            "sharing_request_id": str(uuid.uuid4()),
            "requesting_agent_id": str(uuid.uuid4()),
            "source_agent_id": str(uuid.uuid4()),
            "knowledge_query": "Best practices for API error handling in Python web applications",
            "knowledge_domain": "web_development",
            "sharing_scope": "TEAM",
            "knowledge_items": [
                {
                    "knowledge_id": "error_handling_best_practices",
                    "content": "Always use proper HTTP status codes and provide descriptive error messages",
                    "relevance_score": 0.95,
                    "adapted_content": "For your FastAPI project, implement custom exception handlers...",
                    "adaptation_notes": "Adapted for FastAPI context with specific examples",
                    "usage_constraints": ["project_specific", "team_internal"],
                    "sharing_metadata": {
                        "original_context": "django_project",
                        "adaptation_confidence": 0.88
                    }
                },
                {
                    "knowledge_id": "logging_strategies",
                    "content": "Implement structured logging with correlation IDs for request tracing",
                    "relevance_score": 0.82,
                    "adapted_content": "Use Python's logging module with structured format...",
                    "adaptation_notes": "General logging principles applicable across frameworks"
                }
            ],
            "sharing_status": "SHARED",
            "approval_required": True,
            "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat()
        }
        
        result = validator.validate_data(
            contract_name="knowledge_sharing",
            contract_version="1.0",
            data=valid_sharing
        )
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        # Test all valid sharing scopes
        valid_scopes = ["PUBLIC", "TEAM", "PROJECT", "PRIVATE"]
        
        for scope in valid_scopes:
            scope_variant = valid_sharing.copy()
            scope_variant["sharing_scope"] = scope
            scope_variant["sharing_request_id"] = str(uuid.uuid4())
            
            scope_result = validator.validate_data(
                contract_name="knowledge_sharing",
                contract_version="1.0",
                data=scope_variant
            )
            
            assert scope_result["valid"] is True, f"Sharing scope {scope} should be valid"
        
        # Test all valid sharing statuses
        valid_statuses = ["REQUESTED", "APPROVED", "SHARED", "DENIED"]
        
        for status in valid_statuses:
            status_variant = valid_sharing.copy()
            status_variant["sharing_status"] = status
            status_variant["sharing_request_id"] = str(uuid.uuid4())
            
            status_result = validator.validate_data(
                contract_name="knowledge_sharing",
                contract_version="1.0",
                data=status_variant
            )
            
            assert status_result["valid"] is True, f"Sharing status {status} should be valid"
        
        # Test relevance score boundaries in knowledge items
        relevance_scores = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for score in relevance_scores:
            score_variant = valid_sharing.copy()
            score_variant["knowledge_items"][0]["relevance_score"] = score
            score_variant["sharing_request_id"] = str(uuid.uuid4())
            
            score_result = validator.validate_data(
                contract_name="knowledge_sharing",
                contract_version="1.0",
                data=score_variant
            )
            
            assert score_result["valid"] is True, f"Relevance score {score} should be valid"


@pytest.mark.contracts
@pytest.mark.integration
class TestSemanticMemoryContractIntegration:
    """Test semantic memory contract integration scenarios."""
    
    @pytest.fixture
    def comprehensive_memory_contracts(self):
        """Setup comprehensive memory contract suite."""
        suite = ContractTestSuite()
        suite.setup_context_engine_contracts()
        
        # Also setup extended contracts
        validator = ContractValidator()
        
        # Add all contracts from the extended setup
        extended_validator = extended_validator = ContractValidator()
        
        # Register all contracts in one validator
        for contract_key, contract in suite.validator.contracts.items():
            validator.register_contract(contract)
        
        return validator
    
    async def test_memory_lifecycle_contract_compliance(self, comprehensive_memory_contracts):
        """Test memory operations maintain contract compliance throughout lifecycle."""
        validator = comprehensive_memory_contracts
        
        # 1. Store memory
        memory_data = {
            "memory_id": str(uuid.uuid4()),
            "content": "FastAPI supports dependency injection which makes testing easier",
            "memory_type": "FACT",
            "agent_id": str(uuid.uuid4()),
            "context": {
                "topic": "web_frameworks",
                "subtopic": "testing",
                "learning_session": str(uuid.uuid4())
            },
            "tags": ["fastapi", "testing", "dependency_injection"],
            "importance_score": 0.85,
            "access_count": 0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        storage_result = validator.validate_data(
            contract_name="memory_storage",
            contract_version="1.0",
            data=memory_data
        )
        
        assert storage_result["valid"] is True
        
        # 2. Search for related memories
        search_data = {
            "query_text": "dependency injection testing frameworks",
            "similarity_threshold": 0.6,
            "max_results": 5,
            "filters": {
                "tags": {"include": ["testing"]},
                "importance_score": {"min": 0.7}
            },
            "include_metadata": True
        }
        
        search_result = validator.validate_data(
            contract_name="vector_search",
            contract_version="1.0",
            data=search_data
        )
        
        assert search_result["valid"] is True
        
        # 3. Update memory (simulating access count increment)
        updated_memory = memory_data.copy()
        updated_memory["access_count"] = 3
        updated_memory["updated_at"] = datetime.utcnow().isoformat()
        
        update_result = validator.validate_data(
            contract_name="memory_storage",
            contract_version="1.0",
            data=updated_memory
        )
        
        assert update_result["valid"] is True
    
    async def test_contract_performance_under_load(self, comprehensive_memory_contracts):
        """Test contract validation performance under realistic load."""
        validator = comprehensive_memory_contracts
        
        # Generate large batch of memory operations
        memory_operations = []
        search_operations = []
        
        for i in range(500):
            # Memory storage
            memory_op = {
                "memory_id": str(uuid.uuid4()),
                "content": f"Performance test memory {i}: Content about various programming concepts",
                "memory_type": ["FACT", "SKILL", "PREFERENCE"][i % 3],
                "agent_id": str(uuid.uuid4()),
                "context": {"batch": i, "test": "performance"},
                "tags": [f"tag_{i % 10}", "performance", "test"],
                "importance_score": 0.1 + (i % 9) * 0.1,
                "access_count": i % 5,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            memory_operations.append(memory_op)
            
            # Vector search
            search_op = {
                "query_text": f"search query {i}",
                "similarity_threshold": 0.5 + (i % 5) * 0.1,
                "max_results": 10 + (i % 10),
                "include_metadata": i % 2 == 0
            }
            search_operations.append(search_op)
        
        # Validate all operations and measure performance
        import time
        
        # Memory storage validation
        start_time = time.time()
        memory_results = []
        for memory_op in memory_operations:
            result = validator.validate_data(
                contract_name="memory_storage",
                contract_version="1.0",
                data=memory_op
            )
            memory_results.append(result)
        memory_time = time.time() - start_time
        
        # Search validation
        start_time = time.time()
        search_results = []
        for search_op in search_operations:
            result = validator.validate_data(
                contract_name="vector_search",
                contract_version="1.0",
                data=search_op
            )
            search_results.append(result)
        search_time = time.time() - start_time
        
        # Performance assertions
        total_operations = len(memory_operations) + len(search_operations)
        total_time = memory_time + search_time
        
        assert total_time < 10.0, f"Contract validation took {total_time:.2f}s, should be under 10s for {total_operations} operations"
        
        # Verify all validations succeeded
        successful_memory = sum(1 for result in memory_results if result["valid"])
        successful_search = sum(1 for result in search_results if result["valid"])
        
        assert successful_memory == 500, "All memory operations should pass validation"
        assert successful_search == 500, "All search operations should pass validation"
        
        # Performance metrics
        avg_validation_time = total_time / total_operations
        validations_per_second = total_operations / total_time
        
        print(f"Memory Contract Validation Performance:")
        print(f"  Memory operations: {len(memory_operations)} in {memory_time:.3f}s")
        print(f"  Search operations: {len(search_operations)} in {search_time:.3f}s") 
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average per operation: {avg_validation_time*1000:.2f}ms")
        print(f"  Operations per second: {validations_per_second:.1f}")
        
        assert avg_validation_time < 0.01, "Average validation time should be under 10ms"
        assert validations_per_second > 100, "Should validate at least 100 operations per second"