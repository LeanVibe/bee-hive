"""Add Context Compression and Knowledge Sharing Schema Extensions

Revision ID: 017_context_compression_sharing
Revises: 016_semantic_memory_core
Create Date: 2024-07-29 10:00:00.000000

Adds comprehensive database schema support for:
- Context compression operations and metadata
- Cross-agent knowledge sharing records
- Memory hierarchy management
- Knowledge graph relationships
- Agent expertise tracking
- Quality assessment records
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid


# revision identifiers
revision = '017_context_compression_sharing'
down_revision = '016_semantic_memory_core'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema with context compression and knowledge sharing extensions."""
    
    # =============================================================================
    # CONTEXT COMPRESSION TABLES
    # =============================================================================
    
    # Compression operations table
    op.create_table(
        'compression_operations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('compression_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('agent_id', sa.String(255), nullable=False, index=True),
        sa.Column('original_content_hash', sa.String(64), nullable=False),
        sa.Column('compressed_content_hash', sa.String(64), nullable=False),
        
        # Content and metadata
        sa.Column('original_content', sa.Text, nullable=False),
        sa.Column('compressed_content', sa.Text, nullable=False),
        sa.Column('context_metadata', postgresql.JSONB, nullable=True),
        
        # Size and performance metrics
        sa.Column('original_size', sa.Integer, nullable=False),
        sa.Column('compressed_size', sa.Integer, nullable=False),
        sa.Column('compression_ratio', sa.Float, nullable=False),
        sa.Column('processing_time_ms', sa.Float, nullable=False),
        
        # Quality metrics
        sa.Column('semantic_preservation_score', sa.Float, nullable=False),
        sa.Column('algorithm_used', sa.String(100), nullable=False),
        sa.Column('compression_strategy', sa.String(100), nullable=False),
        sa.Column('compression_quality', sa.String(50), nullable=False),
        
        # Temporal information
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('accessed_count', sa.Integer, default=0, nullable=False),
        sa.Column('last_accessed', sa.DateTime(timezone=True), nullable=True),
        
        # Indexes
        sa.Index('idx_compression_agent_created', 'agent_id', 'created_at'),
        sa.Index('idx_compression_algorithm', 'algorithm_used'),
        sa.Index('idx_compression_ratio', 'compression_ratio'),
        sa.Index('idx_compression_quality', 'semantic_preservation_score')
    )
    
    # Compression configuration table
    op.create_table(
        'compression_configs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('config_name', sa.String(255), nullable=False, unique=True),
        sa.Column('agent_id', sa.String(255), nullable=True, index=True),  # NULL for global configs
        
        # Configuration parameters
        sa.Column('strategy', sa.String(100), nullable=False),
        sa.Column('quality', sa.String(50), nullable=False),
        sa.Column('target_reduction', sa.Float, nullable=False),
        sa.Column('preserve_importance_threshold', sa.Float, nullable=False),
        sa.Column('semantic_similarity_threshold', sa.Float, nullable=False),
        sa.Column('temporal_decay_factor', sa.Float, nullable=False),
        sa.Column('max_clusters', sa.Integer, nullable=False),
        sa.Column('preserve_recent_hours', sa.Integer, nullable=False),
        sa.Column('enable_semantic_validation', sa.Boolean, default=True, nullable=False),
        
        # Metadata and lifecycle
        sa.Column('config_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('usage_count', sa.Integer, default=0, nullable=False),
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        
        sa.Index('idx_compression_config_agent', 'agent_id', 'is_active')
    )
    
    # =============================================================================
    # KNOWLEDGE SHARING TABLES
    # =============================================================================
    
    # Knowledge sharing events table
    op.create_table(
        'knowledge_sharing_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('sharing_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('source_agent_id', sa.String(255), nullable=False, index=True),
        sa.Column('target_agent_id', sa.String(255), nullable=False, index=True),
        sa.Column('knowledge_id', sa.String(255), nullable=False, index=True),
        
        # Sharing details
        sa.Column('sharing_policy', sa.String(100), nullable=False),
        sa.Column('justification', sa.Text, nullable=True),
        sa.Column('approval_status', sa.String(50), default='pending', nullable=False),
        sa.Column('approved_by', sa.String(255), nullable=True),
        sa.Column('approved_at', sa.DateTime(timezone=True), nullable=True),
        
        # Quality and success metrics
        sa.Column('quality_score', sa.Float, nullable=True),
        sa.Column('success', sa.Boolean, nullable=True),
        sa.Column('feedback_score', sa.Float, nullable=True),
        sa.Column('usage_after_sharing', sa.Integer, default=0, nullable=False),
        
        # Content information
        sa.Column('knowledge_type', sa.String(100), nullable=False),
        sa.Column('domain', sa.String(255), nullable=True),
        sa.Column('tags', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('compressed', sa.Boolean, default=False, nullable=False),
        sa.Column('compression_id', sa.String(255), nullable=True),
        
        # Temporal information
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_accessed', sa.DateTime(timezone=True), nullable=True),
        
        # Foreign key constraints
        sa.ForeignKeyConstraint(['compression_id'], ['compression_operations.compression_id'], ondelete='SET NULL'),
        
        # Indexes
        sa.Index('idx_sharing_source_target', 'source_agent_id', 'target_agent_id'),
        sa.Index('idx_sharing_created', 'created_at'),
        sa.Index('idx_sharing_status', 'approval_status'),
        sa.Index('idx_sharing_domain', 'domain'),
        sa.Index('idx_sharing_success', 'success')
    )
    
    # Knowledge quality assessments table
    op.create_table(
        'knowledge_quality_assessments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('knowledge_id', sa.String(255), nullable=False, index=True),
        sa.Column('assessor_agent_id', sa.String(255), nullable=False, index=True),
        
        # Quality metrics
        sa.Column('overall_score', sa.Float, nullable=False),
        sa.Column('accuracy_score', sa.Float, nullable=True),
        sa.Column('usefulness_score', sa.Float, nullable=True),
        sa.Column('completeness_score', sa.Float, nullable=True),
        sa.Column('timeliness_score', sa.Float, nullable=True),
        sa.Column('specificity_score', sa.Float, nullable=True),
        
        # Feedback details
        sa.Column('feedback_type', sa.String(100), nullable=False),
        sa.Column('feedback_comments', sa.Text, nullable=True),
        sa.Column('usage_context', postgresql.JSONB, nullable=True),
        sa.Column('success_outcome', sa.Boolean, nullable=True),
        
        # Temporal information
        sa.Column('assessed_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('usage_session_id', sa.String(255), nullable=True),
        
        # Indexes
        sa.Index('idx_quality_knowledge_assessor', 'knowledge_id', 'assessor_agent_id'),
        sa.Index('idx_quality_overall_score', 'overall_score'),
        sa.Index('idx_quality_assessed_at', 'assessed_at')
    )
    
    # Agent expertise records table
    op.create_table(
        'agent_expertise_records',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('agent_id', sa.String(255), nullable=False, index=True),
        sa.Column('domain', sa.String(255), nullable=False),
        sa.Column('capability', sa.String(255), nullable=False),
        
        # Expertise metrics
        sa.Column('proficiency_level', sa.Float, nullable=False),
        sa.Column('evidence_count', sa.Integer, default=1, nullable=False),
        sa.Column('success_rate', sa.Float, nullable=False),
        sa.Column('confidence_score', sa.Float, nullable=False),
        
        # Evidence and validation
        sa.Column('evidence_sources', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('validation_sources', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('peer_validations', sa.Integer, default=0, nullable=False),
        
        # Temporal tracking
        sa.Column('first_demonstrated', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('last_demonstrated', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        
        # Metadata
        sa.Column('expertise_metadata', postgresql.JSONB, nullable=True),
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        
        # Unique constraint
        sa.UniqueConstraint('agent_id', 'domain', 'capability', name='unique_agent_expertise'),
        
        # Indexes
        sa.Index('idx_expertise_domain_capability', 'domain', 'capability'),
        sa.Index('idx_expertise_proficiency', 'proficiency_level'),
        sa.Index('idx_expertise_success_rate', 'success_rate'),
        sa.Index('idx_expertise_last_demonstrated', 'last_demonstrated')
    )
    
    # =============================================================================
    # MEMORY HIERARCHY TABLES
    # =============================================================================
    
    # Memory hierarchy items table
    op.create_table(
        'memory_hierarchy_items',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('memory_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('agent_id', sa.String(255), nullable=False, index=True),
        
        # Content and classification
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('content_hash', sa.String(64), nullable=False),
        sa.Column('memory_type', sa.String(100), nullable=False),
        sa.Column('memory_level', sa.String(100), nullable=False),
        
        # Quality and importance metrics
        sa.Column('importance_score', sa.Float, default=0.5, nullable=False),
        sa.Column('relevance_score', sa.Float, default=0.5, nullable=False),
        sa.Column('confidence_score', sa.Float, default=0.5, nullable=False),
        sa.Column('completeness_score', sa.Float, default=0.5, nullable=False),
        
        # Context and relationships
        sa.Column('context_metadata', postgresql.JSONB, nullable=True),
        sa.Column('tags', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('related_memory_ids', postgresql.ARRAY(sa.String), nullable=True),
        
        # Task and workflow context
        sa.Column('task_id', sa.String(255), nullable=True),
        sa.Column('workflow_id', sa.String(255), nullable=True),
        sa.Column('session_id', sa.String(255), nullable=True),
        
        # Access and usage tracking
        sa.Column('access_count', sa.Integer, default=0, nullable=False),
        sa.Column('last_accessed', sa.DateTime(timezone=True), nullable=True),
        sa.Column('usage_frequency_score', sa.Float, default=0.0, nullable=False),
        
        # Lifecycle and aging
        sa.Column('decay_rate', sa.Float, default=0.1, nullable=False),
        sa.Column('consolidation_count', sa.Integer, default=0, nullable=False),
        sa.Column('archived', sa.Boolean, default=False, nullable=False),
        sa.Column('archived_at', sa.DateTime(timezone=True), nullable=True),
        
        # Temporal information
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        
        # Indexes
        sa.Index('idx_memory_agent_level', 'agent_id', 'memory_level'),
        sa.Index('idx_memory_type', 'memory_type'),
        sa.Index('idx_memory_importance', 'importance_score'),
        sa.Index('idx_memory_created', 'created_at'),
        sa.Index('idx_memory_last_accessed', 'last_accessed'),
        sa.Index('idx_memory_archived', 'archived', 'archived_at'),
        sa.Index('idx_memory_task_workflow', 'task_id', 'workflow_id')
    )
    
    # Memory consolidation records table
    op.create_table(
        'memory_consolidation_records',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('consolidation_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('agent_id', sa.String(255), nullable=False, index=True),
        sa.Column('trigger_type', sa.String(100), nullable=False),
        
        # Timing information
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('processing_time_ms', sa.Float, nullable=False),
        
        # Processing statistics
        sa.Column('memories_processed', sa.Integer, nullable=False),
        sa.Column('memories_compressed', sa.Integer, nullable=False),
        sa.Column('memories_promoted', sa.Integer, nullable=False),
        sa.Column('memories_archived', sa.Integer, nullable=False),
        sa.Column('memories_deleted', sa.Integer, nullable=False),
        
        # Size and quality metrics
        sa.Column('original_size_bytes', sa.BigInteger, nullable=False),
        sa.Column('compressed_size_bytes', sa.BigInteger, nullable=False),
        sa.Column('compression_ratio', sa.Float, nullable=False),
        sa.Column('avg_semantic_preservation', sa.Float, nullable=False),
        sa.Column('memories_with_high_preservation', sa.Integer, nullable=False),
        
        # Summary and metadata
        sa.Column('consolidation_summary', sa.Text, nullable=True),
        sa.Column('consolidation_metadata', postgresql.JSONB, nullable=True),
        sa.Column('success', sa.Boolean, default=True, nullable=False),
        sa.Column('error_message', sa.Text, nullable=True),
        
        # Indexes
        sa.Index('idx_consolidation_agent_started', 'agent_id', 'started_at'),
        sa.Index('idx_consolidation_trigger', 'trigger_type'),
        sa.Index('idx_consolidation_success', 'success'),
        sa.Index('idx_consolidation_compression_ratio', 'compression_ratio')
    )
    
    # =============================================================================
    # KNOWLEDGE GRAPH TABLES
    # =============================================================================
    
    # Knowledge graph nodes table
    op.create_table(
        'knowledge_graph_nodes',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('node_id', sa.String(255), nullable=False, index=True),
        sa.Column('graph_type', sa.String(100), nullable=False),
        sa.Column('node_type', sa.String(100), nullable=False),
        sa.Column('label', sa.String(500), nullable=False),
        
        # Node properties and metadata
        sa.Column('properties', postgresql.JSONB, nullable=True),
        sa.Column('embedding_vector', postgresql.ARRAY(sa.Float), nullable=True),
        sa.Column('centrality_scores', postgresql.JSONB, nullable=True),
        
        # Temporal information
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('last_analyzed', sa.DateTime(timezone=True), nullable=True),
        
        # Activity tracking
        sa.Column('activity_score', sa.Float, default=0.0, nullable=False),
        sa.Column('connection_count', sa.Integer, default=0, nullable=False),
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        
        # Unique constraint
        sa.UniqueConstraint('node_id', 'graph_type', name='unique_graph_node'),
        
        # Indexes
        sa.Index('idx_graph_node_type', 'graph_type', 'node_type'),
        sa.Index('idx_graph_node_label', 'label'),
        sa.Index('idx_graph_node_activity', 'activity_score'),
        sa.Index('idx_graph_node_updated', 'updated_at')
    )
    
    # Knowledge graph edges table
    op.create_table(
        'knowledge_graph_edges',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('edge_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('graph_type', sa.String(100), nullable=False),
        sa.Column('source_node_id', sa.String(255), nullable=False),
        sa.Column('target_node_id', sa.String(255), nullable=False),
        sa.Column('edge_type', sa.String(100), nullable=False),
        
        # Edge properties and weights
        sa.Column('weight', sa.Float, default=1.0, nullable=False),
        sa.Column('strength', sa.String(50), nullable=False),  # weak/moderate/strong
        sa.Column('properties', postgresql.JSONB, nullable=True),
        
        # Temporal information
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('last_traversed', sa.DateTime(timezone=True), nullable=True),
        
        # Activity and validation
        sa.Column('traversal_count', sa.Integer, default=0, nullable=False),
        sa.Column('validation_score', sa.Float, default=0.5, nullable=False),
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        
        # Foreign key constraints
        sa.ForeignKeyConstraint(['source_node_id', 'graph_type'], ['knowledge_graph_nodes.node_id', 'knowledge_graph_nodes.graph_type']),
        sa.ForeignKeyConstraint(['target_node_id', 'graph_type'], ['knowledge_graph_nodes.node_id', 'knowledge_graph_nodes.graph_type']),
        
        # Indexes
        sa.Index('idx_graph_edge_type', 'graph_type', 'edge_type'),
        sa.Index('idx_graph_edge_source', 'source_node_id'),
        sa.Index('idx_graph_edge_target', 'target_node_id'),
        sa.Index('idx_graph_edge_weight', 'weight'),
        sa.Index('idx_graph_edge_updated', 'last_updated')
    )
    
    # Knowledge graph analyses table
    op.create_table(
        'knowledge_graph_analyses',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('analysis_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('graph_type', sa.String(100), nullable=False),
        sa.Column('analysis_type', sa.String(100), nullable=False),
        
        # Analysis timestamp and performance
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('processing_time_ms', sa.Float, nullable=False),
        
        # Basic graph metrics
        sa.Column('node_count', sa.Integer, nullable=False),
        sa.Column('edge_count', sa.Integer, nullable=False),
        sa.Column('density', sa.Float, nullable=False),
        sa.Column('clustering_coefficient', sa.Float, nullable=False),
        
        # Centrality measures
        sa.Column('top_central_nodes', postgresql.JSONB, nullable=True),
        sa.Column('betweenness_centrality', postgresql.JSONB, nullable=True),
        sa.Column('eigenvector_centrality', postgresql.JSONB, nullable=True),
        
        # Community and flow analysis
        sa.Column('communities', postgresql.JSONB, nullable=True),
        sa.Column('modularity', sa.Float, nullable=True),
        sa.Column('knowledge_flow_rate', sa.Float, nullable=True),
        sa.Column('active_knowledge_paths', sa.Integer, nullable=True),
        sa.Column('bottleneck_nodes', postgresql.ARRAY(sa.String), nullable=True),
        
        # Recommendations
        sa.Column('collaboration_recommendations', postgresql.JSONB, nullable=True),
        sa.Column('knowledge_sharing_opportunities', postgresql.JSONB, nullable=True),
        
        # Analysis metadata
        sa.Column('analysis_metadata', postgresql.JSONB, nullable=True),
        sa.Column('cache_key', sa.String(255), nullable=True, index=True),
        
        # Indexes
        sa.Index('idx_graph_analysis_type', 'graph_type', 'analysis_type'),
        sa.Index('idx_graph_analysis_timestamp', 'timestamp'),
        sa.Index('idx_graph_analysis_cache', 'cache_key')
    )
    
    # =============================================================================
    # CONTEXT RELEVANCE TABLES
    # =============================================================================
    
    # Context scoring sessions table
    op.create_table(
        'context_scoring_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('session_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('agent_id', sa.String(255), nullable=True, index=True),
        sa.Column('task_id', sa.String(255), nullable=True),
        sa.Column('workflow_id', sa.String(255), nullable=True),
        
        # Query and strategy
        sa.Column('query', sa.Text, nullable=False),
        sa.Column('query_hash', sa.String(64), nullable=False, index=True),
        sa.Column('scoring_strategy', sa.String(100), nullable=False),
        
        # Results and performance
        sa.Column('contexts_evaluated', sa.Integer, nullable=False),
        sa.Column('contexts_returned', sa.Integer, nullable=False),
        sa.Column('processing_time_ms', sa.Float, nullable=False),
        sa.Column('top_score', sa.Float, nullable=True),
        sa.Column('avg_score', sa.Float, nullable=True),
        
        # Configuration used
        sa.Column('scoring_config', postgresql.JSONB, nullable=True),
        sa.Column('filter_criteria', postgresql.JSONB, nullable=True),
        
        # Temporal information
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('cache_hit', sa.Boolean, default=False, nullable=False),
        
        # Indexes
        sa.Index('idx_scoring_agent_created', 'agent_id', 'created_at'),
        sa.Index('idx_scoring_strategy', 'scoring_strategy'),
        sa.Index('idx_scoring_performance', 'processing_time_ms'),
        sa.Index('idx_scoring_query_hash', 'query_hash')
    )
    
    # Context relevance scores table
    op.create_table(
        'context_relevance_scores',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('session_id', sa.String(255), nullable=False, index=True),
        sa.Column('context_id', sa.String(255), nullable=False, index=True),
        
        # Overall and factor scores
        sa.Column('overall_score', sa.Float, nullable=False),
        sa.Column('semantic_similarity', sa.Float, nullable=False),
        sa.Column('keyword_overlap', sa.Float, nullable=False),
        sa.Column('recency_score', sa.Float, nullable=False),
        sa.Column('importance_score', sa.Float, nullable=False),
        sa.Column('usage_score', sa.Float, nullable=False),
        sa.Column('task_specificity', sa.Float, nullable=False),
        
        # Scoring metadata
        sa.Column('scoring_time_ms', sa.Float, nullable=False),
        sa.Column('explanation', sa.Text, nullable=True),
        sa.Column('rank_position', sa.Integer, nullable=True),
        
        # Context metadata
        sa.Column('context_type', sa.String(100), nullable=True),
        sa.Column('context_metadata', postgresql.JSONB, nullable=True),
        
        # Feedback tracking
        sa.Column('feedback_provided', sa.Boolean, default=False, nullable=False),
        sa.Column('feedback_score', sa.Float, nullable=True),
        sa.Column('feedback_type', sa.String(100), nullable=True),
        
        # Foreign key constraint
        sa.ForeignKeyConstraint(['session_id'], ['context_scoring_sessions.session_id'], ondelete='CASCADE'),
        
        # Indexes
        sa.Index('idx_relevance_session_score', 'session_id', 'overall_score'),
        sa.Index('idx_relevance_context', 'context_id'),
        sa.Index('idx_relevance_overall_score', 'overall_score'),
        sa.Index('idx_relevance_feedback', 'feedback_provided', 'feedback_score')
    )
    
    # =============================================================================
    # COLLABORATION AND LEARNING TABLES
    # =============================================================================
    
    # Collaborative learning outcomes table
    op.create_table(
        'collaborative_learning_outcomes',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('outcome_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('workflow_id', sa.String(255), nullable=False, index=True),
        sa.Column('collaboration_pattern', sa.String(100), nullable=False),
        
        # Participating agents
        sa.Column('participating_agents', postgresql.ARRAY(sa.String), nullable=False),
        sa.Column('primary_agent_id', sa.String(255), nullable=True),
        sa.Column('knowledge_shared', postgresql.ARRAY(sa.String), nullable=True),
        
        # Success metrics
        sa.Column('success_metrics', postgresql.JSONB, nullable=True),
        sa.Column('overall_success_score', sa.Float, nullable=True),
        sa.Column('completion_time_ms', sa.BigInteger, nullable=True),
        sa.Column('efficiency_score', sa.Float, nullable=True),
        sa.Column('quality_score', sa.Float, nullable=True),
        
        # Learning insights
        sa.Column('lessons_learned', postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column('best_practices', postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column('patterns_identified', postgresql.JSONB, nullable=True),
        sa.Column('improvement_opportunities', postgresql.ARRAY(sa.Text), nullable=True),
        
        # Temporal information
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('workflow_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('workflow_completed_at', sa.DateTime(timezone=True), nullable=True),
        
        # Metadata
        sa.Column('collaboration_metadata', postgresql.JSONB, nullable=True),
        sa.Column('applied_to_future_work', sa.Boolean, default=False, nullable=False),
        
        # Indexes
        sa.Index('idx_learning_workflow', 'workflow_id'),
        sa.Index('idx_learning_pattern', 'collaboration_pattern'),
        sa.Index('idx_learning_success', 'overall_success_score'),
        sa.Index('idx_learning_created', 'created_at')
    )
    
    # Agent collaboration history table
    op.create_table(
        'agent_collaboration_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('collaboration_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('agent1_id', sa.String(255), nullable=False, index=True),
        sa.Column('agent2_id', sa.String(255), nullable=False, index=True),
        
        # Collaboration details
        sa.Column('collaboration_type', sa.String(100), nullable=False),
        sa.Column('context', sa.Text, nullable=True),
        sa.Column('duration_hours', sa.Float, nullable=True),
        sa.Column('success', sa.Boolean, nullable=False),
        sa.Column('outcome_summary', sa.Text, nullable=True),
        
        # Domains and capabilities involved
        sa.Column('domains', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('capabilities_used', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('knowledge_exchanged', sa.Integer, default=0, nullable=False),
        
        # Quality metrics
        sa.Column('effectiveness_score', sa.Float, nullable=True),
        sa.Column('satisfaction_agent1', sa.Float, nullable=True),
        sa.Column('satisfaction_agent2', sa.Float, nullable=True),
        sa.Column('learning_value', sa.Float, nullable=True),
        
        # Temporal information
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        
        # Reference to workflow or task
        sa.Column('workflow_id', sa.String(255), nullable=True),
        sa.Column('task_ids', postgresql.ARRAY(sa.String), nullable=True),
        
        # Metadata
        sa.Column('collaboration_metadata', postgresql.JSONB, nullable=True),
        
        # Indexes
        sa.Index('idx_collaboration_agents', 'agent1_id', 'agent2_id'),
        sa.Index('idx_collaboration_type', 'collaboration_type'),
        sa.Index('idx_collaboration_success', 'success'),
        sa.Index('idx_collaboration_started', 'started_at'),
        sa.Index('idx_collaboration_effectiveness', 'effectiveness_score')
    )
    
    # =============================================================================
    # PERFORMANCE AND ANALYTICS TABLES
    # =============================================================================
    
    # System performance metrics table
    op.create_table(
        'system_performance_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('metric_id', sa.String(255), nullable=False, index=True),
        sa.Column('component', sa.String(100), nullable=False),
        sa.Column('metric_type', sa.String(100), nullable=False),
        sa.Column('metric_name', sa.String(255), nullable=False),
        
        # Metric values
        sa.Column('metric_value', sa.Float, nullable=False),
        sa.Column('metric_unit', sa.String(50), nullable=True),
        sa.Column('target_value', sa.Float, nullable=True),
        sa.Column('threshold_warning', sa.Float, nullable=True),
        sa.Column('threshold_critical', sa.Float, nullable=True),
        
        # Context and metadata
        sa.Column('agent_id', sa.String(255), nullable=True, index=True),
        sa.Column('session_id', sa.String(255), nullable=True),
        sa.Column('operation_id', sa.String(255), nullable=True),
        sa.Column('metric_metadata', postgresql.JSONB, nullable=True),
        
        # Temporal information
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('aggregation_period', sa.String(50), nullable=True),  # e.g., '1h', '1d'
        
        # Indexes
        sa.Index('idx_metrics_component_type', 'component', 'metric_type'),
        sa.Index('idx_metrics_name_recorded', 'metric_name', 'recorded_at'),
        sa.Index('idx_metrics_agent_recorded', 'agent_id', 'recorded_at'),
        sa.Index('idx_metrics_value', 'metric_value')
    )


def downgrade() -> None:
    """Downgrade database schema by dropping context compression and knowledge sharing extensions."""
    
    # Drop tables in reverse order of creation to handle foreign key constraints
    op.drop_table('system_performance_metrics')
    op.drop_table('agent_collaboration_history')
    op.drop_table('collaborative_learning_outcomes')
    op.drop_table('context_relevance_scores')
    op.drop_table('context_scoring_sessions')
    op.drop_table('knowledge_graph_analyses')
    op.drop_table('knowledge_graph_edges')
    op.drop_table('knowledge_graph_nodes')
    op.drop_table('memory_consolidation_records')
    op.drop_table('memory_hierarchy_items')
    op.drop_table('agent_expertise_records')
    op.drop_table('knowledge_quality_assessments')
    op.drop_table('knowledge_sharing_events')
    op.drop_table('compression_configs')
    op.drop_table('compression_operations')