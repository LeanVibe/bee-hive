"""Add Context Engine enhancements - relationships and analytics

Revision ID: 005
Revises: 004
Create Date: 2024-01-01 18:00:00.000000

Adds enhanced schema for:
- Context relationships tracking
- Context retrieval analytics  
- Access level controls
- Performance optimization indexes
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add Context Engine enhancements."""
    
    # Create enums for new schema features
    op.execute("""
        CREATE TYPE relationship_type_enum AS ENUM (
            'SIMILAR',
            'RELATED', 
            'DERIVED_FROM',
            'SUPERSEDES',
            'REFERENCES',
            'CONTRADICTS'
        )
    """)
    
    op.execute("""
        CREATE TYPE access_level_enum AS ENUM (
            'PRIVATE',
            'AGENT_SHARED',
            'SESSION_SHARED', 
            'PUBLIC'
        )
    """)
    
    # Add access_level to existing contexts table
    op.add_column('contexts', sa.Column(
        'access_level', 
        sa.String(30),
        nullable=False,
        server_default='PRIVATE'
    ))
    
    # Create context_relationships table
    op.create_table(
        'context_relationships',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('source_context_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('target_context_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('relationship_type', sa.String(30), nullable=False, index=True),
        sa.Column('similarity_score', sa.Float, nullable=True),
        sa.Column('confidence_score', sa.Float, nullable=False, server_default='1.0'),
        sa.Column('metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['source_context_id'], ['contexts.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['target_context_id'], ['contexts.id'], ondelete='CASCADE'),
        # Prevent duplicate relationships
        sa.UniqueConstraint('source_context_id', 'target_context_id', 'relationship_type', name='unique_context_relationship')
    )
    
    # Create context_retrievals table for analytics
    op.create_table(
        'context_retrievals',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('context_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('requesting_agent_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('query_text', sa.Text, nullable=True),
        sa.Column('query_embedding', Vector(1536), nullable=True),
        sa.Column('similarity_score', sa.Float, nullable=True),
        sa.Column('relevance_score', sa.Float, nullable=True),
        sa.Column('rank_position', sa.Integer, nullable=True),
        sa.Column('was_helpful', sa.Boolean, nullable=True),
        sa.Column('feedback_score', sa.Integer, nullable=True),
        sa.Column('retrieval_method', sa.String(50), nullable=False, server_default='semantic_search'),
        sa.Column('response_time_ms', sa.Float, nullable=True),
        sa.Column('metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('retrieved_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.ForeignKeyConstraint(['context_id'], ['contexts.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['requesting_agent_id'], ['agents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ondelete='SET NULL'),
        # Check constraint for feedback score
        sa.CheckConstraint('feedback_score IS NULL OR (feedback_score >= 1 AND feedback_score <= 5)', name='valid_feedback_score')
    )
    
    # Create context_compression_history table
    op.create_table(
        'context_compression_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('context_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('compression_level', sa.String(20), nullable=False),
        sa.Column('original_token_count', sa.Integer, nullable=False),
        sa.Column('compressed_token_count', sa.Integer, nullable=False),
        sa.Column('compression_ratio', sa.Float, nullable=False),
        sa.Column('key_insights_extracted', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('decisions_preserved', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('patterns_identified', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('compression_time_ms', sa.Float, nullable=True),
        sa.Column('model_used', sa.String(100), nullable=True),
        sa.Column('compressed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.ForeignKeyConstraint(['context_id'], ['contexts.id'], ondelete='CASCADE')
    )
    
    # Create context_usage_analytics table for aggregated metrics
    op.create_table(
        'context_usage_analytics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('context_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('total_retrievals', sa.Integer, nullable=False, server_default='0'),
        sa.Column('successful_retrievals', sa.Integer, nullable=False, server_default='0'),
        sa.Column('average_similarity_score', sa.Float, nullable=True),
        sa.Column('average_relevance_score', sa.Float, nullable=True),
        sa.Column('average_response_time_ms', sa.Float, nullable=True),
        sa.Column('last_retrieved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('cross_agent_shares', sa.Integer, nullable=False, server_default='0'),
        sa.Column('feedback_scores', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['context_id'], ['contexts.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
        # Unique constraint for context-agent combination
        sa.UniqueConstraint('context_id', 'agent_id', name='unique_context_agent_analytics')
    )
    
    # Add performance optimization indexes
    
    # Context relationships indexes
    op.create_index('idx_context_relationships_source', 'context_relationships', ['source_context_id'])
    op.create_index('idx_context_relationships_target', 'context_relationships', ['target_context_id'])
    op.create_index('idx_context_relationships_type_score', 'context_relationships', ['relationship_type', 'similarity_score'])
    
    # Context retrievals indexes for analytics queries
    op.create_index('idx_context_retrievals_context_time', 'context_retrievals', ['context_id', 'retrieved_at'])
    op.create_index('idx_context_retrievals_agent_time', 'context_retrievals', ['requesting_agent_id', 'retrieved_at'])
    op.create_index('idx_context_retrievals_session_time', 'context_retrievals', ['session_id', 'retrieved_at'])
    op.create_index('idx_context_retrievals_similarity', 'context_retrievals', ['similarity_score'])
    op.create_index('idx_context_retrievals_relevance', 'context_retrievals', ['relevance_score'])
    
    # Context compression history indexes
    op.create_index('idx_context_compression_context_time', 'context_compression_history', ['context_id', 'compressed_at'])
    op.create_index('idx_context_compression_ratio', 'context_compression_history', ['compression_ratio'])
    
    # Context usage analytics indexes
    op.create_index('idx_context_analytics_retrievals', 'context_usage_analytics', ['total_retrievals'])
    op.create_index('idx_context_analytics_agent_updated', 'context_usage_analytics', ['agent_id', 'updated_at'])
    
    # Enhanced context indexes for better query performance
    op.create_index('idx_contexts_access_level_importance', 'contexts', ['access_level', 'importance_score'])
    op.create_index('idx_contexts_agent_access_level', 'contexts', ['agent_id', 'access_level'])
    op.create_index('idx_contexts_type_created', 'contexts', ['context_type', 'created_at'])
    op.create_index('idx_contexts_consolidated_importance', 'contexts', ['is_consolidated', 'importance_score'])
    
    # Vector similarity search optimization for relationships
    op.create_index(
        'idx_context_relationships_similarity_score',
        'context_relationships',
        ['similarity_score'],
        postgresql_where=sa.text('similarity_score IS NOT NULL')
    )
    
    # Add triggers for automatic analytics updates
    op.execute("""
        CREATE OR REPLACE FUNCTION update_context_analytics()
        RETURNS TRIGGER AS $$
        BEGIN
            -- Update context access tracking
            UPDATE contexts 
            SET accessed_at = NEW.retrieved_at,
                access_count = CAST(COALESCE(NULLIF(access_count, ''), '0') AS INTEGER) + 1
            WHERE id = NEW.context_id;
            
            -- Update or insert usage analytics
            INSERT INTO context_usage_analytics (
                context_id, 
                agent_id, 
                total_retrievals, 
                last_retrieved_at,
                updated_at
            )
            VALUES (
                NEW.context_id, 
                NEW.requesting_agent_id, 
                1, 
                NEW.retrieved_at,
                NOW()
            )
            ON CONFLICT (context_id, agent_id) 
            DO UPDATE SET
                total_retrievals = context_usage_analytics.total_retrievals + 1,
                last_retrieved_at = NEW.retrieved_at,
                updated_at = NOW();
            
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    op.execute("""
        CREATE TRIGGER context_retrieval_analytics_trigger
        AFTER INSERT ON context_retrievals
        FOR EACH ROW
        EXECUTE FUNCTION update_context_analytics();
    """)
    
    # Add function for cleaning up old analytics data
    op.execute("""
        CREATE OR REPLACE FUNCTION cleanup_old_context_analytics(days_to_keep INTEGER DEFAULT 90)
        RETURNS INTEGER AS $$
        DECLARE
            deleted_count INTEGER;
        BEGIN
            -- Clean up old retrieval records
            DELETE FROM context_retrievals 
            WHERE retrieved_at < NOW() - INTERVAL '1 day' * days_to_keep;
            
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            
            -- Clean up compression history older than 1 year
            DELETE FROM context_compression_history 
            WHERE compressed_at < NOW() - INTERVAL '365 days';
            
            RETURN deleted_count;
        END;
        $$ LANGUAGE plpgsql;
    """)


def downgrade() -> None:
    """Remove Context Engine enhancements."""
    
    # Drop triggers and functions
    op.execute('DROP TRIGGER IF EXISTS context_retrieval_analytics_trigger ON context_retrievals;')
    op.execute('DROP FUNCTION IF EXISTS update_context_analytics();')
    op.execute('DROP FUNCTION IF EXISTS cleanup_old_context_analytics(INTEGER);')
    
    # Drop indexes
    op.drop_index('idx_contexts_consolidated_importance')
    op.drop_index('idx_contexts_type_created')
    op.drop_index('idx_contexts_agent_access_level')
    op.drop_index('idx_contexts_access_level_importance')
    op.drop_index('idx_context_analytics_agent_updated')
    op.drop_index('idx_context_analytics_retrievals')
    op.drop_index('idx_context_compression_ratio')
    op.drop_index('idx_context_compression_context_time')
    op.drop_index('idx_context_retrievals_relevance')
    op.drop_index('idx_context_retrievals_similarity')
    op.drop_index('idx_context_retrievals_session_time')
    op.drop_index('idx_context_retrievals_agent_time')
    op.drop_index('idx_context_retrievals_context_time')
    op.drop_index('idx_context_relationships_type_score')
    op.drop_index('idx_context_relationships_target')
    op.drop_index('idx_context_relationships_source')
    op.drop_index('idx_context_relationships_similarity_score')
    
    # Drop tables in reverse order
    op.drop_table('context_usage_analytics')
    op.drop_table('context_compression_history')
    op.drop_table('context_retrievals')
    op.drop_table('context_relationships')
    
    # Remove access_level column from contexts
    op.drop_column('contexts', 'access_level')
    
    # Drop custom enums
    op.execute('DROP TYPE IF EXISTS access_level_enum')
    op.execute('DROP TYPE IF EXISTS relationship_type_enum')