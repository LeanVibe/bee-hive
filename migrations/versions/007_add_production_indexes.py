"""Add production-grade performance indexes for Context Engine

Revision ID: 007
Revises: 006
Create Date: 2024-01-01 20:00:00.000000

Critical performance optimizations:
- IVFFlat vector indexes for <50ms search performance
- Composite indexes for complex queries
- Query optimization for time-series and analytics
- Memory-efficient index configurations
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add production-grade performance optimizations."""
    
    # Create IVFFlat vector index for primary context search
    # Using optimal parameters for production workload
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_contexts_embedding_ivfflat 
        ON contexts USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100);
    """)
    
    # Create additional IVFFlat index for context_retrievals query embeddings
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_context_retrievals_query_embedding_ivfflat 
        ON context_retrievals USING ivfflat (query_embedding vector_cosine_ops) 
        WITH (lists = 50);
    """)
    
    # Optimize existing indexes for better performance
    op.execute("DROP INDEX IF EXISTS idx_contexts_embedding;")  # Remove old index if exists
    
    # Create optimized composite indexes for frequent query patterns
    
    # Agent + access level + importance for cross-agent sharing queries
    op.create_index(
        'idx_contexts_agent_access_importance_opt',
        'contexts',
        ['agent_id', 'access_level', 'importance_score'],
        postgresql_where=sa.text("embedding IS NOT NULL")
    )
    
    # Time-based queries with embedding availability
    op.create_index(
        'idx_contexts_created_embedding_opt',
        'contexts',
        ['created_at', 'importance_score'],
        postgresql_where=sa.text("embedding IS NOT NULL AND (metadata->>'archived') IS DISTINCT FROM 'true'")
    )
    
    # Accessed frequency for consolidation queries
    op.create_index(
        'idx_contexts_access_consolidation_opt',
        'contexts',
        ['access_count', 'is_consolidated', 'importance_score'],
        postgresql_where=sa.text("embedding IS NOT NULL")
    )
    
    # Context type + importance for filtered searches
    op.create_index(
        'idx_contexts_type_importance_embedding',
        'contexts',
        ['context_type', 'importance_score'],
        postgresql_where=sa.text("embedding IS NOT NULL")
    )
    
    # Analytics optimization indexes
    
    # Context retrievals by agent and time for analytics
    op.create_index(
        'idx_retrievals_agent_time_similarity',
        'context_retrievals',
        ['requesting_agent_id', 'retrieved_at', 'similarity_score']
    )
    
    # Context usage analytics optimization
    op.create_index(
        'idx_usage_analytics_context_retrievals',
        'context_usage_analytics',
        ['context_id', 'total_retrievals'],
        postgresql_where=sa.text("total_retrievals > 0")
    )
    
    # Relationship queries optimization
    op.create_index(
        'idx_relationships_source_type_score',
        'context_relationships',
        ['source_context_id', 'relationship_type', 'similarity_score']
    )
    
    op.create_index(
        'idx_relationships_target_type_score',
        'context_relationships',
        ['target_context_id', 'relationship_type', 'similarity_score']
    )
    
    # Session-based context queries
    op.create_index(
        'idx_contexts_session_importance',
        'contexts',
        ['session_id', 'importance_score'],
        postgresql_where=sa.text("session_id IS NOT NULL AND embedding IS NOT NULL")
    )
    
    # Add partial indexes for archived content
    op.create_index(
        'idx_contexts_archived_cleanup',
        'contexts',
        ['created_at', 'importance_score'],
        postgresql_where=sa.text("(metadata->>'archived') = 'true'")
    )
    
    # Performance monitoring view for index usage
    op.execute("""
        CREATE OR REPLACE VIEW context_index_usage AS
        SELECT 
            schemaname,
            relname as tablename,
            indexrelname as indexname,
            idx_scan,
            idx_tup_read,
            idx_tup_fetch
        FROM pg_stat_user_indexes 
        WHERE schemaname = 'public' 
        AND (relname LIKE 'context%' OR relname IN ('agents', 'sessions'));
    """)
    
    # Add function to monitor search performance
    op.execute("""
        CREATE OR REPLACE FUNCTION analyze_search_performance(
            time_window_hours INTEGER DEFAULT 24
        ) RETURNS TABLE (
            avg_response_time_ms FLOAT,
            total_searches BIGINT,
            cache_hit_rate FLOAT,
            top_queries TEXT[]
        ) AS $$
        DECLARE
            cutoff_time TIMESTAMP := NOW() - INTERVAL '1 hour' * time_window_hours;
        BEGIN
            RETURN QUERY
            WITH top_query_counts AS (
                SELECT cr.query_text, COUNT(*) as query_count
                FROM context_retrievals cr
                WHERE cr.retrieved_at >= cutoff_time
                AND cr.response_time_ms IS NOT NULL
                AND cr.query_text IS NOT NULL
                GROUP BY cr.query_text
                ORDER BY query_count DESC
                LIMIT 10
            )
            SELECT 
                AVG(cr.response_time_ms) as avg_response_time_ms,
                COUNT(*) as total_searches,
                AVG(CASE WHEN cr.metadata->>'cache_hit' = 'true' THEN 1.0 ELSE 0.0 END) as cache_hit_rate,
                ARRAY(SELECT query_text FROM top_query_counts) as top_queries
            FROM context_retrievals cr
            WHERE cr.retrieved_at >= cutoff_time
            AND cr.response_time_ms IS NOT NULL;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Add function to find optimal index configuration
    op.execute("""
        CREATE OR REPLACE FUNCTION suggest_index_optimization()
        RETURNS TABLE (
            table_name TEXT,
            suggestion TEXT,
            estimated_impact TEXT
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                'contexts'::TEXT as table_name,
                'Consider partitioning by created_at for better performance'::TEXT as suggestion,
                'High - for time-based queries'::TEXT as estimated_impact
            WHERE (SELECT COUNT(*) FROM contexts) > 100000
            
            UNION ALL
            
            SELECT 
                'context_retrievals'::TEXT,
                'Consider archiving old retrieval data'::TEXT,
                'Medium - reduces index size'::TEXT
            WHERE (SELECT COUNT(*) FROM context_retrievals WHERE retrieved_at < NOW() - INTERVAL '90 days') > 10000;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Update table statistics for optimal query planning
    op.execute("ANALYZE contexts;")
    op.execute("ANALYZE context_relationships;")
    op.execute("ANALYZE context_retrievals;")
    op.execute("ANALYZE context_usage_analytics;")


def downgrade() -> None:
    """Remove production performance optimizations."""
    
    # Drop custom functions
    op.execute('DROP FUNCTION IF EXISTS suggest_index_optimization();')
    op.execute('DROP FUNCTION IF EXISTS analyze_search_performance(INTEGER);')
    op.execute('DROP VIEW IF EXISTS context_index_usage;')
    
    # Drop performance indexes (in reverse order)
    op.drop_index('idx_contexts_archived_cleanup')
    op.drop_index('idx_contexts_session_importance')
    op.drop_index('idx_relationships_target_type_score')
    op.drop_index('idx_relationships_source_type_score')
    op.drop_index('idx_usage_analytics_context_retrievals')
    op.drop_index('idx_retrievals_agent_time_similarity')
    op.drop_index('idx_contexts_type_importance_embedding')
    op.drop_index('idx_contexts_access_consolidation_opt')
    op.drop_index('idx_contexts_created_embedding_opt')
    op.drop_index('idx_contexts_agent_access_importance_opt')
    
    # Drop IVFFlat indexes
    op.execute('DROP INDEX IF EXISTS idx_context_retrievals_query_embedding_ivfflat;')
    op.execute('DROP INDEX IF EXISTS idx_contexts_embedding_ivfflat;')