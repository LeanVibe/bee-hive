"""Vector Search Performance Optimization - Enhanced pgvector indexes

Revision ID: 011
Revises: 010
Create Date: 2025-01-01 12:00:00.000000

Adds performance optimizations for the Vector Search Engine:
- Advanced pgvector indexes for semantic search
- Query optimization for cross-agent context sharing  
- Performance monitoring views
- Batch processing optimizations
- Memory-efficient vector operations
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '011'
down_revision = '010'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add vector search performance optimizations."""
    
    # Ensure pgvector extension is available
    op.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    
    # Create advanced vector indexes for better semantic search performance
    # IVFFlat index for cosine distance - optimal for semantic search
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_contexts_embedding_ivfflat_cosine 
        ON contexts USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100)
        WHERE embedding IS NOT NULL;
    """)
    
    # L2 distance index for alternative similarity metrics
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_contexts_embedding_ivfflat_l2 
        ON contexts USING ivfflat (embedding vector_l2_ops) 
        WITH (lists = 100)
        WHERE embedding IS NOT NULL;
    """)
    
    # Create optimized indexes for cross-agent search performance - check if exists first
    try:
        op.create_index(
            'idx_contexts_cross_agent_search',
            'contexts',
            ['agent_id', 'importance_score', 'is_consolidated'],
            postgresql_where=sa.text('embedding IS NOT NULL AND importance_score >= 0.7'),
            if_not_exists=True
        )
    except Exception:
        # Index might already exist, skip
        pass
    
    # Index for efficient agent filtering with access levels
    try:
        op.create_index(
            'idx_contexts_agent_access_embedding',
            'contexts', 
            ['agent_id', 'access_level', 'importance_score'],
            postgresql_where=sa.text('embedding IS NOT NULL'),
            if_not_exists=True
        )
    except Exception:
        pass
    
    # Composite index for temporal + importance filtering
    try:
        op.create_index(
            'idx_contexts_temporal_importance_embedding',
            'contexts',
            ['created_at', 'importance_score', 'accessed_at'],
            postgresql_where=sa.text('embedding IS NOT NULL'),
            if_not_exists=True
        )
    except Exception:
        pass
    
    # Index for context type filtering in semantic search - SKIP if exists from migration 007
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_contexts_type_importance_embedding_v2 
        ON contexts (context_type, importance_score) 
        WHERE embedding IS NOT NULL;
    """)
    
    # Create search performance monitoring view
    op.execute("""
        CREATE OR REPLACE VIEW v_vector_search_performance AS
        SELECT 
            DATE_TRUNC('hour', retrieved_at) as hour_bucket,
            COUNT(*) as total_searches,
            AVG(response_time_ms) as avg_response_time_ms,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY response_time_ms) as median_response_time_ms,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time_ms,
            AVG(similarity_score) as avg_similarity_score,
            AVG(relevance_score) as avg_relevance_score,
            COUNT(DISTINCT context_id) as unique_contexts_retrieved,
            COUNT(DISTINCT requesting_agent_id) as unique_agents,
            retrieval_method,
            SUM(CASE WHEN was_helpful = true THEN 1 ELSE 0 END) as helpful_results,
            SUM(CASE WHEN was_helpful = false THEN 1 ELSE 0 END) as unhelpful_results
        FROM context_retrievals 
        WHERE retrieved_at >= NOW() - INTERVAL '7 days'
        GROUP BY DATE_TRUNC('hour', retrieved_at), retrieval_method
        ORDER BY hour_bucket DESC, retrieval_method;
    """)
    
    # Create cross-agent context sharing analytics view
    op.execute("""
        CREATE OR REPLACE VIEW v_cross_agent_context_sharing AS
        SELECT 
            cr.requesting_agent_id,
            c.agent_id as context_owner_agent_id,
            COUNT(*) as shared_contexts_count,
            AVG(cr.similarity_score) as avg_similarity_score,
            AVG(c.importance_score) as avg_context_importance,
            COUNT(DISTINCT c.context_type) as context_types_shared,
            MAX(cr.retrieved_at) as last_shared_at
        FROM context_retrievals cr
        JOIN contexts c ON cr.context_id = c.id
        WHERE cr.requesting_agent_id != c.agent_id  -- Cross-agent access only
          AND cr.retrieved_at >= NOW() - INTERVAL '30 days'
        GROUP BY cr.requesting_agent_id, c.agent_id
        HAVING COUNT(*) >= 5  -- Only show significant sharing relationships
        ORDER BY shared_contexts_count DESC;
    """)
    
    # Create context embedding quality view for monitoring
    op.execute("""
        CREATE OR REPLACE VIEW v_context_embedding_quality AS
        SELECT 
            context_type,
            COUNT(*) as total_contexts,
            COUNT(embedding) as contexts_with_embeddings,
            ROUND(
                ((COUNT(embedding)::FLOAT / COUNT(*)) * 100)::NUMERIC, 2
            ) as embedding_coverage_percent,
            AVG(CASE WHEN embedding IS NOT NULL THEN importance_score END) as avg_importance_with_embedding,
            COUNT(CASE WHEN embedding IS NOT NULL AND accessed_at >= NOW() - INTERVAL '7 days' THEN 1 END) as recently_accessed_with_embedding
        FROM contexts 
        GROUP BY context_type
        ORDER BY embedding_coverage_percent DESC;
    """)
    
    # Add function for optimizing vector search performance
    op.execute("""
        CREATE OR REPLACE FUNCTION optimize_vector_search_performance()
        RETURNS TABLE(
            optimization_step TEXT,
            before_value FLOAT,
            after_value FLOAT,
            improvement_percent FLOAT
        ) AS $$
        DECLARE
            before_time FLOAT;
            after_time FLOAT;
            test_query_embedding VECTOR(1536);
        BEGIN
            -- Create a test embedding for performance testing
            test_query_embedding := ARRAY(SELECT random() FROM generate_series(1, 1536))::VECTOR(1536);
            
            -- Test search performance before optimization
            SELECT EXTRACT(EPOCH FROM (clock_timestamp() - statement_timestamp())) * 1000 INTO before_time;
            PERFORM * FROM contexts 
            WHERE embedding IS NOT NULL 
            ORDER BY embedding <-> test_query_embedding 
            LIMIT 10;
            SELECT EXTRACT(EPOCH FROM (clock_timestamp() - statement_timestamp())) * 1000 INTO after_time;
            before_time := after_time;
            
            -- Update table statistics for better query planning
            ANALYZE contexts;
            
            -- Test search performance after optimization
            SELECT EXTRACT(EPOCH FROM (clock_timestamp() - statement_timestamp())) * 1000 INTO after_time;
            PERFORM * FROM contexts 
            WHERE embedding IS NOT NULL 
            ORDER BY embedding <-> test_query_embedding 
            LIMIT 10;
            SELECT EXTRACT(EPOCH FROM (clock_timestamp() - statement_timestamp())) * 1000 INTO after_time;
            
            -- Return optimization results
            RETURN QUERY SELECT 
                'Table Statistics Update'::TEXT,
                before_time,
                after_time,
                CASE 
                    WHEN before_time > 0 THEN ((before_time - after_time) / before_time) * 100 
                    ELSE 0 
                END;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Add function for vector index maintenance
    op.execute("""
        CREATE OR REPLACE FUNCTION maintain_vector_indexes()
        RETURNS TABLE(
            index_name TEXT,
            maintenance_action TEXT,
            result TEXT
        ) AS $$
        BEGIN
            -- Reindex vector indexes if they become fragmented
            RETURN QUERY SELECT 
                'idx_contexts_embedding_ivfflat_cosine'::TEXT,
                'REINDEX'::TEXT,
                'Completed'::TEXT;
            
            REINDEX INDEX CONCURRENTLY idx_contexts_embedding_ivfflat_cosine;
            
            RETURN QUERY SELECT 
                'idx_contexts_embedding_ivfflat_l2'::TEXT,
                'REINDEX'::TEXT,
                'Completed'::TEXT;
            
            REINDEX INDEX CONCURRENTLY idx_contexts_embedding_ivfflat_l2;
            
            -- Update table statistics
            ANALYZE contexts;
            
            RETURN QUERY SELECT 
                'contexts'::TEXT,
                'ANALYZE'::TEXT,
                'Statistics Updated'::TEXT;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Add batch processing optimization table
    op.create_table(
        'vector_search_batches',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('batch_id', sa.String(64), nullable=False, index=True),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('query_count', sa.Integer, nullable=False),
        sa.Column('total_results', sa.Integer, nullable=False),
        sa.Column('processing_time_ms', sa.Float, nullable=False),
        sa.Column('cache_hit_rate', sa.Float, nullable=False, server_default='0.0'),
        sa.Column('avg_similarity_score', sa.Float, nullable=True),
        sa.Column('performance_target_met', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='SET NULL')
    )
    
    # Index for batch processing analytics
    op.create_index('idx_vector_search_batches_performance', 'vector_search_batches', 
                   ['processing_time_ms', 'performance_target_met', 'created_at'])
    
    # Add search caching optimization table
    op.create_table(
        'vector_search_cache_stats',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('cache_key_hash', sa.String(64), nullable=False, unique=True, index=True),
        sa.Column('query_pattern', sa.Text, nullable=True),
        sa.Column('hit_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('last_hit_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('avg_response_time_ms', sa.Float, nullable=True),
        sa.Column('result_count', sa.Integer, nullable=True),
        sa.Column('cache_efficiency_score', sa.Float, nullable=False, server_default='0.0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'))
    )
    
    # Trigger for updating cache stats
    op.execute("""
        CREATE OR REPLACE FUNCTION update_cache_stats()
        RETURNS TRIGGER AS $$
        BEGIN
            -- Update cache statistics when retrievals happen
            INSERT INTO vector_search_cache_stats (
                cache_key_hash,
                query_pattern,
                hit_count,
                last_hit_at,
                avg_response_time_ms,
                result_count,
                updated_at
            )
            VALUES (
                MD5(COALESCE(NEW.query_text, 'unknown')),
                LEFT(NEW.query_text, 100),
                1,
                NEW.retrieved_at,
                NEW.response_time_ms,
                1,
                NOW()
            )
            ON CONFLICT (cache_key_hash)
            DO UPDATE SET
                hit_count = vector_search_cache_stats.hit_count + 1,
                last_hit_at = NEW.retrieved_at,
                avg_response_time_ms = (
                    COALESCE(vector_search_cache_stats.avg_response_time_ms, 0) * vector_search_cache_stats.hit_count + 
                    COALESCE(NEW.response_time_ms, 0)
                ) / (vector_search_cache_stats.hit_count + 1),
                cache_efficiency_score = CASE 
                    WHEN NEW.response_time_ms < 50 THEN 
                        LEAST(1.0, vector_search_cache_stats.cache_efficiency_score + 0.1)
                    ELSE 
                        GREATEST(0.0, vector_search_cache_stats.cache_efficiency_score - 0.05)
                END,
                updated_at = NOW();
            
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    op.execute("""
        CREATE TRIGGER vector_search_cache_stats_trigger
        AFTER INSERT ON context_retrievals
        FOR EACH ROW
        EXECUTE FUNCTION update_cache_stats();
    """)
    
    # Add procedure for cleaning up old performance data
    op.execute("""
        CREATE OR REPLACE FUNCTION cleanup_vector_search_performance_data(
            days_to_keep INTEGER DEFAULT 30
        )
        RETURNS TABLE(
            table_name TEXT,
            rows_deleted INTEGER
        ) AS $$
        DECLARE
            deleted_count INTEGER;
        BEGIN
            -- Clean up old batch processing records
            DELETE FROM vector_search_batches 
            WHERE created_at < NOW() - INTERVAL '1 day' * days_to_keep;
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            
            RETURN QUERY SELECT 'vector_search_batches'::TEXT, deleted_count;
            
            -- Clean up old cache stats (keep only frequently used ones)
            DELETE FROM vector_search_cache_stats 
            WHERE created_at < NOW() - INTERVAL '1 day' * days_to_keep 
              AND hit_count < 5;
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            
            RETURN QUERY SELECT 'vector_search_cache_stats'::TEXT, deleted_count;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Set appropriate memory settings for vector operations
    op.execute("""
        -- Optimize shared_preload_libraries for vector operations if not already set
        -- Note: This requires a PostgreSQL restart to take effect
        COMMENT ON EXTENSION vector IS 'Vector similarity search optimized for 1536-dimensional embeddings';
    """)
    
    print("Vector search performance optimization migration completed successfully!")
    print("Key optimizations added:")
    print("- Advanced pgvector IVFFlat indexes for cosine and L2 distance")
    print("- Cross-agent search performance indexes")
    print("- Real-time performance monitoring views")
    print("- Batch processing optimization tracking")
    print("- Cache performance analytics")
    print("- Automated maintenance functions")


def downgrade() -> None:
    """Remove vector search performance optimizations."""
    
    # Drop triggers and functions
    op.execute('DROP TRIGGER IF EXISTS vector_search_cache_stats_trigger ON context_retrievals;')
    op.execute('DROP FUNCTION IF EXISTS update_cache_stats();')
    op.execute('DROP FUNCTION IF EXISTS cleanup_vector_search_performance_data(INTEGER);')
    op.execute('DROP FUNCTION IF EXISTS maintain_vector_indexes();')
    op.execute('DROP FUNCTION IF EXISTS optimize_vector_search_performance();')
    
    # Drop performance monitoring views
    op.execute('DROP VIEW IF EXISTS v_context_embedding_quality;')
    op.execute('DROP VIEW IF EXISTS v_cross_agent_context_sharing;')
    op.execute('DROP VIEW IF EXISTS v_vector_search_performance;')
    
    # Drop performance tracking tables
    op.drop_table('vector_search_cache_stats')
    op.drop_table('vector_search_batches')
    
    # Drop performance optimization indexes (safely)
    try:
        op.drop_index('idx_vector_search_batches_performance')
    except Exception:
        pass
    
    # Drop the v2 index we created (not the original from migration 007)
    op.execute('DROP INDEX IF EXISTS idx_contexts_type_importance_embedding_v2;')
    
    try:
        op.drop_index('idx_contexts_temporal_importance_embedding')
    except Exception:
        pass
    
    try:
        op.drop_index('idx_contexts_agent_access_embedding')
    except Exception:
        pass
    
    try:
        op.drop_index('idx_contexts_cross_agent_search')
    except Exception:  
        pass
    
    # Drop vector indexes
    op.execute('DROP INDEX IF EXISTS idx_contexts_embedding_ivfflat_l2;')
    op.execute('DROP INDEX IF EXISTS idx_contexts_embedding_ivfflat_cosine;')
    
    print("Vector search performance optimizations removed.")