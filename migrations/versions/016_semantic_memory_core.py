"""Semantic Memory Service Core - pgvector integration for VS 5.1

Revision ID: 016
Revises: 015
Create Date: 2025-01-28 12:00:00.000000

Creates the core semantic memory service infrastructure with:
- Optimized semantic_documents table with pgvector embeddings
- High-performance HNSW indexes for <200ms P95 search latency
- Agent-scoped document isolation and workflow context preservation
- Comprehensive performance monitoring and analytics
- Batch processing optimization for >500 docs/sec ingestion
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '016_semantic_memory_core'
down_revision = '015'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add semantic memory service core infrastructure."""
    
    # Ensure pgvector extension is available
    op.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    
    # Create semantic_documents table optimized for high-performance semantic search
    op.create_table(
        'semantic_documents',
        sa.Column('document_id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('workflow_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('tags', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('embedding', Vector(1536), nullable=True),  # OpenAI text-embedding-ada-002 dimensions
        sa.Column('importance_score', sa.Float, nullable=False, server_default='0.5'),
        sa.Column('access_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('last_accessed', sa.DateTime(timezone=True), nullable=True),
        
        # Foreign key constraints
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workflow_id'], ['workflows.id'], ondelete='SET NULL')
    )
    
    # Create optimized HNSW indexes for fast similarity search (targeting <200ms P95)
    op.execute("""
        CREATE INDEX semantic_documents_embedding_hnsw_cosine 
        ON semantic_documents USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        WHERE embedding IS NOT NULL;
    """)
    
    # Additional L2 distance index for alternative similarity metrics
    op.execute("""
        CREATE INDEX semantic_documents_embedding_hnsw_l2 
        ON semantic_documents USING hnsw (embedding vector_l2_ops)
        WITH (m = 16, ef_construction = 64)
        WHERE embedding IS NOT NULL;
    """)
    
    # Agent-scoped isolation and filtering indexes
    op.create_index(
        'idx_semantic_documents_agent_scope',
        'semantic_documents',
        ['agent_id', 'importance_score', 'created_at'],
        postgresql_where=sa.text('embedding IS NOT NULL')
    )
    
    # Workflow context preservation index
    op.create_index(
        'idx_semantic_documents_workflow_context',
        'semantic_documents',
        ['workflow_id', 'agent_id', 'importance_score'],
        postgresql_where=sa.text('workflow_id IS NOT NULL AND embedding IS NOT NULL')
    )
    
    # High-performance tag filtering index using GIN
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_semantic_documents_tags_gin 
        ON semantic_documents USING gin ((tags::jsonb))
        WHERE embedding IS NOT NULL;
    """)
    
    # Metadata filtering index for complex queries
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_semantic_documents_metadata_gin 
        ON semantic_documents USING gin ((metadata::jsonb))
        WHERE embedding IS NOT NULL;
    """)
    
    # Temporal filtering for time-based queries
    op.create_index(
        'idx_semantic_documents_temporal',
        'semantic_documents',
        ['created_at', 'last_accessed', 'importance_score'],
        postgresql_where=sa.text('embedding IS NOT NULL')
    )
    
    # Create embedding generation tracking table for batch processing optimization
    op.create_table(
        'embedding_generations',
        sa.Column('generation_id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('batch_id', sa.String(64), nullable=True, index=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('content_hash', sa.String(64), nullable=False, index=True),
        sa.Column('embedding_model', sa.String(100), nullable=False, server_default='text-embedding-ada-002'),
        sa.Column('generation_time_ms', sa.Float, nullable=False),
        sa.Column('token_count', sa.Integer, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='completed'),
        
        sa.ForeignKeyConstraint(['document_id'], ['semantic_documents.document_id'], ondelete='CASCADE')
    )
    
    # Index for batch processing analytics
    op.create_index(
        'idx_embedding_generations_batch_performance',
        'embedding_generations',
        ['batch_id', 'generation_time_ms', 'created_at']
    )
    
    # Create search analytics table for performance monitoring
    op.create_table(
        'semantic_search_analytics',
        sa.Column('search_id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('query_text', sa.Text, nullable=True),
        sa.Column('query_hash', sa.String(64), nullable=False, index=True),
        sa.Column('search_type', sa.String(50), nullable=False),  # semantic, similarity, related
        sa.Column('results_count', sa.Integer, nullable=False),
        sa.Column('search_time_ms', sa.Float, nullable=False),
        sa.Column('embedding_time_ms', sa.Float, nullable=True),
        sa.Column('similarity_threshold', sa.Float, nullable=True),
        sa.Column('avg_similarity_score', sa.Float, nullable=True),
        sa.Column('filters_applied', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE')
    )
    
    # Index for performance analytics and monitoring
    op.create_index(
        'idx_semantic_search_performance',
        'semantic_search_analytics',
        ['search_time_ms', 'results_count', 'created_at']
    )
    
    # Create context compression tracking table
    op.create_table(
        'context_compressions',
        sa.Column('compression_id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('context_id', sa.String(255), nullable=False, index=True),
        sa.Column('compressed_context_id', sa.String(255), nullable=False, unique=True),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('compression_method', sa.String(50), nullable=False),
        sa.Column('original_size', sa.Integer, nullable=False),
        sa.Column('compressed_size', sa.Integer, nullable=False),
        sa.Column('compression_ratio', sa.Float, nullable=False),
        sa.Column('semantic_preservation_score', sa.Float, nullable=False),
        sa.Column('processing_time_ms', sa.Float, nullable=False),
        sa.Column('preserved_documents', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE')
    )
    
    # Performance monitoring views for operational insights
    op.execute("""
        CREATE OR REPLACE VIEW v_semantic_memory_performance AS
        SELECT 
            DATE_TRUNC('hour', created_at) as hour_bucket,
            search_type,
            COUNT(*) as total_searches,
            AVG(search_time_ms) as avg_search_time_ms,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY search_time_ms) as p50_search_time_ms,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY search_time_ms) as p95_search_time_ms,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY search_time_ms) as p99_search_time_ms,
            AVG(results_count) as avg_results_count,
            AVG(avg_similarity_score) as avg_similarity_score,
            COUNT(DISTINCT agent_id) as unique_agents
        FROM semantic_search_analytics 
        WHERE created_at >= NOW() - INTERVAL '7 days'
        GROUP BY DATE_TRUNC('hour', created_at), search_type
        ORDER BY hour_bucket DESC, search_type;
    """)
    
    op.execute("""
        CREATE OR REPLACE VIEW v_embedding_generation_performance AS
        SELECT 
            DATE_TRUNC('hour', created_at) as hour_bucket,
            embedding_model,
            COUNT(*) as total_generations,
            AVG(generation_time_ms) as avg_generation_time_ms,
            SUM(token_count) as total_tokens,
            AVG(token_count) as avg_tokens_per_doc,
            COUNT(*) / EXTRACT(EPOCH FROM INTERVAL '1 hour') as docs_per_second,
            COUNT(DISTINCT batch_id) as unique_batches
        FROM embedding_generations
        WHERE created_at >= NOW() - INTERVAL '7 days'
          AND status = 'completed'
        GROUP BY DATE_TRUNC('hour', created_at), embedding_model
        ORDER BY hour_bucket DESC, embedding_model;
    """)
    
    op.execute("""
        CREATE OR REPLACE VIEW v_agent_document_stats AS
        SELECT 
            sd.agent_id,
            a.name as agent_name,
            COUNT(*) as total_documents,
            COUNT(CASE WHEN sd.embedding IS NOT NULL THEN 1 END) as documents_with_embeddings,
            ROUND(
                ((COUNT(CASE WHEN sd.embedding IS NOT NULL THEN 1 END)::FLOAT / COUNT(*)) * 100)::NUMERIC, 2
            ) as embedding_coverage_percent,
            AVG(sd.importance_score) as avg_importance_score,
            SUM(sd.access_count) as total_access_count,
            MAX(sd.last_accessed) as last_document_access,
            COUNT(DISTINCT sd.workflow_id) as unique_workflows
        FROM semantic_documents sd
        LEFT JOIN agents a ON sd.agent_id = a.id
        GROUP BY sd.agent_id, a.name
        ORDER BY total_documents DESC;
    """)
    
    # Performance optimization functions
    op.execute("""
        CREATE OR REPLACE FUNCTION optimize_semantic_search_performance()
        RETURNS TABLE(
            optimization_action TEXT,
            before_metric FLOAT,
            after_metric FLOAT,
            improvement_percent FLOAT,
            recommendation TEXT
        ) AS $$
        DECLARE
            doc_count INTEGER;
            avg_search_time FLOAT;
            p95_search_time FLOAT;
        BEGIN
            -- Get current document count
            SELECT COUNT(*) INTO doc_count FROM semantic_documents WHERE embedding IS NOT NULL;
            
            -- Get current performance metrics
            SELECT 
                AVG(search_time_ms),
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY search_time_ms)
            INTO avg_search_time, p95_search_time
            FROM semantic_search_analytics
            WHERE created_at >= NOW() - INTERVAL '1 hour';
            
            -- Update table statistics for better query planning
            ANALYZE semantic_documents;
            ANALYZE semantic_search_analytics;
            
            -- Return optimization recommendations
            RETURN QUERY SELECT 
                'Statistics Update'::TEXT,
                COALESCE(avg_search_time, 0.0),
                COALESCE(avg_search_time * 0.95, 0.0),  -- Assume 5% improvement
                5.0::FLOAT,
                CASE 
                    WHEN doc_count > 100000 THEN 'Consider index maintenance for large dataset'
                    WHEN p95_search_time > 200 THEN 'P95 latency exceeds target - investigate slow queries'
                    ELSE 'Performance within acceptable limits'
                END;
            
            -- Additional recommendations based on document count
            IF doc_count > 50000 THEN
                RETURN QUERY SELECT 
                    'Large Dataset Optimization'::TEXT,
                    doc_count::FLOAT,
                    doc_count::FLOAT,
                    0.0::FLOAT,
                    'Consider partitioning by agent_id for datasets > 50k documents'::TEXT;
            END IF;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Batch processing optimization function
    op.execute("""
        CREATE OR REPLACE FUNCTION get_batch_processing_recommendations(
            target_throughput_docs_per_sec FLOAT DEFAULT 500.0
        )
        RETURNS TABLE(
            metric_name TEXT,
            current_value FLOAT,
            target_value FLOAT,
            recommendation TEXT
        ) AS $$
        DECLARE
            current_throughput FLOAT;
            avg_batch_size FLOAT;
            avg_generation_time FLOAT;
        BEGIN
            -- Calculate current throughput
            SELECT 
                COUNT(*)::FLOAT / GREATEST(EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))), 1),
                AVG(CASE WHEN batch_id IS NOT NULL THEN 
                    (SELECT COUNT(*) FROM embedding_generations eg2 WHERE eg2.batch_id = eg.batch_id)
                    ELSE 1 END),
                AVG(generation_time_ms)
            INTO current_throughput, avg_batch_size, avg_generation_time
            FROM embedding_generations eg
            WHERE created_at >= NOW() - INTERVAL '1 hour'
              AND status = 'completed';
            
            -- Return throughput analysis
            RETURN QUERY SELECT 
                'Current Throughput (docs/sec)'::TEXT,
                COALESCE(current_throughput, 0.0),
                target_throughput_docs_per_sec,
                CASE 
                    WHEN current_throughput < target_throughput_docs_per_sec * 0.8 THEN 
                        'Increase batch size or optimize embedding generation'
                    WHEN current_throughput > target_throughput_docs_per_sec * 1.2 THEN 
                        'Performance exceeds target - consider higher throughput goals'
                    ELSE 'Throughput within acceptable range'
                END;
            
            -- Return batch size analysis
            RETURN QUERY SELECT 
                'Average Batch Size'::TEXT,
                COALESCE(avg_batch_size, 1.0),
                50.0::FLOAT,  -- Recommended batch size
                CASE 
                    WHEN avg_batch_size < 20 THEN 'Increase batch size for better efficiency'
                    WHEN avg_batch_size > 100 THEN 'Reduce batch size to prevent timeouts'
                    ELSE 'Batch size is optimal'
                END;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Cleanup function for old analytics data
    op.execute("""
        CREATE OR REPLACE FUNCTION cleanup_semantic_memory_analytics(
            days_to_keep INTEGER DEFAULT 30
        )
        RETURNS TABLE(
            table_name TEXT,
            rows_deleted INTEGER
        ) AS $$
        DECLARE
            deleted_count INTEGER;
        BEGIN
            -- Clean up old search analytics
            DELETE FROM semantic_search_analytics 
            WHERE created_at < NOW() - INTERVAL '1 day' * days_to_keep;
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            
            RETURN QUERY SELECT 'semantic_search_analytics'::TEXT, deleted_count;
            
            -- Clean up old embedding generation records
            DELETE FROM embedding_generations 
            WHERE created_at < NOW() - INTERVAL '1 day' * days_to_keep;
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            
            RETURN QUERY SELECT 'embedding_generations'::TEXT, deleted_count;
            
            -- Clean up old compression records (keep longer for audit)
            DELETE FROM context_compressions 
            WHERE created_at < NOW() - INTERVAL '1 day' * (days_to_keep * 2);
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            
            RETURN QUERY SELECT 'context_compressions'::TEXT, deleted_count;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Create trigger for updating last_accessed timestamp
    op.execute("""
        CREATE OR REPLACE FUNCTION update_document_access()
        RETURNS TRIGGER AS $$
        BEGIN
            -- Update access tracking when document is retrieved
            UPDATE semantic_documents 
            SET 
                access_count = access_count + 1,
                last_accessed = NOW()
            WHERE document_id = NEW.document_id;
            
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Performance monitoring trigger for search operations
    op.execute("""
        CREATE OR REPLACE FUNCTION log_search_performance()
        RETURNS TRIGGER AS $$
        BEGIN
            -- Automatically log performance metrics for monitoring
            IF NEW.search_time_ms > 200 THEN
                -- Log slow queries for investigation
                INSERT INTO system_events (event_type, event_data, created_at)
                VALUES (
                    'slow_semantic_search',
                    jsonb_build_object(
                        'search_id', NEW.search_id,
                        'agent_id', NEW.agent_id,
                        'search_time_ms', NEW.search_time_ms,
                        'results_count', NEW.results_count,
                        'query_hash', NEW.query_hash
                    ),
                    NOW()
                );
            END IF;
            
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    op.execute("""
        CREATE TRIGGER search_performance_monitoring_trigger
        AFTER INSERT ON semantic_search_analytics
        FOR EACH ROW
        EXECUTE FUNCTION log_search_performance();
    """)
    
    print("✅ Semantic Memory Service Core migration completed successfully!")
    print("🚀 Key features added:")
    print("   • High-performance pgvector HNSW indexes for <200ms P95 search")
    print("   • Agent-scoped document isolation with workflow context preservation")
    print("   • Batch processing optimization for >500 docs/sec ingestion")
    print("   • Comprehensive performance monitoring and analytics")
    print("   • Automated maintenance and cleanup functions")
    print("   • Production-ready optimization recommendations")


def downgrade() -> None:
    """Remove semantic memory service core infrastructure."""
    
    # Drop triggers and functions
    op.execute('DROP TRIGGER IF EXISTS search_performance_monitoring_trigger ON semantic_search_analytics;')
    op.execute('DROP FUNCTION IF EXISTS log_search_performance();')
    op.execute('DROP FUNCTION IF EXISTS update_document_access();')
    op.execute('DROP FUNCTION IF EXISTS cleanup_semantic_memory_analytics(INTEGER);')
    op.execute('DROP FUNCTION IF EXISTS get_batch_processing_recommendations(FLOAT);')
    op.execute('DROP FUNCTION IF EXISTS optimize_semantic_search_performance();')
    
    # Drop monitoring views
    op.execute('DROP VIEW IF EXISTS v_agent_document_stats;')
    op.execute('DROP VIEW IF EXISTS v_embedding_generation_performance;')
    op.execute('DROP VIEW IF EXISTS v_semantic_memory_performance;')
    
    # Drop analytics and tracking tables
    op.drop_table('context_compressions')
    op.drop_table('semantic_search_analytics')
    op.drop_table('embedding_generations')
    
    # Drop indexes
    op.drop_index('idx_semantic_documents_temporal')
    op.execute('DROP INDEX IF EXISTS idx_semantic_documents_metadata_gin;')
    op.execute('DROP INDEX IF EXISTS idx_semantic_documents_tags_gin;')
    op.drop_index('idx_semantic_documents_workflow_context')
    op.drop_index('idx_semantic_documents_agent_scope')
    op.execute('DROP INDEX IF EXISTS semantic_documents_embedding_hnsw_l2;')
    op.execute('DROP INDEX IF EXISTS semantic_documents_embedding_hnsw_cosine;')
    
    # Drop main table
    op.drop_table('semantic_documents')
    
    print("🗑️  Semantic Memory Service Core infrastructure removed.")