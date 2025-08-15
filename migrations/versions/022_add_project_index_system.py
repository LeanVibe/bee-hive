"""Add Project Index System for code intelligence and context optimization

Revision ID: 022_add_project_index_system
Revises: d36c23fd2bf9_merge_enum_fixes_and_pgvector_extension
Create Date: 2025-01-15 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '022_add_project_index_system'
down_revision = 'd36c23fd2bf9_merge_enum_fixes_and_pgvector_extension'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema to add Project Index System tables."""
    
    # Create enum types
    op.execute("""
        CREATE TYPE project_status AS ENUM (
            'active', 'inactive', 'archived', 'analyzing', 'failed'
        )
    """)
    
    op.execute("""
        CREATE TYPE file_type AS ENUM (
            'source', 'config', 'documentation', 'test', 'build', 'other'
        )
    """)
    
    op.execute("""
        CREATE TYPE dependency_type AS ENUM (
            'import', 'require', 'include', 'extends', 'implements', 'calls', 'references'
        )
    """)
    
    op.execute("""
        CREATE TYPE snapshot_type AS ENUM (
            'manual', 'scheduled', 'pre_analysis', 'post_analysis', 'git_commit'
        )
    """)
    
    op.execute("""
        CREATE TYPE session_type AS ENUM (
            'full_analysis', 'incremental', 'context_optimization', 'dependency_mapping', 'file_scanning'
        )
    """)
    
    op.execute("""
        CREATE TYPE analysis_status AS ENUM (
            'pending', 'running', 'completed', 'failed', 'cancelled', 'paused'
        )
    """)
    
    # Create project_indexes table
    op.create_table(
        'project_indexes',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(255), nullable=False, index=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('root_path', sa.String(1000), nullable=False, index=True),
        sa.Column('git_repository_url', sa.String(500), nullable=True, index=True),
        sa.Column('git_branch', sa.String(100), nullable=True, index=True),
        sa.Column('git_commit_hash', sa.String(40), nullable=True, index=True),
        sa.Column('status', sa.Enum('active', 'inactive', 'archived', 'analyzing', 'failed', name='project_status'), 
                 nullable=False, server_default='inactive', index=True),
        sa.Column('configuration', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('analysis_settings', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('file_patterns', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('ignore_patterns', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('metadata', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('file_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('dependency_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('last_indexed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_analysis_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), 
                 onupdate=sa.text('now()'), nullable=False),
    )
    
    # Create file_entries table
    op.create_table(
        'file_entries',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('file_path', sa.String(1000), nullable=False, index=True),
        sa.Column('relative_path', sa.String(1000), nullable=False, index=True),
        sa.Column('file_name', sa.String(255), nullable=False, index=True),
        sa.Column('file_extension', sa.String(50), nullable=True, index=True),
        sa.Column('file_type', sa.Enum('source', 'config', 'documentation', 'test', 'build', 'other', name='file_type'), 
                 nullable=False, index=True),
        sa.Column('language', sa.String(50), nullable=True, index=True),
        sa.Column('encoding', sa.String(20), nullable=True, server_default='utf-8'),
        sa.Column('file_size', sa.BigInteger, nullable=True),
        sa.Column('line_count', sa.Integer, nullable=True),
        sa.Column('sha256_hash', sa.String(64), nullable=True, index=True),
        sa.Column('content_preview', sa.Text, nullable=True),
        sa.Column('analysis_data', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('metadata', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('tags', postgresql.ARRAY(sa.String), nullable=True, server_default='{}'),
        sa.Column('is_binary', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('is_generated', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('last_modified', sa.DateTime(timezone=True), nullable=True),
        sa.Column('indexed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), 
                 onupdate=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['project_id'], ['project_indexes.id'], ondelete='CASCADE'),
    )
    
    # Create dependency_relationships table
    op.create_table(
        'dependency_relationships',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('source_file_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('target_file_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('target_path', sa.String(1000), nullable=True, index=True),
        sa.Column('target_name', sa.String(255), nullable=False, index=True),
        sa.Column('dependency_type', sa.Enum('import', 'require', 'include', 'extends', 'implements', 'calls', 'references', name='dependency_type'), 
                 nullable=False, index=True),
        sa.Column('line_number', sa.Integer, nullable=True),
        sa.Column('column_number', sa.Integer, nullable=True),
        sa.Column('source_text', sa.Text, nullable=True),
        sa.Column('is_external', sa.Boolean, nullable=False, server_default='false', index=True),
        sa.Column('is_dynamic', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('confidence_score', sa.Float, nullable=True, server_default='1.0'),
        sa.Column('metadata', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), 
                 onupdate=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['project_id'], ['project_indexes.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['source_file_id'], ['file_entries.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['target_file_id'], ['file_entries.id'], ondelete='SET NULL'),
    )
    
    # Create index_snapshots table
    op.create_table(
        'index_snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('snapshot_name', sa.String(255), nullable=False, index=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('snapshot_type', sa.Enum('manual', 'scheduled', 'pre_analysis', 'post_analysis', 'git_commit', name='snapshot_type'), 
                 nullable=False, index=True),
        sa.Column('git_commit_hash', sa.String(40), nullable=True, index=True),
        sa.Column('git_branch', sa.String(100), nullable=True),
        sa.Column('file_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('dependency_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('changes_since_last', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('analysis_metrics', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('metadata', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('data_checksum', sa.String(64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['project_id'], ['project_indexes.id'], ondelete='CASCADE'),
    )
    
    # Create analysis_sessions table
    op.create_table(
        'analysis_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('session_name', sa.String(255), nullable=False, index=True),
        sa.Column('session_type', sa.Enum('full_analysis', 'incremental', 'context_optimization', 'dependency_mapping', 'file_scanning', name='session_type'), 
                 nullable=False, index=True),
        sa.Column('status', sa.Enum('pending', 'running', 'completed', 'failed', 'cancelled', 'paused', name='analysis_status'), 
                 nullable=False, server_default='pending', index=True),
        sa.Column('progress_percentage', sa.Float, nullable=False, server_default='0.0'),
        sa.Column('current_phase', sa.String(100), nullable=True),
        sa.Column('files_processed', sa.Integer, nullable=False, server_default='0'),
        sa.Column('files_total', sa.Integer, nullable=False, server_default='0'),
        sa.Column('dependencies_found', sa.Integer, nullable=False, server_default='0'),
        sa.Column('errors_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('warnings_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('session_data', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('error_log', postgresql.JSONB, nullable=True, server_default='[]'),
        sa.Column('performance_metrics', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('configuration', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('result_data', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('estimated_completion', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), 
                 onupdate=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['project_id'], ['project_indexes.id'], ondelete='CASCADE'),
    )
    
    # Create performance indexes
    
    # Project indexes for common queries
    op.create_index('idx_project_indexes_status_name', 'project_indexes', ['status', 'name'])
    op.create_index('idx_project_indexes_git_repo', 'project_indexes', ['git_repository_url', 'git_branch'])
    op.create_index('idx_project_indexes_last_indexed', 'project_indexes', ['last_indexed_at'])
    
    # File entries indexes for common queries
    op.create_index('idx_file_entries_project_path', 'file_entries', ['project_id', 'relative_path'])
    op.create_index('idx_file_entries_project_type', 'file_entries', ['project_id', 'file_type'])
    op.create_index('idx_file_entries_project_language', 'file_entries', ['project_id', 'language'])
    op.create_index('idx_file_entries_hash', 'file_entries', ['sha256_hash'])
    op.create_index('idx_file_entries_modified', 'file_entries', ['last_modified'])
    op.create_index('idx_file_entries_extension', 'file_entries', ['file_extension'])
    
    # Dependency relationships indexes for graph queries
    op.create_index('idx_dependency_relationships_source', 'dependency_relationships', ['source_file_id', 'dependency_type'])
    op.create_index('idx_dependency_relationships_target', 'dependency_relationships', ['target_file_id', 'dependency_type'])
    op.create_index('idx_dependency_relationships_project_type', 'dependency_relationships', ['project_id', 'dependency_type'])
    op.create_index('idx_dependency_relationships_external', 'dependency_relationships', ['project_id', 'is_external'])
    op.create_index('idx_dependency_relationships_target_name', 'dependency_relationships', ['target_name'])
    
    # Index snapshots indexes for version comparison
    op.create_index('idx_index_snapshots_project_type', 'index_snapshots', ['project_id', 'snapshot_type'])
    op.create_index('idx_index_snapshots_project_created', 'index_snapshots', ['project_id', 'created_at'])
    op.create_index('idx_index_snapshots_git_commit', 'index_snapshots', ['git_commit_hash'])
    
    # Analysis sessions indexes for monitoring
    op.create_index('idx_analysis_sessions_project_status', 'analysis_sessions', ['project_id', 'status'])
    op.create_index('idx_analysis_sessions_type_status', 'analysis_sessions', ['session_type', 'status'])
    op.create_index('idx_analysis_sessions_started', 'analysis_sessions', ['started_at'])
    op.create_index('idx_analysis_sessions_progress', 'analysis_sessions', ['progress_percentage'])
    
    # Composite indexes for complex queries
    op.create_index('idx_file_entries_composite_search', 'file_entries', 
                   ['project_id', 'file_type', 'language', 'is_binary'])
    op.create_index('idx_dependency_composite_graph', 'dependency_relationships', 
                   ['project_id', 'source_file_id', 'target_file_id', 'dependency_type'])


def downgrade() -> None:
    """Downgrade database schema to remove Project Index System tables."""
    
    # Drop indexes
    op.drop_index('idx_dependency_composite_graph')
    op.drop_index('idx_file_entries_composite_search')
    op.drop_index('idx_analysis_sessions_progress')
    op.drop_index('idx_analysis_sessions_started')
    op.drop_index('idx_analysis_sessions_type_status')
    op.drop_index('idx_analysis_sessions_project_status')
    op.drop_index('idx_index_snapshots_git_commit')
    op.drop_index('idx_index_snapshots_project_created')
    op.drop_index('idx_index_snapshots_project_type')
    op.drop_index('idx_dependency_relationships_target_name')
    op.drop_index('idx_dependency_relationships_external')
    op.drop_index('idx_dependency_relationships_project_type')
    op.drop_index('idx_dependency_relationships_target')
    op.drop_index('idx_dependency_relationships_source')
    op.drop_index('idx_file_entries_extension')
    op.drop_index('idx_file_entries_modified')
    op.drop_index('idx_file_entries_hash')
    op.drop_index('idx_file_entries_project_language')
    op.drop_index('idx_file_entries_project_type')
    op.drop_index('idx_file_entries_project_path')
    op.drop_index('idx_project_indexes_last_indexed')
    op.drop_index('idx_project_indexes_git_repo')
    op.drop_index('idx_project_indexes_status_name')
    
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('analysis_sessions')
    op.drop_table('index_snapshots')
    op.drop_table('dependency_relationships')
    op.drop_table('file_entries')
    op.drop_table('project_indexes')
    
    # Drop enum types
    op.execute('DROP TYPE IF EXISTS analysis_status')
    op.execute('DROP TYPE IF EXISTS session_type')
    op.execute('DROP TYPE IF EXISTS snapshot_type')
    op.execute('DROP TYPE IF EXISTS dependency_type')
    op.execute('DROP TYPE IF EXISTS file_type')
    op.execute('DROP TYPE IF EXISTS project_status')