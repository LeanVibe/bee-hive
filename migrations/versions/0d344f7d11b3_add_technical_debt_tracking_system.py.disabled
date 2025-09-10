"""Add Technical Debt Tracking System for comprehensive code quality monitoring

Revision ID: 0d344f7d11b3
Revises: 28fc7f670c45
Create Date: 2025-08-19 11:32:02.555851

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0d344f7d11b3'
down_revision = '28fc7f670c45'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema to add Technical Debt Tracking System."""
    
    # Create enum types for technical debt classification
    op.execute("""
        CREATE TYPE debt_severity AS ENUM (
            'critical', 'high', 'medium', 'low'
        )
    """)
    
    op.execute("""
        CREATE TYPE debt_category AS ENUM (
            'code_duplication', 'complexity', 'code_smells', 'architecture', 
            'maintainability', 'security', 'performance', 'documentation'
        )
    """)
    
    op.execute("""
        CREATE TYPE debt_status AS ENUM (
            'active', 'resolved', 'acknowledged', 'ignored', 'false_positive'
        )
    """)
    
    # Create debt_snapshots table for tracking debt over time
    op.create_table(
        'debt_snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, 
                 server_default=sa.text('gen_random_uuid()')),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('project_indexes.id', ondelete='CASCADE'), 
                 nullable=False, index=True),
        sa.Column('snapshot_date', sa.DateTime(timezone=True), nullable=False, 
                 server_default=sa.text('now()'), index=True),
        sa.Column('total_debt_score', sa.Float, nullable=False, server_default='0.0'),
        sa.Column('category_scores', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('debt_trend', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('file_count_analyzed', sa.Integer, nullable=False, server_default='0'),
        sa.Column('lines_of_code_analyzed', sa.Integer, nullable=False, server_default='0'),
        sa.Column('metadata', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    
    # Create debt_items table for individual debt tracking
    op.create_table(
        'debt_items',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, 
                 server_default=sa.text('gen_random_uuid()')),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('project_indexes.id', ondelete='CASCADE'), 
                 nullable=False, index=True),
        sa.Column('file_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('file_entries.id', ondelete='CASCADE'), 
                 nullable=False, index=True),
        sa.Column('debt_type', sa.String(100), nullable=False, index=True),
        sa.Column('debt_category', sa.Enum('code_duplication', 'complexity', 'code_smells', 
                                          'architecture', 'maintainability', 'security', 
                                          'performance', 'documentation', name='debt_category'), 
                 nullable=False, index=True),
        sa.Column('severity', sa.Enum('critical', 'high', 'medium', 'low', name='debt_severity'), 
                 nullable=False, index=True),
        sa.Column('status', sa.Enum('active', 'resolved', 'acknowledged', 'ignored', 
                                   'false_positive', name='debt_status'), 
                 nullable=False, server_default='active', index=True),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('evidence', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('location', postgresql.JSONB, nullable=True, server_default='{}'), # line numbers, functions, etc.
        sa.Column('remediation_suggestion', sa.Text, nullable=True),
        sa.Column('estimated_effort_hours', sa.Integer, nullable=True),
        sa.Column('debt_score', sa.Float, nullable=False, server_default='0.0'),
        sa.Column('confidence_score', sa.Float, nullable=False, server_default='1.0'),
        sa.Column('first_detected_at', sa.DateTime(timezone=True), nullable=False, 
                 server_default=sa.text('now()'), index=True),
        sa.Column('last_detected_at', sa.DateTime(timezone=True), nullable=False, 
                 server_default=sa.text('now()'), index=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column('resolved_by', sa.String(255), nullable=True),
        sa.Column('resolution_notes', sa.Text, nullable=True),
        sa.Column('metadata', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), 
                 server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
    )
    
    # Create debt_remediation_plans table for tracking improvement roadmaps
    op.create_table(
        'debt_remediation_plans',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, 
                 server_default=sa.text('gen_random_uuid()')),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('project_indexes.id', ondelete='CASCADE'), 
                 nullable=False, index=True),
        sa.Column('plan_name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('target_debt_reduction', sa.Float, nullable=False), # 0-1 scale
        sa.Column('estimated_effort_hours', sa.Integer, nullable=False),
        sa.Column('priority_level', sa.Integer, nullable=False, server_default='1'), # 1-5 scale
        sa.Column('debt_items', postgresql.JSONB, nullable=False, server_default='[]'), # array of debt item IDs
        sa.Column('remediation_steps', postgresql.JSONB, nullable=False, server_default='[]'),
        sa.Column('status', sa.String(50), nullable=False, server_default='draft'),
        sa.Column('assigned_to', sa.String(255), nullable=True),
        sa.Column('due_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completion_percentage', sa.Float, nullable=False, server_default='0.0'),
        sa.Column('metadata', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), 
                 server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
    )
    
    # Create indexes for better query performance
    op.create_index('idx_debt_snapshots_project_date', 'debt_snapshots', 
                    ['project_id', 'snapshot_date'])
    op.create_index('idx_debt_items_project_severity', 'debt_items', 
                    ['project_id', 'severity', 'status'])
    op.create_index('idx_debt_items_category_status', 'debt_items', 
                    ['debt_category', 'status'])
    op.create_index('idx_debt_items_file_severity', 'debt_items', 
                    ['file_id', 'severity'])
    op.create_index('idx_debt_remediation_project_status', 'debt_remediation_plans', 
                    ['project_id', 'status'])


def downgrade() -> None:
    """Downgrade database schema - remove Technical Debt Tracking System."""
    
    # Drop tables in reverse order
    op.drop_table('debt_remediation_plans')
    op.drop_table('debt_items')
    op.drop_table('debt_snapshots')
    
    # Drop enum types
    op.execute("DROP TYPE IF EXISTS debt_status")
    op.execute("DROP TYPE IF EXISTS debt_category")
    op.execute("DROP TYPE IF EXISTS debt_severity")