"""Add GitHub Integration & Version Control System

Revision ID: 009
Revises: 008
Create Date: 2025-01-27 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '009'
down_revision = '008'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema with GitHub Integration tables."""
    
    # Create custom enums for GitHub Integration
    op.execute("""
        CREATE TYPE work_tree_status AS ENUM (
            'active', 'cleaning', 'archived', 'error'
        );
    """)
    
    op.execute("""
        CREATE TYPE pr_status AS ENUM (
            'open', 'closed', 'merged', 'draft'
        );
    """)
    
    op.execute("""
        CREATE TYPE issue_state AS ENUM (
            'open', 'closed'
        );
    """)
    
    # GitHub repository configurations
    op.create_table(
        'github_repositories',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('repository_full_name', sa.String(255), nullable=False, unique=True, index=True),  # "owner/repo"
        sa.Column('repository_url', sa.String(500), nullable=False),
        sa.Column('clone_url', sa.String(500), nullable=True),  # SSH/HTTPS clone URL
        sa.Column('default_branch', sa.String(100), nullable=False, server_default='main'),
        sa.Column('agent_permissions', postgresql.JSON, nullable=True, server_default='{}'),  # {"read": true, "write": true, "issues": true}
        sa.Column('webhook_secret', sa.String(255), nullable=True),
        sa.Column('webhook_url', sa.String(500), nullable=True),
        sa.Column('access_token_hash', sa.String(255), nullable=True),  # Encrypted token
        sa.Column('repository_config', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('last_sync', sa.DateTime(timezone=True), nullable=True),
        sa.Column('sync_status', sa.String(50), nullable=False, server_default='pending'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    
    # Agent work trees for isolated development
    op.create_table(
        'agent_work_trees',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('repository_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('work_tree_path', sa.String(500), nullable=False, unique=True),
        sa.Column('branch_name', sa.String(255), nullable=False, index=True),
        sa.Column('base_branch', sa.String(255), nullable=True),  # Branch this work tree is based on
        sa.Column('upstream_branch', sa.String(255), nullable=True),  # Remote tracking branch
        sa.Column('status', sa.Enum('active', 'cleaning', 'archived', 'error', name='work_tree_status'), 
                  nullable=False, server_default='active', index=True),
        sa.Column('isolation_config', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('last_commit_hash', sa.String(40), nullable=True),
        sa.Column('uncommitted_changes', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('last_used', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('cleaned_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['repository_id'], ['github_repositories.id'], ondelete='CASCADE'),
    )
    
    # Pull request management
    op.create_table(
        'pull_requests',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('repository_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('work_tree_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('github_pr_number', sa.Integer, nullable=False, index=True),
        sa.Column('github_pr_id', sa.BigInteger, nullable=True),  # GitHub's internal PR ID
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('source_branch', sa.String(255), nullable=False),
        sa.Column('target_branch', sa.String(255), nullable=False),
        sa.Column('status', sa.Enum('open', 'closed', 'merged', 'draft', name='pr_status'), 
                  nullable=False, server_default='open', index=True),
        sa.Column('mergeable', sa.Boolean, nullable=True),
        sa.Column('conflicts', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('review_status', sa.String(50), nullable=True),  # 'pending', 'approved', 'changes_requested'
        sa.Column('ci_status', sa.String(50), nullable=True),  # 'pending', 'success', 'failure'
        sa.Column('labels', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('reviewers', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('pr_metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('merged_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('closed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['repository_id'], ['github_repositories.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['work_tree_id'], ['agent_work_trees.id'], ondelete='SET NULL'),
    )
    
    # Issue tracking and assignment
    op.create_table(
        'github_issues',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('repository_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('github_issue_number', sa.Integer, nullable=False, index=True),
        sa.Column('github_issue_id', sa.BigInteger, nullable=True),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('labels', postgresql.JSON, nullable=True, server_default='[]'),  # ["bug", "enhancement", "priority:high"]
        sa.Column('assignee_agent_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('assignee_github_username', sa.String(255), nullable=True),
        sa.Column('state', sa.Enum('open', 'closed', name='issue_state'), 
                  nullable=False, server_default='open', index=True),
        sa.Column('priority', sa.String(20), nullable=True, index=True),
        sa.Column('issue_type', sa.String(50), nullable=True, index=True),  # 'bug', 'feature', 'enhancement'
        sa.Column('estimated_effort', sa.Integer, nullable=True),  # Story points or hours
        sa.Column('actual_effort', sa.Integer, nullable=True),
        sa.Column('milestone', sa.String(255), nullable=True),
        sa.Column('issue_metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('progress_updates', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('assigned_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('closed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['repository_id'], ['github_repositories.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['assignee_agent_id'], ['agents.id'], ondelete='SET NULL'),
    )
    
    # Code review automation
    op.create_table(
        'code_reviews',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('pull_request_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('reviewer_agent_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('reviewer_type', sa.String(50), nullable=False),  # 'agent', 'human', 'automated'
        sa.Column('review_type', sa.String(50), nullable=False),  # 'security', 'performance', 'style', 'comprehensive'
        sa.Column('review_status', sa.String(50), nullable=False, server_default='pending'),  # 'pending', 'in_progress', 'completed', 'failed'
        sa.Column('findings', postgresql.JSON, nullable=True, server_default='[]'),  # Detailed review findings and suggestions
        sa.Column('security_issues', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('performance_issues', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('style_issues', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('suggestions', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('overall_score', sa.Float, nullable=True),  # 0.0 to 1.0
        sa.Column('approved', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('changes_requested', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('review_metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['pull_request_id'], ['pull_requests.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['reviewer_agent_id'], ['agents.id'], ondelete='SET NULL'),
    )
    
    # Git commit tracking
    op.create_table(
        'git_commits',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('repository_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('work_tree_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('commit_hash', sa.String(40), nullable=False, index=True),
        sa.Column('short_hash', sa.String(10), nullable=True),
        sa.Column('branch_name', sa.String(255), nullable=True, index=True),
        sa.Column('commit_message', sa.Text, nullable=True),
        sa.Column('commit_message_body', sa.Text, nullable=True),
        sa.Column('author_name', sa.String(255), nullable=True),
        sa.Column('author_email', sa.String(255), nullable=True),
        sa.Column('committer_name', sa.String(255), nullable=True),
        sa.Column('committer_email', sa.String(255), nullable=True),
        sa.Column('files_changed', sa.Integer, nullable=True),
        sa.Column('lines_added', sa.Integer, nullable=True),
        sa.Column('lines_deleted', sa.Integer, nullable=True),
        sa.Column('parent_hashes', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('is_merge', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('commit_metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('committed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['repository_id'], ['github_repositories.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['work_tree_id'], ['agent_work_trees.id'], ondelete='SET NULL'),
    )
    
    # Branch management and conflict resolution tracking
    op.create_table(
        'branch_operations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('repository_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('work_tree_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('operation_type', sa.String(50), nullable=False, index=True),  # 'merge', 'rebase', 'cherry_pick', 'sync'
        sa.Column('source_branch', sa.String(255), nullable=True),
        sa.Column('target_branch', sa.String(255), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, server_default='pending'),  # 'pending', 'in_progress', 'completed', 'failed'
        sa.Column('conflicts_detected', sa.Integer, nullable=False, server_default='0'),
        sa.Column('conflicts_resolved', sa.Integer, nullable=False, server_default='0'),
        sa.Column('conflict_details', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('resolution_strategy', sa.String(100), nullable=True),
        sa.Column('operation_result', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['repository_id'], ['github_repositories.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['work_tree_id'], ['agent_work_trees.id'], ondelete='SET NULL'),
    )
    
    # Create indexes for performance optimization
    op.create_index('idx_work_trees_agent_repo_unique', 'agent_work_trees', ['agent_id', 'repository_id'], unique=True)
    op.create_index('idx_pull_requests_status_created', 'pull_requests', ['status', 'created_at'])
    op.create_index('idx_issues_assignee_state', 'github_issues', ['assignee_agent_id', 'state'])
    op.create_index('idx_issues_priority_created', 'github_issues', ['priority', 'created_at'])
    op.create_index('idx_code_reviews_status_type', 'code_reviews', ['review_status', 'review_type'])
    op.create_index('idx_git_commits_hash_unique', 'git_commits', ['repository_id', 'commit_hash'], unique=True)
    op.create_index('idx_git_commits_agent_date', 'git_commits', ['agent_id', 'committed_at'])
    op.create_index('idx_branch_operations_status_type', 'branch_operations', ['status', 'operation_type'])
    op.create_index('idx_repositories_full_name', 'github_repositories', ['repository_full_name'])


def downgrade() -> None:
    """Downgrade database schema - remove GitHub Integration tables."""
    
    # Drop indexes
    op.drop_index('idx_repositories_full_name')
    op.drop_index('idx_branch_operations_status_type')
    op.drop_index('idx_git_commits_agent_date')
    op.drop_index('idx_git_commits_hash_unique')
    op.drop_index('idx_code_reviews_status_type')
    op.drop_index('idx_issues_priority_created')
    op.drop_index('idx_issues_assignee_state')
    op.drop_index('idx_pull_requests_status_created')
    op.drop_index('idx_work_trees_agent_repo_unique')
    
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('branch_operations')
    op.drop_table('git_commits')
    op.drop_table('code_reviews')
    op.drop_table('github_issues')
    op.drop_table('pull_requests')
    op.drop_table('agent_work_trees')
    op.drop_table('github_repositories')
    
    # Drop custom enums
    op.execute('DROP TYPE IF EXISTS issue_state')
    op.execute('DROP TYPE IF EXISTS pr_status')
    op.execute('DROP TYPE IF EXISTS work_tree_status')