"""Add advanced GitHub integration features

Revision ID: 021_add_advanced_github_integration
Revises: 020_fix_enum_columns
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSON


# revision identifiers, used by Alembic.
revision = '021_add_advanced_github_integration'
down_revision = '020_fix_enum_columns'
branch_labels = None
depends_on = None


def upgrade():
    """Add advanced GitHub integration tables and enhancements."""
    
    # Add new columns to existing code_reviews table for enhanced functionality
    op.add_column('code_reviews', 
        sa.Column('ai_confidence_score', sa.Float(), nullable=True, 
                 comment='AI confidence score for automated review (0.0-1.0)')
    )
    op.add_column('code_reviews', 
        sa.Column('complexity_score', sa.Float(), nullable=True,
                 comment='Code complexity score')
    )
    op.add_column('code_reviews', 
        sa.Column('review_duration_seconds', sa.Integer(), nullable=True,
                 comment='Time taken to complete review in seconds')
    )
    op.add_column('code_reviews', 
        sa.Column('automated_fixes_applied', JSON(), nullable=True,
                 comment='List of automated fixes applied during review')
    )
    
    # Create test_results table for automated testing integration
    op.create_table('test_results',
        sa.Column('id', UUID(), primary_key=True),
        sa.Column('pull_request_id', UUID(), sa.ForeignKey('pull_requests.id', ondelete='CASCADE'), 
                 nullable=False, index=True),
        sa.Column('test_run_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('ci_provider', sa.String(50), nullable=False, 
                 comment='CI provider: github_actions, jenkins, etc.'),
        sa.Column('workflow_run_id', sa.String(255), nullable=True),
        sa.Column('workflow_url', sa.String(500), nullable=True),
        
        # Test execution details
        sa.Column('status', sa.String(50), nullable=False, default='pending', index=True,
                 comment='Test execution status: pending, running, passed, failed, cancelled'),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        
        # Test metrics
        sa.Column('total_tests', sa.Integer(), nullable=False, default=0),
        sa.Column('passed_tests', sa.Integer(), nullable=False, default=0),
        sa.Column('failed_tests', sa.Integer(), nullable=False, default=0),
        sa.Column('skipped_tests', sa.Integer(), nullable=False, default=0),
        sa.Column('success_rate', sa.Float(), nullable=True, 
                 comment='Test success rate as percentage'),
        
        # Coverage information
        sa.Column('line_coverage', sa.Float(), nullable=True,
                 comment='Line coverage percentage'),
        sa.Column('branch_coverage', sa.Float(), nullable=True,
                 comment='Branch coverage percentage'),
        sa.Column('function_coverage', sa.Float(), nullable=True,
                 comment='Function coverage percentage'),
        
        # Test suites and detailed results
        sa.Column('test_suites', JSON(), nullable=True,
                 comment='Detailed test suite results and metadata'),
        sa.Column('failure_analysis', JSON(), nullable=True,
                 comment='AI-powered failure analysis and categorization'),
        sa.Column('performance_metrics', JSON(), nullable=True,
                 comment='Performance metrics and benchmarks'),
        
        # Retry and monitoring
        sa.Column('retry_count', sa.Integer(), nullable=False, default=0),
        sa.Column('max_retries', sa.Integer(), nullable=False, default=3),
        sa.Column('retry_reasons', JSON(), nullable=True,
                 comment='Reasons for test retries'),
        
        # Metadata and configuration
        sa.Column('test_configuration', JSON(), nullable=True,
                 comment='Test configuration and environment settings'),
        sa.Column('artifacts', JSON(), nullable=True,
                 comment='List of test artifacts and reports'),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), 
                 onupdate=sa.func.now())
    )
    
    # Create conflict_resolutions table for intelligent merge conflict handling
    op.create_table('conflict_resolutions',
        sa.Column('id', UUID(), primary_key=True),
        sa.Column('pull_request_id', UUID(), sa.ForeignKey('pull_requests.id', ondelete='CASCADE'), 
                 nullable=False, index=True),
        sa.Column('branch_operation_id', UUID(), sa.ForeignKey('branch_operations.id', ondelete='SET NULL'), 
                 nullable=True, index=True),
        
        # Conflict identification
        sa.Column('conflict_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('source_branch', sa.String(255), nullable=False),
        sa.Column('target_branch', sa.String(255), nullable=False),
        
        # Conflict details
        sa.Column('conflicted_files', JSON(), nullable=False,
                 comment='List of files with conflicts and conflict details'),
        sa.Column('conflict_types', JSON(), nullable=True,
                 comment='Types of conflicts detected (import, dependency, logic, etc.)'),
        sa.Column('total_conflicts', sa.Integer(), nullable=False, default=0),
        
        # Resolution strategy and results
        sa.Column('resolution_strategy', sa.String(100), nullable=False,
                 comment='Strategy used: automatic, intelligent_merge, manual_required, etc.'),
        sa.Column('auto_resolvable', sa.Boolean(), nullable=False, default=False,
                 comment='Whether conflicts can be automatically resolved'),
        sa.Column('resolution_confidence', sa.Float(), nullable=True,
                 comment='Confidence in automatic resolution (0.0-1.0)'),
        
        # Resolution execution
        sa.Column('resolution_status', sa.String(50), nullable=False, default='pending', index=True,
                 comment='Resolution status: pending, in_progress, completed, failed'),
        sa.Column('resolved_files', JSON(), nullable=True,
                 comment='List of successfully resolved files'),
        sa.Column('failed_resolutions', JSON(), nullable=True,
                 comment='Files that failed automatic resolution'),
        sa.Column('manual_intervention_required', sa.Boolean(), nullable=False, default=False),
        
        # Resolution details
        sa.Column('resolution_summary', sa.Text(), nullable=True,
                 comment='Human-readable summary of conflict resolution'),
        sa.Column('applied_fixes', JSON(), nullable=True,
                 comment='Detailed list of fixes applied to resolve conflicts'),
        sa.Column('merge_commit_sha', sa.String(40), nullable=True,
                 comment='SHA of merge commit if resolution successful'),
        
        # Performance and quality metrics
        sa.Column('resolution_duration_seconds', sa.Integer(), nullable=True),
        sa.Column('lines_merged', sa.Integer(), nullable=True),
        sa.Column('files_modified', sa.Integer(), nullable=True),
        
        # Timestamps
        sa.Column('detected_at', sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), 
                 onupdate=sa.func.now())
    )
    
    # Create workflow_automations table for intelligent workflow orchestration
    op.create_table('workflow_automations',
        sa.Column('id', UUID(), primary_key=True),
        sa.Column('execution_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('workflow_id', sa.String(255), nullable=False, index=True),
        sa.Column('pull_request_id', UUID(), sa.ForeignKey('pull_requests.id', ondelete='CASCADE'), 
                 nullable=False, index=True),
        
        # Workflow execution details
        sa.Column('trigger_type', sa.String(50), nullable=False, index=True,
                 comment='Trigger: pr_created, pr_updated, manual_trigger, etc.'),
        sa.Column('current_stage', sa.String(50), nullable=False, index=True,
                 comment='Current workflow stage'),
        sa.Column('workflow_config', JSON(), nullable=True,
                 comment='Workflow configuration and parameters'),
        
        # Execution status
        sa.Column('status', sa.String(50), nullable=False, default='initiated', index=True,
                 comment='Workflow status: initiated, running, completed, failed, cancelled'),
        sa.Column('success', sa.Boolean(), nullable=False, default=False),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        
        # Stage tracking
        sa.Column('stages_completed', JSON(), nullable=True,
                 comment='List of completed workflow stages'),
        sa.Column('stages_failed', JSON(), nullable=True,
                 comment='List of failed workflow stages'),
        sa.Column('current_stage_started_at', sa.DateTime(timezone=True), nullable=True),
        
        # Quality gates
        sa.Column('quality_gates_results', JSON(), nullable=True,
                 comment='Results from all quality gate evaluations'),
        sa.Column('quality_gates_passed', sa.Boolean(), nullable=True,
                 comment='Whether all required quality gates passed'),
        sa.Column('quality_score', sa.Float(), nullable=True,
                 comment='Overall quality score (0.0-1.0)'),
        
        # Automation results
        sa.Column('code_formatting_applied', sa.Boolean(), nullable=False, default=False),
        sa.Column('documentation_generated', sa.Boolean(), nullable=False, default=False),
        sa.Column('automated_fixes_count', sa.Integer(), nullable=False, default=0),
        sa.Column('merge_conflicts_resolved', sa.Integer(), nullable=False, default=0),
        
        # Integration results
        sa.Column('test_results_id', UUID(), sa.ForeignKey('test_results.id', ondelete='SET NULL'), 
                 nullable=True, index=True),
        sa.Column('code_review_id', UUID(), sa.ForeignKey('code_reviews.id', ondelete='SET NULL'), 
                 nullable=True, index=True),
        sa.Column('conflict_resolution_id', UUID(), sa.ForeignKey('conflict_resolutions.id', ondelete='SET NULL'), 
                 nullable=True, index=True),
        
        # Workflow metadata and logs
        sa.Column('execution_logs', JSON(), nullable=True,
                 comment='Detailed execution logs and step results'),
        sa.Column('error_messages', JSON(), nullable=True,
                 comment='Error messages from failed stages'),
        sa.Column('warnings', JSON(), nullable=True,
                 comment='Warning messages and non-critical issues'),
        sa.Column('notifications_sent', JSON(), nullable=True,
                 comment='Notifications sent during workflow execution'),
        
        # Performance metrics
        sa.Column('stages_duration', JSON(), nullable=True,
                 comment='Duration of each workflow stage in seconds'),
        sa.Column('resource_usage', JSON(), nullable=True,
                 comment='Resource usage metrics during execution'),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), 
                 onupdate=sa.func.now())
    )
    
    # Create dependency_analyses table for automated dependency management
    op.create_table('dependency_analyses',
        sa.Column('id', UUID(), primary_key=True),
        sa.Column('repository_id', UUID(), sa.ForeignKey('github_repositories.id', ondelete='CASCADE'), 
                 nullable=False, index=True),
        sa.Column('pull_request_id', UUID(), sa.ForeignKey('pull_requests.id', ondelete='SET NULL'), 
                 nullable=True, index=True),
        
        # Analysis identification
        sa.Column('analysis_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('analysis_type', sa.String(50), nullable=False, default='comprehensive',
                 comment='Analysis type: comprehensive, security_focused, update_check'),
        
        # Dependency metrics
        sa.Column('total_dependencies', sa.Integer(), nullable=False, default=0),
        sa.Column('outdated_dependencies', sa.Integer(), nullable=False, default=0),
        sa.Column('vulnerable_dependencies', sa.Integer(), nullable=False, default=0),
        sa.Column('breaking_changes_available', sa.Integer(), nullable=False, default=0),
        
        # Security analysis
        sa.Column('security_vulnerabilities', JSON(), nullable=True,
                 comment='Detailed security vulnerability information'),
        sa.Column('critical_vulnerabilities', sa.Integer(), nullable=False, default=0),
        sa.Column('high_severity_vulnerabilities', sa.Integer(), nullable=False, default=0),
        sa.Column('security_risk_score', sa.Float(), nullable=True,
                 comment='Overall security risk score (0.0-10.0)'),
        
        # Dependency files analysis
        sa.Column('dependency_files', JSON(), nullable=True,
                 comment='Analysis of individual dependency files'),
        sa.Column('languages_detected', JSON(), nullable=True,
                 comment='Programming languages detected in repository'),
        
        # Update recommendations
        sa.Column('update_recommendations', JSON(), nullable=True,
                 comment='Prioritized list of update recommendations'),
        sa.Column('update_strategy', JSON(), nullable=True,
                 comment='Recommended phased update strategy'),
        sa.Column('estimated_effort_hours', sa.Integer(), nullable=True,
                 comment='Estimated effort required for updates'),
        
        # Analysis execution
        sa.Column('status', sa.String(50), nullable=False, default='pending', index=True,
                 comment='Analysis status: pending, running, completed, failed'),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        
        # Analysis results
        sa.Column('analysis_summary', sa.Text(), nullable=True,
                 comment='Human-readable analysis summary'),
        sa.Column('recommendations_applied', JSON(), nullable=True,
                 comment='Recommendations that have been applied'),
        sa.Column('success', sa.Boolean(), nullable=False, default=False),
        sa.Column('error_messages', JSON(), nullable=True,
                 comment='Error messages if analysis failed'),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), 
                 onupdate=sa.func.now())
    )
    
    # Create repository_health_assessments table for comprehensive repository analysis
    op.create_table('repository_health_assessments',
        sa.Column('id', UUID(), primary_key=True),
        sa.Column('repository_id', UUID(), sa.ForeignKey('github_repositories.id', ondelete='CASCADE'), 
                 nullable=False, index=True),
        sa.Column('assessment_id', sa.String(255), nullable=False, unique=True, index=True),
        
        # Health metrics
        sa.Column('overall_health_score', sa.Float(), nullable=False, default=0.0,
                 comment='Overall health score (0.0-100.0)'),
        sa.Column('health_status', sa.String(50), nullable=False, default='unknown', index=True,
                 comment='Health status: excellent, good, warning, critical, unknown'),
        sa.Column('health_grade', sa.String(10), nullable=True,
                 comment='Letter grade: A, B, C, D, F'),
        
        # Branch analysis
        sa.Column('total_branches', sa.Integer(), nullable=False, default=0),
        sa.Column('active_branches', sa.Integer(), nullable=False, default=0),
        sa.Column('stale_branches', sa.Integer(), nullable=False, default=0),
        sa.Column('branch_analysis', JSON(), nullable=True,
                 comment='Detailed analysis of all repository branches'),
        
        # Code quality metrics
        sa.Column('code_quality_score', sa.Float(), nullable=True),
        sa.Column('test_coverage_percentage', sa.Float(), nullable=True),
        sa.Column('documentation_coverage', sa.Float(), nullable=True),
        sa.Column('code_complexity_score', sa.Float(), nullable=True),
        
        # Repository metrics
        sa.Column('total_commits', sa.Integer(), nullable=False, default=0),
        sa.Column('total_contributors', sa.Integer(), nullable=False, default=0),
        sa.Column('last_activity_days', sa.Integer(), nullable=True,
                 comment='Days since last commit'),
        sa.Column('commit_frequency_per_week', sa.Float(), nullable=True),
        
        # File metrics
        sa.Column('total_files', sa.Integer(), nullable=False, default=0),
        sa.Column('code_files', sa.Integer(), nullable=False, default=0),
        sa.Column('test_files', sa.Integer(), nullable=False, default=0),
        sa.Column('documentation_files', sa.Integer(), nullable=False, default=0),
        
        # Dependency health from linked analysis
        sa.Column('dependency_analysis_id', UUID(), sa.ForeignKey('dependency_analyses.id', ondelete='SET NULL'), 
                 nullable=True, index=True),
        sa.Column('dependency_health_score', sa.Float(), nullable=True),
        sa.Column('security_risk_level', sa.String(20), nullable=True,
                 comment='Security risk: low, medium, high, critical'),
        
        # Assessment execution
        sa.Column('assessment_type', sa.String(50), nullable=False, default='comprehensive',
                 comment='Assessment type: comprehensive, quick_check, security_focused'),
        sa.Column('status', sa.String(50), nullable=False, default='pending', index=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        
        # Recommendations and alerts
        sa.Column('recommendations', JSON(), nullable=True,
                 comment='Prioritized list of improvement recommendations'),
        sa.Column('alerts', JSON(), nullable=True,
                 comment='Critical alerts requiring immediate attention'),
        sa.Column('improvement_opportunities', JSON(), nullable=True,
                 comment='Identified opportunities for improvement'),
        
        # Historical tracking
        sa.Column('previous_assessment_id', UUID(), sa.ForeignKey('repository_health_assessments.id'), 
                 nullable=True),
        sa.Column('health_trend', sa.String(20), nullable=True,
                 comment='Health trend: improving, declining, stable'),
        sa.Column('score_change', sa.Float(), nullable=True,
                 comment='Change in score from previous assessment'),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), 
                 onupdate=sa.func.now())
    )
    
    # Create indexes for performance optimization
    op.create_index('idx_test_results_pr_status', 'test_results', ['pull_request_id', 'status'])
    op.create_index('idx_test_results_created_success', 'test_results', ['created_at', 'success_rate'])
    op.create_index('idx_conflict_resolutions_pr_strategy', 'conflict_resolutions', 
                    ['pull_request_id', 'resolution_strategy'])
    op.create_index('idx_conflict_resolutions_auto_resolvable', 'conflict_resolutions', 
                    ['auto_resolvable', 'resolution_status'])
    op.create_index('idx_workflow_automations_pr_status', 'workflow_automations', 
                    ['pull_request_id', 'status'])
    op.create_index('idx_workflow_automations_trigger_stage', 'workflow_automations', 
                    ['trigger_type', 'current_stage'])
    op.create_index('idx_dependency_analyses_repo_status', 'dependency_analyses', 
                    ['repository_id', 'status'])
    op.create_index('idx_dependency_analyses_security_risk', 'dependency_analyses', 
                    ['security_risk_score', 'critical_vulnerabilities'])
    op.create_index('idx_repo_health_score_status', 'repository_health_assessments', 
                    ['overall_health_score', 'health_status'])
    op.create_index('idx_repo_health_repo_created', 'repository_health_assessments', 
                    ['repository_id', 'created_at'])
    
    # Add foreign key relationships to existing tables
    op.add_column('pull_requests', 
        sa.Column('latest_test_result_id', UUID(), sa.ForeignKey('test_results.id', ondelete='SET NULL'), 
                 nullable=True, comment='Reference to latest test execution')
    )
    op.add_column('pull_requests', 
        sa.Column('latest_workflow_automation_id', UUID(), 
                 sa.ForeignKey('workflow_automations.id', ondelete='SET NULL'), 
                 nullable=True, comment='Reference to latest workflow execution')
    )
    op.add_column('pull_requests', 
        sa.Column('automated_merge_eligible', sa.Boolean(), nullable=False, default=False,
                 comment='Whether PR is eligible for automated merge')
    )
    op.add_column('pull_requests', 
        sa.Column('quality_gates_passed', sa.Boolean(), nullable=True,
                 comment='Whether all quality gates have passed')
    )
    
    # Add repository-level automation settings
    op.add_column('github_repositories', 
        sa.Column('automation_config', JSON(), nullable=True,
                 comment='Repository-specific automation configuration')
    )
    op.add_column('github_repositories', 
        sa.Column('quality_gates_config', JSON(), nullable=True,
                 comment='Custom quality gates configuration')
    )
    op.add_column('github_repositories', 
        sa.Column('auto_merge_enabled', sa.Boolean(), nullable=False, default=False,
                 comment='Whether automated merge is enabled')
    )
    op.add_column('github_repositories', 
        sa.Column('latest_health_assessment_id', UUID(), 
                 sa.ForeignKey('repository_health_assessments.id', ondelete='SET NULL'), 
                 nullable=True, comment='Reference to latest health assessment')
    )


def downgrade():
    """Remove advanced GitHub integration enhancements."""
    
    # Remove added columns from existing tables
    op.drop_column('pull_requests', 'latest_test_result_id')
    op.drop_column('pull_requests', 'latest_workflow_automation_id')
    op.drop_column('pull_requests', 'automated_merge_eligible')
    op.drop_column('pull_requests', 'quality_gates_passed')
    
    op.drop_column('github_repositories', 'automation_config')
    op.drop_column('github_repositories', 'quality_gates_config')
    op.drop_column('github_repositories', 'auto_merge_enabled')
    op.drop_column('github_repositories', 'latest_health_assessment_id')
    
    op.drop_column('code_reviews', 'ai_confidence_score')
    op.drop_column('code_reviews', 'complexity_score')
    op.drop_column('code_reviews', 'review_duration_seconds')
    op.drop_column('code_reviews', 'automated_fixes_applied')
    
    # Drop indexes
    op.drop_index('idx_test_results_pr_status', 'test_results')
    op.drop_index('idx_test_results_created_success', 'test_results')
    op.drop_index('idx_conflict_resolutions_pr_strategy', 'conflict_resolutions')
    op.drop_index('idx_conflict_resolutions_auto_resolvable', 'conflict_resolutions')
    op.drop_index('idx_workflow_automations_pr_status', 'workflow_automations')
    op.drop_index('idx_workflow_automations_trigger_stage', 'workflow_automations')
    op.drop_index('idx_dependency_analyses_repo_status', 'dependency_analyses')
    op.drop_index('idx_dependency_analyses_security_risk', 'dependency_analyses')
    op.drop_index('idx_repo_health_score_status', 'repository_health_assessments')
    op.drop_index('idx_repo_health_repo_created', 'repository_health_assessments')
    
    # Drop new tables (in reverse order of creation due to foreign keys)
    op.drop_table('repository_health_assessments')
    op.drop_table('dependency_analyses')
    op.drop_table('workflow_automations')
    op.drop_table('conflict_resolutions')
    op.drop_table('test_results')