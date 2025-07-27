"""Add Security & Authentication System

Revision ID: 008
Revises: 007
Create Date: 2024-01-01 21:00:00.000000

Implements comprehensive OAuth 2.0/OIDC authentication, RBAC authorization,
and audit logging for production-grade security.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '008'
down_revision = '007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add security and authentication tables."""
    
    # Create agent_status enum for agent identities
    op.execute("""
        CREATE TYPE agent_status AS ENUM (
            'active', 'inactive', 'suspended', 'revoked'
        );
    """)
    
    # Create role_scope enum for agent roles
    op.execute("""
        CREATE TYPE role_scope AS ENUM (
            'global', 'session', 'context', 'resource'
        );
    """)
    
    # Agent identities table - OAuth 2.0/OIDC credentials and metadata
    op.create_table(
        'agent_identities',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('agent_name', sa.String(255), nullable=False, index=True),
        sa.Column('human_controller', sa.String(255), nullable=False, index=True),
        sa.Column('oauth_client_id', sa.String(255), nullable=True, unique=True),
        sa.Column('oauth_client_secret_hash', sa.String(255), nullable=True),  # Hashed secret
        sa.Column('public_key', sa.Text, nullable=True),  # For JWT verification
        sa.Column('private_key_encrypted', sa.Text, nullable=True),  # Encrypted private key
        sa.Column('scopes', postgresql.ARRAY(sa.String), nullable=True, server_default='{}'),
        sa.Column('rate_limit_per_minute', sa.Integer, nullable=False, server_default='10'),
        sa.Column('token_expires_in_seconds', sa.Integer, nullable=False, server_default='3600'),  # 1 hour
        sa.Column('refresh_token_expires_in_seconds', sa.Integer, nullable=False, server_default='604800'),  # 7 days
        sa.Column('max_concurrent_tokens', sa.Integer, nullable=False, server_default='5'),
        sa.Column('allowed_redirect_uris', postgresql.ARRAY(sa.String), nullable=True, server_default='{}'),
        sa.Column('metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.Column('last_used', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_token_refresh', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.Enum('active', 'inactive', 'suspended', 'revoked', name='agent_status'), 
                  nullable=False, server_default='active', index=True),
        sa.Column('suspension_reason', sa.Text, nullable=True),
        sa.Column('created_by', sa.String(255), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    
    # Agent roles table - RBAC roles with fine-grained permissions
    op.create_table(
        'agent_roles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('role_name', sa.String(100), nullable=False, unique=True, index=True),
        sa.Column('display_name', sa.String(255), nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('scope', sa.Enum('global', 'session', 'context', 'resource', name='role_scope'), 
                  nullable=False, server_default='resource', index=True),
        sa.Column('permissions', postgresql.JSON, nullable=False, server_default='{}'),
        # permissions structure: {"resources": ["github", "files"], "actions": ["read", "write"], "conditions": {}}
        sa.Column('resource_patterns', postgresql.ARRAY(sa.String), nullable=True, server_default='{}'),
        # e.g., ["github/repos/org/*", "files/workspace/*"]
        sa.Column('max_access_level', sa.String(20), nullable=False, server_default='read'),
        sa.Column('can_delegate', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('auto_expire_hours', sa.Integer, nullable=True),  # Auto-expire role assignment
        sa.Column('is_system_role', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.Column('created_by', sa.String(255), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    
    # Agent role assignments table - Many-to-many with temporal controls
    op.create_table(
        'agent_role_assignments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('role_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('granted_by', sa.String(255), nullable=False),
        sa.Column('granted_reason', sa.Text, nullable=True),
        sa.Column('resource_scope', sa.String(255), nullable=True),  # Specific resource or pattern
        sa.Column('conditions', postgresql.JSON, nullable=True, server_default='{}'),
        # e.g., {"time_restricted": "09:00-17:00", "ip_restricted": ["10.0.0.0/8"]}
        sa.Column('granted_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('revoked_by', sa.String(255), nullable=True),
        sa.Column('revoked_reason', sa.Text, nullable=True),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default='true', index=True),
        sa.Column('metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.ForeignKeyConstraint(['agent_id'], ['agent_identities.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['role_id'], ['agent_roles.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('agent_id', 'role_id', 'resource_scope', name='unique_agent_role_resource')
    )
    
    # Security audit log table - Comprehensive logging with integrity
    op.create_table(
        'security_audit_log',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('human_controller', sa.String(255), nullable=False, index=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('request_id', sa.String(255), nullable=True, index=True),  # For request correlation
        sa.Column('action', sa.String(255), nullable=False, index=True),
        sa.Column('resource', sa.String(255), nullable=True, index=True),
        sa.Column('resource_id', sa.String(255), nullable=True, index=True),
        sa.Column('method', sa.String(10), nullable=True),  # HTTP method
        sa.Column('endpoint', sa.String(255), nullable=True),
        sa.Column('request_data', postgresql.JSON, nullable=True),
        sa.Column('response_data', postgresql.JSON, nullable=True),
        sa.Column('ip_address', postgresql.INET, nullable=True, index=True),
        sa.Column('user_agent', sa.Text, nullable=True),
        sa.Column('success', sa.Boolean, nullable=False, index=True),
        sa.Column('http_status_code', sa.Integer, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('error_code', sa.String(50), nullable=True, index=True),
        sa.Column('duration_ms', sa.Integer, nullable=True),
        sa.Column('tokens_used', sa.Integer, nullable=True),  # For rate limiting tracking
        sa.Column('permission_checked', sa.String(255), nullable=True),
        sa.Column('authorization_result', sa.String(50), nullable=True, index=True),  # granted, denied, error
        sa.Column('risk_score', sa.Float, nullable=True),  # Calculated risk score
        sa.Column('security_labels', postgresql.ARRAY(sa.String), nullable=True, server_default='{}'),
        # e.g., ["suspicious", "bulk_access", "privilege_escalation"]
        sa.Column('geo_location', sa.String(100), nullable=True),  # Country/region
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.Column('log_signature', sa.String(255), nullable=True),  # HMAC signature for integrity
        sa.Column('correlation_id', sa.String(255), nullable=True, index=True),  # For related events
        sa.Column('metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.ForeignKeyConstraint(['agent_id'], ['agent_identities.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ondelete='SET NULL'),
    )
    
    # Token storage table - For JWT token management and blacklisting
    op.create_table(
        'agent_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('token_type', sa.String(20), nullable=False, index=True),  # access, refresh
        sa.Column('token_hash', sa.String(255), nullable=False, unique=True, index=True),  # SHA-256 hash
        sa.Column('jti', sa.String(255), nullable=False, unique=True, index=True),  # JWT ID
        sa.Column('scopes', postgresql.ARRAY(sa.String), nullable=True, server_default='{}'),
        sa.Column('issued_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('usage_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('ip_address', postgresql.INET, nullable=True),
        sa.Column('user_agent', sa.Text, nullable=True),
        sa.Column('is_revoked', sa.Boolean, nullable=False, server_default='false', index=True),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('revoked_reason', sa.String(255), nullable=True),
        sa.Column('metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.ForeignKeyConstraint(['agent_id'], ['agent_identities.id'], ondelete='CASCADE'),
    )
    
    # Security events table - For threat detection and monitoring
    op.create_table(
        'security_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('event_type', sa.String(50), nullable=False, index=True),
        # e.g., "failed_auth", "suspicious_activity", "privilege_escalation"
        sa.Column('severity', sa.String(20), nullable=False, index=True),  # low, medium, high, critical
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('human_controller', sa.String(255), nullable=True, index=True),
        sa.Column('source_ip', postgresql.INET, nullable=True, index=True),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('details', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('risk_score', sa.Float, nullable=True, index=True),
        sa.Column('auto_detected', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('is_resolved', sa.Boolean, nullable=False, server_default='false', index=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolved_by', sa.String(255), nullable=True),
        sa.Column('resolution_notes', sa.Text, nullable=True),
        sa.Column('false_positive', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('related_audit_log_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True, server_default='{}'),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.Column('metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.ForeignKeyConstraint(['agent_id'], ['agent_identities.id'], ondelete='SET NULL'),
    )
    
    # Create comprehensive indexes for performance
    
    # Agent identities indexes
    op.create_index('idx_agent_identities_human_controller', 'agent_identities', ['human_controller'])
    op.create_index('idx_agent_identities_status_created', 'agent_identities', ['status', 'created_at'])
    op.create_index('idx_agent_identities_last_used', 'agent_identities', ['last_used'])
    
    # Agent roles indexes
    op.create_index('idx_agent_roles_scope_system', 'agent_roles', ['scope', 'is_system_role'])
    op.create_index('idx_agent_roles_created_by', 'agent_roles', ['created_by'])
    
    # Role assignments indexes
    op.create_index('idx_role_assignments_agent_active', 'agent_role_assignments', ['agent_id', 'is_active'])
    op.create_index('idx_role_assignments_expires_active', 'agent_role_assignments', ['expires_at', 'is_active'])
    op.create_index('idx_role_assignments_granted_by', 'agent_role_assignments', ['granted_by'])
    
    # Security audit log indexes for fast queries
    op.create_index('idx_audit_log_agent_time', 'security_audit_log', ['agent_id', 'timestamp'])
    op.create_index('idx_audit_log_human_time', 'security_audit_log', ['human_controller', 'timestamp'])
    op.create_index('idx_audit_log_action_success', 'security_audit_log', ['action', 'success'])
    op.create_index('idx_audit_log_resource_time', 'security_audit_log', ['resource', 'timestamp'])
    op.create_index('idx_audit_log_ip_time', 'security_audit_log', ['ip_address', 'timestamp'])
    op.create_index('idx_audit_log_error_code', 'security_audit_log', ['error_code', 'timestamp'])
    op.create_index('idx_audit_log_risk_score', 'security_audit_log', ['risk_score'])
    op.create_index('idx_audit_log_correlation', 'security_audit_log', ['correlation_id'])
    
    # Token management indexes
    op.create_index('idx_agent_tokens_agent_type', 'agent_tokens', ['agent_id', 'token_type'])
    op.create_index('idx_agent_tokens_expires_revoked', 'agent_tokens', ['expires_at', 'is_revoked'])
    op.create_index('idx_agent_tokens_last_used', 'agent_tokens', ['last_used_at'])
    
    # Security events indexes
    op.create_index('idx_security_events_type_severity', 'security_events', ['event_type', 'severity'])
    op.create_index('idx_security_events_agent_time', 'security_events', ['agent_id', 'timestamp'])
    op.create_index('idx_security_events_resolved', 'security_events', ['is_resolved', 'timestamp'])
    op.create_index('idx_security_events_risk_score', 'security_events', ['risk_score'])
    
    # Create partial indexes for active records
    op.create_index(
        'idx_active_role_assignments',
        'agent_role_assignments',
        ['agent_id', 'role_id'],
        postgresql_where=sa.text("is_active = true AND (expires_at IS NULL OR expires_at > now())")
    )
    
    op.create_index(
        'idx_valid_tokens',
        'agent_tokens',
        ['agent_id', 'token_type'],
        postgresql_where=sa.text("is_revoked = false AND expires_at > now()")
    )
    
    # Create views for common security queries
    
    # Active agent permissions view
    op.execute("""
        CREATE OR REPLACE VIEW active_agent_permissions AS
        SELECT 
            ai.id as agent_id,
            ai.agent_name,
            ai.human_controller,
            ai.status as agent_status,
            ar.role_name,
            ar.permissions,
            ar.resource_patterns,
            ar.max_access_level,
            ara.resource_scope,
            ara.conditions,
            ara.expires_at,
            ara.granted_at,
            ara.granted_by
        FROM agent_identities ai
        JOIN agent_role_assignments ara ON ai.id = ara.agent_id
        JOIN agent_roles ar ON ara.role_id = ar.id
        WHERE ai.status = 'active'
        AND ara.is_active = true
        AND (ara.expires_at IS NULL OR ara.expires_at > now())
        AND ara.revoked_at IS NULL;
    """)
    
    # Security dashboard view
    op.execute("""
        CREATE OR REPLACE VIEW security_dashboard AS
        SELECT 
            'audit_summary' as metric_type,
            COUNT(*) as total_events,
            COUNT(*) FILTER (WHERE success = true) as successful_events,
            COUNT(*) FILTER (WHERE success = false) as failed_events,
            AVG(duration_ms) as avg_duration_ms,
            MAX(timestamp) as last_event
        FROM security_audit_log
        WHERE timestamp >= now() - interval '24 hours'
        
        UNION ALL
        
        SELECT 
            'security_events' as metric_type,
            COUNT(*) as total_events,
            COUNT(*) FILTER (WHERE is_resolved = true) as resolved_events,
            COUNT(*) FILTER (WHERE severity IN ('high', 'critical')) as critical_events,
            AVG(risk_score) as avg_risk_score,
            MAX(timestamp) as last_event
        FROM security_events
        WHERE timestamp >= now() - interval '24 hours';
    """)
    
    # Create stored procedures for common operations
    
    # Token cleanup procedure
    op.execute("""
        CREATE OR REPLACE FUNCTION cleanup_expired_tokens()
        RETURNS INTEGER AS $$
        DECLARE
            deleted_count INTEGER;
        BEGIN
            DELETE FROM agent_tokens 
            WHERE expires_at < now() - interval '7 days'
            OR (is_revoked = true AND revoked_at < now() - interval '30 days');
            
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            
            -- Also cleanup old audit logs (configurable retention)
            DELETE FROM security_audit_log 
            WHERE timestamp < now() - interval '90 days';
            
            RETURN deleted_count;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Risk score calculation function
    op.execute("""
        CREATE OR REPLACE FUNCTION calculate_agent_risk_score(
            agent_uuid UUID,
            hours_back INTEGER DEFAULT 24
        ) RETURNS FLOAT AS $$
        DECLARE
            base_risk FLOAT := 0.0;
            failed_auth_count INTEGER;
            suspicious_activity_count INTEGER;
            privilege_escalation_count INTEGER;
            off_hours_activity_count INTEGER;
        BEGIN
            -- Count failed authentications
            SELECT COUNT(*) INTO failed_auth_count
            FROM security_audit_log
            WHERE agent_id = agent_uuid
            AND action LIKE '%auth%'
            AND success = false
            AND timestamp >= now() - interval '1 hour' * hours_back;
            
            -- Count suspicious activities
            SELECT COUNT(*) INTO suspicious_activity_count
            FROM security_events
            WHERE agent_id = agent_uuid
            AND severity IN ('high', 'critical')
            AND timestamp >= now() - interval '1 hour' * hours_back;
            
            -- Count privilege escalation attempts
            SELECT COUNT(*) INTO privilege_escalation_count
            FROM security_audit_log
            WHERE agent_id = agent_uuid
            AND action LIKE '%privilege%'
            AND timestamp >= now() - interval '1 hour' * hours_back;
            
            -- Count off-hours activity (outside 9-17 UTC)
            SELECT COUNT(*) INTO off_hours_activity_count
            FROM security_audit_log
            WHERE agent_id = agent_uuid
            AND EXTRACT(hour FROM timestamp) NOT BETWEEN 9 AND 17
            AND timestamp >= now() - interval '1 hour' * hours_back;
            
            -- Calculate composite risk score (0.0 to 1.0)
            base_risk := LEAST(1.0, 
                (failed_auth_count * 0.15) +
                (suspicious_activity_count * 0.30) +
                (privilege_escalation_count * 0.40) +
                (off_hours_activity_count * 0.05)
            );
            
            RETURN base_risk;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Update table statistics for optimal query planning
    op.execute("ANALYZE agent_identities;")
    op.execute("ANALYZE agent_roles;")
    op.execute("ANALYZE agent_role_assignments;")
    op.execute("ANALYZE security_audit_log;")
    op.execute("ANALYZE agent_tokens;")
    op.execute("ANALYZE security_events;")


def downgrade() -> None:
    """Remove security and authentication tables."""
    
    # Drop stored procedures and views
    op.execute('DROP FUNCTION IF EXISTS calculate_agent_risk_score(UUID, INTEGER);')
    op.execute('DROP FUNCTION IF EXISTS cleanup_expired_tokens();')
    op.execute('DROP VIEW IF EXISTS security_dashboard;')
    op.execute('DROP VIEW IF EXISTS active_agent_permissions;')
    
    # Drop indexes (PostgreSQL automatically drops indexes when tables are dropped,
    # but listing them for completeness)
    op.drop_index('idx_valid_tokens')
    op.drop_index('idx_active_role_assignments')
    op.drop_index('idx_security_events_risk_score')
    op.drop_index('idx_security_events_resolved')
    op.drop_index('idx_security_events_agent_time')
    op.drop_index('idx_security_events_type_severity')
    op.drop_index('idx_agent_tokens_last_used')
    op.drop_index('idx_agent_tokens_expires_revoked')
    op.drop_index('idx_agent_tokens_agent_type')
    op.drop_index('idx_audit_log_correlation')
    op.drop_index('idx_audit_log_risk_score')
    op.drop_index('idx_audit_log_error_code')
    op.drop_index('idx_audit_log_ip_time')
    op.drop_index('idx_audit_log_resource_time')
    op.drop_index('idx_audit_log_action_success')
    op.drop_index('idx_audit_log_human_time')
    op.drop_index('idx_audit_log_agent_time')
    op.drop_index('idx_role_assignments_granted_by')
    op.drop_index('idx_role_assignments_expires_active')
    op.drop_index('idx_role_assignments_agent_active')
    op.drop_index('idx_agent_roles_created_by')
    op.drop_index('idx_agent_roles_scope_system')
    op.drop_index('idx_agent_identities_last_used')
    op.drop_index('idx_agent_identities_status_created')
    op.drop_index('idx_agent_identities_human_controller')
    
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('security_events')
    op.drop_table('agent_tokens')
    op.drop_table('security_audit_log')
    op.drop_table('agent_role_assignments')
    op.drop_table('agent_roles')
    op.drop_table('agent_identities')
    
    # Drop custom enums
    op.execute('DROP TYPE IF EXISTS role_scope;')
    op.execute('DROP TYPE IF EXISTS agent_status;')