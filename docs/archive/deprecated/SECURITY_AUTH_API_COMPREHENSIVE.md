# ⚠️ DEPRECATED - Enterprise Security Auth System - API Reference

**NOTICE**: This file has been deprecated and consolidated into `docs/implementation/security-implementation-guide.md`

**Deprecation Date**: 2025-07-31
**Reason**: Content redundancy - consolidated with other security implementation files

**For current security information, please refer to:**
- `SECURITY.md` (root) - External security policy
- `docs/prd/security-auth-system.md` - Technical specifications
- `docs/implementation/security-implementation-guide.md` - Implementation procedures

---

# Enterprise Security Auth System - API Reference

## Overview

LeanVibe Agent Hive 2.0 implements a comprehensive enterprise-grade security authentication system with OAuth 2.0/OIDC compliance, JWT token management with key rotation, RBAC authorization, and advanced threat detection.

## Base URL

```
Production: https://api.leanvibe.dev/api/v1/auth
Development: http://localhost:8000/api/v1/auth
```

## Authentication Flow

The system implements OAuth 2.0 with OIDC (OpenID Connect) for enterprise-grade authentication:

1. **Agent Identity Registration**
2. **OAuth 2.0 Authorization Code Flow**
3. **JWT Token Generation with Key Rotation**
4. **RBAC Permission Validation**
5. **Audit Logging and Threat Detection**

## Core API Endpoints

### Agent Identity Management

#### Register Agent Identity
```http
POST /api/v1/auth/agents/register
Content-Type: application/json
Authorization: Bearer <admin-token>

{
  "agent_name": "production-agent-01",
  "agent_type": "orchestrator",
  "human_owner": "user@company.com",
  "capabilities": ["code_execution", "github_integration", "context_management"],
  "security_level": "enterprise",
  "metadata": {
    "department": "engineering",
    "project": "leanvibe-hive"
  }
}
```

**Response:**
```json
{
  "agent_id": "agent_550e8400-e29b-41d4-a716-446655440000",
  "agent_name": "production-agent-01",
  "identity_created": "2025-07-29T10:30:00Z",
  "public_key": "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkq...",
  "agent_certificate": "-----BEGIN CERTIFICATE-----\nMIIDXTCC...",
  "status": "active"
}
```

#### Get Agent Identity
```http
GET /api/v1/auth/agents/{agent_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "agent_id": "agent_550e8400-e29b-41d4-a716-446655440000",
  "agent_name": "production-agent-01",
  "agent_type": "orchestrator",
  "human_owner": "user@company.com",
  "capabilities": ["code_execution", "github_integration", "context_management"],
  "security_level": "enterprise",
  "status": "active",
  "created_at": "2025-07-29T10:30:00Z",
  "last_authenticated": "2025-07-29T14:25:30Z",
  "authentication_count": 45,
  "security_events": 0
}
```

### OAuth 2.0/OIDC Authentication

#### OAuth 2.0 Authorization
```http
GET /api/v1/auth/oauth/authorize
  ?client_id=agent_550e8400-e29b-41d4-a716-446655440000
  &response_type=code
  &scope=agent.orchestrate+agent.github+agent.context
  &redirect_uri=https://agent.leanvibe.dev/callback
  &state=random-state-string
  &code_challenge=S256-challenge
  &code_challenge_method=S256
```

**Response:** Redirects to authorization URL with authorization code

#### Token Exchange
```http
POST /api/v1/auth/oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&
code=AUTH_CODE_HERE&
redirect_uri=https://agent.leanvibe.dev/callback&
client_id=agent_550e8400-e29b-41d4-a716-446655440000&
code_verifier=PKCE_VERIFIER
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InJzYTI1Ni0yMDI1LTA3LTI5In0...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "rt_550e8400-e29b-41d4-a716-446655440000",
  "scope": "agent.orchestrate agent.github agent.context",
  "id_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InJzYTI1Ni0yMDI1LTA3LTI5In0..."
}
```

#### Token Refresh
```http
POST /api/v1/auth/oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=refresh_token&
refresh_token=rt_550e8400-e29b-41d4-a716-446655440000&
client_id=agent_550e8400-e29b-41d4-a716-446655440000
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InJzYTI1Ni0yMDI1LTA3LTI5In0...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "rt_550e8400-e29b-41d4-a716-446655440001",
  "scope": "agent.orchestrate agent.github agent.context"
}
```

### JWT Token Management

#### Validate Token
```http
POST /api/v1/auth/tokens/validate
Content-Type: application/json
Authorization: Bearer <token>

{
  "token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InJzYTI1Ni0yMDI1LTA3LTI5In0..."
}
```

**Response:**
```json
{
  "valid": true,
  "agent_id": "agent_550e8400-e29b-41d4-a716-446655440000",
  "scopes": ["agent.orchestrate", "agent.github", "agent.context"],
  "expires_at": "2025-07-29T15:30:00Z",
  "issued_at": "2025-07-29T14:30:00Z",
  "key_id": "rsa256-2025-07-29",
  "human_owner": "user@company.com"
}
```

#### Get Public Keys (JWKS)
```http
GET /api/v1/auth/.well-known/jwks.json
```

**Response:**
```json
{
  "keys": [
    {
      "kty": "RSA",
      "kid": "rsa256-2025-07-29",
      "use": "sig",
      "alg": "RS256",
      "n": "0vx7agoebGcQSuuPiIOXXXX...",
      "e": "AQAB"
    },
    {
      "kty": "RSA",
      "kid": "rsa256-2025-07-28",
      "use": "sig",
      "alg": "RS256",
      "n": "xGbUXXXXXXXXXXXXXXXXXX...",
      "e": "AQAB"
    }
  ]
}
```

### RBAC Authorization

#### Check Permissions
```http
POST /api/v1/auth/permissions/check
Content-Type: application/json
Authorization: Bearer <token>

{
  "resource": "github.repository",
  "action": "write",
  "resource_id": "leanvibe-dev/agent-hive",
  "context": {
    "branch": "feature/security-enhancement",
    "file_path": "app/core/security.py"
  }
}
```

**Response:**
```json
{
  "allowed": true,
  "permission": "github.repository.write",
  "role": "senior-agent",
  "policy_applied": "enterprise-security-policy-v2",
  "conditions": [
    {
      "type": "branch_protection",
      "satisfied": true,
      "details": "Feature branch modifications allowed"
    },
    {
      "type": "file_security",
      "satisfied": true,
      "details": "Security file modifications require enhanced audit"
    }
  ],
  "audit_required": true
}
```

#### Assign Role
```http
POST /api/v1/auth/roles/assign
Content-Type: application/json
Authorization: Bearer <admin-token>

{
  "agent_id": "agent_550e8400-e29b-41d4-a716-446655440000",
  "role": "senior-agent",
  "permissions": [
    "orchestrator.manage",
    "github.repository.write",
    "context.compress",
    "security.audit.read"
  ],
  "conditions": {
    "ip_restrictions": ["10.0.0.0/8", "192.168.1.0/24"],
    "time_restrictions": "business_hours",
    "mfa_required": true
  }
}
```

**Response:**
```json
{
  "role_assignment_id": "ra_660e8400-e29b-41d4-a716-446655440000",
  "agent_id": "agent_550e8400-e29b-41d4-a716-446655440000",
  "role": "senior-agent",
  "permissions": [
    "orchestrator.manage",
    "github.repository.write",
    "context.compress",
    "security.audit.read"
  ],
  "assigned_at": "2025-07-29T14:30:00Z",
  "assigned_by": "admin@company.com",
  "effective_until": "2025-08-29T14:30:00Z"
}
```

### Audit Logging

#### Get Audit Logs
```http
GET /api/v1/auth/audit/logs
  ?agent_id=agent_550e8400-e29b-41d4-a716-446655440000
  &start_time=2025-07-29T00:00:00Z
  &end_time=2025-07-29T23:59:59Z
  &event_type=authentication
  &limit=100
Authorization: Bearer <admin-token>
```

**Response:**
```json
{
  "logs": [
    {
      "event_id": "evt_770e8400-e29b-41d4-a716-446655440000",
      "agent_id": "agent_550e8400-e29b-41d4-a716-446655440000",
      "event_type": "authentication_success",
      "timestamp": "2025-07-29T14:25:30Z",
      "source_ip": "192.168.1.100",
      "user_agent": "LeanVibe-Agent/2.0",
      "details": {
        "method": "oauth2_token",
        "scopes": ["agent.orchestrate", "agent.github"],
        "token_id": "tkn_880e8400-e29b-41d4-a716-446655440000"
      },
      "risk_score": 0.1,
      "signature": "SHA256:a1b2c3d4e5f6..."
    }
  ],
  "total": 45,
  "page": 1,
  "limit": 100
}
```

#### Create Audit Event
```http
POST /api/v1/auth/audit/events
Content-Type: application/json
Authorization: Bearer <token>

{
  "event_type": "permission_granted",
  "resource": "github.repository.leanvibe-dev/agent-hive",
  "action": "code_modification",
  "details": {
    "file_modified": "app/core/enhanced_security.py",
    "lines_changed": 25,
    "commit_hash": "abc123def456"
  },
  "risk_assessment": {
    "risk_level": "medium",
    "factors": ["security_file_modification", "production_environment"]
  }
}
```

**Response:**
```json
{
  "event_id": "evt_990e8400-e29b-41d4-a716-446655440000",
  "logged_at": "2025-07-29T14:30:15Z",
  "signature": "SHA256:f6e5d4c3b2a1...",
  "immutable_hash": "BLAKE3:123abc456def...",
  "audit_trail_position": 12847
}
```

### Threat Detection

#### Get Security Events
```http
GET /api/v1/auth/security/events
  ?severity=high
  &start_time=2025-07-29T00:00:00Z
  &resolved=false
  &limit=50
Authorization: Bearer <security-admin-token>
```

**Response:**
```json
{
  "events": [
    {
      "event_id": "sec_aa0e8400-e29b-41d4-a716-446655440000",
      "agent_id": "agent_550e8400-e29b-41d4-a716-446655440000",
      "event_type": "suspicious_activity",
      "severity": "high",
      "detected_at": "2025-07-29T13:45:00Z",
      "description": "Multiple failed authentication attempts from different IPs",
      "indicators": [
        {
          "type": "ip_anomaly",
          "value": "Multiple source IPs: 203.0.113.1, 198.51.100.1, 192.0.2.1",
          "confidence": 0.85
        },
        {
          "type": "temporal_anomaly",
          "value": "15 attempts in 2 minutes",
          "confidence": 0.92
        }
      ],
      "mitigation": {
        "automatic_actions": ["rate_limit_applied", "ip_temporary_block"],
        "recommended_actions": ["investigate_source", "review_agent_credentials"]
      },
      "resolved": false
    }
  ],
  "total": 3,
  "page": 1,
  "limit": 50
}
```

#### Report Security Incident
```http
POST /api/v1/auth/security/incidents
Content-Type: application/json
Authorization: Bearer <token>

{
  "incident_type": "credential_compromise",
  "severity": "critical",
  "description": "Suspected unauthorized access to agent credentials",
  "affected_agents": ["agent_550e8400-e29b-41d4-a716-446655440000"],
  "evidence": {
    "suspicious_ips": ["203.0.113.1"],
    "unusual_activity": "Access outside business hours",
    "failed_authentications": 25
  },
  "immediate_actions": ["revoke_tokens", "rotate_keys", "notify_human_owner"]
}
```

**Response:**
```json
{
  "incident_id": "inc_bb0e8400-e29b-41d4-a716-446655440000",
  "status": "investigating",
  "assigned_to": "security-team",
  "created_at": "2025-07-29T14:35:00Z",
  "response_actions": [
    {
      "action": "revoke_tokens",
      "status": "completed",
      "timestamp": "2025-07-29T14:35:30Z"
    },
    {
      "action": "rotate_keys",
      "status": "in_progress",
      "estimated_completion": "2025-07-29T14:40:00Z"
    }
  ],
  "escalation_level": "critical"
}
```

## Rate Limiting

The API implements intelligent rate limiting with DDoS protection:

**Headers in Response:**
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1690633200
X-RateLimit-Burst: 50
```

**Rate Limits by Endpoint:**
- Authentication: 10 requests/minute per IP
- Token operations: 100 requests/minute per agent
- Audit logs: 1000 requests/hour per admin
- Security events: 500 requests/hour per security admin

## Error Responses

### Standard Error Format
```json
{
  "error": {
    "code": "INVALID_CREDENTIALS",
    "message": "The provided credentials are invalid or expired",
    "details": {
      "timestamp": "2025-07-29T14:30:00Z",
      "request_id": "req_cc0e8400-e29b-41d4-a716-446655440000",
      "agent_id": "agent_550e8400-e29b-41d4-a716-446655440000"
    },
    "help_url": "https://docs.leanvibe.dev/auth/troubleshooting#invalid-credentials"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| INVALID_CREDENTIALS | 401 | Authentication credentials are invalid |
| EXPIRED_TOKEN | 401 | JWT token has expired |
| INSUFFICIENT_PERMISSIONS | 403 | Agent lacks required permissions |
| RATE_LIMIT_EXCEEDED | 429 | Rate limit exceeded for endpoint |
| SECURITY_VIOLATION | 403 | Security policy violation detected |
| ACCOUNT_LOCKED | 423 | Agent account temporarily locked |
| MFA_REQUIRED | 428 | Multi-factor authentication required |

## Security Features

### Token Security
- **RS256 JWT signatures** with rotating keys
- **Key rotation** every 24 hours automatically
- **Token expiration** with configurable lifetimes
- **Refresh token security** with single-use tokens

### Advanced Protection
- **PKCE (Proof Key for Code Exchange)** for OAuth flows
- **IP allowlisting** with geolocation validation
- **Device fingerprinting** for anomaly detection
- **Behavioral analysis** for threat detection

### Compliance
- **SOC 2 Type II** compliant audit logging
- **GDPR** compliant data handling
- **FIPS 140-2** level 2 cryptographic standards
- **OAuth 2.1** and **OIDC 1.0** specification compliance

## SDK Examples

### Python SDK
```python
from leanvibe_auth import AgentAuth

# Initialize auth client
auth = AgentAuth(
    base_url="https://api.leanvibe.dev",
    client_id="agent_550e8400-e29b-41d4-a716-446655440000",
    client_secret="secret_key_here"
)

# Authenticate agent
token = await auth.authenticate(
    scopes=["agent.orchestrate", "agent.github"]
)

# Check permissions
allowed = await auth.check_permission(
    resource="github.repository",
    action="write",
    resource_id="leanvibe-dev/agent-hive"
)

# Create audit event
await auth.audit_log(
    event_type="code_modification",
    details={"file": "security.py", "lines": 25}
)
```

### JavaScript SDK
```javascript
import { LeanVibeAuth } from '@leanvibe/auth-sdk';

const auth = new LeanVibeAuth({
  baseUrl: 'https://api.leanvibe.dev',
  clientId: 'agent_550e8400-e29b-41d4-a716-446655440000',
  clientSecret: 'secret_key_here'
});

// Authenticate
const token = await auth.authenticate({
  scopes: ['agent.orchestrate', 'agent.github']
});

// Validate token
const isValid = await auth.validateToken(token.access_token);

// Security monitoring
auth.on('security_event', (event) => {
  console.log('Security event detected:', event);
});
```

This comprehensive security auth system provides enterprise-grade authentication, authorization, and audit capabilities with real-time threat detection and advanced security features.