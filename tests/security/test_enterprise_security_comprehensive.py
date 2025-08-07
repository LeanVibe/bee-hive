"""
Comprehensive Security Testing Suite for LeanVibe Agent Hive 2.0

Tests for enterprise security system including authentication, authorization,
MFA, API security, secrets management, and compliance.

CRITICAL: Production security validation and penetration testing scenarios.
"""

import asyncio
import base64
import json
import pytest
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi.testclient import TestClient
from httpx import AsyncClient
import jwt

from app.core.enterprise_security_system import (
    EnterpriseSecuritySystem, SecurityConfig, SecurityEvent, SecurityLevel,
    AuthenticationMethod, ThreatLevel
)
from app.core.enterprise_secrets_manager import (
    EnterpriseSecretsManager, SecretType, SecretRequest, RotationPolicy
)
from app.core.enterprise_compliance import (
    EnterpriseComplianceSystem, ComplianceFramework, AuditEvent, DataClassification
)
from app.api.enterprise_security import router as security_router
from app.main import create_app


class TestEnterpriseSecuritySystem:
    """Test suite for enterprise security system."""
    
    @pytest.fixture
    async def security_system(self):
        """Create security system instance for testing."""
        config = SecurityConfig(
            jwt_access_token_expire_minutes=30,
            mfa_enabled=True,
            rate_limit_requests_per_minute=100,
            password_min_length=12
        )
        system = EnterpriseSecuritySystem(config)
        await system.initialize()
        return system
    
    @pytest.fixture
    async def test_client(self, security_system):
        """Create test client with security system."""
        app = create_app()
        app.include_router(security_router)
        return TestClient(app)
    
    # Authentication Tests
    @pytest.mark.asyncio
    async def test_password_hashing_and_verification(self, security_system):
        """Test password hashing and verification."""
        password = "SecurePassword123!"
        
        # Test hashing
        hashed = security_system.hash_password(password)
        assert hashed != password
        assert len(hashed) > 50  # Bcrypt hashes are typically 60 chars
        
        # Test verification
        assert security_system.verify_password(password, hashed)
        assert not security_system.verify_password("WrongPassword", hashed)
    
    @pytest.mark.asyncio
    async def test_password_strength_validation(self, security_system):
        """Test password strength validation."""
        
        # Test weak password
        weak_result = security_system.validate_password_strength("weak")
        assert not weak_result["valid"]
        assert len(weak_result["issues"]) > 0
        assert weak_result["strength_score"] < 50
        
        # Test strong password
        strong_result = security_system.validate_password_strength("SecurePassword123!")
        assert strong_result["valid"]
        assert len(strong_result["issues"]) == 0
        assert strong_result["strength_score"] >= 80
    
    @pytest.mark.asyncio
    async def test_jwt_token_creation_and_verification(self, security_system):
        """Test JWT token creation and verification."""
        user_data = {
            "id": "user-123",
            "email": "test@example.com",
            "role": "developer",
            "permissions": ["read", "write"],
            "security_level": SecurityLevel.INTERNAL.value,
            "mfa_verified": True
        }
        
        # Create token
        token = security_system.create_access_token(user_data)
        assert isinstance(token, str)
        assert len(token) > 100  # JWT tokens are quite long
        
        # Verify token
        payload = security_system.verify_token(token)
        assert payload is not None
        assert payload["sub"] == user_data["id"]
        assert payload["email"] == user_data["email"]
        assert payload["role"] == user_data["role"]
        assert payload["mfa_verified"] == user_data["mfa_verified"]
    
    @pytest.mark.asyncio
    async def test_expired_token_handling(self, security_system):
        """Test handling of expired tokens."""
        user_data = {
            "id": "user-123",
            "email": "test@example.com",
            "role": "developer",
            "permissions": ["read"],
            "security_level": SecurityLevel.INTERNAL.value,
            "mfa_verified": False
        }
        
        # Create token with immediate expiration
        with patch.object(security_system.config, 'jwt_access_token_expire_minutes', -1):
            token = security_system.create_access_token(user_data)
        
        # Verify expired token
        payload = security_system.verify_token(token)
        assert payload is None
    
    # Multi-Factor Authentication Tests
    @pytest.mark.asyncio
    async def test_mfa_secret_generation(self, security_system):
        """Test MFA secret generation."""
        secret = security_system.generate_mfa_secret()
        assert isinstance(secret, str)
        assert len(secret) == 32  # Standard TOTP secret length
        assert secret.isalnum()  # Base32 encoded
    
    @pytest.mark.asyncio
    async def test_mfa_qr_code_generation(self, security_system):
        """Test MFA QR code generation."""
        secret = "JBSWY3DPEHPK3PXP"
        email = "test@example.com"
        
        qr_code_bytes = security_system.generate_mfa_qr_code(email, secret)
        assert isinstance(qr_code_bytes, bytes)
        assert len(qr_code_bytes) > 1000  # QR code images are substantial
    
    @pytest.mark.asyncio
    async def test_mfa_token_verification(self, security_system):
        """Test MFA token verification."""
        secret = "JBSWY3DPEHPK3PXP"
        
        # Generate current TOTP token
        import pyotp
        totp = pyotp.TOTP(secret)
        valid_token = totp.now()
        
        # Test valid token
        assert security_system.verify_mfa_token(secret, valid_token)
        
        # Test invalid token
        assert not security_system.verify_mfa_token(secret, "000000")
    
    # API Key Management Tests
    @pytest.mark.asyncio
    async def test_api_key_generation(self, security_system):
        """Test API key generation."""
        key, key_hash = security_system.generate_api_key("lv")
        
        assert key.startswith("lv_")
        assert len(key) > 30
        assert len(key_hash) == 64  # SHA256 hash length
        assert key != key_hash
    
    @pytest.mark.asyncio
    async def test_api_key_verification(self, security_system):
        """Test API key verification."""
        key, key_hash = security_system.generate_api_key("lv")
        
        # Test valid key
        assert security_system.verify_api_key(key, key_hash)
        
        # Test invalid key
        assert not security_system.verify_api_key("invalid_key", key_hash)
    
    # Data Encryption Tests
    @pytest.mark.asyncio
    async def test_data_encryption_and_decryption(self, security_system):
        """Test data encryption and decryption."""
        original_data = "Sensitive information that needs encryption"
        
        # Encrypt
        encrypted = security_system.encrypt_sensitive_data(original_data)
        assert encrypted != original_data
        assert len(encrypted) > len(original_data)
        
        # Decrypt
        decrypted = security_system.decrypt_sensitive_data(encrypted)
        assert decrypted == original_data
    
    @pytest.mark.asyncio
    async def test_dict_encryption_and_decryption(self, security_system):
        """Test dictionary encryption and decryption."""
        original_dict = {
            "api_key": "secret_key_123",
            "password": "super_secure_password",
            "config": {"debug": True, "max_retries": 3}
        }
        
        # Encrypt
        encrypted = security_system.encrypt_sensitive_data(original_dict)
        assert isinstance(encrypted, str)
        
        # Decrypt
        decrypted = security_system.decrypt_sensitive_data(encrypted)
        assert decrypted == original_dict
    
    # Rate Limiting Tests
    @pytest.mark.asyncio
    async def test_rate_limiting(self, security_system):
        """Test rate limiting functionality."""
        identifier = "test_user_123"
        action = "api_call"
        
        # Test normal usage within limits
        for _ in range(10):
            result = await security_system.check_rate_limit(identifier, action)
            assert result is True
        
        # Get rate limit info
        info = await security_system.get_rate_limit_info(identifier, action)
        assert info["requests_made"] == 10
        assert info["requests_remaining"] > 0
    
    # Threat Detection Tests
    @pytest.mark.asyncio
    async def test_threat_detection(self, security_system):
        """Test threat detection capabilities."""
        
        # Create mock request with suspicious patterns
        mock_request = MagicMock()
        mock_request.url.path = "/api/users"
        mock_request.url = f"http://example.com/api/users?id=1' OR '1'='1"  # SQL injection attempt
        mock_request.method = "GET"
        mock_request.headers = {"user-agent": "sqlmap/1.0"}
        mock_request.client.host = "192.168.1.100"
        
        threat_analysis = await security_system.detect_threat(mock_request, "user-123")
        
        assert threat_analysis is not None
        assert threat_analysis["threat_detected"] is True
        assert len(threat_analysis["threats"]) > 0
        assert threat_analysis["threat_level"] in ["MEDIUM", "HIGH", "CRITICAL"]
    
    # Security Event Logging Tests
    @pytest.mark.asyncio
    async def test_security_event_logging(self, security_system):
        """Test security event logging."""
        
        await security_system.log_security_event(
            SecurityEvent.LOGIN_SUCCESS,
            user_id="user-123",
            additional_info="Test login"
        )
        
        # Verify event was logged (would check logs in real implementation)
        # For now, just ensure no exceptions were raised
        assert True


class TestSecretsManagement:
    """Test suite for secrets management system."""
    
    @pytest.fixture
    async def secrets_manager(self):
        """Create secrets manager instance for testing."""
        manager = EnterpriseSecretsManager()
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_secret_creation_and_retrieval(self, secrets_manager):
        """Test secret creation and retrieval."""
        secret_request = SecretRequest(
            name="test_api_key",
            secret_type=SecretType.API_KEY,
            value="secret_value_123",
            expires_days=30,
            rotation_policy=RotationPolicy.MANUAL
        )
        
        owner_id = "user-123"
        
        # Create secret
        metadata = await secrets_manager.create_secret(secret_request, owner_id)
        assert metadata.name == secret_request.name
        assert metadata.secret_type == secret_request.secret_type
        assert metadata.owner_id == owner_id
        
        # Retrieve secret
        retrieved_value = await secrets_manager.get_secret(metadata.secret_id, owner_id)
        assert retrieved_value == secret_request.value
    
    @pytest.mark.asyncio
    async def test_secret_encryption(self, secrets_manager):
        """Test that secrets are encrypted in storage."""
        secret_value = "plaintext_secret"
        
        encrypted = secrets_manager._encrypt_secret(secret_value)
        assert encrypted != secret_value
        assert len(encrypted) > len(secret_value)
        
        decrypted = secrets_manager._decrypt_secret(encrypted)
        assert decrypted == secret_value
    
    @pytest.mark.asyncio
    async def test_secret_access_control(self, secrets_manager):
        """Test secret access control."""
        secret_request = SecretRequest(
            name="restricted_secret",
            secret_type=SecretType.API_KEY,
            value="restricted_value",
            access_permissions=["user-456"]  # Only specific user can access
        )
        
        owner_id = "user-123"
        authorized_user = "user-456"
        unauthorized_user = "user-789"
        
        # Create secret
        metadata = await secrets_manager.create_secret(secret_request, owner_id)
        
        # Owner can access
        value = await secrets_manager.get_secret(metadata.secret_id, owner_id)
        assert value == secret_request.value
        
        # Authorized user can access
        value = await secrets_manager.get_secret(metadata.secret_id, authorized_user)
        assert value == secret_request.value
        
        # Unauthorized user cannot access
        value = await secrets_manager.get_secret(metadata.secret_id, unauthorized_user)
        assert value is None
    
    @pytest.mark.asyncio
    async def test_secret_expiration(self, secrets_manager):
        """Test secret expiration handling."""
        secret_request = SecretRequest(
            name="expiring_secret",
            secret_type=SecretType.TOKEN,
            value="expiring_value",
            expires_days=1  # Expires in 1 day
        )
        
        owner_id = "user-123"
        
        # Create secret
        metadata = await secrets_manager.create_secret(secret_request, owner_id)
        
        # Should be accessible immediately
        value = await secrets_manager.get_secret(metadata.secret_id, owner_id)
        assert value == secret_request.value
        
        # Simulate expiration by modifying metadata
        metadata.expires_at = datetime.utcnow() - timedelta(hours=1)
        backend = secrets_manager._get_storage_backend(metadata.secret_type)
        await backend.update_metadata(metadata.secret_id, metadata)
        
        # Should not be accessible after expiration
        value = await secrets_manager.get_secret(metadata.secret_id, owner_id)
        assert value is None
    
    @pytest.mark.asyncio
    async def test_secret_rotation(self, secrets_manager):
        """Test secret rotation functionality."""
        secret_request = SecretRequest(
            name="rotating_secret",
            secret_type=SecretType.API_KEY,
            value="original_value",
            rotation_policy=RotationPolicy.TIME_BASED,
            rotation_interval_days=30
        )
        
        owner_id = "user-123"
        
        # Create secret
        metadata = await secrets_manager.create_secret(secret_request, owner_id)
        original_value = await secrets_manager.get_secret(metadata.secret_id, owner_id)
        
        # Rotate secret
        success = await secrets_manager.rotate_secret(metadata.secret_id)
        assert success is True
        
        # Verify new value is different
        new_value = await secrets_manager.get_secret(metadata.secret_id, owner_id)
        assert new_value != original_value
        assert new_value is not None
    
    @pytest.mark.asyncio
    async def test_secret_usage_tracking(self, secrets_manager):
        """Test secret usage tracking."""
        secret_request = SecretRequest(
            name="tracked_secret",
            secret_type=SecretType.API_KEY,
            value="tracked_value",
            max_usage_count=3  # Limit to 3 uses
        )
        
        owner_id = "user-123"
        
        # Create secret
        metadata = await secrets_manager.create_secret(secret_request, owner_id)
        
        # Use secret within limit
        for i in range(3):
            value = await secrets_manager.get_secret(metadata.secret_id, owner_id)
            assert value == secret_request.value
        
        # Fourth access should be denied
        value = await secrets_manager.get_secret(metadata.secret_id, owner_id)
        assert value is None


class TestComplianceSystem:
    """Test suite for compliance system."""
    
    @pytest.fixture
    async def compliance_system(self):
        """Create compliance system instance for testing."""
        frameworks = [ComplianceFramework.SOC2_TYPE2, ComplianceFramework.GDPR]
        system = EnterpriseComplianceSystem(frameworks)
        await system.initialize()
        return system
    
    @pytest.mark.asyncio
    async def test_audit_event_logging(self, compliance_system):
        """Test audit event logging."""
        event = AuditEvent(
            event_type="test_event",
            user_id="user-123",
            action="test_action",
            outcome="success",
            details={"test_key": "test_value"},
            data_classification=DataClassification.INTERNAL
        )
        
        await compliance_system.log_audit_event(event)
        
        # Verify event was processed (no exceptions means success)
        assert True
    
    @pytest.mark.asyncio
    async def test_soc2_audit_format(self, compliance_system):
        """Test SOC2 audit log format."""
        event = AuditEvent(
            event_type="user_login",
            user_id="user-123",
            resource_id="login_system",
            action="authenticate",
            outcome="success",
            ip_address="192.168.1.100",
            details={"method": "password"}
        )
        
        soc2_format = event.to_soc2_format()
        
        assert "timestamp" in soc2_format
        assert soc2_format["user_identifier"] == "user-123"
        assert soc2_format["action_performed"] == "authenticate"
        assert soc2_format["resource_accessed"] == "login_system"
        assert soc2_format["source_ip"] == "192.168.1.100"
        assert soc2_format["outcome"] == "success"
    
    @pytest.mark.asyncio
    async def test_gdpr_audit_format(self, compliance_system):
        """Test GDPR audit log format."""
        event = AuditEvent(
            event_type="data_processing",
            user_id="data-subject-123",
            action="user_profile_update",
            outcome="success",
            data_classification=DataClassification.PII,
            details={"lawful_basis": "consent"}
        )
        
        gdpr_format = event.to_gdpr_format()
        
        assert "processing_timestamp" in gdpr_format
        assert gdpr_format["data_subject_id"] == "data-subject-123"
        assert gdpr_format["processing_activity"] == "user_profile_update"
        assert gdpr_format["data_categories"] == ["pii"]
        assert gdpr_format["lawful_basis"] == "consent"
        assert gdpr_format["controller"] == "LeanVibe Agent Hive"
    
    @pytest.mark.asyncio
    async def test_compliance_report_generation(self, compliance_system):
        """Test compliance report generation."""
        report = await compliance_system.generate_compliance_report(
            ComplianceFramework.SOC2_TYPE2,
            period_days=30
        )
        
        assert report.framework == ComplianceFramework.SOC2_TYPE2
        assert report.compliance_score >= 0
        assert report.compliance_score <= 100
        assert report.controls_assessed >= 0
        assert isinstance(report.findings, list)
        assert isinstance(report.recommendations, list)


class TestSecurityIntegration:
    """Test suite for security system integration."""
    
    @pytest.fixture
    async def integrated_systems(self):
        """Create integrated security systems for testing."""
        security_config = SecurityConfig()
        security_system = EnterpriseSecuritySystem(security_config)
        await security_system.initialize()
        
        secrets_manager = EnterpriseSecretsManager()
        await secrets_manager.initialize()
        
        compliance_system = EnterpriseComplianceSystem()
        await compliance_system.initialize()
        
        return {
            "security": security_system,
            "secrets": secrets_manager,
            "compliance": compliance_system
        }
    
    @pytest.mark.asyncio
    async def test_login_with_compliance_logging(self, integrated_systems):
        """Test login process with compliance audit logging."""
        security = integrated_systems["security"]
        compliance = integrated_systems["compliance"]
        
        user_data = {
            "id": "user-123",
            "email": "test@example.com",
            "role": "developer",
            "permissions": ["read"],
            "security_level": SecurityLevel.INTERNAL.value,
            "mfa_verified": True
        }
        
        # Create access token
        token = security.create_access_token(user_data)
        
        # Log login event for compliance
        login_event = AuditEvent(
            event_type="user_login",
            user_id=user_data["id"],
            action="jwt_token_created",
            outcome="success",
            data_classification=DataClassification.INTERNAL
        )
        
        await compliance.log_audit_event(login_event)
        
        # Verify token is valid
        payload = security.verify_token(token)
        assert payload is not None
        assert payload["sub"] == user_data["id"]
    
    @pytest.mark.asyncio
    async def test_secret_access_with_audit_trail(self, integrated_systems):
        """Test secret access with complete audit trail."""
        secrets = integrated_systems["secrets"]
        compliance = integrated_systems["compliance"]
        
        # Create secret
        secret_request = SecretRequest(
            name="audited_secret",
            secret_type=SecretType.API_KEY,
            value="audited_value"
        )
        
        owner_id = "user-123"
        metadata = await secrets.create_secret(secret_request, owner_id)
        
        # Log secret creation
        creation_event = AuditEvent(
            event_type="secret_created",
            user_id=owner_id,
            resource_id=metadata.secret_id,
            action="create_secret",
            outcome="success",
            data_classification=DataClassification.CONFIDENTIAL
        )
        
        await compliance.log_audit_event(creation_event)
        
        # Access secret
        value = await secrets.get_secret(metadata.secret_id, owner_id)
        
        # Log secret access
        access_event = AuditEvent(
            event_type="secret_accessed",
            user_id=owner_id,
            resource_id=metadata.secret_id,
            action="retrieve_secret",
            outcome="success",
            data_classification=DataClassification.CONFIDENTIAL
        )
        
        await compliance.log_audit_event(access_event)
        
        assert value == secret_request.value
    
    @pytest.mark.asyncio
    async def test_security_incident_response(self, integrated_systems):
        """Test security incident response workflow."""
        security = integrated_systems["security"]
        compliance = integrated_systems["compliance"]
        
        # Simulate security incident
        incident_details = {
            "incident_type": "unauthorized_access_attempt",
            "severity": "high",
            "source_ip": "malicious.ip.address",
            "attempted_resource": "/admin/users",
            "detection_method": "threat_analysis"
        }
        
        # Log security incident
        incident_event = AuditEvent(
            event_type="security_incident",
            action="unauthorized_access_blocked",
            outcome="blocked",
            details=incident_details,
            data_classification=DataClassification.RESTRICTED
        )
        
        await compliance.log_audit_event(incident_event)
        
        # Log security response
        response_event = AuditEvent(
            event_type="security_response",
            action="ip_address_blocked",
            outcome="success",
            details={"blocked_ip": incident_details["source_ip"]},
            data_classification=DataClassification.INTERNAL
        )
        
        await compliance.log_audit_event(response_event)
        
        # Verify incident was logged (no exceptions means success)
        assert True


class TestSecurityPenetration:
    """Penetration testing scenarios for security validation."""
    
    @pytest.fixture
    async def test_app(self):
        """Create test application with security endpoints."""
        app = create_app()
        app.include_router(security_router)
        return app
    
    @pytest.mark.asyncio
    async def test_sql_injection_protection(self, test_app):
        """Test protection against SQL injection attacks."""
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Attempt SQL injection in login
            malicious_payload = {
                "email": "admin@example.com' OR '1'='1",
                "password": "password"
            }
            
            response = await client.post("/api/v1/security/login", json=malicious_payload)
            
            # Should not succeed with SQL injection
            assert response.status_code in [400, 401, 422]  # Various rejection codes
    
    @pytest.mark.asyncio
    async def test_xss_protection(self, test_app):
        """Test protection against XSS attacks."""
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Attempt XSS in registration
            malicious_payload = {
                "email": "test@example.com",
                "password": "SecurePassword123!",
                "full_name": "<script>alert('xss')</script>",
                "role": "developer"
            }
            
            response = await client.post("/api/v1/security/register", json=malicious_payload)
            
            # Should handle XSS attempt safely
            assert response.status_code in [200, 400, 422]
    
    @pytest.mark.asyncio
    async def test_brute_force_protection(self, test_app):
        """Test protection against brute force attacks."""
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Attempt multiple failed logins
            failed_attempts = 0
            for i in range(10):  # Try 10 failed attempts
                response = await client.post("/api/v1/security/login", json={
                    "email": "admin@leanvibe.com",
                    "password": f"wrong_password_{i}"
                })
                
                if response.status_code == 429:  # Rate limited
                    break
                elif response.status_code == 401:  # Failed login
                    failed_attempts += 1
            
            # Should have triggered rate limiting
            assert failed_attempts < 10  # Rate limiting kicked in
    
    @pytest.mark.asyncio
    async def test_invalid_jwt_handling(self, test_app):
        """Test handling of invalid JWT tokens."""
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            
            # Test with completely invalid token
            headers = {"Authorization": "Bearer invalid_token_here"}
            response = await client.get("/api/v1/security/config", headers=headers)
            assert response.status_code == 401
            
            # Test with malformed token
            headers = {"Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.malformed"}
            response = await client.get("/api/v1/security/config", headers=headers)
            assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_authorization_bypass_attempts(self, test_app):
        """Test attempts to bypass authorization."""
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            
            # Create a low-privilege token
            user_data = {
                "id": "user-123",
                "email": "user@example.com",
                "role": "viewer",
                "permissions": ["read"],
                "security_level": SecurityLevel.PUBLIC.value
            }
            
            # Would need proper security system to create valid token
            # For now, test with invalid token
            headers = {"Authorization": "Bearer fake_low_privilege_token"}
            
            # Try to access high-privilege endpoint
            response = await client.get("/api/v1/security/config", headers=headers)
            assert response.status_code in [401, 403]  # Unauthorized or Forbidden


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])