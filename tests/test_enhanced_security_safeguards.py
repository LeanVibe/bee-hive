"""
Comprehensive tests for Enhanced Security Safeguards & Deterministic Control System.

Tests include:
- Deterministic control engine rule evaluation
- Agent behavior monitoring and anomaly detection
- Security decision making and policy enforcement
- Code execution security validation
- Integration with existing security infrastructure
- Performance and reliability testing
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from app.core.enhanced_security_safeguards import (
    EnhancedSecuritySafeguards,
    DeterministicControlEngine,
    AgentBehaviorMonitor,
    SecurityRule,
    SecurityContext,
    AgentBehaviorProfile,
    ControlDecision,
    SecurityRiskLevel,
    AgentBehaviorState,
    SecurityPolicyType,
    validate_agent_action,
    validate_code_execution,
    get_agent_security_status
)
from app.core.code_execution import CodeBlock, CodeLanguage, SecurityLevel


class TestSecurityRule:
    """Test security rule evaluation and logic."""
    
    def test_security_rule_creation(self):
        """Test security rule creation and properties."""
        rule = SecurityRule(
            id="test_rule",
            name="Test Security Rule",
            policy_type=SecurityPolicyType.CODE_EXECUTION,
            conditions={"risk_score": {"operator": ">=", "value": 0.8}},
            decision=ControlDecision.DENY,
            priority=500
        )
        
        assert rule.id == "test_rule"
        assert rule.name == "Test Security Rule"
        assert rule.policy_type == SecurityPolicyType.CODE_EXECUTION
        assert rule.decision == ControlDecision.DENY
        assert rule.priority == 500
        assert rule.enabled is True
    
    def test_rule_evaluation_simple_conditions(self):
        """Test rule evaluation with simple conditions."""
        rule = SecurityRule(
            id="simple_rule",
            name="Simple Rule",
            policy_type=SecurityPolicyType.DATA_ACCESS,
            conditions={
                "action_type": "data_access",
                "data_sensitivity": "confidential"
            },
            decision=ControlDecision.REQUIRE_APPROVAL,
            priority=100
        )
        
        # Matching context
        context = {
            "action_type": "data_access",
            "data_sensitivity": "confidential",
            "risk_score": 0.5
        }
        matches, reason = rule.evaluate(context)
        assert matches is True
        assert "All conditions met" in reason
        
        # Non-matching context
        context = {
            "action_type": "code_execution",
            "data_sensitivity": "normal"
        }
        matches, reason = rule.evaluate(context)
        assert matches is False
        assert "!=" in reason
    
    def test_rule_evaluation_operator_conditions(self):
        """Test rule evaluation with operator-based conditions."""
        rule = SecurityRule(
            id="operator_rule",
            name="Operator Rule",
            policy_type=SecurityPolicyType.BEHAVIORAL_ANALYSIS,
            conditions={
                "risk_score": {"operator": ">=", "value": 0.7},
                "failure_rate": {"operator": "<", "value": 0.3},
                "action_count": {"operator": ">", "value": 100}
            },
            decision=ControlDecision.ESCALATE,
            priority=200
        )
        
        # Test matching conditions
        context = {
            "risk_score": 0.8,
            "failure_rate": 0.2,
            "action_count": 150
        }
        matches, reason = rule.evaluate(context)
        assert matches is True
        
        # Test non-matching conditions
        context = {
            "risk_score": 0.6,  # Below threshold
            "failure_rate": 0.2,
            "action_count": 150
        }
        matches, reason = rule.evaluate(context)
        assert matches is False
        assert "risk_score >= 0.7 failed" in reason
    
    def test_rule_evaluation_list_conditions(self):
        """Test rule evaluation with list-based conditions."""
        rule = SecurityRule(
            id="list_rule",
            name="List Rule",
            policy_type=SecurityPolicyType.DATA_ACCESS,
            conditions={
                "behavior_state": ["ANOMALOUS", "COMPROMISED"],
                "agent_type": ["backend_agent", "data_agent"]
            },
            decision=ControlDecision.QUARANTINE,
            priority=800
        )
        
        # Test matching conditions
        context = {
            "behavior_state": "ANOMALOUS",
            "agent_type": "backend_agent"
        }
        matches, reason = rule.evaluate(context)
        assert matches is True
        
        # Test non-matching conditions
        context = {
            "behavior_state": "NORMAL",
            "agent_type": "backend_agent"
        }
        matches, reason = rule.evaluate(context)
        assert matches is False
        assert "not in allowed list" in reason
    
    def test_rule_evaluation_error_handling(self):
        """Test rule evaluation error handling."""
        rule = SecurityRule(
            id="error_rule",
            name="Error Rule",
            policy_type=SecurityPolicyType.CODE_EXECUTION,
            conditions={
                "invalid_operator": {"operator": "invalid", "value": 0.5}
            },
            decision=ControlDecision.DENY,
            priority=100
        )
        
        context = {"invalid_operator": 0.7}
        matches, reason = rule.evaluate(context)
        assert matches is False
        assert "failed" in reason.lower()


class TestDeterministicControlEngine:
    """Test deterministic control engine functionality."""
    
    @pytest.fixture
    def control_engine(self):
        """Control engine instance for testing."""
        return DeterministicControlEngine()
    
    @pytest.fixture
    def sample_security_context(self):
        """Sample security context for testing."""
        return SecurityContext(
            agent_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            action_type="data_access",
            resource_type="context",
            resource_id="ctx_123",
            timestamp=datetime.utcnow(),
            ip_address="127.0.0.1",
            user_agent="test-agent",
            agent_type="backend_agent",
            agent_capabilities=["read", "write"],
            current_risk_score=0.3,
            behavior_state=AgentBehaviorState.NORMAL,
            data_sensitivity="normal",
            operation_impact="low"
        )
    
    def test_control_engine_initialization(self, control_engine):
        """Test control engine initialization with default rules."""
        assert len(control_engine.security_rules) > 0
        assert control_engine.default_decision == ControlDecision.DENY
        
        # Check that default rules are present
        rule_names = [rule.name for rule in control_engine.security_rules]
        assert "Block Dangerous Code Execution" in rule_names
        assert "Quarantine Compromised Agents" in rule_names
        assert "Default Allow for Normal Operations" in rule_names
    
    def test_add_remove_rules(self, control_engine):
        """Test adding and removing security rules."""
        initial_count = len(control_engine.security_rules)
        
        # Add new rule
        new_rule = SecurityRule(
            id="test_rule_123",
            name="Test Custom Rule",
            policy_type=SecurityPolicyType.DATA_ACCESS,
            conditions={"test_condition": True},
            decision=ControlDecision.ALLOW,
            priority=500
        )
        
        control_engine.add_rule(new_rule)
        assert len(control_engine.security_rules) == initial_count + 1
        
        # Check rule is in list
        rule_ids = [rule.id for rule in control_engine.security_rules]
        assert "test_rule_123" in rule_ids
        
        # Remove rule
        removed = control_engine.remove_rule("test_rule_123")
        assert removed is True
        assert len(control_engine.security_rules) == initial_count
        
        # Try to remove non-existent rule
        removed = control_engine.remove_rule("non_existent")
        assert removed is False
    
    @pytest.mark.asyncio
    async def test_decision_making_normal_agent(self, control_engine, sample_security_context):
        """Test decision making for normal agent behavior."""
        # Normal agent with low risk should be allowed
        sample_security_context.current_risk_score = 0.2
        sample_security_context.behavior_state = AgentBehaviorState.NORMAL
        
        decision, reason, applied_rules = await control_engine.make_decision(sample_security_context)
        
        assert decision == ControlDecision.ALLOW
        assert "Default Allow for Normal Operations" in reason
        assert len(applied_rules) > 0
    
    @pytest.mark.asyncio
    async def test_decision_making_high_risk_agent(self, control_engine, sample_security_context):
        """Test decision making for high risk agent."""
        # High risk agent accessing confidential data should require approval
        sample_security_context.current_risk_score = 0.85
        sample_security_context.data_sensitivity = "confidential"
        
        decision, reason, applied_rules = await control_engine.make_decision(sample_security_context)
        
        assert decision == ControlDecision.REQUIRE_APPROVAL
        assert "High Risk Operations" in reason
        assert len(applied_rules) > 0
    
    @pytest.mark.asyncio
    async def test_decision_making_compromised_agent(self, control_engine, sample_security_context):
        """Test decision making for compromised agent."""
        # Compromised agent should be quarantined
        sample_security_context.behavior_state = AgentBehaviorState.COMPROMISED
        sample_security_context.current_risk_score = 0.95
        
        decision, reason, applied_rules = await control_engine.make_decision(sample_security_context)
        
        assert decision == ControlDecision.QUARANTINE
        assert "Quarantine Compromised Agents" in reason
        assert len(applied_rules) > 0
    
    @pytest.mark.asyncio
    async def test_decision_caching(self, control_engine, sample_security_context):
        """Test decision caching functionality."""
        # Make first decision
        decision1, reason1, rules1 = await control_engine.make_decision(sample_security_context)
        
        # Make second identical decision
        decision2, reason2, rules2 = await control_engine.make_decision(sample_security_context)
        
        assert decision1 == decision2
        assert reason1 == reason2
        
        # Check cache hit in metrics
        metrics = control_engine.get_metrics()
        engine_metrics = metrics["deterministic_control_engine"]
        assert engine_metrics["cache_hits"] > 0
    
    @pytest.mark.asyncio
    async def test_off_hours_access_blocking(self, control_engine, sample_security_context):
        """Test off-hours access blocking."""
        # Set time to off-hours (2 AM)
        off_hours_time = datetime.utcnow().replace(hour=2, minute=0, second=0)
        sample_security_context.timestamp = off_hours_time
        sample_security_context.data_sensitivity = "sensitive"
        
        decision, reason, applied_rules = await control_engine.make_decision(sample_security_context)
        
        # Should be blocked due to off-hours + sensitive data
        assert decision == ControlDecision.DENY
        assert "Off-Hours" in reason
    
    def test_context_to_dict_conversion(self, control_engine, sample_security_context):
        """Test security context to dictionary conversion."""
        context_dict = control_engine._context_to_dict(sample_security_context)
        
        assert "agent_id" in context_dict
        assert "action_type" in context_dict
        assert "risk_score" in context_dict
        assert "behavior_state" in context_dict
        assert "is_off_hours" in context_dict
        assert "hour" in context_dict
        
        # Check off-hours detection
        if 6 <= sample_security_context.timestamp.hour <= 22:
            assert context_dict["is_off_hours"] is False
        else:
            assert context_dict["is_off_hours"] is True
    
    def test_metrics_tracking(self, control_engine):
        """Test metrics tracking functionality."""
        initial_metrics = control_engine.get_metrics()
        engine_metrics = initial_metrics["deterministic_control_engine"]
        
        assert "decisions_made" in engine_metrics
        assert "allow_decisions" in engine_metrics
        assert "deny_decisions" in engine_metrics
        assert "cache_hits" in engine_metrics
        assert "avg_decision_time_ms" in engine_metrics
        
        # Check that metrics are numeric
        assert isinstance(engine_metrics["decisions_made"], int)
        assert isinstance(engine_metrics["avg_decision_time_ms"], float)


class TestAgentBehaviorMonitor:
    """Test agent behavior monitoring and anomaly detection."""
    
    @pytest.fixture
    def behavior_monitor(self):
        """Behavior monitor instance for testing."""
        return AgentBehaviorMonitor()
    
    @pytest.fixture
    def sample_agent_id(self):
        """Sample agent ID for testing."""
        return uuid.uuid4()
    
    @pytest.mark.asyncio
    async def test_behavior_profile_creation(self, behavior_monitor, sample_agent_id):
        """Test behavior profile creation and initialization."""
        # Update behavior for new agent
        profile = await behavior_monitor.update_agent_behavior(
            agent_id=sample_agent_id,
            action_type="data_access",
            success=True
        )
        
        assert profile.agent_id == sample_agent_id
        assert profile.behavior_state == AgentBehaviorState.NORMAL
        assert profile.risk_score >= 0.0
        assert profile.action_count_24h >= 1
        assert profile.failed_attempts_24h == 0
        assert isinstance(profile.last_updated, datetime)
    
    @pytest.mark.asyncio
    async def test_behavior_metrics_update(self, behavior_monitor, sample_agent_id):
        """Test behavior metrics updating."""
        # Simulate multiple actions
        for i in range(10):
            success = i < 8  # 80% success rate
            await behavior_monitor.update_agent_behavior(
                agent_id=sample_agent_id,
                action_type="code_execution",
                success=success
            )
        
        profile = behavior_monitor.get_agent_profile(sample_agent_id)
        assert profile is not None
        assert profile.action_count_24h == 10
        assert profile.failed_attempts_24h == 2  # 20% failures
        assert profile.code_executions_24h == 10
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_high_failure_rate(self, behavior_monitor, sample_agent_id):
        """Test anomaly detection for high failure rates."""
        # Simulate high failure rate (60%)
        for i in range(20):
            success = i < 8  # 40% success rate (high failure)
            await behavior_monitor.update_agent_behavior(
                agent_id=sample_agent_id,
                action_type="data_access",
                success=success
            )
        
        profile = behavior_monitor.get_agent_profile(sample_agent_id)
        
        # Should detect high failure rate as anomalous
        assert profile.anomaly_score > 0.0
        assert profile.behavior_state in [AgentBehaviorState.SUSPICIOUS, AgentBehaviorState.ANOMALOUS]
    
    @pytest.mark.asyncio
    async def test_off_hours_activity_detection(self, behavior_monitor, sample_agent_id):
        """Test off-hours activity detection."""
        # Simulate off-hours activity (2 AM)
        off_hours_time = datetime.utcnow().replace(hour=2, minute=0, second=0)
        
        for i in range(10):
            await behavior_monitor.update_agent_behavior(
                agent_id=sample_agent_id,
                action_type="data_access",
                success=True,
                timestamp=off_hours_time
            )
        
        profile = behavior_monitor.get_agent_profile(sample_agent_id)
        
        # Should detect anomaly due to off-hours activity
        assert profile.anomaly_score > 0.0
    
    @pytest.mark.asyncio
    async def test_agent_quarantine(self, behavior_monitor, sample_agent_id):
        """Test agent quarantine functionality."""
        # Quarantine agent
        await behavior_monitor.quarantine_agent(
            agent_id=sample_agent_id,
            reason="Security violation detected",
            duration_hours=24
        )
        
        profile = behavior_monitor.get_agent_profile(sample_agent_id)
        
        assert profile.behavior_state == AgentBehaviorState.QUARANTINED
        assert profile.quarantine_reason == "Security violation detected"
        assert profile.quarantine_until is not None
        assert profile.risk_score == 1.0
        
        # Check quarantine expiration
        expected_expiry = datetime.utcnow() + timedelta(hours=24)
        time_diff = abs((profile.quarantine_until - expected_expiry).total_seconds())
        assert time_diff < 60  # Within 1 minute
    
    @pytest.mark.asyncio
    async def test_risk_score_calculation(self, behavior_monitor, sample_agent_id):
        """Test risk score calculation logic."""
        # Start with normal behavior
        profile = await behavior_monitor.update_agent_behavior(
            agent_id=sample_agent_id,
            action_type="data_access",
            success=True
        )
        initial_risk = profile.risk_score
        
        # Add some failures
        for i in range(5):
            await behavior_monitor.update_agent_behavior(
                agent_id=sample_agent_id,
                action_type="data_access",
                success=False
            )
        
        profile = behavior_monitor.get_agent_profile(sample_agent_id)
        
        # Risk score should increase due to failures
        assert profile.risk_score > initial_risk
        assert 0.0 <= profile.risk_score <= 1.0
    
    def test_monitoring_metrics(self, behavior_monitor, sample_agent_id):
        """Test monitoring metrics retrieval."""
        # Create some behavior profiles
        behavior_monitor.behavior_profiles[sample_agent_id] = AgentBehaviorProfile(
            agent_id=sample_agent_id,
            behavior_state=AgentBehaviorState.SUSPICIOUS,
            risk_score=0.7
        )
        
        another_agent_id = uuid.uuid4()
        behavior_monitor.behavior_profiles[another_agent_id] = AgentBehaviorProfile(
            agent_id=another_agent_id,
            behavior_state=AgentBehaviorState.NORMAL,
            risk_score=0.2
        )
        
        metrics = behavior_monitor.get_monitoring_metrics()
        
        assert "total_agents_monitored" in metrics
        assert "behavior_state_distribution" in metrics
        assert "risk_score_distribution" in metrics
        assert metrics["total_agents_monitored"] == 2
        assert "SUSPICIOUS" in metrics["behavior_state_distribution"]
        assert "NORMAL" in metrics["behavior_state_distribution"]


class TestEnhancedSecuritySafeguards:
    """Test the comprehensive enhanced security safeguards system."""
    
    @pytest.fixture
    async def security_safeguards(self):
        """Enhanced security safeguards instance for testing."""
        with patch('app.core.enhanced_security_safeguards.get_async_session') as mock_session, \
             patch('app.core.enhanced_security_safeguards.SecurityAuditSystem') as mock_audit, \
             patch('app.core.enhanced_security_safeguards.SecurityAnalyzer') as mock_analyzer:
            
            # Mock database session
            mock_db_session = AsyncMock()
            mock_session.return_value.__anext__ = AsyncMock(return_value=mock_db_session)
            
            # Mock audit system
            mock_audit_instance = AsyncMock()
            mock_audit.return_value = mock_audit_instance
            
            # Mock security analyzer
            mock_analyzer_instance = AsyncMock()
            mock_analyzer.return_value = mock_analyzer_instance
            
            safeguards = EnhancedSecuritySafeguards(
                db_session=mock_db_session,
                security_audit_system=mock_audit_instance,
                code_security_analyzer=mock_analyzer_instance
            )
            
            return safeguards
    
    @pytest.fixture
    def sample_agent_id(self):
        """Sample agent ID for testing."""
        return uuid.uuid4()
    
    @pytest.mark.asyncio
    async def test_validate_normal_agent_action(self, security_safeguards, sample_agent_id):
        """Test validation of normal agent action."""
        decision, reason, additional_data = await security_safeguards.validate_agent_action(
            agent_id=sample_agent_id,
            action_type="data_access",
            resource_type="context",
            resource_id="ctx_123",
            metadata={"data_sensitivity": "normal"}
        )
        
        assert decision in [ControlDecision.ALLOW, ControlDecision.REQUIRE_APPROVAL]
        assert isinstance(reason, str)
        assert "behavior_profile" in additional_data
        assert "applied_rules" in additional_data
    
    @pytest.mark.asyncio
    async def test_validate_high_risk_agent_action(self, security_safeguards, sample_agent_id):
        """Test validation of high risk agent action."""
        # First, create some high-risk behavior
        for i in range(15):
            await security_safeguards.behavior_monitor.update_agent_behavior(
                agent_id=sample_agent_id,
                action_type="data_access",
                success=i < 5  # High failure rate
            )
        
        decision, reason, additional_data = await security_safeguards.validate_agent_action(
            agent_id=sample_agent_id,
            action_type="data_access",
            resource_type="sensitive_data",
            metadata={"data_sensitivity": "confidential"}
        )
        
        # High risk agent accessing confidential data should require approval or be denied
        assert decision in [ControlDecision.REQUIRE_APPROVAL, ControlDecision.DENY, ControlDecision.ESCALATE]
        assert additional_data["behavior_profile"]["risk_score"] > 0.0
    
    @pytest.mark.asyncio
    async def test_validate_code_execution_safe(self, security_safeguards, sample_agent_id):
        """Test validation of safe code execution."""
        # Mock security analyzer to return safe code
        mock_security_result = Mock()
        mock_security_result.security_level = SecurityLevel.SAFE
        mock_security_result.safe_to_execute = True
        mock_security_result.threats_detected = []
        mock_security_result.warnings = []
        mock_security_result.has_file_operations = False
        mock_security_result.has_network_access = False
        mock_security_result.has_system_calls = False
        mock_security_result.has_dangerous_imports = False
        mock_security_result.has_exec_statements = False
        mock_security_result.confidence_score = 0.9
        
        security_safeguards.code_analyzer.analyze_security = AsyncMock(return_value=mock_security_result)
        
        code_block = CodeBlock(
            id="test_code_123",
            language=CodeLanguage.PYTHON,
            content="print('Hello, World!')",
            description="Simple print statement"
        )
        
        decision, reason, additional_data = await security_safeguards.validate_code_execution(
            agent_id=sample_agent_id,
            code_block=code_block
        )
        
        assert decision == ControlDecision.ALLOW
        assert "security_analysis" in additional_data
        assert additional_data["code_block_id"] == "test_code_123"
    
    @pytest.mark.asyncio
    async def test_validate_code_execution_dangerous(self, security_safeguards, sample_agent_id):
        """Test validation of dangerous code execution."""
        # Mock security analyzer to return dangerous code
        mock_security_result = Mock()
        mock_security_result.security_level = SecurityLevel.DANGEROUS
        mock_security_result.safe_to_execute = False
        mock_security_result.threats_detected = ["Dangerous system calls detected"]
        mock_security_result.warnings = ["File deletion operations found"]
        mock_security_result.has_file_operations = True
        mock_security_result.has_network_access = False
        mock_security_result.has_system_calls = True
        mock_security_result.has_dangerous_imports = True
        mock_security_result.has_exec_statements = True
        mock_security_result.confidence_score = 0.95
        
        security_safeguards.code_analyzer.analyze_security = AsyncMock(return_value=mock_security_result)
        
        code_block = CodeBlock(
            id="dangerous_code_456",
            language=CodeLanguage.PYTHON,
            content="import os; os.system('rm -rf /')",
            description="Dangerous system command"
        )
        
        decision, reason, additional_data = await security_safeguards.validate_code_execution(
            agent_id=sample_agent_id,
            code_block=code_block
        )
        
        assert decision == ControlDecision.DENY
        assert "security_analysis" in additional_data
        assert additional_data["security_analysis"].security_level == SecurityLevel.DANGEROUS
    
    @pytest.mark.asyncio
    async def test_agent_quarantine_handling(self, security_safeguards, sample_agent_id):
        """Test agent quarantine handling."""
        # Create behavior that triggers quarantine
        profile = AgentBehaviorProfile(
            agent_id=sample_agent_id,
            behavior_state=AgentBehaviorState.COMPROMISED,
            risk_score=0.95
        )
        security_safeguards.behavior_monitor.behavior_profiles[sample_agent_id] = profile
        
        decision, reason, additional_data = await security_safeguards.validate_agent_action(
            agent_id=sample_agent_id,
            action_type="data_access",
            resource_type="sensitive_data"
        )
        
        assert decision == ControlDecision.QUARANTINE
        assert "quarantine_duration_hours" in additional_data
        assert additional_data["quarantine_duration_hours"] == 24
        
        # Verify agent is quarantined
        updated_profile = security_safeguards.behavior_monitor.get_agent_profile(sample_agent_id)
        assert updated_profile.behavior_state == AgentBehaviorState.QUARANTINED
    
    @pytest.mark.asyncio
    async def test_get_agent_security_status(self, security_safeguards, sample_agent_id):
        """Test getting agent security status."""
        # Create some behavior data
        await security_safeguards.behavior_monitor.update_agent_behavior(
            agent_id=sample_agent_id,
            action_type="data_access",
            success=True
        )
        
        status = await security_safeguards.get_agent_security_status(sample_agent_id)
        
        assert "agent_id" in status
        assert "behavior_state" in status
        assert "risk_score" in status
        assert "anomaly_score" in status
        assert "metrics_24h" in status
        assert "security_status" in status
        
        assert status["agent_id"] == str(sample_agent_id)
        assert status["security_status"] == "normal"
        assert 0.0 <= status["risk_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_get_agent_security_status_unknown(self, security_safeguards):
        """Test getting security status for unknown agent."""
        unknown_agent_id = uuid.uuid4()
        
        status = await security_safeguards.get_agent_security_status(unknown_agent_id)
        
        assert status["status"] == "unknown"
        assert "No behavior profile found" in status["message"]
    
    def test_comprehensive_metrics(self, security_safeguards):
        """Test comprehensive metrics retrieval."""
        metrics = security_safeguards.get_comprehensive_metrics()
        
        assert "enhanced_security_safeguards" in metrics
        assert "deterministic_control_engine" in metrics
        assert "behavior_monitoring" in metrics
        assert "configuration" in metrics
        
        # Check that metrics contain expected keys
        safeguard_metrics = metrics["enhanced_security_safeguards"]
        assert "security_checks_performed" in safeguard_metrics
        assert "decisions_made" in safeguard_metrics
        assert "threats_blocked" in safeguard_metrics
        
        engine_metrics = metrics["deterministic_control_engine"]
        assert "decisions_made" in engine_metrics
        assert "rules_count" in engine_metrics
        
        monitoring_metrics = metrics["behavior_monitoring"]
        assert "total_agents_monitored" in monitoring_metrics


class TestConvenienceFunctions:
    """Test convenience functions for common operations."""
    
    @pytest.mark.asyncio
    async def test_validate_agent_action_convenience(self):
        """Test convenience function for agent action validation."""
        with patch('app.core.enhanced_security_safeguards.get_enhanced_security_safeguards') as mock_get:
            mock_safeguards = AsyncMock()
            mock_safeguards.validate_agent_action = AsyncMock(
                return_value=(ControlDecision.ALLOW, "Test reason", {"test": "data"})
            )
            mock_get.return_value = mock_safeguards
            
            result = await validate_agent_action(
                agent_id=uuid.uuid4(),
                action_type="test_action",
                resource_type="test_resource"
            )
            
            assert result == (ControlDecision.ALLOW, "Test reason", {"test": "data"})
            mock_safeguards.validate_agent_action.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_code_execution_convenience(self):
        """Test convenience function for code execution validation."""
        with patch('app.core.enhanced_security_safeguards.get_enhanced_security_safeguards') as mock_get:
            mock_safeguards = AsyncMock()
            mock_safeguards.validate_code_execution = AsyncMock(
                return_value=(ControlDecision.DENY, "Dangerous code", {"analysis": "complete"})
            )
            mock_get.return_value = mock_safeguards
            
            code_block = CodeBlock(
                id="test_code",
                language=CodeLanguage.PYTHON,
                content="test code",
                description="test"
            )
            
            result = await validate_code_execution(
                agent_id=uuid.uuid4(),
                code_block=code_block
            )
            
            assert result == (ControlDecision.DENY, "Dangerous code", {"analysis": "complete"})
            mock_safeguards.validate_code_execution.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_agent_security_status_convenience(self):
        """Test convenience function for getting agent security status."""
        with patch('app.core.enhanced_security_safeguards.get_enhanced_security_safeguards') as mock_get:
            mock_safeguards = AsyncMock()
            mock_safeguards.get_agent_security_status = AsyncMock(
                return_value={"status": "normal", "risk_score": 0.2}
            )
            mock_get.return_value = mock_safeguards
            
            agent_id = uuid.uuid4()
            result = await get_agent_security_status(agent_id)
            
            assert result == {"status": "normal", "risk_score": 0.2}
            mock_safeguards.get_agent_security_status.assert_called_once_with(agent_id)


class TestIntegrationAndPerformance:
    """Test integration scenarios and performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_validations(self):
        """Test concurrent agent validations."""
        with patch('app.core.enhanced_security_safeguards.get_async_session'), \
             patch('app.core.enhanced_security_safeguards.SecurityAuditSystem'), \
             patch('app.core.enhanced_security_safeguards.SecurityAnalyzer'):
            
            safeguards = EnhancedSecuritySafeguards(
                db_session=AsyncMock(),
                security_audit_system=AsyncMock(),
                code_security_analyzer=AsyncMock()
            )
            
            # Create multiple concurrent validation tasks
            agent_ids = [uuid.uuid4() for _ in range(10)]
            tasks = [
                safeguards.validate_agent_action(
                    agent_id=agent_id,
                    action_type="data_access",
                    resource_type="context",
                    metadata={"test": f"agent_{i}"}
                )
                for i, agent_id in enumerate(agent_ids)
            ]
            
            # Execute concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should succeed
            for result in results:
                assert isinstance(result, tuple)
                assert len(result) == 3
                decision, reason, data = result
                assert isinstance(decision, ControlDecision)
                assert isinstance(reason, str)
                assert isinstance(data, dict)
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under load."""
        with patch('app.core.enhanced_security_safeguards.get_async_session'), \
             patch('app.core.enhanced_security_safeguards.SecurityAuditSystem'), \
             patch('app.core.enhanced_security_safeguards.SecurityAnalyzer'):
            
            safeguards = EnhancedSecuritySafeguards(
                db_session=AsyncMock(),
                security_audit_system=AsyncMock(),
                code_security_analyzer=AsyncMock()
            )
            
            start_time = datetime.utcnow()
            
            # Simulate high load
            for i in range(100):
                await safeguards.validate_agent_action(
                    agent_id=uuid.uuid4(),
                    action_type="data_access",
                    resource_type="context"
                )
            
            end_time = datetime.utcnow()
            total_time = (end_time - start_time).total_seconds()
            
            # Should complete 100 validations in reasonable time
            assert total_time < 10.0  # Less than 10 seconds
            
            # Check performance metrics
            metrics = safeguards.get_comprehensive_metrics()
            engine_metrics = metrics["deterministic_control_engine"]
            assert engine_metrics["avg_decision_time_ms"] < 100  # Less than 100ms average
    
    @pytest.mark.asyncio
    async def test_memory_usage_efficiency(self):
        """Test memory usage efficiency with many agents."""
        behavior_monitor = AgentBehaviorMonitor()
        
        # Create many agents
        agent_ids = [uuid.uuid4() for _ in range(1000)]
        
        for agent_id in agent_ids:
            await behavior_monitor.update_agent_behavior(
                agent_id=agent_id,
                action_type="data_access",
                success=True
            )
        
        # Check that all profiles are created
        assert len(behavior_monitor.behavior_profiles) == 1000
        
        # Check that behavior history is properly limited
        for agent_id in agent_ids:
            history = behavior_monitor.behavior_history[agent_id]
            assert len(history) <= 1000  # Should respect maxlen
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and system recovery."""
        with patch('app.core.enhanced_security_safeguards.get_async_session'), \
             patch('app.core.enhanced_security_safeguards.SecurityAuditSystem'), \
             patch('app.core.enhanced_security_safeguards.SecurityAnalyzer'):
            
            safeguards = EnhancedSecuritySafeguards(
                db_session=AsyncMock(),
                security_audit_system=AsyncMock(),
                code_security_analyzer=AsyncMock()
            )
            
            # Simulate error in behavior monitoring
            with patch.object(safeguards.behavior_monitor, 'update_agent_behavior', side_effect=Exception("Test error")):
                
                decision, reason, data = await safeguards.validate_agent_action(
                    agent_id=uuid.uuid4(),
                    action_type="data_access",
                    resource_type="context"
                )
                
                # Should default to DENY on error
                assert decision == ControlDecision.DENY
                assert "error" in reason.lower()
    
    def test_rule_priority_enforcement(self):
        """Test that rules are evaluated in priority order."""
        engine = DeterministicControlEngine()
        
        # Add rules with different priorities
        high_priority_rule = SecurityRule(
            id="high_priority",
            name="High Priority Rule",
            policy_type=SecurityPolicyType.DATA_ACCESS,
            conditions={"test_condition": True},
            decision=ControlDecision.DENY,
            priority=1000
        )
        
        low_priority_rule = SecurityRule(
            id="low_priority",
            name="Low Priority Rule",
            policy_type=SecurityPolicyType.DATA_ACCESS,
            conditions={"test_condition": True},
            decision=ControlDecision.ALLOW,
            priority=100
        )
        
        engine.add_rule(high_priority_rule)
        engine.add_rule(low_priority_rule)
        
        # Check that rules are sorted by priority
        assert engine.security_rules[0].priority >= engine.security_rules[1].priority
        
        # High priority rule should be first
        high_priority_index = next(
            i for i, rule in enumerate(engine.security_rules) 
            if rule.id == "high_priority"
        )
        low_priority_index = next(
            i for i, rule in enumerate(engine.security_rules) 
            if rule.id == "low_priority"
        )
        
        assert high_priority_index < low_priority_index