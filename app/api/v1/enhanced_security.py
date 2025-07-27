"""
Enhanced Security API with Deterministic Control and Safeguards.

Provides comprehensive security endpoints for:
- Agent action validation and authorization
- Code execution security validation  
- Agent behavior monitoring and status
- Security policy management
- Real-time security metrics and alerts
- Incident response and investigation
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import structlog
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.enhanced_security_safeguards import (
    get_enhanced_security_safeguards,
    validate_agent_action,
    validate_code_execution,
    get_agent_security_status,
    ControlDecision,
    AgentBehaviorState,
    SecurityPolicyType,
    SecurityRule
)
from ...core.code_execution import CodeBlock, CodeLanguage
from ...core.database import get_async_session
from ...schemas.security import SecurityValidationRequest, SecurityValidationResponse
from ...core.config import get_settings

logger = structlog.get_logger()

router = APIRouter(prefix="/enhanced-security", tags=["Enhanced Security"])


# Request/Response Schemas
class AgentActionValidationRequest(BaseModel):
    """Request schema for agent action validation."""
    agent_id: str
    action_type: str
    resource_type: str
    resource_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CodeExecutionValidationRequest(BaseModel):
    """Request schema for code execution validation."""
    agent_id: str
    code_content: str
    code_language: str
    description: Optional[str] = None
    session_id: Optional[str] = None


class SecurityRuleRequest(BaseModel):
    """Request schema for security rule management."""
    id: str
    name: str
    policy_type: str
    conditions: Dict[str, Any]
    decision: str
    priority: int
    enabled: bool = True
    metadata: Optional[Dict[str, Any]] = None


class SecurityValidationResponse(BaseModel):
    """Response schema for security validation."""
    decision: str
    reason: str
    risk_score: float
    behavior_state: str
    additional_data: Dict[str, Any]
    timestamp: str


class AgentSecurityStatusResponse(BaseModel):
    """Response schema for agent security status."""
    agent_id: str
    behavior_state: str
    risk_score: float
    anomaly_score: float
    security_status: str
    last_updated: str
    metrics_24h: Dict[str, int]
    quarantine_info: Optional[Dict[str, Any]] = None


@router.post("/validate/agent-action", response_model=SecurityValidationResponse)
async def validate_agent_action_endpoint(
    request: AgentActionValidationRequest
) -> SecurityValidationResponse:
    """
    Validate agent action through enhanced security safeguards.
    
    Performs comprehensive security validation including:
    - Risk assessment based on agent behavior
    - Policy compliance checking
    - Deterministic control decision making
    """
    try:
        # Convert string UUIDs
        agent_id = uuid.UUID(request.agent_id)
        session_id = uuid.UUID(request.session_id) if request.session_id else None
        
        # Validate the action
        decision, reason, additional_data = await validate_agent_action(
            agent_id=agent_id,
            action_type=request.action_type,
            resource_type=request.resource_type,
            resource_id=request.resource_id,
            session_id=session_id,
            metadata=request.metadata or {}
        )
        
        # Extract behavior information
        behavior_profile = additional_data.get("behavior_profile", {})
        
        return SecurityValidationResponse(
            decision=decision.value,
            reason=reason,
            risk_score=behavior_profile.get("risk_score", 0.0),
            behavior_state=behavior_profile.get("state", "UNKNOWN"),
            additional_data=additional_data,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid UUID format: {e}")
    except Exception as e:
        logger.error(f"Agent action validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/code-execution", response_model=SecurityValidationResponse)
async def validate_code_execution_endpoint(
    request: CodeExecutionValidationRequest
) -> SecurityValidationResponse:
    """
    Validate code execution through enhanced security analysis.
    
    Performs multi-layer security analysis including:
    - Static code analysis for dangerous patterns
    - AI-powered security assessment
    - Agent behavior risk evaluation
    - Policy compliance verification
    """
    try:
        # Convert string UUIDs
        agent_id = uuid.UUID(request.agent_id)
        session_id = uuid.UUID(request.session_id) if request.session_id else None
        
        # Create code block
        code_block = CodeBlock(
            id=str(uuid.uuid4()),
            language=CodeLanguage(request.code_language),
            content=request.code_content,
            description=request.description or "Code execution request",
            agent_id=str(agent_id)
        )
        
        # Validate code execution
        decision, reason, additional_data = await validate_code_execution(
            agent_id=agent_id,
            code_block=code_block,
            session_id=session_id
        )
        
        # Extract security analysis
        security_analysis = additional_data.get("security_analysis")
        behavior_profile = additional_data.get("behavior_profile", {})
        
        # Build enhanced response data
        enhanced_data = {
            **additional_data,
            "security_analysis_summary": {
                "security_level": security_analysis.security_level.value if security_analysis else "unknown",
                "threats_detected": security_analysis.threats_detected if security_analysis else [],
                "safe_to_execute": security_analysis.safe_to_execute if security_analysis else False,
                "confidence_score": security_analysis.confidence_score if security_analysis else 0.0
            }
        }
        
        return SecurityValidationResponse(
            decision=decision.value,
            reason=reason,
            risk_score=behavior_profile.get("risk_score", 0.0),
            behavior_state=behavior_profile.get("state", "UNKNOWN"),
            additional_data=enhanced_data,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except Exception as e:
        logger.error(f"Code execution validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent/{agent_id}/status", response_model=AgentSecurityStatusResponse)
async def get_agent_security_status_endpoint(
    agent_id: str
) -> AgentSecurityStatusResponse:
    """
    Get comprehensive security status for a specific agent.
    
    Returns:
    - Current behavior state and risk assessment
    - 24-hour activity metrics
    - Quarantine status if applicable
    - Security recommendations
    """
    try:
        agent_uuid = uuid.UUID(agent_id)
        
        status = await get_agent_security_status(agent_uuid)
        
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["message"])
        
        # Build quarantine info if applicable
        quarantine_info = None
        if status.get("security_status") == "quarantined":
            quarantine_info = {
                "reason": status.get("quarantine_reason"),
                "until": status.get("quarantine_until"),
                "is_active": True
            }
        
        return AgentSecurityStatusResponse(
            agent_id=status["agent_id"],
            behavior_state=status["behavior_state"],
            risk_score=status["risk_score"],
            anomaly_score=status["anomaly_score"],
            security_status=status["security_status"],
            last_updated=status["last_updated"],
            metrics_24h=status["metrics_24h"],
            quarantine_info=quarantine_info
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid agent ID: {e}")
    except Exception as e:
        logger.error(f"Get agent status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/overview")
async def get_agents_security_overview(
    limit: int = Query(default=50, le=200),
    behavior_state: Optional[str] = Query(default=None),
    min_risk_score: Optional[float] = Query(default=None, ge=0.0, le=1.0)
) -> JSONResponse:
    """
    Get security overview for multiple agents with filtering.
    
    Provides high-level security metrics and alerts across agents.
    """
    try:
        safeguards = await get_enhanced_security_safeguards()
        
        # Get all agent profiles
        all_profiles = safeguards.behavior_monitor.behavior_profiles
        
        # Apply filters
        filtered_profiles = []
        for agent_id, profile in all_profiles.items():
            
            # Filter by behavior state
            if behavior_state and profile.behavior_state.value != behavior_state:
                continue
                
            # Filter by minimum risk score
            if min_risk_score is not None and profile.risk_score < min_risk_score:
                continue
                
            filtered_profiles.append({
                "agent_id": str(agent_id),
                "behavior_state": profile.behavior_state.value,
                "risk_score": profile.risk_score,
                "anomaly_score": profile.anomaly_score,
                "action_count_24h": profile.action_count_24h,
                "failed_attempts_24h": profile.failed_attempts_24h,
                "last_updated": profile.last_updated.isoformat()
            })
        
        # Sort by risk score (descending) and limit
        filtered_profiles.sort(key=lambda x: x["risk_score"], reverse=True)
        filtered_profiles = filtered_profiles[:limit]
        
        # Calculate summary statistics
        total_agents = len(all_profiles)
        high_risk_agents = sum(1 for p in all_profiles.values() if p.risk_score >= 0.7)
        quarantined_agents = sum(1 for p in all_profiles.values() if p.behavior_state == AgentBehaviorState.QUARANTINED)
        
        return JSONResponse({
            "summary": {
                "total_agents": total_agents,
                "high_risk_agents": high_risk_agents,
                "quarantined_agents": quarantined_agents,
                "filters_applied": {
                    "behavior_state": behavior_state,
                    "min_risk_score": min_risk_score,
                    "limit": limit
                }
            },
            "agents": filtered_profiles,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Get agents overview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/{agent_id}/quarantine")
async def quarantine_agent_endpoint(
    agent_id: str,
    duration_hours: int = Body(default=24, ge=1, le=168),  # 1 hour to 1 week
    reason: str = Body(..., min_length=10, max_length=500)
) -> JSONResponse:
    """
    Manually quarantine an agent for security reasons.
    
    Requires human authorization for quarantine actions.
    """
    try:
        agent_uuid = uuid.UUID(agent_id)
        
        safeguards = await get_enhanced_security_safeguards()
        
        # Quarantine the agent
        await safeguards.behavior_monitor.quarantine_agent(
            agent_id=agent_uuid,
            reason=reason,
            duration_hours=duration_hours
        )
        
        logger.warning(
            f"Agent manually quarantined",
            agent_id=agent_id,
            reason=reason,
            duration_hours=duration_hours
        )
        
        return JSONResponse({
            "success": True,
            "message": f"Agent {agent_id} quarantined for {duration_hours} hours",
            "quarantine_until": (datetime.utcnow() + timedelta(hours=duration_hours)).isoformat(),
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid agent ID: {e}")
    except Exception as e:
        logger.error(f"Agent quarantine failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/agent/{agent_id}/quarantine")
async def release_agent_quarantine_endpoint(
    agent_id: str,
    reason: str = Body(..., min_length=10, max_length=500)
) -> JSONResponse:
    """
    Release an agent from quarantine early.
    
    Requires human authorization and justification.
    """
    try:
        agent_uuid = uuid.UUID(agent_id)
        
        safeguards = await get_enhanced_security_safeguards()
        
        # Get agent profile
        profile = safeguards.behavior_monitor.get_agent_profile(agent_uuid)
        if not profile:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        if profile.behavior_state != AgentBehaviorState.QUARANTINED:
            raise HTTPException(status_code=400, detail="Agent is not quarantined")
        
        # Release from quarantine
        profile.behavior_state = AgentBehaviorState.NORMAL
        profile.quarantine_reason = None
        profile.quarantine_until = None
        profile.risk_score = min(profile.risk_score, 0.5)  # Reduce risk score
        
        logger.info(
            f"Agent released from quarantine",
            agent_id=agent_id,
            reason=reason
        )
        
        return JSONResponse({
            "success": True,
            "message": f"Agent {agent_id} released from quarantine",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid agent ID: {e}")
    except Exception as e:
        logger.error(f"Agent quarantine release failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security-rules")
async def list_security_rules() -> JSONResponse:
    """
    List all active security rules in the deterministic control engine.
    
    Returns rules sorted by priority for policy review.
    """
    try:
        safeguards = await get_enhanced_security_safeguards()
        
        rules_data = []
        for rule in safeguards.control_engine.security_rules:
            rules_data.append({
                "id": rule.id,
                "name": rule.name,
                "policy_type": rule.policy_type.value,
                "decision": rule.decision.value,
                "priority": rule.priority,
                "enabled": rule.enabled,
                "conditions": rule.conditions,
                "metadata": rule.metadata
            })
        
        return JSONResponse({
            "rules": rules_data,
            "total_count": len(rules_data),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"List security rules failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/security-rules")
async def create_security_rule(
    rule_request: SecurityRuleRequest
) -> JSONResponse:
    """
    Create a new security rule for the deterministic control engine.
    
    Requires careful validation of rule logic and conditions.
    """
    try:
        safeguards = await get_enhanced_security_safeguards()
        
        # Validate policy type and decision
        try:
            policy_type = SecurityPolicyType(rule_request.policy_type)
            decision = ControlDecision(rule_request.decision)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid enum value: {e}")
        
        # Create security rule
        security_rule = SecurityRule(
            id=rule_request.id,
            name=rule_request.name,
            policy_type=policy_type,
            conditions=rule_request.conditions,
            decision=decision,
            priority=rule_request.priority,
            enabled=rule_request.enabled,
            metadata=rule_request.metadata or {}
        )
        
        # Add to engine
        safeguards.control_engine.add_rule(security_rule)
        
        logger.info(
            f"Security rule created",
            rule_id=rule_request.id,
            rule_name=rule_request.name,
            priority=rule_request.priority
        )
        
        return JSONResponse({
            "success": True,
            "message": f"Security rule '{rule_request.name}' created",
            "rule_id": rule_request.id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Create security rule failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/security-rules/{rule_id}")
async def delete_security_rule(rule_id: str) -> JSONResponse:
    """
    Delete a security rule from the deterministic control engine.
    
    Requires careful consideration of policy impact.
    """
    try:
        safeguards = await get_enhanced_security_safeguards()
        
        # Remove rule
        removed = safeguards.control_engine.remove_rule(rule_id)
        
        if not removed:
            raise HTTPException(status_code=404, detail="Security rule not found")
        
        logger.info(
            f"Security rule deleted",
            rule_id=rule_id
        )
        
        return JSONResponse({
            "success": True,
            "message": f"Security rule '{rule_id}' deleted",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Delete security rule failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_security_metrics() -> JSONResponse:
    """
    Get comprehensive security metrics and performance data.
    
    Provides insights into system security posture and performance.
    """
    try:
        safeguards = await get_enhanced_security_safeguards()
        
        metrics = safeguards.get_comprehensive_metrics()
        
        # Add timestamp and system info
        metrics["timestamp"] = datetime.utcnow().isoformat()
        metrics["system_info"] = {
            "deterministic_control_enabled": safeguards.config["enable_deterministic_control"],
            "behavioral_analysis_enabled": safeguards.config["enable_behavioral_analysis"],
            "real_time_monitoring_enabled": safeguards.config["enable_real_time_monitoring"]
        }
        
        return JSONResponse(metrics)
        
    except Exception as e:
        logger.error(f"Get security metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_security_health() -> JSONResponse:
    """
    Get security system health status.
    
    Returns overall health and component status for monitoring.
    """
    try:
        safeguards = await get_enhanced_security_safeguards()
        
        # Get metrics for health assessment
        metrics = safeguards.get_comprehensive_metrics()
        
        # Assess health status
        health_status = "healthy"
        health_checks = {}
        
        # Check deterministic control engine
        engine_metrics = metrics["deterministic_control_engine"]
        if engine_metrics["avg_decision_time_ms"] > 500:  # Slow decisions
            health_status = "degraded"
            health_checks["decision_performance"] = "degraded"
        else:
            health_checks["decision_performance"] = "healthy"
        
        # Check behavior monitoring
        monitoring_metrics = metrics["behavior_monitoring"]
        if monitoring_metrics["total_agents_monitored"] == 0:
            health_checks["behavior_monitoring"] = "no_data"
        else:
            health_checks["behavior_monitoring"] = "healthy"
        
        # Check safeguards performance
        safeguard_metrics = metrics["enhanced_security_safeguards"]
        if safeguard_metrics["avg_check_time_ms"] > 1000:  # Very slow checks
            health_status = "degraded"
            health_checks["safeguards_performance"] = "degraded"
        else:
            health_checks["safeguards_performance"] = "healthy"
        
        # Overall health determination
        if any(status == "degraded" for status in health_checks.values()):
            health_status = "degraded"
        
        return JSONResponse({
            "status": health_status,
            "timestamp": datetime.utcnow().isoformat(),
            "health_checks": health_checks,
            "metrics_summary": {
                "total_decisions": engine_metrics["decisions_made"],
                "total_agents_monitored": monitoring_metrics["total_agents_monitored"],
                "avg_decision_time_ms": engine_metrics["avg_decision_time_ms"],
                "threats_blocked": safeguard_metrics["threats_blocked"]
            }
        })
        
    except Exception as e:
        logger.error(f"Security health check failed: {e}")
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }, status_code=500)


@router.post("/test/trigger-security-event")
async def trigger_test_security_event(
    agent_id: str,
    event_type: str = Body(...),
    severity: str = Body(default="info"),
    metadata: Optional[Dict[str, Any]] = Body(default=None)
) -> JSONResponse:
    """
    Trigger a test security event for testing and development.
    
    Allows manual testing of security event processing and alerting.
    """
    try:
        agent_uuid = uuid.UUID(agent_id)
        
        safeguards = await get_enhanced_security_safeguards()
        
        # Update agent behavior to simulate the event
        await safeguards.behavior_monitor.update_agent_behavior(
            agent_id=agent_uuid,
            action_type=event_type,
            success=severity not in ["error", "critical"],
            metadata={
                **(metadata or {}),
                "test_event": True,
                "severity": severity,
                "triggered_manually": True
            }
        )
        
        logger.info(
            f"Test security event triggered",
            agent_id=agent_id,
            event_type=event_type,
            severity=severity
        )
        
        return JSONResponse({
            "success": True,
            "message": f"Test security event '{event_type}' triggered for agent {agent_id}",
            "event_type": event_type,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid agent ID: {e}")
    except Exception as e:
        logger.error(f"Trigger test security event failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))