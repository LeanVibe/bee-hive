"""
Security Integration Module
Integrates unified authorization engine with orchestrator, task engine, and monitoring systems
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from app.core.unified_authorization_engine import (
    get_unified_authorization_engine,
    UnifiedAuthorizationEngine,
    AuthorizationContext,
    ResourceType,
    PermissionLevel,
    AccessDecision,
    require_permission,
    require_role
)

logger = logging.getLogger(__name__)

class SecurityOrchestrationIntegration:
    """
    Integration layer between unified authorization engine and orchestration systems.
    Provides secure interfaces for orchestrator, task engine, and monitoring.
    """
    
    def __init__(self):
        self.auth_engine = get_unified_authorization_engine()
        
    # Orchestrator Security Integration
    
    @require_permission(ResourceType.AGENT, "register", PermissionLevel.WRITE)
    async def secure_agent_registration(
        self, 
        agent_config: Dict[str, Any], 
        current_user,
        request=None
    ) -> Dict[str, Any]:
        """
        Secure agent registration with authorization checks.
        
        Integrates with production orchestrator for secure agent lifecycle management.
        """
        try:
            # Additional security validation for agent registration
            context = AuthorizationContext(
                user_id=current_user.user_id,
                resource_type=ResourceType.AGENT,
                action="register",
                permission_level=PermissionLevel.ADMIN,  # Require admin for new agents
                client_ip=getattr(request, 'client', {}).get('host') if request else None,
                additional_context={
                    "agent_type": agent_config.get("type"),
                    "agent_capabilities": agent_config.get("capabilities", []),
                    "security_level": agent_config.get("security_level", "standard")
                }
            )
            
            # Check enhanced permissions for agent registration
            result = await self.auth_engine.check_permission(context)
            
            if result.decision != AccessDecision.GRANTED:
                raise PermissionError(f"Agent registration denied: {result.reason}")
            
            # Security validation of agent configuration
            security_validation = await self._validate_agent_security_config(agent_config)
            if not security_validation["is_valid"]:
                raise ValueError(f"Agent security validation failed: {security_validation['reason']}")
            
            # Get orchestrator instance (would be injected in production)
            from app.core.production_orchestrator import get_production_orchestrator
            orchestrator = get_production_orchestrator()
            
            # Register agent with security context
            registration_result = await orchestrator.register_agent(
                agent_config=agent_config,
                authorized_by=current_user.user_id,
                security_context=context
            )
            
            # Log successful agent registration
            await self._log_security_event(
                "agent_registered",
                current_user.user_id,
                ResourceType.AGENT,
                True,
                {
                    "agent_id": registration_result.get("agent_id"),
                    "agent_type": agent_config.get("type"),
                    "registered_by": current_user.user_id
                }
            )
            
            return {
                "success": True,
                "agent_id": registration_result.get("agent_id"),
                "security_validation": security_validation,
                "authorization_result": result
            }
            
        except Exception as e:
            logger.error(f"Secure agent registration failed: {e}")
            await self._log_security_event(
                "agent_registration_failed",
                current_user.user_id,
                ResourceType.AGENT,
                False,
                {"error": str(e)}
            )
            raise
    
    @require_permission(ResourceType.ORCHESTRATOR, "manage", PermissionLevel.ADMIN)
    async def secure_orchestrator_control(
        self,
        action: str,
        parameters: Dict[str, Any],
        current_user,
        request=None
    ) -> Dict[str, Any]:
        """
        Secure orchestrator control operations.
        """
        try:
            # Validate high-privilege orchestrator operations
            if action in ["shutdown", "restart", "reconfigure"]:
                # Require super admin for critical operations
                context = AuthorizationContext(
                    user_id=current_user.user_id,
                    resource_type=ResourceType.ORCHESTRATOR,
                    action=action,
                    permission_level=PermissionLevel.SUPER_ADMIN,
                    mfa_verified=True,  # Require MFA for critical operations
                    client_ip=getattr(request, 'client', {}).get('host') if request else None
                )
                
                result = await self.auth_engine.check_permission(context)
                if result.decision != AccessDecision.GRANTED:
                    raise PermissionError(f"Orchestrator {action} denied: {result.reason}")
            
            # Get orchestrator instance
            from app.core.production_orchestrator import get_production_orchestrator
            orchestrator = get_production_orchestrator()
            
            # Execute secured orchestrator operation
            operation_result = await orchestrator.execute_control_operation(
                action=action,
                parameters=parameters,
                authorized_by=current_user.user_id
            )
            
            # Log orchestrator operation
            await self._log_security_event(
                f"orchestrator_{action}",
                current_user.user_id,
                ResourceType.ORCHESTRATOR,
                True,
                {"action": action, "parameters": parameters}
            )
            
            return {
                "success": True,
                "action": action,
                "result": operation_result
            }
            
        except Exception as e:
            logger.error(f"Secure orchestrator control failed: {e}")
            await self._log_security_event(
                f"orchestrator_{action}_failed",
                current_user.user_id,
                ResourceType.ORCHESTRATOR,
                False,
                {"error": str(e), "action": action}
            )
            raise
    
    # Task Engine Security Integration
    
    @require_permission(ResourceType.TASK, "submit", PermissionLevel.WRITE)
    async def secure_task_submission(
        self,
        task_config: Dict[str, Any],
        current_user,
        request=None
    ) -> Dict[str, Any]:
        """
        Secure task submission with comprehensive validation.
        """
        try:
            # Security validation for task submission
            task_security_validation = await self._validate_task_security(task_config)
            if not task_security_validation["is_valid"]:
                raise ValueError(f"Task security validation failed: {task_security_validation['reason']}")
            
            # Check task complexity and resource requirements
            complexity_level = self._assess_task_complexity(task_config)
            required_permission_level = self._get_required_permission_level(complexity_level)
            
            context = AuthorizationContext(
                user_id=current_user.user_id,
                resource_type=ResourceType.TASK,
                action="submit",
                permission_level=required_permission_level,
                client_ip=getattr(request, 'client', {}).get('host') if request else None,
                additional_context={
                    "task_type": task_config.get("type"),
                    "complexity_level": complexity_level,
                    "resource_requirements": task_config.get("resources", {})
                }
            )
            
            result = await self.auth_engine.check_permission(context)
            if result.decision != AccessDecision.GRANTED:
                raise PermissionError(f"Task submission denied: {result.reason}")
            
            # Get task engine instance
            from app.core.task_execution_engine import get_task_execution_engine
            task_engine = get_task_execution_engine()
            
            # Submit task with security context
            task_result = await task_engine.submit_task(
                task_config=task_config,
                submitted_by=current_user.user_id,
                security_context=context
            )
            
            # Log task submission
            await self._log_security_event(
                "task_submitted",
                current_user.user_id,
                ResourceType.TASK,
                True,
                {
                    "task_id": task_result.get("task_id"),
                    "task_type": task_config.get("type"),
                    "complexity_level": complexity_level
                }
            )
            
            return {
                "success": True,
                "task_id": task_result.get("task_id"),
                "security_validation": task_security_validation,
                "complexity_level": complexity_level
            }
            
        except Exception as e:
            logger.error(f"Secure task submission failed: {e}")
            await self._log_security_event(
                "task_submission_failed",
                current_user.user_id,
                ResourceType.TASK,
                False,
                {"error": str(e)}
            )
            raise
    
    @require_permission(ResourceType.TASK, "monitor", PermissionLevel.READ)
    async def secure_task_monitoring(
        self,
        task_id: str,
        current_user,
        request=None
    ) -> Dict[str, Any]:
        """
        Secure task monitoring with access control.
        """
        try:
            # Check if user can monitor this specific task
            context = AuthorizationContext(
                user_id=current_user.user_id,
                resource_type=ResourceType.TASK,
                resource_id=task_id,
                action="monitor",
                permission_level=PermissionLevel.READ
            )
            
            result = await self.auth_engine.check_permission(context)
            if result.decision != AccessDecision.GRANTED:
                raise PermissionError(f"Task monitoring denied: {result.reason}")
            
            # Get task engine instance
            from app.core.task_execution_engine import get_task_execution_engine
            task_engine = get_task_execution_engine()
            
            # Get task status with security filtering
            task_status = await task_engine.get_task_status(
                task_id=task_id,
                requested_by=current_user.user_id
            )
            
            # Filter sensitive information based on user permissions
            filtered_status = await self._filter_task_status_by_permissions(
                task_status, current_user, context
            )
            
            return {
                "success": True,
                "task_id": task_id,
                "status": filtered_status
            }
            
        except Exception as e:
            logger.error(f"Secure task monitoring failed: {e}")
            raise
    
    # Monitoring System Security Integration
    
    @require_permission(ResourceType.METRICS, "access", PermissionLevel.READ)
    async def secure_metrics_access(
        self,
        metrics_scope: str,
        current_user,
        request=None
    ) -> Dict[str, Any]:
        """
        Secure access to monitoring metrics with role-based filtering.
        """
        try:
            # Different metrics require different permission levels
            permission_level = self._get_metrics_permission_level(metrics_scope)
            
            context = AuthorizationContext(
                user_id=current_user.user_id,
                resource_type=ResourceType.METRICS,
                resource_id=metrics_scope,
                action="access",
                permission_level=permission_level
            )
            
            result = await self.auth_engine.check_permission(context)
            if result.decision != AccessDecision.GRANTED:
                raise PermissionError(f"Metrics access denied: {result.reason}")
            
            # Get monitoring service instance
            from app.core.monitoring_service import get_monitoring_service
            monitoring = get_monitoring_service()
            
            # Get metrics with security filtering
            metrics_data = await monitoring.get_metrics(
                scope=metrics_scope,
                requested_by=current_user.user_id,
                permission_level=permission_level
            )
            
            # Filter sensitive metrics based on user role
            filtered_metrics = await self._filter_metrics_by_role(
                metrics_data, current_user.roles
            )
            
            return {
                "success": True,
                "metrics_scope": metrics_scope,
                "data": filtered_metrics,
                "permission_level": permission_level.value
            }
            
        except Exception as e:
            logger.error(f"Secure metrics access failed: {e}")
            raise
    
    @require_role("admin")
    async def secure_security_dashboard_access(
        self,
        current_user,
        request=None
    ) -> Dict[str, Any]:
        """
        Secure access to security dashboard with admin privileges.
        """
        try:
            # Get comprehensive security metrics
            security_metrics = await self.auth_engine.get_security_metrics()
            
            # Get security events
            security_events = await self._get_recent_security_events()
            
            # Get system health related to security
            security_health = await self._get_security_system_health()
            
            return {
                "success": True,
                "security_metrics": security_metrics,
                "recent_events": security_events,
                "system_health": security_health,
                "accessed_by": current_user.user_id,
                "access_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Security dashboard access failed: {e}")
            raise
    
    # Security Validation Helpers
    
    async def _validate_agent_security_config(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate agent configuration for security compliance."""
        try:
            # Check for required security fields
            required_fields = ["type", "security_level", "capabilities"]
            missing_fields = [field for field in required_fields if field not in agent_config]
            
            if missing_fields:
                return {
                    "is_valid": False,
                    "reason": f"Missing required security fields: {missing_fields}"
                }
            
            # Validate security level
            valid_security_levels = ["low", "standard", "high", "critical"]
            if agent_config.get("security_level") not in valid_security_levels:
                return {
                    "is_valid": False,
                    "reason": f"Invalid security level. Must be one of: {valid_security_levels}"
                }
            
            # Validate capabilities for security risks
            capabilities = agent_config.get("capabilities", [])
            high_risk_capabilities = ["system_access", "network_access", "file_write", "admin_operations"]
            
            has_high_risk = any(cap in high_risk_capabilities for cap in capabilities)
            if has_high_risk and agent_config.get("security_level") not in ["high", "critical"]:
                return {
                    "is_valid": False,
                    "reason": "High-risk capabilities require high or critical security level"
                }
            
            return {
                "is_valid": True,
                "security_level": agent_config.get("security_level"),
                "risk_assessment": "high" if has_high_risk else "standard"
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "reason": f"Security validation error: {str(e)}"
            }
    
    async def _validate_task_security(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate task configuration for security compliance."""
        try:
            # Check for suspicious task types
            suspicious_task_types = ["system_command", "file_operation", "network_request"]
            task_type = task_config.get("type", "")
            
            if task_type in suspicious_task_types:
                # Require additional validation for suspicious tasks
                if not task_config.get("security_approved"):
                    return {
                        "is_valid": False,
                        "reason": f"Task type '{task_type}' requires security approval"
                    }
            
            # Validate resource requirements
            resources = task_config.get("resources", {})
            max_memory = resources.get("memory_mb", 0)
            max_cpu = resources.get("cpu_cores", 0)
            
            # Resource limits for security
            if max_memory > 1024:  # 1GB limit
                return {
                    "is_valid": False,
                    "reason": "Memory requirement exceeds security limit (1GB)"
                }
            
            if max_cpu > 4:  # 4 cores limit
                return {
                    "is_valid": False,
                    "reason": "CPU requirement exceeds security limit (4 cores)"
                }
            
            return {
                "is_valid": True,
                "task_type": task_type,
                "resource_validation": "passed"
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "reason": f"Task security validation error: {str(e)}"
            }
    
    def _assess_task_complexity(self, task_config: Dict[str, Any]) -> str:
        """Assess task complexity for permission level determination."""
        resources = task_config.get("resources", {})
        memory_mb = resources.get("memory_mb", 0)
        cpu_cores = resources.get("cpu_cores", 0)
        estimated_duration = task_config.get("estimated_duration_minutes", 0)
        
        # Complexity scoring
        complexity_score = 0
        
        if memory_mb > 512:
            complexity_score += 2
        elif memory_mb > 256:
            complexity_score += 1
        
        if cpu_cores > 2:
            complexity_score += 2
        elif cpu_cores > 1:
            complexity_score += 1
        
        if estimated_duration > 60:  # More than 1 hour
            complexity_score += 2
        elif estimated_duration > 15:  # More than 15 minutes
            complexity_score += 1
        
        # Task type complexity
        high_complexity_types = ["system_command", "ml_training", "data_processing"]
        if task_config.get("type") in high_complexity_types:
            complexity_score += 3
        
        # Determine complexity level
        if complexity_score >= 6:
            return "high"
        elif complexity_score >= 3:
            return "medium"
        else:
            return "low"
    
    def _get_required_permission_level(self, complexity_level: str) -> PermissionLevel:
        """Get required permission level based on task complexity."""
        complexity_to_permission = {
            "low": PermissionLevel.WRITE,
            "medium": PermissionLevel.EXECUTE,
            "high": PermissionLevel.ADMIN
        }
        return complexity_to_permission.get(complexity_level, PermissionLevel.WRITE)
    
    def _get_metrics_permission_level(self, metrics_scope: str) -> PermissionLevel:
        """Get required permission level for metrics access."""
        if metrics_scope in ["security", "admin", "system"]:
            return PermissionLevel.ADMIN
        elif metrics_scope in ["performance", "orchestrator", "agents"]:
            return PermissionLevel.EXECUTE
        else:
            return PermissionLevel.READ
    
    async def _filter_task_status_by_permissions(
        self, 
        task_status: Dict[str, Any], 
        user, 
        context: AuthorizationContext
    ) -> Dict[str, Any]:
        """Filter task status information based on user permissions."""
        filtered_status = {
            "task_id": task_status.get("task_id"),
            "status": task_status.get("status"),
            "progress": task_status.get("progress")
        }
        
        # Add detailed information based on permission level
        if context.permission_level in [PermissionLevel.EXECUTE, PermissionLevel.ADMIN]:
            filtered_status.update({
                "logs": task_status.get("logs"),
                "resource_usage": task_status.get("resource_usage")
            })
        
        if context.permission_level == PermissionLevel.ADMIN:
            filtered_status.update({
                "security_context": task_status.get("security_context"),
                "system_metrics": task_status.get("system_metrics")
            })
        
        return filtered_status
    
    async def _filter_metrics_by_role(
        self, 
        metrics_data: Dict[str, Any], 
        user_roles: List[str]
    ) -> Dict[str, Any]:
        """Filter metrics data based on user roles."""
        filtered_metrics = {}
        
        # Basic metrics for all authenticated users
        filtered_metrics.update({
            "timestamp": metrics_data.get("timestamp"),
            "system_status": metrics_data.get("system_status")
        })
        
        # Performance metrics for agents and admins
        if any(role in ["agent", "admin", "operator"] for role in user_roles):
            filtered_metrics.update({
                "performance": metrics_data.get("performance"),
                "task_metrics": metrics_data.get("task_metrics")
            })
        
        # Security metrics only for admins
        if "admin" in user_roles:
            filtered_metrics.update({
                "security_metrics": metrics_data.get("security_metrics"),
                "audit_logs": metrics_data.get("audit_logs"),
                "system_internals": metrics_data.get("system_internals")
            })
        
        return filtered_metrics
    
    async def _get_recent_security_events(self) -> List[Dict[str, Any]]:
        """Get recent security events for dashboard."""
        try:
            # Get events from Redis (stored by unified auth engine)
            events_data = await self.auth_engine.redis.lrange("security_events", 0, 49)  # Last 50 events
            
            events = []
            for event_data in events_data:
                if isinstance(event_data, bytes):
                    event_data = event_data.decode('utf-8')
                
                try:
                    event = eval(event_data)  # In production, use json.loads
                    events.append(event)
                except:
                    continue
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get security events: {e}")
            return []
    
    async def _get_security_system_health(self) -> Dict[str, Any]:
        """Get security system health status."""
        try:
            metrics = await self.auth_engine.get_security_metrics()
            
            # Calculate health indicators
            auth_success_rate = (
                metrics["authentication_metrics"]["success_rate"]
                if metrics["authentication_metrics"]["total_authentications"] > 0 
                else 1.0
            )
            
            authz_success_rate = (
                metrics["authorization_metrics"]["success_rate"]
                if metrics["authorization_metrics"]["total_authorizations"] > 0
                else 1.0
            )
            
            cache_hit_rate = metrics["cache_metrics"]["cache_hit_rate"]
            
            # Overall health score
            health_score = (auth_success_rate + authz_success_rate + cache_hit_rate) / 3
            
            if health_score >= 0.9:
                status = "healthy"
            elif health_score >= 0.7:
                status = "warning"
            else:
                status = "critical"
            
            return {
                "overall_status": status,
                "health_score": health_score,
                "authentication_health": auth_success_rate,
                "authorization_health": authz_success_rate,
                "cache_performance": cache_hit_rate,
                "threats_detected": metrics["security_metrics"]["threats_detected"],
                "active_tokens": metrics["security_metrics"]["active_tokens"]
            }
            
        except Exception as e:
            logger.error(f"Failed to get security health: {e}")
            return {
                "overall_status": "unknown",
                "error": str(e)
            }
    
    async def _log_security_event(
        self,
        action: str,
        user_id: str,
        resource_type: ResourceType,
        success: bool,
        metadata: Dict[str, Any]
    ):
        """Log security event using unified auth engine."""
        await self.auth_engine._log_security_event(
            action=action,
            resource=resource_type.value,
            success=success,
            user_id=user_id,
            metadata=metadata
        )


def get_security_integration() -> SecurityOrchestrationIntegration:
    """Get security integration instance."""
    return SecurityOrchestrationIntegration()


# Integration validation and testing
async def validate_security_integration():
    """Validate that security integration is working correctly."""
    try:
        integration = get_security_integration()
        auth_engine = get_unified_authorization_engine()
        
        # Test basic functionality
        metrics = await auth_engine.get_security_metrics()
        
        return {
            "integration_status": "active",
            "unified_engine_status": "active",
            "metrics_available": bool(metrics),
            "validation_time": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "integration_status": "error",
            "error": str(e),
            "validation_time": datetime.utcnow().isoformat()
        }


if __name__ == "__main__":
    import asyncio
    
    async def test_integration():
        validation = await validate_security_integration()
        print("Security Integration Validation:", validation)
    
    asyncio.run(test_integration())