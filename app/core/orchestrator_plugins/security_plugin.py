"""
Security Plugin for Orchestrator

Consolidates functionality from:
- security_orchestrator_integration.py
- security_monitoring_system.py
- integrated_security_system.py
- security_audit.py
- security.py
"""

import asyncio
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from . import OrchestratorPlugin, PluginMetadata, PluginType
from ..config import settings
from ..redis import get_redis
from ..database import get_session
from ..logging_service import get_component_logger

logger = get_component_logger("security_plugin")


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_FAILURE = "authz_failure"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_TOKEN = "invalid_token"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_ACCESS_VIOLATION = "data_access_violation"


class SecurityThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_id: str
    event_type: SecurityEventType
    threat_level: SecurityThreatLevel
    timestamp: datetime
    source_ip: Optional[str]
    user_id: Optional[str]
    agent_id: Optional[str]
    description: str
    metadata: Dict[str, Any]


@dataclass
class SecurityMetrics:
    """Security monitoring metrics."""
    timestamp: datetime
    auth_failures_last_hour: int
    suspicious_activities_last_hour: int
    blocked_requests_last_hour: int
    active_sessions: int
    threat_detections: int


class SecurityPlugin(OrchestratorPlugin):
    """Plugin for security monitoring and enforcement."""
    
    def __init__(self):
        metadata = PluginMetadata(
            name="security_plugin",
            version="1.0.0",
            plugin_type=PluginType.SECURITY,
            description="Authentication, authorization, and security monitoring",
            dependencies=["redis", "database"]
        )
        super().__init__(metadata)
        
        self.security_events: List[SecurityEvent] = []
        self.blocked_ips: Set[str] = set()
        self.rate_limits: Dict[str, List[float]] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Security thresholds
        self.max_auth_failures_per_hour = 10
        self.max_requests_per_minute = 60
        self.suspicious_activity_threshold = 5
        
    async def initialize(self, orchestrator_context: Dict[str, Any]) -> bool:
        """Initialize security monitoring."""
        try:
            self.redis = await get_redis()
            
            # Load blocked IPs from Redis
            blocked_ips_data = await self.redis.smembers("security:blocked_ips")
            self.blocked_ips = set(blocked_ips_data) if blocked_ips_data else set()
            
            logger.info("Security plugin initialized successfully")
            
            # Start background monitoring
            self._monitoring_task = asyncio.create_task(self._monitor_security())
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize security plugin: {e}")
            return False
            
    async def cleanup(self) -> bool:
        """Cleanup security monitoring resources."""
        try:
            if self._monitoring_task:
                self._monitoring_task.cancel()
                
            # Save blocked IPs to Redis
            if self.blocked_ips:
                await self.redis.sadd("security:blocked_ips", *self.blocked_ips)
                
            logger.info("Security plugin cleaned up successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup security plugin: {e}")
            return False
            
    async def pre_task_execution(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Security checks before task execution."""
        # Check if source IP is blocked
        source_ip = task_context.get("source_ip")
        if source_ip and source_ip in self.blocked_ips:
            raise SecurityError(f"IP {source_ip} is blocked")
            
        # Rate limiting check
        user_id = task_context.get("user_id")
        if user_id:
            if not await self._check_rate_limit(user_id):
                await self._log_security_event(
                    SecurityEventType.RATE_LIMIT_EXCEEDED,
                    SecurityThreatLevel.MEDIUM,
                    source_ip,
                    user_id,
                    task_context.get("agent_id"),
                    f"Rate limit exceeded for user {user_id}"
                )
                raise SecurityError("Rate limit exceeded")
                
        # Add security context
        task_context["security_validated"] = True
        task_context["security_timestamp"] = datetime.utcnow()
        
        return task_context
        
    async def post_task_execution(self, task_context: Dict[str, Any], result: Any) -> Any:
        """Security audit after task execution."""
        # Log successful task execution
        await self._log_successful_execution(task_context)
        
        # Check for suspicious patterns
        await self._analyze_execution_patterns(task_context, result)
        
        return result
        
    async def health_check(self) -> Dict[str, Any]:
        """Return security plugin health status."""
        metrics = await self._collect_security_metrics()
        
        health_status = "healthy"
        issues = []
        
        if metrics.auth_failures_last_hour > self.max_auth_failures_per_hour:
            health_status = "warning"
            issues.append(f"High authentication failures: {metrics.auth_failures_last_hour}")
            
        if metrics.threat_detections > 0:
            health_status = "critical"
            issues.append(f"Active threat detections: {metrics.threat_detections}")
            
        return {
            "plugin": self.metadata.name,
            "enabled": self.enabled,
            "status": health_status,
            "issues": issues,
            "metrics": {
                "auth_failures_last_hour": metrics.auth_failures_last_hour,
                "suspicious_activities_last_hour": metrics.suspicious_activities_last_hour,
                "blocked_requests_last_hour": metrics.blocked_requests_last_hour,
                "active_sessions": metrics.active_sessions,
                "threat_detections": metrics.threat_detections,
                "blocked_ips_count": len(self.blocked_ips)
            }
        }
        
    async def _monitor_security(self):
        """Background task for security monitoring."""
        while True:
            try:
                # Clean old rate limit entries
                await self._cleanup_rate_limits()
                
                # Check for threat patterns
                await self._detect_threat_patterns()
                
                # Update security metrics
                metrics = await self._collect_security_metrics()
                await self._store_security_metrics(metrics)
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in security monitoring: {e}")
                await asyncio.sleep(60)
                
    async def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits."""
        current_time = time.time()
        window_start = current_time - 60  # 1-minute window
        
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []
            
        # Remove old entries
        self.rate_limits[user_id] = [
            timestamp for timestamp in self.rate_limits[user_id]
            if timestamp > window_start
        ]
        
        # Check limit
        if len(self.rate_limits[user_id]) >= self.max_requests_per_minute:
            return False
            
        # Add current request
        self.rate_limits[user_id].append(current_time)
        return True
        
    async def _cleanup_rate_limits(self):
        """Clean up old rate limit entries."""
        current_time = time.time()
        window_start = current_time - 60
        
        for user_id in list(self.rate_limits.keys()):
            self.rate_limits[user_id] = [
                timestamp for timestamp in self.rate_limits[user_id]
                if timestamp > window_start
            ]
            
            # Remove empty entries
            if not self.rate_limits[user_id]:
                del self.rate_limits[user_id]
                
    async def _log_security_event(
        self,
        event_type: SecurityEventType,
        threat_level: SecurityThreatLevel,
        source_ip: Optional[str],
        user_id: Optional[str],
        agent_id: Optional[str],
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a security event."""
        event_id = hashlib.sha256(
            f"{event_type.value}:{datetime.utcnow().isoformat()}:{source_ip}:{user_id}".encode()
        ).hexdigest()[:16]
        
        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            threat_level=threat_level,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            user_id=user_id,
            agent_id=agent_id,
            description=description,
            metadata=metadata or {}
        )
        
        self.security_events.append(event)
        
        # Keep only last 1000 events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
            
        # Store in Redis
        event_data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "threat_level": event.threat_level.value,
            "timestamp": event.timestamp.isoformat(),
            "source_ip": event.source_ip,
            "user_id": event.user_id,
            "agent_id": event.agent_id,
            "description": event.description,
            "metadata": event.metadata
        }
        
        await self.redis.lpush("security:events", str(event_data))
        await self.redis.ltrim("security:events", 0, 1000)
        
        # Log critical events
        if threat_level in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]:
            logger.warning(f"Security event [{threat_level.value}]: {description}")
            
    async def _log_successful_execution(self, task_context: Dict[str, Any]):
        """Log successful task execution for audit."""
        audit_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": task_context.get("user_id"),
            "agent_id": task_context.get("agent_id"),
            "task_id": task_context.get("task_id"),
            "source_ip": task_context.get("source_ip"),
            "action": "task_execution",
            "status": "success"
        }
        
        await self.redis.lpush("security:audit_log", str(audit_data))
        await self.redis.ltrim("security:audit_log", 0, 10000)  # Keep last 10k entries
        
    async def _analyze_execution_patterns(self, task_context: Dict[str, Any], result: Any):
        """Analyze execution patterns for suspicious activity."""
        user_id = task_context.get("user_id")
        source_ip = task_context.get("source_ip")
        
        if not user_id:
            return
            
        # Check for suspicious patterns
        recent_executions = await self._get_recent_executions(user_id)
        
        # Pattern 1: Too many executions in short time
        if len(recent_executions) > 100:  # More than 100 in last hour
            await self._log_security_event(
                SecurityEventType.SUSPICIOUS_ACTIVITY,
                SecurityThreatLevel.MEDIUM,
                source_ip,
                user_id,
                task_context.get("agent_id"),
                f"Excessive task executions: {len(recent_executions)} in last hour"
            )
            
        # Pattern 2: Multiple different IPs for same user
        unique_ips = set(ex.get("source_ip") for ex in recent_executions if ex.get("source_ip"))
        if len(unique_ips) > 5:
            await self._log_security_event(
                SecurityEventType.SUSPICIOUS_ACTIVITY,
                SecurityThreatLevel.HIGH,
                source_ip,
                user_id,
                task_context.get("agent_id"),
                f"Multiple source IPs detected: {len(unique_ips)} different IPs"
            )
            
    async def _get_recent_executions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get recent executions for a user."""
        try:
            audit_entries = await self.redis.lrange("security:audit_log", 0, 1000)
            user_executions = []
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            
            for entry in audit_entries:
                try:
                    data = eval(entry)  # In production, use json.loads
                    if (data.get("user_id") == user_id and 
                        datetime.fromisoformat(data["timestamp"]) > cutoff_time):
                        user_executions.append(data)
                except Exception:
                    continue
                    
            return user_executions
        except Exception:
            return []
            
    async def _detect_threat_patterns(self):
        """Detect threat patterns in security events."""
        recent_events = [
            event for event in self.security_events
            if event.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        
        # Group by source IP
        ip_events = {}
        for event in recent_events:
            if event.source_ip:
                if event.source_ip not in ip_events:
                    ip_events[event.source_ip] = []
                ip_events[event.source_ip].append(event)
                
        # Check for IPs with multiple security events
        for ip, events in ip_events.items():
            if len(events) >= self.suspicious_activity_threshold:
                if ip not in self.blocked_ips:
                    self.blocked_ips.add(ip)
                    await self.redis.sadd("security:blocked_ips", ip)
                    
                    await self._log_security_event(
                        SecurityEventType.SUSPICIOUS_ACTIVITY,
                        SecurityThreatLevel.HIGH,
                        ip,
                        None,
                        None,
                        f"IP {ip} blocked due to {len(events)} security events"
                    )
                    
    async def _collect_security_metrics(self) -> SecurityMetrics:
        """Collect current security metrics."""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        recent_events = [
            event for event in self.security_events
            if event.timestamp > cutoff_time
        ]
        
        auth_failures = len([
            event for event in recent_events
            if event.event_type == SecurityEventType.AUTHENTICATION_FAILURE
        ])
        
        suspicious_activities = len([
            event for event in recent_events
            if event.event_type == SecurityEventType.SUSPICIOUS_ACTIVITY
        ])
        
        blocked_requests = len([
            event for event in recent_events
            if event.event_type == SecurityEventType.RATE_LIMIT_EXCEEDED
        ])
        
        # Get active sessions from Redis
        try:
            session_keys = await self.redis.keys("session:*:active")
            active_sessions = len(session_keys)
        except Exception:
            active_sessions = 0
            
        # Count high/critical threat detections
        threat_detections = len([
            event for event in recent_events
            if event.threat_level in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]
        ])
        
        return SecurityMetrics(
            timestamp=datetime.utcnow(),
            auth_failures_last_hour=auth_failures,
            suspicious_activities_last_hour=suspicious_activities,
            blocked_requests_last_hour=blocked_requests,
            active_sessions=active_sessions,
            threat_detections=threat_detections
        )
        
    async def _store_security_metrics(self, metrics: SecurityMetrics):
        """Store security metrics in Redis."""
        try:
            metrics_data = {
                "timestamp": metrics.timestamp.isoformat(),
                "auth_failures_last_hour": metrics.auth_failures_last_hour,
                "suspicious_activities_last_hour": metrics.suspicious_activities_last_hour,
                "blocked_requests_last_hour": metrics.blocked_requests_last_hour,
                "active_sessions": metrics.active_sessions,
                "threat_detections": metrics.threat_detections
            }
            
            await self.redis.set("security:current_metrics", str(metrics_data))
            await self.redis.lpush("security:metrics_history", str(metrics_data))
            await self.redis.ltrim("security:metrics_history", 0, 1000)
            
        except Exception as e:
            logger.error(f"Error storing security metrics: {e}")
            
    async def authenticate_request(self, token: str, required_permissions: List[str] = None) -> Dict[str, Any]:
        """Authenticate and authorize a request."""
        try:
            # Basic token validation (in production, implement proper JWT validation)
            if not token or len(token) < 32:
                await self._log_security_event(
                    SecurityEventType.INVALID_TOKEN,
                    SecurityThreatLevel.MEDIUM,
                    None,
                    None,
                    None,
                    "Invalid token format"
                )
                raise SecurityError("Invalid token")
                
            # Get user info from token (simplified)
            user_data = await self._validate_token(token)
            
            # Check permissions if required
            if required_permissions:
                user_permissions = user_data.get("permissions", [])
                missing_permissions = set(required_permissions) - set(user_permissions)
                
                if missing_permissions:
                    await self._log_security_event(
                        SecurityEventType.AUTHORIZATION_FAILURE,
                        SecurityThreatLevel.MEDIUM,
                        None,
                        user_data.get("user_id"),
                        None,
                        f"Missing permissions: {missing_permissions}"
                    )
                    raise SecurityError(f"Missing permissions: {missing_permissions}")
                    
            return user_data
            
        except SecurityError:
            raise
        except Exception as e:
            await self._log_security_event(
                SecurityEventType.AUTHENTICATION_FAILURE,
                SecurityThreatLevel.HIGH,
                None,
                None,
                None,
                f"Authentication error: {str(e)}"
            )
            raise SecurityError("Authentication failed")
            
    async def _validate_token(self, token: str) -> Dict[str, Any]:
        """Validate authentication token."""
        # In production, implement proper JWT validation
        # For now, return mock user data
        return {
            "user_id": "user_" + hashlib.sha256(token.encode()).hexdigest()[:8],
            "permissions": ["read", "write", "execute"],
            "token_valid": True
        }
        
    async def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security summary."""
        metrics = await self._collect_security_metrics()
        
        recent_critical_events = [
            {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "threat_level": event.threat_level.value,
                "timestamp": event.timestamp.isoformat(),
                "description": event.description
            }
            for event in self.security_events[-50:]  # Last 50 events
            if event.threat_level in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]
        ]
        
        return {
            "current_metrics": {
                "auth_failures_last_hour": metrics.auth_failures_last_hour,
                "suspicious_activities_last_hour": metrics.suspicious_activities_last_hour,
                "blocked_requests_last_hour": metrics.blocked_requests_last_hour,
                "active_sessions": metrics.active_sessions,
                "threat_detections": metrics.threat_detections
            },
            "blocked_ips_count": len(self.blocked_ips),
            "recent_critical_events": recent_critical_events,
            "security_status": "secure" if metrics.threat_detections == 0 else "threats_detected"
        }


class SecurityError(Exception):
    """Security-related error."""
    pass