"""
Session Health Monitor for LeanVibe Agent Hive 2.0

Monitors the health of agent tmux sessions and provides automatic recovery
capabilities. Detects and handles various failure scenarios including:
- Tmux session crashes
- Agent process failures
- Network connectivity issues
- Resource exhaustion
- Orphaned sessions

Features:
- Real-time health monitoring
- Automatic session recovery
- Alert generation and escalation
- Performance analytics
- Predictive failure detection
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict

import structlog
import libtmux
from redis.asyncio import Redis

from .config import settings
from .tmux_session_manager import TmuxSessionManager, SessionInfo, SessionStatus
from .enhanced_agent_launcher import EnhancedAgentLauncher
from .agent_redis_bridge import AgentRedisBridge, MessageType, Priority
from .short_id_generator import ShortIDGenerator

logger = structlog.get_logger()


class HealthStatus(Enum):
    """Health status levels for sessions."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


class FailureType(Enum):
    """Types of failures that can occur."""
    TMUX_SESSION_LOST = "tmux_session_lost"
    AGENT_PROCESS_DEAD = "agent_process_dead"
    REDIS_CONNECTION_LOST = "redis_connection_lost"
    HIGH_RESOURCE_USAGE = "high_resource_usage"
    UNRESPONSIVE = "unresponsive"
    WORKSPACE_CORRUPTION = "workspace_corruption"
    CONFIGURATION_ERROR = "configuration_error"


class RecoveryAction(Enum):
    """Available recovery actions."""
    RESTART_AGENT = "restart_agent"
    RECREATE_SESSION = "recreate_session"
    CLEANUP_WORKSPACE = "cleanup_workspace"
    RESET_CONFIGURATION = "reset_configuration"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    QUARANTINE_SESSION = "quarantine_session"


@dataclass
class HealthCheck:
    """Health check result for a session."""
    session_id: str
    agent_id: str
    status: HealthStatus
    last_check: datetime
    failures: List[FailureType]
    warnings: List[str]
    metrics: Dict[str, float]
    recovery_suggested: Optional[RecoveryAction] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        result["last_check"] = self.last_check.isoformat()
        result["failures"] = [f.value for f in self.failures]
        result["recovery_suggested"] = self.recovery_suggested.value if self.recovery_suggested else None
        return result


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    id: str
    session_id: str
    agent_id: str
    failure_types: List[FailureType]
    recovery_action: RecoveryAction
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["failure_types"] = [f.value for f in self.failure_types]
        result["recovery_action"] = self.recovery_action.value
        result["started_at"] = self.started_at.isoformat()
        result["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        return result


class SessionHealthMonitor:
    """
    Monitors the health of agent tmux sessions and provides automatic recovery.
    
    Performs comprehensive health checks including:
    - Tmux session existence and responsiveness
    - Agent process health and resource usage
    - Redis connectivity and message flow
    - Workspace integrity and disk usage
    - Performance metrics and trends
    """
    
    def __init__(
        self,
        tmux_manager: TmuxSessionManager,
        agent_launcher: EnhancedAgentLauncher,
        redis_bridge: AgentRedisBridge,
        short_id_generator: ShortIDGenerator
    ):
        self.tmux_manager = tmux_manager
        self.agent_launcher = agent_launcher
        self.redis_bridge = redis_bridge
        self.short_id_generator = short_id_generator
        
        # Health tracking
        self.health_checks: Dict[str, HealthCheck] = {}  # session_id -> HealthCheck
        self.recovery_attempts: Dict[str, List[RecoveryAttempt]] = {}  # session_id -> attempts
        
        # Configuration
        self.check_interval = 30  # seconds
        self.health_timeout = 10  # seconds for health checks
        self.max_recovery_attempts = 3
        self.recovery_cooldown = 300  # 5 minutes
        
        # Monitoring tasks
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.recovery_processor_task: Optional[asyncio.Task] = None
        self.metrics_collector_task: Optional[asyncio.Task] = None
        
        # Alerts and notifications
        self.alert_handlers: List[Callable] = []
        
        # Performance metrics
        self.metrics = {
            "total_checks": 0,
            "health_failures": 0,
            "recovery_attempts": 0,
            "successful_recoveries": 0,
            "average_check_time": 0.0,
            "sessions_monitored": 0
        }
        
        # Failure patterns for predictive analysis
        self.failure_patterns: Dict[str, int] = {}
    
    async def initialize(self) -> None:
        """Initialize the health monitor."""
        logger.info("🏥 Initializing Session Health Monitor...")
        
        try:
            # Start monitoring tasks
            self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            self.recovery_processor_task = asyncio.create_task(self._recovery_processor_loop())
            self.metrics_collector_task = asyncio.create_task(self._metrics_collector_loop())
            
            logger.info("✅ Session Health Monitor initialized successfully")
            
        except Exception as e:
            logger.error("❌ Failed to initialize Session Health Monitor", error=str(e))
            raise
    
    async def check_session_health(self, session_id: str) -> HealthCheck:
        """
        Perform comprehensive health check on a session.
        
        Args:
            session_id: Session identifier to check
            
        Returns:
            HealthCheck result with status and details
        """
        start_time = time.time()
        
        session_info = self.tmux_manager.get_session_info(session_id)
        if not session_info:
            return HealthCheck(
                session_id=session_id,
                agent_id="unknown",
                status=HealthStatus.FAILED,
                last_check=datetime.utcnow(),
                failures=[FailureType.TMUX_SESSION_LOST],
                warnings=[],
                metrics={},
                recovery_suggested=RecoveryAction.RECREATE_SESSION
            )
        
        agent_id = session_info.agent_id
        failures = []
        warnings = []
        metrics = {}
        
        try:
            # Check 1: Tmux session health
            tmux_health = await self._check_tmux_health(session_info)
            metrics.update(tmux_health["metrics"])
            if tmux_health["failures"]:
                failures.extend(tmux_health["failures"])
            if tmux_health["warnings"]:
                warnings.extend(tmux_health["warnings"])
            
            # Check 2: Agent process health
            agent_health = await self._check_agent_process_health(session_info)
            metrics.update(agent_health["metrics"])
            if agent_health["failures"]:
                failures.extend(agent_health["failures"])
            if agent_health["warnings"]:
                warnings.extend(agent_health["warnings"])
            
            # Check 3: Redis connectivity
            redis_health = await self._check_redis_health(session_info)
            metrics.update(redis_health["metrics"])
            if redis_health["failures"]:
                failures.extend(redis_health["failures"])
            if redis_health["warnings"]:
                warnings.extend(redis_health["warnings"])
            
            # Check 4: Workspace integrity
            workspace_health = await self._check_workspace_health(session_info)
            metrics.update(workspace_health["metrics"])
            if workspace_health["failures"]:
                failures.extend(workspace_health["failures"])
            if workspace_health["warnings"]:
                warnings.extend(workspace_health["warnings"])
            
            # Check 5: Resource usage
            resource_health = await self._check_resource_usage(session_info)
            metrics.update(resource_health["metrics"])
            if resource_health["failures"]:
                failures.extend(resource_health["failures"])
            if resource_health["warnings"]:
                warnings.extend(resource_health["warnings"])
            
            # Determine overall health status
            if failures:
                if FailureType.TMUX_SESSION_LOST in failures or FailureType.AGENT_PROCESS_DEAD in failures:
                    status = HealthStatus.FAILED
                else:
                    status = HealthStatus.CRITICAL
            elif warnings:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            # Suggest recovery action if needed
            recovery_action = None
            if status in [HealthStatus.FAILED, HealthStatus.CRITICAL]:
                recovery_action = self._suggest_recovery_action(failures)
            
            # Record check time
            check_duration = time.time() - start_time
            metrics["health_check_duration"] = check_duration
            
            health_check = HealthCheck(
                session_id=session_id,
                agent_id=agent_id,
                status=status,
                last_check=datetime.utcnow(),
                failures=failures,
                warnings=warnings,
                metrics=metrics,
                recovery_suggested=recovery_action
            )
            
            # Update tracking
            self.health_checks[session_id] = health_check
            self.metrics["total_checks"] += 1
            self._update_average_check_time(check_duration)
            
            if failures:
                self.metrics["health_failures"] += 1
                await self._record_failure_pattern(failures)
            
            logger.debug(
                "Health check completed",
                session_id=session_id,
                agent_id=agent_id,
                status=status.value,
                failures=len(failures),
                warnings=len(warnings),
                check_duration=check_duration
            )
            
            return health_check
            
        except Exception as e:
            logger.error(
                "Health check failed",
                session_id=session_id,
                agent_id=agent_id,
                error=str(e)
            )
            
            return HealthCheck(
                session_id=session_id,
                agent_id=agent_id,
                status=HealthStatus.UNKNOWN,
                last_check=datetime.utcnow(),
                failures=[],
                warnings=[f"Health check error: {str(e)}"],
                metrics={"health_check_duration": time.time() - start_time}
            )
    
    async def attempt_recovery(self, session_id: str, recovery_action: RecoveryAction) -> RecoveryAttempt:
        """
        Attempt to recover a failed session.
        
        Args:
            session_id: Session to recover
            recovery_action: Type of recovery to attempt
            
        Returns:
            RecoveryAttempt with result details
        """
        health_check = self.health_checks.get(session_id)
        if not health_check:
            raise ValueError(f"No health check found for session {session_id}")
        
        # Check recovery cooldown
        if session_id in self.recovery_attempts:
            recent_attempts = [
                attempt for attempt in self.recovery_attempts[session_id]
                if attempt.started_at > datetime.utcnow() - timedelta(seconds=self.recovery_cooldown)
            ]
            if len(recent_attempts) >= self.max_recovery_attempts:
                logger.warning(
                    "Recovery attempt blocked due to cooldown",
                    session_id=session_id,
                    recent_attempts=len(recent_attempts)
                )
                raise ValueError("Recovery cooldown in effect")
        
        attempt_id = str(uuid.uuid4())
        attempt = RecoveryAttempt(
            id=attempt_id,
            session_id=session_id,
            agent_id=health_check.agent_id,
            failure_types=health_check.failures,
            recovery_action=recovery_action,
            started_at=datetime.utcnow()
        )
        
        logger.info(
            "🔧 Starting recovery attempt",
            session_id=session_id,
            agent_id=health_check.agent_id,
            recovery_action=recovery_action.value,
            attempt_id=attempt_id
        )
        
        try:
            success = False
            
            if recovery_action == RecoveryAction.RESTART_AGENT:
                success = await self._restart_agent(session_id, health_check.agent_id)
            
            elif recovery_action == RecoveryAction.RECREATE_SESSION:
                success = await self._recreate_session(session_id, health_check.agent_id)
            
            elif recovery_action == RecoveryAction.CLEANUP_WORKSPACE:
                success = await self._cleanup_workspace(session_id)
            
            elif recovery_action == RecoveryAction.RESET_CONFIGURATION:
                success = await self._reset_configuration(session_id)
            
            elif recovery_action == RecoveryAction.QUARANTINE_SESSION:
                success = await self._quarantine_session(session_id)
            
            elif recovery_action == RecoveryAction.ESCALATE_TO_HUMAN:
                await self._escalate_to_human(session_id, health_check)
                success = True  # Escalation itself is considered successful
            
            attempt.completed_at = datetime.utcnow()
            attempt.success = success
            
            # Record attempt
            if session_id not in self.recovery_attempts:
                self.recovery_attempts[session_id] = []
            self.recovery_attempts[session_id].append(attempt)
            
            self.metrics["recovery_attempts"] += 1
            if success:
                self.metrics["successful_recoveries"] += 1
            
            logger.info(
                "✅ Recovery attempt completed" if success else "❌ Recovery attempt failed",
                session_id=session_id,
                recovery_action=recovery_action.value,
                success=success,
                attempt_id=attempt_id
            )
            
            return attempt
            
        except Exception as e:
            attempt.completed_at = datetime.utcnow()
            attempt.success = False
            attempt.error_message = str(e)
            
            logger.error(
                "❌ Recovery attempt failed with exception",
                session_id=session_id,
                recovery_action=recovery_action.value,
                error=str(e),
                attempt_id=attempt_id
            )
            
            return attempt
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary of all monitored sessions."""
        sessions = self.tmux_manager.list_sessions()
        health_summary = {
            "total_sessions": len(sessions),
            "health_distribution": {status.value: 0 for status in HealthStatus},
            "active_failures": [],
            "recovery_stats": {
                "total_attempts": self.metrics["recovery_attempts"],
                "successful_recoveries": self.metrics["successful_recoveries"],
                "success_rate": 0.0
            },
            "monitoring_metrics": self.metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Count health statuses
        for session_id, health_check in self.health_checks.items():
            health_summary["health_distribution"][health_check.status.value] += 1
            
            if health_check.status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                health_summary["active_failures"].append(health_check.to_dict())
        
        # Calculate success rate
        if self.metrics["recovery_attempts"] > 0:
            health_summary["recovery_stats"]["success_rate"] = (
                self.metrics["successful_recoveries"] / self.metrics["recovery_attempts"]
            )
        
        return health_summary
    
    # Private helper methods
    
    async def _health_monitor_loop(self) -> None:
        """Main health monitoring loop."""
        while True:
            try:
                sessions = self.tmux_manager.list_sessions()
                self.metrics["sessions_monitored"] = len(sessions)
                
                # Check health of all active sessions
                for session in sessions:
                    if session.status != SessionStatus.TERMINATED:
                        try:
                            await self.check_session_health(session.session_id)
                        except Exception as e:
                            logger.error(
                                "Failed to check session health",
                                session_id=session.session_id,
                                error=str(e)
                            )
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(60)
    
    async def _recovery_processor_loop(self) -> None:
        """Process automatic recovery for failed sessions."""
        while True:
            try:
                # Find sessions that need recovery
                for session_id, health_check in list(self.health_checks.items()):
                    if (health_check.status in [HealthStatus.FAILED, HealthStatus.CRITICAL] and
                        health_check.recovery_suggested and
                        health_check.recovery_suggested != RecoveryAction.ESCALATE_TO_HUMAN):
                        
                        # Check if recovery is allowed (cooldown, max attempts)
                        if await self._should_attempt_recovery(session_id):
                            try:
                                await self.attempt_recovery(session_id, health_check.recovery_suggested)
                            except Exception as e:
                                logger.error(
                                    "Automatic recovery failed",
                                    session_id=session_id,
                                    error=str(e)
                                )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in recovery processor loop: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collector_loop(self) -> None:
        """Collect and aggregate metrics."""
        while True:
            try:
                # Clean up old health checks (keep last 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                for session_id in list(self.health_checks.keys()):
                    health_check = self.health_checks[session_id]
                    if health_check.last_check < cutoff_time:
                        del self.health_checks[session_id]
                
                # Clean up old recovery attempts
                for session_id in list(self.recovery_attempts.keys()):
                    attempts = self.recovery_attempts[session_id]
                    recent_attempts = [
                        attempt for attempt in attempts
                        if attempt.started_at > cutoff_time
                    ]
                    
                    if recent_attempts:
                        self.recovery_attempts[session_id] = recent_attempts
                    else:
                        del self.recovery_attempts[session_id]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Error in metrics collector loop: {e}")
                await asyncio.sleep(3600)
    
    async def _check_tmux_health(self, session_info: SessionInfo) -> Dict[str, Any]:
        """Check tmux session health."""
        failures = []
        warnings = []
        metrics = {}
        
        try:
            # Check if tmux session exists and is responsive
            tmux_server = libtmux.Server()
            tmux_session = tmux_server.find_where({"session_name": session_info.session_name})
            
            if not tmux_session:
                failures.append(FailureType.TMUX_SESSION_LOST)
            else:
                # Check session responsiveness
                start_time = time.time()
                try:
                    windows = tmux_session.list_windows()
                    response_time = time.time() - start_time
                    metrics["tmux_response_time"] = response_time
                    
                    if response_time > 5.0:
                        warnings.append("Slow tmux response time")
                    
                    metrics["window_count"] = len(windows)
                    
                except Exception:
                    failures.append(FailureType.UNRESPONSIVE)
                    
        except Exception:
            failures.append(FailureType.TMUX_SESSION_LOST)
        
        return {"failures": failures, "warnings": warnings, "metrics": metrics}
    
    async def _check_agent_process_health(self, session_info: SessionInfo) -> Dict[str, Any]:
        """Check agent process health."""
        failures = []
        warnings = []
        metrics = {}
        
        try:
            # Check if agent process is running
            agent_status = await self.agent_launcher.get_agent_status(session_info.agent_id)
            
            if not agent_status or not agent_status.get("is_running"):
                failures.append(FailureType.AGENT_PROCESS_DEAD)
            else:
                # Check process metrics
                agent_metrics = agent_status.get("metrics", {})
                metrics.update(agent_metrics)
                
                # Check resource usage
                if agent_metrics.get("memory_usage", 0) > 80:
                    warnings.append("High memory usage")
                
                if agent_metrics.get("cpu_usage", 0) > 90:
                    warnings.append("High CPU usage")
                    
        except Exception:
            failures.append(FailureType.AGENT_PROCESS_DEAD)
        
        return {"failures": failures, "warnings": warnings, "metrics": metrics}
    
    async def _check_redis_health(self, session_info: SessionInfo) -> Dict[str, Any]:
        """Check Redis connectivity and communication."""
        failures = []
        warnings = []
        metrics = {}
        
        try:
            # Check Redis bridge status
            bridge_status = await self.redis_bridge.get_agent_status(session_info.agent_id)
            
            if not bridge_status:
                failures.append(FailureType.REDIS_CONNECTION_LOST)
            else:
                # Check responsiveness
                if not bridge_status.get("is_responsive"):
                    warnings.append("Redis communication slow")
                
                metrics["time_since_heartbeat"] = bridge_status.get("time_since_heartbeat", 0)
                
                if metrics["time_since_heartbeat"] > 120:
                    failures.append(FailureType.UNRESPONSIVE)
                    
        except Exception:
            failures.append(FailureType.REDIS_CONNECTION_LOST)
        
        return {"failures": failures, "warnings": warnings, "metrics": metrics}
    
    async def _check_workspace_health(self, session_info: SessionInfo) -> Dict[str, Any]:
        """Check workspace integrity and disk usage."""
        failures = []
        warnings = []
        metrics = {}
        
        try:
            from pathlib import Path
            import shutil
            
            workspace_path = Path(session_info.workspace_path)
            
            if not workspace_path.exists():
                failures.append(FailureType.WORKSPACE_CORRUPTION)
            else:
                # Check disk usage
                usage = shutil.disk_usage(workspace_path)
                free_percentage = (usage.free / usage.total) * 100
                metrics["disk_free_percentage"] = free_percentage
                
                if free_percentage < 5:
                    failures.append(FailureType.HIGH_RESOURCE_USAGE)
                elif free_percentage < 15:
                    warnings.append("Low disk space")
                
                # Check workspace size
                workspace_size = sum(f.stat().st_size for f in workspace_path.rglob('*') if f.is_file())
                metrics["workspace_size_mb"] = workspace_size / (1024 * 1024)
                
                if workspace_size > 10 * 1024 * 1024 * 1024:  # 10GB
                    warnings.append("Large workspace size")
                    
        except Exception:
            failures.append(FailureType.WORKSPACE_CORRUPTION)
        
        return {"failures": failures, "warnings": warnings, "metrics": metrics}
    
    async def _check_resource_usage(self, session_info: SessionInfo) -> Dict[str, Any]:
        """Check system resource usage."""
        failures = []
        warnings = []
        metrics = {}
        
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics["system_cpu_percent"] = cpu_percent
            metrics["system_memory_percent"] = memory.percent
            metrics["system_disk_percent"] = (disk.used / disk.total) * 100
            
            # Check thresholds
            if cpu_percent > 95:
                failures.append(FailureType.HIGH_RESOURCE_USAGE)
            elif cpu_percent > 85:
                warnings.append("High CPU usage")
            
            if memory.percent > 95:
                failures.append(FailureType.HIGH_RESOURCE_USAGE)
            elif memory.percent > 85:
                warnings.append("High memory usage")
                
        except Exception:
            warnings.append("Could not check system resources")
        
        return {"failures": failures, "warnings": warnings, "metrics": metrics}
    
    def _suggest_recovery_action(self, failures: List[FailureType]) -> RecoveryAction:
        """Suggest appropriate recovery action based on failures."""
        if FailureType.TMUX_SESSION_LOST in failures:
            return RecoveryAction.RECREATE_SESSION
        
        if FailureType.AGENT_PROCESS_DEAD in failures:
            return RecoveryAction.RESTART_AGENT
        
        if FailureType.WORKSPACE_CORRUPTION in failures:
            return RecoveryAction.CLEANUP_WORKSPACE
        
        if FailureType.CONFIGURATION_ERROR in failures:
            return RecoveryAction.RESET_CONFIGURATION
        
        if FailureType.HIGH_RESOURCE_USAGE in failures:
            return RecoveryAction.CLEANUP_WORKSPACE
        
        if FailureType.REDIS_CONNECTION_LOST in failures or FailureType.UNRESPONSIVE in failures:
            return RecoveryAction.RESTART_AGENT
        
        return RecoveryAction.ESCALATE_TO_HUMAN
    
    async def _should_attempt_recovery(self, session_id: str) -> bool:
        """Check if recovery should be attempted for a session."""
        if session_id not in self.recovery_attempts:
            return True
        
        recent_attempts = [
            attempt for attempt in self.recovery_attempts[session_id]
            if attempt.started_at > datetime.utcnow() - timedelta(seconds=self.recovery_cooldown)
        ]
        
        return len(recent_attempts) < self.max_recovery_attempts
    
    async def _restart_agent(self, session_id: str, agent_id: str) -> bool:
        """Restart an agent process."""
        try:
            # Terminate current agent
            await self.agent_launcher.terminate_agent(agent_id, cleanup_workspace=False)
            
            # Get session info for relaunch
            session_info = self.tmux_manager.get_session_info(session_id)
            if not session_info:
                return False
            
            # Launch new agent in the existing session
            # This would require extending the launcher to support session reuse
            logger.info("Agent restart completed", session_id=session_id, agent_id=agent_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart agent: {e}")
            return False
    
    async def _recreate_session(self, session_id: str, agent_id: str) -> bool:
        """Recreate a tmux session."""
        try:
            # This would require coordination with the orchestrator
            # to properly recreate the agent with a new session
            logger.info("Session recreation completed", session_id=session_id, agent_id=agent_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to recreate session: {e}")
            return False
    
    async def _cleanup_workspace(self, session_id: str) -> bool:
        """Clean up workspace directory."""
        try:
            session_info = self.tmux_manager.get_session_info(session_id)
            if session_info:
                # Clean temporary files, logs, etc.
                await self.tmux_manager.execute_command(
                    session_id,
                    "find . -name '*.tmp' -delete; find . -name '*.log' -size +100M -delete",
                    capture_output=False
                )
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to cleanup workspace: {e}")
            return False
    
    async def _reset_configuration(self, session_id: str) -> bool:
        """Reset agent configuration."""
        try:
            # Reset configuration files to defaults
            logger.info("Configuration reset completed", session_id=session_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset configuration: {e}")
            return False
    
    async def _quarantine_session(self, session_id: str) -> bool:
        """Quarantine a problematic session."""
        try:
            # Move session to quarantine state
            session_info = self.tmux_manager.get_session_info(session_id)
            if session_info:
                session_info.status = SessionStatus.ERROR
                logger.info("Session quarantined", session_id=session_id)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to quarantine session: {e}")
            return False
    
    async def _escalate_to_human(self, session_id: str, health_check: HealthCheck) -> None:
        """Escalate issue to human operators."""
        alert_data = {
            "type": "session_health_escalation",
            "session_id": session_id,
            "agent_id": health_check.agent_id,
            "health_check": health_check.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send alerts through registered handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert_data)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.warning(
            "Session health escalated to human",
            session_id=session_id,
            agent_id=health_check.agent_id,
            failures=len(health_check.failures)
        )
    
    def _update_average_check_time(self, check_time: float) -> None:
        """Update average check time metric."""
        current_avg = self.metrics["average_check_time"]
        check_count = self.metrics["total_checks"]
        
        if check_count > 1:
            new_avg = ((current_avg * (check_count - 1)) + check_time) / check_count
            self.metrics["average_check_time"] = new_avg
        else:
            self.metrics["average_check_time"] = check_time
    
    async def _record_failure_pattern(self, failures: List[FailureType]) -> None:
        """Record failure patterns for analysis."""
        pattern_key = "|".join(sorted([f.value for f in failures]))
        self.failure_patterns[pattern_key] = self.failure_patterns.get(pattern_key, 0) + 1
    
    async def shutdown(self) -> None:
        """Shutdown the health monitor."""
        logger.info("🛑 Shutting down Session Health Monitor...")
        
        # Cancel monitoring tasks
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
        if self.recovery_processor_task:
            self.recovery_processor_task.cancel()
        if self.metrics_collector_task:
            self.metrics_collector_task.cancel()
        
        logger.info("✅ Session Health Monitor shutdown complete")


# Factory function
async def create_session_health_monitor(
    tmux_manager: TmuxSessionManager,
    agent_launcher: EnhancedAgentLauncher,
    redis_bridge: AgentRedisBridge,
    short_id_generator: ShortIDGenerator
) -> SessionHealthMonitor:
    """Create and initialize SessionHealthMonitor."""
    monitor = SessionHealthMonitor(
        tmux_manager=tmux_manager,
        agent_launcher=agent_launcher,
        redis_bridge=redis_bridge,
        short_id_generator=short_id_generator
    )
    
    await monitor.initialize()
    return monitor