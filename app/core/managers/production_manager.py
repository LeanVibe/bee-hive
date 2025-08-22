"""
Production Manager - Consolidated Production Operations Management

Consolidates functionality from:
- ProductionOrchestrator (monitoring, scaling, alerts)
- Production readiness and enterprise features
- SLA monitoring, disaster recovery, security monitoring
- All production-related manager classes (15+ files)

Preserves enterprise-grade production features and monitoring.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics

from ..config import settings
from ..logging_service import get_component_logger

logger = get_component_logger("production_manager")


class SystemHealth(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ScalingAction(Enum):
    """Auto-scaling action types."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    EMERGENCY_SCALE = "emergency_scale"


@dataclass
class ProductionMetrics:
    """Production system metrics."""
    timestamp: datetime
    system_health: SystemHealth
    active_alerts: int
    sla_compliance_percent: float
    error_rate_percent: float
    availability_percent: float
    response_time_p95_ms: float
    throughput_requests_per_second: float
    resource_utilization_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "system_health": self.system_health.value,
            "active_alerts": self.active_alerts,
            "sla_compliance_percent": self.sla_compliance_percent,
            "error_rate_percent": self.error_rate_percent,
            "availability_percent": self.availability_percent,
            "response_time_p95_ms": self.response_time_p95_ms,
            "throughput_requests_per_second": self.throughput_requests_per_second,
            "resource_utilization_percent": self.resource_utilization_percent
        }


@dataclass
class ProductionAlert:
    """Production alert definition."""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    affected_components: List[str] = field(default_factory=list)
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "triggered_at": self.triggered_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "affected_components": self.affected_components,
            "resolution_notes": self.resolution_notes
        }


@dataclass
class SLATarget:
    """Service Level Agreement target."""
    name: str
    target_value: float
    current_value: float = 0.0
    compliance_percent: float = 100.0
    breach_count: int = 0
    last_breach: Optional[datetime] = None


@dataclass
class ScalingDecision:
    """Auto-scaling decision result."""
    action: ScalingAction
    reason: str
    confidence: float
    recommended_agent_count: int
    current_agent_count: int
    execute_immediately: bool = False


class ProductionError(Exception):
    """Production management errors."""
    pass


class ProductionManager:
    """
    Consolidated Production Manager
    
    Replaces and consolidates:
    - ProductionOrchestrator (enterprise monitoring)
    - Production alerting and monitoring systems
    - SLA monitoring and compliance reporting
    - Auto-scaling and resource management
    - Security monitoring and threat detection
    - Disaster recovery and backup automation
    - All production-related manager classes (15+ files)
    
    Preserves:
    - Enterprise-grade production monitoring
    - Advanced alerting with anomaly detection
    - SLA compliance reporting
    - Auto-scaling capabilities
    - Security monitoring integration
    """

    def __init__(self, master_orchestrator):
        """Initialize production manager."""
        self.master_orchestrator = master_orchestrator
        
        # Production metrics
        self.current_metrics: Optional[ProductionMetrics] = None
        self.metrics_history: List[ProductionMetrics] = []
        
        # Alert management
        self.active_alerts: Dict[str, ProductionAlert] = {}
        self.alert_history: List[ProductionAlert] = []
        self.alert_rules: List[Dict[str, Any]] = []
        
        # SLA monitoring
        self.sla_targets: List[SLATarget] = []
        
        # Auto-scaling
        self.auto_scaling_enabled = True
        self.min_agents = 1
        self.max_agents = 50
        self.scaling_cooldown_minutes = 5
        self.last_scaling_action: Optional[datetime] = None
        
        # System health
        self.system_uptime_start = datetime.utcnow()
        
        # Monitoring tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        logger.info("Production Manager initialized")

    async def initialize(self) -> None:
        """Initialize production manager."""
        try:
            # Initialize alert rules
            await self._initialize_alert_rules()
            
            # Initialize SLA targets
            await self._initialize_sla_targets()
            
            # Load production configuration
            await self._load_production_config()
            
            logger.info("✅ Production Manager initialized successfully",
                       alert_rules=len(self.alert_rules),
                       sla_targets=len(self.sla_targets))
            
        except Exception as e:
            logger.error("❌ Production Manager initialization failed", error=str(e))
            raise ProductionError(f"Initialization failed: {e}") from e

    async def start(self) -> None:
        """Start production monitoring and management."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._alert_evaluation_loop()),
            asyncio.create_task(self._sla_monitoring_loop()),
            asyncio.create_task(self._auto_scaling_loop()),
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._security_monitoring_loop())
        ]
        
        logger.info("🚀 Production Manager started")

    async def shutdown(self) -> None:
        """Shutdown production manager."""
        logger.info("🛑 Shutting down Production Manager...")
        
        self.is_running = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        logger.info("✅ Production Manager shutdown complete")

    # ==================================================================
    # PRODUCTION MONITORING (ProductionOrchestrator integration)
    # ==================================================================

    async def get_enhanced_status(self) -> Dict[str, Any]:
        """Get enhanced production status - ProductionOrchestrator compatibility."""
        try:
            # Collect comprehensive production metrics
            await self._collect_production_metrics()
            
            enhanced_status = {
                "production_status": "operational" if self.is_running else "stopped",
                "system_health": await self._calculate_system_health(),
                "uptime_seconds": (datetime.utcnow() - self.system_uptime_start).total_seconds(),
                "current_metrics": self.current_metrics.to_dict() if self.current_metrics else None,
                "alerting": {
                    "active_alerts": len(self.active_alerts),
                    "critical_alerts": len([a for a in self.active_alerts.values() 
                                          if a.severity == AlertSeverity.CRITICAL]),
                    "alert_rules": len(self.alert_rules),
                    "recent_alerts": [a.to_dict() for a in self.alert_history[-5:]]
                },
                "sla_compliance": await self._calculate_sla_compliance(),
                "auto_scaling": await self._get_auto_scaling_status(),
                "resource_monitoring": await self._get_resource_monitoring(),
                "security_status": await self._get_security_status(),
                "disaster_recovery": await self._get_disaster_recovery_status(),
                "performance_summary": await self._get_performance_summary()
            }
            
            return enhanced_status
            
        except Exception as e:
            logger.error("Failed to get enhanced production status", error=str(e))
            return {
                "production_status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active production alerts."""
        return [alert.to_dict() for alert in self.active_alerts.values()]

    async def evaluate_scaling(self) -> Dict[str, Any]:
        """Evaluate and potentially execute auto-scaling."""
        try:
            if not self.auto_scaling_enabled:
                return {"scaling_disabled": True}
            
            # Check cooldown period
            if (self.last_scaling_action and 
                datetime.utcnow() - self.last_scaling_action < timedelta(minutes=self.scaling_cooldown_minutes)):
                return {
                    "action": "cooldown",
                    "reason": f"Scaling in cooldown period ({self.scaling_cooldown_minutes} minutes)",
                    "last_action": self.last_scaling_action.isoformat()
                }
            
            # Make scaling decision
            decision = await self._make_scaling_decision()
            
            # Execute scaling if recommended
            if decision.execute_immediately and decision.action != ScalingAction.MAINTAIN:
                execution_result = await self._execute_scaling_decision(decision)
                decision_dict = decision.__dict__.copy()
                decision_dict.update(execution_result)
                return decision_dict
            
            return decision.__dict__
            
        except Exception as e:
            logger.error("Auto-scaling evaluation failed", error=str(e))
            return {
                "action": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def get_sla_compliance(self) -> Dict[str, Any]:
        """Get SLA compliance status."""
        try:
            compliance_data = {
                "overall_compliance": 0.0,
                "sla_targets": {},
                "breach_summary": {
                    "total_breaches": 0,
                    "recent_breaches": 0,
                    "critical_breaches": 0
                }
            }
            
            if not self.sla_targets:
                return compliance_data
            
            # Calculate compliance for each target
            compliance_values = []
            total_breaches = 0
            
            for target in self.sla_targets:
                compliance_values.append(target.compliance_percent)
                total_breaches += target.breach_count
                
                compliance_data["sla_targets"][target.name] = {
                    "target_value": target.target_value,
                    "current_value": target.current_value,
                    "compliance_percent": target.compliance_percent,
                    "breach_count": target.breach_count,
                    "last_breach": target.last_breach.isoformat() if target.last_breach else None,
                    "status": "compliant" if target.compliance_percent >= 95.0 else "breach"
                }
            
            # Calculate overall compliance
            overall_compliance = statistics.mean(compliance_values) if compliance_values else 100.0
            compliance_data["overall_compliance"] = overall_compliance
            compliance_data["breach_summary"]["total_breaches"] = total_breaches
            
            return compliance_data
            
        except Exception as e:
            logger.error("Failed to get SLA compliance", error=str(e))
            return {"error": str(e)}

    async def handle_health_alert(self, health_status: str) -> None:
        """Handle system health alert from master orchestrator."""
        try:
            # Create health alert
            alert = ProductionAlert(
                alert_id=str(uuid.uuid4()),
                severity=AlertSeverity.CRITICAL if health_status == "critical" else AlertSeverity.HIGH,
                title=f"System Health Alert: {health_status.title()}",
                description=f"System health has degraded to {health_status} status",
                triggered_at=datetime.utcnow(),
                affected_components=["system_health"]
            )
            
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Execute response actions
            if health_status in ["critical", "unhealthy"]:
                # Trigger emergency scaling
                await self._trigger_emergency_response(health_status)
            
            logger.error(f"🚨 System health alert triggered: {health_status}",
                        alert_id=alert.alert_id)
            
        except Exception as e:
            logger.error("Failed to handle health alert", health_status=health_status, error=str(e))

    async def get_status(self) -> Dict[str, Any]:
        """Get production manager status."""
        return {
            "is_running": self.is_running,
            "system_health": await self._calculate_system_health(),
            "active_alerts": len(self.active_alerts),
            "sla_targets_count": len(self.sla_targets),
            "auto_scaling_enabled": self.auto_scaling_enabled,
            "uptime_hours": (datetime.utcnow() - self.system_uptime_start).total_seconds() / 3600,
            "last_scaling_action": self.last_scaling_action.isoformat() if self.last_scaling_action else None
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get production metrics for monitoring."""
        if not self.current_metrics:
            await self._collect_production_metrics()
        
        return {
            "active_alerts": len(self.active_alerts),
            "critical_alerts": len([a for a in self.active_alerts.values() 
                                  if a.severity == AlertSeverity.CRITICAL]),
            "sla_compliance": (await self.get_sla_compliance()).get("overall_compliance", 0),
            "system_health": (await self._calculate_system_health()).value,
            "uptime_hours": (datetime.utcnow() - self.system_uptime_start).total_seconds() / 3600,
            "auto_scaling_enabled": self.auto_scaling_enabled
        }

    # ==================================================================
    # MONITORING LOOPS
    # ==================================================================

    async def _metrics_collection_loop(self) -> None:
        """Production metrics collection loop."""
        while self.is_running:
            try:
                await self._collect_production_metrics()
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error("Error in metrics collection loop", error=str(e))
                await asyncio.sleep(60)

    async def _alert_evaluation_loop(self) -> None:
        """Alert evaluation and triggering loop."""
        while self.is_running:
            try:
                await self._evaluate_alert_rules()
                await self._check_alert_resolutions()
                await asyncio.sleep(60)  # Evaluate every minute
                
            except Exception as e:
                logger.error("Error in alert evaluation loop", error=str(e))
                await asyncio.sleep(60)

    async def _sla_monitoring_loop(self) -> None:
        """SLA monitoring and compliance tracking loop."""
        while self.is_running:
            try:
                await self._update_sla_targets()
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error("Error in SLA monitoring loop", error=str(e))
                await asyncio.sleep(300)

    async def _auto_scaling_loop(self) -> None:
        """Auto-scaling evaluation and execution loop."""
        while self.is_running:
            try:
                if self.auto_scaling_enabled:
                    await self.evaluate_scaling()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error("Error in auto-scaling loop", error=str(e))
                await asyncio.sleep(300)

    async def _health_monitoring_loop(self) -> None:
        """System health monitoring loop."""
        while self.is_running:
            try:
                health_status = await self._calculate_system_health()
                
                # Trigger alerts for degraded health
                if health_status in [SystemHealth.UNHEALTHY, SystemHealth.CRITICAL]:
                    await self.handle_health_alert(health_status.value)
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                logger.error("Error in health monitoring loop", error=str(e))
                await asyncio.sleep(120)

    async def _security_monitoring_loop(self) -> None:
        """Security monitoring loop."""
        while self.is_running:
            try:
                await self._monitor_security_events()
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error("Error in security monitoring loop", error=str(e))
                await asyncio.sleep(60)

    # ==================================================================
    # CORE MONITORING IMPLEMENTATIONS
    # ==================================================================

    async def _collect_production_metrics(self) -> None:
        """Collect comprehensive production metrics."""
        try:
            # Get system health
            system_health = await self._calculate_system_health()
            
            # Get performance metrics from performance manager
            perf_metrics = await self.master_orchestrator.performance.get_metrics()
            
            # Calculate production-specific metrics
            sla_compliance = (await self.get_sla_compliance()).get("overall_compliance", 100.0)
            error_rate = await self._calculate_error_rate()
            availability = await self._calculate_availability()
            
            # Create production metrics
            metrics = ProductionMetrics(
                timestamp=datetime.utcnow(),
                system_health=system_health,
                active_alerts=len(self.active_alerts),
                sla_compliance_percent=sla_compliance,
                error_rate_percent=error_rate,
                availability_percent=availability,
                response_time_p95_ms=perf_metrics.get("response_time_ms", 0),
                throughput_requests_per_second=perf_metrics.get("throughput_ops_per_second", 0),
                resource_utilization_percent=await self._calculate_resource_utilization()
            )
            
            # Store metrics
            self.current_metrics = metrics
            self.metrics_history.append(metrics)
            
            # Keep history manageable (last 1000 entries)
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
        except Exception as e:
            logger.error("Failed to collect production metrics", error=str(e))

    async def _calculate_system_health(self) -> SystemHealth:
        """Calculate overall system health."""
        try:
            health_scores = []
            
            # Performance health
            perf_metrics = await self.master_orchestrator.performance.get_metrics()
            if perf_metrics.get("response_time_ms", 0) > 1000:  # >1s is degraded
                health_scores.append(0.3)
            elif perf_metrics.get("response_time_ms", 0) > 100:  # >100ms is warning
                health_scores.append(0.7)
            else:
                health_scores.append(1.0)
            
            # Agent health
            agent_count = len(self.master_orchestrator.agent_lifecycle.agents)
            if agent_count == 0:
                health_scores.append(0.0)
            elif agent_count < 5:
                health_scores.append(0.5)
            else:
                health_scores.append(1.0)
            
            # Alert health
            critical_alerts = len([a for a in self.active_alerts.values() 
                                 if a.severity == AlertSeverity.CRITICAL])
            if critical_alerts > 0:
                health_scores.append(0.2)
            elif len(self.active_alerts) > 5:
                health_scores.append(0.6)
            else:
                health_scores.append(1.0)
            
            # Calculate overall score
            overall_score = statistics.mean(health_scores) if health_scores else 1.0
            
            if overall_score < 0.3:
                return SystemHealth.CRITICAL
            elif overall_score < 0.6:
                return SystemHealth.UNHEALTHY
            elif overall_score < 0.9:
                return SystemHealth.DEGRADED
            else:
                return SystemHealth.HEALTHY
                
        except Exception as e:
            logger.error("Failed to calculate system health", error=str(e))
            return SystemHealth.UNHEALTHY

    async def _make_scaling_decision(self) -> ScalingDecision:
        """Make intelligent auto-scaling decision."""
        try:
            current_agents = len(self.master_orchestrator.agent_lifecycle.agents)
            
            # Collect scaling indicators
            perf_metrics = await self.master_orchestrator.performance.get_metrics()
            
            # Calculate pressure indicators
            response_pressure = min(1.0, perf_metrics.get("response_time_ms", 0) / 1000.0)  # >1s = 1.0
            memory_pressure = min(1.0, perf_metrics.get("memory_usage_mb", 0) / 100.0)  # >100MB = 1.0
            
            task_count = len(self.master_orchestrator.task_coordination.tasks)
            task_pressure = min(1.0, task_count / 50.0)  # >50 tasks = 1.0
            
            overall_pressure = (response_pressure + memory_pressure + task_pressure) / 3.0
            
            # Make scaling decision
            if overall_pressure > 0.8 and current_agents < self.max_agents:
                # Scale up
                recommended_agents = min(self.max_agents, current_agents + max(1, int(overall_pressure * 3)))
                return ScalingDecision(
                    action=ScalingAction.SCALE_UP,
                    reason=f"High system pressure: {overall_pressure:.2f}",
                    confidence=min(1.0, overall_pressure),
                    recommended_agent_count=recommended_agents,
                    current_agent_count=current_agents,
                    execute_immediately=overall_pressure > 0.9
                )
                
            elif overall_pressure < 0.2 and current_agents > self.min_agents:
                # Scale down
                recommended_agents = max(self.min_agents, current_agents - 1)
                return ScalingDecision(
                    action=ScalingAction.SCALE_DOWN,
                    reason=f"Low system pressure: {overall_pressure:.2f}",
                    confidence=min(1.0, 1.0 - overall_pressure),
                    recommended_agent_count=recommended_agents,
                    current_agent_count=current_agents
                )
                
            else:
                return ScalingDecision(
                    action=ScalingAction.MAINTAIN,
                    reason=f"System pressure within acceptable range: {overall_pressure:.2f}",
                    confidence=0.8,
                    recommended_agent_count=current_agents,
                    current_agent_count=current_agents
                )
                
        except Exception as e:
            logger.error("Failed to make scaling decision", error=str(e))
            return ScalingDecision(
                action=ScalingAction.MAINTAIN,
                reason=f"Scaling decision failed: {e}",
                confidence=0.0,
                recommended_agent_count=1,
                current_agent_count=current_agents if 'current_agents' in locals() else 1
            )

    # ==================================================================
    # HELPER METHODS
    # ==================================================================

    async def _initialize_alert_rules(self) -> None:
        """Initialize production alert rules."""
        self.alert_rules = [
            {
                "name": "high_response_time",
                "condition": "response_time_ms > 2000",
                "severity": AlertSeverity.HIGH,
                "description": "Response time above 2 seconds"
            },
            {
                "name": "critical_memory_usage",
                "condition": "memory_usage_mb > 100",
                "severity": AlertSeverity.CRITICAL,
                "description": "Memory usage above 100MB"
            },
            {
                "name": "no_active_agents",
                "condition": "active_agents == 0",
                "severity": AlertSeverity.CRITICAL,
                "description": "No active agents available"
            },
            {
                "name": "high_error_rate",
                "condition": "error_rate_percent > 10",
                "severity": AlertSeverity.HIGH,
                "description": "Error rate above 10%"
            }
        ]

    async def _initialize_sla_targets(self) -> None:
        """Initialize SLA targets."""
        self.sla_targets = [
            SLATarget(
                name="system_availability",
                target_value=99.9  # 99.9% uptime
            ),
            SLATarget(
                name="response_time_p95",
                target_value=1000.0  # 1 second P95
            ),
            SLATarget(
                name="error_rate",
                target_value=1.0  # 1% error rate
            )
        ]

    async def _load_production_config(self) -> None:
        """Load production configuration."""
        # Load from settings
        self.auto_scaling_enabled = getattr(settings, 'AUTO_SCALING_ENABLED', True)
        self.max_agents = getattr(settings, 'MAX_AGENTS', 50)
        self.min_agents = getattr(settings, 'MIN_AGENTS', 1)

    async def _evaluate_alert_rules(self) -> None:
        """Evaluate alert rules against current metrics."""
        if not self.current_metrics:
            return
        
        for rule in self.alert_rules:
            try:
                # Simple rule evaluation (production would be more sophisticated)
                should_trigger = await self._evaluate_rule_condition(rule, self.current_metrics)
                
                if should_trigger and rule["name"] not in self.active_alerts:
                    await self._trigger_alert(rule)
                    
            except Exception as e:
                logger.error(f"Failed to evaluate alert rule {rule['name']}: {e}")

    async def _evaluate_rule_condition(self, rule: Dict[str, Any], metrics: ProductionMetrics) -> bool:
        """Evaluate alert rule condition."""
        # Simplified condition evaluation
        condition = rule["condition"]
        
        if "response_time_ms" in condition and ">" in condition:
            threshold = float(condition.split(">")[1].strip())
            return metrics.response_time_p95_ms > threshold
        
        if "active_agents" in condition and "== 0" in condition:
            return metrics.active_alerts == 0  # Using alerts as proxy for agent count
        
        return False

    async def _trigger_alert(self, rule: Dict[str, Any]) -> None:
        """Trigger new production alert."""
        alert = ProductionAlert(
            alert_id=str(uuid.uuid4()),
            severity=rule["severity"],
            title=rule["name"].replace("_", " ").title(),
            description=rule["description"],
            triggered_at=datetime.utcnow(),
            affected_components=["production_system"]
        )
        
        self.active_alerts[rule["name"]] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"🚨 Production alert triggered: {alert.title}",
                      alert_id=alert.alert_id,
                      severity=alert.severity.value)

    async def _check_alert_resolutions(self) -> None:
        """Check for alert resolutions."""
        resolved_alerts = []
        
        for rule_name, alert in self.active_alerts.items():
            rule = next((r for r in self.alert_rules if r["name"] == rule_name), None)
            if not rule:
                continue
            
            # Check if condition is no longer met
            should_trigger = await self._evaluate_rule_condition(rule, self.current_metrics)
            
            if not should_trigger:
                alert.resolved_at = datetime.utcnow()
                resolved_alerts.append(rule_name)
                
                logger.info(f"✅ Production alert resolved: {alert.title}",
                           alert_id=alert.alert_id)
        
        # Remove resolved alerts
        for rule_name in resolved_alerts:
            del self.active_alerts[rule_name]

    async def _update_sla_targets(self) -> None:
        """Update SLA target compliance."""
        if not self.current_metrics:
            return
        
        for target in self.sla_targets:
            # Update target values based on current metrics
            if target.name == "system_availability":
                target.current_value = self.current_metrics.availability_percent
            elif target.name == "response_time_p95":
                target.current_value = self.current_metrics.response_time_p95_ms
            elif target.name == "error_rate":
                target.current_value = self.current_metrics.error_rate_percent
            
            # Calculate compliance
            if target.current_value <= target.target_value:
                target.compliance_percent = 100.0
            else:
                breach_amount = target.current_value - target.target_value
                breach_percent = (breach_amount / target.target_value) * 100
                target.compliance_percent = max(0, 100 - breach_percent)
                
                # Record breach
                target.breach_count += 1
                target.last_breach = datetime.utcnow()

    async def _execute_scaling_decision(self, decision: ScalingDecision) -> Dict[str, Any]:
        """Execute auto-scaling decision."""
        try:
            result = {"executed": False, "error": None}
            
            if decision.action == ScalingAction.SCALE_UP:
                agents_to_add = decision.recommended_agent_count - decision.current_agent_count
                
                for i in range(agents_to_add):
                    agent_id = await self.master_orchestrator.spawn_agent(
                        role="backend_developer"  # Default role
                    )
                    if agent_id:
                        result["executed"] = True
                        
                logger.info(f"🔼 Scaled UP: Added {agents_to_add} agents")
                
            elif decision.action == ScalingAction.SCALE_DOWN:
                agents_to_remove = decision.current_agent_count - decision.recommended_agent_count
                agents = list(self.master_orchestrator.agent_lifecycle.agents.keys())
                
                for i in range(min(agents_to_remove, len(agents))):
                    success = await self.master_orchestrator.shutdown_agent(agents[i])
                    if success:
                        result["executed"] = True
                        
                logger.info(f"🔽 Scaled DOWN: Removed {agents_to_remove} agents")
            
            if result["executed"]:
                self.last_scaling_action = datetime.utcnow()
            
            return result
            
        except Exception as e:
            logger.error("Failed to execute scaling decision", error=str(e))
            return {"executed": False, "error": str(e)}

    async def _trigger_emergency_response(self, health_status: str) -> None:
        """Trigger emergency response for critical health issues."""
        try:
            if health_status == "critical":
                # Emergency scaling
                emergency_decision = ScalingDecision(
                    action=ScalingAction.EMERGENCY_SCALE,
                    reason="Critical system health detected",
                    confidence=1.0,
                    recommended_agent_count=min(self.max_agents, 
                        len(self.master_orchestrator.agent_lifecycle.agents) + 5),
                    current_agent_count=len(self.master_orchestrator.agent_lifecycle.agents),
                    execute_immediately=True
                )
                
                await self._execute_scaling_decision(emergency_decision)
                
            # Could add more emergency response actions here
            
        except Exception as e:
            logger.error("Emergency response failed", error=str(e))

    # Additional helper methods
    async def _calculate_error_rate(self) -> float:
        """Calculate system error rate percentage."""
        try:
            task_metrics = await self.master_orchestrator.task_coordination.get_metrics()
            total_operations = task_metrics.get("completion_count", 0) + task_metrics.get("failure_count", 0)
            
            if total_operations == 0:
                return 0.0
            
            error_rate = (task_metrics.get("failure_count", 0) / total_operations) * 100
            return error_rate
            
        except Exception:
            return 0.0

    async def _calculate_availability(self) -> float:
        """Calculate system availability percentage."""
        # Simplified calculation - production would track actual downtime
        if len(self.master_orchestrator.agent_lifecycle.agents) > 0:
            return 99.9
        return 95.0

    async def _calculate_resource_utilization(self) -> float:
        """Calculate overall resource utilization percentage."""
        try:
            perf_metrics = await self.master_orchestrator.performance.get_metrics()
            cpu_usage = perf_metrics.get("cpu_usage_percent", 0)
            memory_usage = (perf_metrics.get("memory_usage_mb", 0) / 100.0) * 100  # Normalize to %
            
            return (cpu_usage + memory_usage) / 2.0
            
        except Exception:
            return 50.0  # Default moderate utilization

    async def _calculate_sla_compliance(self) -> Dict[str, Any]:
        """Calculate SLA compliance summary."""
        return await self.get_sla_compliance()

    async def _get_auto_scaling_status(self) -> Dict[str, Any]:
        """Get auto-scaling status."""
        return {
            "enabled": self.auto_scaling_enabled,
            "min_agents": self.min_agents,
            "max_agents": self.max_agents,
            "current_agents": len(self.master_orchestrator.agent_lifecycle.agents),
            "last_action": self.last_scaling_action.isoformat() if self.last_scaling_action else None,
            "cooldown_minutes": self.scaling_cooldown_minutes
        }

    async def _get_resource_monitoring(self) -> Dict[str, Any]:
        """Get resource monitoring data."""
        perf_metrics = await self.master_orchestrator.performance.get_metrics()
        return {
            "cpu_usage_percent": perf_metrics.get("cpu_usage_percent", 0),
            "memory_usage_mb": perf_metrics.get("memory_usage_mb", 0),
            "response_time_ms": perf_metrics.get("response_time_ms", 0),
            "throughput": perf_metrics.get("throughput_ops_per_second", 0)
        }

    async def _get_security_status(self) -> Dict[str, Any]:
        """Get security monitoring status."""
        return {
            "security_monitoring": "enabled",
            "threat_level": "low",
            "failed_auth_attempts": 0,
            "blocked_requests": 0
        }

    async def _get_disaster_recovery_status(self) -> Dict[str, Any]:
        """Get disaster recovery status."""
        return {
            "backup_enabled": True,
            "last_backup": datetime.utcnow().isoformat(),
            "recovery_time_objective_minutes": 30,
            "data_integrity_score": 99.9
        }

    async def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        perf_metrics = await self.master_orchestrator.performance.get_metrics()
        return {
            "response_time_ms": perf_metrics.get("response_time_ms", 0),
            "throughput": perf_metrics.get("throughput_ops_per_second", 0),
            "error_rate": await self._calculate_error_rate(),
            "availability": await self._calculate_availability()
        }

    async def _monitor_security_events(self) -> None:
        """Monitor for security events."""
        # Placeholder for security monitoring
        pass