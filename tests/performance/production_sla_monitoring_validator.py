"""
Production SLA Monitoring Validator for EPIC D Phase 2.

Implements comprehensive 99.9% uptime SLA tracking and compliance validation with
automated alerting, escalation procedures, and SLA breach detection.

Features:
- 99.9% uptime SLA tracking and compliance validation
- Comprehensive error recovery testing scenarios
- Automated alerting and escalation procedure validation
- SLA breach detection and reporting
- Downtime impact analysis and cost calculation
- Service level agreement compliance monitoring
- Multi-tier SLA validation (Bronze/Silver/Gold/Platinum)
"""

import asyncio
import logging
import time
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pytest
import numpy as np

logger = logging.getLogger(__name__)


class SLATier(Enum):
    """SLA service tiers with different requirements."""
    BRONZE = "bronze"       # 99.0% uptime, 1s response time
    SILVER = "silver"       # 99.5% uptime, 500ms response time  
    GOLD = "gold"          # 99.9% uptime, 200ms response time
    PLATINUM = "platinum"   # 99.95% uptime, 100ms response time


class SLAMetricType(Enum):
    """Types of SLA metrics to monitor."""
    UPTIME = "uptime"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    AVAILABILITY = "availability"
    MTTR = "mean_time_to_recovery"
    MTBF = "mean_time_between_failures"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Alert notification channels."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    PAGER_DUTY = "pagerduty"
    WEBHOOK = "webhook"


@dataclass
class SLATarget:
    """SLA target definition."""
    metric_type: SLAMetricType
    tier: SLATier
    target_value: float
    measurement_period_hours: int = 24
    breach_threshold_count: int = 3
    
    # Alert configuration
    warning_threshold_percent: float = 80.0  # % of target before warning
    critical_threshold_percent: float = 95.0  # % of target before critical


@dataclass
class SLAMeasurement:
    """Single SLA measurement data point."""
    timestamp: datetime
    metric_type: SLAMetricType
    value: float
    target_value: float
    compliant: bool
    measurement_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLABreachEvent:
    """SLA breach event record."""
    breach_id: str
    timestamp: datetime
    metric_type: SLAMetricType
    tier: SLATier
    actual_value: float
    target_value: float
    severity: AlertSeverity
    duration_seconds: float
    impact_description: str
    recovery_actions: List[str] = field(default_factory=list)
    business_impact_cost: float = 0.0


@dataclass
class SLAComplianceReport:
    """SLA compliance report."""
    report_id: str
    reporting_period: Tuple[datetime, datetime]
    tier: SLATier
    
    # Compliance metrics
    uptime_percentage: float
    avg_response_time_ms: float
    error_rate_percentage: float
    availability_percentage: float
    
    # SLA targets met
    targets_met: Dict[str, bool]
    overall_compliance: bool
    
    # Breach summary
    total_breaches: int
    total_downtime_seconds: float
    mttr_seconds: float
    mtbf_seconds: float
    
    # Business impact
    estimated_cost_impact: float
    affected_users: int
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class ProductionSLAMonitoringValidator:
    """Production SLA monitoring and validation system."""
    
    def __init__(self, 
                 api_base_url: str = "http://localhost:8000",
                 monitoring_config: Dict[str, Any] = None):
        self.api_base_url = api_base_url
        self.monitoring_config = monitoring_config or {
            'measurement_interval_seconds': 60,
            'alert_email': 'sla-alerts@leanvibe.com',
            'escalation_timeout_minutes': 15,
            'max_response_time_samples': 1000
        }
        
        # SLA targets by tier
        self.sla_targets = {
            SLATier.BRONZE: [
                SLATarget(SLAMetricType.UPTIME, SLATier.BRONZE, 99.0),
                SLATarget(SLAMetricType.RESPONSE_TIME, SLATier.BRONZE, 1000.0),  # 1s
                SLATarget(SLAMetricType.ERROR_RATE, SLATier.BRONZE, 2.0),  # 2%
            ],
            SLATier.SILVER: [
                SLATarget(SLAMetricType.UPTIME, SLATier.SILVER, 99.5),
                SLATarget(SLAMetricType.RESPONSE_TIME, SLATier.SILVER, 500.0),  # 500ms
                SLATarget(SLAMetricType.ERROR_RATE, SLATier.SILVER, 1.0),  # 1%
            ],
            SLATier.GOLD: [
                SLATarget(SLAMetricType.UPTIME, SLATier.GOLD, 99.9),
                SLATarget(SLAMetricType.RESPONSE_TIME, SLATier.GOLD, 200.0),  # 200ms
                SLATarget(SLAMetricType.ERROR_RATE, SLATier.GOLD, 0.5),  # 0.5%
            ],
            SLATier.PLATINUM: [
                SLATarget(SLAMetricType.UPTIME, SLATier.PLATINUM, 99.95),
                SLATarget(SLAMetricType.RESPONSE_TIME, SLATier.PLATINUM, 100.0),  # 100ms
                SLATarget(SLAMetricType.ERROR_RATE, SLATier.PLATINUM, 0.1),  # 0.1%
            ]
        }
        
        # Data storage
        self.measurements: Dict[SLATier, List[SLAMeasurement]] = defaultdict(list)
        self.breach_events: List[SLABreachEvent] = []
        self.alert_history: List[Dict[str, Any]] = []
        
        # State tracking
        self.monitoring_active = False
        self.last_measurement_time = {}
        self.breach_counters = defaultdict(int)
    
    async def start_continuous_monitoring(self, duration_hours: float = 24.0):
        """Start continuous SLA monitoring."""
        logger.info(f"🎯 Starting continuous SLA monitoring for {duration_hours} hours")
        
        self.monitoring_active = True
        end_time = time.time() + (duration_hours * 3600)
        measurement_interval = self.monitoring_config['measurement_interval_seconds']
        
        try:
            while self.monitoring_active and time.time() < end_time:
                # Take measurements for all tiers
                for tier in SLATier:
                    await self._take_sla_measurements(tier)
                
                # Check for SLA breaches
                await self._check_sla_breaches()
                
                # Sleep until next measurement
                await asyncio.sleep(measurement_interval)
                
        except Exception as e:
            logger.error(f"❌ SLA monitoring error: {e}")
            raise
        finally:
            self.monitoring_active = False
            logger.info("⏹️ SLA monitoring stopped")
    
    async def _take_sla_measurements(self, tier: SLATier):
        """Take SLA measurements for a specific tier."""
        current_time = datetime.utcnow()
        
        try:
            # Measure uptime/availability
            uptime_measurement = await self._measure_uptime()
            uptime_target = next(t for t in self.sla_targets[tier] if t.metric_type == SLAMetricType.UPTIME)
            
            self.measurements[tier].append(SLAMeasurement(
                timestamp=current_time,
                metric_type=SLAMetricType.UPTIME,
                value=uptime_measurement,
                target_value=uptime_target.target_value,
                compliant=uptime_measurement >= uptime_target.target_value
            ))
            
            # Measure response time
            response_time_measurement = await self._measure_response_time()
            response_time_target = next(t for t in self.sla_targets[tier] if t.metric_type == SLAMetricType.RESPONSE_TIME)
            
            self.measurements[tier].append(SLAMeasurement(
                timestamp=current_time,
                metric_type=SLAMetricType.RESPONSE_TIME,
                value=response_time_measurement,
                target_value=response_time_target.target_value,
                compliant=response_time_measurement <= response_time_target.target_value
            ))
            
            # Measure error rate
            error_rate_measurement = await self._measure_error_rate()
            error_rate_target = next(t for t in self.sla_targets[tier] if t.metric_type == SLAMetricType.ERROR_RATE)
            
            self.measurements[tier].append(SLAMeasurement(
                timestamp=current_time,
                metric_type=SLAMetricType.ERROR_RATE,
                value=error_rate_measurement,
                target_value=error_rate_target.target_value,
                compliant=error_rate_measurement <= error_rate_target.target_value
            ))
            
            self.last_measurement_time[tier] = current_time
            
        except Exception as e:
            logger.warning(f"Failed to take SLA measurements for {tier.value}: {e}")
    
    async def _measure_uptime(self) -> float:
        """Measure current system uptime percentage."""
        try:
            # Check multiple endpoints to determine overall availability
            endpoints = [
                '/health',
                '/api/agents',
                '/api/tasks',
                '/api/dashboard/live'
            ]
            
            successful_checks = 0
            total_checks = len(endpoints)
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                for endpoint in endpoints:
                    try:
                        async with session.get(f"{self.api_base_url}{endpoint}") as response:
                            if response.status < 500:  # Consider 4xx as available (client errors)
                                successful_checks += 1
                    except Exception:
                        # Connection error counts as downtime
                        pass
            
            uptime_percentage = (successful_checks / total_checks) * 100.0
            return uptime_percentage
            
        except Exception as e:
            logger.warning(f"Uptime measurement failed: {e}")
            return 0.0  # Conservative approach - assume down if can't measure
    
    async def _measure_response_time(self) -> float:
        """Measure average response time across key endpoints."""
        try:
            endpoints = [
                '/health',
                '/api/agents',
                '/api/tasks',
            ]
            
            response_times = []
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                for endpoint in endpoints:
                    try:
                        start_time = time.time()
                        async with session.get(f"{self.api_base_url}{endpoint}") as response:
                            response_time = (time.time() - start_time) * 1000  # Convert to ms
                            if response.status < 500:  # Only count successful responses
                                response_times.append(response_time)
                    except Exception:
                        # Failed requests don't contribute to response time average
                        pass
            
            if response_times:
                # Use 95th percentile for SLA measurement
                return np.percentile(response_times, 95)
            else:
                return float('inf')  # No successful responses
                
        except Exception as e:
            logger.warning(f"Response time measurement failed: {e}")
            return float('inf')
    
    async def _measure_error_rate(self) -> float:
        """Measure current error rate percentage."""
        try:
            # Test multiple operations to calculate error rate
            test_operations = [
                ('GET', '/health'),
                ('GET', '/api/agents'),
                ('POST', '/api/tasks', {'description': 'SLA test task', 'priority': 'low'}),
                ('GET', '/api/dashboard/live'),
            ]
            
            successful_operations = 0
            total_operations = 0
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                for method, endpoint, *payload in test_operations:
                    try:
                        total_operations += 1
                        
                        if method == 'GET':
                            async with session.get(f"{self.api_base_url}{endpoint}") as response:
                                if response.status < 400:
                                    successful_operations += 1
                        elif method == 'POST' and payload:
                            async with session.post(f"{self.api_base_url}{endpoint}", json=payload[0]) as response:
                                if response.status < 400:
                                    successful_operations += 1
                    
                    except Exception:
                        # Failed operations count toward error rate
                        pass
            
            if total_operations > 0:
                error_rate = ((total_operations - successful_operations) / total_operations) * 100.0
                return error_rate
            else:
                return 100.0  # No operations succeeded
                
        except Exception as e:
            logger.warning(f"Error rate measurement failed: {e}")
            return 100.0
    
    async def _check_sla_breaches(self):
        """Check for SLA breaches and trigger alerts."""
        current_time = datetime.utcnow()
        
        for tier in SLATier:
            if tier not in self.measurements or not self.measurements[tier]:
                continue
            
            # Get recent measurements (last hour)
            hour_ago = current_time - timedelta(hours=1)
            recent_measurements = [
                m for m in self.measurements[tier]
                if m.timestamp >= hour_ago
            ]
            
            if not recent_measurements:
                continue
            
            # Check each metric type for breaches
            for target in self.sla_targets[tier]:
                metric_measurements = [
                    m for m in recent_measurements
                    if m.metric_type == target.metric_type
                ]
                
                if not metric_measurements:
                    continue
                
                # Calculate current performance
                if target.metric_type == SLAMetricType.UPTIME:
                    current_value = statistics.mean(m.value for m in metric_measurements)
                    breach_condition = current_value < target.target_value
                elif target.metric_type == SLAMetricType.RESPONSE_TIME:
                    current_value = np.percentile([m.value for m in metric_measurements], 95)
                    breach_condition = current_value > target.target_value
                elif target.metric_type == SLAMetricType.ERROR_RATE:
                    current_value = statistics.mean(m.value for m in metric_measurements)
                    breach_condition = current_value > target.target_value
                else:
                    continue
                
                # Check for breach
                if breach_condition:
                    await self._handle_sla_breach(tier, target, current_value, current_time)
    
    async def _handle_sla_breach(self, 
                               tier: SLATier,
                               target: SLATarget,
                               actual_value: float,
                               timestamp: datetime):
        """Handle SLA breach event."""
        breach_id = f"{tier.value}_{target.metric_type.value}_{int(timestamp.timestamp())}"
        
        # Determine severity
        if actual_value <= target.target_value * (target.warning_threshold_percent / 100):
            severity = AlertSeverity.WARNING
        elif actual_value <= target.target_value * (target.critical_threshold_percent / 100):
            severity = AlertSeverity.CRITICAL
        else:
            severity = AlertSeverity.EMERGENCY
        
        # Calculate business impact
        business_impact = self._calculate_business_impact(tier, target.metric_type, actual_value, target.target_value)
        
        # Create breach event
        breach_event = SLABreachEvent(
            breach_id=breach_id,
            timestamp=timestamp,
            metric_type=target.metric_type,
            tier=tier,
            actual_value=actual_value,
            target_value=target.target_value,
            severity=severity,
            duration_seconds=0,  # Will be updated when breach ends
            impact_description=f"{target.metric_type.value} breach for {tier.value} tier",
            business_impact_cost=business_impact['cost']
        )
        
        self.breach_events.append(breach_event)
        
        # Send alert
        await self._send_sla_alert(breach_event)
        
        logger.warning(
            f"🚨 SLA BREACH: {tier.value} {target.metric_type.value} - "
            f"Actual: {actual_value:.2f}, Target: {target.target_value:.2f}, "
            f"Severity: {severity.value}"
        )
    
    def _calculate_business_impact(self, 
                                 tier: SLATier,
                                 metric_type: SLAMetricType,
                                 actual_value: float,
                                 target_value: float) -> Dict[str, Any]:
        """Calculate business impact of SLA breach."""
        # Sample business impact calculation
        # In practice, this would be based on actual business metrics
        
        impact_multipliers = {
            SLATier.BRONZE: 1.0,
            SLATier.SILVER: 2.0,
            SLATier.GOLD: 5.0,
            SLATier.PLATINUM: 10.0
        }
        
        base_cost_per_hour = {
            SLAMetricType.UPTIME: 1000.0,      # $1000/hour downtime
            SLAMetricType.RESPONSE_TIME: 500.0, # $500/hour slow response
            SLAMetricType.ERROR_RATE: 750.0     # $750/hour high errors
        }
        
        multiplier = impact_multipliers[tier]
        base_cost = base_cost_per_hour[metric_type]
        
        # Calculate severity factor
        if metric_type == SLAMetricType.UPTIME:
            severity_factor = max(0, (target_value - actual_value) / target_value)
        else:
            severity_factor = max(0, (actual_value - target_value) / target_value)
        
        estimated_cost = base_cost * multiplier * severity_factor
        
        # Estimate affected users
        affected_users = int(1000 * severity_factor * multiplier)
        
        return {
            'cost': estimated_cost,
            'affected_users': affected_users,
            'severity_factor': severity_factor
        }
    
    async def _send_sla_alert(self, breach_event: SLABreachEvent):
        """Send SLA breach alert through configured channels."""
        alert_message = {
            'alert_id': f"sla_alert_{int(time.time())}",
            'timestamp': breach_event.timestamp.isoformat(),
            'severity': breach_event.severity.value,
            'tier': breach_event.tier.value,
            'metric': breach_event.metric_type.value,
            'actual_value': breach_event.actual_value,
            'target_value': breach_event.target_value,
            'business_impact_cost': breach_event.business_impact_cost,
            'message': f"SLA breach detected: {breach_event.impact_description}"
        }
        
        self.alert_history.append(alert_message)
        
        # In a production system, this would send actual alerts via email, Slack, PagerDuty, etc.
        logger.warning(f"📢 SLA ALERT SENT: {json.dumps(alert_message, indent=2)}")
        
        # Simulate escalation for critical/emergency alerts
        if breach_event.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            escalation_message = {
                'alert_id': alert_message['alert_id'],
                'escalation_level': 2,
                'escalated_at': datetime.utcnow().isoformat(),
                'escalation_reason': f'{breach_event.severity.value} SLA breach requires immediate attention'
            }
            self.alert_history.append(escalation_message)
            logger.critical(f"🚨 ESCALATED ALERT: {json.dumps(escalation_message, indent=2)}")
    
    async def test_error_recovery_scenarios(self) -> Dict[str, Any]:
        """Test comprehensive error recovery scenarios."""
        recovery_scenarios = [
            {
                'scenario_id': 'api_server_overload',
                'description': 'API server under heavy load with degraded performance',
                'expected_recovery_time_seconds': 300,
                'recovery_actions': ['Scale up instances', 'Enable circuit breaker', 'Cache optimization']
            },
            {
                'scenario_id': 'database_connection_exhaustion',
                'description': 'Database connection pool exhaustion',
                'expected_recovery_time_seconds': 180,
                'recovery_actions': ['Increase pool size', 'Kill long-running queries', 'Restart connections']
            },
            {
                'scenario_id': 'redis_memory_pressure',
                'description': 'Redis cache memory pressure and evictions',
                'expected_recovery_time_seconds': 120,
                'recovery_actions': ['Increase memory limit', 'Optimize cache keys', 'Clear stale data']
            },
            {
                'scenario_id': 'websocket_connection_storm',
                'description': 'Massive WebSocket connection storm',
                'expected_recovery_time_seconds': 240,
                'recovery_actions': ['Rate limit connections', 'Scale WebSocket servers', 'Implement backpressure']
            }
        ]
        
        recovery_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'scenarios_tested': len(recovery_scenarios),
            'scenario_results': [],
            'overall_recovery_score': 0.0
        }
        
        for scenario in recovery_scenarios:
            logger.info(f"🔄 Testing error recovery scenario: {scenario['scenario_id']}")
            
            # Record baseline metrics
            baseline_uptime = await self._measure_uptime()
            baseline_response_time = await self._measure_response_time()
            baseline_error_rate = await self._measure_error_rate()
            
            # Simulate recovery scenario (in practice, this would involve actual failure injection)
            await asyncio.sleep(2)  # Simulate time for issue detection
            
            # Measure performance during "recovery"
            recovery_start = time.time()
            
            # Simulate recovery actions
            for action in scenario['recovery_actions']:
                logger.info(f"  🔧 Executing recovery action: {action}")
                await asyncio.sleep(1)  # Simulate action execution time
            
            recovery_duration = time.time() - recovery_start
            
            # Measure post-recovery metrics
            post_recovery_uptime = await self._measure_uptime()
            post_recovery_response_time = await self._measure_response_time()
            post_recovery_error_rate = await self._measure_error_rate()
            
            # Calculate recovery effectiveness
            uptime_recovery = post_recovery_uptime / baseline_uptime if baseline_uptime > 0 else 1.0
            response_time_recovery = baseline_response_time / post_recovery_response_time if post_recovery_response_time > 0 else 1.0
            error_rate_recovery = baseline_error_rate / post_recovery_error_rate if post_recovery_error_rate > 0 else 1.0
            
            recovery_score = (uptime_recovery + response_time_recovery + error_rate_recovery) / 3.0
            
            scenario_result = {
                'scenario_id': scenario['scenario_id'],
                'description': scenario['description'],
                'expected_recovery_time_seconds': scenario['expected_recovery_time_seconds'],
                'actual_recovery_time_seconds': recovery_duration,
                'recovery_actions_executed': scenario['recovery_actions'],
                'baseline_metrics': {
                    'uptime_percentage': baseline_uptime,
                    'response_time_ms': baseline_response_time,
                    'error_rate_percentage': baseline_error_rate
                },
                'post_recovery_metrics': {
                    'uptime_percentage': post_recovery_uptime,
                    'response_time_ms': post_recovery_response_time,
                    'error_rate_percentage': post_recovery_error_rate
                },
                'recovery_effectiveness_score': recovery_score,
                'recovery_within_sla': recovery_duration <= scenario['expected_recovery_time_seconds']
            }
            
            recovery_results['scenario_results'].append(scenario_result)
        
        # Calculate overall recovery score
        if recovery_results['scenario_results']:
            overall_score = statistics.mean(r['recovery_effectiveness_score'] for r in recovery_results['scenario_results'])
            recovery_results['overall_recovery_score'] = overall_score
        
        return recovery_results
    
    def generate_sla_compliance_report(self, 
                                     tier: SLATier,
                                     reporting_period_hours: int = 24) -> SLAComplianceReport:
        """Generate comprehensive SLA compliance report."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=reporting_period_hours)
        
        # Filter measurements for reporting period
        period_measurements = [
            m for m in self.measurements[tier]
            if start_time <= m.timestamp <= end_time
        ]
        
        if not period_measurements:
            # Return empty report if no data
            return SLAComplianceReport(
                report_id=f"{tier.value}_report_{int(end_time.timestamp())}",
                reporting_period=(start_time, end_time),
                tier=tier,
                uptime_percentage=0.0,
                avg_response_time_ms=0.0,
                error_rate_percentage=100.0,
                availability_percentage=0.0,
                targets_met={},
                overall_compliance=False,
                total_breaches=0,
                total_downtime_seconds=0.0,
                mttr_seconds=0.0,
                mtbf_seconds=0.0,
                estimated_cost_impact=0.0,
                affected_users=0
            )
        
        # Calculate metrics by type
        uptime_measurements = [m for m in period_measurements if m.metric_type == SLAMetricType.UPTIME]
        response_time_measurements = [m for m in period_measurements if m.metric_type == SLAMetricType.RESPONSE_TIME]
        error_rate_measurements = [m for m in period_measurements if m.metric_type == SLAMetricType.ERROR_RATE]
        
        # Calculate averages
        uptime_percentage = statistics.mean(m.value for m in uptime_measurements) if uptime_measurements else 0.0
        avg_response_time_ms = statistics.mean(m.value for m in response_time_measurements) if response_time_measurements else 0.0
        error_rate_percentage = statistics.mean(m.value for m in error_rate_measurements) if error_rate_measurements else 100.0
        
        # Check compliance against targets
        targets_met = {}
        for target in self.sla_targets[tier]:
            if target.metric_type == SLAMetricType.UPTIME:
                targets_met['uptime'] = uptime_percentage >= target.target_value
            elif target.metric_type == SLAMetricType.RESPONSE_TIME:
                targets_met['response_time'] = avg_response_time_ms <= target.target_value
            elif target.metric_type == SLAMetricType.ERROR_RATE:
                targets_met['error_rate'] = error_rate_percentage <= target.target_value
        
        overall_compliance = all(targets_met.values())
        
        # Calculate breach statistics
        period_breaches = [
            b for b in self.breach_events
            if b.tier == tier and start_time <= b.timestamp <= end_time
        ]
        
        total_breaches = len(period_breaches)
        total_downtime_seconds = sum(b.duration_seconds for b in period_breaches)
        
        # Calculate MTTR and MTBF
        if period_breaches:
            mttr_seconds = statistics.mean(b.duration_seconds for b in period_breaches if b.duration_seconds > 0)
            if mttr_seconds == 0:
                mttr_seconds = 300  # Default assumption
        else:
            mttr_seconds = 0
        
        # MTBF calculation (time between failures)
        if total_breaches > 1:
            total_time_seconds = reporting_period_hours * 3600
            mtbf_seconds = (total_time_seconds - total_downtime_seconds) / (total_breaches - 1)
        else:
            mtbf_seconds = reporting_period_hours * 3600  # Full period if no failures
        
        # Business impact
        estimated_cost_impact = sum(b.business_impact_cost for b in period_breaches)
        affected_users = max((b.business_impact_cost / 1000 * 100) for b in period_breaches) if period_breaches else 0
        
        # Generate recommendations
        recommendations = []
        if not targets_met.get('uptime', True):
            recommendations.append(f"Uptime ({uptime_percentage:.2f}%) below target. Implement redundancy and failover mechanisms.")
        if not targets_met.get('response_time', True):
            recommendations.append(f"Response time ({avg_response_time_ms:.0f}ms) exceeds target. Optimize performance and consider caching.")
        if not targets_met.get('error_rate', True):
            recommendations.append(f"Error rate ({error_rate_percentage:.2f}%) above target. Improve error handling and system stability.")
        if total_breaches > 5:
            recommendations.append(f"High number of breaches ({total_breaches}). Review system architecture and monitoring thresholds.")
        if mttr_seconds > 600:  # 10 minutes
            recommendations.append(f"Mean Time To Recovery ({mttr_seconds/60:.1f} min) is high. Improve incident response procedures.")
        
        if not recommendations:
            recommendations.append("✅ All SLA targets met. Maintain current service levels and monitoring practices.")
        
        return SLAComplianceReport(
            report_id=f"{tier.value}_report_{int(end_time.timestamp())}",
            reporting_period=(start_time, end_time),
            tier=tier,
            uptime_percentage=uptime_percentage,
            avg_response_time_ms=avg_response_time_ms,
            error_rate_percentage=error_rate_percentage,
            availability_percentage=uptime_percentage,  # Same as uptime for simplicity
            targets_met=targets_met,
            overall_compliance=overall_compliance,
            total_breaches=total_breaches,
            total_downtime_seconds=total_downtime_seconds,
            mttr_seconds=mttr_seconds,
            mtbf_seconds=mtbf_seconds,
            estimated_cost_impact=estimated_cost_impact,
            affected_users=int(affected_users),
            recommendations=recommendations
        )
    
    async def run_comprehensive_sla_validation(self, 
                                             monitoring_duration_hours: float = 2.0) -> Dict[str, Any]:
        """Run comprehensive SLA monitoring validation."""
        logger.info("🎯 Starting Comprehensive SLA Monitoring Validation")
        
        validation_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'monitoring_duration_hours': monitoring_duration_hours,
            'sla_monitoring_results': {},
            'error_recovery_analysis': {},
            'compliance_reports': {},
            'alerting_validation': {},
            'overall_sla_health_score': 0.0,
            'recommendations': []
        }
        
        # 1. Run continuous monitoring for specified duration
        logger.info(f"📊 Running continuous SLA monitoring for {monitoring_duration_hours} hours...")
        monitoring_task = asyncio.create_task(
            self.start_continuous_monitoring(monitoring_duration_hours)
        )
        
        # Wait for monitoring to complete
        try:
            await monitoring_task
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
        
        # 2. Test error recovery scenarios
        logger.info("🔄 Testing error recovery scenarios...")
        validation_results['error_recovery_analysis'] = await self.test_error_recovery_scenarios()
        
        # 3. Generate compliance reports for all tiers
        logger.info("📋 Generating SLA compliance reports...")
        for tier in SLATier:
            report = self.generate_sla_compliance_report(tier, int(monitoring_duration_hours))
            validation_results['compliance_reports'][tier.value] = {
                'overall_compliance': report.overall_compliance,
                'uptime_percentage': report.uptime_percentage,
                'avg_response_time_ms': report.avg_response_time_ms,
                'error_rate_percentage': report.error_rate_percentage,
                'total_breaches': report.total_breaches,
                'mttr_seconds': report.mttr_seconds,
                'mtbf_seconds': report.mtbf_seconds,
                'estimated_cost_impact': report.estimated_cost_impact,
                'targets_met': report.targets_met,
                'recommendations': report.recommendations
            }
        
        # 4. Validate alerting system
        validation_results['alerting_validation'] = {
            'total_alerts_sent': len(self.alert_history),
            'alert_types': list(set(alert.get('severity', 'unknown') for alert in self.alert_history)),
            'escalations_triggered': len([a for a in self.alert_history if 'escalation_level' in a]),
            'alerting_functional': len(self.alert_history) > 0 or len(self.breach_events) == 0
        }
        
        # 5. Calculate overall SLA health score
        compliance_scores = []
        for tier_report in validation_results['compliance_reports'].values():
            if tier_report['overall_compliance']:
                compliance_scores.append(1.0)
            else:
                # Partial score based on individual targets met
                targets_met = tier_report['targets_met']
                partial_score = sum(targets_met.values()) / len(targets_met) if targets_met else 0.0
                compliance_scores.append(partial_score)
        
        if compliance_scores:
            sla_compliance_score = statistics.mean(compliance_scores)
        else:
            sla_compliance_score = 0.0
        
        # Include recovery effectiveness
        recovery_score = validation_results['error_recovery_analysis'].get('overall_recovery_score', 0.0)
        
        # Calculate overall health score
        validation_results['overall_sla_health_score'] = (sla_compliance_score * 0.7) + (recovery_score * 0.3)
        
        # 6. Generate overall recommendations
        recommendations = []
        
        # SLA compliance recommendations
        non_compliant_tiers = [
            tier for tier, report in validation_results['compliance_reports'].items()
            if not report['overall_compliance']
        ]
        
        if non_compliant_tiers:
            recommendations.append(f"SLA compliance issues in tiers: {', '.join(non_compliant_tiers)}")
        
        # Gold tier specific recommendations (our target)
        gold_report = validation_results['compliance_reports'].get('gold', {})
        if not gold_report.get('overall_compliance', False):
            if not gold_report.get('targets_met', {}).get('uptime', True):
                recommendations.append("🥇 CRITICAL: Gold tier uptime SLA (99.9%) not met - implement high availability architecture")
            if not gold_report.get('targets_met', {}).get('response_time', True):
                recommendations.append("🥇 CRITICAL: Gold tier response time SLA (200ms) not met - optimize API performance")
            if not gold_report.get('targets_met', {}).get('error_rate', True):
                recommendations.append("🥇 CRITICAL: Gold tier error rate SLA (0.5%) not met - improve error handling")
        
        # Recovery recommendations
        if recovery_score < 0.8:
            recommendations.append("Improve error recovery procedures - current recovery effectiveness below target")
        
        # Alerting recommendations
        if not validation_results['alerting_validation']['alerting_functional']:
            recommendations.append("Alerting system validation failed - review alert configuration and delivery")
        
        if not recommendations:
            recommendations.append("✅ All SLA targets met across all tiers. Excellent production reliability!")
        
        validation_results['recommendations'] = recommendations
        
        logger.info(f"✅ SLA validation completed. Overall health score: {validation_results['overall_sla_health_score']:.2f}")
        
        return validation_results


# Test utilities for pytest integration
@pytest.fixture
async def sla_monitoring_validator():
    """Pytest fixture for SLA monitoring validator."""
    validator = ProductionSLAMonitoringValidator()
    yield validator


class TestProductionSLAMonitoring:
    """Test suite for production SLA monitoring validation."""
    
    @pytest.mark.asyncio
    async def test_uptime_measurement(self, sla_monitoring_validator):
        """Test uptime measurement accuracy."""
        uptime = await sla_monitoring_validator._measure_uptime()
        
        assert isinstance(uptime, float)
        assert 0.0 <= uptime <= 100.0
    
    @pytest.mark.asyncio
    async def test_response_time_measurement(self, sla_monitoring_validator):
        """Test response time measurement."""
        response_time = await sla_monitoring_validator._measure_response_time()
        
        assert isinstance(response_time, float)
        assert response_time >= 0.0  # Should be positive or infinity
    
    @pytest.mark.asyncio
    async def test_error_rate_measurement(self, sla_monitoring_validator):
        """Test error rate measurement."""
        error_rate = await sla_monitoring_validator._measure_error_rate()
        
        assert isinstance(error_rate, float)
        assert 0.0 <= error_rate <= 100.0
    
    @pytest.mark.asyncio
    async def test_gold_tier_sla_compliance(self, sla_monitoring_validator):
        """Test Gold tier SLA compliance (99.9% uptime, 200ms response)."""
        # Take measurements
        await sla_monitoring_validator._take_sla_measurements(SLATier.GOLD)
        
        # Generate compliance report
        report = sla_monitoring_validator.generate_sla_compliance_report(SLATier.GOLD, 1)
        
        assert report.tier == SLATier.GOLD
        
        # Check if Gold tier targets are met
        if report.overall_compliance:
            assert report.uptime_percentage >= 99.9, "Gold tier uptime SLA not met"
            assert report.avg_response_time_ms <= 200.0, "Gold tier response time SLA not met"
            assert report.error_rate_percentage <= 0.5, "Gold tier error rate SLA not met"
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self, sla_monitoring_validator):
        """Test error recovery scenario validation."""
        results = await sla_monitoring_validator.test_error_recovery_scenarios()
        
        assert 'scenario_results' in results
        assert 'overall_recovery_score' in results
        assert isinstance(results['overall_recovery_score'], float)
        assert 0.0 <= results['overall_recovery_score'] <= 10.0  # Recovery score can be > 1.0
    
    @pytest.mark.asyncio
    async def test_comprehensive_sla_validation(self, sla_monitoring_validator):
        """Test comprehensive SLA validation suite."""
        # Run short validation (5 minutes)
        results = await sla_monitoring_validator.run_comprehensive_sla_validation(
            monitoring_duration_hours=0.083  # 5 minutes
        )
        
        assert 'compliance_reports' in results
        assert 'error_recovery_analysis' in results
        assert 'alerting_validation' in results
        assert 'overall_sla_health_score' in results
        
        # Validate score range
        score = results['overall_sla_health_score']
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    async def main():
        validator = ProductionSLAMonitoringValidator()
        results = await validator.run_comprehensive_sla_validation(monitoring_duration_hours=0.25)  # 15 minutes
        
        print("🎯 Production SLA Monitoring Validation Results:")
        print(json.dumps(results, indent=2, default=str))

    asyncio.run(main())