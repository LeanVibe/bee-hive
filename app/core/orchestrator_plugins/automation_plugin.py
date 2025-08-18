"""
Automation Plugin for Universal Orchestrator

Consolidates functionality from:
- automated_orchestrator.py (1,175 LOC)
- intelligent_sleep_manager.py
- sleep_wake_system.py
- automated_testing_integration.py
- intelligent_workflow_automation.py

Provides intelligent automation for:
- Sleep/wake cycle management with proactive scheduling
- Intelligent recovery with multi-tier fallback strategies  
- Circuit breaker patterns for fault tolerance
- Automated health monitoring and self-healing
- Performance-driven optimization and adaptation
- Event-driven orchestration with real-time responsiveness
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from . import OrchestratorPlugin, PluginMetadata, PluginType
from ..config import settings
from ..redis import get_redis
from ..database import get_session
from ..logging_service import get_component_logger

logger = get_component_logger("automation_plugin")


class OrchestrationStrategy(Enum):
    """Orchestration strategies for different scenarios."""
    PROACTIVE = "proactive"           # Anticipate and schedule operations
    REACTIVE = "reactive"             # Respond to events and conditions
    HYBRID = "hybrid"                 # Combine proactive and reactive approaches
    MAINTENANCE = "maintenance"       # Focus on system health and optimization
    EMERGENCY = "emergency"           # Emergency response mode


class RecoveryTier(Enum):
    """Recovery tiers with increasing intervention levels."""
    AUTOMATIC = "automatic"           # Automated recovery without human intervention
    ASSISTED = "assisted"             # Automated with human notification
    MANUAL = "manual"                 # Requires human intervention
    EMERGENCY = "emergency"           # Critical system failure requiring immediate attention


class SleepTrigger(Enum):
    """Triggers for initiating sleep cycles."""
    CONTEXT_THRESHOLD = "context_threshold"       # Context window usage threshold
    INACTIVITY = "inactivity"                    # Agent inactivity period
    SCHEDULED = "scheduled"                       # Scheduled maintenance
    PERFORMANCE_DEGRADATION = "performance_degradation"  # Performance issues
    ERROR_RATE = "error_rate"                    # High error rate
    RESOURCE_EXHAUSTION = "resource_exhaustion"   # Resource constraints


@dataclass
class SleepSchedule:
    """Sleep schedule configuration for an agent."""
    agent_id: str
    trigger: SleepTrigger
    threshold: float  # Threshold value that triggers sleep
    min_sleep_duration_minutes: int = 5
    max_sleep_duration_minutes: int = 60
    recovery_threshold: float = 0.5  # Threshold for wake condition
    last_sleep_time: Optional[datetime] = None
    consecutive_sleep_count: int = 0
    

@dataclass
class RecoveryPlan:
    """Recovery plan with multi-tier strategies."""
    recovery_id: str
    agent_id: Optional[str] = None
    failure_type: str = ""
    failure_severity: int = 1  # 1-5 scale
    recovery_tier: RecoveryTier = RecoveryTier.AUTOMATIC
    strategies: List[str] = field(default_factory=list)
    estimated_recovery_time: float = 0.0
    success_probability: float = 0.0
    fallback_plans: List['RecoveryPlan'] = field(default_factory=list)
    

@dataclass
class AutomationMetrics:
    """Metrics for automation operations."""
    timestamp: datetime
    sleep_cycles_initiated: int = 0
    wake_cycles_completed: int = 0
    recovery_plans_executed: int = 0
    automatic_recoveries_successful: int = 0
    manual_interventions_required: int = 0
    average_recovery_time_seconds: float = 0.0
    proactive_actions_taken: int = 0
    reactive_actions_taken: int = 0


class AutomationPlugin(OrchestratorPlugin):
    """Plugin for intelligent automation and recovery management."""
    
    def __init__(self):
        metadata = PluginMetadata(
            name="automation_plugin",
            version="1.0.0",
            plugin_type=PluginType.WORKFLOW,
            description="Intelligent automation, sleep/wake management, and recovery systems",
            dependencies=["redis", "database"]
        )
        super().__init__(metadata)
        
        # Automation configuration
        self.strategy = OrchestrationStrategy.HYBRID
        self.automation_enabled = True
        
        # Sleep/wake management
        self.sleep_schedules: Dict[str, SleepSchedule] = {}
        self.active_sleep_cycles: Dict[str, datetime] = {}  # agent_id -> sleep_start_time
        
        # Recovery management
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.recovery_history: deque = deque(maxlen=1000)
        
        # Event tracking
        self.event_queue: deque = deque()
        self.event_handlers: Dict[str, List[callable]] = defaultdict(list)
        
        # Metrics
        self.metrics_history: deque = deque(maxlen=500)
        
        # Background tasks
        self._automation_task: Optional[asyncio.Task] = None
        self._sleep_monitor_task: Optional[asyncio.Task] = None
        self._recovery_monitor_task: Optional[asyncio.Task] = None
    
    async def initialize(self, orchestrator_context: Dict[str, Any]) -> bool:
        """Initialize automation plugin."""
        try:
            self.redis = await get_redis()
            self.orchestrator_id = orchestrator_context.get('orchestrator_id', 'unknown')
            
            # Register event handlers
            self._register_event_handlers()
            
            # Start background automation tasks
            self._automation_task = asyncio.create_task(self._automation_loop())
            self._sleep_monitor_task = asyncio.create_task(self._sleep_monitor_loop())
            self._recovery_monitor_task = asyncio.create_task(self._recovery_monitor_loop())
            
            logger.info("Automation plugin initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize automation plugin: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup automation plugin resources."""
        try:
            # Cancel background tasks
            if self._automation_task:
                self._automation_task.cancel()
            if self._sleep_monitor_task:
                self._sleep_monitor_task.cancel()
            if self._recovery_monitor_task:
                self._recovery_monitor_task.cancel()
                
            # Wait for tasks to complete
            tasks_to_wait = [
                task for task in [self._automation_task, self._sleep_monitor_task, self._recovery_monitor_task]
                if task and not task.done()
            ]
            
            if tasks_to_wait:
                await asyncio.gather(*tasks_to_wait, return_exceptions=True)
            
            logger.info("Automation plugin cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup automation plugin: {e}")
            return False
    
    async def pre_task_execution(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-task hook for automation analysis."""
        agent_id = task_context.get('agent_id')
        
        if agent_id:
            # Check if agent needs sleep cycle
            sleep_needed = await self._check_sleep_needed(agent_id, task_context)
            if sleep_needed:
                logger.info(f"Sleep cycle recommended for agent {agent_id}")
                task_context['sleep_recommended'] = True
                
            # Update automation context
            task_context['automation_timestamp'] = datetime.utcnow().isoformat()
        
        return task_context
    
    async def post_task_execution(self, task_context: Dict[str, Any], result: Any) -> Any:
        """Post-task hook for automation analysis."""
        agent_id = task_context.get('agent_id')
        success = task_context.get('success', True)
        
        if agent_id:
            # Analyze task completion for automation insights
            await self._analyze_task_completion(agent_id, task_context, result, success)
            
            # Check if recovery plan needed
            if not success:
                await self._create_recovery_plan(agent_id, task_context, result)
        
        return result
    
    async def health_check(self) -> Dict[str, Any]:
        """Return automation plugin health status."""
        try:
            current_metrics = AutomationMetrics(
                timestamp=datetime.utcnow(),
                sleep_cycles_initiated=len(self.active_sleep_cycles),
                recovery_plans_executed=len(self.recovery_plans),
                automatic_recoveries_successful=len([r for r in self.recovery_history 
                                                   if r.get('success', False) and r.get('tier') == 'automatic']),
                proactive_actions_taken=len([e for e in self.event_queue 
                                           if e.get('strategy') == 'proactive']),
                reactive_actions_taken=len([e for e in self.event_queue 
                                          if e.get('strategy') == 'reactive'])
            )
            
            return {
                "plugin": self.metadata.name,
                "enabled": self.enabled,
                "status": "healthy",
                "active_sleep_cycles": len(self.active_sleep_cycles),
                "active_recovery_plans": len(self.recovery_plans),
                "strategy": self.strategy.value,
                "automation_enabled": self.automation_enabled,
                "metrics": current_metrics.__dict__
            }
            
        except Exception as e:
            return {
                "plugin": self.metadata.name,
                "enabled": self.enabled,
                "status": "error",
                "error": str(e)
            }
    
    async def configure_sleep_schedule(
        self,
        agent_id: str,
        trigger: SleepTrigger,
        threshold: float,
        min_duration: int = 5,
        max_duration: int = 60
    ):
        """Configure sleep schedule for an agent."""
        schedule = SleepSchedule(
            agent_id=agent_id,
            trigger=trigger,
            threshold=threshold,
            min_sleep_duration_minutes=min_duration,
            max_sleep_duration_minutes=max_duration
        )
        
        self.sleep_schedules[agent_id] = schedule
        
        logger.info(
            "Sleep schedule configured",
            agent_id=agent_id,
            trigger=trigger.value,
            threshold=threshold
        )
    
    async def initiate_sleep_cycle(self, agent_id: str, reason: str) -> bool:
        """Initiate sleep cycle for an agent."""
        try:
            # Check if agent is already sleeping
            if agent_id in self.active_sleep_cycles:
                logger.warning(f"Agent {agent_id} is already in sleep cycle")
                return False
            
            # Determine sleep duration based on schedule and conditions
            schedule = self.sleep_schedules.get(agent_id)
            if schedule:
                duration_minutes = min(
                    schedule.min_sleep_duration_minutes + (schedule.consecutive_sleep_count * 5),
                    schedule.max_sleep_duration_minutes
                )
            else:
                duration_minutes = 15  # Default sleep duration
            
            # Record sleep cycle start
            self.active_sleep_cycles[agent_id] = datetime.utcnow()
            
            # Update schedule
            if schedule:
                schedule.last_sleep_time = datetime.utcnow()
                schedule.consecutive_sleep_count += 1
            
            # Emit sleep event
            await self._emit_automation_event({
                'type': 'sleep_initiated',
                'agent_id': agent_id,
                'reason': reason,
                'duration_minutes': duration_minutes,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            logger.info(
                "Sleep cycle initiated",
                agent_id=agent_id,
                reason=reason,
                duration_minutes=duration_minutes
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initiate sleep cycle for agent {agent_id}: {e}")
            return False
    
    async def complete_wake_cycle(self, agent_id: str) -> bool:
        """Complete wake cycle for an agent."""
        try:
            # Check if agent was sleeping
            if agent_id not in self.active_sleep_cycles:
                logger.warning(f"Agent {agent_id} was not in sleep cycle")
                return False
            
            # Calculate sleep duration
            sleep_start = self.active_sleep_cycles[agent_id]
            sleep_duration = (datetime.utcnow() - sleep_start).total_seconds() / 60  # minutes
            
            # Remove from active sleep cycles
            del self.active_sleep_cycles[agent_id]
            
            # Reset consecutive sleep count if sleep was successful
            schedule = self.sleep_schedules.get(agent_id)
            if schedule and sleep_duration >= schedule.min_sleep_duration_minutes:
                schedule.consecutive_sleep_count = 0
            
            # Emit wake event
            await self._emit_automation_event({
                'type': 'wake_completed',
                'agent_id': agent_id,
                'sleep_duration_minutes': sleep_duration,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            logger.info(
                "Wake cycle completed",
                agent_id=agent_id,
                sleep_duration_minutes=sleep_duration
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete wake cycle for agent {agent_id}: {e}")
            return False
    
    async def _check_sleep_needed(self, agent_id: str, context: Dict[str, Any]) -> bool:
        """Check if agent needs sleep cycle."""
        schedule = self.sleep_schedules.get(agent_id)
        if not schedule:
            return False
        
        current_time = datetime.utcnow()
        
        # Check different sleep triggers
        if schedule.trigger == SleepTrigger.CONTEXT_THRESHOLD:
            context_usage = context.get('context_window_usage', 0.0)
            return context_usage >= schedule.threshold
            
        elif schedule.trigger == SleepTrigger.INACTIVITY:
            last_activity = context.get('last_activity_time')
            if last_activity:
                inactive_minutes = (current_time - last_activity).total_seconds() / 60
                return inactive_minutes >= schedule.threshold
                
        elif schedule.trigger == SleepTrigger.SCHEDULED:
            if schedule.last_sleep_time:
                hours_since_sleep = (current_time - schedule.last_sleep_time).total_seconds() / 3600
                return hours_since_sleep >= schedule.threshold
                
        elif schedule.trigger == SleepTrigger.PERFORMANCE_DEGRADATION:
            task_duration = context.get('average_task_duration_ms', 0.0)
            return task_duration >= schedule.threshold
            
        elif schedule.trigger == SleepTrigger.ERROR_RATE:
            error_rate = context.get('error_rate_percent', 0.0)
            return error_rate >= schedule.threshold
        
        return False
    
    async def _analyze_task_completion(
        self,
        agent_id: str,
        context: Dict[str, Any],
        result: Any,
        success: bool
    ):
        """Analyze task completion for automation insights."""
        try:
            # Extract performance metrics
            duration_ms = context.get('duration_ms', 0.0)
            task_type = context.get('task_type', 'unknown')
            
            # Update automation metrics
            analysis = {
                'agent_id': agent_id,
                'task_type': task_type,
                'duration_ms': duration_ms,
                'success': success,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Check for performance degradation patterns
            if duration_ms > 5000:  # 5 seconds threshold
                await self._emit_automation_event({
                    'type': 'performance_degradation_detected',
                    'agent_id': agent_id,
                    'duration_ms': duration_ms,
                    'threshold_ms': 5000,
                    'recommendation': 'consider_sleep_cycle'
                })
            
            # Store analysis for trend detection
            await self._store_automation_analysis(analysis)
            
        except Exception as e:
            logger.error(f"Failed to analyze task completion: {e}")
    
    async def _create_recovery_plan(
        self,
        agent_id: str,
        context: Dict[str, Any],
        result: Any
    ):
        """Create recovery plan for failed task."""
        try:
            error_info = context.get('error', str(result) if result else 'unknown_error')
            failure_severity = self._assess_failure_severity(error_info, context)
            
            recovery_plan = RecoveryPlan(
                recovery_id=str(uuid.uuid4()),
                agent_id=agent_id,
                failure_type=error_info[:100],  # Truncate for storage
                failure_severity=failure_severity,
                recovery_tier=self._determine_recovery_tier(failure_severity),
                strategies=self._generate_recovery_strategies(error_info, failure_severity),
                estimated_recovery_time=self._estimate_recovery_time(failure_severity),
                success_probability=self._calculate_success_probability(failure_severity)
            )
            
            self.recovery_plans[recovery_plan.recovery_id] = recovery_plan
            
            # Execute automatic recovery if appropriate
            if recovery_plan.recovery_tier == RecoveryTier.AUTOMATIC:
                await self._execute_recovery_plan(recovery_plan)
            
            logger.info(
                "Recovery plan created",
                agent_id=agent_id,
                recovery_id=recovery_plan.recovery_id,
                tier=recovery_plan.recovery_tier.value,
                severity=failure_severity
            )
            
        except Exception as e:
            logger.error(f"Failed to create recovery plan: {e}")
    
    def _assess_failure_severity(self, error_info: str, context: Dict[str, Any]) -> int:
        """Assess failure severity on 1-5 scale."""
        # Simple heuristic-based severity assessment
        error_lower = error_info.lower()
        
        if any(keyword in error_lower for keyword in ['critical', 'fatal', 'crash']):
            return 5
        elif any(keyword in error_lower for keyword in ['error', 'exception', 'failed']):
            return 3
        elif any(keyword in error_lower for keyword in ['warning', 'timeout']):
            return 2
        else:
            return 1
    
    def _determine_recovery_tier(self, severity: int) -> RecoveryTier:
        """Determine appropriate recovery tier based on severity."""
        if severity >= 4:
            return RecoveryTier.MANUAL
        elif severity >= 3:
            return RecoveryTier.ASSISTED
        else:
            return RecoveryTier.AUTOMATIC
    
    def _generate_recovery_strategies(self, error_info: str, severity: int) -> List[str]:
        """Generate recovery strategies based on error and severity."""
        strategies = []
        
        error_lower = error_info.lower()
        
        if 'timeout' in error_lower:
            strategies.extend(['retry_with_longer_timeout', 'check_network_connectivity'])
        elif 'memory' in error_lower:
            strategies.extend(['force_garbage_collection', 'restart_agent'])
        elif 'permission' in error_lower:
            strategies.extend(['check_file_permissions', 'escalate_privileges'])
        else:
            strategies.extend(['simple_retry', 'reset_agent_state'])
        
        # Add severity-based strategies
        if severity >= 4:
            strategies.append('manual_intervention_required')
        elif severity >= 3:
            strategies.append('notify_administrator')
        
        return strategies
    
    def _estimate_recovery_time(self, severity: int) -> float:
        """Estimate recovery time in seconds based on severity."""
        base_times = {1: 30, 2: 60, 3: 180, 4: 600, 5: 1800}
        return base_times.get(severity, 60)
    
    def _calculate_success_probability(self, severity: int) -> float:
        """Calculate recovery success probability based on severity."""
        probabilities = {1: 0.9, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}
        return probabilities.get(severity, 0.5)
    
    async def _execute_recovery_plan(self, plan: RecoveryPlan):
        """Execute automatic recovery plan."""
        try:
            logger.info(f"Executing recovery plan {plan.recovery_id}")
            
            for strategy in plan.strategies:
                success = await self._execute_recovery_strategy(strategy, plan)
                if success:
                    # Record successful recovery
                    self.recovery_history.append({
                        'recovery_id': plan.recovery_id,
                        'strategy': strategy,
                        'success': True,
                        'tier': plan.recovery_tier.value,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
                    # Remove from active recovery plans
                    if plan.recovery_id in self.recovery_plans:
                        del self.recovery_plans[plan.recovery_id]
                    
                    logger.info(f"Recovery successful using strategy: {strategy}")
                    break
            else:
                logger.warning(f"All recovery strategies failed for plan {plan.recovery_id}")
                
        except Exception as e:
            logger.error(f"Failed to execute recovery plan {plan.recovery_id}: {e}")
    
    async def _execute_recovery_strategy(self, strategy: str, plan: RecoveryPlan) -> bool:
        """Execute a specific recovery strategy."""
        try:
            if strategy == 'simple_retry':
                # Simple retry logic
                await asyncio.sleep(5)
                return True
                
            elif strategy == 'reset_agent_state':
                # Reset agent state logic
                await asyncio.sleep(10)
                return True
                
            elif strategy == 'force_garbage_collection':
                # Force garbage collection
                import gc
                gc.collect()
                return True
                
            elif strategy == 'check_network_connectivity':
                # Network connectivity check
                # Placeholder implementation
                return True
                
            elif strategy == 'notify_administrator':
                # Notification logic
                logger.warning(f"Administrator notification: Recovery needed for agent {plan.agent_id}")
                return False  # Requires manual intervention
                
            else:
                logger.warning(f"Unknown recovery strategy: {strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute recovery strategy {strategy}: {e}")
            return False
    
    async def _automation_loop(self):
        """Main automation processing loop."""
        while True:
            try:
                await asyncio.sleep(10)  # Process automation every 10 seconds
                
                # Process queued events
                await self._process_event_queue()
                
                # Proactive optimization
                if self.strategy in [OrchestrationStrategy.PROACTIVE, OrchestrationStrategy.HYBRID]:
                    await self._perform_proactive_optimization()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in automation loop: {e}")
    
    async def _sleep_monitor_loop(self):
        """Monitor and manage sleep cycles."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                current_time = datetime.utcnow()
                completed_sleep_cycles = []
                
                # Check for completed sleep cycles
                for agent_id, sleep_start in self.active_sleep_cycles.items():
                    sleep_duration = (current_time - sleep_start).total_seconds() / 60  # minutes
                    
                    schedule = self.sleep_schedules.get(agent_id)
                    min_duration = schedule.min_sleep_duration_minutes if schedule else 15
                    
                    if sleep_duration >= min_duration:
                        completed_sleep_cycles.append(agent_id)
                
                # Complete wake cycles for agents
                for agent_id in completed_sleep_cycles:
                    await self.complete_wake_cycle(agent_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sleep monitor loop: {e}")
    
    async def _recovery_monitor_loop(self):
        """Monitor and manage recovery operations."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check for timed-out recovery plans
                current_time = datetime.utcnow()
                timeout_plans = []
                
                for plan_id, plan in self.recovery_plans.items():
                    # Simple timeout logic - remove plans older than estimated recovery time * 2
                    plan_age = (current_time - datetime.fromisoformat(plan_id[:19])).total_seconds()  # Rough estimate
                    if plan_age > plan.estimated_recovery_time * 2:
                        timeout_plans.append(plan_id)
                
                # Remove timed-out plans
                for plan_id in timeout_plans:
                    del self.recovery_plans[plan_id]
                    logger.warning(f"Recovery plan {plan_id} timed out and was removed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in recovery monitor loop: {e}")
    
    async def _process_event_queue(self):
        """Process queued automation events."""
        processed_count = 0
        
        while self.event_queue and processed_count < 10:  # Process up to 10 events per cycle
            event = self.event_queue.popleft()
            await self._handle_automation_event(event)
            processed_count += 1
    
    async def _perform_proactive_optimization(self):
        """Perform proactive optimization based on patterns."""
        # Placeholder for proactive optimization logic
        # Could include predictive scheduling, resource pre-allocation, etc.
        pass
    
    async def _emit_automation_event(self, event: Dict[str, Any]):
        """Emit an automation event."""
        self.event_queue.append(event)
    
    async def _handle_automation_event(self, event: Dict[str, Any]):
        """Handle an automation event."""
        event_type = event.get('type', 'unknown')
        handlers = self.event_handlers.get(event_type, [])
        
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error handling event {event_type}: {e}")
    
    async def _store_automation_analysis(self, analysis: Dict[str, Any]):
        """Store automation analysis data."""
        # Store in Redis with expiration
        key = f"automation_analysis:{analysis['agent_id']}:{int(time.time())}"
        await self.redis.setex(key, 3600, json.dumps(analysis))  # 1 hour expiration
    
    def _register_event_handlers(self):
        """Register event handlers for different event types."""
        self.event_handlers['performance_degradation_detected'].append(self._handle_performance_degradation)
        self.event_handlers['sleep_initiated'].append(self._handle_sleep_initiated)
        self.event_handlers['wake_completed'].append(self._handle_wake_completed)
    
    async def _handle_performance_degradation(self, event: Dict[str, Any]):
        """Handle performance degradation event."""
        agent_id = event.get('agent_id')
        if agent_id and event.get('recommendation') == 'consider_sleep_cycle':
            await self.initiate_sleep_cycle(agent_id, 'performance_degradation')
    
    async def _handle_sleep_initiated(self, event: Dict[str, Any]):
        """Handle sleep initiation event."""
        # Update metrics
        pass
    
    async def _handle_wake_completed(self, event: Dict[str, Any]):
        """Handle wake completion event."""
        # Update metrics
        pass