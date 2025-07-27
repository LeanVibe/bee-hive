"""
Sleep Scheduler Service for automated sleep window management.

Provides APScheduler-based cron-like scheduling for agent sleep cycles with:
- Configurable sleep windows per agent or system-wide defaults
- Timezone-aware scheduling with daylight savings support
- Graceful agent quiescence and state transition management
- Priority-based sleep window resolution
- Health monitoring and error recovery
"""

import asyncio
import logging
import statistics
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID
import pytz
from collections import defaultdict, deque

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func

from ..models.sleep_wake import SleepWindow, SleepState, SleepWakeCycle
from ..models.agent import Agent
from ..models.task import Task
from ..core.database import get_async_session
from ..core.config import get_settings
from ..core.redis import get_redis


logger = logging.getLogger(__name__)


class ActivityProfile:
    """Agent activity pattern analysis."""
    
    def __init__(self, agent_id: UUID):
        self.agent_id = agent_id
        self.hourly_activity = defaultdict(int)  # Hour -> activity count
        self.daily_patterns = defaultdict(list)  # Day -> activity hours
        self.task_durations = deque(maxlen=100)  # Recent task durations
        self.sleep_efficiency = {}  # Sleep metrics
        self.peak_hours = []
        self.idle_periods = []
        self.optimal_sleep_windows = []
        
    def calculate_activity_score(self, hour: int) -> float:
        """Calculate activity score for a given hour (0-23)."""
        total_activity = sum(self.hourly_activity.values())
        if total_activity == 0:
            return 0.0
        return self.hourly_activity[hour] / total_activity
    
    def get_optimal_sleep_window(self) -> Tuple[int, int]:
        """Get optimal sleep window (start_hour, end_hour) based on activity."""
        if not self.hourly_activity:
            return (2, 6)  # Default 2-6 AM
        
        # Find 4-hour window with lowest activity
        min_activity = float('inf')
        optimal_start = 2
        
        for start_hour in range(24):
            window_activity = sum(
                self.hourly_activity[(start_hour + i) % 24] 
                for i in range(4)
            )
            if window_activity < min_activity:
                min_activity = window_activity
                optimal_start = start_hour
        
        return (optimal_start, (optimal_start + 4) % 24)


class ScheduleOptimization:
    """Sleep schedule optimization result."""
    
    def __init__(self):
        self.recommended_windows: List[Dict[str, Any]] = []
        self.efficiency_score: float = 0.0
        self.token_reduction_potential: float = 0.0
        self.activity_analysis: Dict[str, Any] = {}
        self.optimization_rationale: List[str] = []


class SleepMetrics:
    """Sleep cycle effectiveness metrics."""
    
    def __init__(self):
        self.total_cycles: int = 0
        self.successful_cycles: int = 0
        self.average_duration_hours: float = 0.0
        self.average_token_reduction: float = 0.0
        self.consolidation_efficiency: float = 0.0
        self.recovery_time_seconds: float = 0.0
        self.activity_improvement: float = 0.0
        self.optimal_schedule_adherence: float = 0.0


class SleepScheduler:
    """
    Manages automated sleep scheduling for agents using APScheduler.
    
    Features:
    - Cron-like sleep window configuration
    - Agent-specific and system-wide scheduling
    - Timezone-aware scheduling with DST support
    - Priority-based window resolution
    - Graceful agent state transitions
    """
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.settings = get_settings()
        self._active_sleep_jobs: Dict[str, str] = {}  # job_id -> agent_id
        self._sleep_windows_cache: Dict[UUID, List[SleepWindow]] = {}
        self._system_windows_cache: List[SleepWindow] = []
        self._last_cache_update: Optional[datetime] = None
        self._cache_ttl_seconds = 300  # 5 minutes
        
        # Intelligence components
        self._activity_profiles: Dict[UUID, ActivityProfile] = {}
        self._sleep_metrics: Dict[UUID, SleepMetrics] = {}
        self._optimization_cache: Dict[UUID, ScheduleOptimization] = {}
        self._last_activity_analysis: Optional[datetime] = None
        
        # Intelligent scheduling settings
        self.enable_intelligent_scheduling = True
        self.activity_analysis_interval_hours = 6
        self.min_activity_data_points = 10
        self.sleep_efficiency_threshold = 0.8
        
        # Event handlers
        self.scheduler.add_listener(self._on_job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._on_job_error, EVENT_JOB_ERROR)
    
    async def start(self) -> None:
        """Start the sleep scheduler."""
        logger.info("Starting Sleep Scheduler with Intelligence")
        try:
            self.scheduler.start()
            await self._refresh_sleep_windows()
            
            # Initialize intelligent scheduling
            if self.enable_intelligent_scheduling:
                await self._initialize_activity_profiles()
                await self._schedule_intelligence_jobs()
            
            await self._schedule_all_windows()
            logger.info("Sleep Scheduler started successfully")
        except Exception as e:
            logger.error(f"Failed to start Sleep Scheduler: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the sleep scheduler gracefully."""
        logger.info("Stopping Sleep Scheduler")
        try:
            self.scheduler.shutdown(wait=True)
            logger.info("Sleep Scheduler stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping Sleep Scheduler: {e}")
    
    async def force_sleep(self, agent_id: UUID, duration_minutes: Optional[int] = None) -> bool:
        """
        Force an agent into immediate sleep state.
        
        Args:
            agent_id: Agent to put to sleep
            duration_minutes: Optional sleep duration, defaults to next wake window
            
        Returns:
            True if sleep was initiated successfully
        """
        async with get_async_session() as session:
            try:
                # Check if agent is already sleeping
                agent = await session.get(Agent, agent_id)
                if not agent:
                    logger.error(f"Agent {agent_id} not found")
                    return False
                
                if agent.current_sleep_state != SleepState.AWAKE:
                    logger.warning(f"Agent {agent_id} is already in state {agent.current_sleep_state}")
                    return False
                
                # Calculate wake time
                wake_time = None
                if duration_minutes:
                    wake_time = datetime.utcnow() + timedelta(minutes=duration_minutes)
                else:
                    wake_time = await self._calculate_next_wake_time(agent_id)
                
                # Initiate sleep cycle
                from ..core.sleep_wake_manager import SleepWakeManager
                sleep_manager = SleepWakeManager()
                
                success = await sleep_manager.initiate_sleep_cycle(
                    agent_id=agent_id,
                    cycle_type="manual",
                    expected_wake_time=wake_time
                )
                
                if success:
                    logger.info(f"Successfully initiated manual sleep for agent {agent_id}")
                else:
                    logger.error(f"Failed to initiate manual sleep for agent {agent_id}")
                
                return success
                
            except Exception as e:
                logger.error(f"Error forcing sleep for agent {agent_id}: {e}")
                return False
    
    async def force_wake(self, agent_id: UUID) -> bool:
        """
        Force an agent to wake up immediately.
        
        Args:
            agent_id: Agent to wake up
            
        Returns:
            True if wake was successful
        """
        async with get_async_session() as session:
            try:
                agent = await session.get(Agent, agent_id)
                if not agent:
                    logger.error(f"Agent {agent_id} not found")
                    return False
                
                if agent.current_sleep_state == SleepState.AWAKE:
                    logger.info(f"Agent {agent_id} is already awake")
                    return True
                
                # Initiate wake process
                from ..core.sleep_wake_manager import SleepWakeManager
                sleep_manager = SleepWakeManager()
                
                success = await sleep_manager.initiate_wake_cycle(agent_id)
                
                if success:
                    logger.info(f"Successfully woke up agent {agent_id}")
                else:
                    logger.error(f"Failed to wake up agent {agent_id}")
                
                return success
                
            except Exception as e:
                logger.error(f"Error forcing wake for agent {agent_id}: {e}")
                return False
    
    async def add_sleep_window(self, sleep_window: SleepWindow) -> bool:
        """
        Add a new sleep window configuration.
        
        Args:
            sleep_window: Sleep window configuration
            
        Returns:
            True if added successfully
        """
        try:
            # Validate sleep window
            if not self._validate_sleep_window(sleep_window):
                return False
            
            # Save to database
            async with get_async_session() as session:
                session.add(sleep_window)
                await session.commit()
            
            # Refresh cache and reschedule
            await self._refresh_sleep_windows()
            await self._schedule_agent_windows(sleep_window.agent_id)
            
            logger.info(f"Added sleep window for agent {sleep_window.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding sleep window: {e}")
            return False
    
    async def update_sleep_window(self, window_id: int, updates: Dict) -> bool:
        """
        Update an existing sleep window.
        
        Args:
            window_id: Window ID to update
            updates: Dictionary of field updates
            
        Returns:
            True if updated successfully
        """
        try:
            async with get_async_session() as session:
                sleep_window = await session.get(SleepWindow, window_id)
                if not sleep_window:
                    logger.error(f"Sleep window {window_id} not found")
                    return False
                
                # Apply updates
                for field, value in updates.items():
                    if hasattr(sleep_window, field):
                        setattr(sleep_window, field, value)
                
                sleep_window.updated_at = datetime.utcnow()
                await session.commit()
            
            # Refresh cache and reschedule
            await self._refresh_sleep_windows()
            await self._schedule_agent_windows(sleep_window.agent_id)
            
            logger.info(f"Updated sleep window {window_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating sleep window {window_id}: {e}")
            return False
    
    async def remove_sleep_window(self, window_id: int) -> bool:
        """
        Remove a sleep window configuration.
        
        Args:
            window_id: Window ID to remove
            
        Returns:
            True if removed successfully
        """
        try:
            async with get_async_session() as session:
                sleep_window = await session.get(SleepWindow, window_id)
                if not sleep_window:
                    logger.error(f"Sleep window {window_id} not found")
                    return False
                
                agent_id = sleep_window.agent_id
                await session.delete(sleep_window)
                await session.commit()
            
            # Refresh cache and reschedule
            await self._refresh_sleep_windows()
            await self._schedule_agent_windows(agent_id)
            
            logger.info(f"Removed sleep window {window_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing sleep window {window_id}: {e}")
            return False
    
    async def get_agent_sleep_windows(self, agent_id: Optional[UUID] = None) -> List[SleepWindow]:
        """
        Get sleep windows for an agent or all system windows.
        
        Args:
            agent_id: Agent ID, or None for system-wide windows
            
        Returns:
            List of sleep windows
        """
        await self._refresh_sleep_windows_if_needed()
        
        if agent_id:
            return self._sleep_windows_cache.get(agent_id, [])
        else:
            return self._system_windows_cache
    
    async def get_next_sleep_time(self, agent_id: UUID) -> Optional[datetime]:
        """
        Calculate the next scheduled sleep time for an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Next sleep time or None if no windows configured
        """
        windows = await self.get_agent_sleep_windows(agent_id)
        if not windows:
            windows = await self.get_agent_sleep_windows(None)  # System default
        
        if not windows:
            return None
        
        # Find the next sleep window
        now = datetime.now()
        next_sleep = None
        
        for window in windows:
            if not window.active:
                continue
                
            # Convert to agent's timezone
            tz = pytz.timezone(window.timezone)
            local_now = tz.localize(now.replace(tzinfo=None))
            
            # Check today and tomorrow
            for days_ahead in [0, 1]:
                check_date = (local_now + timedelta(days=days_ahead)).date()
                
                # Check if this day is in the window's days_of_week
                weekday = check_date.weekday() + 1  # Convert to 1=Monday format
                if weekday not in window.days_of_week:
                    continue
                
                # Create datetime for sleep time
                sleep_datetime = tz.localize(
                    datetime.combine(check_date, window.start_time)
                )
                
                # Skip if this time has already passed today
                if days_ahead == 0 and sleep_datetime <= local_now:
                    continue
                
                # Convert back to UTC
                sleep_utc = sleep_datetime.astimezone(pytz.UTC).replace(tzinfo=None)
                
                if next_sleep is None or sleep_utc < next_sleep:
                    next_sleep = sleep_utc
        
        return next_sleep
    
    async def schedule_sleep_windows(self, agent_id: UUID, schedule: str) -> None:
        """Schedule sleep windows using cron format."""
        try:
            # Parse cron schedule and create sleep windows
            # Format: "minute hour * * day_of_week"
            parts = schedule.split()
            if len(parts) != 5:
                raise ValueError("Invalid cron format. Expected: 'minute hour * * day_of_week'")
            
            minute, hour, _, _, dow = parts
            
            # Convert cron day of week (0=Sunday) to our format (1=Monday)
            days_of_week = []
            if dow == '*':
                days_of_week = [1, 2, 3, 4, 5, 6, 7]  # All days
            else:
                for d in dow.split(','):
                    cron_day = int(d)
                    our_day = (cron_day + 1) if cron_day < 6 else 1  # Convert Sunday(0) to 7
                    days_of_week.append(our_day)
            
            # Create sleep window
            start_time = time(int(hour), int(minute))
            end_time = time((int(hour) + 4) % 24, int(minute))  # 4-hour window
            
            sleep_window = SleepWindow(
                agent_id=agent_id,
                start_time=start_time,
                end_time=end_time,
                timezone="UTC",
                active=True,
                days_of_week=days_of_week,
                priority=50
            )
            
            await self.add_sleep_window(sleep_window)
            logger.info(f"Scheduled sleep windows for agent {agent_id} with cron: {schedule}")
            
        except Exception as e:
            logger.error(f"Error scheduling sleep windows for agent {agent_id}: {e}")
            raise
    
    async def trigger_intelligent_sleep(self, agent_id: UUID, threshold: float = 0.8) -> bool:
        """Trigger intelligent sleep based on activity and efficiency thresholds."""
        try:
            # Analyze current activity
            profile = await self._get_or_create_activity_profile(agent_id)
            current_hour = datetime.now().hour
            
            # Check if agent is in low-activity period
            activity_score = profile.calculate_activity_score(current_hour)
            
            if activity_score < threshold:
                # Check if agent has been active long enough to benefit from sleep
                async with get_async_session() as session:
                    agent = await session.get(Agent, agent_id)
                    if agent and agent.current_sleep_state == SleepState.AWAKE:
                        time_since_wake = None
                        if agent.last_wake_time:
                            time_since_wake = datetime.utcnow() - agent.last_wake_time
                        
                        # Only trigger if awake for at least 2 hours
                        if not time_since_wake or time_since_wake > timedelta(hours=2):
                            logger.info(f"Triggering intelligent sleep for agent {agent_id} (activity: {activity_score:.2f})")
                            return await self.force_sleep(agent_id)
            
            return False
            
        except Exception as e:
            logger.error(f"Error in intelligent sleep trigger for agent {agent_id}: {e}")
            return False
    
    async def optimize_sleep_schedule(self, agent_id: UUID) -> ScheduleOptimization:
        """Optimize sleep schedule based on activity patterns."""
        try:
            optimization = ScheduleOptimization()
            profile = await self._get_or_create_activity_profile(agent_id)
            
            # Analyze activity patterns
            await self._analyze_agent_activity(agent_id, profile)
            
            # Generate optimal sleep windows
            optimal_start, optimal_end = profile.get_optimal_sleep_window()
            
            # Calculate efficiency metrics
            optimization.efficiency_score = await self._calculate_schedule_efficiency(agent_id)
            optimization.token_reduction_potential = await self._estimate_token_reduction(agent_id)
            
            # Create recommended windows
            recommendation = {
                "start_hour": optimal_start,
                "end_hour": optimal_end,
                "duration_hours": 4,
                "days_of_week": [1, 2, 3, 4, 5, 6, 7],  # Daily
                "confidence": min(0.95, len(profile.task_durations) / 50),  # Max 95% confidence
                "rationale": f"Optimal window based on {len(profile.task_durations)} activity samples"
            }
            
            optimization.recommended_windows.append(recommendation)
            optimization.activity_analysis = {
                "peak_hours": profile.peak_hours,
                "idle_periods": profile.idle_periods,
                "average_task_duration": statistics.mean(profile.task_durations) if profile.task_durations else 0
            }
            
            optimization.optimization_rationale = [
                f"Low activity detected between {optimal_start}:00-{optimal_end}:00",
                f"Potential {optimization.token_reduction_potential:.1%} token reduction",
                f"Current schedule efficiency: {optimization.efficiency_score:.1%}"
            ]
            
            # Cache optimization
            self._optimization_cache[agent_id] = optimization
            
            logger.info(f"Optimized sleep schedule for agent {agent_id}: {optimal_start}:00-{optimal_end}:00")
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing sleep schedule for agent {agent_id}: {e}")
            return ScheduleOptimization()
    
    async def monitor_sleep_effectiveness(self) -> SleepMetrics:
        """Monitor and analyze sleep cycle effectiveness across all agents."""
        try:
            metrics = SleepMetrics()
            
            async with get_async_session() as session:
                # Get recent sleep cycles (last 30 days)
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                
                cycles_result = await session.execute(
                    select(SleepWakeCycle).where(
                        SleepWakeCycle.created_at >= cutoff_date
                    )
                )
                cycles = cycles_result.scalars().all()
                
                if cycles:
                    metrics.total_cycles = len(cycles)
                    successful_cycles = [c for c in cycles if c.sleep_state == SleepState.AWAKE]
                    metrics.successful_cycles = len(successful_cycles)
                    
                    # Calculate average metrics
                    if successful_cycles:
                        durations = []
                        token_reductions = []
                        recovery_times = []
                        
                        for cycle in successful_cycles:
                            if cycle.sleep_time and cycle.wake_time:
                                duration = (cycle.wake_time - cycle.sleep_time).total_seconds() / 3600
                                durations.append(duration)
                            
                            if cycle.token_reduction_achieved:
                                token_reductions.append(cycle.token_reduction_achieved)
                            
                            if cycle.recovery_time_ms:
                                recovery_times.append(cycle.recovery_time_ms / 1000)
                        
                        if durations:
                            metrics.average_duration_hours = statistics.mean(durations)
                        if token_reductions:
                            metrics.average_token_reduction = statistics.mean(token_reductions)
                        if recovery_times:
                            metrics.recovery_time_seconds = statistics.mean(recovery_times)
                
                # Calculate system-wide consolidation efficiency
                metrics.consolidation_efficiency = await self._calculate_consolidation_efficiency()
                
                # Calculate schedule adherence
                metrics.optimal_schedule_adherence = await self._calculate_schedule_adherence()
                
            logger.info(f"Sleep effectiveness monitoring: {metrics.successful_cycles}/{metrics.total_cycles} successful cycles")
            return metrics
            
        except Exception as e:
            logger.error(f"Error monitoring sleep effectiveness: {e}")
            return SleepMetrics()
    
    async def _refresh_sleep_windows(self) -> None:
        """Refresh the sleep windows cache from database."""
        try:
            async with get_async_session() as session:
                # Get all active sleep windows
                result = await session.execute(
                    select(SleepWindow).where(SleepWindow.active == True)
                )
                windows = result.scalars().all()
                
                # Clear caches
                self._sleep_windows_cache.clear()
                self._system_windows_cache.clear()
                
                # Group by agent_id
                for window in windows:
                    if window.agent_id:
                        if window.agent_id not in self._sleep_windows_cache:
                            self._sleep_windows_cache[window.agent_id] = []
                        self._sleep_windows_cache[window.agent_id].append(window)
                    else:
                        self._system_windows_cache.append(window)
                
                # Sort by priority (higher priority first)
                for agent_windows in self._sleep_windows_cache.values():
                    agent_windows.sort(key=lambda w: w.priority, reverse=True)
                
                self._system_windows_cache.sort(key=lambda w: w.priority, reverse=True)
                self._last_cache_update = datetime.utcnow()
                
                logger.debug(f"Refreshed sleep windows cache: {len(windows)} windows loaded")
                
        except Exception as e:
            logger.error(f"Error refreshing sleep windows cache: {e}")
    
    async def _refresh_sleep_windows_if_needed(self) -> None:
        """Refresh cache if TTL expired."""
        if (not self._last_cache_update or 
            datetime.utcnow() - self._last_cache_update > timedelta(seconds=self._cache_ttl_seconds)):
            await self._refresh_sleep_windows()
    
    async def _schedule_all_windows(self) -> None:
        """Schedule all sleep windows."""
        await self._refresh_sleep_windows_if_needed()
        
        # Schedule agent-specific windows
        for agent_id in self._sleep_windows_cache.keys():
            await self._schedule_agent_windows(agent_id)
        
        # Schedule system-wide windows for agents without specific windows
        if self._system_windows_cache:
            async with get_async_session() as session:
                # Get agents without specific sleep windows
                result = await session.execute(
                    select(Agent.id).where(
                        ~Agent.id.in_(list(self._sleep_windows_cache.keys()))
                    )
                )
                agents_without_windows = [row[0] for row in result]
                
                for agent_id in agents_without_windows:
                    await self._schedule_agent_windows(agent_id, use_system_default=True)
    
    async def _schedule_agent_windows(self, agent_id: Optional[UUID], use_system_default: bool = False) -> None:
        """Schedule sleep windows for a specific agent."""
        if not agent_id:
            return
        
        # Remove existing jobs for this agent
        existing_jobs = [job_id for job_id, aid in self._active_sleep_jobs.items() if aid == str(agent_id)]
        for job_id in existing_jobs:
            try:
                self.scheduler.remove_job(job_id)
                del self._active_sleep_jobs[job_id]
            except Exception as e:
                logger.warning(f"Error removing job {job_id}: {e}")
        
        # Get windows to schedule
        if use_system_default:
            windows = self._system_windows_cache
        else:
            windows = self._sleep_windows_cache.get(agent_id, [])
        
        # Schedule each window
        for window in windows:
            await self._schedule_window(agent_id, window)
    
    async def _schedule_window(self, agent_id: UUID, window: SleepWindow) -> None:
        """Schedule a specific sleep window."""
        try:
            # Create cron trigger
            tz = pytz.timezone(window.timezone)
            
            # Convert days_of_week to cron format (0=Sunday in cron)
            cron_days = [(day % 7) for day in window.days_of_week]
            cron_days_str = ','.join(map(str, cron_days))
            
            trigger = CronTrigger(
                hour=window.start_time.hour,
                minute=window.start_time.minute,
                second=0,
                day_of_week=cron_days_str,
                timezone=tz
            )
            
            # Create unique job ID
            job_id = f"sleep_{agent_id}_{window.id}"
            
            # Add job
            self.scheduler.add_job(
                func=self._execute_sleep_window,
                trigger=trigger,
                args=[agent_id, window],
                id=job_id,
                max_instances=1,
                replace_existing=True
            )
            
            self._active_sleep_jobs[job_id] = str(agent_id)
            
            logger.debug(f"Scheduled sleep window {window.id} for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Error scheduling window {window.id} for agent {agent_id}: {e}")
    
    async def _execute_sleep_window(self, agent_id: UUID, window: SleepWindow) -> None:
        """Execute a scheduled sleep window."""
        try:
            logger.info(f"Executing sleep window {window.id} for agent {agent_id}")
            
            # Calculate wake time
            wake_time = await self._calculate_wake_time(window)
            
            # Check if agent is available for sleep
            async with get_async_session() as session:
                agent = await session.get(Agent, agent_id)
                if not agent or agent.current_sleep_state != SleepState.AWAKE:
                    logger.info(f"Agent {agent_id} not available for sleep (state: {agent.current_sleep_state if agent else 'NOT_FOUND'})")
                    return
            
            # Initiate sleep cycle
            from ..core.sleep_wake_manager import SleepWakeManager
            sleep_manager = SleepWakeManager()
            
            success = await sleep_manager.initiate_sleep_cycle(
                agent_id=agent_id,
                cycle_type="scheduled",
                expected_wake_time=wake_time
            )
            
            if success:
                logger.info(f"Successfully initiated scheduled sleep for agent {agent_id}")
            else:
                logger.error(f"Failed to initiate scheduled sleep for agent {agent_id}")
        
        except Exception as e:
            logger.error(f"Error executing sleep window for agent {agent_id}: {e}")
    
    async def _calculate_wake_time(self, window: SleepWindow) -> datetime:
        """Calculate wake time based on sleep window."""
        tz = pytz.timezone(window.timezone)
        now_local = datetime.now(tz)
        
        # Calculate end time for today
        if window.end_time > window.start_time:
            # Same day window
            wake_time_local = now_local.replace(
                hour=window.end_time.hour,
                minute=window.end_time.minute,
                second=0,
                microsecond=0
            )
        else:
            # Overnight window - wake tomorrow
            wake_time_local = (now_local + timedelta(days=1)).replace(
                hour=window.end_time.hour,
                minute=window.end_time.minute,
                second=0,
                microsecond=0
            )
        
        # Convert to UTC
        return wake_time_local.astimezone(pytz.UTC).replace(tzinfo=None)
    
    async def _calculate_next_wake_time(self, agent_id: UUID) -> datetime:
        """Calculate next wake time for manual sleep."""
        windows = await self.get_agent_sleep_windows(agent_id)
        if not windows:
            windows = await self.get_agent_sleep_windows(None)
        
        if windows:
            return await self._calculate_wake_time(windows[0])  # Use highest priority window
        else:
            # Default to 4 hours if no windows configured
            return datetime.utcnow() + timedelta(hours=4)
    
    def _validate_sleep_window(self, window: SleepWindow) -> bool:
        """Validate sleep window configuration."""
        try:
            # Validate timezone
            pytz.timezone(window.timezone)
            
            # Validate days of week
            if not window.days_of_week or not all(1 <= day <= 7 for day in window.days_of_week):
                logger.error("Invalid days_of_week: must be list of integers 1-7")
                return False
            
            # Validate time format
            if not isinstance(window.start_time, time) or not isinstance(window.end_time, time):
                logger.error("Invalid time format")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Sleep window validation error: {e}")
            return False
    
    async def _initialize_activity_profiles(self) -> None:
        """Initialize activity profiles for all agents."""
        try:
            async with get_async_session() as session:
                agents_result = await session.execute(select(Agent))
                agents = agents_result.scalars().all()
                
                for agent in agents:
                    if agent.id not in self._activity_profiles:
                        profile = ActivityProfile(agent.id)
                        await self._populate_activity_profile(agent.id, profile)
                        self._activity_profiles[agent.id] = profile
                
                logger.info(f"Initialized activity profiles for {len(agents)} agents")
                
        except Exception as e:
            logger.error(f"Error initializing activity profiles: {e}")
    
    async def _schedule_intelligence_jobs(self) -> None:
        """Schedule periodic intelligence analysis jobs."""
        try:
            # Schedule activity analysis every 6 hours
            self.scheduler.add_job(
                func=self._periodic_activity_analysis,
                trigger="interval",
                hours=self.activity_analysis_interval_hours,
                id="activity_analysis",
                max_instances=1,
                replace_existing=True
            )
            
            # Schedule optimization updates every 12 hours
            self.scheduler.add_job(
                func=self._periodic_optimization_updates,
                trigger="interval",
                hours=12,
                id="optimization_updates",
                max_instances=1,
                replace_existing=True
            )
            
            logger.info("Scheduled intelligence analysis jobs")
            
        except Exception as e:
            logger.error(f"Error scheduling intelligence jobs: {e}")
    
    async def _periodic_activity_analysis(self) -> None:
        """Periodic analysis of agent activity patterns."""
        try:
            logger.info("Running periodic activity analysis")
            
            for agent_id, profile in self._activity_profiles.items():
                await self._analyze_agent_activity(agent_id, profile)
                
                # Check if schedule optimization is needed
                if len(profile.task_durations) >= self.min_activity_data_points:
                    optimization = await self.optimize_sleep_schedule(agent_id)
                    
                    # Apply optimization if efficiency is low
                    if optimization.efficiency_score < self.sleep_efficiency_threshold:
                        await self._apply_optimization(agent_id, optimization)
            
            self._last_activity_analysis = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error in periodic activity analysis: {e}")
    
    async def _periodic_optimization_updates(self) -> None:
        """Periodic updates to sleep schedule optimizations."""
        try:
            logger.info("Running periodic optimization updates")
            
            for agent_id in self._activity_profiles.keys():
                # Update activity profile with recent data
                profile = self._activity_profiles[agent_id]
                await self._populate_activity_profile(agent_id, profile)
                
                # Re-optimize if needed
                current_optimization = self._optimization_cache.get(agent_id)
                if not current_optimization or current_optimization.efficiency_score < 0.9:
                    await self.optimize_sleep_schedule(agent_id)
            
        except Exception as e:
            logger.error(f"Error in periodic optimization updates: {e}")
    
    async def _get_or_create_activity_profile(self, agent_id: UUID) -> ActivityProfile:
        """Get or create activity profile for an agent."""
        if agent_id not in self._activity_profiles:
            profile = ActivityProfile(agent_id)
            await self._populate_activity_profile(agent_id, profile)
            self._activity_profiles[agent_id] = profile
        return self._activity_profiles[agent_id]
    
    async def _populate_activity_profile(self, agent_id: UUID, profile: ActivityProfile) -> None:
        """Populate activity profile with historical data."""
        try:
            async with get_async_session() as session:
                # Get recent task data (last 30 days)
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                
                tasks_result = await session.execute(
                    select(Task).where(
                        and_(
                            Task.agent_id == agent_id,
                            Task.created_at >= cutoff_date
                        )
                    )
                )
                tasks = tasks_result.scalars().all()
                
                # Analyze task patterns
                for task in tasks:
                    if task.created_at:
                        hour = task.created_at.hour
                        profile.hourly_activity[hour] += 1
                        
                        # Track task duration if completed
                        if task.completed_at and task.created_at:
                            duration = (task.completed_at - task.created_at).total_seconds() / 60
                            profile.task_durations.append(duration)
                
                # Get sleep cycle efficiency data
                cycles_result = await session.execute(
                    select(SleepWakeCycle).where(
                        and_(
                            SleepWakeCycle.agent_id == agent_id,
                            SleepWakeCycle.created_at >= cutoff_date
                        )
                    )
                )
                cycles = cycles_result.scalars().all()
                
                for cycle in cycles:
                    if cycle.token_reduction_achieved:
                        profile.sleep_efficiency[str(cycle.id)] = {
                            "token_reduction": cycle.token_reduction_achieved,
                            "duration_hours": ((cycle.wake_time - cycle.sleep_time).total_seconds() / 3600) if cycle.wake_time and cycle.sleep_time else 0
                        }
                
                # Calculate peak hours and idle periods
                if profile.hourly_activity:
                    avg_activity = statistics.mean(profile.hourly_activity.values())
                    profile.peak_hours = [h for h, activity in profile.hourly_activity.items() if activity > avg_activity * 1.5]
                    profile.idle_periods = [h for h, activity in profile.hourly_activity.items() if activity < avg_activity * 0.5]
                
        except Exception as e:
            logger.error(f"Error populating activity profile for agent {agent_id}: {e}")
    
    async def _analyze_agent_activity(self, agent_id: UUID, profile: ActivityProfile) -> None:
        """Analyze agent activity patterns for optimization insights."""
        try:
            # Get real-time activity data from Redis
            redis_client = get_redis()
            
            # Check current task queue length
            queue_key = f"agent:{agent_id}:tasks"
            queue_length = await redis_client.llen(queue_key)
            
            # Update current activity
            current_hour = datetime.now().hour
            if queue_length > 0:
                profile.hourly_activity[current_hour] += 1
            
            # Calculate optimal sleep windows based on updated data
            profile.optimal_sleep_windows = [profile.get_optimal_sleep_window()]
            
        except Exception as e:
            logger.error(f"Error analyzing activity for agent {agent_id}: {e}")
    
    async def _calculate_schedule_efficiency(self, agent_id: UUID) -> float:
        """Calculate current sleep schedule efficiency."""
        try:
            async with get_async_session() as session:
                # Get recent sleep cycles
                cutoff_date = datetime.utcnow() - timedelta(days=7)
                
                cycles_result = await session.execute(
                    select(SleepWakeCycle).where(
                        and_(
                            SleepWakeCycle.agent_id == agent_id,
                            SleepWakeCycle.created_at >= cutoff_date,
                            SleepWakeCycle.sleep_state == SleepState.AWAKE
                        )
                    )
                )
                cycles = cycles_result.scalars().all()
                
                if not cycles:
                    return 0.5  # Default efficiency
                
                # Calculate efficiency based on token reduction and duration
                efficiency_scores = []
                for cycle in cycles:
                    if cycle.token_reduction_achieved and cycle.sleep_time and cycle.wake_time:
                        duration_hours = (cycle.wake_time - cycle.sleep_time).total_seconds() / 3600
                        # Efficiency = token_reduction / duration_hours (normalized)
                        efficiency = min(1.0, cycle.token_reduction_achieved / max(0.1, duration_hours / 4))
                        efficiency_scores.append(efficiency)
                
                return statistics.mean(efficiency_scores) if efficiency_scores else 0.5
                
        except Exception as e:
            logger.error(f"Error calculating schedule efficiency for agent {agent_id}: {e}")
            return 0.5
    
    async def _estimate_token_reduction(self, agent_id: UUID) -> float:
        """Estimate potential token reduction with optimized scheduling."""
        try:
            profile = self._activity_profiles.get(agent_id)
            if not profile or not profile.sleep_efficiency:
                return 0.55  # Target reduction
            
            # Calculate average token reduction from historical data
            reductions = [data["token_reduction"] for data in profile.sleep_efficiency.values()]
            if reductions:
                current_avg = statistics.mean(reductions)
                # Estimate 10-20% improvement with optimization
                return min(0.8, current_avg * 1.15)
            
            return 0.55  # Default target
            
        except Exception as e:
            logger.error(f"Error estimating token reduction for agent {agent_id}: {e}")
            return 0.55
    
    async def _calculate_consolidation_efficiency(self) -> float:
        """Calculate system-wide consolidation efficiency."""
        try:
            async with get_async_session() as session:
                # Get recent consolidation jobs
                cutoff_date = datetime.utcnow() - timedelta(days=7)
                
                from ..models.sleep_wake import ConsolidationJob, ConsolidationStatus
                
                jobs_result = await session.execute(
                    select(ConsolidationJob).where(
                        ConsolidationJob.created_at >= cutoff_date
                    )
                )
                jobs = jobs_result.scalars().all()
                
                if not jobs:
                    return 0.8  # Default efficiency
                
                successful_jobs = [j for j in jobs if j.status == ConsolidationStatus.COMPLETED]
                efficiency = len(successful_jobs) / len(jobs)
                
                return efficiency
                
        except Exception as e:
            logger.error(f"Error calculating consolidation efficiency: {e}")
            return 0.8
    
    async def _calculate_schedule_adherence(self) -> float:
        """Calculate how well agents adhere to optimal schedules."""
        try:
            adherence_scores = []
            
            for agent_id, profile in self._activity_profiles.items():
                optimization = self._optimization_cache.get(agent_id)
                if not optimization or not optimization.recommended_windows:
                    continue
                
                recommended_window = optimization.recommended_windows[0]
                optimal_start = recommended_window["start_hour"]
                optimal_end = recommended_window["end_hour"]
                
                # Check recent sleep cycles for adherence
                async with get_async_session() as session:
                    cutoff_date = datetime.utcnow() - timedelta(days=7)
                    
                    cycles_result = await session.execute(
                        select(SleepWakeCycle).where(
                            and_(
                                SleepWakeCycle.agent_id == agent_id,
                                SleepWakeCycle.created_at >= cutoff_date,
                                SleepWakeCycle.sleep_time.isnot(None)
                            )
                        )
                    )
                    cycles = cycles_result.scalars().all()
                    
                    if cycles:
                        adherence_count = 0
                        for cycle in cycles:
                            sleep_hour = cycle.sleep_time.hour
                            # Check if within 1 hour of optimal window
                            if abs(sleep_hour - optimal_start) <= 1:
                                adherence_count += 1
                        
                        agent_adherence = adherence_count / len(cycles)
                        adherence_scores.append(agent_adherence)
            
            return statistics.mean(adherence_scores) if adherence_scores else 0.7
            
        except Exception as e:
            logger.error(f"Error calculating schedule adherence: {e}")
            return 0.7
    
    async def _apply_optimization(self, agent_id: UUID, optimization: ScheduleOptimization) -> None:
        """Apply sleep schedule optimization for an agent."""
        try:
            if not optimization.recommended_windows:
                return
            
            recommendation = optimization.recommended_windows[0]
            
            # Create optimized sleep window
            optimized_window = SleepWindow(
                agent_id=agent_id,
                start_time=time(recommendation["start_hour"], 0),
                end_time=time(recommendation["end_hour"], 0),
                timezone="UTC",
                active=True,
                days_of_week=recommendation["days_of_week"],
                priority=90  # High priority for optimized windows
            )
            
            # Remove existing windows for this agent
            async with get_async_session() as session:
                existing_windows_result = await session.execute(
                    select(SleepWindow).where(
                        and_(
                            SleepWindow.agent_id == agent_id,
                            SleepWindow.active == True
                        )
                    )
                )
                existing_windows = existing_windows_result.scalars().all()
                
                for window in existing_windows:
                    window.active = False
                
                await session.commit()
            
            # Add optimized window
            await self.add_sleep_window(optimized_window)
            
            logger.info(f"Applied sleep optimization for agent {agent_id}: {recommendation['start_hour']}:00-{recommendation['end_hour']}:00")
            
        except Exception as e:
            logger.error(f"Error applying optimization for agent {agent_id}: {e}")
    
    def _on_job_executed(self, event):
        """Handle successful job execution."""
        logger.debug(f"Sleep scheduler job executed: {event.job_id}")
    
    def _on_job_error(self, event):
        """Handle job execution errors."""
        logger.error(f"Sleep scheduler job error: {event.job_id} - {event.exception}")


# Global scheduler instance
_scheduler_instance: Optional[SleepScheduler] = None


async def get_sleep_scheduler() -> SleepScheduler:
    """Get the global sleep scheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = SleepScheduler()
        await _scheduler_instance.start()
    return _scheduler_instance


async def shutdown_sleep_scheduler() -> None:
    """Shutdown the global sleep scheduler."""
    global _scheduler_instance
    if _scheduler_instance:
        await _scheduler_instance.stop()
        _scheduler_instance = None


# Utility functions for intelligent scheduling
async def analyze_agent_activity_patterns(agent_id: UUID, days_back: int = 30) -> Dict[str, Any]:
    """Analyze agent activity patterns for the specified period."""
    scheduler = await get_sleep_scheduler()
    profile = await scheduler._get_or_create_activity_profile(agent_id)
    await scheduler._populate_activity_profile(agent_id, profile)
    
    return {
        "agent_id": str(agent_id),
        "hourly_activity": dict(profile.hourly_activity),
        "peak_hours": profile.peak_hours,
        "idle_periods": profile.idle_periods,
        "average_task_duration": statistics.mean(profile.task_durations) if profile.task_durations else 0,
        "optimal_sleep_window": profile.get_optimal_sleep_window(),
        "analysis_date": datetime.utcnow().isoformat()
    }


async def get_system_sleep_analytics() -> Dict[str, Any]:
    """Get comprehensive sleep analytics for the entire system."""
    scheduler = await get_sleep_scheduler()
    metrics = await scheduler.monitor_sleep_effectiveness()
    
    analytics = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_cycles": metrics.total_cycles,
        "successful_cycles": metrics.successful_cycles,
        "success_rate": metrics.successful_cycles / max(1, metrics.total_cycles),
        "average_duration_hours": metrics.average_duration_hours,
        "average_token_reduction": metrics.average_token_reduction,
        "consolidation_efficiency": metrics.consolidation_efficiency,
        "recovery_time_seconds": metrics.recovery_time_seconds,
        "schedule_adherence": metrics.optimal_schedule_adherence,
        "agent_profiles": len(scheduler._activity_profiles),
        "optimizations_cached": len(scheduler._optimization_cache)
    }
    
    return analytics