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
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Set
from uuid import UUID
import pytz

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from ..models.sleep_wake import SleepWindow, SleepState
from ..models.agent import Agent
from ..core.database import get_async_session
from ..core.config import get_settings


logger = logging.getLogger(__name__)


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
        
        # Event handlers
        self.scheduler.add_listener(self._on_job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._on_job_error, EVENT_JOB_ERROR)
    
    async def start(self) -> None:
        """Start the sleep scheduler."""
        logger.info("Starting Sleep Scheduler")
        try:
            self.scheduler.start()
            await self._refresh_sleep_windows()
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