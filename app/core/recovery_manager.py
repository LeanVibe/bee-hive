"""
Recovery Manager for checkpoint restoration and fallback logic.

Provides robust recovery capabilities with:
- Checkpoint validation and restoration
- Multi-generation fallback logic with automatic recovery
- State rehydration and service restart coordination
- Health checks and recovery verification
- Automated rollback on restoration failure
- Recovery metrics and alerting integration
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func
import redis.asyncio as redis

from ..models.sleep_wake import (
    Checkpoint, CheckpointType, SleepWakeCycle, SleepState, SleepWakeAnalytics
)
from ..models.agent import Agent
from ..core.database import get_async_session
from ..core.redis import get_redis
from ..core.checkpoint_manager import get_checkpoint_manager
from ..core.config import get_settings


logger = logging.getLogger(__name__)


class RecoveryError(Exception):
    """Raised when recovery operations fail."""
    pass


class RecoveryManager:
    """
    Manages checkpoint restoration and system recovery operations.
    
    Features:
    - Checkpoint validation and restoration
    - Multi-generation fallback with automatic retry
    - State rehydration and service coordination
    - Health verification and rollback on failure
    - Recovery metrics and performance tracking
    - Automated alerting and monitoring integration
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.checkpoint_manager = get_checkpoint_manager()
        
        # Recovery settings
        self.max_fallback_generations = 3
        self.recovery_timeout_minutes = 10
        self.health_check_timeout_seconds = 30
        self.rollback_timeout_seconds = 60
        
        # Health check thresholds
        self.min_success_rate = 0.8  # 80% health check success rate
        self.max_error_rate = 0.1   # 10% maximum error rate
        
        # Recovery tracking
        self._active_recoveries: Dict[UUID, Dict[str, Any]] = {}
        self._recovery_history: List[Dict[str, Any]] = []
        self._last_successful_recovery: Optional[datetime] = None
    
    async def initiate_recovery(
        self,
        agent_id: Optional[UUID] = None,
        target_checkpoint_id: Optional[UUID] = None,
        recovery_type: str = "automatic"
    ) -> Tuple[bool, Optional[Checkpoint]]:
        """
        Initiate recovery process for an agent or system.
        
        Args:
            agent_id: Agent ID for agent-specific recovery, None for system-wide
            target_checkpoint_id: Specific checkpoint to restore, None for latest
            recovery_type: Type of recovery (automatic, manual, emergency)
            
        Returns:
            Tuple of (success, restored_checkpoint)
        """
        recovery_id = UUID()
        start_time = time.time()
        
        try:
            logger.info(f"Initiating {recovery_type} recovery for agent {agent_id}")
            
            # Track active recovery
            self._active_recoveries[recovery_id] = {
                "agent_id": agent_id,
                "start_time": start_time,
                "recovery_type": recovery_type,
                "status": "starting"
            }
            
            # Get checkpoint for recovery
            if target_checkpoint_id:
                checkpoint = await self._get_specific_checkpoint(target_checkpoint_id)
                fallback_checkpoints = []
            else:
                checkpoint, fallback_checkpoints = await self._get_recovery_checkpoints(agent_id)
            
            if not checkpoint:
                logger.error(f"No valid checkpoint found for recovery of agent {agent_id}")
                await self._record_recovery_failure(recovery_id, "no_checkpoint_found")
                return False, None
            
            # Pre-recovery health check
            pre_recovery_health = await self._perform_health_check(agent_id)
            
            # Attempt recovery with fallback logic
            success, restored_checkpoint = await self._attempt_recovery_with_fallbacks(
                recovery_id, checkpoint, fallback_checkpoints, agent_id
            )
            
            if success:
                # Post-recovery verification
                verification_success = await self._verify_recovery(agent_id, restored_checkpoint)
                
                if not verification_success:
                    logger.error("Recovery verification failed, initiating rollback")
                    await self._initiate_rollback(recovery_id, agent_id, pre_recovery_health)
                    success = False
                    restored_checkpoint = None
            
            # Record recovery result
            recovery_time = time.time() - start_time
            await self._record_recovery_result(
                recovery_id, success, restored_checkpoint, recovery_time, agent_id
            )
            
            # Update analytics
            await self._update_recovery_analytics(agent_id, success, recovery_time)
            
            if success:
                self._last_successful_recovery = datetime.utcnow()
                logger.info(
                    f"Recovery completed successfully for agent {agent_id} "
                    f"using checkpoint {restored_checkpoint.id} in {recovery_time:.2f}s"
                )
            else:
                logger.error(f"Recovery failed for agent {agent_id} after {recovery_time:.2f}s")
            
            return success, restored_checkpoint
            
        except Exception as e:
            logger.error(f"Error during recovery initiation: {e}")
            await self._record_recovery_failure(recovery_id, f"exception: {str(e)}")
            return False, None
            
        finally:
            # Cleanup active recovery tracking
            if recovery_id in self._active_recoveries:
                del self._active_recoveries[recovery_id]
    
    async def emergency_recovery(self, agent_id: Optional[UUID] = None) -> bool:
        """
        Perform emergency recovery with minimal checks.
        
        Args:
            agent_id: Agent ID for emergency recovery
            
        Returns:
            True if emergency recovery succeeded
        """
        try:
            logger.warning(f"Performing emergency recovery for agent {agent_id}")
            
            # Get the most recent valid checkpoint
            checkpoints = await self.checkpoint_manager.get_checkpoint_fallbacks(
                agent_id, max_generations=1
            )
            
            if not checkpoints:
                logger.error("No checkpoints available for emergency recovery")
                return False
            
            checkpoint = checkpoints[0]
            
            # Attempt direct restoration without extensive verification
            success, state_data = await self.checkpoint_manager.restore_checkpoint(checkpoint.id)
            
            if success:
                # Minimal state restoration
                await self._restore_minimal_agent_state(agent_id, state_data)
                
                # Update agent state
                async with get_async_session() as session:
                    if agent_id:
                        agent = await session.get(Agent, agent_id)
                        if agent:
                            agent.current_sleep_state = SleepState.AWAKE
                            agent.last_wake_time = datetime.utcnow()
                            await session.commit()
                
                logger.info(f"Emergency recovery completed for agent {agent_id}")
                return True
            else:
                logger.error(f"Emergency recovery failed for agent {agent_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error during emergency recovery: {e}")
            return False
    
    async def validate_recovery_readiness(self, agent_id: Optional[UUID] = None) -> Dict[str, Any]:
        """
        Validate system readiness for recovery operations.
        
        Args:
            agent_id: Agent ID to validate
            
        Returns:
            Dictionary with readiness status and details
        """
        readiness = {
            "ready": False,
            "agent_id": str(agent_id) if agent_id else None,
            "checks": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check checkpoint availability
            checkpoints = await self.checkpoint_manager.get_checkpoint_fallbacks(agent_id)
            readiness["checks"]["checkpoints_available"] = len(checkpoints) > 0
            readiness["checks"]["checkpoint_count"] = len(checkpoints)
            
            if len(checkpoints) == 0:
                readiness["errors"].append("No valid checkpoints available")
            elif len(checkpoints) < 2:
                readiness["warnings"].append("Only one checkpoint available, no fallback")
            
            # Check database connectivity
            try:
                async with get_async_session() as session:
                    await session.execute("SELECT 1")
                readiness["checks"]["database_connected"] = True
            except Exception as e:
                readiness["checks"]["database_connected"] = False
                readiness["errors"].append(f"Database connectivity error: {e}")
            
            # Check Redis connectivity
            try:
                redis_client = get_redis()
                await redis_client.ping()
                readiness["checks"]["redis_connected"] = True
            except Exception as e:
                readiness["checks"]["redis_connected"] = False
                readiness["errors"].append(f"Redis connectivity error: {e}")
            
            # Check agent state if specific agent
            if agent_id:
                try:
                    async with get_async_session() as session:
                        agent = await session.get(Agent, agent_id)
                        if agent:
                            readiness["checks"]["agent_exists"] = True
                            readiness["checks"]["agent_state"] = agent.current_sleep_state.value
                        else:
                            readiness["checks"]["agent_exists"] = False
                            readiness["errors"].append(f"Agent {agent_id} not found")
                except Exception as e:
                    readiness["checks"]["agent_exists"] = False
                    readiness["errors"].append(f"Agent check error: {e}")
            
            # Check active recoveries
            active_count = len(self._active_recoveries)
            readiness["checks"]["no_active_recoveries"] = active_count == 0
            
            if active_count > 0:
                readiness["warnings"].append(f"{active_count} recovery operations in progress")
            
            # Overall readiness determination
            critical_checks = [
                "checkpoints_available",
                "database_connected", 
                "redis_connected"
            ]
            
            if agent_id:
                critical_checks.append("agent_exists")
            
            readiness["ready"] = (
                len(readiness["errors"]) == 0 and
                all(readiness["checks"].get(check, False) for check in critical_checks)
            )
            
        except Exception as e:
            readiness["errors"].append(f"Readiness validation error: {e}")
            readiness["ready"] = False
        
        return readiness
    
    async def get_recovery_history(
        self,
        agent_id: Optional[UUID] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get recovery operation history.
        
        Args:
            agent_id: Filter by agent ID
            limit: Maximum number of records
            
        Returns:
            List of recovery history records
        """
        try:
            # Filter history by agent if specified
            if agent_id:
                filtered_history = [
                    record for record in self._recovery_history
                    if record.get("agent_id") == str(agent_id)
                ]
            else:
                filtered_history = self._recovery_history
            
            # Sort by timestamp and limit
            sorted_history = sorted(
                filtered_history,
                key=lambda x: x.get("timestamp", ""),
                reverse=True
            )
            
            return sorted_history[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recovery history: {e}")
            return []
    
    async def _get_recovery_checkpoints(
        self,
        agent_id: Optional[UUID]
    ) -> Tuple[Optional[Checkpoint], List[Checkpoint]]:
        """Get primary checkpoint and fallbacks for recovery."""
        try:
            # Get all valid checkpoints for fallback
            fallback_checkpoints = await self.checkpoint_manager.get_checkpoint_fallbacks(
                agent_id, max_generations=self.max_fallback_generations
            )
            
            if not fallback_checkpoints:
                return None, []
            
            # Primary checkpoint is the most recent
            primary_checkpoint = fallback_checkpoints[0]
            
            # Validate primary checkpoint
            is_valid, validation_errors = await self.checkpoint_manager.validate_checkpoint(
                primary_checkpoint.id
            )
            
            if not is_valid:
                logger.warning(
                    f"Primary checkpoint {primary_checkpoint.id} failed validation: {validation_errors}"
                )
                # Remove invalid primary and use next available
                fallback_checkpoints = fallback_checkpoints[1:]
                primary_checkpoint = fallback_checkpoints[0] if fallback_checkpoints else None
            
            return primary_checkpoint, fallback_checkpoints[1:]
            
        except Exception as e:
            logger.error(f"Error getting recovery checkpoints: {e}")
            return None, []
    
    async def _get_specific_checkpoint(self, checkpoint_id: UUID) -> Optional[Checkpoint]:
        """Get a specific checkpoint and validate it."""
        try:
            async with get_async_session() as session:
                checkpoint = await session.get(Checkpoint, checkpoint_id)
                
                if not checkpoint:
                    return None
                
                # Validate checkpoint
                is_valid, validation_errors = await self.checkpoint_manager.validate_checkpoint(
                    checkpoint_id
                )
                
                if not is_valid:
                    logger.error(f"Checkpoint {checkpoint_id} validation failed: {validation_errors}")
                    return None
                
                return checkpoint
                
        except Exception as e:
            logger.error(f"Error getting specific checkpoint {checkpoint_id}: {e}")
            return None
    
    async def _attempt_recovery_with_fallbacks(
        self,
        recovery_id: UUID,
        primary_checkpoint: Checkpoint,
        fallback_checkpoints: List[Checkpoint],
        agent_id: Optional[UUID]
    ) -> Tuple[bool, Optional[Checkpoint]]:
        """Attempt recovery with fallback logic."""
        checkpoints_to_try = [primary_checkpoint] + fallback_checkpoints
        
        for i, checkpoint in enumerate(checkpoints_to_try):
            try:
                logger.info(f"Attempting recovery with checkpoint {checkpoint.id} (attempt {i+1})")
                
                # Update recovery status
                self._active_recoveries[recovery_id]["status"] = f"attempting_checkpoint_{i+1}"
                self._active_recoveries[recovery_id]["current_checkpoint"] = str(checkpoint.id)
                
                # Attempt restoration
                success, state_data = await self.checkpoint_manager.restore_checkpoint(checkpoint.id)
                
                if success:
                    # Restore agent state
                    await self._restore_agent_state(agent_id, state_data)
                    
                    # Quick health check
                    health_check_passed = await self._quick_health_check(agent_id)
                    
                    if health_check_passed:
                        logger.info(f"Recovery successful with checkpoint {checkpoint.id}")
                        return True, checkpoint
                    else:
                        logger.warning(f"Health check failed after restoration with checkpoint {checkpoint.id}")
                        continue
                else:
                    logger.warning(f"Restoration failed with checkpoint {checkpoint.id}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error attempting recovery with checkpoint {checkpoint.id}: {e}")
                continue
        
        logger.error("All recovery attempts failed")
        return False, None
    
    async def _restore_agent_state(self, agent_id: Optional[UUID], state_data: Dict[str, Any]) -> None:
        """Restore agent state from checkpoint data."""
        try:
            if not agent_id:
                # System-wide restoration
                await self._restore_system_state(state_data)
                return
            
            async with get_async_session() as session:
                agent = await session.get(Agent, agent_id)
                if not agent:
                    logger.error(f"Agent {agent_id} not found during state restoration")
                    return
                
                # Restore agent sleep state
                agent.current_sleep_state = SleepState.AWAKE
                agent.last_wake_time = datetime.utcnow()
                agent.current_cycle_id = None
                
                # Update agent configuration if present in state data
                if "agent_states" in state_data:
                    agent_states = state_data["agent_states"]
                    
                    if isinstance(agent_states, list):
                        # Find agent in list
                        for agent_state in agent_states:
                            if agent_state.get("id") == str(agent_id):
                                if "config" in agent_state:
                                    agent.config = agent_state["config"]
                                break
                    elif isinstance(agent_states, dict) and "config" in agent_states:
                        agent.config = agent_states["config"]
                
                await session.commit()
                
                logger.info(f"Agent {agent_id} state restored successfully")
                
        except Exception as e:
            logger.error(f"Error restoring agent state for {agent_id}: {e}")
            raise RecoveryError(f"Agent state restoration failed: {e}")
    
    async def _restore_minimal_agent_state(self, agent_id: Optional[UUID], state_data: Dict[str, Any]) -> None:
        """Restore minimal agent state for emergency recovery."""
        try:
            if agent_id:
                async with get_async_session() as session:
                    agent = await session.get(Agent, agent_id)
                    if agent:
                        agent.current_sleep_state = SleepState.AWAKE
                        agent.last_wake_time = datetime.utcnow()
                        await session.commit()
                        
        except Exception as e:
            logger.error(f"Error restoring minimal agent state: {e}")
    
    async def _restore_system_state(self, state_data: Dict[str, Any]) -> None:
        """Restore system-wide state from checkpoint data."""
        try:
            # Restore all agent states
            if "agent_states" in state_data and isinstance(state_data["agent_states"], list):
                async with get_async_session() as session:
                    for agent_state in state_data["agent_states"]:
                        try:
                            agent_id = UUID(agent_state["id"])
                            agent = await session.get(Agent, agent_id)
                            
                            if agent:
                                agent.current_sleep_state = SleepState.AWAKE
                                agent.last_wake_time = datetime.utcnow()
                                agent.current_cycle_id = None
                                
                                if "config" in agent_state:
                                    agent.config = agent_state["config"]
                                    
                        except Exception as e:
                            logger.error(f"Error restoring agent {agent_state.get('id')}: {e}")
                            continue
                    
                    await session.commit()
            
            logger.info("System state restored successfully")
            
        except Exception as e:
            logger.error(f"Error restoring system state: {e}")
            raise RecoveryError(f"System state restoration failed: {e}")
    
    async def _perform_health_check(self, agent_id: Optional[UUID]) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": str(agent_id) if agent_id else None,
            "checks": {},
            "overall_healthy": False
        }
        
        try:
            # Database health
            try:
                async with get_async_session() as session:
                    await session.execute("SELECT 1")
                health_status["checks"]["database"] = True
            except Exception:
                health_status["checks"]["database"] = False
            
            # Redis health
            try:
                redis_client = get_redis()
                await redis_client.ping()
                health_status["checks"]["redis"] = True
            except Exception:
                health_status["checks"]["redis"] = False
            
            # Agent-specific health
            if agent_id:
                try:
                    async with get_async_session() as session:
                        agent = await session.get(Agent, agent_id)
                        health_status["checks"]["agent_exists"] = agent is not None
                        
                        if agent:
                            health_status["checks"]["agent_state"] = agent.current_sleep_state.value
                except Exception:
                    health_status["checks"]["agent_exists"] = False
            
            # Overall health determination
            critical_checks = ["database", "redis"]
            if agent_id:
                critical_checks.append("agent_exists")
            
            health_status["overall_healthy"] = all(
                health_status["checks"].get(check, False) for check in critical_checks
            )
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
            health_status["checks"]["health_check_error"] = str(e)
        
        return health_status
    
    async def _quick_health_check(self, agent_id: Optional[UUID]) -> bool:
        """Perform quick health check for recovery validation."""
        try:
            # Quick database check
            async with get_async_session() as session:
                await session.execute("SELECT 1")
            
            # Quick Redis check
            redis_client = get_redis()
            await redis_client.ping()
            
            # Quick agent check if specified
            if agent_id:
                async with get_async_session() as session:
                    agent = await session.get(Agent, agent_id)
                    if not agent:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Quick health check failed: {e}")
            return False
    
    async def _verify_recovery(self, agent_id: Optional[UUID], checkpoint: Checkpoint) -> bool:
        """Verify recovery operation success."""
        try:
            # Comprehensive health check
            health_status = await self._perform_health_check(agent_id)
            
            if not health_status["overall_healthy"]:
                logger.error("Post-recovery health check failed")
                return False
            
            # Verify agent state if specific agent
            if agent_id:
                async with get_async_session() as session:
                    agent = await session.get(Agent, agent_id)
                    
                    if not agent:
                        logger.error(f"Agent {agent_id} not found after recovery")
                        return False
                    
                    if agent.current_sleep_state != SleepState.AWAKE:
                        logger.error(f"Agent {agent_id} not in AWAKE state after recovery")
                        return False
            
            logger.info("Recovery verification passed")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying recovery: {e}")
            return False
    
    async def _initiate_rollback(
        self,
        recovery_id: UUID,
        agent_id: Optional[UUID],
        pre_recovery_health: Dict[str, Any]
    ) -> None:
        """Initiate rollback on recovery failure."""
        try:
            logger.warning(f"Initiating rollback for recovery {recovery_id}")
            
            # Update recovery status
            if recovery_id in self._active_recoveries:
                self._active_recoveries[recovery_id]["status"] = "rolling_back"
            
            # Attempt to restore previous state
            if agent_id:
                async with get_async_session() as session:
                    agent = await session.get(Agent, agent_id)
                    if agent:
                        # Set to error state for manual intervention
                        agent.current_sleep_state = SleepState.ERROR
                        await session.commit()
            
            logger.info(f"Rollback completed for recovery {recovery_id}")
            
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
    
    async def _record_recovery_result(
        self,
        recovery_id: UUID,
        success: bool,
        checkpoint: Optional[Checkpoint],
        recovery_time: float,
        agent_id: Optional[UUID]
    ) -> None:
        """Record recovery operation result."""
        try:
            record = {
                "recovery_id": str(recovery_id),
                "agent_id": str(agent_id) if agent_id else None,
                "success": success,
                "checkpoint_id": str(checkpoint.id) if checkpoint else None,
                "recovery_time_seconds": recovery_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self._recovery_history.append(record)
            
            # Keep only last 1000 records
            if len(self._recovery_history) > 1000:
                self._recovery_history = self._recovery_history[-1000:]
            
        except Exception as e:
            logger.error(f"Error recording recovery result: {e}")
    
    async def _record_recovery_failure(self, recovery_id: UUID, reason: str) -> None:
        """Record recovery failure."""
        try:
            record = {
                "recovery_id": str(recovery_id),
                "success": False,
                "failure_reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self._recovery_history.append(record)
            
        except Exception as e:
            logger.error(f"Error recording recovery failure: {e}")
    
    async def _update_recovery_analytics(
        self,
        agent_id: Optional[UUID],
        success: bool,
        recovery_time: float
    ) -> None:
        """Update recovery analytics."""
        try:
            async with get_async_session() as session:
                # Update daily analytics
                today = datetime.utcnow().date()
                
                analytics = await session.execute(
                    select(SleepWakeAnalytics).where(
                        and_(
                            SleepWakeAnalytics.agent_id == agent_id,
                            SleepWakeAnalytics.date == today
                        )
                    )
                )
                daily_analytics = analytics.scalars().first()
                
                if not daily_analytics:
                    daily_analytics = SleepWakeAnalytics(
                        agent_id=agent_id,
                        date=today
                    )
                    session.add(daily_analytics)
                
                # Update recovery metrics
                if success:
                    # Update average recovery time
                    if daily_analytics.average_recovery_time_ms:
                        daily_analytics.average_recovery_time_ms = (
                            daily_analytics.average_recovery_time_ms * 0.8 + 
                            recovery_time * 1000 * 0.2
                        )
                    else:
                        daily_analytics.average_recovery_time_ms = recovery_time * 1000
                else:
                    daily_analytics.fallback_recoveries += 1
                
                daily_analytics.updated_at = datetime.utcnow()
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error updating recovery analytics: {e}")


# Global recovery manager instance
_recovery_manager_instance: Optional[RecoveryManager] = None


def get_recovery_manager() -> RecoveryManager:
    """Get the global recovery manager instance."""
    global _recovery_manager_instance
    if _recovery_manager_instance is None:
        _recovery_manager_instance = RecoveryManager()
    return _recovery_manager_instance