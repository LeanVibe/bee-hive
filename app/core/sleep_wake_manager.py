"""
Sleep-Wake Manager - Main orchestrator for autonomous consolidation and recovery.

Coordinates all sleep-wake operations including:
- Sleep cycle initiation and management
- Wake cycle restoration and verification
- Integration with all sleep-wake components
- Health monitoring and error handling
- Performance metrics and optimization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func

from ..models.sleep_wake import (
    SleepWakeCycle, SleepState, CheckpointType, Checkpoint
)
from ..models.agent import Agent
from ..core.database import get_async_session
from ..core.sleep_scheduler import get_sleep_scheduler
from ..core.checkpoint_manager import get_checkpoint_manager
from ..core.consolidation_engine import get_consolidation_engine
from ..core.recovery_manager import get_recovery_manager
from ..core.config import get_settings


logger = logging.getLogger(__name__)


class SleepWakeManagerError(Exception):
    """Raised when sleep-wake operations fail."""
    pass


class SleepWakeManager:
    """
    Main orchestrator for sleep-wake operations.
    
    Coordinates all components to provide seamless autonomous
    consolidation and recovery for agent systems.
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Component instances will be initialized on first use
        self._scheduler = None
        self._checkpoint_manager = None
        self._consolidation_engine = None
        self._recovery_manager = None
        
        # Operation tracking
        self._active_operations: Dict[UUID, Dict[str, Any]] = {}
        
        # Performance tracking
        self._operation_metrics: Dict[str, Any] = {
            "total_sleep_cycles": 0,
            "successful_sleep_cycles": 0,
            "total_wake_cycles": 0,
            "successful_wake_cycles": 0,
            "total_consolidations": 0,
            "successful_consolidations": 0,
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "average_sleep_time_ms": 0,
            "average_wake_time_ms": 0,
            "average_consolidation_time_ms": 0,
            "average_recovery_time_ms": 0
        }
    
    async def initialize(self) -> None:
        """Initialize the sleep-wake manager and all components."""
        try:
            logger.info("Initializing Sleep-Wake Manager")
            
            # Initialize components
            self._scheduler = await get_sleep_scheduler()
            self._checkpoint_manager = get_checkpoint_manager()
            self._consolidation_engine = get_consolidation_engine()
            self._recovery_manager = get_recovery_manager()
            
            logger.info("Sleep-Wake Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Sleep-Wake Manager: {e}")
            raise SleepWakeManagerError(f"Initialization failed: {e}")
    
    async def initiate_sleep_cycle(
        self,
        agent_id: UUID,
        cycle_type: str = "scheduled",
        expected_wake_time: Optional[datetime] = None
    ) -> bool:
        """
        Initiate a complete sleep cycle for an agent.
        
        Args:
            agent_id: Agent ID to put to sleep
            cycle_type: Type of sleep cycle (scheduled, manual, emergency)
            expected_wake_time: When the agent should wake up
            
        Returns:
            True if sleep cycle was initiated successfully
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Initiating sleep cycle for agent {agent_id}, type: {cycle_type}")
            
            # Validate agent state
            async with get_async_session() as session:
                agent = await session.get(Agent, agent_id)
                if not agent:
                    raise SleepWakeManagerError(f"Agent {agent_id} not found")
                
                if agent.current_sleep_state != SleepState.AWAKE:
                    raise SleepWakeManagerError(
                        f"Agent {agent_id} is already in state {agent.current_sleep_state.value}"
                    )
                
                # Create sleep-wake cycle record
                cycle = SleepWakeCycle(
                    agent_id=agent_id,
                    cycle_type=cycle_type,
                    sleep_state=SleepState.PREPARING_SLEEP,
                    sleep_time=start_time,
                    expected_wake_time=expected_wake_time
                )
                
                session.add(cycle)
                await session.commit()
                await session.refresh(cycle)
                
                # Update agent state
                agent.current_sleep_state = SleepState.PREPARING_SLEEP
                agent.current_cycle_id = cycle.id
                agent.last_sleep_time = start_time
                await session.commit()
            
            # Track operation
            self._active_operations[cycle.id] = {
                "operation_type": "sleep",
                "agent_id": agent_id,
                "start_time": start_time,
                "status": "preparing"
            }
            
            # Phase 1: Create pre-sleep checkpoint
            logger.info(f"Creating pre-sleep checkpoint for agent {agent_id}")
            checkpoint = await self._checkpoint_manager.create_checkpoint(
                agent_id=agent_id,
                checkpoint_type=CheckpointType.PRE_SLEEP,
                metadata={"cycle_id": str(cycle.id), "cycle_type": cycle_type}
            )
            
            if not checkpoint:
                await self._handle_sleep_error(cycle.id, "Failed to create pre-sleep checkpoint")
                return False
            
            # Update cycle with checkpoint
            async with get_async_session() as session:
                await session.refresh(cycle)
                cycle.pre_sleep_checkpoint_id = checkpoint.id
                cycle.sleep_state = SleepState.SLEEPING
                await session.commit()
                
                # Update agent state
                agent = await session.get(Agent, agent_id)
                agent.current_sleep_state = SleepState.SLEEPING
                await session.commit()
            
            # Phase 2: Start consolidation process
            logger.info(f"Starting consolidation for agent {agent_id}")
            consolidation_success = await self._consolidation_engine.start_consolidation_cycle(
                cycle.id, agent_id
            )
            
            if not consolidation_success:
                logger.warning(f"Consolidation failed for agent {agent_id}, but sleep continues")
            
            # Update operation tracking
            self._active_operations[cycle.id]["status"] = "sleeping"
            self._active_operations[cycle.id]["checkpoint_id"] = str(checkpoint.id)
            
            # Update metrics
            self._operation_metrics["total_sleep_cycles"] += 1
            if consolidation_success:
                self._operation_metrics["successful_sleep_cycles"] += 1
            
            sleep_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_average_metric("average_sleep_time_ms", sleep_time_ms)
            
            logger.info(f"Sleep cycle initiated successfully for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error initiating sleep cycle for agent {agent_id}: {e}")
            await self._handle_sleep_error(cycle.id if 'cycle' in locals() else None, str(e))
            return False
            
        finally:
            # Cleanup operation tracking
            if 'cycle' in locals() and cycle.id in self._active_operations:
                if self._active_operations[cycle.id]["status"] not in ["sleeping", "consolidating"]:
                    del self._active_operations[cycle.id]
    
    async def initiate_wake_cycle(self, agent_id: UUID) -> bool:
        """
        Initiate wake cycle for an agent.
        
        Args:
            agent_id: Agent ID to wake up
            
        Returns:
            True if wake cycle was initiated successfully
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Initiating wake cycle for agent {agent_id}")
            
            # Get current cycle
            async with get_async_session() as session:
                agent = await session.get(Agent, agent_id)
                if not agent:
                    raise SleepWakeManagerError(f"Agent {agent_id} not found")
                
                if agent.current_sleep_state == SleepState.AWAKE:
                    logger.info(f"Agent {agent_id} is already awake")
                    return True
                
                if not agent.current_cycle_id:
                    raise SleepWakeManagerError(f"Agent {agent_id} has no active cycle")
                
                cycle = await session.get(SleepWakeCycle, agent.current_cycle_id)
                if not cycle:
                    raise SleepWakeManagerError(f"Sleep cycle {agent.current_cycle_id} not found")
                
                # Update cycle state
                cycle.sleep_state = SleepState.PREPARING_WAKE
                cycle.wake_time = start_time
                cycle.updated_at = start_time
                await session.commit()
                
                # Update agent state
                agent.current_sleep_state = SleepState.PREPARING_WAKE
                await session.commit()
            
            # Track operation
            self._active_operations[cycle.id] = {
                "operation_type": "wake",
                "agent_id": agent_id,
                "start_time": start_time,
                "status": "preparing"
            }
            
            # Phase 1: Create post-wake checkpoint
            logger.info(f"Creating post-wake checkpoint for agent {agent_id}")
            checkpoint = await self._checkpoint_manager.create_checkpoint(
                agent_id=agent_id,
                checkpoint_type=CheckpointType.SCHEDULED,
                metadata={"cycle_id": str(cycle.id), "wake_operation": True}
            )
            
            if checkpoint:
                async with get_async_session() as session:
                    await session.refresh(cycle)
                    cycle.post_wake_checkpoint_id = checkpoint.id
                    await session.commit()
            
            # Phase 2: Restore agent state
            logger.info(f"Restoring agent state for {agent_id}")
            
            # Calculate recovery time
            recovery_start = datetime.utcnow()
            
            async with get_async_session() as session:
                await session.refresh(cycle)
                await session.refresh(agent)
                
                # Finalize cycle
                cycle.sleep_state = SleepState.AWAKE
                cycle.wake_time = start_time
                cycle.recovery_time_ms = (datetime.utcnow() - recovery_start).total_seconds() * 1000
                cycle.updated_at = datetime.utcnow()
                await session.commit()
                
                # Update agent state
                agent.current_sleep_state = SleepState.AWAKE
                agent.last_wake_time = start_time
                agent.current_cycle_id = None
                await session.commit()
            
            # Update operation tracking
            self._active_operations[cycle.id]["status"] = "completed"
            
            # Update metrics
            self._operation_metrics["total_wake_cycles"] += 1
            self._operation_metrics["successful_wake_cycles"] += 1
            
            wake_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_average_metric("average_wake_time_ms", wake_time_ms)
            
            logger.info(f"Wake cycle completed successfully for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error initiating wake cycle for agent {agent_id}: {e}")
            await self._handle_wake_error(agent_id, str(e))
            return False
            
        finally:
            # Cleanup operation tracking
            if 'cycle' in locals() and cycle.id in self._active_operations:
                del self._active_operations[cycle.id]
    
    async def emergency_shutdown(self, agent_id: Optional[UUID] = None) -> bool:
        """
        Perform emergency shutdown for an agent or system.
        
        Args:
            agent_id: Agent ID for agent-specific shutdown, None for system-wide
            
        Returns:
            True if emergency shutdown was successful
        """
        try:
            logger.warning(f"Performing emergency shutdown for agent {agent_id}")
            
            if agent_id:
                # Agent-specific emergency shutdown
                async with get_async_session() as session:
                    agent = await session.get(Agent, agent_id)
                    if agent:
                        agent.current_sleep_state = SleepState.ERROR
                        agent.current_cycle_id = None
                        await session.commit()
                
                # Attempt emergency recovery
                success = await self._recovery_manager.emergency_recovery(agent_id)
                return success
            else:
                # System-wide emergency shutdown
                async with get_async_session() as session:
                    agents = await session.execute(select(Agent))
                    
                    for agent in agents.scalars():
                        agent.current_sleep_state = SleepState.ERROR
                        agent.current_cycle_id = None
                    
                    await session.commit()
                
                # Attempt system recovery
                success = await self._recovery_manager.emergency_recovery(None)
                return success
                
        except Exception as e:
            logger.error(f"Error during emergency shutdown: {e}")
            return False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            status = {
                "timestamp": datetime.utcnow().isoformat(),
                "system_healthy": True,
                "metrics": self._operation_metrics.copy(),
                "active_operations": len(self._active_operations),
                "agents": {},
                "errors": []
            }
            
            # Get agent states
            async with get_async_session() as session:
                agents = await session.execute(select(Agent))
                
                for agent in agents.scalars():
                    status["agents"][str(agent.id)] = {
                        "name": agent.name,
                        "sleep_state": agent.current_sleep_state.value,
                        "current_cycle_id": str(agent.current_cycle_id) if agent.current_cycle_id else None,
                        "last_sleep_time": agent.last_sleep_time.isoformat() if agent.last_sleep_time else None,
                        "last_wake_time": agent.last_wake_time.isoformat() if agent.last_wake_time else None
                    }
                    
                    # Check for error states
                    if agent.current_sleep_state == SleepState.ERROR:
                        status["system_healthy"] = False
                        status["errors"].append(f"Agent {agent.id} in error state")
            
            # Get recent cycles statistics
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            recent_cycles = await session.execute(
                select(func.count(SleepWakeCycle.id)).where(
                    SleepWakeCycle.created_at >= cutoff_time
                )
            )
            status["metrics"]["recent_cycles_24h"] = recent_cycles.scalar() or 0
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "system_healthy": False,
                "error": str(e)
            }
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Optimize system performance based on metrics."""
        try:
            optimization_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "optimizations_applied": [],
                "performance_gains": {}
            }
            
            # Cleanup old checkpoints
            cleanup_count = await self._checkpoint_manager.cleanup_old_checkpoints()
            optimization_results["optimizations_applied"].append(f"Cleaned up {cleanup_count} old checkpoints")
            
            # Analyze performance metrics
            metrics = self._operation_metrics
            
            # Check sleep cycle success rate
            if metrics["total_sleep_cycles"] > 0:
                sleep_success_rate = metrics["successful_sleep_cycles"] / metrics["total_sleep_cycles"]
                if sleep_success_rate < 0.9:  # Below 90% success rate
                    optimization_results["optimizations_applied"].append(
                        "Sleep cycle success rate below 90%, reviewing checkpoint creation"
                    )
            
            # Check wake cycle success rate
            if metrics["total_wake_cycles"] > 0:
                wake_success_rate = metrics["successful_wake_cycles"] / metrics["total_wake_cycles"]
                if wake_success_rate < 0.95:  # Below 95% success rate
                    optimization_results["optimizations_applied"].append(
                        "Wake cycle success rate below 95%, reviewing recovery procedures"
                    )
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error optimizing performance: {e}")
            return {"error": str(e)}
    
    async def _handle_sleep_error(self, cycle_id: Optional[UUID], error_message: str) -> None:
        """Handle errors during sleep cycle."""
        try:
            if cycle_id:
                async with get_async_session() as session:
                    cycle = await session.get(SleepWakeCycle, cycle_id)
                    if cycle:
                        cycle.sleep_state = SleepState.ERROR
                        cycle.error_details = {"error": error_message}
                        cycle.updated_at = datetime.utcnow()
                        await session.commit()
                        
                        # Update agent state
                        agent = await session.get(Agent, cycle.agent_id)
                        if agent:
                            agent.current_sleep_state = SleepState.ERROR
                            await session.commit()
            
            logger.error(f"Sleep cycle error: {error_message}")
            
        except Exception as e:
            logger.error(f"Error handling sleep error: {e}")
    
    async def _handle_wake_error(self, agent_id: UUID, error_message: str) -> None:
        """Handle errors during wake cycle."""
        try:
            async with get_async_session() as session:
                agent = await session.get(Agent, agent_id)
                if agent:
                    agent.current_sleep_state = SleepState.ERROR
                    await session.commit()
                    
                    # Update current cycle if exists
                    if agent.current_cycle_id:
                        cycle = await session.get(SleepWakeCycle, agent.current_cycle_id)
                        if cycle:
                            cycle.sleep_state = SleepState.ERROR
                            cycle.error_details = {"error": error_message}
                            cycle.updated_at = datetime.utcnow()
                            await session.commit()
            
            logger.error(f"Wake cycle error for agent {agent_id}: {error_message}")
            
        except Exception as e:
            logger.error(f"Error handling wake error: {e}")
    
    def _update_average_metric(self, metric_name: str, new_value: float) -> None:
        """Update a rolling average metric."""
        current_avg = self._operation_metrics.get(metric_name, 0)
        # Simple exponential moving average with alpha = 0.1
        self._operation_metrics[metric_name] = current_avg * 0.9 + new_value * 0.1


# Global sleep-wake manager instance
_sleep_wake_manager_instance: Optional[SleepWakeManager] = None


async def get_sleep_wake_manager() -> SleepWakeManager:
    """Get the global sleep-wake manager instance."""
    global _sleep_wake_manager_instance
    if _sleep_wake_manager_instance is None:
        _sleep_wake_manager_instance = SleepWakeManager()
        await _sleep_wake_manager_instance.initialize()
    return _sleep_wake_manager_instance


async def shutdown_sleep_wake_manager() -> None:
    """Shutdown the global sleep-wake manager."""
    global _sleep_wake_manager_instance
    if _sleep_wake_manager_instance:
        # Perform any necessary cleanup
        _sleep_wake_manager_instance = None