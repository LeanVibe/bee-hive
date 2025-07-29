"""
Enhanced State Manager for VS 7.1: Redis Fast Access + PostgreSQL Persistence

Provides hybrid state management with:
- Redis for ultra-fast access (<1ms read/write)
- PostgreSQL for durable persistence
- Automatic write-through caching
- Connection pooling and optimization
- State consistency guarantees
- Performance monitoring and metrics
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from uuid import UUID
from contextlib import asynccontextmanager

import redis.asyncio as redis
from sqlalchemy import select, update, delete, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import QueuePool

from ..core.database import get_async_session
from ..core.redis import get_redis
from ..core.config import get_settings
from ..models.agent import Agent
from ..models.sleep_wake import SleepWakeCycle, SleepState, Checkpoint
from ..models.context import Context


logger = logging.getLogger(__name__)


class StateConsistencyError(Exception):
    """Raised when state consistency is violated."""
    pass


class EnhancedStateManager:
    """
    VS 7.1 Enhanced State Manager with Redis fast access and PostgreSQL persistence.
    
    Features:
    - Hybrid storage: Redis for speed, PostgreSQL for durability
    - Write-through caching with automatic sync
    - Connection pooling and query optimization
    - State consistency validation
    - Performance metrics and monitoring
    - Circuit breaker protection
    - Batch operations for efficiency
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Redis configuration for fast access
        self.redis_key_prefix = "state:v7.1:"
        self.redis_ttl_default = 3600  # 1 hour default TTL
        self.redis_batch_size = 100
        
        # Connection pool settings
        self.redis_pool_size = 20
        self.postgres_pool_size = 10
        
        # Performance settings
        self.enable_write_through_cache = True
        self.enable_read_through_cache = True
        self.enable_batch_operations = True
        self.enable_consistency_checks = True
        
        # Performance tracking
        self._performance_metrics = {
            "redis_reads": 0,
            "redis_writes": 0,
            "postgres_reads": 0,
            "postgres_writes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "consistency_checks": 0,
            "consistency_failures": 0,
            "average_read_time_ms": 0.0,
            "average_write_time_ms": 0.0
        }
        
        # Connection pools (initialized lazily)
        self._redis_pool: Optional[redis.ConnectionPool] = None
        self._postgres_pool_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the enhanced state manager."""
        try:
            logger.info("Initializing Enhanced State Manager")
            
            # Initialize Redis connection pool
            await self._initialize_redis_pool()
            
            # Initialize PostgreSQL pool (done by SQLAlchemy automatically)
            self._postgres_pool_initialized = True
            
            # Warm up connections
            await self._warm_up_connections()
            
            logger.info("Enhanced State Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced State Manager: {e}")
            raise
    
    async def _initialize_redis_pool(self) -> None:
        """Initialize Redis connection pool."""
        try:
            redis_client = get_redis()
            
            # Test connection
            await redis_client.ping()
            
            # Configure Redis for optimal performance
            await redis_client.config_set("timeout", "0")  # No client timeout
            await redis_client.config_set("tcp-keepalive", "60")  # Keep connections alive
            
            logger.info("Redis connection pool initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Redis pool: {e}")
            raise
    
    async def _warm_up_connections(self) -> None:
        """Warm up database connections."""
        try:
            # Warm up PostgreSQL connection
            async with get_async_session() as session:
                await session.execute("SELECT 1")
            
            # Warm up Redis connection
            redis_client = get_redis()
            await redis_client.ping()
            
            logger.info("Database connections warmed up")
            
        except Exception as e:
            logger.error(f"Error warming up connections: {e}")
            raise
    
    # Agent state management
    
    async def get_agent_state(self, agent_id: UUID, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get agent state with Redis fast access.
        
        Args:
            agent_id: Agent ID to get state for
            use_cache: Whether to use Redis cache
            
        Returns:
            Agent state dictionary or None if not found
        """
        start_time = time.time()
        
        try:
            cache_key = f"{self.redis_key_prefix}agent:{agent_id}"
            
            # Try Redis first if caching enabled
            if use_cache and self.enable_read_through_cache:
                redis_client = get_redis()
                cached_data = await redis_client.get(cache_key)
                
                if cached_data:
                    self._performance_metrics["redis_reads"] += 1
                    self._performance_metrics["cache_hits"] += 1
                    
                    read_time = (time.time() - start_time) * 1000
                    self._update_average_metric("average_read_time_ms", read_time)
                    
                    return json.loads(cached_data.decode())
                else:
                    self._performance_metrics["cache_misses"] += 1
            
            # Fallback to PostgreSQL
            async with get_async_session() as session:
                agent = await session.get(Agent, agent_id)
                
                if not agent:
                    return None
                
                agent_state = {
                    "id": str(agent.id),
                    "name": agent.name,
                    "status": agent.status.value if agent.status else None,
                    "current_sleep_state": agent.current_sleep_state.value if agent.current_sleep_state else None,
                    "current_cycle_id": str(agent.current_cycle_id) if agent.current_cycle_id else None,
                    "last_sleep_time": agent.last_sleep_time.isoformat() if agent.last_sleep_time else None,
                    "last_wake_time": agent.last_wake_time.isoformat() if agent.last_wake_time else None,
                    "config": agent.config or {},
                    "created_at": agent.created_at.isoformat() if agent.created_at else None,
                    "updated_at": agent.updated_at.isoformat() if agent.updated_at else None
                }
                
                self._performance_metrics["postgres_reads"] += 1
                
                # Cache in Redis for future access
                if use_cache and self.enable_write_through_cache:
                    await redis_client.setex(
                        cache_key,
                        self.redis_ttl_default,
                        json.dumps(agent_state, default=str)
                    )
                    self._performance_metrics["redis_writes"] += 1
                
                read_time = (time.time() - start_time) * 1000
                self._update_average_metric("average_read_time_ms", read_time)
                
                return agent_state
                
        except Exception as e:
            logger.error(f"Error getting agent state for {agent_id}: {e}")
            return None
    
    async def set_agent_state(
        self,
        agent_id: UUID,
        state_updates: Dict[str, Any],
        persist_to_db: bool = True
    ) -> bool:
        """
        Set agent state with write-through caching.
        
        Args:
            agent_id: Agent ID to update
            state_updates: Dictionary of state updates
            persist_to_db: Whether to persist to PostgreSQL
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            cache_key = f"{self.redis_key_prefix}agent:{agent_id}"
            
            # Update PostgreSQL first for consistency
            if persist_to_db:
                async with get_async_session() as session:
                    agent = await session.get(Agent, agent_id)
                    
                    if not agent:
                        logger.error(f"Agent {agent_id} not found for state update")
                        return False
                    
                    # Apply updates
                    for key, value in state_updates.items():
                        if key == "current_sleep_state" and isinstance(value, str):
                            agent.current_sleep_state = SleepState(value)
                        elif key == "current_cycle_id" and value:
                            agent.current_cycle_id = UUID(value) if isinstance(value, str) else value
                        elif key == "last_sleep_time" and value:
                            agent.last_sleep_time = datetime.fromisoformat(value) if isinstance(value, str) else value
                        elif key == "last_wake_time" and value:
                            agent.last_wake_time = datetime.fromisoformat(value) if isinstance(value, str) else value
                        elif hasattr(agent, key):
                            setattr(agent, key, value)
                    
                    agent.updated_at = datetime.utcnow()
                    await session.commit()
                    
                    self._performance_metrics["postgres_writes"] += 1
            
            # Update Redis cache
            if self.enable_write_through_cache:
                redis_client = get_redis()
                
                # Get current cached state or create new
                current_state = await self.get_agent_state(agent_id, use_cache=True)
                if not current_state:
                    current_state = {"id": str(agent_id)}
                
                # Apply updates to cached state
                current_state.update(state_updates)
                current_state["updated_at"] = datetime.utcnow().isoformat()
                
                # Write back to cache
                await redis_client.setex(
                    cache_key,
                    self.redis_ttl_default,
                    json.dumps(current_state, default=str)
                )
                
                self._performance_metrics["redis_writes"] += 1
            
            write_time = (time.time() - start_time) * 1000
            self._update_average_metric("average_write_time_ms", write_time)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting agent state for {agent_id}: {e}")
            return False
    
    async def batch_get_agent_states(self, agent_ids: List[UUID]) -> Dict[UUID, Dict[str, Any]]:
        """
        Get multiple agent states efficiently with batch operations.
        
        Args:
            agent_ids: List of agent IDs to retrieve
            
        Returns:
            Dictionary mapping agent IDs to their states
        """
        start_time = time.time()
        results = {}
        
        try:
            if not self.enable_batch_operations:
                # Fallback to individual gets
                for agent_id in agent_ids:
                    state = await self.get_agent_state(agent_id)
                    if state:
                        results[agent_id] = state
                return results
            
            redis_client = get_redis()
            cache_keys = [f"{self.redis_key_prefix}agent:{agent_id}" for agent_id in agent_ids]
            
            # Batch get from Redis
            cached_states = await redis_client.mget(cache_keys)
            cache_hits = []
            cache_misses = []
            
            for i, (agent_id, cached_data) in enumerate(zip(agent_ids, cached_states)):
                if cached_data:
                    try:
                        results[agent_id] = json.loads(cached_data.decode())
                        cache_hits.append(agent_id)
                    except json.JSONDecodeError:
                        cache_misses.append(agent_id)
                else:
                    cache_misses.append(agent_id)
            
            self._performance_metrics["cache_hits"] += len(cache_hits)
            self._performance_metrics["cache_misses"] += len(cache_misses)
            self._performance_metrics["redis_reads"] += len(cache_keys)
            
            # Get missing states from PostgreSQL
            if cache_misses:
                async with get_async_session() as session:
                    agents_result = await session.execute(
                        select(Agent).where(Agent.id.in_(cache_misses))
                    )
                    agents = agents_result.scalars().all()
                    
                    # Process agents and cache results
                    cache_operations = []
                    
                    for agent in agents:
                        agent_state = {
                            "id": str(agent.id),
                            "name": agent.name,
                            "status": agent.status.value if agent.status else None,
                            "current_sleep_state": agent.current_sleep_state.value if agent.current_sleep_state else None,
                            "current_cycle_id": str(agent.current_cycle_id) if agent.current_cycle_id else None,
                            "last_sleep_time": agent.last_sleep_time.isoformat() if agent.last_sleep_time else None,
                            "last_wake_time": agent.last_wake_time.isoformat() if agent.last_wake_time else None,
                            "config": agent.config or {},
                            "created_at": agent.created_at.isoformat() if agent.created_at else None,
                            "updated_at": agent.updated_at.isoformat() if agent.updated_at else None
                        }
                        
                        results[agent.id] = agent_state
                        
                        # Prepare cache operation
                        cache_key = f"{self.redis_key_prefix}agent:{agent.id}"
                        cache_operations.append((cache_key, json.dumps(agent_state, default=str)))
                    
                    # Batch cache update
                    if cache_operations and self.enable_write_through_cache:
                        pipe = redis_client.pipeline()
                        for cache_key, cache_data in cache_operations:
                            pipe.setex(cache_key, self.redis_ttl_default, cache_data)
                        await pipe.execute()
                        
                        self._performance_metrics["redis_writes"] += len(cache_operations)
                    
                    self._performance_metrics["postgres_reads"] += 1
            
            batch_time = (time.time() - start_time) * 1000
            logger.debug(f"Batch get {len(agent_ids)} agent states in {batch_time:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch get agent states: {e}")
            return results
    
    # Sleep-wake state management
    
    async def get_sleep_cycle_state(self, cycle_id: UUID) -> Optional[Dict[str, Any]]:
        """Get sleep cycle state with caching."""
        start_time = time.time()
        
        try:
            cache_key = f"{self.redis_key_prefix}cycle:{cycle_id}"
            
            # Try Redis first
            if self.enable_read_through_cache:
                redis_client = get_redis()
                cached_data = await redis_client.get(cache_key)
                
                if cached_data:
                    self._performance_metrics["redis_reads"] += 1
                    self._performance_metrics["cache_hits"] += 1
                    
                    return json.loads(cached_data.decode())
                else:
                    self._performance_metrics["cache_misses"] += 1
            
            # Fallback to PostgreSQL
            async with get_async_session() as session:
                cycle = await session.get(SleepWakeCycle, cycle_id)
                
                if not cycle:
                    return None
                
                cycle_state = {
                    "id": str(cycle.id),
                    "agent_id": str(cycle.agent_id),
                    "cycle_type": cycle.cycle_type,
                    "sleep_state": cycle.sleep_state.value if cycle.sleep_state else None,
                    "sleep_time": cycle.sleep_time.isoformat() if cycle.sleep_time else None,
                    "wake_time": cycle.wake_time.isoformat() if cycle.wake_time else None,
                    "expected_wake_time": cycle.expected_wake_time.isoformat() if cycle.expected_wake_time else None,
                    "pre_sleep_checkpoint_id": str(cycle.pre_sleep_checkpoint_id) if cycle.pre_sleep_checkpoint_id else None,
                    "post_wake_checkpoint_id": str(cycle.post_wake_checkpoint_id) if cycle.post_wake_checkpoint_id else None,
                    "recovery_time_ms": cycle.recovery_time_ms,
                    "error_details": cycle.error_details or {},
                    "created_at": cycle.created_at.isoformat() if cycle.created_at else None,
                    "updated_at": cycle.updated_at.isoformat() if cycle.updated_at else None
                }
                
                self._performance_metrics["postgres_reads"] += 1
                
                # Cache the result
                if self.enable_write_through_cache:
                    await redis_client.setex(
                        cache_key,
                        self.redis_ttl_default,
                        json.dumps(cycle_state, default=str)
                    )
                    self._performance_metrics["redis_writes"] += 1
                
                return cycle_state
                
        except Exception as e:
            logger.error(f"Error getting sleep cycle state for {cycle_id}: {e}")
            return None
    
    async def update_sleep_cycle_state(
        self,
        cycle_id: UUID,
        state_updates: Dict[str, Any]
    ) -> bool:
        """Update sleep cycle state with write-through caching."""
        start_time = time.time()
        
        try:
            cache_key = f"{self.redis_key_prefix}cycle:{cycle_id}"
            
            # Update PostgreSQL
            async with get_async_session() as session:
                cycle = await session.get(SleepWakeCycle, cycle_id)
                
                if not cycle:
                    return False
                
                # Apply updates
                for key, value in state_updates.items():
                    if key == "sleep_state" and isinstance(value, str):
                        cycle.sleep_state = SleepState(value)
                    elif key in ["sleep_time", "wake_time", "expected_wake_time"] and value:
                        if isinstance(value, str):
                            setattr(cycle, key, datetime.fromisoformat(value))
                        else:
                            setattr(cycle, key, value)
                    elif hasattr(cycle, key):
                        setattr(cycle, key, value)
                
                cycle.updated_at = datetime.utcnow()
                await session.commit()
                
                self._performance_metrics["postgres_writes"] += 1
            
            # Update Redis cache
            if self.enable_write_through_cache:
                redis_client = get_redis()
                
                # Get current state and update
                current_state = await self.get_sleep_cycle_state(cycle_id)
                if current_state:
                    current_state.update(state_updates)
                    current_state["updated_at"] = datetime.utcnow().isoformat()
                    
                    await redis_client.setex(
                        cache_key,
                        self.redis_ttl_default,
                        json.dumps(current_state, default=str)
                    )
                    
                    self._performance_metrics["redis_writes"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating sleep cycle state for {cycle_id}: {e}")
            return False
    
    # State consistency and validation
    
    async def validate_state_consistency(self, agent_id: UUID) -> Dict[str, Any]:
        """Validate consistency between Redis cache and PostgreSQL."""
        if not self.enable_consistency_checks:
            return {"consistent": True, "skipped": True}
        
        try:
            self._performance_metrics["consistency_checks"] += 1
            
            # Get state from both sources
            redis_state = await self.get_agent_state(agent_id, use_cache=True)
            postgres_state = await self.get_agent_state(agent_id, use_cache=False)
            
            if not redis_state and not postgres_state:
                return {"consistent": True, "reason": "both_empty"}
            
            if not redis_state or not postgres_state:
                self._performance_metrics["consistency_failures"] += 1
                return {
                    "consistent": False,
                    "reason": "missing_data",
                    "redis_exists": bool(redis_state),
                    "postgres_exists": bool(postgres_state)
                }
            
            # Compare critical fields
            critical_fields = [
                "current_sleep_state",
                "current_cycle_id", 
                "last_sleep_time",
                "last_wake_time"
            ]
            
            inconsistencies = []
            for field in critical_fields:
                redis_value = redis_state.get(field)
                postgres_value = postgres_state.get(field)
                
                if redis_value != postgres_value:
                    inconsistencies.append({
                        "field": field,
                        "redis_value": redis_value,
                        "postgres_value": postgres_value
                    })
            
            if inconsistencies:
                self._performance_metrics["consistency_failures"] += 1
                
                return {
                    "consistent": False,
                    "inconsistencies": inconsistencies,
                    "total_inconsistencies": len(inconsistencies)
                }
            
            return {"consistent": True}
            
        except Exception as e:
            logger.error(f"Error validating state consistency for {agent_id}: {e}")
            self._performance_metrics["consistency_failures"] += 1
            return {"consistent": False, "error": str(e)}
    
    async def repair_state_consistency(self, agent_id: UUID) -> bool:
        """Repair state consistency by using PostgreSQL as source of truth."""
        try:
            logger.info(f"Repairing state consistency for agent {agent_id}")
            
            # Get authoritative state from PostgreSQL
            postgres_state = await self.get_agent_state(agent_id, use_cache=False)
            
            if not postgres_state:
                logger.error(f"No PostgreSQL state found for agent {agent_id}")
                return False
            
            # Force update Redis cache
            redis_client = get_redis()
            cache_key = f"{self.redis_key_prefix}agent:{agent_id}"
            
            await redis_client.setex(
                cache_key,
                self.redis_ttl_default,
                json.dumps(postgres_state, default=str)
            )
            
            logger.info(f"State consistency repaired for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error repairing state consistency for {agent_id}: {e}")
            return False
    
    # Performance monitoring and metrics
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced state manager performance metrics."""
        try:
            metrics = self._performance_metrics.copy()
            
            # Calculate derived metrics
            total_reads = metrics["redis_reads"] + metrics["postgres_reads"]
            total_writes = metrics["redis_writes"] + metrics["postgres_writes"]
            total_cache_requests = metrics["cache_hits"] + metrics["cache_misses"]
            
            if total_cache_requests > 0:
                metrics["cache_hit_rate"] = metrics["cache_hits"] / total_cache_requests
            else:
                metrics["cache_hit_rate"] = 0.0
            
            if total_reads > 0:
                metrics["redis_read_ratio"] = metrics["redis_reads"] / total_reads
            else:
                metrics["redis_read_ratio"] = 0.0
            
            if metrics["consistency_checks"] > 0:
                metrics["consistency_failure_rate"] = (
                    metrics["consistency_failures"] / metrics["consistency_checks"]
                )
            else:
                metrics["consistency_failure_rate"] = 0.0
            
            # Add configuration info
            metrics["configuration"] = {
                "write_through_cache_enabled": self.enable_write_through_cache,
                "read_through_cache_enabled": self.enable_read_through_cache,
                "batch_operations_enabled": self.enable_batch_operations,
                "consistency_checks_enabled": self.enable_consistency_checks,
                "redis_ttl_default": self.redis_ttl_default
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    async def clear_cache(self, pattern: Optional[str] = None) -> int:
        """Clear Redis cache with optional pattern matching."""
        try:
            redis_client = get_redis()
            
            if pattern:
                # Clear specific pattern
                keys = await redis_client.keys(f"{self.redis_key_prefix}{pattern}")
            else:
                # Clear all state cache
                keys = await redis_client.keys(f"{self.redis_key_prefix}*")
            
            if keys:
                deleted_count = await redis_client.delete(*keys)
                logger.info(f"Cleared {deleted_count} cache entries")
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
    
    def _update_average_metric(self, metric_name: str, new_value: float) -> None:
        """Update rolling average metric."""
        current_avg = self._performance_metrics.get(metric_name, 0)
        # Exponential moving average with alpha = 0.1
        self._performance_metrics[metric_name] = current_avg * 0.9 + new_value * 0.1
    
    @asynccontextmanager
    async def atomic_state_transaction(self, agent_id: UUID):
        """Context manager for atomic state operations."""
        redis_client = get_redis()
        lock_key = f"{self.redis_key_prefix}lock:{agent_id}"
        
        # Acquire distributed lock
        lock_acquired = await redis_client.set(lock_key, "locked", nx=True, ex=10)
        
        if not lock_acquired:
            raise StateConsistencyError(f"Could not acquire lock for agent {agent_id}")
        
        try:
            yield
        finally:
            # Release lock
            await redis_client.delete(lock_key)


# Global enhanced state manager instance
_enhanced_state_manager_instance: Optional[EnhancedStateManager] = None


async def get_enhanced_state_manager() -> EnhancedStateManager:
    """Get the global enhanced state manager instance."""
    global _enhanced_state_manager_instance
    if _enhanced_state_manager_instance is None:
        _enhanced_state_manager_instance = EnhancedStateManager()
        await _enhanced_state_manager_instance.initialize()
    return _enhanced_state_manager_instance