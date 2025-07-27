"""
Consolidation Engine for context compression and performance optimization.

Provides automated consolidation capabilities during sleep cycles with:
- Context Engine API integration for compression
- Redis stream state preservation and optimization
- Database transaction management during consolidation
- Performance audit and metrics collection
- Multi-stage consolidation pipeline with job tracking
- Token reduction optimization and reporting
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
import redis.asyncio as redis

from ..models.sleep_wake import (
    ConsolidationJob, ConsolidationStatus, SleepWakeCycle, SleepState
)
from ..models.context import Context
from ..models.agent import Agent
from ..core.database import get_async_session
from ..core.redis import get_redis
from ..core.context_manager import ContextManager
from ..core.config import get_settings


logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """
    Manages automated consolidation processes during sleep cycles.
    
    Features:
    - Context compression via Context Engine integration
    - Redis stream preservation and optimization
    - Database transaction consolidation
    - Performance metrics collection
    - Multi-stage pipeline with job tracking
    - Token reduction optimization
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.context_manager = ContextManager()
        
        # Consolidation settings
        self.max_concurrent_jobs = 3
        self.job_timeout_minutes = 30
        self.compression_target_ratio = 0.6  # Target 60% size reduction
        self.min_context_age_hours = 1  # Only consolidate contexts older than 1 hour
        
        # Performance thresholds
        self.token_reduction_target = 0.55  # 55% reduction goal
        self.max_processing_time_per_mb = 30000  # 30 seconds per MB
        
        # Active consolidation tracking
        self._active_jobs: Dict[UUID, ConsolidationJob] = {}
        self._consolidation_metrics: Dict[str, Any] = {}
    
    async def start_consolidation_cycle(
        self,
        cycle_id: UUID,
        agent_id: Optional[UUID] = None
    ) -> bool:
        """
        Start a complete consolidation cycle for a sleep-wake cycle.
        
        Args:
            cycle_id: Sleep-wake cycle ID
            agent_id: Agent ID for agent-specific consolidation
            
        Returns:
            True if consolidation cycle started successfully
        """
        try:
            logger.info(f"Starting consolidation cycle {cycle_id} for agent {agent_id}")
            
            async with get_async_session() as session:
                cycle = await session.get(SleepWakeCycle, cycle_id)
                if not cycle:
                    logger.error(f"Sleep-wake cycle {cycle_id} not found")
                    return False
                
                # Update cycle state
                cycle.sleep_state = SleepState.CONSOLIDATING
                cycle.updated_at = datetime.utcnow()
                await session.commit()
            
            # Create consolidation jobs pipeline
            jobs = await self._create_consolidation_pipeline(cycle_id, agent_id)
            
            if not jobs:
                logger.warning(f"No consolidation jobs created for cycle {cycle_id}")
                return False
            
            # Execute jobs in priority order
            success = await self._execute_consolidation_pipeline(jobs)
            
            # Update cycle with results
            await self._finalize_consolidation_cycle(cycle_id, success)
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting consolidation cycle {cycle_id}: {e}")
            await self._handle_consolidation_error(cycle_id, str(e))
            return False
    
    async def _create_consolidation_pipeline(
        self,
        cycle_id: UUID,
        agent_id: Optional[UUID]
    ) -> List[ConsolidationJob]:
        """Create the consolidation job pipeline."""
        jobs = []
        
        try:
            async with get_async_session() as session:
                # Job 1: Context compression (highest priority)
                context_job = ConsolidationJob(
                    cycle_id=cycle_id,
                    job_type="context_compression",
                    status=ConsolidationStatus.PENDING,
                    priority=100,
                    input_data={"agent_id": str(agent_id) if agent_id else None},
                    max_retries=2
                )
                session.add(context_job)
                jobs.append(context_job)
                
                # Job 2: Vector index update (high priority)
                vector_job = ConsolidationJob(
                    cycle_id=cycle_id,
                    job_type="vector_index_update",
                    status=ConsolidationStatus.PENDING,
                    priority=80,
                    input_data={"agent_id": str(agent_id) if agent_id else None},
                    max_retries=3
                )
                session.add(vector_job)
                jobs.append(vector_job)
                
                # Job 3: Redis stream cleanup (medium priority)
                redis_job = ConsolidationJob(
                    cycle_id=cycle_id,
                    job_type="redis_stream_cleanup",
                    status=ConsolidationStatus.PENDING,
                    priority=60,
                    input_data={"agent_id": str(agent_id) if agent_id else None},
                    max_retries=2
                )
                session.add(redis_job)
                jobs.append(redis_job)
                
                # Job 4: Performance audit (low priority)
                audit_job = ConsolidationJob(
                    cycle_id=cycle_id,
                    job_type="performance_audit",
                    status=ConsolidationStatus.PENDING,
                    priority=40,
                    input_data={"agent_id": str(agent_id) if agent_id else None},
                    max_retries=1
                )
                session.add(audit_job)
                jobs.append(audit_job)
                
                # Job 5: Database maintenance (lowest priority)
                db_job = ConsolidationJob(
                    cycle_id=cycle_id,
                    job_type="database_maintenance",
                    status=ConsolidationStatus.PENDING,
                    priority=20,
                    input_data={"agent_id": str(agent_id) if agent_id else None},
                    max_retries=1
                )
                session.add(db_job)
                jobs.append(db_job)
                
                await session.commit()
                
                # Sort by priority
                jobs.sort(key=lambda j: j.priority, reverse=True)
                
                logger.info(f"Created {len(jobs)} consolidation jobs for cycle {cycle_id}")
                return jobs
                
        except Exception as e:
            logger.error(f"Error creating consolidation pipeline: {e}")
            return []
    
    async def _execute_consolidation_pipeline(self, jobs: List[ConsolidationJob]) -> bool:
        """Execute consolidation jobs with concurrency control."""
        try:
            # Group jobs by priority for batch execution
            priority_groups = {}
            for job in jobs:
                if job.priority not in priority_groups:
                    priority_groups[job.priority] = []
                priority_groups[job.priority].append(job)
            
            overall_success = True
            
            # Execute groups in priority order
            for priority in sorted(priority_groups.keys(), reverse=True):
                group_jobs = priority_groups[priority]
                logger.info(f"Executing priority {priority} jobs: {[j.job_type for j in group_jobs]}")
                
                # Execute jobs in this priority group concurrently
                semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
                tasks = [
                    self._execute_consolidation_job(job, semaphore)
                    for job in group_jobs
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Job {group_jobs[i].job_type} failed with exception: {result}")
                        overall_success = False
                    elif not result:
                        logger.error(f"Job {group_jobs[i].job_type} failed")
                        overall_success = False
            
            return overall_success
            
        except Exception as e:
            logger.error(f"Error executing consolidation pipeline: {e}")
            return False
    
    async def _execute_consolidation_job(
        self,
        job: ConsolidationJob,
        semaphore: asyncio.Semaphore
    ) -> bool:
        """Execute a single consolidation job."""
        async with semaphore:
            start_time = time.time()
            
            try:
                async with get_async_session() as session:
                    # Update job status
                    await session.refresh(job)
                    job.status = ConsolidationStatus.IN_PROGRESS
                    job.started_at = datetime.utcnow()
                    await session.commit()
                
                self._active_jobs[job.id] = job
                
                logger.info(f"Executing consolidation job: {job.job_type}")
                
                # Execute based on job type
                success = False
                if job.job_type == "context_compression":
                    success = await self._execute_context_compression(job)
                elif job.job_type == "vector_index_update":
                    success = await self._execute_vector_index_update(job)
                elif job.job_type == "redis_stream_cleanup":
                    success = await self._execute_redis_cleanup(job)
                elif job.job_type == "performance_audit":
                    success = await self._execute_performance_audit(job)
                elif job.job_type == "database_maintenance":
                    success = await self._execute_database_maintenance(job)
                else:
                    logger.error(f"Unknown job type: {job.job_type}")
                    success = False
                
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Update job status
                async with get_async_session() as session:
                    await session.refresh(job)
                    job.status = ConsolidationStatus.COMPLETED if success else ConsolidationStatus.FAILED
                    job.completed_at = datetime.utcnow()
                    job.processing_time_ms = processing_time_ms
                    job.progress_percentage = 100.0 if success else job.progress_percentage
                    
                    if not success:
                        job.retry_count += 1
                        if job.can_retry:
                            job.status = ConsolidationStatus.PENDING
                            job.completed_at = None
                            logger.info(f"Job {job.job_type} will be retried (attempt {job.retry_count})")
                    
                    await session.commit()
                
                del self._active_jobs[job.id]
                
                if success:
                    logger.info(f"Job {job.job_type} completed successfully in {processing_time_ms:.0f}ms")
                else:
                    logger.error(f"Job {job.job_type} failed after {processing_time_ms:.0f}ms")
                
                return success
                
            except Exception as e:
                logger.error(f"Error executing job {job.job_type}: {e}")
                
                # Update job with error
                async with get_async_session() as session:
                    await session.refresh(job)
                    job.status = ConsolidationStatus.FAILED
                    job.error_message = str(e)
                    job.completed_at = datetime.utcnow()
                    job.processing_time_ms = (time.time() - start_time) * 1000
                    await session.commit()
                
                if job.id in self._active_jobs:
                    del self._active_jobs[job.id]
                
                return False
    
    async def _execute_context_compression(self, job: ConsolidationJob) -> bool:
        """Execute context compression job."""
        try:
            agent_id = UUID(job.input_data["agent_id"]) if job.input_data.get("agent_id") else None
            
            # Get contexts to compress
            contexts_to_compress = await self._get_contexts_for_compression(agent_id)
            
            if not contexts_to_compress:
                logger.info("No contexts found for compression")
                job.output_data = {"contexts_processed": 0, "tokens_saved": 0}
                return True
            
            total_tokens_processed = 0
            total_tokens_saved = 0
            compressed_count = 0
            
            for context in contexts_to_compress:
                try:
                    # Update progress
                    progress = (compressed_count / len(contexts_to_compress)) * 100
                    async with get_async_session() as session:
                        await session.refresh(job)
                        job.progress_percentage = progress
                        await session.commit()
                    
                    # Compress context using Context Engine
                    compression_result = await self.context_manager.compress_context(
                        context.id,
                        compression_level="aggressive"
                    )
                    
                    if compression_result:
                        total_tokens_processed += compression_result.get("original_tokens", 0)
                        total_tokens_saved += compression_result.get("tokens_saved", 0)
                        compressed_count += 1
                        
                        logger.debug(f"Compressed context {context.id}: saved {compression_result.get('tokens_saved', 0)} tokens")
                    
                except Exception as e:
                    logger.error(f"Error compressing context {context.id}: {e}")
                    continue
            
            # Calculate efficiency metrics
            compression_ratio = total_tokens_saved / total_tokens_processed if total_tokens_processed > 0 else 0
            
            job.output_data = {
                "contexts_processed": compressed_count,
                "total_contexts": len(contexts_to_compress),
                "tokens_processed": total_tokens_processed,
                "tokens_saved": total_tokens_saved,
                "compression_ratio": compression_ratio
            }
            job.tokens_processed = total_tokens_processed
            job.tokens_saved = total_tokens_saved
            
            logger.info(
                f"Context compression completed: {compressed_count}/{len(contexts_to_compress)} contexts, "
                f"{total_tokens_saved} tokens saved ({compression_ratio:.2%} reduction)"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error in context compression: {e}")
            return False
    
    async def _execute_vector_index_update(self, job: ConsolidationJob) -> bool:
        """Execute vector index update job."""
        try:
            agent_id = UUID(job.input_data["agent_id"]) if job.input_data.get("agent_id") else None
            
            # Update vector indexes for compressed contexts
            updated_count = await self.context_manager.rebuild_vector_indexes(agent_id)
            
            job.output_data = {
                "indexes_updated": updated_count,
                "agent_id": str(agent_id) if agent_id else "system"
            }
            
            logger.info(f"Vector index update completed: {updated_count} indexes updated")
            return True
            
        except Exception as e:
            logger.error(f"Error in vector index update: {e}")
            return False
    
    async def _execute_redis_cleanup(self, job: ConsolidationJob) -> bool:
        """Execute Redis stream cleanup job."""
        try:
            agent_id = UUID(job.input_data["agent_id"]) if job.input_data.get("agent_id") else None
            
            redis_client = get_redis()
            
            # Define cleanup patterns
            if agent_id:
                patterns = [
                    f"agent:{agent_id}:*",
                    f"tasks:{agent_id}:*",
                    f"temp:{agent_id}:*"
                ]
            else:
                patterns = ["temp:*", "cache:*"]
            
            cleaned_keys = 0
            cleaned_streams = 0
            
            for pattern in patterns:
                keys = await redis_client.keys(pattern)
                
                for key in keys:
                    try:
                        key_str = key.decode() if isinstance(key, bytes) else key
                        
                        # Check if it's a stream
                        key_type = await redis_client.type(key)
                        
                        if key_type == b"stream":
                            # Trim old messages from streams
                            try:
                                info = await redis_client.xinfo_stream(key)
                                length = info.get("length", 0)
                                
                                if length > 1000:  # Keep only last 1000 messages
                                    await redis_client.xtrim(key, maxlen=1000, approximate=True)
                                    cleaned_streams += 1
                            except Exception as e:
                                logger.warning(f"Could not trim stream {key_str}: {e}")
                        
                        elif "temp:" in key_str or "cache:" in key_str:
                            # Delete temporary keys
                            await redis_client.delete(key)
                            cleaned_keys += 1
                            
                    except Exception as e:
                        logger.warning(f"Error cleaning Redis key {key}: {e}")
            
            job.output_data = {
                "cleaned_keys": cleaned_keys,
                "cleaned_streams": cleaned_streams,
                "agent_id": str(agent_id) if agent_id else "system"
            }
            
            logger.info(f"Redis cleanup completed: {cleaned_keys} keys deleted, {cleaned_streams} streams trimmed")
            return True
            
        except Exception as e:
            logger.error(f"Error in Redis cleanup: {e}")
            return False
    
    async def _execute_performance_audit(self, job: ConsolidationJob) -> bool:
        """Execute performance audit job."""
        try:
            agent_id = UUID(job.input_data["agent_id"]) if job.input_data.get("agent_id") else None
            
            # Collect performance metrics
            metrics = await self._collect_performance_metrics(agent_id)
            
            job.output_data = metrics
            
            # Store metrics for monitoring
            self._consolidation_metrics[str(job.cycle_id)] = metrics
            
            logger.info(f"Performance audit completed for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error in performance audit: {e}")
            return False
    
    async def _execute_database_maintenance(self, job: ConsolidationJob) -> bool:
        """Execute database maintenance job."""
        try:
            async with get_async_session() as session:
                # Run cleanup functions
                result = await session.execute(
                    "SELECT cleanup_old_context_analytics(90)"
                )
                analytics_cleaned = result.scalar()
                
                # Vacuum analyze for performance
                await session.execute("VACUUM ANALYZE contexts")
                await session.execute("VACUUM ANALYZE sleep_wake_cycles")
                
                job.output_data = {
                    "analytics_records_cleaned": analytics_cleaned or 0,
                    "tables_vacuumed": ["contexts", "sleep_wake_cycles"]
                }
            
            logger.info("Database maintenance completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in database maintenance: {e}")
            return False
    
    async def _get_contexts_for_compression(self, agent_id: Optional[UUID]) -> List[Context]:
        """Get contexts that are candidates for compression."""
        try:
            async with get_async_session() as session:
                # Get contexts older than threshold that haven't been compressed recently
                cutoff_time = datetime.utcnow() - timedelta(hours=self.min_context_age_hours)
                
                query = select(Context).where(
                    and_(
                        Context.created_at < cutoff_time,
                        or_(
                            Context.is_consolidated == False,
                            Context.is_consolidated.is_(None)
                        )
                    )
                )
                
                if agent_id:
                    query = query.where(Context.agent_id == agent_id)
                
                # Limit to avoid overwhelming the system
                query = query.limit(100)
                
                result = await session.execute(query)
                return list(result.scalars().all())
                
        except Exception as e:
            logger.error(f"Error getting contexts for compression: {e}")
            return []
    
    async def _collect_performance_metrics(self, agent_id: Optional[UUID]) -> Dict[str, Any]:
        """Collect performance metrics for audit."""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": str(agent_id) if agent_id else None
        }
        
        try:
            async with get_async_session() as session:
                # Context metrics
                context_query = select(func.count(Context.id)).where(
                    Context.agent_id == agent_id if agent_id else True
                )
                total_contexts = await session.scalar(context_query)
                
                compressed_query = select(func.count(Context.id)).where(
                    and_(
                        Context.agent_id == agent_id if agent_id else True,
                        Context.is_consolidated == True
                    )
                )
                compressed_contexts = await session.scalar(compressed_query)
                
                metrics.update({
                    "total_contexts": total_contexts or 0,
                    "compressed_contexts": compressed_contexts or 0,
                    "compression_percentage": (compressed_contexts / total_contexts * 100) if total_contexts > 0 else 0
                })
                
                # Sleep cycle metrics
                cycle_query = select(func.count(SleepWakeCycle.id)).where(
                    SleepWakeCycle.agent_id == agent_id if agent_id else True
                )
                total_cycles = await session.scalar(cycle_query)
                
                metrics["total_sleep_cycles"] = total_cycles or 0
                
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
        
        return metrics
    
    async def _finalize_consolidation_cycle(self, cycle_id: UUID, success: bool) -> None:
        """Finalize consolidation cycle with results."""
        try:
            async with get_async_session() as session:
                cycle = await session.get(SleepWakeCycle, cycle_id)
                if not cycle:
                    return
                
                # Get consolidation job results
                jobs_query = select(ConsolidationJob).where(ConsolidationJob.cycle_id == cycle_id)
                result = await session.execute(jobs_query)
                jobs = result.scalars().all()
                
                # Calculate overall metrics
                total_tokens_saved = sum(job.tokens_saved or 0 for job in jobs)
                total_tokens_processed = sum(job.tokens_processed or 0 for job in jobs)
                total_processing_time = sum(job.processing_time_ms or 0 for job in jobs)
                
                token_reduction = total_tokens_saved / total_tokens_processed if total_tokens_processed > 0 else 0
                
                # Update cycle with consolidation results
                cycle.token_reduction_achieved = token_reduction
                cycle.consolidation_time_ms = total_processing_time
                cycle.sleep_state = SleepState.SLEEPING  # Return to sleeping state
                cycle.performance_metrics = {
                    "tokens_saved": total_tokens_saved,
                    "tokens_processed": total_tokens_processed,
                    "processing_time_ms": total_processing_time,
                    "jobs_completed": len([j for j in jobs if j.status == ConsolidationStatus.COMPLETED]),
                    "jobs_failed": len([j for j in jobs if j.status == ConsolidationStatus.FAILED])
                }
                cycle.updated_at = datetime.utcnow()
                
                await session.commit()
                
                logger.info(
                    f"Consolidation cycle {cycle_id} finalized: "
                    f"{token_reduction:.2%} token reduction, "
                    f"{total_processing_time:.0f}ms processing time"
                )
                
        except Exception as e:
            logger.error(f"Error finalizing consolidation cycle {cycle_id}: {e}")
    
    async def _handle_consolidation_error(self, cycle_id: UUID, error_message: str) -> None:
        """Handle consolidation cycle errors."""
        try:
            async with get_async_session() as session:
                cycle = await session.get(SleepWakeCycle, cycle_id)
                if cycle:
                    cycle.sleep_state = SleepState.ERROR
                    cycle.error_details = {"consolidation_error": error_message}
                    cycle.updated_at = datetime.utcnow()
                    await session.commit()
                    
        except Exception as e:
            logger.error(f"Error handling consolidation error for cycle {cycle_id}: {e}")
    
    async def get_consolidation_status(self, cycle_id: UUID) -> Dict[str, Any]:
        """Get status of consolidation for a cycle."""
        try:
            async with get_async_session() as session:
                jobs_query = select(ConsolidationJob).where(ConsolidationJob.cycle_id == cycle_id)
                result = await session.execute(jobs_query)
                jobs = result.scalars().all()
                
                status = {
                    "cycle_id": str(cycle_id),
                    "total_jobs": len(jobs),
                    "completed_jobs": len([j for j in jobs if j.status == ConsolidationStatus.COMPLETED]),
                    "failed_jobs": len([j for j in jobs if j.status == ConsolidationStatus.FAILED]),
                    "in_progress_jobs": len([j for j in jobs if j.status == ConsolidationStatus.IN_PROGRESS]),
                    "pending_jobs": len([j for j in jobs if j.status == ConsolidationStatus.PENDING]),
                    "jobs": [job.to_dict() for job in jobs]
                }
                
                return status
                
        except Exception as e:
            logger.error(f"Error getting consolidation status for cycle {cycle_id}: {e}")
            return {}


# Global consolidation engine instance
_consolidation_engine_instance: Optional[ConsolidationEngine] = None


def get_consolidation_engine() -> ConsolidationEngine:
    """Get the global consolidation engine instance."""
    global _consolidation_engine_instance
    if _consolidation_engine_instance is None:
        _consolidation_engine_instance = ConsolidationEngine()
    return _consolidation_engine_instance