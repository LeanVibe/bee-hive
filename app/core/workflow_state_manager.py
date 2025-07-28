"""
Workflow State Manager for LeanVibe Agent Hive 2.0 Workflow Engine

Production-ready state persistence and recovery system for complex multi-step workflows
with checkpoint management, rollback capabilities, and disaster recovery.
"""

import uuid
import json
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import structlog

from .database import get_session
from ..models.workflow import Workflow, WorkflowStatus
from ..models.task import Task, TaskStatus
from sqlalchemy import select, insert, update, delete, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()


class SnapshotType(Enum):
    """Types of workflow state snapshots."""
    CHECKPOINT = "checkpoint"
    BATCH_COMPLETION = "batch_completion"
    ERROR_STATE = "error_state"
    MANUAL = "manual"
    SCHEDULED = "scheduled"


class RecoveryStrategy(Enum):
    """Workflow recovery strategies."""
    RESUME_FROM_LAST_CHECKPOINT = "resume_from_last_checkpoint"
    RESTART_FAILED_BATCH = "restart_failed_batch"
    RESTART_FROM_BEGINNING = "restart_from_beginning"
    SKIP_FAILED_TASKS = "skip_failed_tasks"


@dataclass
class TaskState:
    """State information for a single task."""
    task_id: str
    status: TaskStatus
    agent_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None
    execution_time_ms: int = 0


@dataclass
class WorkflowStateSnapshot:
    """Complete workflow state snapshot."""
    snapshot_id: str
    workflow_id: str
    execution_id: str
    snapshot_type: SnapshotType
    batch_number: int
    timestamp: datetime
    workflow_status: WorkflowStatus
    task_states: Dict[str, TaskState]
    execution_context: Dict[str, Any]
    batch_progress: Dict[str, Any]
    error_information: Optional[Dict[str, Any]] = None
    can_resume_from: bool = True


@dataclass
class RecoveryPlan:
    """Plan for recovering a failed workflow."""
    strategy: RecoveryStrategy
    target_snapshot_id: str
    tasks_to_retry: List[str]
    tasks_to_skip: List[str]
    estimated_recovery_time_ms: int
    recovery_context: Dict[str, Any]


class WorkflowStateManager:
    """
    Advanced workflow state management system.
    
    Features:
    - Automatic checkpoint creation at batch boundaries
    - State persistence with configurable retention policies
    - Recovery planning with multiple strategies
    - Rollback capabilities for failed workflows
    - State comparison and diff analysis
    - Performance-optimized state storage
    - Disaster recovery with state reconstruction
    """
    
    def __init__(
        self,
        checkpoint_interval_minutes: int = 5,
        max_snapshots_per_workflow: int = 50,
        snapshot_retention_days: int = 30
    ):
        """Initialize the workflow state manager."""
        self.checkpoint_interval_minutes = checkpoint_interval_minutes
        self.max_snapshots_per_workflow = max_snapshots_per_workflow
        self.snapshot_retention_days = snapshot_retention_days
        
        # In-memory state cache for active workflows
        self.active_workflow_states: Dict[str, WorkflowStateSnapshot] = {}
        self.last_checkpoint_times: Dict[str, datetime] = {}
        
        # Performance metrics
        self.metrics = {
            'snapshots_created': 0,
            'snapshots_loaded': 0,
            'recoveries_performed': 0,
            'average_snapshot_size_kb': 0.0,
            'average_snapshot_time_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(
            "WorkflowStateManager initialized",
            checkpoint_interval_minutes=checkpoint_interval_minutes,
            max_snapshots_per_workflow=max_snapshots_per_workflow,
            snapshot_retention_days=snapshot_retention_days
        )
    
    async def create_snapshot(
        self,
        workflow_id: str,
        execution_id: str,
        workflow_status: WorkflowStatus,
        task_states: Dict[str, TaskState],
        batch_number: int,
        snapshot_type: SnapshotType = SnapshotType.CHECKPOINT,
        execution_context: Dict[str, Any] = None,
        batch_progress: Dict[str, Any] = None,
        error_information: Dict[str, Any] = None
    ) -> str:
        """
        Create a workflow state snapshot.
        
        Args:
            workflow_id: ID of the workflow
            execution_id: ID of the current execution
            workflow_status: Current workflow status
            task_states: Dictionary of task states
            batch_number: Current batch number
            snapshot_type: Type of snapshot being created
            execution_context: Additional execution context
            batch_progress: Batch progress information
            error_information: Error details if applicable
            
        Returns:
            Snapshot ID of the created snapshot
        """
        start_time = datetime.utcnow()
        snapshot_id = str(uuid.uuid4())
        
        try:
            # Create snapshot object
            snapshot = WorkflowStateSnapshot(
                snapshot_id=snapshot_id,
                workflow_id=workflow_id,
                execution_id=execution_id,
                snapshot_type=snapshot_type,
                batch_number=batch_number,
                timestamp=start_time,
                workflow_status=workflow_status,
                task_states=task_states,
                execution_context=execution_context or {},
                batch_progress=batch_progress or {},
                error_information=error_information,
                can_resume_from=(snapshot_type in [SnapshotType.CHECKPOINT, SnapshotType.BATCH_COMPLETION])
            )
            
            # Store in database
            await self._persist_snapshot(snapshot)
            
            # Update in-memory cache
            self.active_workflow_states[workflow_id] = snapshot
            self.last_checkpoint_times[workflow_id] = start_time
            
            # Clean up old snapshots
            await self._cleanup_old_snapshots(workflow_id)
            
            # Update metrics
            snapshot_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self._update_snapshot_metrics(snapshot, snapshot_time_ms)
            
            logger.info(
                "✅ Workflow snapshot created",
                workflow_id=workflow_id,
                snapshot_id=snapshot_id,
                snapshot_type=snapshot_type.value,
                batch_number=batch_number,
                task_count=len(task_states),
                snapshot_time_ms=snapshot_time_ms
            )
            
            return snapshot_id
            
        except Exception as e:
            logger.error(
                "❌ Failed to create workflow snapshot",
                workflow_id=workflow_id,
                snapshot_type=snapshot_type.value,
                error=str(e)
            )
            raise
    
    async def load_snapshot(self, snapshot_id: str) -> Optional[WorkflowStateSnapshot]:
        """
        Load a workflow state snapshot by ID.
        
        Args:
            snapshot_id: ID of the snapshot to load
            
        Returns:
            WorkflowStateSnapshot if found, None otherwise
        """
        try:
            # Check in-memory cache first
            for workflow_id, cached_snapshot in self.active_workflow_states.items():
                if cached_snapshot.snapshot_id == snapshot_id:
                    self.metrics['cache_hits'] += 1
                    return cached_snapshot
            
            # Load from database
            snapshot = await self._load_snapshot_from_db(snapshot_id)
            
            if snapshot:
                self.metrics['snapshots_loaded'] += 1
                self.metrics['cache_misses'] += 1
                
                # Cache the loaded snapshot
                self.active_workflow_states[snapshot.workflow_id] = snapshot
                
                logger.info(
                    "✅ Workflow snapshot loaded",
                    snapshot_id=snapshot_id,
                    workflow_id=snapshot.workflow_id,
                    batch_number=snapshot.batch_number
                )
            else:
                logger.warning(f"Snapshot {snapshot_id} not found")
            
            return snapshot
            
        except Exception as e:
            logger.error(f"❌ Failed to load snapshot {snapshot_id}", error=str(e))
            return None
    
    async def get_latest_snapshot(
        self, 
        workflow_id: str, 
        execution_id: Optional[str] = None
    ) -> Optional[WorkflowStateSnapshot]:
        """
        Get the latest snapshot for a workflow.
        
        Args:
            workflow_id: ID of the workflow
            execution_id: Optional execution ID filter
            
        Returns:
            Latest WorkflowStateSnapshot if found
        """
        try:
            # Check in-memory cache first
            if workflow_id in self.active_workflow_states:
                cached_snapshot = self.active_workflow_states[workflow_id]
                if not execution_id or cached_snapshot.execution_id == execution_id:
                    self.metrics['cache_hits'] += 1
                    return cached_snapshot
            
            # Load from database
            async with get_session() as db_session:
                query = select(WorkflowStateSnapshotModel).where(
                    WorkflowStateSnapshotModel.workflow_id == workflow_id
                )
                
                if execution_id:
                    query = query.where(WorkflowStateSnapshotModel.execution_id == execution_id)
                
                query = query.order_by(desc(WorkflowStateSnapshotModel.created_at)).limit(1)
                
                result = await db_session.execute(query)
                snapshot_row = result.scalar_one_or_none()
                
                if snapshot_row:
                    snapshot = self._row_to_snapshot(snapshot_row)
                    self.active_workflow_states[workflow_id] = snapshot
                    self.metrics['snapshots_loaded'] += 1
                    self.metrics['cache_misses'] += 1
                    return snapshot
                
                return None
                
        except Exception as e:
            logger.error(
                f"❌ Failed to get latest snapshot for workflow {workflow_id}",
                error=str(e)
            )
            return None
    
    async def list_snapshots(
        self, 
        workflow_id: str, 
        limit: int = 10,
        snapshot_type: Optional[SnapshotType] = None
    ) -> List[WorkflowStateSnapshot]:
        """
        List snapshots for a workflow.
        
        Args:
            workflow_id: ID of the workflow
            limit: Maximum number of snapshots to return
            snapshot_type: Optional filter by snapshot type
            
        Returns:
            List of WorkflowStateSnapshots ordered by timestamp (newest first)
        """
        try:
            async with get_session() as db_session:
                query = select(WorkflowStateSnapshotModel).where(
                    WorkflowStateSnapshotModel.workflow_id == workflow_id
                )
                
                if snapshot_type:
                    query = query.where(WorkflowStateSnapshotModel.snapshot_type == snapshot_type.value)
                
                query = query.order_by(desc(WorkflowStateSnapshotModel.created_at)).limit(limit)
                
                result = await db_session.execute(query)
                snapshot_rows = result.scalars().all()
                
                snapshots = [self._row_to_snapshot(row) for row in snapshot_rows]
                
                logger.info(
                    f"✅ Listed {len(snapshots)} snapshots for workflow {workflow_id}"
                )
                
                return snapshots
                
        except Exception as e:
            logger.error(
                f"❌ Failed to list snapshots for workflow {workflow_id}",
                error=str(e)
            )
            return []
    
    async def create_recovery_plan(
        self, 
        workflow_id: str, 
        execution_id: str,
        failed_batch_number: Optional[int] = None
    ) -> Optional[RecoveryPlan]:
        """
        Create a recovery plan for a failed workflow.
        
        Args:
            workflow_id: ID of the failed workflow
            execution_id: ID of the failed execution
            failed_batch_number: Batch number where failure occurred
            
        Returns:
            RecoveryPlan with recommended recovery strategy
        """
        try:
            # Get snapshots for analysis
            snapshots = await self.list_snapshots(workflow_id, limit=20)
            
            if not snapshots:
                logger.warning(f"No snapshots found for workflow {workflow_id}")
                return None
            
            # Analyze failure context
            latest_snapshot = snapshots[0]
            
            # Determine recovery strategy based on failure analysis
            strategy = await self._determine_recovery_strategy(
                latest_snapshot, 
                failed_batch_number
            )
            
            # Find target snapshot for recovery
            target_snapshot = await self._find_recovery_target_snapshot(
                snapshots, 
                strategy, 
                failed_batch_number
            )
            
            if not target_snapshot:
                logger.error(f"No suitable recovery target found for workflow {workflow_id}")
                return None
            
            # Identify tasks to retry/skip
            tasks_to_retry, tasks_to_skip = await self._analyze_task_recovery_needs(
                target_snapshot, 
                latest_snapshot,
                strategy
            )
            
            # Estimate recovery time
            estimated_time = await self._estimate_recovery_time(
                target_snapshot, 
                tasks_to_retry
            )
            
            recovery_plan = RecoveryPlan(
                strategy=strategy,
                target_snapshot_id=target_snapshot.snapshot_id,
                tasks_to_retry=tasks_to_retry,
                tasks_to_skip=tasks_to_skip,
                estimated_recovery_time_ms=estimated_time,
                recovery_context={
                    "target_batch": target_snapshot.batch_number,
                    "failed_batch": failed_batch_number,
                    "recovery_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(
                "✅ Recovery plan created",
                workflow_id=workflow_id,
                strategy=strategy.value,
                target_snapshot_id=target_snapshot.snapshot_id,
                tasks_to_retry=len(tasks_to_retry),
                tasks_to_skip=len(tasks_to_skip)
            )
            
            return recovery_plan
            
        except Exception as e:
            logger.error(
                f"❌ Failed to create recovery plan for workflow {workflow_id}",
                error=str(e)
            )
            return None
    
    async def execute_recovery(
        self, 
        recovery_plan: RecoveryPlan,
        workflow_engine: 'WorkflowEngine'
    ) -> bool:
        """
        Execute a recovery plan to restore workflow execution.
        
        Args:
            recovery_plan: The recovery plan to execute
            workflow_engine: Workflow engine instance for execution
            
        Returns:
            True if recovery was successful
        """
        try:
            # Load target snapshot
            target_snapshot = await self.load_snapshot(recovery_plan.target_snapshot_id)
            if not target_snapshot:
                logger.error(f"Target snapshot {recovery_plan.target_snapshot_id} not found")
                return False
            
            # Update workflow status to resuming
            await self._update_workflow_status(
                target_snapshot.workflow_id, 
                WorkflowStatus.RUNNING
            )
            
            # Restore task states
            await self._restore_task_states(target_snapshot, recovery_plan)
            
            # Create recovery checkpoint
            recovery_snapshot_id = await self.create_snapshot(
                workflow_id=target_snapshot.workflow_id,
                execution_id=target_snapshot.execution_id,
                workflow_status=WorkflowStatus.RUNNING,
                task_states=target_snapshot.task_states,
                batch_number=target_snapshot.batch_number,
                snapshot_type=SnapshotType.CHECKPOINT,
                execution_context=recovery_plan.recovery_context
            )
            
            # Update metrics
            self.metrics['recoveries_performed'] += 1
            
            logger.info(
                "✅ Recovery executed successfully",
                workflow_id=target_snapshot.workflow_id,
                strategy=recovery_plan.strategy.value,
                recovery_snapshot_id=recovery_snapshot_id
            )
            
            return True
            
        except Exception as e:
            logger.error(
                f"❌ Failed to execute recovery plan",
                target_snapshot_id=recovery_plan.target_snapshot_id,
                error=str(e)
            )
            return False
    
    async def should_create_checkpoint(self, workflow_id: str) -> bool:
        """
        Determine if a checkpoint should be created based on time interval.
        
        Args:
            workflow_id: ID of the workflow to check
            
        Returns:
            True if a checkpoint should be created
        """
        if workflow_id not in self.last_checkpoint_times:
            return True
        
        last_checkpoint = self.last_checkpoint_times[workflow_id]
        time_since_checkpoint = datetime.utcnow() - last_checkpoint
        
        return time_since_checkpoint.total_seconds() >= (self.checkpoint_interval_minutes * 60)
    
    async def compare_snapshots(
        self, 
        snapshot_id_1: str, 
        snapshot_id_2: str
    ) -> Dict[str, Any]:
        """
        Compare two workflow snapshots to identify differences.
        
        Args:
            snapshot_id_1: ID of the first snapshot
            snapshot_id_2: ID of the second snapshot
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            snapshot_1 = await self.load_snapshot(snapshot_id_1)
            snapshot_2 = await self.load_snapshot(snapshot_id_2)
            
            if not snapshot_1 or not snapshot_2:
                return {"error": "One or both snapshots not found"}
            
            # Compare workflow status
            status_changed = snapshot_1.workflow_status != snapshot_2.workflow_status
            
            # Compare task states
            task_changes = []
            all_task_ids = set(snapshot_1.task_states.keys()) | set(snapshot_2.task_states.keys())
            
            for task_id in all_task_ids:
                task_1 = snapshot_1.task_states.get(task_id)
                task_2 = snapshot_2.task_states.get(task_id)
                
                if not task_1:
                    task_changes.append({"task_id": task_id, "change": "added", "new_status": task_2.status.value})
                elif not task_2:
                    task_changes.append({"task_id": task_id, "change": "removed", "old_status": task_1.status.value})
                elif task_1.status != task_2.status:
                    task_changes.append({
                        "task_id": task_id,
                        "change": "status_changed",
                        "old_status": task_1.status.value,
                        "new_status": task_2.status.value
                    })
            
            # Compare batch progress
            batch_changed = snapshot_1.batch_number != snapshot_2.batch_number
            
            return {
                "snapshot_1_id": snapshot_id_1,
                "snapshot_2_id": snapshot_id_2,
                "time_difference_seconds": (snapshot_2.timestamp - snapshot_1.timestamp).total_seconds(),
                "workflow_status_changed": status_changed,
                "batch_changed": batch_changed,
                "batch_difference": snapshot_2.batch_number - snapshot_1.batch_number,
                "task_changes": task_changes,
                "task_changes_count": len(task_changes)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to compare snapshots", error=str(e))
            return {"error": str(e)}
    
    async def cleanup_workflow_state(self, workflow_id: str) -> None:
        """
        Clean up state information for a completed workflow.
        
        Args:
            workflow_id: ID of the workflow to clean up
        """
        try:
            # Remove from in-memory cache
            self.active_workflow_states.pop(workflow_id, None)
            self.last_checkpoint_times.pop(workflow_id, None)
            
            # Keep the most recent snapshots, remove older ones
            await self._cleanup_old_snapshots(workflow_id, keep_count=5)
            
            logger.info(f"✅ Workflow state cleaned up for {workflow_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup workflow state for {workflow_id}", error=str(e))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get state manager performance metrics."""
        return {
            **self.metrics,
            "active_workflows": len(self.active_workflow_states),
            "cached_snapshots": len(self.active_workflow_states),
            "cache_hit_rate": (
                self.metrics['cache_hits'] / 
                max(self.metrics['cache_hits'] + self.metrics['cache_misses'], 1)
            )
        }
    
    # Private methods
    
    async def _persist_snapshot(self, snapshot: WorkflowStateSnapshot) -> None:
        """Persist a snapshot to the database."""
        async with get_session() as db_session:
            # Serialize task states
            task_states_json = {
                task_id: {
                    'task_id': state.task_id,
                    'status': state.status.value,
                    'agent_id': state.agent_id,
                    'started_at': state.started_at.isoformat() if state.started_at else None,
                    'completed_at': state.completed_at.isoformat() if state.completed_at else None,
                    'retry_count': state.retry_count,
                    'error_message': state.error_message,
                    'result_data': state.result_data,
                    'execution_time_ms': state.execution_time_ms
                }
                for task_id, state in snapshot.task_states.items()
            }
            
            # Insert snapshot record
            await db_session.execute(
                insert(WorkflowStateSnapshotModel).values(
                    id=snapshot.snapshot_id,
                    workflow_id=snapshot.workflow_id,
                    execution_id=snapshot.execution_id,
                    snapshot_type=snapshot.snapshot_type.value,
                    batch_number=snapshot.batch_number,
                    state_data=asdict(snapshot),
                    task_states=task_states_json,
                    created_at=snapshot.timestamp,
                    can_resume_from=snapshot.can_resume_from
                )
            )
            
            await db_session.commit()
    
    async def _load_snapshot_from_db(self, snapshot_id: str) -> Optional[WorkflowStateSnapshot]:
        """Load a snapshot from the database."""
        async with get_session() as db_session:
            result = await db_session.execute(
                select(WorkflowStateSnapshotModel).where(
                    WorkflowStateSnapshotModel.id == snapshot_id
                )
            )
            snapshot_row = result.scalar_one_or_none()
            
            if snapshot_row:
                return self._row_to_snapshot(snapshot_row)
            
            return None
    
    def _row_to_snapshot(self, row) -> WorkflowStateSnapshot:
        """Convert database row to WorkflowStateSnapshot object."""
        # Deserialize task states
        task_states = {}
        if row.task_states:
            for task_id, state_data in row.task_states.items():
                task_states[task_id] = TaskState(
                    task_id=state_data['task_id'],
                    status=TaskStatus(state_data['status']),
                    agent_id=state_data.get('agent_id'),
                    started_at=datetime.fromisoformat(state_data['started_at']) if state_data.get('started_at') else None,
                    completed_at=datetime.fromisoformat(state_data['completed_at']) if state_data.get('completed_at') else None,
                    retry_count=state_data.get('retry_count', 0),
                    error_message=state_data.get('error_message'),
                    result_data=state_data.get('result_data'),
                    execution_time_ms=state_data.get('execution_time_ms', 0)
                )
        
        # Create snapshot object
        return WorkflowStateSnapshot(
            snapshot_id=str(row.id),
            workflow_id=str(row.workflow_id),
            execution_id=str(row.execution_id),
            snapshot_type=SnapshotType(row.snapshot_type),
            batch_number=row.batch_number,
            timestamp=row.created_at,
            workflow_status=WorkflowStatus(row.state_data.get('workflow_status', 'running')),
            task_states=task_states,
            execution_context=row.state_data.get('execution_context', {}),
            batch_progress=row.state_data.get('batch_progress', {}),
            error_information=row.state_data.get('error_information'),
            can_resume_from=row.can_resume_from
        )
    
    async def _cleanup_old_snapshots(self, workflow_id: str, keep_count: Optional[int] = None) -> None:
        """Clean up old snapshots for a workflow."""
        keep_count = keep_count or self.max_snapshots_per_workflow
        
        try:
            async with get_session() as db_session:
                # Get snapshot count
                count_result = await db_session.execute(
                    select(func.count(WorkflowStateSnapshotModel.id)).where(
                        WorkflowStateSnapshotModel.workflow_id == workflow_id
                    )
                )
                total_snapshots = count_result.scalar()
                
                if total_snapshots <= keep_count:
                    return
                
                # Delete oldest snapshots
                snapshots_to_delete = total_snapshots - keep_count
                
                # Get IDs of oldest snapshots
                old_snapshots_result = await db_session.execute(
                    select(WorkflowStateSnapshotModel.id).where(
                        WorkflowStateSnapshotModel.workflow_id == workflow_id
                    ).order_by(WorkflowStateSnapshotModel.created_at).limit(snapshots_to_delete)
                )
                old_snapshot_ids = [row[0] for row in old_snapshots_result]
                
                # Delete old snapshots
                if old_snapshot_ids:
                    await db_session.execute(
                        delete(WorkflowStateSnapshotModel).where(
                            WorkflowStateSnapshotModel.id.in_(old_snapshot_ids)
                        )
                    )
                    await db_session.commit()
                    
                    logger.info(
                        f"✅ Cleaned up {len(old_snapshot_ids)} old snapshots for workflow {workflow_id}"
                    )
        
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old snapshots for workflow {workflow_id}", error=str(e))
    
    async def _determine_recovery_strategy(
        self, 
        latest_snapshot: WorkflowStateSnapshot,
        failed_batch_number: Optional[int]
    ) -> RecoveryStrategy:
        """Determine the best recovery strategy based on failure analysis."""
        # Analyze failure context
        failed_tasks = [
            task_id for task_id, state in latest_snapshot.task_states.items()
            if state.status == TaskStatus.FAILED
        ]
        
        error_count = len(failed_tasks)
        total_tasks = len(latest_snapshot.task_states)
        
        # If few tasks failed and we have a clear failed batch, restart that batch
        if error_count <= 3 and failed_batch_number is not None:
            return RecoveryStrategy.RESTART_FAILED_BATCH
        
        # If many tasks failed, might need to restart from beginning
        elif error_count > total_tasks * 0.3:
            return RecoveryStrategy.RESTART_FROM_BEGINNING
        
        # If moderate failures, skip failed tasks and continue
        elif error_count > 0:
            return RecoveryStrategy.SKIP_FAILED_TASKS
        
        # Default to resuming from last checkpoint
        else:
            return RecoveryStrategy.RESUME_FROM_LAST_CHECKPOINT
    
    async def _find_recovery_target_snapshot(
        self,
        snapshots: List[WorkflowStateSnapshot],
        strategy: RecoveryStrategy,
        failed_batch_number: Optional[int]
    ) -> Optional[WorkflowStateSnapshot]:
        """Find the target snapshot for recovery based on strategy."""
        for snapshot in snapshots:
            if not snapshot.can_resume_from:
                continue
            
            if strategy == RecoveryStrategy.RESTART_FAILED_BATCH:
                if failed_batch_number and snapshot.batch_number < failed_batch_number:
                    return snapshot
            
            elif strategy == RecoveryStrategy.RESTART_FROM_BEGINNING:
                if snapshot.batch_number == 0:
                    return snapshot
            
            elif strategy in [RecoveryStrategy.RESUME_FROM_LAST_CHECKPOINT, RecoveryStrategy.SKIP_FAILED_TASKS]:
                return snapshot
        
        # Fallback to latest resumable snapshot
        return next((s for s in snapshots if s.can_resume_from), None)
    
    async def _analyze_task_recovery_needs(
        self,
        target_snapshot: WorkflowStateSnapshot,
        latest_snapshot: WorkflowStateSnapshot,
        strategy: RecoveryStrategy
    ) -> Tuple[List[str], List[str]]:
        """Analyze which tasks need to be retried or skipped."""
        tasks_to_retry = []
        tasks_to_skip = []
        
        for task_id, latest_state in latest_snapshot.task_states.items():
            target_state = target_snapshot.task_states.get(task_id)
            
            if strategy == RecoveryStrategy.SKIP_FAILED_TASKS:
                if latest_state.status == TaskStatus.FAILED:
                    tasks_to_skip.append(task_id)
                elif not target_state or target_state.status != TaskStatus.COMPLETED:
                    tasks_to_retry.append(task_id)
            
            elif strategy == RecoveryStrategy.RESTART_FAILED_BATCH:
                if not target_state or target_state.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.FAILED]:
                    tasks_to_retry.append(task_id)
            
            elif strategy == RecoveryStrategy.RESTART_FROM_BEGINNING:
                tasks_to_retry.append(task_id)
            
            else:  # RESUME_FROM_LAST_CHECKPOINT
                if not target_state or target_state.status != TaskStatus.COMPLETED:
                    tasks_to_retry.append(task_id)
        
        return tasks_to_retry, tasks_to_skip
    
    async def _estimate_recovery_time(
        self,
        target_snapshot: WorkflowStateSnapshot,
        tasks_to_retry: List[str]
    ) -> int:
        """Estimate recovery time based on task history."""
        if not tasks_to_retry:
            return 0
        
        # Simple estimation based on average task execution time
        total_estimated_time = 0
        
        for task_id in tasks_to_retry:
            task_state = target_snapshot.task_states.get(task_id)
            if task_state and task_state.execution_time_ms > 0:
                total_estimated_time += task_state.execution_time_ms
            else:
                # Default estimate: 30 minutes per task
                total_estimated_time += 30 * 60 * 1000
        
        # Add 20% buffer for overhead
        return int(total_estimated_time * 1.2)
    
    async def _update_workflow_status(self, workflow_id: str, status: WorkflowStatus) -> None:
        """Update workflow status in the database."""
        async with get_session() as db_session:
            await db_session.execute(
                update(Workflow).where(Workflow.id == workflow_id).values(
                    status=status,
                    updated_at=datetime.utcnow()
                )
            )
            await db_session.commit()
    
    async def _restore_task_states(
        self, 
        target_snapshot: WorkflowStateSnapshot,
        recovery_plan: RecoveryPlan
    ) -> None:
        """Restore task states from a target snapshot."""
        async with get_session() as db_session:
            for task_id, task_state in target_snapshot.task_states.items():
                # Skip tasks that should be skipped in recovery
                if task_id in recovery_plan.tasks_to_skip:
                    continue
                
                # Reset tasks that should be retried
                if task_id in recovery_plan.tasks_to_retry:
                    status = TaskStatus.PENDING
                    completed_at = None
                    error_message = None
                else:
                    status = task_state.status
                    completed_at = task_state.completed_at
                    error_message = task_state.error_message
                
                await db_session.execute(
                    update(Task).where(Task.id == task_id).values(
                        status=status,
                        started_at=task_state.started_at,
                        completed_at=completed_at,
                        error_message=error_message,
                        updated_at=datetime.utcnow()
                    )
                )
            
            await db_session.commit()
    
    def _update_snapshot_metrics(self, snapshot: WorkflowStateSnapshot, snapshot_time_ms: int) -> None:
        """Update performance metrics after snapshot creation."""
        self.metrics['snapshots_created'] += 1
        
        # Estimate snapshot size (rough approximation)
        snapshot_size_kb = len(json.dumps(asdict(snapshot), default=str)) / 1024
        
        # Update average snapshot size
        current_avg_size = self.metrics['average_snapshot_size_kb']
        total_snapshots = self.metrics['snapshots_created']
        new_avg_size = ((current_avg_size * (total_snapshots - 1)) + snapshot_size_kb) / total_snapshots
        self.metrics['average_snapshot_size_kb'] = new_avg_size
        
        # Update average snapshot time
        current_avg_time = self.metrics['average_snapshot_time_ms']
        new_avg_time = ((current_avg_time * (total_snapshots - 1)) + snapshot_time_ms) / total_snapshots
        self.metrics['average_snapshot_time_ms'] = new_avg_time


# Database model placeholder (would be defined in models/)
class WorkflowStateSnapshotModel:
    """Placeholder for actual database model."""
    pass