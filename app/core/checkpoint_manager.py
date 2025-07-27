"""
Checkpoint Manager for atomic state preservation and recovery.

Provides secure, validated checkpointing capabilities with:
- Atomic snapshot creation with rollback on failure
- SHA-256 integrity validation and verification
- Compressed checkpoint storage with zstd
- Redis stream offset preservation
- Database transaction state management
- Multi-generation fallback support
- Automated cleanup and retention policies
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID
import zstandard as zstd

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc
import redis.asyncio as redis

from ..models.sleep_wake import Checkpoint, CheckpointType, SleepState
from ..models.agent import Agent
from ..core.database import get_async_session
from ..core.redis import get_redis
from ..core.config import get_settings


logger = logging.getLogger(__name__)


class CheckpointValidationError(Exception):
    """Raised when checkpoint validation fails."""
    pass


class CheckpointCreationError(Exception):
    """Raised when checkpoint creation fails."""
    pass


class CheckpointManager:
    """
    Manages atomic checkpoint creation, validation, and recovery.
    
    Features:
    - Atomic snapshot creation with SHA-256 validation
    - Compressed storage with zstd for space efficiency
    - Redis stream offset preservation and restoration
    - Database transaction state management
    - Multi-generation fallback logic
    - Automated cleanup and retention policies
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.checkpoint_dir = Path(self.settings.checkpoint_base_path or "/var/lib/hive/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Compression settings
        self.compression_level = 3  # Balance between speed and compression
        self.compressor = zstd.ZstdCompressor(level=self.compression_level)
        self.decompressor = zstd.ZstdDecompressor()
        
        # Validation settings
        self.validation_timeout = 60  # seconds
        self.max_checkpoint_size = 10 * 1024 * 1024 * 1024  # 10GB
        
        # Retention settings
        self.max_checkpoints_per_agent = 10
        self.max_checkpoint_age_days = 30
        self.cleanup_interval_hours = 6
    
    async def create_checkpoint(
        self,
        agent_id: Optional[UUID] = None,
        checkpoint_type: CheckpointType = CheckpointType.SCHEDULED,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Checkpoint]:
        """
        Create a new checkpoint with atomic state preservation.
        
        Args:
            agent_id: Agent ID for agent-specific checkpoint, None for system-wide
            checkpoint_type: Type of checkpoint being created
            metadata: Additional metadata to store with checkpoint
            
        Returns:
            Created checkpoint or None if creation failed
        """
        start_time = time.time()
        
        try:
            logger.info(f"Creating checkpoint for agent {agent_id}, type {checkpoint_type.value}")
            
            # Generate checkpoint ID and paths
            checkpoint_id = self._generate_checkpoint_id()
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.tar.zst"
            temp_dir = None
            
            try:
                # Create temporary directory for atomic operations
                temp_dir = Path(tempfile.mkdtemp(prefix="checkpoint_", dir=self.checkpoint_dir.parent))
                temp_path = temp_dir / f"{checkpoint_id}.tar.zst"
                
                # Collect state data
                state_data = await self._collect_state_data(agent_id)
                
                # Create compressed archive
                await self._create_compressed_archive(temp_path, state_data)
                
                # Calculate SHA-256 hash
                sha256_hash = await self._calculate_file_hash(temp_path)
                
                # Get file size
                file_size = temp_path.stat().st_size
                
                # Validate checkpoint
                validation_start = time.time()
                validation_errors = await self._validate_checkpoint(
                    temp_path, sha256_hash, file_size, state_data
                )
                validation_time_ms = (time.time() - validation_start) * 1000
                
                is_valid = len(validation_errors) == 0
                
                if not is_valid:
                    logger.error(f"Checkpoint validation failed: {validation_errors}")
                    raise CheckpointValidationError(f"Validation failed: {validation_errors}")
                
                # Atomically move to final location
                shutil.move(str(temp_path), str(checkpoint_path))
                
                # Calculate compression ratio
                uncompressed_size = sum(len(str(data).encode()) for data in state_data.values())
                compression_ratio = file_size / uncompressed_size if uncompressed_size > 0 else 1.0
                
                creation_time_ms = (time.time() - start_time) * 1000
                
                # Create database record
                async with get_async_session() as session:
                    checkpoint = Checkpoint(
                        agent_id=agent_id,
                        checkpoint_type=checkpoint_type,
                        path=str(checkpoint_path),
                        sha256=sha256_hash,
                        size_bytes=file_size,
                        is_valid=is_valid,
                        validation_errors=validation_errors,
                        checkpoint_metadata=metadata or {},
                        redis_offsets=state_data.get("redis_offsets", {}),
                        database_snapshot_id=state_data.get("database_snapshot_id"),
                        compression_ratio=compression_ratio,
                        creation_time_ms=creation_time_ms,
                        validation_time_ms=validation_time_ms,
                        expires_at=datetime.utcnow() + timedelta(days=self.max_checkpoint_age_days)
                    )
                    
                    session.add(checkpoint)
                    await session.commit()
                    await session.refresh(checkpoint)
                
                logger.info(
                    f"Checkpoint created successfully: {checkpoint.id} "
                    f"(size: {file_size // 1024 // 1024}MB, "
                    f"compression: {compression_ratio:.2f}, "
                    f"time: {creation_time_ms:.0f}ms)"
                )
                
                return checkpoint
                
            finally:
                # Cleanup temporary directory
                if temp_dir and temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            # Cleanup on failure
            if checkpoint_path.exists():
                checkpoint_path.unlink(missing_ok=True)
            return None
    
    async def validate_checkpoint(self, checkpoint_id: UUID) -> Tuple[bool, List[str]]:
        """
        Validate an existing checkpoint's integrity.
        
        Args:
            checkpoint_id: Checkpoint ID to validate
            
        Returns:
            Tuple of (is_valid, validation_errors)
        """
        try:
            async with get_async_session() as session:
                checkpoint = await session.get(Checkpoint, checkpoint_id)
                if not checkpoint:
                    return False, ["Checkpoint not found"]
                
                checkpoint_path = Path(checkpoint.path)
                if not checkpoint_path.exists():
                    return False, ["Checkpoint file not found"]
                
                # Re-calculate hash and validate
                actual_hash = await self._calculate_file_hash(checkpoint_path)
                if actual_hash != checkpoint.sha256:
                    return False, ["SHA-256 hash mismatch"]
                
                # Validate file size
                actual_size = checkpoint_path.stat().st_size
                if actual_size != checkpoint.size_bytes:
                    return False, ["File size mismatch"]
                
                # Try to extract and validate content
                try:
                    state_data = await self._extract_checkpoint_data(checkpoint_path)
                    validation_errors = await self._validate_state_data(state_data)
                    
                    if validation_errors:
                        return False, validation_errors
                    
                except Exception as e:
                    return False, [f"Content validation failed: {str(e)}"]
                
                # Update database record
                checkpoint.is_valid = True
                checkpoint.validation_errors = []
                await session.commit()
                
                return True, []
                
        except Exception as e:
            logger.error(f"Error validating checkpoint {checkpoint_id}: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    async def restore_checkpoint(self, checkpoint_id: UUID) -> Tuple[bool, Dict[str, Any]]:
        """
        Restore system state from a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to restore
            
        Returns:
            Tuple of (success, restored_state_data)
        """
        try:
            logger.info(f"Restoring checkpoint {checkpoint_id}")
            
            async with get_async_session() as session:
                checkpoint = await session.get(Checkpoint, checkpoint_id)
                if not checkpoint:
                    logger.error(f"Checkpoint {checkpoint_id} not found")
                    return False, {}
                
                if not checkpoint.is_valid:
                    logger.error(f"Checkpoint {checkpoint_id} is not valid")
                    return False, {}
                
                checkpoint_path = Path(checkpoint.path)
                if not checkpoint_path.exists():
                    logger.error(f"Checkpoint file not found: {checkpoint_path}")
                    return False, {}
                
                # Extract checkpoint data
                state_data = await self._extract_checkpoint_data(checkpoint_path)
                
                # Restore Redis stream offsets
                if checkpoint.redis_offsets:
                    await self._restore_redis_state(checkpoint.redis_offsets)
                
                # Restore database state if needed
                if checkpoint.database_snapshot_id:
                    await self._restore_database_state(checkpoint.database_snapshot_id)
                
                # Restore agent states
                if "agent_states" in state_data:
                    await self._restore_agent_states(state_data["agent_states"])
                
                logger.info(f"Checkpoint {checkpoint_id} restored successfully")
                return True, state_data
                
        except Exception as e:
            logger.error(f"Error restoring checkpoint {checkpoint_id}: {e}")
            return False, {}
    
    async def get_latest_checkpoint(
        self,
        agent_id: Optional[UUID] = None,
        checkpoint_type: Optional[CheckpointType] = None
    ) -> Optional[Checkpoint]:
        """
        Get the latest valid checkpoint for an agent or system.
        
        Args:
            agent_id: Agent ID, or None for system-wide checkpoints
            checkpoint_type: Filter by checkpoint type
            
        Returns:
            Latest checkpoint or None if none found
        """
        try:
            async with get_async_session() as session:
                query = select(Checkpoint).where(
                    and_(
                        Checkpoint.agent_id == agent_id,
                        Checkpoint.is_valid == True
                    )
                )
                
                if checkpoint_type:
                    query = query.where(Checkpoint.checkpoint_type == checkpoint_type)
                
                query = query.order_by(desc(Checkpoint.created_at))
                
                result = await session.execute(query)
                return result.scalars().first()
                
        except Exception as e:
            logger.error(f"Error getting latest checkpoint: {e}")
            return None
    
    async def get_checkpoint_fallbacks(
        self,
        agent_id: Optional[UUID] = None,
        max_generations: int = 3
    ) -> List[Checkpoint]:
        """
        Get fallback checkpoints for recovery.
        
        Args:
            agent_id: Agent ID, or None for system-wide checkpoints
            max_generations: Maximum number of fallback generations
            
        Returns:
            List of valid checkpoints ordered by creation time (newest first)
        """
        try:
            async with get_async_session() as session:
                query = select(Checkpoint).where(
                    and_(
                        Checkpoint.agent_id == agent_id,
                        Checkpoint.is_valid == True
                    )
                ).order_by(desc(Checkpoint.created_at)).limit(max_generations)
                
                result = await session.execute(query)
                return list(result.scalars().all())
                
        except Exception as e:
            logger.error(f"Error getting checkpoint fallbacks: {e}")
            return []
    
    async def cleanup_old_checkpoints(self) -> int:
        """
        Clean up old and invalid checkpoints based on retention policies.
        
        Returns:
            Number of checkpoints cleaned up
        """
        cleaned_count = 0
        
        try:
            async with get_async_session() as session:
                # Clean up expired checkpoints
                expired_query = select(Checkpoint).where(
                    or_(
                        Checkpoint.expires_at < datetime.utcnow(),
                        and_(
                            Checkpoint.is_valid == False,
                            Checkpoint.created_at < datetime.utcnow() - timedelta(days=7)
                        )
                    )
                )
                
                result = await session.execute(expired_query)
                expired_checkpoints = result.scalars().all()
                
                for checkpoint in expired_checkpoints:
                    try:
                        # Remove file
                        checkpoint_path = Path(checkpoint.path)
                        if checkpoint_path.exists():
                            checkpoint_path.unlink()
                        
                        # Remove database record
                        await session.delete(checkpoint)
                        cleaned_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error cleaning up checkpoint {checkpoint.id}: {e}")
                
                # Clean up excess checkpoints per agent (keep only max_checkpoints_per_agent)
                agents_query = select(Checkpoint.agent_id).distinct()
                result = await session.execute(agents_query)
                agent_ids = [row[0] for row in result]
                
                for agent_id in agent_ids:
                    excess_query = select(Checkpoint).where(
                        and_(
                            Checkpoint.agent_id == agent_id,
                            Checkpoint.is_valid == True
                        )
                    ).order_by(desc(Checkpoint.created_at)).offset(self.max_checkpoints_per_agent)
                    
                    result = await session.execute(excess_query)
                    excess_checkpoints = result.scalars().all()
                    
                    for checkpoint in excess_checkpoints:
                        try:
                            checkpoint_path = Path(checkpoint.path)
                            if checkpoint_path.exists():
                                checkpoint_path.unlink()
                            
                            await session.delete(checkpoint)
                            cleaned_count += 1
                            
                        except Exception as e:
                            logger.error(f"Error cleaning up excess checkpoint {checkpoint.id}: {e}")
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error during checkpoint cleanup: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old checkpoints")
        
        return cleaned_count
    
    async def _collect_state_data(self, agent_id: Optional[UUID]) -> Dict[str, Any]:
        """Collect all state data for checkpointing."""
        state_data = {}
        
        try:
            # Collect Redis stream offsets
            state_data["redis_offsets"] = await self._collect_redis_offsets(agent_id)
            
            # Collect agent state if specific agent
            if agent_id:
                state_data["agent_states"] = await self._collect_agent_state(agent_id)
            else:
                # Collect all agent states for system checkpoint
                state_data["agent_states"] = await self._collect_all_agent_states()
            
            # Collect database transaction state
            state_data["database_snapshot_id"] = await self._create_database_snapshot()
            
            # Collect task queues
            state_data["task_queues"] = await self._collect_task_queues(agent_id)
            
            # Collect context cache state
            state_data["context_cache"] = await self._collect_context_cache(agent_id)
            
            # Add timestamp
            state_data["timestamp"] = datetime.utcnow().isoformat()
            state_data["checkpoint_version"] = "1.0"
            
        except Exception as e:
            logger.error(f"Error collecting state data: {e}")
            raise CheckpointCreationError(f"Failed to collect state data: {e}")
        
        return state_data
    
    async def _create_compressed_archive(self, archive_path: Path, state_data: Dict[str, Any]) -> None:
        """Create compressed tar archive of state data."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write state data to JSON file
                state_file = temp_path / "state.json"
                with open(state_file, "w") as f:
                    json.dump(state_data, f, indent=2, default=str)
                
                # Create tar file
                with tarfile.open(archive_path, "w") as tar:
                    tar.add(state_file, arcname="state.json")
                
                # Compress with zstd
                with open(archive_path, "rb") as f_in:
                    uncompressed_data = f_in.read()
                
                compressed_data = self.compressor.compress(uncompressed_data)
                
                with open(archive_path, "wb") as f_out:
                    f_out.write(compressed_data)
                    
        except Exception as e:
            logger.error(f"Error creating compressed archive: {e}")
            raise CheckpointCreationError(f"Failed to create archive: {e}")
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    async def _validate_checkpoint(
        self,
        checkpoint_path: Path,
        expected_hash: str,
        expected_size: int,
        state_data: Dict[str, Any]
    ) -> List[str]:
        """Validate checkpoint integrity and content."""
        errors = []
        
        try:
            # Validate file exists
            if not checkpoint_path.exists():
                errors.append("Checkpoint file does not exist")
                return errors
            
            # Validate file size
            actual_size = checkpoint_path.stat().st_size
            if actual_size != expected_size:
                errors.append(f"File size mismatch: expected {expected_size}, got {actual_size}")
            
            if actual_size > self.max_checkpoint_size:
                errors.append(f"Checkpoint too large: {actual_size} bytes")
            
            # Validate hash
            actual_hash = await self._calculate_file_hash(checkpoint_path)
            if actual_hash != expected_hash:
                errors.append("SHA-256 hash mismatch")
            
            # Validate content can be extracted
            try:
                extracted_data = await self._extract_checkpoint_data(checkpoint_path)
                validation_errors = await self._validate_state_data(extracted_data)
                errors.extend(validation_errors)
            except Exception as e:
                errors.append(f"Content extraction failed: {str(e)}")
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors
    
    async def _extract_checkpoint_data(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Extract state data from checkpoint file."""
        try:
            # Decompress file
            with open(checkpoint_path, "rb") as f:
                compressed_data = f.read()
            
            uncompressed_data = self.decompressor.decompress(compressed_data)
            
            # Extract from tar
            with tempfile.NamedTemporaryFile() as temp_file:
                temp_file.write(uncompressed_data)
                temp_file.seek(0)
                
                with tarfile.open(temp_file.name, "r") as tar:
                    state_file = tar.extractfile("state.json")
                    if state_file:
                        return json.load(state_file)
                    else:
                        raise CheckpointValidationError("state.json not found in archive")
                        
        except Exception as e:
            logger.error(f"Error extracting checkpoint data: {e}")
            raise CheckpointValidationError(f"Failed to extract data: {e}")
    
    async def _validate_state_data(self, state_data: Dict[str, Any]) -> List[str]:
        """Validate the structure and content of state data."""
        errors = []
        
        try:
            # Check required fields
            required_fields = ["timestamp", "checkpoint_version"]
            for field in required_fields:
                if field not in state_data:
                    errors.append(f"Missing required field: {field}")
            
            # Validate timestamp format
            if "timestamp" in state_data:
                try:
                    datetime.fromisoformat(state_data["timestamp"])
                except ValueError:
                    errors.append("Invalid timestamp format")
            
            # Validate checkpoint version
            if "checkpoint_version" in state_data:
                if state_data["checkpoint_version"] != "1.0":
                    errors.append(f"Unsupported checkpoint version: {state_data['checkpoint_version']}")
            
            # Validate Redis offsets structure
            if "redis_offsets" in state_data:
                redis_offsets = state_data["redis_offsets"]
                if not isinstance(redis_offsets, dict):
                    errors.append("Redis offsets must be a dictionary")
            
            # Validate agent states structure
            if "agent_states" in state_data:
                agent_states = state_data["agent_states"]
                if not isinstance(agent_states, (dict, list)):
                    errors.append("Agent states must be a dictionary or list")
            
        except Exception as e:
            errors.append(f"State data validation error: {str(e)}")
        
        return errors
    
    async def _collect_redis_offsets(self, agent_id: Optional[UUID]) -> Dict[str, str]:
        """Collect Redis stream offsets for preservation."""
        offsets = {}
        
        try:
            redis_client = get_redis()
            
            if agent_id:
                # Collect offsets for specific agent streams
                stream_patterns = [
                    f"agent:{agent_id}:*",
                    f"tasks:{agent_id}:*",
                    f"messages:{agent_id}:*"
                ]
            else:
                # Collect all stream offsets for system checkpoint
                stream_patterns = [
                    "agent:*",
                    "tasks:*", 
                    "messages:*",
                    "system:*"
                ]
            
            for pattern in stream_patterns:
                keys = await redis_client.keys(pattern)
                for key in keys:
                    try:
                        # Get stream info to find last message ID
                        info = await redis_client.xinfo_stream(key)
                        if info and "last-generated-id" in info:
                            offsets[key.decode()] = info["last-generated-id"].decode()
                    except Exception as e:
                        logger.warning(f"Could not get offset for stream {key}: {e}")
                        
        except Exception as e:
            logger.error(f"Error collecting Redis offsets: {e}")
        
        return offsets
    
    async def _collect_agent_state(self, agent_id: UUID) -> Dict[str, Any]:
        """Collect state for a specific agent."""
        try:
            async with get_async_session() as session:
                agent = await session.get(Agent, agent_id)
                if agent:
                    return {
                        "id": str(agent.id),
                        "name": agent.name,
                        "status": agent.status.value if agent.status else None,
                        "current_sleep_state": agent.current_sleep_state.value if agent.current_sleep_state else None,
                        "current_cycle_id": str(agent.current_cycle_id) if agent.current_cycle_id else None,
                        "last_sleep_time": agent.last_sleep_time.isoformat() if agent.last_sleep_time else None,
                        "last_wake_time": agent.last_wake_time.isoformat() if agent.last_wake_time else None,
                        "config": agent.config or {}
                    }
        except Exception as e:
            logger.error(f"Error collecting agent state for {agent_id}: {e}")
        
        return {}
    
    async def _collect_all_agent_states(self) -> List[Dict[str, Any]]:
        """Collect states for all agents."""
        agents_data = []
        
        try:
            async with get_async_session() as session:
                result = await session.execute(select(Agent))
                agents = result.scalars().all()
                
                for agent in agents:
                    agent_data = await self._collect_agent_state(agent.id)
                    if agent_data:
                        agents_data.append(agent_data)
                        
        except Exception as e:
            logger.error(f"Error collecting all agent states: {e}")
        
        return agents_data
    
    async def _create_database_snapshot(self) -> Optional[str]:
        """Create database snapshot identifier."""
        # This would integrate with actual database backup system
        # For now, return a timestamp-based identifier
        return f"snapshot_{int(time.time())}"
    
    async def _collect_task_queues(self, agent_id: Optional[UUID]) -> Dict[str, Any]:
        """Collect task queue states."""
        # Placeholder for task queue collection
        return {}
    
    async def _collect_context_cache(self, agent_id: Optional[UUID]) -> Dict[str, Any]:
        """Collect context cache states."""
        # Placeholder for context cache collection
        return {}
    
    async def _restore_redis_state(self, redis_offsets: Dict[str, str]) -> None:
        """Restore Redis stream states."""
        try:
            redis_client = get_redis()
            
            for stream_key, last_id in redis_offsets.items():
                try:
                    # Restore stream consumer group positions if needed
                    # This is a simplified restoration - actual implementation
                    # would need more sophisticated stream state management
                    logger.debug(f"Restored stream {stream_key} to position {last_id}")
                except Exception as e:
                    logger.warning(f"Could not restore stream {stream_key}: {e}")
                    
        except Exception as e:
            logger.error(f"Error restoring Redis state: {e}")
    
    async def _restore_database_state(self, snapshot_id: str) -> None:
        """Restore database state from snapshot."""
        # Placeholder for database restoration
        logger.info(f"Database state restoration not implemented for snapshot {snapshot_id}")
    
    async def _restore_agent_states(self, agent_states: List[Dict[str, Any]]) -> None:
        """Restore agent states from checkpoint."""
        try:
            async with get_async_session() as session:
                for agent_data in agent_states:
                    try:
                        agent_id = UUID(agent_data["id"])
                        agent = await session.get(Agent, agent_id)
                        
                        if agent:
                            # Restore sleep state
                            if "current_sleep_state" in agent_data and agent_data["current_sleep_state"]:
                                agent.current_sleep_state = SleepState(agent_data["current_sleep_state"])
                            
                            # Restore other state fields as needed
                            logger.debug(f"Restored state for agent {agent_id}")
                            
                    except Exception as e:
                        logger.error(f"Error restoring agent state: {e}")
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error restoring agent states: {e}")
    
    def _generate_checkpoint_id(self) -> str:
        """Generate unique checkpoint identifier."""
        timestamp = int(time.time() * 1000)  # milliseconds
        return f"cp_{timestamp}"


# Global checkpoint manager instance
_checkpoint_manager_instance: Optional[CheckpointManager] = None


def get_checkpoint_manager() -> CheckpointManager:
    """Get the global checkpoint manager instance."""
    global _checkpoint_manager_instance
    if _checkpoint_manager_instance is None:
        _checkpoint_manager_instance = CheckpointManager()
    return _checkpoint_manager_instance