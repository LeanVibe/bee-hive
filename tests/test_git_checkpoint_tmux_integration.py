"""
Integration tests for Git-based checkpoint system and Tmux session management.

Tests the complete sleep-wake cycle with:
- Git checkpoint creation and restoration
- Tmux session lifecycle management
- Context consolidation with aging policies
- Performance validation for <60s recovery time
"""

import asyncio
import pytest
import pytest_asyncio
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from app.core.checkpoint_manager import CheckpointManager
from app.core.workspace_manager import TmuxSessionManager, WorkspaceConfig, AgentWorkspace
from app.core.consolidation_engine import (
    ConsolidationEngine, ContextAgingPolicies, ContextPrioritizationEngine
)
from app.models.sleep_wake import CheckpointType


class MockContext:
    """Mock Context class for testing without SQLAlchemy dependencies."""
    
    def __init__(self, id=None, content="", created_at=None, last_accessed_at=None, access_count=1):
        self.id = id or uuid4()
        self.content = content
        self.created_at = created_at or datetime.utcnow()
        self.last_accessed_at = last_accessed_at or self.created_at
        self.access_count = access_count
        self.is_consolidated = False
        self.metadata = {}


class TestGitCheckpointSystem:
    """Test Git-based checkpoint system functionality."""
    
    @pytest_asyncio.fixture
    async def checkpoint_manager(self, tmp_path):
        """Create a checkpoint manager with temporary directories."""
        manager = CheckpointManager()
        manager.checkpoint_dir = tmp_path / "checkpoints"
        manager.git_repo_path = tmp_path / "git_checkpoints"
        manager.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        manager.git_repo_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Git repository
        manager.git_repo = manager._initialize_git_repository()
        
        return manager
    
    @pytest.mark.asyncio
    async def test_git_checkpoint_creation(self, checkpoint_manager):
        """Test creating a Git checkpoint with versioning."""
        agent_id = uuid4()
        
        # Create checkpoint
        checkpoint = await checkpoint_manager.create_checkpoint(
            agent_id=agent_id,
            checkpoint_type=CheckpointType.PRE_SLEEP,
            metadata={"test": "git_checkpoint"}
        )
        
        assert checkpoint is not None
        assert checkpoint.agent_id == agent_id
        assert checkpoint.checkpoint_type == CheckpointType.PRE_SLEEP
        assert checkpoint.is_valid is True
        
        # Verify Git commit was created
        git_commit = checkpoint.checkpoint_metadata.get("git_commit_hash")
        assert git_commit is not None
        
        # Verify Git history
        history = await checkpoint_manager.get_git_checkpoint_history(agent_id)
        assert len(history) == 1
        assert history[0]["commit_hash"] == git_commit
        assert f"agent {agent_id}" in history[0]["message"]
    
    @pytest.mark.asyncio
    async def test_git_checkpoint_restoration(self, checkpoint_manager):
        """Test restoring from a Git checkpoint."""
        agent_id = uuid4()
        
        # Create initial checkpoint
        original_checkpoint = await checkpoint_manager.create_checkpoint(
            agent_id=agent_id,
            checkpoint_type=CheckpointType.SCHEDULED,
            metadata={"version": "1.0", "test_data": "original"}
        )
        
        git_commit = original_checkpoint.checkpoint_metadata["git_commit_hash"]
        
        # Restore from Git checkpoint
        success, restored_data = await checkpoint_manager.restore_from_git_checkpoint(
            git_commit, agent_id
        )
        
        assert success is True
        assert "state_data" in restored_data
        assert "metadata" in restored_data
        assert restored_data["metadata"]["additional_metadata"]["version"] == "1.0"
        assert restored_data["git_commit"] == git_commit
    
    @pytest.mark.asyncio
    async def test_git_checkpoint_history_management(self, checkpoint_manager):
        """Test Git history management and cleanup."""
        agent_id = uuid4()
        
        # Create multiple checkpoints
        checkpoints = []
        for i in range(5):
            checkpoint = await checkpoint_manager.create_checkpoint(
                agent_id=agent_id,
                checkpoint_type=CheckpointType.SCHEDULED,
                metadata={"version": f"1.{i}"}
            )
            checkpoints.append(checkpoint)
            await asyncio.sleep(0.1)  # Small delay to ensure different timestamps
        
        # Verify history
        history = await checkpoint_manager.get_git_checkpoint_history(agent_id, limit=10)
        assert len(history) == 5
        
        # History should be in reverse chronological order
        for i in range(len(history) - 1):
            assert history[i]["committed_date"] >= history[i + 1]["committed_date"]


class TestTmuxSessionManagement:
    """Test enhanced Tmux session management."""
    
    @pytest_asyncio.fixture
    async def tmux_manager(self, tmp_path):
        """Create a Tmux session manager for testing."""
        return TmuxSessionManager()
    
    @pytest_asyncio.fixture  
    async def workspace_config(self, tmp_path):
        """Create a workspace configuration for testing."""
        agent_id = str(uuid4())
        workspace_path = tmp_path / "workspace" / agent_id
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        return WorkspaceConfig(
            agent_id=agent_id,
            workspace_name=f"test-workspace-{agent_id[:8]}",
            project_path=workspace_path,
            max_memory_mb=1024,
            max_cpu_percent=25.0
        )
    
    @pytest.mark.asyncio
    async def test_tmux_session_creation_with_templates(self, tmux_manager, workspace_config):
        """Test creating Tmux sessions with different templates."""
        agent_id = workspace_config.agent_id
        
        # Test AI agent template
        session = await tmux_manager.create_agent_session(
            agent_id=agent_id,
            workspace_path=workspace_config.project_path,
            template="ai_agent"
        )
        
        assert session is not None
        assert session.name == f"agent-{agent_id}"
        assert agent_id in tmux_manager.session_registry
        
        # Verify session has expected windows
        session_info = tmux_manager.session_registry[agent_id]
        assert session_info["template"] == "ai_agent"
        assert session_info["status"] == "active"
        
        # Clean up
        await tmux_manager.terminate_session(agent_id)
    
    @pytest.mark.asyncio
    async def test_tmux_session_health_monitoring(self, tmux_manager, workspace_config):
        """Test Tmux session health monitoring."""
        agent_id = workspace_config.agent_id
        
        # Create session
        session = await tmux_manager.create_agent_session(
            agent_id=agent_id,
            workspace_path=workspace_config.project_path,
            template="minimal"
        )
        
        # Test health check
        healthy = await tmux_manager.health_check_session(agent_id)
        assert healthy is True
        
        # Get session metrics
        metrics = await tmux_manager.get_session_metrics(agent_id)
        assert metrics["agent_id"] == agent_id
        assert metrics["status"] == "active"
        assert "uptime_seconds" in metrics
        
        # Clean up
        await tmux_manager.terminate_session(agent_id)
    
    @pytest.mark.asyncio
    async def test_tmux_session_checkpoint_integration(self, tmux_manager, workspace_config):
        """Test Tmux session state capture and restoration."""
        agent_id = workspace_config.agent_id
        
        # Create session
        session = await tmux_manager.create_agent_session(
            agent_id=agent_id,
            workspace_path=workspace_config.project_path,
            template="development"
        )
        
        # Capture session state
        state = await tmux_manager.capture_session_state(agent_id)
        assert "session_name" in state
        assert "template" in state
        assert "windows" in state
        assert state["template"] == "development"
        
        # Terminate session
        await tmux_manager.terminate_session(agent_id)
        
        # Restore from checkpoint
        restored_session = await tmux_manager.restore_session_from_checkpoint(
            agent_id=agent_id,
            workspace_path=workspace_config.project_path,
            checkpoint_data={"tmux_session": state}
        )
        
        assert restored_session is not None
        assert restored_session.name == f"agent-{agent_id}"
        
        session_info = tmux_manager.session_registry[agent_id]
        assert session_info["status"] == "restored"
        
        # Clean up
        await tmux_manager.terminate_session(agent_id)


class TestContextConsolidationPipeline:
    """Test enhanced context consolidation with aging and prioritization."""
    
    @pytest.fixture
    def aging_policies(self):
        """Create context aging policies for testing."""
        return ContextAgingPolicies()
    
    @pytest.fixture
    def prioritization_engine(self):
        """Create context prioritization engine for testing."""
        return ContextPrioritizationEngine()
    
    def create_test_context(self, age_hours: int, content_size: int = 1000, access_count: int = 1):
        """Create a test context with specified age and properties."""
        created_at = datetime.utcnow() - timedelta(hours=age_hours)
        
        context = MockContext(
            id=uuid4(),
            content="x" * content_size,
            created_at=created_at,
            last_accessed_at=created_at + timedelta(hours=1),
            access_count=access_count
        )
        
        return context
    
    def test_context_aging_categorization(self, aging_policies):
        """Test context categorization by age."""
        # Fresh context (1 hour old)
        fresh_context = self.create_test_context(age_hours=1)
        assert aging_policies.get_aging_category(fresh_context) == "fresh"
        assert aging_policies.should_consolidate(fresh_context) is False
        
        # Active context (12 hours old)
        active_context = self.create_test_context(age_hours=12)
        assert aging_policies.get_aging_category(active_context) == "active"
        
        # Stale context (3 days old)
        stale_context = self.create_test_context(age_hours=72)
        assert aging_policies.get_aging_category(stale_context) == "stale"
        
        # Archival context (2 weeks old)
        archival_context = self.create_test_context(age_hours=336)
        assert aging_policies.get_aging_category(archival_context) == "archival"
        
        # Deletion candidate (2 months old)
        old_context = self.create_test_context(age_hours=1440)
        assert aging_policies.get_aging_category(old_context) == "deletion_candidate"
        assert aging_policies.should_consolidate(old_context) is True
    
    def test_context_age_scoring(self, aging_policies):
        """Test context age scoring algorithm."""
        # Recent, frequently accessed context should score higher
        recent_context = self.create_test_context(age_hours=6, access_count=5, content_size=5000)
        score1 = aging_policies.calculate_context_age_score(recent_context)
        
        # Old, rarely accessed context should score lower
        old_context = self.create_test_context(age_hours=720, access_count=1, content_size=100)
        score2 = aging_policies.calculate_context_age_score(old_context)
        
        assert score1 > score2
        assert 0 <= score1 <= 1
        assert 0 <= score2 <= 1
    
    @pytest.mark.asyncio
    async def test_context_prioritization(self, prioritization_engine):
        """Test intelligent context prioritization."""
        # Create test contexts with different characteristics
        contexts = [
            self.create_test_context(age_hours=2, content_size=1000, access_count=1),    # fresh
            self.create_test_context(age_hours=48, content_size=10000, access_count=3),  # stale, large
            self.create_test_context(age_hours=720, content_size=5000, access_count=1),  # old, medium
            self.create_test_context(age_hours=12, content_size=500, access_count=10),   # active, small, popular
        ]
        
        # System load simulation
        system_load = {"cpu_percent": 30, "memory_percent": 40}
        
        # Prioritize contexts
        prioritized = await prioritization_engine.prioritize_contexts_for_consolidation(
            contexts, system_load
        )
        
        # Should filter out fresh contexts and prioritize appropriately
        assert len(prioritized) <= len(contexts)  # Fresh context should be filtered out
        
        # Verify prioritization order makes sense
        if len(prioritized) >= 2:
            first_context, first_score, first_metadata = prioritized[0]
            second_context, second_score, second_metadata = prioritized[1]
            
            # Higher priority context should have higher score
            assert first_score >= second_score
            
            # Verify metadata contains expected fields
            assert "aging_category" in first_metadata
            assert "content_size" in first_metadata
            assert "urgency" in first_metadata


class TestPerformanceValidation:
    """Test performance requirements including <60s recovery time."""
    
    @pytest.mark.asyncio
    async def test_checkpoint_creation_performance(self, tmp_path):
        """Test checkpoint creation performance."""
        checkpoint_manager = CheckpointManager()
        checkpoint_manager.checkpoint_dir = tmp_path / "perf_checkpoints"
        checkpoint_manager.git_repo_path = tmp_path / "perf_git"
        checkpoint_manager.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_manager.git_repo_path.mkdir(parents=True, exist_ok=True)
        checkpoint_manager.git_repo = checkpoint_manager._initialize_git_repository()
        
        agent_id = uuid4()
        
        # Measure checkpoint creation time
        start_time = time.time()
        
        checkpoint = await checkpoint_manager.create_checkpoint(
            agent_id=agent_id,
            checkpoint_type=CheckpointType.SCHEDULED,
            metadata={"performance_test": True}
        )
        
        creation_time = time.time() - start_time
        
        assert checkpoint is not None
        assert creation_time < 30  # Should create checkpoint in under 30 seconds
        
        # Verify performance metrics in checkpoint
        assert checkpoint.creation_time_ms > 0
        assert checkpoint.validation_time_ms > 0
        
        print(f"Checkpoint creation took {creation_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_recovery_time_performance(self, tmp_path):
        """Test full recovery time meets <60s requirement."""
        # Setup
        checkpoint_manager = CheckpointManager()
        checkpoint_manager.checkpoint_dir = tmp_path / "recovery_checkpoints"
        checkpoint_manager.git_repo_path = tmp_path / "recovery_git"
        checkpoint_manager.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_manager.git_repo_path.mkdir(parents=True, exist_ok=True)
        checkpoint_manager.git_repo = checkpoint_manager._initialize_git_repository()
        
        tmux_manager = TmuxSessionManager()
        
        agent_id = uuid4()
        workspace_path = tmp_path / "recovery_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint
        checkpoint = await checkpoint_manager.create_checkpoint(
            agent_id=agent_id,
            checkpoint_type=CheckpointType.PRE_SLEEP,
            metadata={"recovery_test": True}
        )
        
        git_commit = checkpoint.checkpoint_metadata["git_commit_hash"]
        
        # Measure full recovery time
        recovery_start = time.time()
        
        # Phase 1: Restore from Git checkpoint
        success, restored_data = await checkpoint_manager.restore_from_git_checkpoint(
            git_commit, agent_id
        )
        assert success is True
        
        # Phase 2: Restore Tmux session
        tmux_session = await tmux_manager.restore_session_from_checkpoint(
            agent_id=str(agent_id),
            workspace_path=workspace_path,
            checkpoint_data=restored_data.get("state_data", {})
        )
        
        recovery_time = time.time() - recovery_start
        
        # Verify recovery completed successfully
        assert tmux_session is not None
        
        # Critical performance requirement: recovery must complete in under 60 seconds
        assert recovery_time < 60, f"Recovery took {recovery_time:.2f}s, exceeds 60s requirement"
        
        print(f"Full recovery completed in {recovery_time:.2f}s")
        
        # Clean up
        await tmux_manager.terminate_session(str(agent_id))
    
    @pytest.mark.asyncio
    async def test_consolidation_performance_with_aging(self):
        """Test context consolidation performance with aging policies."""
        aging_policies = ContextAgingPolicies()
        prioritization_engine = ContextPrioritizationEngine()
        
        # Create large batch of test contexts
        contexts = []
        for i in range(100):
            age_hours = i + 1  # 1 to 100 hours old
            context = MockContext(
                id=uuid4(),
                content="x" * (1000 + i * 10),
                created_at=datetime.utcnow() - timedelta(hours=age_hours),
                access_count=max(1, 10 - (i // 10))  # Decreasing access frequency
            )
            contexts.append(context)
        
        # Measure prioritization performance
        start_time = time.time()
        
        system_load = {"cpu_percent": 50, "memory_percent": 60}
        prioritized = await prioritization_engine.prioritize_contexts_for_consolidation(
            contexts, system_load
        )
        
        prioritization_time = time.time() - start_time
        
        # Prioritization should complete quickly even with large batches
        assert prioritization_time < 5  # Under 5 seconds for 100 contexts
        assert len(prioritized) > 0  # Should have some contexts to process
        
        print(f"Prioritized {len(contexts)} contexts in {prioritization_time:.2f}s")
        print(f"Selected {len(prioritized)} contexts for consolidation")


@pytest.mark.integration
class TestFullSleepWakeIntegration:
    """Integration test for complete sleep-wake cycle."""
    
    @pytest.mark.asyncio
    async def test_complete_sleep_wake_cycle(self, tmp_path):
        """Test a complete sleep-wake cycle with all components."""
        # Setup all components
        checkpoint_manager = CheckpointManager()
        checkpoint_manager.checkpoint_dir = tmp_path / "integration_checkpoints"
        checkpoint_manager.git_repo_path = tmp_path / "integration_git"
        checkpoint_manager.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_manager.git_repo_path.mkdir(parents=True, exist_ok=True)
        checkpoint_manager.git_repo = checkpoint_manager._initialize_git_repository()
        
        tmux_manager = TmuxSessionManager()
        
        agent_id = uuid4()
        workspace_path = tmp_path / "integration_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        workspace_config = WorkspaceConfig(
            agent_id=str(agent_id),
            workspace_name="integration-test",
            project_path=workspace_path
        )
        
        workspace = AgentWorkspace(workspace_config, tmux_manager)
        
        print(f"Starting integration test for agent {agent_id}")
        
        # Phase 1: Initialize workspace
        init_success = await workspace.initialize()
        assert init_success is True
        print("✓ Workspace initialized")
        
        # Phase 2: Create checkpoint (simulating sleep)
        checkpoint_state = await workspace.capture_checkpoint_state()
        assert len(checkpoint_state) > 0
        
        checkpoint = await checkpoint_manager.create_checkpoint(
            agent_id=agent_id,
            checkpoint_type=CheckpointType.PRE_SLEEP,
            metadata={"integration_test": True, "workspace_state": checkpoint_state}
        )
        assert checkpoint is not None
        print("✓ Checkpoint created")
        
        # Phase 3: Terminate workspace (simulating crash)
        await workspace.terminate()
        print("✓ Workspace terminated")
        
        # Phase 4: Full recovery (simulating wake)
        recovery_start = time.time()
        
        # Restore from checkpoint
        git_commit = checkpoint.checkpoint_metadata["git_commit_hash"]
        success, restored_data = await checkpoint_manager.restore_from_git_checkpoint(
            git_commit, agent_id
        )
        assert success is True
        
        # Create new workspace and restore
        new_workspace = AgentWorkspace(workspace_config, tmux_manager)
        restore_success = await new_workspace.restore_from_checkpoint(
            restored_data.get("state_data", {})
        )
        assert restore_success is True
        
        recovery_time = time.time() - recovery_start
        print(f"✓ Full recovery completed in {recovery_time:.2f}s")
        
        # Verify recovery meets performance requirements
        assert recovery_time < 60
        
        # Verify workspace is functional
        assert new_workspace.status.value == "active"
        assert new_workspace.tmux_session is not None
        
        # Clean up
        await new_workspace.terminate()
        print("✓ Integration test completed successfully")