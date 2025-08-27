"""
Basic validation tests for Context Engine components.

Tests core functionality without complex async fixtures.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from app.core.context_engine_integration import (
    ContextEngineConfig,
    ContextEngineStatus,
    ContextEngineMetrics
)
from app.core.context_consolidation_triggers import (
    TriggerType,
    TriggerPriority,
    ConsolidationTrigger,
    AgentUsagePattern
)
from app.core.context_memory_manager import (
    CleanupPolicy,
    CleanupPolicyConfig,
    MemoryPressureLevel,
    MemoryUsageSnapshot
)
from app.core.context_cache_manager import (
    CacheLevel,
    CachePolicy,
    CacheEntry
)
from app.core.context_lifecycle_manager import (
    ContextLifecycleState,
    VersionAction,
    ContextVersion
)
from app.core.context_orchestrator_integration import (
    SleepPhase,
    OrchestratorEvent,
    SleepCycleContext,
    WakeContext
)


class TestContextEngineConfig:
    """Test context engine configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ContextEngineConfig()
        
        assert config.auto_consolidation_enabled is True
        assert config.consolidation_usage_threshold == 10
        assert config.memory_cleanup_enabled is True
        assert config.cache_enabled is True
        assert config.max_search_time_ms == 500.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ContextEngineConfig(
            consolidation_usage_threshold=20,
            memory_cleanup_interval_hours=12,
            cache_ttl_seconds=7200
        )
        
        assert config.consolidation_usage_threshold == 20
        assert config.memory_cleanup_interval_hours == 12
        assert config.cache_ttl_seconds == 7200


class TestConsolidationTrigger:
    """Test consolidation trigger data structures."""
    
    def test_trigger_creation(self):
        """Test creating consolidation trigger."""
        agent_id = uuid.uuid4()
        
        trigger = ConsolidationTrigger(
            trigger_id="test_trigger_123",
            trigger_type=TriggerType.USAGE_THRESHOLD,
            priority=TriggerPriority.MEDIUM,
            agent_id=agent_id,
            triggered_at=datetime.utcnow(),
            expected_processing_time_ms=1000.0,
            context_count_estimate=25,
            memory_pressure_mb=150.0,
            trigger_metadata={"test": True}
        )
        
        assert trigger.trigger_type == TriggerType.USAGE_THRESHOLD
        assert trigger.priority == TriggerPriority.MEDIUM
        assert trigger.agent_id == agent_id
        assert trigger.context_count_estimate == 25
        
        # Test serialization
        trigger_dict = trigger.to_dict()
        assert "trigger_id" in trigger_dict
        assert "agent_id" in trigger_dict
        assert trigger_dict["trigger_type"] == "usage_threshold"
    
    def test_agent_usage_pattern(self):
        """Test agent usage pattern creation."""
        agent_id = uuid.uuid4()
        
        pattern = AgentUsagePattern(
            agent_id=agent_id,
            contexts_created_per_hour=5.0,
            contexts_accessed_per_hour=10.0,
            avg_session_duration_minutes=45.0,
            peak_activity_hours=[9, 10, 11, 14, 15, 16],
            consolidation_frequency_hours=6.0,
            last_consolidation=datetime.utcnow() - timedelta(hours=8),
            current_unconsolidated_count=25,
            memory_usage_mb=150.0,
            is_active=True
        )
        
        assert pattern.agent_id == agent_id
        assert pattern.is_active is True
        assert len(pattern.peak_activity_hours) == 6
        assert pattern.current_unconsolidated_count == 25


class TestCleanupPolicies:
    """Test memory cleanup policies."""
    
    def test_policy_configurations(self):
        """Test different cleanup policy configurations."""
        conservative = CleanupPolicyConfig.conservative()
        balanced = CleanupPolicyConfig.balanced()
        aggressive = CleanupPolicyConfig.aggressive()
        emergency = CleanupPolicyConfig.emergency()
        
        # Test age thresholds (conservative should keep longer)
        assert conservative.max_age_days > balanced.max_age_days
        assert balanced.max_age_days > aggressive.max_age_days
        assert aggressive.max_age_days > emergency.max_age_days
        
        # Test importance thresholds (emergency should require higher importance)
        assert conservative.min_importance_threshold < emergency.min_importance_threshold
        
        # Test memory thresholds
        assert conservative.memory_threshold_mb > emergency.memory_threshold_mb
        
        # Test context limits
        assert conservative.max_contexts_per_agent > emergency.max_contexts_per_agent
    
    def test_memory_usage_snapshot(self):
        """Test memory usage snapshot creation."""
        snapshot = MemoryUsageSnapshot(
            timestamp=datetime.utcnow(),
            total_memory_mb=8192.0,
            available_memory_mb=2048.0,
            context_count=1000,
            consolidated_context_count=300,
            avg_context_size_kb=5.0,
            memory_pressure_level=MemoryPressureLevel.MEDIUM,
            gc_collections=150,
            cache_size_mb=256.0
        )
        
        assert snapshot.context_count == 1000
        assert snapshot.consolidated_context_count == 300
        assert snapshot.memory_pressure_level == MemoryPressureLevel.MEDIUM
        assert snapshot.total_memory_mb == 8192.0


class TestCacheManagement:
    """Test cache management components."""
    
    def test_cache_entry_creation(self):
        """Test cache entry creation and properties."""
        entry = CacheEntry(
            key="test_key",
            data={"test": "data"},
            created_at=datetime.utcnow(),
            accessed_at=datetime.utcnow(),
            access_count=1,
            size_bytes=1024,
            importance_score=0.7,
            cache_level=CacheLevel.L1_MEMORY,
            ttl_seconds=3600,
            tags={"test", "cache"}
        )
        
        assert entry.key == "test_key"
        assert entry.cache_level == CacheLevel.L1_MEMORY
        assert entry.ttl_seconds == 3600
        assert "test" in entry.tags
        assert entry.is_expired is False  # Should not be expired yet
    
    def test_cache_entry_expiration(self):
        """Test cache entry expiration logic."""
        # Create expired entry
        old_time = datetime.utcnow() - timedelta(hours=2)
        entry = CacheEntry(
            key="expired_key",
            data={"test": "data"},
            created_at=old_time,
            accessed_at=old_time,
            access_count=1,
            size_bytes=1024,
            importance_score=0.5,
            cache_level=CacheLevel.L1_MEMORY,
            ttl_seconds=3600  # 1 hour TTL, but created 2 hours ago
        )
        
        assert entry.is_expired is True
        assert entry.age_seconds > 3600
    
    def test_cache_scoring_policies(self):
        """Test cache scoring for different policies."""
        entry = CacheEntry(
            key="test_key",
            data={"test": "data"},
            created_at=datetime.utcnow() - timedelta(minutes=30),
            accessed_at=datetime.utcnow() - timedelta(minutes=5),
            access_count=10,
            size_bytes=1024,
            importance_score=0.8,
            cache_level=CacheLevel.L1_MEMORY
        )
        
        lru_score = entry.calculate_score(CachePolicy.LRU)
        lfu_score = entry.calculate_score(CachePolicy.LFU)
        importance_score = entry.calculate_score(CachePolicy.IMPORTANCE_BASED)
        
        assert lru_score > 0  # Should be positive (time since last access)
        assert lfu_score < 0  # Should be negative (negative access count)
        assert importance_score < 0  # Should be negative (negative importance score)


class TestLifecycleManagement:
    """Test context lifecycle management."""
    
    def test_context_version_creation(self):
        """Test context version creation."""
        context_id = uuid.uuid4()
        
        version = ContextVersion(
            version_id="version_123",
            context_id=context_id,
            version_number=1,
            action=VersionAction.CREATE,
            content_hash="abc123def456",
            content_snapshot={"title": "Test", "content": "Test content"},
            metadata_snapshot={"created_at": datetime.utcnow().isoformat()},
            created_at=datetime.utcnow(),
            created_by="test_user",
            parent_version_id=None,
            changes_summary="Initial version",
            size_bytes=1024
        )
        
        assert version.context_id == context_id
        assert version.version_number == 1
        assert version.action == VersionAction.CREATE
        assert version.created_by == "test_user"
        assert version.size_bytes == 1024
        
        # Test serialization
        version_dict = version.to_dict()
        assert "version_id" in version_dict
        assert "context_id" in version_dict
        assert version_dict["action"] == "create"
    
    def test_lifecycle_states(self):
        """Test context lifecycle states."""
        states = [
            ContextLifecycleState.DRAFT,
            ContextLifecycleState.ACTIVE,
            ContextLifecycleState.CONSOLIDATED,
            ContextLifecycleState.ARCHIVED,
            ContextLifecycleState.DELETED,
            ContextLifecycleState.RECOVERED
        ]
        
        for state in states:
            assert isinstance(state.value, str)
            assert len(state.value) > 0


class TestOrchestratorIntegration:
    """Test orchestrator integration components."""
    
    def test_sleep_cycle_context(self):
        """Test sleep cycle context creation."""
        agent_id = uuid.uuid4()
        session_id = uuid.uuid4()
        
        sleep_context = SleepCycleContext(
            agent_id=agent_id,
            session_id=session_id,
            sleep_initiated_at=datetime.utcnow(),
            expected_wake_time=datetime.utcnow() + timedelta(hours=2),
            sleep_phase=SleepPhase.DEEP_SLEEP,
            contexts_count=50,
            unconsolidated_count=25,
            memory_usage_mb=200.0,
            last_activity_at=datetime.utcnow() - timedelta(minutes=30),
            consolidation_priority=TriggerPriority.HIGH
        )
        
        assert sleep_context.agent_id == agent_id
        assert sleep_context.session_id == session_id
        assert sleep_context.sleep_phase == SleepPhase.DEEP_SLEEP
        assert sleep_context.contexts_count == 50
        assert sleep_context.consolidation_priority == TriggerPriority.HIGH
        
        # Test serialization
        context_dict = sleep_context.to_dict()
        assert "agent_id" in context_dict
        assert "sleep_phase" in context_dict
        assert context_dict["sleep_phase"] == "deep_sleep"
    
    def test_wake_context(self):
        """Test wake context creation."""
        agent_id = uuid.uuid4()
        session_id = uuid.uuid4()
        
        wake_context = WakeContext(
            agent_id=agent_id,
            session_id=session_id,
            wake_initiated_at=datetime.utcnow(),
            sleep_duration_minutes=120.0,
            contexts_processed_during_sleep=15,
            memory_freed_mb=75.0,
            consolidation_ratio=0.3,
            cache_warmup_required=True,
            priority_contexts=["context_1", "context_2"]
        )
        
        assert wake_context.agent_id == agent_id
        assert wake_context.sleep_duration_minutes == 120.0
        assert wake_context.cache_warmup_required is True
        assert len(wake_context.priority_contexts) == 2
        
        # Test serialization
        context_dict = wake_context.to_dict()
        assert "agent_id" in context_dict
        assert "sleep_duration_minutes" in context_dict
        assert context_dict["cache_warmup_required"] is True
    
    def test_orchestrator_events(self):
        """Test orchestrator event enumeration."""
        events = [
            OrchestratorEvent.AGENT_SLEEP_INITIATED,
            OrchestratorEvent.AGENT_SLEEP_COMPLETED,
            OrchestratorEvent.AGENT_WAKE_INITIATED,
            OrchestratorEvent.AGENT_WAKE_COMPLETED,
            OrchestratorEvent.SESSION_STARTED,
            OrchestratorEvent.SESSION_ENDED
        ]
        
        for event in events:
            assert isinstance(event.value, str)
            assert len(event.value) > 0


class TestContextEngineMetrics:
    """Test context engine metrics."""
    
    def test_metrics_creation(self):
        """Test creating context engine metrics."""
        metrics = ContextEngineMetrics(
            total_contexts=1000,
            consolidated_contexts=300,
            cached_contexts=150,
            active_searches=5,
            avg_search_time_ms=45.0,
            avg_consolidation_time_ms=1500.0,
            cache_hit_rate=0.75,
            memory_usage_mb=512.0,
            error_rate=0.02,
            uptime_hours=24.5,
            searches_per_hour=120.0,
            consolidations_per_hour=8.0,
            contexts_created_per_hour=25.0
        )
        
        assert metrics.total_contexts == 1000
        assert metrics.consolidated_contexts == 300
        assert metrics.cache_hit_rate == 0.75
        assert metrics.error_rate == 0.02
        assert metrics.uptime_hours == 24.5


def test_integration_data_flow():
    """Test data flow between components (mock-based)."""
    # This test validates that the data structures work together
    agent_id = uuid.uuid4()
    
    # Create usage pattern
    pattern = AgentUsagePattern(
        agent_id=agent_id,
        contexts_created_per_hour=10.0,
        contexts_accessed_per_hour=20.0,
        avg_session_duration_minutes=60.0,
        peak_activity_hours=[9, 10, 11, 14, 15, 16],
        consolidation_frequency_hours=6.0,
        last_consolidation=None,
        current_unconsolidated_count=30,
        memory_usage_mb=200.0,
        is_active=True
    )
    
    # Create trigger based on pattern
    trigger = ConsolidationTrigger(
        trigger_id=f"usage_{agent_id}_{int(datetime.utcnow().timestamp())}",
        trigger_type=TriggerType.USAGE_THRESHOLD,
        priority=TriggerPriority.MEDIUM,
        agent_id=agent_id,
        triggered_at=datetime.utcnow(),
        expected_processing_time_ms=pattern.current_unconsolidated_count * 150,
        context_count_estimate=pattern.current_unconsolidated_count,
        memory_pressure_mb=pattern.memory_usage_mb,
        trigger_metadata={
            "threshold_contexts": 15,
            "actual_contexts": pattern.current_unconsolidated_count,
            "contexts_created_per_hour": pattern.contexts_created_per_hour
        }
    )
    
    # Create sleep context
    sleep_context = SleepCycleContext(
        agent_id=agent_id,
        session_id=uuid.uuid4(),
        sleep_initiated_at=datetime.utcnow(),
        expected_wake_time=datetime.utcnow() + timedelta(hours=2),
        sleep_phase=SleepPhase.LIGHT_SLEEP,
        contexts_count=pattern.current_unconsolidated_count,
        unconsolidated_count=pattern.current_unconsolidated_count,
        memory_usage_mb=pattern.memory_usage_mb,
        last_activity_at=datetime.utcnow() - timedelta(minutes=15),
        consolidation_priority=trigger.priority
    )
    
    # Validate data consistency
    assert trigger.agent_id == pattern.agent_id == sleep_context.agent_id
    assert trigger.context_count_estimate == pattern.current_unconsolidated_count
    assert sleep_context.consolidation_priority == trigger.priority
    assert sleep_context.memory_usage_mb == pattern.memory_usage_mb


if __name__ == "__main__":
    pytest.main([__file__, "-v"])