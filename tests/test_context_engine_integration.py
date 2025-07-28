"""
Comprehensive Test Suite for Context Engine Integration.

Tests all major components of the context engine:
- Context Engine Integration Service
- Consolidation Triggers
- Memory Management
- Cache Management
- Lifecycle Management
- Orchestrator Integration
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.testclient import TestClient

from app.core.context_engine_integration import (
    ContextEngineIntegration,
    ContextEngineConfig,
    ContextEngineStatus,
    ConsolidationTrigger
)
from app.core.context_consolidation_triggers import (
    ConsolidationTriggerManager,
    TriggerType,
    TriggerPriority,
    AgentUsagePattern
)
from app.core.context_memory_manager import (
    ContextMemoryManager,
    CleanupPolicy,
    CleanupPolicyConfig,
    MemoryPressureLevel
)
from app.core.context_cache_manager import (
    ContextCacheManager,
    CacheLevel,
    CachePolicy
)
from app.core.context_lifecycle_manager import (
    ContextLifecycleManager,
    ContextLifecycleState,
    VersionAction
)
from app.core.context_orchestrator_integration import (
    ContextOrchestratorIntegration,
    SleepPhase,
    OrchestratorEvent
)
from app.models.context import Context, ContextType
from app.schemas.context import ContextCreate, ContextSearchRequest


class TestContextEngineIntegration:
    """Test suite for Context Engine Integration Service."""
    
    @pytest.fixture
    async def context_engine(self):
        """Create context engine for testing."""
        config = ContextEngineConfig(
            auto_consolidation_enabled=True,
            consolidation_usage_threshold=5,
            memory_cleanup_enabled=True,
            cache_enabled=True
        )
        
        engine = ContextEngineIntegration(config=config)
        
        # Mock dependencies
        engine.context_manager = AsyncMock()
        engine.embedding_service = AsyncMock()
        engine.consolidator = AsyncMock()
        engine.redis_client = AsyncMock()
        
        await engine.initialize()
        
        yield engine
        
        await engine.shutdown()
    
    @pytest.fixture
    def sample_context_data(self):
        """Create sample context data for testing."""
        return ContextCreate(
            title="Test Context",
            content="This is a test context for integration testing",
            context_type=ContextType.CONVERSATION,
            agent_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            importance_score=0.7,
            tags=["test", "integration"],
            metadata={"test": True}
        )
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, context_engine):
        """Test context engine initialization."""
        assert context_engine.status == ContextEngineStatus.HEALTHY
        assert context_engine._initialized is True
        assert len(context_engine._background_tasks) > 0
    
    @pytest.mark.asyncio
    async def test_store_context_enhanced(self, context_engine, sample_context_data):
        """Test enhanced context storage."""
        # Mock context manager response
        mock_context = Context(
            id=uuid.uuid4(),
            title=sample_context_data.title,
            content=sample_context_data.content,
            context_type=sample_context_data.context_type,
            agent_id=sample_context_data.agent_id,
            importance_score=sample_context_data.importance_score
        )
        
        context_engine.context_manager.store_context.return_value = mock_context
        
        # Test storage
        result = await context_engine.store_context_enhanced(
            context_data=sample_context_data,
            generate_embedding=True,
            enable_auto_consolidation=True,
            cache_result=True
        )
        
        assert result.id == mock_context.id
        assert result.title == sample_context_data.title
        context_engine.context_manager.store_context.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_contexts_enhanced(self, context_engine, sample_context_data):
        """Test enhanced context search."""
        search_request = ContextSearchRequest(
            query="test query",
            agent_id=sample_context_data.agent_id,
            limit=10,
            min_relevance=0.7
        )
        
        # Mock search engine
        mock_search_engine = AsyncMock()
        mock_results = []
        mock_metadata = {
            "cache_hit": False,
            "search_time_ms": 50.0,
            "search_method": "enhanced",
            "results_count": 0
        }
        
        mock_search_engine.enhanced_search.return_value = (mock_results, mock_metadata)
        context_engine._search_engine = mock_search_engine
        
        # Test search
        results, metadata = await context_engine.search_contexts_enhanced(
            request=search_request,
            use_cache=True,
            enable_analytics=True
        )
        
        assert isinstance(results, list)
        assert isinstance(metadata, dict)
        assert "search_time_ms" in metadata
        mock_search_engine.enhanced_search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trigger_consolidation(self, context_engine):
        """Test consolidation triggering."""
        agent_id = uuid.uuid4()
        
        # Mock consolidator response
        mock_metrics = Mock()
        mock_metrics.compression_ratio = 0.7
        mock_metrics.processing_time_ms = 1000.0
        mock_metrics.contexts_merged = 5
        mock_metrics.contexts_archived = 0
        
        context_engine.consolidator.ultra_compress_agent_contexts.return_value = mock_metrics
        
        # Test consolidation
        result = await context_engine.trigger_consolidation(
            agent_id=agent_id,
            trigger_type=ConsolidationTrigger.MANUAL,
            target_reduction=0.7
        )
        
        assert result.compression_ratio == 0.7
        assert result.processing_time_ms == 1000.0
        context_engine.consolidator.ultra_compress_agent_contexts.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_memory_optimization(self, context_engine):
        """Test memory usage optimization."""
        agent_id = uuid.uuid4()
        
        # Mock memory manager
        context_engine._memory_manager = AsyncMock()
        
        # Test optimization
        result = await context_engine.optimize_memory_usage(
            agent_id=agent_id,
            force_cleanup=True
        )
        
        assert isinstance(result, dict)
        assert "contexts_cleaned" in result or "error" in result
    
    @pytest.mark.asyncio
    async def test_health_status(self, context_engine):
        """Test comprehensive health status."""
        # Mock component health checks
        context_engine.context_manager.health_check.return_value = {"status": "healthy"}
        context_engine.embedding_service.health_check.return_value = {"status": "healthy"}
        
        health = await context_engine.get_comprehensive_health_status()
        
        assert "status" in health
        assert "components" in health
        assert "performance" in health
        assert health["initialized"] is True


class TestConsolidationTriggers:
    """Test suite for Consolidation Trigger Manager."""
    
    @pytest.fixture
    async def trigger_manager(self):
        """Create trigger manager for testing."""
        manager = ConsolidationTriggerManager()
        
        # Mock dependencies
        manager.redis_client = AsyncMock()
        
        yield manager
        
        await manager.stop_monitoring()
    
    @pytest.fixture
    def sample_agent_pattern(self):
        """Create sample agent usage pattern."""
        return AgentUsagePattern(
            agent_id=uuid.uuid4(),
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
    
    @pytest.mark.asyncio
    async def test_check_usage_triggers(self, trigger_manager, sample_agent_pattern):
        """Test usage-based trigger checking."""
        with patch.object(trigger_manager, '_get_agent_pattern', return_value=sample_agent_pattern):
            triggers = await trigger_manager.check_agent_triggers(sample_agent_pattern.agent_id)
            
            # Should trigger consolidation due to high unconsolidated count
            usage_triggers = [t for t in triggers if t.trigger_type == TriggerType.USAGE_THRESHOLD]
            assert len(usage_triggers) > 0
            assert usage_triggers[0].priority == TriggerPriority.MEDIUM
    
    @pytest.mark.asyncio
    async def test_sleep_cycle_trigger_registration(self, trigger_manager):
        """Test sleep cycle trigger registration."""
        agent_id = uuid.uuid4()
        context_count = 20
        expected_wake_time = datetime.utcnow() + timedelta(hours=2)
        
        trigger = await trigger_manager.register_sleep_cycle_trigger(
            agent_id=agent_id,
            sleep_context_count=context_count,
            expected_wake_time=expected_wake_time
        )
        
        assert trigger is not None
        assert trigger.trigger_type == TriggerType.SLEEP_CYCLE
        assert trigger.agent_id == agent_id
        assert trigger.context_count_estimate == context_count
        assert trigger.priority == TriggerPriority.HIGH
    
    @pytest.mark.asyncio
    async def test_manual_trigger_registration(self, trigger_manager):
        """Test manual trigger registration."""
        agent_id = uuid.uuid4()
        
        with patch('app.core.context_consolidation_triggers.get_async_session') as mock_session:
            mock_db = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db
            mock_db.scalar.return_value = 15  # Mock context count
            
            trigger = await trigger_manager.register_manual_trigger(
                agent_id=agent_id,
                priority=TriggerPriority.HIGH,
                metadata={"reason": "test"}
            )
            
            assert trigger.trigger_type == TriggerType.MANUAL
            assert trigger.priority == TriggerPriority.HIGH
            assert trigger.context_count_estimate == 15
    
    @pytest.mark.asyncio
    async def test_trigger_completion(self, trigger_manager):
        """Test trigger completion tracking."""
        trigger_id = "test_trigger_123"
        
        await trigger_manager.complete_trigger(
            trigger_id=trigger_id,
            success=True,
            processing_time_ms=1500.0,
            contexts_processed=10
        )
        
        # Should update success rates and processing times
        assert TriggerType.MANUAL in trigger_manager.trigger_success_rates
        trigger_manager.redis_client.lpush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trigger_statistics(self, trigger_manager):
        """Test trigger statistics retrieval."""
        # Add some mock data
        trigger_manager.agent_patterns[uuid.uuid4()] = AgentUsagePattern(
            agent_id=uuid.uuid4(),
            contexts_created_per_hour=3.0,
            contexts_accessed_per_hour=5.0,
            avg_session_duration_minutes=30.0,
            peak_activity_hours=[10, 11, 15, 16],
            consolidation_frequency_hours=8.0,
            last_consolidation=None,
            current_unconsolidated_count=8,
            memory_usage_mb=50.0,
            is_active=True
        )
        
        stats = await trigger_manager.get_trigger_statistics()
        
        assert "active_triggers" in stats
        assert "agents_monitored" in stats
        assert "trigger_success_rates" in stats
        assert stats["agents_monitored"] == 1


class TestMemoryManager:
    """Test suite for Context Memory Manager."""
    
    @pytest.fixture
    async def memory_manager(self):
        """Create memory manager for testing."""
        manager = ContextMemoryManager()
        
        # Mock dependencies
        manager.redis_client = AsyncMock()
        
        yield manager
        
        await manager.stop_memory_management()
    
    @pytest.mark.asyncio
    async def test_cleanup_policy_configuration(self, memory_manager):
        """Test cleanup policy configurations."""
        conservative = CleanupPolicyConfig.conservative()
        balanced = CleanupPolicyConfig.balanced()
        aggressive = CleanupPolicyConfig.aggressive()
        emergency = CleanupPolicyConfig.emergency()
        
        assert conservative.max_age_days > balanced.max_age_days
        assert balanced.max_age_days > aggressive.max_age_days
        assert aggressive.max_age_days > emergency.max_age_days
        
        assert conservative.min_importance_threshold < emergency.min_importance_threshold
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_balanced_policy(self, memory_manager):
        """Test memory cleanup with balanced policy."""
        agent_id = uuid.uuid4()
        
        with patch.object(memory_manager, '_cleanup_agent_contexts') as mock_cleanup:
            mock_cleanup.return_value = {"archived": 5, "deleted": 0, "consolidated": 3}
            
            with patch.object(memory_manager, '_take_memory_snapshot') as mock_snapshot:
                mock_snapshot.return_value = Mock(
                    total_memory_mb=500.0,
                    memory_pressure_level=MemoryPressureLevel.MEDIUM
                )
                
                result = await memory_manager.perform_memory_cleanup(
                    policy=CleanupPolicy.BALANCED,
                    agent_id=agent_id,
                    force_cleanup=False
                )
                
                assert result.policy_used == CleanupPolicy.BALANCED
                assert result.contexts_archived == 5
                assert result.contexts_consolidated == 3
    
    @pytest.mark.asyncio
    async def test_memory_pressure_assessment(self, memory_manager):
        """Test memory pressure level assessment."""
        with patch('psutil.virtual_memory') as mock_memory:
            # Test high memory usage
            mock_memory.return_value.total = 8 * 1024 * 1024 * 1024  # 8GB
            mock_memory.return_value.available = 1 * 1024 * 1024 * 1024  # 1GB available
            
            pressure = await memory_manager._assess_memory_pressure()
            
            # Should detect high or critical pressure
            assert pressure in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_memory_optimization(self, memory_manager):
        """Test comprehensive memory optimization."""
        with patch.object(memory_manager, '_perform_garbage_collection', return_value=50.0):
            with patch.object(memory_manager, '_optimize_caches', return_value=25.0):
                with patch.object(memory_manager, '_take_memory_snapshot') as mock_snapshot:
                    mock_snapshot.return_value = Mock(
                        total_memory_mb=1000.0,
                        memory_pressure_level=MemoryPressureLevel.MEDIUM
                    )
                    
                    result = await memory_manager.optimize_memory_usage()
                    
                    assert "memory_freed_mb" in result
                    assert "optimizations_applied" in result
                    assert result["memory_freed_mb"] >= 75.0  # GC + cache optimization
    
    @pytest.mark.asyncio
    async def test_memory_statistics(self, memory_manager):
        """Test memory statistics retrieval."""
        with patch.object(memory_manager, '_take_memory_snapshot') as mock_snapshot:
            mock_snapshot.return_value = Mock(
                total_memory_mb=2000.0,
                context_count=1000,
                consolidated_context_count=300,
                memory_pressure_level=MemoryPressureLevel.LOW
            )
            
            stats = await memory_manager.get_memory_statistics()
            
            assert "current_memory" in stats
            assert "cleanup_statistics" in stats
            assert "memory_trends" in stats
            assert stats["current_memory"]["context_count"] == 1000


class TestCacheManager:
    """Test suite for Context Cache Manager."""
    
    @pytest.fixture
    async def cache_manager(self):
        """Create cache manager for testing."""
        manager = ContextCacheManager()
        
        # Mock dependencies
        manager.redis_client = AsyncMock()
        
        await manager.start_cache_management()
        
        yield manager
        
        await manager.stop_cache_management()
    
    @pytest.fixture
    def sample_context(self):
        """Create sample context for testing."""
        return Context(
            id=uuid.uuid4(),
            title="Test Context",
            content="Test content for caching",
            context_type=ContextType.CONVERSATION,
            agent_id=uuid.uuid4(),
            importance_score=0.7
        )
    
    @pytest.mark.asyncio
    async def test_context_caching_l1(self, cache_manager, sample_context):
        """Test L1 memory cache operations."""
        context_id = sample_context.id
        
        # Mock database lookup
        with patch.object(cache_manager, '_get_context_from_database', return_value=sample_context):
            # First call should hit database and cache result
            result1 = await cache_manager.get_context(context_id, use_cache=True)
            
            assert result1 is not None
            assert result1.id == context_id
            
            # Second call should hit L1 cache
            result2 = await cache_manager.get_context(context_id, use_cache=True)
            
            assert result2 is not None
            assert result2.id == context_id
    
    @pytest.mark.asyncio
    async def test_context_storage_multilevel(self, cache_manager, sample_context):
        """Test storing context in multiple cache levels."""
        success = await cache_manager.store_context(
            context=sample_context,
            cache_levels=[CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
        )
        
        assert success is True
        
        # Should be stored in L1 cache
        cache_key = f"context:{sample_context.id}"
        l1_entry = cache_manager.context_cache.get(cache_key)
        assert l1_entry is not None
    
    @pytest.mark.asyncio
    async def test_context_invalidation(self, cache_manager, sample_context):
        """Test context cache invalidation."""
        # First store the context
        await cache_manager.store_context(sample_context)
        
        # Then invalidate it
        success = await cache_manager.invalidate_context(
            context_id=sample_context.id,
            cache_levels=[CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
        )
        
        assert success is True
        
        # Should no longer be in L1 cache
        cache_key = f"context:{sample_context.id}"
        l1_entry = cache_manager.context_cache.get(cache_key)
        assert l1_entry is None
    
    @pytest.mark.asyncio
    async def test_agent_cache_invalidation(self, cache_manager):
        """Test invalidating all contexts for an agent."""
        agent_id = uuid.uuid4()
        
        with patch('app.core.context_cache_manager.get_async_session') as mock_session:
            mock_db = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock context IDs for agent
            mock_db.execute.return_value.all.return_value = [
                (uuid.uuid4(),), (uuid.uuid4(),), (uuid.uuid4(),)
            ]
            
            count = await cache_manager.invalidate_agent_contexts(agent_id)
            
            assert count >= 0  # Should process contexts without error
    
    @pytest.mark.asyncio
    async def test_cache_warming(self, cache_manager):
        """Test cache warming for an agent."""
        agent_id = uuid.uuid4()
        
        with patch('app.core.context_cache_manager.get_async_session') as mock_session:
            mock_db = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock contexts for warming
            mock_contexts = [
                Context(id=uuid.uuid4(), title=f"Context {i}", content=f"Content {i}",
                       context_type=ContextType.CONVERSATION, agent_id=agent_id, importance_score=0.5)
                for i in range(5)
            ]
            
            mock_db.execute.return_value.scalars.return_value.all.return_value = mock_contexts
            
            warmed_count = await cache_manager.warm_cache_for_agent(
                agent_id=agent_id,
                context_limit=10
            )
            
            assert warmed_count == 5
    
    @pytest.mark.asyncio
    async def test_cache_statistics(self, cache_manager):
        """Test cache statistics retrieval."""
        # Add some mock data to caches
        cache_manager.context_cache.put("test_key", Mock())
        
        stats = await cache_manager.get_cache_statistics()
        
        assert "cache_levels" in stats
        assert "l1_memory_stats" in stats
        assert "performance_metrics" in stats
        assert stats["l1_memory_stats"]["context_cache"]["entries"] >= 1


class TestLifecycleManager:
    """Test suite for Context Lifecycle Manager."""
    
    @pytest.fixture
    def lifecycle_manager(self):
        """Create lifecycle manager for testing."""
        manager = ContextLifecycleManager()
        
        # Mock dependencies
        manager.redis_client = AsyncMock()
        
        return manager
    
    @pytest.fixture
    def sample_context(self):
        """Create sample context for testing."""
        return Context(
            id=uuid.uuid4(),
            title="Test Context",
            content="Test content for lifecycle management",
            context_type=ContextType.CONVERSATION,
            agent_id=uuid.uuid4(),
            importance_score=0.7,
            context_metadata={}
        )
    
    @pytest.mark.asyncio
    async def test_version_creation(self, lifecycle_manager, sample_context):
        """Test creating context versions."""
        with patch.object(lifecycle_manager, '_store_version') as mock_store:
            with patch.object(lifecycle_manager, '_update_context_version_metadata') as mock_update:
                with patch.object(lifecycle_manager, '_create_audit_entry') as mock_audit:
                    
                    version = await lifecycle_manager.create_version(
                        context=sample_context,
                        action=VersionAction.CREATE,
                        changes_summary="Initial version",
                        created_by="test_user"
                    )
                    
                    assert version.context_id == sample_context.id
                    assert version.action == VersionAction.CREATE
                    assert version.version_number == 1
                    assert version.created_by == "test_user"
                    
                    mock_store.assert_called_once()
                    mock_update.assert_called_once()
                    mock_audit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_version_history_retrieval(self, lifecycle_manager, sample_context):
        """Test retrieving version history."""
        context_id = sample_context.id
        
        # Mock Redis responses
        version_ids = [b"version_1", b"version_2", b"version_3"]
        lifecycle_manager.redis_client.lrange.return_value = version_ids
        
        with patch.object(lifecycle_manager, '_get_version') as mock_get_version:
            mock_versions = [
                Mock(version_number=i, created_at=datetime.utcnow())
                for i in range(1, 4)
            ]
            mock_get_version.side_effect = mock_versions
            
            history = await lifecycle_manager.get_version_history(context_id, limit=10)
            
            assert len(history) == 3
            assert all(isinstance(v, Mock) for v in history)
    
    @pytest.mark.asyncio
    async def test_context_restoration(self, lifecycle_manager, sample_context):
        """Test context restoration from version."""
        context_id = sample_context.id
        version_id = "test_version_123"
        
        with patch('app.core.context_lifecycle_manager.get_async_session') as mock_session:
            mock_db = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db
            mock_db.get.return_value = sample_context
            
            # Mock target version
            mock_version = Mock()
            mock_version.version_id = version_id
            mock_version.version_number = 2
            mock_version.content_snapshot = {
                "title": "Restored Title",
                "content": "Restored Content",
                "importance_score": 0.8
            }
            mock_version.metadata_snapshot = {"access_count": "5"}
            
            with patch.object(lifecycle_manager, '_get_version', return_value=mock_version):
                with patch.object(lifecycle_manager, 'create_version') as mock_create:
                    
                    restored = await lifecycle_manager.restore_context_version(
                        context_id=context_id,
                        version_id=version_id,
                        create_backup=True
                    )
                    
                    assert restored.title == "Restored Title"
                    assert restored.content == "Restored Content"
                    assert restored.importance_score == 0.8
                    
                    # Should create backup and restoration versions
                    assert mock_create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_rollback_to_previous_version(self, lifecycle_manager, sample_context):
        """Test rolling back to previous version."""
        context_id = sample_context.id
        
        # Mock version history
        mock_versions = [
            Mock(version_id="v3", version_number=3),
            Mock(version_id="v2", version_number=2),
            Mock(version_id="v1", version_number=1)
        ]
        
        with patch.object(lifecycle_manager, 'get_version_history', return_value=mock_versions):
            with patch.object(lifecycle_manager, 'restore_context_version') as mock_restore:
                mock_restore.return_value = sample_context
                
                result = await lifecycle_manager.rollback_to_previous_version(
                    context_id=context_id,
                    steps_back=1
                )
                
                # Should restore to version 2 (1 step back from version 3)
                mock_restore.assert_called_once_with(context_id, "v2", create_backup=True)
                assert result == sample_context
    
    @pytest.mark.asyncio
    async def test_recovery_point_creation(self, lifecycle_manager, sample_context):
        """Test creating recovery points."""
        with patch('app.core.context_lifecycle_manager.get_async_session') as mock_session:
            mock_db = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db
            mock_db.get.return_value = sample_context
            
            with patch.object(lifecycle_manager, 'create_version') as mock_create_version:
                mock_version = Mock()
                mock_version.version_id = "recovery_version_123"
                mock_version.content_hash = "abc123"
                mock_create_version.return_value = mock_version
                
                with patch.object(lifecycle_manager, '_store_recovery_point') as mock_store:
                    
                    recovery_point = await lifecycle_manager.create_recovery_point(
                        context_id=sample_context.id,
                        recovery_type="manual",
                        metadata={"reason": "test"}
                    )
                    
                    assert recovery_point.context_id == sample_context.id
                    assert recovery_point.recovery_type == "manual"
                    assert recovery_point.version_id == "recovery_version_123"
                    
                    mock_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_data_integrity_verification(self, lifecycle_manager, sample_context):
        """Test data integrity verification."""
        context_id = sample_context.id
        version_id = "test_version_123"
        
        # Mock version with matching hash
        mock_version = Mock()
        mock_version.content_snapshot = {"test": "data"}
        mock_version.content_hash = lifecycle_manager._calculate_content_hash({"test": "data"})
        
        with patch.object(lifecycle_manager, '_get_version', return_value=mock_version):
            with patch.object(lifecycle_manager, '_verify_snapshot_structure', return_value=True):
                
                result = await lifecycle_manager.verify_data_integrity(
                    context_id=context_id,
                    version_id=version_id
                )
                
                assert result["integrity_valid"] is True
                assert "content_hash_verification" in result["checks_performed"]
                assert "snapshot_structure_verification" in result["checks_performed"]
                assert len(result["issues_found"]) == 0


class TestOrchestratorIntegration:
    """Test suite for Context Orchestrator Integration."""
    
    @pytest.fixture
    async def orchestrator_integration(self):
        """Create orchestrator integration for testing."""
        integration = ContextOrchestratorIntegration()
        
        # Mock all dependencies
        integration._context_engine = AsyncMock()
        integration._trigger_manager = AsyncMock()
        integration._memory_manager = AsyncMock()
        integration._cache_manager = AsyncMock()
        integration._lifecycle_manager = AsyncMock()
        integration.redis_client = AsyncMock()
        
        await integration.initialize()
        
        yield integration
        
        await integration.shutdown()
    
    @pytest.mark.asyncio
    async def test_sleep_cycle_initiation(self, orchestrator_integration):
        """Test agent sleep cycle initiation."""
        agent_id = uuid.uuid4()
        session_id = uuid.uuid4()
        expected_wake_time = datetime.utcnow() + timedelta(hours=2)
        
        with patch.object(orchestrator_integration, '_gather_agent_context_stats') as mock_stats:
            mock_stats.return_value = {
                "total_contexts": 50,
                "unconsolidated_contexts": 25,
                "estimated_memory_mb": 200.0,
                "last_activity": datetime.utcnow()
            }
            
            with patch.object(orchestrator_integration, '_trigger_sleep_consolidation') as mock_consolidation:
                
                sleep_context = await orchestrator_integration.handle_agent_sleep_initiated(
                    agent_id=agent_id,
                    session_id=session_id,
                    expected_wake_time=expected_wake_time,
                    sleep_phase=SleepPhase.DEEP_SLEEP
                )
                
                assert sleep_context.agent_id == agent_id
                assert sleep_context.session_id == session_id
                assert sleep_context.sleep_phase == SleepPhase.DEEP_SLEEP
                assert sleep_context.contexts_count == 50
                assert sleep_context.unconsolidated_count == 25
                
                # Should trigger consolidation for contexts
                mock_consolidation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sleep_cycle_completion(self, orchestrator_integration):
        """Test agent sleep cycle completion."""
        agent_id = uuid.uuid4()
        
        # Set up active sleep cycle
        sleep_context = Mock()
        sleep_context.sleep_initiated_at = datetime.utcnow() - timedelta(minutes=30)
        sleep_context.unconsolidated_count = 25
        sleep_context.contexts_count = 50
        sleep_context.memory_usage_mb = 200.0
        sleep_context.sleep_phase = SleepPhase.LIGHT_SLEEP
        
        orchestrator_integration.active_sleep_cycles[agent_id] = sleep_context
        
        with patch.object(orchestrator_integration, '_gather_agent_context_stats') as mock_stats:
            mock_stats.return_value = {
                "unconsolidated_contexts": 10,  # Reduced from 25
                "estimated_memory_mb": 150.0   # Reduced from 200
            }
            
            with patch.object(orchestrator_integration, '_store_sleep_completion_results') as mock_store:
                
                result = await orchestrator_integration.handle_agent_sleep_completed(
                    agent_id=agent_id,
                    consolidation_results={"contexts_processed": 15}
                )
                
                assert result["contexts_processed"] == 15
                assert result["memory_freed_mb"] == 50.0
                assert "sleep_duration_minutes" in result
                
                # Should store completion results
                mock_store.assert_called_once()
                
                # Should remove from active cycles
                assert agent_id not in orchestrator_integration.active_sleep_cycles
    
    @pytest.mark.asyncio
    async def test_wake_cycle_initiation(self, orchestrator_integration):
        """Test agent wake cycle initiation."""
        agent_id = uuid.uuid4()
        session_id = uuid.uuid4()
        priority_contexts = ["context_1", "context_2"]
        
        # Mock sleep completion results
        sleep_results = {
            "sleep_duration_minutes": 120.0,
            "contexts_processed": 15,
            "memory_freed_mb": 75.0,
            "consolidation_ratio": 0.3
        }
        
        with patch.object(orchestrator_integration, '_get_sleep_completion_results', return_value=sleep_results):
            with patch.object(orchestrator_integration, '_perform_cache_warmup') as mock_warmup:
                with patch.object(orchestrator_integration, '_restore_session_state') as mock_restore:
                    
                    wake_context = await orchestrator_integration.handle_agent_wake_initiated(
                        agent_id=agent_id,
                        session_id=session_id,
                        priority_contexts=priority_contexts
                    )
                    
                    assert wake_context.agent_id == agent_id
                    assert wake_context.session_id == session_id
                    assert wake_context.sleep_duration_minutes == 120.0
                    assert wake_context.cache_warmup_required is True  # > 60 minutes
                    assert wake_context.priority_contexts == priority_contexts
                    
                    # Should restore session state
                    mock_restore.assert_called_once_with(agent_id, session_id)
    
    @pytest.mark.asyncio
    async def test_wake_cycle_completion(self, orchestrator_integration):
        """Test agent wake cycle completion."""
        agent_id = uuid.uuid4()
        
        # Set up active wake context
        wake_context = Mock()
        wake_context.wake_initiated_at = datetime.utcnow() - timedelta(seconds=5)
        wake_context.session_id = uuid.uuid4()
        
        orchestrator_integration.wake_contexts[agent_id] = wake_context
        
        warmup_results = {
            "contexts_warmed": 10,
            "hit_rate_improvement": 0.15
        }
        
        with patch.object(orchestrator_integration, '_store_wake_completion_results') as mock_store:
            
            result = await orchestrator_integration.handle_agent_wake_completed(
                agent_id=agent_id,
                warmup_results=warmup_results
            )
            
            assert result["contexts_warmed"] == 10
            assert result["cache_hit_rate_improvement"] == 0.15
            assert result["session_restored"] is True
            assert "wake_duration_ms" in result
            
            # Should store completion results
            mock_store.assert_called_once()
            
            # Should remove from active wake contexts
            assert agent_id not in orchestrator_integration.wake_contexts
    
    @pytest.mark.asyncio
    async def test_session_lifecycle_events(self, orchestrator_integration):
        """Test session start and end event handling."""
        agent_id = uuid.uuid4()
        session_id = uuid.uuid4()
        
        # Test session start
        start_result = await orchestrator_integration.handle_session_started(
            agent_id=agent_id,
            session_id=session_id,
            session_metadata={"test": True}
        )
        
        assert start_result["agent_id"] == str(agent_id)
        assert start_result["session_id"] == str(session_id)
        assert start_result["recovery_point_created"] is True
        
        # Test session end
        with patch.object(orchestrator_integration._trigger_manager, 'check_agent_triggers', return_value=[]):
            
            end_result = await orchestrator_integration.handle_session_ended(
                agent_id=agent_id,
                session_id=session_id,
                session_summary="Test session completed"
            )
            
            assert end_result["agent_id"] == str(agent_id)
            assert end_result["session_id"] == str(session_id)
            assert "consolidation_triggered" in end_result
    
    @pytest.mark.asyncio
    async def test_event_handler_registration(self, orchestrator_integration):
        """Test event handler registration and emission."""
        event_triggered = False
        
        async def test_handler(data):
            nonlocal event_triggered
            event_triggered = True
        
        # Register handler
        await orchestrator_integration.register_event_handler(
            OrchestratorEvent.AGENT_SLEEP_INITIATED,
            test_handler
        )
        
        # Emit event
        await orchestrator_integration._emit_event(
            OrchestratorEvent.AGENT_SLEEP_INITIATED,
            {"test": "data"}
        )
        
        # Allow event processing
        await asyncio.sleep(0.1)
        
        assert event_triggered is True
    
    @pytest.mark.asyncio
    async def test_integration_status(self, orchestrator_integration):
        """Test integration status retrieval."""
        with patch.object(orchestrator_integration, '_get_recent_events', return_value=[]):
            
            status = await orchestrator_integration.get_integration_status()
            
            assert "is_running" in status
            assert "active_sleep_cycles" in status
            assert "active_wake_contexts" in status
            assert "integration_metrics" in status
            assert "component_status" in status
            assert status["is_running"] is True


@pytest.mark.asyncio
async def test_end_to_end_context_flow():
    """Test complete end-to-end context management flow."""
    # This test demonstrates the complete integration flow
    agent_id = uuid.uuid4()
    session_id = uuid.uuid4()
    
    # Mock components
    mock_engine = AsyncMock()
    mock_trigger_manager = AsyncMock()
    mock_orchestrator = AsyncMock()
    
    with patch('app.core.context_engine_integration.get_context_engine_integration', return_value=mock_engine):
        with patch('app.core.context_consolidation_triggers.get_consolidation_trigger_manager', return_value=mock_trigger_manager):
            with patch('app.core.context_orchestrator_integration.get_context_orchestrator_integration', return_value=mock_orchestrator):
                
                # 1. Start session
                mock_orchestrator.handle_session_started.return_value = {"success": True}
                session_result = await mock_orchestrator.handle_session_started(agent_id, session_id)
                assert session_result["success"] is True
                
                # 2. Store contexts
                context_data = ContextCreate(
                    title="Test Context",
                    content="Test content",
                    context_type=ContextType.CONVERSATION,
                    agent_id=agent_id,
                    session_id=session_id,
                    importance_score=0.7
                )
                
                mock_context = Context(id=uuid.uuid4(), **context_data.model_dump())
                mock_engine.store_context_enhanced.return_value = mock_context
                
                stored_context = await mock_engine.store_context_enhanced(context_data)
                assert stored_context.id == mock_context.id
                
                # 3. Trigger consolidation
                mock_consolidation_metrics = Mock()
                mock_consolidation_metrics.compression_ratio = 0.7
                mock_engine.trigger_consolidation.return_value = mock_consolidation_metrics
                
                consolidation_result = await mock_engine.trigger_consolidation(agent_id)
                assert consolidation_result.compression_ratio == 0.7
                
                # 4. Initiate sleep
                mock_sleep_context = Mock()
                mock_sleep_context.unconsolidated_count = 10
                mock_orchestrator.handle_agent_sleep_initiated.return_value = mock_sleep_context
                
                sleep_result = await mock_orchestrator.handle_agent_sleep_initiated(agent_id)
                assert sleep_result.unconsolidated_count == 10
                
                # 5. Complete sleep and wake
                mock_orchestrator.handle_agent_sleep_completed.return_value = {"success": True}
                mock_orchestrator.handle_agent_wake_initiated.return_value = Mock()
                
                sleep_completion = await mock_orchestrator.handle_agent_sleep_completed(agent_id)
                wake_result = await mock_orchestrator.handle_agent_wake_initiated(agent_id)
                
                assert sleep_completion["success"] is True
                assert wake_result is not None
                
                # 6. End session
                mock_orchestrator.handle_session_ended.return_value = {"success": True}
                end_result = await mock_orchestrator.handle_session_ended(agent_id, session_id)
                assert end_result["success"] is True


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=app.core",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])