"""
Comprehensive tests for Enhanced Git Checkpoint Manager and Sleep-Wake Integration.

Tests include:
- Enhanced checkpoint creation with sleep cycle data
- Context threshold monitoring and automatic checkpoints  
- Recovery validation and state consistency
- Performance analytics and optimization detection
- Integration with existing infrastructure
"""

import asyncio
import pytest
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from app.core.enhanced_git_checkpoint_manager import (
    EnhancedGitCheckpointManager, get_enhanced_git_checkpoint_manager
)
from app.core.enhanced_sleep_wake_integration import (
    EnhancedSleepWakeIntegration, get_enhanced_sleep_wake_integration
)
from app.models.sleep_wake import CheckpointType, SleepState
from app.models.agent import Agent


class TestEnhancedGitCheckpointManager:
    """Test suite for Enhanced Git Checkpoint Manager."""
    
    @pytest.fixture
    async def enhanced_checkpoint_manager(self):
        """Create enhanced checkpoint manager for testing."""
        with patch('app.core.enhanced_git_checkpoint_manager.get_checkpoint_manager') as mock_base:
            mock_base_manager = Mock()
            mock_base_manager._create_git_checkpoint = AsyncMock(return_value="commit_hash_123")
            mock_base_manager.restore_checkpoint = AsyncMock(return_value=(True, {"restored": True}))
            mock_base.return_value = mock_base_manager
            
            manager = EnhancedGitCheckpointManager(mock_base_manager)
            yield manager
    
    @pytest.fixture
    def sample_agent_id(self):
        """Sample agent ID for testing."""
        return uuid.uuid4()
    
    @pytest.fixture
    def sample_sleep_cycle_data(self):
        """Sample sleep cycle data for testing."""
        return {
            "context_usage_percent": 78.5,
            "consolidation_trigger": "scheduled",
            "trigger_reason": "high_context_usage",
            "cycle_type": "scheduled",
            "memory_usage_mb": 150.0,
            "active_context_count": 12,
            "consolidation_performed": True,
            "expected_wake_time": (datetime.utcnow() + timedelta(hours=8)).isoformat()
        }
    
    @pytest.mark.asyncio
    async def test_create_enhanced_sleep_cycle_checkpoint_success(
        self, 
        enhanced_checkpoint_manager, 
        sample_agent_id, 
        sample_sleep_cycle_data
    ):
        """Test successful creation of enhanced sleep cycle checkpoint."""
        
        # Mock the comprehensive state collection
        with patch.object(
            enhanced_checkpoint_manager, 
            '_collect_comprehensive_state',
            return_value={"agent": {"id": str(sample_agent_id)}, "state": "active"}
        ), patch.object(
            enhanced_checkpoint_manager,
            '_store_checkpoint_analytics'
        ) as mock_store_analytics, patch.object(
            enhanced_checkpoint_manager,
            '_update_performance_metrics'
        ) as mock_update_metrics:
            
            # Execute checkpoint creation
            result = await enhanced_checkpoint_manager.create_enhanced_sleep_cycle_checkpoint(
                agent_id=sample_agent_id,
                sleep_cycle_data=sample_sleep_cycle_data,
                cycle_id=uuid.uuid4()
            )
            
            # Assertions
            assert result == "commit_hash_123"
            
            # Verify base manager was called with enhanced metadata
            enhanced_checkpoint_manager.base_manager._create_git_checkpoint.assert_called_once()
            call_args = enhanced_checkpoint_manager.base_manager._create_git_checkpoint.call_args
            
            assert call_args[1]["agent_id"] == sample_agent_id
            assert call_args[1]["checkpoint_type"] == CheckpointType.SLEEP_CYCLE
            
            # Verify metadata contains enhanced information
            metadata = call_args[1]["metadata"]
            assert "sleep_cycle" in metadata
            assert metadata["sleep_cycle"]["context_usage_percent"] == 78.5
            assert metadata["sleep_cycle"]["consolidation_trigger"] == "scheduled"
            assert "performance" in metadata
            assert "context" in metadata
            
            # Verify analytics and metrics were updated
            mock_store_analytics.assert_called_once()
            mock_update_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_enhanced_checkpoint_failure_handling(
        self,
        enhanced_checkpoint_manager,
        sample_agent_id,
        sample_sleep_cycle_data
    ):
        """Test proper handling of checkpoint creation failures."""
        
        # Mock base manager to return failure
        enhanced_checkpoint_manager.base_manager._create_git_checkpoint.return_value = None
        
        with patch.object(
            enhanced_checkpoint_manager,
            '_collect_comprehensive_state',
            return_value={}
        ):
            
            result = await enhanced_checkpoint_manager.create_enhanced_sleep_cycle_checkpoint(
                agent_id=sample_agent_id,
                sleep_cycle_data=sample_sleep_cycle_data
            )
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_restore_from_enhanced_checkpoint_success(
        self,
        enhanced_checkpoint_manager,
        sample_agent_id
    ):
        """Test successful restoration from enhanced checkpoint."""
        
        git_commit_hash = "commit_hash_123"
        
        with patch.object(
            enhanced_checkpoint_manager,
            '_load_enhanced_metadata',
            return_value={
                "sleep_cycle": {"context_usage_percent": 75},
                "performance": {"success_rate": 95},
                "context": {"total_contexts": 10}
            }
        ), patch.object(
            enhanced_checkpoint_manager,
            '_validate_context_consistency',
            return_value={"valid": True, "issues": []}
        ), patch.object(
            enhanced_checkpoint_manager,
            '_restore_sleep_cycle_specific_state',
            return_value={"restoration_successful": True}
        ):
            
            success, metadata = await enhanced_checkpoint_manager.restore_from_enhanced_checkpoint(
                git_commit_hash=git_commit_hash,
                agent_id=sample_agent_id,
                validate_context=True
            )
            
            assert success is True
            assert "enhanced_metadata" in metadata
            assert "restoration_time_seconds" in metadata
            assert "context_validation_performed" in metadata
            assert metadata["context_validation_performed"] is True
            
            # Verify base manager restoration was called
            enhanced_checkpoint_manager.base_manager.restore_checkpoint.assert_called_once_with(
                git_commit_hash, sample_agent_id
            )
    
    @pytest.mark.asyncio
    async def test_restore_with_context_validation_failure(
        self,
        enhanced_checkpoint_manager,
        sample_agent_id
    ):
        """Test restoration with context validation failure."""
        
        git_commit_hash = "commit_hash_123"
        
        with patch.object(
            enhanced_checkpoint_manager,
            '_load_enhanced_metadata',
            return_value={"sleep_cycle": {}}
        ), patch.object(
            enhanced_checkpoint_manager,
            '_validate_context_consistency',
            return_value={"valid": False, "issues": ["Agent ID mismatch", "Timestamp inconsistency"]}
        ), patch.object(
            enhanced_checkpoint_manager,
            '_restore_sleep_cycle_specific_state',
            return_value={"restoration_successful": True}
        ):
            
            success, metadata = await enhanced_checkpoint_manager.restore_from_enhanced_checkpoint(
                git_commit_hash=git_commit_hash,
                agent_id=sample_agent_id,
                validate_context=True
            )
            
            # Should still succeed but with validation issues noted
            assert success is True
            assert "context_validation_issues" in metadata
            assert len(metadata["context_validation_issues"]) == 2
    
    @pytest.mark.asyncio
    async def test_create_context_threshold_checkpoint(
        self,
        enhanced_checkpoint_manager,
        sample_agent_id
    ):
        """Test context threshold checkpoint creation."""
        
        with patch.object(
            enhanced_checkpoint_manager,
            'create_enhanced_sleep_cycle_checkpoint',
            return_value="threshold_commit_hash"
        ) as mock_create:
            
            result = await enhanced_checkpoint_manager.create_context_threshold_checkpoint(
                agent_id=sample_agent_id,
                context_usage_percent=87.5,
                threshold_trigger="85_percent",
                consolidation_opportunity=True
            )
            
            assert result == "threshold_commit_hash"
            
            # Verify the call was made with correct threshold data
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            
            assert call_args[1]["agent_id"] == sample_agent_id
            
            sleep_cycle_data = call_args[1]["sleep_cycle_data"]
            assert sleep_cycle_data["context_usage_percent"] == 87.5
            assert sleep_cycle_data["threshold_trigger"] == "85_percent"
            assert sleep_cycle_data["consolidation_opportunity"] is True
            assert sleep_cycle_data["checkpoint_reason"] == "context_threshold"
    
    @pytest.mark.asyncio
    async def test_get_checkpoint_analytics(self, enhanced_checkpoint_manager):
        """Test comprehensive checkpoint analytics retrieval."""
        
        with patch.object(
            enhanced_checkpoint_manager,
            '_get_base_checkpoint_analytics',
            return_value={"total_checkpoints": 50, "successful_checkpoints": 48}
        ), patch.object(
            enhanced_checkpoint_manager,
            '_analyze_performance_trends',
            return_value={"trend_direction": "improving", "improvement_score": 85}
        ), patch.object(
            enhanced_checkpoint_manager,
            '_identify_optimization_opportunities',
            return_value={"opportunities": ["context_compression"], "priority_score": 75}
        ), patch.object(
            enhanced_checkpoint_manager,
            '_calculate_health_metrics',
            return_value={"overall_health": "good", "checkpoint_success_rate": 96.0}
        ):
            
            analytics = await enhanced_checkpoint_manager.get_checkpoint_analytics(
                include_performance_trends=True
            )
            
            assert "summary" in analytics
            assert "performance_trends" in analytics
            assert "optimization_opportunities" in analytics
            assert "health_metrics" in analytics
            
            assert analytics["summary"]["total_checkpoints"] == 50
            assert analytics["performance_trends"]["trend_direction"] == "improving"
            assert analytics["health_metrics"]["overall_health"] == "good"
    
    def test_generate_sleep_cycle_branch_name(self, enhanced_checkpoint_manager, sample_agent_id):
        """Test sleep cycle branch name generation."""
        
        cycle_id = uuid.uuid4()
        
        branch_name = enhanced_checkpoint_manager._generate_sleep_cycle_branch_name(
            sample_agent_id, cycle_id
        )
        
        assert branch_name.startswith("sleep-cycle")
        assert str(sample_agent_id) in branch_name
        assert str(cycle_id)[:8] in branch_name
        
        # Test without cycle_id
        branch_name_no_cycle = enhanced_checkpoint_manager._generate_sleep_cycle_branch_name(
            sample_agent_id, None
        )
        
        assert "manual" in branch_name_no_cycle


class TestEnhancedSleepWakeIntegration:
    """Test suite for Enhanced Sleep-Wake Integration."""
    
    @pytest.fixture
    async def enhanced_integration(self):
        """Create enhanced sleep-wake integration for testing."""
        with patch('app.core.enhanced_sleep_wake_integration.get_sleep_wake_manager') as mock_sleep_wake, \
             patch('app.core.enhanced_sleep_wake_integration.get_enhanced_git_checkpoint_manager') as mock_checkpoint:
            
            # Mock sleep-wake manager
            mock_sleep_manager = Mock()
            mock_sleep_manager.initiate_sleep_cycle = AsyncMock(return_value=True)
            mock_sleep_manager.initiate_wake_cycle = AsyncMock(return_value=True)
            mock_sleep_wake.return_value = mock_sleep_manager
            
            # Mock checkpoint manager
            mock_checkpoint_manager = Mock()
            mock_checkpoint_manager.create_enhanced_sleep_cycle_checkpoint = AsyncMock(
                return_value="commit_hash_456"
            )
            mock_checkpoint_manager.restore_from_enhanced_checkpoint = AsyncMock(
                return_value=(True, {"restored": True})
            )
            mock_checkpoint_manager.create_context_threshold_checkpoint = AsyncMock(
                return_value="threshold_commit_789"
            )
            mock_checkpoint.return_value = mock_checkpoint_manager
            
            integration = EnhancedSleepWakeIntegration(mock_sleep_manager, mock_checkpoint_manager)
            yield integration
    
    @pytest.fixture
    def sample_agent_id(self):
        """Sample agent ID for testing."""
        return uuid.uuid4()
    
    @pytest.mark.asyncio
    async def test_initiate_enhanced_sleep_cycle_success(
        self,
        enhanced_integration,
        sample_agent_id
    ):
        """Test successful enhanced sleep cycle initiation."""
        
        expected_wake_time = datetime.utcnow() + timedelta(hours=6)
        
        with patch.object(
            enhanced_integration,
            '_prepare_sleep_cycle_data',
            return_value={
                "cycle_id": uuid.uuid4(),
                "context_usage_percent": 82.0,
                "consolidation_trigger": "scheduled"
            }
        ), patch.object(
            enhanced_integration,
            '_store_enhanced_sleep_metadata'
        ) as mock_store_metadata, patch.object(
            enhanced_integration,
            '_analyze_sleep_performance'
        ) as mock_analyze_performance:
            
            success, metadata = await enhanced_integration.initiate_enhanced_sleep_cycle(
                agent_id=sample_agent_id,
                cycle_type="scheduled",
                expected_wake_time=expected_wake_time,
                context_usage_percent=82.0,
                consolidation_trigger="high_context_usage"
            )
            
            assert success is True
            assert metadata["sleep_cycle_initiated"] is True
            assert metadata["checkpoint_created"] is True
            assert metadata["git_commit_hash"] == "commit_hash_456"
            assert metadata["cycle_type"] == "scheduled"
            assert metadata["consolidation_trigger"] == "high_context_usage"
            assert metadata["context_usage_percent"] == 82.0
            assert "operation_time_seconds" in metadata
            
            # Verify checkpoint creation was called
            enhanced_integration.checkpoint_manager.create_enhanced_sleep_cycle_checkpoint.assert_called_once()
            
            # Verify sleep cycle initiation was called
            enhanced_integration.sleep_wake_manager.initiate_sleep_cycle.assert_called_once_with(
                agent_id=sample_agent_id,
                cycle_type="scheduled",
                expected_wake_time=expected_wake_time
            )
            
            # Verify metadata storage and analysis
            mock_store_metadata.assert_called_once()
            mock_analyze_performance.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initiate_enhanced_sleep_cycle_checkpoint_failure(
        self,
        enhanced_integration,
        sample_agent_id
    ):
        """Test enhanced sleep cycle with checkpoint creation failure."""
        
        # Mock checkpoint creation to fail
        enhanced_integration.checkpoint_manager.create_enhanced_sleep_cycle_checkpoint.return_value = None
        
        with patch.object(
            enhanced_integration,
            '_prepare_sleep_cycle_data',
            return_value={"cycle_id": uuid.uuid4()}
        ), patch.object(
            enhanced_integration,
            '_store_enhanced_sleep_metadata'
        ), patch.object(
            enhanced_integration,
            '_analyze_sleep_performance'
        ):
            
            success, metadata = await enhanced_integration.initiate_enhanced_sleep_cycle(
                agent_id=sample_agent_id,
                cycle_type="manual"
            )
            
            # Should still succeed even if checkpoint fails
            assert success is True
            assert metadata["checkpoint_created"] is False
            assert metadata["git_commit_hash"] is None
    
    @pytest.mark.asyncio
    async def test_initiate_enhanced_wake_cycle_with_restoration(
        self,
        enhanced_integration,
        sample_agent_id
    ):
        """Test enhanced wake cycle with checkpoint restoration."""
        
        git_commit_hash = "restore_commit_123"
        
        with patch.object(
            enhanced_integration,
            '_perform_wake_validation',
            return_value={"validation_score": 95, "agent_responsive": True}
        ) as mock_validation:
            
            success, metadata = await enhanced_integration.initiate_enhanced_wake_cycle(
                agent_id=sample_agent_id,
                git_commit_hash=git_commit_hash,
                validate_context=True,
                performance_validation=True
            )
            
            assert success is True
            assert metadata["wake_cycle_initiated"] is True
            assert metadata["checkpoint_restored"] is True
            assert metadata["git_commit_hash"] == git_commit_hash
            assert metadata["context_validation_performed"] is True
            assert metadata["performance_validation_performed"] is True
            
            # Verify checkpoint restoration was called
            enhanced_integration.checkpoint_manager.restore_from_enhanced_checkpoint.assert_called_once_with(
                git_commit_hash=git_commit_hash,
                agent_id=sample_agent_id,
                validate_context=True
            )
            
            # Verify wake cycle initiation was called
            enhanced_integration.sleep_wake_manager.initiate_wake_cycle.assert_called_once_with(
                sample_agent_id
            )
            
            # Verify validation was performed
            mock_validation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_monitor_context_thresholds_critical(
        self,
        enhanced_integration,
        sample_agent_id
    ):
        """Test context threshold monitoring with critical threshold reached."""
        
        critical_usage = 96.0  # Above critical threshold (95%)
        
        with patch.object(
            enhanced_integration,
            'initiate_enhanced_sleep_cycle',
            return_value=(True, {"emergency_sleep": True})
        ) as mock_emergency_sleep:
            
            result = await enhanced_integration.monitor_context_thresholds(
                agent_id=sample_agent_id,
                current_context_usage=critical_usage,
                memory_usage_mb=200.0,
                auto_trigger_checkpoint=True
            )
            
            assert result["agent_id"] == str(sample_agent_id)
            assert result["context_usage_percent"] == critical_usage
            assert result["threshold_status"] == "critical"
            assert "Emergency checkpoint created" in result["actions_taken"]
            assert "Emergency sleep cycle initiated" in result["actions_taken"]
            assert result["emergency_checkpoint_hash"] == "threshold_commit_789"
            
            # Verify emergency checkpoint was created
            enhanced_integration.checkpoint_manager.create_context_threshold_checkpoint.assert_called_once_with(
                agent_id=sample_agent_id,
                context_usage_percent=critical_usage,
                threshold_trigger="critical_95_percent",
                consolidation_opportunity=True
            )
            
            # Verify emergency sleep cycle was initiated
            mock_emergency_sleep.assert_called_once()
            call_kwargs = mock_emergency_sleep.call_args[1]
            assert call_kwargs["cycle_type"] == "emergency"
            assert call_kwargs["consolidation_trigger"] == "critical_context_threshold"
            assert call_kwargs["force_checkpoint"] is True
    
    @pytest.mark.asyncio
    async def test_monitor_context_thresholds_high(
        self,
        enhanced_integration,
        sample_agent_id
    ):
        """Test context threshold monitoring with high threshold reached."""
        
        high_usage = 87.0  # Above high threshold (85%) but below critical (95%)
        
        result = await enhanced_integration.monitor_context_thresholds(
            agent_id=sample_agent_id,
            current_context_usage=high_usage,
            auto_trigger_checkpoint=True
        )
        
        assert result["threshold_status"] == "high"
        assert "Preventive checkpoint created" in result["actions_taken"]
        assert result["preventive_checkpoint_hash"] == "threshold_commit_789"
        assert "Consolidation recommended soon" in result["recommendations"]
        
        # Verify preventive checkpoint was created
        enhanced_integration.checkpoint_manager.create_context_threshold_checkpoint.assert_called_once_with(
            agent_id=sample_agent_id,
            context_usage_percent=high_usage,
            threshold_trigger="high_85_percent",
            consolidation_opportunity=True
        )
    
    @pytest.mark.asyncio
    async def test_monitor_context_thresholds_low(
        self,
        enhanced_integration,
        sample_agent_id
    ):
        """Test context threshold monitoring with low context usage."""
        
        low_usage = 25.0  # Below low threshold (30%)
        
        with patch.object(
            enhanced_integration,
            '_analyze_low_context_usage',
            return_value={"optimization_potential": "high", "savings_estimate": "40%"}
        ) as mock_analyze_low:
            
            result = await enhanced_integration.monitor_context_thresholds(
                agent_id=sample_agent_id,
                current_context_usage=low_usage,
                auto_trigger_checkpoint=True
            )
            
            assert result["threshold_status"] == "low"
            assert "Context usage is low - possible over-allocation" in result["recommendations"]
            assert "optimization_analysis" in result
            
            # Verify low usage analysis was performed
            mock_analyze_low.assert_called_once_with(sample_agent_id, low_usage)
    
    @pytest.mark.asyncio
    async def test_monitor_context_thresholds_normal(
        self,
        enhanced_integration,
        sample_agent_id
    ):
        """Test context threshold monitoring with normal usage."""
        
        normal_usage = 65.0  # Between low (30%) and high (85%) thresholds
        
        result = await enhanced_integration.monitor_context_thresholds(
            agent_id=sample_agent_id,
            current_context_usage=normal_usage,
            auto_trigger_checkpoint=True
        )
        
        assert result["threshold_status"] == "normal"
        assert len(result["actions_taken"]) == 0
        assert len(result["recommendations"]) == 0
    
    @pytest.mark.asyncio
    async def test_get_integration_analytics(self, enhanced_integration):
        """Test comprehensive integration analytics retrieval."""
        
        with patch.object(
            enhanced_integration.checkpoint_manager,
            'get_checkpoint_analytics',
            return_value={
                "summary": {"total_checkpoints": 100},
                "performance_trends": {"trend": "improving"}
            }
        ), patch.object(
            enhanced_integration,
            '_get_sleep_wake_analytics',
            return_value={"total_cycles": 50, "success_rate": 98}
        ), patch.object(
            enhanced_integration,
            '_analyze_integration_performance_trends',
            return_value={"integration_trend": "stable"}
        ), patch.object(
            enhanced_integration,
            '_generate_optimization_recommendations',
            return_value=["Optimize checkpoint timing", "Improve context monitoring"]
        ), patch.object(
            enhanced_integration,
            '_calculate_integration_health',
            return_value={"overall_health": "excellent", "checkpoint_success_rate": 97.5}
        ):
            
            analytics = await enhanced_integration.get_integration_analytics(
                include_performance_trends=True,
                include_optimization_recommendations=True
            )
            
            assert "integration_metrics" in analytics
            assert "checkpoint_analytics" in analytics
            assert "sleep_wake_analytics" in analytics
            assert "performance_trends" in analytics
            assert "optimization_recommendations" in analytics
            assert "health_status" in analytics
            
            # Verify integration metrics
            assert analytics["integration_metrics"]["total_enhanced_sleep_cycles"] >= 0
            assert analytics["integration_metrics"]["successful_checkpoint_creations"] >= 0
            
            # Verify analytics data structure
            assert analytics["checkpoint_analytics"]["summary"]["total_checkpoints"] == 100
            assert analytics["sleep_wake_analytics"]["success_rate"] == 98
            assert len(analytics["optimization_recommendations"]) == 2
            assert analytics["health_status"]["overall_health"] == "excellent"


class TestIntegrationErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_checkpoint_creation_exception_handling(self):
        """Test proper exception handling during checkpoint creation."""
        
        with patch('app.core.enhanced_git_checkpoint_manager.get_checkpoint_manager') as mock_base:
            mock_base_manager = Mock()
            mock_base_manager._create_git_checkpoint = AsyncMock(
                side_effect=Exception("Git operation failed")
            )
            mock_base.return_value = mock_base_manager
            
            manager = EnhancedGitCheckpointManager(mock_base_manager)
            
            result = await manager.create_enhanced_sleep_cycle_checkpoint(
                agent_id=uuid.uuid4(),
                sleep_cycle_data={"context_usage_percent": 50},
                cycle_id=uuid.uuid4()
            )
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_restoration_exception_handling(self):
        """Test proper exception handling during restoration."""
        
        with patch('app.core.enhanced_git_checkpoint_manager.get_checkpoint_manager') as mock_base:
            mock_base_manager = Mock()
            mock_base_manager.restore_checkpoint = AsyncMock(
                side_effect=Exception("Restoration failed")
            )
            mock_base.return_value = mock_base_manager
            
            manager = EnhancedGitCheckpointManager(mock_base_manager)
            
            success, metadata = await manager.restore_from_enhanced_checkpoint(
                git_commit_hash="invalid_hash",
                agent_id=uuid.uuid4()
            )
            
            assert success is False
            assert "error" in metadata
    
    @pytest.mark.asyncio
    async def test_integration_sleep_cycle_exception_handling(self):
        """Test integration exception handling during sleep cycle initiation."""
        
        with patch('app.core.enhanced_sleep_wake_integration.get_sleep_wake_manager') as mock_sleep_wake:
            mock_sleep_manager = Mock()
            mock_sleep_manager.initiate_sleep_cycle = AsyncMock(
                side_effect=Exception("Sleep cycle failed")
            )
            mock_sleep_wake.return_value = mock_sleep_manager
            
            integration = EnhancedSleepWakeIntegration(mock_sleep_manager, Mock())
            
            success, metadata = await integration.initiate_enhanced_sleep_cycle(
                agent_id=uuid.uuid4(),
                cycle_type="test"
            )
            
            assert success is False
            assert "error" in metadata


# Performance and stress tests
class TestPerformanceAndScale:
    """Test performance characteristics and scalability."""
    
    @pytest.mark.asyncio
    async def test_concurrent_checkpoint_creation(self):
        """Test concurrent checkpoint creation for multiple agents."""
        
        with patch('app.core.enhanced_git_checkpoint_manager.get_checkpoint_manager') as mock_base:
            mock_base_manager = Mock()
            mock_base_manager._create_git_checkpoint = AsyncMock(return_value="concurrent_hash")
            mock_base.return_value = mock_base_manager
            
            manager = EnhancedGitCheckpointManager(mock_base_manager)
            
            # Create multiple concurrent checkpoint operations
            agent_ids = [uuid.uuid4() for _ in range(5)]
            tasks = []
            
            for agent_id in agent_ids:
                task = manager.create_enhanced_sleep_cycle_checkpoint(
                    agent_id=agent_id,
                    sleep_cycle_data={"context_usage_percent": 60},
                    cycle_id=uuid.uuid4()
                )
                tasks.append(task)
            
            # Execute concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all succeeded
            for result in results:
                assert isinstance(result, str)  # Should return commit hash
                assert result == "concurrent_hash"
    
    @pytest.mark.asyncio
    async def test_large_metadata_handling(self):
        """Test handling of large metadata payloads."""
        
        with patch('app.core.enhanced_git_checkpoint_manager.get_checkpoint_manager') as mock_base:
            mock_base_manager = Mock()
            mock_base_manager._create_git_checkpoint = AsyncMock(return_value="large_metadata_hash")
            mock_base.return_value = mock_base_manager
            
            manager = EnhancedGitCheckpointManager(mock_base_manager)
            
            # Create large sleep cycle data
            large_sleep_data = {
                "context_usage_percent": 75,
                "consolidation_trigger": "large_data_test",
                "large_metadata": {f"key_{i}": f"value_{i}" * 100 for i in range(1000)}
            }
            
            result = await manager.create_enhanced_sleep_cycle_checkpoint(
                agent_id=uuid.uuid4(),
                sleep_cycle_data=large_sleep_data,
                cycle_id=uuid.uuid4()
            )
            
            assert result == "large_metadata_hash"