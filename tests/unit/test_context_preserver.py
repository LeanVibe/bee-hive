"""
Unit Tests for Production Context Preserver

Tests the complete context preservation engine for agent handoff continuity.
Validates compression, integrity checking, and context restoration.
"""

import pytest
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any

from app.core.communication.context_preserver import ProductionContextPreserver
from app.core.communication.protocol_models import ContextPackage
from app.core.agents.universal_agent_interface import AgentType


class TestProductionContextPreserver:
    """Test suite for ProductionContextPreserver."""
    
    @pytest.fixture
    def context_preserver(self):
        """Create a context preserver instance."""
        return ProductionContextPreserver()
    
    @pytest.fixture
    def sample_execution_context(self):
        """Create a sample execution context for testing."""
        return {
            "variables": {
                "project_name": "bee-hive",
                "current_task": "context_preservation",
                "file_count": 42,
                "status": "in_progress"
            },
            "current_state": {
                "working_directory": "/tmp/workspace",
                "last_command": "implement_context_preserver",
                "active_files": ["context_preserver.py", "protocol_models.py"]
            },
            "task_history": [
                {
                    "task_id": "task_001",
                    "task_type": "implementation",
                    "completed_at": "2024-01-15T10:30:00Z",
                    "result": "success"
                },
                {
                    "task_id": "task_002", 
                    "task_type": "testing",
                    "completed_at": "2024-01-15T11:00:00Z",
                    "result": "in_progress"
                }
            ],
            "intermediate_results": [
                {"step": "analysis", "data": {"complexity": "high"}},
                {"step": "design", "data": {"pattern": "producer_consumer"}}
            ],
            "files_created": [
                "app/core/communication/context_preserver.py",
                "tests/unit/test_context_preserver.py"
            ],
            "files_modified": [
                "app/core/communication/__init__.py"
            ],
            "workflow_position": "implementation_phase",
            "required_capabilities": ["code_implementation", "testing"]
        }
    
    @pytest.mark.asyncio
    async def test_package_context_basic(self, context_preserver, sample_execution_context):
        """Test basic context packaging functionality."""
        # Test with default compression
        package = await context_preserver.package_context(
            execution_context=sample_execution_context,
            target_agent_type=AgentType.CLAUDE_CODE
        )
        
        # Validate package structure
        assert package.package_id is not None
        assert package.source_agent_id == "production_context_preserver"
        assert package.target_agent_id == "claude_code_agent"
        assert package.context_format_version == "2.0"
        assert package.compression_used is True
        assert package.validation_status == "valid"
        assert package.context_integrity_hash is not None
        assert package.package_size_bytes > 0
        
        # Validate metadata
        assert "compressed_data" in package.metadata
        assert "compression_level" in package.metadata
        assert "original_size_bytes" in package.metadata
        assert "compression_ratio" in package.metadata
        assert "packaging_time_ms" in package.metadata
        
        # Validate compression ratio
        compression_ratio = package.metadata["compression_ratio"]
        assert 0.1 <= compression_ratio <= 1.0  # Should achieve some compression
    
    @pytest.mark.asyncio
    async def test_package_context_no_compression(self, context_preserver, sample_execution_context):
        """Test context packaging without compression."""
        package = await context_preserver.package_context(
            execution_context=sample_execution_context,
            target_agent_type=AgentType.CURSOR,
            compression_level=0
        )
        
        assert package.compression_used is False
        assert package.metadata["compression_level"] == 0
        assert package.metadata["compression_ratio"] == 1.0
    
    @pytest.mark.asyncio
    async def test_package_context_maximum_compression(self, context_preserver, sample_execution_context):
        """Test context packaging with maximum compression."""
        package = await context_preserver.package_context(
            execution_context=sample_execution_context,
            target_agent_type=AgentType.GITHUB_COPILOT,
            compression_level=9
        )
        
        assert package.compression_used is True
        assert package.metadata["compression_level"] == 9
        # Maximum compression should achieve better ratio than default
        assert package.metadata["compression_ratio"] <= 0.8
    
    @pytest.mark.asyncio
    async def test_agent_specific_optimizations(self, context_preserver, sample_execution_context):
        """Test agent-specific context optimizations."""
        # Test Claude Code optimizations
        claude_package = await context_preserver.package_context(
            execution_context=sample_execution_context,
            target_agent_type=AgentType.CLAUDE_CODE
        )
        claude_opts = claude_package.metadata["target_optimizations"]
        assert claude_opts["file_format_preference"] == "markdown"
        assert claude_opts["context_style"] == "detailed"
        assert claude_opts["include_history"] is True
        
        # Test Cursor optimizations
        cursor_package = await context_preserver.package_context(
            execution_context=sample_execution_context,
            target_agent_type=AgentType.CURSOR
        )
        cursor_opts = cursor_package.metadata["target_optimizations"]
        assert cursor_opts["file_format_preference"] == "json"
        assert cursor_opts["context_style"] == "minimal"
        assert cursor_opts["include_history"] is False
    
    @pytest.mark.asyncio
    async def test_validate_context_integrity_valid(self, context_preserver, sample_execution_context):
        """Test integrity validation for valid context package."""
        package = await context_preserver.package_context(
            execution_context=sample_execution_context,
            target_agent_type=AgentType.CLAUDE_CODE
        )
        
        validation_result = await context_preserver.validate_context_integrity(package)
        
        assert validation_result["is_valid"] is True
        assert validation_result["integrity_score"] == 1.0
        assert validation_result["error"] is None
        assert len(validation_result["checks_failed"]) == 0
        assert "package_structure" in validation_result["checks_passed"]
        assert "sha256_integrity" in validation_result["checks_passed"]
        assert "metadata_presence" in validation_result["checks_passed"]
    
    @pytest.mark.asyncio
    async def test_validate_context_integrity_corrupted(self, context_preserver, sample_execution_context):
        """Test integrity validation for corrupted context package."""
        package = await context_preserver.package_context(
            execution_context=sample_execution_context,
            target_agent_type=AgentType.CLAUDE_CODE
        )
        
        # Corrupt the data
        package.metadata["compressed_data"] = b"corrupted_data"
        
        validation_result = await context_preserver.validate_context_integrity(package)
        
        assert validation_result["is_valid"] is False
        assert "sha256_integrity" in validation_result["checks_failed"]
        assert "SHA256 integrity check failed" in validation_result["error"]
    
    @pytest.mark.asyncio
    async def test_validate_context_integrity_missing_data(self, context_preserver):
        """Test integrity validation for package with missing data."""
        # Create incomplete package
        package = ContextPackage(
            source_agent_id="test_agent",
            target_agent_id="target_agent"
        )
        package.metadata = {}  # Missing compressed data
        
        validation_result = await context_preserver.validate_context_integrity(package)
        
        assert validation_result["is_valid"] is False
        assert "compressed_data" in validation_result["checks_failed"]
        assert "Missing compressed data" in validation_result["error"]
    
    @pytest.mark.asyncio
    async def test_restore_context_basic(self, context_preserver, sample_execution_context):
        """Test basic context restoration functionality."""
        # Package context first
        package = await context_preserver.package_context(
            execution_context=sample_execution_context,
            target_agent_type=AgentType.CLAUDE_CODE
        )
        
        # Restore context
        restored_context = await context_preserver.restore_context(package)
        
        # Validate restored context
        assert "variables" in restored_context
        assert "current_state" in restored_context
        assert "task_history" in restored_context
        assert "restoration_metadata" in restored_context
        
        # Check specific values are preserved
        assert restored_context["variables"]["project_name"] == "bee-hive"
        assert restored_context["variables"]["current_task"] == "context_preservation"
        assert restored_context["current_state"]["working_directory"] == "/tmp/workspace"
        assert len(restored_context["task_history"]) == 2
        
        # Check restoration metadata
        restoration_meta = restored_context["restoration_metadata"]
        assert "restored_at" in restoration_meta
        assert "restoration_time_ms" in restoration_meta
        assert restoration_meta["format_version"] == "2.0"
        assert restoration_meta["source_agent_type"] == "context_preserver"
    
    @pytest.mark.asyncio
    async def test_full_roundtrip_preservation(self, context_preserver, sample_execution_context):
        """Test complete roundtrip context preservation."""
        original_context = sample_execution_context.copy()
        
        # Package context
        package = await context_preserver.package_context(
            execution_context=original_context,
            target_agent_type=AgentType.CLAUDE_CODE,
            compression_level=6
        )
        
        # Validate package
        validation_result = await context_preserver.validate_context_integrity(package)
        assert validation_result["is_valid"] is True
        
        # Restore context
        restored_context = await context_preserver.restore_context(package)
        
        # Verify all original data is preserved
        assert restored_context["variables"] == original_context["variables"]
        assert restored_context["current_state"] == original_context["current_state"]
        assert restored_context["task_history"] == original_context["task_history"]
        assert restored_context["intermediate_results"] == original_context["intermediate_results"]
        assert restored_context["files_created"] == original_context["files_created"]
        assert restored_context["files_modified"] == original_context["files_modified"]
    
    @pytest.mark.asyncio
    async def test_large_context_handling(self, context_preserver):
        """Test handling of large contexts."""
        # Create large context
        large_context = {
            "variables": {f"var_{i}": f"value_{i}" for i in range(1000)},
            "current_state": {"large_data": "x" * 10000},  # 10KB string
            "task_history": [{"task": f"task_{i}", "data": "x" * 100} for i in range(100)],
            "files_created": [f"file_{i}.py" for i in range(100)],
            "files_modified": [f"modified_{i}.py" for i in range(50)]
        }
        
        # Package with high compression
        package = await context_preserver.package_context(
            execution_context=large_context,
            target_agent_type=AgentType.CLAUDE_CODE,
            compression_level=9
        )
        
        # Verify compression is effective
        compression_ratio = package.metadata["compression_ratio"]
        assert compression_ratio < 0.5  # Should achieve good compression
        
        # Verify restoration works
        restored = await context_preserver.restore_context(package)
        assert len(restored["variables"]) == 1000
        assert len(restored["task_history"]) == 100
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self, context_preserver, sample_execution_context):
        """Test performance requirements are met."""
        import time
        
        # Test packaging performance (<1s requirement)
        start_time = time.time()
        package = await context_preserver.package_context(
            execution_context=sample_execution_context,
            target_agent_type=AgentType.CLAUDE_CODE
        )
        packaging_time = (time.time() - start_time) * 1000
        
        assert packaging_time < 1000  # Less than 1 second
        assert package.metadata["packaging_time_ms"] < 1000
        
        # Test restoration performance (<500ms requirement)
        start_time = time.time()
        restored = await context_preserver.restore_context(package)
        restoration_time = (time.time() - start_time) * 1000
        
        assert restoration_time < 500  # Less than 500ms
        assert restored["restoration_metadata"]["restoration_time_ms"] < 500
    
    @pytest.mark.asyncio
    async def test_error_handling(self, context_preserver):
        """Test error handling and recovery."""
        # Test restoration with invalid package
        invalid_package = ContextPackage()
        invalid_package.metadata = {"compressed_data": b"invalid_json_data"}
        invalid_package.context_integrity_hash = "invalid_hash"
        
        with pytest.raises(Exception) as exc_info:
            await context_preserver.restore_context(invalid_package)
        
        assert "Context package validation failed" in str(exc_info.value)
    
    def test_compression_strategies(self, context_preserver):
        """Test different compression strategies."""
        test_data = b"This is test data that should compress well" * 100
        
        # Test no compression
        no_compression = context_preserver._apply_compression(test_data, 0)
        assert no_compression == test_data
        
        # Test different compression levels
        level_1 = context_preserver._apply_compression(test_data, 1)
        level_6 = context_preserver._apply_compression(test_data, 6)
        level_9 = context_preserver._apply_compression(test_data, 9)
        
        # Higher compression should result in smaller size
        assert len(level_9) <= len(level_6) <= len(level_1) < len(test_data)
        
        # Test decompression
        assert context_preserver._decompress_data(level_6, 6) == test_data
        assert context_preserver._decompress_data(level_9, 9) == test_data
    
    def test_helper_methods(self, context_preserver):
        """Test helper method functionality."""
        # Test agent type and ID getters
        assert context_preserver._get_current_agent_type() == "context_preserver"
        assert context_preserver._get_current_agent_id() == "production_context_preserver"
        
        # Test agent-specific optimizations
        test_data = {"test": "data"}
        
        claude_optimized = context_preserver._optimize_for_target_agent(
            test_data, AgentType.CLAUDE_CODE
        )
        assert claude_optimized["target_optimizations"]["context_style"] == "detailed"
        
        cursor_optimized = context_preserver._optimize_for_target_agent(
            test_data, AgentType.CURSOR
        )
        assert cursor_optimized["target_optimizations"]["context_style"] == "minimal"


# Performance benchmark tests
class TestContextPreserverPerformance:
    """Performance benchmark tests for context preserver."""
    
    @pytest.mark.asyncio
    async def test_scaling_performance(self):
        """Test performance scaling with different context sizes."""
        context_preserver = ProductionContextPreserver()
        
        # Test different context sizes
        sizes = [100, 1000, 5000]  # Number of variables
        
        for size in sizes:
            context = {
                "variables": {f"var_{i}": f"value_{i}_{'x' * 10}" for i in range(size)},
                "current_state": {"size": size},
                "task_history": [],
                "files_created": [],
                "files_modified": []
            }
            
            import time
            start_time = time.time()
            
            package = await context_preserver.package_context(
                execution_context=context,
                target_agent_type=AgentType.CLAUDE_CODE
            )
            
            packaging_time = (time.time() - start_time) * 1000
            
            # Performance should scale reasonably
            assert packaging_time < size * 2  # Linear scaling allowance
            
            # Restoration performance
            start_time = time.time()
            restored = await context_preserver.restore_context(package)
            restoration_time = (time.time() - start_time) * 1000
            
            assert restoration_time < 500  # Always under 500ms requirement
            assert len(restored["variables"]) == size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])