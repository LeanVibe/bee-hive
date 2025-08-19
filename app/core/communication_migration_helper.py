"""
Communication Protocol Migration Helper - Phase 0 POC Week 1
LeanVibe Agent Hive 2.0 - Migration Assistant

This module helps migrate from fragmented communication implementations
to the unified communication protocol. It provides:
- Compatibility adapters for existing code
- Migration analysis and recommendations
- Performance comparison tools
- Gradual migration support

MIGRATION TARGET COMPONENTS:
1. Redis implementations (6 files) → UnifiedRedisClient
2. Message models (10+ formats) → StandardUniversalMessage
3. Communication managers (5 files) → UnifiedCommunicationManager
4. Protocol adapters → Built-in protocol selection

MIGRATION STRATEGY:
- Phase 1: Add compatibility layer (maintain existing APIs)
- Phase 2: Gradual migration with performance monitoring
- Phase 3: Deprecate old implementations
- Phase 4: Remove legacy code
"""

import asyncio
import inspect
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, Callable

import structlog

from .unified_communication_protocol import (
    UnifiedCommunicationManager,
    StandardUniversalMessage,
    MessageType,
    MessagePriority,
    DeliveryGuarantee,
    get_communication_manager
)

logger = structlog.get_logger("communication_migration")

# ================================================================================
# Legacy Component Detection and Analysis
# ================================================================================

class LegacyComponentAnalyzer:
    """Analyzes existing codebase for legacy communication components."""
    
    LEGACY_IMPORTS = [
        "from .redis import",
        "from .optimized_redis import",
        "from .redis_integration import",
        "from .redis_pubsub_manager import",
        "from .enhanced_redis_streams_manager import",
        "from .team_coordination_redis import",
        "from .communication_manager import",
        "from .communication import",
        "from .communication_hub.communication_hub import",
        "from .message_processor import",
        "from .messaging_service import",
        "from .agent_messaging_service import",
        "from .agent_communication_service import"
    ]
    
    LEGACY_PATTERNS = [
        r"redis\.asyncio",
        r"RedisStreamMessage",
        r"UniversalMessage",
        r"CLIMessage",
        r"BridgeConnection",
        r"CommunicationBridge",
        r"CommunicationHub",
        r"MessagingService"
    ]
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.analysis_results = {
            "legacy_files": [],
            "migration_candidates": [],
            "high_priority_files": [],
            "estimated_effort": {},
            "compatibility_issues": []
        }
    
    async def analyze_project(self) -> Dict[str, Any]:
        """Analyze entire project for legacy communication usage."""
        logger.info("Starting legacy communication analysis")
        
        # Scan all Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                await self._analyze_file(file_path)
            except Exception as e:
                logger.error("Failed to analyze file", file=str(file_path), error=str(e))
        
        # Generate migration recommendations
        self._generate_migration_plan()
        
        logger.info("Legacy communication analysis complete", 
                   legacy_files=len(self.analysis_results["legacy_files"]),
                   migration_candidates=len(self.analysis_results["migration_candidates"]))
        
        return self.analysis_results
    
    async def _analyze_file(self, file_path: Path):
        """Analyze individual file for legacy usage."""
        try:
            content = file_path.read_text()
            
            # Check for legacy imports
            legacy_imports = []
            for legacy_import in self.LEGACY_IMPORTS:
                if legacy_import in content:
                    legacy_imports.append(legacy_import)
            
            if legacy_imports:
                file_info = {
                    "file_path": str(file_path),
                    "legacy_imports": legacy_imports,
                    "lines_of_code": len(content.split('\n')),
                    "complexity_score": self._calculate_complexity_score(content),
                    "migration_priority": self._calculate_migration_priority(content)
                }
                
                self.analysis_results["legacy_files"].append(file_info)
                
                if file_info["migration_priority"] >= 7:
                    self.analysis_results["high_priority_files"].append(file_info)
                
                if file_info["complexity_score"] < 5:  # Easy to migrate
                    self.analysis_results["migration_candidates"].append(file_info)
        
        except Exception as e:
            logger.error("Error analyzing file", file=str(file_path), error=str(e))
    
    def _calculate_complexity_score(self, content: str) -> int:
        """Calculate migration complexity (1-10 scale)."""
        score = 1
        
        # Add complexity for multiple legacy components
        legacy_count = sum(1 for pattern in self.LEGACY_IMPORTS if pattern in content)
        score += min(legacy_count, 5)
        
        # Add complexity for custom implementations
        if "class" in content and ("Redis" in content or "Message" in content):
            score += 2
        
        # Add complexity for async patterns
        if "async def" in content and "await" in content:
            score += 1
        
        # Add complexity for error handling
        if "try:" in content and "except" in content:
            score += 1
        
        return min(score, 10)
    
    def _calculate_migration_priority(self, content: str) -> int:
        """Calculate migration priority (1-10 scale)."""
        priority = 1
        
        # High priority for core orchestrator files
        if "orchestrator" in content.lower():
            priority += 4
        
        # High priority for manager files
        if "manager" in content.lower():
            priority += 3
        
        # High priority for frequently used components
        if "async def send" in content or "async def publish" in content:
            priority += 3
        
        # High priority for performance-critical paths
        if "performance" in content.lower() or "optimization" in content.lower():
            priority += 2
        
        # High priority for files with many dependencies
        import_count = content.count("import ")
        priority += min(import_count // 5, 2)
        
        return min(priority, 10)
    
    def _generate_migration_plan(self):
        """Generate detailed migration plan."""
        # Sort files by priority and complexity
        high_priority = sorted(
            self.analysis_results["high_priority_files"],
            key=lambda x: (x["migration_priority"], -x["complexity_score"]),
            reverse=True
        )
        
        easy_migrations = sorted(
            self.analysis_results["migration_candidates"],
            key=lambda x: x["complexity_score"]
        )
        
        # Estimate effort
        total_files = len(self.analysis_results["legacy_files"])
        total_loc = sum(f["lines_of_code"] for f in self.analysis_results["legacy_files"])
        
        self.analysis_results["estimated_effort"] = {
            "total_files": total_files,
            "total_lines_of_code": total_loc,
            "estimated_hours": total_loc // 50,  # Rough estimate
            "phase_1_files": len(high_priority),
            "phase_2_files": len(easy_migrations),
            "phase_3_files": total_files - len(high_priority) - len(easy_migrations)
        }

# ================================================================================
# Compatibility Layer for Legacy Code
# ================================================================================

class LegacyRedisAdapter:
    """
    Compatibility adapter for legacy Redis implementations.
    Provides familiar API while using unified Redis client underneath.
    """
    
    def __init__(self):
        self._unified_manager: Optional[UnifiedCommunicationManager] = None
    
    async def _get_manager(self) -> UnifiedCommunicationManager:
        """Get unified communication manager."""
        if self._unified_manager is None:
            self._unified_manager = await get_communication_manager()
        return self._unified_manager
    
    # Legacy redis.py compatibility
    async def publish(self, channel: str, message: str) -> int:
        """Legacy publish method compatibility."""
        manager = await self._get_manager()
        
        # Convert legacy message to unified format
        unified_message = StandardUniversalMessage(
            message_type=MessageType.SYSTEM_EVENT,
            priority=MessagePriority.NORMAL,
            payload={"legacy_message": message},
            metadata={"legacy_channel": channel, "migration_source": "redis.py"}
        )
        
        success = await manager.send_message(unified_message)
        return 1 if success else 0
    
    async def xadd(self, stream: str, fields: Dict[str, Any]) -> str:
        """Legacy stream add compatibility."""
        manager = await self._get_manager()
        
        # Convert legacy stream message to unified format
        unified_message = StandardUniversalMessage(
            message_type=MessageType.TASK_REQUEST,
            priority=MessagePriority.NORMAL,
            payload=fields,
            metadata={"legacy_stream": stream, "migration_source": "redis_streams"}
        )
        
        # Use unified Redis client
        stream_id = await manager.redis_client.send_stream_message(stream, unified_message)
        return stream_id or f"{int(time.time() * 1000)}-0"

class LegacyMessageAdapter:
    """Adapter for legacy message formats."""
    
    @staticmethod
    def from_redis_stream_message(legacy_message: Any) -> StandardUniversalMessage:
        """Convert RedisStreamMessage to StandardUniversalMessage."""
        return StandardUniversalMessage(
            message_id=getattr(legacy_message, 'message_id', str(time.time())),
            from_agent=getattr(legacy_message, 'from_agent', 'unknown'),
            to_agent=getattr(legacy_message, 'to_agent', 'broadcast'),
            message_type=MessageType.TASK_REQUEST,
            priority=MessagePriority.NORMAL,
            payload=getattr(legacy_message, 'fields', {}),
            metadata={"migration_source": "RedisStreamMessage"}
        )
    
    @staticmethod
    def from_universal_message(legacy_message: Any) -> StandardUniversalMessage:
        """Convert legacy UniversalMessage to StandardUniversalMessage."""
        return StandardUniversalMessage(
            message_id=getattr(legacy_message, 'id', str(time.time())),
            from_agent=getattr(legacy_message, 'source', 'unknown'),
            to_agent=getattr(legacy_message, 'target', 'broadcast'),
            message_type=MessageType(getattr(legacy_message, 'type', 'task_request')),
            priority=MessagePriority(getattr(legacy_message, 'priority', 'normal')),
            payload=getattr(legacy_message, 'data', {}),
            metadata={"migration_source": "UniversalMessage"}
        )
    
    @staticmethod
    def from_cli_message(legacy_message: Any) -> StandardUniversalMessage:
        """Convert legacy CLIMessage to StandardUniversalMessage."""
        return StandardUniversalMessage(
            message_id=getattr(legacy_message, 'message_id', str(time.time())),
            from_agent=getattr(legacy_message, 'cli_source', 'cli_agent'),
            to_agent=getattr(legacy_message, 'target_agent', 'system'),
            message_type=MessageType.TASK_REQUEST,
            priority=MessagePriority.NORMAL,
            payload=getattr(legacy_message, 'payload', {}),
            metadata={"migration_source": "CLIMessage"}
        )

# ================================================================================
# Migration Performance Monitor
# ================================================================================

class MigrationPerformanceMonitor:
    """Monitor performance during migration from legacy to unified system."""
    
    def __init__(self):
        self.legacy_metrics = {
            "message_count": 0,
            "avg_latency_ms": 0.0,
            "error_count": 0,
            "start_time": time.time()
        }
        
        self.unified_metrics = {
            "message_count": 0,
            "avg_latency_ms": 0.0,
            "error_count": 0,
            "start_time": time.time()
        }
    
    def record_legacy_operation(self, latency_ms: float, success: bool):
        """Record legacy system operation."""
        self.legacy_metrics["message_count"] += 1
        
        if success:
            # Update running average
            current_avg = self.legacy_metrics["avg_latency_ms"]
            count = self.legacy_metrics["message_count"]
            new_avg = ((current_avg * (count - 1)) + latency_ms) / count
            self.legacy_metrics["avg_latency_ms"] = new_avg
        else:
            self.legacy_metrics["error_count"] += 1
    
    def record_unified_operation(self, latency_ms: float, success: bool):
        """Record unified system operation."""
        self.unified_metrics["message_count"] += 1
        
        if success:
            # Update running average
            current_avg = self.unified_metrics["avg_latency_ms"]
            count = self.unified_metrics["message_count"]
            new_avg = ((current_avg * (count - 1)) + latency_ms) / count
            self.unified_metrics["avg_latency_ms"] = new_avg
        else:
            self.unified_metrics["error_count"] += 1
    
    def get_comparison_report(self) -> Dict[str, Any]:
        """Get performance comparison between legacy and unified systems."""
        legacy_runtime = time.time() - self.legacy_metrics["start_time"]
        unified_runtime = time.time() - self.unified_metrics["start_time"]
        
        return {
            "legacy_system": {
                **self.legacy_metrics,
                "runtime_seconds": legacy_runtime,
                "throughput_msg_per_sec": self.legacy_metrics["message_count"] / max(legacy_runtime, 1),
                "error_rate": self.legacy_metrics["error_count"] / max(self.legacy_metrics["message_count"], 1)
            },
            "unified_system": {
                **self.unified_metrics,
                "runtime_seconds": unified_runtime,
                "throughput_msg_per_sec": self.unified_metrics["message_count"] / max(unified_runtime, 1),
                "error_rate": self.unified_metrics["error_count"] / max(self.unified_metrics["message_count"], 1)
            },
            "improvement": {
                "latency_improvement_percent": (
                    (self.legacy_metrics["avg_latency_ms"] - self.unified_metrics["avg_latency_ms"]) /
                    max(self.legacy_metrics["avg_latency_ms"], 0.001) * 100
                ),
                "throughput_improvement_percent": (
                    ((self.unified_metrics["message_count"] / max(unified_runtime, 1)) -
                     (self.legacy_metrics["message_count"] / max(legacy_runtime, 1))) /
                    max(self.legacy_metrics["message_count"] / max(legacy_runtime, 1), 0.001) * 100
                ),
                "error_reduction_percent": (
                    ((self.legacy_metrics["error_count"] / max(self.legacy_metrics["message_count"], 1)) -
                     (self.unified_metrics["error_count"] / max(self.unified_metrics["message_count"], 1))) * 100
                )
            }
        }

# ================================================================================
# Migration Utilities
# ================================================================================

class MigrationHelper:
    """Helper class for migration operations."""
    
    @staticmethod
    async def run_migration_analysis(project_root: str = ".") -> Dict[str, Any]:
        """Run complete migration analysis."""
        analyzer = LegacyComponentAnalyzer(Path(project_root))
        return await analyzer.analyze_project()
    
    @staticmethod
    def create_compatibility_shim(old_class_name: str, new_implementation: Type) -> Type:
        """Create compatibility shim for old class names."""
        
        class CompatibilityShim(new_implementation):
            """Compatibility shim for legacy code."""
            
            def __init__(self, *args, **kwargs):
                logger.warning(
                    "Using deprecated class - please migrate to unified communication",
                    old_class=old_class_name,
                    new_class=new_implementation.__name__
                )
                super().__init__(*args, **kwargs)
        
        CompatibilityShim.__name__ = old_class_name
        return CompatibilityShim
    
    @staticmethod
    async def validate_migration(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate migration by running test cases."""
        results = {
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        for test_case in test_cases:
            try:
                # Run test case
                if await test_case["test_function"]():
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Test {test_case['name']} failed")
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Test {test_case['name']} error: {str(e)}")
        
        return results

# ================================================================================
# Global Migration Utilities
# ================================================================================

# Global migration monitor for tracking performance
_migration_monitor = MigrationPerformanceMonitor()

def get_migration_monitor() -> MigrationPerformanceMonitor:
    """Get global migration monitor instance."""
    return _migration_monitor

# Convenience function for quick migration
async def quick_migrate_message_send(
    old_send_function: Callable,
    from_agent: str,
    to_agent: str,
    payload: Dict[str, Any]
) -> bool:
    """
    Quick migration helper for message sending functions.
    
    Usage:
    # Old code:
    # await redis_client.publish("channel", json.dumps(data))
    
    # Migrated code:
    # await quick_migrate_message_send(
    #     redis_client.publish, "agent1", "agent2", data
    # )
    """
    start_time = time.time()
    
    try:
        # Try unified system first
        manager = await get_communication_manager()
        message = StandardUniversalMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=MessageType.TASK_REQUEST,
            priority=MessagePriority.NORMAL,
            payload=payload
        )
        
        success = await manager.send_message(message)
        latency = (time.time() - start_time) * 1000
        _migration_monitor.record_unified_operation(latency, success)
        
        return success
        
    except Exception as e:
        logger.error("Unified system failed, falling back to legacy", error=str(e))
        
        # Fall back to legacy system
        try:
            await old_send_function("legacy_channel", json.dumps(payload))
            latency = (time.time() - start_time) * 1000
            _migration_monitor.record_legacy_operation(latency, True)
            return True
        except Exception as legacy_error:
            latency = (time.time() - start_time) * 1000
            _migration_monitor.record_legacy_operation(latency, False)
            logger.error("Both unified and legacy systems failed", 
                        unified_error=str(e), legacy_error=str(legacy_error))
            return False