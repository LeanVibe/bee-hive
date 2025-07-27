"""Self-Modification Engine Core Components."""

from .code_analysis_engine import CodeAnalysisEngine, ProjectAnalysis, FileAnalysis, CodePattern
from .modification_generator import ModificationGenerator, ModificationContext
from .sandbox_environment import SandboxEnvironment, SandboxResult, ResourceLimits, SecurityPolicy
from .version_control_manager import VersionControlManager, CommitInfo, BranchInfo
from .safety_validator import SafetyValidator, SafetyValidationReport, SecurityIssue
from .performance_monitor import PerformanceMonitor, PerformanceMetric, BenchmarkResult
from .self_modification_service import SelfModificationService

__all__ = [
    "CodeAnalysisEngine",
    "ProjectAnalysis", 
    "FileAnalysis",
    "CodePattern",
    "ModificationGenerator",
    "ModificationContext",
    "SandboxEnvironment",
    "SandboxResult",
    "ResourceLimits",
    "SecurityPolicy",
    "VersionControlManager", 
    "CommitInfo",
    "BranchInfo",
    "SafetyValidator",
    "SafetyValidationReport",
    "SecurityIssue",
    "PerformanceMonitor",
    "PerformanceMetric",
    "BenchmarkResult",
    "SelfModificationService",
]