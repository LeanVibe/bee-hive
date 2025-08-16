"""
Consolidation Testing Framework

This module provides testing infrastructure for safely validating
the consolidation of 313 files into 50 modules during Epic 1-4 transformation.

The framework ensures:
1. Functionality preservation across consolidations
2. API compatibility maintenance
3. Performance regression prevention
4. Integration integrity validation
"""

__version__ = "1.0.0"
__author__ = "LeanVibe Agent Hive"

from .consolidation_framework import ConsolidationTestFramework
from .module_compatibility import ModuleCompatibilityTester
from .performance_regression import PerformanceRegressionDetector

__all__ = [
    "ConsolidationTestFramework",
    "ModuleCompatibilityTester", 
    "PerformanceRegressionDetector"
]