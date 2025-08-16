"""
Quality Gates Framework

This module provides automated quality gates for the Epic 1-4 transformation.
Quality gates ensure that consolidation only proceeds when all safety criteria are met.
"""

__version__ = "1.0.0"
__author__ = "LeanVibe Agent Hive"

from .consolidation_quality_gates import ConsolidationQualityGates

__all__ = [
    "ConsolidationQualityGates"
]