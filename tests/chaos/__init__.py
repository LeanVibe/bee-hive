"""
Chaos Testing Suite for LeanVibe Agent Hive 2.0 - Phase 5.1 Foundational Reliability

This module provides comprehensive chaos engineering capabilities for validating system resilience
and reliability under various failure conditions. It integrates with VS 3.3 (Error Handling Framework)
and VS 4.3 (DLQ System) to ensure >99.95% availability under all chaos scenarios.

Components:
- chaos_testing_framework: Core chaos infrastructure and utilities
- test_phase_5_1_chaos_scenarios: Main chaos testing suite with 7 scenarios
- performance_under_chaos: Performance validation during failure conditions
- resilience_validation: Availability and recovery time validation
"""

__version__ = "1.0.0"
__all__ = [
    "chaos_testing_framework",
    "test_phase_5_1_chaos_scenarios", 
    "performance_under_chaos",
    "resilience_validation"
]