"""
Foundation Testing Layer - Testing Pyramid Base Level

This module provides comprehensive foundation testing that validates:
1. Import Resolution Testing - All modules can be imported without errors
2. Configuration Validation Testing - All environment configurations work
3. Model Integrity Testing - Database and Pydantic models validate correctly  
4. Core Dependency Testing - Essential dependencies are functional

Foundation tests are designed to be fast (<30s), reliable, and comprehensive
to provide confidence in basic system integrity before running higher-level tests.
"""

__version__ = "1.0.0"