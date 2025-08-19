"""
Shared utility patterns for LeanVibe Agent Hive 2.0

This package contains standardized patterns and utilities for eliminating
code duplication across the entire codebase.

REFACTORED: Phase 1.2 Technical Debt Remediation - Standardized __init__.py pattern
Contains Phase 1.1 main() pattern elimination and Phase 1.2 standardization tools.
"""

# Standard utility imports
from typing import Any, Callable, Dict, List, Optional

# Utility imports - Consolidated patterns from Phase 1.1-1.2
from .shared_patterns import (
    BaseScript, ScriptConfig, ScriptResult, ExecutionMode,
    standard_main_wrapper, async_main_wrapper, simple_main_wrapper,
    StandardArgumentParser, standard_logging_setup, standard_error_handling
)
from .init_file_standardizer import (
    InitFileStandardizer, InitFileType, InitFileMetadata, StandardizedTemplate
)

# Utility exports - Phase 1.1-1.2 Consolidation
__all__ = [
    # Phase 1.1 - Main function pattern elimination
    "BaseScript", "ScriptConfig", "ScriptResult", "ExecutionMode",
    "standard_main_wrapper", "async_main_wrapper", "simple_main_wrapper", 
    "StandardArgumentParser", "standard_logging_setup", "standard_error_handling",
    
    # Phase 1.2 - __init__.py standardization
    "InitFileStandardizer", "InitFileType", "InitFileMetadata", "StandardizedTemplate",
]

# Standard module initialization
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Shared utilities package initialized: {__name__} - Phase 1.1-1.2 patterns loaded")