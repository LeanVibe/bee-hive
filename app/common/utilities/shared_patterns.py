#!/usr/bin/env python3
"""
Shared Pattern Utilities for LeanVibe Agent Hive 2.0

This module provides standardized patterns to eliminate the massive code duplication
across 221+ main() functions and common script patterns throughout the codebase.

Phase 1.1 Implementation of Technical Debt Remediation Plan - ROI: 1283.0
Targeting ~15,000+ lines of duplicate main() function code elimination.
"""

import asyncio
import logging
import sys
import traceback
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class ExecutionMode(str, Enum):
    """Execution mode for scripts."""
    SYNC = "sync"
    ASYNC = "async"


@dataclass
class ScriptConfig:
    """Standard configuration for script execution."""
    name: str
    description: str = ""
    execution_mode: ExecutionMode = ExecutionMode.SYNC
    enable_logging: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    timeout_seconds: Optional[int] = None
    enable_json_output: bool = False
    enable_exit_codes: bool = True


@dataclass  
class ScriptResult:
    """Standard result format for script execution."""
    success: bool
    exit_code: int
    data: Dict[str, Any]
    error: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: Optional[str] = None


class BaseScript(ABC):
    """
    Base class for all scripts with standardized patterns.
    
    This eliminates the 221+ duplicated main() function patterns by providing
    a unified interface for script execution with proper error handling,
    logging, and exit code management.
    """
    
    def __init__(self, config: ScriptConfig):
        self.config = config
        self.start_time = time.time()
        self.logger = self._setup_logging()
        self.result: Optional[ScriptResult] = None
    
    def _setup_logging(self) -> logging.Logger:
        """Setup standardized logging pattern."""
        logger = logging.getLogger(self.config.name)
        
        if self.config.enable_logging:
            # Clear existing handlers
            logger.handlers.clear()
            
            # Set level
            logger.setLevel(getattr(logging, self.config.log_level.upper()))
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler if specified
            if self.config.log_file:
                file_handler = logging.FileHandler(self.config.log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """Execute the main script logic. Must be implemented by subclasses."""
        pass
    
    def run(self) -> ScriptResult:
        """
        Standard script execution pattern.
        
        This replaces the duplicated main() patterns across 221+ files
        with a unified execution framework.
        """
        try:
            self.logger.info(f"🚀 Starting {self.config.name}")
            
            # Execute main logic based on mode
            if self.config.execution_mode == ExecutionMode.ASYNC:
                data = asyncio.run(self._run_async())
            else:
                data = self.execute()
            
            execution_time = time.time() - self.start_time
            
            self.result = ScriptResult(
                success=True,
                exit_code=0,
                data=data or {},
                execution_time=execution_time,
                timestamp=datetime.utcnow().isoformat()
            )
            
            self.logger.info(
                f"✅ {self.config.name} completed successfully "
                f"in {execution_time:.2f}s"
            )
            
        except Exception as e:
            execution_time = time.time() - self.start_time
            error_msg = str(e)
            
            self.result = ScriptResult(
                success=False,
                exit_code=1,
                data={},
                error=error_msg,
                execution_time=execution_time,
                timestamp=datetime.utcnow().isoformat()
            )
            
            self.logger.error(f"❌ {self.config.name} failed: {error_msg}")
            if self.config.log_level.upper() == "DEBUG":
                self.logger.error(traceback.format_exc())
        
        return self.result
    
    async def _run_async(self) -> Dict[str, Any]:
        """Execute async script logic."""
        return await self.execute_async()
    
    async def execute_async(self) -> Dict[str, Any]:
        """Async execution method for async scripts."""
        # Default to sync execute if not overridden
        return self.execute()


def standard_main_wrapper(
    script_class: type,
    config: Optional[ScriptConfig] = None,
    args: Optional[List[str]] = None
) -> None:
    """
    Standard main() function wrapper that eliminates duplication.
    
    This single function replaces 221+ duplicated main() function patterns
    with a standardized approach that handles:
    - Error handling and logging
    - Exit code management
    - JSON output (optional)
    - Command line argument parsing
    - Async/sync execution
    
    Usage:
        if __name__ == "__main__":
            standard_main_wrapper(MyScriptClass)
    """
    try:
        # Create default config if none provided
        if config is None:
            script_name = getattr(script_class, '__name__', 'unknown_script')
            config = ScriptConfig(name=script_name)
        
        # Initialize and run script
        script = script_class(config)
        result = script.run()
        
        # Handle output
        if config.enable_json_output:
            output = {
                'success': result.success,
                'data': result.data,
                'execution_time': result.execution_time,
                'timestamp': result.timestamp
            }
            if result.error:
                output['error'] = result.error
            
            print(json.dumps(output, indent=2))
        
        # Handle exit codes
        if config.enable_exit_codes:
            sys.exit(result.exit_code)
            
    except KeyboardInterrupt:
        print(f"\n⏹️  {config.name if config else 'Script'} interrupted by user")
        sys.exit(130)  # Standard interrupt exit code
    except Exception as e:
        print(f"❌ Fatal error in {config.name if config else 'script'}: {e}")
        sys.exit(1)


def async_main_wrapper(
    main_func: Callable,
    script_name: str = "async_script",
    enable_logging: bool = True
) -> None:
    """
    Async main() function wrapper for async scripts.
    
    Replaces patterns like:
        if __name__ == "__main__":
            import sys
            exit_code = asyncio.run(main())
            sys.exit(exit_code)
    
    With:
        if __name__ == "__main__":
            async_main_wrapper(main, "my_script")
    """
    if enable_logging:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    try:
        start_time = time.time()
        exit_code = asyncio.run(main_func())
        execution_time = time.time() - start_time
        
        if enable_logging:
            logger = logging.getLogger(script_name)
            logger.info(f"✅ {script_name} completed in {execution_time:.2f}s")
        
        sys.exit(exit_code if isinstance(exit_code, int) else 0)
        
    except KeyboardInterrupt:
        if enable_logging:
            logger = logging.getLogger(script_name)
            logger.info(f"⏹️ {script_name} interrupted by user")
        sys.exit(130)
    except Exception as e:
        if enable_logging:
            logger = logging.getLogger(script_name)
            logger.error(f"❌ {script_name} failed: {e}")
        sys.exit(1)


def simple_main_wrapper(main_func: Callable, script_name: str = "script") -> None:
    """
    Simple main() function wrapper for basic scripts.
    
    Replaces the simplest pattern:
        if __name__ == "__main__":
            main()
    
    With:
        if __name__ == "__main__":
            simple_main_wrapper(main, "my_script")
    """
    try:
        main_func()
    except KeyboardInterrupt:
        print(f"\n⏹️  {script_name} interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ {script_name} failed: {e}")
        sys.exit(1)


class StandardArgumentParser:
    """
    Standardized argument parser to eliminate duplicated argparse patterns.
    
    Provides common arguments that many scripts use.
    """
    
    @staticmethod
    def create_parser(
        description: str,
        add_output: bool = True,
        add_config: bool = True,
        add_verbose: bool = True
    ) -> argparse.ArgumentParser:
        """Create standardized argument parser."""
        parser = argparse.ArgumentParser(description=description)
        
        if add_output:
            parser.add_argument(
                '--output', '-o', 
                type=str, 
                help='Output file for results (JSON)'
            )
        
        if add_config:
            parser.add_argument(
                '--config', '-c',
                type=str,
                help='Configuration file path'
            )
        
        if add_verbose:
            parser.add_argument(
                '--verbose', '-v',
                action='store_true',
                help='Enable verbose/debug output'
            )
        
        return parser


# Common initialization patterns
def standard_logging_setup(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Standard logging setup pattern used across many scripts.
    
    Eliminates duplicated logging configuration code.
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def standard_error_handling(
    func: Callable,
    script_name: str,
    logger: Optional[logging.Logger] = None
) -> Any:
    """
    Standard error handling wrapper for common patterns.
    
    Eliminates duplicated try/catch blocks across scripts.
    """
    try:
        return func()
    except KeyboardInterrupt:
        msg = f"⏹️  {script_name} interrupted by user"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        sys.exit(130)
    except Exception as e:
        msg = f"❌ {script_name} failed: {e}"
        if logger:
            logger.error(msg)
            logger.debug(traceback.format_exc())
        else:
            print(msg)
        sys.exit(1)


def get_project_root() -> Path:
    """Get project root directory - common pattern across many scripts."""
    return Path(__file__).parent.parent.parent.parent


def add_project_root_to_path() -> None:
    """Add project root to Python path - common pattern across scripts."""
    project_root = get_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


# Success/failure determination helpers
def determine_exit_code(success_conditions: List[bool]) -> int:
    """Determine exit code based on success conditions."""
    return 0 if all(success_conditions) else 1


def create_success_message(script_name: str, details: Dict[str, Any] = None) -> str:
    """Create standardized success message."""
    msg = f"🎉 {script_name} completed successfully!"
    if details:
        for key, value in details.items():
            msg += f"\n  - {key}: {value}"
    return msg


def create_failure_message(script_name: str, error: str, details: Dict[str, Any] = None) -> str:
    """Create standardized failure message."""
    msg = f"❌ {script_name} failed: {error}"
    if details:
        for key, value in details.items():
            msg += f"\n  - {key}: {value}"
    return msg


# Example usage patterns
class ExampleValidationScript(BaseScript):
    """Example of how to use BaseScript to eliminate main() duplication."""
    
    def execute(self) -> Dict[str, Any]:
        """Execute validation logic."""
        # Your validation logic here
        self.logger.info("Running validation...")
        
        # Simulate some work
        import time
        time.sleep(0.1)
        
        return {
            "tests_passed": 5,
            "tests_failed": 0,
            "validation_status": "success"
        }


# Migration helper functions
def migrate_simple_main(old_main_func: Callable, script_name: str) -> None:
    """Helper to migrate simple main() patterns."""
    simple_main_wrapper(old_main_func, script_name)


def migrate_async_main(old_async_main: Callable, script_name: str) -> None:
    """Helper to migrate async main() patterns."""
    async_main_wrapper(old_async_main, script_name)


if __name__ == "__main__":
    # Example usage
    config = ScriptConfig(
        name="shared_patterns_demo",
        description="Demonstration of shared pattern utilities",
        enable_json_output=True
    )
    
    standard_main_wrapper(ExampleValidationScript, config)