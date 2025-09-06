"""
Standardized Script Base for LeanVibe Agent Hive 2.0

This module provides a standardized pattern for script execution,
eliminating the need for boilerplate main() patterns across the codebase.

Usage:
    from app.common.script_base import ScriptBase
    
    class MyScript(ScriptBase):
        async def run(self):
            # Your script logic here
            return {"status": "success", "data": "result"}
            
    # At module level:
    script = MyScript()
    
    # Optional: for direct execution
    if __name__ == "__main__":
        script.execute()

Benefits:
- Eliminates ~30 lines of boilerplate per file
- Consistent error handling and logging
- Standardized JSON output format
- Automatic asyncio management
- Consistent CLI interface
"""

import asyncio
import json
import sys
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional
import structlog

logger = structlog.get_logger()


class ScriptBase(ABC):
    """
    Base class for standardized script execution.
    
    Provides consistent patterns for:
    - Async execution management
    - Error handling and logging  
    - JSON output formatting
    - CLI argument processing
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the script base.
        
        Args:
            name: Optional script name (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.started_at = None
        self.completed_at = None
        
    @abstractmethod
    async def run(self) -> Dict[str, Any]:
        """
        Execute the main script logic.
        
        Returns:
            Dict containing script results in standardized format
            
        Should return a dictionary with at least:
        - status: "success" | "error" | "warning"  
        - data: Any results data
        - message: Optional status message
        """
        pass
        
    def execute(self, output_json: bool = True, exit_on_error: bool = True) -> Dict[str, Any]:
        """
        Execute the script with standardized error handling.
        
        Args:
            output_json: Whether to print JSON output to stdout
            exit_on_error: Whether to call sys.exit() on errors
            
        Returns:
            Dict containing execution results
        """
        try:
            self.started_at = datetime.utcnow()
            logger.info(f"🚀 Starting {self.name}...")
            
            # Run the async script
            results = asyncio.run(self._run_with_metrics())
            
            self.completed_at = datetime.utcnow()
            duration = (self.completed_at - self.started_at).total_seconds()
            
            # Add execution metadata
            final_results = {
                **results,
                "script_name": self.name,
                "started_at": self.started_at.isoformat(),
                "completed_at": self.completed_at.isoformat(),
                "duration_seconds": duration,
                "success": results.get("status") == "success"
            }
            
            if output_json:
                print(json.dumps(final_results, indent=2, default=str))
                
            logger.info(
                f"✅ {self.name} completed successfully",
                duration_seconds=duration,
                status=results.get("status", "unknown")
            )
            
            return final_results
            
        except KeyboardInterrupt:
            logger.warning(f"⚠️ {self.name} interrupted by user")
            error_result = {
                "script_name": self.name,
                "status": "interrupted",
                "error": "Script interrupted by user",
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": datetime.utcnow().isoformat(),
                "success": False
            }
            
            if output_json:
                print(json.dumps(error_result, indent=2))
                
            if exit_on_error:
                sys.exit(1)
                
            return error_result
            
        except Exception as e:
            self.completed_at = datetime.utcnow()
            duration = (self.completed_at - self.started_at).total_seconds() if self.started_at else 0
            
            logger.error(
                f"❌ {self.name} failed",
                error=str(e),
                duration_seconds=duration,
                traceback=traceback.format_exc()
            )
            
            error_result = {
                "script_name": self.name,
                "status": "error", 
                "error": str(e),
                "traceback": traceback.format_exc(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat(),
                "duration_seconds": duration,
                "success": False
            }
            
            if output_json:
                print(json.dumps(error_result, indent=2))
                
            if exit_on_error:
                sys.exit(1)
                
            return error_result
    
    async def _run_with_metrics(self) -> Dict[str, Any]:
        """Run the script with performance metrics collection."""
        try:
            results = await self.run()
            
            # Ensure results have required fields
            if not isinstance(results, dict):
                results = {"data": results}
                
            if "status" not in results:
                results["status"] = "success"
                
            return results
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }


class SimpleScriptBase(ScriptBase):
    """
    Simplified script base for scripts with sync main functions.
    
    Usage:
        class MyScript(SimpleScriptBase):
            def run_sync(self):
                return {"message": "Hello World"}
                
        script = MyScript()
        if __name__ == "__main__":
            script.execute()
    """
    
    def run_sync(self) -> Dict[str, Any]:
        """
        Synchronous version of run() for simple scripts.
        Override this instead of run() for sync scripts.
        """
        return {"status": "success", "message": "No operation performed"}
    
    async def run(self) -> Dict[str, Any]:
        """Async wrapper around run_sync()."""
        return self.run_sync()


# Utility function for backwards compatibility
def execute_script(script_instance: ScriptBase) -> Dict[str, Any]:
    """
    Execute a script instance with standard error handling.
    
    Args:
        script_instance: Instance of ScriptBase to execute
        
    Returns:
        Dict containing execution results
    """
    return script_instance.execute()


# CLI entry point helper
def cli_main(script_class, *args, **kwargs):
    """
    Standard CLI entry point for scripts.
    
    Usage:
        if __name__ == "__main__":
            from app.common.script_base import cli_main
            cli_main(MyScriptClass)
    """
    script = script_class(*args, **kwargs)
    return script.execute()