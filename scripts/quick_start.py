import asyncio
#!/usr/bin/env python3
"""
Quick Start Script for LeanVibe Agent Hive 2.0
Ensures all services are operational with intelligent recovery

REFACTORED: Phase 1.1 Technical Debt Remediation - Using shared patterns to eliminate main() duplication
"""

import subprocess
import time
import sys
import json
from pathlib import Path

# Import shared patterns to eliminate main() function duplication
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.common.utilities.shared_patterns import (
    BaseScript, ScriptConfig, ExecutionMode, simple_main_wrapper
)

class HiveQuickStart(BaseScript):
    """LeanVibe Agent Hive 2.0 Quick Start - Refactored to use shared patterns."""
    
    def __init__(self, config: ScriptConfig):
        super().__init__(config)
        self.base_path = Path(__file__).parent.parent
        
    def run_command(self, cmd, timeout=30):
        """Run shell command with timeout"""
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=self.base_path
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out: {cmd}")
            return False, "", "Timeout"
        except Exception as e:
            self.logger.error(f"Command failed: {cmd} - {e}")
            return False, "", str(e)
    
    def check_docker_services(self):
        """Check if PostgreSQL and Redis containers are running"""
        self.logger.info("🐳 Checking Docker services...")
        
        success, stdout, stderr = self.run_command(
            "docker ps --format '{{.Names}}\t{{.Status}}' | grep -E '(postgres|redis)'"
        )
        
        if success and stdout:
            services = stdout.strip().split('\n')
            self.logger.info(f"✅ Found {len(services)} Docker services running")
            for service in services:
                self.logger.info(f"  - {service}")
            return len(services) >= 2
        else:
            self.logger.warning("❌ Docker services not found or not running")
            return False
    
    def start_docker_services(self):
        """Start required Docker services"""
        self.logger.info("🚀 Starting Docker services...")
        
        success, stdout, stderr = self.run_command(
            "docker compose up -d postgres redis", 
            timeout=60
        )
        
        if success:
            self.logger.info("✅ Docker services started")
            time.sleep(5)  # Wait for services to initialize
            return True
        else:
            self.logger.error(f"❌ Failed to start Docker services: {stderr}")
            return False
    
    def check_api_health(self, timeout=10):
        """Check if API server is responding"""
        self.logger.info("🏥 Checking API health...")
        
        success, stdout, stderr = self.run_command(
            f"timeout {timeout} curl -sf http://localhost:8000/health",
            timeout=timeout + 5
        )
        
        if success:
            self.logger.info("✅ API server is healthy")
            return True
        else:
            self.logger.warning("❌ API server not responding")
            return False
    
    def start_api_server(self):
        """Start the FastAPI server"""
        self.logger.info("🌐 Starting API server...")
        
        # Kill any existing processes
        self.run_command("pkill -f 'uvicorn.*app.main:app' || true")
        time.sleep(2)
        
        # Start new server
        success, stdout, stderr = self.run_command(
            "python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > api_server.log 2>&1 &"
        )
        
        # Wait for startup
        self.logger.info("🔄 Waiting for API server startup...")
        for i in range(30):
            if self.check_api_health(timeout=5):
                self.logger.info("✅ API server started successfully")
                return True
            time.sleep(1)
        
        self.logger.error("❌ API server failed to start within 30 seconds")
        return False
    
    def get_system_status(self):
        """Get comprehensive system status"""
        status = {
            "docker_services": self.check_docker_services(),
            "api_health": self.check_api_health(timeout=5),
            "timestamp": time.time()
        }
        
        # Get agent count
        success, stdout, stderr = self.run_command(
            "timeout 3 curl -s http://localhost:8000/debug-agents 2>/dev/null"
        )
        
        if success and stdout:
            try:
                data = json.loads(stdout)
                status["agent_count"] = data.get("agent_count", 0)
            except:
                status["agent_count"] = 0
        else:
            status["agent_count"] = 0
        
        return status
    
    def run_full_startup(self):
        """Run complete system startup sequence"""
        self.logger.info("🚀 Starting LeanVibe Agent Hive 2.0...")
        
        # Step 1: Check/Start Docker services
        if not self.check_docker_services():
            if not self.start_docker_services():
                self.logger.error("❌ Failed to start Docker services")
                return False
        
        # Step 2: Check/Start API server
        if not self.check_api_health():
            if not self.start_api_server():
                self.logger.error("❌ Failed to start API server")
                return False
        
        # Step 3: Final status check
        status = self.get_system_status()
        
        self.logger.info("📊 System Status:")
        self.logger.info(f"  - Docker Services: {'✅' if status['docker_services'] else '❌'}")
        self.logger.info(f"  - API Health: {'✅' if status['api_health'] else '❌'}")
        self.logger.info(f"  - Agent Count: {status['agent_count']}")
        
        if status["docker_services"] and status["api_health"]:
            self.logger.info("🎉 LeanVibe Agent Hive 2.0 is fully operational!")
            return True
        else:
            self.logger.error("❌ System startup incomplete")
            return False
    
    def execute(self) -> dict:
        """
        Execute the quick start logic based on command line arguments.
        
        REFACTORED: Eliminates duplicated main() function pattern using shared utilities.
        This replaces the old main() function with standardized execution logic.
        """
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
        else:
            command = "start"
        
        if command == "status":
            status = self.get_system_status()
            print(json.dumps(status, indent=2))
            return {"command": "status", "status": status, "success": True}
            
        elif command == "start":
            success = self.run_full_startup()
            return {
                "command": "start", 
                "success": success,
                "exit_code": 0 if success else 1
            }
            
        elif command == "health":
            healthy = self.check_api_health()
            return {
                "command": "health",
                "healthy": healthy,
                "success": healthy,
                "exit_code": 0 if healthy else 1
            }
        else:
            self.logger.info("Usage: python quick_start.py [start|status|health]")
            return {"command": "invalid", "success": False, "exit_code": 1}


def main():
    """Legacy main function - kept for backward compatibility."""
    # Use simple wrapper for basic compatibility
    def legacy_main():
        config = ScriptConfig(
            name="hive_quick_start",
            description="LeanVibe Agent Hive 2.0 Quick Start",
            enable_logging=True,
            log_level="INFO"
        )
        
        quick_start = HiveQuickStart(config)
        result = quick_start.run()
        
        # Handle exit codes for backward compatibility
        if result.success:
            exit_code = result.data.get("exit_code", 0)
            sys.exit(exit_code)
        else:
            sys.exit(1)
    
    simple_main_wrapper(legacy_main, "hive_quick_start")


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class QuickStartScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            main()
            
            return {"status": "completed"}
    
    script_main(QuickStartScript)