#!/usr/bin/env python3
"""
Quick Start Script for LeanVibe Agent Hive 2.0
Ensures all services are operational with intelligent recovery
"""

import asyncio
import subprocess
import time
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HiveQuickStart:
    def __init__(self):
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
            logger.error(f"Command timed out: {cmd}")
            return False, "", "Timeout"
        except Exception as e:
            logger.error(f"Command failed: {cmd} - {e}")
            return False, "", str(e)
    
    def check_docker_services(self):
        """Check if PostgreSQL and Redis containers are running"""
        logger.info("🐳 Checking Docker services...")
        
        success, stdout, stderr = self.run_command(
            "docker ps --format '{{.Names}}\t{{.Status}}' | grep -E '(postgres|redis)'"
        )
        
        if success and stdout:
            services = stdout.strip().split('\n')
            logger.info(f"✅ Found {len(services)} Docker services running")
            for service in services:
                logger.info(f"  - {service}")
            return len(services) >= 2
        else:
            logger.warning("❌ Docker services not found or not running")
            return False
    
    def start_docker_services(self):
        """Start required Docker services"""
        logger.info("🚀 Starting Docker services...")
        
        success, stdout, stderr = self.run_command(
            "docker compose up -d postgres redis", 
            timeout=60
        )
        
        if success:
            logger.info("✅ Docker services started")
            time.sleep(5)  # Wait for services to initialize
            return True
        else:
            logger.error(f"❌ Failed to start Docker services: {stderr}")
            return False
    
    def check_api_health(self, timeout=10):
        """Check if API server is responding"""
        logger.info("🏥 Checking API health...")
        
        success, stdout, stderr = self.run_command(
            f"timeout {timeout} curl -sf http://localhost:8000/health",
            timeout=timeout + 5
        )
        
        if success:
            logger.info("✅ API server is healthy")
            return True
        else:
            logger.warning("❌ API server not responding")
            return False
    
    def start_api_server(self):
        """Start the FastAPI server"""
        logger.info("🌐 Starting API server...")
        
        # Kill any existing processes
        self.run_command("pkill -f 'uvicorn.*app.main:app' || true")
        time.sleep(2)
        
        # Start new server
        success, stdout, stderr = self.run_command(
            "python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > api_server.log 2>&1 &"
        )
        
        # Wait for startup
        logger.info("🔄 Waiting for API server startup...")
        for i in range(30):
            if self.check_api_health(timeout=5):
                logger.info("✅ API server started successfully")
                return True
            time.sleep(1)
        
        logger.error("❌ API server failed to start within 30 seconds")
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
        logger.info("🚀 Starting LeanVibe Agent Hive 2.0...")
        
        # Step 1: Check/Start Docker services
        if not self.check_docker_services():
            if not self.start_docker_services():
                logger.error("❌ Failed to start Docker services")
                return False
        
        # Step 2: Check/Start API server
        if not self.check_api_health():
            if not self.start_api_server():
                logger.error("❌ Failed to start API server")
                return False
        
        # Step 3: Final status check
        status = self.get_system_status()
        
        logger.info("📊 System Status:")
        logger.info(f"  - Docker Services: {'✅' if status['docker_services'] else '❌'}")
        logger.info(f"  - API Health: {'✅' if status['api_health'] else '❌'}")
        logger.info(f"  - Agent Count: {status['agent_count']}")
        
        if status["docker_services"] and status["api_health"]:
            logger.info("🎉 LeanVibe Agent Hive 2.0 is fully operational!")
            return True
        else:
            logger.error("❌ System startup incomplete")
            return False

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
    else:
        command = "start"
    
    hive = HiveQuickStart()
    
    if command == "status":
        status = hive.get_system_status()
        print(json.dumps(status, indent=2))
    elif command == "start":
        success = hive.run_full_startup()
        sys.exit(0 if success else 1)
    elif command == "health":
        healthy = hive.check_api_health()
        sys.exit(0 if healthy else 1)
    else:
        logger.info("Usage: python quick_start.py [start|status|health]")
        sys.exit(1)

if __name__ == "__main__":
    main()