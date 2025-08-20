import asyncio
#!/usr/bin/env python3
"""
LeanVibe Agent Hive 2.0 - Intelligent Error Recovery System
Automatically detects and fixes common system issues
"""

import subprocess
import time
import json
import os
import sys
import socket
from pathlib import Path

class IntelligentRecovery:
    def __init__(self):
        self.project_root = Path("/Users/bogdan/work/leanvibe-dev/bee-hive")
        self.recovery_log = []
        
    def log_action(self, action, status="INFO"):
        """Log recovery actions"""
        log_entry = f"[{status}] {action}"
        self.recovery_log.append(log_entry)
        print(log_entry)
        
    def check_port_availability(self, port):
        """Check if a port is available"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) != 0
            
    def kill_process_on_port(self, port):
        """Kill process using specific port"""
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=10
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    subprocess.run(["kill", "-9", pid], timeout=5)
                    self.log_action(f"Killed process {pid} on port {port}", "SUCCESS")
                return True
        except Exception as e:
            self.log_action(f"Failed to kill process on port {port}: {e}", "ERROR")
        return False
        
    def fix_port_conflicts(self):
        """Resolve port conflicts for key services"""
        ports_to_check = {
            8000: "FastAPI Server",
            5173: "Dashboard",
            5432: "PostgreSQL", 
            6380: "Redis"
        }
        
        conflicts_fixed = 0
        for port, service in ports_to_check.items():
            if not self.check_port_availability(port):
                self.log_action(f"Port conflict detected for {service} (port {port})", "WARNING")
                if self.kill_process_on_port(port):
                    conflicts_fixed += 1
                    time.sleep(2)  # Wait for port to be released
                    
        return conflicts_fixed > 0
        
    def check_docker_status(self):
        """Check if Docker is running"""
        try:
            result = subprocess.run(
                ["docker", "info"], 
                capture_output=True, timeout=10
            )
            return result.returncode == 0
        except:
            return False
            
    def start_docker_services(self):
        """Start required Docker services"""
        try:
            os.chdir(self.project_root)
            result = subprocess.run(
                ["docker", "compose", "up", "-d", "postgres", "redis"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                self.log_action("Docker services started successfully", "SUCCESS")
                return True
            else:
                self.log_action(f"Docker services failed: {result.stderr}", "ERROR")
        except Exception as e:
            self.log_action(f"Error starting Docker services: {e}", "ERROR")
        return False
        
    def check_api_health(self):
        """Check if API server is responding"""
        try:
            result = subprocess.run(
                ["curl", "-sf", "http://localhost:8000/health"],
                capture_output=True, timeout=10
            )
            return result.returncode == 0
        except:
            return False
            
    def start_api_server(self):
        """Start the FastAPI server"""
        try:
            os.chdir(self.project_root)
            # Kill any existing API processes
            self.kill_process_on_port(8000)
            time.sleep(2)
            
            # Start API server in background
            subprocess.Popen(
                ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for startup and verify
            for i in range(30):  # 30 second timeout
                time.sleep(1)
                if self.check_api_health():
                    self.log_action("API server started successfully", "SUCCESS")
                    return True
                    
            self.log_action("API server failed to start within timeout", "ERROR")
            return False
        except Exception as e:
            self.log_action(f"Error starting API server: {e}", "ERROR")
            return False
            
    def check_database_connection(self):
        """Check database connectivity"""
        try:
            result = subprocess.run([
                "psql", 
                "postgresql://postgres:password@localhost:5432/agent_hive",
                "-c", "SELECT 1;"
            ], capture_output=True, timeout=10)
            return result.returncode == 0
        except:
            return False
            
    def run_database_migrations(self):
        """Run database migrations"""
        try:
            os.chdir(self.project_root)
            result = subprocess.run(
                ["python", "-m", "alembic", "upgrade", "head"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                self.log_action("Database migrations completed", "SUCCESS")
                return True
            else:
                self.log_action(f"Migration failed: {result.stderr}", "ERROR")
        except Exception as e:
            self.log_action(f"Migration error: {e}", "ERROR")
        return False
        
    def check_agent_system(self):
        """Check if agents are active"""
        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:8000/api/agents/debug"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                agent_count = len(data.get("agents", []))
                self.log_action(f"Found {agent_count} active agents", "INFO")
                return agent_count >= 3  # Minimum viable agent count
        except Exception as e:
            self.log_action(f"Agent check failed: {e}", "ERROR")
        return False
        
    def activate_agent_team(self):
        """Activate a basic agent team"""
        try:
            payload = {
                "team_size": 5,
                "project_type": "autonomous_testing",
                "specialized_roles": [
                    "product_manager", 
                    "architect", 
                    "backend_developer", 
                    "qa_engineer", 
                    "devops_engineer"
                ]
            }
            
            result = subprocess.run([
                "curl", "-X", "POST", 
                "http://localhost:8000/api/coordination/activate-team",
                "-H", "Content-Type: application/json",
                "-d", json.dumps(payload)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.log_action("Agent team activated successfully", "SUCCESS")
                return True
        except Exception as e:
            self.log_action(f"Agent activation failed: {e}", "ERROR")
        return False
        
    def full_system_recovery(self):
        """Complete system recovery sequence"""
        self.log_action("🚀 Starting intelligent system recovery", "INFO")
        recovery_steps = []
        
        # Step 1: Fix port conflicts
        self.log_action("Step 1: Checking port conflicts", "INFO")
        if self.fix_port_conflicts():
            recovery_steps.append("✅ Port conflicts resolved")
        else:
            recovery_steps.append("ℹ️ No port conflicts detected")
            
        # Step 2: Ensure Docker is running
        self.log_action("Step 2: Checking Docker services", "INFO")
        if not self.check_docker_status():
            self.log_action("Docker not running, attempting to start services", "WARNING")
            if self.start_docker_services():
                recovery_steps.append("✅ Docker services started")
            else:
                recovery_steps.append("❌ Docker services failed")
                return False
        else:
            if self.start_docker_services():  # Ensure services are up
                recovery_steps.append("✅ Docker services verified")
                
        # Step 3: Database connectivity and migrations
        self.log_action("Step 3: Checking database", "INFO")
        time.sleep(5)  # Wait for DB startup
        if not self.check_database_connection():
            self.log_action("Database connection failed", "ERROR")
            recovery_steps.append("❌ Database connection failed")
        else:
            if self.run_database_migrations():
                recovery_steps.append("✅ Database migrations completed")
            else:
                recovery_steps.append("⚠️ Database migrations failed")
                
        # Step 4: Start API server
        self.log_action("Step 4: Starting API server", "INFO")
        if not self.check_api_health():
            if self.start_api_server():
                recovery_steps.append("✅ API server started")
            else:
                recovery_steps.append("❌ API server failed")
                return False
        else:
            recovery_steps.append("✅ API server already running")
            
        # Step 5: Activate agent system
        self.log_action("Step 5: Checking agent system", "INFO")
        if not self.check_agent_system():
            if self.activate_agent_team():
                recovery_steps.append("✅ Agent team activated")
            else:
                recovery_steps.append("⚠️ Agent activation failed")
        else:
            recovery_steps.append("✅ Agents already active")
            
        # Final status
        self.log_action("🎉 System recovery completed", "SUCCESS")
        return True
        
    def get_system_status(self):
        """Get comprehensive system status"""
        status = {
            "api_healthy": self.check_api_health(),
            "docker_running": self.check_docker_status(),
            "database_connected": self.check_database_connection(),
            "agents_active": self.check_agent_system(),
            "ports_clear": all(self.check_port_availability(p) for p in [8000, 5173]),
            "recovery_log": self.recovery_log
        }
        return status

def main():
    """Main recovery execution"""
    recovery = IntelligentRecovery()
    
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        # Just check status
        status = recovery.get_system_status()
        print("🔍 LeanVibe System Status:")
        print(f"   API Health: {'✅' if status['api_healthy'] else '❌'}")
        print(f"   Docker: {'✅' if status['docker_running'] else '❌'}")
        print(f"   Database: {'✅' if status['database_connected'] else '❌'}")
        print(f"   Agents: {'✅' if status['agents_active'] else '❌'}")
        print(f"   Ports: {'✅' if status['ports_clear'] else '❌'}")
        return status
    else:
        # Full recovery
        success = recovery.full_system_recovery()
        final_status = recovery.get_system_status()
        
        print("\n" + "="*60)
        print("🎯 Recovery Summary:")
        for log_entry in recovery.recovery_log:
            print(f"   {log_entry}")
            
        print(f"\n🚀 System Status: {'✅ OPERATIONAL' if success else '❌ NEEDS ATTENTION'}")
        return success

if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class IntelligentRecoveryScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            main()
            
            return {"status": "completed"}
    
    script_main(IntelligentRecoveryScript)