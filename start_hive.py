#!/usr/bin/env python3
"""
Simple Hive Startup Script
Gets LeanVibe Agent Hive 2.0 operational with error recovery
"""

import subprocess
import time
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, timeout=30, check_output=False):
    """Run command with timeout and error handling"""
    try:
        if check_output:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=True, timeout=timeout)
            return result.returncode == 0, "", ""
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)

def check_docker_services():
    """Check if Docker services are running"""
    logger.info("🐳 Checking Docker services...")
    
    success, stdout, stderr = run_command(
        "docker ps --format '{{.Names}}\t{{.Status}}' | grep -E '(postgres|redis)'",
        check_output=True
    )
    
    if success and stdout:
        services = [line for line in stdout.strip().split('\n') if line]
        logger.info(f"✅ Found {len(services)} Docker services:")
        for service in services:
            logger.info(f"   {service}")
        return len(services) >= 2
    else:
        logger.warning("❌ Required Docker services not found")
        return False

def start_docker_services():
    """Start Docker services if needed"""
    if not check_docker_services():
        logger.info("🚀 Starting Docker services...")
        success, _, stderr = run_command(
            "docker compose -f docker-compose.fast.yml up -d postgres redis", 
            timeout=60
        )
        if success:
            logger.info("✅ Docker services started, waiting for health checks...")
            time.sleep(10)
            return check_docker_services()
        else:
            logger.error(f"❌ Failed to start Docker services: {stderr}")
            return False
    return True

def fix_database_enums():
    """Fix database enum casting issues"""
    logger.info("🔧 Fixing database enum issues...")
    
    # SQL to fix the enum casting issue
    fix_sql = """
    DO $$
    BEGIN
        -- Create a function to cast text to taskstatus enum safely
        CREATE OR REPLACE FUNCTION cast_to_taskstatus(text_value text)
        RETURNS taskstatus AS $func$
        BEGIN
            CASE text_value
                WHEN 'PENDING' THEN RETURN 'PENDING'::taskstatus;
                WHEN 'IN_PROGRESS' THEN RETURN 'IN_PROGRESS'::taskstatus;
                WHEN 'COMPLETED' THEN RETURN 'COMPLETED'::taskstatus;
                WHEN 'FAILED' THEN RETURN 'FAILED'::taskstatus;
                WHEN 'CANCELLED' THEN RETURN 'CANCELLED'::taskstatus;
                ELSE RETURN 'PENDING'::taskstatus;
            END CASE;
        END;
        $func$ LANGUAGE plpgsql;
        
        EXCEPTION WHEN OTHERS THEN
            -- Ignore errors if enum doesn't exist yet
            NULL;
    END $$;
    """
    
    # Try to run the SQL fix
    success, _, stderr = run_command(
        f'docker exec leanvibe_postgres_fast psql -U leanvibe_user -d leanvibe_agent_hive -c "{fix_sql}"',
        timeout=10,
        check_output=True
    )
    
    if success:
        logger.info("✅ Database enum issues fixed")
    else:
        logger.warning(f"⚠️ Could not fix database enums (may not be needed): {stderr}")
    
    return True

def start_api_server():
    """Start the FastAPI server in background"""
    logger.info("🌐 Starting API server...")
    
    # Kill any existing processes
    run_command("pkill -f 'uvicorn.*app.main:app' || true")
    time.sleep(2)
    
    # Start server in background
    success, _, stderr = run_command(
        "nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level warning > api_server.log 2>&1 &",
        timeout=5
    )
    
    if success:
        logger.info("🔄 Waiting for server startup...")
        
        # Wait for server to become responsive
        for i in range(20):
            time.sleep(1)
            success, _, _ = run_command(
                "curl -sf http://localhost:8000/health > /dev/null 2>&1",
                timeout=3
            )
            if success:
                logger.info("✅ API server is running!")
                return True
        
        logger.error("❌ API server failed to start properly")
        return False
    else:
        logger.error(f"❌ Failed to start API server: {stderr}")
        return False

def get_system_status():
    """Get system status"""
    logger.info("📊 System Status:")
    
    # Check Docker services
    docker_ok = check_docker_services()
    logger.info(f"   Docker Services: {'✅' if docker_ok else '❌'}")
    
    # Check API health
    api_ok, stdout, _ = run_command(
        "curl -s http://localhost:8000/health",
        timeout=5,
        check_output=True
    )
    
    if api_ok and stdout:
        try:
            health_data = json.loads(stdout)
            status = health_data.get('status', 'unknown')
            logger.info(f"   API Status: ✅ {status}")
            
            # Show component status
            components = health_data.get('components', {})
            for name, info in components.items():
                comp_status = info.get('status', 'unknown')
                icon = '✅' if comp_status == 'healthy' else '⚠️' if comp_status == 'degraded' else '❌'
                logger.info(f"   {name}: {icon} {comp_status}")
                
        except:
            logger.info("   API Status: ✅ responding")
    else:
        logger.info("   API Status: ❌ not responding")
    
    # Check debug endpoint
    debug_ok, stdout, _ = run_command(
        "curl -s http://localhost:8000/debug-agents",
        timeout=3,
        check_output=True
    )
    
    if debug_ok and stdout:
        try:
            debug_data = json.loads(stdout)
            agent_count = debug_data.get('agent_count', 0)
            logger.info(f"   Active Agents: {agent_count}")
        except:
            logger.info("   Active Agents: unknown")
    
    return docker_ok and api_ok

def main():
    """Main startup sequence"""
    logger.info("🚀 LeanVibe Agent Hive 2.0 Startup")
    logger.info("=" * 50)
    
    # Step 1: Docker services
    if not start_docker_services():
        logger.error("❌ Failed to start Docker services")
        return 1
    
    # Step 2: Fix database issues
    fix_database_enums()
    
    # Step 3: Start API server
    if not start_api_server():
        logger.error("❌ Failed to start API server")
        return 1
    
    # Step 4: System status
    logger.info("=" * 50)
    system_ok = get_system_status()
    
    if system_ok:
        logger.info("=" * 50)
        logger.info("🎉 LeanVibe Agent Hive 2.0 is OPERATIONAL!")
        logger.info("")
        logger.info("📱 Access points:")
        logger.info("   • API: http://localhost:8000")
        logger.info("   • Health: http://localhost:8000/health")
        logger.info("   • Docs: http://localhost:8000/docs")
        logger.info("")
        logger.info("🧪 Test with: python scripts/simple_test.py")
        return 0
    else:
        logger.error("❌ System startup incomplete")
        return 1

if __name__ == "__main__":
    sys.exit(main())