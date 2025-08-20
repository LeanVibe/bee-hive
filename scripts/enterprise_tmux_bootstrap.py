#!/usr/bin/env python3
"""
Enterprise Tmux Bootstrap Script for LeanVibe Agent Hive 2.0

Replaces the current tmux session with enterprise-grade session management
featuring automatic recovery, process monitoring, and fault tolerance.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.enterprise_tmux_manager import EnterpriseTmuxManager, create_enterprise_services
import structlog

logger = structlog.get_logger()

async def bootstrap_enterprise_session():
    """Bootstrap the enterprise tmux session."""
    
    print("🚀 LeanVibe Agent Hive 2.0 - Enterprise Tmux Bootstrap")
    print("=" * 60)
    
    try:
        # Stop existing session if it exists
        print("🔄 Stopping existing tmux session...")
        import subprocess
        try:
            subprocess.run(["tmux", "kill-session", "-t", "agent-hive"], 
                         capture_output=True, check=False)
            print("✅ Existing session stopped")
        except:
            print("ℹ️ No existing session found")
        
        # Initialize enterprise manager
        print("🏗️ Initializing enterprise tmux manager...")
        manager = EnterpriseTmuxManager(session_name="leanvibe-enterprise")
        await manager.initialize()
        print("✅ Enterprise tmux manager initialized")
        
        # Add enterprise services
        print("📦 Adding enterprise services...")
        services = create_enterprise_services()
        for service_config in services:
            success = await manager.add_service(service_config)
            if success:
                print(f"  ✅ Added service: {service_config.name}")
            else:
                print(f"  ❌ Failed to add service: {service_config.name}")
        
        # Start services with proper dependency order
        print("🚀 Starting services in dependency order...")
        dependency_order = ["infrastructure", "api-server", "observability", "agent-pool", "monitoring"]
        
        for service_name in dependency_order:
            print(f"  🔄 Starting {service_name}...")
            success = await manager.start_service(service_name)
            if success:
                print(f"  ✅ Started {service_name}")
                # Allow service to stabilize
                await asyncio.sleep(3)
            else:
                print(f"  ❌ Failed to start {service_name}")
        
        # Start health monitoring
        print("💚 Starting health monitoring...")
        health_task = asyncio.create_task(manager.start_health_monitoring(check_interval=30))
        print("✅ Health monitoring started")
        
        # Display session information
        print("\n🎯 Enterprise Tmux Session Operational")
        print("=" * 60)
        print(f"Session Name: {manager.session_name}")
        print("Services:")
        
        health_status = await manager.get_all_service_health()
        for service_name, health in health_status.items():
            status_emoji = "✅" if health.status.value == "healthy" else "⚠️" if health.status.value == "degraded" else "❌"
            print(f"  {status_emoji} {service_name}: {health.status.value}")
        
        print("\nAccess URLs:")
        print("  🌐 API Server: http://localhost:8000")
        print("  📊 API Docs: http://localhost:8000/docs")
        print("  📈 Observability: http://localhost:8001/metrics")
        print("  🔍 Grafana: http://localhost:3001")
        print("  📊 Prometheus: http://localhost:9090")
        
        print("\nTmux Commands:")
        print("  📋 Attach: tmux attach-session -t leanvibe-enterprise")
        print("  🔍 List sessions: tmux list-sessions")
        print("  🪟 List windows: tmux list-windows -t leanvibe-enterprise")
        
        print("\n🎉 Enterprise platform ready for autonomous development!")
        print("Press Ctrl+C to gracefully shutdown...")
        
        # Keep running and handle shutdown
        try:
            while True:
                await asyncio.sleep(10)
                # Quick health check
                all_healthy = True
                health_status = await manager.get_all_service_health()
                for service_name, health in health_status.items():
                    if health.status.value not in ["healthy", "degraded"]:
                        all_healthy = False
                        break
                
                if not all_healthy:
                    print("⚠️ Some services are unhealthy - automatic recovery in progress...")
        
        except KeyboardInterrupt:
            print("\n🛑 Received shutdown signal...")
        
        finally:
            print("🔄 Gracefully shutting down enterprise session...")
            health_task.cancel()
            await manager.graceful_shutdown()
            print("✅ Shutdown complete")
        
    except Exception as e:
        print(f"❌ Bootstrap failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


async def quick_status_check():
    """Quick status check of the enterprise session."""
    try:
        manager = EnterpriseTmuxManager(session_name="leanvibe-enterprise")
        await manager.initialize()
        
        print("🔍 Enterprise Tmux Session Status")
        print("=" * 40)
        
        health_status = await manager.get_all_service_health()
        for service_name, health in health_status.items():
            status_emoji = "✅" if health.status.value == "healthy" else "⚠️" if health.status.value == "degraded" else "❌"
            uptime = f"{health.uptime_seconds:.0f}s" if health.uptime_seconds > 0 else "N/A"
            print(f"{status_emoji} {service_name}: {health.status.value} (uptime: {uptime})")
        
        await manager.graceful_shutdown()
        
    except Exception as e:
        print(f"❌ Status check failed: {e}")


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class EnterpriseTmuxBootstrapScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            if len(sys.argv) > 1 and sys.argv[1] == "--status":
            await quick_status_check()
            else:
            await bootstrap_enterprise_session()
            
            return {"status": "completed"}
    
    script_main(EnterpriseTmuxBootstrapScript)