#!/usr/bin/env python3
"""
Remote Multi-Agent Oversight Demo for LeanVibe Agent Hive 2.0

Creates a comprehensive tmux session showcasing:
- Real-time agent coordination
- Mobile dashboard for remote oversight
- Human-in-the-loop decision points
- Live autonomous development demonstration

This implements the pragmatic vertical slice plan for remote multi-agent oversight.
"""

import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

def run_command(cmd, shell=True, capture_output=False):
    """Run a command with proper error handling."""
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else None
        else:
            subprocess.run(cmd, shell=shell, check=True)
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {e}")
        return False

def check_tmux_session_exists(session_name):
    """Check if a tmux session exists."""
    result = run_command(f"tmux has-session -t {session_name} 2>/dev/null", capture_output=True)
    return result is not None

def create_remote_oversight_session():
    """Create comprehensive tmux session for remote oversight demo."""
    
    session_name = "agent-hive-remote"
    
    print("🚀 Creating LeanVibe Agent Hive Remote Oversight Demo Session...")
    print("=" * 70)
    
    # Kill existing session if it exists
    if check_tmux_session_exists(session_name):
        print(f"🔄 Killing existing session: {session_name}")
        run_command(f"tmux kill-session -t {session_name}")
        time.sleep(2)
    
    # Create new session with main control window
    print("📊 Creating main control center...")
    run_command(f"""tmux new-session -d -s {session_name} -n control \\
        'echo "🎛️ AGENT HIVE REMOTE OVERSIGHT CONTROL CENTER"; \\
         echo "========================================="; \\
         echo ""; \\
         echo "🌐 ACCESS URLS:"; \\
         echo "  📡 API Server: http://localhost:8000"; \\
         echo "  📊 API Docs: http://localhost:8000/docs"; \\
         echo "  🎛️ Dashboard: http://localhost:8000/dashboard"; \\
         echo "  📈 Health: http://localhost:8000/health"; \\
         echo "  📊 Prometheus: http://localhost:9090"; \\
         echo "  📈 Grafana: http://localhost:3001"; \\
         echo ""; \\
         echo "🤖 REMOTE OVERSIGHT COMMANDS:"; \\
         echo "  agent-hive start        # Start platform"; \\
         echo "  agent-hive dashboard    # Open dashboard"; \\
         echo "  agent-hive develop \"task\" # Start autonomous dev"; \\
         echo "  agent-hive demo         # Run full demo"; \\
         echo ""; \\
         echo "📱 MOBILE OVERSIGHT:"; \\
         echo "  Open dashboard URL on mobile device for remote oversight"; \\
         echo "  Real-time agent status, task progress, decision alerts"; \\
         echo ""; \\
         echo "⚡ Quick Status Check:"; \\
         curl -s http://localhost:8000/health | jq -r '\''.status'\'' 2>/dev/null || echo "API not responding"; \\
         echo ""; \\
         echo "🎯 Ready for remote multi-agent oversight!"; \\
         exec bash'""")
    
    # Window 1: API Server with enhanced logging
    print("📡 Setting up API server with detailed logging...")
    run_command(f"""tmux new-window -t {session_name} -n api-server \\
        'echo "📡 Starting LeanVibe Agent Hive API Server..."; \\
         echo "Real-time multi-agent coordination endpoint"; \\
         echo "WebSocket dashboard feeds active"; \\
         source venv/bin/activate && \\
         uvicorn app.main:app --reload --host 0.0.0.0 --port 8000'""")
    
    # Window 2: Dashboard with PWA serving
    print("🎛️ Setting up dashboard with PWA serving...")
    run_command(f"""tmux new-window -t {session_name} -n dashboard \\
        'echo "🎛️ Starting Dashboard & PWA Services..."; \\
         echo ""; \\
         echo "Dashboard Features:"; \\
         echo "✅ Real-time agent status monitoring"; \\
         echo "✅ Live task progress tracking"; \\
         echo "✅ WebSocket feeds for remote oversight"; \\
         echo "✅ Mobile-optimized responsive interface"; \\
         echo "✅ Human-in-the-loop decision alerts"; \\
         echo ""; \\
         echo "Waiting for API server startup..."; \\
         sleep 5; \\
         echo "Opening dashboard in browser..."; \\
         python -c "import webbrowser; webbrowser.open('\''http://localhost:8000/dashboard'\'')" || true; \\
         echo ""; \\
         echo "📱 For mobile access: http://[YOUR_IP]:8000/dashboard"; \\
         echo ""; \\
         echo "Dashboard ready for remote oversight!"; \\
         exec bash'""")
    
    # Window 3: Autonomous Development Engine
    print("🤖 Setting up autonomous development engine...")
    run_command(f"""tmux new-window -t {session_name} -n autonomous \\
        'echo "🤖 Autonomous Development Engine"; \\
         echo "=============================="; \\
         echo ""; \\
         echo "Ready to start autonomous development with remote oversight:"; \\
         echo ""; \\
         echo "Commands:"; \\
         echo "  python scripts/demos/autonomous_development_demo.py"; \\
         echo "  agent-hive develop \"Build user authentication API\""; \\
         echo "  agent-hive demo  # Full showcase"; \\
         echo ""; \\
         echo "Features:"; \\
         echo "✅ Multi-agent coordination"; \\
         echo "✅ Real-time progress on dashboard"; \\
         echo "✅ Human approval checkpoints"; \\
         echo "✅ Mobile notifications for decisions"; \\
         echo "✅ Live code generation and testing"; \\
         echo ""; \\
         echo "Waiting for you to start autonomous development..."; \\
         exec bash'""")
    
    # Window 4: Infrastructure & Monitoring
    print("🏗️ Setting up infrastructure monitoring...")
    run_command(f"""tmux new-window -t {session_name} -n infrastructure \\
        'echo "🏗️ Infrastructure & Monitoring"; \\
         echo "============================="; \\
         echo ""; \\
         echo "Docker Services Status:"; \\
         docker compose ps --format "table {{{{.Name}}}}\\t{{{{.Status}}}}\\t{{{{.Ports}}}}" 2>/dev/null || echo "Docker Compose not available"; \\
         echo ""; \\
         echo "Monitoring URLs:"; \\
         echo "📊 Prometheus: http://localhost:9090"; \\
         echo "📈 Grafana: http://localhost:3001"; \\
         echo ""; \\
         echo "Starting monitoring stack..."; \\
         docker compose --profile monitoring up -d prometheus grafana 2>/dev/null || echo "Monitoring services already running"; \\
         echo ""; \\
         echo "🔄 Continuous Infrastructure Status:"; \\
         while true; do \\
           echo "$(date): Infrastructure Status - $(docker compose ps --format json | jq -r length) services running"; \\
           sleep 30; \\
         done'""")
    
    # Window 5: Live Logs & Events
    print("📜 Setting up live event monitoring...")
    run_command(f"""tmux new-window -t {session_name} -n logs \\
        'echo "📜 Live Agent Hive Events & Logs"; \\
         echo "==============================="; \\
         echo ""; \\
         echo "Real-time system events for remote oversight:"; \\
         echo "- Agent coordination events"; \\
         echo "- Task execution progress"; \\
         echo "- Human decision points"; \\
         echo "- System health updates"; \\
         echo ""; \\
         echo "Waiting for system startup..."; \\
         sleep 8; \\
         echo ""; \\
         echo "🔄 Starting live log feed..."; \\
         docker compose logs -f --tail=50 2>/dev/null || \\
         tail -f /tmp/agent-hive.log 2>/dev/null || \\
         echo "Live logging will appear here when agents are active"'""")
    
    # Window 6: CLI Command Center
    print("⚡ Setting up CLI command center...")
    run_command(f"""tmux new-window -t {session_name} -n cli \\
        'echo "⚡ Agent Hive CLI Command Center"; \\
         echo "=============================="; \\
         echo ""; \\
         echo "Ultimate agent-hive commands for remote oversight:"; \\
         echo ""; \\
         echo "🚀 Platform Management:"; \\
         echo "  agent-hive start     # Start entire platform"; \\
         echo "  agent-hive stop      # Stop all services"; \\
         echo "  agent-hive restart   # Restart platform"; \\
         echo "  agent-hive status    # Show system status"; \\
         echo ""; \\
         echo "🤖 Autonomous Development:"; \\
         echo "  agent-hive develop \"Build auth API\"  # Start development"; \\
         echo "  agent-hive demo                        # Full showcase demo"; \\
         echo ""; \\
         echo "📊 Monitoring:"; \\
         echo "  agent-hive dashboard  # Open dashboard"; \\
         echo "  agent-hive health     # Health check"; \\
         echo "  agent-hive logs       # View logs"; \\
         echo ""; \\
         echo "Ready for commands! Type any agent-hive command above."; \\
         exec bash'""")
    
    return session_name

def open_dashboard_on_mobile():
    """Provide mobile access information."""
    print("\n📱 Mobile Dashboard Access:")
    print("=" * 30)
    
    # Get local IP for mobile access
    local_ip = run_command("hostname -I | awk '{print $1}'", capture_output=True)
    if local_ip:
        mobile_url = f"http://{local_ip}:8000/dashboard"
        print(f"🔗 Mobile URL: {mobile_url}")
        print("📲 Scan QR code or enter URL on mobile device")
        print("🎛️ Full remote oversight capabilities available")
    else:
        print("🔗 Mobile URL: http://localhost:8000/dashboard")
    
    print("\n📊 Desktop Dashboard:")
    print("🖥️  http://localhost:8000/dashboard")

def main():
    """Main function to start the remote oversight demo."""
    
    print("🚀 LeanVibe Agent Hive 2.0 - Remote Multi-Agent Oversight Demo")
    print("=" * 70)
    print()
    print("Creating comprehensive remote oversight environment...")
    print("✅ Real-time agent coordination")
    print("✅ Mobile dashboard for remote management") 
    print("✅ Human-in-the-loop decision points")
    print("✅ Live autonomous development capabilities")
    print()
    
    # Create the tmux session
    session_name = create_remote_oversight_session()
    
    # Wait for services to start
    print("\n⏳ Waiting for services to initialize...")
    time.sleep(5)
    
    # Open dashboard
    print("🖥️  Opening dashboard...")
    try:
        webbrowser.open("http://localhost:8000/dashboard")
    except:
        print("⚠️  Could not auto-open browser. Please visit: http://localhost:8000/dashboard")
    
    # Show mobile access info
    open_dashboard_on_mobile()
    
    # Final instructions
    print("\n🎯 Remote Oversight Demo Ready!")
    print("=" * 40)
    print(f"📺 Tmux Session: {session_name}")
    print("💻 Attach with: tmux attach -t agent-hive-remote")
    print("🎛️  Dashboard: http://localhost:8000/dashboard")
    print("📊 API Docs: http://localhost:8000/docs")
    print()
    print("🤖 To start autonomous development:")
    print("   agent-hive develop \"Build authentication API with JWT\"")
    print("   agent-hive demo  # Full showcase")
    print()
    print("📱 Monitor agents remotely via dashboard on mobile device!")
    print("✨ LeanVibe Agent Hive remote oversight is ready!")

if __name__ == "__main__":
    main()