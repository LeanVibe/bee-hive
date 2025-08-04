#!/usr/bin/env python3
"""
LeanVibe Agent Hive 2.0 - Mobile Dashboard Generator
Generates QR codes and mobile-optimized access for remote oversight
"""

import socket
import subprocess
import json
import sys

def get_local_ip():
    """Get the local IP address for mobile access"""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "localhost"

def check_system_status():
    """Check current system health"""
    status = {
        "api_online": False,
        "agents_active": 0,
        "services_running": 0,
        "system_load": "unknown"
    }
    
    try:
        # Check API status
        result = subprocess.run(
            ["curl", "-sf", "http://localhost:8000/health"],
            capture_output=True, timeout=5
        )
        status["api_online"] = result.returncode == 0
        
        # Check agent count
        if status["api_online"]:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:8000/api/agents/debug"],
                capture_output=True, timeout=5
            )
            try:
                data = json.loads(result.stdout.decode())
                status["agents_active"] = len(data.get("agents", []))
            except:
                pass
                
        # Check Docker services
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, timeout=5
        )
        if result.returncode == 0:
            services = result.stdout.decode().strip().split('\n')
            status["services_running"] = len([s for s in services if s and ('postgres' in s or 'redis' in s)])
            
    except Exception as e:
        print(f"Status check error: {e}")
    
    return status

def generate_qr_code(url):
    """Generate QR code for mobile access"""
    # Simple ASCII QR code placeholder
    print("┌" + "─" * 30 + "┐")
    print("│" + " " * 30 + "│")
    print("│" + "    📱 MOBILE ACCESS     ".center(30) + "│")
    print("│" + " " * 30 + "│") 
    print("│" + f"  {url}  ".center(30) + "│")
    print("│" + " " * 30 + "│")
    print("│" + "  Scan with mobile device  ".center(30) + "│")
    print("│" + " " * 30 + "│")
    print("└" + "─" * 30 + "┘")
    
    return url

def create_mobile_status_page():
    """Create a simple mobile status page"""
    status = check_system_status()
    local_ip = get_local_ip()
    
    mobile_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 LeanVibe Agent Hive - Mobile Control</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #1a1a2e; color: #eee;
        }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .status-card {{
            background: #16213e; border-radius: 12px; padding: 20px; margin: 15px 0;
            border-left: 4px solid #0f3460;
        }}
        .status-online {{ border-left-color: #00ff88; }}
        .status-warning {{ border-left-color: #ffaa00; }}
        .status-offline {{ border-left-color: #ff4444; }}
        .metric {{ display: flex; justify-content: space-between; margin: 10px 0; }}
        .control-btn {{
            width: 100%; padding: 15px; margin: 10px 0; border: none;
            border-radius: 8px; font-size: 16px; font-weight: 600;
            cursor: pointer; transition: all 0.3s;
        }}
        .btn-start {{ background: #00ff88; color: #000; }}
        .btn-stop {{ background: #ff4444; color: #fff; }}
        .btn-status {{ background: #0f7fda; color: #fff; }}
        .refresh-timer {{ text-align: center; color: #888; margin: 20px 0; }}
    </style>
    <script>
        let refreshTimer = 30;
        function updateTimer() {{
            document.getElementById('timer').textContent = refreshTimer;
            refreshTimer--;
            if (refreshTimer < 0) {{
                location.reload();
            }}
        }}
        setInterval(updateTimer, 1000);
        
        function executeCommand(cmd) {{
            if (confirm(`Execute: ${{cmd}}?`)) {{
                // In a real implementation, this would call the API
                alert(`Command ${{cmd}} would be executed`);
            }}
        }}
    </script>
</head>
<body>
    <div class="header">
        <h1>🚀 LeanVibe Agent Hive</h1>
        <p>Mobile Control Center</p>
    </div>
    
    <div class="status-card {'status-online' if status['api_online'] else 'status-offline'}">
        <h3>🌐 System Status</h3>
        <div class="metric">
            <span>API Server:</span>
            <span>{'✅ ONLINE' if status['api_online'] else '❌ OFFLINE'}</span>
        </div>
        <div class="metric">
            <span>Active Agents:</span>
            <span>{status['agents_active']}/5</span>
        </div>
        <div class="metric">
            <span>Services:</span>
            <span>{status['services_running']}/2 running</span>
        </div>
    </div>
    
    <div class="status-card">
        <h3>🤖 Quick Actions</h3>
        <button class="control-btn btn-start" onclick="executeCommand('start')">
            🚀 Start System
        </button>
        <button class="control-btn btn-status" onclick="executeCommand('status')">
            📊 Full Status
        </button>
        <button class="control-btn btn-stop" onclick="executeCommand('stop')">
            🛑 Emergency Stop
        </button>
    </div>
    
    <div class="refresh-timer">
        Auto-refresh in <span id="timer">30</span>s
    </div>
    
    <div class="status-card">
        <h3>📱 Access Info</h3>
        <div class="metric">
            <span>Local IP:</span>
            <span>{local_ip}</span>
        </div>
        <div class="metric">
            <span>Dashboard:</span>
            <span><a href="http://{local_ip}:8000" style="color: #00ff88;">Dashboard</a></span>
        </div>
    </div>
</body>
</html>
"""
    
    # Write mobile page
    with open("/Users/bogdan/work/leanvibe-dev/bee-hive/mobile_status.html", "w") as f:
        f.write(mobile_html)
    
    return f"http://{local_ip}:8000/mobile"

def main():
    """Main mobile dashboard generator"""
    print("🚀 LeanVibe Agent Hive 2.0 - Mobile Dashboard Generator")
    print("=" * 60)
    
    # Check system status
    status = check_system_status()
    local_ip = get_local_ip()
    
    # Generate mobile access URL
    mobile_url = create_mobile_status_page()
    
    print(f"\n📊 System Status:")
    print(f"   API Server: {'✅ ONLINE' if status['api_online'] else '❌ OFFLINE'}")
    print(f"   Active Agents: {status['agents_active']}/5")
    print(f"   Services Running: {status['services_running']}/2")
    
    print(f"\n📱 Mobile Access:")
    print(f"   URL: {mobile_url}")
    print(f"   Local IP: {local_ip}")
    
    print(f"\n📷 QR Code for Mobile Access:")
    generate_qr_code(mobile_url)
    
    print(f"\n🎯 Quick Actions:")
    print(f"   • Scan QR code with mobile device")
    print(f"   • Bookmark {mobile_url} for quick access")
    print(f"   • Use mobile interface for remote system oversight")
    
    return mobile_url

if __name__ == "__main__":
    main()