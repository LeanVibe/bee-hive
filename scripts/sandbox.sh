#!/bin/bash

# LeanVibe Agent Hive 2.0 - Sandbox Demo Script
# Interactive demonstration environment for autonomous development
#
# Usage: ./scripts/sandbox.sh [MODE] [OPTIONS]
# Modes: interactive (default), demo, auto, showcase
#
# Environment Variables:
#   SANDBOX_MODE      - Override mode selection (interactive|demo|auto|showcase)
#   DEMO_DURATION     - Demo duration in seconds (default: 300)
#   SKIP_SETUP        - Skip sandbox setup (true/false)
#   BROWSER_OPEN      - Open browser automatically (true/false)

set -euo pipefail

# Color codes for professional output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Script metadata
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_VERSION="2.0.0"

# Configuration
readonly DEFAULT_MODE="interactive"
readonly DEFAULT_DEMO_DURATION=300
readonly SANDBOX_PORT=8001
readonly API_PORT=8000

# Mode configurations (using functions for compatibility)
get_sandbox_mode_description() {
    case "$1" in
        "interactive") echo "Interactive sandbox with guided demonstrations" ;;
        "demo") echo "Automated demo mode for presentations" ;;
        "auto") echo "Fully autonomous development showcase" ;;
        "showcase") echo "Best-of showcase for external audiences" ;;
        *) echo "Unknown mode" ;;
    esac
}

get_sandbox_mode_features() {
    case "$1" in
        "interactive") echo "guided_tour user_interaction real_time_feedback" ;;
        "demo") echo "automated_sequence presentation_mode progress_tracking" ;;
        "auto") echo "autonomous_agents feature_development self_healing" ;;
        "showcase") echo "highlight_reel success_stories performance_metrics" ;;
        *) echo "" ;;
    esac
}

# Global variables
SANDBOX_MODE="${SANDBOX_MODE:-${1:-$DEFAULT_MODE}}"
DEMO_DURATION="${DEMO_DURATION:-$DEFAULT_DEMO_DURATION}"
SKIP_SETUP="${SKIP_SETUP:-false}"
BROWSER_OPEN="${BROWSER_OPEN:-true}"
START_TIME=""
SANDBOX_LOG=""
SANDBOX_PID=""

#======================================
# Utility Functions
#======================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")  echo -e "${BLUE}[INFO]${NC}  $message" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC}  $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $message" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} $message" ;;
        "STEP") echo -e "${PURPLE}[STEP]${NC} $message" ;;
        "DEMO") echo -e "${CYAN}[DEMO]${NC} $message" ;;
    esac
    
    # Log to file if available
    if [[ -n "$SANDBOX_LOG" ]]; then
        echo "[$timestamp] [$level] $message" >> "$SANDBOX_LOG"
    fi
}

show_header() {
    clear
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                          LeanVibe Agent Hive 2.0                            ║
║                          Sandbox Demonstration                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo
    log "INFO" "Starting sandbox in mode: ${SANDBOX_MODE}"
    log "INFO" "Mode: $(get_sandbox_mode_description "$SANDBOX_MODE")"
    echo
    START_TIME=$(date +%s)
}

show_help() {
    cat << EOF
${CYAN}LeanVibe Agent Hive 2.0 - Sandbox Demonstration Script${NC}

${YELLOW}USAGE:${NC}
    $SCRIPT_NAME [MODE] [OPTIONS]

${YELLOW}SANDBOX MODES:${NC}
EOF
    for mode in interactive demo auto showcase; do
        echo "    ${GREEN}$mode${NC} - $(get_sandbox_mode_description "$mode")"
    done
    cat << EOF

${YELLOW}ENVIRONMENT VARIABLES:${NC}
    SANDBOX_MODE    Override mode selection
    DEMO_DURATION   Demo duration in seconds (default: 300)
    SKIP_SETUP      Skip sandbox setup (true/false)
    BROWSER_OPEN    Open browser automatically (true/false)

${YELLOW}EXAMPLES:${NC}
    $SCRIPT_NAME                        # Interactive sandbox (default)
    $SCRIPT_NAME demo                   # Automated demo mode
    $SCRIPT_NAME auto                   # Autonomous development showcase
    DEMO_DURATION=600 $SCRIPT_NAME demo # 10-minute demo

${YELLOW}INTERACTIVE CONTROLS:${NC}
    Ctrl+C          Exit sandbox gracefully
    Space           Pause/resume demo
    Enter           Continue to next step
    'q'             Quit immediately

${YELLOW}ACCESS POINTS:${NC}
    Sandbox UI:     http://localhost:$SANDBOX_PORT
    API Docs:       http://localhost:$API_PORT/docs
    Health Check:   http://localhost:$API_PORT/health

${YELLOW}MORE INFO:${NC}
    Sandbox Guide: docs/SANDBOX_MODE_GUIDE.md
    Demo Scripts: scripts/demos/
EOF
}

setup_logging() {
    local log_dir="$PROJECT_ROOT/logs"
    mkdir -p "$log_dir"
    SANDBOX_LOG="$log_dir/sandbox-$(date '+%Y%m%d-%H%M%S').log"
    log "INFO" "Sandbox logging to: $SANDBOX_LOG"
}

check_prerequisites() {
    log "STEP" "Checking sandbox prerequisites..."
    
    local errors=0
    
    # Check virtual environment
    if [[ ! -d "$PROJECT_ROOT/venv" ]]; then
        log "ERROR" "Python virtual environment not found. Run 'make setup' first."
        errors=$((errors + 1))
    fi
    
    # Check if services are running
    if ! curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
        log "WARN" "API service not running on port $API_PORT"
        log "INFO" "Start services with: make start"
        # Don't count as error - we can start it
    fi
    
    # Check demo scripts
    if [[ ! -f "$PROJECT_ROOT/scripts/demos/autonomous_development_demo.py" ]]; then
        log "ERROR" "Demo scripts not found"
        errors=$((errors + 1))
    fi
    
    # Check sandbox demo script
    if [[ ! -f "$PROJECT_ROOT/start-sandbox-demo.sh" ]]; then
        log "WARN" "Legacy sandbox script not found - using integrated mode"
    fi
    
    if [[ $errors -gt 0 ]]; then
        log "ERROR" "Prerequisites check failed"
        exit 1
    fi
    
    log "SUCCESS" "Prerequisites check passed"
}

start_sandbox_services() {
    if [[ "$SKIP_SETUP" == "true" ]]; then
        log "INFO" "Skipping sandbox setup (SKIP_SETUP=true)"
        return 0
    fi
    
    log "STEP" "Starting sandbox services..."
    
    cd "$PROJECT_ROOT"
    
    # Start core services if not running
    if ! curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
        log "INFO" "Starting core services..."
        source venv/bin/activate
        
        # Start services in background
        nohup uvicorn app.main:app --host 0.0.0.0 --port $API_PORT > "$SANDBOX_LOG.api" 2>&1 &
        local api_pid=$!
        echo $api_pid > "$PROJECT_ROOT/.sandbox_api.pid"
        
        # Wait for API to be ready
        local retries=20
        while [[ $retries -gt 0 ]]; do
            if curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
                log "SUCCESS" "API service is ready"
                break
            fi
            retries=$((retries - 1))
            sleep 2
            echo -n "."
        done
        
        if [[ $retries -eq 0 ]]; then
            log "ERROR" "API service failed to start"
            return 1
        fi
    else
        log "INFO" "API service already running"
    fi
    
    # Start sandbox-specific demo service
    log "INFO" "Starting sandbox demo service..."
    source venv/bin/activate
    
    # Create sandbox demo server
    cat > "$PROJECT_ROOT/sandbox_server.py" << 'EOF'
#!/usr/bin/env python3
"""
LeanVibe Agent Hive 2.0 - Sandbox Demo Server
Interactive demonstration environment
"""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LeanVibe Agent Hive 2.0 - Sandbox",
    description="Interactive demonstration environment",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo state
demo_state = {
    "mode": "interactive",
    "current_step": 0,
    "total_steps": 10,
    "is_running": False,
    "is_paused": False,
    "start_time": None,
    "demo_scenarios": []
}

# WebSocket connections
active_connections: list[WebSocket] = []

async def broadcast_message(message: Dict[str, Any]):
    """Broadcast message to all connected clients"""
    if active_connections:
        message_str = json.dumps(message)
        for connection in active_connections:
            try:
                await connection.send_text(message_str)
            except:
                pass

@app.get("/", response_class=HTMLResponse)
async def get_sandbox_ui():
    """Serve the sandbox UI"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LeanVibe Agent Hive 2.0 - Sandbox</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }
            .demo-controls { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .demo-area { background: white; padding: 20px; border-radius: 10px; min-height: 400px; }
            .btn { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
            .btn-primary { background: #007bff; color: white; }
            .btn-success { background: #28a745; color: white; }
            .btn-warning { background: #ffc107; color: black; }
            .btn-danger { background: #dc3545; color: white; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .status-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status-info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
            .status-warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
            .demo-log { background: #f8f9fa; padding: 15px; border-radius: 5px; 
                       font-family: monospace; font-size: 14px; max-height: 300px; overflow-y: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 LeanVibe Agent Hive 2.0 - Sandbox</h1>
                <p>Interactive Autonomous Development Demonstration</p>
            </div>
            
            <div class="demo-controls">
                <h3>Demo Controls</h3>
                <button class="btn btn-primary" onclick="startDemo()">Start Demo</button>
                <button class="btn btn-warning" onclick="pauseDemo()">Pause/Resume</button>
                <button class="btn btn-danger" onclick="stopDemo()">Stop Demo</button>
                <button class="btn btn-success" onclick="runAutonomousDemo()">Autonomous Development</button>
                
                <div class="status status-info" id="status">
                    Ready to start demonstration
                </div>
            </div>
            
            <div class="demo-area">
                <h3>Live Demo Feed</h3>
                <div class="demo-log" id="demoLog">
                    Waiting for demo to start...
                </div>
            </div>
        </div>
        
        <script>
            let ws = null;
            
            function connectWebSocket() {
                ws = new WebSocket(`ws://localhost:8001/ws`);
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDemoLog(data);
                };
                ws.onclose = function() {
                    setTimeout(connectWebSocket, 1000);
                };
            }
            
            function updateDemoLog(data) {
                const log = document.getElementById('demoLog');
                const status = document.getElementById('status');
                
                log.innerHTML += `<div>[${data.timestamp}] ${data.message}</div>`;
                log.scrollTop = log.scrollHeight;
                
                if (data.status) {
                    status.textContent = data.status;
                    status.className = `status status-${data.level || 'info'}`;
                }
            }
            
            function startDemo() {
                fetch('/demo/start', { method: 'POST' });
            }
            
            function pauseDemo() {
                fetch('/demo/pause', { method: 'POST' });
            }
            
            function stopDemo() {
                fetch('/demo/stop', { method: 'POST' });
                document.getElementById('demoLog').innerHTML = 'Demo stopped.';
            }
            
            function runAutonomousDemo() {
                fetch('/demo/autonomous', { method: 'POST' });
            }
            
            connectWebSocket();
        </script>
    </body>
    </html>
    """

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)

@app.post("/demo/start")
async def start_demo():
    """Start the demo"""
    demo_state["is_running"] = True
    demo_state["is_paused"] = False
    demo_state["start_time"] = datetime.now()
    demo_state["current_step"] = 0
    
    await broadcast_message({
        "timestamp": datetime.now().isoformat(),
        "message": "🚀 Starting LeanVibe Agent Hive demonstration...",
        "level": "success",
        "status": "Demo starting..."
    })
    
    # Run demo steps
    asyncio.create_task(run_demo_sequence())
    return {"status": "started"}

@app.post("/demo/pause")
async def pause_demo():
    """Pause/resume the demo"""
    demo_state["is_paused"] = not demo_state["is_paused"]
    status = "paused" if demo_state["is_paused"] else "resumed"
    
    await broadcast_message({
        "timestamp": datetime.now().isoformat(),
        "message": f"⏸️ Demo {status}",
        "level": "warning",
        "status": f"Demo {status}"
    })
    
    return {"status": status}

@app.post("/demo/stop")
async def stop_demo():
    """Stop the demo"""
    demo_state["is_running"] = False
    demo_state["is_paused"] = False
    
    await broadcast_message({
        "timestamp": datetime.now().isoformat(),
        "message": "🛑 Demo stopped",
        "level": "info",
        "status": "Demo stopped"
    })
    
    return {"status": "stopped"}

@app.post("/demo/autonomous")
async def run_autonomous_demo():
    """Run autonomous development demo"""
    await broadcast_message({
        "timestamp": datetime.now().isoformat(),
        "message": "🤖 Starting autonomous development demonstration...",
        "level": "success",
        "status": "Autonomous demo starting..."
    })
    
    # Run autonomous demo
    asyncio.create_task(run_autonomous_sequence())
    return {"status": "autonomous_started"}

async def run_demo_sequence():
    """Run the demo sequence"""
    demo_steps = [
        "Initializing Agent Hive system...",
        "Loading autonomous development agents...",
        "Setting up multi-agent coordination...",
        "Demonstrating context memory system...",
        "Showing intelligent task routing...",
        "Running real-time agent collaboration...",
        "Executing autonomous feature development...",
        "Validating self-healing capabilities...",
        "Demonstrating GitHub integration...",
        "Completing autonomous development cycle..."
    ]
    
    for i, step in enumerate(demo_steps):
        if not demo_state["is_running"]:
            break
            
        while demo_state["is_paused"]:
            await asyncio.sleep(1)
            
        demo_state["current_step"] = i + 1
        
        await broadcast_message({
            "timestamp": datetime.now().isoformat(),
            "message": f"Step {i+1}/{len(demo_steps)}: {step}",
            "level": "info",
            "status": f"Step {i+1} of {len(demo_steps)}"
        })
        
        await asyncio.sleep(3)  # Wait between steps
    
    if demo_state["is_running"]:
        await broadcast_message({
            "timestamp": datetime.now().isoformat(),
            "message": "✅ Demo completed successfully!",
            "level": "success",
            "status": "Demo completed"
        })

async def run_autonomous_sequence():
    """Run autonomous development sequence"""
    import subprocess
    import sys
    
    try:
        # Run the actual autonomous development demo
        process = subprocess.Popen([
            sys.executable, "scripts/demos/autonomous_development_demo.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Stream output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                await broadcast_message({
                    "timestamp": datetime.now().isoformat(),
                    "message": output.strip(),
                    "level": "info",
                    "status": "Autonomous demo running..."
                })
        
        await broadcast_message({
            "timestamp": datetime.now().isoformat(),
            "message": "🎉 Autonomous development demo completed!",
            "level": "success",
            "status": "Autonomous demo completed"
        })
        
    except Exception as e:
        await broadcast_message({
            "timestamp": datetime.now().isoformat(),
            "message": f"❌ Error running autonomous demo: {str(e)}",
            "level": "error",
            "status": "Demo error"
        })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "demo_state": demo_state
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
EOF
    
    # Start sandbox server
    nohup python sandbox_server.py > "$SANDBOX_LOG.sandbox" 2>&1 &
    SANDBOX_PID=$!
    echo $SANDBOX_PID > "$PROJECT_ROOT/.sandbox.pid"
    
    # Wait for sandbox to be ready
    local retries=15
    while [[ $retries -gt 0 ]]; do
        if curl -s http://localhost:$SANDBOX_PORT/health > /dev/null 2>&1; then
            log "SUCCESS" "Sandbox service is ready"
            break
        fi
        retries=$((retries - 1))
        sleep 2
        echo -n "."
    done
    
    if [[ $retries -eq 0 ]]; then
        log "ERROR" "Sandbox service failed to start"
        return 1
    fi
    
    log "SUCCESS" "Sandbox services started successfully"
}

open_browser() {
    if [[ "$BROWSER_OPEN" != "true" ]]; then
        return 0
    fi
    
    log "STEP" "Opening sandbox in browser..."
    
    # Wait a moment for services to be fully ready
    sleep 3
    
    # Open browser based on OS
    if command -v open &> /dev/null; then
        # macOS
        open "http://localhost:$SANDBOX_PORT" &
    elif command -v xdg-open &> /dev/null; then
        # Linux
        xdg-open "http://localhost:$SANDBOX_PORT" &
    elif command -v start &> /dev/null; then
        # Windows
        start "http://localhost:$SANDBOX_PORT" &
    else
        log "INFO" "Please open http://localhost:$SANDBOX_PORT in your browser"
    fi
    
    log "SUCCESS" "Browser opened"
}

run_interactive_mode() {
    log "STEP" "Starting interactive sandbox mode..."
    
    echo
    echo -e "${YELLOW}🎮 INTERACTIVE SANDBOX MODE${NC}"
    echo "=============================="
    echo
    echo "The sandbox is now running at: http://localhost:$SANDBOX_PORT"
    echo "API documentation available at: http://localhost:$API_PORT/docs"
    echo
    echo -e "${CYAN}Available demonstrations:${NC}"
    echo "1. Autonomous Development - Watch AI agents build features"
    echo "2. Multi-Agent Coordination - See agents collaborate in real-time"
    echo "3. Context Memory System - Intelligent learning and adaptation"
    echo "4. Self-Healing Platform - Automatic error recovery"
    echo
    echo -e "${YELLOW}Controls:${NC}"
    echo "  Ctrl+C    - Exit sandbox"
    echo "  Browser   - Interactive controls available in web UI"
    echo
    
    # Keep running until interrupted
    trap 'log "INFO" "Shutting down sandbox..."; cleanup_sandbox; exit 0' INT TERM
    
    while true; do
        sleep 5
        # Check if services are still running
        if ! kill -0 "$SANDBOX_PID" 2>/dev/null; then
            log "ERROR" "Sandbox service stopped unexpectedly"
            break
        fi
    done
}

run_demo_mode() {
    log "STEP" "Starting automated demo mode..."
    
    echo
    echo -e "${YELLOW}🎬 AUTOMATED DEMO MODE${NC}"
    echo "======================="
    echo
    echo "Running $DEMO_DURATION second automated demonstration..."
    echo "Demo will showcase autonomous development capabilities"
    echo
    
    # Run automated demo sequence
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Start the autonomous development demo
    timeout "$DEMO_DURATION" python scripts/demos/autonomous_development_demo.py || true
    
    log "SUCCESS" "Demo mode completed"
}

run_auto_mode() {
    log "STEP" "Starting autonomous development showcase..."
    
    echo
    echo -e "${YELLOW}🤖 AUTONOMOUS DEVELOPMENT SHOWCASE${NC}"
    echo "===================================="
    echo
    echo "Fully autonomous AI agents will now:"
    echo "1. Analyze requirements"
    echo "2. Design architecture"
    echo "3. Implement features"
    echo "4. Run tests"
    echo "5. Deploy changes"
    echo
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Run standalone autonomous demo
    python scripts/demos/standalone_autonomous_demo.py
    
    log "SUCCESS" "Autonomous showcase completed"
}

run_showcase_mode() {
    log "STEP" "Starting showcase mode..."
    
    echo
    echo -e "${YELLOW}🏆 SHOWCASE MODE${NC}"
    echo "================="
    echo
    echo "Presenting the best of LeanVibe Agent Hive:"
    echo
    
    # Run multiple demo scenarios
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    local demos=(
        "scripts/demos/autonomous_development_demo.py"
        "scripts/demos/real_multiagent_workflow_demo.py"
        "scripts/demos/phase_3_milestone_demonstration.py"
    )
    
    for demo in "${demos[@]}"; do
        if [[ -f "$demo" ]]; then
            log "DEMO" "Running $(basename "$demo")..."
            python "$demo" || log "WARN" "Demo $(basename "$demo") completed with warnings"
            echo
            sleep 2
        fi
    done
    
    log "SUCCESS" "Showcase completed"
}

cleanup_sandbox() {
    log "INFO" "Cleaning up sandbox services..."
    
    # Stop sandbox server
    if [[ -n "$SANDBOX_PID" ]] && kill -0 "$SANDBOX_PID" 2>/dev/null; then
        kill -TERM "$SANDBOX_PID" || true
    fi
    
    # Stop API if we started it
    if [[ -f "$PROJECT_ROOT/.sandbox_api.pid" ]]; then
        local api_pid=$(cat "$PROJECT_ROOT/.sandbox_api.pid")
        if kill -0 "$api_pid" 2>/dev/null; then
            kill -TERM "$api_pid" || true
        fi
        rm -f "$PROJECT_ROOT/.sandbox_api.pid"
    fi
    
    # Clean up PID files
    rm -f "$PROJECT_ROOT/.sandbox.pid"
    rm -f "$PROJECT_ROOT/sandbox_server.py"
    
    log "SUCCESS" "Sandbox cleanup completed"
}

show_sandbox_summary() {
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    echo
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                            SANDBOX COMPLETE                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo
    log "SUCCESS" "Sandbox session completed in ${minutes}m ${seconds}s"
    echo
    
    echo -e "${YELLOW}WHAT YOU EXPERIENCED:${NC}"
    echo "✅ Autonomous AI development in action"
    echo "✅ Multi-agent coordination and collaboration"
    echo "✅ Real-time context memory and learning"
    echo "✅ Self-healing and error recovery"
    echo "✅ Professional development workflows"
    echo
    
    echo -e "${YELLOW}NEXT STEPS:${NC}"
    echo "🚀 Ready to set up your own system? Run: make setup"
    echo "📚 Learn more: docs/GETTING_STARTED.md"
    echo "🏢 Enterprise evaluation: docs/enterprise/"
    echo "💬 Questions? Check: docs/TROUBLESHOOTING_GUIDE_COMPREHENSIVE.md"
    echo
    
    if [[ -n "$SANDBOX_LOG" ]]; then
        echo -e "${CYAN}Sandbox log saved to:${NC} $SANDBOX_LOG"
    fi
    echo
}

handle_sandbox_error() {
    local exit_code=$?
    log "ERROR" "Sandbox failed with exit code $exit_code"
    cleanup_sandbox
    exit $exit_code
}

#======================================
# Main Sandbox Flow
#======================================

main() {
    # Handle help request
    if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
        show_help
        exit 0
    fi
    
    # Validate mode
    if [[ -n "${1:-}" ]] && [[ "$1" != "interactive" && "$1" != "demo" && "$1" != "auto" && "$1" != "showcase" ]]; then
        log "ERROR" "Invalid sandbox mode: $1"
        echo "Valid modes: interactive demo auto showcase"
        exit 1
    fi
    
    # Set up error handling
    trap handle_sandbox_error ERR
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Initialize
    show_header
    setup_logging
    check_prerequisites
    start_sandbox_services
    
    # Open browser
    open_browser
    
    # Run mode-specific demo
    case "$SANDBOX_MODE" in
        "interactive")
            run_interactive_mode
            ;;
        "demo")
            run_demo_mode
            ;;
        "auto")
            run_auto_mode
            ;;
        "showcase")
            run_showcase_mode
            ;;
    esac
    
    # Cleanup and summary
    cleanup_sandbox
    show_sandbox_summary
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi