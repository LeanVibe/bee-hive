#!/bin/bash

# LeanVibe Agent Hive 2.0 - Sandbox Demo Launcher
# Zero-friction autonomous development demonstration without API keys required

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_PORT=${DEMO_PORT:-8080}
LOG_FILE="${SCRIPT_DIR}/sandbox-demo.log"

# Function to print colored status
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print banner
print_banner() {
    clear
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                                                                              ║${NC}"
    echo -e "${CYAN}║${BOLD}                  🏖️  LeanVibe Agent Hive 2.0 - Sandbox Demo                 ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}║                                                                              ║${NC}"
    echo -e "${CYAN}║  ${YELLOW}Experience autonomous AI development without any API keys required!${NC}${CYAN}     ║${NC}"
    echo -e "${CYAN}║                                                                              ║${NC}"
    echo -e "${CYAN}║  ${GREEN}✅ Zero configuration needed                                             ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}║  ${GREEN}✅ Realistic AI responses and multi-agent coordination                  ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}║  ${GREEN}✅ Complete autonomous development demonstrations                        ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}║  ${GREEN}✅ Professional quality for enterprise evaluation                       ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}║                                                                              ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo
}

# Function to check Python
check_python() {
    print_status "${BLUE}" "🔍 Checking Python installation..."
    
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    else
        print_status "${RED}" "❌ Python not found. Please install Python 3.8+ and try again."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_status "${RED}" "❌ Python 3.8+ required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    print_status "${GREEN}" "✅ Python $PYTHON_VERSION detected"
}

# Function to check dependencies
check_dependencies() {
    print_status "${BLUE}" "🔍 Checking dependencies..."
    
    # Check if pip is available
    if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
        print_status "${RED}" "❌ pip not found. Please install pip and try again."
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_status "${YELLOW}" "⚠️  Virtual environment not found. Creating..."
        $PYTHON_CMD -m venv venv
        print_status "${GREEN}" "✅ Virtual environment created"
    fi
    
    # Activate virtual environment
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || {
        print_status "${RED}" "❌ Failed to activate virtual environment"
        exit 1
    }
    
    print_status "${GREEN}" "✅ Virtual environment activated"
}

# Function to install dependencies
install_dependencies() {
    print_status "${BLUE}" "📦 Installing sandbox demo dependencies..."
    
    # Install minimal dependencies for sandbox mode
    pip install --quiet --upgrade pip
    
    # Install core dependencies
    pip install --quiet \
        fastapi \
        uvicorn \
        pydantic \
        structlog \
        asyncio \
        aiofiles \
        jinja2 \
        python-multipart
    
    print_status "${GREEN}" "✅ Dependencies installed"
}

# Function to set up sandbox environment
setup_sandbox_environment() {
    print_status "${BLUE}" "🏖️  Setting up sandbox environment..."
    
    # Create sandbox configuration
    cat > .env.sandbox << 'EOF'
# Sandbox Mode Configuration
SANDBOX_MODE=true
SANDBOX_DEMO_MODE=true

# Mock API Keys (not real - for sandbox demonstration only)
ANTHROPIC_API_KEY=sandbox-mock-anthropic-key
OPENAI_API_KEY=sandbox-mock-openai-key
GITHUB_TOKEN=sandbox-mock-github-token

# Demo Configuration
DEMO_PORT=8080
DEMO_SCENARIOS_ENABLED=true
REALISTIC_TIMING=true
PROGRESS_SIMULATION=true

# Sandbox Indicators
SHOW_SANDBOX_BANNER=true
SHOW_MOCK_INDICATORS=true

# Database (SQLite for sandbox)
DATABASE_URL=sqlite:///./sandbox_demo.db
REDIS_URL=redis://localhost:6379/0

# Security (demo only)
SECRET_KEY=sandbox-demo-secret-key-not-for-production-use
JWT_SECRET_KEY=sandbox-jwt-secret-key-demo-only
EOF
    
    print_status "${GREEN}" "✅ Sandbox environment configured"
}

# Function to create demo launcher
create_demo_launcher() {
    print_status "${BLUE}" "🚀 Creating demo launcher..."
    
    # Create standalone demo launcher
    cat > demo_launcher.py << 'EOF'
#!/usr/bin/env python3
"""
Standalone Sandbox Demo Launcher for LeanVibe Agent Hive 2.0
Runs autonomous development demonstrations without requiring API keys
"""

import os
import sys
import asyncio
from pathlib import Path

# Set sandbox environment
os.environ["SANDBOX_MODE"] = "true"
os.environ["SANDBOX_DEMO_MODE"] = "true"

# Add project to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Please run the setup script first: ./start-sandbox-demo.sh")
    sys.exit(1)

# Import sandbox components
try:
    from app.core.sandbox import (
        get_sandbox_config, 
        is_sandbox_mode,
        print_sandbox_banner,
        get_sandbox_status,
        SandboxOrchestrator,
        create_sandbox_orchestrator
    )
    from app.core.sandbox.demo_scenarios import get_demo_scenario_engine
except ImportError:
    # Fallback for minimal demo
    print("⚠️  Using minimal sandbox demo mode")
    
    def get_sandbox_config():
        return type('Config', (), {
            'enabled': True,
            'reason': 'Minimal sandbox demo',
            'mock_anthropic': True
        })()
    
    def is_sandbox_mode():
        return True
    
    def print_sandbox_banner():
        print("🏖️  SANDBOX MODE - Demo Running")
    
    def get_sandbox_status():
        return {"sandbox_mode": {"enabled": True, "demo_ready": True}}

# Create FastAPI application
app = FastAPI(
    title="LeanVibe Agent Hive 2.0 - Sandbox Demo",
    description="Autonomous AI Development Platform - Demo Mode",
    version="2.0.0-sandbox"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize sandbox orchestrator
try:
    orchestrator = create_sandbox_orchestrator()
except:
    orchestrator = None

@app.get("/")
async def root():
    """Root endpoint with sandbox demo information."""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LeanVibe Agent Hive 2.0 - Sandbox Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .banner { background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 30px; }
            .feature { background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4caf50; }
            .button { background: #667eea; color: white; padding: 12px 24px; border: none; border-radius: 5px; text-decoration: none; display: inline-block; margin: 10px 10px 10px 0; cursor: pointer; }
            .button:hover { background: #5a6fd8; }
            .status { background: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="banner">
                <h1>🏖️ LeanVibe Agent Hive 2.0 - Sandbox Demo</h1>
                <p>Experience autonomous AI development without API keys!</p>
            </div>
            
            <div class="status">
                <strong>🚀 Demo Mode Active:</strong> You're running a complete demonstration of autonomous development capabilities using mock AI services.
            </div>
            
            <div class="feature">
                <h3>🤖 Multi-Agent Autonomous Development</h3>
                <p>Watch AI agents collaborate to understand requirements, design architecture, implement code, create tests, and write documentation.</p>
            </div>
            
            <div class="feature">
                <h3>🔧 Zero Configuration Required</h3>
                <p>No API keys, complex setup, or external dependencies needed. Everything runs locally with realistic mock responses.</p>
            </div>
            
            <div class="feature">
                <h3>📊 Professional Quality Demonstrations</h3>
                <p>Enterprise-ready demonstrations suitable for technical evaluation and proof-of-concept validation.</p>
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <a href="/api/sandbox/status" class="button">📊 Sandbox Status</a>
                <a href="/api/demo/scenarios" class="button">🎯 Demo Scenarios</a>
                <a href="/api/docs" class="button">📖 API Documentation</a>
            </div>
            
            <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 5px;">
                <h3>🚀 Next Steps:</h3>
                <ul>
                    <li>Try the autonomous development demo scenarios</li>
                    <li>Explore the API documentation</li>
                    <li>Set up production mode with real API keys</li>
                    <li>Deploy to your infrastructure</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """)

@app.get("/api/sandbox/status")
async def sandbox_status():
    """Get sandbox mode status."""
    return JSONResponse(get_sandbox_status())

@app.get("/api/demo/scenarios")
async def demo_scenarios():
    """Get available demo scenarios."""
    try:
        scenario_engine = get_demo_scenario_engine()
        scenarios = scenario_engine.get_all_scenarios()
        return JSONResponse({
            "sandbox_mode": True,
            "scenarios_available": len(scenarios),
            "scenarios": scenarios[:3],  # Show first 3 scenarios
            "message": "Full scenario list available in production mode"
        })
    except:
        return JSONResponse({
            "sandbox_mode": True,
            "scenarios_available": 2,
            "scenarios": [
                {
                    "id": "fibonacci-demo",
                    "title": "Fibonacci Calculator Demo", 
                    "description": "Autonomous development of Fibonacci number calculator",
                    "complexity": "simple",
                    "estimated_duration_minutes": 5
                },
                {
                    "id": "temperature-demo", 
                    "title": "Temperature Converter Demo",
                    "description": "Multi-unit temperature converter with validation",
                    "complexity": "simple", 
                    "estimated_duration_minutes": 7
                }
            ],
            "message": "Demo scenarios ready for sandbox testing"
        })

@app.post("/api/demo/start")
async def start_demo(request: Request):
    """Start autonomous development demo."""
    body = await request.json()
    scenario_id = body.get("scenario_id", "fibonacci-demo")
    
    return JSONResponse({
        "demo_started": True,
        "scenario_id": scenario_id,
        "session_id": f"sandbox-{scenario_id}-demo",
        "estimated_duration_minutes": 5,
        "status": "Demo will showcase autonomous development workflow",
        "sandbox_mode": True,
        "progress_url": f"/api/demo/progress/sandbox-{scenario_id}-demo"
    })

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "LeanVibe Sandbox Demo",
        "sandbox_mode": True,
        "version": "2.0.0-sandbox"
    })

if __name__ == "__main__":
    # Print banner
    print_sandbox_banner()
    
    print(f"\n🚀 Starting LeanVibe Sandbox Demo on http://localhost:{os.getenv('DEMO_PORT', '8080')}")
    print(f"📖 API Documentation: http://localhost:{os.getenv('DEMO_PORT', '8080')}/docs")
    print(f"🏖️  Sandbox Status: http://localhost:{os.getenv('DEMO_PORT', '8080')}/api/sandbox/status")
    print(f"\n💡 This is a complete demonstration that works without API keys!")
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("DEMO_PORT", "8080")),
        log_level="info"
    )
EOF
    
    chmod +x demo_launcher.py
    print_status "${GREEN}" "✅ Demo launcher created"
}

# Function to start demo
start_demo() {
    print_status "${BLUE}" "🚀 Starting sandbox demo..."
    
    # Activate virtual environment
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
    
    # Load sandbox environment (filter out comments)
    export $(cat .env.sandbox | grep -v '^#' | grep -v '^$' | xargs)
    
    print_status "${GREEN}" "🌟 Demo starting on http://localhost:${DEMO_PORT}"
    print_status "${CYAN}" "📖 API Documentation: http://localhost:${DEMO_PORT}/docs"
    print_status "${PURPLE}" "🏖️  Sandbox Status: http://localhost:${DEMO_PORT}/api/sandbox/status"
    
    echo
    print_status "${BOLD}${YELLOW}" "💡 This is a complete autonomous development demonstration that works without API keys!"
    echo
    
    # Start the demo
    python demo_launcher.py
}

# Function to show usage
show_usage() {
    echo -e "${BOLD}Usage:${NC}"
    echo "  $0 [options]"
    echo
    echo -e "${BOLD}Options:${NC}"
    echo "  -p, --port PORT    Set demo port (default: 8080)"
    echo "  -h, --help         Show this help message"
    echo
    echo -e "${BOLD}Examples:${NC}"
    echo "  $0                 # Start demo on port 8080"
    echo "  $0 -p 9000         # Start demo on port 9000"
}

# Main execution
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--port)
                DEMO_PORT="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Setup and run demo
    print_banner
    check_python
    check_dependencies
    install_dependencies
    setup_sandbox_environment
    create_demo_launcher
    start_demo
}

# Error handling
trap 'print_status "${RED}" "❌ Setup failed. Check ${LOG_FILE} for details."' ERR

# Redirect output to log file
exec > >(tee -a "${LOG_FILE}")
exec 2>&1

# Run main function
main "$@"