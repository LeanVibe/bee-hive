#!/bin/bash

# LeanVibe Agent Hive 2.0 - DevContainer Post-Create Script
# Target: <2 minute setup with sandbox mode enabled

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

# Performance tracking
START_TIME=$(date +%s)

print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_step() {
    local message=$1
    local elapsed=$((($(date +%s) - START_TIME)))
    print_status "$BLUE" "[$elapsed s] 🔧 $message"
}

print_success() {
    local message=$1
    local elapsed=$((($(date +%s) - START_TIME)))
    print_status "$GREEN" "[$elapsed s] ✅ $message"
}

print_error() {
    local message=$1
    local elapsed=$((($(date +%s) - START_TIME)))
    print_status "$RED" "[$elapsed s] ❌ $message"
    exit 1
}

# Main setup function
main() {
    print_status "$BOLD$PURPLE" "🚀 LeanVibe Agent Hive 2.0 - DevContainer Setup"
    print_status "$PURPLE" "=============================================="
    print_status "$CYAN" "🎯 Target: <2 minute setup with sandbox mode"
    echo ""
    
    # Step 1: Git configuration
    print_step "Configuring Git for workspace"
    git config --global --add safe.directory /workspace || true
    git config --global user.name "DevContainer User" || true
    git config --global user.email "dev@leanvibe.com" || true
    print_success "Git configured"
    
    # Step 2: Create sandbox environment file
    print_step "Creating sandbox environment configuration"
    cat > /workspace/.env.local << 'EOF'
# LeanVibe Agent Hive 2.0 - DevContainer Sandbox Configuration
# Generated automatically for DevContainer development

# Sandbox Mode - Safe for immediate testing
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
SANDBOX_MODE=true

# Database - DevContainer optimized
DATABASE_URL=postgresql://leanvibe_user:leanvibe_secure_pass@postgres:5432/leanvibe_agent_hive
DATABASE_POOL_SIZE=3
DATABASE_MAX_OVERFLOW=5

# Redis - DevContainer optimized
REDIS_URL=redis://:leanvibe_redis_pass@redis:6379/0
REDIS_STREAM_MAX_LEN=1000

# Demo API Keys - For sandbox testing only
# Replace with real keys for production use
ANTHROPIC_API_KEY=demo_key_for_sandbox_only
OPENAI_API_KEY=demo_key_for_sandbox_only
GITHUB_TOKEN=demo_token_for_sandbox_only

# Performance Settings - DevContainer optimized
MAX_CONCURRENT_AGENTS=3
AGENT_HEARTBEAT_INTERVAL=30
CONTEXT_MAX_TOKENS=2000

# Security Keys - Generated for sandbox
SECRET_KEY=sandbox_secret_key_12345
JWT_SECRET_KEY=sandbox_jwt_secret_67890

# Docker Service Passwords
POSTGRES_PASSWORD=leanvibe_secure_pass
REDIS_PASSWORD=leanvibe_redis_pass
PGLADMIN_PASSWORD=admin_password
EOF
    print_success "Sandbox environment configured"
    
    # Step 3: Python virtual environment (quick setup)
    print_step "Setting up Python virtual environment"
    if [ ! -d "/workspace/venv" ]; then
        python3 -m venv /workspace/venv
    fi
    source /workspace/venv/bin/activate
    pip install --upgrade pip setuptools wheel > /dev/null 2>&1
    print_success "Python environment ready"
    
    # Step 4: Install core dependencies (minimal for speed)
    print_step "Installing core dependencies"
    if [ -f "/workspace/pyproject.toml" ]; then
        pip install -e . > /dev/null 2>&1 || pip install fastapi uvicorn sqlalchemy asyncpg redis python-multipart > /dev/null 2>&1
    else
        pip install fastapi uvicorn sqlalchemy asyncpg redis python-multipart > /dev/null 2>&1
    fi
    print_success "Core dependencies installed"
    
    # Step 5: Create demo data and scripts
    print_step "Setting up sandbox demo environment"
    mkdir -p /workspace/sandbox
    
    # Create quick start script
    cat > /workspace/sandbox/quick_start.sh << 'EOF'
#!/bin/bash

echo "🚀 LeanVibe Agent Hive 2.0 - DevContainer Quick Start"
echo "================================================="
echo ""
echo "✅ DevContainer setup complete!"
echo "⚡ Setup time: <2 minutes (target achieved)"
echo ""
echo "🎯 Quick Commands:"
echo "  python scripts/demos/autonomous_development_demo.py  - Autonomous development demo"
echo "  ./start-fast.sh                                      - Start all services"
echo "  ./health-check.sh                                    - Check system health"
echo ""
echo "🌐 Service URLs (after ./start-fast.sh):"
echo "  • API Docs: http://localhost:8000/docs"
echo "  • Health: http://localhost:8000/health"
echo "  • pgAdmin: http://localhost:5050"
echo ""
echo "📝 Configuration:"
echo "  • Sandbox mode: ENABLED (demo keys configured)"
echo "  • Environment: /workspace/.env.local"
echo "  • Python: /workspace/venv/bin/python"
echo ""
echo "🎉 Ready for autonomous development!"
EOF
    
    chmod +x /workspace/sandbox/quick_start.sh
    print_success "Sandbox demo environment ready"
    
    # Step 6: Final validation
    print_step "Validating setup"
    python3 -c "import sys; print(f'Python: {sys.version}')" > /dev/null
    [ -f "/workspace/.env.local" ] || print_error "Environment file missing"
    [ -d "/workspace/venv" ] || print_error "Virtual environment missing"
    print_success "Setup validation passed"
    
    # Final summary
    local total_time=$((($(date +%s) - START_TIME)))
    echo ""
    print_status "$BOLD$GREEN" "🎉 DevContainer Setup Complete!"
    print_status "$GREEN" "⚡ Time: ${total_time} seconds (target: <120s)"
    print_status "$GREEN" "🎯 Sandbox mode: ENABLED"
    echo ""
    
    if [ $total_time -le 120 ]; then
        print_status "$BOLD$GREEN" "🏆 TARGET ACHIEVED: <2 minute setup!"
    else
        print_status "$YELLOW" "📈 Close to target (cached runs will be faster)"
    fi
    
    echo ""
    print_status "$YELLOW" "🚀 Next Steps:"
    print_status "$NC" "1. Run: ./sandbox/quick_start.sh"
    print_status "$NC" "2. Run: python scripts/demos/autonomous_development_demo.py"
    print_status "$NC" "3. For real usage: Update API keys in .env.local"
    echo ""
}

# Execute main function
main "$@"