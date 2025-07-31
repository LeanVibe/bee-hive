#!/bin/bash

# LeanVibe Agent Hive 2.0 - Troubleshooting Script
# Automated diagnostics and fixes for common deployment issues

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TROUBLESHOOT_LOG="${SCRIPT_DIR}/troubleshoot.log"

print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    local title=$1
    echo ""
    print_status "$PURPLE" "🔧 $title"
    print_status "$PURPLE" "$(printf '=%.0s' $(seq 1 $((${#title} + 3))))"
}

log_command() {
    echo "=== $(date): $1 ===" >> "$TROUBLESHOOT_LOG"
    eval "$1" >> "$TROUBLESHOOT_LOG" 2>&1
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

check_and_fix_docker() {
    print_header "Docker Issues"
    
    if ! command_exists docker; then
        print_status "$RED" "❌ Docker not installed"
        print_status "$CYAN" "Installing Docker..."
        
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker $USER
            rm get-docker.sh
            print_status "$GREEN" "✅ Docker installed. Please log out and back in."
        else
            print_status "$YELLOW" "⚠️  Please install Docker Desktop manually"
        fi
        return
    fi
    
    if ! docker info >/dev/null 2>&1; then
        print_status "$RED" "❌ Docker daemon not running"
        print_status "$CYAN" "Attempting to start Docker..."
        
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo systemctl start docker
            sudo systemctl enable docker
            print_status "$GREEN" "✅ Docker service started"
        else
            print_status "$YELLOW" "⚠️  Please start Docker Desktop manually"
        fi
    else
        print_status "$GREEN" "✅ Docker is running"
    fi
    
    # Check Docker Compose
    if ! docker compose version >/dev/null 2>&1; then
        if ! command_exists docker-compose; then
            print_status "$RED" "❌ Docker Compose not available"
            print_status "$CYAN" "Docker Compose should be included with Docker Desktop"
        else
            print_status "$YELLOW" "⚠️  Using legacy docker-compose"
        fi
    else
        print_status "$GREEN" "✅ Docker Compose available"
    fi
}

check_and_fix_ports() {
    print_header "Port Conflicts"
    
    local ports=("5432" "6380" "8000" "3000")
    local conflicts_found=false
    
    for port in "${ports[@]}"; do
        if command_exists lsof; then
            local process=$(lsof -ti:$port 2>/dev/null || echo "")
            if [[ -n "$process" ]]; then
                local process_name=$(ps -p $process -o comm= 2>/dev/null || echo "unknown")
                if [[ "$process_name" != "docker-proxy" ]]; then
                    print_status "$RED" "❌ Port $port is in use by $process_name (PID: $process)"
                    conflicts_found=true
                    
                    read -p "Kill process on port $port? (y/N): " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        kill -9 $process 2>/dev/null || true
                        print_status "$GREEN" "✅ Process killed"
                    fi
                fi
            else
                print_status "$GREEN" "✅ Port $port is available"
            fi
        elif command_exists netstat; then
            if netstat -tuln 2>/dev/null | grep -q ":$port "; then
                print_status "$YELLOW" "⚠️  Port $port appears to be in use"
                conflicts_found=true
            else
                print_status "$GREEN" "✅ Port $port is available"
            fi
        else
            print_status "$YELLOW" "⚠️  Cannot check port $port (no lsof or netstat)"
        fi
    done
    
    if ! $conflicts_found; then
        print_status "$GREEN" "✅ No port conflicts detected"
    fi
}

check_and_fix_permissions() {
    print_header "File Permissions"
    
    # Check script permissions
    local scripts=("setup.sh" "health-check.sh" "validate-setup.sh" "start.sh" "stop.sh")
    local fixed_permissions=false
    
    for script in "${scripts[@]}"; do
        if [[ -f "${SCRIPT_DIR}/$script" ]]; then
            if [[ ! -x "${SCRIPT_DIR}/$script" ]]; then
                print_status "$YELLOW" "⚠️  $script is not executable"
                chmod +x "${SCRIPT_DIR}/$script"
                print_status "$GREEN" "✅ Fixed permissions for $script"
                fixed_permissions=true
            else
                print_status "$GREEN" "✅ $script has correct permissions"
            fi
        fi
    done
    
    # Check directory permissions
    local dirs=("workspaces" "logs" "temp")
    for dir in "${dirs[@]}"; do
        local dir_path="${SCRIPT_DIR}/$dir"
        if [[ ! -d "$dir_path" ]]; then
            mkdir -p "$dir_path"
            print_status "$GREEN" "✅ Created directory: $dir"
        elif [[ ! -w "$dir_path" ]]; then
            chmod 755 "$dir_path"
            print_status "$GREEN" "✅ Fixed permissions for $dir directory"
            fixed_permissions=true
        else
            print_status "$GREEN" "✅ Directory $dir has correct permissions"
        fi
    done
    
    if ! $fixed_permissions; then
        print_status "$GREEN" "✅ All permissions are correct"
    fi
}

check_and_fix_environment() {
    print_header "Environment Configuration"
    
    local env_file="${SCRIPT_DIR}/.env.local"
    
    if [[ ! -f "$env_file" ]]; then
        print_status "$RED" "❌ Environment file missing"
        print_status "$CYAN" "Creating default environment file..."
        
        # Generate secure keys
        local secret_key=$(python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null || openssl rand -hex 32)
        local jwt_secret=$(python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null || openssl rand -hex 32)
        
        cat > "$env_file" << EOF
# LeanVibe Agent Hive 2.0 - Auto-generated Configuration
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=$secret_key
JWT_SECRET_KEY=$jwt_secret
DATABASE_URL=postgresql://leanvibe_user:leanvibe_secure_pass@localhost:5432/leanvibe_agent_hive
REDIS_URL=redis://:leanvibe_redis_pass@localhost:6380/0
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GITHUB_TOKEN=your_github_token_here
BASE_URL=http://localhost:8000
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://localhost:8000
EOF
        print_status "$GREEN" "✅ Environment file created"
        print_status "$YELLOW" "⚠️  Don't forget to update API keys in .env.local"
    else
        print_status "$GREEN" "✅ Environment file exists"
        
        # Check for placeholder values
        if grep -q "your_.*_key_here" "$env_file"; then
            print_status "$YELLOW" "⚠️  Environment file contains placeholder API keys"
            print_status "$CYAN" "Please update the following in .env.local:"
            grep "your_.*_key_here" "$env_file" | sed 's/^/    /'
        else
            print_status "$GREEN" "✅ No placeholder values detected"
        fi
    fi
}

check_and_fix_python_env() {
    print_header "Python Environment"
    
    if [[ ! -d "${SCRIPT_DIR}/venv" ]]; then
        print_status "$RED" "❌ Virtual environment missing"
        print_status "$CYAN" "Creating virtual environment..."
        
        python3 -m venv "${SCRIPT_DIR}/venv"
        source "${SCRIPT_DIR}/venv/bin/activate"
        pip install --upgrade pip setuptools wheel
        pip install -e .[dev,monitoring,ai-extended]
        
        print_status "$GREEN" "✅ Virtual environment created and configured"
    else
        print_status "$GREEN" "✅ Virtual environment exists"
        
        # Check if it's properly configured
        source "${SCRIPT_DIR}/venv/bin/activate"
        
        if ! pip show fastapi >/dev/null 2>&1; then
            print_status "$YELLOW" "⚠️  Dependencies missing, installing..."
            pip install --upgrade pip
            pip install -e .[dev,monitoring,ai-extended]
            print_status "$GREEN" "✅ Dependencies installed"
        else
            print_status "$GREEN" "✅ Dependencies are installed"
        fi
    fi
}

check_and_fix_docker_services() {
    print_header "Docker Services"
    
    # Try to start services
    print_status "$CYAN" "Starting required services..."
    if docker compose up -d postgres redis 2>/dev/null; then
        print_status "$GREEN" "✅ Services started"
    else
        print_status "$RED" "❌ Failed to start services"
        print_status "$CYAN" "Checking for issues..."
        
        # Check if images need to be pulled
        if ! docker images | grep -q "pgvector/pgvector"; then
            print_status "$CYAN" "Pulling PostgreSQL image..."
            docker pull pgvector/pgvector:pg15
        fi
        
        if ! docker images | grep -q "redis.*alpine"; then
            print_status "$CYAN" "Pulling Redis image..."
            docker pull redis:7-alpine
        fi
        
        # Try again
        if docker compose up -d postgres redis; then
            print_status "$GREEN" "✅ Services started after pulling images"
        else
            print_status "$RED" "❌ Still unable to start services. Check docker compose logs"
            return
        fi
    fi
    
    # Wait for services to be ready
    print_status "$CYAN" "Waiting for services to be ready..."
    local max_attempts=30
    local postgres_ready=false
    local redis_ready=false
    
    for ((i=1; i<=max_attempts; i++)); do
        if ! $postgres_ready && docker compose exec -T postgres pg_isready -U leanvibe_user -d leanvibe_agent_hive >/dev/null 2>&1; then
            print_status "$GREEN" "✅ PostgreSQL is ready"
            postgres_ready=true
        fi
        
        if ! $redis_ready && docker compose exec -T redis redis-cli ping >/dev/null 2>&1; then
            print_status "$GREEN" "✅ Redis is ready"
            redis_ready=true
        fi
        
        if $postgres_ready && $redis_ready; then
            break
        fi
        
        sleep 2
    done
    
    if ! $postgres_ready; then
        print_status "$RED" "❌ PostgreSQL failed to become ready"
    fi
    
    if ! $redis_ready; then
        print_status "$RED" "❌ Redis failed to become ready"
    fi
}

check_and_fix_database() {
    print_header "Database Setup"
    
    if ! docker compose exec -T postgres pg_isready -U leanvibe_user -d leanvibe_agent_hive >/dev/null 2>&1; then
        print_status "$RED" "❌ Database not accessible"
        return
    fi
    
    # Check if tables exist
    local table_count=$(docker compose exec -T postgres psql -U leanvibe_user -d leanvibe_agent_hive -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';" 2>/dev/null | tr -d ' \n' || echo "0")
    
    if [[ "$table_count" -eq 0 ]]; then
        print_status "$YELLOW" "⚠️  No tables found, running migrations..."
        
        source "${SCRIPT_DIR}/venv/bin/activate"
        if alembic upgrade head; then
            print_status "$GREEN" "✅ Database migrations completed"
        else
            print_status "$RED" "❌ Migration failed"
        fi
    else
        print_status "$GREEN" "✅ Database has $table_count tables"
    fi
}

run_comprehensive_test() {
    print_header "Comprehensive Test"
    
    print_status "$CYAN" "Running full health check..."
    if ./health-check.sh >/dev/null 2>&1; then
        print_status "$GREEN" "✅ Health check passed"
    else
        print_status "$YELLOW" "⚠️  Health check found issues. Run './health-check.sh' for details"
    fi
    
    print_status "$CYAN" "Testing API startup..."
    source "${SCRIPT_DIR}/venv/bin/activate"
    
    # Start API in background for testing
    uvicorn app.main:app --host 127.0.0.1 --port 8001 &
    local api_pid=$!
    
    # Wait for API to start
    sleep 5
    
    if curl -s http://127.0.0.1:8001/health >/dev/null; then
        print_status "$GREEN" "✅ API starts successfully"
    else
        print_status "$RED" "❌ API failed to start"
    fi
    
    # Kill test API
    kill $api_pid 2>/dev/null || true
    wait $api_pid 2>/dev/null || true
}

provide_final_report() {
    print_header "Troubleshooting Complete"
    
    print_status "$CYAN" "Summary of actions taken:"
    if [[ -s "$TROUBLESHOOT_LOG" ]]; then
        print_status "$NC" "  • Detailed logs saved to: $TROUBLESHOOT_LOG"
    fi
    
    print_status "$CYAN" "Next steps:"
    print_status "$NC" "  1. Run './validate-setup.sh' for quick validation"
    print_status "$NC" "  2. Run './health-check.sh' for detailed health check"
    print_status "$NC" "  3. Update API keys in .env.local if needed"
    print_status "$NC" "  4. Start the system with './start.sh' or 'make dev'"
    echo ""
    
    print_status "$GREEN" "🎉 Troubleshooting completed!"
    print_status "$BLUE" "If issues persist, check the logs or run './setup.sh' for a fresh setup."
}

main() {
    # Clear log
    > "$TROUBLESHOOT_LOG"
    
    print_status "$PURPLE" "🔧 LeanVibe Agent Hive 2.0 - Automated Troubleshooter"
    print_status "$PURPLE" "======================================================"
    print_status "$CYAN" "Diagnosing and fixing common deployment issues..."
    echo ""
    
    cd "$SCRIPT_DIR"
    
    check_and_fix_docker
    check_and_fix_ports
    check_and_fix_permissions
    check_and_fix_environment
    check_and_fix_python_env
    check_and_fix_docker_services
    check_and_fix_database
    run_comprehensive_test
    provide_final_report
}

# Handle interruption
trap 'print_status "$RED" "Troubleshooting interrupted"; exit 1' INT TERM

main "$@"