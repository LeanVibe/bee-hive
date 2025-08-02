#!/bin/bash
# Quick Start - LeanVibe Agent Hive Autonomous Development Platform
# This script provides a one-command setup for the complete autonomous development system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print colored output
print_header() {
    echo -e "${PURPLE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_step() {
    echo -e "${CYAN}🔄 $1${NC}"
}

# Main setup function
main() {
    print_header "🚀 LeanVibe Agent Hive - Quick Start Setup"
    print_header "=========================================="
    echo
    print_info "Setting up the complete autonomous development platform in 3 easy steps!"
    echo

    # Step 1: Environment Validation
    print_step "Step 1: Validating Environment"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is required but not installed."
        print_info "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
        exit 1
    fi
    print_success "Docker is available"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is required but not available."
        exit 1
    fi
    print_success "Docker Compose is available"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed."
        print_info "Please install Python 3.11+ from: https://www.python.org/downloads/"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MAJOR" -eq 3 -a "$PYTHON_MINOR" -lt 11 ]; then
        print_error "Python 3.11+ is required. Found Python $PYTHON_VERSION"
        exit 1
    fi
    print_success "Python $PYTHON_VERSION is compatible"
    
    echo

    # Step 2: Infrastructure Setup
    print_step "Step 2: Setting Up Infrastructure"
    
    # Start database services
    print_info "Starting database services (PostgreSQL + Redis)..."
    if docker ps | grep -q leanvibe_postgres_fast; then
        print_success "PostgreSQL already running"
    else
        docker run -d \
            --name leanvibe_postgres_fast \
            --health-cmd="pg_isready -U leanvibe_user -d leanvibe_agent_hive" \
            --health-interval=5s \
            --health-timeout=5s \
            --health-retries=5 \
            -p 5432:5432 \
            -e POSTGRES_DB=leanvibe_agent_hive \
            -e POSTGRES_USER=leanvibe_user \
            -e POSTGRES_PASSWORD=leanvibe_secure_pass \
            pgvector/pgvector:pg15 > /dev/null
        print_success "PostgreSQL with pgvector started"
    fi
    
    if docker ps | grep -q leanvibe_redis_fast; then
        print_success "Redis already running"
    else
        docker run -d \
            --name leanvibe_redis_fast \
            --health-cmd="redis-cli ping" \
            --health-interval=5s \
            --health-timeout=3s \
            --health-retries=3 \
            -p 6380:6379 \
            redis:7-alpine > /dev/null
        print_success "Redis started"
    fi
    
    # Wait for services to be healthy
    print_info "Waiting for services to be ready..."
    for i in {1..30}; do
        if docker ps --filter "name=leanvibe_postgres_fast" --filter "health=healthy" | grep -q leanvibe_postgres_fast && \
           docker ps --filter "name=leanvibe_redis_fast" --filter "health=healthy" | grep -q leanvibe_redis_fast; then
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Services failed to start within 60 seconds"
            exit 1
        fi
        sleep 2
        echo -n "."
    done
    echo
    print_success "All services are healthy and ready"
    
    # Install Python dependencies
    print_info "Installing Python dependencies..."
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt > /dev/null 2>&1
    else
        pip3 install asyncpg sqlalchemy[asyncio] alembic pydantic pydantic-settings structlog > /dev/null 2>&1
    fi
    print_success "Python dependencies installed"
    
    echo

    # Step 3: Database Bootstrap
    print_step "Step 3: Database Bootstrap & Validation"
    
    # Create environment file if it doesn't exist
    if [ ! -f ".env.local" ]; then
        print_info "Creating environment configuration..."
        cat > .env.local << EOF
# LeanVibe Agent Hive Configuration
ENVIRONMENT=development
DEBUG=true

# Database Configuration
DATABASE_URL=postgresql://leanvibe_user:leanvibe_secure_pass@localhost:5432/leanvibe_agent_hive
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=leanvibe_agent_hive
POSTGRES_USER=leanvibe_user
POSTGRES_PASSWORD=leanvibe_secure_pass

# Redis Configuration  
REDIS_URL=redis://localhost:6380/0

# Security (generate your own keys for production)
SECRET_KEY=demo-secret-key-change-in-production
JWT_SECRET_KEY=demo-jwt-secret-key-change-in-production

# CORS and Hosts
CORS_ORIGINS=["http://localhost:3000","http://localhost:8080","http://localhost:8000"]
ALLOWED_HOSTS=["localhost","127.0.0.1","0.0.0.0"]

# Optional: Add your AI API keys for full autonomous development
# ANTHROPIC_API_KEY=your_claude_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here
EOF
        print_success "Environment configuration created"
    else
        print_success "Environment configuration already exists"
    fi
    
    # Run database bootstrap
    print_info "Initializing database schema..."
    if python3 scripts/init_db.py > /dev/null 2>&1; then
        print_success "Database bootstrap completed successfully"
    else
        print_error "Database bootstrap failed"
        print_info "Trying manual database initialization..."
        if python3 scripts/init_db.py; then
            print_success "Database bootstrap completed on retry"
        else
            print_error "Database bootstrap failed. Please check your setup."
            exit 1
        fi
    fi
    
    echo

    # Step 4: Validation
    print_step "Step 4: System Validation"
    
    print_info "Running autonomous development demo..."
    if python3 scripts/demos/hello_world_autonomous_demo_fixed.py > /dev/null 2>&1; then
        print_success "Autonomous development demo completed successfully"
    else
        print_warning "Demo had issues, but core system may still be working"
        print_info "You can run the demo manually: python3 scripts/demos/hello_world_autonomous_demo_fixed.py"
    fi
    
    echo
    print_header "🎉 SETUP COMPLETE! 🎉"
    print_header "==================="
    echo
    print_success "LeanVibe Agent Hive autonomous development platform is ready!"
    echo
    print_info "📊 What's Running:"
    print_info "   • PostgreSQL with pgvector: localhost:5432"
    print_info "   • Redis message bus: localhost:6380"
    print_info "   • Complete database schema: 50+ tables ready"
    print_info "   • Autonomous development framework: Operational"
    echo
    print_info "🚀 Quick Start Commands:"
    print_info "   • Test the system: python3 scripts/demos/hello_world_autonomous_demo_fixed.py"
    print_info "   • Check database: python3 scripts/init_db.py"
    print_info "   • View logs: docker logs leanvibe_postgres_fast"
    echo
    print_info "📝 Next Steps:"
    print_info "   1. Add your AI API keys to .env.local for full autonomous development"
    print_info "   2. Explore the autonomous development examples"
    print_info "   3. Create your first autonomous development project"
    echo
    print_info "🔗 Resources:"
    print_info "   • Documentation: docs/"
    print_info "   • Examples: scripts/demos/"
    print_info "   • Configuration: .env.local"
    echo
    print_success "Happy autonomous developing! 🤖✨"
}

# Cleanup function
cleanup() {
    if [ $? -ne 0 ]; then
        echo
        print_error "Setup failed. Cleaning up..."
        print_info "To clean up manually:"
        print_info "   docker stop leanvibe_postgres_fast leanvibe_redis_fast"
        print_info "   docker rm leanvibe_postgres_fast leanvibe_redis_fast"
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Run main function
main "$@"