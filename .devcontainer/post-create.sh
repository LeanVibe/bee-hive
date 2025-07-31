#!/bin/bash

# LeanVibe Agent Hive 2.0 - Dev Container Post-Create Script
# Runs after the container is created to set up the development environment

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 Setting up LeanVibe Agent Hive 2.0 development environment...${NC}"

# Create virtual environment if it doesn't exist
if [[ ! -d "/workspace/venv" ]]; then
    echo -e "${BLUE}📦 Creating Python virtual environment...${NC}"
    python3 -m venv /workspace/venv
fi

# Activate virtual environment
source /workspace/venv/bin/activate

# Upgrade pip and install dependencies
echo -e "${BLUE}📚 Installing Python dependencies...${NC}"
pip install --upgrade pip setuptools wheel
pip install -e .[dev,monitoring,ai-extended]

# Install pre-commit hooks
echo -e "${BLUE}🔧 Setting up pre-commit hooks...${NC}"
pre-commit install

# Create necessary directories
echo -e "${BLUE}📁 Creating project directories...${NC}"
mkdir -p /workspace/logs
mkdir -p /workspace/temp
mkdir -p /workspace/workspaces
mkdir -p /workspace/dev-state/checkpoints
mkdir -p /workspace/dev-state/repositories

# Set up git configuration if not already set
if [[ -z "$(git config --global user.name 2>/dev/null || true)" ]]; then
    echo -e "${YELLOW}⚠️  Git user not configured. Setting default values...${NC}"
    git config --global user.name "Developer"
    git config --global user.email "developer@leanvibe.dev"
    git config --global init.defaultBranch main
fi

# Create environment file if it doesn't exist
if [[ ! -f "/workspace/.env.local" ]]; then
    echo -e "${BLUE}⚙️  Creating development environment configuration...${NC}"
    
    # Generate secure random keys
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    
    cat > /workspace/.env.local << EOF
# LeanVibe Agent Hive 2.0 - Development Container Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
SECRET_KEY=$SECRET_KEY
JWT_SECRET_KEY=$JWT_SECRET

# Database (using docker-compose services)
DATABASE_URL=postgresql://leanvibe_user:leanvibe_secure_pass@postgres:5432/leanvibe_agent_hive

# Redis (using docker-compose services)
REDIS_URL=redis://:leanvibe_redis_pass@redis:6379/0

# AI Services (placeholder - update with real keys)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# GitHub Integration (optional)
GITHUB_TOKEN=your_github_token_here
BASE_URL=http://localhost:8000

# Development optimizations
MAX_CONCURRENT_AGENTS=5
CONTEXT_MAX_TOKENS=4000
METRICS_ENABLED=true
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://localhost:8000
EOF
fi

# Install frontend dependencies if frontend exists
if [[ -d "/workspace/frontend" ]]; then
    echo -e "${BLUE}🎨 Installing frontend dependencies...${NC}"
    cd /workspace/frontend
    npm install
    cd /workspace
fi

# Set up shell aliases and environment
echo -e "${BLUE}🐚 Configuring shell environment...${NC}"
cat >> ~/.zshrc << 'EOF'

# LeanVibe Agent Hive aliases
alias activate='source /workspace/venv/bin/activate'
alias health='/workspace/health-check.sh'
alias start='/workspace/start.sh'
alias stop='/workspace/stop.sh'
alias setup='/workspace/setup.sh'
alias logs='docker compose logs -f'
alias ps='docker compose ps'
alias test='pytest -v'
alias test-cov='pytest --cov=app --cov-report=html'
alias lint='ruff check . && black --check .'
alias format='black . && ruff --fix .'
alias mypy='mypy app'
alias dev='uvicorn app.main:app --reload --host 0.0.0.0 --port 8000'

# Git aliases
alias gst='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git pull'
alias gpo='git push origin'
alias gco='git checkout'
alias gb='git branch'
alias gl='git log --oneline -10'

# Docker aliases
alias dc='docker compose'
alias dcup='docker compose up -d'
alias dcdown='docker compose down'
alias dcps='docker compose ps'
alias dclogs='docker compose logs -f'

# Utility aliases
alias ll='ls -la'
alias la='ls -la'
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'

# Auto-activate virtual environment
if [[ -f "/workspace/venv/bin/activate" ]]; then
    source /workspace/venv/bin/activate
fi

# Set PYTHONPATH
export PYTHONPATH="/workspace:$PYTHONPATH"
EOF

# Wait for database to be ready and run migrations
echo -e "${BLUE}🗄️  Waiting for database and running migrations...${NC}"
timeout=60
counter=0

while ! docker compose exec -T postgres pg_isready -U leanvibe_user -d leanvibe_agent_hive >/dev/null 2>&1; do
    if [[ $counter -ge $timeout ]]; then
        echo -e "${YELLOW}⚠️  Database not ready after ${timeout}s. You may need to run migrations manually.${NC}"
        break
    fi
    sleep 2
    counter=$((counter + 2))
done

if docker compose exec -T postgres pg_isready -U leanvibe_user -d leanvibe_agent_hive >/dev/null 2>&1; then
    echo -e "${BLUE}🔄 Running database migrations...${NC}"
    alembic upgrade head || echo -e "${YELLOW}⚠️  Migration failed. You may need to run 'alembic upgrade head' manually.${NC}"
fi

echo -e "${GREEN}✅ Development environment setup complete!${NC}"
echo -e "${BLUE}🔧 Available commands:${NC}"
echo -e "  ${GREEN}health${NC}  - Check system health"
echo -e "  ${GREEN}start${NC}   - Start the application"
echo -e "  ${GREEN}test${NC}    - Run tests"
echo -e "  ${GREEN}lint${NC}    - Check code quality"
echo -e "  ${GREEN}dev${NC}     - Start development server"
echo ""
echo -e "${YELLOW}⚠️  Don't forget to update API keys in .env.local${NC}"