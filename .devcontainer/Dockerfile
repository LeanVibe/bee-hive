# LeanVibe Agent Hive 2.0 - Development Container
FROM mcr.microsoft.com/vscode/devcontainers/python:3.11-bullseye as base

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    tmux \
    postgresql-client \
    redis-tools \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    zsh \
    vim \
    nano \
    htop \
    tree \
    jq \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for frontend development
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Install Docker CLI (for Docker-in-Docker)
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce-cli docker-compose-plugin

# Create vscode user (if not exists)
RUN if ! id -u vscode > /dev/null 2>&1; then \
        groupadd --gid 1000 vscode \
        && useradd --uid 1000 --gid vscode --shell /bin/zsh --create-home vscode; \
    fi

# Add vscode user to docker group
RUN groupadd -f docker && usermod -aG docker vscode

# Development stage
FROM base as development

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip install -e .[dev,monitoring,ai-extended]

# Install pre-commit hooks
RUN pip install pre-commit

# Switch to vscode user
USER vscode

# Set up oh-my-zsh
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Configure shell
RUN echo 'export PATH="/workspace/venv/bin:$PATH"' >> ~/.zshrc \
    && echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> ~/.zshrc \
    && echo 'alias ll="ls -la"' >> ~/.zshrc \
    && echo 'alias la="ls -la"' >> ~/.zshrc \
    && echo 'alias ..="cd .."' >> ~/.zshrc \
    && echo 'alias ...="cd ../.."' >> ~/.zshrc \
    && echo 'alias gs="git status"' >> ~/.zshrc \
    && echo 'alias gp="git pull"' >> ~/.zshrc \
    && echo 'alias gc="git commit"' >> ~/.zshrc \
    && echo 'alias ga="git add"' >> ~/.zshrc \
    && echo 'alias health="./health-check.sh"' >> ~/.zshrc \
    && echo 'alias start="./start.sh"' >> ~/.zshrc \
    && echo 'alias stop="./stop.sh"' >> ~/.zshrc \
    && echo 'alias logs="docker compose logs -f"' >> ~/.zshrc \
    && echo 'alias test="pytest"' >> ~/.zshrc \
    && echo 'alias lint="ruff check . && black --check ."' >> ~/.zshrc \
    && echo 'alias format="black . && ruff --fix ."' >> ~/.zshrc

# Expose common ports
EXPOSE 8000 3000 5432 6380 9090

# Default command
CMD ["sleep", "infinity"]