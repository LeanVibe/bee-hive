# Quick Setup Guide

Get LeanVibe Agent Hive running in under 2 minutes with our optimized setup options. Choose the method that best fits your workflow and technical requirements.

## 🎯 Setup Options Overview

<div class="setup-comparison">
  <div class="comparison-table">
    <table>
      <thead>
        <tr>
          <th>Method</th>
          <th>Time</th>
          <th>Prerequisites</th>
          <th>Best For</th>
        </tr>
      </thead>
      <tbody>
        <tr class="recommended">
          <td><strong>DevContainer</strong></td>
          <td>&lt;2 min</td>
          <td>VS Code + Docker</td>
          <td>Zero configuration</td>
        </tr>
        <tr>
          <td><strong>Professional</strong></td>
          <td>&lt;5 min</td>
          <td>Docker + Make</td>
          <td>Production setup</td>
        </tr>
        <tr>
          <td><strong>Sandbox</strong></td>
          <td>0 min</td>
          <td>Web browser</td>
          <td>Quick evaluation</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

## 🎯 Option 1: DevContainer (Recommended)

**Perfect for developers who want zero-configuration autonomous development**

The DevContainer approach provides a complete, pre-configured development environment that works identically across all platforms.

### Prerequisites
- [VS Code](https://code.visualstudio.com/) installed
- [DevContainers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/LeanVibe/bee-hive.git
   cd bee-hive
   ```

2. **Open in VS Code**
   ```bash
   code .
   ```

3. **Reopen in Container**
   - VS Code will detect the DevContainer configuration
   - Click **"Reopen in Container"** when prompted
   - Or use Command Palette: `Dev Containers: Reopen in Container`

4. **Add your API key**
   ```bash
   echo "ANTHROPIC_API_KEY=your_key_here" >> .env.local
   ```

5. **Verify setup**
   ```bash
   make health
   ```

::: tip Success Indicator
You should see all services running and health checks passing. The autonomous development system is now ready at http://localhost:8000
:::

### DevContainer Features

- **Pre-installed tools**: Python, Node.js, Docker, PostgreSQL, Redis
- **VS Code extensions**: Python, Docker, YAML, Markdown, and more
- **Port forwarding**: Automatic port mapping for all services
- **Consistent environment**: Identical setup across all platforms
- **Integrated terminal**: Full terminal access within the container

## ⚡ Option 2: Professional Setup

**Ideal for production environments and advanced users who want full control**

This method provides maximum flexibility and is designed for production deployments.

### Prerequisites
- Docker and Docker Compose
- Python 3.12+
- Make (optional but recommended)
- Git

### Setup Steps

1. **Clone and enter directory**
   ```bash
   git clone https://github.com/LeanVibe/bee-hive.git
   cd bee-hive
   ```

2. **Run automated setup**
   ```bash
   make setup
   ```
   
   Or manually:
   ```bash
   ./setup.sh
   ```

3. **Configure environment**
   ```bash
   echo "ANTHROPIC_API_KEY=your_key_here" >> .env.local
   ```

4. **Start services**
   ```bash
   make start
   ```

5. **Verify installation**
   ```bash
   make health
   curl http://localhost:8000/health
   ```

### Professional Setup Features

- **Full system control**: Direct access to all configuration
- **Production ready**: Optimized for enterprise deployment
- **Monitoring included**: Prometheus and Grafana dashboards
- **Security hardened**: Enterprise-grade security configuration
- **Scalable**: Horizontal scaling capabilities

## 🎮 Option 3: Sandbox Mode

**Try autonomous development immediately without any installation**

Perfect for evaluation, demos, and learning without local setup.

### Access Sandbox

<div class="sandbox-access">
  <a href="/demo" target="_blank" class="sandbox-button">
    🎮 Launch Sandbox Environment
  </a>
  <p>Opens in a new tab with full interactive environment</p>
</div>

### Sandbox Features

- **No installation required**: Runs entirely in your browser
- **Full functionality**: Complete autonomous development capabilities
- **Interactive tutorials**: Guided learning experience
- **Shareable sessions**: Share your experiments with others
- **Reset anytime**: Fresh environment with one click

## 🔧 Post-Setup Configuration

Once you have LeanVibe Agent Hive running, complete these essential configuration steps:

### 1. API Key Configuration

You'll need an Anthropic API key for autonomous agents:

```bash
# Add your API key to environment
echo "ANTHROPIC_API_KEY=your_key_here" >> .env.local

# Restart services to pick up the key
make restart
```

::: warning Security Note
Never commit API keys to version control. The `.env.local` file is automatically ignored by Git.
:::

### 2. Verify System Health

```bash
# Comprehensive health check
make health

# Quick API check
curl http://localhost:8000/health

# Check all services
make ps
```

Expected output:
```json
{
  "status": "healthy",
  "services": {
    "api": "running",
    "database": "ready",
    "redis": "connected",
    "agents": "initialized"
  },
  "version": "2.0.0",
  "uptime": "00:02:34"
}
```

### 3. Access Key Interfaces

- **API Documentation**: http://localhost:8000/docs
- **System Health**: http://localhost:8000/health  
- **Web Dashboard**: http://localhost:3000 (if enabled)
- **Monitoring**: http://localhost:9090 (Prometheus)

## 🎯 Your First Demo

Test your setup with the autonomous development demo:

<InteractiveGuide />

```bash
# Run the autonomous development demo
python scripts/demos/autonomous_development_demo.py

# Or use the make command
make demo
```

This demo will:
1. Initialize multiple AI agents
2. Create a sample project
3. Demonstrate multi-agent coordination
4. Show real-time progress monitoring

## 🔍 Troubleshooting Common Issues

### Issue: Docker containers not starting

**Symptoms**: Services fail to start, port conflicts
**Solution**:
```bash
# Stop any conflicting services
make stop
docker system prune -f

# Restart with clean state
make start
```

### Issue: API key not recognized

**Symptoms**: "Authentication failed" errors
**Solution**:
```bash
# Verify API key format
echo $ANTHROPIC_API_KEY

# Re-add key and restart
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env.local
make restart
```

### Issue: Port conflicts

**Symptoms**: "Port already in use" errors
**Solution**:
```bash
# Check what's using the ports
netstat -tulpn | grep :8000
netstat -tulpn | grep :5432

# Kill conflicting processes or change ports in docker-compose.yml
```

### Issue: Slow performance

**Symptoms**: Long response times, timeouts
**Solution**:
```bash
# Check system resources
make health
docker stats

# Optimize for your system
make setup-minimal  # For resource-constrained systems
```

## 🚀 Next Steps

With LeanVibe Agent Hive successfully running, you're ready to:

1. **[Try Your First Demo](/learn/getting-started/first-demo)** - Run an autonomous development workflow
2. **[Learn Core Concepts](/learn/getting-started/concepts)** - Understand agents and coordination  
3. **[Explore Custom Commands](/learn/fundamentals/commands)** - Create your own workflows
4. **[Join the Community](/community/)** - Connect with other developers

## 📚 Additional Resources

- **[Deployment Guide](/enterprise/deployment/)** - Production deployment strategies
- **[API Reference](/api/)** - Complete API documentation
- **[Troubleshooting Guide](/learn/advanced/troubleshooting)** - Advanced problem solving
- **[Community Support](/community/support)** - Get help from the community

---

::: tip Congratulations! 🎉
You now have a working autonomous development environment. The future of software development is at your fingertips!
:::

<style>
.setup-comparison {
  margin: 2rem 0;
}

.comparison-table table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
}

.comparison-table th,
.comparison-table td {
  padding: 1rem;
  text-align: left;
  border-bottom: 1px solid var(--vp-c-border);
}

.comparison-table th {
  background: var(--vp-c-bg-soft);
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.comparison-table tr.recommended {
  background: rgba(99, 102, 241, 0.05);
  border-left: 4px solid var(--lv-primary);
}

.sandbox-access {
  text-align: center;
  margin: 2rem 0;
  padding: 2rem;
  background: var(--lv-glass-bg);
  backdrop-filter: blur(10px);
  border: 1px solid var(--lv-glass-border);
  border-radius: 12px;
}

.sandbox-button {
  display: inline-block;
  padding: 1rem 2rem;
  background: var(--lv-gradient-primary);
  color: white;
  text-decoration: none;
  border-radius: 12px;
  font-size: 1.1rem;
  font-weight: 600;
  transition: all 0.3s ease;
  margin-bottom: 1rem;
}

.sandbox-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
}

.sandbox-access p {
  color: var(--vp-c-text-2);
  margin: 0;
}

@media (max-width: 768px) {
  .comparison-table table {
    font-size: 0.875rem;
  }
  
  .comparison-table th,
  .comparison-table td {
    padding: 0.75rem 0.5rem;
  }
}
</style>