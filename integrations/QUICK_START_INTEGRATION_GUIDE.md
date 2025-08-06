# 🚀 LeanVibe Agent Hive Integration Quick Start Guide

**Get up and running with LeanVibe Agent Hive integrations in minutes, not hours.**

## 🎯 Choose Your Integration Path

### 🔧 Developer Workflow Integration (2-5 minutes)
**Best for:** Individual developers wanting AI-powered development assistance
- [VS Code Extension](#-vs-code-extension-2-minutes)
- [GitHub Actions](#-github-actions-5-minutes)

### 🐳 Container Deployment (5-10 minutes) 
**Best for:** Development teams wanting consistent environments
- [Docker Development](#-docker-development-5-minutes)
- [Docker Production](#-docker-production-10-minutes)

### ☁️ Cloud Platform Deployment (15-30 minutes)
**Best for:** Production deployments and enterprise scale
- [AWS One-Click Deploy](#-aws-one-click-deploy-15-minutes)
- [Kubernetes Deployment](#-kubernetes-deployment-20-minutes)

---

## 🔧 VS Code Extension (2 minutes)

### Prerequisites
- VS Code 1.74+
- LeanVibe Agent Hive running locally or accessible API endpoint

### Installation & Setup

1. **Install Extension:**
   ```bash
   # From VS Code Marketplace (recommended)
   ext install leanvibe.leanvibe-agent-hive
   
   # Or from local build
   cd integrations/vscode-extension
   npm install && npm run package
   code --install-extension leanvibe-agent-hive-1.0.0.vsix
   ```

2. **Configure Settings:**
   ```json
   // VS Code settings.json
   {
     "leanvibe.apiUrl": "http://localhost:8000",
     "leanvibe.autoStart": true,
     "leanvibe.enableCodeSuggestions": true,
     "leanvibe.maxAgents": 5
   }
   ```

3. **Start Using:**
   - Open any project with `.leanvibe/`, `pyproject.toml`, or `CLAUDE.md`
   - Press `Ctrl+Shift+A, S` to start Agent Hive
   - View agents in the sidebar
   - Right-click files for AI optimization

### ✅ Verification
- Agent Hive sidebar shows "Connected" status
- Can see active agents and tasks
- Code suggestions appear in supported files

---

## ⚡ GitHub Actions (5 minutes)

### Prerequisites
- GitHub repository
- Anthropic API key
- LeanVibe Agent Hive instance (can be started by workflow)

### Setup Steps

1. **Copy Workflow File:**
   ```bash
   mkdir -p .github/workflows
   cp integrations/github-actions/leanvibe-autonomous-development.yml .github/workflows/
   ```

2. **Configure Repository Secrets:**
   ```bash
   # In GitHub repository settings → Secrets and variables → Actions
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   LEANVIBE_API_URL=http://localhost:8000  # Optional for local development
   ```

3. **Trigger Workflow:**
   ```yaml
   # Manual trigger with options
   on:
     workflow_dispatch:
       inputs:
         autonomous_mode:
           description: 'Enable fully autonomous development mode'
           default: 'false'
           type: boolean
         max_agents:
           description: 'Maximum number of concurrent agents'
           default: '3'
   ```

4. **Commit and Push:**
   ```bash
   git add .github/workflows/leanvibe-autonomous-development.yml
   git commit -m "Add LeanVibe Agent Hive autonomous development workflow"
   git push
   ```

### ✅ Verification
- Workflow appears in Actions tab
- Can trigger manually with custom parameters
- Quality gates pass (tests, security, coverage)
- Autonomous development agents deploy successfully

---

## 🐳 Docker Development (5 minutes)

### Prerequisites
- Docker 20.10+
- Docker Compose v2
- 4GB RAM available for containers

### Quick Start

1. **Clone and Navigate:**
   ```bash
   git clone https://github.com/leanvibe-dev/bee-hive.git
   cd bee-hive
   ```

2. **Set Environment Variables:**
   ```bash
   # Create .env.local file
   cat > .env.local << EOF
   ANTHROPIC_API_KEY=your_api_key_here
   POSTGRES_PASSWORD=secure_db_password
   REDIS_PASSWORD=secure_redis_password
   SECRET_KEY=your_secure_secret_key
   EOF
   ```

3. **Start Development Environment:**
   ```bash
   # Use optimized multi-stage build
   docker-compose -f integrations/docker/docker-compose.yml up -d
   
   # Or use the fast setup
   make setup  # Automated 5-12 minute setup
   ```

4. **Access Services:**
   ```bash
   # Agent Hive API
   curl http://localhost:8000/health
   
   # Dashboard
   open http://localhost:3000
   
   # Redis Insight
   open http://localhost:8001
   
   # pgAdmin
   open http://localhost:5050
   ```

### ✅ Verification
- All containers healthy: `docker ps`
- API responds: `curl http://localhost:8000/health`
- Can create and manage agents via API
- Dashboard shows real-time agent status

---

## 🐳 Docker Production (10 minutes)

### Prerequisites
- Docker Swarm or Kubernetes
- Load balancer (nginx/traefik)
- SSL certificates
- Monitoring setup

### Production Deployment

1. **Prepare Production Configuration:**
   ```bash
   # Copy production compose file
   cp integrations/docker/docker-compose.production.yml docker-compose.prod.yml
   
   # Configure production environment
   cat > .env.prod << EOF
   ENVIRONMENT=production
   POSTGRES_PASSWORD=$(openssl rand -hex 32)
   REDIS_PASSWORD=$(openssl rand -hex 32)
   SECRET_KEY=$(openssl rand -hex 64)
   ANTHROPIC_API_KEY=your_production_api_key
   GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 16)
   EOF
   ```

2. **Start Production Stack:**
   ```bash
   # Deploy with production optimizations
   docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d
   ```

3. **Configure Reverse Proxy:**
   ```nginx
   # nginx configuration example
   server {
       listen 443 ssl;
       server_name agent-hive.yourdomain.com;
       
       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

4. **Set Up Monitoring:**
   ```bash
   # Access monitoring dashboards
   open http://localhost:9090  # Prometheus
   open http://localhost:3000  # Grafana
   open http://localhost:9093  # Alertmanager
   ```

### ✅ Verification
- All production services healthy
- HTTPS access working
- Monitoring dashboards populated
- Backup processes running
- Performance metrics within targets

---

## ☁️ AWS One-Click Deploy (15 minutes)

### Prerequisites
- AWS CLI configured with appropriate permissions
- Domain name (optional)
- SSL certificate in AWS Certificate Manager (optional)

### Deployment Steps

1. **Prepare Parameters:**
   ```bash
   # Create parameters file
   cat > aws-parameters.json << EOF
   [
     {
       "ParameterKey": "Environment",
       "ParameterValue": "production"
     },
     {
       "ParameterKey": "AnthropicApiKey",
       "ParameterValue": "your_anthropic_api_key"
     },
     {
       "ParameterKey": "NotificationEmail",
       "ParameterValue": "admin@yourdomain.com"
     },
     {
       "ParameterKey": "InstanceType",
       "ParameterValue": "t3.large"
     },
     {
       "ParameterKey": "DomainName",
       "ParameterValue": "agent-hive.yourdomain.com"
     }
   ]
   EOF
   ```

2. **Deploy CloudFormation Stack:**
   ```bash
   aws cloudformation create-stack \
     --stack-name leanvibe-agent-hive-prod \
     --template-body file://integrations/aws/cloudformation/leanvibe-agent-hive.yml \
     --parameters file://aws-parameters.json \
     --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
     --on-failure ROLLBACK
   ```

3. **Monitor Deployment:**
   ```bash
   # Watch stack creation progress
   aws cloudformation describe-stacks \
     --stack-name leanvibe-agent-hive-prod \
     --query 'Stacks[0].StackStatus'
   
   # Get stack outputs when complete
   aws cloudformation describe-stacks \
     --stack-name leanvibe-agent-hive-prod \
     --query 'Stacks[0].Outputs'
   ```

4. **Configure DNS (if using custom domain):**
   ```bash
   # Get ALB DNS name from stack outputs
   ALB_DNS=$(aws cloudformation describe-stacks \
     --stack-name leanvibe-agent-hive-prod \
     --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue' \
     --output text)
   
   # Create CNAME record: agent-hive.yourdomain.com → $ALB_DNS
   ```

### ✅ Verification
- CloudFormation stack in `CREATE_COMPLETE` status
- Application accessible via load balancer URL
- Database and Redis endpoints responding
- Auto-scaling group healthy
- CloudWatch alarms configured

---

## 🚢 Kubernetes Deployment (20 minutes)

### Prerequisites
- Kubernetes cluster (1.21+)
- Helm 3.7+
- kubectl configured
- Ingress controller (nginx/traefik)
- Storage class available

### Deployment Steps

1. **Add Helm Repository:**
   ```bash
   # Add LeanVibe repository (when published)
   helm repo add leanvibe https://charts.leanvibe.dev
   helm repo update
   
   # Or use local charts
   cd integrations/kubernetes/helm
   ```

2. **Create Values File:**
   ```yaml
   # values.production.yaml
   replicaCount: 3
   
   image:
     tag: "2.0.0"
   
   ingress:
     enabled: true
     className: "nginx"
     hosts:
       - host: agent-hive.yourdomain.com
         paths:
           - path: /
             pathType: Prefix
     tls:
       - secretName: agent-hive-tls
         hosts:
           - agent-hive.yourdomain.com
   
   postgresql:
     auth:
       postgresPassword: "secure_postgres_password"
       password: "secure_app_password"
     primary:
       persistence:
         size: 50Gi
   
   redis:
     auth:
       password: "secure_redis_password"
   
   secrets:
     create: true
     data:
       anthropic-api-key: "your_anthropic_api_key"
       secret-key: "your_secure_secret_key"
   
   monitoring:
     enabled: true
     prometheus:
       enabled: true
     grafana:
       enabled: true
   
   autoscaling:
     enabled: true
     minReplicas: 3
     maxReplicas: 20
     targetCPUUtilizationPercentage: 70
   ```

3. **Deploy with Helm:**
   ```bash
   # Create namespace
   kubectl create namespace leanvibe-agent-hive
   
   # Install with custom values
   helm install agent-hive ./leanvibe-agent-hive \
     --namespace leanvibe-agent-hive \
     --values values.production.yaml \
     --wait --timeout 10m
   ```

4. **Verify Deployment:**
   ```bash
   # Check pod status
   kubectl get pods -n leanvibe-agent-hive
   
   # Check services
   kubectl get svc -n leanvibe-agent-hive
   
   # Check ingress
   kubectl get ingress -n leanvibe-agent-hive
   
   # View logs
   kubectl logs -l app.kubernetes.io/name=leanvibe-agent-hive -n leanvibe-agent-hive
   ```

### ✅ Verification
- All pods in `Running` status
- Services have endpoints
- Ingress has assigned IP/hostname
- Application accessible via ingress URL
- Horizontal Pod Autoscaler active
- Monitoring dashboards available

---

## 🔧 Post-Deployment Configuration

### Security Hardening
```bash
# Rotate default passwords
# Enable HTTPS/TLS everywhere
# Configure network policies
# Set up proper RBAC (Kubernetes)
# Enable audit logging
```

### Performance Tuning
```bash
# Adjust resource limits based on workload
# Configure database connection pooling
# Optimize Redis memory settings
# Set up appropriate caching strategies
```

### Monitoring Setup
```bash
# Configure alerts for key metrics
# Set up log aggregation
# Enable distributed tracing
# Configure backup strategies
```

---

## ❗ Troubleshooting

### Common Issues & Solutions

**Connection Refused:**
```bash
# Check if service is running
curl http://localhost:8000/health
docker ps  # or kubectl get pods

# Verify network connectivity
telnet localhost 8000
```

**Authentication Errors:**
```bash
# Verify API keys are set correctly
echo $ANTHROPIC_API_KEY
kubectl get secret agent-hive-secrets -o yaml
```

**Performance Issues:**
```bash
# Check resource usage
docker stats  # or kubectl top pods
htop

# Monitor database performance
# Check Redis memory usage
```

**Deployment Failures:**
```bash
# Check logs
docker logs container_name
kubectl logs -l app=agent-hive

# Verify configuration
docker exec -it container env
kubectl describe pod pod_name
```

---

## 📞 Getting Help

### Quick Support Channels
- **GitHub Issues:** [Bug reports and feature requests](https://github.com/leanvibe-dev/bee-hive/issues)
- **Discord Community:** [Real-time chat support](https://discord.gg/leanvibe)
- **Documentation:** [Comprehensive guides](../docs/README.md)

### Enterprise Support
- **Email:** enterprise@leanvibe.dev
- **Custom Integration:** Available for enterprise clients
- **Training Programs:** Hands-on integration workshops

---

## 🎉 Success! What's Next?

### Immediate Next Steps
1. **Test Core Functionality:** Create agents, run tasks, monitor performance
2. **Customize Configuration:** Adjust settings for your use case
3. **Set Up Monitoring:** Configure alerts and dashboards
4. **Plan Scaling:** Prepare for growth and increased usage

### Advanced Integrations
- **Custom Agents:** Develop specialized agents for your domain
- **API Integration:** Connect with your existing tools and workflows
- **Advanced Orchestration:** Complex multi-agent coordination
- **Enterprise Features:** SSO, RBAC, audit logging

### Community Contribution
- **Share Your Setup:** Help others with similar configurations
- **Contribute Integrations:** Add support for new platforms
- **Report Issues:** Help improve the platform for everyone

**🚀 You're now ready to harness the power of autonomous development with LeanVibe Agent Hive!**