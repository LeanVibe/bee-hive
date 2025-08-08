# Sandbox to Production Migration Guide

## Overview
This guide helps you migrate from LeanVibe's sandbox demonstration mode to full production deployment with real API services.

## Quick Migration Checklist

### ✅ 1. Obtain Required API Keys
- **Anthropic API Key** (Required)  
  - Sign up at: https://console.anthropic.com/
  - Create API key in dashboard
  - Ensure sufficient credits/billing setup
  
- **OpenAI API Key** (Optional - for embeddings)
  - Sign up at: https://platform.openai.com/
  - Create API key in dashboard
  - Used for semantic memory and embeddings

- **GitHub Token** (Optional - for repository integration)
  - Generate at: https://github.com/settings/tokens
  - Required scopes: `repo`, `workflow`, `write:packages`

### ✅ 2. Configure Environment Variables

Create or update your `.env.local` file:

```bash
# Production Configuration
SANDBOX_MODE=false
SANDBOX_DEMO_MODE=false

# Required API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Optional API Keys
OPENAI_API_KEY=your_openai_api_key_here
GITHUB_TOKEN=your_github_token_here

# Database Configuration (PostgreSQL recommended for production)
DATABASE_URL=postgresql://user:password@localhost:5432/leanvibe_prod
REDIS_URL=redis://localhost:6379/0

# Security Configuration
SECRET_KEY=your_secure_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_key_here

# Production Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
```

### ✅ 3. Database Migration

**From SQLite (sandbox) to PostgreSQL (production):**

```bash
# 1. Install PostgreSQL
# Ubuntu/Debian:
sudo apt install postgresql postgresql-contrib

# macOS:
brew install postgresql

# 2. Create production database
sudo -u postgres createdb leanvibe_prod
sudo -u postgres createuser -s leanvibe_user

# 3. Run migrations
alembic upgrade head

# 4. Verify database connection
python -c "from app.core.database import get_session; print('Database connected successfully')"
```

### ✅ 4. Redis Configuration

**Production Redis setup:**

```bash
# Install Redis
# Ubuntu/Debian:
sudo apt install redis-server

# macOS:
brew install redis

# Start Redis service
sudo systemctl start redis-server  # Linux
brew services start redis          # macOS

# Test Redis connection
redis-cli ping  # Should return "PONG"
```

### ✅ 5. Validate Production Configuration

Run the configuration validator:

```bash
# Check all production requirements
python scripts/validate_production_config.py

# Expected output:
# ✅ Anthropic API key configured and valid
# ✅ Database connection successful
# ✅ Redis connection successful
# ✅ All security settings configured
# ✅ Production mode ready
```

### ✅ 6. Performance Testing

Test production performance:

```bash
# Run performance benchmarks
python scripts/benchmark_production_performance.py

# Expected results:
# - API response time: <2 seconds
# - Database query time: <500ms
# - Redis operations: <10ms
# - Concurrent agent handling: 10+ agents
```

## Migration Commands

### Quick Migration Script

```bash
#!/bin/bash
# Quick migration from sandbox to production

echo "🚀 Migrating from Sandbox to Production..."

# 1. Check environment file
if [ ! -f ".env.local" ]; then
    echo "❌ .env.local not found. Please create it with your API keys."
    exit 1
fi

# 2. Disable sandbox mode
sed -i 's/SANDBOX_MODE=true/SANDBOX_MODE=false/' .env.local
sed -i 's/SANDBOX_DEMO_MODE=true/SANDBOX_DEMO_MODE=false/' .env.local

# 3. Validate API keys
if grep -q "sandbox-mock" .env.local; then
    echo "❌ Mock API keys found. Please replace with real keys."
    exit 1
fi

# 4. Run database migrations
echo "📊 Running database migrations..."
alembic upgrade head

# 5. Start services
echo "🔥 Starting production services..."
docker-compose -f docker-compose.yml up -d postgres redis

# 6. Start application
echo "🚀 Starting LeanVibe in production mode..."
uvicorn app.main:app --host 0.0.0.0 --port 8000

echo "✅ Migration complete! LeanVibe is running in production mode."
```

## Feature Comparison

| Feature | Sandbox Mode | Production Mode |
|---------|--------------|-----------------|
| **AI Responses** | Mock/Pre-defined | Real Claude API |
| **Multi-Agent Coordination** | Simulated | Real AI agents |
| **Code Generation** | Template-based | AI-generated |
| **Database** | SQLite | PostgreSQL |
| **Performance** | Instant responses | 2-10 second AI calls |
| **Cost** | Free | API usage costs |
| **Scalability** | Demo purposes | Production ready |
| **Real Development** | No | Yes |

## Cost Considerations

### Anthropic API Costs
- **Claude 3.5 Sonnet**: ~$3 per 1M input tokens, ~$15 per 1M output tokens
- **Average autonomous development task**: 10,000-50,000 tokens
- **Estimated cost per task**: $0.20 - $2.00
- **Monthly usage (100 tasks)**: $20 - $200

### OpenAI API Costs (Optional)
- **Text Embeddings**: ~$0.10 per 1M tokens
- **Monthly embedding costs**: <$10 for typical usage

### Infrastructure Costs
- **PostgreSQL**: Free (self-hosted) or $20-50/month (managed)
- **Redis**: Free (self-hosted) or $15-30/month (managed)
- **Server/Hosting**: $50-200/month depending on scale

## Production Optimizations

### 1. API Rate Limiting
```python
# Configure rate limiting in .env.local
ANTHROPIC_RATE_LIMIT_RPM=1000
ANTHROPIC_RATE_LIMIT_TPM=50000
OPENAI_RATE_LIMIT_RPM=3000
OPENAI_RATE_LIMIT_TPM=150000
```

### 2. Caching Strategy
```python
# Enable production caching
REDIS_CACHE_ENABLED=true
CACHE_TTL_HOURS=24
EMBEDDING_CACHE_ENABLED=true
```

### 3. Monitoring & Alerting
```python
# Production monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
ALERT_WEBHOOK_URL=your_slack_webhook_url
```

### 4. Security Hardening
```python
# Production security
CORS_ORIGINS=["https://yourdomain.com"]
ALLOWED_HOSTS=["yourdomain.com"]
RATE_LIMITING_ENABLED=true
API_KEY_ROTATION_ENABLED=true
```

## Troubleshooting

### Common Issues

**1. API Key Authentication Errors**
```bash
# Test API keys
curl -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.anthropic.com/v1/messages
```

**2. Database Connection Errors**
```bash
# Test database connection
python -c "
import psycopg2
conn = psycopg2.connect('postgresql://user:pass@localhost/db')
print('Database connected')
conn.close()
"
```

**3. Redis Connection Errors**
```bash
# Test Redis connection
redis-cli -h localhost -p 6379 ping
```

**4. Performance Issues**
- Check API rate limits
- Monitor database query performance
- Verify Redis cache hit rates
- Review agent concurrency settings

### Getting Help

- **Documentation**: https://github.com/leanvibe/agent-hive/docs
- **Issues**: https://github.com/leanvibe/agent-hive/issues
- **Discord**: https://discord.gg/leanvibe
- **Email**: support@leanvibe.dev

## Advanced Production Features

### Multi-Environment Setup
```bash
# Development
ENVIRONMENT=development
DATABASE_URL=postgresql://localhost/leanvibe_dev

# Staging
ENVIRONMENT=staging  
DATABASE_URL=postgresql://staging-db/leanvibe_staging

# Production
ENVIRONMENT=production
DATABASE_URL=postgresql://prod-db/leanvibe_prod
```

### Horizontal Scaling
```bash
# Docker Compose scaling
docker-compose up --scale app=3 --scale worker=5

# Kubernetes deployment
kubectl apply -f k8s/production/
kubectl scale deployment leanvibe-app --replicas=3
```

### Backup & Recovery
```bash
# Database backups
pg_dump leanvibe_prod > backup_$(date +%Y%m%d).sql

# Redis backups  
redis-cli --rdb backup_$(date +%Y%m%d).rdb

# Automated backups
crontab -e
# Add: 0 2 * * * /usr/local/bin/backup_leanvibe.sh
```

## Success Metrics

After migration, monitor these key metrics:

- **Availability**: >99.9% uptime
- **Response Time**: <2 second average for AI operations
- **Error Rate**: <0.1% of requests
- **Cost per Task**: Within $0.20-$2.00 range
- **User Satisfaction**: Positive feedback on real AI capabilities

## Next Steps

1. **Set up monitoring and alerting**
2. **Configure automated backups** 
3. **Implement CI/CD pipeline**
4. **Scale based on usage patterns**
5. **Optimize costs based on actual usage**

---

🎉 **Congratulations!** You've successfully migrated from sandbox to production. Your LeanVibe Agent Hive is now ready for real autonomous AI development at scale.