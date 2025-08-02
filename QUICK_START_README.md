# 🚀 LeanVibe Agent Hive - Quick Start Guide

**Get autonomous development running in under 5 minutes!**

## ⚡ One-Command Setup

```bash
./quick-start.sh
```

That's it! The script will:
- ✅ Validate your environment (Docker, Python 3.11+)
- ✅ Start PostgreSQL + Redis services  
- ✅ Bootstrap the complete database schema
- ✅ Run autonomous development validation
- ✅ Provide next steps and resources

## 📋 Prerequisites

- **Docker**: For database services
- **Python 3.11+**: For the autonomous development platform
- **5 minutes**: For complete setup

## 🎯 What You Get

After setup, you'll have a **complete autonomous development platform**:

### **Infrastructure Ready** 🛠️
- PostgreSQL with pgvector for semantic operations
- Redis for real-time agent coordination  
- 50+ database tables for multi-agent workflows
- Production-grade error handling and monitoring

### **Autonomous Development Framework** 🤖
- Multi-agent coordination system
- Task orchestration and workflow management
- AI-ready integration points
- Comprehensive logging and observability

### **Working Demonstrations** 🎮
- Hello World autonomous development demo
- Database connectivity validation
- End-to-end system integration tests

## 🚀 Quick Test

After setup, test the system:

```bash
python3 scripts/demos/hello_world_autonomous_demo_fixed.py
```

Expected output:
```
🎉 HELLO WORLD AUTONOMOUS DEVELOPMENT DEMO COMPLETE!
✅ Database Infrastructure: Fully operational
✅ Autonomous Workflow Framework: Ready
✅ Multi-Agent Coordination: Infrastructure complete
```

## 🔧 Manual Setup (Alternative)

If you prefer manual setup:

```bash
# 1. Start services
docker run -d --name leanvibe_postgres_fast -p 5432:5432 \
  -e POSTGRES_DB=leanvibe_agent_hive \
  -e POSTGRES_USER=leanvibe_user \
  -e POSTGRES_PASSWORD=leanvibe_secure_pass \
  pgvector/pgvector:pg15

docker run -d --name leanvibe_redis_fast -p 6380:6379 redis:7-alpine

# 2. Install dependencies
pip3 install asyncpg sqlalchemy[asyncio] alembic pydantic structlog

# 3. Bootstrap database
python3 scripts/init_db.py

# 4. Test system
python3 scripts/demos/hello_world_autonomous_demo_fixed.py
```

## 🎯 Next Steps

1. **Add AI Integration** 📡
   ```bash
   # Add to .env.local
   ANTHROPIC_API_KEY=your_claude_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

2. **Explore Examples** 📚
   ```bash
   ls scripts/demos/
   cat docs/getting-started.md
   ```

3. **Start Developing** 💻
   - Create autonomous development workflows
   - Build multi-agent coordination projects
   - Integrate with your existing development tools

## 📊 System Health

Check system status:

```bash
# Database health
python3 scripts/init_db.py

# Service status  
docker ps | grep leanvibe

# Demo validation
python3 scripts/demos/hello_world_autonomous_demo_fixed.py
```

## 🆘 Troubleshooting

**Services won't start?**
```bash
docker stop leanvibe_postgres_fast leanvibe_redis_fast
docker rm leanvibe_postgres_fast leanvibe_redis_fast
./quick-start.sh
```

**Database connection issues?**
```bash
# Check if PostgreSQL is ready
docker logs leanvibe_postgres_fast | tail -10

# Verify database exists
docker exec leanvibe_postgres_fast psql -U leanvibe_user -d leanvibe_agent_hive -c "SELECT 1;"
```

**Python dependency issues?**
```bash
# Update pip and retry
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```

## 🏆 Success Criteria

You'll know setup is successful when:
- ✅ `./quick-start.sh` completes without errors
- ✅ Demo shows "Demo completed successfully! 🎉"
- ✅ All services show as healthy in `docker ps`
- ✅ Database validation passes in demo output

## 🎉 Welcome to Autonomous Development!

You now have a **production-ready autonomous development platform** ready for AI integration and real-world projects.

**Ready to build the future of software development! 🤖✨**