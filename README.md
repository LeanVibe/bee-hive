# 🤖 LeanVibe Agent Hive

**The next-generation multi-agent orchestration system for autonomous software development**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://www.postgresql.org)
[![Redis](https://img.shields.io/badge/Redis-7+-red.svg)](https://redis.io)

---

## 🌟 Overview

LeanVibe Agent Hive is a production-ready multi-agent orchestration platform that enables autonomous software development through intelligent agent coordination, real-time communication, and advanced context management. Built with modern Python async/await patterns and designed for scale.

### 🎯 Key Features

- **🤖 Multi-Agent Orchestration**: Coordinate multiple AI agents with role-based task assignment
- **📱 Mobile PWA Dashboard**: Real-time monitoring and control via progressive web app
- **⚡ Real-time Communication**: Redis Streams-based message bus with WebSocket support
- **🧠 Advanced Context Engine**: PostgreSQL + pgvector for intelligent memory management
- **🔄 Sleep-Wake Management**: Automated context consolidation and recovery cycles
- **📊 Production Observability**: Comprehensive monitoring with Prometheus + Grafana
- **🔐 Security-First**: JWT authentication, RBAC, and audit logging
- **🌐 GitHub Integration**: Native CI/CD workflow automation
- **📦 Offline-First**: IndexedDB caching and optimistic updates
- **🔔 Push Notifications**: Firebase Cloud Messaging for critical alerts

### 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mobile PWA    │    │   Vue.js Web    │    │  FastAPI API    │
│   Dashboard     │◄──►│   Dashboard     │◄──►│   Gateway       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                         ┌─────────────────────────────┼─────────────────────────────┐
                         │                             │                             │
                ┌─────────▼──────┐          ┌─────────▼──────┐          ┌─────────▼──────┐
                │  Agent         │          │  Context       │          │  Communication │
                │  Orchestrator  │          │  Engine        │          │  Bus           │
                └────────────────┘          └────────────────┘          └────────────────┘
                         │                             │                             │
                ┌─────────▼──────┐          ┌─────────▼──────┐          ┌─────────▼──────┐
                │  PostgreSQL    │          │  pgvector      │          │  Redis         │
                │  Database      │          │  Embeddings    │          │  Streams       │
                └────────────────┘          └────────────────┘          └────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** with async/await support
- **Docker & Docker Compose** for containerized services
- **Node.js 18+** for frontend development
- **Git** with SSH keys configured

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/LeanVibe/bee-hive.git
cd bee-hive

# Start infrastructure services
docker-compose up -d postgres redis

# Install Python dependencies
pip install -e .

# Run database migrations
alembic upgrade head

# Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend Setup

#### Vue.js Web Dashboard
```bash
cd frontend
npm install
npm run dev
# Available at http://localhost:3000
```

#### Mobile PWA Dashboard
```bash
cd mobile-pwa
npm install
npm run dev
# Available at http://localhost:3001
```

### 3. Access the Platform

- **API Documentation**: http://localhost:8000/docs
- **Web Dashboard**: http://localhost:3000
- **Mobile PWA**: http://localhost:3001
- **WebSocket Endpoint**: ws://localhost:8000/ws/observability

---

## 📱 Mobile PWA Dashboard

The Mobile PWA Dashboard provides a complete mobile-first experience for monitoring and controlling your multi-agent operations:

### Features
- **📊 Real-time Agent Monitoring**: Live status updates and performance metrics
- **📋 Kanban Task Management**: Drag-and-drop task organization with offline support
- **🔔 Push Notifications**: Instant alerts for build failures, agent errors, and approval requests
- **📱 Offline-First**: Full functionality when disconnected with automatic sync
- **🔐 Biometric Authentication**: WebAuthn support for secure, passwordless login
- **⚡ High Performance**: <2s load time, 45+ FPS on low-end Android devices

### Installation
The PWA can be installed directly from your browser on mobile devices:

1. Open http://localhost:3001 in your mobile browser
2. Look for the "Install" prompt or use your browser's "Add to Home Screen" option
3. The app will be available as a native-like experience on your device

### Technology Stack
- **Framework**: Lit (Web Components) + TypeScript
- **Build Tool**: Vite with PWA plugin
- **Styling**: Tailwind CSS with mobile-first responsive design
- **Offline**: Workbox service worker with IndexedDB caching
- **State Management**: Zustand for reactive state
- **Notifications**: Firebase Cloud Messaging (FCM)

---

## 🛠️ Development

### Project Structure

```
bee-hive/
├── app/                     # FastAPI application
│   ├── api/v1/             # API endpoints
│   ├── core/               # Core business logic
│   ├── models/             # Database models
│   ├── schemas/            # Pydantic schemas
│   └── observability/      # Monitoring & logging
├── mobile-pwa/             # Mobile PWA Dashboard (Lit + TypeScript)
├── frontend/               # Web Dashboard (Vue.js)
├── migrations/             # Database migrations
├── tests/                  # Test suite
├── monitoring/             # Grafana dashboards
└── docs/                   # Documentation
```

### Running Tests

```bash
# Backend tests
pytest -v --cov=app

# Frontend tests (Vue.js)
cd frontend && npm test

# Mobile PWA tests
cd mobile-pwa && npm test

# E2E tests
cd mobile-pwa && npm run test:e2e
```

### Code Quality

```bash
# Python formatting & linting
black app/ tests/
ruff check app/ tests/

# TypeScript linting (Mobile PWA)
cd mobile-pwa && npm run lint

# Frontend linting (Vue.js)
cd frontend && npm run lint
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/agent_hive

# Redis
REDIS_URL=redis://localhost:6379/0

# Authentication
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Anthropic API
ANTHROPIC_API_KEY=your-anthropic-api-key

# Firebase (for push notifications)
FIREBASE_PROJECT_ID=your-firebase-project
FIREBASE_PRIVATE_KEY_ID=your-private-key-id
FIREBASE_PRIVATE_KEY=your-private-key
FIREBASE_CLIENT_EMAIL=your-client-email

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true

# Development
DEBUG=true
LOG_LEVEL=INFO
```

### Docker Compose Services

The included `docker-compose.yml` provides:

- **PostgreSQL 15** with pgvector extension
- **Redis 7** with persistence
- **Prometheus** for metrics collection
- **Grafana** for dashboards and alerting

---

## 📊 Monitoring & Observability

### Built-in Monitoring

- **Health Checks**: `/health` endpoint with dependency validation
- **Metrics**: Prometheus-compatible metrics at `/metrics`
- **Structured Logging**: JSON logs with correlation IDs
- **Performance Tracking**: Request/response times and error rates
- **Real-time Events**: WebSocket streaming for live monitoring

### Grafana Dashboards

Pre-built dashboards for:
- Agent performance and status
- API response times and error rates
- Database connection pooling
- Redis stream processing
- System resource utilization

Access Grafana at http://localhost:3000 (admin/admin)

---

## 🔐 Security

### Authentication & Authorization

- **JWT-based authentication** with refresh token rotation
- **Role-based access control (RBAC)** with granular permissions
- **WebAuthn support** for passwordless authentication
- **Session management** with automatic timeout
- **Audit logging** for all security-relevant events

### Security Headers

- CORS properly configured
- CSP headers for XSS protection
- Secure cookie settings
- Rate limiting on authentication endpoints

---

## 🚀 Deployment

### Production Requirements

- **Python 3.11+** with uvloop for performance
- **PostgreSQL 15+** with pgvector extension
- **Redis 7+** with persistence enabled
- **Nginx** for reverse proxy and static file serving
- **SSL/TLS** certificates for HTTPS

### Docker Deployment

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy with environment-specific configuration
docker-compose -f docker-compose.prod.yml up -d
```

### Environment-Specific Configurations

- **Development**: Hot reloading, debug logging, local services
- **Staging**: Production-like setup with test data
- **Production**: Optimized for performance, security, and reliability

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes with tests
4. Ensure all tests pass: `pytest` and `npm test`
5. Submit a pull request

### Code Standards

- **Python**: Follow PEP 8, use Black for formatting, type hints required
- **TypeScript**: Strict mode enabled, ESLint + Prettier
- **Testing**: Minimum 90% code coverage required
- **Documentation**: Update docs for any API changes

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **FastAPI** for the excellent async web framework
- **Lit** for lightweight, efficient web components
- **PostgreSQL** and **pgvector** for vector similarity search
- **Redis** for reliable message streaming
- **Anthropic** for Claude AI integration

---

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/LeanVibe/bee-hive/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LeanVibe/bee-hive/discussions)

---

**Built with ❤️ by the LeanVibe team**