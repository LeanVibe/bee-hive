# Claude Code Master Prompt: LeanVibe Agent Hive System Transformation

## 🎯 Primary Mission

You are the lead architect for transforming the LeanVibe Agent Hive from a feature-rich but complex system into a production-ready, enterprise-grade platform. Your task is to **plan comprehensively** and **execute systematically** a 4-epic transformation that consolidates 200+ files into 50 core modules, builds 90% test coverage from zero, and achieves enterprise production readiness.

## 📊 Critical System Analysis

### ✅ **Foundation Strengths (Build Upon These)**
**Project Index Feature - PRODUCTION READY**:
- ✅ Complete database schema (5 tables, migrations up to 023)
- ✅ Core infrastructure: ProjectIndexer, CodeAnalyzer, file monitoring, Redis caching
- ✅ 8 REST API endpoints + AI-powered context optimization  
- ✅ 4 WebSocket event types with real-time subscription management
- ✅ 8 interactive PWA dashboard components with D3.js visualizations
- ✅ Multi-agent integration with context-aware coordination
- ✅ Quality assurance patterns established

### 🚨 **Critical Transformation Targets**

#### 1. **System Complexity Crisis** (IMMEDIATE)
```
Current: app/core/ = 200+ files with massive redundancy
Target: app/core/ = 50 core modules, single unified orchestrator
Impact: 75% complexity reduction, 300% maintenance efficiency
Timeline: Epic 1 (Weeks 1-3)
```

#### 2. **Testing Infrastructure Void** (CRITICAL)
```
Current: ~0% comprehensive test coverage
Target: 90% test coverage with automated quality gates
Impact: Production deployment confidence, regression prevention
Timeline: Epic 2 (Weeks 4-6)
```

#### 3. **Integration Gap** (HIGH PRIORITY)
```
Current: Project Index isolated, not integrated into main app
Target: Full integration with main app, unified WebSocket, auth flow
Impact: Immediate user value, feature accessibility
Timeline: Epic 1, Phase 2 (Week 2)
```

#### 4. **Documentation Chaos** (MEDIUM PRIORITY)
```
Current: 500+ fragmented documentation files
Target: 50 canonical sources with living documentation
Impact: 50% faster developer onboarding, knowledge accessibility
Timeline: Epic 3 (Weeks 7-9)
```

## 🏗️ **Strategic Implementation Framework**

### **Phase 1: Strategic Planning & Assessment (Week 1)**

**Your First Task: Create Master Implementation Plan**

1. **System Architecture Audit**
   ```bash
   # Analyze current complexity
   find app/core/ -name "*.py" | xargs wc -l | sort -rn
   ls app/core/ | grep -E "(orchestrator|coordinator|manager)" | wc -l
   find app/api/ -name "*.py" | grep -v __pycache__ | wc -l
   
   # Map dependencies and redundancy
   grep -r "from app.core" app/ | cut -d: -f2 | sort | uniq -c | sort -rn
   ```

2. **Integration Assessment**
   ```bash
   # Check Project Index integration status
   grep -r "project_index" app/main.py || echo "NOT INTEGRATED"
   grep -r "project-index" mobile-pwa/src/main.ts || echo "FRONTEND NOT INTEGRATED"
   ```

3. **Create Detailed Execution Roadmap**
   - Map all 200+ files in app/core/ by function and redundancy
   - Identify 15+ orchestrator variants for consolidation 
   - Plan Project Index integration points
   - Design testing infrastructure architecture
   - Structure agent delegation strategy

**Deliverable**: Comprehensive execution plan with specific file consolidation map, integration points, and agent task assignments.

### **Phase 2: Agent Delegation Strategy**

**Use Task Tool for Complex Operations** - Delegate these to specialized agents:

#### **Backend Engineer Agent** (Primary for Epic 1)
```
Mission: System consolidation and Project Index integration
Scope: 
- Consolidate app/core/ from 200+ to 50 modules
- Integrate Project Index into main application
- Unify orchestrator implementations
- Optimize database and API performance

Specific Tasks:
1. Audit and map all app/core/ files for consolidation
2. Merge 15+ orchestrator variants into unified implementation  
3. Register Project Index routes in app/main.py
4. Integrate Project Index WebSocket events with dashboard
5. Optimize database queries and connection pooling

Context Limits: Focus on 2-3 major consolidation tasks per session
Success Criteria: app/core/ <50 files, Project Index fully integrated
```

#### **QA Test Guardian Agent** (Primary for Epic 2)
```
Mission: Build comprehensive testing infrastructure from zero
Scope:
- Create testing framework (pytest, vitest, Playwright)
- Achieve 90% test coverage across all modules
- Build automated quality gates and CI/CD enhancement
- Implement performance benchmarking

Specific Tasks:
1. Set up pytest framework with fixtures and mocks
2. Create API testing suite for all endpoints
3. Build WebSocket testing framework
4. Implement performance benchmarking
5. Create CI/CD quality gates

Context Limits: Focus on one testing domain per session
Success Criteria: 90% coverage, automated quality gates
```

#### **Frontend Builder Agent** (Supporting Epic 1 & 4)
```
Mission: Project Index PWA integration and UX optimization
Scope:
- Integrate Project Index components into main PWA
- Add navigation and routing for Project Index
- Optimize user experience and performance
- Build responsive dashboard layouts

Specific Tasks:
1. Add Project Index routes to PWA navigation
2. Integrate dashboard components with existing layout
3. Implement real-time WebSocket integration
4. Optimize component performance and loading
5. Add mobile-responsive layouts

Context Limits: Focus on 2-3 components per session
Success Criteria: Project Index accessible via PWA, responsive UX
```

#### **DevOps Deployer Agent** (Primary for Epic 4)
```
Mission: Production infrastructure and enterprise readiness
Scope:
- Production deployment pipeline
- Monitoring and observability
- Security hardening and compliance
- Scalability and performance optimization

Specific Tasks:
1. Docker containerization optimization
2. Kubernetes deployment manifests
3. Prometheus metrics integration
4. Security audit and hardening
5. Load balancing and auto-scaling

Context Limits: Focus on one infrastructure domain per session
Success Criteria: 99.9% availability, enterprise security compliance
```

#### **General Purpose Agent** (Primary for Epic 3)
```
Mission: Documentation consolidation and knowledge management
Scope:
- Audit and consolidate 500+ documentation files
- Create living documentation system
- Build searchable knowledge base
- Optimize developer onboarding experience

Specific Tasks:
1. Audit all documentation for redundancy
2. Consolidate overlapping content
3. Create automated documentation system
4. Build developer onboarding guides
5. Implement searchable knowledge base

Context Limits: Focus on one documentation domain per session
Success Criteria: 50 canonical docs, 50% faster onboarding
```

## 📋 **Detailed Implementation Roadmap**

### **Epic 1: System Consolidation & Integration (Weeks 1-3)**

#### **Week 1: Assessment & Project Index Integration**
```
Day 1-2: System Analysis & Planning
- Backend Engineer: Audit app/core/ files and create consolidation map
- You: Create master implementation plan and agent task assignments

Day 3-4: Project Index Integration  
- Backend Engineer: Register Project Index routes in app/main.py
- Backend Engineer: Integrate WebSocket events with dashboard
- Frontend Builder: Add Project Index to PWA navigation

Day 5: Integration Testing
- QA Test Guardian: Create basic integration tests for Project Index
- You: Validate end-to-end Project Index functionality

Success Criteria:
✅ Project Index accessible via main app
✅ WebSocket events integrated with dashboard  
✅ PWA navigation includes Project Index
✅ Basic integration tests passing
```

#### **Week 2: Core System Consolidation**
```
Day 1-3: Orchestrator Consolidation
- Backend Engineer: Merge 15+ orchestrator variants into unified implementation
- Backend Engineer: Consolidate agent lifecycle management
- QA Test Guardian: Create tests for unified orchestrator

Day 4-5: API Route Cleanup
- Backend Engineer: Merge overlapping API modules in app/api/v1/
- Backend Engineer: Standardize error handling across APIs
- QA Test Guardian: Update API tests for consolidated routes

Success Criteria:
✅ Single unified orchestrator implementation
✅ API routes consolidated with no redundancy
✅ All existing functionality preserved
✅ Performance maintained or improved
```

#### **Week 3: Performance Optimization**
```
Day 1-2: Database Optimization
- Backend Engineer: Optimize database queries and indexing
- Backend Engineer: Configure connection pooling
- QA Test Guardian: Create performance tests

Day 3-4: API Performance & Caching
- Backend Engineer: Implement response caching strategy
- Backend Engineer: Add rate limiting and throttling
- DevOps Deployer: Set up performance monitoring

Day 5: Validation & Documentation
- You: Validate all Epic 1 success criteria
- General Purpose: Document consolidation decisions

Success Criteria:
✅ app/core/ reduced to <50 modules
✅ API response times <500ms for 95% requests
✅ Database queries optimized
✅ Performance monitoring operational
```

### **Epic 2: Testing Infrastructure & Quality Gates (Weeks 4-6)**

#### **Week 4: Core Testing Framework**
```
Day 1-2: Testing Infrastructure Setup
- QA Test Guardian: Set up pytest framework with fixtures
- QA Test Guardian: Create database testing with isolated environments
- QA Test Guardian: Set up mock services for external dependencies

Day 3-4: API Testing Suite
- QA Test Guardian: Create REST API endpoint testing
- QA Test Guardian: WebSocket connection and message testing
- QA Test Guardian: Authentication and authorization testing

Day 5: Frontend Testing Setup
- Frontend Builder: Set up Vitest for component testing
- Frontend Builder: Configure Playwright for E2E testing
- QA Test Guardian: Cross-browser compatibility testing

Success Criteria:
✅ pytest framework operational with fixtures
✅ 50+ unit tests for core functionality  
✅ API testing suite with 100% endpoint coverage
✅ Frontend testing framework setup
```

#### **Week 5: Integration & Performance Testing**
```
Day 1-2: Integration Testing Suite
- QA Test Guardian: Database-API-Frontend integration tests
- QA Test Guardian: Multi-agent coordination testing
- QA Test Guardian: Real-time WebSocket integration testing

Day 3-4: Performance Testing Framework
- QA Test Guardian: Load testing with k6/Artillery
- QA Test Guardian: Database performance benchmarking
- DevOps Deployer: WebSocket connection stress testing

Day 5: Security Testing
- QA Test Guardian: Authentication/authorization testing
- QA Test Guardian: Input validation and injection testing
- DevOps Deployer: API security and rate limiting testing

Success Criteria:
✅ Integration testing suite operational
✅ Performance benchmarking framework
✅ Security testing validation
✅ Load testing scenarios defined
```

#### **Week 6: Quality Gates & CI/CD**
```
Day 1-2: Automated Quality Gates
- QA Test Guardian: Test coverage enforcement (90%+ target)
- QA Test Guardian: Performance regression detection
- DevOps Deployer: Security vulnerability scanning

Day 3-4: CI/CD Pipeline Enhancement
- DevOps Deployer: Automated testing on all PRs
- DevOps Deployer: Performance benchmarking in CI
- DevOps Deployer: Security scanning integration

Day 5: Monitoring & Alerting
- DevOps Deployer: Test result monitoring and alerting
- DevOps Deployer: Performance metric tracking
- You: Validate all Epic 2 success criteria

Success Criteria:
✅ 90%+ test coverage across all modules
✅ Automated quality gates in CI/CD
✅ Performance regression detection
✅ Security vulnerability prevention
```

### **Epic 3: Documentation & Knowledge Management (Weeks 7-9)**

#### **Week 7: Documentation Audit & Consolidation**
```
Day 1-2: Documentation Audit
- General Purpose: Catalog all 500+ documentation files
- General Purpose: Identify redundancy and overlap
- General Purpose: Extract valuable unique content

Day 3-4: Content Consolidation
- General Purpose: Merge overlapping PRDs into single source
- General Purpose: Consolidate technical specifications
- General Purpose: Create canonical architecture documentation

Day 5: Information Architecture
- General Purpose: Design clear documentation hierarchy
- General Purpose: Create navigation and discovery system
- Frontend Builder: Implement content categorization UI

Success Criteria:
✅ Documentation audit completed
✅ Content reduced from 500+ to 100 intermediate files
✅ Clear information architecture designed
```

#### **Week 8: Living Documentation System**
```
Day 1-2: Automated Documentation
- General Purpose: API documentation auto-generation
- General Purpose: Code example validation system
- General Purpose: Link checking and maintenance

Day 3-4: Knowledge Base Implementation
- Frontend Builder: Searchable documentation system
- Frontend Builder: Tag-based content discovery
- General Purpose: Cross-reference mapping

Day 5: Documentation Standards
- General Purpose: Writing and formatting standards
- General Purpose: Template system for consistency
- General Purpose: Review and approval workflow

Success Criteria:
✅ Automated documentation system
✅ Searchable knowledge base
✅ Documentation standards established
```

#### **Week 9: Developer Experience Enhancement**
```
Day 1-2: Onboarding Optimization
- General Purpose: Complete developer setup guide
- General Purpose: Quick start tutorials and examples
- General Purpose: Troubleshooting and FAQ system

Day 3-4: API & Integration Guides
- General Purpose: Complete API reference documentation
- General Purpose: Integration examples and patterns
- General Purpose: Best practices and conventions

Day 5: Final Consolidation
- General Purpose: Final reduction to 50 canonical sources
- You: Validate all Epic 3 success criteria

Success Criteria:
✅ 50 canonical documentation sources
✅ Developer onboarding time reduced by 50%
✅ Searchable knowledge base operational
✅ 95% documentation accuracy and freshness
```

### **Epic 4: Production Optimization & Enterprise Features (Weeks 10-12)**

#### **Week 10: Production Infrastructure**
```
Day 1-2: Production Deployment Pipeline
- DevOps Deployer: Docker containerization optimization
- DevOps Deployer: Kubernetes deployment manifests
- DevOps Deployer: Production configuration management

Day 3-4: Monitoring & Observability
- DevOps Deployer: Prometheus metrics integration
- DevOps Deployer: Distributed tracing with OpenTelemetry
- DevOps Deployer: Performance monitoring dashboards

Day 5: High Availability & Scaling
- DevOps Deployer: Database clustering and replication
- DevOps Deployer: Redis clustering configuration
- DevOps Deployer: Load balancing and auto-scaling

Success Criteria:
✅ Production deployment pipeline
✅ Comprehensive monitoring system
✅ High availability architecture
```

#### **Week 11: Enterprise Security & Compliance**
```
Day 1-2: Security Hardening
- DevOps Deployer: Authentication and authorization enhancement
- DevOps Deployer: API security and rate limiting
- DevOps Deployer: Data encryption and protection

Day 3-4: Compliance Framework
- DevOps Deployer: GDPR/SOC2 compliance preparation
- DevOps Deployer: Audit logging and reporting
- DevOps Deployer: Data privacy and protection

Day 5: Enterprise Features
- Backend Engineer: Multi-tenancy support
- Backend Engineer: Role-based access control (RBAC)
- DevOps Deployer: Enterprise SSO integration

Success Criteria:
✅ Enterprise security framework
✅ Compliance monitoring system
✅ Multi-tenant architecture
```

#### **Week 12: Performance & Scalability Optimization**
```
Day 1-2: Performance Optimization
- Backend Engineer: Database query optimization
- Backend Engineer: API response time improvement
- Backend Engineer: Memory usage optimization

Day 3-4: Scalability Enhancements
- DevOps Deployer: Horizontal scaling capabilities
- DevOps Deployer: Microservices architecture preparation
- DevOps Deployer: Caching strategy optimization

Day 5: Final Validation
- You: Validate all Epic 4 success criteria
- You: Conduct final system validation
- You: Prepare production readiness report

Success Criteria:
✅ 99.9% system availability target
✅ Enterprise security compliance
✅ 10x scaling capacity
✅ Production deployment ready
```

## 🔧 **Technical Implementation Guidelines**

### **Context Management Strategy**
```
Session Management:
- Plan comprehensive roadmap (this session)
- Use Task tool for agent delegation (avoid context rot)
- Focus each agent on 2-3 specific deliverables per session
- Regular validation checkpoints every 2-3 days
- Document decisions and progress continuously

Agent Coordination:
- Clear task boundaries to prevent overlap
- Shared deliverables and integration points
- Regular cross-agent validation sessions
- Centralized progress tracking and reporting
```

### **Quality Assurance Requirements**
```
Every Change Must Include:
✅ Unit tests for new functionality
✅ Integration tests for system interactions
✅ Performance impact assessment
✅ Documentation updates
✅ Security impact review

Validation Checkpoints:
- Daily: Basic functionality preserved
- Weekly: Epic success criteria progress
- Bi-weekly: Cross-system integration validation
- Monthly: Performance and security audits
```

### **Risk Mitigation Protocols**
```
Change Management:
- Incremental changes with rollback capability
- Feature flags for major integrations
- Database migration safety (backup/restore)
- Performance monitoring during changes

Integration Safety:
- Comprehensive testing before major integrations
- Staged rollout for critical changes
- Rollback procedures for all modifications
- Real-time monitoring during deployments
```

## 📊 **Success Metrics & Validation**

### **Epic 1 Success Criteria**
- [ ] app/core/ reduced from 200+ to <50 modules
- [ ] Project Index fully integrated into main application
- [ ] Single unified orchestrator replacing 15+ variants
- [ ] API response times <500ms for 95% of requests
- [ ] Memory usage <2GB for typical workloads

### **Epic 2 Success Criteria**
- [ ] 90%+ test coverage across all core modules
- [ ] Comprehensive integration and performance testing
- [ ] Automated quality gates in CI/CD pipeline
- [ ] Performance benchmarking framework operational
- [ ] Zero critical security vulnerabilities

### **Epic 3 Success Criteria**
- [ ] Documentation reduced from 500+ to 50 canonical sources
- [ ] Living documentation system with automated maintenance
- [ ] Developer onboarding time reduced by 50%
- [ ] Searchable knowledge base operational
- [ ] 95% documentation accuracy and freshness

### **Epic 4 Success Criteria**
- [ ] Production deployment pipeline operational
- [ ] Enterprise security and compliance framework
- [ ] 99.9% system availability target
- [ ] 10x scaling capacity with performance optimization
- [ ] Enterprise multi-tenancy support

## 🚀 **Immediate Actions (Start Here)**

### **First Session Tasks**
1. **Review Current State** (30 minutes)
   ```bash
   # System health check
   python -c "import app.main; print('Main app loads')"
   python -c "import app.project_index.core; print('Project Index loads')"
   
   # Complexity assessment
   find app/core/ -name "*.py" | wc -l
   ls app/core/ | grep orchestrator | wc -l
   
   # Integration status
   grep -r "project_index" app/main.py || echo "NOT INTEGRATED"
   ```

2. **Create Master Plan** (60 minutes)
   - Map all 200+ app/core/ files by function
   - Identify consolidation opportunities
   - Plan Project Index integration points
   - Design agent delegation strategy

3. **Begin Agent Delegation** (30 minutes)
   - Use Task tool to delegate Project Index integration to Backend Engineer
   - Use Task tool to delegate testing setup to QA Test Guardian
   - Schedule regular coordination checkpoints

### **Critical Success Factors**
1. **Start with Project Index integration** (immediate user value)
2. **Use agent delegation early and often** (avoid context rot)
3. **Maintain quality gates throughout** (no compromises on testing)
4. **Document all decisions** (knowledge preservation)
5. **Monitor performance continuously** (prevent regressions)

## ⚠️ **Critical Warnings & Constraints**

### **Absolute Requirements**
- **NEVER break existing functionality** during consolidation
- **ALL changes must include comprehensive tests**
- **PERFORMANCE must be maintained or improved**
- **PROJECT INDEX must remain functional throughout**

### **Agent Delegation Constraints**
- **Max 3 major tasks per agent per session** (context limits)
- **Clear deliverables and success criteria** for each delegation
- **Regular coordination between agents** (prevent conflicts)
- **Shared documentation and decision tracking**

### **Production Safety**
- **All database changes require migration rollback plans**
- **API changes require backward compatibility**
- **WebSocket changes require graceful degradation**
- **Performance changes require before/after benchmarks**

## 🎯 **Final Success Definition**

**Mission Complete When**:
✅ System transformed from 200+ chaotic modules to 50 clean, maintainable components
✅ Comprehensive testing infrastructure providing 90%+ coverage with automated quality gates  
✅ Project Index fully integrated and accessible to users via main application
✅ Documentation consolidated into living knowledge base supporting autonomous development
✅ Production-ready platform with enterprise security, compliance, and 10x scaling capacity

**Transformation Summary**: 
Complex feature-rich system → Production-ready enterprise platform
Maintenance nightmare → Developer productivity paradise
Knowledge silos → Living documentation ecosystem
Quality uncertainty → Automated quality assurance

**Your success is measured by transforming complexity into clarity, features into reliability, and potential into production-ready value.**

---

*This comprehensive prompt provides everything needed for successful system transformation through strategic planning and intelligent agent delegation. Begin with the immediate actions and use the Task tool extensively to avoid context rot while maintaining high-quality outcomes.*