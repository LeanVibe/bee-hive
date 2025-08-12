# LeanVibe Agent Hive - Documentation Index

**The comprehensive documentation navigation hub for LeanVibe Agent Hive**

---

## 🚀 Quick Start

**New to Agent Hive?** Start here:

1. **[Getting Started Guide](../GETTING_STARTED.md)** - Installation and first steps
2. **[System Architecture](../ARCHITECTURE.md)** - Implementation-aligned architecture
3. **[Core Overview](../CORE.md)** - Product and scope overview
4. **[API Reference](api/API_REFERENCE_COMPREHENSIVE.md)** - Complete API documentation

**Common Tasks:**
- 🔧 **Implementation Status**: [Implementation → Progress Tracker](implementation/progress-tracker.md)
- 📊 **Monitoring System**: [User → Coordination Dashboard](user/COORDINATION_DASHBOARD_USER_GUIDE.md)
- 🛠️ **Development Setup**: [Developer Guide](user/DEVELOPER_GUIDE.md)
- 🔐 **Security Config**: [Enterprise → Security Guide](enterprise/ENTERPRISE_DEPLOYMENT_SECURITY_GUIDE.md)

---

## 📂 Documentation Structure

### 📋 Product Requirements (`/prd`)
*The authoritative specifications for all system components*

**Core System Specifications:**
- **[System Architecture](prd/system-architecture.md)** - Master system design and component relationships
- **[Agent Orchestrator](prd/agent-orchestrator-prd.md)** - Multi-agent coordination system
- **[Context Engine](prd/PRD-context-engine.md)** - Advanced memory and context management
- **[Sleep-Wake Manager](prd/PRD-sleep-wake-manager.md)** - Automated context consolidation
- **[Security & Auth System](prd/security-auth-system-prd.md)** - Authentication and authorization

**Specialized Systems:**
- **[Observability System](prd/observability-system-prd.md)** - Monitoring and performance tracking
- **[Communication System](prd/communication-prd.md)** - Inter-agent messaging and coordination
- **[GitHub Integration](prd/github-integration-prd.md)** - CI/CD workflow automation
- **[Self-Modification Engine](prd/self-modification-engine-prd.md)** - Autonomous system improvements
- **[Prompt Optimization](prd/prompt-optimization-system-prd.md)** - AI prompt enhancement system
- **[Mobile PWA Dashboard](prd/PRD-mobile-pwa-dashboard.md)** - Mobile progressive web app

### 🛠️ Implementation Guides (`/implementation`)
*Step-by-step implementation documentation and current status*

- **[🎯 Implementation Progress Tracker](implementation/progress-tracker.md)** - **⭐ MASTER STATUS DOCUMENT** - Live implementation progress with validation results
- **[Current Status](implementation/CURRENT_STATUS.md)** - Historical implementation progress and roadmap
- **[QA Validation Report](implementation/qa-validation-report.md)** - Quality assurance validation results
- **[Vertical Slice 1](implementation/VERTICAL_SLICE_1_IMPLEMENTATION.md)** - Basic agent orchestration
- **[Vertical Slice 2](implementation/VERTICAL_SLICE_2_IMPLEMENTATION.md)** - Advanced coordination features
- **[Vertical Slice 2.1](implementation/VERTICAL_SLICE_2_1_IMPLEMENTATION.md)** - Enhanced orchestration
- **[VS 4.2 Migration Guide](implementation/VS_4_2_MIGRATION_GUIDE.md)** - Database migration procedures
- **[VS 6.2 Implementation](implementation/VS_6_2_IMPLEMENTATION_GUIDE.md)** - Dashboard integration
- **[Workflow Orchestration](implementation/WORKFLOW_ORCHESTRATION_OPTIMIZATION.md)** - Advanced workflow management

### 🏢 Enterprise Materials (`/enterprise`)
*Enterprise deployment, security, and user documentation*

- **[Enterprise User Guide](enterprise/ENTERPRISE_USER_GUIDE.md)** - Complete enterprise user documentation
- **[Deployment Security Guide](enterprise/ENTERPRISE_DEPLOYMENT_SECURITY_GUIDE.md)** - Security hardening and compliance
- **[System Architecture](enterprise/ENTERPRISE_SYSTEM_ARCHITECTURE.md)** - Enterprise-specific architecture considerations

### 🔌 API Documentation (`/api`)
*Complete API references and integration guides*

- **[Comprehensive API Reference](api/API_REFERENCE_COMPREHENSIVE.md)** - All API endpoints and schemas
- **[GitHub Integration API](api/GITHUB_INTEGRATION_API_COMPREHENSIVE.md)** - GitHub workflow automation APIs
- **[Security & Auth API](api/SECURITY_AUTH_API_COMPREHENSIVE.md)** - Authentication and authorization APIs
- **[Semantic Memory API](api/SEMANTIC_MEMORY_API.md)** - Vector search and memory management APIs

### 👥 User Documentation (`/user`)
*User guides, tutorials, and how-to documentation*

- **[Developer Guide](user/DEVELOPER_GUIDE.md)** - Complete development workflow and best practices
- **[Comprehensive Tutorial](user/USER_TUTORIAL_COMPREHENSIVE.md)** - Step-by-step platform tutorial  
- **[Coordination Dashboard Guide](user/COORDINATION_DASHBOARD_USER_GUIDE.md)** - Real-time monitoring and control
- **[Custom Commands Guide](user/CUSTOM_COMMANDS_USER_GUIDE.md)** - Creating and managing custom commands
- **[External Tools Guide](user/EXTERNAL_TOOLS_GUIDE.md)** - Integrating third-party tools
- **[Multi-Agent Coordination](user/MULTI_AGENT_COORDINATION_GUIDE.md)** - Advanced agent coordination patterns

### 📚 Specialized Guides
*Technical deep-dives and specialized documentation*

- **[Troubleshooting Guide](TROUBLESHOOTING_GUIDE_COMPREHENSIVE.md)** - Comprehensive problem resolution
- **[Production Deployment](PRODUCTION_DEPLOYMENT_RUNBOOK.md)** - Production deployment procedures
- **[Quality Gates Automation](QUALITY_GATES_AUTOMATION.md)** - Automated quality assurance
- **[Hook Integration Guide](HOOK_INTEGRATION_GUIDE.md)** - System integration hooks
- **[Agent Specialization Templates](AGENT_SPECIALIZATION_TEMPLATES.md)** - Creating specialized agents
- **[Semantic Memory Performance](SEMANTIC_MEMORY_PERFORMANCE.md)** - Vector search optimization
- **[Observability Event Schema](OBSERVABILITY_EVENT_SCHEMA.md)** - Event monitoring specifications

### 📦 Archive (`/archive`)
*Historical documents and deprecated content*

- **[`/archive/phase-reports`](archive/phase-reports/)** - Historical phase implementation reports
- **[`/archive/vertical-slice-reports`](archive/vertical-slice-reports/)** - Archived vertical slice implementation reports
- **[`/archive/implementation-reports`](archive/implementation-reports/)** - Consolidated implementation audit and validation reports
- **[`/archive/old-versions`](archive/old-versions/)** - Previous versions of consolidated documents  
- **[`/archive/deprecated`](archive/deprecated/)** - Deprecated documentation (watermarked for safety)

---

## 🤖 AI Agent Navigation Guide

### For AI Agents Working with Agent Hive:

**Finding Information by Type:**
- **System Behavior**: Start with [System Architecture](prd/system-architecture.md), then check [Implementation Progress Tracker](implementation/progress-tracker.md)
- **API Integration**: Use [API Reference](api/API_REFERENCE_COMPREHENSIVE.md) as primary source
- **Implementation Status**: Check [Implementation Progress Tracker](implementation/progress-tracker.md) for current validated status
- **Error Resolution**: Consult [Troubleshooting Guide](TROUBLESHOOTING_GUIDE_COMPREHENSIVE.md)
- **Configuration**: See [Enterprise Security Guide](enterprise/ENTERPRISE_DEPLOYMENT_SECURITY_GUIDE.md)

**Authoritative Source Resolution:**
- **Single Source of Truth**: Each topic has ONE authoritative document (linked above)
- **Historical Context**: Check [Archive](archive/) if current docs reference deprecated content
- **Cross-References**: Follow links within documents for related information
- **Status Updates**: [Implementation Progress Tracker](implementation/progress-tracker.md) is the live project tracker with validation results

**Creating New Documentation:**
- **Product Specs**: Add to `/prd` directory
- **Implementation Guides**: Add to `/implementation` directory  
- **User Documentation**: Add to `/user` directory
- **API Documentation**: Add to `/api` directory
- **Enterprise Content**: Add to `/enterprise` directory

---

## 📖 Documentation Standards

### Document Types and Purposes

**Product Requirements Documents (PRDs)**
- Authoritative system specifications
- Technical architecture definitions
- Feature requirements and acceptance criteria
- Used by development teams for implementation

**Implementation Guides**
- Step-by-step implementation instructions
- Migration procedures and upgrade paths
- Integration workflows and examples
- Current project status and progress tracking

**User Documentation**
- End-user guides and tutorials
- Administrative procedures
- Troubleshooting and support information
- Best practices and workflow recommendations

**API Documentation**
- Endpoint specifications and schemas
- Authentication and authorization procedures
- Integration examples and sample code
- Performance and rate limiting information

**Enterprise Materials**
- Deployment and security hardening guides
- Compliance and audit documentation
- Enterprise-specific architecture considerations
- Business continuity and disaster recovery

### Quality Standards

**All Documentation Must:**
- Have a clear, descriptive title and purpose statement
- Include table of contents for documents >500 words
- Use consistent markdown formatting and structure
- Include last updated date and version information
- Cross-reference related documents appropriately
- Avoid redundancy with other documents

**Technical Documentation Must:**
- Include code examples where applicable
- Specify version compatibility and requirements
- Provide troubleshooting information
- Include performance considerations
- Reference relevant test cases

---

## 🔄 Document Lifecycle

### Creation Process
1. Determine appropriate directory based on document type
2. Follow naming conventions: `descriptive-name.md`
3. Include proper frontmatter and metadata
4. Cross-reference related existing documents
5. Update this index with new document links

### Update Process
1. Update content while preserving document structure
2. Update "last modified" date and version
3. Review cross-references for accuracy
4. Validate links and references
5. Update related documents if necessary

### Archival Process
1. Move deprecated documents to `/archive/deprecated/`
2. Add watermark indicating deprecation status
3. Update cross-references to point to new authoritative source
4. Maintain historical context in archive structure

---

## 📞 Documentation Support

**For Questions About:**
- **Content**: Check related documents in same directory
- **Structure**: Consult [Contributing Guide](../CONTRIBUTING.md)
- **Missing Information**: Create GitHub issue with documentation request
- **Corrections**: Submit pull request with proposed changes

**Documentation Maintainers:**
- Architecture: System Architecture Team
- Implementation: Development Team  
- User Guides: Product Team
- API Docs: Engineering Team

---

**Last Updated**: July 31, 2025  
**Version**: 2.0.0  
**Maintained By**: Documentation Architecture Team

---

*This index serves as the central navigation hub for all Agent Hive documentation. For specific technical questions, consult the appropriate specialized guide above.*