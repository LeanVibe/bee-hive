# CLAUDE.md - Documentation & Knowledge Management

## 🎯 **Context: Living Documentation System**

You are working in the **documentation and knowledge management** system of LeanVibe Agent Hive 2.0. This directory contains comprehensive technical documentation, strategic plans, and knowledge artifacts that support autonomous development and enterprise deployment.

## 📚 **Documentation Architecture**

### **Current State: Comprehensive but Fragmented**
- **500+ documentation files** with massive redundancy
- **Strong Content Quality**: High-value strategic analyses and implementation guides
- **Organization Challenge**: Information scattered across multiple subdirectories
- **Opportunity**: Consolidate into living documentation that stays current

### **Documentation Categories**
```
docs/
├── core/                   # PRD, Architecture, System Specifications
├── guides/                 # Implementation and User Guides  
├── reference/              # API Documentation, Schemas
├── reports/                # Status Reports, Progress Tracking
├── tutorials/              # Step-by-step Learning Materials
├── enterprise/             # Business Strategy, Market Analysis
├── runbooks/               # Operational Procedures
└── archive/                # Historical Documents (500+ files)
```

## 🎨 **Documentation Standards**

### **System Design Guidelines**
- **Port Configuration**: All services should use non-standard ports that are easy to configure/change to avoid conflicts with other apps running on the same system

### **Living Documentation Principles**
1. **Single Source of Truth**: No duplicate information across files
2. **Executable Documentation**: Code examples that actually work
3. **Version Control**: Track changes and maintain accuracy
4. **Automated Validation**: Ensure links, examples, and references stay current

### **Markdown Standards**
```markdown
# Document Title - Clear and Specific

## 🎯 **Purpose Statement**
Clear, concise explanation of what this document provides

## 📋 **Prerequisites** 
What readers need to know/have before reading

## 🚀 **Quick Start**
Immediate actionable steps (when applicable)

## 🏗️ **Detailed Implementation**
Comprehensive information with code examples

## ⚠️ **Critical Considerations**
Important warnings, gotchas, or limitations

## ✅ **Validation**
How to verify success or implementation

## 🔗 **Related Resources**
Links to related documents, avoiding redundancy
```

### **Code Example Standards**
```python
# All code examples must be:
# 1. Executable (tested and working)
# 2. Complete (not just fragments)
# 3. Commented for clarity
# 4. Following project coding standards

from app.core.orchestrator import ProductionOrchestrator
from app.models.agent import AgentSpec

async def example_agent_creation():
    """
    Complete working example of agent creation
    This example demonstrates the full workflow from
    orchestrator initialization to agent registration
    """
    # Initialize orchestrator with production config
    config = OrchestratorConfig(
        max_agents=50,
        task_timeout=300,
        redis_url="redis://localhost:6379"
    )
    orchestrator = ProductionOrchestrator(config)
    
    # Create and register agent
    agent_spec = AgentSpec(
        name="documentation-agent",
        type="general-purpose",
        capabilities=["documentation", "markdown", "analysis"]
    )
    
    agent_id = await orchestrator.register_agent(agent_spec)
    print(f"Agent registered with ID: {agent_id}")
    
    return agent_id

# Usage:
# agent_id = await example_agent_creation()
```

## 📖 **Content Creation Guidelines**

### **PRD and Specification Documents** (`core/`)
**Purpose**: Define system requirements and architecture

```markdown
# PRD Template Structure

## 🎯 **Product Overview**
- Problem statement and user needs
- Success criteria and business value

## 🏗️ **Technical Requirements**
- Functional requirements with acceptance criteria
- Non-functional requirements (performance, security, etc.)
- Integration requirements and dependencies

## 📊 **Success Metrics** 
- Quantifiable success criteria
- Monitoring and validation approaches

## 🚧 **Implementation Phases**
- Phased delivery approach
- Dependencies and risk mitigation

## 🔄 **Maintenance and Evolution**
- Update procedures and ownership
- Long-term evolution strategy
```

### **Implementation Guides** (`guides/`)
**Purpose**: Step-by-step implementation instructions

```markdown
# Implementation Guide Template

## 🎯 **What You'll Build**
Clear description of end result

## 📋 **Prerequisites**
- Required knowledge and skills
- System requirements and dependencies
- Preparation steps

## 🚀 **Step-by-Step Implementation**
### Step 1: Foundation Setup
Detailed instructions with code examples

### Step 2: Core Implementation  
Build primary functionality

### Step 3: Integration and Testing
Connect components and validate

## ✅ **Validation and Troubleshooting**
- How to verify success
- Common issues and solutions
- Performance optimization tips

## 🎯 **Next Steps**
- Advanced configurations
- Integration with other systems
- Further learning resources
```

### **API Documentation** (`reference/`)
**Purpose**: Comprehensive API reference with examples

```markdown
# API Endpoint Documentation Template

## `POST /api/v1/agents`
Create a new agent in the hive system

### Request
```json
{
  "name": "backend-engineer-01",
  "type": "backend-engineer", 
  "capabilities": ["python", "fastapi", "postgresql"],
  "max_concurrent_tasks": 5
}
```

### Response
```json
{
  "id": "agent-uuid-here",
  "name": "backend-engineer-01",
  "type": "backend-engineer", 
  "status": "initializing",
  "created_at": "2025-08-15T10:00:00Z",
  "capabilities": ["python", "fastapi", "postgresql"]
}
```

### Error Responses
- `400 Bad Request`: Invalid request format
- `409 Conflict`: Agent name already exists
- `503 Service Unavailable`: Maximum agent capacity reached

### Example Usage
```python
import httpx

async def create_agent(name: str, agent_type: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/agents",
            json={
                "name": name,
                "type": agent_type,
                "capabilities": ["python", "fastapi"]
            }
        )
        return response.json()

# Usage
agent = await create_agent("my-agent", "backend-engineer")
```
```

## 📊 **Documentation Quality Assurance**

### **Content Validation**
- **Link Checking**: All internal and external links must be valid
- **Code Testing**: All code examples must execute successfully  
- **Accuracy Verification**: Technical details verified against implementation
- **Completeness**: No missing sections or incomplete explanations

### **Automated Documentation Maintenance**
```python
# scripts/validate_documentation.py
import ast
import subprocess
from pathlib import Path

class DocumentationValidator:
    """Automated documentation quality assurance"""
    
    def validate_code_examples(self, doc_path: Path):
        """Extract and test all Python code blocks"""
        content = doc_path.read_text()
        code_blocks = self.extract_python_blocks(content)
        
        for block in code_blocks:
            try:
                # Syntax validation
                ast.parse(block)
                
                # Optional: execution testing in isolated environment
                if self.is_executable_example(block):
                    self.test_code_execution(block)
                    
            except SyntaxError as e:
                raise DocumentationError(
                    f"Invalid Python syntax in {doc_path}: {e}"
                )
    
    def validate_links(self, doc_path: Path):
        """Verify all links are valid"""
        internal_links = self.extract_internal_links(doc_path)
        external_links = self.extract_external_links(doc_path)
        
        for link in internal_links:
            if not self.internal_link_exists(link):
                raise DocumentationError(f"Broken internal link: {link}")
        
        for link in external_links:
            if not self.external_link_accessible(link):
                self.log_warning(f"External link may be broken: {link}")
    
    def generate_metrics(self):
        """Generate documentation quality metrics"""
        return {
            "total_documents": self.count_documents(),
            "code_example_coverage": self.calculate_code_coverage(),
            "link_validity_rate": self.calculate_link_validity(),
            "content_freshness": self.assess_content_age(),
            "redundancy_score": self.detect_duplicate_content()
        }
```

## 🔄 **Epic Integration Documentation**

### **Epic 1: Agent Orchestration Documentation**
- **PRD Updates**: Keep agent orchestrator PRD current with implementation
- **API Documentation**: Document all new orchestration endpoints
- **Integration Guides**: How to integrate with consolidated orchestrator
- **Performance Documentation**: Benchmark results and optimization guides

### **Epic 2: Testing Documentation**
- **Testing Strategy**: Document the comprehensive testing pyramid approach
- **Test Writing Guides**: How to contribute effective tests
- **Quality Gates**: Document automated quality assurance processes
- **Coverage Reports**: Automated test coverage documentation

### **Epic 3: Security & Operations Documentation**
- **Security Policies**: Comprehensive security implementation documentation
- **Operational Runbooks**: Production deployment and maintenance procedures
- **Monitoring Guides**: How to set up and use monitoring systems
- **Incident Response**: Procedures for handling system issues

### **Epic 4: Context Engine Documentation**
- **Semantic Memory Guide**: How the context engine works and how to optimize it
- **Knowledge Management**: Best practices for cross-agent knowledge sharing
- **Context Optimization**: Performance tuning and memory management
- **Integration Patterns**: How to integrate context awareness into applications

## 🎯 **Documentation Success Metrics**

### **Quality Metrics**
- **Accuracy**: 100% of code examples execute successfully
- **Completeness**: All API endpoints documented with examples
- **Currency**: Documentation updated within 48 hours of code changes
- **Accessibility**: Clear writing that serves both technical and business audiences

### **Usage Metrics**
- **Developer Onboarding Time**: Reduce new developer ramp-up by 40%
- **Support Ticket Reduction**: Comprehensive documentation reduces questions
- **Implementation Success Rate**: Higher success rate for following guides
- **Search Effectiveness**: Easy discovery of relevant information

### **Maintenance Metrics**
- **Update Frequency**: Regular content updates and accuracy verification
- **Redundancy Reduction**: Single source of truth for all information
- **Broken Link Rate**: <1% broken links across all documentation
- **Content Freshness**: No documentation older than 6 months without review

## ⚠️ **Critical Documentation Priorities**

### **Immediate Consolidation Needs**
- **Archive Analysis**: Review 500+ archived files for valuable content
- **Redundancy Elimination**: Merge duplicate information into authoritative sources
- **Navigation Improvement**: Create clear documentation hierarchy and navigation
- **Search Enhancement**: Enable effective content discovery

### **Living Documentation Implementation**
- **Automation Integration**: Connect documentation updates to code changes
- **Version Synchronization**: Ensure docs stay current with implementation
- **Feedback Loops**: Enable users to report issues and suggest improvements
- **Metrics Dashboard**: Track documentation quality and usage metrics

## ✅ **Success Criteria**

Your work in `/docs/` is successful when:
- **Completeness**: All system components have comprehensive documentation
- **Accuracy**: All code examples and references are verified as working
- **Organization**: Clear information architecture with easy navigation
- **Currency**: Documentation stays current with system evolution
- **Usability**: New developers can successfully onboard using documentation alone

Focus on **consolidating the archive** and creating **living documentation patterns** that automatically stay current with system evolution. This supports all four epics by ensuring implementation knowledge is captured and accessible.