# Documentation Contributing Guide

**How to contribute to and maintain the LeanVibe Agent Hive documentation system**

---

## 🎯 Overview

This guide explains how to work with the Agent Hive documentation system, including our new organizational structure, contribution workflows, and maintenance procedures. Following these guidelines ensures consistent, discoverable, and maintainable documentation.

---

## 📂 New Documentation Structure

### Directory Organization

The documentation is organized into six primary directories:

```
docs/
├── prd/           # Product Requirements Documents
├── implementation/# Implementation guides and status
├── enterprise/    # Enterprise materials and deployment
├── api/           # API references and integration
├── user/          # User guides and tutorials
└── archive/       # Historical and deprecated content
```

### Directory Purposes

| Directory | Purpose | Content Types | Audience |
|-----------|---------|---------------|----------|
| `prd/` | Authoritative specifications | System architecture, feature requirements, technical specs | Developers, Architects |
| `implementation/` | Current progress & guides | Status updates, migration guides, implementation steps | Development Teams |
| `enterprise/` | Enterprise-focused materials | Deployment guides, security procedures, business docs | IT Admin, Enterprise Users |
| `api/` | API documentation | Endpoint specs, schemas, integration examples | API Consumers, Integrators |
| `user/` | End-user documentation | Tutorials, how-tos, troubleshooting | End Users, Administrators |
| `archive/` | Historical content | Deprecated docs, old versions, phase reports | Historical Reference Only |

---

## 📝 Document Placement Decision Matrix

### Where Should Your Document Go?

**Ask These Questions:**

1. **What is the primary purpose?**
   - System specification → `prd/`
   - Implementation guide → `implementation/`
   - User instruction → `user/`
   - API documentation → `api/`
   - Enterprise procedure → `enterprise/`

2. **Who is the primary audience?**
   - Developers building the system → `prd/` or `implementation/`
   - Users operating the system → `user/` or `enterprise/`
   - Integrators using APIs → `api/`
   - Historical reference → `archive/`

3. **What is the document lifecycle?**
   - Living specification → `prd/`
   - Current status/progress → `implementation/`
   - Stable user guide → `user/`
   - Deprecated content → `archive/`

### Decision Tree

```
Is this a system specification or requirement?
├─ YES → prd/
└─ NO → Is this implementation guidance or status?
    ├─ YES → implementation/
    └─ NO → Is this API documentation?
        ├─ YES → api/
        └─ NO → Is this enterprise-specific?
            ├─ YES → enterprise/
            └─ NO → Is this user-facing?
                ├─ YES → user/
                └─ NO → Is this historical/deprecated?
                    ├─ YES → archive/
                    └─ NO → Consult maintainers
```

### Common Document Types by Directory

**`prd/` Examples:**
- System architecture specifications
- Feature requirements documents
- Technical design documents
- Component interface definitions

**`implementation/` Examples:**
- Current project status
- Migration procedures
- Implementation roadmaps
- Integration guides with code examples

**`enterprise/` Examples:**
- Deployment runbooks
- Security configuration guides
- Compliance documentation
- Business continuity procedures

**`api/` Examples:**
- REST API specifications
- GraphQL schemas
- Authentication procedures
- SDK documentation

**`user/` Examples:**
- Getting started tutorials
- User interface guides
- Troubleshooting procedures
- Best practices documentation

**`archive/` Examples:**
- Completed phase reports
- Previous versions of specifications
- Deprecated procedures
- Historical design decisions

---

## 🔄 Contribution Workflow

### Creating New Documentation

1. **Determine Placement**
   - Use the decision matrix above to identify the correct directory
   - Check if similar documentation already exists
   - Consider if you're updating existing content vs. creating new

2. **Follow Naming Conventions**
   ```
   descriptive-name.md           # Good: clear and descriptive
   COMPONENT_SYSTEM_SPEC.md      # Avoid: ALL_CAPS, use kebab-case
   temp-notes.md                 # Avoid: temporary or vague names
   ```

3. **Use Document Template**
   ```markdown
   # Document Title
   
   **Brief description of document purpose**
   
   ---
   
   ## Overview
   [Introduction and context]
   
   ## [Main Sections]
   [Content organized logically]
   
   ---
   
   **Last Updated**: [Date]
   **Version**: [Version Number]
   **Maintained By**: [Team/Person]
   ```

4. **Update Master Index**
   - Add your document to the appropriate section in `docs/README.md`
   - Include a brief description of the document's purpose
   - Ensure the link is correct and functional

### Updating Existing Documentation

1. **Identify the Authoritative Source**
   - Check `docs/README.md` for the canonical version
   - Avoid creating duplicate content
   - If multiple versions exist, consolidate or update references

2. **Preserve Structure**
   - Maintain existing section organization
   - Update content while keeping formatting consistent
   - Preserve important historical context where relevant

3. **Update Metadata**
   - Change "Last Updated" date
   - Increment version number appropriately
   - Update maintainer information if necessary

4. **Review Cross-References**
   - Check that all internal links still work
   - Update references to moved or renamed documents
   - Ensure related documents are updated as needed

### Quality Checklist

Before submitting documentation:

- [ ] **Clarity**: Is the purpose clear from the title and first paragraph?
- [ ] **Completeness**: Does it cover all necessary information for the intended audience?
- [ ] **Accuracy**: Is all information current and correct?
- [ ] **Structure**: Is content organized logically with proper headings?
- [ ] **Links**: Do all internal and external links work?
- [ ] **Formatting**: Is markdown syntax correct and consistent?
- [ ] **Index**: Is the document properly linked in `docs/README.md`?

---

## 🗂️ Archival Process

### When to Archive Documents

Archive documents when they are:
- **Superseded**: Replaced by newer, more comprehensive versions
- **Outdated**: No longer relevant to current system state
- **Deprecated**: Related to discontinued features or approaches
- **Historical**: Valuable for context but not for current operations

### Archival Procedure

1. **Determine Archive Category**
   ```
   archive/
   ├── deprecated/     # Documents that are no longer valid
   ├── old-versions/   # Previous versions of current documents  
   └── phase-reports/  # Completed project phase documentation
   ```

2. **Add Deprecation Watermark**
   Add this header to archived documents:
   ```markdown
   > **--- ARCHIVED DOCUMENT ---**
   > **This document is historical and no longer maintained. Do not use for current work.**
   > **The authoritative source for this topic is now [Link to new document].**
   > ---
   ```

3. **Update Cross-References**
   - Find all documents that link to the archived content
   - Update links to point to current authoritative sources
   - Use search tools to find references: `grep -r "old-filename" docs/`

4. **Update Master Index**
   - Remove the document from active sections in `docs/README.md`
   - Add reference to archived location if historically significant
   - Update any related navigation elements

### Archive Validation

After archiving:

- [ ] Original document moved to appropriate archive subdirectory
- [ ] Deprecation watermark added with link to replacement
- [ ] All cross-references updated to point to current sources
- [ ] Master index updated to reflect changes
- [ ] No broken links remain in active documentation

---

## 🔍 Link Management and Validation

### Link Checking Process

Regular link validation helps maintain documentation quality:

1. **Automated Checking**
   ```bash
   # Check for broken links in all markdown files
   find docs/ -name "*.md" -exec grep -l "http" {} \; | xargs -I {} echo "Checking: {}"
   
   # Find references to specific files
   grep -r "filename.md" docs/
   ```

2. **Manual Validation**
   - Test all links in newly created or updated documents
   - Verify that cross-references resolve correctly
   - Check that external links are current and accessible

3. **Common Link Issues**
   - Relative paths that break when documents move
   - References to archived documents without watermarks
   - External links that have become invalid
   - Case-sensitive filename mismatches

### Link Best Practices

- **Use relative paths** for internal links: `../user/guide.md`
- **Use descriptive link text**: `[User Guide](user/guide.md)` not `[here](user/guide.md)`
- **Verify links before submitting** changes
- **Update related documents** when moving or renaming files

---

## 👥 Team Adoption and Enforcement

### Communication Strategy

**Phase 1: Introduction (Week 1)**
- Share this contributing guide with all team members
- Conduct training session on new structure
- Identify documentation champions in each team

**Phase 2: Transition (Weeks 2-4)**
- Update existing workflows to use new structure
- Provide feedback and support for early adopters
- Address questions and refinements

**Phase 3: Enforcement (Ongoing)**
- Include documentation review in PR processes
- Regular audits of documentation placement
- Continuous improvement based on usage patterns

### Review Process

**Documentation Reviews Should Check:**
- Correct directory placement using decision matrix
- Proper formatting and structure
- Updated master index entries
- Valid cross-references and links
- No duplicate or redundant content

**Review Checklist for PRs:**
- [ ] New documents in correct directory
- [ ] Master index updated
- [ ] No duplicate content created
- [ ] Links tested and functional
- [ ] Follows formatting standards

### Enforcement Guidelines

**Soft Enforcement (Preferred)**
- Educational comments in PR reviews
- Slack reminders about documentation standards
- Regular team updates on documentation improvements

**Hard Enforcement (When Necessary)**
- PR blocking for significant violations
- Required documentation updates before feature merging
- Team lead intervention for repeated issues

---

## 🛠️ Tools and Automation

### Recommended Tools

**Link Checking:**
- `markdown-link-check` for automated link validation
- `grep` for finding cross-references
- Browser bookmarks for quick manual testing

**Writing and Editing:**
- VS Code with Markdown extensions
- Grammarly or similar for writing quality
- Git diff for tracking changes

**Structure Validation:**
- Directory listing tools for organization checks
- Markdown linters for format consistency
- Custom scripts for automation

### Automation Opportunities

**Future Improvements:**
- Automated link checking in CI/CD pipeline
- Document freshness monitoring
- Cross-reference validation scripts
- Archive watermark automation

---

## 📊 Success Metrics

### Documentation Quality Indicators

**Quantitative Metrics:**
- Reduced number of duplicate documents
- Decreased time to find information
- Lower documentation maintenance overhead
- Increased cross-reference accuracy

**Qualitative Metrics:**
- Improved developer onboarding experience
- Better user satisfaction with documentation
- Reduced support questions about finding information
- Enhanced team collaboration on documentation

### Monitoring and Improvement

**Regular Assessments:**
- Monthly documentation audits
- Quarterly user feedback surveys
- Annual structure effectiveness reviews
- Continuous improvement based on usage patterns

---

## 🚨 Common Issues and Solutions

### Document Placement Confusion

**Problem**: Unclear where to place new documentation
**Solution**: Use the decision matrix and ask maintainers when uncertain

### Duplicate Content Creation

**Problem**: Creating content that already exists elsewhere
**Solution**: Search existing documentation before creating new files

### Broken Cross-References

**Problem**: Links break when documents are moved or renamed
**Solution**: Use systematic search and replace when restructuring

### Archive Management

**Problem**: Archived documents without proper watermarking
**Solution**: Follow the archival procedure consistently

### Index Maintenance

**Problem**: Master index becomes outdated
**Solution**: Include index updates in all documentation PRs

---

## 📞 Getting Help

### Escalation Path

1. **Self-Service**: Check this guide and the master index
2. **Team Discussion**: Ask in documentation channel
3. **Maintainer Consultation**: Contact documentation team leads
4. **Team Lead Intervention**: For persistent issues or conflicts

### Contact Information

**Documentation Team:**
- **Primary Maintainer**: [Team Lead]
- **Technical Writing**: [Technical Writer]
- **Architecture Documentation**: [System Architect]
- **User Documentation**: [Product Manager]

**Communication Channels:**
- **Slack**: `#documentation`
- **Email**: `docs@leanvibe.dev`
- **GitHub Issues**: Use `documentation` label

---

## 📈 Continuous Improvement

### Feedback Mechanisms

**We Welcome Feedback On:**
- Documentation structure effectiveness
- Contribution process improvements
- Tool recommendations
- Training needs identification

**How to Provide Feedback:**
- GitHub issues with `documentation` label
- Direct messages to documentation team
- Team retrospective discussions
- Quarterly documentation surveys

### Evolution of Standards

This guide will evolve based on:
- Team usage patterns and needs
- Tool availability and improvements
- Project growth and complexity changes
- Industry best practices adoption

---

**Last Updated**: July 31, 2025  
**Version**: 1.0.0  
**Maintained By**: Documentation Architecture Team

---

*This guide ensures consistent, discoverable, and maintainable documentation for the LeanVibe Agent Hive project. Follow these standards to contribute effectively to our documentation ecosystem.*