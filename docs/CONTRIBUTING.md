# Documentation Contributing Guide

## Overview
This guide establishes the peer review process and quality standards for ongoing documentation consolidation and maintenance in the LeanVibe Agent Hive project.

## Peer Review Process

### Mandatory Review Requirements
**Every consolidation action must have peer review:**
- **Consolidation**: Merging 2+ documents requires review by second person
- **Content Changes**: Significant updates to authoritative documents require review
- **Archive Decisions**: Moving files to archive requires review approval
- **Cross-Reference Updates**: Link changes across multiple files require review

### Review Process Steps

#### 1. **Pre-Review Preparation**
Before requesting review:
- [ ] Document consolidation plan in scratchpad
- [ ] Identify unique content in all source files
- [ ] Create consolidated document draft
- [ ] Update cross-references
- [ ] Apply archive watermarks

#### 2. **Review Request Format**
Create review request with:
```markdown
## Consolidation Review Request

**Type**: [Tier 1/Tier 2/Tier 3] Consolidation
**Files Involved**: [List source files]
**Target Document**: [Consolidated file location]
**Unique Content Preserved**: [Summary]
**Cross-References Updated**: [List]
**Archive Actions**: [Files being archived]

**Validation Checklist**:
- [ ] No unique content lost
- [ ] All cross-references updated
- [ ] Archive watermarks applied
- [ ] Single source of truth established
```

#### 3. **Review Validation Checklist**
Reviewers must verify:
- [ ] **Content Preservation**: All unique information preserved
- [ ] **Quality Improvement**: Consolidated version is superior to sources
- [ ] **Cross-Reference Accuracy**: All links updated correctly
- [ ] **Archive Compliance**: Watermarks applied, historical access maintained
- [ ] **Documentation Standards**: Follows established format and style

#### 4. **Review Approval Process**
- **Approval**: Reviewer validates all checklist items ✅
- **Conditional Approval**: Minor fixes required before merge ⚠️
- **Rejection**: Major issues require rework ❌

### Quality Assurance Standards

#### Documentation Quality Requirements
- **Single Source of Truth**: One authoritative document per topic
- **Clear Ownership**: Each document has defined purpose and scope
- **Consistent Format**: Follow established templates and structure
- **Link Integrity**: All cross-references valid and updated
- **Archive Compliance**: Deprecated files properly watermarked

#### Content Standards
- **Accuracy**: Information must be current and validated
- **Completeness**: Cover all aspects of the topic comprehensively
- **Clarity**: Written for intended audience with clear examples
- **Consistency**: Terminology and style consistent across documents
- **Maintainability**: Structure supports ongoing updates

### Automation Integration

#### Automated Quality Checks
Run before review request:
```bash
# Link validation
python scripts/validate_documentation_links.py --output scratchpad/link_check.json

# Consolidation validation
/consolidate validate

# Cross-reference check
grep -r "old-filename" docs/
```

#### Review Documentation
Document all reviews in `scratchpad/peer_reviews/`:
- Review request details
- Reviewer feedback and validation
- Actions taken to address feedback
- Final approval confirmation

### Team Adoption Strategy

#### Training Requirements
All team members must understand:
- New documentation structure and organization
- Decision matrix for document placement
- Consolidation and archival processes
- Quality standards and review requirements

#### Enforcement Guidelines
- **No exceptions**: All consolidation requires peer review
- **Quality gates**: Automated checks must pass before review
- **Historical preservation**: Never delete, always archive
- **Documentation debt**: Address during regular development cycles

### Success Metrics

#### Measurable Outcomes
- **Reduced Redundancy**: <10% content overlap across documents
- **Improved Navigation**: <30 seconds to find any information
- **Maintenance Efficiency**: 70% reduction in update effort
- **Quality Consistency**: 95% link validity across all documentation

#### Monitoring Process
- Monthly documentation health checks
- Quarterly redundancy analysis
- Annual documentation architecture review
- Continuous link validation monitoring

## Implementation

### Immediate Actions
1. **Establish Review Team**: Identify peer reviewers for different domains
2. **Create Review Templates**: Standardize review request and approval formats
3. **Set Up Automation**: Configure automated quality checks
4. **Document Process**: Communicate new standards to all team members

### Ongoing Maintenance
- **Regular Reviews**: Schedule quarterly documentation health assessments
- **Process Refinement**: Continuously improve based on experience
- **Tool Enhancement**: Upgrade automation based on identified needs
- **Team Training**: Regular updates on documentation standards and processes

---

This peer review process ensures the systematic consolidation of 120+ markdown files maintains quality while eliminating redundancy and improving navigation for both human developers and AI agents.