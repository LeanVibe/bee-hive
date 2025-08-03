# Root Directory Cleanup Plan - LeanVibe Agent Hive

## Current State Analysis
**Root directory contains**: 77 files and directories (**Way too many!**)  
**Goal**: Reduce to ~15-20 essential items with logical organization

## Problem Statement
The project root is cluttered with:
- Multiple temporary/generated files (JSON reports, logs)
- Demo and test scripts scattered throughout
- Development artifacts mixed with core project files  
- Lack of clear separation between production code and development tools

## Proposed Organization Strategy

### ✅ **KEEP IN ROOT** (Essential Project Files - 15 items)

#### Core Configuration Files (6)
- `pyproject.toml` - Python project configuration
- `docker-compose.yml` - Container orchestration
- `Dockerfile` - Container definition
- `alembic.ini` - Database migration config
- `pytest.ini` - Test configuration
- `.gitignore` - Git configuration

#### Essential Documentation (5)
- `README.md` - Project overview
- `CLAUDE.md` - AI agent instructions
- `CONTRIBUTING.md` - Development guidelines
- `GETTING_STARTED.md` - User onboarding
- `SECURITY.md` - Security policy
- `LICENSE` - Legal requirements

#### Core Directories (4)
- `app/` - Main application code
- `tests/` - Test suite
- `docs/` - Documentation (already organized)
- `migrations/` - Database migrations

### 🗂️ **ORGANIZE INTO SUBDIRECTORIES**

#### 1. **Development Tools** → `dev/`
**Purpose**: Development scripts, demos, and utilities
```
dev/
├── demos/           # Demo scripts and examples
├── validation/      # Validation scripts
├── performance/     # Performance testing
└── utilities/       # Development utilities
```

**Files to move**:
- `*_demo.py` files (7 files)
- `validate_*.py` files (2 files)  
- `test_*.py` files in root (13 files)
- `comprehensive_qa_validation.py`
- `configuration_audit.py`
- `dashboard_integration_validation.py`
- `fix_database_compatibility.py`
- `phase_*_demonstration.py` files (4 files)

#### 2. **Generated Reports** → `reports/`
**Purpose**: Generated analysis and validation reports
```
reports/
├── audits/          # Security and configuration audits
├── performance/     # Performance validation results
├── qa/             # Quality assurance reports  
└── logs/           # Execution logs
```

**Files to move**:
- `audit_report_*.json` files (4 files)
- `bandit_report.json`
- `coverage.json`
- `performance_validation_results*.json` files (2 files)
- `qa_comprehensive_validation_report.json`
- `security_validation_post_fix.json`
- `phase_1_*.json` files (3 files)
- `phase_4_milestone_report.json`
- `phase2_demonstration.log`
- `logs/` directory

#### 3. **Infrastructure** → `infrastructure/`
**Purpose**: Deployment, monitoring, and infrastructure code
```
infrastructure/
├── monitoring/      # Grafana, Prometheus configs
├── mobile-pwa/      # PWA application
├── frontend/        # Frontend application
└── config/          # Configuration files
```

**Directories to move**:
- `monitoring/`
- `mobile-pwa/`
- `frontend/`
- `config/`

#### 4. **Development Resources** → `resources/`
**Purpose**: Schemas, contracts, examples, and reference materials
```
resources/
├── schemas/         # Data schemas
├── api_contracts/   # API contracts
├── examples/        # Code examples
├── mock_servers/    # Mock server implementations
└── workspaces/      # Development workspaces
```

**Directories to move**:
- `schemas/`
- `api_contracts/`
- `examples/`
- `mock_servers/`
- `workspaces/`

#### 5. **Development State** → `dev-state/`
**Purpose**: Checkpoints, repositories, temporary development state
```
dev-state/
├── checkpoints/     # Git checkpoints
├── repositories/    # Repository clones
└── cache/          # Development cache
```

**Directories to move**:
- `checkpoints/`
- `repositories/`

### 🗑️ **REMOVE/CLEAN**

#### Build Artifacts
- `__pycache__/` - Python cache (should be in .gitignore)
- Any other build artifacts

#### Duplicate/Obsolete
- `demo/` directory (merge with dev/demos/)

## Implementation Steps

### Phase 1: Create New Directory Structure
```bash
mkdir -p dev/{demos,validation,performance,utilities}
mkdir -p reports/{audits,performance,qa,logs}
mkdir -p infrastructure/
mkdir -p resources/
mkdir -p dev-state/
```

### Phase 2: Move Files Systematically
```bash
# Move demo and test scripts
mv *_demo.py dev/demos/
mv test_*.py dev/validation/
mv validate_*.py dev/validation/
mv phase_*_demonstration.py dev/demos/

# Move generated reports
mv audit_report_*.json reports/audits/
mv *_report.json reports/qa/
mv *_results.json reports/performance/
mv *.log reports/logs/
mv coverage.json reports/qa/

# Move infrastructure
mv monitoring/ infrastructure/
mv mobile-pwa/ infrastructure/
mv frontend/ infrastructure/
mv config/ infrastructure/

# Move resources
mv schemas/ resources/
mv api_contracts/ resources/
mv examples/ resources/
mv mock_servers/ resources/
mv workspaces/ resources/

# Move development state
mv checkpoints/ dev-state/
mv repositories/ dev-state/
```

### Phase 3: Clean Build Artifacts
```bash
# Remove Python cache
rm -rf __pycache__/

# Remove any other build artifacts
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

### Phase 4: Update References
- Update any scripts that reference moved files
- Update Docker configurations if needed
- Update CI/CD pipelines
- Update documentation

## Expected Results

### Before Cleanup
```
/ (77 items - cluttered and confusing)
├── Essential files mixed with temporary files
├── Demo scripts scattered throughout
├── Generated reports in root
└── No clear organization
```

### After Cleanup
```
/ (15-20 items - clean and organized)
├── pyproject.toml, docker-compose.yml, etc. (config)
├── README.md, CLAUDE.md, etc. (docs)
├── app/, tests/, docs/, migrations/ (core)
├── dev/ (development tools and scripts)
├── reports/ (generated reports and logs)
├── infrastructure/ (deployment and infra)
├── resources/ (schemas, examples, workspaces)
└── dev-state/ (checkpoints and temp state)
```

## Benefits

### For Developers
- **Clear project structure** - easy to understand project layout
- **Faster navigation** - know exactly where to find things
- **Reduced confusion** - no more mixing of core files with temporary files
- **Better onboarding** - new developers can understand structure quickly

### For CI/CD
- **Cleaner builds** - no unnecessary files in build context
- **Better caching** - can cache different directories differently
- **Improved security** - sensitive files clearly separated

### For Maintenance
- **Easier cleanup** - can clean entire directories of temporary files
- **Better backups** - can backup core vs generated content differently
- **Improved organization** - logical grouping of related files

## Risk Mitigation

### Before Moving Files
1. **Create backup** of current state
2. **Check all references** to files being moved
3. **Update configurations** that reference file paths
4. **Test functionality** after each major move

### During Implementation
1. **Move incrementally** - one category at a time
2. **Test after each phase** - ensure nothing breaks
3. **Update documentation** as files are moved
4. **Communicate changes** to team members

## Success Metrics

- **Root directory items**: Reduced from 77 to ~15-20 items
- **Organization clarity**: Clear separation of concerns
- **Developer satisfaction**: Faster navigation and understanding
- **Build performance**: Improved due to cleaner structure
- **Maintenance efficiency**: Easier to manage and clean up

---

This plan transforms the cluttered root directory into a clean, well-organized project structure that follows industry best practices for Python/FastAPI projects.