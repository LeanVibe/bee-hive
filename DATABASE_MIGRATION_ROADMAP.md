# Database Migration Roadmap - LeanVibe Agent Hive 2.0

## 📋 Migration Overview

This document provides a detailed roadmap for database schema changes required to support LeanVibe Agent Hive 2.0 enhancements. The migration strategy ensures zero downtime deployment with comprehensive rollback capabilities.

**Migration Status:** Ready for execution with comprehensive validation

---

## 🗄️ Migration Strategy

### Core Principles
1. **Zero Downtime** - All migrations use non-blocking operations
2. **Backward Compatibility** - Existing systems continue to function
3. **Incremental Deployment** - Changes applied in small, safe steps
4. **Comprehensive Rollback** - Full rollback capability at each stage
5. **Performance Preservation** - No degradation of existing operations

### Migration Phases
- **Phase 1:** Index Creation (Non-blocking)
- **Phase 2:** Schema Enhancement (Additive only)
- **Phase 3:** Data Population (Background processing)
- **Phase 4:** Feature Enablement (Application layer)

---

## 📊 Current Database State Analysis

### Existing Tables Assessment

**✅ Already Implemented:**
```sql
-- Project Management Hierarchy (Complete)
project_management_projects         -- Projects with short_id support
project_management_epics           -- Epics with project relationships
project_management_prds            -- PRDs with epic relationships  
project_management_tasks           -- Tasks with full hierarchy

-- Agent System (Functional)
agents                             -- Agent definitions with short_id
agent_sessions                     -- Session tracking

-- Support Tables (Operational)
short_id_registry                  -- Short ID collision tracking
project_index                      -- Project indexing system
```

**🔧 Enhancement Needed:**
- Performance indexes for short ID lookups
- Agent-to-project assignment tracking
- Tmux session management integration
- Cross-project resource allocation tables

---

## 🎯 Migration Phase 1: Performance Indexes

### Objective
Create high-performance indexes for short ID lookups and project hierarchy queries without affecting existing operations.

### Duration: 2-4 hours (depending on data size)

### Migration Script 1.1: Short ID Indexes
```sql
-- Migration: 001_add_short_id_indexes.sql
-- Description: Add performance indexes for short ID lookups
-- Risk Level: LOW (read-only index creation)

BEGIN;

-- Project Management Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_project_short_id 
    ON project_management_projects(short_id) 
    WHERE short_id IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_epic_short_id 
    ON project_management_epics(short_id) 
    WHERE short_id IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_prd_short_id 
    ON project_management_prds(short_id) 
    WHERE short_id IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_short_id 
    ON project_management_tasks(short_id) 
    WHERE short_id IS NOT NULL;

-- Agent System Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_short_id 
    ON agents(short_id) 
    WHERE short_id IS NOT NULL;

-- Project Hierarchy Queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_epic_project_id 
    ON project_management_epics(project_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_prd_epic_id 
    ON project_management_prds(epic_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_prd_id 
    ON project_management_tasks(prd_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_project_id 
    ON project_management_tasks(project_id);

-- Performance Optimization Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_status_project 
    ON project_management_tasks(project_id, kanban_state) 
    WHERE kanban_state IN ('backlog', 'ready', 'in_progress');

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_assigned_agent 
    ON project_management_tasks(assigned_agent_id) 
    WHERE assigned_agent_id IS NOT NULL;

COMMIT;
```

### Migration Script 1.2: Search Optimization
```sql
-- Migration: 002_search_optimization_indexes.sql  
-- Description: Optimize search and filtering operations
-- Risk Level: LOW (read-only index creation)

BEGIN;

-- Text Search Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_project_name_search 
    ON project_management_projects 
    USING gin(to_tsvector('english', name || ' ' || coalesce(description, '')));

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_name_search 
    ON project_management_tasks 
    USING gin(to_tsvector('english', title || ' ' || coalesce(description, '')));

-- Status and Priority Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_project_status_created 
    ON project_management_projects(status, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_priority_status 
    ON project_management_tasks(priority, kanban_state, created_at DESC);

-- Agent Workload Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_status_type 
    ON agents(status, agent_type) 
    WHERE status IN ('idle', 'busy');

COMMIT;
```

### Validation Script 1.1
```sql
-- Verify index creation and performance
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE indexname LIKE 'idx_%short_id%' 
   OR indexname LIKE 'idx_%project%'
   OR indexname LIKE 'idx_%task%'
ORDER BY tablename, indexname;

-- Performance test queries
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM project_management_projects WHERE short_id = 'PRJ-A7B2';

EXPLAIN (ANALYZE, BUFFERS)
SELECT t.* FROM project_management_tasks t 
JOIN project_management_projects p ON t.project_id = p.id 
WHERE p.short_id = 'PRJ-A7B2' AND t.kanban_state = 'in_progress';
```

### Rollback Script 1.1
```sql
-- Rollback: Remove all Phase 1 indexes
DROP INDEX CONCURRENTLY IF EXISTS idx_project_short_id;
DROP INDEX CONCURRENTLY IF EXISTS idx_epic_short_id;
DROP INDEX CONCURRENTLY IF EXISTS idx_prd_short_id;
DROP INDEX CONCURRENTLY IF EXISTS idx_task_short_id;
DROP INDEX CONCURRENTLY IF EXISTS idx_agent_short_id;
DROP INDEX CONCURRENTLY IF EXISTS idx_epic_project_id;
DROP INDEX CONCURRENTLY IF EXISTS idx_prd_epic_id;
DROP INDEX CONCURRENTLY IF EXISTS idx_task_prd_id;
DROP INDEX CONCURRENTLY IF EXISTS idx_task_project_id;
DROP INDEX CONCURRENTLY IF EXISTS idx_task_status_project;
DROP INDEX CONCURRENTLY IF EXISTS idx_task_assigned_agent;
DROP INDEX CONCURRENTLY IF EXISTS idx_project_name_search;
DROP INDEX CONCURRENTLY IF EXISTS idx_task_name_search;
DROP INDEX CONCURRENTLY IF EXISTS idx_project_status_created;
DROP INDEX CONCURRENTLY IF EXISTS idx_task_priority_status;
DROP INDEX CONCURRENTLY IF EXISTS idx_agent_status_type;
```

---

## 🔧 Migration Phase 2: Schema Enhancement

### Objective
Add new columns and tables required for enhanced multi-project coordination without breaking existing functionality.

### Duration: 1-2 hours

### Migration Script 2.1: Agent Enhancement
```sql
-- Migration: 003_agent_project_assignment.sql
-- Description: Add project assignment tracking to agents
-- Risk Level: MEDIUM (schema changes)

BEGIN;

-- Add project assignment columns to agents table
ALTER TABLE agents 
ADD COLUMN IF NOT EXISTS current_project_id UUID 
    REFERENCES project_management_projects(id) ON DELETE SET NULL;

ALTER TABLE agents 
ADD COLUMN IF NOT EXISTS project_assignments JSONB DEFAULT '[]';

ALTER TABLE agents 
ADD COLUMN IF NOT EXISTS tmux_session_id VARCHAR(255);

ALTER TABLE agents 
ADD COLUMN IF NOT EXISTS max_concurrent_projects INTEGER DEFAULT 1 CHECK (max_concurrent_projects > 0);

ALTER TABLE agents 
ADD COLUMN IF NOT EXISTS project_capacity_percentage INTEGER DEFAULT 100 
    CHECK (project_capacity_percentage > 0 AND project_capacity_percentage <= 100);

-- Add metadata tracking
ALTER TABLE agents 
ADD COLUMN IF NOT EXISTS last_project_assignment TIMESTAMP;

ALTER TABLE agents 
ADD COLUMN IF NOT EXISTS total_projects_completed INTEGER DEFAULT 0;

-- Create index for efficient project queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_current_project 
    ON agents(current_project_id) 
    WHERE current_project_id IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_project_capacity 
    ON agents(project_capacity_percentage, status) 
    WHERE status IN ('idle', 'busy');

COMMIT;
```

### Migration Script 2.2: Task Orchestration Enhancement
```sql
-- Migration: 004_task_orchestration_enhancement.sql
-- Description: Add orchestration tracking to tasks
-- Risk Level: MEDIUM (schema changes)

BEGIN;

-- Add orchestration tracking to tasks
ALTER TABLE project_management_tasks 
ADD COLUMN IF NOT EXISTS orchestrator_session_id UUID;

ALTER TABLE project_management_tasks 
ADD COLUMN IF NOT EXISTS agent_assignment_history JSONB DEFAULT '[]';

ALTER TABLE project_management_tasks 
ADD COLUMN IF NOT EXISTS delegation_metadata JSONB DEFAULT '{}';

ALTER TABLE project_management_tasks 
ADD COLUMN IF NOT EXISTS tmux_session_info JSONB DEFAULT '{}';

-- Add performance tracking
ALTER TABLE project_management_tasks 
ADD COLUMN IF NOT EXISTS estimated_duration_minutes INTEGER;

ALTER TABLE project_management_tasks 
ADD COLUMN IF NOT EXISTS actual_duration_minutes INTEGER;

ALTER TABLE project_management_tasks 
ADD COLUMN IF NOT EXISTS complexity_score INTEGER DEFAULT 5 
    CHECK (complexity_score >= 1 AND complexity_score <= 10);

-- Add cross-project dependency tracking
ALTER TABLE project_management_tasks 
ADD COLUMN IF NOT EXISTS depends_on_tasks UUID[] DEFAULT '{}';

ALTER TABLE project_management_tasks 
ADD COLUMN IF NOT EXISTS blocks_tasks UUID[] DEFAULT '{}';

-- Create performance indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_orchestrator_session 
    ON project_management_tasks(orchestrator_session_id) 
    WHERE orchestrator_session_id IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_complexity_status 
    ON project_management_tasks(complexity_score, kanban_state);

-- GIN index for dependency arrays
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_dependencies 
    ON project_management_tasks USING gin(depends_on_tasks);

COMMIT;
```

### Migration Script 2.3: Session Management Tables
```sql
-- Migration: 005_session_management_tables.sql
-- Description: Create comprehensive session tracking
-- Risk Level: LOW (new tables only)

BEGIN;

-- Enhanced agent sessions table
CREATE TABLE IF NOT EXISTS agent_tmux_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Session identification
    session_name VARCHAR(255) NOT NULL UNIQUE,
    tmux_session_id VARCHAR(255) NOT NULL,
    
    -- Relationships
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    project_id UUID REFERENCES project_management_projects(id) ON DELETE CASCADE,
    
    -- Session configuration
    workspace_path TEXT NOT NULL,
    environment_vars JSONB DEFAULT '{}',
    
    -- Status tracking
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('creating', 'active', 'idle', 'busy', 'sleeping', 'error', 'terminated')),
    
    -- Performance metrics
    cpu_usage_percent DECIMAL(5,2) DEFAULT 0.0,
    memory_usage_mb INTEGER DEFAULT 0,
    disk_usage_mb INTEGER DEFAULT 0,
    
    -- Timing
    created_at TIMESTAMP DEFAULT NOW(),
    last_activity TIMESTAMP DEFAULT NOW(),
    terminated_at TIMESTAMP,
    
    -- Metadata
    session_metadata JSONB DEFAULT '{}',
    performance_history JSONB DEFAULT '[]'
);

-- Project resource allocation tracking
CREATE TABLE IF NOT EXISTS project_resource_allocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Resource allocation
    project_id UUID REFERENCES project_management_projects(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    
    -- Allocation details
    capacity_percentage INTEGER NOT NULL CHECK (capacity_percentage > 0 AND capacity_percentage <= 100),
    allocated_at TIMESTAMP DEFAULT NOW(),
    deallocated_at TIMESTAMP,
    
    -- Status
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'completed', 'cancelled')),
    
    -- Performance tracking
    tasks_completed INTEGER DEFAULT 0,
    average_task_duration_minutes DECIMAL(10,2),
    efficiency_score DECIMAL(5,2),
    
    -- Metadata
    allocation_metadata JSONB DEFAULT '{}',
    
    UNIQUE(project_id, agent_id, allocated_at)
);

-- Cross-project coordination tracking
CREATE TABLE IF NOT EXISTS cross_project_dependencies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Dependency relationship
    source_project_id UUID REFERENCES project_management_projects(id) ON DELETE CASCADE,
    target_project_id UUID REFERENCES project_management_projects(id) ON DELETE CASCADE,
    
    -- Dependency details
    dependency_type VARCHAR(50) NOT NULL CHECK (dependency_type IN ('blocks', 'requires', 'shares_resources', 'sequential')),
    description TEXT,
    
    -- Status
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'resolved', 'cancelled')),
    
    -- Timing
    created_at TIMESTAMP DEFAULT NOW(),
    resolved_at TIMESTAMP,
    
    -- Metadata
    dependency_metadata JSONB DEFAULT '{}',
    
    CHECK (source_project_id != target_project_id)
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_tmux_sessions_agent 
    ON agent_tmux_sessions(agent_id, status);

CREATE INDEX IF NOT EXISTS idx_tmux_sessions_project 
    ON agent_tmux_sessions(project_id, status);

CREATE INDEX IF NOT EXISTS idx_resource_allocations_project 
    ON project_resource_allocations(project_id, status);

CREATE INDEX IF NOT EXISTS idx_resource_allocations_agent 
    ON project_resource_allocations(agent_id, status);

CREATE INDEX IF NOT EXISTS idx_cross_project_deps_source 
    ON cross_project_dependencies(source_project_id, status);

CREATE INDEX IF NOT EXISTS idx_cross_project_deps_target 
    ON cross_project_dependencies(target_project_id, status);

COMMIT;
```

### Validation Script 2.1
```sql
-- Verify schema changes
SELECT 
    table_name, 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_name IN (
    'agents', 
    'project_management_tasks',
    'agent_tmux_sessions',
    'project_resource_allocations',
    'cross_project_dependencies'
)
AND column_name LIKE '%project%' 
   OR column_name LIKE '%session%' 
   OR column_name LIKE '%orchestrator%'
ORDER BY table_name, column_name;

-- Test constraint validation
INSERT INTO agent_tmux_sessions (session_name, tmux_session_id, workspace_path, status) 
VALUES ('test-session-' || gen_random_uuid(), 'tmux-test', '/tmp/test', 'active');

DELETE FROM agent_tmux_sessions WHERE session_name LIKE 'test-session-%';
```

### Rollback Script 2.1
```sql
-- Rollback Phase 2 changes
BEGIN;

-- Drop new tables
DROP TABLE IF EXISTS cross_project_dependencies;
DROP TABLE IF EXISTS project_resource_allocations;  
DROP TABLE IF EXISTS agent_tmux_sessions;

-- Remove columns from agents table
ALTER TABLE agents DROP COLUMN IF EXISTS current_project_id;
ALTER TABLE agents DROP COLUMN IF EXISTS project_assignments;
ALTER TABLE agents DROP COLUMN IF EXISTS tmux_session_id;
ALTER TABLE agents DROP COLUMN IF EXISTS max_concurrent_projects;
ALTER TABLE agents DROP COLUMN IF EXISTS project_capacity_percentage;
ALTER TABLE agents DROP COLUMN IF EXISTS last_project_assignment;
ALTER TABLE agents DROP COLUMN IF EXISTS total_projects_completed;

-- Remove columns from tasks table
ALTER TABLE project_management_tasks DROP COLUMN IF EXISTS orchestrator_session_id;
ALTER TABLE project_management_tasks DROP COLUMN IF EXISTS agent_assignment_history;
ALTER TABLE project_management_tasks DROP COLUMN IF EXISTS delegation_metadata;
ALTER TABLE project_management_tasks DROP COLUMN IF EXISTS tmux_session_info;
ALTER TABLE project_management_tasks DROP COLUMN IF EXISTS estimated_duration_minutes;
ALTER TABLE project_management_tasks DROP COLUMN IF EXISTS actual_duration_minutes;
ALTER TABLE project_management_tasks DROP COLUMN IF EXISTS complexity_score;
ALTER TABLE project_management_tasks DROP COLUMN IF EXISTS depends_on_tasks;
ALTER TABLE project_management_tasks DROP COLUMN IF EXISTS blocks_tasks;

COMMIT;
```

---

## 📊 Migration Phase 3: Data Population

### Objective
Populate new columns with default values and ensure data consistency without affecting system performance.

### Duration: 30 minutes - 2 hours (depending on data volume)

### Migration Script 3.1: Short ID Backfill
```sql
-- Migration: 006_short_id_backfill.sql
-- Description: Generate short IDs for existing records
-- Risk Level: LOW (data updates with proper validation)

BEGIN;

-- Create temporary function for safe short ID generation
CREATE OR REPLACE FUNCTION generate_safe_short_id(entity_prefix text) 
RETURNS text LANGUAGE plpgsql AS $$
DECLARE
    new_short_id text;
    counter integer := 0;
BEGIN
    LOOP
        -- Generate random short ID
        new_short_id := entity_prefix || '-' || 
            translate(encode(gen_random_bytes(3), 'base32'), '018', '239');
        
        -- Check for collisions across all entities
        IF NOT EXISTS (
            SELECT 1 FROM project_management_projects WHERE short_id = new_short_id
            UNION ALL
            SELECT 1 FROM project_management_epics WHERE short_id = new_short_id  
            UNION ALL
            SELECT 1 FROM project_management_prds WHERE short_id = new_short_id
            UNION ALL
            SELECT 1 FROM project_management_tasks WHERE short_id = new_short_id
            UNION ALL
            SELECT 1 FROM agents WHERE short_id = new_short_id
        ) THEN
            RETURN new_short_id;
        END IF;
        
        counter := counter + 1;
        IF counter > 10 THEN
            RAISE EXCEPTION 'Unable to generate unique short ID after 10 attempts';
        END IF;
    END LOOP;
END;
$$;

-- Backfill projects
UPDATE project_management_projects 
SET short_id = generate_safe_short_id('PRJ')
WHERE short_id IS NULL;

-- Backfill epics  
UPDATE project_management_epics 
SET short_id = generate_safe_short_id('EPC')
WHERE short_id IS NULL;

-- Backfill PRDs
UPDATE project_management_prds 
SET short_id = generate_safe_short_id('PRD')
WHERE short_id IS NULL;

-- Backfill tasks
UPDATE project_management_tasks 
SET short_id = generate_safe_short_id('TSK')
WHERE short_id IS NULL;

-- Backfill agents
UPDATE agents 
SET short_id = generate_safe_short_id('AGT')
WHERE short_id IS NULL;

-- Clean up temporary function
DROP FUNCTION generate_safe_short_id(text);

-- Add NOT NULL constraints now that data is populated
ALTER TABLE project_management_projects 
ALTER COLUMN short_id SET NOT NULL;

ALTER TABLE project_management_epics 
ALTER COLUMN short_id SET NOT NULL;

ALTER TABLE project_management_prds 
ALTER COLUMN short_id SET NOT NULL;

ALTER TABLE project_management_tasks 
ALTER COLUMN short_id SET NOT NULL;

ALTER TABLE agents 
ALTER COLUMN short_id SET NOT NULL;

COMMIT;
```

### Migration Script 3.2: Default Value Population
```sql
-- Migration: 007_default_value_population.sql
-- Description: Set sensible defaults for new columns
-- Risk Level: LOW (safe default updates)

BEGIN;

-- Update agent metadata with defaults
UPDATE agents 
SET 
    project_assignments = '[]'::jsonb,
    max_concurrent_projects = CASE 
        WHEN agent_type = 'meta_agent' THEN 5
        WHEN agent_type IN ('senior_developer', 'architect') THEN 3
        ELSE 1
    END,
    project_capacity_percentage = 100,
    total_projects_completed = 0
WHERE project_assignments IS NULL OR max_concurrent_projects IS NULL;

-- Update task metadata with estimates
UPDATE project_management_tasks 
SET 
    agent_assignment_history = '[]'::jsonb,
    delegation_metadata = '{}'::jsonb,
    tmux_session_info = '{}'::jsonb,
    complexity_score = CASE 
        WHEN task_type IN ('architecture', 'research') THEN 8
        WHEN task_type IN ('feature_development', 'refactoring') THEN 6
        WHEN task_type IN ('bug_fix', 'testing') THEN 4
        ELSE 5
    END,
    estimated_duration_minutes = CASE 
        WHEN complexity_score >= 8 THEN 480  -- 8 hours
        WHEN complexity_score >= 6 THEN 240  -- 4 hours  
        WHEN complexity_score >= 4 THEN 120  -- 2 hours
        ELSE 60  -- 1 hour
    END,
    depends_on_tasks = '{}',
    blocks_tasks = '{}'
WHERE agent_assignment_history IS NULL OR delegation_metadata IS NULL;

COMMIT;
```

### Validation Script 3.1
```sql
-- Verify data population
SELECT 
    'Projects' as entity_type,
    COUNT(*) as total_records,
    COUNT(short_id) as records_with_short_id,
    COUNT(DISTINCT short_id) as unique_short_ids
FROM project_management_projects
UNION ALL
SELECT 
    'Epics' as entity_type,
    COUNT(*) as total_records,
    COUNT(short_id) as records_with_short_id,
    COUNT(DISTINCT short_id) as unique_short_ids
FROM project_management_epics
UNION ALL
SELECT 
    'Tasks' as entity_type,
    COUNT(*) as total_records,
    COUNT(short_id) as records_with_short_id,
    COUNT(DISTINCT short_id) as unique_short_ids
FROM project_management_tasks
UNION ALL
SELECT 
    'Agents' as entity_type,
    COUNT(*) as total_records,
    COUNT(short_id) as records_with_short_id,
    COUNT(DISTINCT short_id) as unique_short_ids
FROM agents;

-- Check for any short ID collisions
SELECT short_id, COUNT(*) 
FROM (
    SELECT short_id FROM project_management_projects
    UNION ALL SELECT short_id FROM project_management_epics
    UNION ALL SELECT short_id FROM project_management_prds  
    UNION ALL SELECT short_id FROM project_management_tasks
    UNION ALL SELECT short_id FROM agents
) all_short_ids
GROUP BY short_id 
HAVING COUNT(*) > 1;
```

---

## ⚡ Migration Phase 4: Performance Optimization

### Objective
Fine-tune database performance for the new multi-project coordination features.

### Duration: 1-2 hours

### Migration Script 4.1: Advanced Indexes
```sql
-- Migration: 008_advanced_performance_indexes.sql
-- Description: Create specialized indexes for complex queries
-- Risk Level: LOW (performance enhancement only)

BEGIN;

-- Multi-column indexes for common query patterns
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_project_agent_status 
    ON project_management_tasks(project_id, assigned_agent_id, kanban_state);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_project_capacity 
    ON agents(current_project_id, project_capacity_percentage, status)
    WHERE status IN ('idle', 'busy');

-- Partial indexes for active records
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_active_project_sessions 
    ON agent_tmux_sessions(project_id, agent_id) 
    WHERE status IN ('active', 'busy');

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_active_resource_allocations 
    ON project_resource_allocations(project_id, capacity_percentage)
    WHERE status = 'active';

-- Functional indexes for JSON queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_complexity_metadata 
    ON project_management_tasks USING gin(delegation_metadata) 
    WHERE delegation_metadata != '{}'::jsonb;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_project_assignments 
    ON agents USING gin(project_assignments) 
    WHERE project_assignments != '[]'::jsonb;

-- Time-based indexes for performance queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_duration_performance 
    ON project_management_tasks(actual_duration_minutes, estimated_duration_minutes)
    WHERE actual_duration_minutes IS NOT NULL AND estimated_duration_minutes IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_session_performance_history 
    ON agent_tmux_sessions(last_activity DESC, cpu_usage_percent, memory_usage_mb)
    WHERE status = 'active';

COMMIT;
```

### Migration Script 4.2: Database Optimization
```sql
-- Migration: 009_database_optimization.sql
-- Description: Optimize database settings for multi-project workload
-- Risk Level: MEDIUM (configuration changes)

BEGIN;

-- Update table statistics for better query planning
ANALYZE project_management_projects;
ANALYZE project_management_epics;
ANALYZE project_management_prds;
ANALYZE project_management_tasks;
ANALYZE agents;
ANALYZE agent_tmux_sessions;
ANALYZE project_resource_allocations;
ANALYZE cross_project_dependencies;

-- Add table partitioning for large session logs (if needed)
-- Note: This would be implemented based on data volume assessment

-- Create materialized view for project dashboard queries
CREATE MATERIALIZED VIEW IF NOT EXISTS project_dashboard_summary AS
SELECT 
    p.id as project_id,
    p.short_id as project_short_id,
    p.name as project_name,
    p.status as project_status,
    COUNT(DISTINCT e.id) as epic_count,
    COUNT(DISTINCT t.id) as task_count,
    COUNT(DISTINCT t.id) FILTER (WHERE t.kanban_state = 'completed') as completed_tasks,
    COUNT(DISTINCT a.id) as assigned_agents,
    AVG(t.complexity_score) as avg_complexity,
    p.updated_at as last_updated
FROM project_management_projects p
LEFT JOIN project_management_epics e ON e.project_id = p.id
LEFT JOIN project_management_tasks t ON t.project_id = p.id
LEFT JOIN agents a ON a.current_project_id = p.id
WHERE p.status != 'archived'
GROUP BY p.id, p.short_id, p.name, p.status, p.updated_at;

-- Create index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_project_dashboard_project_id 
    ON project_dashboard_summary(project_id);

CREATE INDEX IF NOT EXISTS idx_project_dashboard_status 
    ON project_dashboard_summary(project_status, last_updated DESC);

-- Refresh schedule for materialized view (would be handled by application)
-- REFRESH MATERIALIZED VIEW CONCURRENTLY project_dashboard_summary;

COMMIT;
```

### Performance Validation Script
```sql
-- Test query performance after optimization
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT 
    p.short_id,
    p.name,
    COUNT(t.id) as total_tasks,
    COUNT(t.id) FILTER (WHERE t.kanban_state = 'in_progress') as active_tasks,
    array_agg(DISTINCT a.short_id) as assigned_agents
FROM project_management_projects p
LEFT JOIN project_management_tasks t ON t.project_id = p.id  
LEFT JOIN agents a ON a.current_project_id = p.id
WHERE p.status = 'active'
GROUP BY p.id, p.short_id, p.name
ORDER BY p.updated_at DESC
LIMIT 10;

-- Test short ID lookup performance
EXPLAIN (ANALYZE, BUFFERS)
SELECT p.*, 
       (SELECT COUNT(*) FROM project_management_tasks WHERE project_id = p.id) as task_count
FROM project_management_projects p 
WHERE p.short_id = 'PRJ-A7B2';

-- Test cross-project resource queries
EXPLAIN (ANALYZE, BUFFERS)
SELECT 
    a.short_id as agent_id,
    a.current_project_id,
    a.project_capacity_percentage,
    COUNT(t.id) as assigned_tasks
FROM agents a
LEFT JOIN project_management_tasks t ON t.assigned_agent_id = a.id AND t.kanban_state IN ('ready', 'in_progress')
WHERE a.status IN ('idle', 'busy')
GROUP BY a.id, a.short_id, a.current_project_id, a.project_capacity_percentage
ORDER BY a.project_capacity_percentage ASC;
```

---

## 📋 Pre-Migration Checklist

### System Preparation
- [ ] **Database Backup**: Full backup completed and verified
- [ ] **Staging Environment**: Identical to production, migrations tested
- [ ] **Monitoring Setup**: Database performance monitoring active
- [ ] **Rollback Scripts**: All rollback procedures tested
- [ ] **Application Compatibility**: Existing APIs validated with new schema

### Risk Assessment
- [ ] **Peak Hours Avoided**: Migration scheduled during low-traffic period
- [ ] **Resource Availability**: Sufficient database resources allocated
- [ ] **Team Availability**: Database and application teams on standby
- [ ] **Communication Plan**: Stakeholders notified of migration timeline

### Performance Baseline
```sql
-- Capture performance baseline before migration
SELECT 
    schemaname,
    tablename,
    n_tup_ins + n_tup_upd + n_tup_del as total_operations,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch
FROM pg_stat_user_tables 
WHERE tablename LIKE '%project%' OR tablename LIKE '%agent%'
ORDER BY total_operations DESC;

-- Query performance baseline
\timing on
SELECT COUNT(*) FROM project_management_projects;
SELECT COUNT(*) FROM project_management_tasks;
SELECT COUNT(*) FROM agents;
```

---

## 🚨 Emergency Rollback Procedures

### Immediate Rollback (< 5 minutes)
```sql
-- Emergency: Disable new features at application level
-- This would be handled by application configuration
UPDATE system_config SET value = 'false' WHERE key = 'feature.short_id_enabled';
UPDATE system_config SET value = 'false' WHERE key = 'feature.multi_project_enabled';
```

### Partial Rollback (< 30 minutes)
```sql
-- Drop new indexes if they cause performance issues
DROP INDEX CONCURRENTLY IF EXISTS idx_project_short_id;
DROP INDEX CONCURRENTLY IF EXISTS idx_task_project_agent_status;
-- ... (continue with specific problematic indexes)
```

### Full Schema Rollback (< 2 hours)
```sql
-- Complete rollback to pre-migration state
-- Execute all phase rollback scripts in reverse order:
-- 1. Phase 4 rollback (drop optimization indexes)
-- 2. Phase 3 rollback (revert data changes)
-- 3. Phase 2 rollback (drop new columns/tables)  
-- 4. Phase 1 rollback (drop performance indexes)
```

### Data Integrity Verification Post-Rollback
```sql
-- Verify system integrity after rollback
SELECT COUNT(*) FROM project_management_projects;
SELECT COUNT(*) FROM project_management_tasks WHERE assigned_agent_id IS NOT NULL;
SELECT COUNT(*) FROM agents WHERE status IN ('idle', 'busy');

-- Test critical application queries
SELECT p.id, p.name FROM project_management_projects p LIMIT 5;
SELECT t.id, t.title FROM project_management_tasks t LIMIT 5;
```

---

## 📊 Success Metrics and Monitoring

### Performance Metrics
- **Index Creation Time**: Target <4 hours, Alert >6 hours
- **Query Performance**: <50ms for short ID lookups
- **Migration Downtime**: Target 0 seconds, Alert >30 seconds
- **Data Integrity**: 100% data preservation

### Monitoring Queries
```sql
-- Monitor migration progress
SELECT 
    phase,
    start_time,
    end_time,
    status,
    records_processed,
    errors_encountered
FROM migration_log 
ORDER BY start_time DESC;

-- Monitor index creation progress
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE indexname LIKE 'idx_%'
ORDER BY idx_scan DESC;

-- Monitor query performance post-migration
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    min_time,
    max_time
FROM pg_stat_statements 
WHERE query LIKE '%short_id%' OR query LIKE '%project_management_%'
ORDER BY total_time DESC;
```

### Alert Thresholds
- **Query Response Time** >200ms: Warning, >500ms: Critical
- **Index Usage** <70%: Investigation needed
- **Migration Duration** >Expected +50%: Escalation required
- **Error Rate** >0.1%: Immediate attention

---

## 🎯 Post-Migration Validation

### Comprehensive Validation Script
```sql
-- Migration validation suite
DO $$ 
DECLARE 
    validation_errors TEXT := '';
    test_result BOOLEAN;
    record_count INTEGER;
BEGIN
    -- Test 1: All tables exist
    SELECT COUNT(*) INTO record_count 
    FROM information_schema.tables 
    WHERE table_name IN (
        'agent_tmux_sessions',
        'project_resource_allocations', 
        'cross_project_dependencies'
    );
    
    IF record_count != 3 THEN
        validation_errors := validation_errors || 'Missing required tables. ';
    END IF;
    
    -- Test 2: All indexes exist
    SELECT COUNT(*) INTO record_count
    FROM pg_indexes 
    WHERE indexname LIKE 'idx_%short_id%';
    
    IF record_count < 5 THEN
        validation_errors := validation_errors || 'Missing short ID indexes. ';
    END IF;
    
    -- Test 3: Short IDs populated
    SELECT CASE 
        WHEN (SELECT COUNT(*) FROM project_management_projects WHERE short_id IS NULL) > 0
        THEN FALSE ELSE TRUE 
    END INTO test_result;
    
    IF NOT test_result THEN
        validation_errors := validation_errors || 'Short IDs not populated for all projects. ';
    END IF;
    
    -- Test 4: Performance test
    PERFORM 1 FROM project_management_projects WHERE short_id = 'PRJ-TEST' LIMIT 1;
    -- If this takes >50ms, something is wrong
    
    -- Output results
    IF validation_errors = '' THEN
        RAISE NOTICE 'All migration validations passed successfully!';
    ELSE
        RAISE EXCEPTION 'Migration validation failed: %', validation_errors;
    END IF;
END $$;
```

### Application Integration Tests
```bash
# Test CLI integration with new schema
hive project list --format json | jq '.[] | .short_id'
hive task create PRJ-TEST "Test task" --type testing
hive agent list --format table

# Test API endpoints
curl -s "$API_BASE/api/v1/projects" | jq '.[] | .short_id'
curl -s "$API_BASE/api/v1/agents" | jq '.[] | {short_id, current_project_id}'
```

---

## 📈 Future Migration Considerations

### Scalability Planning
- **Data Growth**: Plan for 10x data growth over 2 years
- **Concurrent Projects**: Support for 1000+ active projects
- **Agent Scale**: Support for 500+ concurrent agents
- **Session Management**: 1000+ active tmux sessions

### Performance Evolution
- **Partitioning Strategy**: Time-based partitioning for large tables
- **Read Replicas**: Separate read workloads for analytics
- **Caching Layer**: Redis caching for frequently accessed data
- **Archive Strategy**: Automated archival of completed projects

### Monitoring Evolution
```sql
-- Future performance monitoring queries
CREATE OR REPLACE VIEW project_performance_metrics AS
SELECT 
    p.short_id,
    p.name,
    COUNT(DISTINCT t.id) as total_tasks,
    AVG(t.actual_duration_minutes / NULLIF(t.estimated_duration_minutes, 0)) as estimation_accuracy,
    COUNT(DISTINCT a.id) as agents_used,
    AVG(ra.efficiency_score) as avg_efficiency
FROM project_management_projects p
LEFT JOIN project_management_tasks t ON t.project_id = p.id
LEFT JOIN project_resource_allocations ra ON ra.project_id = p.id
LEFT JOIN agents a ON ra.agent_id = a.id
WHERE p.status != 'archived'
  AND p.created_at > NOW() - INTERVAL '30 days'
GROUP BY p.id, p.short_id, p.name;
```

---

**Migration Status: 🚀 READY FOR EXECUTION**

*This comprehensive database migration roadmap ensures zero-downtime deployment of LeanVibe Agent Hive 2.0 enhancements with full rollback capabilities and performance optimization.*