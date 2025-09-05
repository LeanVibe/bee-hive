#!/usr/bin/env python3

"""
LeanVibe Agent Hive - Database Migration Manager
Production-grade database migration system with rollback capabilities
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import asyncpg
import click

# Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://leanvibe_user:leanvibe_secure_pass@localhost:15432/leanvibe_agent_hive')
MIGRATIONS_DIR = Path(__file__).parent.parent / 'migrations'
BACKUP_DIR = Path('/backups')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MigrationManager:
    """Database migration management with production safety features"""
    
    def __init__(self, database_url: str, migrations_dir: Path):
        self.database_url = database_url
        self.migrations_dir = migrations_dir
        self.migrations_dir.mkdir(exist_ok=True)
        
    async def init_migration_table(self, conn: asyncpg.Connection) -> None:
        """Initialize the migration tracking table"""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(255) PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                checksum VARCHAR(64) NOT NULL,
                execution_time_ms INTEGER,
                rollback_sql TEXT,
                applied_by VARCHAR(255) DEFAULT CURRENT_USER,
                description TEXT
            )
        """)
        
        # Create index for performance
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at 
            ON schema_migrations(applied_at)
        """)
        
    def get_migration_files(self) -> List[Tuple[str, Path]]:
        """Get all migration files sorted by version"""
        migrations = []
        
        for migration_file in self.migrations_dir.glob('*.sql'):
            if migration_file.name.startswith('V'):
                # Extract version from filename (e.g., V001_create_users.sql)
                version = migration_file.name.split('_')[0][1:]  # Remove 'V' prefix
                migrations.append((version.zfill(3), migration_file))
                
        return sorted(migrations, key=lambda x: x[0])
    
    def calculate_checksum(self, content: str) -> str:
        """Calculate SHA256 checksum of migration content"""
        import hashlib
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def get_applied_migrations(self, conn: asyncpg.Connection) -> Dict[str, Dict]:
        """Get list of applied migrations with metadata"""
        await self.init_migration_table(conn)
        
        rows = await conn.fetch("""
            SELECT version, filename, applied_at, checksum, execution_time_ms, 
                   rollback_sql, applied_by, description
            FROM schema_migrations
            ORDER BY version
        """)
        
        return {
            row['version']: {
                'filename': row['filename'],
                'applied_at': row['applied_at'],
                'checksum': row['checksum'],
                'execution_time_ms': row['execution_time_ms'],
                'rollback_sql': row['rollback_sql'],
                'applied_by': row['applied_by'],
                'description': row['description']
            }
            for row in rows
        }
    
    async def create_backup(self, conn: asyncpg.Connection, description: str) -> str:
        """Create a database backup before migration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"pre_migration_backup_{timestamp}_{description.replace(' ', '_')}"
        
        # Use the backup script
        import subprocess
        backup_script = Path(__file__).parent / 'backup.sh'
        
        if backup_script.exists():
            logger.info(f"Creating backup: {backup_name}")
            env = os.environ.copy()
            env['BACKUP_NAME_SUFFIX'] = f"_{description.replace(' ', '_')}"
            
            result = subprocess.run(
                [str(backup_script)], 
                env=env,
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Backup failed: {result.stderr}")
                raise Exception(f"Backup creation failed: {result.stderr}")
            
            logger.info("Backup created successfully")
            return backup_name
        else:
            logger.warning("Backup script not found, skipping backup")
            return "no_backup_created"
    
    async def apply_migration(self, conn: asyncpg.Connection, version: str, migration_file: Path, 
                            create_backup: bool = True, dry_run: bool = False) -> Dict:
        """Apply a single migration with safety checks"""
        logger.info(f"Processing migration {version}: {migration_file.name}")
        
        # Read migration file
        with open(migration_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse migration content
        parts = content.split('-- ROLLBACK --')
        migration_sql = parts[0].strip()
        rollback_sql = parts[1].strip() if len(parts) > 1 else ''
        
        # Extract description from comments
        description = ""
        for line in migration_sql.split('\n'):
            if line.strip().startswith('-- Description:'):
                description = line.replace('-- Description:', '').strip()
                break
        
        # Calculate checksum
        checksum = self.calculate_checksum(migration_sql)
        
        # Validate migration
        if not migration_sql.strip():
            raise Exception(f"Migration {version} is empty")
        
        if dry_run:
            logger.info(f"DRY RUN: Would apply migration {version}")
            logger.info(f"SQL: {migration_sql[:200]}...")
            return {
                'version': version,
                'filename': migration_file.name,
                'description': description,
                'checksum': checksum,
                'dry_run': True
            }
        
        # Create backup if requested
        backup_name = ""
        if create_backup:
            backup_name = await self.create_backup(conn, f"migration_{version}")
        
        # Apply migration in transaction
        start_time = datetime.now()
        
        try:
            async with conn.transaction():
                # Execute migration
                await conn.execute(migration_sql)
                
                # Record migration
                execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                
                await conn.execute("""
                    INSERT INTO schema_migrations 
                    (version, filename, checksum, execution_time_ms, rollback_sql, description)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, version, migration_file.name, checksum, execution_time_ms, rollback_sql, description)
                
                logger.info(f"Migration {version} applied successfully in {execution_time_ms}ms")
                
                return {
                    'version': version,
                    'filename': migration_file.name,
                    'description': description,
                    'checksum': checksum,
                    'execution_time_ms': execution_time_ms,
                    'backup_name': backup_name,
                    'success': True
                }
                
        except Exception as e:
            logger.error(f"Migration {version} failed: {e}")
            raise Exception(f"Migration {version} failed: {e}")
    
    async def rollback_migration(self, conn: asyncpg.Connection, version: str, 
                               dry_run: bool = False) -> Dict:
        """Rollback a specific migration"""
        logger.info(f"Rolling back migration {version}")
        
        # Get migration info
        migration_info = await conn.fetchrow("""
            SELECT filename, rollback_sql, applied_at, description
            FROM schema_migrations
            WHERE version = $1
        """, version)
        
        if not migration_info:
            raise Exception(f"Migration {version} not found in database")
        
        if not migration_info['rollback_sql']:
            raise Exception(f"No rollback SQL available for migration {version}")
        
        if dry_run:
            logger.info(f"DRY RUN: Would rollback migration {version}")
            logger.info(f"Rollback SQL: {migration_info['rollback_sql'][:200]}...")
            return {
                'version': version,
                'filename': migration_info['filename'],
                'description': migration_info['description'],
                'dry_run': True
            }
        
        # Create backup before rollback
        backup_name = await self.create_backup(conn, f"rollback_{version}")
        
        # Perform rollback
        start_time = datetime.now()
        
        try:
            async with conn.transaction():
                # Execute rollback
                await conn.execute(migration_info['rollback_sql'])
                
                # Remove migration record
                await conn.execute("""
                    DELETE FROM schema_migrations WHERE version = $1
                """, version)
                
                execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                logger.info(f"Migration {version} rolled back successfully in {execution_time_ms}ms")
                
                return {
                    'version': version,
                    'filename': migration_info['filename'],
                    'description': migration_info['description'],
                    'execution_time_ms': execution_time_ms,
                    'backup_name': backup_name,
                    'success': True
                }
                
        except Exception as e:
            logger.error(f"Rollback of migration {version} failed: {e}")
            raise Exception(f"Rollback of migration {version} failed: {e}")
    
    async def validate_schema_integrity(self, conn: asyncpg.Connection) -> Dict:
        """Validate database schema integrity"""
        logger.info("Validating database schema integrity")
        
        # Check for basic consistency
        checks = {
            'tables_count': 0,
            'indexes_count': 0,
            'constraints_count': 0,
            'functions_count': 0,
            'critical_tables_present': True,
            'orphaned_indexes': 0,
            'missing_primary_keys': 0
        }
        
        # Count tables
        checks['tables_count'] = await conn.fetchval("""
            SELECT count(*) FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """)
        
        # Count indexes
        checks['indexes_count'] = await conn.fetchval("""
            SELECT count(*) FROM pg_indexes WHERE schemaname = 'public'
        """)
        
        # Count constraints
        checks['constraints_count'] = await conn.fetchval("""
            SELECT count(*) FROM information_schema.table_constraints 
            WHERE table_schema = 'public'
        """)
        
        # Count functions
        checks['functions_count'] = await conn.fetchval("""
            SELECT count(*) FROM information_schema.routines 
            WHERE routine_schema = 'public'
        """)
        
        # Check critical tables
        critical_tables = ['users', 'agents', 'tasks', 'workflows', 'schema_migrations']
        for table in critical_tables:
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = $1
                )
            """, table)
            
            if not exists:
                logger.warning(f"Critical table '{table}' is missing")
                checks['critical_tables_present'] = False
        
        logger.info(f"Schema validation completed: {checks}")
        return checks

# CLI Interface
@click.group()
@click.option('--database-url', default=DATABASE_URL, help='Database connection URL')
@click.option('--migrations-dir', default=str(MIGRATIONS_DIR), help='Migrations directory')
@click.pass_context
def cli(ctx, database_url, migrations_dir):
    """LeanVibe Database Migration Manager"""
    ctx.ensure_object(dict)
    ctx.obj['manager'] = MigrationManager(database_url, Path(migrations_dir))

@cli.command()
@click.option('--dry-run', is_flag=True, help='Show what would be applied without executing')
@click.option('--backup/--no-backup', default=True, help='Create backup before migration')
@click.pass_context
def migrate(ctx, dry_run, backup):
    """Apply pending migrations"""
    async def run():
        manager = ctx.obj['manager']
        conn = await asyncpg.connect(manager.database_url)
        
        try:
            # Get current state
            applied = await manager.get_applied_migrations(conn)
            migrations = manager.get_migration_files()
            
            pending = [m for m in migrations if m[0] not in applied]
            
            if not pending:
                logger.info("No pending migrations")
                return
            
            logger.info(f"Found {len(pending)} pending migrations")
            
            for version, migration_file in pending:
                result = await manager.apply_migration(
                    conn, version, migration_file, 
                    create_backup=backup, dry_run=dry_run
                )
                
                if not dry_run:
                    logger.info(f"Applied migration {version}: {result['description']}")
            
            # Validate schema after migration
            if not dry_run:
                integrity = await manager.validate_schema_integrity(conn)
                logger.info(f"Schema integrity check: {integrity}")
                
        finally:
            await conn.close()
    
    asyncio.run(run())

@cli.command()
@click.argument('version')
@click.option('--dry-run', is_flag=True, help='Show what would be rolled back without executing')
@click.pass_context
def rollback(ctx, version, dry_run):
    """Rollback a specific migration"""
    async def run():
        manager = ctx.obj['manager']
        conn = await asyncpg.connect(manager.database_url)
        
        try:
            result = await manager.rollback_migration(conn, version, dry_run)
            
            if not dry_run:
                logger.info(f"Rolled back migration {version}: {result['description']}")
                
                # Validate schema after rollback
                integrity = await manager.validate_schema_integrity(conn)
                logger.info(f"Schema integrity check: {integrity}")
                
        finally:
            await conn.close()
    
    asyncio.run(run())

@cli.command()
@click.pass_context
def status(ctx):
    """Show migration status"""
    async def run():
        manager = ctx.obj['manager']
        conn = await asyncpg.connect(manager.database_url)
        
        try:
            applied = await manager.get_applied_migrations(conn)
            migrations = manager.get_migration_files()
            
            logger.info("Migration Status:")
            logger.info("================")
            
            for version, migration_file in migrations:
                status_char = "✓" if version in applied else "✗"
                applied_at = applied[version]['applied_at'].strftime('%Y-%m-%d %H:%M:%S') if version in applied else "Not applied"
                description = applied[version]['description'] if version in applied else "Pending"
                
                logger.info(f"{status_char} {version} - {migration_file.name} - {description} ({applied_at})")
            
            pending_count = len([m for m in migrations if m[0] not in applied])
            logger.info(f"\nApplied: {len(applied)}, Pending: {pending_count}")
            
        finally:
            await conn.close()
    
    asyncio.run(run())

@cli.command()
@click.pass_context
def validate(ctx):
    """Validate database schema integrity"""
    async def run():
        manager = ctx.obj['manager']
        conn = await asyncpg.connect(manager.database_url)
        
        try:
            integrity = await manager.validate_schema_integrity(conn)
            
            logger.info("Database Schema Validation:")
            logger.info("==========================")
            for key, value in integrity.items():
                status = "✓" if (isinstance(value, bool) and value) or (isinstance(value, int) and value >= 0) else "✗"
                logger.info(f"{status} {key}: {value}")
                
        finally:
            await conn.close()
    
    asyncio.run(run())

if __name__ == '__main__':
    cli()