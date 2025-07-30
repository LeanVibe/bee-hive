#!/usr/bin/env python3
"""
Configuration Audit Script for LeanVibe Agent Hive 2.0

Validates all environment variables, database schema, service configurations,
and system readiness before proceeding with integration resolution.

Run this before any integration fixes to ensure a stable foundation.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis
import docker
from alembic.config import Config
from alembic import command
from alembic.script import ScriptDirectory
from alembic.runtime.environment import EnvironmentContext

# Import project modules  
from app.core.database import get_session, init_database
from app.core.redis import get_redis, init_redis
from app.core.config import settings
from app.core.correlation import CorrelationManager, EnhancedLogger

logger = structlog.get_logger(__name__)


class ConfigurationAuditor:
    """
    Comprehensive configuration audit for LeanVibe Agent Hive 2.0.
    
    Validates:
    - Environment variables and configuration
    - Database schema and migrations
    - Redis configuration and connectivity
    - Docker services and health
    - File system permissions and structure
    """
    
    def __init__(self):
        self.audit_results: Dict[str, Any] = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'UNKNOWN',
            'checks': {},
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        self.correlation_id = CorrelationManager.generate_id()
        
    async def run_comprehensive_audit(self) -> Dict[str, Any]:
        """
        Run complete configuration audit.
        
        Returns:
            Comprehensive audit results
        """
        with CorrelationManager.correlation_context(
            self.correlation_id, 
            {'audit_type': 'configuration', 'component': 'system_audit'}
        ):
            audit_logger = EnhancedLogger.get_logger("configuration_auditor")
            audit_logger.info("Starting comprehensive configuration audit")
            
            # Initialize services for testing
            try:
                audit_logger.info("Initializing database for audit")
                await init_database()
            except Exception as e:
                audit_logger.warning("Database initialization failed, will test connection separately", error=str(e))
            
            try:
                audit_logger.info("Initializing Redis for audit")
                await init_redis()
            except Exception as e:
                audit_logger.warning("Redis initialization failed, will test connection separately", error=str(e))
            
            # Run all audit checks
            await self._audit_environment_variables()
            await self._audit_database_configuration()
            await self._audit_redis_configuration()
            await self._audit_docker_services()
            await self._audit_file_system_structure()
            await self._audit_security_configuration()
            
            # Determine overall status
            self._determine_overall_status()
            
            audit_logger.info(
                "Configuration audit completed",
                overall_status=self.audit_results['overall_status'],
                total_checks=len(self.audit_results['checks']),
                errors=len(self.audit_results['errors']),
                warnings=len(self.audit_results['warnings'])
            )
            
            return self.audit_results
    
    async def _audit_environment_variables(self) -> None:
        """Audit environment variables and configuration settings."""
        logger = EnhancedLogger.get_logger("config_audit_env")
        logger.info("Auditing environment variables")
        
        check_name = "environment_variables"
        try:
            # Required environment variables
            required_vars = [
                'DATABASE_URL',
                'REDIS_URL', 
                'SECRET_KEY',
                'ALLOWED_HOSTS',
                'CORS_ORIGINS'
            ]
            
            missing_vars = []
            present_vars = {}
            
            for var in required_vars:
                value = getattr(settings, var.lower(), None) or os.getenv(var)
                if not value:
                    missing_vars.append(var)
                else:
                    # Don't log sensitive values
                    if 'SECRET' in var or 'PASSWORD' in var:
                        present_vars[var] = "***REDACTED***"
                    else:
                        present_vars[var] = value
            
            # Check optional but recommended variables
            optional_vars = ['JWT_SECRET', 'ANTHROPIC_API_KEY', 'GITHUB_TOKEN']
            optional_missing = []
            
            for var in optional_vars:
                value = getattr(settings, var.lower(), None) or os.getenv(var)
                if not value:
                    optional_missing.append(var)
                else:
                    present_vars[var] = "***REDACTED***"
            
            # Validate settings object
            config_issues = []
            if not settings.DEBUG and not settings.SECRET_KEY:
                config_issues.append("SECRET_KEY required in production mode")
            
            if settings.DATABASE_URL and 'sqlite' in settings.DATABASE_URL.lower():
                config_issues.append("SQLite not recommended for production")
            
            # Record results
            self.audit_results['checks'][check_name] = {
                'status': 'PASS' if not missing_vars and not config_issues else 'FAIL',
                'required_present': present_vars,
                'missing_required': missing_vars,
                'missing_optional': optional_missing,
                'config_issues': config_issues,
                'settings_debug': settings.DEBUG,
                'settings_environment': getattr(settings, 'ENVIRONMENT', 'development')
            }
            
            if missing_vars:
                self.audit_results['errors'].append(
                    f"Missing required environment variables: {', '.join(missing_vars)}"
                )
            
            if optional_missing:
                self.audit_results['warnings'].append(
                    f"Missing optional environment variables: {', '.join(optional_missing)}"
                )
            
            if config_issues:
                self.audit_results['errors'].extend(config_issues)
            
            logger.info(
                "Environment variables audit completed",
                status=self.audit_results['checks'][check_name]['status'],
                missing_required=len(missing_vars),
                missing_optional=len(optional_missing)
            )
            
        except Exception as e:
            logger.error("Environment variables audit failed", error=str(e))
            self.audit_results['checks'][check_name] = {
                'status': 'ERROR',
                'error': str(e)
            }
            self.audit_results['errors'].append(f"Environment audit failed: {str(e)}")
    
    async def _audit_database_configuration(self) -> None:
        """Audit database configuration and schema."""
        logger = EnhancedLogger.get_logger("config_audit_db")
        logger.info("Auditing database configuration")
        
        check_name = "database_configuration"
        try:
            # Test database connection
            async with get_session() as session:
                # Basic connectivity test
                result = await session.execute(text("SELECT 1 as test"))
                assert result.scalar() == 1
                
                # Check database version
                db_version_result = await session.execute(text("SELECT version()"))
                db_version = db_version_result.scalar()
                
                # Check if pgvector extension is available (for semantic search)
                try:
                    await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    pgvector_available = True
                except Exception:
                    pgvector_available = False
                
                # Check alembic migration status
                try:
                    # Get current migration head
                    alembic_cfg = Config(project_root / "alembic.ini")
                    script_dir = ScriptDirectory.from_config(alembic_cfg)
                    heads = script_dir.get_heads()
                    
                    # Check current database revision
                    current_rev_result = await session.execute(
                        text("SELECT version_num FROM alembic_version ORDER BY version_num DESC LIMIT 1")
                    )
                    current_rev = current_rev_result.scalar()
                    
                    migrations_current = current_rev in heads if current_rev else False
                    
                except Exception as e:
                    migrations_current = None
                    migration_error = str(e)
                
                # Count tables to verify schema
                tables_result = await session.execute(text("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                """))
                table_count = tables_result.scalar()
                
                # Check for key tables
                key_tables = ['agents', 'tasks', 'sessions', 'workflows', 'contexts']
                existing_tables = []
                missing_tables = []
                
                for table in key_tables:
                    table_check = await session.execute(text(f"""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' AND table_name = '{table}'
                        )
                    """))
                    if table_check.scalar():
                        existing_tables.append(table)
                    else:
                        missing_tables.append(table)
                
                # Record results
                self.audit_results['checks'][check_name] = {
                    'status': 'PASS' if migrations_current and not missing_tables else 'WARN',
                    'connection': 'SUCCESS',
                    'database_version': db_version,
                    'pgvector_available': pgvector_available,
                    'table_count': table_count,
                    'existing_tables': existing_tables,
                    'missing_tables': missing_tables,
                    'migrations_current': migrations_current
                }
                
                if not pgvector_available:
                    self.audit_results['warnings'].append(
                        "pgvector extension not available - semantic search features may not work"
                    )
                
                if missing_tables:
                    self.audit_results['warnings'].append(
                        f"Missing database tables: {', '.join(missing_tables)} - run migrations"
                    )
                
                if not migrations_current:
                    self.audit_results['warnings'].append(
                        "Database migrations may not be current - run 'alembic upgrade head'"
                    )
                
                logger.info(
                    "Database audit completed",
                    status=self.audit_results['checks'][check_name]['status'],
                    table_count=table_count,
                    pgvector_available=pgvector_available
                )
                
        except Exception as e:
            logger.error("Database audit failed", error=str(e))
            self.audit_results['checks'][check_name] = {
                'status': 'ERROR',
                'error': str(e)
            }
            self.audit_results['errors'].append(f"Database audit failed: {str(e)}")
    
    async def _audit_redis_configuration(self) -> None:
        """Audit Redis configuration and connectivity."""
        logger = EnhancedLogger.get_logger("config_audit_redis")
        logger.info("Auditing Redis configuration")
        
        check_name = "redis_configuration"
        try:
            # Test Redis connection
            redis_client = get_redis()
            
            # Basic connectivity and operations test
            test_key = f"audit_test_{self.correlation_id}"
            await redis_client.set(test_key, "test_value", ex=10)
            test_value = await redis_client.get(test_key)
            await redis_client.delete(test_key)
            
            assert test_value.decode() == "test_value"
            
            # Get Redis info
            redis_info = await redis_client.info()
            redis_version = redis_info.get('redis_version', 'unknown')
            memory_used = redis_info.get('used_memory_human', 'unknown')
            connected_clients = redis_info.get('connected_clients', 0)
            
            # Test Redis Streams (critical for multi-agent coordination)
            stream_name = f"audit_stream_{self.correlation_id}"
            try:
                # Test stream creation and reading
                await redis_client.xadd(stream_name, {'test': 'message'})
                messages = await redis_client.xread({stream_name: 0}, count=1)
                await redis_client.delete(stream_name)
                
                streams_functional = len(messages) > 0
            except Exception as e:
                streams_functional = False
                stream_error = str(e)
            
            # Check Redis configuration
            config_info = await redis_client.config_get('*')
            max_memory = config_info.get('maxmemory', '0')
            persistence = config_info.get('save', '')
            
            # Record results
            self.audit_results['checks'][check_name] = {
                'status': 'PASS' if streams_functional else 'WARN',
                'connection': 'SUCCESS',
                'redis_version': redis_version,
                'memory_used': memory_used,
                'connected_clients': connected_clients,
                'streams_functional': streams_functional,
                'max_memory': max_memory,
                'persistence_config': persistence
            }
            
            if not streams_functional:
                self.audit_results['warnings'].append(
                    "Redis Streams not functional - multi-agent coordination may fail"
                )
            
            if max_memory == '0':
                self.audit_results['warnings'].append(
                    "Redis max memory not set - may cause memory issues under load"
                )
            
            logger.info(
                "Redis audit completed",
                status=self.audit_results['checks'][check_name]['status'],
                version=redis_version,
                streams_functional=streams_functional
            )
            
        except Exception as e:
            logger.error("Redis audit failed", error=str(e))
            self.audit_results['checks'][check_name] = {
                'status': 'ERROR',
                'error': str(e)
            }
            self.audit_results['errors'].append(f"Redis audit failed: {str(e)}")
    
    async def _audit_docker_services(self) -> None:
        """Audit Docker services and container health."""
        logger = EnhancedLogger.get_logger("config_audit_docker")
        logger.info("Auditing Docker services")
        
        check_name = "docker_services"
        try:
            client = docker.from_env()
            
            # Check if Docker is running
            client.ping()
            
            # Get running containers
            containers = client.containers.list()
            container_info = []
            
            for container in containers:
                container_info.append({
                    'name': container.name,
                    'image': container.image.tags[0] if container.image.tags else 'unknown',
                    'status': container.status,
                    'health': container.attrs.get('State', {}).get('Health', {}).get('Status', 'unknown')
                })
            
            # Check for expected services
            expected_services = ['postgres', 'redis']
            found_services = []
            missing_services = []
            
            for service in expected_services:
                found = any(service in container['name'].lower() for container in container_info)
                if found:
                    found_services.append(service)
                else:
                    missing_services.append(service)
            
            # Check Docker Compose file
            compose_file = project_root / "docker-compose.yml"
            compose_exists = compose_file.exists()
            
            # Record results
            self.audit_results['checks'][check_name] = {
                'status': 'PASS' if not missing_services else 'WARN',
                'docker_running': True,
                'containers': container_info,
                'found_services': found_services,
                'missing_services': missing_services,
                'compose_file_exists': compose_exists
            }
            
            if missing_services:
                self.audit_results['warnings'].append(
                    f"Expected Docker services not running: {', '.join(missing_services)}"
                )
            
            logger.info(
                "Docker audit completed",
                status=self.audit_results['checks'][check_name]['status'],
                running_containers=len(containers),
                found_services=len(found_services)
            )
            
        except Exception as e:
            logger.error("Docker audit failed", error=str(e))
            self.audit_results['checks'][check_name] = {
                'status': 'ERROR',
                'error': str(e)
            }
            self.audit_results['errors'].append(f"Docker audit failed: {str(e)}")
    
    async def _audit_file_system_structure(self) -> None:
        """Audit file system structure and permissions."""
        logger = EnhancedLogger.get_logger("config_audit_fs")
        logger.info("Auditing file system structure")
        
        check_name = "file_system_structure"
        try:
            # Check critical directories
            critical_dirs = [
                'app',
                'app/core',
                'app/api',
                'app/models',
                'migrations',
                'tests',
                'docs'
            ]
            
            existing_dirs = []
            missing_dirs = []
            
            for dir_path in critical_dirs:
                full_path = project_root / dir_path
                if full_path.exists() and full_path.is_dir():
                    existing_dirs.append(dir_path)
                else:
                    missing_dirs.append(dir_path)
            
            # Check critical files
            critical_files = [
                'app/main.py',
                'app/core/config.py',
                'app/core/database.py',
                'app/core/redis.py',
                'docker-compose.yml',
                'pyproject.toml',
                'alembic.ini'
            ]
            
            existing_files = []
            missing_files = []
            
            for file_path in critical_files:
                full_path = project_root / file_path
                if full_path.exists() and full_path.is_file():
                    existing_files.append(file_path)
                else:
                    missing_files.append(file_path)
            
            # Check write permissions for logs and checkpoints
            writable_dirs = ['logs', 'checkpoints']
            permission_issues = []
            
            for dir_path in writable_dirs:
                full_path = project_root / dir_path
                try:
                    full_path.mkdir(exist_ok=True)
                    # Test write permission
                    test_file = full_path / f"test_{self.correlation_id}.tmp"
                    test_file.write_text("test")
                    test_file.unlink()
                except Exception as e:
                    permission_issues.append(f"{dir_path}: {str(e)}")
            
            # Record results
            self.audit_results['checks'][check_name] = {
                'status': 'PASS' if not missing_files and not permission_issues else 'WARN',
                'existing_dirs': existing_dirs,
                'missing_dirs': missing_dirs,
                'existing_files': existing_files,
                'missing_files': missing_files,
                'permission_issues': permission_issues
            }
            
            if missing_files:
                self.audit_results['warnings'].append(
                    f"Missing critical files: {', '.join(missing_files)}"
                )
            
            if permission_issues:
                self.audit_results['warnings'].append(
                    f"File permission issues: {', '.join(permission_issues)}"
                )
            
            logger.info(
                "File system audit completed",
                status=self.audit_results['checks'][check_name]['status'],
                existing_files=len(existing_files),
                missing_files=len(missing_files)
            )
            
        except Exception as e:
            logger.error("File system audit failed", error=str(e))
            self.audit_results['checks'][check_name] = {
                'status': 'ERROR',
                'error': str(e)
            }
            self.audit_results['errors'].append(f"File system audit failed: {str(e)}")
    
    async def _audit_security_configuration(self) -> None:
        """Audit security configuration and settings."""
        logger = EnhancedLogger.get_logger("config_audit_security")
        logger.info("Auditing security configuration")
        
        check_name = "security_configuration"
        try:
            security_issues = []
            security_warnings = []
            
            # Check secret key strength
            if hasattr(settings, 'SECRET_KEY') and settings.SECRET_KEY:
                if len(settings.SECRET_KEY) < 32:
                    security_issues.append("SECRET_KEY too short (minimum 32 characters)")
                if settings.SECRET_KEY in ['changeme', 'development', 'secret']:
                    security_issues.append("SECRET_KEY uses default/weak value")
            
            # Check debug mode in production
            if not settings.DEBUG and os.getenv('ENVIRONMENT') == 'production':
                if settings.DEBUG:
                    security_issues.append("DEBUG mode enabled in production")
            
            # Check CORS configuration
            cors_origins = getattr(settings, 'CORS_ORIGINS', [])
            if '*' in cors_origins and not settings.DEBUG:
                security_warnings.append("CORS allows all origins in production")
            
            # Check allowed hosts
            allowed_hosts = getattr(settings, 'ALLOWED_HOSTS', [])
            if '*' in allowed_hosts and not settings.DEBUG:
                security_warnings.append("All hosts allowed in production")
            
            # Check database URL for security
            if settings.DATABASE_URL:
                if 'password' in settings.DATABASE_URL.lower() and '@localhost' not in settings.DATABASE_URL:
                    # This is a remote database with password in URL - potential security issue
                    security_warnings.append("Database password in connection URL - consider using environment variables")
            
            # Record results
            self.audit_results['checks'][check_name] = {
                'status': 'PASS' if not security_issues else 'FAIL',
                'security_issues': security_issues,
                'security_warnings': security_warnings,
                'debug_mode': settings.DEBUG,
                'cors_origins_count': len(cors_origins) if cors_origins else 0,
                'allowed_hosts_count': len(allowed_hosts) if allowed_hosts else 0
            }
            
            if security_issues:
                self.audit_results['errors'].extend(security_issues)
            
            if security_warnings:
                self.audit_results['warnings'].extend(security_warnings)
            
            logger.info(
                "Security audit completed",
                status=self.audit_results['checks'][check_name]['status'],
                security_issues=len(security_issues),
                security_warnings=len(security_warnings)
            )
            
        except Exception as e:
            logger.error("Security audit failed", error=str(e))
            self.audit_results['checks'][check_name] = {
                'status': 'ERROR',
                'error': str(e)
            }
            self.audit_results['errors'].append(f"Security audit failed: {str(e)}")
    
    def _determine_overall_status(self) -> None:
        """Determine overall audit status based on individual checks."""
        checks = self.audit_results['checks']
        
        # Count status types
        error_count = sum(1 for check in checks.values() if check.get('status') == 'ERROR')
        fail_count = sum(1 for check in checks.values() if check.get('status') == 'FAIL')
        warn_count = sum(1 for check in checks.values() if check.get('status') == 'WARN')
        pass_count = sum(1 for check in checks.values() if check.get('status') == 'PASS')
        
        # Determine overall status
        if error_count > 0:
            self.audit_results['overall_status'] = 'ERROR'
            self.audit_results['recommendations'].append(
                "CRITICAL: Fix error conditions before proceeding with integration"
            )
        elif fail_count > 0:
            self.audit_results['overall_status'] = 'FAIL'
            self.audit_results['recommendations'].append(
                "IMPORTANT: Fix failing checks before proceeding with integration"
            )
        elif warn_count > 0:
            self.audit_results['overall_status'] = 'WARN'
            self.audit_results['recommendations'].append(
                "RECOMMENDED: Address warnings for optimal system performance"
            )
        else:
            self.audit_results['overall_status'] = 'PASS'
            self.audit_results['recommendations'].append(
                "System configuration is ready for integration resolution"
            )
        
        # Add specific recommendations
        if any('database' in str(check) for check in self.audit_results['errors']):
            self.audit_results['recommendations'].append(
                "Run: alembic upgrade head"
            )
        
        if any('redis' in str(check) for check in self.audit_results['errors']):
            self.audit_results['recommendations'].append(
                "Start Redis service: docker-compose up -d redis"
            )
        
        if any('docker' in str(check) for check in self.audit_results['errors']):
            self.audit_results['recommendations'].append(
                "Start required services: docker-compose up -d"
            )
        
        # Summary statistics
        self.audit_results['summary'] = {
            'total_checks': len(checks),
            'passed': pass_count,
            'warnings': warn_count,
            'failures': fail_count,
            'errors': error_count
        }
    
    def save_audit_report(self, filename: Optional[str] = None) -> str:
        """
        Save audit results to file.
        
        Args:
            filename: Optional filename (default: audit_report_timestamp.json)
            
        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"audit_report_{timestamp}.json"
        
        filepath = project_root / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.audit_results, f, indent=2)
        
        return str(filepath)
    
    def print_audit_summary(self) -> None:
        """Print a human-readable audit summary."""
        print("\n" + "="*80)
        print("LEANVIBE AGENT HIVE 2.0 - CONFIGURATION AUDIT REPORT")
        print("="*80)
        
        print(f"\nOverall Status: {self.audit_results['overall_status']}")
        print(f"Audit Time: {self.audit_results['timestamp']}")
        print(f"Correlation ID: {self.correlation_id}")
        
        # Summary statistics
        summary = self.audit_results['summary']
        print(f"\nSummary:")
        print(f"  Total Checks: {summary['total_checks']}")
        print(f"  Passed: {summary['passed']}")
        print(f"  Warnings: {summary['warnings']}")
        print(f"  Failures: {summary['failures']}")
        print(f"  Errors: {summary['errors']}")
        
        # Errors
        if self.audit_results['errors']:
            print(f"\n🚨 ERRORS ({len(self.audit_results['errors'])}):")
            for error in self.audit_results['errors']:
                print(f"  - {error}")
        
        # Warnings
        if self.audit_results['warnings']:
            print(f"\n⚠️  WARNINGS ({len(self.audit_results['warnings'])}):")
            for warning in self.audit_results['warnings']:
                print(f"  - {warning}")
        
        # Recommendations
        if self.audit_results['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in self.audit_results['recommendations']:
                print(f"  - {rec}")
        
        print("\n" + "="*80)


async def main():
    """Run configuration audit and display results."""
    auditor = ConfigurationAuditor()
    
    print("Starting LeanVibe Agent Hive 2.0 Configuration Audit...")
    
    audit_results = await auditor.run_comprehensive_audit()
    
    # Print summary
    auditor.print_audit_summary()
    
    # Save detailed report
    report_file = auditor.save_audit_report()
    print(f"\nDetailed audit report saved to: {report_file}")
    
    # Exit with appropriate code
    if audit_results['overall_status'] in ['ERROR', 'FAIL']:
        print("\n❌ Configuration audit failed. Please address issues before proceeding.")
        sys.exit(1)
    elif audit_results['overall_status'] == 'WARN':
        print("\n⚠️  Configuration audit completed with warnings. Review recommendations.")
        sys.exit(0)
    else:
        print("\n✅ Configuration audit passed. System ready for integration resolution.")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())