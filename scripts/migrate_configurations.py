import asyncio
#!/usr/bin/env python3
"""
Configuration Migration Tool for LeanVibe Agent Hive 2.0

This script migrates from scattered configuration files to the unified configuration system.
It provides:
- Automated migration of existing Settings to UnifiedSystemConfig
- Configuration validation and integrity checking
- Backup and rollback procedures
- Environment-specific migration support

USAGE:
    python scripts/migrate_configurations.py --environment development --backup
    python scripts/migrate_configurations.py --validate-only
    python scripts/migrate_configurations.py --rollback backup_20240101_120000.json
"""

import os
import json
import shutil
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.config.unified_config import (
    UnifiedSystemConfig, 
    Environment,
    ConfigurationManager,
    initialize_unified_config
)
from app.core.config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigurationMigrationTool:
    """
    Tool for migrating from legacy configuration to unified configuration system.
    """
    
    def __init__(self, backup_dir: str = "./config_backups"):
        """Initialize migration tool."""
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Legacy configuration files to migrate
        self.legacy_config_files = [
            "app/core/config.py",
            "app/config/semantic_memory_config.py", 
            "app/core/configuration_service.py",
            "app/core/error_handling_config.py",
            "app/observability/hooks/hooks_config.py",
            "bee-hive-config.json",
            "config/project-size-profiles.yml"
        ]
        
        self.migration_stats = {
            "files_migrated": 0,
            "settings_migrated": 0,
            "validation_errors": [],
            "migration_warnings": []
        }
    
    def create_backup(self, environment: Environment) -> str:
        """Create backup of current configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"config_backup_{environment.value}_{timestamp}.json"
        backup_path = self.backup_dir / backup_name
        
        backup_data = {
            "timestamp": timestamp,
            "environment": environment.value,
            "files": {},
            "environment_variables": self._get_env_vars(),
            "migration_metadata": {
                "tool_version": "2.0.0",
                "source_system": "scattered_configs",
                "target_system": "unified_config"
            }
        }
        
        # Backup existing configuration files
        for config_file in self.legacy_config_files:
            file_path = Path(config_file)
            if file_path.exists():
                try:
                    if file_path.suffix == '.py':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            backup_data["files"][config_file] = f.read()
                    elif file_path.suffix in ['.json', '.yml', '.yaml']:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            backup_data["files"][config_file] = f.read()
                    
                    logger.info(f"📦 Backed up {config_file}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to backup {config_file}: {e}")
        
        # Save backup
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)
        
        logger.info(f"✅ Configuration backup created: {backup_path}")
        return str(backup_path)
    
    def _get_env_vars(self) -> Dict[str, str]:
        """Get relevant environment variables."""
        env_prefixes = [
            "DATABASE_", "REDIS_", "JWT_", "ANTHROPIC_", "GITHUB_", "OPENAI_",
            "ENVIRONMENT", "DEBUG", "LOG_LEVEL", "SECRET_KEY", "CORS_",
            "SECURITY_", "COMPLIANCE_", "HOOK_", "TMUX_", "PROMETHEUS_"
        ]
        
        env_vars = {}
        for key, value in os.environ.items():
            for prefix in env_prefixes:
                if key.startswith(prefix):
                    env_vars[key] = value
                    break
        
        return env_vars
    
    def migrate_legacy_settings(self, target_environment: Environment) -> UnifiedSystemConfig:
        """Migrate legacy Settings to UnifiedSystemConfig."""
        logger.info("🔄 Starting migration from legacy Settings...")
        
        try:
            # Load legacy settings
            legacy_settings = Settings()
            logger.info("✅ Legacy settings loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load legacy settings: {e}")
            # Create default unified config if legacy fails
            return UnifiedSystemConfig(environment=target_environment)
        
        # Create new unified config with migrated values
        unified_config = self._create_unified_from_legacy(legacy_settings, target_environment)
        
        # Validate migrated configuration
        validation_result = self.validate_configuration(unified_config)
        if not validation_result.is_valid:
            logger.error(f"❌ Migration validation failed: {validation_result.errors}")
            raise ValueError(f"Configuration migration validation failed: {validation_result.errors}")
        
        logger.info("✅ Legacy settings migration completed successfully")
        self.migration_stats["settings_migrated"] = len(unified_config.__dict__)
        
        return unified_config
    
    def _create_unified_from_legacy(self, legacy: Settings, environment: Environment) -> UnifiedSystemConfig:
        """Create UnifiedSystemConfig from legacy Settings."""
        
        # Create unified config with environment
        unified = UnifiedSystemConfig(environment=environment)
        
        # Migrate core application settings
        unified.app_name = legacy.APP_NAME
        unified.debug = legacy.DEBUG
        unified.log_level = legacy.LOG_LEVEL
        
        # Migrate orchestrator settings
        unified.orchestrator.max_agents = legacy.MAX_CONCURRENT_AGENTS
        unified.orchestrator.health_check_interval = legacy.AGENT_HEARTBEAT_INTERVAL
        
        # Migrate context manager settings
        unified.managers.context_manager.max_context_size = legacy.CONTEXT_MAX_TOKENS
        unified.managers.context_manager.compression_threshold = legacy.CONTEXT_COMPRESSION_THRESHOLD
        unified.managers.context_manager.embedding_model = legacy.CONTEXT_EMBEDDING_MODEL
        
        # Migrate security manager settings
        unified.managers.security_manager.jwt_secret_key = legacy.JWT_SECRET_KEY
        unified.managers.security_manager.jwt_algorithm = legacy.JWT_ALGORITHM
        unified.managers.security_manager.jwt_expiration_hours = legacy.JWT_ACCESS_TOKEN_EXPIRE_MINUTES // 60
        unified.managers.security_manager.gdpr_compliance = legacy.GDPR_COMPLIANCE
        unified.managers.security_manager.soc2_compliance = legacy.SOC2_COMPLIANCE
        unified.managers.security_manager.audit_log_retention_days = legacy.AUDIT_LOG_RETENTION_DAYS
        
        # Migrate task manager settings  
        unified.managers.task_manager.max_concurrent_tasks = legacy.MAX_CONCURRENT_AGENTS * 2  # Reasonable default
        unified.managers.task_manager.default_task_timeout = legacy.AGENT_TIMEOUT
        
        # Migrate communication manager settings
        unified.managers.communication_manager.max_message_size = legacy.MAX_MESSAGE_SIZE_BYTES
        unified.managers.communication_manager.message_compression_enabled = legacy.COMPRESSION_ENABLED
        unified.managers.communication_manager.batch_size = legacy.MESSAGE_BATCH_SIZE
        unified.managers.communication_manager.dlq_enabled = True
        unified.managers.communication_manager.dlq_max_retries = legacy.DLQ_MAX_RETRIES
        
        # Migrate communication hub settings
        unified.communication_hub.redis_host = self._extract_redis_host(legacy.REDIS_URL)
        unified.communication_hub.redis_port = self._extract_redis_port(legacy.REDIS_URL)
        unified.communication_hub.message_compression_enabled = legacy.COMPRESSION_ENABLED
        
        # Migrate database settings
        unified.database.url = legacy.DATABASE_URL
        unified.database.pool_size = legacy.DATABASE_POOL_SIZE
        unified.database.max_overflow = legacy.DATABASE_MAX_OVERFLOW
        
        # Migrate Redis settings
        unified.redis.url = legacy.REDIS_URL
        unified.redis.connection_pool_size = legacy.REDIS_CONNECTION_POOL_SIZE
        unified.redis.max_connections = legacy.REDIS_MAX_CONNECTIONS
        unified.redis.connection_timeout = legacy.REDIS_CONNECTION_TIMEOUT
        unified.redis.compression_enabled = legacy.COMPRESSION_ENABLED
        unified.redis.compression_algorithm = legacy.COMPRESSION_ALGORITHM
        unified.redis.compression_level = legacy.COMPRESSION_LEVEL
        
        # Migrate monitoring settings
        unified.monitoring.metrics_enabled = legacy.METRICS_ENABLED
        unified.monitoring.metrics_port = legacy.PROMETHEUS_PORT
        unified.monitoring.log_level = legacy.LOG_LEVEL
        unified.monitoring.performance_monitoring = True
        unified.monitoring.slow_query_threshold = legacy.TARGET_P95_LATENCY_MS
        
        # Migrate security settings
        unified.security.jwt_secret_key = legacy.JWT_SECRET_KEY
        unified.security.jwt_algorithm = legacy.JWT_ALGORITHM
        unified.security.jwt_expiration_minutes = legacy.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        unified.security.api_key_required = legacy.SECURITY_ENABLED
        unified.security.rate_limiting_enabled = legacy.API_RATE_LIMIT_ENABLED
        unified.security.cors_origins = legacy.CORS_ORIGINS
        unified.security.encryption_at_rest = legacy.SECRETS_ENCRYPTION_ENABLED
        unified.security.audit_logging_enabled = legacy.SECURITY_AUDIT_ENABLED
        unified.security.vulnerability_scanning = legacy.THREAT_DETECTION_ENABLED
        
        # Migrate performance settings
        unified.performance.max_concurrent_agents = legacy.MAX_CONCURRENT_AGENTS
        unified.performance.target_response_time_ms = legacy.TARGET_P95_LATENCY_MS
        unified.performance.target_throughput_rps = legacy.TARGET_THROUGHPUT_MSG_PER_SEC
        unified.performance.target_availability = legacy.TARGET_SUCCESS_RATE
        
        # Migrate engine settings
        unified.engines.communication_engine.max_connections = legacy.REDIS_MAX_CONNECTIONS
        unified.engines.data_processing_engine.max_batch_size = legacy.MESSAGE_BATCH_SIZE
        unified.engines.monitoring_engine.metrics_collection_interval = legacy.STREAM_MONITORING_INTERVAL
        unified.engines.monitoring_engine.prometheus_enabled = legacy.PROMETHEUS_METRICS_ENABLED
        unified.engines.security_engine.vulnerability_scanning_enabled = legacy.THREAT_DETECTION_ENABLED
        unified.engines.security_engine.security_monitoring_enabled = legacy.SECURITY_AUDIT_ENABLED
        
        return unified
    
    def _extract_redis_host(self, redis_url: str) -> str:
        """Extract Redis host from URL."""
        try:
            # Parse redis://host:port/db format
            if redis_url.startswith('redis://'):
                url_part = redis_url[8:]  # Remove redis://
                if '@' in url_part:
                    url_part = url_part.split('@')[1]  # Remove credentials
                host_port = url_part.split('/')[0]  # Remove database number
                return host_port.split(':')[0]
            return "localhost"
        except:
            return "localhost"
    
    def _extract_redis_port(self, redis_url: str) -> int:
        """Extract Redis port from URL."""
        try:
            # Parse redis://host:port/db format
            if redis_url.startswith('redis://'):
                url_part = redis_url[8:]  # Remove redis://
                if '@' in url_part:
                    url_part = url_part.split('@')[1]  # Remove credentials
                host_port = url_part.split('/')[0]  # Remove database number
                if ':' in host_port:
                    return int(host_port.split(':')[1])
            return 6379  # Default Redis port
        except:
            return 6379
    
    def validate_configuration(self, config: UnifiedSystemConfig) -> 'ValidationResult':
        """Validate migrated configuration."""
        logger.info("🔍 Validating migrated configuration...")
        
        errors = []
        warnings = []
        
        # Validate required fields
        if not config.security.jwt_secret_key:
            errors.append("JWT secret key is missing")
        
        if not config.database.url:
            errors.append("Database URL is missing")
        
        if not config.redis.url:
            errors.append("Redis URL is missing")
        
        # Validate numeric ranges
        if config.orchestrator.max_agents < 1:
            errors.append("Max agents must be >= 1")
        
        if config.database.pool_size < 1:
            errors.append("Database pool size must be >= 1")
        
        if config.performance.max_concurrent_agents < 1:
            errors.append("Max concurrent agents must be >= 1")
        
        # Validate environment-specific settings
        if config.environment == Environment.PRODUCTION:
            if config.debug:
                warnings.append("Debug mode is enabled in production")
            
            if len(config.security.jwt_secret_key) < 32:
                errors.append("JWT secret key too short for production (< 32 chars)")
            
            if not config.security.api_key_required:
                warnings.append("API key requirement disabled in production")
            
            if not config.security.rate_limiting_enabled:
                warnings.append("Rate limiting disabled in production")
        
        # Validate performance settings
        if config.performance.target_response_time_ms > 1000:
            warnings.append("Target response time > 1000ms may impact user experience")
        
        if config.managers.context_manager.compression_threshold > 0.95:
            warnings.append("Context compression threshold very high (> 95%)")
        
        # Log validation results
        if errors:
            logger.error(f"❌ Validation errors: {errors}")
            self.migration_stats["validation_errors"].extend(errors)
        
        if warnings:
            logger.warning(f"⚠️ Validation warnings: {warnings}")
            self.migration_stats["migration_warnings"].extend(warnings)
        
        if not errors and not warnings:
            logger.info("✅ Configuration validation passed")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def perform_migration(
        self, 
        target_environment: Environment,
        create_backup: bool = True,
        validate_only: bool = False
    ) -> Tuple[bool, str]:
        """Perform complete configuration migration."""
        logger.info(f"🚀 Starting configuration migration to {target_environment.value}")
        
        try:
            # Create backup if requested
            backup_path = None
            if create_backup:
                backup_path = self.create_backup(target_environment)
            
            # Migrate configuration
            unified_config = self.migrate_legacy_settings(target_environment)
            
            # If validate-only mode, stop here
            if validate_only:
                logger.info("✅ Validation completed successfully")
                return True, f"Validation passed for {target_environment.value}"
            
            # Initialize unified config system
            config_manager = initialize_unified_config(
                environment=target_environment,
                enable_hot_reload=target_environment == Environment.DEVELOPMENT
            )
            
            # Update with migrated configuration
            config_manager.update_config(unified_config)
            
            # Save migrated configuration to file
            config_file = f"config/unified_config_{target_environment.value}.json"
            Path("config").mkdir(exist_ok=True)
            unified_config.save_to_file(config_file, include_sensitive=False)
            
            logger.info("✅ Configuration migration completed successfully")
            
            # Print migration summary
            self._print_migration_summary(backup_path, config_file)
            
            return True, f"Migration completed successfully. Config saved to {config_file}"
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False, f"Migration failed: {str(e)}"
    
    def rollback_migration(self, backup_file: str) -> bool:
        """Rollback to previous configuration from backup."""
        logger.info(f"🔄 Rolling back configuration from {backup_file}")
        
        try:
            backup_path = Path(backup_file)
            if not backup_path.exists():
                backup_path = self.backup_dir / backup_file
                if not backup_path.exists():
                    raise FileNotFoundError(f"Backup file not found: {backup_file}")
            
            # Load backup data
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            # Restore configuration files
            for config_file, content in backup_data.get("files", {}).items():
                file_path = Path(config_file)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, 'w') as f:
                    f.write(content)
                
                logger.info(f"📄 Restored {config_file}")
            
            # Restore environment variables (optional - requires restart)
            env_vars = backup_data.get("environment_variables", {})
            if env_vars:
                logger.info("⚠️ Environment variables found in backup. Manual restoration required:")
                for key, value in env_vars.items():
                    print(f"export {key}='{value}'")
            
            logger.info("✅ Configuration rollback completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False
    
    def _print_migration_summary(self, backup_path: Optional[str], config_file: str):
        """Print migration summary."""
        print("\n" + "="*60)
        print("🎉 CONFIGURATION MIGRATION SUMMARY")
        print("="*60)
        print(f"Files migrated: {self.migration_stats['files_migrated']}")
        print(f"Settings migrated: {self.migration_stats['settings_migrated']}")
        print(f"Validation errors: {len(self.migration_stats['validation_errors'])}")
        print(f"Warnings: {len(self.migration_stats['migration_warnings'])}")
        
        if backup_path:
            print(f"Backup created: {backup_path}")
        
        print(f"New config file: {config_file}")
        
        if self.migration_stats['validation_errors']:
            print("\n❌ VALIDATION ERRORS:")
            for error in self.migration_stats['validation_errors']:
                print(f"  - {error}")
        
        if self.migration_stats['migration_warnings']:
            print("\n⚠️ WARNINGS:")
            for warning in self.migration_stats['migration_warnings']:
                print(f"  - {warning}")
        
        print("\n✅ Migration completed successfully!")
        print("="*60)


class ValidationResult:
    """Configuration validation result."""
    
    def __init__(self, is_valid: bool, errors: List[str], warnings: List[str]):
        self.is_valid = is_valid
        self.errors = errors
        self.warnings = warnings


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate LeanVibe Agent Hive configurations to unified system"
    )
    
    parser.add_argument(
        "--environment", 
        choices=["development", "staging", "production", "testing"],
        default="development",
        help="Target environment for migration"
    )
    
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup before migration"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true", 
        help="Only validate configuration, don't perform migration"
    )
    
    parser.add_argument(
        "--rollback",
        type=str,
        help="Rollback to previous configuration from backup file"
    )
    
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="./config_backups",
        help="Directory for configuration backups"
    )
    
    args = parser.parse_args()
    
    # Initialize migration tool
    migration_tool = ConfigurationMigrationTool(backup_dir=args.backup_dir)
    
    try:
        if args.rollback:
            # Perform rollback
            success = migration_tool.rollback_migration(args.rollback)
            exit(0 if success else 1)
            
        else:
            # Perform migration
            target_env = Environment(args.environment)
            success, message = migration_tool.perform_migration(
                target_environment=target_env,
                create_backup=args.backup,
                validate_only=args.validate_only
            )
            
            print(f"\n{message}")
            exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Migration cancelled by user")
        exit(1)
    except Exception as e:
        logger.error(f"❌ Migration tool error: {e}")
        exit(1)


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class MigrateConfigurationsScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            main()
            
            return {"status": "completed"}
    
    script_main(MigrateConfigurationsScript)