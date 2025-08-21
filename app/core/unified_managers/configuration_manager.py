#!/usr/bin/env python3
"""
ConfigurationManager - Settings, Secrets, and Configuration Consolidation
Phase 2.1 Implementation of Technical Debt Remediation Plan

This manager consolidates all configuration management, settings handling, secrets
management, feature flags, and environment configuration into a unified,
high-performance system built on the BaseManager framework.

TARGET CONSOLIDATION: 12+ configuration-related manager classes → 1 unified ConfigurationManager
- Application settings and configuration
- Environment variable management
- Secrets and credentials management
- Feature flags and toggles
- Configuration validation and schema
- Dynamic configuration updates
- Configuration encryption and security
- Multi-environment configuration
- Configuration versioning and rollback
- Configuration monitoring and auditing
"""

import asyncio
import json
import os
import yaml
from pathlib import Path
import base64
import hashlib
import time
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
import threading
from contextlib import asynccontextmanager
import re

import structlog
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field, ValidationError

# Import BaseManager framework
from .base_manager import (
    BaseManager, ManagerConfig, ManagerDomain, ManagerStatus, ManagerMetrics,
    PluginInterface, PluginType
)

# Import shared patterns from Phase 1
from ...common.utilities.shared_patterns import (
    standard_logging_setup, standard_error_handling
)

logger = structlog.get_logger(__name__)


class ConfigurationSource(str, Enum):
    """Sources of configuration data."""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    REMOTE = "remote"
    MEMORY = "memory"
    VAULT = "vault"


class ConfigurationType(str, Enum):
    """Types of configuration values."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    SECRET = "secret"
    JSON = "json"
    YAML = "yaml"


class SecurityLevel(str, Enum):
    """Security levels for configuration values."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


class ChangeType(str, Enum):
    """Types of configuration changes."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    ENCRYPTED = "encrypted"
    DECRYPTED = "decrypted"


@dataclass
class ConfigurationValue:
    """Represents a configuration value with metadata."""
    key: str
    value: Any
    type: ConfigurationType
    source: ConfigurationSource
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    description: str = ""
    default_value: Optional[Any] = None
    required: bool = False
    encrypted: bool = False
    validation_pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    tags: Dict[str, str] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate the configuration value."""
        try:
            # Type validation
            if self.type == ConfigurationType.STRING and not isinstance(self.value, str):
                return False
            elif self.type == ConfigurationType.INTEGER and not isinstance(self.value, int):
                return False
            elif self.type == ConfigurationType.FLOAT and not isinstance(self.value, (int, float)):
                return False
            elif self.type == ConfigurationType.BOOLEAN and not isinstance(self.value, bool):
                return False
            elif self.type == ConfigurationType.LIST and not isinstance(self.value, list):
                return False
            elif self.type == ConfigurationType.DICT and not isinstance(self.value, dict):
                return False
            
            # Pattern validation
            if self.validation_pattern and isinstance(self.value, str):
                if not re.match(self.validation_pattern, self.value):
                    return False
            
            # Allowed values validation
            if self.allowed_values and self.value not in self.allowed_values:
                return False
            
            return True
            
        except Exception:
            return False
    
    def is_secret(self) -> bool:
        """Check if this configuration value is a secret."""
        return (self.type == ConfigurationType.SECRET or 
                self.security_level == SecurityLevel.SECRET or 
                self.encrypted)


@dataclass
class FeatureFlag:
    """Represents a feature flag configuration."""
    name: str
    enabled: bool = False
    rollout_percentage: float = 0.0
    target_groups: Set[str] = field(default_factory=set)
    conditions: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    def is_enabled_for(self, user_id: str = None, group: str = None, context: Dict[str, Any] = None) -> bool:
        """Check if feature is enabled for specific user/group/context."""
        if not self.enabled:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        # Check group targeting
        if group and self.target_groups and group not in self.target_groups:
            return False
        
        # Check rollout percentage
        if user_id and self.rollout_percentage < 100.0:
            # Deterministic rollout based on user ID hash
            user_hash = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
            user_percentage = (user_hash % 10000) / 100.0
            if user_percentage >= self.rollout_percentage:
                return False
        
        # Check custom conditions
        if self.conditions and context:
            for condition_key, condition_value in self.conditions.items():
                if context.get(condition_key) != condition_value:
                    return False
        
        return True


@dataclass
class ConfigurationChange:
    """Represents a configuration change event."""
    id: str = field(default_factory=lambda: str(int(time.time() * 1000000)))
    key: str = ""
    old_value: Any = None
    new_value: Any = None
    change_type: ChangeType = ChangeType.UPDATED
    source: ConfigurationSource = ConfigurationSource.MEMORY
    user_id: Optional[str] = None
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    rollback_data: Optional[Dict[str, Any]] = None


@dataclass
class ConfigurationSchema:
    """Schema definition for configuration validation."""
    required_keys: Set[str] = field(default_factory=set)
    optional_keys: Set[str] = field(default_factory=set)
    key_types: Dict[str, ConfigurationType] = field(default_factory=dict)
    key_patterns: Dict[str, str] = field(default_factory=dict)
    key_allowed_values: Dict[str, List[Any]] = field(default_factory=dict)


@dataclass
class ConfigurationManagerMetrics:
    """ConfigurationManager-specific metrics."""
    total_configurations: int = 0
    configurations_by_source: Dict[ConfigurationSource, int] = field(default_factory=dict)
    configurations_by_type: Dict[ConfigurationType, int] = field(default_factory=dict)
    encrypted_configurations: int = 0
    secret_configurations: int = 0
    feature_flags_count: int = 0
    active_feature_flags: int = 0
    configuration_changes: int = 0
    validation_errors: int = 0
    encryption_operations: int = 0
    decryption_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_lookup_time_ms: float = 0.0


class ConfigurationPlugin(PluginInterface):
    """Base class for configuration plugins."""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.CONFIGURATION
    
    async def pre_get_hook(self, key: str) -> Dict[str, Any]:
        """Hook called before getting a configuration value."""
        return {}
    
    async def post_get_hook(self, key: str, value: Any, found: bool) -> None:
        """Hook called after getting a configuration value."""
        pass
    
    async def pre_set_hook(self, key: str, value: Any, config_type: ConfigurationType) -> Dict[str, Any]:
        """Hook called before setting a configuration value."""
        return {}
    
    async def post_set_hook(self, key: str, old_value: Any, new_value: Any) -> None:
        """Hook called after setting a configuration value."""
        pass


class ConfigurationManager(BaseManager):
    """
    Unified manager for all configuration and settings operations.
    
    CONSOLIDATION TARGET: Replaces 12+ specialized configuration managers:
    - SettingsManager
    - EnvironmentManager
    - SecretsManager
    - FeatureFlagManager
    - ConfigValidationManager
    - DynamicConfigManager
    - EncryptionManager
    - MultiEnvConfigManager
    - ConfigVersionManager
    - ConfigAuditManager
    - ConfigCacheManager
    - ConfigReloadManager
    
    Built on BaseManager framework with Phase 2 enhancements.
    """
    
    def __init__(self, config: Optional[ManagerConfig] = None):
        # Create default config if none provided
        if config is None:
            config = ManagerConfig(
                name="ConfigurationManager",
                domain=ManagerDomain.CONFIGURATION,
                max_concurrent_operations=200,
                health_check_interval=30,
                circuit_breaker_enabled=True,
                circuit_breaker_failure_threshold=5
            )
        
        super().__init__(config)
        
        # Configuration-specific state
        self.configurations: Dict[str, ConfigurationValue] = {}
        self.feature_flags: Dict[str, FeatureFlag] = {}
        self.configuration_schemas: Dict[str, ConfigurationSchema] = {}
        self.change_history: List[ConfigurationChange] = []
        self.configuration_metrics = ConfigurationManagerMetrics()
        
        # Encryption
        self._encryption_key = self._generate_encryption_key()
        self._cipher_suite = Fernet(self._encryption_key)
        
        # Caching
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl = 300  # 5 minutes default
        
        # File watching
        self._watched_files: Dict[str, datetime] = {}
        self._file_watchers: List[asyncio.Task] = []
        
        # Background tasks
        self._reload_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._audit_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._config_lock = threading.RLock()
        self._flags_lock = threading.RLock()
        self._cache_lock = threading.RLock()
        
        # Default configuration paths
        self._config_paths = [
            Path("config/app.yaml"),
            Path("config/secrets.yaml"),
            Path(".env"),
            Path("config/feature_flags.yaml")
        ]
        
        self.logger = standard_logging_setup(
            name="ConfigurationManager",
            level="INFO"
        )
    
    # BaseManager Implementation
    
    async def _setup(self) -> None:
        """Initialize configuration management systems."""
        self.logger.info("Setting up ConfigurationManager")
        
        # Load configurations from various sources
        await self._load_from_environment()
        await self._load_from_files()
        await self._load_feature_flags()
        
        # Start background tasks
        self._reload_task = asyncio.create_task(self._reload_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._audit_task = asyncio.create_task(self._audit_loop())
        
        # Start file watchers
        for config_path in self._config_paths:
            if config_path.exists():
                watcher_task = asyncio.create_task(self._watch_file(config_path))
                self._file_watchers.append(watcher_task)
        
        self.logger.info("ConfigurationManager setup completed")
    
    async def _cleanup(self) -> None:
        """Clean up configuration management systems."""
        self.logger.info("Cleaning up ConfigurationManager")
        
        # Cancel background tasks
        tasks = [self._reload_task, self._cleanup_task, self._audit_task] + self._file_watchers
        
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Clear sensitive data
        with self._config_lock:
            for config_value in self.configurations.values():
                if config_value.is_secret():
                    config_value.value = None
        
        self._cache.clear()
        
        self.logger.info("ConfigurationManager cleanup completed")
    
    async def _health_check_internal(self) -> Dict[str, Any]:
        """Configuration-specific health check."""
        with self._config_lock:
            total_configs = len(self.configurations)
            encrypted_configs = sum(1 for c in self.configurations.values() if c.encrypted)
            secret_configs = sum(1 for c in self.configurations.values() if c.is_secret())
        
        with self._flags_lock:
            total_flags = len(self.feature_flags)
            active_flags = sum(1 for f in self.feature_flags.values() if f.enabled)
        
        cache_size = len(self._cache)
        
        return {
            "total_configurations": total_configs,
            "encrypted_configurations": encrypted_configs,
            "secret_configurations": secret_configs,
            "total_feature_flags": total_flags,
            "active_feature_flags": active_flags,
            "cache_size": cache_size,
            "watched_files": len(self._watched_files),
            "configuration_metrics": {
                "total_configurations": self.configuration_metrics.total_configurations,
                "configuration_changes": self.configuration_metrics.configuration_changes,
                "validation_errors": self.configuration_metrics.validation_errors,
                "cache_hits": self.configuration_metrics.cache_hits,
                "cache_misses": self.configuration_metrics.cache_misses
            }
        }
    
    # Core Configuration Operations
    
    async def get(
        self,
        key: str,
        default: Any = None,
        decrypt: bool = True,
        use_cache: bool = True
    ) -> Any:
        """
        Get a configuration value.
        
        CONSOLIDATES: SettingsGetter, SecretRetriever, CachedConfigLookup patterns
        """
        async with self.execute_with_monitoring("get_configuration"):
            start_time = time.time()
            
            try:
                # Pre-get hooks
                for plugin in self.plugins.values():
                    if isinstance(plugin, ConfigurationPlugin):
                        await plugin.pre_get_hook(key)
                
                # Check cache first
                if use_cache:
                    cached_value = self._get_from_cache(key)
                    if cached_value is not None:
                        self.configuration_metrics.cache_hits += 1
                        lookup_time = (time.time() - start_time) * 1000
                        self._update_lookup_time_metrics(lookup_time)
                        
                        # Post-get hooks
                        for plugin in self.plugins.values():
                            if isinstance(plugin, ConfigurationPlugin):
                                await plugin.post_get_hook(key, cached_value, True)
                        
                        return cached_value
                    else:
                        self.configuration_metrics.cache_misses += 1
                
                # Get from configurations
                with self._config_lock:
                    config_value = self.configurations.get(key)
                
                if config_value is None:
                    # Post-get hooks
                    for plugin in self.plugins.values():
                        if isinstance(plugin, ConfigurationPlugin):
                            await plugin.post_get_hook(key, default, False)
                    
                    return default
                
                value = config_value.value
                
                # Decrypt if needed
                if decrypt and config_value.encrypted and value:
                    value = self._decrypt_value(value)
                    self.configuration_metrics.decryption_operations += 1
                
                # Cache the result
                if use_cache:
                    self._set_in_cache(key, value)
                
                # Post-get hooks
                for plugin in self.plugins.values():
                    if isinstance(plugin, ConfigurationPlugin):
                        await plugin.post_get_hook(key, value, True)
                
                # Update metrics
                lookup_time = (time.time() - start_time) * 1000
                self._update_lookup_time_metrics(lookup_time)
                
                return value
                
            except Exception as e:
                self.logger.error(f"Failed to get configuration: {e}", key=key)
                return default
    
    async def set(
        self,
        key: str,
        value: Any,
        config_type: ConfigurationType = ConfigurationType.STRING,
        source: ConfigurationSource = ConfigurationSource.MEMORY,
        security_level: SecurityLevel = SecurityLevel.INTERNAL,
        encrypt: bool = False,
        description: str = "",
        user_id: Optional[str] = None,
        reason: str = ""
    ) -> bool:
        """
        Set a configuration value.
        
        CONSOLIDATES: SettingsSetter, SecretStorer, ConfigValidator patterns
        """
        async with self.execute_with_monitoring("set_configuration"):
            try:
                # Pre-set hooks
                for plugin in self.plugins.values():
                    if isinstance(plugin, ConfigurationPlugin):
                        await plugin.pre_set_hook(key, value, config_type)
                
                old_value = None
                with self._config_lock:
                    old_config = self.configurations.get(key)
                    old_value = old_config.value if old_config else None
                
                # Encrypt if requested
                actual_value = value
                if encrypt or security_level == SecurityLevel.SECRET:
                    actual_value = self._encrypt_value(value)
                    encrypt = True
                    self.configuration_metrics.encryption_operations += 1
                
                # Create configuration value
                config_value = ConfigurationValue(
                    key=key,
                    value=actual_value,
                    type=config_type,
                    source=source,
                    security_level=security_level,
                    description=description,
                    encrypted=encrypt
                )
                
                # Validate configuration
                if not config_value.validate():
                    self.configuration_metrics.validation_errors += 1
                    raise ValueError(f"Configuration validation failed for key: {key}")
                
                # Store configuration
                with self._config_lock:
                    if key in self.configurations:
                        # Update existing
                        old_config = self.configurations[key]
                        config_value.version = old_config.version + 1
                        config_value.created_at = old_config.created_at
                        
                        change_type = ChangeType.UPDATED
                    else:
                        # New configuration
                        change_type = ChangeType.CREATED
                        self.configuration_metrics.total_configurations += 1
                    
                    self.configurations[key] = config_value
                
                # Update metrics
                self._update_configuration_metrics(config_value)
                self.configuration_metrics.configuration_changes += 1
                
                # Record change
                change = ConfigurationChange(
                    key=key,
                    old_value=old_value,
                    new_value=value,  # Store the unencrypted value in change log
                    change_type=change_type,
                    source=source,
                    user_id=user_id,
                    reason=reason,
                    rollback_data={"old_config": old_config.__dict__ if old_config else None}
                )
                self.change_history.append(change)
                
                # Invalidate cache
                self._invalidate_cache(key)
                
                # Post-set hooks
                for plugin in self.plugins.values():
                    if isinstance(plugin, ConfigurationPlugin):
                        await plugin.post_set_hook(key, old_value, value)
                
                self.logger.info(
                    f"Configuration {'updated' if change_type == ChangeType.UPDATED else 'created'}",
                    key=key,
                    type=config_type.value,
                    encrypted=encrypt,
                    security_level=security_level.value
                )
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to set configuration: {e}", key=key)
                return False
    
    async def delete(self, key: str, user_id: Optional[str] = None, reason: str = "") -> bool:
        """Delete a configuration value."""
        async with self.execute_with_monitoring("delete_configuration"):
            try:
                with self._config_lock:
                    config_value = self.configurations.get(key)
                    if not config_value:
                        return False
                    
                    old_value = config_value.value
                    del self.configurations[key]
                
                # Record change
                change = ConfigurationChange(
                    key=key,
                    old_value=old_value,
                    new_value=None,
                    change_type=ChangeType.DELETED,
                    user_id=user_id,
                    reason=reason,
                    rollback_data={"deleted_config": config_value.__dict__}
                )
                self.change_history.append(change)
                
                # Update metrics
                self.configuration_metrics.total_configurations -= 1
                self.configuration_metrics.configuration_changes += 1
                
                # Invalidate cache
                self._invalidate_cache(key)
                
                self.logger.info(f"Configuration deleted", key=key)
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to delete configuration: {e}", key=key)
                return False
    
    # Feature Flag Management
    
    async def create_feature_flag(
        self,
        name: str,
        enabled: bool = False,
        rollout_percentage: float = 0.0,
        target_groups: Optional[Set[str]] = None,
        conditions: Optional[Dict[str, Any]] = None,
        description: str = "",
        expires_at: Optional[datetime] = None
    ) -> bool:
        """
        Create a feature flag.
        
        CONSOLIDATES: FeatureFlagManager, ToggleManager patterns
        """
        async with self.execute_with_monitoring("create_feature_flag"):
            try:
                feature_flag = FeatureFlag(
                    name=name,
                    enabled=enabled,
                    rollout_percentage=rollout_percentage,
                    target_groups=target_groups or set(),
                    conditions=conditions or {},
                    description=description,
                    expires_at=expires_at
                )
                
                with self._flags_lock:
                    self.feature_flags[name] = feature_flag
                
                self.configuration_metrics.feature_flags_count += 1
                if enabled:
                    self.configuration_metrics.active_feature_flags += 1
                
                self.logger.info(f"Feature flag created", name=name, enabled=enabled)
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to create feature flag: {e}", name=name)
                return False
    
    async def is_feature_enabled(
        self,
        name: str,
        user_id: Optional[str] = None,
        group: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if a feature flag is enabled for given context."""
        async with self.execute_with_monitoring("check_feature_flag"):
            try:
                with self._flags_lock:
                    feature_flag = self.feature_flags.get(name)
                    if not feature_flag:
                        return False
                    
                    return feature_flag.is_enabled_for(user_id, group, context)
                
            except Exception as e:
                self.logger.error(f"Failed to check feature flag: {e}", name=name)
                return False
    
    async def update_feature_flag(
        self,
        name: str,
        enabled: Optional[bool] = None,
        rollout_percentage: Optional[float] = None,
        target_groups: Optional[Set[str]] = None
    ) -> bool:
        """Update a feature flag."""
        async with self.execute_with_monitoring("update_feature_flag"):
            try:
                with self._flags_lock:
                    feature_flag = self.feature_flags.get(name)
                    if not feature_flag:
                        return False
                    
                    old_enabled = feature_flag.enabled
                    
                    if enabled is not None:
                        feature_flag.enabled = enabled
                    if rollout_percentage is not None:
                        feature_flag.rollout_percentage = rollout_percentage
                    if target_groups is not None:
                        feature_flag.target_groups = target_groups
                    
                    feature_flag.updated_at = datetime.utcnow()
                    
                    # Update active flags count
                    if old_enabled != feature_flag.enabled:
                        if feature_flag.enabled:
                            self.configuration_metrics.active_feature_flags += 1
                        else:
                            self.configuration_metrics.active_feature_flags -= 1
                
                self.logger.info(f"Feature flag updated", name=name)
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to update feature flag: {e}", name=name)
                return False
    
    # Configuration Loading
    
    async def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        env_prefix = "APP_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower().replace('_', '.')
                
                # Determine type
                config_type = ConfigurationType.STRING
                actual_value = value
                
                if value.lower() in ('true', 'false'):
                    config_type = ConfigurationType.BOOLEAN
                    actual_value = value.lower() == 'true'
                elif value.isdigit():
                    config_type = ConfigurationType.INTEGER
                    actual_value = int(value)
                elif self._is_float(value):
                    config_type = ConfigurationType.FLOAT
                    actual_value = float(value)
                
                # Check if it's a secret (contains sensitive keywords)
                security_level = SecurityLevel.INTERNAL
                if any(secret_word in key.lower() for secret_word in ['password', 'key', 'secret', 'token']):
                    security_level = SecurityLevel.SECRET
                
                config_value = ConfigurationValue(
                    key=config_key,
                    value=actual_value,
                    type=config_type,
                    source=ConfigurationSource.ENVIRONMENT,
                    security_level=security_level,
                    description=f"Environment variable {key}"
                )
                
                with self._config_lock:
                    self.configurations[config_key] = config_value
                
                self.configuration_metrics.total_configurations += 1
                self._update_configuration_metrics(config_value)
        
        self.logger.info("Configuration loaded from environment variables")
    
    async def _load_from_files(self) -> None:
        """Load configuration from files."""
        for config_path in self._config_paths:
            if config_path.exists():
                try:
                    await self._load_config_file(config_path)
                    self._watched_files[str(config_path)] = datetime.fromtimestamp(config_path.stat().st_mtime)
                except Exception as e:
                    self.logger.error(f"Failed to load config file: {e}", path=str(config_path))
    
    async def _load_config_file(self, config_path: Path) -> None:
        """Load a specific configuration file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    data = json.load(f)
                elif config_path.name == '.env':
                    # Simple .env file parsing
                    data = {}
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            data[key.strip()] = value.strip()
                else:
                    self.logger.warning(f"Unsupported config file format", path=str(config_path))
                    return
            
            await self._process_config_data(data, ConfigurationSource.FILE, str(config_path))
            
        except Exception as e:
            self.logger.error(f"Failed to load config file: {e}", path=str(config_path))
    
    async def _process_config_data(
        self,
        data: Dict[str, Any],
        source: ConfigurationSource,
        source_description: str
    ) -> None:
        """Process configuration data from a source."""
        def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
            result = {}
            for key, value in d.items():
                new_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    result.update(flatten_dict(value, new_key))
                else:
                    result[new_key] = value
            return result
        
        flattened_data = flatten_dict(data)
        
        for key, value in flattened_data.items():
            # Determine type and security level
            config_type = self._determine_type(value)
            security_level = SecurityLevel.INTERNAL
            
            if any(secret_word in key.lower() for secret_word in ['password', 'key', 'secret', 'token']):
                security_level = SecurityLevel.SECRET
            
            config_value = ConfigurationValue(
                key=key,
                value=value,
                type=config_type,
                source=source,
                security_level=security_level,
                description=f"Loaded from {source_description}"
            )
            
            with self._config_lock:
                self.configurations[key] = config_value
            
            self.configuration_metrics.total_configurations += 1
            self._update_configuration_metrics(config_value)
    
    async def _load_feature_flags(self) -> None:
        """Load feature flags from configuration."""
        feature_flags_file = Path("config/feature_flags.yaml")
        if feature_flags_file.exists():
            try:
                with open(feature_flags_file, 'r') as f:
                    flags_data = yaml.safe_load(f)
                
                for flag_name, flag_config in flags_data.items():
                    await self.create_feature_flag(
                        name=flag_name,
                        enabled=flag_config.get('enabled', False),
                        rollout_percentage=flag_config.get('rollout_percentage', 0.0),
                        target_groups=set(flag_config.get('target_groups', [])),
                        conditions=flag_config.get('conditions', {}),
                        description=flag_config.get('description', ''),
                        expires_at=datetime.fromisoformat(flag_config['expires_at']) if flag_config.get('expires_at') else None
                    )
                
                self.logger.info("Feature flags loaded from configuration file")
                
            except Exception as e:
                self.logger.error(f"Failed to load feature flags: {e}")
    
    # Private Helper Methods
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for secrets."""
        # In production, this should come from a secure key management system
        key_file = Path(".encryption_key")
        if key_file.exists():
            return key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            return key
    
    def _encrypt_value(self, value: Any) -> str:
        """Encrypt a configuration value."""
        try:
            value_bytes = json.dumps(value).encode()
            encrypted_bytes = self._cipher_suite.encrypt(value_bytes)
            return base64.b64encode(encrypted_bytes).decode()
        except Exception as e:
            self.logger.error(f"Failed to encrypt value: {e}")
            raise
    
    def _decrypt_value(self, encrypted_value: str) -> Any:
        """Decrypt a configuration value."""
        try:
            encrypted_bytes = base64.b64decode(encrypted_value.encode())
            decrypted_bytes = self._cipher_suite.decrypt(encrypted_bytes)
            return json.loads(decrypted_bytes.decode())
        except Exception as e:
            self.logger.error(f"Failed to decrypt value: {e}")
            raise
    
    def _determine_type(self, value: Any) -> ConfigurationType:
        """Determine configuration type from value."""
        if isinstance(value, bool):
            return ConfigurationType.BOOLEAN
        elif isinstance(value, int):
            return ConfigurationType.INTEGER
        elif isinstance(value, float):
            return ConfigurationType.FLOAT
        elif isinstance(value, list):
            return ConfigurationType.LIST
        elif isinstance(value, dict):
            return ConfigurationType.DICT
        else:
            return ConfigurationType.STRING
    
    def _is_float(self, value: str) -> bool:
        """Check if string represents a float."""
        try:
            float(value)
            return '.' in value
        except ValueError:
            return False
    
    def _get_from_cache(self, key: str) -> Any:
        """Get value from cache."""
        with self._cache_lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if (datetime.utcnow() - timestamp).total_seconds() < self._cache_ttl:
                    return value
                else:
                    del self._cache[key]
        return None
    
    def _set_in_cache(self, key: str, value: Any) -> None:
        """Set value in cache."""
        with self._cache_lock:
            self._cache[key] = (value, datetime.utcnow())
    
    def _invalidate_cache(self, key: str) -> None:
        """Invalidate cache entry."""
        with self._cache_lock:
            self._cache.pop(key, None)
    
    def _update_configuration_metrics(self, config_value: ConfigurationValue) -> None:
        """Update configuration-specific metrics."""
        # Update by source
        current = self.configuration_metrics.configurations_by_source.get(config_value.source, 0)
        self.configuration_metrics.configurations_by_source[config_value.source] = current + 1
        
        # Update by type
        current = self.configuration_metrics.configurations_by_type.get(config_value.type, 0)
        self.configuration_metrics.configurations_by_type[config_value.type] = current + 1
        
        # Update special counts
        if config_value.encrypted:
            self.configuration_metrics.encrypted_configurations += 1
        
        if config_value.is_secret():
            self.configuration_metrics.secret_configurations += 1
    
    def _update_lookup_time_metrics(self, lookup_time_ms: float) -> None:
        """Update lookup time metrics."""
        total_lookups = self.configuration_metrics.cache_hits + self.configuration_metrics.cache_misses
        current_avg = self.configuration_metrics.avg_lookup_time_ms
        
        if total_lookups == 1:
            self.configuration_metrics.avg_lookup_time_ms = lookup_time_ms
        else:
            self.configuration_metrics.avg_lookup_time_ms = (
                (current_avg * (total_lookups - 1) + lookup_time_ms) / total_lookups
            )
    
    # Background Tasks
    
    async def _reload_loop(self) -> None:
        """Periodically reload configurations from sources."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                if self._shutdown_event.is_set():
                    break
                
                # Reload from environment (in case it changed)
                await self._load_from_environment()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Reload loop error: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Clean up old change history and cache entries."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(3600)  # Run every hour
                if self._shutdown_event.is_set():
                    break
                
                # Clean up old change history (keep last 1000 entries)
                if len(self.change_history) > 1000:
                    self.change_history = self.change_history[-1000:]
                
                # Clean up expired cache entries
                current_time = datetime.utcnow()
                with self._cache_lock:
                    expired_keys = []
                    for key, (value, timestamp) in self._cache.items():
                        if (current_time - timestamp).total_seconds() > self._cache_ttl:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self._cache[key]
                
                if expired_keys:
                    self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    async def _audit_loop(self) -> None:
        """Audit configuration changes and security."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                if self._shutdown_event.is_set():
                    break
                
                # Log audit summary
                recent_changes = [
                    change for change in self.change_history
                    if (datetime.utcnow() - change.timestamp).total_seconds() < 1800
                ]
                
                if recent_changes:
                    self.logger.info(
                        "Configuration audit summary",
                        recent_changes=len(recent_changes),
                        change_types={
                            change_type.value: len([c for c in recent_changes if c.change_type == change_type])
                            for change_type in ChangeType
                        }
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Audit loop error: {e}")
    
    async def _watch_file(self, config_path: Path) -> None:
        """Watch a configuration file for changes."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                if self._shutdown_event.is_set():
                    break
                
                if not config_path.exists():
                    continue
                
                current_mtime = datetime.fromtimestamp(config_path.stat().st_mtime)
                last_mtime = self._watched_files.get(str(config_path))
                
                if last_mtime and current_mtime > last_mtime:
                    self.logger.info(f"Configuration file changed, reloading", path=str(config_path))
                    await self._load_config_file(config_path)
                    self._watched_files[str(config_path)] = current_mtime
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"File watcher error: {e}", path=str(config_path))
    
    # Public API Extensions
    
    def get_configuration_metrics(self) -> ConfigurationManagerMetrics:
        """Get current configuration manager metrics."""
        return self.configuration_metrics
    
    def list_configurations(self, pattern: Optional[str] = None, include_secrets: bool = False) -> List[str]:
        """List all configuration keys, optionally filtered by pattern."""
        with self._config_lock:
            keys = []
            for key, config_value in self.configurations.items():
                if not include_secrets and config_value.is_secret():
                    continue
                
                if pattern and pattern not in key:
                    continue
                
                keys.append(key)
        
        return sorted(keys)
    
    def list_feature_flags(self) -> List[str]:
        """List all feature flag names."""
        with self._flags_lock:
            return sorted(self.feature_flags.keys())
    
    async def export_configuration(self, format: str = "yaml", include_secrets: bool = False) -> str:
        """Export configuration in specified format."""
        config_data = {}
        
        with self._config_lock:
            for key, config_value in self.configurations.items():
                if not include_secrets and config_value.is_secret():
                    continue
                
                # Reconstruct nested structure
                keys = key.split('.')
                current = config_data
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                
                value = config_value.value
                if config_value.encrypted and include_secrets:
                    try:
                        value = self._decrypt_value(value)
                    except Exception:
                        value = "<encrypted>"
                
                current[keys[-1]] = value
        
        if format.lower() == "yaml":
            return yaml.dump(config_data, default_flow_style=False)
        elif format.lower() == "json":
            return json.dumps(config_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def rollback_change(self, change_id: str) -> bool:
        """Rollback a configuration change."""
        change = next((c for c in self.change_history if c.id == change_id), None)
        if not change or not change.rollback_data:
            return False
        
        try:
            if change.change_type == ChangeType.CREATED:
                # Delete the created configuration
                return await self.delete(change.key, reason=f"Rollback of change {change_id}")
            
            elif change.change_type == ChangeType.UPDATED:
                # Restore old value
                old_config_data = change.rollback_data.get("old_config")
                if old_config_data:
                    return await self.set(
                        key=change.key,
                        value=change.old_value,
                        config_type=ConfigurationType(old_config_data["type"]),
                        source=ConfigurationSource(old_config_data["source"]),
                        security_level=SecurityLevel(old_config_data["security_level"]),
                        encrypt=old_config_data["encrypted"],
                        reason=f"Rollback of change {change_id}"
                    )
            
            elif change.change_type == ChangeType.DELETED:
                # Recreate the deleted configuration
                deleted_config_data = change.rollback_data.get("deleted_config")
                if deleted_config_data:
                    return await self.set(
                        key=change.key,
                        value=change.old_value,
                        config_type=ConfigurationType(deleted_config_data["type"]),
                        source=ConfigurationSource(deleted_config_data["source"]),
                        security_level=SecurityLevel(deleted_config_data["security_level"]),
                        encrypt=deleted_config_data["encrypted"],
                        reason=f"Rollback of change {change_id}"
                    )
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to rollback change: {e}", change_id=change_id)
            return False


# Plugin Examples

class ConfigurationValidationPlugin(ConfigurationPlugin):
    """Plugin for advanced configuration validation."""
    
    def __init__(self, schema: ConfigurationSchema):
        self.schema = schema
    
    @property
    def name(self) -> str:
        return "ConfigurationValidation"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, manager: BaseManager) -> None:
        pass
    
    async def cleanup(self) -> None:
        pass
    
    async def pre_set_hook(self, key: str, value: Any, config_type: ConfigurationType) -> Dict[str, Any]:
        # Validate against schema
        if key in self.schema.required_keys or key in self.schema.optional_keys:
            expected_type = self.schema.key_types.get(key)
            if expected_type and expected_type != config_type:
                raise ValueError(f"Type mismatch for {key}: expected {expected_type}, got {config_type}")
            
            pattern = self.schema.key_patterns.get(key)
            if pattern and isinstance(value, str) and not re.match(pattern, value):
                raise ValueError(f"Value for {key} does not match pattern {pattern}")
            
            allowed_values = self.schema.key_allowed_values.get(key)
            if allowed_values and value not in allowed_values:
                raise ValueError(f"Value for {key} not in allowed values: {allowed_values}")
        
        return {}


class ConfigurationAuditPlugin(ConfigurationPlugin):
    """Plugin for configuration change auditing."""
    
    @property
    def name(self) -> str:
        return "ConfigurationAudit"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, manager: BaseManager) -> None:
        self.audit_log: List[Dict[str, Any]] = []
    
    async def cleanup(self) -> None:
        pass
    
    async def post_set_hook(self, key: str, old_value: Any, new_value: Any) -> None:
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": "set_configuration",
            "key": key,
            "old_value": "<redacted>" if self._is_sensitive(key) else old_value,
            "new_value": "<redacted>" if self._is_sensitive(key) else new_value
        }
        self.audit_log.append(audit_entry)
        
        # Keep only last 1000 entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
    
    def _is_sensitive(self, key: str) -> bool:
        return any(word in key.lower() for word in ['password', 'key', 'secret', 'token'])