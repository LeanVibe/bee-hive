"""
Production Logging Configuration

Comprehensive logging configuration for production environments with
log aggregation, monitoring integration, and enterprise-grade features.
"""

import json
import os
import sys
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

import structlog
from pythonjsonlogger import jsonlogger

from .enhanced_logging import log_aggregator, correlation_context


class ProductionLogConfig:
    """
    Production logging configuration with enterprise features.
    
    Features:
    - Structured JSON logging for log aggregation systems
    - Multiple log levels and outputs
    - Log rotation and retention policies
    - Performance monitoring integration
    - Security event separation
    - Compliance-ready audit trails
    """
    
    def __init__(
        self,
        log_level: str = "INFO",
        log_dir: str = "/var/log/bee-hive",
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 10,
        enable_console: bool = True,
        enable_syslog: bool = False,
        enable_metrics: bool = True,
        enable_audit_separation: bool = True
    ):
        """Initialize production logging configuration."""
        self.log_level = log_level.upper()
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_syslog = enable_syslog
        self.enable_metrics = enable_metrics
        self.enable_audit_separation = enable_audit_separation
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        self._configure_production_logging()
    
    def _configure_production_logging(self) -> None:
        """Configure production logging with multiple outputs."""
        # Clear existing handlers
        logging.root.handlers = []
        
        # Configure structlog processors
        processors = self._get_structlog_processors()
        
        # Configure structlog
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=self._get_logger_factory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Set root logging level
        numeric_level = getattr(logging, self.log_level, logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            handlers=self._create_handlers(),
            format='%(message)s'  # structlog handles formatting
        )
        
        # Configure specific loggers
        self._configure_component_loggers()
        
        # Set up log rotation monitoring
        if self.enable_metrics:
            self._setup_log_monitoring()
    
    def _get_structlog_processors(self) -> List:
        """Get structlog processors for production."""
        processors = [
            # Filter by level first for performance
            structlog.stdlib.filter_by_level,
            
            # Add standard context
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            
            # Add correlation context
            self._correlation_processor,
            
            # Add timestamp with high precision
            structlog.processors.TimeStamper(
                fmt="%Y-%m-%dT%H:%M:%S.%fZ",
                utc=True,
                key="@timestamp"  # Elasticsearch/ELK compatible
            ),
            
            # Add performance context
            self._performance_processor,
            
            # Add environment context
            self._environment_processor,
            
            # Handle stack traces
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            
            # Ensure unicode compatibility
            structlog.processors.UnicodeDecoder(),
            
            # Final JSON rendering
            structlog.processors.JSONRenderer(
                serializer=self._custom_json_serializer,
                sort_keys=True
            )
        ]
        
        return processors
    
    def _correlation_processor(self, logger, method_name, event_dict):
        """Add correlation context to logs."""
        context = correlation_context.get_full_context()
        event_dict.update(context)
        
        # Add process and thread info for distributed tracing
        import threading
        event_dict["process_id"] = os.getpid()
        event_dict["thread_id"] = threading.get_ident()
        
        return event_dict
    
    def _performance_processor(self, logger, method_name, event_dict):
        """Add performance context to logs."""
        # Add log processing timestamp for latency analysis
        event_dict["log_processed_at"] = datetime.utcnow().isoformat() + "Z"
        
        # Add memory usage if available
        try:
            import psutil
            process = psutil.Process()
            event_dict["memory_usage_mb"] = round(process.memory_info().rss / 1024 / 1024, 2)
        except ImportError:
            pass
        
        # Classify log importance for indexing
        if "error" in str(event_dict.get("event", "")).lower():
            event_dict["log_importance"] = "high"
        elif "warning" in str(event_dict.get("event", "")).lower():
            event_dict["log_importance"] = "medium"
        else:
            event_dict["log_importance"] = "low"
        
        return event_dict
    
    def _environment_processor(self, logger, method_name, event_dict):
        """Add environment context to logs."""
        event_dict["environment"] = os.environ.get("ENVIRONMENT", "development")
        event_dict["service_name"] = "bee-hive"
        event_dict["service_version"] = os.environ.get("SERVICE_VERSION", "unknown")
        event_dict["hostname"] = os.environ.get("HOSTNAME", "unknown")
        event_dict["datacenter"] = os.environ.get("DATACENTER", "unknown")
        
        return event_dict
    
    def _custom_json_serializer(self, obj: Any) -> str:
        """Custom JSON serializer for complex objects."""
        def default_serializer(o):
            if isinstance(o, datetime):
                return o.isoformat() + "Z"
            if hasattr(o, 'to_dict'):
                return o.to_dict()
            if hasattr(o, '__dict__'):
                return {k: v for k, v in o.__dict__.items() 
                       if not k.startswith('_')}
            return str(o)
        
        return json.dumps(obj, default=default_serializer, separators=(',', ':'))
    
    def _get_logger_factory(self):
        """Get logger factory for structlog."""
        return structlog.stdlib.LoggerFactory()
    
    def _create_handlers(self) -> List[logging.Handler]:
        """Create logging handlers for production."""
        handlers = []
        
        # Console handler for development/debugging
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self._get_console_formatter())
            handlers.append(console_handler)
        
        # Main application log file
        app_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "bee-hive.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        app_handler.setFormatter(self._get_json_formatter())
        handlers.append(app_handler)
        
        # Error log file (errors and above)
        error_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "bee-hive-errors.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self._get_json_formatter())
        handlers.append(error_handler)
        
        # Audit log file (separate for compliance)
        if self.enable_audit_separation:
            audit_handler = logging.handlers.RotatingFileHandler(
                filename=self.log_dir / "bee-hive-audit.log",
                maxBytes=self.max_file_size,
                backupCount=self.backup_count * 2,  # Keep more audit logs
                encoding='utf-8'
            )
            audit_handler.addFilter(self._audit_filter)
            audit_handler.setFormatter(self._get_json_formatter())
            handlers.append(audit_handler)
        
        # Security events log file
        security_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "bee-hive-security.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count * 2,  # Keep more security logs
            encoding='utf-8'
        )
        security_handler.addFilter(self._security_filter)
        security_handler.setFormatter(self._get_json_formatter())
        handlers.append(security_handler)
        
        # Performance metrics log file
        if self.enable_metrics:
            metrics_handler = logging.handlers.RotatingFileHandler(
                filename=self.log_dir / "bee-hive-metrics.log",
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            metrics_handler.addFilter(self._metrics_filter)
            metrics_handler.setFormatter(self._get_json_formatter())
            handlers.append(metrics_handler)
        
        # Syslog handler for centralized logging
        if self.enable_syslog:
            try:
                syslog_handler = logging.handlers.SysLogHandler(
                    address=('localhost', 514),
                    facility=logging.handlers.SysLogHandler.LOG_LOCAL0
                )
                syslog_handler.setFormatter(self._get_syslog_formatter())
                handlers.append(syslog_handler)
            except Exception as e:
                print(f"Warning: Could not configure syslog handler: {e}")
        
        return handlers
    
    def _get_console_formatter(self) -> logging.Formatter:
        """Get formatter for console output."""
        return logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _get_json_formatter(self) -> jsonlogger.JsonFormatter:
        """Get JSON formatter for log files."""
        return jsonlogger.JsonFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S.%fZ'
        )
    
    def _get_syslog_formatter(self) -> logging.Formatter:
        """Get formatter for syslog."""
        return logging.Formatter(
            fmt='bee-hive[%(process)d]: %(levelname)s %(name)s %(message)s'
        )
    
    def _audit_filter(self, record) -> bool:
        """Filter for audit events."""
        message = getattr(record, 'getMessage', lambda: '')()
        return 'audit_event' in message or hasattr(record, 'audit_event')
    
    def _security_filter(self, record) -> bool:
        """Filter for security events."""
        message = getattr(record, 'getMessage', lambda: '')()
        return (
            'security_event' in message or 
            'authentication' in message or
            'authorization' in message or
            hasattr(record, 'security_event')
        )
    
    def _metrics_filter(self, record) -> bool:
        """Filter for metrics events."""
        message = getattr(record, 'getMessage', lambda: '')()
        return (
            'performance_metric' in message or
            'metric_type' in message or
            hasattr(record, 'metric_type')
        )
    
    def _configure_component_loggers(self) -> None:
        """Configure specific component loggers."""
        # Suppress noisy third-party loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
        
        # Configure important component loggers
        important_loggers = [
            'app.core.simple_orchestrator',
            'app.api_v2',
            'app.core.auth',
            'app.core.database'
        ]
        
        for logger_name in important_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG if self.log_level == 'DEBUG' else logging.INFO)
    
    def _setup_log_monitoring(self) -> None:
        """Set up log file monitoring and alerts."""
        # This would integrate with monitoring systems like Prometheus
        # For now, we'll just log configuration
        structlog.get_logger("log_config").info(
            "production_logging_configured",
            log_level=self.log_level,
            log_dir=str(self.log_dir),
            max_file_size=self.max_file_size,
            backup_count=self.backup_count,
            handlers_count=len(self._create_handlers())
        )
    
    def get_log_aggregation_config(self) -> Dict[str, Any]:
        """Get configuration for log aggregation systems."""
        return {
            "service": "bee-hive",
            "environment": os.environ.get("ENVIRONMENT", "development"),
            "log_files": {
                "application": str(self.log_dir / "bee-hive.log"),
                "errors": str(self.log_dir / "bee-hive-errors.log"),
                "audit": str(self.log_dir / "bee-hive-audit.log"),
                "security": str(self.log_dir / "bee-hive-security.log"),
                "metrics": str(self.log_dir / "bee-hive-metrics.log")
            },
            "log_format": "json",
            "retention_days": 30,
            "compression": True,
            "shipping_method": "filebeat"
        }
    
    def get_elasticsearch_index_template(self) -> Dict[str, Any]:
        """Get Elasticsearch index template for logs."""
        return {
            "index_patterns": ["bee-hive-logs-*"],
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "index.refresh_interval": "5s",
                "index.codec": "best_compression"
            },
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "level": {"type": "keyword"},
                    "logger": {"type": "keyword"},
                    "message": {"type": "text", "analyzer": "standard"},
                    "correlation_id": {"type": "keyword"},
                    "request_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "operation": {"type": "keyword"},
                    "component": {"type": "keyword"},
                    "duration_ms": {"type": "float"},
                    "status_code": {"type": "integer"},
                    "error_type": {"type": "keyword"},
                    "log_importance": {"type": "keyword"},
                    "environment": {"type": "keyword"},
                    "service_name": {"type": "keyword"},
                    "hostname": {"type": "keyword"}
                }
            }
        }


class LogRotationManager:
    """Manage log rotation and cleanup policies."""
    
    def __init__(self, log_config: ProductionLogConfig):
        self.log_config = log_config
        self.logger = structlog.get_logger("log_rotation")
    
    def cleanup_old_logs(self, retention_days: int = 30) -> None:
        """Clean up old log files based on retention policy."""
        import time
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
        
        for log_file in self.log_config.log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                self.logger.info(
                    "removing_old_log_file",
                    file=str(log_file),
                    age_days=(time.time() - log_file.stat().st_mtime) / (24 * 60 * 60)
                )
                log_file.unlink()
    
    def compress_old_logs(self) -> None:
        """Compress old log files to save space."""
        import gzip
        import shutil
        
        for log_file in self.log_config.log_dir.glob("*.log.[0-9]*"):
            if not log_file.name.endswith('.gz'):
                compressed_file = log_file.with_suffix(log_file.suffix + '.gz')
                
                with open(log_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                log_file.unlink()
                
                self.logger.info(
                    "compressed_log_file",
                    original=str(log_file),
                    compressed=str(compressed_file)
                )


# Factory function
def create_production_logging(
    environment: str = None,
    log_level: str = None,
    log_dir: str = None
) -> ProductionLogConfig:
    """
    Create production logging configuration.
    
    Args:
        environment: Environment name (development, staging, production)
        log_level: Logging level override
        log_dir: Log directory override
        
    Returns:
        Configured ProductionLogConfig instance
    """
    # Environment-specific defaults
    env = environment or os.environ.get("ENVIRONMENT", "development")
    
    if env == "production":
        config = ProductionLogConfig(
            log_level=log_level or "INFO",
            log_dir=log_dir or "/var/log/bee-hive",
            enable_console=False,
            enable_syslog=True,
            enable_metrics=True,
            enable_audit_separation=True
        )
    elif env == "staging":
        config = ProductionLogConfig(
            log_level=log_level or "DEBUG",
            log_dir=log_dir or "/tmp/bee-hive-logs",
            enable_console=True,
            enable_syslog=False,
            enable_metrics=True,
            enable_audit_separation=True
        )
    else:  # development
        config = ProductionLogConfig(
            log_level=log_level or "DEBUG",
            log_dir=log_dir or "./logs",
            enable_console=True,
            enable_syslog=False,
            enable_metrics=False,
            enable_audit_separation=False
        )
    
    return config


# Global configuration instance
production_log_config = None


def initialize_production_logging(
    environment: str = None,
    log_level: str = None,
    log_dir: str = None
) -> ProductionLogConfig:
    """Initialize global production logging configuration."""
    global production_log_config
    production_log_config = create_production_logging(environment, log_level, log_dir)
    return production_log_config


def get_production_log_config() -> Optional[ProductionLogConfig]:
    """Get the global production logging configuration."""
    return production_log_config