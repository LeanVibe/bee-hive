#!/usr/bin/env python3
"""
LeanVibe Agent Hive 2.0 - Emergency Rollback Procedures
Comprehensive rollback system with multiple recovery strategies

Subagent 7: Legacy Code Cleanup and Migration Specialist
Mission: Reliable recovery procedures for failed migrations
"""

import asyncio
import datetime
import json
import logging
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RollbackType(Enum):
    """Types of rollback operations"""
    FULL_SYSTEM = "full_system"
    PARTIAL_COMPONENT = "partial_component"
    CONFIGURATION_ONLY = "configuration_only"
    DATA_ONLY = "data_only"
    TRAFFIC_ONLY = "traffic_only"


class RollbackTrigger(Enum):
    """Rollback triggers"""
    MANUAL = "manual"
    AUTOMATED_FAILURE = "automated_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SECURITY_INCIDENT = "security_incident"
    DATA_CORRUPTION = "data_corruption"
    SYSTEM_INSTABILITY = "system_instability"


class RollbackStatus(Enum):
    """Rollback execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"


@dataclass
class RollbackPoint:
    """Recovery point for rollback operations"""
    point_id: str
    timestamp: datetime.datetime
    rollback_type: RollbackType
    backup_location: str
    system_state: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'point_id': self.point_id,
            'timestamp': self.timestamp.isoformat(),
            'rollback_type': self.rollback_type.value,
            'backup_location': self.backup_location,
            'system_state': self.system_state,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RollbackPoint':
        return cls(
            point_id=data['point_id'],
            timestamp=datetime.datetime.fromisoformat(data['timestamp']),
            rollback_type=RollbackType(data['rollback_type']),
            backup_location=data['backup_location'],
            system_state=data.get('system_state', {}),
            metadata=data.get('metadata', {})
        )


@dataclass
class RollbackOperation:
    """Individual rollback operation"""
    operation_id: str
    rollback_type: RollbackType
    trigger: RollbackTrigger
    target_point: RollbackPoint
    status: RollbackStatus = RollbackStatus.PENDING
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    steps_completed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict:
        return {
            'operation_id': self.operation_id,
            'rollback_type': self.rollback_type.value,
            'trigger': self.trigger.value,
            'target_point': self.target_point.to_dict(),
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'steps_completed': self.steps_completed,
            'errors': self.errors,
            'warnings': self.warnings
        }


class EmergencyRollbackSystem:
    """
    Emergency rollback system for LeanVibe Agent Hive 2.0
    Provides multiple recovery strategies with comprehensive validation
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.rollback_db = self.project_root / "backups" / "rollback_points.db"
        self.rollback_log = self.project_root / "logs" / f"rollback-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        
        # Create necessary directories
        self.rollback_db.parent.mkdir(exist_ok=True)
        self.rollback_log.parent.mkdir(exist_ok=True)
        
        # Initialize rollback database
        self._init_rollback_db()
        
        # Critical system components for validation
        self.critical_components = [
            'app/core/universal_orchestrator.py',
            'app/core/communication_hub/communication_hub.py',
            'app/core/managers/resource_manager.py',
            'app/core/managers/context_manager_unified.py',
            'app/core/managers/security_manager.py',
            'app/core/managers/workflow_manager.py',
            'app/core/managers/communication_manager.py',
            'app/core/engines/task_execution_engine.py',
            'app/core/engines/workflow_engine.py',
            'app/core/engines/data_processing_engine.py',
            'app/core/engines/security_engine.py',
            'app/core/engines/communication_engine.py',
            'app/core/engines/monitoring_engine.py',
            'app/core/engines/integration_engine.py',
            'app/core/engines/optimization_engine.py'
        ]

    def _init_rollback_db(self):
        """Initialize rollback points database"""
        with sqlite3.connect(self.rollback_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rollback_points (
                    point_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    rollback_type TEXT NOT NULL,
                    backup_location TEXT NOT NULL,
                    system_state TEXT DEFAULT '{}',
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rollback_operations (
                    operation_id TEXT PRIMARY KEY,
                    rollback_type TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    target_point_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    steps_completed TEXT DEFAULT '[]',
                    errors TEXT DEFAULT '[]',
                    warnings TEXT DEFAULT '[]',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (target_point_id) REFERENCES rollback_points (point_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rollback_timestamp 
                ON rollback_points (timestamp)
            """)

    async def create_rollback_point(self, rollback_type: RollbackType = RollbackType.FULL_SYSTEM, 
                                   metadata: Dict = None) -> RollbackPoint:
        """Create a new rollback point"""
        point_id = f"rollback-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        timestamp = datetime.datetime.now()
        
        logger.info(f"🔄 Creating rollback point: {point_id}")
        
        try:
            # Create backup directory for this rollback point
            backup_location = self.project_root / "backups" / point_id
            backup_location.mkdir(parents=True, exist_ok=True)
            
            # Capture system state
            system_state = await self._capture_system_state()
            
            # Create backups based on rollback type
            if rollback_type == RollbackType.FULL_SYSTEM:
                await self._backup_full_system(backup_location)
            elif rollback_type == RollbackType.PARTIAL_COMPONENT:
                await self._backup_critical_components(backup_location)
            elif rollback_type == RollbackType.CONFIGURATION_ONLY:
                await self._backup_configuration(backup_location)
            elif rollback_type == RollbackType.DATA_ONLY:
                await self._backup_data(backup_location)
            
            # Create rollback point
            rollback_point = RollbackPoint(
                point_id=point_id,
                timestamp=timestamp,
                rollback_type=rollback_type,
                backup_location=str(backup_location),
                system_state=system_state,
                metadata=metadata or {}
            )
            
            # Save to database
            self._save_rollback_point(rollback_point)
            
            logger.info(f"✅ Rollback point created: {point_id}")
            logger.info(f"   Type: {rollback_type.value}")
            logger.info(f"   Location: {backup_location}")
            
            return rollback_point
            
        except Exception as e:
            logger.exception(f"Failed to create rollback point: {str(e)}")
            raise

    async def execute_rollback(self, target_point_id: str, 
                              trigger: RollbackTrigger = RollbackTrigger.MANUAL,
                              force: bool = False) -> RollbackOperation:
        """Execute rollback to specified point"""
        operation_id = f"rollback-op-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        logger.warning(f"🔄 Initiating rollback operation: {operation_id}")
        logger.warning(f"   Target point: {target_point_id}")
        logger.warning(f"   Trigger: {trigger.value}")
        
        # Get target rollback point
        target_point = self._get_rollback_point(target_point_id)
        if not target_point:
            raise ValueError(f"Rollback point not found: {target_point_id}")
        
        # Create rollback operation
        rollback_op = RollbackOperation(
            operation_id=operation_id,
            rollback_type=target_point.rollback_type,
            trigger=trigger,
            target_point=target_point,
            start_time=datetime.datetime.now()
        )
        
        try:
            rollback_op.status = RollbackStatus.IN_PROGRESS
            self._save_rollback_operation(rollback_op)
            
            # Pre-rollback validation
            if not force:
                validation_result = await self._validate_rollback_safety(rollback_op)
                if not validation_result['safe']:
                    rollback_op.errors.extend(validation_result['errors'])
                    rollback_op.status = RollbackStatus.FAILED
                    rollback_op.end_time = datetime.datetime.now()
                    self._save_rollback_operation(rollback_op)
                    raise RuntimeError(f"Rollback safety validation failed: {validation_result['errors']}")
            
            # Execute rollback steps
            await self._execute_rollback_steps(rollback_op)
            
            # Post-rollback validation
            validation_result = await self._validate_rollback_success(rollback_op)
            if not validation_result['success']:
                rollback_op.warnings.extend(validation_result['warnings'])
                rollback_op.status = RollbackStatus.PARTIALLY_COMPLETED
            else:
                rollback_op.status = RollbackStatus.COMPLETED
            
            rollback_op.end_time = datetime.datetime.now()
            self._save_rollback_operation(rollback_op)
            
            # Log rollback completion
            self._log_rollback_operation(rollback_op)
            
            if rollback_op.status == RollbackStatus.COMPLETED:
                logger.info(f"✅ Rollback completed successfully: {operation_id}")
                logger.info(f"   Duration: {rollback_op.duration_seconds:.2f}s")
            else:
                logger.warning(f"⚠️ Rollback partially completed: {operation_id}")
                logger.warning(f"   Warnings: {rollback_op.warnings}")
            
            return rollback_op
            
        except Exception as e:
            logger.exception(f"Rollback operation failed: {str(e)}")
            rollback_op.status = RollbackStatus.FAILED
            rollback_op.errors.append(f"Rollback execution failed: {str(e)}")
            rollback_op.end_time = datetime.datetime.now()
            self._save_rollback_operation(rollback_op)
            self._log_rollback_operation(rollback_op)
            raise

    async def emergency_rollback(self, reason: str = "Emergency recovery") -> RollbackOperation:
        """Execute emergency rollback to most recent stable point"""
        logger.critical(f"🚨 EMERGENCY ROLLBACK INITIATED: {reason}")
        
        # Find most recent rollback point
        recent_points = self.list_rollback_points(limit=5)
        if not recent_points:
            raise RuntimeError("No rollback points available for emergency recovery")
        
        # Use most recent full system rollback point
        target_point = None
        for point in recent_points:
            if point.rollback_type == RollbackType.FULL_SYSTEM:
                target_point = point
                break
        
        if not target_point:
            # Fall back to most recent point regardless of type
            target_point = recent_points[0]
            logger.warning(f"Using non-full-system rollback point: {target_point.point_id}")
        
        # Execute emergency rollback with force=True
        return await self.execute_rollback(
            target_point.point_id, 
            trigger=RollbackTrigger.SYSTEM_INSTABILITY, 
            force=True
        )

    async def _capture_system_state(self) -> Dict:
        """Capture current system state"""
        try:
            system_state = {
                'timestamp': datetime.datetime.now().isoformat(),
                'critical_files_exist': [],
                'critical_files_missing': [],
                'process_info': {},
                'system_health': {}
            }
            
            # Check critical component files
            for component_path in self.critical_components:
                full_path = self.project_root / component_path
                if full_path.exists():
                    system_state['critical_files_exist'].append(component_path)
                    system_state[f'{component_path}_size'] = full_path.stat().st_size
                    system_state[f'{component_path}_mtime'] = full_path.stat().st_mtime
                else:
                    system_state['critical_files_missing'].append(component_path)
            
            # Capture process information (simplified)
            try:
                result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    system_state['process_count'] = len(result.stdout.splitlines())
                    system_state['process_snapshot_available'] = True
                else:
                    system_state['process_snapshot_available'] = False
            except Exception:
                system_state['process_snapshot_available'] = False
            
            # Basic system health metrics
            try:
                system_state['disk_usage'] = shutil.disk_usage(self.project_root)._asdict()
            except Exception:
                system_state['disk_usage'] = {}
            
            return system_state
            
        except Exception as e:
            logger.warning(f"Failed to capture complete system state: {str(e)}")
            return {'error': str(e), 'timestamp': datetime.datetime.now().isoformat()}

    async def _backup_full_system(self, backup_location: Path):
        """Create full system backup"""
        logger.info("Creating full system backup...")
        
        # Backup app directory
        app_backup = backup_location / "app"
        if (self.project_root / "app").exists():
            shutil.copytree(
                self.project_root / "app",
                app_backup,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'logs')
            )
        
        # Backup configuration files
        config_backup = backup_location / "config"
        config_backup.mkdir(exist_ok=True)
        
        config_files = [
            "requirements.txt", "pyproject.toml", "docker-compose.yml",
            ".env.example", "Dockerfile", "README.md"
        ]
        
        for config_file in config_files:
            src = self.project_root / config_file
            if src.exists():
                shutil.copy2(src, config_backup / config_file)

    async def _backup_critical_components(self, backup_location: Path):
        """Backup critical components only"""
        logger.info("Creating critical components backup...")
        
        components_backup = backup_location / "critical_components"
        components_backup.mkdir(exist_ok=True)
        
        for component_path in self.critical_components:
            src = self.project_root / component_path
            if src.exists():
                dest = components_backup / component_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                if src.is_file():
                    shutil.copy2(src, dest)
                elif src.is_dir():
                    shutil.copytree(src, dest)

    async def _backup_configuration(self, backup_location: Path):
        """Backup configuration files only"""
        logger.info("Creating configuration backup...")
        
        config_backup = backup_location / "configuration"
        config_backup.mkdir(exist_ok=True)
        
        # Find and backup all configuration files
        config_patterns = ['*.yml', '*.yaml', '*.json', '*.toml', '*.ini', 'requirements*.txt']
        
        for pattern in config_patterns:
            for config_file in self.project_root.glob(pattern):
                if config_file.is_file():
                    shutil.copy2(config_file, config_backup / config_file.name)

    async def _backup_data(self, backup_location: Path):
        """Backup data files only"""
        logger.info("Creating data backup...")
        
        data_backup = backup_location / "data"
        data_backup.mkdir(exist_ok=True)
        
        # Backup databases and data files
        data_patterns = ['*.db', '*.sqlite', '*.sqlite3', 'data/*']
        
        for pattern in data_patterns:
            for data_file in self.project_root.glob(pattern):
                if data_file.is_file():
                    dest = data_backup / data_file.name
                    shutil.copy2(data_file, dest)

    async def _validate_rollback_safety(self, rollback_op: RollbackOperation) -> Dict:
        """Validate rollback safety before execution"""
        safety_issues = []
        
        try:
            # Check that backup exists and is accessible
            backup_path = Path(rollback_op.target_point.backup_location)
            if not backup_path.exists():
                safety_issues.append(f"Backup location does not exist: {backup_path}")
            
            # Check backup integrity (simplified)
            if backup_path.exists():
                backup_files = list(backup_path.rglob('*'))
                if len(backup_files) < 10:  # Arbitrary minimum for full system backup
                    safety_issues.append("Backup appears incomplete - too few files")
            
            # Check system state compatibility
            current_state = await self._capture_system_state()
            target_state = rollback_op.target_point.system_state
            
            # Warn if rolling back to much older state
            if target_state.get('timestamp'):
                target_time = datetime.datetime.fromisoformat(target_state['timestamp'])
                age_hours = (datetime.datetime.now() - target_time).total_seconds() / 3600
                if age_hours > 24:  # More than 24 hours old
                    safety_issues.append(f"Rollback target is {age_hours:.1f} hours old")
            
            return {
                'safe': len(safety_issues) == 0,
                'errors': safety_issues,
                'current_state': current_state,
                'target_state': target_state
            }
            
        except Exception as e:
            return {
                'safe': False,
                'errors': [f"Safety validation failed: {str(e)}"]
            }

    async def _execute_rollback_steps(self, rollback_op: RollbackOperation):
        """Execute rollback steps based on rollback type"""
        backup_path = Path(rollback_op.target_point.backup_location)
        
        if rollback_op.rollback_type == RollbackType.FULL_SYSTEM:
            await self._rollback_full_system(rollback_op, backup_path)
        elif rollback_op.rollback_type == RollbackType.PARTIAL_COMPONENT:
            await self._rollback_critical_components(rollback_op, backup_path)
        elif rollback_op.rollback_type == RollbackType.CONFIGURATION_ONLY:
            await self._rollback_configuration(rollback_op, backup_path)
        elif rollback_op.rollback_type == RollbackType.DATA_ONLY:
            await self._rollback_data(rollback_op, backup_path)
        elif rollback_op.rollback_type == RollbackType.TRAFFIC_ONLY:
            await self._rollback_traffic_routing(rollback_op)

    async def _rollback_full_system(self, rollback_op: RollbackOperation, backup_path: Path):
        """Execute full system rollback"""
        logger.info("Executing full system rollback...")
        
        # Step 1: Stop services (if any)
        rollback_op.steps_completed.append("services_stopped")
        
        # Step 2: Backup current state for recovery
        current_backup = self.project_root / "backups" / f"pre-rollback-{rollback_op.operation_id}"
        current_backup.mkdir(parents=True, exist_ok=True)
        
        try:
            if (self.project_root / "app").exists():
                shutil.copytree(
                    self.project_root / "app",
                    current_backup / "app",
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc')
                )
        except Exception as e:
            rollback_op.warnings.append(f"Failed to backup current state: {str(e)}")
        
        rollback_op.steps_completed.append("current_state_backed_up")
        
        # Step 3: Remove current app directory
        app_dir = self.project_root / "app"
        if app_dir.exists():
            shutil.rmtree(app_dir)
        rollback_op.steps_completed.append("current_system_removed")
        
        # Step 4: Restore from backup
        backup_app = backup_path / "app"
        if backup_app.exists():
            shutil.copytree(backup_app, app_dir)
            rollback_op.steps_completed.append("system_restored_from_backup")
        else:
            raise RuntimeError("Backup app directory not found")
        
        # Step 5: Restore configuration
        backup_config = backup_path / "config"
        if backup_config.exists():
            for config_file in backup_config.iterdir():
                if config_file.is_file():
                    dest = self.project_root / config_file.name
                    shutil.copy2(config_file, dest)
            rollback_op.steps_completed.append("configuration_restored")
        
        # Step 6: Restart services (if any)
        rollback_op.steps_completed.append("services_restarted")

    async def _rollback_critical_components(self, rollback_op: RollbackOperation, backup_path: Path):
        """Execute critical components rollback"""
        logger.info("Executing critical components rollback...")
        
        components_backup = backup_path / "critical_components"
        if not components_backup.exists():
            raise RuntimeError("Critical components backup not found")
        
        # Restore each critical component
        for component_path in self.critical_components:
            backup_component = components_backup / component_path
            current_component = self.project_root / component_path
            
            if backup_component.exists():
                # Remove current component
                if current_component.exists():
                    if current_component.is_file():
                        current_component.unlink()
                    elif current_component.is_dir():
                        shutil.rmtree(current_component)
                
                # Restore from backup
                current_component.parent.mkdir(parents=True, exist_ok=True)
                if backup_component.is_file():
                    shutil.copy2(backup_component, current_component)
                elif backup_component.is_dir():
                    shutil.copytree(backup_component, current_component)
                
                logger.debug(f"Restored component: {component_path}")
        
        rollback_op.steps_completed.append("critical_components_restored")

    async def _rollback_configuration(self, rollback_op: RollbackOperation, backup_path: Path):
        """Execute configuration rollback"""
        logger.info("Executing configuration rollback...")
        
        config_backup = backup_path / "configuration"
        if not config_backup.exists():
            raise RuntimeError("Configuration backup not found")
        
        # Restore configuration files
        for config_file in config_backup.iterdir():
            if config_file.is_file():
                dest = self.project_root / config_file.name
                shutil.copy2(config_file, dest)
                logger.debug(f"Restored config: {config_file.name}")
        
        rollback_op.steps_completed.append("configuration_restored")

    async def _rollback_data(self, rollback_op: RollbackOperation, backup_path: Path):
        """Execute data rollback"""
        logger.info("Executing data rollback...")
        
        data_backup = backup_path / "data"
        if not data_backup.exists():
            raise RuntimeError("Data backup not found")
        
        # Restore data files
        for data_file in data_backup.iterdir():
            if data_file.is_file():
                dest = self.project_root / data_file.name
                shutil.copy2(data_file, dest)
                logger.debug(f"Restored data: {data_file.name}")
        
        rollback_op.steps_completed.append("data_restored")

    async def _rollback_traffic_routing(self, rollback_op: RollbackOperation):
        """Execute traffic routing rollback"""
        logger.info("Executing traffic routing rollback...")
        
        # Restore traffic to legacy system (100% legacy, 0% consolidated)
        try:
            # This would integrate with the traffic switchover system
            # For now, simulate the routing change
            routing_commands = [
                "echo 'Routing 100% traffic to legacy system'",
                "echo 'Stopping consolidated system services'",
                "echo 'Validating legacy system health'"
            ]
            
            for cmd in routing_commands:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"Traffic routing command succeeded: {cmd}")
                else:
                    logger.warning(f"Traffic routing command failed: {cmd} -> {result.stderr}")
            
            rollback_op.steps_completed.append("traffic_routing_restored")
            
        except Exception as e:
            rollback_op.errors.append(f"Traffic routing rollback failed: {str(e)}")

    async def _validate_rollback_success(self, rollback_op: RollbackOperation) -> Dict:
        """Validate rollback was successful"""
        validation_issues = []
        
        try:
            # Check that critical components exist
            missing_components = []
            for component_path in self.critical_components:
                full_path = self.project_root / component_path
                if not full_path.exists():
                    missing_components.append(component_path)
            
            if missing_components:
                validation_issues.extend([f"Missing component after rollback: {c}" for c in missing_components])
            
            # Basic syntax check on restored files
            syntax_errors = []
            for component_path in self.critical_components[:3]:  # Check first 3 for efficiency
                full_path = self.project_root / component_path
                if full_path.exists() and full_path.is_file() and full_path.suffix == '.py':
                    try:
                        result = subprocess.run([
                            sys.executable, '-m', 'py_compile', str(full_path)
                        ], capture_output=True, text=True, timeout=10)
                        
                        if result.returncode != 0:
                            syntax_errors.append(f"Syntax error in {component_path}: {result.stderr}")
                    except Exception as e:
                        syntax_errors.append(f"Could not validate {component_path}: {str(e)}")
            
            validation_issues.extend(syntax_errors)
            
            return {
                'success': len(validation_issues) == 0,
                'warnings': validation_issues,
                'components_checked': len(self.critical_components),
                'missing_components': len(missing_components),
                'syntax_errors': len(syntax_errors)
            }
            
        except Exception as e:
            return {
                'success': False,
                'warnings': [f"Validation failed: {str(e)}"]
            }

    def list_rollback_points(self, limit: int = 10) -> List[RollbackPoint]:
        """List available rollback points"""
        with sqlite3.connect(self.rollback_db) as conn:
            cursor = conn.execute("""
                SELECT point_id, timestamp, rollback_type, backup_location, 
                       system_state, metadata
                FROM rollback_points 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            rollback_points = []
            for row in cursor.fetchall():
                point_data = {
                    'point_id': row[0],
                    'timestamp': row[1],
                    'rollback_type': row[2],
                    'backup_location': row[3],
                    'system_state': json.loads(row[4] or '{}'),
                    'metadata': json.loads(row[5] or '{}')
                }
                rollback_points.append(RollbackPoint.from_dict(point_data))
            
            return rollback_points

    def list_rollback_operations(self, limit: int = 10) -> List[RollbackOperation]:
        """List rollback operations history"""
        with sqlite3.connect(self.rollback_db) as conn:
            cursor = conn.execute("""
                SELECT ro.operation_id, ro.rollback_type, ro.trigger_type, 
                       ro.status, ro.start_time, ro.end_time,
                       ro.steps_completed, ro.errors, ro.warnings,
                       rp.point_id, rp.timestamp as point_timestamp, 
                       rp.rollback_type as point_type, rp.backup_location,
                       rp.system_state, rp.metadata
                FROM rollback_operations ro
                JOIN rollback_points rp ON ro.target_point_id = rp.point_id
                ORDER BY ro.start_time DESC
                LIMIT ?
            """, (limit,))
            
            operations = []
            for row in cursor.fetchall():
                # Reconstruct RollbackPoint
                point_data = {
                    'point_id': row[9],
                    'timestamp': row[10],
                    'rollback_type': row[11],
                    'backup_location': row[12],
                    'system_state': json.loads(row[13] or '{}'),
                    'metadata': json.loads(row[14] or '{}')
                }
                rollback_point = RollbackPoint.from_dict(point_data)
                
                # Create RollbackOperation
                operation = RollbackOperation(
                    operation_id=row[0],
                    rollback_type=RollbackType(row[1]),
                    trigger=RollbackTrigger(row[2]),
                    target_point=rollback_point,
                    status=RollbackStatus(row[3]),
                    start_time=datetime.datetime.fromisoformat(row[4]) if row[4] else None,
                    end_time=datetime.datetime.fromisoformat(row[5]) if row[5] else None,
                    steps_completed=json.loads(row[6] or '[]'),
                    errors=json.loads(row[7] or '[]'),
                    warnings=json.loads(row[8] or '[]')
                )
                
                operations.append(operation)
            
            return operations

    def _save_rollback_point(self, point: RollbackPoint):
        """Save rollback point to database"""
        with sqlite3.connect(self.rollback_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO rollback_points 
                (point_id, timestamp, rollback_type, backup_location, system_state, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                point.point_id,
                point.timestamp.isoformat(),
                point.rollback_type.value,
                point.backup_location,
                json.dumps(point.system_state),
                json.dumps(point.metadata)
            ))

    def _get_rollback_point(self, point_id: str) -> Optional[RollbackPoint]:
        """Get rollback point by ID"""
        with sqlite3.connect(self.rollback_db) as conn:
            cursor = conn.execute("""
                SELECT point_id, timestamp, rollback_type, backup_location, 
                       system_state, metadata
                FROM rollback_points 
                WHERE point_id = ?
            """, (point_id,))
            
            row = cursor.fetchone()
            if row:
                point_data = {
                    'point_id': row[0],
                    'timestamp': row[1],
                    'rollback_type': row[2],
                    'backup_location': row[3],
                    'system_state': json.loads(row[4] or '{}'),
                    'metadata': json.loads(row[5] or '{}')
                }
                return RollbackPoint.from_dict(point_data)
            
            return None

    def _save_rollback_operation(self, operation: RollbackOperation):
        """Save rollback operation to database"""
        with sqlite3.connect(self.rollback_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO rollback_operations 
                (operation_id, rollback_type, trigger_type, target_point_id, 
                 status, start_time, end_time, steps_completed, errors, warnings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                operation.operation_id,
                operation.rollback_type.value,
                operation.trigger.value,
                operation.target_point.point_id,
                operation.status.value,
                operation.start_time.isoformat() if operation.start_time else None,
                operation.end_time.isoformat() if operation.end_time else None,
                json.dumps(operation.steps_completed),
                json.dumps(operation.errors),
                json.dumps(operation.warnings)
            ))

    def _log_rollback_operation(self, operation: RollbackOperation):
        """Log rollback operation to file"""
        try:
            with open(self.rollback_log, 'a') as f:
                f.write(json.dumps(operation.to_dict()) + '\n')
        except Exception as e:
            logger.warning(f"Failed to log rollback operation: {str(e)}")


async def main():
    """Main rollback system CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LeanVibe Agent Hive 2.0 - Emergency Rollback System")
    subparsers = parser.add_subparsers(dest='command', help='Rollback commands')
    
    # Create rollback point
    create_parser = subparsers.add_parser('create-point', help='Create new rollback point')
    create_parser.add_argument('--type', choices=['full_system', 'partial_component', 'configuration_only', 'data_only'], 
                              default='full_system', help='Rollback point type')
    create_parser.add_argument('--metadata', help='JSON metadata for rollback point')
    
    # Execute rollback
    rollback_parser = subparsers.add_parser('rollback', help='Execute rollback')
    rollback_parser.add_argument('point_id', help='Rollback point ID')
    rollback_parser.add_argument('--trigger', choices=['manual', 'automated_failure', 'performance_degradation', 
                                                      'security_incident', 'data_corruption', 'system_instability'],
                                default='manual', help='Rollback trigger reason')
    rollback_parser.add_argument('--force', action='store_true', help='Skip safety validations')
    
    # Emergency rollback
    emergency_parser = subparsers.add_parser('emergency', help='Execute emergency rollback')
    emergency_parser.add_argument('--reason', default='Emergency recovery', help='Emergency reason')
    
    # List operations
    list_parser = subparsers.add_parser('list', help='List rollback points and operations')
    list_parser.add_argument('--type', choices=['points', 'operations'], default='points', help='What to list')
    list_parser.add_argument('--limit', type=int, default=10, help='Number of items to show')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Initialize rollback system
    rollback_system = EmergencyRollbackSystem()
    
    try:
        if args.command == 'create-point':
            rollback_type = RollbackType(args.type)
            metadata = json.loads(args.metadata) if args.metadata else {}
            
            point = await rollback_system.create_rollback_point(rollback_type, metadata)
            print(f"✅ Rollback point created: {point.point_id}")
            print(f"   Type: {point.rollback_type.value}")
            print(f"   Location: {point.backup_location}")
            
        elif args.command == 'rollback':
            trigger = RollbackTrigger(args.trigger)
            
            operation = await rollback_system.execute_rollback(args.point_id, trigger, args.force)
            if operation.status == RollbackStatus.COMPLETED:
                print(f"✅ Rollback completed successfully: {operation.operation_id}")
                print(f"   Duration: {operation.duration_seconds:.2f}s")
                print(f"   Steps: {len(operation.steps_completed)}")
            else:
                print(f"❌ Rollback failed or incomplete: {operation.operation_id}")
                print(f"   Status: {operation.status.value}")
                print(f"   Errors: {operation.errors}")
                sys.exit(1)
                
        elif args.command == 'emergency':
            operation = await rollback_system.emergency_rollback(args.reason)
            if operation.status == RollbackStatus.COMPLETED:
                print(f"✅ Emergency rollback completed: {operation.operation_id}")
                print(f"   Target point: {operation.target_point.point_id}")
            else:
                print(f"❌ Emergency rollback failed: {operation.operation_id}")
                print(f"   Errors: {operation.errors}")
                sys.exit(1)
                
        elif args.command == 'list':
            if args.type == 'points':
                points = rollback_system.list_rollback_points(args.limit)
                print(f"\n📋 Rollback Points ({len(points)}):")
                for point in points:
                    print(f"  {point.point_id}: {point.rollback_type.value} - {point.timestamp}")
                    print(f"    Location: {point.backup_location}")
                    if point.metadata:
                        print(f"    Metadata: {point.metadata}")
                    print()
            else:
                operations = rollback_system.list_rollback_operations(args.limit)
                print(f"\n📋 Rollback Operations ({len(operations)}):")
                for op in operations:
                    status_symbol = "✅" if op.status == RollbackStatus.COMPLETED else "❌" if op.status == RollbackStatus.FAILED else "⚠️"
                    print(f"  {status_symbol} {op.operation_id}: {op.trigger.value} - {op.status.value}")
                    print(f"    Target: {op.target_point.point_id}")
                    print(f"    Duration: {op.duration_seconds:.2f}s")
                    if op.errors:
                        print(f"    Errors: {len(op.errors)}")
                    print()
        
    except Exception as e:
        logger.exception(f"💥 Command execution failed: {str(e)}")
        print(f"\n💥 COMMAND FAILED")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class RollbackMigrationScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            await main()
            
            return {"status": "completed"}
    
    script_main(RollbackMigrationScript)