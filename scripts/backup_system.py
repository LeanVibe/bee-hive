#!/usr/bin/env python3
"""
LeanVibe Agent Hive 2.0 - Comprehensive Backup System
Complete system backup with granular restore capabilities

Subagent 7: Legacy Code Cleanup and Migration Specialist
Mission: Comprehensive backup system with point-in-time recovery
"""

import asyncio
import datetime
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
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


class BackupType(Enum):
    """Types of backups supported"""
    FULL_SYSTEM = "full_system"
    CODE_ONLY = "code_only"
    CONFIG_ONLY = "config_only"
    DATABASE_ONLY = "database_only"
    INCREMENTAL = "incremental"


class BackupStatus(Enum):
    """Backup operation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"


@dataclass
class BackupMetadata:
    """Metadata for backup operations"""
    backup_id: str
    backup_type: BackupType
    timestamp: datetime.datetime
    status: BackupStatus
    source_path: str
    backup_path: str
    file_count: int = 0
    total_size_bytes: int = 0
    checksum: str = ""
    compression_ratio: float = 0.0
    duration_seconds: float = 0.0
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'backup_id': self.backup_id,
            'backup_type': self.backup_type.value,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'source_path': self.source_path,
            'backup_path': self.backup_path,
            'file_count': self.file_count,
            'total_size_bytes': self.total_size_bytes,
            'checksum': self.checksum,
            'compression_ratio': self.compression_ratio,
            'duration_seconds': self.duration_seconds,
            'tags': self.tags,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BackupMetadata':
        """Create from dictionary"""
        return cls(
            backup_id=data['backup_id'],
            backup_type=BackupType(data['backup_type']),
            timestamp=datetime.datetime.fromisoformat(data['timestamp']),
            status=BackupStatus(data['status']),
            source_path=data['source_path'],
            backup_path=data['backup_path'],
            file_count=data.get('file_count', 0),
            total_size_bytes=data.get('total_size_bytes', 0),
            checksum=data.get('checksum', ''),
            compression_ratio=data.get('compression_ratio', 0.0),
            duration_seconds=data.get('duration_seconds', 0.0),
            tags=data.get('tags', []),
            notes=data.get('notes', '')
        )


@dataclass
class RestoreOptions:
    """Options for restore operations"""
    target_path: str
    overwrite_existing: bool = False
    verify_checksums: bool = True
    restore_permissions: bool = True
    exclude_patterns: List[str] = field(default_factory=list)


class ComprehensiveBackupSystem:
    """
    Comprehensive backup system for LeanVibe Agent Hive 2.0
    Supports full, incremental, and targeted backups with verification
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.backup_root = self.project_root / "backups"
        self.metadata_db = self.backup_root / "backup_metadata.db"
        
        # Ensure backup directory exists
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata database
        self._init_metadata_db()
        
        # Default exclusion patterns
        self.default_exclude_patterns = [
            '__pycache__',
            '*.pyc',
            '*.pyo',
            '.git',
            'node_modules',
            '.env',
            '*.log',
            'logs/*',
            'temp/*',
            '*.tmp'
        ]

    def _init_metadata_db(self):
        """Initialize backup metadata database"""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backups (
                    backup_id TEXT PRIMARY KEY,
                    backup_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    backup_path TEXT NOT NULL,
                    file_count INTEGER DEFAULT 0,
                    total_size_bytes INTEGER DEFAULT 0,
                    checksum TEXT DEFAULT '',
                    compression_ratio REAL DEFAULT 0.0,
                    duration_seconds REAL DEFAULT 0.0,
                    tags TEXT DEFAULT '',
                    notes TEXT DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backup_files (
                    backup_id TEXT,
                    file_path TEXT,
                    file_size INTEGER,
                    file_checksum TEXT,
                    FOREIGN KEY (backup_id) REFERENCES backups (backup_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_backup_timestamp 
                ON backups (timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_backup_type 
                ON backups (backup_type)
            """)

    async def create_full_system_backup(self, tags: List[str] = None, notes: str = "") -> BackupMetadata:
        """Create complete system backup"""
        backup_id = f"full-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        logger.info(f"🔄 Creating full system backup: {backup_id}")
        start_time = time.time()
        
        # Create backup metadata
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=BackupType.FULL_SYSTEM,
            timestamp=datetime.datetime.now(),
            status=BackupStatus.IN_PROGRESS,
            source_path=str(self.project_root),
            backup_path=str(self.backup_root / f"{backup_id}.tar.gz"),
            tags=tags or ["full", "migration"],
            notes=notes
        )
        
        try:
            # Save initial metadata
            self._save_backup_metadata(metadata)
            
            # Create compressed archive
            archive_path = Path(metadata.backup_path)
            file_count, total_size = await self._create_compressed_archive(
                self.project_root,
                archive_path,
                exclude_patterns=self.default_exclude_patterns + ["backups"]
            )
            
            # Calculate checksums
            checksum = self._calculate_file_checksum(archive_path)
            compression_ratio = self._calculate_compression_ratio(total_size, archive_path.stat().st_size)
            
            # Update metadata
            metadata.status = BackupStatus.COMPLETED
            metadata.file_count = file_count
            metadata.total_size_bytes = total_size
            metadata.checksum = checksum
            metadata.compression_ratio = compression_ratio
            metadata.duration_seconds = time.time() - start_time
            
            # Save updated metadata
            self._save_backup_metadata(metadata)
            
            logger.info(f"✅ Full system backup completed: {backup_id}")
            logger.info(f"   Files: {file_count}, Size: {total_size / (1024*1024):.2f} MB")
            logger.info(f"   Compression: {compression_ratio:.2f}x, Duration: {metadata.duration_seconds:.2f}s")
            
            return metadata
            
        except Exception as e:
            logger.exception(f"❌ Full system backup failed: {str(e)}")
            metadata.status = BackupStatus.FAILED
            metadata.notes = f"Error: {str(e)}"
            metadata.duration_seconds = time.time() - start_time
            self._save_backup_metadata(metadata)
            raise

    async def create_code_backup(self, tags: List[str] = None, notes: str = "") -> BackupMetadata:
        """Create backup of code files only"""
        backup_id = f"code-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        logger.info(f"🔄 Creating code backup: {backup_id}")
        start_time = time.time()
        
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=BackupType.CODE_ONLY,
            timestamp=datetime.datetime.now(),
            status=BackupStatus.IN_PROGRESS,
            source_path=str(self.project_root / "app"),
            backup_path=str(self.backup_root / f"{backup_id}.tar.gz"),
            tags=tags or ["code", "app"],
            notes=notes
        )
        
        try:
            self._save_backup_metadata(metadata)
            
            # Backup app directory
            source_dir = self.project_root / "app"
            archive_path = Path(metadata.backup_path)
            
            file_count, total_size = await self._create_compressed_archive(
                source_dir,
                archive_path,
                exclude_patterns=self.default_exclude_patterns
            )
            
            # Update metadata
            checksum = self._calculate_file_checksum(archive_path)
            compression_ratio = self._calculate_compression_ratio(total_size, archive_path.stat().st_size)
            
            metadata.status = BackupStatus.COMPLETED
            metadata.file_count = file_count
            metadata.total_size_bytes = total_size
            metadata.checksum = checksum
            metadata.compression_ratio = compression_ratio
            metadata.duration_seconds = time.time() - start_time
            
            self._save_backup_metadata(metadata)
            
            logger.info(f"✅ Code backup completed: {backup_id}")
            return metadata
            
        except Exception as e:
            logger.exception(f"❌ Code backup failed: {str(e)}")
            metadata.status = BackupStatus.FAILED
            metadata.notes = f"Error: {str(e)}"
            metadata.duration_seconds = time.time() - start_time
            self._save_backup_metadata(metadata)
            raise

    async def create_config_backup(self, tags: List[str] = None, notes: str = "") -> BackupMetadata:
        """Create backup of configuration files"""
        backup_id = f"config-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        logger.info(f"🔄 Creating configuration backup: {backup_id}")
        start_time = time.time()
        
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=BackupType.CONFIG_ONLY,
            timestamp=datetime.datetime.now(),
            status=BackupStatus.IN_PROGRESS,
            source_path=str(self.project_root),
            backup_path=str(self.backup_root / f"{backup_id}.tar.gz"),
            tags=tags or ["config", "settings"],
            notes=notes
        )
        
        try:
            self._save_backup_metadata(metadata)
            
            # Configuration files to backup
            config_patterns = [
                "*.yml", "*.yaml", "*.json", "*.toml", "*.ini", "*.env.example",
                "requirements*.txt", "pyproject.toml", "Dockerfile*",
                "docker-compose*.yml", ".gitignore", "Makefile"
            ]
            
            # Create temporary directory for config files
            temp_config_dir = self.backup_root / f"temp-config-{backup_id}"
            temp_config_dir.mkdir(exist_ok=True)
            
            file_count = 0
            total_size = 0
            
            try:
                # Collect configuration files
                for pattern in config_patterns:
                    for file_path in self.project_root.glob(pattern):
                        if file_path.is_file():
                            relative_path = file_path.relative_to(self.project_root)
                            dest_path = temp_config_dir / relative_path
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(file_path, dest_path)
                            file_count += 1
                            total_size += file_path.stat().st_size
                
                # Include app/config directory if exists
                app_config_dir = self.project_root / "app" / "config"
                if app_config_dir.exists():
                    dest_config_dir = temp_config_dir / "app" / "config"
                    shutil.copytree(app_config_dir, dest_config_dir)
                    for file_path in dest_config_dir.rglob('*'):
                        if file_path.is_file():
                            file_count += 1
                            total_size += file_path.stat().st_size
                
                # Create archive from temp directory
                archive_path = Path(metadata.backup_path)
                _, _ = await self._create_compressed_archive(temp_config_dir, archive_path)
                
                # Calculate metadata
                checksum = self._calculate_file_checksum(archive_path)
                compression_ratio = self._calculate_compression_ratio(total_size, archive_path.stat().st_size)
                
                metadata.status = BackupStatus.COMPLETED
                metadata.file_count = file_count
                metadata.total_size_bytes = total_size
                metadata.checksum = checksum
                metadata.compression_ratio = compression_ratio
                metadata.duration_seconds = time.time() - start_time
                
                self._save_backup_metadata(metadata)
                
                logger.info(f"✅ Configuration backup completed: {backup_id}")
                return metadata
                
            finally:
                # Clean up temp directory
                if temp_config_dir.exists():
                    shutil.rmtree(temp_config_dir)
            
        except Exception as e:
            logger.exception(f"❌ Configuration backup failed: {str(e)}")
            metadata.status = BackupStatus.FAILED
            metadata.notes = f"Error: {str(e)}"
            metadata.duration_seconds = time.time() - start_time
            self._save_backup_metadata(metadata)
            raise

    async def create_incremental_backup(self, base_backup_id: str, tags: List[str] = None, notes: str = "") -> BackupMetadata:
        """Create incremental backup based on previous backup"""
        backup_id = f"incremental-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        logger.info(f"🔄 Creating incremental backup: {backup_id} (base: {base_backup_id})")
        start_time = time.time()
        
        # Get base backup metadata
        base_metadata = self._get_backup_metadata(base_backup_id)
        if not base_metadata:
            raise ValueError(f"Base backup not found: {base_backup_id}")
        
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=BackupType.INCREMENTAL,
            timestamp=datetime.datetime.now(),
            status=BackupStatus.IN_PROGRESS,
            source_path=str(self.project_root),
            backup_path=str(self.backup_root / f"{backup_id}.tar.gz"),
            tags=tags or ["incremental", f"base-{base_backup_id}"],
            notes=f"Incremental from {base_backup_id}. {notes}"
        )
        
        try:
            self._save_backup_metadata(metadata)
            
            # Find files modified since base backup
            modified_files = await self._find_modified_files(base_metadata.timestamp)
            
            if not modified_files:
                logger.info("No modified files found - creating empty incremental backup")
                metadata.status = BackupStatus.COMPLETED
                metadata.file_count = 0
                metadata.total_size_bytes = 0
                metadata.duration_seconds = time.time() - start_time
                self._save_backup_metadata(metadata)
                return metadata
            
            # Create temp directory for modified files
            temp_dir = self.backup_root / f"temp-incremental-{backup_id}"
            temp_dir.mkdir(exist_ok=True)
            
            file_count = 0
            total_size = 0
            
            try:
                # Copy modified files maintaining directory structure
                for file_path in modified_files:
                    src_path = self.project_root / file_path
                    if src_path.exists() and src_path.is_file():
                        dest_path = temp_dir / file_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_path, dest_path)
                        file_count += 1
                        total_size += src_path.stat().st_size
                
                # Create archive
                archive_path = Path(metadata.backup_path)
                _, _ = await self._create_compressed_archive(temp_dir, archive_path)
                
                # Update metadata
                checksum = self._calculate_file_checksum(archive_path)
                compression_ratio = self._calculate_compression_ratio(total_size, archive_path.stat().st_size)
                
                metadata.status = BackupStatus.COMPLETED
                metadata.file_count = file_count
                metadata.total_size_bytes = total_size
                metadata.checksum = checksum
                metadata.compression_ratio = compression_ratio
                metadata.duration_seconds = time.time() - start_time
                
                self._save_backup_metadata(metadata)
                
                logger.info(f"✅ Incremental backup completed: {backup_id}")
                logger.info(f"   Modified files: {file_count}, Size: {total_size / (1024*1024):.2f} MB")
                return metadata
                
            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            
        except Exception as e:
            logger.exception(f"❌ Incremental backup failed: {str(e)}")
            metadata.status = BackupStatus.FAILED
            metadata.notes = f"Error: {str(e)}"
            metadata.duration_seconds = time.time() - start_time
            self._save_backup_metadata(metadata)
            raise

    async def verify_backup(self, backup_id: str) -> Dict:
        """Verify backup integrity"""
        logger.info(f"🔍 Verifying backup: {backup_id}")
        
        metadata = self._get_backup_metadata(backup_id)
        if not metadata:
            return {'valid': False, 'error': f'Backup not found: {backup_id}'}
        
        backup_path = Path(metadata.backup_path)
        if not backup_path.exists():
            return {'valid': False, 'error': f'Backup file not found: {backup_path}'}
        
        try:
            # Verify checksum
            current_checksum = self._calculate_file_checksum(backup_path)
            checksum_valid = current_checksum == metadata.checksum
            
            # Verify archive can be opened
            archive_valid = False
            file_count = 0
            
            try:
                with tarfile.open(backup_path, 'r:gz') as tar:
                    file_count = len(tar.getnames())
                    archive_valid = True
            except Exception as e:
                logger.warning(f"Archive verification failed: {str(e)}")
            
            result = {
                'valid': checksum_valid and archive_valid,
                'checksum_valid': checksum_valid,
                'archive_valid': archive_valid,
                'expected_checksum': metadata.checksum,
                'actual_checksum': current_checksum,
                'expected_file_count': metadata.file_count,
                'actual_file_count': file_count
            }
            
            if result['valid']:
                logger.info(f"✅ Backup verification passed: {backup_id}")
            else:
                logger.warning(f"❌ Backup verification failed: {backup_id}")
            
            return result
            
        except Exception as e:
            logger.exception(f"Backup verification error: {str(e)}")
            return {'valid': False, 'error': str(e)}

    async def restore_backup(self, backup_id: str, options: RestoreOptions) -> Dict:
        """Restore from backup"""
        logger.info(f"🔄 Restoring backup: {backup_id} to {options.target_path}")
        
        metadata = self._get_backup_metadata(backup_id)
        if not metadata:
            return {'success': False, 'error': f'Backup not found: {backup_id}'}
        
        backup_path = Path(metadata.backup_path)
        if not backup_path.exists():
            return {'success': False, 'error': f'Backup file not found: {backup_path}'}
        
        # Verify backup if requested
        if options.verify_checksums:
            verification = await self.verify_backup(backup_id)
            if not verification['valid']:
                return {'success': False, 'error': f'Backup verification failed: {verification}'}
        
        try:
            target_path = Path(options.target_path)
            
            # Create target directory
            if not target_path.exists():
                target_path.mkdir(parents=True, exist_ok=True)
            elif not options.overwrite_existing:
                return {'success': False, 'error': 'Target path exists and overwrite_existing=False'}
            
            # Extract archive
            with tarfile.open(backup_path, 'r:gz') as tar:
                # Filter files based on exclude patterns
                if options.exclude_patterns:
                    members = []
                    for member in tar.getmembers():
                        exclude = False
                        for pattern in options.exclude_patterns:
                            if pattern in member.name:
                                exclude = True
                                break
                        if not exclude:
                            members.append(member)
                else:
                    members = tar.getmembers()
                
                # Extract files
                for member in members:
                    tar.extract(member, target_path)
                    
                    # Restore permissions if requested
                    if options.restore_permissions and member.isfile():
                        extracted_path = target_path / member.name
                        if extracted_path.exists():
                            extracted_path.chmod(member.mode)
            
            result = {
                'success': True,
                'files_restored': len(members) if 'members' in locals() else 0,
                'restore_path': str(target_path)
            }
            
            logger.info(f"✅ Backup restore completed: {backup_id}")
            return result
            
        except Exception as e:
            logger.exception(f"Backup restore failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def list_backups(self, backup_type: BackupType = None, tags: List[str] = None) -> List[BackupMetadata]:
        """List available backups"""
        with sqlite3.connect(self.metadata_db) as conn:
            query = "SELECT * FROM backups WHERE 1=1"
            params = []
            
            if backup_type:
                query += " AND backup_type = ?"
                params.append(backup_type.value)
            
            if tags:
                for tag in tags:
                    query += " AND tags LIKE ?"
                    params.append(f"%{tag}%")
            
            query += " ORDER BY timestamp DESC"
            
            cursor = conn.execute(query, params)
            backups = []
            
            for row in cursor.fetchall():
                backup_data = {
                    'backup_id': row[0],
                    'backup_type': row[1],
                    'timestamp': row[2],
                    'status': row[3],
                    'source_path': row[4],
                    'backup_path': row[5],
                    'file_count': row[6],
                    'total_size_bytes': row[7],
                    'checksum': row[8],
                    'compression_ratio': row[9],
                    'duration_seconds': row[10],
                    'tags': row[11].split(',') if row[11] else [],
                    'notes': row[12] or ''
                }
                backups.append(BackupMetadata.from_dict(backup_data))
            
            return backups

    async def cleanup_old_backups(self, keep_count: int = 10, keep_days: int = 30) -> Dict:
        """Clean up old backups based on retention policy"""
        logger.info(f"🧹 Cleaning up old backups (keep {keep_count} recent, {keep_days} days)")
        
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=keep_days)
        
        with sqlite3.connect(self.metadata_db) as conn:
            # Get backups to delete
            cursor = conn.execute("""
                SELECT backup_id, backup_path, timestamp 
                FROM backups 
                WHERE timestamp < ? 
                AND backup_id NOT IN (
                    SELECT backup_id FROM backups 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                )
                ORDER BY timestamp ASC
            """, (cutoff_date.isoformat(), keep_count))
            
            backups_to_delete = cursor.fetchall()
            
            deleted_count = 0
            deleted_size = 0
            errors = []
            
            for backup_id, backup_path, timestamp in backups_to_delete:
                try:
                    backup_file = Path(backup_path)
                    if backup_file.exists():
                        file_size = backup_file.stat().st_size
                        backup_file.unlink()
                        deleted_size += file_size
                    
                    # Remove from database
                    conn.execute("DELETE FROM backup_files WHERE backup_id = ?", (backup_id,))
                    conn.execute("DELETE FROM backups WHERE backup_id = ?", (backup_id,))
                    deleted_count += 1
                    
                    logger.info(f"Deleted old backup: {backup_id}")
                    
                except Exception as e:
                    error_msg = f"Failed to delete backup {backup_id}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            conn.commit()
        
        result = {
            'deleted_count': deleted_count,
            'deleted_size_mb': deleted_size / (1024 * 1024),
            'errors': errors
        }
        
        logger.info(f"✅ Cleanup completed: deleted {deleted_count} backups ({deleted_size / (1024*1024):.2f} MB)")
        return result

    async def _create_compressed_archive(self, source_path: Path, archive_path: Path, 
                                       exclude_patterns: List[str] = None) -> Tuple[int, int]:
        """Create compressed tar archive"""
        exclude_patterns = exclude_patterns or []
        
        def exclude_filter(tarinfo):
            # Check if file should be excluded
            for pattern in exclude_patterns:
                if pattern in tarinfo.name:
                    return None
            return tarinfo
        
        file_count = 0
        total_size = 0
        
        with tarfile.open(archive_path, 'w:gz') as tar:
            if source_path.is_file():
                tar.add(source_path, arcname=source_path.name, filter=exclude_filter)
                file_count = 1
                total_size = source_path.stat().st_size
            else:
                for file_path in source_path.rglob('*'):
                    if file_path.is_file():
                        # Check exclusion patterns
                        exclude = False
                        relative_path = file_path.relative_to(source_path)
                        for pattern in exclude_patterns:
                            if pattern in str(relative_path) or pattern in file_path.name:
                                exclude = True
                                break
                        
                        if not exclude:
                            tar.add(file_path, arcname=relative_path)
                            file_count += 1
                            total_size += file_path.stat().st_size
        
        return file_count, total_size

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _calculate_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio"""
        if compressed_size == 0:
            return 0.0
        return original_size / compressed_size

    def _save_backup_metadata(self, metadata: BackupMetadata):
        """Save backup metadata to database"""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO backups (
                    backup_id, backup_type, timestamp, status, source_path, 
                    backup_path, file_count, total_size_bytes, checksum,
                    compression_ratio, duration_seconds, tags, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.backup_id,
                metadata.backup_type.value,
                metadata.timestamp.isoformat(),
                metadata.status.value,
                metadata.source_path,
                metadata.backup_path,
                metadata.file_count,
                metadata.total_size_bytes,
                metadata.checksum,
                metadata.compression_ratio,
                metadata.duration_seconds,
                ','.join(metadata.tags),
                metadata.notes
            ))
            conn.commit()

    def _get_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get backup metadata from database"""
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute(
                "SELECT * FROM backups WHERE backup_id = ?", 
                (backup_id,)
            )
            row = cursor.fetchone()
            
            if row:
                backup_data = {
                    'backup_id': row[0],
                    'backup_type': row[1],
                    'timestamp': row[2],
                    'status': row[3],
                    'source_path': row[4],
                    'backup_path': row[5],
                    'file_count': row[6],
                    'total_size_bytes': row[7],
                    'checksum': row[8],
                    'compression_ratio': row[9],
                    'duration_seconds': row[10],
                    'tags': row[11].split(',') if row[11] else [],
                    'notes': row[12] or ''
                }
                return BackupMetadata.from_dict(backup_data)
            
            return None

    async def _find_modified_files(self, since_timestamp: datetime.datetime) -> List[str]:
        """Find files modified since timestamp"""
        modified_files = []
        
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file():
                # Skip backup directory and excluded patterns
                relative_path = file_path.relative_to(self.project_root)
                if 'backups' in str(relative_path):
                    continue
                
                exclude = False
                for pattern in self.default_exclude_patterns:
                    if pattern in str(relative_path) or pattern in file_path.name:
                        exclude = True
                        break
                
                if not exclude:
                    # Check modification time
                    mtime = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime > since_timestamp:
                        modified_files.append(str(relative_path))
        
        return modified_files


async def main():
    """Main backup system CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LeanVibe Agent Hive 2.0 - Comprehensive Backup System")
    parser.add_argument('action', choices=['full', 'code', 'config', 'incremental', 'list', 'verify', 'restore', 'cleanup'])
    parser.add_argument('--backup-id', help='Backup ID for verify/restore operations')
    parser.add_argument('--base-backup', help='Base backup ID for incremental backup')
    parser.add_argument('--target-path', help='Target path for restore operation')
    parser.add_argument('--tags', nargs='*', help='Tags for backup')
    parser.add_argument('--notes', default='', help='Notes for backup')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files during restore')
    parser.add_argument('--keep-count', type=int, default=10, help='Number of backups to keep during cleanup')
    parser.add_argument('--keep-days', type=int, default=30, help='Number of days to keep backups during cleanup')
    
    args = parser.parse_args()
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Initialize backup system
    backup_system = ComprehensiveBackupSystem()
    
    try:
        if args.action == 'full':
            result = await backup_system.create_full_system_backup(args.tags, args.notes)
            print(f"✅ Full backup created: {result.backup_id}")
            
        elif args.action == 'code':
            result = await backup_system.create_code_backup(args.tags, args.notes)
            print(f"✅ Code backup created: {result.backup_id}")
            
        elif args.action == 'config':
            result = await backup_system.create_config_backup(args.tags, args.notes)
            print(f"✅ Configuration backup created: {result.backup_id}")
            
        elif args.action == 'incremental':
            if not args.base_backup:
                print("❌ --base-backup required for incremental backup")
                sys.exit(1)
            result = await backup_system.create_incremental_backup(args.base_backup, args.tags, args.notes)
            print(f"✅ Incremental backup created: {result.backup_id}")
            
        elif args.action == 'list':
            backups = await backup_system.list_backups()
            print(f"\n📋 Available backups ({len(backups)}):")
            for backup in backups:
                print(f"  {backup.backup_id}: {backup.backup_type.value} - {backup.timestamp} - {backup.status.value}")
                print(f"    Size: {backup.total_size_bytes / (1024*1024):.2f} MB, Files: {backup.file_count}")
                if backup.tags:
                    print(f"    Tags: {', '.join(backup.tags)}")
                print()
                
        elif args.action == 'verify':
            if not args.backup_id:
                print("❌ --backup-id required for verify operation")
                sys.exit(1)
            result = await backup_system.verify_backup(args.backup_id)
            if result['valid']:
                print(f"✅ Backup verification passed: {args.backup_id}")
            else:
                print(f"❌ Backup verification failed: {result}")
                
        elif args.action == 'restore':
            if not args.backup_id or not args.target_path:
                print("❌ --backup-id and --target-path required for restore operation")
                sys.exit(1)
            
            options = RestoreOptions(
                target_path=args.target_path,
                overwrite_existing=args.overwrite
            )
            result = await backup_system.restore_backup(args.backup_id, options)
            if result['success']:
                print(f"✅ Backup restore completed: {args.backup_id}")
                print(f"   Files restored: {result['files_restored']}")
            else:
                print(f"❌ Backup restore failed: {result['error']}")
                
        elif args.action == 'cleanup':
            result = await backup_system.cleanup_old_backups(args.keep_count, args.keep_days)
            print(f"✅ Cleanup completed: deleted {result['deleted_count']} backups ({result['deleted_size_mb']:.2f} MB)")
            if result['errors']:
                print(f"⚠️  Errors: {result['errors']}")
        
    except Exception as e:
        logger.exception(f"❌ Operation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class BackupSystemScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            await main()
            
            return {"status": "completed"}
    
    script_main(BackupSystemScript)