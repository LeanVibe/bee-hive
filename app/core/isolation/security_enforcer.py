"""
Security Enforcer for Resource Monitoring and Constraint Enforcement

This module provides comprehensive security enforcement including resource
limits, process monitoring, and command validation to prevent abuse and
ensure secure operation of isolated worktree environments.
"""

import asyncio
import logging
import os
import signal
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Optional dependency handling
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

# ================================================================================
# Data Models
# ================================================================================

@dataclass
class ResourceLimits:
    """Resource limits for a worktree environment."""
    
    # Disk usage limits
    max_disk_mb: float = 100.0
    max_files: int = 1000
    max_directories: int = 100
    
    # Process limits
    max_cpu_percent: float = 80.0
    max_memory_mb: float = 512.0
    max_execution_time_seconds: int = 3600
    max_processes: int = 5
    
    # Network limits (if applicable)
    block_network_access: bool = True
    allowed_domains: List[str] = field(default_factory=list)
    
    # File operation limits
    max_file_size_mb: float = 10.0
    max_read_operations_per_minute: int = 1000
    max_write_operations_per_minute: int = 500

@dataclass
class ProcessMonitoring:
    """Process monitoring information."""
    
    process_id: int
    command: List[str]
    start_time: datetime
    worktree_path: str
    resource_limits: ResourceLimits
    
    # Current resource usage
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    disk_usage_mb: float = 0.0
    file_count: int = 0
    
    # Violations
    violations: List[str] = field(default_factory=list)
    terminated: bool = False

@dataclass
class SecurityEvent:
    """Security event for monitoring and alerting."""
    
    event_type: str
    severity: str  # "low", "medium", "high", "critical"
    worktree_id: str
    description: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    process_id: Optional[int] = None
    resource_usage: Optional[Dict[str, float]] = None
    action_taken: Optional[str] = None

# ================================================================================
# Security Enforcer
# ================================================================================

class SecurityEnforcer:
    """
    Enforces security constraints and monitors resource usage.
    
    Key Features:
    - Process resource monitoring and limits
    - Command validation and sanitization
    - File operation monitoring
    - Automatic violation response
    - Security event logging
    - Resource usage reporting
    """
    
    def __init__(self):
        """Initialize security enforcer."""
        
        # Configuration
        self._monitoring_interval = 5.0  # seconds
        self._violation_threshold = 3  # strikes before termination
        self._cleanup_interval = 300  # 5 minutes
        
        # State tracking
        self._monitored_processes: Dict[int, ProcessMonitoring] = {}
        self._worktree_monitors: Dict[str, ResourceLimits] = {}
        self._security_events: List[SecurityEvent] = []
        self._blocked_commands: Set[str] = {
            'sudo', 'su', 'rm', 'rmdir', 'del', 'format', 'fdisk',
            'mount', 'umount', 'chmod', 'chown', 'passwd', 'useradd',
            'userdel', 'systemctl', 'service', 'kill', 'killall',
            'wget', 'curl', 'nc', 'netcat', 'telnet', 'ssh', 'scp',
            'rsync', 'ping', 'nmap', 'iptables', 'ufw'
        }
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        
        logger.info("SecurityEnforcer initialized")
    
    # ================================================================================
    # Core Security Methods
    # ================================================================================
    
    def validate_command_safety(self, command_args: List[str]) -> Tuple[bool, str]:
        """
        Validate that a command is safe to execute.
        
        Args:
            command_args: Command and arguments to validate
            
        Returns:
            Tuple[bool, str]: (is_safe, reason_if_unsafe)
        """
        try:
            if not command_args:
                return False, "Empty command"
            
            command = command_args[0].lower()
            
            # Check against blocked commands
            if command in self._blocked_commands:
                return False, f"Command '{command}' is blocked for security"
            
            # Check for shell injection patterns
            full_command = ' '.join(command_args)
            dangerous_patterns = [
                ';', '&&', '||', '|', '`', '$(',
                '../', '/etc/', '/usr/', '/bin/',
                '$(', '${', '>', '>>', '<'
            ]
            
            for pattern in dangerous_patterns:
                if pattern in full_command:
                    return False, f"Dangerous pattern '{pattern}' detected"
            
            # Check for network-related commands
            network_commands = {'wget', 'curl', 'ping', 'nc', 'netcat'}
            if command in network_commands:
                return False, f"Network command '{command}' blocked"
            
            # Validate file paths in arguments
            for arg in command_args[1:]:
                if self._contains_dangerous_path(arg):
                    return False, f"Dangerous path in argument: {arg}"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Command validation error: {e}")
            return False, f"Validation error: {e}"
    
    def enforce_resource_limits(self, process_id: int) -> bool:
        """
        Enforce resource limits for a process.
        
        Args:
            process_id: Process ID to monitor
            
        Returns:
            bool: True if limits are being enforced
        """
        try:
            if process_id not in self._monitored_processes:
                logger.warning(f"Process {process_id} not found in monitoring")
                return False
            
            monitor = self._monitored_processes[process_id]
            limits = monitor.resource_limits
            
            # Get process handle (requires psutil)
            if not PSUTIL_AVAILABLE:
                logger.warning("psutil not available, skipping detailed process monitoring")
                return self._basic_process_check(process_id, limits, monitor)
            
            try:
                process = psutil.Process(process_id)
            except psutil.NoSuchProcess:
                logger.info(f"Process {process_id} no longer exists")
                self._remove_process_monitoring(process_id)
                return True
            
            # Check CPU usage
            cpu_percent = process.cpu_percent()
            if cpu_percent > limits.max_cpu_percent:
                violation = f"CPU usage {cpu_percent:.1f}% exceeds limit {limits.max_cpu_percent}%"
                self._record_violation(process_id, violation)
                return False
            
            # Check memory usage
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            if memory_mb > limits.max_memory_mb:
                violation = f"Memory usage {memory_mb:.1f}MB exceeds limit {limits.max_memory_mb}MB"
                self._record_violation(process_id, violation)
                return False
            
            # Check execution time
            execution_time = (datetime.utcnow() - monitor.start_time).total_seconds()
            if execution_time > limits.max_execution_time_seconds:
                violation = f"Execution time {execution_time:.0f}s exceeds limit {limits.max_execution_time_seconds}s"
                self._record_violation(process_id, violation)
                return False
            
            # Update monitoring data
            monitor.cpu_percent = cpu_percent
            monitor.memory_mb = memory_mb
            
            return True
            
        except Exception as e:
            logger.error(f"Resource limit enforcement error for PID {process_id}: {e}")
            return False
    
    async def setup_monitoring(
        self, 
        worktree_path: str, 
        max_disk_mb: float, 
        max_files: int
    ) -> None:
        """
        Set up monitoring for a worktree.
        
        Args:
            worktree_path: Path to monitor
            max_disk_mb: Maximum disk usage in MB
            max_files: Maximum number of files
        """
        try:
            # Create resource limits
            limits = ResourceLimits(
                max_disk_mb=max_disk_mb,
                max_files=max_files
            )
            
            self._worktree_monitors[worktree_path] = limits
            
            # Start monitoring if not already running
            if not self._is_monitoring:
                await self._start_monitoring()
            
            logger.debug(f"Monitoring setup for worktree: {worktree_path}")
            
        except Exception as e:
            logger.error(f"Monitoring setup failed for {worktree_path}: {e}")
            raise
    
    def monitor_file_operations(self, worktree_path: str) -> Dict[str, Any]:
        """
        Monitor file operations in a worktree.
        
        Args:
            worktree_path: Worktree path to monitor
            
        Returns:
            Dict[str, Any]: File operation statistics
        """
        try:
            if not os.path.exists(worktree_path):
                return {"error": "Worktree path does not exist"}
            
            # Count files and directories
            file_count = 0
            dir_count = 0
            total_size = 0
            
            for root, dirs, files in os.walk(worktree_path):
                dir_count += len(dirs)
                file_count += len(files)
                
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
                    except Exception:
                        continue
            
            disk_usage_mb = total_size / (1024 * 1024)
            
            # Check against limits
            limits = self._worktree_monitors.get(worktree_path)
            violations = []
            
            if limits:
                if file_count > limits.max_files:
                    violations.append(f"File count {file_count} exceeds limit {limits.max_files}")
                
                if disk_usage_mb > limits.max_disk_mb:
                    violations.append(f"Disk usage {disk_usage_mb:.1f}MB exceeds limit {limits.max_disk_mb}MB")
                
                if dir_count > limits.max_directories:
                    violations.append(f"Directory count {dir_count} exceeds limit {limits.max_directories}")
            
            # Record violations
            if violations:
                for violation in violations:
                    self._record_security_event(
                        "file_limit_violation",
                        "medium",
                        worktree_path,
                        violation
                    )
            
            return {
                "file_count": file_count,
                "directory_count": dir_count,
                "disk_usage_mb": disk_usage_mb,
                "violations": violations,
                "limits": limits.__dict__ if limits else None
            }
            
        except Exception as e:
            logger.error(f"File operation monitoring error: {e}")
            return {"error": str(e)}
    
    # ================================================================================
    # Process Monitoring
    # ================================================================================
    
    def start_process_monitoring(
        self, 
        process_id: int, 
        command: List[str],
        worktree_path: str,
        resource_limits: Optional[ResourceLimits] = None
    ) -> None:
        """Start monitoring a process."""
        try:
            if resource_limits is None:
                resource_limits = ResourceLimits()
            
            monitor = ProcessMonitoring(
                process_id=process_id,
                command=command,
                start_time=datetime.utcnow(),
                worktree_path=worktree_path,
                resource_limits=resource_limits
            )
            
            self._monitored_processes[process_id] = monitor
            
            logger.debug(f"Started monitoring process {process_id}: {' '.join(command)}")
            
        except Exception as e:
            logger.error(f"Failed to start process monitoring: {e}")
    
    def stop_process_monitoring(self, process_id: int) -> None:
        """Stop monitoring a process."""
        if process_id in self._monitored_processes:
            del self._monitored_processes[process_id]
            logger.debug(f"Stopped monitoring process {process_id}")
    
    async def _start_monitoring(self) -> None:
        """Start the monitoring task."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._is_monitoring = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Security monitoring started")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        try:
            while self._is_monitoring:
                await self._check_all_processes()
                await self._check_all_worktrees()
                await self._cleanup_old_events()
                await asyncio.sleep(self._monitoring_interval)
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
        finally:
            self._is_monitoring = False
    
    async def _check_all_processes(self) -> None:
        """Check all monitored processes."""
        for process_id in list(self._monitored_processes.keys()):
            try:
                if not self.enforce_resource_limits(process_id):
                    # Handle violation
                    monitor = self._monitored_processes.get(process_id)
                    if monitor and len(monitor.violations) >= self._violation_threshold:
                        await self._terminate_process(process_id)
            except Exception as e:
                logger.error(f"Process check error for PID {process_id}: {e}")
    
    async def _check_all_worktrees(self) -> None:
        """Check all monitored worktrees."""
        for worktree_path in list(self._worktree_monitors.keys()):
            try:
                self.monitor_file_operations(worktree_path)
            except Exception as e:
                logger.error(f"Worktree check error for {worktree_path}: {e}")
    
    async def _terminate_process(self, process_id: int) -> None:
        """Terminate a process due to violations."""
        try:
            monitor = self._monitored_processes.get(process_id)
            if not monitor:
                return
            
            # Try graceful termination first
            try:
                if PSUTIL_AVAILABLE:
                    process = psutil.Process(process_id)
                    process.terminate()
                    
                    # Wait for graceful termination
                    await asyncio.sleep(5)
                    
                    # Force kill if still running
                    if process.is_running():
                        process.kill()
                else:
                    # Fallback to os.kill
                    try:
                        os.kill(process_id, signal.SIGTERM)
                        await asyncio.sleep(5)
                        # Force kill if needed
                        os.kill(process_id, signal.SIGKILL)
                    except OSError:
                        pass  # Process may have already terminated
                    
            except (psutil.NoSuchProcess if PSUTIL_AVAILABLE else OSError):
                pass  # Process already terminated
            
            # Record termination
            self._record_security_event(
                "process_terminated",
                "high",
                monitor.worktree_path,
                f"Process {process_id} terminated due to violations: {monitor.violations}",
                process_id=process_id,
                action_taken="process_termination"
            )
            
            monitor.terminated = True
            logger.warning(f"Terminated process {process_id} due to violations")
            
        except Exception as e:
            logger.error(f"Process termination error: {e}")
    
    def _basic_process_check(
        self, 
        process_id: int, 
        limits: ResourceLimits, 
        monitor: ProcessMonitoring
    ) -> bool:
        """Basic process checking without psutil."""
        try:
            # Check if process still exists using os.kill
            try:
                os.kill(process_id, 0)  # Send signal 0 to check existence
            except OSError:
                logger.info(f"Process {process_id} no longer exists")
                self._remove_process_monitoring(process_id)
                return True
            
            # Check execution time (basic limit we can enforce)
            execution_time = (datetime.utcnow() - monitor.start_time).total_seconds()
            if execution_time > limits.max_execution_time_seconds:
                violation = f"Execution time {execution_time:.0f}s exceeds limit {limits.max_execution_time_seconds}s"
                self._record_violation(process_id, violation)
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Basic process check error: {e}")
            return False
    
    # ================================================================================
    # Helper Methods
    # ================================================================================
    
    def _contains_dangerous_path(self, path: str) -> bool:
        """Check if path contains dangerous elements."""
        dangerous_paths = [
            '/etc/', '/usr/', '/bin/', '/sbin/', '/root/',
            '../', '..\\', '/dev/', '/proc/', '/sys/'
        ]
        
        path_lower = path.lower()
        return any(dangerous in path_lower for dangerous in dangerous_paths)
    
    def _record_violation(self, process_id: int, violation: str) -> None:
        """Record a resource violation."""
        monitor = self._monitored_processes.get(process_id)
        if monitor:
            monitor.violations.append(violation)
            
            self._record_security_event(
                "resource_violation",
                "medium",
                monitor.worktree_path,
                violation,
                process_id=process_id
            )
    
    def _record_security_event(
        self,
        event_type: str,
        severity: str,
        worktree_id: str,
        description: str,
        process_id: Optional[int] = None,
        action_taken: Optional[str] = None
    ) -> None:
        """Record a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            worktree_id=worktree_id,
            description=description,
            process_id=process_id,
            action_taken=action_taken
        )
        
        self._security_events.append(event)
        
        # Log based on severity
        if severity == "critical":
            logger.critical(f"Security event: {description}")
        elif severity == "high":
            logger.error(f"Security event: {description}")
        elif severity == "medium":
            logger.warning(f"Security event: {description}")
        else:
            logger.info(f"Security event: {description}")
    
    def _remove_process_monitoring(self, process_id: int) -> None:
        """Remove process from monitoring."""
        if process_id in self._monitored_processes:
            del self._monitored_processes[process_id]
    
    async def _cleanup_old_events(self) -> None:
        """Clean up old security events."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self._security_events = [
            event for event in self._security_events 
            if event.timestamp > cutoff_time
        ]
    
    # ================================================================================
    # Public Interface
    # ================================================================================
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            "is_monitoring": self._is_monitoring,
            "monitored_processes": len(self._monitored_processes),
            "monitored_worktrees": len(self._worktree_monitors),
            "recent_events": len(self._security_events),
            "monitoring_interval": self._monitoring_interval
        }
    
    def get_process_statistics(self) -> List[Dict[str, Any]]:
        """Get statistics for all monitored processes."""
        stats = []
        for process_id, monitor in self._monitored_processes.items():
            stats.append({
                "process_id": process_id,
                "command": ' '.join(monitor.command),
                "worktree_path": monitor.worktree_path,
                "start_time": monitor.start_time.isoformat(),
                "cpu_percent": monitor.cpu_percent,
                "memory_mb": monitor.memory_mb,
                "violations": len(monitor.violations),
                "terminated": monitor.terminated
            })
        return stats
    
    def get_security_events(
        self, 
        severity: Optional[str] = None,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get recent security events."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        events = [
            event for event in self._security_events
            if event.timestamp > cutoff_time
        ]
        
        if severity:
            events = [event for event in events if event.severity == severity]
        
        return [
            {
                "event_type": event.event_type,
                "severity": event.severity,
                "worktree_id": event.worktree_id,
                "description": event.description,
                "timestamp": event.timestamp.isoformat(),
                "process_id": event.process_id,
                "action_taken": event.action_taken
            }
            for event in sorted(events, key=lambda x: x.timestamp, reverse=True)
        ]
    
    async def shutdown(self) -> None:
        """Shutdown the security enforcer."""
        self._is_monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Terminate any remaining monitored processes
        for process_id in list(self._monitored_processes.keys()):
            try:
                if PSUTIL_AVAILABLE:
                    process = psutil.Process(process_id)
                    if process.is_running():
                        process.terminate()
                else:
                    # Fallback to os.kill
                    try:
                        os.kill(process_id, signal.SIGTERM)
                    except OSError:
                        pass
            except Exception:
                pass
        
        logger.info("SecurityEnforcer shutdown completed")