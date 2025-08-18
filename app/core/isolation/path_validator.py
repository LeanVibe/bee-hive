"""
Path Validator for Secure File System Access

This module provides comprehensive path validation and sanitization to prevent
security vulnerabilities including path traversal attacks, symlink escapes,
and unauthorized access to system directories.
"""

import os
import logging
import re
from pathlib import Path
from typing import List, Set, Optional, Tuple
from urllib.parse import unquote

logger = logging.getLogger(__name__)

# ================================================================================
# Path Validator
# ================================================================================

class PathValidator:
    """
    Validates file system paths for security and prevents unauthorized access.
    
    Key Security Features:
    - Path traversal attack prevention (../../../etc/passwd)
    - Symlink escape protection
    - System directory access blocking
    - File extension validation
    - URL decode attack prevention
    - Null byte injection protection
    """
    
    def __init__(self):
        """Initialize path validator with security configuration."""
        
        # Restricted system directories (never allow access)
        self._restricted_paths = {
            "/etc", "/usr", "/bin", "/sbin", "/root", "/sys", "/proc",
            "/boot", "/dev", "/lib", "/lib64", "/opt/system", "/var/lib",
            "/run", "/tmp/systemd", "/tmp/dbus", "/home/.ssh"
        }
        
        # Dangerous path patterns
        self._dangerous_patterns = [
            r'\.\./',           # Path traversal
            r'\.\.\/',          # Windows-style path traversal  
            r'\.\.\\',          # Windows backslash traversal
            r'%2e%2e%2f',      # URL encoded path traversal
            r'%2e%2e%5c',      # URL encoded Windows traversal
            r'\x00',           # Null byte injection
            r'\/\/+',          # Multiple slashes
            r'~root',          # Root home directory
            r'~0',             # Root user reference
        ]
        
        # Compile patterns for performance
        self._compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self._dangerous_patterns]
        
        # Allowed file extensions (whitelist approach)
        self._allowed_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx',           # Code files
            '.md', '.txt', '.rst', '.adoc',                # Documentation
            '.json', '.yaml', '.yml', '.toml', '.ini',     # Configuration
            '.csv', '.xml', '.html', '.css',               # Data/Web files
            '.sh', '.bat', '.ps1',                         # Scripts (with caution)
            '.sql', '.graphql',                            # Query files
            '.dockerfile', '.containerfile',                # Container files
            '.gitignore', '.gitattributes',                # Git files
            '.log',                                        # Log files (read-only)
        }
        
        # Maximum path length (security limit)
        self._max_path_length = 4096
        
        # Maximum filename length
        self._max_filename_length = 255
        
        logger.debug("PathValidator initialized with security constraints")
    
    # ================================================================================
    # Core Validation Methods
    # ================================================================================
    
    def validate_file_access(self, worktree_path: str, file_path: str) -> bool:
        """
        Validate that file access is safe within the worktree.
        
        Args:
            worktree_path: Base worktree directory path
            file_path: Relative file path to validate
            
        Returns:
            bool: True if access is safe, False otherwise
        """
        try:
            # 1. Basic input validation
            if not self._validate_basic_constraints(file_path):
                return False
            
            # 2. Sanitize and normalize paths
            sanitized_file_path = self.sanitize_path(file_path)
            if not sanitized_file_path:
                return False
            
            # 3. Resolve absolute paths
            abs_worktree = os.path.abspath(worktree_path)
            abs_file_path = os.path.abspath(os.path.join(abs_worktree, sanitized_file_path))
            
            # 4. Check path containment (prevent directory traversal)
            if not self._is_path_contained(abs_worktree, abs_file_path):
                logger.warning(f"Path traversal attempt detected: {file_path}")
                return False
            
            # 5. Check against restricted system paths
            if not self.check_security_constraints(abs_file_path):
                return False
            
            # 6. Validate file extension
            if not self._validate_file_extension(sanitized_file_path):
                logger.warning(f"Unauthorized file extension: {sanitized_file_path}")
                return False
            
            # 7. Check for symlink escapes
            if not self._validate_symlink_safety(abs_file_path, abs_worktree):
                logger.warning(f"Symlink escape attempt detected: {file_path}")
                return False
            
            logger.debug(f"File access validated: {sanitized_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Path validation error: {e}")
            return False
    
    def sanitize_path(self, path: str) -> Optional[str]:
        """
        Sanitize and normalize a file path.
        
        Args:
            path: Raw file path to sanitize
            
        Returns:
            Optional[str]: Sanitized path or None if invalid
        """
        try:
            if not path:
                return None
            
            # 1. URL decode to handle encoded attacks
            decoded_path = unquote(path)
            
            # 2. Check for dangerous patterns after decoding
            for pattern in self._compiled_patterns:
                if pattern.search(decoded_path):
                    logger.warning(f"Dangerous pattern detected in path: {path}")
                    return None
            
            # 3. Normalize path separators
            normalized = decoded_path.replace('\\', '/')
            
            # 4. Remove leading slashes (force relative paths)
            normalized = normalized.lstrip('/')
            
            # 5. Normalize path (resolve . and .. components safely)
            # Only allow .. if it doesn't escape the worktree
            parts = []
            for part in normalized.split('/'):
                if part == '' or part == '.':
                    continue
                elif part == '..':
                    if parts:  # Only go up if we have directories to go up from
                        parts.pop()
                    # Otherwise ignore the .. (prevents escape)
                else:
                    parts.append(part)
            
            if not parts:
                return "."  # Current directory
            
            # 6. Rejoin path
            sanitized = '/'.join(parts)
            
            # 7. Final validation
            if len(sanitized) > self._max_path_length:
                logger.warning(f"Path too long: {len(sanitized)} chars")
                return None
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Path sanitization error: {e}")
            return None
    
    def check_security_constraints(self, path: str) -> bool:
        """
        Check if path violates security constraints.
        
        Args:
            path: Absolute path to check
            
        Returns:
            bool: True if path is safe
        """
        try:
            # 1. Resolve real path (follows symlinks)
            try:
                real_path = os.path.realpath(path)
            except Exception:
                # If we can't resolve the path, be conservative
                real_path = path
            
            # 2. Check against restricted system directories
            for restricted in self._restricted_paths:
                if real_path.startswith(restricted):
                    logger.warning(f"Access to restricted path blocked: {real_path}")
                    return False
            
            # 3. Additional system-specific checks
            if self._is_system_critical_path(real_path):
                logger.warning(f"Access to system critical path blocked: {real_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Security constraint check error: {e}")
            return False
    
    # ================================================================================
    # Helper Methods
    # ================================================================================
    
    def _validate_basic_constraints(self, path: str) -> bool:
        """Validate basic path constraints."""
        if not path:
            return False
        
        # Check length limits
        if len(path) > self._max_path_length:
            return False
        
        # Check for null bytes
        if '\x00' in path:
            logger.warning("Null byte injection attempt detected")
            return False
        
        # Check filename length
        filename = os.path.basename(path)
        if len(filename) > self._max_filename_length:
            return False
        
        return True
    
    def _is_path_contained(self, base_path: str, target_path: str) -> bool:
        """Check if target path is contained within base path."""
        try:
            # Normalize both paths
            base_real = os.path.realpath(base_path)
            target_real = os.path.realpath(target_path)
            
            # Check if target starts with base
            return target_real.startswith(base_real + os.sep) or target_real == base_real
            
        except Exception:
            return False
    
    def _validate_file_extension(self, path: str) -> bool:
        """Validate file extension against whitelist."""
        if not path or path == ".":
            return True  # Allow directory access
        
        # Extract extension
        _, ext = os.path.splitext(path.lower())
        
        # Allow files without extensions (directories, etc.)
        if not ext:
            return True
        
        # Check against whitelist
        return ext in self._allowed_extensions
    
    def _validate_symlink_safety(self, target_path: str, base_path: str) -> bool:
        """Validate that symlinks don't escape the base directory."""
        try:
            # If file doesn't exist yet, it's safe
            if not os.path.exists(target_path):
                return True
            
            # If it's not a symlink, it's safe
            if not os.path.islink(target_path):
                return True
            
            # Resolve the symlink target
            link_target = os.readlink(target_path)
            
            # If relative, resolve relative to the symlink's directory
            if not os.path.isabs(link_target):
                link_dir = os.path.dirname(target_path)
                resolved_target = os.path.abspath(os.path.join(link_dir, link_target))
            else:
                resolved_target = os.path.abspath(link_target)
            
            # Check if symlink target is contained within base path
            return self._is_path_contained(base_path, resolved_target)
            
        except Exception as e:
            logger.error(f"Symlink validation error: {e}")
            return False
    
    def _is_system_critical_path(self, path: str) -> bool:
        """Check if path is system-critical."""
        # Additional system-specific critical paths
        critical_patterns = [
            r'/etc/passwd',
            r'/etc/shadow',
            r'/etc/hosts',
            r'/proc/self/environ',
            r'/var/run/secrets',
            r'\.ssh/',
            r'\.aws/',
            r'\.kube/',
            r'password',
            r'secret',
            r'\.key$',
            r'\.pem$',
        ]
        
        path_lower = path.lower()
        for pattern in critical_patterns:
            if re.search(pattern, path_lower):
                return True
        
        return False
    
    # ================================================================================
    # Utility Methods
    # ================================================================================
    
    def get_safe_filename(self, filename: str) -> str:
        """
        Generate a safe filename by removing dangerous characters.
        
        Args:
            filename: Original filename
            
        Returns:
            str: Safe filename
        """
        if not filename:
            return "unnamed"
        
        # Remove dangerous characters
        safe = re.sub(r'[^\w\-_\.]', '_', filename)
        
        # Limit length
        if len(safe) > self._max_filename_length:
            name, ext = os.path.splitext(safe)
            max_name_len = self._max_filename_length - len(ext)
            safe = name[:max_name_len] + ext
        
        # Ensure it doesn't start with dot (hidden files)
        if safe.startswith('.'):
            safe = 'f' + safe
        
        return safe
    
    def get_relative_path(self, base_path: str, target_path: str) -> Optional[str]:
        """
        Get safe relative path from base to target.
        
        Args:
            base_path: Base directory path
            target_path: Target file path
            
        Returns:
            Optional[str]: Relative path or None if unsafe
        """
        try:
            base_abs = os.path.abspath(base_path)
            target_abs = os.path.abspath(target_path)
            
            # Check containment
            if not self._is_path_contained(base_abs, target_abs):
                return None
            
            # Calculate relative path
            return os.path.relpath(target_abs, base_abs)
            
        except Exception:
            return None
    
    def validate_directory_creation(self, base_path: str, dir_path: str) -> bool:
        """
        Validate that directory creation is safe.
        
        Args:
            base_path: Base directory
            dir_path: Directory to create
            
        Returns:
            bool: True if safe to create
        """
        return self.validate_file_access(base_path, dir_path)
    
    def get_validation_report(self, path: str) -> dict:
        """
        Get detailed validation report for a path.
        
        Args:
            path: Path to analyze
            
        Returns:
            dict: Validation report with details
        """
        report = {
            "path": path,
            "is_valid": False,
            "issues": [],
            "sanitized_path": None,
            "security_level": "unknown"
        }
        
        try:
            # Basic validation
            if not self._validate_basic_constraints(path):
                report["issues"].append("Basic constraint violation")
                return report
            
            # Sanitization
            sanitized = self.sanitize_path(path)
            if not sanitized:
                report["issues"].append("Path sanitization failed")
                return report
            
            report["sanitized_path"] = sanitized
            
            # Pattern check
            for pattern in self._compiled_patterns:
                if pattern.search(path):
                    report["issues"].append(f"Dangerous pattern detected")
                    break
            
            # Extension check
            if not self._validate_file_extension(path):
                report["issues"].append("File extension not allowed")
            
            # Overall assessment
            if not report["issues"]:
                report["is_valid"] = True
                report["security_level"] = "safe"
            else:
                report["security_level"] = "dangerous"
            
        except Exception as e:
            report["issues"].append(f"Validation error: {e}")
        
        return report