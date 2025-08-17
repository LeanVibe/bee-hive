#!/usr/bin/env python3
"""
Install Backward Compatibility Layer for Epic 1 Phase 1.2 Consolidation

This script safely installs the backward compatibility layer that allows
the transition from 338 scattered files to 7 unified managers with zero
breaking changes.

The script:
1. Backs up original files with .backup extension
2. Installs compatibility adapters in their place
3. Provides rollback capability if needed

Usage:
    python install_compatibility_layer.py [--rollback]
"""

import os
import shutil
import argparse
from pathlib import Path

# File mappings: original_file -> compatibility_file
COMPATIBILITY_MAPPINGS = {
    'agent_spawner.py': 'agent_spawner_compat.py',
    'messaging_service.py': 'messaging_service_compat.py', 
    'context_compression.py': 'context_compression_compat.py',
    'workflow_engine.py': 'workflow_engine_compat.py',
    'performance_optimizer.py': 'performance_optimizer_compat.py',
    'security_audit.py': 'security_audit_compat.py',
}

CORE_DIR = Path(__file__).parent / 'app' / 'core'

def backup_original_files():
    """Backup original files with .backup extension."""
    print("Backing up original files...")
    
    backed_up = []
    for original_file in COMPATIBILITY_MAPPINGS.keys():
        original_path = CORE_DIR / original_file
        backup_path = CORE_DIR / f"{original_file}.backup"
        
        if original_path.exists():
            print(f"  Backing up {original_file} -> {original_file}.backup")
            shutil.copy2(original_path, backup_path)
            backed_up.append(original_file)
        else:
            print(f"  Warning: {original_file} not found, skipping backup")
    
    return backed_up

def install_compatibility_files():
    """Install compatibility files in place of originals."""
    print("Installing compatibility layer...")
    
    installed = []
    for original_file, compat_file in COMPATIBILITY_MAPPINGS.items():
        original_path = CORE_DIR / original_file
        compat_path = CORE_DIR / compat_file
        
        if compat_path.exists():
            print(f"  Installing {compat_file} -> {original_file}")
            shutil.copy2(compat_path, original_path)
            installed.append(original_file)
        else:
            print(f"  Error: {compat_file} not found!")
            return False
    
    return True

def rollback_files():
    """Rollback to original files from backups."""
    print("Rolling back to original files...")
    
    for original_file in COMPATIBILITY_MAPPINGS.keys():
        original_path = CORE_DIR / original_file
        backup_path = CORE_DIR / f"{original_file}.backup"
        
        if backup_path.exists():
            print(f"  Restoring {original_file} from backup")
            shutil.copy2(backup_path, original_path)
            # Optionally remove backup
            # backup_path.unlink()
        else:
            print(f"  Warning: No backup found for {original_file}")

def verify_unified_managers():
    """Verify that all unified managers exist."""
    print("Verifying unified managers...")
    
    required_managers = [
        'agent_manager.py',
        'context_manager_unified.py',
        'resource_manager.py', 
        'communication_manager.py',
        'security_manager.py',
        'workflow_manager.py',
        'storage_manager.py',
        'unified_manager_base.py'
    ]
    
    missing = []
    for manager in required_managers:
        manager_path = CORE_DIR / manager
        if not manager_path.exists():
            missing.append(manager)
    
    if missing:
        print(f"  Error: Missing unified managers: {missing}")
        return False
    
    print("  All unified managers found ✓")
    return True

def verify_compatibility_files():
    """Verify that all compatibility files exist."""
    print("Verifying compatibility files...")
    
    # Check main adapter file
    adapter_path = CORE_DIR / '_compatibility_adapters.py'
    if not adapter_path.exists():
        print(f"  Error: Main adapter file not found: {adapter_path}")
        return False
    
    # Check individual compatibility files
    missing = []
    for compat_file in COMPATIBILITY_MAPPINGS.values():
        compat_path = CORE_DIR / compat_file
        if not compat_path.exists():
            missing.append(compat_file)
    
    if missing:
        print(f"  Error: Missing compatibility files: {missing}")
        return False
    
    print("  All compatibility files found ✓")
    return True

def main():
    parser = argparse.ArgumentParser(description="Install backward compatibility layer")
    parser.add_argument('--rollback', action='store_true', 
                       help="Rollback to original files from backups")
    parser.add_argument('--verify', action='store_true',
                       help="Only verify that all files exist")
    
    args = parser.parse_args()
    
    if args.verify:
        print("Verification mode - checking all required files...")
        managers_ok = verify_unified_managers()
        compat_ok = verify_compatibility_files()
        
        if managers_ok and compat_ok:
            print("\n✓ All files verified successfully!")
            return 0
        else:
            print("\n✗ Verification failed!")
            return 1
    
    if args.rollback:
        rollback_files()
        print("\n✓ Rollback completed!")
        return 0
    
    # Standard installation process
    print("Installing Epic 1 Phase 1.2 Backward Compatibility Layer")
    print("=" * 60)
    
    # Verify prerequisites
    if not verify_unified_managers():
        print("\n✗ Installation failed: Missing unified managers")
        return 1
        
    if not verify_compatibility_files():
        print("\n✗ Installation failed: Missing compatibility files")
        return 1
    
    # Backup and install
    backed_up = backup_original_files()
    
    if install_compatibility_files():
        print(f"\n✓ Compatibility layer installed successfully!")
        print(f"   Backed up {len(backed_up)} original files")
        print(f"   Installed {len(COMPATIBILITY_MAPPINGS)} compatibility adapters")
        print("\nTo rollback: python install_compatibility_layer.py --rollback")
        return 0
    else:
        print("\n✗ Installation failed!")
        return 1

if __name__ == '__main__':
    exit(main())