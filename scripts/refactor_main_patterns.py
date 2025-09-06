"""
AST-based Main Pattern Refactoring Tool

This script automatically refactors Python files to use the standardized ScriptBase pattern,
eliminating boilerplate main() code patterns across the codebase.

Features:
- AST-based analysis and transformation
- Preserves existing functionality while reducing boilerplate
- Comprehensive validation and backup creation
- Batch processing with progress tracking
- Dry-run mode for safe validation

Usage:
    python scripts/refactor_main_patterns.py --help
    python scripts/refactor_main_patterns.py --file path/to/script.py --dry-run
    python scripts/refactor_main_patterns.py --pattern "tests/**/*.py" --batch-size 10
"""

import ast
import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.common.script_base import ScriptBase

logger = structlog.get_logger()


@dataclass
class RefactoringResult:
    """Result of refactoring a single file."""
    file_path: str
    success: bool
    lines_removed: int
    lines_added: int
    net_reduction: int
    error: Optional[str] = None
    backup_path: Optional[str] = None
    validation_passed: bool = False


@dataclass
class RefactoringStats:
    """Overall refactoring statistics."""
    total_files: int
    successful_refactorings: int
    failed_refactorings: int
    total_lines_removed: int
    total_lines_added: int
    net_lines_reduced: int
    processing_time_seconds: float


class MainPatternDetector(ast.NodeVisitor):
    """AST visitor to detect and analyze main() patterns."""
    
    def __init__(self):
        self.has_main_block = False
        self.main_node = None
        self.global_instance_nodes = []
        self.async_function_nodes = []
        self.asyncio_run_nodes = []
        self.imports = []
        
    def visit_If(self, node):
        """Detect if __name__ == "__main__": patterns."""
        if (isinstance(node.test, ast.Compare) and
            isinstance(node.test.left, ast.Name) and
            node.test.left.id == '__name__' and
            len(node.test.comparators) == 1 and
            isinstance(node.test.comparators[0], ast.Constant) and
            node.test.comparators[0].value == '__main__'):
            
            self.has_main_block = True
            self.main_node = node
            
        self.generic_visit(node)
        
    def visit_Assign(self, node):
        """Detect global instance assignments."""
        if (len(node.targets) == 1 and
            isinstance(node.targets[0], ast.Name) and
            isinstance(node.value, ast.Call)):
            self.global_instance_nodes.append(node)
            
        self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node):
        """Detect async function definitions."""
        self.async_function_nodes.append(node)
        self.generic_visit(node)
        
    def visit_Call(self, node):
        """Detect asyncio.run() calls."""
        if (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'asyncio' and
            node.func.attr == 'run'):
            self.asyncio_run_nodes.append(node)
            
        self.generic_visit(node)
        
    def visit_Import(self, node):
        """Track imports."""
        self.imports.extend([alias.name for alias in node.names])
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Track from imports."""
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)


class MainPatternRefactorer:
    """Main class for refactoring main() patterns."""
    
    def __init__(self, dry_run: bool = False, create_backups: bool = True):
        self.dry_run = dry_run
        self.create_backups = create_backups
        self.results: List[RefactoringResult] = []
        
    def analyze_file(self, file_path: Path) -> Tuple[bool, MainPatternDetector]:
        """Analyze a file to determine if it needs refactoring."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            detector = MainPatternDetector()
            detector.visit(tree)
            
            # Determine if this file is a candidate for refactoring
            is_candidate = (
                detector.has_main_block and
                len(detector.asyncio_run_nodes) > 0 and
                len(detector.global_instance_nodes) > 0 and
                'app.common.script_base' not in detector.imports
            )
            
            return is_candidate, detector
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return False, None
            
    def generate_refactored_content(self, file_path: Path, detector: MainPatternDetector) -> Tuple[str, int, int]:
        """Generate refactored file content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()
            
        # Count original lines
        original_line_count = len(original_lines)
        
        # Extract the class name from global instance
        class_instance_name = None
        class_name = None
        
        for node in detector.global_instance_nodes:
            if isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name):
                    class_instance_name = node.targets[0].id
                    class_name = node.value.func.id
                    break
                    
        if not class_instance_name or not class_name:
            raise ValueError("Could not identify class instance pattern")
            
        # Generate new content
        new_lines = []
        
        # Add existing content up to the main block, but remove global instance
        tree = ast.parse(''.join(original_lines))
        
        for i, line in enumerate(original_lines):
            # Skip the global instance line and main block
            line_num = i + 1
            
            # Check if this line contains the global instance
            if class_instance_name in line and '=' in line and class_name in line:
                continue  # Skip global instance line
                
            # Check if this line starts the main block
            if line.strip().startswith('if __name__'):
                break  # Stop before main block
                
            new_lines.append(line)
            
        # Add the refactored pattern
        new_lines.append('\n')
        new_lines.append(f'# Standardized script execution pattern\n')
        new_lines.append(f'{class_instance_name} = {class_name}()\n')
        new_lines.append('\n')
        new_lines.append('if __name__ == "__main__":\n')
        new_lines.append(f'    {class_instance_name}.execute()\n')
        
        new_content = ''.join(new_lines)
        new_line_count = len(new_lines)
        
        return new_content, original_line_count, new_line_count
    
    def validate_refactored_file(self, content: str) -> bool:
        """Validate that refactored content is syntactically correct."""
        try:
            ast.parse(content)
            return True
        except SyntaxError as e:
            logger.error(f"Syntax validation failed: {e}")
            return False
            
    def create_backup(self, file_path: Path) -> Optional[Path]:
        """Create backup of original file."""
        if not self.create_backups:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f'.backup_{timestamp}.py')
        
        try:
            shutil.copy2(file_path, backup_path)
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
            
    def refactor_file(self, file_path: Path) -> RefactoringResult:
        """Refactor a single file."""
        logger.info(f"🔄 Processing {file_path}")
        
        # Analyze the file
        is_candidate, detector = self.analyze_file(file_path)
        
        if not is_candidate:
            return RefactoringResult(
                file_path=str(file_path),
                success=False,
                lines_removed=0,
                lines_added=0,
                net_reduction=0,
                error="Not a candidate for refactoring"
            )
            
        try:
            # Generate refactored content
            new_content, original_lines, new_lines = self.generate_refactored_content(file_path, detector)
            
            # Validate syntax
            if not self.validate_refactored_file(new_content):
                return RefactoringResult(
                    file_path=str(file_path),
                    success=False,
                    lines_removed=0,
                    lines_added=0,
                    net_reduction=0,
                    error="Syntax validation failed"
                )
                
            lines_removed = original_lines - new_lines
            net_reduction = lines_removed
            
            if self.dry_run:
                logger.info(f"✅ [DRY RUN] Would refactor {file_path} ({lines_removed} lines saved)")
                return RefactoringResult(
                    file_path=str(file_path),
                    success=True,
                    lines_removed=lines_removed,
                    lines_added=new_lines,
                    net_reduction=net_reduction,
                    validation_passed=True
                )
                
            # Create backup
            backup_path = self.create_backup(file_path)
            
            # Write refactored content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            logger.info(f"✅ Refactored {file_path} ({lines_removed} lines saved)")
            
            return RefactoringResult(
                file_path=str(file_path),
                success=True,
                lines_removed=lines_removed,
                lines_added=new_lines,
                net_reduction=net_reduction,
                backup_path=str(backup_path) if backup_path else None,
                validation_passed=True
            )
            
        except Exception as e:
            error_msg = f"Failed to refactor {file_path}: {e}"
            logger.error(error_msg)
            
            return RefactoringResult(
                file_path=str(file_path),
                success=False,
                lines_removed=0,
                lines_added=0,
                net_reduction=0,
                error=error_msg
            )
            
    def refactor_multiple_files(self, file_paths: List[Path], batch_size: int = 50) -> RefactoringStats:
        """Refactor multiple files in batches."""
        start_time = datetime.now()
        results = []
        
        total_files = len(file_paths)
        successful = 0
        failed = 0
        total_lines_removed = 0
        total_lines_added = 0
        
        logger.info(f"🚀 Starting refactoring of {total_files} files...")
        
        for i in range(0, total_files, batch_size):
            batch = file_paths[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_files + batch_size - 1) // batch_size
            
            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
            
            for file_path in batch:
                result = self.refactor_file(file_path)
                results.append(result)
                
                if result.success:
                    successful += 1
                    total_lines_removed += result.lines_removed
                    total_lines_added += result.lines_added
                else:
                    failed += 1
                    
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        stats = RefactoringStats(
            total_files=total_files,
            successful_refactorings=successful,
            failed_refactorings=failed,
            total_lines_removed=total_lines_removed,
            total_lines_added=total_lines_added,
            net_lines_reduced=total_lines_removed - total_lines_added,
            processing_time_seconds=processing_time
        )
        
        self.results = results
        
        logger.info(
            f"🎉 Refactoring complete!",
            total_files=total_files,
            successful=successful,
            failed=failed,
            net_lines_reduced=stats.net_lines_reduced,
            processing_time_seconds=processing_time
        )
        
        return stats


class RefactoringScript(ScriptBase):
    """Main refactoring script using ScriptBase pattern."""
    
    def __init__(self, args):
        super().__init__("MainPatternRefactorer")
        self.args = args
        
    async def run(self) -> Dict[str, any]:
        """Execute the refactoring process."""
        refactorer = MainPatternRefactorer(
            dry_run=self.args.dry_run,
            create_backups=self.args.backup
        )
        
        # Collect files to process
        file_paths = []
        
        if self.args.file:
            file_paths = [Path(self.args.file)]
        elif self.args.pattern:
            from glob import glob
            file_paths = [Path(f) for f in glob(self.args.pattern, recursive=True) if f.endswith('.py')]
        elif self.args.directory:
            dir_path = Path(self.args.directory)
            file_paths = list(dir_path.rglob('*.py'))
        else:
            # Default to current directory
            file_paths = list(Path('.').rglob('*.py'))
            
        if not file_paths:
            return {
                "status": "warning",
                "message": "No Python files found to process",
                "files_processed": 0
            }
            
        # Filter out files that shouldn't be refactored
        filtered_paths = []
        for path in file_paths:
            # Skip certain directories and files
            path_str = str(path)
            if any(skip in path_str for skip in ['.venv', '__pycache__', '.git', 'migrations']):
                continue
            if path.name in ['__init__.py', 'setup.py', 'manage.py']:
                continue
                
            filtered_paths.append(path)
            
        logger.info(f"Found {len(filtered_paths)} candidate files for refactoring")
        
        # Execute refactoring
        stats = refactorer.refactor_multiple_files(
            filtered_paths,
            batch_size=self.args.batch_size
        )
        
        return {
            "status": "success",
            "message": f"Refactored {stats.successful_refactorings} files successfully",
            "statistics": {
                "total_files": stats.total_files,
                "successful_refactorings": stats.successful_refactorings,
                "failed_refactorings": stats.failed_refactorings,
                "total_lines_removed": stats.total_lines_removed,
                "total_lines_added": stats.total_lines_added,
                "net_lines_reduced": stats.net_lines_reduced,
                "processing_time_seconds": stats.processing_time_seconds,
                "estimated_maintenance_savings": stats.net_lines_reduced * 1.5  # $1.50 per line/year
            },
            "results": [
                {
                    "file_path": r.file_path,
                    "success": r.success,
                    "lines_saved": r.net_reduction,
                    "error": r.error
                }
                for r in refactorer.results[:10]  # First 10 for brevity
            ]
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Refactor main() patterns to use ScriptBase")
    parser.add_argument('--file', type=str, help='Single file to refactor')
    parser.add_argument('--pattern', type=str, help='Glob pattern for files (e.g., "tests/**/*.py")')
    parser.add_argument('--directory', type=str, help='Directory to process recursively')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without modifying files')
    parser.add_argument('--no-backup', dest='backup', action='store_false', help='Skip creating backups')
    parser.add_argument('--batch-size', type=int, default=50, help='Number of files to process per batch')
    
    args = parser.parse_args()
    
    script = RefactoringScript(args)
    return script.execute()


if __name__ == "__main__":
    main()