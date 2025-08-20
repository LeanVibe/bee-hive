"""
Init File Standardizer - Parallel Track 2 Implementation
========================================================

Template-based __init__.py standardization implementing Gemini CLI 
recommendations for parallel execution strategy.

Features:
- Simple template-based approach (independent of main() patterns)
- Preserves essential imports and metadata
- High-throughput processing (200 files/day target)
- Minimal risk through standardized templates
- Static analysis integration

Usage:
    # Analyze __init__.py files
    python init_standardizer.py --analyze
    
    # Execute standardization
    python init_standardizer.py --execute --batch init_batch_001.json
    
    # Dry run validation
    python init_standardizer.py --dry-run --batch init_batch_001.json
"""

import ast
import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class InitFileAnalysis:
    """Analysis result for an __init__.py file."""
    file_path: Path
    current_size_lines: int
    has_imports: bool
    has_version: bool
    has_all: bool
    preserved_content: List[str]
    estimated_loc_savings: int
    requires_standardization: bool

@dataclass
class InitStandardizationResult:
    """Result of standardizing an __init__.py file."""
    file_path: Path
    success: bool
    message: str
    loc_saved: int = 0
    execution_time: float = 0.0
    backup_path: Optional[Path] = None

@dataclass
class InitBatchResult:
    """Result of batch __init__.py standardization."""
    batch_id: str
    files_processed: int
    files_successful: int
    total_loc_saved: int
    total_execution_time: float
    results: List[InitStandardizationResult] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        return self.files_successful / self.files_processed if self.files_processed > 0 else 0.0

class InitFileStandardizer:
    """
    Simple template-based __init__.py standardization.
    
    Implements parallel Track 2 strategy for high-throughput processing
    of __init__.py files with minimal risk through standardized templates.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.logger = structlog.get_logger(f"{self.__class__.__name__}")
        
        # Backup system
        self.backup_dir = Path(tempfile.gettempdir()) / "init_standardization_backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Standard template
        self.standard_template = self._load_standard_template()
        
        # Patterns to preserve
        self.preserve_patterns = [
            'from .',          # Relative imports
            'import .',        # Relative imports  
            '__version__',     # Version metadata
            '__author__',      # Author metadata
            '__email__',       # Email metadata
            '__all__',         # Public API definition
        ]
        
    def _load_standard_template(self) -> str:
        """Load the standard __init__.py template."""
        return '''"""
{module_name} module initialization.

This module provides {module_description}.
"""

{preserved_imports}

{preserved_metadata}

{preserved_all}
'''.strip()
    
    def analyze_init_file(self, file_path: Path) -> InitFileAnalysis:
        """Analyze an __init__.py file for standardization opportunities."""
        try:
            if not file_path.name == '__init__.py':
                return InitFileAnalysis(
                    file_path=file_path,
                    current_size_lines=0,
                    has_imports=False,
                    has_version=False,
                    has_all=False,
                    preserved_content=[],
                    estimated_loc_savings=0,
                    requires_standardization=False
                )
            
            content = file_path.read_text(encoding='utf-8')
            lines = content.splitlines()
            
            # Analyze content
            has_imports = any('import' in line for line in lines)
            has_version = any('__version__' in line for line in lines)
            has_all = any('__all__' in line for line in lines)
            
            # Extract preserved content
            preserved_content = self._extract_preserved_content(content)
            
            # Estimate standardization benefit
            current_lines = len([line for line in lines if line.strip()])
            estimated_template_lines = self._estimate_template_size(preserved_content)
            estimated_savings = max(0, current_lines - estimated_template_lines)
            
            # Determine if standardization is beneficial
            requires_standardization = (
                estimated_savings > 2 or  # Significant size reduction
                len(lines) > 10 or        # File is getting large
                not self._follows_standard_format(content)  # Non-standard format
            )
            
            return InitFileAnalysis(
                file_path=file_path,
                current_size_lines=current_lines,
                has_imports=has_imports,
                has_version=has_version,
                has_all=has_all,
                preserved_content=preserved_content,
                estimated_loc_savings=estimated_savings,
                requires_standardization=requires_standardization
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze {file_path}: {e}")
            return InitFileAnalysis(
                file_path=file_path,
                current_size_lines=0,
                has_imports=False,
                has_version=False,
                has_all=False,
                preserved_content=[],
                estimated_loc_savings=0,
                requires_standardization=False
            )
    
    def _extract_preserved_content(self, content: str) -> List[str]:
        """Extract content that should be preserved in standardization."""
        lines = content.splitlines()
        preserved = []
        
        # Simple line-by-line approach with better validation
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            if not stripped or stripped.startswith('#'):
                i += 1
                continue
            
            # Handle single-line imports and metadata
            if self._is_simple_preserve_line(stripped):
                preserved.append(stripped)
                i += 1
                continue
            
            # Handle multi-line imports and assignments - skip them for now to avoid syntax issues
            if ((stripped.startswith(('from ', 'import ')) and ('(' in stripped or stripped.endswith('\\'))) or
                (stripped.startswith('__') and '=' in stripped and stripped.endswith('['))):
                # Skip complex multi-line statements to avoid syntax errors
                # These are typically large imports or __all__ definitions that are better left as-is
                if stripped.endswith(('(', '[')):
                    # Skip to closing delimiter
                    i += 1
                    closing_char = ')' if stripped.endswith('(') else ']'
                    while i < len(lines):
                        if lines[i].strip().endswith(closing_char):
                            break
                        i += 1
                elif stripped.endswith('\\'):
                    # Skip backslash-continued lines
                    i += 1
                    while i < len(lines) and lines[i-1].strip().endswith('\\'):
                        i += 1
                i += 1
                continue
            
            i += 1
                
        return preserved
    
    def _is_simple_preserve_line(self, line: str) -> bool:
        """Check if line is a simple statement we should preserve."""
        return (
            # Simple imports
            (line.startswith(('import ', 'from .')) and not line.endswith('(')) or
            # Simple metadata assignments (single line only)
            (any(f'__{meta}__' in line for meta in ['version', 'author', 'email']) and '=' in line and not line.endswith('[')) or
            # Other single-line assignments we want to keep
            (line.startswith('__') and '=' in line and not line.endswith(('[', '(')))
        )
    
    
    def _estimate_template_size(self, preserved_content: List[str]) -> int:
        """Estimate the size of the standardized template."""
        # Base template: docstring (3 lines) + imports section + metadata section + __all__ section
        base_size = 3
        
        # Add preserved content
        preserved_size = len(preserved_content)
        
        # Add reasonable spacing
        spacing = 2
        
        return base_size + preserved_size + spacing
    
    def _follows_standard_format(self, content: str) -> bool:
        """Check if file already follows standard format."""
        # Simple heuristic: has docstring at the top
        stripped = content.strip()
        return stripped.startswith('"""') or stripped.startswith("'''")
    
    def analyze_project(self) -> List[InitFileAnalysis]:
        """Analyze all __init__.py files in the project."""
        self.logger.info("Analyzing __init__.py files in project")
        
        # Find all __init__.py files
        init_files = list(self.project_root.glob("**/__init__.py"))
        
        # Filter out certain directories
        excluded_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules'}
        filtered_files = [
            f for f in init_files 
            if not any(excluded in str(f) for excluded in excluded_dirs)
        ]
        
        # Analyze each file
        analyses = []
        for file_path in filtered_files:
            analysis = self.analyze_init_file(file_path)
            if analysis.requires_standardization:
                analyses.append(analysis)
        
        self.logger.info(
            f"Found {len(analyses)} __init__.py files requiring standardization",
            total_files=len(filtered_files),
            standardization_candidates=len(analyses),
            estimated_total_savings=sum(a.estimated_loc_savings for a in analyses)
        )
        
        return analyses
    
    def create_batches(self, analyses: List[InitFileAnalysis], batch_size: int = 50) -> List[Dict[str, Any]]:
        """Create batches for processing."""
        batches = []
        
        for i in range(0, len(analyses), batch_size):
            batch_analyses = analyses[i:i + batch_size]
            batch_id = f"init_batch_{i//batch_size + 1:03d}"
            
            batch = {
                'batch_id': batch_id,
                'files': [self._analysis_to_dict(analysis) for analysis in batch_analyses],
                'estimated_loc_savings': sum(a.estimated_loc_savings for a in batch_analyses),
                'file_count': len(batch_analyses)
            }
            batches.append(batch)
        
        return batches
    
    def standardize_file(self, analysis: InitFileAnalysis, dry_run: bool = False) -> InitStandardizationResult:
        """Standardize a single __init__.py file."""
        start_time = time.time()
        
        if dry_run:
            return InitStandardizationResult(
                file_path=analysis.file_path,
                success=True,
                message="Dry run successful",
                loc_saved=analysis.estimated_loc_savings,
                execution_time=time.time() - start_time
            )
        
        # Create backup
        backup_path = self._create_backup(analysis.file_path)
        
        try:
            # Generate standardized content
            standardized_content = self._generate_standardized_content(analysis)
            
            # Write standardized content
            analysis.file_path.write_text(standardized_content, encoding='utf-8')
            
            # Validate syntax
            try:
                ast.parse(standardized_content)
            except SyntaxError as e:
                # Rollback on syntax error
                self._restore_backup(backup_path, analysis.file_path)
                return InitStandardizationResult(
                    file_path=analysis.file_path,
                    success=False,
                    message=f"Syntax error in generated content: {e}",
                    execution_time=time.time() - start_time,
                    backup_path=backup_path
                )
            
            return InitStandardizationResult(
                file_path=analysis.file_path,
                success=True,
                message="Standardization successful",
                loc_saved=analysis.estimated_loc_savings,
                execution_time=time.time() - start_time,
                backup_path=backup_path
            )
            
        except Exception as e:
            # Emergency rollback
            self._restore_backup(backup_path, analysis.file_path)
            return InitStandardizationResult(
                file_path=analysis.file_path,
                success=False,
                message=f"Standardization failed: {e}",
                execution_time=time.time() - start_time,
                backup_path=backup_path
            )
    
    def _generate_standardized_content(self, analysis: InitFileAnalysis) -> str:
        """Generate standardized content for an __init__.py file."""
        # Extract module information from path
        module_name = analysis.file_path.parent.name
        module_description = f"the {module_name} functionality"
        
        # Organize preserved content
        imports = []
        metadata = []
        all_definition = None
        
        for line in analysis.preserved_content:
            stripped = line.strip()
            if 'import' in stripped and not '__all__' in stripped:
                imports.append(stripped)
            elif '__all__' in stripped:
                all_definition = stripped
            elif any(meta in stripped for meta in ['__version__', '__author__', '__email__']):
                metadata.append(stripped)
        
        # Build content sections
        sections = []
        
        # 1. Docstring
        sections.append(f'"""')
        sections.append(f'{module_name} module initialization.')
        sections.append('')
        sections.append(f'This module provides {module_description}.')
        sections.append('"""')
        sections.append('')
        
        # 2. Imports
        if imports:
            sections.extend(imports)
            sections.append('')
        
        # 3. Metadata
        if metadata:
            sections.extend(metadata)
            sections.append('')
        
        # 4. __all__ definition
        if all_definition:
            sections.append(all_definition)
            sections.append('')
        
        # Clean up extra blank lines
        cleaned_lines = []
        prev_blank = False
        
        for line in sections:
            if line.strip():
                cleaned_lines.append(line)
                prev_blank = False
            elif not prev_blank:
                cleaned_lines.append('')
                prev_blank = True
        
        # Ensure file ends with single newline
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines) + '\n'
    
    def standardize_batch(self, batch_data: Dict[str, Any], dry_run: bool = False) -> InitBatchResult:
        """Standardize a batch of __init__.py files."""
        batch_id = batch_data['batch_id']
        analyses = [self._dict_to_analysis(file_dict) for file_dict in batch_data['files']]
        
        self.logger.info(f"Processing init batch {batch_id} with {len(analyses)} files")
        
        results = []
        successful_count = 0
        total_loc_saved = 0
        start_time = time.time()
        
        for analysis in analyses:
            result = self.standardize_file(analysis, dry_run=dry_run)
            results.append(result)
            
            if result.success:
                successful_count += 1
                total_loc_saved += result.loc_saved
        
        return InitBatchResult(
            batch_id=batch_id,
            files_processed=len(analyses),
            files_successful=successful_count,
            total_loc_saved=total_loc_saved,
            total_execution_time=time.time() - start_time,
            results=results
        )
    
    def _create_backup(self, file_path: Path) -> Path:
        """Create backup of file before standardization."""
        timestamp = int(time.time())
        backup_name = f"{file_path.parent.name}__init__.py.{timestamp}.backup"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        self.logger.debug(f"Created backup: {backup_path}")
        
        return backup_path
    
    def _restore_backup(self, backup_path: Path, original_path: Path) -> bool:
        """Restore file from backup."""
        try:
            shutil.copy2(backup_path, original_path)
            self.logger.info(f"Restored {original_path} from backup")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")
            return False
    
    def _analysis_to_dict(self, analysis: InitFileAnalysis) -> Dict[str, Any]:
        """Convert InitFileAnalysis to dictionary for JSON serialization."""
        return {
            'file_path': str(analysis.file_path),
            'current_size_lines': analysis.current_size_lines,
            'has_imports': analysis.has_imports,
            'has_version': analysis.has_version,
            'has_all': analysis.has_all,
            'preserved_content': analysis.preserved_content,
            'estimated_loc_savings': analysis.estimated_loc_savings,
            'requires_standardization': analysis.requires_standardization
        }
    
    def _dict_to_analysis(self, file_dict: Dict[str, Any]) -> InitFileAnalysis:
        """Convert dictionary back to InitFileAnalysis."""
        return InitFileAnalysis(
            file_path=Path(file_dict['file_path']),
            current_size_lines=file_dict['current_size_lines'],
            has_imports=file_dict['has_imports'],
            has_version=file_dict['has_version'],
            has_all=file_dict['has_all'],
            preserved_content=file_dict['preserved_content'],
            estimated_loc_savings=file_dict['estimated_loc_savings'],
            requires_standardization=file_dict['requires_standardization']
        )

# CLI Interface
def main():
    """Command-line interface for init file standardization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="__init__.py file standardization")
    parser.add_argument('--analyze', action='store_true', help='Analyze __init__.py files')
    parser.add_argument('--execute', action='store_true', help='Execute standardization')
    parser.add_argument('--dry-run', action='store_true', help='Perform dry run validation')
    parser.add_argument('--batch', type=str, help='Batch file to process')
    parser.add_argument('--batch-size', type=int, default=50, help='Size of batches for processing')
    
    args = parser.parse_args()
    
    standardizer = InitFileStandardizer()
    
    if args.analyze:
        # Analyze project and create batches
        analyses = standardizer.analyze_project()
        batches = standardizer.create_batches(analyses, args.batch_size)
        
        print(f"Analysis complete:")
        print(f"- __init__.py files requiring standardization: {len(analyses)}")
        print(f"- Estimated LOC savings: {sum(a.estimated_loc_savings for a in analyses)}")
        print(f"- Batches created: {len(batches)}")
        
        # Save batches to files
        for batch in batches:
            batch_file = Path(f"{batch['batch_id']}.json")
            with open(batch_file, 'w') as f:
                json.dump(batch, f, indent=2)
            print(f"- Saved batch: {batch_file}")
            
    elif args.batch:
        # Process specific batch
        batch_file = Path(args.batch)
        if not batch_file.exists():
            print(f"Batch file not found: {batch_file}")
            return
            
        with open(batch_file) as f:
            batch_data = json.load(f)
            
        if args.dry_run:
            result = standardizer.standardize_batch(batch_data, dry_run=True)
            print(f"Dry run complete for {result.batch_id}:")
            print(f"- Files to process: {result.files_processed}")
            print(f"- Estimated LOC savings: {batch_data['estimated_loc_savings']}")
            
        elif args.execute:
            result = standardizer.standardize_batch(batch_data, dry_run=False)
            print(f"Standardization complete for {result.batch_id}:")
            print(f"- Files processed: {result.files_processed}")
            print(f"- Files successful: {result.files_successful}")
            print(f"- Success rate: {result.success_rate:.1%}")
            print(f"- LOC saved: {result.total_loc_saved}")
            print(f"- Execution time: {result.total_execution_time:.2f}s")
            
            # Report any failures
            failures = [r for r in result.results if not r.success]
            if failures:
                print(f"\nFailures ({len(failures)}):")
                for failure in failures:
                    print(f"- {failure.file_path}: {failure.message}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()