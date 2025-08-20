#!/usr/bin/env python3
"""
Obsolete File Analyzer
======================

Phase 8: Final cleanup phase - identifies files that are no longer needed
and can be safely removed after our comprehensive technical debt remediation.

Detection Categories:
- Duplicate files (exact content matches)
- Superseded files (replaced by consolidated versions)  
- Dead code files (no import references)
- Temporary/backup files (safe cleanup patterns)
- Empty or near-empty files
- Unused configuration files
- Deprecated scripts and tools
- Old test files with no coverage
"""

import ast
import os
import re
import hashlib
import subprocess
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import tempfile
import shutil
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class ObsoleteFile:
    """Represents a file identified as potentially obsolete."""
    file_path: Path
    reason: str
    confidence: float  # 0.0 to 1.0
    size_bytes: int
    last_modified: datetime
    references_found: List[str]
    replacement_suggestion: Optional[str] = None
    safety_notes: List[str] = None

class ObsoleteFileAnalyzer:
    """Identifies files that can be safely removed from the project."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.analysis_results_dir = self.project_root / "analysis" / "obsolete_files"
        self.backup_dir = Path(tempfile.mkdtemp(prefix="obsolete_backups_"))
        
        # Files to skip (critical system files)
        self.critical_files = {
            'requirements.txt', 'pyproject.toml', 'setup.py', 'setup.cfg',
            'Dockerfile', 'docker-compose.yml', '.gitignore', '.gitattributes',
            'README.md', 'LICENSE', 'MANIFEST.in', '__init__.py'
        }
        
        # Patterns for temporary/backup files
        self.temp_patterns = [
            r'\.tmp$', r'\.temp$', r'\.bak$', r'\.backup$', r'\.old$',
            r'~$', r'\.swp$', r'\.swo$', r'\.orig$', r'\.rej$',
            r'\.DS_Store$', r'Thumbs\.db$', r'\.pyc$', r'\.pyo$',
            r'__pycache__', r'\.egg-info', r'\.coverage$', r'\.pytest_cache'
        ]
        
        # Patterns for generated/cache files
        self.generated_patterns = [
            r'\.log$', r'\.cache$', r'\.pid$', r'\.lock$',
            r'node_modules/', r'\.venv/', r'venv/', r'\.git/',
            r'build/', r'dist/', r'\.tox/', r'htmlcov/'
        ]
        
        # File extensions to analyze
        self.analyzable_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala'
        }
        
        # Store analysis results
        self.file_references = defaultdict(set)
        self.file_metadata = {}
        self.duplicate_groups = []
        self.obsolete_candidates = []
    
    def analyze_project_files(self) -> Dict[str, List[ObsoleteFile]]:
        """Main analysis function to identify obsolete files."""
        print("🗑️ Obsolete File Analyzer - Phase 8: Dead Code Elimination")
        print("=" * 60)
        
        # Step 1: Build comprehensive file metadata
        print("📊 Building file metadata and reference map...")
        self.build_file_metadata()
        
        # Step 2: Analyze file references and dependencies
        print("🔍 Analyzing file references and import dependencies...")
        self.analyze_file_references()
        
        # Step 3: Find duplicate files
        print("🔄 Identifying duplicate files...")
        self.find_duplicate_files()
        
        # Step 4: Identify dead code files
        print("💀 Identifying dead code and unreferenced files...")
        dead_files = self.identify_dead_code_files()
        
        # Step 5: Find temporary and backup files
        print("🧹 Finding temporary and backup files...")
        temp_files = self.find_temporary_files()
        
        # Step 6: Identify empty or minimal files
        print("📭 Identifying empty or minimal content files...")
        empty_files = self.find_empty_files()
        
        # Step 7: Find superseded files from our consolidation
        print("🔄 Identifying files superseded by consolidation...")
        superseded_files = self.find_superseded_files()
        
        # Step 8: Analyze unused configuration files
        print("⚙️ Analyzing unused configuration files...")
        unused_configs = self.find_unused_configs()
        
        # Compile results
        results = {
            'duplicate_files': self.categorize_duplicates(),
            'dead_code_files': dead_files,
            'temporary_files': temp_files,
            'empty_files': empty_files,
            'superseded_files': superseded_files,
            'unused_configs': unused_configs
        }
        
        # Calculate totals and generate report
        self.generate_comprehensive_report(results)
        
        return results
    
    def build_file_metadata(self) -> None:
        """Build comprehensive metadata for all project files."""
        all_files = []
        
        # Get all files, excluding common ignore patterns
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                path_str = str(file_path)
                if not any(pattern in path_str for pattern in ['.git/', '__pycache__/', '.venv/', 'node_modules/']):
                    all_files.append(file_path)
        
        print(f"   📄 Analyzing {len(all_files)} files...")
        
        for file_path in all_files:
            try:
                stat = file_path.stat()
                self.file_metadata[str(file_path)] = {
                    'path': file_path,
                    'size': stat.st_size,
                    'last_modified': datetime.fromtimestamp(stat.st_mtime),
                    'extension': file_path.suffix,
                    'name': file_path.name,
                    'content_hash': self.calculate_file_hash(file_path) if stat.st_size < 10_000_000 else None  # Skip huge files
                }
            except (OSError, PermissionError):
                continue
        
        print(f"   ✅ Built metadata for {len(self.file_metadata)} files")
    
    def calculate_file_hash(self, file_path: Path) -> Optional[str]:
        """Calculate MD5 hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except (OSError, PermissionError, UnicodeDecodeError):
            return None
    
    def analyze_file_references(self) -> None:
        """Analyze which files are referenced by others."""
        python_files = [path for path, meta in self.file_metadata.items() 
                       if meta['extension'] == '.py']
        
        print(f"   🐍 Analyzing imports in {len(python_files)} Python files...")
        
        for file_path_str in python_files:
            file_path = Path(file_path_str)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find all imports and references
                imports = self.extract_imports(content)
                file_references = self.extract_file_references(content)
                
                for imported_module in imports:
                    # Convert module imports to potential file paths
                    potential_files = self.module_to_file_paths(imported_module, file_path)
                    for potential_file in potential_files:
                        if potential_file in self.file_metadata:
                            self.file_references[potential_file].add(file_path_str)
                
                for referenced_file in file_references:
                    if referenced_file in self.file_metadata:
                        self.file_references[referenced_file].add(file_path_str)
                        
            except (OSError, UnicodeDecodeError):
                continue
        
        referenced_count = len(self.file_references)
        print(f"   ✅ Found {referenced_count} files with references")
    
    def extract_imports(self, content: str) -> List[str]:
        """Extract import statements from Python code."""
        imports = []
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                        for alias in node.names:
                            imports.append(f"{node.module}.{alias.name}")
        except SyntaxError:
            # Fallback to regex for files with syntax errors
            import_patterns = [
                r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',
                r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
            ]
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                imports.extend(matches)
        
        return imports
    
    def extract_file_references(self, content: str) -> List[str]:
        """Extract file path references from content."""
        references = []
        
        # Look for file path patterns
        file_patterns = [
            r'["\']([^"\']*\.py)["\']',  # Python file references
            r'["\']([^"\']*\.json)["\']',  # JSON file references
            r'["\']([^"\']*\.yaml?)["\']',  # YAML file references
            r'["\']([^"\']*\.md)["\']',  # Markdown file references
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # Convert relative paths to absolute
                potential_path = self.project_root / match
                if potential_path.exists():
                    references.append(str(potential_path))
        
        return references
    
    def module_to_file_paths(self, module_name: str, current_file: Path) -> List[str]:
        """Convert module import to potential file paths."""
        potential_paths = []
        
        # Handle relative imports
        if module_name.startswith('.'):
            current_dir = current_file.parent
            relative_path = module_name.lstrip('.')
            if relative_path:
                potential_file = current_dir / f"{relative_path.replace('.', '/')}.py"
            else:
                potential_file = current_dir / "__init__.py"
            
            if potential_file.exists():
                potential_paths.append(str(potential_file))
        
        # Handle absolute imports from project
        else:
            # Try to find in project structure
            module_parts = module_name.split('.')
            
            # Try different root directories
            for root in [self.project_root, self.project_root / "app", self.project_root / "src"]:
                potential_file = root / f"{'/'.join(module_parts)}.py"
                if potential_file.exists():
                    potential_paths.append(str(potential_file))
                
                # Also check for __init__.py in package
                potential_package = root / '/'.join(module_parts) / "__init__.py"
                if potential_package.exists():
                    potential_paths.append(str(potential_package))
        
        return potential_paths
    
    def find_duplicate_files(self) -> None:
        """Find files with identical content."""
        hash_to_files = defaultdict(list)
        
        for file_path_str, metadata in self.file_metadata.items():
            content_hash = metadata.get('content_hash')
            if content_hash and metadata['size'] > 0:  # Skip empty files
                hash_to_files[content_hash].append(file_path_str)
        
        # Find groups with duplicates
        for content_hash, file_paths in hash_to_files.items():
            if len(file_paths) > 1:
                # Sort by importance (keep the most "canonical" file)
                sorted_files = self.sort_files_by_importance(file_paths)
                self.duplicate_groups.append({
                    'hash': content_hash,
                    'files': file_paths,
                    'keep_file': sorted_files[0],
                    'remove_files': sorted_files[1:]
                })
        
        print(f"   🔄 Found {len(self.duplicate_groups)} duplicate file groups")
    
    def sort_files_by_importance(self, file_paths: List[str]) -> List[str]:
        """Sort files by importance to determine which to keep."""
        def importance_score(file_path: str) -> int:
            path = Path(file_path)
            score = 0
            
            # Prefer files in main directories
            if 'app/' in file_path or 'src/' in file_path:
                score += 100
            
            # Prefer files with more references
            score += len(self.file_references.get(file_path, set())) * 10
            
            # Prefer shorter paths (closer to root)
            score -= len(path.parts)
            
            # Prefer files without "test" in path
            if 'test' not in file_path.lower():
                score += 50
            
            # Prefer files without backup/temp indicators
            if not any(pattern in path.name.lower() for pattern in ['backup', 'old', 'temp', 'copy']):
                score += 25
            
            return score
        
        return sorted(file_paths, key=importance_score, reverse=True)
    
    def identify_dead_code_files(self) -> List[ObsoleteFile]:
        """Identify files that appear to be dead code."""
        dead_files = []
        
        for file_path_str, metadata in self.file_metadata.items():
            file_path = metadata['path']
            
            # Skip critical files
            if file_path.name in self.critical_files:
                continue
            
            # Skip non-analyzable files
            if metadata['extension'] not in self.analyzable_extensions:
                continue
            
            # Check if file has no references
            references = self.file_references.get(file_path_str, set())
            
            # Additional checks for dead code
            is_potentially_dead = (
                len(references) == 0 and  # No references found
                not self.is_entry_point(file_path) and  # Not an entry point
                not self.is_test_file(file_path) and  # Not a test file (tests might not be imported)
                not self.has_main_block(file_path) and  # Not a script with __main__
                metadata['size'] > 0  # Not empty
            )
            
            if is_potentially_dead:
                confidence = 0.7  # Base confidence
                
                # Increase confidence if very old and small
                days_old = (datetime.now() - metadata['last_modified']).days
                if days_old > 180 and metadata['size'] < 1000:
                    confidence = 0.9
                
                dead_files.append(ObsoleteFile(
                    file_path=file_path,
                    reason="No references found - appears to be dead code",
                    confidence=confidence,
                    size_bytes=metadata['size'],
                    last_modified=metadata['last_modified'],
                    references_found=list(references),
                    safety_notes=[
                        "Verify not used in configuration files",
                        "Check for dynamic imports not detected by static analysis",
                        "Confirm not referenced in documentation"
                    ]
                ))
        
        print(f"   💀 Found {len(dead_files)} potential dead code files")
        return dead_files
    
    def is_entry_point(self, file_path: Path) -> bool:
        """Check if file appears to be an entry point."""
        entry_point_names = {
            'main.py', 'run.py', 'app.py', 'server.py', 'cli.py',
            'manage.py', 'start.py', 'init.py'
        }
        return file_path.name in entry_point_names or 'main' in file_path.name.lower()
    
    def is_test_file(self, file_path: Path) -> bool:
        """Check if file appears to be a test file."""
        test_indicators = ['test_', '_test', 'tests/', 'test/', 'spec_', '_spec']
        file_path_str = str(file_path).lower()
        return any(indicator in file_path_str for indicator in test_indicators)
    
    def has_main_block(self, file_path: Path) -> bool:
        """Check if file has if __name__ == '__main__' block."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return 'if __name__ == "__main__"' in content or "if __name__ == '__main__'" in content
        except:
            return False
    
    def find_temporary_files(self) -> List[ObsoleteFile]:
        """Find temporary and backup files that can be safely removed."""
        temp_files = []
        
        for file_path_str, metadata in self.file_metadata.items():
            file_path = metadata['path']
            
            # Check against temporary file patterns
            for pattern in self.temp_patterns:
                if re.search(pattern, str(file_path)):
                    temp_files.append(ObsoleteFile(
                        file_path=file_path,
                        reason=f"Temporary/backup file matching pattern: {pattern}",
                        confidence=0.95,
                        size_bytes=metadata['size'],
                        last_modified=metadata['last_modified'],
                        references_found=[],
                        safety_notes=["Verify not a critical backup file"]
                    ))
                    break
            
            # Check against generated file patterns
            for pattern in self.generated_patterns:
                if re.search(pattern, str(file_path)):
                    temp_files.append(ObsoleteFile(
                        file_path=file_path,
                        reason=f"Generated/cache file matching pattern: {pattern}",
                        confidence=0.9,
                        size_bytes=metadata['size'],
                        last_modified=metadata['last_modified'],
                        references_found=[],
                        safety_notes=["Can be regenerated"]
                    ))
                    break
        
        print(f"   🧹 Found {len(temp_files)} temporary/generated files")
        return temp_files
    
    def find_empty_files(self) -> List[ObsoleteFile]:
        """Find empty or near-empty files."""
        empty_files = []
        
        for file_path_str, metadata in self.file_metadata.items():
            file_path = metadata['path']
            
            # Skip __init__.py files (they can legitimately be empty)
            if file_path.name == '__init__.py':
                continue
            
            if metadata['size'] == 0:
                empty_files.append(ObsoleteFile(
                    file_path=file_path,
                    reason="File is empty (0 bytes)",
                    confidence=0.8,
                    size_bytes=0,
                    last_modified=metadata['last_modified'],
                    references_found=list(self.file_references.get(file_path_str, set())),
                    safety_notes=["Verify not a placeholder file"]
                ))
            elif metadata['size'] < 50 and metadata['extension'] in self.analyzable_extensions:
                # Check if file has only whitespace or comments
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        
                    # Remove comments and check if anything remains
                    content_no_comments = re.sub(r'#.*', '', content)  # Python comments
                    content_no_comments = re.sub(r'//.*', '', content_no_comments)  # JS comments
                    content_no_comments = content_no_comments.strip()
                    
                    if len(content_no_comments) < 10:  # Very minimal content
                        empty_files.append(ObsoleteFile(
                            file_path=file_path,
                            reason="File contains minimal content (mostly comments/whitespace)",
                            confidence=0.6,
                            size_bytes=metadata['size'],
                            last_modified=metadata['last_modified'],
                            references_found=list(self.file_references.get(file_path_str, set())),
                            safety_notes=["May be a template or placeholder"]
                        ))
                        
                except (OSError, UnicodeDecodeError):
                    continue
        
        print(f"   📭 Found {len(empty_files)} empty or minimal files")
        return empty_files
    
    def find_superseded_files(self) -> List[ObsoleteFile]:
        """Find files that have been superseded by our consolidation efforts."""
        superseded_files = []
        
        # Check if our consolidated modules exist
        consolidated_dirs = [
            self.project_root / "app" / "common" / "utilities",
            self.project_root / "app" / "common" / "patterns",
            self.project_root / "docs" / "consolidated",
            self.project_root / "config" / "consolidated"
        ]
        
        for consolidated_dir in consolidated_dirs:
            if consolidated_dir.exists():
                print(f"   🔄 Checking for files superseded by {consolidated_dir}")
                
                # Read consolidated files to understand what they replace
                for consolidated_file in consolidated_dir.glob("*"):
                    if consolidated_file.is_file():
                        superseded = self.find_files_superseded_by(consolidated_file)
                        superseded_files.extend(superseded)
        
        print(f"   🔄 Found {len(superseded_files)} potentially superseded files")
        return superseded_files
    
    def find_files_superseded_by(self, consolidated_file: Path) -> List[ObsoleteFile]:
        """Find files that are superseded by a specific consolidated file."""
        superseded = []
        
        try:
            with open(consolidated_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for references to original files in consolidated documentation
            original_file_patterns = [
                r'Original implementations found in:\s*\n((?:\s*-\s*[^\n]+\n?)+)',
                r'- `([^`]+\.py)`',
                r'Location.*`([^`]+)`',
                r'from\s+(\d+)\s+similar implementations',
            ]
            
            for pattern in original_file_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                for match in matches:
                    if isinstance(match, str) and '.py' in match:
                        # Extract file path
                        potential_path = Path(match.strip().split(':')[0])
                        if potential_path.exists() and str(potential_path) in self.file_metadata:
                            superseded.append(ObsoleteFile(
                                file_path=potential_path,
                                reason=f"Superseded by consolidated module: {consolidated_file.name}",
                                confidence=0.4,  # Low confidence, needs manual review
                                size_bytes=self.file_metadata[str(potential_path)]['size'],
                                last_modified=self.file_metadata[str(potential_path)]['last_modified'],
                                references_found=list(self.file_references.get(str(potential_path), set())),
                                replacement_suggestion=str(consolidated_file),
                                safety_notes=[
                                    "Manual review required",
                                    "Ensure all functionality is preserved in consolidated version",
                                    "Update all imports to use consolidated module"
                                ]
                            ))
        except (OSError, UnicodeDecodeError):
            pass
        
        return superseded
    
    def find_unused_configs(self) -> List[ObsoleteFile]:
        """Find configuration files that appear to be unused."""
        unused_configs = []
        
        config_extensions = {'.json', '.yaml', '.yml', '.ini', '.cfg', '.conf', '.env'}
        
        for file_path_str, metadata in self.file_metadata.items():
            if metadata['extension'] in config_extensions:
                file_path = metadata['path']
                
                # Skip if referenced
                if len(self.file_references.get(file_path_str, set())) > 0:
                    continue
                
                # Additional checks for config files
                is_likely_unused = (
                    not self.is_standard_config(file_path) and
                    not self.is_environment_config(file_path) and
                    self.is_old_file(metadata['last_modified'])
                )
                
                if is_likely_unused:
                    unused_configs.append(ObsoleteFile(
                        file_path=file_path,
                        reason="Configuration file with no detected references",
                        confidence=0.5,  # Lower confidence for config files
                        size_bytes=metadata['size'],
                        last_modified=metadata['last_modified'],
                        references_found=[],
                        safety_notes=[
                            "May be used by deployment scripts",
                            "Check environment-specific usage",
                            "Verify not used by CI/CD pipelines"
                        ]
                    ))
        
        print(f"   ⚙️ Found {len(unused_configs)} potentially unused config files")
        return unused_configs
    
    def is_standard_config(self, file_path: Path) -> bool:
        """Check if this is a standard configuration file."""
        standard_configs = {
            'config.json', 'config.yaml', 'config.yml', 'settings.json',
            'package.json', 'tsconfig.json', '.env', 'requirements.txt'
        }
        return file_path.name in standard_configs
    
    def is_environment_config(self, file_path: Path) -> bool:
        """Check if this is an environment-specific config."""
        env_indicators = ['prod', 'dev', 'test', 'staging', 'local']
        file_name_lower = file_path.name.lower()
        return any(env in file_name_lower for env in env_indicators)
    
    def is_old_file(self, last_modified: datetime) -> bool:
        """Check if file is old (> 6 months without modification)."""
        return datetime.now() - last_modified > timedelta(days=180)
    
    def categorize_duplicates(self) -> List[ObsoleteFile]:
        """Categorize duplicate files for removal."""
        duplicate_files = []
        
        for group in self.duplicate_groups:
            for file_path_str in group['remove_files']:
                metadata = self.file_metadata[file_path_str]
                
                duplicate_files.append(ObsoleteFile(
                    file_path=metadata['path'],
                    reason=f"Duplicate of {group['keep_file']}",
                    confidence=0.95,
                    size_bytes=metadata['size'],
                    last_modified=metadata['last_modified'],
                    references_found=list(self.file_references.get(file_path_str, set())),
                    replacement_suggestion=group['keep_file'],
                    safety_notes=["Verify references point to kept file"]
                ))
        
        return duplicate_files
    
    def generate_comprehensive_report(self, results: Dict[str, List[ObsoleteFile]]) -> None:
        """Generate comprehensive analysis report."""
        self.analysis_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate summary statistics
        total_files = sum(len(files) for files in results.values())
        total_size = sum(sum(f.size_bytes for f in files) for files in results.values())
        high_confidence_files = sum(1 for files in results.values() for f in files if f.confidence >= 0.8)
        
        print(f"\n📊 Obsolete File Analysis Results:")
        print(f"   🗑️ {total_files} potentially obsolete files identified")
        print(f"   📦 {total_size:,} bytes ({total_size/1024/1024:.1f} MB) potential cleanup")
        print(f"   ✅ {high_confidence_files} high-confidence removals (>80%)")
        
        print(f"\n🔍 Breakdown by Category:")
        for category, files in results.items():
            if files:
                category_size = sum(f.size_bytes for f in files)
                avg_confidence = sum(f.confidence for f in files) / len(files)
                print(f"   • {category.replace('_', ' ').title()}: {len(files)} files, "
                      f"{category_size:,} bytes, {avg_confidence:.1%} avg confidence")
        
        # Generate detailed JSON report
        json_report = {
            'analysis_date': datetime.now().isoformat(),
            'summary': {
                'total_files': total_files,
                'total_size_bytes': total_size,
                'high_confidence_files': high_confidence_files,
                'categories': {category: len(files) for category, files in results.items()}
            },
            'detailed_results': {}
        }
        
        for category, files in results.items():
            json_report['detailed_results'][category] = [
                {
                    'file_path': str(f.file_path),
                    'reason': f.reason,
                    'confidence': f.confidence,
                    'size_bytes': f.size_bytes,
                    'last_modified': f.last_modified.isoformat(),
                    'references_found': f.references_found,
                    'replacement_suggestion': f.replacement_suggestion,
                    'safety_notes': f.safety_notes or []
                }
                for f in files
            ]
        
        report_path = self.analysis_results_dir / "obsolete_files_report.json"
        with open(report_path, 'w') as f:
            json.dump(json_report, f, indent=2, sort_keys=True)
        
        print(f"\n📋 Detailed report saved to: {report_path}")
        
        # Generate removal script for high-confidence files
        self.generate_removal_script(results)
    
    def generate_removal_script(self, results: Dict[str, List[ObsoleteFile]]) -> None:
        """Generate safe removal script for high-confidence files."""
        script_path = self.analysis_results_dir / "safe_removal_script.py"
        
        script_content = f'''#!/usr/bin/env python3
"""
Safe File Removal Script
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This script removes files identified as obsolete with high confidence (>80%).
IMPORTANT: Review the list before running and create backups if needed.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def create_backup(file_path, backup_dir):
    """Create backup of file before deletion."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / file_path.name
    shutil.copy2(file_path, backup_path)
    return backup_path

def main():
    project_root = Path("{self.project_root}")
    backup_dir = project_root / "backups" / f"removed_files_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}"
    
    # High-confidence files for removal
    files_to_remove = [
'''
        
        # Add high-confidence files
        high_conf_files = []
        for category, files in results.items():
            high_conf_files.extend([f for f in files if f.confidence >= 0.8])
        
        for file_obj in high_conf_files:
            script_content += f'''        {{
            "path": "{file_obj.file_path}",
            "reason": "{file_obj.reason}",
            "confidence": {file_obj.confidence:.2f},
            "size": {file_obj.size_bytes}
        }},
'''
        
        script_content += '''    ]
    
    print(f"🗑️ Safe File Removal - {len(files_to_remove)} files identified")
    print("=" * 60)
    
    if not files_to_remove:
        print("✅ No high-confidence files to remove")
        return
    
    total_size = sum(f["size"] for f in files_to_remove)
    print(f"📦 Total size to be freed: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    
    # Ask for confirmation
    response = input("\\nProceed with removal? (type 'yes' to continue): ")
    if response.lower() != 'yes':
        print("❌ Removal cancelled")
        return
    
    # Create backups and remove files
    removed_count = 0
    errors = []
    
    for file_info in files_to_remove:
        file_path = Path(file_info["path"])
        
        if not file_path.exists():
            print(f"⚠️ File not found: {file_path}")
            continue
        
        try:
            # Create backup
            backup_path = create_backup(file_path, backup_dir)
            
            # Remove original file
            file_path.unlink()
            
            print(f"✅ Removed: {file_path} (backed up to {backup_path})")
            removed_count += 1
            
        except Exception as e:
            error_msg = f"❌ Failed to remove {file_path}: {e}"
            print(error_msg)
            errors.append(error_msg)
    
    print(f"\\n📊 Removal Summary:")
    print(f"   ✅ Successfully removed: {removed_count} files")
    print(f"   ❌ Errors: {len(errors)} files")
    print(f"   💾 Backups created in: {backup_dir}")
    
    if errors:
        print("\\n❌ Errors encountered:")
        for error in errors:
            print(f"   {error}")

if __name__ == "__main__":
    main()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        script_path.chmod(0o755)
        
        print(f"🗑️ Safe removal script generated: {script_path}")
        print(f"   Execute with: python {script_path}")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Identify obsolete files for safe removal')
    parser.add_argument('--analyze', action='store_true', help='Analyze project for obsolete files')
    
    args = parser.parse_args()
    
    if not args.analyze:
        args.analyze = True  # Default to analysis
    
    analyzer = ObsoleteFileAnalyzer()
    results = analyzer.analyze_project_files()
    
    print(f"\n✅ Analysis complete! Review the results and use the generated removal script for cleanup.")
    print(f"📂 Results saved in: {analyzer.analysis_results_dir}")

if __name__ == "__main__":
    main()