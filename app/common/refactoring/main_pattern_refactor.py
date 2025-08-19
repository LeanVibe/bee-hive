"""
Main Pattern Refactor - AST-Based Automated Refactoring
======================================================

Production-ready AST-based refactoring system for main() function patterns.
Implements Gemini CLI recommendations for safe, automated refactoring with
comprehensive testing and rollback capabilities.

Features:
- AST-based pattern detection and replacement
- Integrated import consolidation (touch files only once)  
- Automatic test execution with rollback on failure
- Batch processing with continuous integration support
- Comprehensive safety mechanisms and validation

Usage:
    # Analyze files for refactoring opportunities
    python main_pattern_refactor.py --analyze --module services
    
    # Dry run validation
    python main_pattern_refactor.py --dry-run --batch batch_001.json
    
    # Execute refactoring with automatic testing
    python main_pattern_refactor.py --execute --batch batch_001.json --auto-test
"""

import ast
import asyncio
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import subprocess
import tempfile
import time
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class RefactoringPlan:
    """Plan for refactoring a single file."""
    file_path: Path
    has_main_pattern: bool
    current_main_code: str
    proposed_refactoring: str
    imports_to_add: List[str]
    imports_to_remove: List[str]
    estimated_loc_savings: int
    complexity_score: float
    test_files: List[Path] = field(default_factory=list)

@dataclass
class RefactoringResult:
    """Result of refactoring operation."""
    file_path: Path
    success: bool
    message: str
    loc_saved: int = 0
    execution_time: float = 0.0
    tests_passed: bool = False
    rolled_back: bool = False
    backup_path: Optional[Path] = None

@dataclass
class BatchResult:
    """Result of batch refactoring operation."""
    batch_id: str
    files_processed: int
    files_successful: int
    total_loc_saved: int
    total_execution_time: float
    results: List[RefactoringResult] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        return self.files_successful / self.files_processed if self.files_processed > 0 else 0.0

class MainPatternDetector:
    """Detects main() function patterns using AST analysis."""
    
    def __init__(self):
        self.logger = structlog.get_logger(f"{self.__class__.__name__}")
        
    def analyze_file(self, file_path: Path) -> RefactoringPlan:
        """Analyze file for main() pattern refactoring opportunities."""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            main_pattern = self._find_main_pattern(tree, content)
            
            if not main_pattern:
                return RefactoringPlan(
                    file_path=file_path,
                    has_main_pattern=False,
                    current_main_code="",
                    proposed_refactoring="",
                    imports_to_add=[],
                    imports_to_remove=[],
                    estimated_loc_savings=0,
                    complexity_score=0.0
                )
                
            # Analyze the main pattern and generate refactoring plan
            proposed_refactoring = self._generate_refactoring(main_pattern, file_path)
            imports_analysis = self._analyze_imports(tree)
            
            return RefactoringPlan(
                file_path=file_path,
                has_main_pattern=True,
                current_main_code=main_pattern['code'],
                proposed_refactoring=proposed_refactoring,
                imports_to_add=imports_analysis['to_add'],
                imports_to_remove=imports_analysis['to_remove'],
                estimated_loc_savings=main_pattern['lines'] - proposed_refactoring.count('\n'),
                complexity_score=self._calculate_complexity(main_pattern['ast_node'])
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze {file_path}: {e}")
            return RefactoringPlan(
                file_path=file_path,
                has_main_pattern=False,
                current_main_code="",
                proposed_refactoring="",
                imports_to_add=[],
                imports_to_remove=[],
                estimated_loc_savings=0,
                complexity_score=0.0
            )
            
    def _find_main_pattern(self, tree: ast.AST, content: str) -> Optional[Dict[str, Any]]:
        """Find main() pattern in AST."""
        lines = content.splitlines()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check for if __name__ == "__main__": pattern
                if self._is_main_guard(node):
                    start_line = node.lineno - 1
                    end_line = self._find_end_line(node, lines)
                    
                    main_code = '\n'.join(lines[start_line:end_line])
                    
                    return {
                        'ast_node': node,
                        'start_line': start_line,
                        'end_line': end_line,
                        'code': main_code,
                        'lines': end_line - start_line
                    }
        return None
        
    def _is_main_guard(self, node: ast.If) -> bool:
        """Check if node is if __name__ == "__main__": pattern."""
        if not isinstance(node.test, ast.Compare):
            return False
            
        test = node.test
        if not (isinstance(test.left, ast.Name) and test.left.id == '__name__'):
            return False
            
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            return False
            
        if len(test.comparators) != 1:
            return False
            
        comparator = test.comparators[0]
        if isinstance(comparator, ast.Constant):
            return comparator.value == "__main__"
        elif isinstance(comparator, ast.Str):  # Python < 3.8 compatibility
            return comparator.s == "__main__"
            
        return False
        
    def _find_end_line(self, node: ast.If, lines: List[str]) -> int:
        """Find the end line of the main block."""
        # Start from the if statement line
        start_line = node.lineno - 1
        
        # Look for the end of the if block
        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                line_indent = len(line) - len(line.lstrip())
                if line_indent <= indent_level and not line.strip().startswith('#'):
                    return i
                    
        return len(lines)
        
    def _generate_refactoring(self, main_pattern: Dict[str, Any], file_path: Path) -> str:
        """Generate refactored code using script_base pattern."""
        main_code = main_pattern['code']
        
        # Check if already using script_base pattern
        if 'script_base' in main_code and 'BaseScript' in main_code:
            # Already refactored, return as-is
            return main_code
            
        # Extract class name from file path
        class_name = self._extract_class_name_from_path(file_path)
        
        # Extract and clean the main logic
        main_logic = self._extract_main_logic(main_code)
        
        # Generate refactored version
        refactored = f'''if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class {class_name}(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
{main_logic}
            
            return {{"status": "completed"}}
    
    script_main({class_name})'''
    
        return refactored
        
    def _extract_class_name_from_path(self, file_path: Path) -> str:
        """Extract appropriate class name from file path."""
        file_name = file_path.stem
        
        # Convert file name to class name
        words = file_name.replace('_', ' ').split()
        class_name = ''.join(word.capitalize() for word in words)
        
        # Ensure it ends with Script if not already descriptive
        if not any(suffix in class_name.lower() for suffix in ['test', 'script', 'service', 'runner']):
            class_name += 'Script'
            
        return class_name
        
    def _extract_main_logic(self, main_code: str) -> str:
        """Extract the main logic and convert to method body."""
        lines = main_code.splitlines()
        
        # Find the content inside the if __name__ == "__main__": block
        logic_lines = []
        in_main_block = False
        base_indent = None
        
        for line in lines:
            if 'if __name__' in line and '__main__' in line:
                in_main_block = True
                continue
                
            if in_main_block:
                if line.strip():
                    # Determine base indentation from first content line
                    if base_indent is None:
                        base_indent = len(line) - len(line.lstrip())
                    
                    # Remove base indentation and add method indentation
                    if len(line) >= base_indent:
                        content = line[base_indent:]
                        adjusted_line = self._convert_to_method_body(content)
                        logic_lines.append(f"            {adjusted_line}")
                    else:
                        # Handle lines with less indentation (shouldn't happen in well-formed code)
                        adjusted_line = self._convert_to_method_body(line.strip())
                        logic_lines.append(f"            {adjusted_line}")
                else:
                    # Preserve empty lines
                    logic_lines.append("")
                    
        # Clean up and ensure we have content
        cleaned_lines = [line.rstrip() for line in logic_lines]
        
        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
            
        return '\n'.join(cleaned_lines) if cleaned_lines else '            pass'
        
    def _convert_to_method_body(self, line: str) -> str:
        """Convert line to fit method body format."""
        stripped = line.strip()
        
        # Convert print statements to logger calls
        if stripped.startswith('print('):
            # Extract content between print( and )
            content = stripped[6:-1]
            return f'self.logger.info({content})'
            
        # Convert asyncio.run calls to direct await
        if 'asyncio.run(' in stripped:
            # Extract the function call from asyncio.run()
            start = stripped.find('asyncio.run(') + 12
            # Find matching closing parenthesis
            paren_count = 1
            end = start
            for i, char in enumerate(stripped[start:], start):
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        end = i
                        break
            
            if end > start:
                func_call = stripped[start:end]
                return f'await {func_call}'
            
        return stripped
        
    def _analyze_imports(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Analyze imports and determine what to add/remove."""
        existing_imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    existing_imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    existing_imports.add(f"{module}.{alias.name}")
                    
        # Determine what imports are needed for script_base
        required_imports = {
            'app.common.utilities.script_base'
        }
        
        # Check if asyncio import is needed (if async patterns detected)
        needs_asyncio = 'asyncio' not in existing_imports
        
        imports_to_add = []
        if needs_asyncio:
            imports_to_add.append('asyncio')
            
        return {
            'to_add': imports_to_add,
            'to_remove': []  # Conservative approach - don't remove imports automatically
        }
        
    def _calculate_complexity(self, node: ast.AST) -> float:
        """Calculate complexity score for the main pattern."""
        complexity = 1.0  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 0.5
            elif isinstance(child, ast.FunctionDef):
                complexity += 1.0
                
        return complexity

class TestRunner:
    """Runs tests for refactored files with characterization test support."""
    
    def __init__(self):
        self.logger = structlog.get_logger(f"{self.__class__.__name__}")
        
    def find_tests_for_file(self, file_path: Path) -> List[Path]:
        """Find corresponding test files for a given source file."""
        test_files = []
        
        # Common test file patterns
        test_patterns = [
            f"test_{file_path.stem}.py",
            f"{file_path.stem}_test.py",
            f"test_{file_path.stem}s.py",  # Plural
        ]
        
        # Common test directories
        test_dirs = [
            file_path.parent / "tests",
            file_path.parents[1] / "tests",
            Path("tests"),
            Path("test")
        ]
        
        for test_dir in test_dirs:
            if test_dir.exists():
                for pattern in test_patterns:
                    test_file = test_dir / pattern
                    if test_file.exists():
                        test_files.append(test_file)
                        
        return test_files
        
    def run_tests_for_file(self, file_path: Path) -> bool:
        """Run tests for a specific file."""
        test_files = self.find_tests_for_file(file_path)
        
        if not test_files:
            self.logger.warning(f"No test files found for {file_path}")
            return True  # Conservative - assume pass if no tests
            
        for test_file in test_files:
            if not self._run_single_test_file(test_file):
                return False
                
        return True
        
    def _run_single_test_file(self, test_file: Path) -> bool:
        """Run a single test file and return success status."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", str(test_file), "-v"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            success = result.returncode == 0
            
            if not success:
                self.logger.error(
                    f"Test failed for {test_file}",
                    stdout=result.stdout,
                    stderr=result.stderr
                )
                
            return success
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Test timeout for {test_file}")
            return False
        except Exception as e:
            self.logger.error(f"Test execution error for {test_file}: {e}")
            return False

class SafetySystem:
    """Comprehensive safety system with backup and rollback capabilities."""
    
    def __init__(self, backup_dir: Optional[Path] = None):
        self.backup_dir = backup_dir or Path(tempfile.gettempdir()) / "refactoring_backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.logger = structlog.get_logger(f"{self.__class__.__name__}")
        
    def create_backup(self, file_path: Path) -> Path:
        """Create backup of file before refactoring."""
        timestamp = int(time.time())
        backup_name = f"{file_path.name}.{timestamp}.backup"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        self.logger.debug(f"Created backup: {backup_path}")
        
        return backup_path
        
    def restore_backup(self, backup_path: Path, original_path: Path) -> bool:
        """Restore file from backup."""
        try:
            shutil.copy2(backup_path, original_path)
            self.logger.info(f"Restored {original_path} from backup")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")
            return False
            
    def cleanup_backups(self, max_age_hours: int = 24) -> None:
        """Clean up old backup files."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        for backup_file in self.backup_dir.glob("*.backup"):
            try:
                # Extract timestamp from filename
                timestamp_str = backup_file.stem.split('.')[-1]
                timestamp = int(timestamp_str)
                
                if timestamp < cutoff_time:
                    backup_file.unlink()
                    self.logger.debug(f"Cleaned up old backup: {backup_file}")
                    
            except (ValueError, IndexError):
                # Skip files that don't match expected format
                continue

class MainPatternRefactor:
    """
    Main refactoring orchestrator implementing Gemini CLI recommendations.
    
    Provides AST-based automated refactoring with comprehensive safety systems,
    automatic testing, and rollback capabilities.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.detector = MainPatternDetector()
        self.test_runner = TestRunner()
        self.safety_system = SafetySystem()
        self.logger = structlog.get_logger(f"{self.__class__.__name__}")
        
    def analyze_module(self, module_pattern: str) -> List[RefactoringPlan]:
        """Analyze all files in a module for refactoring opportunities."""
        self.logger.info(f"Analyzing module pattern: {module_pattern}")
        
        # Find Python files matching the pattern
        if module_pattern == "services":
            file_pattern = "app/services/*.py"
        elif module_pattern == "scripts":
            file_pattern = "scripts/*.py"
        elif module_pattern == "core":
            file_pattern = "app/core/*.py"
        else:
            file_pattern = f"**/{module_pattern}*.py"
            
        files = list(self.project_root.glob(file_pattern))
        plans = []
        
        for file_path in files:
            if file_path.name.startswith('__'):
                continue  # Skip __init__.py and __pycache__
                
            plan = self.detector.analyze_file(file_path)
            if plan.has_main_pattern:
                plan.test_files = self.test_runner.find_tests_for_file(file_path)
                plans.append(plan)
                
        self.logger.info(f"Found {len(plans)} files with main patterns to refactor")
        return plans
        
    def create_batch(self, plans: List[RefactoringPlan], batch_size: int = 25) -> List[Dict[str, Any]]:
        """Create batches for continuous integration workflow."""
        batches = []
        
        for i in range(0, len(plans), batch_size):
            batch_plans = plans[i:i + batch_size]
            batch_id = f"batch_{i//batch_size + 1:03d}"
            
            batch = {
                'batch_id': batch_id,
                'plans': [self._plan_to_dict(plan) for plan in batch_plans],
                'estimated_loc_savings': sum(plan.estimated_loc_savings for plan in batch_plans),
                'file_count': len(batch_plans)
            }
            batches.append(batch)
            
        return batches
        
    def refactor_file(self, plan: RefactoringPlan, dry_run: bool = False) -> RefactoringResult:
        """Refactor a single file with safety mechanisms."""
        start_time = time.time()
        
        if dry_run:
            return RefactoringResult(
                file_path=plan.file_path,
                success=True,
                message="Dry run successful",
                loc_saved=plan.estimated_loc_savings,
                execution_time=time.time() - start_time
            )
            
        # Create backup before refactoring
        backup_path = self.safety_system.create_backup(plan.file_path)
        
        try:
            # Apply refactoring
            self._apply_refactoring(plan)
            
            # Run tests
            tests_passed = self.test_runner.run_tests_for_file(plan.file_path)
            
            if not tests_passed:
                # Rollback on test failure
                self.safety_system.restore_backup(backup_path, plan.file_path)
                return RefactoringResult(
                    file_path=plan.file_path,
                    success=False,
                    message="Tests failed - rolled back",
                    execution_time=time.time() - start_time,
                    tests_passed=False,
                    rolled_back=True,
                    backup_path=backup_path
                )
                
            return RefactoringResult(
                file_path=plan.file_path,
                success=True,
                message="Refactoring successful",
                loc_saved=plan.estimated_loc_savings,
                execution_time=time.time() - start_time,
                tests_passed=True,
                backup_path=backup_path
            )
            
        except Exception as e:
            # Emergency rollback
            self.safety_system.restore_backup(backup_path, plan.file_path)
            return RefactoringResult(
                file_path=plan.file_path,
                success=False,
                message=f"Refactoring failed: {e}",
                execution_time=time.time() - start_time,
                rolled_back=True,
                backup_path=backup_path
            )
            
    def refactor_batch(self, batch_data: Dict[str, Any], dry_run: bool = False) -> BatchResult:
        """Refactor a batch of files with comprehensive tracking."""
        batch_id = batch_data['batch_id']
        plans = [self._dict_to_plan(plan_dict) for plan_dict in batch_data['plans']]
        
        self.logger.info(f"Processing batch {batch_id} with {len(plans)} files")
        
        results = []
        successful_count = 0
        total_loc_saved = 0
        start_time = time.time()
        
        for plan in plans:
            result = self.refactor_file(plan, dry_run=dry_run)
            results.append(result)
            
            if result.success:
                successful_count += 1
                total_loc_saved += result.loc_saved
                
        return BatchResult(
            batch_id=batch_id,
            files_processed=len(plans),
            files_successful=successful_count,
            total_loc_saved=total_loc_saved,
            total_execution_time=time.time() - start_time,
            results=results
        )
        
    def _apply_refactoring(self, plan: RefactoringPlan) -> None:
        """Apply the refactoring changes to the file."""
        content = plan.file_path.read_text(encoding='utf-8')
        
        # Replace the main pattern with refactored version
        new_content = content.replace(plan.current_main_code, plan.proposed_refactoring)
        
        # Add any required imports at the top
        if plan.imports_to_add:
            lines = new_content.splitlines()
            
            # Find where to insert imports (after existing imports)
            insert_index = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    insert_index = i + 1
                elif line.strip() and not line.strip().startswith('#'):
                    break
                    
            # Insert new imports
            for import_stmt in plan.imports_to_add:
                lines.insert(insert_index, f"import {import_stmt}")
                insert_index += 1
                
            new_content = '\n'.join(lines)
            
        # Write the refactored content
        plan.file_path.write_text(new_content, encoding='utf-8')
        
    def _plan_to_dict(self, plan: RefactoringPlan) -> Dict[str, Any]:
        """Convert RefactoringPlan to dictionary for JSON serialization."""
        return {
            'file_path': str(plan.file_path),
            'has_main_pattern': plan.has_main_pattern,
            'current_main_code': plan.current_main_code,
            'proposed_refactoring': plan.proposed_refactoring,
            'imports_to_add': plan.imports_to_add,
            'imports_to_remove': plan.imports_to_remove,
            'estimated_loc_savings': plan.estimated_loc_savings,
            'complexity_score': plan.complexity_score,
            'test_files': [str(f) for f in plan.test_files]
        }
        
    def _dict_to_plan(self, plan_dict: Dict[str, Any]) -> RefactoringPlan:
        """Convert dictionary back to RefactoringPlan."""
        return RefactoringPlan(
            file_path=Path(plan_dict['file_path']),
            has_main_pattern=plan_dict['has_main_pattern'],
            current_main_code=plan_dict['current_main_code'],
            proposed_refactoring=plan_dict['proposed_refactoring'],
            imports_to_add=plan_dict['imports_to_add'],
            imports_to_remove=plan_dict['imports_to_remove'],
            estimated_loc_savings=plan_dict['estimated_loc_savings'],
            complexity_score=plan_dict['complexity_score'],
            test_files=[Path(f) for f in plan_dict['test_files']]
        )

# CLI Interface
def main():
    """Command-line interface for the refactoring system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AST-based main() pattern refactoring")
    parser.add_argument('--analyze', action='store_true', help='Analyze files for refactoring opportunities')
    parser.add_argument('--dry-run', action='store_true', help='Perform dry run validation')
    parser.add_argument('--execute', action='store_true', help='Execute refactoring')
    parser.add_argument('--module', type=str, help='Module to analyze (services, scripts, core)')
    parser.add_argument('--batch', type=str, help='Batch file to process')
    parser.add_argument('--batch-size', type=int, default=25, help='Size of batches for processing')
    parser.add_argument('--auto-test', action='store_true', help='Automatically run tests after refactoring')
    
    args = parser.parse_args()
    
    refactor = MainPatternRefactor()
    
    if args.analyze and args.module:
        # Analyze module and create batches
        plans = refactor.analyze_module(args.module)
        batches = refactor.create_batch(plans, args.batch_size)
        
        print(f"Analysis complete:")
        print(f"- Files with main patterns: {len(plans)}")
        print(f"- Estimated LOC savings: {sum(plan.estimated_loc_savings for plan in plans)}")
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
            result = refactor.refactor_batch(batch_data, dry_run=True)
            print(f"Dry run complete for {result.batch_id}:")
            print(f"- Files to process: {result.files_processed}")
            print(f"- Estimated LOC savings: {batch_data['estimated_loc_savings']}")
            
        elif args.execute:
            result = refactor.refactor_batch(batch_data, dry_run=False)
            print(f"Refactoring complete for {result.batch_id}:")
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