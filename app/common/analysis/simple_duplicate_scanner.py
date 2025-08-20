import asyncio
"""
Simple Duplicate Scanner - Pragmatic 80/20 Approach
==================================================

Focus on the highest-ROI duplicates identified in TECHNICAL_DEBT_REMEDIATION_PLAN:
1. 15,000+ LOC duplicate patterns (ROI 1283.0) - main() functions
2. 29 duplicate __init__.py files (ROI 1031.0)
3. Common import patterns (ROI 508.0)

This scanner uses simple string matching to quickly identify these patterns
and generate immediate actionable consolidation tasks.

Based on first principles:
- Working software > theoretical perfection
- 80% results with 20% effort
- Actionable output over comprehensive analysis
"""

import re
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class QuickWin:
    """Represents an immediate consolidation opportunity."""
    pattern_name: str
    files: List[Path]
    duplicate_content: str
    lines_saved: int
    roi_score: float
    action_plan: str
    estimated_hours: int

class SimpleDuplicateScanner:
    """
    Pragmatic duplicate scanner focusing on high-ROI quick wins.
    
    Uses simple pattern matching to find the three highest-value
    duplicate categories from the technical debt analysis.
    """
    
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.logger = structlog.get_logger(f"{self.__class__.__name__}")
        
    def scan_quick_wins(self) -> List[QuickWin]:
        """Scan for the three highest-ROI duplicate patterns."""
        self.logger.info("🎯 Scanning for high-ROI duplicate patterns")
        
        quick_wins = []
        
        # 1. __init__.py duplicates (ROI: 1031.0)
        init_duplicates = self._find_init_duplicates()
        if init_duplicates:
            quick_wins.append(init_duplicates)
            
        # 2. main() function patterns (ROI: 1283.0)
        main_duplicates = self._find_main_function_duplicates()
        if main_duplicates:
            quick_wins.append(main_duplicates)
            
        # 3. Import pattern duplicates (ROI: 508.0)
        import_duplicates = self._find_import_duplicates()
        if import_duplicates:
            quick_wins.append(import_duplicates)
            
        self.logger.info(f"✅ Found {len(quick_wins)} high-ROI quick wins")
        return quick_wins
        
    def _find_init_duplicates(self) -> QuickWin:
        """Find duplicate __init__.py files."""
        init_files = list(self.project_root.rglob("__init__.py"))
        
        # Group by content hash
        content_groups = defaultdict(list)
        
        for file_path in init_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                # Normalize content (remove comments, normalize imports)
                normalized = self._normalize_init_content(content)
                content_hash = hashlib.md5(normalized.encode()).hexdigest()
                content_groups[content_hash].append(file_path)
            except Exception as e:
                self.logger.warning(f"Failed to read {file_path}: {e}")
                
        # Find largest duplicate group
        largest_group = max(content_groups.values(), key=len, default=[])
        
        if len(largest_group) >= 5:  # Significant duplication
            return QuickWin(
                pattern_name="__init__.py duplicates",
                files=largest_group,
                duplicate_content=self._get_representative_content(largest_group[0]),
                lines_saved=(len(largest_group) - 1) * 12,  # Conservative estimate
                roi_score=1031.0,
                action_plan="Create app/common/templates/__init__.py template and standardize all init files",
                estimated_hours=16
            )
        return None
        
    def _find_main_function_duplicates(self) -> QuickWin:
        """Find duplicate main() function patterns."""
        python_files = list(self.project_root.rglob("*.py"))
        main_pattern_files = []
        
        # Common main patterns to detect
        main_patterns = [
            r'if __name__ == ["\']__main__["\']:',
            r'def main\(\):',
            r'async def main\(\):'
        ]
        
        for file_path in python_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Check for main patterns
                has_main_pattern = any(re.search(pattern, content) for pattern in main_patterns)
                if has_main_pattern:
                    main_pattern_files.append(file_path)
                    
            except Exception as e:
                self.logger.warning(f"Failed to read {file_path}: {e}")
                
        if len(main_pattern_files) >= 20:  # Significant duplication
            return QuickWin(
                pattern_name="main() function patterns", 
                files=main_pattern_files,
                duplicate_content="Standard main() function boilerplate",
                lines_saved=len(main_pattern_files) * 15,  # Conservative estimate
                roi_score=1283.0,
                action_plan="Create app/common/utilities/script_base.py with standard main wrapper",
                estimated_hours=40
            )
        return None
        
    def _find_import_duplicates(self) -> QuickWin:
        """Find common import patterns."""
        python_files = list(self.project_root.rglob("*.py"))
        import_groups = defaultdict(list)
        
        for file_path in python_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                imports = self._extract_import_block(content)
                
                if imports and len(imports.splitlines()) >= 3:
                    # Normalize imports for grouping
                    normalized = self._normalize_imports(imports)
                    import_hash = hashlib.md5(normalized.encode()).hexdigest()
                    import_groups[import_hash].append(file_path)
                    
            except Exception as e:
                self.logger.warning(f"Failed to read {file_path}: {e}")
                
        # Find largest group
        largest_group = max(import_groups.values(), key=len, default=[])
        
        if len(largest_group) >= 10:  # Significant duplication
            return QuickWin(
                pattern_name="Common import patterns",
                files=largest_group,
                duplicate_content=self._get_representative_imports(largest_group[0]),
                lines_saved=len(largest_group) * 5,
                roi_score=508.0,
                action_plan="Create app/common/imports/ modules for standard import patterns",
                estimated_hours=24
            )
        return None
        
    def _normalize_init_content(self, content: str) -> str:
        """Normalize __init__.py content for comparison."""
        lines = []
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                # Normalize common patterns
                line = re.sub(r'__version__ = ["\'][^"\']*["\']', '__version__ = "VERSION"', line)
                line = re.sub(r'from \. import \w+', 'from . import MODULE', line)
                lines.append(line)
        return '\n'.join(lines)
        
    def _extract_import_block(self, content: str) -> str:
        """Extract import block from Python file."""
        lines = content.splitlines()
        import_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                import_lines.append(line)
            elif stripped and not stripped.startswith('#') and import_lines:
                # End of import block
                break
                
        return '\n'.join(import_lines)
        
    def _normalize_imports(self, imports: str) -> str:
        """Normalize imports for comparison."""
        lines = []
        for line in imports.splitlines():
            line = line.strip()
            if line:
                # Sort imports alphabetically for consistent comparison
                if line.startswith('from '):
                    # Extract module and sort imported names
                    match = re.match(r'from ([\w.]+) import (.+)', line)
                    if match:
                        module, names = match.groups()
                        sorted_names = ', '.join(sorted(names.split(', ')))
                        line = f"from {module} import {sorted_names}"
                lines.append(line)
        
        return '\n'.join(sorted(lines))
        
    def _get_representative_content(self, file_path: Path) -> str:
        """Get representative content from a file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            return content[:200] + "..." if len(content) > 200 else content
        except Exception:
            return "Could not read file content"
            
    def _get_representative_imports(self, file_path: Path) -> str:
        """Get representative imports from a file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            imports = self._extract_import_block(content)
            return imports[:200] + "..." if len(imports) > 200 else imports
        except Exception:
            return "Could not read imports"
            
    def generate_action_plan(self, quick_wins: List[QuickWin]) -> str:
        """Generate immediate action plan."""
        if not quick_wins:
            return "No high-ROI duplicates found."
            
        # Sort by ROI score
        sorted_wins = sorted(quick_wins, key=lambda w: w.roi_score, reverse=True)
        
        plan = "# Immediate Technical Debt Action Plan\n\n"
        total_savings = sum(w.lines_saved for w in sorted_wins)
        total_hours = sum(w.estimated_hours for w in sorted_wins)
        
        plan += f"**Total Impact**: {total_savings} LOC eliminated in {total_hours} hours\n\n"
        
        for i, win in enumerate(sorted_wins, 1):
            plan += f"## Phase {i}: {win.pattern_name}\n"
            plan += f"- **Files affected**: {len(win.files)}\n"
            plan += f"- **Lines saved**: {win.lines_saved}\n" 
            plan += f"- **ROI**: {win.roi_score}\n"
            plan += f"- **Effort**: {win.estimated_hours} hours\n"
            plan += f"- **Action**: {win.action_plan}\n\n"
            
        return plan
        
def run_quick_scan():
    """Run a quick scan and generate action plan."""
    scanner = SimpleDuplicateScanner("/Users/bogdan/work/leanvibe-dev/bee-hive")
    quick_wins = scanner.scan_quick_wins()
    action_plan = scanner.generate_action_plan(quick_wins)
    
    print(action_plan)
    
    return quick_wins

if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class SimpleDuplicateScannerScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            run_quick_scan()
            
            return {"status": "completed"}
    
    script_main(SimpleDuplicateScannerScript)