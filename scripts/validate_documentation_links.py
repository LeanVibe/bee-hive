#!/usr/bin/env python3
"""
Automated Link Validation System for LeanVibe Agent Hive Documentation
Implements Gemini CLI recommendation for automated link checking
"""

import os
import re
import glob
import json
import urllib.parse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LinkResult:
    """Container for link validation results"""
    file_path: str
    line_number: int
    link_text: str
    link_target: str
    link_type: str  # 'internal', 'external', 'anchor'
    status: str     # 'valid', 'broken', 'warning'
    message: str

class DocumentationLinkValidator:
    """
    Automated link validation system for markdown documentation
    Per Gemini CLI recommendation for reliable cross-reference tracking
    """
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.results: List[LinkResult] = []
        self.file_map: Dict[str, Path] = {}
        self.anchor_map: Dict[str, Set[str]] = {}
        
        # Patterns for different link types
        self.link_patterns = {
            'markdown': re.compile(r'\[([^\]]*)\]\(([^)]+)\)'),
            'reference': re.compile(r'\[([^\]]*)\]:\s*(.+)'),
            'html': re.compile(r'<a\s+href=["\']([^"\']+)["\'][^>]*>([^<]*)</a>')
        }
        
        # Known archived files pattern (from our consolidation)
        self.archived_files = set()
        self._build_file_maps()
    
    def _build_file_maps(self):
        """Build maps of all markdown files and their anchors"""
        markdown_files = glob.glob(str(self.base_path / "**/*.md"), recursive=True)
        
        for file_path in markdown_files:
            path_obj = Path(file_path)
            relative_path = path_obj.relative_to(self.base_path)
            
            # Build file map for internal link resolution
            self.file_map[str(relative_path)] = path_obj
            self.file_map[path_obj.name] = path_obj
            
            # Track archived files
            if 'archive' in str(relative_path):
                self.archived_files.add(str(relative_path))
            
            # Build anchor map
            self._extract_anchors(path_obj)
    
    def _extract_anchors(self, file_path: Path):
        """Extract all anchors (headers) from a markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find all markdown headers
            header_pattern = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
            headers = header_pattern.findall(content)
            
            # Convert headers to anchor format
            anchors = set()
            for header in headers:
                # GitHub-style anchor generation
                anchor = header.lower()
                anchor = re.sub(r'[^\w\s-]', '', anchor)
                anchor = re.sub(r'[-\s]+', '-', anchor)
                anchors.add(anchor.strip('-'))
            
            self.anchor_map[str(file_path)] = anchors
            
        except Exception as e:
            print(f"Warning: Could not extract anchors from {file_path}: {e}")
    
    def validate_links(self) -> List[LinkResult]:
        """Main validation method - validates all links in all markdown files"""
        markdown_files = glob.glob(str(self.base_path / "**/*.md"), recursive=True)
        
        for file_path in markdown_files:
            self._validate_file_links(Path(file_path))
        
        return self.results
    
    def _validate_file_links(self, file_path: Path):
        """Validate all links in a single markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                # Check markdown links
                for match in self.link_patterns['markdown'].finditer(line):
                    link_text, link_target = match.groups()
                    self._validate_single_link(file_path, line_num, link_text, link_target)
                
                # Check reference links
                for match in self.link_patterns['reference'].finditer(line):
                    link_text, link_target = match.groups()
                    self._validate_single_link(file_path, line_num, link_text, link_target, is_reference=True)
                    
        except Exception as e:
            self.results.append(LinkResult(
                file_path=str(file_path),
                line_number=0,
                link_text="FILE_READ_ERROR",
                link_target="",
                link_type="error",
                status="broken",
                message=f"Could not read file: {e}"
            ))
    
    def _validate_single_link(self, file_path: Path, line_num: int, link_text: str, link_target: str, is_reference: bool = False):
        """Validate a single link"""
        # Skip empty links
        if not link_target.strip():
            return
        
        # Determine link type
        if link_target.startswith(('http://', 'https://')):
            link_type = 'external'
            status, message = self._validate_external_link(link_target)
        elif link_target.startswith('#'):
            link_type = 'anchor'
            status, message = self._validate_anchor_link(file_path, link_target)
        elif link_target.startswith('mailto:'):
            link_type = 'email'
            status, message = 'valid', 'Email link (not validated)'
        else:
            link_type = 'internal'
            status, message = self._validate_internal_link(file_path, link_target)
        
        self.results.append(LinkResult(
            file_path=str(file_path.relative_to(self.base_path)),
            line_number=line_num,
            link_text=link_text,
            link_target=link_target,
            link_type=link_type,
            status=status,
            message=message
        ))
    
    def _validate_internal_link(self, source_file: Path, link_target: str) -> Tuple[str, str]:
        """Validate internal file links"""
        # Handle anchor links within files
        if '#' in link_target:
            file_part, anchor_part = link_target.split('#', 1)
            target_file = file_part if file_part else source_file
        else:
            target_file = link_target
            anchor_part = None
        
        # Resolve relative paths
        if not target_file:
            resolved_path = source_file
        else:
            if target_file.startswith('/'):
                # Absolute path from base
                resolved_path = self.base_path / target_file.lstrip('/')
            else:
                # Relative path from source file
                resolved_path = (source_file.parent / target_file).resolve()
        
        # Check if file exists
        if not resolved_path.exists():
            # Try to find in file map
            if target_file in self.file_map:
                resolved_path = self.file_map[target_file]
            else:
                return 'broken', f'File not found: {target_file}'
        
        # Check if linking to archived file
        relative_resolved = resolved_path.relative_to(self.base_path)
        if str(relative_resolved) in self.archived_files:
            return 'warning', f'Links to archived file: {relative_resolved}'
        
        # Validate anchor if present
        if anchor_part:
            if str(resolved_path) in self.anchor_map:
                if anchor_part not in self.anchor_map[str(resolved_path)]:
                    return 'warning', f'Anchor not found: #{anchor_part}'
        
        return 'valid', 'Internal link valid'
    
    def _validate_anchor_link(self, source_file: Path, anchor: str) -> Tuple[str, str]:
        """Validate anchor links within the same file"""
        anchor_name = anchor.lstrip('#')
        
        if str(source_file) in self.anchor_map:
            if anchor_name in self.anchor_map[str(source_file)]:
                return 'valid', 'Anchor exists'
            else:
                return 'broken', f'Anchor not found: {anchor}'
        
        return 'warning', 'Could not validate anchor (file not in map)'
    
    def _validate_external_link(self, url: str) -> Tuple[str, str]:
        """Basic validation for external links (URL format only)"""
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme and parsed.netloc:
                return 'valid', 'External URL format valid (not checked)'
            else:
                return 'broken', 'Invalid URL format'
        except Exception as e:
            return 'broken', f'URL parsing error: {e}'
    
    def generate_report(self, output_file: str = None) -> Dict:
        """Generate comprehensive validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'base_path': str(self.base_path),
            'total_links': len(self.results),
            'summary': {
                'valid': sum(1 for r in self.results if r.status == 'valid'),
                'broken': sum(1 for r in self.results if r.status == 'broken'),
                'warnings': sum(1 for r in self.results if r.status == 'warning')
            },
            'by_type': {},
            'by_file': {},
            'issues': [
                {
                    'file': r.file_path,
                    'line': r.line_number,
                    'text': r.link_text,
                    'target': r.link_target,
                    'type': r.link_type,
                    'status': r.status,
                    'message': r.message
                }
                for r in self.results if r.status in ['broken', 'warning']
            ]
        }
        
        # Group by type
        for result in self.results:
            if result.link_type not in report['by_type']:
                report['by_type'][result.link_type] = {'valid': 0, 'broken': 0, 'warning': 0}
            report['by_type'][result.link_type][result.status] += 1
        
        # Group by file
        for result in self.results:
            if result.file_path not in report['by_file']:
                report['by_file'][result.file_path] = {'valid': 0, 'broken': 0, 'warning': 0}
            report['by_file'][result.file_path][result.status] += 1
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def print_summary(self):
        """Print human-readable summary"""
        report = self.generate_report()
        
        print("=" * 60)
        print("📋 DOCUMENTATION LINK VALIDATION REPORT")
        print("=" * 60)
        print(f"🕐 Timestamp: {report['timestamp']}")
        print(f"📁 Base Path: {report['base_path']}")
        print(f"🔗 Total Links: {report['total_links']}")
        print()
        
        print("📊 SUMMARY")
        print("-" * 30)
        print(f"✅ Valid Links:   {report['summary']['valid']:4d}")
        print(f"❌ Broken Links:  {report['summary']['broken']:4d}")
        print(f"⚠️  Warnings:      {report['summary']['warnings']:4d}")
        print()
        
        if report['summary']['broken'] > 0 or report['summary']['warnings'] > 0:
            print("🔍 ISSUES FOUND")
            print("-" * 30)
            for issue in report['issues']:
                status_icon = "❌" if issue['status'] == 'broken' else "⚠️"
                print(f"{status_icon} {issue['file']}:{issue['line']}")
                print(f"   Link: [{issue['text']}]({issue['target']})")
                print(f"   Issue: {issue['message']}")
                print()
        
        print("📈 BY FILE TYPE")
        print("-" * 30)
        for link_type, counts in report['by_type'].items():
            total = counts['valid'] + counts['broken'] + counts['warning']
            print(f"{link_type.title()}: {total} total "
                  f"(✅{counts['valid']} ❌{counts['broken']} ⚠️{counts['warning']})")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate documentation links')
    parser.add_argument('--base-path', default='.', help='Base path for documentation')
    parser.add_argument('--output', help='Output JSON report file')
    parser.add_argument('--quiet', action='store_true', help='Suppress console output')
    
    args = parser.parse_args()
    
    validator = DocumentationLinkValidator(args.base_path)
    validator.validate_links()
    
    if not args.quiet:
        validator.print_summary()
    
    if args.output:
        validator.generate_report(args.output)
        if not args.quiet:
            print(f"\n📄 Detailed report saved to: {args.output}")
    
    # Exit with error code if broken links found
    broken_count = sum(1 for r in validator.results if r.status == 'broken')
    if broken_count > 0:
        exit(1)


if __name__ == "__main__":
    main()