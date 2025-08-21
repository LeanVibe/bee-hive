#!/usr/bin/env python3
"""
Documentation Debt Analyzer
============================

Phase 6: Analyzes documentation patterns across 500+ files to identify
duplicate content, redundant explanations, and consolidation opportunities.

Target Areas:
- API documentation duplicates
- Setup instruction redundancy  
- Architectural description overlap
- Duplicate code examples
- Repeated configuration guides
"""

import os
import re
import hashlib
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import tempfile
import shutil
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class DocumentationPattern:
    """Represents a documentation consolidation pattern."""
    pattern_type: str
    pattern_name: str
    similar_documents: List[Dict]
    content_similarity: float
    consolidation_potential: int
    affected_files: List[Path]

class DocumentationDebtAnalyzer:
    """Analyzes documentation debt and identifies consolidation opportunities."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.consolidated_docs_dir = self.project_root / "docs" / "consolidated"
        self.backup_dir = Path(tempfile.mkdtemp(prefix="docs_backups_"))
        
        # Documentation file extensions to analyze
        self.doc_extensions = {'.md', '.rst', '.txt', '.adoc', '.wiki'}
        
        # Pattern categories for documentation analysis
        self.doc_patterns = {
            'api_documentation': ['api', 'endpoint', 'swagger', 'openapi', 'rest'],
            'setup_instructions': ['setup', 'install', 'getting_started', 'quickstart', 'configure'],
            'architectural_docs': ['architecture', 'design', 'system', 'overview', 'structure'],
            'code_examples': ['example', 'sample', 'demo', 'tutorial', 'howto'],
            'configuration_guides': ['config', 'environment', 'deployment', 'settings'],
            'troubleshooting': ['troubleshoot', 'debug', 'faq', 'issues', 'problems'],
            'user_guides': ['guide', 'manual', 'documentation', 'help', 'usage']
        }
        
        # Common documentation templates and patterns
        self.common_sections = [
            'installation', 'configuration', 'usage', 'examples', 'api reference',
            'troubleshooting', 'faq', 'contributing', 'license', 'changelog'
        ]
    
    def discover_documentation_files(self) -> List[Path]:
        """Discover all documentation files in the project."""
        doc_files = []
        
        for ext in self.doc_extensions:
            doc_files.extend(self.project_root.rglob(f"*{ext}"))
        
        # Filter out generated files and dependencies
        filtered_files = []
        for doc_file in doc_files:
            file_str = str(doc_file).lower()
            if not any(skip in file_str for skip in ['.venv', 'venv', 'node_modules', '__pycache__', '.git']):
                filtered_files.append(doc_file)
        
        logger.info(f"Discovered {len(filtered_files)} documentation files")
        return filtered_files
    
    def analyze_documentation_content(self, doc_files: List[Path]) -> Dict[str, List[Dict]]:
        """Analyze content of documentation files for patterns."""
        print("📚 Analyzing documentation content patterns...")
        
        doc_analysis = {}
        content_patterns = defaultdict(list)
        
        for doc_file in doc_files:
            try:
                analysis = self.analyze_single_document(doc_file)
                if analysis:
                    doc_analysis[str(doc_file)] = analysis
                    
                    # Group by content patterns
                    for pattern_type in analysis.get('patterns', []):
                        content_patterns[pattern_type].append({
                            'file': doc_file,
                            'analysis': analysis
                        })
                        
            except Exception as e:
                logger.warning(f"Failed to analyze {doc_file}: {e}")
        
        print(f"   📄 Analyzed {len(doc_analysis)} documentation files")
        print(f"   🔍 Found {len(content_patterns)} content pattern types")
        
        return dict(content_patterns)
    
    def analyze_single_document(self, doc_file: Path) -> Optional[Dict]:
        """Analyze a single documentation file."""
        try:
            with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract basic metadata
            word_count = len(content.split())
            line_count = len(content.split('\n'))
            
            # Analyze content structure
            sections = self.extract_sections(content)
            code_blocks = self.extract_code_blocks(content)
            
            # Determine document patterns
            patterns = self.classify_document_patterns(doc_file, content)
            
            # Calculate content fingerprint
            content_hash = hashlib.md5(self.normalize_content(content).encode()).hexdigest()[:12]
            
            return {
                'file_path': doc_file,
                'word_count': word_count,
                'line_count': line_count,
                'sections': sections,
                'code_blocks': code_blocks,
                'patterns': patterns,
                'content_hash': content_hash,
                'content_preview': content[:500]  # First 500 chars
            }
            
        except Exception as e:
            logger.debug(f"Error analyzing document {doc_file}: {e}")
            return None
    
    def extract_sections(self, content: str) -> List[str]:
        """Extract section headings from documentation."""
        sections = []
        
        # Markdown headers
        markdown_headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        sections.extend(markdown_headers)
        
        # RST headers (underlined)
        rst_pattern = r'^(.+)\n[=-~^"\'`#*+<>!@$%&()_\\|;:,.?/]+$'
        rst_headers = re.findall(rst_pattern, content, re.MULTILINE)
        sections.extend(rst_headers)
        
        # Clean and normalize section names
        cleaned_sections = []
        for section in sections:
            cleaned = re.sub(r'[^\w\s]', '', section).strip().lower()
            if cleaned:
                cleaned_sections.append(cleaned)
        
        return cleaned_sections
    
    def extract_code_blocks(self, content: str) -> List[Dict]:
        """Extract code blocks from documentation."""
        code_blocks = []
        
        # Markdown code blocks
        markdown_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', content, re.DOTALL)
        for language, code in markdown_blocks:
            code_blocks.append({
                'language': language or 'unknown',
                'code': code.strip(),
                'length': len(code.split('\n')),
                'type': 'markdown'
            })
        
        # RST code blocks
        rst_blocks = re.findall(r'.. code-block::\s*(\w+)\n\n((?:\s{4}.*\n?)*)', content)
        for language, code in rst_blocks:
            code_blocks.append({
                'language': language,
                'code': code.strip(),
                'length': len(code.split('\n')),
                'type': 'rst'
            })
        
        # Inline code
        inline_code = re.findall(r'`([^`]+)`', content)
        for code in inline_code:
            if len(code.split()) > 2:  # Only consider multi-word code
                code_blocks.append({
                    'language': 'inline',
                    'code': code,
                    'length': 1,
                    'type': 'inline'
                })
        
        return code_blocks
    
    def classify_document_patterns(self, doc_file: Path, content: str) -> List[str]:
        """Classify document into pattern categories."""
        file_name = doc_file.name.lower()
        content_lower = content.lower()
        
        patterns = []
        
        for pattern_type, keywords in self.doc_patterns.items():
            # Check filename
            filename_matches = sum(1 for keyword in keywords if keyword in file_name)
            
            # Check content
            content_matches = sum(1 for keyword in keywords if keyword in content_lower)
            
            # Pattern strength scoring
            total_score = filename_matches * 3 + content_matches  # Filename weighted more
            
            if total_score >= 2:  # Minimum threshold
                patterns.append(pattern_type)
        
        return patterns if patterns else ['general']
    
    def normalize_content(self, content: str) -> str:
        """Normalize content for similarity comparison."""
        # Remove markdown/rst formatting
        content = re.sub(r'#+\s+', '', content)  # Headers
        content = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', content)  # Bold/italic
        content = re.sub(r'`([^`]+)`', r'\1', content)  # Inline code
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)  # Links
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip().lower()
        
        return content
    
    def find_similar_documents(self, content_patterns: Dict[str, List[Dict]]) -> List[DocumentationPattern]:
        """Find similar documents for consolidation."""
        print("🔍 Finding similar documentation patterns...")
        
        similar_patterns = []
        
        for pattern_type, documents in content_patterns.items():
            if len(documents) < 2:  # Need at least 2 documents
                continue
            
            print(f"   📝 Analyzing {pattern_type}: {len(documents)} documents")
            
            # Group similar documents within this pattern type
            similar_groups = self.group_similar_documents(documents)
            
            for group in similar_groups:
                if len(group) >= 2:  # At least 2 similar documents
                    pattern = self.create_documentation_pattern(pattern_type, group)
                    similar_patterns.append(pattern)
        
        return similar_patterns
    
    def group_similar_documents(self, documents: List[Dict]) -> List[List[Dict]]:
        """Group similar documents for consolidation."""
        groups = []
        processed = set()
        
        for i, doc1 in enumerate(documents):
            if i in processed:
                continue
            
            group = [doc1]
            processed.add(i)
            
            for j, doc2 in enumerate(documents[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self.calculate_document_similarity(
                    doc1['analysis'], doc2['analysis']
                )
                
                if similarity > 0.7:  # High similarity threshold
                    group.append(doc2)
                    processed.add(j)
            
            if len(group) >= 2:
                groups.append(group)
        
        return groups
    
    def calculate_document_similarity(self, doc1: Dict, doc2: Dict) -> float:
        """Calculate similarity between two documents."""
        similarities = []
        
        # Section similarity
        sections1 = set(doc1.get('sections', []))
        sections2 = set(doc2.get('sections', []))
        if sections1 or sections2:
            section_sim = len(sections1 & sections2) / len(sections1 | sections2)
            similarities.append(section_sim)
        
        # Content length similarity
        len1, len2 = doc1.get('word_count', 0), doc2.get('word_count', 0)
        if len1 and len2:
            length_sim = min(len1, len2) / max(len1, len2)
            similarities.append(length_sim)
        
        # Code block similarity
        code1 = doc1.get('code_blocks', [])
        code2 = doc2.get('code_blocks', [])
        if code1 or code2:
            code_sim = self.calculate_code_similarity(code1, code2)
            similarities.append(code_sim)
        
        # Content preview similarity (simple text comparison)
        content1 = doc1.get('content_preview', '')
        content2 = doc2.get('content_preview', '')
        if content1 and content2:
            content_sim = self.calculate_text_similarity(content1, content2)
            similarities.append(content_sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def calculate_code_similarity(self, code_blocks1: List[Dict], code_blocks2: List[Dict]) -> float:
        """Calculate similarity between code block collections."""
        if not code_blocks1 and not code_blocks2:
            return 1.0
        if not code_blocks1 or not code_blocks2:
            return 0.0
        
        # Compare by language distribution
        langs1 = Counter(block['language'] for block in code_blocks1)
        langs2 = Counter(block['language'] for block in code_blocks2)
        
        all_langs = set(langs1.keys()) | set(langs2.keys())
        similarity = 0.0
        
        for lang in all_langs:
            count1, count2 = langs1.get(lang, 0), langs2.get(lang, 0)
            if count1 and count2:
                similarity += min(count1, count2) / max(count1, count2)
        
        return similarity / len(all_langs) if all_langs else 0.0
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple word overlap."""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        return len(words1 & words2) / len(words1 | words2)
    
    def create_documentation_pattern(self, pattern_type: str, documents: List[Dict]) -> DocumentationPattern:
        """Create a documentation consolidation pattern."""
        # Calculate consolidation potential
        total_words = sum(doc['analysis'].get('word_count', 0) for doc in documents)
        consolidation_potential = int(total_words * 0.4)  # Estimate 40% reduction
        
        # Calculate average similarity
        similarities = []
        for i, doc1 in enumerate(documents):
            for doc2 in documents[i+1:]:
                sim = self.calculate_document_similarity(doc1['analysis'], doc2['analysis'])
                similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Generate pattern name
        file_names = [doc['file'].stem for doc in documents]
        common_words = self.find_common_words(file_names)
        pattern_name = f"{pattern_type}_{common_words}" if common_words else f"{pattern_type}_consolidated"
        
        return DocumentationPattern(
            pattern_type=pattern_type,
            pattern_name=pattern_name,
            similar_documents=documents,
            content_similarity=avg_similarity,
            consolidation_potential=consolidation_potential,
            affected_files=[doc['file'] for doc in documents]
        )
    
    def find_common_words(self, names: List[str]) -> str:
        """Find common words in document names."""
        words = []
        for name in names:
            words.extend(re.findall(r'[a-zA-Z]+', name.lower()))
        
        word_counts = Counter(words)
        common_words = [word for word, count in word_counts.most_common(2) if count > 1]
        return '_'.join(common_words[:2]) if common_words else 'unified'
    
    def generate_consolidation_report(self, patterns: List[DocumentationPattern], dry_run: bool = False) -> Dict:
        """Generate documentation consolidation report."""
        print("📋 Generating documentation consolidation report...")
        
        total_patterns = len(patterns)
        total_files = sum(len(pattern.affected_files) for pattern in patterns)
        total_savings = sum(pattern.consolidation_potential for pattern in patterns)
        
        # Group by pattern type
        by_type = defaultdict(list)
        for pattern in patterns:
            by_type[pattern.pattern_type].append(pattern)
        
        type_analysis = {}
        for pattern_type, type_patterns in by_type.items():
            type_analysis[pattern_type] = {
                'count': len(type_patterns),
                'total_files': sum(len(p.affected_files) for p in type_patterns),
                'total_savings': sum(p.consolidation_potential for p in type_patterns),
                'avg_similarity': sum(p.content_similarity for p in type_patterns) / len(type_patterns)
            }
        
        # Create consolidated documentation if not dry run
        if not dry_run:
            self.create_consolidated_documentation(patterns)
        
        return {
            'summary': {
                'total_documentation_patterns': total_patterns,
                'total_files_affected': total_files,
                'total_consolidation_potential': total_savings,
                'top_opportunities': sorted(by_type.items(), 
                                          key=lambda x: type_analysis[x[0]]['total_savings'], 
                                          reverse=True)[:5]
            },
            'by_type': type_analysis,
            'patterns': patterns
        }
    
    def create_consolidated_documentation(self, patterns: List[DocumentationPattern]) -> None:
        """Create consolidated documentation files."""
        print("📝 Creating consolidated documentation...")
        
        self.consolidated_docs_dir.mkdir(parents=True, exist_ok=True)
        
        created_count = 0
        for pattern in patterns:
            if len(pattern.similar_documents) >= 2:
                consolidated_file = self.create_consolidated_doc_file(pattern)
                if consolidated_file:
                    created_count += 1
                    logger.info(f"Created consolidated documentation: {consolidated_file}")
        
        print(f"   ✅ Created {created_count} consolidated documentation files")
    
    def create_consolidated_doc_file(self, pattern: DocumentationPattern) -> Optional[Path]:
        """Create a consolidated documentation file from a pattern."""
        consolidated_path = self.consolidated_docs_dir / f"{pattern.pattern_name}.md"
        
        # Generate consolidated content
        content = self.generate_consolidated_content(pattern)
        
        try:
            consolidated_path.write_text(content)
            return consolidated_path
        except Exception as e:
            logger.error(f"Failed to create consolidated doc {consolidated_path}: {e}")
            return None
    
    def generate_consolidated_content(self, pattern: DocumentationPattern) -> str:
        """Generate consolidated documentation content."""
        content = f"""# {pattern.pattern_name.replace('_', ' ').title()}

*Consolidated {pattern.pattern_type.replace('_', ' ')} documentation*

## Overview

This documentation consolidates {len(pattern.similar_documents)} similar {pattern.pattern_type.replace('_', ' ')} documents:

"""
        
        # List original documents
        for doc in pattern.similar_documents:
            file_path = doc['file']
            analysis = doc['analysis']
            content += f"- `{file_path.name}` ({analysis.get('word_count', 0)} words)\n"
        
        content += f"""

**Consolidation Benefits:**
- Estimated word reduction: {pattern.consolidation_potential:,} words
- Content similarity: {pattern.content_similarity:.1%}
- Single source of truth for {pattern.pattern_type.replace('_', ' ')} information

## Consolidated Content

"""
        
        # Extract and merge common sections
        common_sections = self.extract_common_sections(pattern.similar_documents)
        
        for section_name, section_content in common_sections.items():
            content += f"### {section_name.title()}\n\n{section_content}\n\n"
        
        content += f"""
## Original Documents

The following documents have been consolidated into this unified documentation:

"""
        
        for doc in pattern.similar_documents:
            content += f"### {doc['file'].name}\n\n"
            content += f"**Location:** `{doc['file']}`\n"
            content += f"**Word Count:** {doc['analysis'].get('word_count', 0)}\n"
            content += f"**Sections:** {', '.join(doc['analysis'].get('sections', [])[:5])}\n\n"
        
        content += f"""
---

*This consolidated documentation was generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Consolidation potential: {pattern.consolidation_potential:,} words*
"""
        
        return content
    
    def extract_common_sections(self, documents: List[Dict]) -> Dict[str, str]:
        """Extract common sections from similar documents."""
        common_sections = {}
        
        # Find sections that appear in multiple documents
        all_sections = []
        for doc in documents:
            all_sections.extend(doc['analysis'].get('sections', []))
        
        section_counts = Counter(all_sections)
        common_section_names = [name for name, count in section_counts.items() if count >= 2]
        
        # For each common section, try to extract representative content
        for section_name in common_section_names[:5]:  # Limit to top 5 sections
            common_sections[section_name] = f"*Content consolidated from {section_counts[section_name]} documents with '{section_name}' sections*"
        
        if not common_sections:
            common_sections['overview'] = "*Consolidated content from similar documentation files*"
        
        return common_sections

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze documentation debt and consolidation opportunities')
    parser.add_argument('--analyze', action='store_true', help='Analyze documentation consolidation opportunities')
    parser.add_argument('--consolidate', action='store_true', help='Create consolidated documentation files')
    
    args = parser.parse_args()
    
    analyzer = DocumentationDebtAnalyzer()
    
    print("📚 Documentation Debt Analyzer - Phase 6")
    print("=" * 60)
    
    # Discover documentation files
    doc_files = analyzer.discover_documentation_files()
    
    if not doc_files:
        print("❌ No documentation files found.")
        return
    
    # Analyze content patterns
    content_patterns = analyzer.analyze_documentation_content(doc_files)
    
    # Find similar documents
    similar_patterns = analyzer.find_similar_documents(content_patterns)
    
    if not similar_patterns:
        print("❌ No documentation consolidation patterns found.")
        return
    
    # Generate report
    if args.consolidate:
        print("🚀 Creating consolidated documentation files...")
        report = analyzer.generate_consolidation_report(similar_patterns, dry_run=False)
    else:
        print("📋 Analyzing documentation consolidation opportunities...")
        report = analyzer.generate_consolidation_report(similar_patterns, dry_run=True)
    
    # Print results
    summary = report['summary']
    print(f"\n📊 Documentation Debt Analysis Results:")
    print(f"   📚 {len(doc_files)} total documentation files discovered")
    print(f"   🔍 {summary['total_documentation_patterns']} consolidation patterns found")
    print(f"   📄 {summary['total_files_affected']} files affected")
    print(f"   💰 {summary['total_consolidation_potential']:,} words consolidation potential")
    
    print(f"\n🏆 Top Documentation Consolidation Opportunities:")
    for pattern_type, patterns in summary['top_opportunities']:
        analysis = report['by_type'][pattern_type]
        print(f"   • {pattern_type.title()}: {analysis['count']} patterns, {analysis['total_savings']:,} words savings")
    
    if args.consolidate:
        consolidated_count = len([p for p in similar_patterns if len(p.similar_documents) >= 2])
        print(f"\n✅ Created {consolidated_count} consolidated documentation files")
        print(f"📂 Consolidated documentation in: docs/consolidated/")

if __name__ == "__main__":
    main()