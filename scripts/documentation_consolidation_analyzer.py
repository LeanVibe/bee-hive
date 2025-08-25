#!/usr/bin/env python3
"""
Documentation Consolidation Analyzer and Roadmap Generator

Analyzes the massive documentation ecosystem (835+ files) and creates a strategic
consolidation roadmap to reduce redundancy while preserving valuable content.

Features:
- Comprehensive documentation discovery and categorization
- Content similarity analysis and redundancy detection
- Strategic consolidation roadmap with prioritized actions
- Automated merge opportunity identification
- Preservation strategy for unique high-value content
"""

import asyncio
import json
import logging
import os
import re
import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import difflib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentationConsolidationAnalyzer:
    """Comprehensive analyzer for documentation consolidation opportunities."""
    
    def __init__(self, project_root: str = "/Users/bogdan/work/leanvibe-dev/bee-hive"):
        self.project_root = Path(project_root)
        self.all_docs = []
        self.analysis_cache = {}
        
        # Document categories for strategic organization
        self.strategic_categories = {
            "core_strategic": {
                "keywords": ["PLAN", "PROMPT", "ROADMAP", "STRATEGY", "VISION"],
                "priority": "critical",
                "consolidation_approach": "preserve_all"
            },
            "architecture": {
                "keywords": ["ARCHITECTURE", "TECHNICAL", "DESIGN", "SYSTEM", "CORE"],
                "priority": "high",
                "consolidation_approach": "merge_similar"
            },
            "implementation": {
                "keywords": ["IMPLEMENTATION", "COMPLETE", "MISSION", "PHASE", "EPIC"],
                "priority": "medium",
                "consolidation_approach": "timeline_based"
            },
            "operational": {
                "keywords": ["DEPLOYMENT", "OPERATION", "RUNBOOK", "GUIDE", "SETUP"],
                "priority": "high",
                "consolidation_approach": "merge_by_function"
            },
            "testing": {
                "keywords": ["TEST", "VALIDATION", "BENCHMARK", "QA"],
                "priority": "medium", 
                "consolidation_approach": "merge_by_type"
            },
            "documentation_meta": {
                "keywords": ["README", "DOCS", "DOCUMENTATION", "INDEX", "MANIFEST"],
                "priority": "high",
                "consolidation_approach": "hierarchical_organization"
            },
            "reports_status": {
                "keywords": ["REPORT", "ANALYSIS", "AUDIT", "STATUS", "SUMMARY"],
                "priority": "low",
                "consolidation_approach": "archive_old"
            },
            "legacy_deprecated": {
                "keywords": ["OLD", "DEPRECATED", "BACKUP", "TEMP", "DRAFT"],
                "priority": "low",
                "consolidation_approach": "safe_removal"
            }
        }

    async def discover_all_documentation(self) -> Dict[str, Any]:
        """Discover and catalog all documentation files in the project."""
        logger.info("Discovering all documentation files...")
        
        discovery_result = {
            "discovery_timestamp": datetime.now().isoformat(),
            "file_patterns": ["*.md", "*.rst", "*.txt", "*.adoc"],
            "total_files": 0,
            "files_by_category": {},
            "files_by_directory": {},
            "size_analysis": {},
            "content_analysis": {}
        }
        
        # Find all documentation files
        all_docs = []
        for pattern in discovery_result["file_patterns"]:
            docs = list(self.project_root.rglob(pattern))
            all_docs.extend(docs)
        
        # Remove duplicates and filter
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc not in seen and doc.is_file():
                seen.add(doc)
                unique_docs.append(doc)
        
        discovery_result["total_files"] = len(unique_docs)
        self.all_docs = unique_docs
        
        # Analyze each document
        logger.info(f"Analyzing {len(unique_docs)} documentation files...")
        
        categorized_files = defaultdict(list)
        directory_files = defaultdict(list) 
        total_size = 0
        
        for doc in unique_docs:
            # Get basic file info
            file_info = await self._analyze_single_document(doc)
            total_size += file_info["size_bytes"]
            
            # Categorize by strategic category
            category = self._categorize_document(doc)
            categorized_files[category].append(file_info)
            
            # Group by directory
            rel_dir = str(doc.parent.relative_to(self.project_root))
            directory_files[rel_dir].append(file_info)
        
        # Compile results
        discovery_result["files_by_category"] = {
            category: {
                "count": len(files),
                "files": files,
                "total_size": sum(f["size_bytes"] for f in files)
            }
            for category, files in categorized_files.items()
        }
        
        discovery_result["files_by_directory"] = {
            directory: {
                "count": len(files),
                "files": files,
                "total_size": sum(f["size_bytes"] for f in files)
            }
            for directory, files in directory_files.items()
        }
        
        discovery_result["size_analysis"] = {
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "average_file_size": round(total_size / len(unique_docs)) if unique_docs else 0,
            "largest_files": sorted(
                [{"path": f["relative_path"], "size": f["size_bytes"]} 
                 for category_files in categorized_files.values() 
                 for f in category_files],
                key=lambda x: x["size"], reverse=True
            )[:10]
        }
        
        return discovery_result

    async def _analyze_single_document(self, doc_path: Path) -> Dict[str, Any]:
        """Analyze a single document for metadata and characteristics."""
        file_info = {
            "path": str(doc_path),
            "name": doc_path.name,
            "relative_path": str(doc_path.relative_to(self.project_root)),
            "directory": str(doc_path.parent.relative_to(self.project_root)),
            "size_bytes": 0,
            "lines": 0,
            "words": 0,
            "last_modified": "",
            "content_hash": "",
            "headings": [],
            "keywords": [],
            "creation_date_estimate": None
        }
        
        try:
            # Basic file stats
            stat = doc_path.stat()
            file_info["size_bytes"] = stat.st_size
            file_info["last_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            
            # Read content
            with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Content analysis
            lines = content.split('\n')
            file_info["lines"] = len(lines)
            file_info["words"] = len(content.split())
            file_info["content_hash"] = hashlib.md5(content.encode()).hexdigest()
            
            # Extract headings
            headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
            file_info["headings"] = headings[:10]  # First 10 headings
            
            # Extract keywords from filename and content
            filename_words = re.findall(r'\w+', doc_path.stem.upper())
            content_keywords = re.findall(r'\b[A-Z]{2,}\b', content)  # All caps words
            file_info["keywords"] = list(set(filename_words + content_keywords[:20]))
            
            # Try to extract creation date from content
            date_patterns = [
                r'(?:Date|Created|Updated):\s*(\d{4}-\d{2}-\d{2})',
                r'(\d{4}-\d{2}-\d{2})'
            ]
            for pattern in date_patterns:
                match = re.search(pattern, content[:1000])  # Check first 1000 chars
                if match:
                    file_info["creation_date_estimate"] = match.group(1)
                    break
                    
        except Exception as e:
            file_info["analysis_error"] = str(e)
        
        return file_info

    def _categorize_document(self, doc_path: Path) -> str:
        """Categorize a document based on filename and path patterns."""
        path_upper = str(doc_path).upper()
        name_upper = doc_path.name.upper()
        
        # Check each strategic category
        for category, config in self.strategic_categories.items():
            if any(keyword in path_upper or keyword in name_upper 
                   for keyword in config["keywords"]):
                return category
        
        # Default category
        return "uncategorized"

    async def analyze_content_similarity(self) -> Dict[str, Any]:
        """Analyze content similarity to identify merge opportunities."""
        logger.info("Analyzing content similarity for merge opportunities...")
        
        similarity_analysis = {
            "analysis_timestamp": datetime.now().isoformat(),
            "similar_groups": [],
            "duplicate_candidates": [],
            "merge_opportunities": [],
            "content_clusters": {}
        }
        
        if not self.all_docs:
            await self.discover_all_documentation()
        
        # Group documents by category for focused analysis
        docs_by_category = defaultdict(list)
        for doc in self.all_docs:
            category = self._categorize_document(doc)
            docs_by_category[category].append(doc)
        
        # Analyze similarity within each category
        for category, docs in docs_by_category.items():
            if len(docs) < 2:
                continue
            
            logger.info(f"Analyzing similarity in category: {category} ({len(docs)} docs)")
            category_analysis = await self._analyze_category_similarity(category, docs)
            
            similarity_analysis["similar_groups"].extend(category_analysis["similar_groups"])
            similarity_analysis["duplicate_candidates"].extend(category_analysis["duplicate_candidates"])
            similarity_analysis["merge_opportunities"].extend(category_analysis["merge_opportunities"])
        
        return similarity_analysis

    async def _analyze_category_similarity(self, category: str, docs: List[Path]) -> Dict[str, Any]:
        """Analyze similarity within a specific category."""
        category_analysis = {
            "category": category,
            "similar_groups": [],
            "duplicate_candidates": [],
            "merge_opportunities": []
        }
        
        # Compare documents pairwise
        for i, doc1 in enumerate(docs):
            for doc2 in docs[i+1:]:
                try:
                    similarity_score = await self._calculate_document_similarity(doc1, doc2)
                    
                    if similarity_score > 0.9:  # Very high similarity - potential duplicates
                        category_analysis["duplicate_candidates"].append({
                            "doc1": str(doc1.relative_to(self.project_root)),
                            "doc2": str(doc2.relative_to(self.project_root)),
                            "similarity": similarity_score,
                            "recommendation": "Consider removing duplicate"
                        })
                    
                    elif similarity_score > 0.7:  # High similarity - merge candidates
                        category_analysis["merge_opportunities"].append({
                            "doc1": str(doc1.relative_to(self.project_root)),
                            "doc2": str(doc2.relative_to(self.project_root)),
                            "similarity": similarity_score,
                            "recommendation": "Consider merging similar content"
                        })
                    
                    elif similarity_score > 0.5:  # Moderate similarity - group for review
                        category_analysis["similar_groups"].append({
                            "doc1": str(doc1.relative_to(self.project_root)),
                            "doc2": str(doc2.relative_to(self.project_root)),
                            "similarity": similarity_score,
                            "recommendation": "Review for potential consolidation"
                        })
                        
                except Exception as e:
                    logger.warning(f"Error comparing {doc1.name} and {doc2.name}: {e}")
        
        return category_analysis

    async def _calculate_document_similarity(self, doc1: Path, doc2: Path) -> float:
        """Calculate similarity score between two documents."""
        try:
            # Read both documents
            with open(doc1, 'r', encoding='utf-8', errors='ignore') as f:
                content1 = f.read()
            
            with open(doc2, 'r', encoding='utf-8', errors='ignore') as f:
                content2 = f.read()
            
            # Normalize content (remove extra whitespace, convert to lowercase)
            content1_norm = ' '.join(content1.lower().split())
            content2_norm = ' '.join(content2.lower().split())
            
            # Calculate similarity using difflib
            similarity = difflib.SequenceMatcher(None, content1_norm, content2_norm).ratio()
            
            return similarity
            
        except Exception:
            return 0.0

    async def generate_consolidation_roadmap(self) -> Dict[str, Any]:
        """Generate comprehensive consolidation roadmap with strategic priorities."""
        logger.info("Generating documentation consolidation roadmap...")
        
        # Ensure we have discovery and similarity analysis
        discovery = await self.discover_all_documentation()
        similarity = await self.analyze_content_similarity()
        
        roadmap = {
            "roadmap_timestamp": datetime.now().isoformat(),
            "current_state": {
                "total_files": discovery["total_files"],
                "total_size_mb": discovery["size_analysis"]["total_size_mb"],
                "categories": len(discovery["files_by_category"]),
                "directories": len(discovery["files_by_directory"])
            },
            "consolidation_goals": {
                "target_file_reduction": "60-70%",  # Aggressive but achievable
                "target_final_count": f"{discovery['total_files'] // 3}-{discovery['total_files'] // 2}",
                "preserve_content_value": "95%+",
                "improve_findability": "90%+"
            },
            "phase_1_immediate": await self._generate_phase_1_actions(discovery, similarity),
            "phase_2_strategic": await self._generate_phase_2_actions(discovery, similarity),
            "phase_3_optimization": await self._generate_phase_3_actions(discovery, similarity),
            "preservation_strategy": await self._generate_preservation_strategy(discovery),
            "implementation_timeline": await self._generate_implementation_timeline(discovery, similarity)
        }
        
        # Save roadmap
        roadmap_path = self.project_root / "docs" / "DOCUMENTATION_CONSOLIDATION_ROADMAP.md"
        await self._generate_roadmap_document(roadmap, roadmap_path)
        
        # Save detailed analysis
        analysis_path = self.project_root / "reports" / f"documentation_consolidation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        analysis_path.parent.mkdir(exist_ok=True)
        
        detailed_analysis = {
            "discovery": discovery,
            "similarity": similarity,
            "roadmap": roadmap
        }
        
        with open(analysis_path, 'w') as f:
            json.dump(detailed_analysis, f, indent=2)
        
        logger.info(f"Consolidation roadmap generated: {roadmap_path}")
        logger.info(f"Detailed analysis saved: {analysis_path}")
        
        return roadmap

    async def _generate_phase_1_actions(self, discovery: Dict[str, Any], similarity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Phase 1 immediate consolidation actions."""
        phase_1 = {
            "title": "Phase 1: Immediate Wins (Week 1)",
            "objective": "Remove obvious duplicates and organize critical documents",
            "actions": [],
            "estimated_reduction": "30-40%",
            "risk_level": "low"
        }
        
        # Identify safe removal candidates
        reports_category = discovery["files_by_category"].get("reports_status", {})
        if reports_category.get("count", 0) > 10:
            phase_1["actions"].append({
                "action": "Consolidate status reports and analysis files",
                "description": f"Merge {reports_category['count']} report files into 3-5 current status documents",
                "files_affected": reports_category["count"],
                "impact": "high",
                "effort": "medium"
            })
        
        # Handle duplicates
        duplicate_count = len(similarity.get("duplicate_candidates", []))
        if duplicate_count > 0:
            phase_1["actions"].append({
                "action": "Remove duplicate documents",
                "description": f"Remove {duplicate_count} identified duplicate files after content verification",
                "files_affected": duplicate_count,
                "impact": "high",
                "effort": "low"
            })
        
        # Organize README files
        readme_files = [
            doc for category_data in discovery["files_by_category"].values()
            for doc in category_data.get("files", [])
            if "README" in doc["name"].upper()
        ]
        
        if len(readme_files) > 10:
            phase_1["actions"].append({
                "action": "Consolidate README files",
                "description": f"Merge {len(readme_files)} README files into directory-specific guides",
                "files_affected": len(readme_files),
                "impact": "medium",
                "effort": "medium"
            })
        
        return phase_1

    async def _generate_phase_2_actions(self, discovery: Dict[str, Any], similarity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Phase 2 strategic consolidation actions."""
        phase_2 = {
            "title": "Phase 2: Strategic Consolidation (Week 2-3)",
            "objective": "Merge related content and create unified documents",
            "actions": [],
            "estimated_reduction": "20-30%",
            "risk_level": "medium"
        }
        
        # Architecture document consolidation
        arch_category = discovery["files_by_category"].get("architecture", {})
        if arch_category.get("count", 0) > 5:
            phase_2["actions"].append({
                "action": "Consolidate architecture documents",
                "description": f"Merge {arch_category['count']} architecture files into unified system documentation",
                "files_affected": arch_category["count"],
                "impact": "high",
                "effort": "high"
            })
        
        # Implementation/completion reports
        impl_category = discovery["files_by_category"].get("implementation", {})
        if impl_category.get("count", 0) > 8:
            phase_2["actions"].append({
                "action": "Create implementation timeline document",
                "description": f"Consolidate {impl_category['count']} implementation reports into chronological timeline",
                "files_affected": impl_category["count"],
                "impact": "medium",
                "effort": "high"
            })
        
        # Merge similar documents
        merge_candidates = len(similarity.get("merge_opportunities", []))
        if merge_candidates > 0:
            phase_2["actions"].append({
                "action": "Merge similar content",
                "description": f"Intelligently merge {merge_candidates} pairs of similar documents",
                "files_affected": merge_candidates,
                "impact": "high", 
                "effort": "high"
            })
        
        return phase_2

    async def _generate_phase_3_actions(self, discovery: Dict[str, Any], similarity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Phase 3 optimization actions."""
        phase_3 = {
            "title": "Phase 3: Organization & Optimization (Week 4)",
            "objective": "Create hierarchical structure and living documentation",
            "actions": [],
            "estimated_reduction": "10-15%",
            "risk_level": "low"
        }
        
        phase_3["actions"].extend([
            {
                "action": "Create hierarchical documentation index",
                "description": "Build master index with logical navigation paths",
                "files_affected": 1,  # New file
                "impact": "high",
                "effort": "medium"
            },
            {
                "action": "Implement living documentation templates",
                "description": "Convert static docs to auto-updating living documents",
                "files_affected": len(discovery["files_by_category"].get("core_strategic", {}).get("files", [])),
                "impact": "high",
                "effort": "high"
            },
            {
                "action": "Archive historical documents",
                "description": "Move outdated documents to archive with preservation links",
                "files_affected": discovery["total_files"] // 10,  # ~10% estimated
                "impact": "medium",
                "effort": "low"
            }
        ])
        
        return phase_3

    async def _generate_preservation_strategy(self, discovery: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy for preserving valuable content during consolidation."""
        return {
            "critical_documents": [
                "PLAN.md", "PROMPT.md", "README.md", "ARCHITECTURE.md",
                "API_REFERENCE.md", "GETTING_STARTED.md"
            ],
            "preservation_rules": [
                "Always backup before consolidation",
                "Preserve unique technical details even in merges",
                "Maintain git history for all changes",
                "Create redirect index for moved content",
                "Preserve authorship and creation dates"
            ],
            "content_verification": [
                "Verify no critical information lost in merges",
                "Test all preserved links and references",
                "Validate technical accuracy after consolidation",
                "Ensure searchability of consolidated content"
            ]
        }

    async def _generate_implementation_timeline(self, discovery: Dict[str, Any], similarity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate implementation timeline with milestones."""
        return {
            "total_duration": "4 weeks",
            "phases": [
                {
                    "phase": "Phase 1",
                    "duration": "1 week",
                    "milestone": f"Reduce from {discovery['total_files']} to {discovery['total_files'] * 0.6:.0f} files",
                    "success_criteria": ["All duplicates removed", "Critical docs organized", "No content loss"]
                },
                {
                    "phase": "Phase 2", 
                    "duration": "2 weeks",
                    "milestone": f"Reduce to {discovery['total_files'] * 0.4:.0f} files with strategic consolidation",
                    "success_criteria": ["Related content merged", "Navigation improved", "Quality maintained"]
                },
                {
                    "phase": "Phase 3",
                    "duration": "1 week", 
                    "milestone": f"Final organization with {discovery['total_files'] * 0.3:.0f} optimized files",
                    "success_criteria": ["Living docs active", "Archive complete", "Quality gates passing"]
                }
            ],
            "resources_required": [
                "Documentation architect (full-time)",
                "Content reviewer (part-time)",
                "Quality assurance validation"
            ],
            "risk_mitigation": [
                "Comprehensive backup strategy",
                "Incremental implementation with rollback",
                "Stakeholder approval for major changes"
            ]
        }

    async def _generate_roadmap_document(self, roadmap: Dict[str, Any], output_path: Path):
        """Generate the consolidation roadmap markdown document."""
        
        content = f"""# Documentation Consolidation Roadmap
*Generated: {roadmap['roadmap_timestamp']}*

## Executive Summary

**Current State**: {roadmap['current_state']['total_files']} documentation files ({roadmap['current_state']['total_size_mb']} MB)
**Target State**: {roadmap['consolidation_goals']['target_final_count']} strategically organized files
**Reduction Goal**: {roadmap['consolidation_goals']['target_file_reduction']} reduction while preserving {roadmap['consolidation_goals']['preserve_content_value']} content value

## Strategic Consolidation Plan

### {roadmap['phase_1_immediate']['title']}
**Objective**: {roadmap['phase_1_immediate']['objective']}
**Estimated Reduction**: {roadmap['phase_1_immediate']['estimated_reduction']}

"""
        
        for action in roadmap['phase_1_immediate']['actions']:
            content += f"- **{action['action']}**: {action['description']} ({action['files_affected']} files affected)\n"
        
        content += f"""
### {roadmap['phase_2_strategic']['title']}
**Objective**: {roadmap['phase_2_strategic']['objective']}
**Estimated Reduction**: {roadmap['phase_2_strategic']['estimated_reduction']}

"""
        
        for action in roadmap['phase_2_strategic']['actions']:
            content += f"- **{action['action']}**: {action['description']} ({action['files_affected']} files affected)\n"
        
        content += f"""
### {roadmap['phase_3_optimization']['title']}
**Objective**: {roadmap['phase_3_optimization']['objective']}
**Estimated Reduction**: {roadmap['phase_3_optimization']['estimated_reduction']}

"""
        
        for action in roadmap['phase_3_optimization']['actions']:
            content += f"- **{action['action']}**: {action['description']} ({action['files_affected']} files affected)\n"
        
        content += f"""
## Implementation Timeline

**Total Duration**: {roadmap['implementation_timeline']['total_duration']}

"""
        
        for phase in roadmap['implementation_timeline']['phases']:
            content += f"### {phase['phase']} ({phase['duration']})\n"
            content += f"**Milestone**: {phase['milestone']}\n\n"
            content += "**Success Criteria**:\n"
            for criteria in phase['success_criteria']:
                content += f"- {criteria}\n"
            content += "\n"
        
        content += f"""
## Preservation Strategy

### Critical Documents (Never Remove)
"""
        
        for doc in roadmap['preservation_strategy']['critical_documents']:
            content += f"- {doc}\n"
        
        content += f"""
### Preservation Rules
"""
        
        for rule in roadmap['preservation_strategy']['preservation_rules']:
            content += f"- {rule}\n"
        
        content += f"""
## Success Metrics

- **File Count Reduction**: {roadmap['current_state']['total_files']} → {roadmap['consolidation_goals']['target_final_count']}
- **Content Value Preservation**: {roadmap['consolidation_goals']['preserve_content_value']}
- **Findability Improvement**: {roadmap['consolidation_goals']['improve_findability']}
- **Maintenance Effort Reduction**: 70% less documentation maintenance overhead

## Risk Mitigation

"""
        
        for risk in roadmap['implementation_timeline']['risk_mitigation']:
            content += f"- {risk}\n"
        
        content += f"""
---

*This roadmap was generated by the Documentation Consolidation Analyzer*
*For detailed analysis and implementation support, see accompanying analysis report*
"""
        
        # Write the roadmap document
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)


async def main():
    """Main function to run documentation consolidation analysis."""
    analyzer = DocumentationConsolidationAnalyzer()
    
    print("Documentation Consolidation Analysis")
    print("===================================")
    
    # Generate comprehensive roadmap
    print("Generating consolidation roadmap...")
    roadmap = await analyzer.generate_consolidation_roadmap()
    
    # Display summary
    current = roadmap["current_state"]
    goals = roadmap["consolidation_goals"]
    
    print(f"\nCurrent State:")
    print(f"  Total Files: {current['total_files']}")
    print(f"  Total Size: {current['total_size_mb']} MB")
    print(f"  Categories: {current['categories']}")
    print(f"  Directories: {current['directories']}")
    
    print(f"\nConsolidation Goals:")
    print(f"  Target Reduction: {goals['target_file_reduction']}")
    print(f"  Final File Count: {goals['target_final_count']}")
    print(f"  Content Preservation: {goals['preserve_content_value']}")
    print(f"  Findability Improvement: {goals['improve_findability']}")
    
    print(f"\nImplementation Plan:")
    for phase in roadmap["implementation_timeline"]["phases"]:
        print(f"  {phase['phase']}: {phase['milestone']} ({phase['duration']})")
    
    print(f"\nRoadmap generated successfully!")
    print(f"See: docs/DOCUMENTATION_CONSOLIDATION_ROADMAP.md")


if __name__ == "__main__":
    asyncio.run(main())