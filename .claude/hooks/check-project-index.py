#!/usr/bin/env python3
"""
Pre-file creation hook that checks project index to prevent duplicate logic and overwrites.
This hook is automatically triggered when Claude considers creating a new file.
"""

import os
import sys
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class FileConflictResult:
    """Result of file conflict analysis"""
    should_create: bool
    conflicts: List[Dict[str, Any]]
    recommendations: List[str]
    similar_files: List[str]
    risk_level: str  # low, medium, high
    reasoning: str

class ProjectIndexChecker:
    """Intelligent file creation checker using project index"""
    
    def __init__(self):
        self.project_root = project_root
        self.cache_file = self.project_root / ".claude" / "cache" / "project_index_cache.json"
        self.cache_file.parent.mkdir(exist_ok=True)
        
    async def analyze_file_creation(self, proposed_path: str, purpose: str = "") -> FileConflictResult:
        """
        Analyze whether a new file should be created based on project index.
        
        Args:
            proposed_path: Path where Claude wants to create a file
            purpose: Brief description of what the file is intended for
            
        Returns:
            FileConflictResult with analysis and recommendations
        """
        
        # Convert to absolute path
        if not os.path.isabs(proposed_path):
            proposed_path = str(self.project_root / proposed_path)
        
        proposed_path = os.path.abspath(proposed_path)
            
        # Check if file already exists
        if os.path.exists(proposed_path):
            return FileConflictResult(
                should_create=False,
                conflicts=[{"type": "file_exists", "path": proposed_path}],
                recommendations=[f"File already exists at {proposed_path}. Consider editing existing file instead."],
                similar_files=[proposed_path],
                risk_level="high",
                reasoning="File already exists - would overwrite existing content"
            )
        
        # Get project structure analysis
        similar_files = self._find_similar_files(proposed_path, purpose)
        conflicts = self._detect_potential_conflicts(proposed_path, purpose, similar_files)
        
        # Determine risk level
        risk_level = self._calculate_risk_level(conflicts, similar_files)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            proposed_path, purpose, conflicts, similar_files, risk_level
        )
        
        # Final decision
        should_create = self._should_create_file(risk_level, conflicts)
        
        reasoning = self._generate_reasoning(
            proposed_path, purpose, conflicts, similar_files, risk_level, should_create
        )
        
        return FileConflictResult(
            should_create=should_create,
            conflicts=conflicts,
            recommendations=recommendations,
            similar_files=similar_files,
            risk_level=risk_level,
            reasoning=reasoning
        )
    
    def _find_similar_files(self, proposed_path: str, purpose: str) -> List[str]:
        """Find files with similar names, purposes, or functionality"""
        similar_files = []
        
        proposed_name = Path(proposed_path).name
        proposed_stem = Path(proposed_path).stem
        proposed_dir = Path(proposed_path).parent
        
        # Search for files with similar names
        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden directories and common ignore patterns
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]
            
            for file in files:
                if file.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, file)
                file_stem = Path(file).stem
                
                # Check for similar names
                if (file_stem.lower() == proposed_stem.lower() or 
                    proposed_stem.lower() in file_stem.lower() or
                    file_stem.lower() in proposed_stem.lower()):
                    similar_files.append(file_path)
                
                # Check for files in same directory with similar extensions
                if Path(file_path).parent == proposed_dir:
                    if Path(file_path).suffix == Path(proposed_path).suffix:
                        similar_files.append(file_path)
        
        # Remove duplicates and sort by similarity
        return list(set(similar_files))[:10]  # Limit to top 10 matches
    
    def _detect_potential_conflicts(self, proposed_path: str, purpose: str, similar_files: List[str]) -> List[Dict[str, Any]]:
        """Detect potential conflicts with existing code"""
        conflicts = []
        
        proposed_name = Path(proposed_path).name
        proposed_stem = Path(proposed_path).stem
        
        for similar_file in similar_files:
            similar_stem = Path(similar_file).stem
            
            # Name conflicts
            if similar_stem.lower() == proposed_stem.lower():
                conflicts.append({
                    "type": "name_conflict",
                    "severity": "high",
                    "file": similar_file,
                    "message": f"Similar name to existing file: {similar_file}"
                })
            
            # Directory structure conflicts
            if Path(similar_file).parent == Path(proposed_path).parent:
                conflicts.append({
                    "type": "directory_conflict", 
                    "severity": "medium",
                    "file": similar_file,
                    "message": f"Another file in same directory: {similar_file}"
                })
            
            # Check for potential functionality overlap
            if purpose and self._check_functionality_overlap(similar_file, purpose):
                conflicts.append({
                    "type": "functionality_conflict",
                    "severity": "high", 
                    "file": similar_file,
                    "message": f"Potential duplicate functionality in: {similar_file}"
                })
        
        return conflicts
    
    def _check_functionality_overlap(self, existing_file: str, proposed_purpose: str) -> bool:
        """Check if an existing file might have overlapping functionality"""
        try:
            with open(existing_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Simple keyword matching for now
            purpose_keywords = proposed_purpose.lower().split()
            content_lower = content.lower()
            
            overlap_count = sum(1 for keyword in purpose_keywords 
                              if len(keyword) > 3 and keyword in content_lower)
            
            return overlap_count >= len(purpose_keywords) * 0.5
            
        except Exception:
            return False
    
    def _calculate_risk_level(self, conflicts: List[Dict[str, Any]], similar_files: List[str]) -> str:
        """Calculate overall risk level for file creation"""
        if not conflicts and len(similar_files) <= 1:
            return "low"
        
        high_severity_conflicts = [c for c in conflicts if c.get("severity") == "high"]
        if high_severity_conflicts:
            return "high"
            
        if len(conflicts) > 2 or len(similar_files) > 5:
            return "medium"
            
        return "low"
    
    def _should_create_file(self, risk_level: str, conflicts: List[Dict[str, Any]]) -> bool:
        """Decide whether file should be created"""
        if risk_level == "high":
            # Block high-risk creations
            high_severity = [c for c in conflicts if c.get("severity") == "high"]
            if any(c.get("type") in ["name_conflict", "functionality_conflict"] for c in high_severity):
                return False
        
        return True
    
    def _generate_recommendations(self, proposed_path: str, purpose: str, 
                                conflicts: List[Dict[str, Any]], similar_files: List[str],
                                risk_level: str) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if risk_level == "high":
            recommendations.append("🚨 HIGH RISK: Consider alternatives before creating this file")
        
        # Specific conflict recommendations
        name_conflicts = [c for c in conflicts if c["type"] == "name_conflict"]
        if name_conflicts:
            recommendations.append(f"📝 Consider editing existing file: {name_conflicts[0]['file']}")
            recommendations.append(f"📝 Or use a more specific name to avoid confusion")
        
        functionality_conflicts = [c for c in conflicts if c["type"] == "functionality_conflict"]  
        if functionality_conflicts:
            recommendations.append(f"🔄 Check if functionality already exists in: {functionality_conflicts[0]['file']}")
            recommendations.append(f"🔄 Consider extending existing code instead of duplicating")
        
        # Directory recommendations
        if len(similar_files) > 3:
            recommendations.append(f"📁 Many similar files found - consider organizing in subdirectories")
        
        # General best practices
        if not recommendations:
            recommendations.append("✅ No major conflicts detected - proceed with caution")
            recommendations.append("📋 Consider adding clear documentation for the new file's purpose")
        
        return recommendations
    
    def _generate_reasoning(self, proposed_path: str, purpose: str,
                          conflicts: List[Dict[str, Any]], similar_files: List[str],
                          risk_level: str, should_create: bool) -> str:
        """Generate human-readable reasoning for the decision"""
        
        parts = [
            f"Analysis for creating: {proposed_path}",
            f"Purpose: {purpose or 'Not specified'}",
            f"Risk Level: {risk_level.upper()}",
            f"Similar files found: {len(similar_files)}",
            f"Conflicts detected: {len(conflicts)}"
        ]
        
        if conflicts:
            parts.append("\nKey conflicts:")
            for conflict in conflicts[:3]:  # Show top 3 conflicts
                parts.append(f"  • {conflict['message']}")
        
        if similar_files:
            parts.append(f"\nSimilar files to review:")
            for file in similar_files[:3]:  # Show top 3 similar files
                parts.append(f"  • {os.path.relpath(file, self.project_root)}")
        
        decision = "✅ APPROVED" if should_create else "❌ BLOCKED"
        parts.append(f"\nDecision: {decision}")
        
        return "\n".join(parts)

def format_result_for_claude(result: FileConflictResult, proposed_path: str) -> str:
    """Format the analysis result for Claude to understand"""
    
    status_emoji = "✅" if result.should_create else "❌"
    risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}[result.risk_level]
    
    output = [
        f"\n🔍 PROJECT INDEX FILE ANALYSIS",
        f"{'='*50}",
        f"📁 Proposed file: {os.path.relpath(proposed_path, project_root)}",
        f"{status_emoji} Decision: {'APPROVED' if result.should_create else 'BLOCKED'}",
        f"{risk_emoji} Risk Level: {result.risk_level.upper()}",
        f""
    ]
    
    if result.conflicts:
        output.append("⚠️  CONFLICTS DETECTED:")
        for conflict in result.conflicts:
            severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(conflict.get("severity", "low"), "⚪")
            conflict_file = os.path.relpath(conflict.get("file", ""), project_root)
            output.append(f"  {severity_emoji} {conflict['message']}")
            output.append(f"     File: {conflict_file}")
        output.append("")
    
    if result.similar_files:
        output.append(f"📋 SIMILAR FILES ({len(result.similar_files)} found):")
        for file in result.similar_files[:5]:  # Show top 5
            rel_path = os.path.relpath(file, project_root)
            output.append(f"  📄 {rel_path}")
        if len(result.similar_files) > 5:
            output.append(f"  ... and {len(result.similar_files) - 5} more")
        output.append("")
    
    if result.recommendations:
        output.append("💡 RECOMMENDATIONS:")
        for rec in result.recommendations:
            output.append(f"  {rec}")
        output.append("")
    
    output.append("🤖 REASONING:")
    for line in result.reasoning.split('\n'):
        if line.strip():
            output.append(f"  {line}")
    
    output.extend([
        "",
        f"{'='*50}",
        "💡 To override this analysis, use: --force-create flag",
        ""
    ])
    
    return "\n".join(output)

async def main():
    """Main entry point for the hook"""
    
    if len(sys.argv) < 2:
        print("Usage: check-project-index.py <proposed_file_path> [purpose]")
        sys.exit(1)
    
    proposed_path = sys.argv[1]
    purpose = sys.argv[2] if len(sys.argv) > 2 else ""
    
    # Check for force flag
    force_create = "--force-create" in sys.argv
    
    if force_create:
        print("🚨 FORCE CREATE FLAG DETECTED - Skipping project index analysis")
        sys.exit(0)
    
    try:
        checker = ProjectIndexChecker()
        result = await checker.analyze_file_creation(proposed_path, purpose)
        
        # Print formatted result
        print(format_result_for_claude(result, proposed_path))
        
        # Exit with appropriate code
        if result.should_create:
            sys.exit(0)  # Success - allow creation
        else:
            sys.exit(1)  # Block creation
            
    except Exception as e:
        print(f"❌ Error in project index analysis: {e}")
        print("⚠️  Proceeding without analysis due to error")
        sys.exit(0)  # Allow creation on error (fail-safe)

if __name__ == "__main__":
    asyncio.run(main())