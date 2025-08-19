#!/usr/bin/env python3
"""
Simple pre-file creation hook that checks for existing files and conflicts.
"""

import os
import sys
from pathlib import Path

def check_file_creation(proposed_path: str, purpose: str = ""):
    """Check if a file should be created"""
    
    # Convert to absolute path
    if not os.path.isabs(proposed_path):
        project_root = Path(__file__).parent.parent.parent
        proposed_path = str(project_root / proposed_path)
    
    proposed_path = os.path.abspath(proposed_path)
    
    print(f"\n🔍 FILE CREATION ANALYSIS")
    print(f"{'='*50}")
    print(f"📁 Proposed file: {os.path.relpath(proposed_path)}")
    print(f"💭 Purpose: {purpose or 'Not specified'}")
    
    # Check if file already exists
    if os.path.exists(proposed_path):
        print(f"❌ Decision: BLOCKED")
        print(f"🔴 Risk Level: HIGH") 
        print(f"⚠️  FILE ALREADY EXISTS")
        print(f"💡 RECOMMENDATION: Edit existing file instead of overwriting")
        print(f"{'='*50}")
        return False
    
    # Look for similar files in the same directory
    directory = os.path.dirname(proposed_path)
    filename = os.path.basename(proposed_path)
    name_stem = os.path.splitext(filename)[0]
    
    similar_files = []
    conflicts = []
    
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.startswith('.'):
                continue
                
            file_stem = os.path.splitext(file)[0]
            
            # Check for similar names
            if (file_stem.lower() == name_stem.lower() and file != filename):
                conflicts.append(f"Exact name match: {file}")
            elif name_stem.lower() in file_stem.lower() or file_stem.lower() in name_stem.lower():
                similar_files.append(file)
    
    # Determine risk level
    if conflicts:
        risk_level = "HIGH"
        decision = "REVIEW REQUIRED"
        emoji = "🔴"
    elif len(similar_files) > 3:
        risk_level = "MEDIUM" 
        decision = "APPROVED"
        emoji = "🟡"
    elif similar_files:
        risk_level = "LOW"
        decision = "APPROVED"
        emoji = "🟢"
    else:
        risk_level = "LOW"
        decision = "APPROVED"
        emoji = "🟢"
    
    print(f"✅ Decision: {decision}")
    print(f"{emoji} Risk Level: {risk_level}")
    
    if conflicts:
        print(f"\n⚠️  NAME CONFLICTS:")
        for conflict in conflicts:
            print(f"  🔴 {conflict}")
    
    if similar_files:
        print(f"\n📋 SIMILAR FILES ({len(similar_files)} found):")
        for file in similar_files[:5]:
            print(f"  📄 {file}")
        if len(similar_files) > 5:
            print(f"  ... and {len(similar_files) - 5} more")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if conflicts:
        print(f"  🚨 Review naming conflicts before proceeding")
        print(f"  📝 Consider using a more specific name")
    elif risk_level == "MEDIUM":
        print(f"  📁 Consider organizing similar files in subdirectories") 
    else:
        print(f"  ✅ No major conflicts detected - safe to proceed")
    
    print(f"{'='*50}\n")
    
    # Allow creation unless there are exact name conflicts
    return len(conflicts) == 0

def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        print("Usage: simple-file-check.py <proposed_file_path> [purpose]")
        sys.exit(1)
    
    proposed_path = sys.argv[1]
    purpose = sys.argv[2] if len(sys.argv) > 2 else ""
    
    # Check for force flag
    if "--force-create" in sys.argv:
        print("🚨 FORCE CREATE FLAG - Skipping analysis")
        sys.exit(0)
    
    try:
        should_create = check_file_creation(proposed_path, purpose)
        sys.exit(0 if should_create else 1)
    except Exception as e:
        print(f"❌ Error in file analysis: {e}")
        print("⚠️  Proceeding without analysis due to error")
        sys.exit(0)  # Fail-safe: allow creation on error

if __name__ == "__main__":
    main()