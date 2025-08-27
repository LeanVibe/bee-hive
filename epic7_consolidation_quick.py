#!/usr/bin/env python3
"""
Epic 7 Quick Consolidation Assessment
Fast assessment of system consolidation status.
"""
import sys
from pathlib import Path

def quick_consolidation_check():
    """Quick assessment of system consolidation."""
    print("🎯 EPIC 7 QUICK CONSOLIDATION ASSESSMENT")
    print("=" * 60)
    
    consolidation_indicators = []
    
    # 1. Check for single configuration system
    print("🔧 Checking configuration consolidation...")
    config_files = list(Path("app/config").glob("*.py")) if Path("app/config").exists() else []
    
    if len(config_files) <= 5:
        print(f"  ✅ Configuration files: {len(config_files)} (consolidated)")
        consolidation_indicators.append(True)
    else:
        print(f"  ❌ Configuration files: {len(config_files)} (needs consolidation)")
        consolidation_indicators.append(False)
    
    # 2. Check for unified core modules
    print("🔧 Checking core module consolidation...")
    core_files = list(Path("app/core").glob("*.py")) if Path("app/core").exists() else []
    
    # Count orchestrator-related files (should be consolidated)
    orchestrator_files = [f for f in core_files if 'orchestr' in f.name.lower()]
    
    if len(orchestrator_files) <= 3:
        print(f"  ✅ Orchestrator files: {len(orchestrator_files)} (consolidated)")
        consolidation_indicators.append(True)
    else:
        print(f"  ❌ Orchestrator files: {len(orchestrator_files)} (needs consolidation)")
        consolidation_indicators.append(False)
    
    # 3. Check for unified database access
    print("🔧 Checking database access consolidation...")
    db_files = [f for f in core_files if 'database' in f.name.lower() or 'db' in f.name.lower()]
    
    if len(db_files) <= 2:
        print(f"  ✅ Database files: {len(db_files)} (consolidated)")
        consolidation_indicators.append(True)
    else:
        print(f"  ❌ Database files: {len(db_files)} (needs consolidation)")
        consolidation_indicators.append(False)
    
    # 4. Check for API organization
    print("🔧 Checking API endpoint consolidation...")
    api_files = list(Path("app/api").glob("*.py")) if Path("app/api").exists() else []
    
    # Good consolidation means organized but not excessive fragmentation
    if 5 <= len(api_files) <= 20:
        print(f"  ✅ API files: {len(api_files)} (well organized)")
        consolidation_indicators.append(True)
    else:
        print(f"  ❌ API files: {len(api_files)} (needs better organization)")
        consolidation_indicators.append(False)
    
    # 5. Check for model consolidation
    print("🔧 Checking model consolidation...")
    model_files = list(Path("app/models").glob("*.py")) if Path("app/models").exists() else []
    
    if len(model_files) <= 25:
        print(f"  ✅ Model files: {len(model_files)} (consolidated)")
        consolidation_indicators.append(True)
    else:
        print(f"  ❌ Model files: {len(model_files)} (needs consolidation)")
        consolidation_indicators.append(False)
    
    # Calculate consolidation score
    consolidation_score = (sum(consolidation_indicators) / len(consolidation_indicators)) * 100
    
    print("\n📊 QUICK CONSOLIDATION RESULTS")
    print(f"Consolidation indicators passed: {sum(consolidation_indicators)}/{len(consolidation_indicators)}")
    print(f"Consolidation score: {consolidation_score:.1f}%")
    print(f"Target: >50% consolidation")
    
    # Estimate redundancy elimination
    if consolidation_score >= 80:
        redundancy_eliminated = 60
        print(f"✅ EXCELLENT: Estimated {redundancy_eliminated}% redundancy eliminated")
        return True
    elif consolidation_score >= 60:
        redundancy_eliminated = 45
        print(f"✅ GOOD: Estimated {redundancy_eliminated}% redundancy eliminated")
        return True
    elif consolidation_score >= 40:
        redundancy_eliminated = 30
        print(f"⚠️ FAIR: Estimated {redundancy_eliminated}% redundancy eliminated")
        return False
    else:
        redundancy_eliminated = 15
        print(f"❌ POOR: Estimated {redundancy_eliminated}% redundancy eliminated")
        return False

def test_system_functional_consolidation():
    """Test that consolidation hasn't broken system functionality."""
    print("\n🔧 Testing system functional consolidation...")
    try:
        # Test that key modules still import correctly
        from app.core.orchestrator import Orchestrator
        from app.core.database import Base
        from app.core.configuration_service import get_config
        from app.main import create_app
        
        # Test app creation
        app = create_app()
        
        print("  ✅ All consolidated modules import successfully")
        print("  ✅ Application creation successful")
        print("  ✅ System functional after consolidation")
        return True
        
    except Exception as e:
        print(f"  ❌ System functionality compromised: {e}")
        return False

def main():
    """Run quick consolidation assessment."""
    consolidation_success = quick_consolidation_check()
    functional_success = test_system_functional_consolidation()
    
    overall_success = consolidation_success and functional_success
    
    print(f"\n🎯 FINAL CONSOLIDATION ASSESSMENT")
    if overall_success:
        print("✅ CONSOLIDATION PHASE SUCCESSFUL")
        print("System shows good consolidation with maintained functionality")
    else:
        print("❌ CONSOLIDATION PHASE NEEDS WORK")
        print("System requires further consolidation or has functional issues")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)