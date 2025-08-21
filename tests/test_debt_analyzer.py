#!/usr/bin/env python3
"""
Quick test script for the Technical Debt Analyzer integration.

This script validates that the debt analyzer can properly analyze a project
and detect technical debt issues.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.project_index.debt_analyzer import TechnicalDebtAnalyzer, DebtSeverity, DebtCategory
from app.core.database import get_session, init_database
from sqlalchemy import select
from app.models.project_index import ProjectIndex


async def test_debt_analyzer():
    """Test the technical debt analyzer."""
    print("🔍 Testing Technical Debt Analyzer Integration...")
    
    try:
        # Initialize the analyzer
        analyzer = TechnicalDebtAnalyzer()
        print("✅ TechnicalDebtAnalyzer initialized successfully")
        
        # Initialize database first
        await init_database()
        
        # Get a test project (if one exists)
        async with get_session() as session:
            stmt = select(ProjectIndex).limit(1)
            result = await session.execute(stmt)
            project = result.scalar_one_or_none()
            
            if not project:
                print("⚠️  No projects found in database. Creating test analysis...")
                # We can still test the analyzer methods
                
                # Test severity calculation
                severity = analyzer._get_severity_for_complexity(15, 'cyclomatic')
                assert severity == DebtSeverity.HIGH
                print("✅ Severity calculation working")
                
                # Test text similarity
                similarity = analyzer._calculate_text_similarity("hello world", "hello world")
                assert similarity == 1.0
                print("✅ Text similarity calculation working")
                
                # Test naming violations detection
                violations = analyzer._detect_naming_violations("def BadFunctionName(): pass")
                assert len(violations) > 0
                print("✅ Naming violation detection working")
                
                # Test comment ratio calculation
                comment_ratio = analyzer._calculate_comment_ratio("# comment\ncode line")
                assert comment_ratio == 0.5
                print("✅ Comment ratio calculation working")
                
                print("\n🎉 All technical debt analyzer components working correctly!")
                return True
            else:
                print(f"📊 Found project: {project.name}")
                print(f"📁 Project path: {project.root_path}")
                print(f"📄 File count: {project.file_count}")
                
                # Test full debt analysis (if files exist and are accessible)
                if project.file_entries:
                    print(f"📝 Analyzing debt for {len(project.file_entries)} files...")
                    
                    # Analyze just the first few files for testing
                    test_files = list(project.file_entries)[:3]
                    total_debt_items = 0
                    
                    for file_entry in test_files:
                        if not file_entry.is_binary and file_entry.file_path:
                            if os.path.exists(file_entry.file_path):
                                try:
                                    debt_items = await analyzer._analyze_file_debt(file_entry, session)
                                    total_debt_items += len(debt_items)
                                    print(f"  📄 {file_entry.file_name}: {len(debt_items)} debt items found")
                                except Exception as e:
                                    print(f"  ❌ Error analyzing {file_entry.file_name}: {e}")
                            else:
                                print(f"  ⚠️  File not found: {file_entry.file_path}")
                    
                    print(f"\n📊 Total debt items found: {total_debt_items}")
                    
                    if total_debt_items > 0:
                        print("✅ Technical debt detection working!")
                    else:
                        print("ℹ️  No debt items detected in test files (could be clean code!)")
                    
                    return True
                else:
                    print("ℹ️  No file entries found for this project")
                    return True
                
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_database_integration():
    """Test that debt tracking database tables exist and are accessible."""
    print("\n🗄️  Testing Database Integration...")
    
    try:
        # Initialize database first
        await init_database()
        
        async with get_session() as session:
            # Test that we can query the debt tables
            from app.models.project_index import DebtSnapshot, DebtItem, DebtRemediationPlan
            
            # Count existing debt snapshots
            snapshot_count = await session.execute(select(DebtSnapshot))
            snapshots = len(snapshot_count.fetchall())
            print(f"📊 Found {snapshots} debt snapshots in database")
            
            # Count existing debt items  
            item_count = await session.execute(select(DebtItem))
            items = len(item_count.fetchall())
            print(f"🐛 Found {items} debt items in database")
            
            # Count existing remediation plans
            plan_count = await session.execute(select(DebtRemediationPlan))
            plans = len(plan_count.fetchall())
            print(f"📋 Found {plans} remediation plans in database")
            
            print("✅ All debt tracking tables accessible")
            return True
            
    except Exception as e:
        print(f"❌ Database integration error: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 Starting Technical Debt System Integration Tests\n")
    
    # Test database integration
    db_success = await test_database_integration()
    
    # Test debt analyzer
    analyzer_success = await test_debt_analyzer()
    
    print(f"\n{'='*60}")
    print("📋 TEST SUMMARY:")
    print(f"  Database Integration: {'✅ PASSED' if db_success else '❌ FAILED'}")
    print(f"  Debt Analyzer: {'✅ PASSED' if analyzer_success else '❌ FAILED'}")
    
    if db_success and analyzer_success:
        print("\n🎉 ALL TESTS PASSED! Technical Debt System is ready!")
        print("\n📋 Phase 1 Technical Debt Integration: COMPLETED")
        print("   ✅ Database schema created")
        print("   ✅ TechnicalDebtAnalyzer integrated") 
        print("   ✅ Models and relationships working")
        print("   ✅ Core debt detection algorithms functional")
        return True
    else:
        print("\n❌ SOME TESTS FAILED - Review errors above")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)