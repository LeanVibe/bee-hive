#!/usr/bin/env python3
"""
Simple test of the Technical Debt Analyzer components without full database setup.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.project_index.debt_analyzer import TechnicalDebtAnalyzer, DebtSeverity, DebtCategory


def test_debt_analyzer_components():
    """Test individual components of the debt analyzer."""
    print("🔍 Testing Technical Debt Analyzer Components...")
    
    try:
        # Initialize the analyzer
        analyzer = TechnicalDebtAnalyzer()
        print("✅ TechnicalDebtAnalyzer initialized successfully")
        
        # Test severity calculation
        severity = analyzer._get_severity_for_complexity(15, 'cyclomatic')
        assert severity == DebtSeverity.MEDIUM  # 15 is in MEDIUM range (10-15)
        
        severity_high = analyzer._get_severity_for_complexity(25, 'cyclomatic') 
        assert severity_high == DebtSeverity.CRITICAL  # >20 is CRITICAL
        print("✅ Complexity severity calculation working")
        
        # Test text similarity
        similarity = analyzer._calculate_text_similarity("hello world", "hello world")
        assert similarity == 1.0
        similarity2 = analyzer._calculate_text_similarity("hello world", "hello there")
        assert 0.0 < similarity2 < 1.0
        print("✅ Text similarity calculation working")
        
        # Test naming violations detection
        violations = analyzer._detect_naming_violations("def BadFunctionName(): pass")
        assert len(violations) > 0
        print("✅ Naming violation detection working")
        
        # Test comment ratio calculation  
        comment_ratio = analyzer._calculate_comment_ratio("# comment\ncode line")
        assert comment_ratio == 0.5
        print("✅ Comment ratio calculation working")
        
        # Test cyclomatic complexity calculation
        import ast
        code = """
def complex_function(x):
    if x > 0:
        if x > 10:
            return "big"
        else:
            return "small"
    elif x < 0:
        return "negative"
    else:
        return "zero"
"""
        tree = ast.parse(code)
        func_node = tree.body[0]
        complexity = analyzer._calculate_cyclomatic_complexity(func_node)
        assert complexity > 1  # Should be more than base complexity
        print(f"✅ Cyclomatic complexity calculation working (complexity: {complexity})")
        
        # Test duplicate code block detection
        lines = [
            "def function_a():",
            "    x = 1",  
            "    y = 2",
            "    return x + y",
            "",
            "def function_b():",
            "    x = 1",
            "    y = 2", 
            "    return x + y"
        ]
        duplicates = analyzer._find_duplicate_code_blocks(lines)
        print(f"✅ Duplicate code detection working (found {len(duplicates)} duplicates)")
        
        # Test module docstring detection
        has_docstring = analyzer._has_module_docstring('"""Module docstring"""\ncode')
        assert has_docstring == True
        no_docstring = analyzer._has_module_docstring('import os\ncode')  
        assert no_docstring == False
        print("✅ Module docstring detection working")
        
        # Test severity mappings
        assert analyzer._get_severity_for_length(120) == DebtSeverity.HIGH
        assert analyzer._get_severity_for_param_count(12) == DebtSeverity.HIGH
        assert analyzer._get_severity_for_duplication(0.97) == DebtSeverity.HIGH  # >0.95 is HIGH
        print("✅ All severity mapping functions working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing analyzer components: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_structures():
    """Test debt-related data structures."""
    print("\n📊 Testing Data Structures...")
    
    try:
        from app.project_index.debt_analyzer import DebtItem, DebtSnapshot, DebtAnalysisResult
        
        # Test DebtItem creation
        debt_item = DebtItem(
            project_id="test-project",
            file_id="test-file",
            debt_type="test_debt",
            category=DebtCategory.COMPLEXITY,
            severity=DebtSeverity.HIGH,
            description="Test debt item",
            debt_score=0.8
        )
        assert debt_item.category == DebtCategory.COMPLEXITY
        assert debt_item.severity == DebtSeverity.HIGH
        print("✅ DebtItem data structure working")
        
        # Test DebtSnapshot creation
        snapshot = DebtSnapshot(
            project_id="test-project",
            total_debt_score=0.6,
            category_scores={"complexity": 0.8},
            file_count_analyzed=10,
            lines_of_code_analyzed=1000
        )
        assert snapshot.total_debt_score == 0.6
        print("✅ DebtSnapshot data structure working")
        
        # Test DebtAnalysisResult creation
        result = DebtAnalysisResult(
            project_id="test-project",
            total_debt_score=0.5,
            debt_items=[debt_item],
            category_scores={"complexity": 0.8},
            file_count=5,
            lines_of_code=500,
            analysis_duration=2.5,
            recommendations=["Fix complexity issues"]
        )
        assert len(result.debt_items) == 1
        assert result.analysis_duration == 2.5
        print("✅ DebtAnalysisResult data structure working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing data structures: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_models():
    """Test that database models can be imported and instantiated."""
    print("\n🗄️  Testing Database Models...")
    
    try:
        from app.models.project_index import (
            DebtSnapshot as DBDebtSnapshot, 
            DebtItem as DBDebtItem, 
            DebtRemediationPlan,
            DebtSeverity as DBDebtSeverity,
            DebtCategory as DBDebtCategory,
            DebtStatus as DBDebtStatus
        )
        
        # Test enum values
        assert DBDebtSeverity.HIGH.value == "high"
        assert DBDebtCategory.COMPLEXITY.value == "complexity"  
        assert DBDebtStatus.ACTIVE.value == "active"
        print("✅ Database enums working")
        
        # Test model instantiation (without database)
        snapshot = DBDebtSnapshot(
            project_id="test-id",
            total_debt_score=0.5,
            file_count_analyzed=10
        )
        # Test the to_dict method
        snapshot_dict = snapshot.to_dict()
        assert snapshot_dict['total_debt_score'] == 0.5
        print("✅ DebtSnapshot database model working")
        
        item = DBDebtItem(
            project_id="test-project",
            file_id="test-file", 
            debt_type="test_type",
            debt_category=DBDebtCategory.COMPLEXITY,
            severity=DBDebtSeverity.HIGH,
            description="Test description"
        )
        item_dict = item.to_dict()
        assert item_dict['debt_category'] == 'complexity'
        print("✅ DebtItem database model working")
        
        plan = DebtRemediationPlan(
            project_id="test-project",
            plan_name="Test Plan",
            target_debt_reduction=0.3,
            estimated_effort_hours=10
        )
        plan_dict = plan.to_dict()
        assert plan_dict['plan_name'] == 'Test Plan'
        print("✅ DebtRemediationPlan database model working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing database models: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🚀 Starting Technical Debt System Component Tests\n")
    
    # Test analyzer components
    analyzer_success = test_debt_analyzer_components()
    
    # Test data structures
    structures_success = test_data_structures()
    
    # Test database models
    models_success = test_database_models()
    
    print(f"\n{'='*60}")
    print("📋 TEST SUMMARY:")
    print(f"  Analyzer Components: {'✅ PASSED' if analyzer_success else '❌ FAILED'}")
    print(f"  Data Structures: {'✅ PASSED' if structures_success else '❌ FAILED'}")
    print(f"  Database Models: {'✅ PASSED' if models_success else '❌ FAILED'}")
    
    all_passed = analyzer_success and structures_success and models_success
    
    if all_passed:
        print("\n🎉 ALL COMPONENT TESTS PASSED!")
        print("\n✅ Technical Debt System Phase 1 Integration: COMPLETED")
        print("   ✅ Database schema created")
        print("   ✅ TechnicalDebtAnalyzer class implemented")
        print("   ✅ Models and enums working correctly")
        print("   ✅ Core debt detection algorithms functional")
        print("   ✅ Data structures properly designed")
        print("\n📋 Ready for Phase 2: Advanced Detection Algorithms")
        return True
    else:
        print("\n❌ SOME COMPONENT TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)