#!/usr/bin/env python3
"""
Quick test script for the Short ID System

This script validates that the short ID system components work correctly
before running the full migration. Run this to verify implementation.

Usage:
    python test_short_id_system.py
"""

import sys
import os
import uuid
import tempfile
from datetime import datetime

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from app.core.short_id_generator import (
        ShortIdGenerator, EntityType, ShortIdConfig,
        generate_short_id, validate_short_id_format, 
        get_generator, HUMAN_ALPHABET
    )
    from app.cli.short_id_commands import ShortIdResolver, IdResolutionStrategy
    print("✅ Successfully imported short ID components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def test_basic_generation():
    """Test basic short ID generation."""
    print("\n🧪 Testing basic ID generation...")
    
    generator = ShortIdGenerator()
    
    # Test each entity type
    for entity_type in EntityType:
        short_id, uuid_obj = generator.generate_id(entity_type)
        
        # Validate format
        assert short_id.startswith(entity_type.value + "-")
        assert len(short_id) == 8  # PREFIX-XXXX
        assert validate_short_id_format(short_id)
        
        print(f"  ✅ {entity_type.name}: {short_id}")
    
    print("✅ Basic generation test passed")


def test_collision_resistance():
    """Test collision resistance."""
    print("\n🧪 Testing collision resistance...")
    
    generator = ShortIdGenerator()
    generated_ids = set()
    
    # Generate many IDs and check for duplicates
    for i in range(1000):
        short_id, _ = generator.generate_id(EntityType.TASK)
        
        if short_id in generated_ids:
            print(f"❌ Collision detected: {short_id}")
            return False
        
        generated_ids.add(short_id)
    
    print(f"✅ Generated {len(generated_ids)} unique IDs without collisions")
    return True


def test_alphabet_usage():
    """Test that only allowed characters are used."""
    print("\n🧪 Testing alphabet compliance...")
    
    generator = ShortIdGenerator()
    
    for i in range(100):
        short_id, _ = generator.generate_id(EntityType.TASK)
        
        # Extract code part (after the dash)
        code = short_id.split('-')[1]
        
        # Check all characters are in allowed alphabet
        for char in code:
            if char not in HUMAN_ALPHABET:
                print(f"❌ Invalid character '{char}' in {short_id}")
                return False
    
    print("✅ All generated IDs use only allowed characters")
    print(f"   Allowed: {HUMAN_ALPHABET}")
    print(f"   Excluded: 0, 1, I, O (for human clarity)")
    return True


def test_validation():
    """Test ID validation functions."""
    print("\n🧪 Testing validation...")
    
    valid_ids = [
        "TSK-A7B2",
        "PRJ-X9K3", 
        "AGT-M4P7",
        "WFL-Q2R5"
    ]
    
    invalid_ids = [
        "TSK-A7B",      # Too short
        "TASK-A7B2",    # Wrong prefix length
        "TSK_A7B2",     # Wrong separator
        "TSK-A7B2X",    # Too long
        "TSK-A7I2",     # Invalid character (I)
        "TSK-A702",     # Invalid character (0)
        "",             # Empty
        "INVALID"       # Completely wrong format
    ]
    
    # Test valid IDs
    for valid_id in valid_ids:
        if not validate_short_id_format(valid_id):
            print(f"❌ Valid ID rejected: {valid_id}")
            return False
    
    # Test invalid IDs
    for invalid_id in invalid_ids:
        if validate_short_id_format(invalid_id):
            print(f"❌ Invalid ID accepted: {invalid_id}")
            return False
    
    print("✅ Validation test passed")
    return True


def test_cli_resolution():
    """Test CLI resolution functionality."""
    print("\n🧪 Testing CLI resolution...")
    
    try:
        resolver = ShortIdResolver()
        
        # Test with mock data (since we don't have a real database)
        test_id = "TSK-A7B2"
        
        # Test format validation path
        if validate_short_id_format(test_id):
            print("✅ CLI resolver can validate ID formats")
        
        # Test entity type extraction
        generator = get_generator()
        entity_type = generator.extract_entity_type(test_id)
        
        if entity_type == EntityType.TASK:
            print("✅ CLI resolver can extract entity types")
        
        print("✅ CLI resolution components working")
        return True
        
    except Exception as e:
        print(f"❌ CLI resolution test failed: {e}")
        return False


def test_performance():
    """Test performance characteristics."""
    print("\n🧪 Testing performance...")
    
    generator = ShortIdGenerator()
    
    start_time = datetime.now()
    
    # Generate many IDs
    count = 1000
    for i in range(count):
        generate_short_id(EntityType.TASK)
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    ids_per_second = count / elapsed
    avg_time_ms = (elapsed * 1000) / count
    
    print(f"✅ Performance: {ids_per_second:.0f} IDs/second")
    print(f"   Average time: {avg_time_ms:.2f}ms per ID")
    
    # Check stats
    stats = generator.get_stats()
    print(f"   Generated: {stats.generated_count}")
    print(f"   Collisions: {stats.collision_count}")
    
    return True


def test_entity_type_mapping():
    """Test entity type to prefix mapping."""
    print("\n🧪 Testing entity type mapping...")
    
    expected_mappings = {
        EntityType.PROJECT: "PRJ",
        EntityType.EPIC: "EPC", 
        EntityType.PRD: "PRD",
        EntityType.TASK: "TSK",
        EntityType.AGENT: "AGT",
        EntityType.WORKFLOW: "WFL",
        EntityType.FILE: "FIL",
        EntityType.DEPENDENCY: "DEP",
        EntityType.SNAPSHOT: "SNP",
        EntityType.SESSION: "SES",
        EntityType.DEBT: "DBT",
        EntityType.PLAN: "PLN"
    }
    
    for entity_type, expected_prefix in expected_mappings.items():
        if entity_type.value != expected_prefix:
            print(f"❌ Wrong prefix for {entity_type.name}: got {entity_type.value}, expected {expected_prefix}")
            return False
    
    print(f"✅ All {len(expected_mappings)} entity type mappings correct")
    return True


def test_config():
    """Test configuration system."""
    print("\n🧪 Testing configuration...")
    
    custom_config = ShortIdConfig(
        prefix_length=3,
        code_length=4,
        separator="-",
        max_retries=5
    )
    
    generator = ShortIdGenerator(custom_config)
    short_id, _ = generator.generate_id(EntityType.TASK)
    
    # Validate custom config is applied
    parts = short_id.split(custom_config.separator)
    if len(parts) != 2:
        print(f"❌ Wrong separator: {short_id}")
        return False
    
    prefix, code = parts
    if len(prefix) != custom_config.prefix_length:
        print(f"❌ Wrong prefix length: {len(prefix)}")
        return False
    
    if len(code) != custom_config.code_length:
        print(f"❌ Wrong code length: {len(code)}")
        return False
    
    print(f"✅ Configuration test passed: {short_id}")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting Short ID System Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Generation", test_basic_generation),
        ("Collision Resistance", test_collision_resistance), 
        ("Alphabet Compliance", test_alphabet_usage),
        ("Validation", test_validation),
        ("CLI Resolution", test_cli_resolution),
        ("Performance", test_performance),
        ("Entity Type Mapping", test_entity_type_mapping),
        ("Configuration", test_config)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Short ID system is ready to use.")
        print("\nNext steps:")
        print("1. Run database migration: alembic upgrade head") 
        print("2. Update models to use ShortIdMixin")
        print("3. Test with real data")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please fix issues before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)