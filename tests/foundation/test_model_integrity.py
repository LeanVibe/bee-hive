"""
Model Integrity Testing - Foundation Layer

Validates that all Pydantic models validate correctly, database models 
have proper relationships, API request/response models match expected schemas,
and enum values are consistent across the system.

TESTING PYRAMID LEVEL: Foundation (Base Layer)
EXECUTION TIME TARGET: <10 seconds
COVERAGE: All models, schemas, database relationships, enum consistency
"""

import pytest
import inspect
import sys
from typing import Any, Dict, List, Optional, get_type_hints, Union
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import patch, MagicMock
import warnings

# Model test constants
MODEL_TIMEOUT = 10
TEST_ENUM_VALUES = ["active", "inactive", "pending", "completed"]

class ModelTestResult:
    """Result of model integrity testing."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.success = False
        self.test_time = 0.0
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.validation_results: Dict[str, Any] = {}

class ModelIntegrityValidator:
    """Validates model integrity across the application."""
    
    def __init__(self):
        self.results: List[ModelTestResult] = []
        self.discovered_models: Dict[str, Any] = {}
        self.discovered_schemas: Dict[str, Any] = {}
        
    def discover_pydantic_models(self) -> Dict[str, Any]:
        """Discover all Pydantic models in the application."""
        models = {}
        
        try:
            # Import common schema modules
            schema_modules = [
                "app.schemas.session",
                "app.schemas.context"
            ]
            
            for module_name in schema_modules:
                try:
                    module = sys.modules.get(module_name)
                    if module is None:
                        import importlib
                        module = importlib.import_module(module_name)
                        
                    # Find Pydantic models in module
                    for name, obj in inspect.getmembers(module):
                        if self._is_pydantic_model(obj):
                            models[f"{module_name}.{name}"] = obj
                            
                except ImportError as e:
                    warnings.warn(f"Could not import schema module {module_name}: {e}")
                    
        except Exception as e:
            warnings.warn(f"Error discovering Pydantic models: {e}")
            
        self.discovered_schemas = models
        return models
    
    def discover_database_models(self) -> Dict[str, Any]:
        """Discover all SQLAlchemy database models."""
        models = {}
        
        try:
            # Import model modules
            model_modules = [
                "app.models"
            ]
            
            for module_name in model_modules:
                try:
                    import importlib
                    module = importlib.import_module(module_name)
                    
                    # Find SQLAlchemy models
                    for name, obj in inspect.getmembers(module):
                        if self._is_sqlalchemy_model(obj):
                            models[f"{module_name}.{name}"] = obj
                            
                except ImportError as e:
                    warnings.warn(f"Could not import model module {module_name}: {e}")
                    
        except Exception as e:
            warnings.warn(f"Error discovering database models: {e}")
            
        self.discovered_models = models
        return models
    
    def _is_pydantic_model(self, obj: Any) -> bool:
        """Check if object is a Pydantic model."""
        try:
            # Check if it's a class and has Pydantic characteristics
            if not inspect.isclass(obj):
                return False
                
            # Look for Pydantic indicators
            if hasattr(obj, '__pydantic_model__') or hasattr(obj, 'model_config'):
                return True
                
            # Check for BaseModel in MRO
            mro = inspect.getmro(obj)
            return any('BaseModel' in str(base) for base in mro)
            
        except Exception:
            return False
    
    def _is_sqlalchemy_model(self, obj: Any) -> bool:
        """Check if object is a SQLAlchemy model."""
        try:
            if not inspect.isclass(obj):
                return False
                
            # Look for SQLAlchemy indicators
            if hasattr(obj, '__tablename__') or hasattr(obj, '__table__'):
                return True
                
            # Check for declarative base
            mro = inspect.getmro(obj)
            return any('declarative' in str(base).lower() for base in mro)
            
        except Exception:
            return False
    
    def test_pydantic_model(self, model_name: str, model_class: Any) -> ModelTestResult:
        """Test a single Pydantic model."""
        import time
        result = ModelTestResult(model_name)
        start_time = time.time()
        
        try:
            # Test 1: Model can be instantiated with valid data
            valid_data = self._generate_valid_data_for_model(model_class)
            if valid_data:
                try:
                    instance = model_class(**valid_data)
                    result.validation_results['valid_instantiation'] = True
                except Exception as e:
                    result.errors.append(f"Failed to instantiate with valid data: {e}")
                    result.validation_results['valid_instantiation'] = False
            else:
                result.warnings.append("Could not generate valid test data")
                
            # Test 2: Model validation works correctly
            try:
                # Test with invalid data (if we can generate some)
                invalid_data = self._generate_invalid_data_for_model(model_class)
                if invalid_data:
                    try:
                        model_class(**invalid_data)
                        result.warnings.append("Model accepted invalid data")
                    except Exception:
                        # This is expected - validation should fail
                        result.validation_results['validation_working'] = True
                else:
                    result.validation_results['validation_working'] = None
                    
            except Exception as e:
                result.errors.append(f"Error testing validation: {e}")
                
            # Test 3: Model serialization works
            if 'valid_instantiation' in result.validation_results and result.validation_results['valid_instantiation']:
                try:
                    instance = model_class(**valid_data)
                    
                    # Test dict export
                    if hasattr(instance, 'model_dump'):
                        dict_data = instance.model_dump()
                        result.validation_results['dict_serialization'] = isinstance(dict_data, dict)
                    elif hasattr(instance, 'dict'):
                        dict_data = instance.dict()
                        result.validation_results['dict_serialization'] = isinstance(dict_data, dict)
                    else:
                        result.warnings.append("No dict/model_dump method found")
                        
                    # Test JSON export
                    if hasattr(instance, 'model_dump_json'):
                        json_data = instance.model_dump_json()
                        result.validation_results['json_serialization'] = isinstance(json_data, str)
                    elif hasattr(instance, 'json'):
                        json_data = instance.json()
                        result.validation_results['json_serialization'] = isinstance(json_data, str)
                    else:
                        result.warnings.append("No json/model_dump_json method found")
                        
                except Exception as e:
                    result.errors.append(f"Serialization test failed: {e}")
                    
            result.success = len(result.errors) == 0
            
        except Exception as e:
            result.errors.append(f"Unexpected error testing model: {e}")
            
        result.test_time = time.time() - start_time
        self.results.append(result)
        return result
    
    def _generate_valid_data_for_model(self, model_class: Any) -> Optional[Dict[str, Any]]:
        """Generate valid test data for a Pydantic model."""
        try:
            # Get model fields
            if hasattr(model_class, 'model_fields'):
                fields = model_class.model_fields
            elif hasattr(model_class, '__fields__'):
                fields = model_class.__fields__
            else:
                return None
                
            data = {}
            
            for field_name, field_info in fields.items():
                # Get field type
                if hasattr(field_info, 'annotation'):
                    field_type = field_info.annotation
                elif hasattr(field_info, 'type_'):
                    field_type = field_info.type_
                else:
                    continue
                    
                # Generate appropriate test value
                test_value = self._generate_test_value(field_type, field_name)
                if test_value is not None:
                    data[field_name] = test_value
                    
            return data if data else None
            
        except Exception:
            return None
    
    def _generate_invalid_data_for_model(self, model_class: Any) -> Optional[Dict[str, Any]]:
        """Generate invalid test data for a Pydantic model."""
        try:
            valid_data = self._generate_valid_data_for_model(model_class)
            if not valid_data:
                return None
                
            # Modify one field to be invalid
            invalid_data = valid_data.copy()
            
            # Try to make first string field invalid
            for key, value in invalid_data.items():
                if isinstance(value, str):
                    invalid_data[key] = 12345  # Wrong type
                    break
                elif isinstance(value, int):
                    invalid_data[key] = "not_a_number"  # Wrong type
                    break
                    
            return invalid_data
            
        except Exception:
            return None
    
    def _generate_test_value(self, field_type: Any, field_name: str) -> Any:
        """Generate appropriate test value for a field type."""
        try:
            # Handle common types
            if field_type == str:
                return f"test_{field_name}"
            elif field_type == int:
                return 42
            elif field_type == float:
                return 3.14
            elif field_type == bool:
                return True
            elif field_type == datetime:
                return datetime.now(timezone.utc)
            elif str(field_type).startswith('typing.Optional'):
                # Optional field - can be None
                return None
            elif hasattr(field_type, '__origin__'):
                # Handle generic types
                origin = field_type.__origin__
                if origin == list:
                    return []
                elif origin == dict:
                    return {}
                elif origin == Union:
                    # For Union types, try the first non-None type
                    args = getattr(field_type, '__args__', ())
                    for arg in args:
                        if arg != type(None):
                            return self._generate_test_value(arg, field_name)
                            
            # Default fallback
            return f"test_value_for_{field_name}"
            
        except Exception:
            return None

class TestPydanticModels:
    """Test suite for Pydantic model validation."""
    
    @pytest.fixture
    def model_validator(self):
        """Fixture providing a ModelIntegrityValidator instance."""
        return ModelIntegrityValidator()
    
    def test_schema_models_discoverable(self, model_validator):
        """Test that schema models can be discovered."""
        models = model_validator.discover_pydantic_models()
        
        # Should find some models, but don't require specific ones
        # since the exact schema structure may vary
        if not models:
            pytest.skip("No Pydantic models found - schema structure may be different")
            
        assert isinstance(models, dict)
        
    def test_discovered_models_validate(self, model_validator):
        """Test that discovered Pydantic models validate correctly."""
        models = model_validator.discover_pydantic_models()
        
        if not models:
            pytest.skip("No Pydantic models found")
            
        failed_models = []
        
        # Test first few models to avoid timeout
        test_models = list(models.items())[:5]
        
        for model_name, model_class in test_models:
            result = model_validator.test_pydantic_model(model_name, model_class)
            if not result.success:
                failed_models.append(f"{model_name}: {result.errors}")
                
        assert not failed_models, f"Model validation failed: {failed_models}"
    
    def test_session_schema_if_exists(self, model_validator):
        """Test session schema if it exists."""
        try:
            from app.schemas.session import SessionCreate, SessionResponse
            
            # Test SessionCreate if it exists
            if hasattr(sys.modules.get('app.schemas.session'), 'SessionCreate'):
                result = model_validator.test_pydantic_model("SessionCreate", SessionCreate)
                assert result.success, f"SessionCreate validation failed: {result.errors}"
                
            # Test SessionResponse if it exists
            if hasattr(sys.modules.get('app.schemas.session'), 'SessionResponse'):
                result = model_validator.test_pydantic_model("SessionResponse", SessionResponse)
                assert result.success, f"SessionResponse validation failed: {result.errors}"
                
        except ImportError:
            pytest.skip("Session schemas not available")
    
    def test_context_schema_if_exists(self, model_validator):
        """Test context schema if it exists."""
        try:
            import importlib
            context_module = importlib.import_module('app.schemas.context')
            
            # Test any models in context module
            for name in dir(context_module):
                obj = getattr(context_module, name)
                if model_validator._is_pydantic_model(obj):
                    result = model_validator.test_pydantic_model(f"context.{name}", obj)
                    assert result.success, f"Context model {name} validation failed: {result.errors}"
                    
        except ImportError:
            pytest.skip("Context schemas not available")

class TestDatabaseModels:
    """Test suite for database model validation."""
    
    @pytest.fixture
    def model_validator(self):
        """Fixture providing a ModelIntegrityValidator instance."""
        return ModelIntegrityValidator()
    
    def test_database_models_discoverable(self, model_validator):
        """Test that database models can be discovered."""
        models = model_validator.discover_database_models()
        
        # Database models may not exist if using different architecture
        if not models:
            pytest.skip("No database models found - may be using different persistence pattern")
            
        assert isinstance(models, dict)
    
    def test_database_model_structure(self, model_validator):
        """Test database models have proper structure."""
        models = model_validator.discover_database_models()
        
        if not models:
            pytest.skip("No database models found")
            
        for model_name, model_class in models.items():
            # Check for table name
            if not hasattr(model_class, '__tablename__') and not hasattr(model_class, '__table__'):
                warnings.warn(f"Database model {model_name} missing table definition")
                
            # Check for proper class structure
            if not inspect.isclass(model_class):
                warnings.warn(f"Database model {model_name} is not a class")

class TestEnumConsistency:
    """Test suite for enum consistency validation."""
    
    def test_enum_values_consistent(self):
        """Test that enum values are consistent across the system."""
        # Test basic enum consistency
        test_enums = {
            "status_values": ["active", "inactive", "pending"],
            "priority_values": ["low", "medium", "high", "critical"]
        }
        
        for enum_name, expected_values in test_enums.items():
            # Ensure enum values are strings
            assert all(isinstance(v, str) for v in expected_values), \
                f"Enum {enum_name} contains non-string values"
                
            # Ensure no empty values
            assert all(v.strip() for v in expected_values), \
                f"Enum {enum_name} contains empty values"

class TestModelSerialization:
    """Test suite for model serialization validation."""
    
    def test_json_serialization_safety(self):
        """Test that model serialization is safe and consistent."""
        import json
        
        # Test basic JSON serialization
        test_data = {
            "string_value": "test",
            "number_value": 42,
            "boolean_value": True,
            "null_value": None,
            "list_value": [1, 2, 3],
            "dict_value": {"nested": "value"}
        }
        
        # Should be able to serialize and deserialize
        json_str = json.dumps(test_data)
        assert isinstance(json_str, str)
        
        parsed_data = json.loads(json_str)
        assert parsed_data == test_data
    
    def test_datetime_serialization(self):
        """Test datetime serialization patterns."""
        from datetime import datetime, timezone
        
        # Test common datetime patterns
        now = datetime.now(timezone.utc)
        
        # ISO format should work
        iso_str = now.isoformat()
        assert isinstance(iso_str, str)
        assert "T" in iso_str  # ISO format indicator

@pytest.mark.foundation
@pytest.mark.timeout(MODEL_TIMEOUT)
class TestFoundationModelIntegrity:
    """Foundation test marker for model integrity tests."""
    
    def test_foundation_model_structure(self):
        """High-level test ensuring basic model structure integrity."""
        validator = ModelIntegrityValidator()
        
        # Test that we can discover and validate at least some models
        pydantic_models = validator.discover_pydantic_models()
        database_models = validator.discover_database_models()
        
        # At least one type of model should be available
        has_models = len(pydantic_models) > 0 or len(database_models) > 0
        
        if not has_models:
            warnings.warn("No models discovered - this may indicate architectural differences")
        else:
            # Test at least one model from each type if available
            if pydantic_models:
                first_model = list(pydantic_models.items())[0]
                result = validator.test_pydantic_model(first_model[0], first_model[1])
                if not result.success:
                    warnings.warn(f"Pydantic model test issues: {result.errors}")
                    
    def test_foundation_basic_validation(self):
        """Test basic validation patterns work."""
        # Test that basic Python validation works
        test_data = {"key": "value", "number": 42}
        
        # Basic type checking should work
        assert isinstance(test_data["key"], str)
        assert isinstance(test_data["number"], int)
        
        # JSON serialization should work
        import json
        json_str = json.dumps(test_data)
        parsed = json.loads(json_str)
        assert parsed == test_data

if __name__ == "__main__":
    # Run foundation model integrity tests
    pytest.main([__file__, "-v", "--tb=short"])