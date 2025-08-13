#!/usr/bin/env python3
"""
Contract Validation Script

Pre-commit hook and CI/CD script for validating contract schemas and ensuring
contract test coverage exists for all defined schemas.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
from jsonschema import Draft202012Validator, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


class ContractValidationResult:
    """Results of contract validation."""
    
    def __init__(self):
        self.schema_validation_results: List[Tuple[Path, bool, str]] = []
        self.test_coverage_results: List[Tuple[str, bool, str]] = []
        self.compatibility_results: List[Tuple[str, bool, str]] = []
        self.performance_results: List[Tuple[str, bool, str]] = []
    
    @property
    def all_passed(self) -> bool:
        """Check if all validations passed."""
        all_results = [
            all(result[1] for result in self.schema_validation_results),
            all(result[1] for result in self.test_coverage_results),
            all(result[1] for result in self.compatibility_results),
            all(result[1] for result in self.performance_results)
        ]
        return all(all_results)
    
    def print_summary(self):
        """Print validation summary."""
        total_checks = (
            len(self.schema_validation_results) + 
            len(self.test_coverage_results) +
            len(self.compatibility_results) +
            len(self.performance_results)
        )
        
        passed_checks = sum([
            sum(1 for _, passed, _ in self.schema_validation_results if passed),
            sum(1 for _, passed, _ in self.test_coverage_results if passed),
            sum(1 for _, passed, _ in self.compatibility_results if passed),
            sum(1 for _, passed, _ in self.performance_results if passed)
        ])
        
        logger.info(f"Contract Validation Summary: {passed_checks}/{total_checks} checks passed")
        
        if self.all_passed:
            logger.info("✅ All contract validations passed!")
        else:
            logger.error("❌ Contract validation failed!")
            self._print_failures()
    
    def _print_failures(self):
        """Print detailed failure information."""
        failure_categories = [
            ("Schema Validation", self.schema_validation_results),
            ("Test Coverage", self.test_coverage_results),
            ("Compatibility", self.compatibility_results),
            ("Performance", self.performance_results)
        ]
        
        for category, results in failure_categories:
            failures = [result for result in results if not result[1]]
            if failures:
                logger.error(f"\n{category} Failures:")
                for item, _, error in failures:
                    logger.error(f"  ❌ {item}: {error}")


class ContractValidator:
    """Validates contract schemas and associated tests."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.schemas_dir = project_root / "schemas"
        self.contracts_test_dir = project_root / "tests" / "contracts"
        self.result = ContractValidationResult()
    
    def validate_all(self) -> ContractValidationResult:
        """Run all contract validations."""
        logger.info("Starting comprehensive contract validation...")
        
        # Core validations
        self.validate_schema_syntax()
        self.validate_test_coverage()
        self.validate_schema_examples()
        
        # Advanced validations
        self.validate_schema_compatibility()
        self.validate_performance_requirements()
        
        return self.result
    
    def validate_schema_syntax(self):
        """Validate that all JSON schemas are syntactically correct."""
        logger.info("Validating JSON schema syntax...")
        
        if not self.schemas_dir.exists():
            self.result.schema_validation_results.append((
                self.schemas_dir, False, "Schemas directory does not exist"
            ))
            return
        
        schema_files = list(self.schemas_dir.glob("*.json"))
        if not schema_files:
            self.result.schema_validation_results.append((
                self.schemas_dir, False, "No schema files found"
            ))
            return
        
        for schema_file in schema_files:
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema_content = json.load(f)
                
                # Validate schema itself
                Draft202012Validator.check_schema(schema_content)
                
                # Validate required metadata
                self._validate_schema_metadata(schema_file, schema_content)
                
                self.result.schema_validation_results.append((
                    schema_file, True, "Valid schema"
                ))
                logger.info(f"  ✅ {schema_file.name} - Valid schema")
                
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON: {e}"
                self.result.schema_validation_results.append((
                    schema_file, False, error_msg
                ))
                logger.error(f"  ❌ {schema_file.name} - {error_msg}")
                
            except Exception as e:
                error_msg = f"Schema validation error: {e}"
                self.result.schema_validation_results.append((
                    schema_file, False, error_msg
                ))
                logger.error(f"  ❌ {schema_file.name} - {error_msg}")
    
    def _validate_schema_metadata(self, schema_file: Path, schema: Dict):
        """Validate required schema metadata."""
        required_fields = ["$schema", "$id", "title", "type"]
        missing_fields = [field for field in required_fields if field not in schema]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Validate schema ID format
        schema_id = schema.get("$id", "")
        if not schema_id.startswith("https://leanvibe.ai/schemas/"):
            raise ValueError("Schema $id must use leanvibe.ai domain")
        
        # Validate that examples exist and are valid
        if "examples" in schema:
            validator = Draft202012Validator(schema)
            for i, example in enumerate(schema["examples"]):
                try:
                    validator.validate(example)
                except ValidationError as e:
                    raise ValueError(f"Example {i} invalid: {e}")
    
    def validate_test_coverage(self):
        """Validate that contract tests exist for all schemas."""
        logger.info("Validating contract test coverage...")
        
        # Define expected test files for each schema
        schema_to_test_mapping = {
            "redis_agent_messages.schema.json": "test_redis_contracts.py",
            "ws_messages.schema.json": "test_websocket_contracts.py", 
            "live_dashboard_data.schema.json": "test_api_contracts.py",
            "database_models.schema.json": "test_database_contracts.py"
        }
        
        schema_files = set(self.schemas_dir.glob("*.json"))
        
        for schema_file in schema_files:
            expected_test_file = schema_to_test_mapping.get(schema_file.name)
            
            if expected_test_file:
                test_file_path = self.contracts_test_dir / expected_test_file
                
                if test_file_path.exists():
                    # Validate test file content
                    if self._validate_test_file_content(test_file_path, schema_file):
                        self.result.test_coverage_results.append((
                            schema_file.name, True, f"Test coverage exists: {expected_test_file}"
                        ))
                        logger.info(f"  ✅ {schema_file.name} - Test coverage exists")
                    else:
                        self.result.test_coverage_results.append((
                            schema_file.name, False, f"Test file incomplete: {expected_test_file}"
                        ))
                        logger.warning(f"  ⚠️ {schema_file.name} - Test file incomplete")
                else:
                    self.result.test_coverage_results.append((
                        schema_file.name, False, f"Missing test file: {expected_test_file}"
                    ))
                    logger.error(f"  ❌ {schema_file.name} - Missing test file: {expected_test_file}")
            else:
                # Schema doesn't have defined test mapping
                self.result.test_coverage_results.append((
                    schema_file.name, False, "No test mapping defined"
                ))
                logger.warning(f"  ⚠️ {schema_file.name} - No test mapping defined")
    
    def _validate_test_file_content(self, test_file: Path, schema_file: Path) -> bool:
        """Validate that test file contains required contract tests."""
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for required test patterns
            required_patterns = [
                "@pytest.mark.contract",
                "validate(instance=",
                "ValidationError",
                "schema=" 
            ]
            
            missing_patterns = [
                pattern for pattern in required_patterns 
                if pattern not in content
            ]
            
            if missing_patterns:
                logger.warning(f"    Missing patterns in {test_file.name}: {missing_patterns}")
                return False
            
            # Check that the schema file is referenced
            if schema_file.name not in content:
                logger.warning(f"    Schema {schema_file.name} not referenced in {test_file.name}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"    Error reading test file {test_file}: {e}")
            return False
    
    def validate_schema_examples(self):
        """Validate that schema examples are correct."""
        logger.info("Validating schema examples...")
        
        for schema_file in self.schemas_dir.glob("*.json"):
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
                
                if "examples" not in schema:
                    logger.warning(f"  ⚠️ {schema_file.name} - No examples provided")
                    continue
                
                validator = Draft202012Validator(schema)
                examples = schema["examples"]
                
                for i, example in enumerate(examples):
                    try:
                        validator.validate(example)
                        logger.debug(f"    Example {i} in {schema_file.name} is valid")
                    except ValidationError as e:
                        error_msg = f"Example {i} invalid: {e.message}"
                        self.result.schema_validation_results.append((
                            schema_file, False, error_msg
                        ))
                        logger.error(f"  ❌ {schema_file.name} - {error_msg}")
                        return
                
                logger.info(f"  ✅ {schema_file.name} - All examples valid")
                
            except Exception as e:
                error_msg = f"Error validating examples: {e}"
                self.result.schema_validation_results.append((
                    schema_file, False, error_msg
                ))
                logger.error(f"  ❌ {schema_file.name} - {error_msg}")
    
    def validate_schema_compatibility(self):
        """Validate schema backward compatibility."""
        logger.info("Validating schema backward compatibility...")
        
        # This would compare with previous versions stored in git
        # For now, we implement basic compatibility checks
        
        for schema_file in self.schemas_dir.glob("*.json"):
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
                
                # Check for potential breaking changes
                breaking_issues = self._check_breaking_changes(schema)
                
                if breaking_issues:
                    for issue in breaking_issues:
                        self.result.compatibility_results.append((
                            schema_file.name, False, issue
                        ))
                        logger.warning(f"  ⚠️ {schema_file.name} - {issue}")
                else:
                    self.result.compatibility_results.append((
                        schema_file.name, True, "No obvious breaking changes"
                    ))
                    logger.info(f"  ✅ {schema_file.name} - No obvious breaking changes")
                    
            except Exception as e:
                error_msg = f"Error checking compatibility: {e}"
                self.result.compatibility_results.append((
                    schema_file.name, False, error_msg
                ))
                logger.error(f"  ❌ {schema_file.name} - {error_msg}")
    
    def _check_breaking_changes(self, schema: Dict) -> List[str]:
        """Check for potential breaking changes in schema."""
        issues = []
        
        # Check for overly strict constraints that might break existing data
        if "required" in schema and len(schema["required"]) > 10:
            issues.append("Large number of required fields may cause compatibility issues")
        
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            # Check for very restrictive string lengths
            if prop_schema.get("type") == "string":
                max_length = prop_schema.get("maxLength")
                if max_length and max_length < 10:
                    issues.append(f"Property '{prop_name}' has very restrictive maxLength ({max_length})")
            
            # Check for very restrictive number ranges
            if prop_schema.get("type") in ["integer", "number"]:
                minimum = prop_schema.get("minimum")
                maximum = prop_schema.get("maximum")
                if minimum is not None and maximum is not None and maximum - minimum < 10:
                    issues.append(f"Property '{prop_name}' has very restrictive range")
        
        # Check additionalProperties setting
        if schema.get("additionalProperties") is False:
            issues.append("additionalProperties: false may cause compatibility issues with extensions")
        
        return issues
    
    def validate_performance_requirements(self):
        """Validate performance-related requirements in schemas."""
        logger.info("Validating performance requirements...")
        
        for schema_file in self.schemas_dir.glob("*.json"):
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
                
                # Check for performance issues
                perf_issues = self._check_performance_issues(schema)
                
                if perf_issues:
                    for issue in perf_issues:
                        self.result.performance_results.append((
                            schema_file.name, False, issue
                        ))
                        logger.warning(f"  ⚠️ {schema_file.name} - {issue}")
                else:
                    self.result.performance_results.append((
                        schema_file.name, True, "No performance issues detected"
                    ))
                    logger.info(f"  ✅ {schema_file.name} - No performance issues detected")
                    
            except Exception as e:
                error_msg = f"Error checking performance: {e}"
                self.result.performance_results.append((
                    schema_file.name, False, error_msg
                ))
                logger.error(f"  ❌ {schema_file.name} - {error_msg}")
    
    def _check_performance_issues(self, schema: Dict) -> List[str]:
        """Check for potential performance issues in schema."""
        issues = []
        
        properties = schema.get("properties", {})
        
        # Check for very large string limits that could impact memory
        for prop_name, prop_schema in properties.items():
            if prop_schema.get("type") == "string":
                max_length = prop_schema.get("maxLength")
                if max_length and max_length > 100000:  # 100KB
                    issues.append(f"Property '{prop_name}' allows very large strings ({max_length} chars)")
            
            # Check for deeply nested objects
            if prop_schema.get("type") == "object" and "properties" in prop_schema:
                if self._count_nesting_depth(prop_schema) > 5:
                    issues.append(f"Property '{prop_name}' has deep nesting (>5 levels)")
            
            # Check for large arrays
            if prop_schema.get("type") == "array":
                max_items = prop_schema.get("maxItems")
                if max_items and max_items > 10000:
                    issues.append(f"Property '{prop_name}' allows very large arrays ({max_items} items)")
        
        return issues
    
    def _count_nesting_depth(self, schema: Dict, depth: int = 0) -> int:
        """Count maximum nesting depth in a schema."""
        if depth > 10:  # Prevent infinite recursion
            return depth
        
        max_depth = depth
        
        if "properties" in schema:
            for prop_schema in schema["properties"].values():
                if prop_schema.get("type") == "object":
                    prop_depth = self._count_nesting_depth(prop_schema, depth + 1)
                    max_depth = max(max_depth, prop_depth)
        
        return max_depth


def main():
    """Main entry point for contract validation."""
    project_root = Path(__file__).parent.parent
    
    logger.info("=== LeanVibe Agent Hive Contract Validation ===")
    logger.info(f"Project root: {project_root}")
    
    validator = ContractValidator(project_root)
    result = validator.validate_all()
    
    result.print_summary()
    
    if result.all_passed:
        logger.info("\n🎉 All contract validations passed! Ready for deployment.")
        return 0
    else:
        logger.error("\n💥 Contract validation failed! Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())