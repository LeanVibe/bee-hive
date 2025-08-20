"""
Validation Validate - LeanVibe Agent Hive 2.0
============================================================

Consolidated validation pattern from 3 implementations.
Estimated LOC savings: 420

Original implementations consolidated:
- /Users/bogdan/work/leanvibe-dev/bee-hive/test_autonomous_development_scenarios.py:272 (_validate_api_endpoint_code)
- /Users/bogdan/work/leanvibe-dev/bee-hive/test_autonomous_development_scenarios.py:279 (_validate_database_models)
- /Users/bogdan/work/leanvibe-dev/bee-hive/test_autonomous_development_scenarios.py:286 (_validate_multi_file_feature)


Generated on: 2025-08-20 12:15:13
"""

"""
Consolidated Validation Pattern
Generated from 3 similar implementations
"""

from typing import Any, Dict, List, Optional, Callable, Union
from abc import ABC, abstractmethod
import re
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationResult:
    """Represents validation result."""
    
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)

class UnifiedValidator:
    """Consolidated validation pattern."""
    
    def __init__(self):
        self.validation_rules = {}
        self.custom_validators = {}
        self._setup_default_validators()
    
    def _setup_default_validators(self):
        """Setup default validation rules."""
        self.validation_rules.update({
            'email': self._validate_email,
            'url': self._validate_url,
            'phone': self._validate_phone,
            'required': self._validate_required,
            'min_length': self._validate_min_length,
            'max_length': self._validate_max_length,
            'numeric': self._validate_numeric,
            'alphanumeric': self._validate_alphanumeric
        })
    
    def add_validator(self, name: str, validator_func: Callable) -> None:
        """Add custom validator function."""
        self.custom_validators[name] = validator_func
        logger.info(f"Added custom validator: {name}")
    
    def validate(self, data: Any, rules: Union[Dict, List[str]]) -> ValidationResult:
        """Validate data against rules."""
        result = ValidationResult(is_valid=True)
        
        if isinstance(rules, list):
            # Simple rule list
            for rule in rules:
                self._apply_rule(data, rule, result)
        elif isinstance(rules, dict):
            # Complex rule dictionary
            for field, field_rules in rules.items():
                field_data = data.get(field) if isinstance(data, dict) else data
                if isinstance(field_rules, list):
                    for rule in field_rules:
                        self._apply_rule(field_data, rule, result, field)
                else:
                    self._apply_rule(field_data, field_rules, result, field)
        
        return result
    
    def _apply_rule(self, data: Any, rule: str, result: ValidationResult, field: Optional[str] = None):
        """Apply single validation rule."""
        rule_parts = rule.split(':')
        rule_name = rule_parts[0]
        rule_params = rule_parts[1:] if len(rule_parts) > 1 else []
        
        validator_func = self.validation_rules.get(rule_name) or self.custom_validators.get(rule_name)
        if not validator_func:
            result.add_warning(f"Unknown validation rule: {rule_name}")
            return
        
        try:
            is_valid, message = validator_func(data, *rule_params)
            if not is_valid:
                field_prefix = f"{field}: " if field else ""
                result.add_error(f"{field_prefix}{message}")
        except Exception as e:
            result.add_error(f"Validation error for rule {rule_name}: {e}")
    
    def _validate_email(self, value: Any) -> tuple[bool, str]:
        """Validate email format."""
        if not isinstance(value, str):
            return False, "Email must be a string"
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, value):
            return True, ""
        return False, "Invalid email format"
    
    def _validate_url(self, value: Any) -> tuple[bool, str]:
        """Validate URL format."""
        if not isinstance(value, str):
            return False, "URL must be a string"
        
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if re.match(url_pattern, value, re.IGNORECASE):
            return True, ""
        return False, "Invalid URL format"
    
    def _validate_phone(self, value: Any) -> tuple[bool, str]:
        """Validate phone number format."""
        if not isinstance(value, str):
            return False, "Phone number must be a string"
        
        phone_pattern = r'^\+?[\d\s\-\(\)]+$'
        if re.match(phone_pattern, value) and len(re.sub(r'\D', '', value)) >= 7:
            return True, ""
        return False, "Invalid phone number format"
    
    def _validate_required(self, value: Any) -> tuple[bool, str]:
        """Validate required field."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return False, "Field is required"
        return True, ""
    
    def _validate_min_length(self, value: Any, min_len: str) -> tuple[bool, str]:
        """Validate minimum length."""
        try:
            min_length = int(min_len)
            if hasattr(value, '__len__') and len(value) < min_length:
                return False, f"Minimum length is {min_length}"
            return True, ""
        except ValueError:
            return False, "Invalid min_length parameter"
    
    def _validate_max_length(self, value: Any, max_len: str) -> tuple[bool, str]:
        """Validate maximum length."""
        try:
            max_length = int(max_len)
            if hasattr(value, '__len__') and len(value) > max_length:
                return False, f"Maximum length is {max_length}"
            return True, ""
        except ValueError:
            return False, "Invalid max_length parameter"
    
    def _validate_numeric(self, value: Any) -> tuple[bool, str]:
        """Validate numeric value."""
        try:
            float(value)
            return True, ""
        except (ValueError, TypeError):
            return False, "Value must be numeric"
    
    def _validate_alphanumeric(self, value: Any) -> tuple[bool, str]:
        """Validate alphanumeric value."""
        if not isinstance(value, str):
            return False, "Value must be a string"
        
        if value.isalnum():
            return True, ""
        return False, "Value must be alphanumeric"

# Global validator instance
global_validator = UnifiedValidator()

