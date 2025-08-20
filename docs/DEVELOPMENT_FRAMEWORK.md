# 🏗️ LeanVibe Agent Hive - Enhanced Development Framework

## 🎯 **Mission: Prevent Redundant Development & Ensure Quality**

This framework establishes systematic processes to avoid rebuilding existing functionality and ensure comprehensive testing and documentation.

## 📋 **Phase 0: Discovery & Inventory (MANDATORY)**

### **Step 1: Comprehensive Component Analysis**
Before any development work, ALWAYS execute this discovery sequence:

```bash
# 1. Systematic codebase analysis
/analyze-existing-implementations --scope=<feature-area> --output=inventory.md

# 2. API endpoint discovery  
curl http://localhost:8000/docs | jq '.paths' > api-inventory.json

# 3. CLI command discovery
hive help > cli-inventory.txt
find app/cli -name "*.py" -exec grep -l "def.*" {} \; > cli-functions.txt

# 4. Frontend component discovery
find app/static -name "*.ts" -exec grep -l "@customElement" {} \; > pwa-components.txt

# 5. Database model discovery
find app/models -name "*.py" -exec grep -l "class.*" {} \; > data-models.txt
```

### **Step 2: Documentation-First Validation**
```markdown
# Required before any development:

## Existing Implementation Check ✅/❌
- [ ] API endpoints exist and functional
- [ ] CLI commands cover use case  
- [ ] Frontend components available
- [ ] Database models support requirements
- [ ] Tests exist and pass
- [ ] Documentation current

## Gap Analysis
- Missing functionality: [list specific gaps]
- Enhancement opportunities: [list improvements needed]
- Integration requirements: [list integration points]

## Development Strategy Decision
- [ ] **Enhance existing** (preferred)
- [ ] **Extend existing** (add new features)  
- [ ] **Create new** (only if justified)

Justification: [required for any new development]
```

## 📚 **Enhanced Documentation Hierarchy**

### **Root Level Documentation**
```
/CLAUDE.md                    # Main development guidelines
/docs/
  ├── DEVELOPMENT_FRAMEWORK.md # This file - systematic processes
  ├── PLAN.md                 # Current development plan & status  
  ├── PROMPT.md               # AI interaction guidelines
  ├── COMPONENT_INVENTORY.md  # Live inventory of all components
  ├── API_CONTRACTS.md        # API interface contracts
  ├── TESTING_STRATEGY.md     # Comprehensive testing approach
  └── INTEGRATION_MAP.md      # Component integration matrix
```

### **Component-Specific Documentation**
```
/app/cli/CLAUDE.md           # CLI development guidelines
/app/api/CLAUDE.md           # API development guidelines  
/app/core/CLAUDE.md          # Core system guidelines (already exists)
/app/static/CLAUDE.md        # Frontend/PWA guidelines
/app/models/CLAUDE.md        # Database/data modeling guidelines
/app/services/CLAUDE.md      # Service layer guidelines
/tests/CLAUDE.md             # Testing standards and utilities
```

## 🧪 **Bottom-Up Testing Strategy**

### **Level 1: Unit Testing (Foundation)**
```python
# Component isolation testing template
# tests/unit/test_<component>.py

import pytest
from unittest.mock import Mock, patch
from app.<module>.<component> import <ComponentClass>

class TestComponentIsolation:
    """Test component in complete isolation."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies."""
        with patch.multiple(
            'app.<module>.<component>',
            dependency1=Mock(),
            dependency2=Mock(),
            external_service=Mock()
        ):
            yield
    
    def test_component_core_functionality(self, mock_dependencies):
        """Test core functionality with mocked dependencies."""
        component = ComponentClass()
        result = component.primary_method(test_input)
        
        assert result.success == True
        assert result.expected_output == expected_value
    
    def test_component_error_handling(self, mock_dependencies):
        """Test error scenarios."""
        component = ComponentClass()
        
        with pytest.raises(ExpectedException):
            component.primary_method(invalid_input)
    
    def test_component_boundary_conditions(self, mock_dependencies):
        """Test edge cases and limits."""
        pass
```

### **Level 2: Integration Testing**
```python
# tests/integration/test_<component>_integration.py

class TestComponentIntegration:
    """Test component integration with real dependencies."""
    
    @pytest.fixture
    def real_dependencies(self):
        """Setup real but controlled dependencies."""
        # Use test database, test Redis, etc.
        pass
    
    def test_component_with_database(self, real_dependencies):
        """Test component with real database."""
        pass
    
    def test_component_with_external_apis(self, real_dependencies):
        """Test component with external service integration."""
        pass
```

### **Level 3: Contract Testing**
```python
# tests/contracts/test_<service>_contracts.py

class TestAPIContracts:
    """Test API contracts and interfaces."""
    
    def test_endpoint_request_schema(self):
        """Validate request schema matches documentation."""
        pass
    
    def test_endpoint_response_schema(self):
        """Validate response schema matches documentation."""
        pass
    
    def test_error_response_consistency(self):
        """Ensure consistent error response format."""
        pass
```

### **Level 4: End-to-End Testing**
```python
# tests/e2e/test_<workflow>_e2e.py

class TestCompleteWorkflow:
    """Test complete user workflows."""
    
    def test_cli_to_api_to_database_flow(self):
        """Test complete CLI command execution."""
        pass
    
    def test_web_ui_to_api_workflow(self):
        """Test complete web UI interactions."""
        pass
    
    def test_mobile_pwa_workflow(self):
        """Test mobile PWA functionality."""
        pass
```

## 🤖 **Sub-Agent Specialization Strategy**

### **Agent Roles & Responsibilities**

#### **Discovery Agent**
```python
# Role: Comprehensive system analysis
class DiscoveryAgent:
    responsibilities = [
        "Analyze existing implementations",
        "Generate component inventory", 
        "Identify integration points",
        "Document API surfaces",
        "Map data flow"
    ]
    
    def analyze_system_before_development(self, feature_request):
        """MANDATORY: Run before any development work."""
        return {
            "existing_implementations": self.find_existing_code(feature_request),
            "api_endpoints": self.discover_apis(feature_request),
            "cli_commands": self.discover_cli(feature_request),
            "frontend_components": self.discover_frontend(feature_request),
            "gaps_identified": self.identify_gaps(feature_request),
            "enhancement_recommendations": self.suggest_enhancements()
        }
```

#### **Testing Agent**
```python
# Role: Comprehensive testing strategy
class TestingAgent:
    responsibilities = [
        "Generate unit tests",
        "Create integration tests", 
        "Validate contracts",
        "Run E2E scenarios",
        "Performance testing"
    ]
    
    def create_comprehensive_test_suite(self, component):
        """Generate complete test coverage."""
        return {
            "unit_tests": self.generate_unit_tests(component),
            "integration_tests": self.generate_integration_tests(component),
            "contract_tests": self.generate_contract_tests(component),
            "e2e_tests": self.generate_e2e_tests(component),
            "performance_tests": self.generate_performance_tests(component)
        }
```

#### **Documentation Agent**
```python
# Role: Living documentation maintenance  
class DocumentationAgent:
    responsibilities = [
        "Update component documentation",
        "Maintain API documentation",
        "Keep PLAN.md current",
        "Update PROMPT.md",
        "Generate integration maps"
    ]
    
    def update_documentation_continuously(self, changes):
        """Keep all documentation current."""
        return {
            "updated_component_docs": self.update_component_docs(changes),
            "updated_api_docs": self.update_api_docs(changes),
            "updated_plan": self.update_plan_md(changes),
            "updated_prompts": self.update_prompt_md(changes)
        }
```

#### **Integration Agent**
```python
# Role: System integration and validation
class IntegrationAgent:
    responsibilities = [
        "Test component integration",
        "Validate API contracts",
        "Test CLI-API integration", 
        "Test PWA-API integration",
        "Performance validation"
    ]
    
    def validate_system_integration(self, components):
        """Comprehensive integration testing."""
        return {
            "integration_test_results": self.run_integration_tests(components),
            "contract_validation": self.validate_contracts(components),
            "performance_metrics": self.measure_performance(components),
            "integration_issues": self.identify_issues(components)
        }
```

## 📋 **Mandatory Process Checkpoints**

### **Before Any Development Work:**
- [ ] **Discovery Complete**: Full component inventory and gap analysis
- [ ] **Documentation Review**: All relevant CLAUDE.md files reviewed
- [ ] **Integration Map**: Understanding of component relationships
- [ ] **Test Strategy**: Clear testing approach defined

### **During Development:**
- [ ] **Test-First Development**: Tests written before implementation
- [ ] **Documentation Updates**: CLAUDE.md files updated with changes  
- [ ] **Integration Validation**: Component integration tested continuously
- [ ] **Contract Compliance**: API contracts maintained

### **After Development:**
- [ ] **Full Test Suite**: All test levels passing
- [ ] **Documentation Current**: All documentation updated
- [ ] **Integration Verified**: End-to-end workflows tested
- [ ] **Performance Validated**: Performance requirements met

## 🎯 **Success Metrics**

### **Process Efficiency**
- **Discovery Time**: <30 minutes for comprehensive analysis
- **Redundant Work**: <5% of development effort wasted on existing functionality
- **Documentation Currency**: <24 hours lag between code and documentation
- **Test Coverage**: >90% across all test levels

### **Quality Metrics**
- **Integration Issues**: <2 per major component
- **Contract Violations**: 0 API contract breaking changes
- **Performance Regression**: 0 performance degradation
- **Documentation Accuracy**: >95% accuracy in component descriptions

This framework ensures we **build on existing strengths** rather than **rebuild from scratch**, while maintaining **comprehensive quality** through systematic testing and documentation.