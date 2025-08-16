#!/bin/bash

# Project Index Validation Framework Installation Script

set -e

echo "🚀 Installing Project Index Validation Framework..."

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Install validation framework dependencies
echo "📦 Installing validation framework dependencies..."
pip install -r requirements-validation.txt

echo "🔧 Setting up validation framework..."

# Make scripts executable
chmod +x comprehensive_validation_suite.py
chmod +x install_validation_framework.sh

# Test basic functionality
echo "🧪 Testing framework installation..."
python3 -c "
try:
    from validation_framework import ValidationConfig, ValidationLevel
    from comprehensive_validation_suite import ComprehensiveValidationSuite
    config = ValidationConfig()
    suite = ComprehensiveValidationSuite(config)
    print('✅ Framework installation successful!')
except Exception as e:
    print(f'❌ Installation test failed: {e}')
    exit(1)
"

echo ""
echo "🎉 Project Index Validation Framework installed successfully!"
echo ""
echo "📋 Quick Start:"
echo "  # Quick validation check"
echo "  python3 comprehensive_validation_suite.py --quick-check"
echo ""
echo "  # Standard validation"
echo "  python3 comprehensive_validation_suite.py --level standard"
echo ""
echo "  # Comprehensive validation with report"
echo "  python3 comprehensive_validation_suite.py --level comprehensive --output report.json"
echo ""
echo "📚 Documentation: VALIDATION_FRAMEWORK_DOCUMENTATION.md"
echo "🛠️  Requirements: requirements-validation.txt"
echo ""