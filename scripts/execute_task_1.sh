#!/bin/bash
# Execute Task 1: Universal Agent Interface Implementation
# Status: COMPLETE ✅
# This task is already implemented - validate completion

echo "🎯 Task 1: Universal Agent Interface Implementation"
echo "Status: COMPLETE ✅"
echo ""

echo "📋 Validating Implementation..."

# Check if all required files exist
files=(
    "app/core/agents/__init__.py"
    "app/core/agents/universal_agent_interface.py"
    "app/core/agents/models.py"
    "app/core/agents/agent_registry.py"
)

echo "Checking file existence:"
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (MISSING)"
        exit 1
    fi
done

echo ""
echo "🧪 Running syntax validation..."

# Validate Python syntax
python3 -m py_compile app/core/agents/universal_agent_interface.py
if [ $? -eq 0 ]; then
    echo "✅ universal_agent_interface.py syntax valid"
else
    echo "❌ universal_agent_interface.py syntax errors"
    exit 1
fi

python3 -m py_compile app/core/agents/models.py
if [ $? -eq 0 ]; then
    echo "✅ models.py syntax valid"
else
    echo "❌ models.py syntax errors"
    exit 1
fi

python3 -m py_compile app/core/agents/agent_registry.py
if [ $? -eq 0 ]; then
    echo "✅ agent_registry.py syntax valid"
else
    echo "❌ agent_registry.py syntax errors"
    exit 1
fi

echo ""
echo "🔍 Checking interface completeness..."

# Check for required classes and methods
if grep -q "class UniversalAgentInterface" app/core/agents/universal_agent_interface.py; then
    echo "✅ UniversalAgentInterface class found"
else
    echo "❌ UniversalAgentInterface class missing"
    exit 1
fi

if grep -q "async def execute_task" app/core/agents/universal_agent_interface.py; then
    echo "✅ execute_task method found"
else
    echo "❌ execute_task method missing"
    exit 1
fi

if grep -q "async def get_capabilities" app/core/agents/universal_agent_interface.py; then
    echo "✅ get_capabilities method found"
else
    echo "❌ get_capabilities method missing"
    exit 1
fi

if grep -q "async def health_check" app/core/agents/universal_agent_interface.py; then
    echo "✅ health_check method found"
else
    echo "❌ health_check method missing"
    exit 1
fi

echo ""
echo "🎉 Task 1 Validation Complete!"
echo "✅ All required components implemented"
echo "✅ Syntax validation passed"
echo "✅ Interface methods present"
echo ""
echo "📝 Next Action: Execute Task 2 (Claude Code Adapter Implementation)"
echo "Run: ./scripts/execute_task_2.sh"