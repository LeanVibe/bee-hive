#!/bin/bash
# Execute Task 3: Git Worktree Isolation System
# Priority: CRITICAL - Security foundation for multi-CLI coordination

echo "🎯 Task 3: Git Worktree Isolation System Implementation"
echo "Priority: CRITICAL - Security and isolation foundation"
echo ""

echo "📋 Checking Prerequisites..."

# Check git availability
if command -v git &> /dev/null; then
    echo "✅ Git available: $(git --version)"
else
    echo "❌ Git not available - required for worktree management"
    exit 1
fi

# Check if we're in a git repository
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "✅ Current directory is a git repository"
else
    echo "❌ Not in a git repository - required for worktree operations"
    exit 1
fi

echo ""
echo "📁 Creating Required Directory Structure..."

# Create isolation directory if it doesn't exist
mkdir -p app/core/isolation
echo "✅ Created app/core/isolation/ directory"

# Check if files exist
files=(
    "app/core/isolation/__init__.py"
    "app/core/isolation/worktree_manager.py"
    "app/core/isolation/path_validator.py"
    "app/core/isolation/security_enforcer.py"
)

echo ""
echo "🔍 Checking Implementation Status..."

missing_files=()
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "🔨 $file - REQUIRES CREATION"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    echo ""
    echo "🧪 Validating Existing Implementation..."
    
    # Check syntax of existing files
    for file in "${files[@]}"; do
        if [[ $file == *.py ]]; then
            python3 -m py_compile "$file" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "✅ $file syntax valid"
            else
                echo "❌ $file syntax errors"
            fi
        fi
    done
    
    echo ""
    echo "🔍 Checking Implementation Completeness..."
    
    # Check for key classes and methods
    if grep -q "class WorktreeManager" app/core/isolation/worktree_manager.py 2>/dev/null; then
        echo "✅ WorktreeManager class found"
        
        if grep -q "async def create_worktree" app/core/isolation/worktree_manager.py; then
            echo "✅ create_worktree method found"
        else
            echo "🔨 create_worktree method - REQUIRES IMPLEMENTATION"
        fi
        
        if grep -q "async def cleanup_worktree" app/core/isolation/worktree_manager.py; then
            echo "✅ cleanup_worktree method found"
        else
            echo "🔨 cleanup_worktree method - REQUIRES IMPLEMENTATION"
        fi
        
        if grep -q "async def validate_path_access" app/core/isolation/worktree_manager.py; then
            echo "✅ validate_path_access method found"
        else
            echo "🔨 validate_path_access method - REQUIRES IMPLEMENTATION"
        fi
    else
        echo "🔨 WorktreeManager class - REQUIRES IMPLEMENTATION"
    fi
    
    if grep -q "class PathValidator" app/core/isolation/path_validator.py 2>/dev/null; then
        echo "✅ PathValidator class found"
    else
        echo "🔨 PathValidator class - REQUIRES IMPLEMENTATION"
    fi
    
else
    echo ""
    echo "📝 IMPLEMENTATION REQUIRED:"
    echo ""
    
    for file in "${missing_files[@]}"; do
        case $file in
            "app/core/isolation/__init__.py")
                echo "🔨 Create $file:"
                echo '   """Git Worktree Isolation System"""'
                echo '   from .worktree_manager import WorktreeManager'
                echo '   from .path_validator import PathValidator'
                echo ""
                ;;
            "app/core/isolation/worktree_manager.py")
                echo "🔨 Create $file:"
                echo "   Implement WorktreeManager class with methods:"
                echo "   - async def create_worktree(agent_id, branch, base_path)"
                echo "   - async def cleanup_worktree(worktree_id)"
                echo "   - async def validate_path_access(worktree_id, path)"
                echo ""
                ;;
            "app/core/isolation/path_validator.py")
                echo "🔨 Create $file:"
                echo "   Implement PathValidator class with methods:"
                echo "   - def validate_file_access(worktree_path, file_path)"
                echo "   - def sanitize_path(path)"
                echo "   - def check_security_constraints(path)"
                echo ""
                ;;
            "app/core/isolation/security_enforcer.py")
                echo "🔨 Create $file:"
                echo "   Implement SecurityEnforcer class with methods:"
                echo "   - def enforce_resource_limits(process_id)"
                echo "   - def monitor_file_operations(worktree_path)"
                echo "   - def validate_command_safety(command_args)"
                echo ""
                ;;
        esac
    done
fi

echo ""
echo "📖 Implementation Requirements:"
echo ""
echo "🔒 Security Requirements:"
echo "   1. Path Traversal Prevention: Block ../../../etc/passwd attacks"
echo "   2. Symlink Protection: Validate real paths after resolution"
echo "   3. System Directory Blocking: Prevent access to /etc, /usr, /bin"
echo "   4. Resource Limits: Enforce disk space, file count, execution time"
echo "   5. Process Isolation: Sandbox execution environment"
echo ""

echo "⚡ Performance Requirements:"
echo "   1. Worktree Creation: <5 seconds"
echo "   2. Path Validation: <100ms per check"
echo "   3. Resource Monitoring: <1% CPU overhead"
echo "   4. Cleanup Operations: <10 seconds"
echo ""

echo "🧪 Testing Requirements:"
echo "   1. Security Tests: Path traversal attack prevention"
echo "   2. Performance Tests: Creation/cleanup timing"
echo "   3. Stress Tests: Multiple concurrent worktrees"
echo "   4. Integration Tests: With Claude Code adapter"
echo ""

echo "📚 Implementation Reference:"
echo "   Technical Specs: docs/TECHNICAL_SPECIFICATIONS.md (Section 3)"
echo "   Test Framework: tests/git_worktree_isolation_tests.py"
echo "   Security Patterns: Validate ALL paths, block system directories"
echo ""

echo "🚀 Key Implementation Pattern:"
echo '   ```python'
echo '   async def create_worktree(self, agent_id: str, branch: str, base_path: str):'
echo '       # 1. Validate inputs and check permissions'
echo '       # 2. Execute: git worktree add [path] [branch]'
echo '       # 3. Set up directory permissions and limits'
echo '       # 4. Return WorktreeContext with metadata'
echo '   ```'
echo ""

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "📝 Next Action: Validate implementation with ./scripts/validate_task_3.sh"
else
    echo "📝 Next Action: Implement the missing files listed above"
fi

echo ""
echo "⚠️  CRITICAL: This task blocks all secure agent execution"
echo "💡 TIP: Start with basic WorktreeManager, add security incrementally"