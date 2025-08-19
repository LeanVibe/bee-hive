# Check Project Index Before File Creation

This command automatically checks the project index before creating any new files to prevent duplicate logic, overwrites, and maintain codebase organization.

## Usage

```bash
# Automatically run before any file creation
python .claude/hooks/simple-file-check.py <proposed_file_path> [purpose]

# Example usage by Claude:
python .claude/hooks/simple-file-check.py src/services/new-service.ts "API service for user management"
```

## Integration Instructions for Claude

**MANDATORY: Before creating any new file, you MUST run this command:**

1. **Always check first**: Run the project index analysis before any `Write` tool usage
2. **Include purpose**: Provide a brief description of what the file will do
3. **Review recommendations**: Follow the analysis recommendations
4. **Use force flag only when necessary**: Add `--force-create` flag only if you're certain the analysis is wrong

## Example Workflow

```bash
# Step 1: Check before creating
python .claude/hooks/simple-file-check.py mobile-pwa/src/components/UserProfile.tsx "React component for displaying user profile information"

# Step 2: If approved (exit code 0), proceed with creation
# If blocked (exit code 1), review conflicts and alternatives

# Step 3: Follow recommendations from the analysis
```

## What This Command Checks

- **File conflicts**: Existing files with similar names or purposes
- **Directory organization**: Files in similar locations
- **Functionality overlap**: Potential duplicate logic
- **Naming conventions**: Consistency with existing patterns
- **Risk assessment**: Overall impact of creating the new file

## Risk Levels

- 🟢 **LOW**: Safe to create, minimal conflicts
- 🟡 **MEDIUM**: Some concerns, review recommendations
- 🔴 **HIGH**: Major conflicts detected, consider alternatives

## Override Options

```bash
# Force creation despite analysis (use with caution)
python .claude/hooks/simple-file-check.py path/to/file.ext "purpose" --force-create
```

## Integration with Claude Workflow

This command should be integrated into Claude's file creation process:

1. **Before every `Write` tool call**
2. **Before every `MultiEdit` tool call that creates new files**
3. **When user requests new file creation**

## Expected Output

The command provides structured analysis including:
- ✅/❌ Creation decision
- 🟢🟡🔴 Risk level indicator
- ⚠️ Conflict details
- 📋 Similar files list
- 💡 Actionable recommendations
- 🤖 Reasoning explanation

This ensures intelligent file creation that maintains codebase quality and prevents accidental duplication or overwrites.