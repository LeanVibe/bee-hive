# LeanVibe Agent Hive 2.0 - Command Documentation Template

## Standard Command Documentation Format

This template provides a standardized format for documenting all Hive commands with comprehensive examples, mobile optimization guidelines, and usage patterns.

---

## Command: /hive:{command_name}

**Category:** {Development|Monitoring|Management|Utilities}  
**Version:** 2.0+  
**Mobile Optimized:** {Yes|Partial|No}  
**Estimated Execution Time:** {< 1 min | 1-5 min | 5+ min}

### Quick Summary

{One-line description of what the command does}

### Description

{Detailed description of the command's purpose, functionality, and use cases. Include when and why users would use this command.}

### Syntax

```bash
/hive:{command_name} [required_param] [--optional-flag] [--optional=value]
```

### Parameters

#### Required Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `param_name` | `string` | Description of what this parameter does | `"example value"` |

#### Optional Parameters

| Flag | Type | Default | Description | Mobile Optimized |
|------|------|---------|-------------|------------------|
| `--flag` | `boolean` | `false` | Description of the flag | ✅ |
| `--option=value` | `string` | `medium` | Description with possible values | ✅ |

### Mobile Optimization

**Mobile Compatibility:** {Fully Compatible|Partially Compatible|Desktop Only}

#### Mobile-Specific Flags

- `--mobile`: Enable mobile-optimized response format
- `--priority=high`: Faster processing for mobile users
- `--compact`: Reduced output for mobile screens

#### Mobile Performance

- **Cached Response Time:** < 5ms
- **Live Response Time:** < 50ms  
- **Memory Usage:** < 10MB
- **Network Efficiency:** Optimized for 3G/4G

### Usage Examples

#### Basic Usage

```bash
# Simple command execution
/hive:{command_name}

# Expected output:
{
  "success": true,
  "message": "Command executed successfully",
  "execution_time_ms": 1250
}
```

#### Advanced Usage

```bash
# Advanced usage with multiple flags
/hive:{command_name} "parameter value" --flag --option=custom

# Expected output:
{
  "success": true,
  "detailed_results": {...},
  "execution_time_ms": 2100
}
```

#### Mobile-Optimized Usage

```bash
# Mobile-optimized execution
/hive:{command_name} --mobile --priority=high

# Expected mobile output:
{
  "success": true,
  "mobile_optimized": true,
  "quick_actions": [...],
  "execution_time_ms": 150
}
```

#### Context-Aware Examples

```bash
# When system is not ready
/hive:{command_name}
# Suggested: /hive:start first

# When mobile context detected
/hive:{command_name} --mobile
# Optimized for touch interface
```

### Integration Examples

#### JavaScript Integration

```javascript
// Using HiveJS interface
const hive = new HiveCommandInterface({
  mobileOptimized: true
});

// Execute command
const result = await hive.executeCommand('/hive:{command_name}', {
  priority: 'high',
  useCache: true
});

console.log('Result:', result);
```

#### Python Integration

```python
# Using hive command system
from app.core.hive_slash_commands import execute_hive_command

# Execute command
result = await execute_hive_command(
    "/hive:{command_name}",
    context={"mobile_optimized": True}
)

print(f"Success: {result['success']}")
```

### Error Handling

#### Common Errors

| Error Code | Description | Resolution |
|------------|-------------|------------|
| `INVALID_PARAMS` | Required parameter missing | Provide all required parameters |
| `SYSTEM_NOT_READY` | Platform not initialized | Run `/hive:start` first |
| `INSUFFICIENT_PERMISSIONS` | Missing required permissions | Check user permissions |

#### Error Recovery

```bash
# If command fails, try:
/hive:status --detailed  # Check system state
/hive:{command_name} --help  # Get specific help
```

### Performance Considerations

#### Execution Time

- **Minimum:** {time}
- **Average:** {time}  
- **Maximum:** {time}
- **Timeout:** {time}

#### Resource Usage

- **CPU Usage:** {low|medium|high}
- **Memory Usage:** {amount}MB
- **Network Activity:** {none|light|moderate|heavy}
- **Disk I/O:** {none|light|moderate|heavy}

#### Scalability

- **Agent Count Impact:** {none|linear|exponential}
- **Data Size Impact:** {none|linear|exponential}
- **Parallel Execution:** {supported|not supported}

### Security Considerations

#### Security Level: {Low|Medium|High|Critical}

#### Permissions Required

- `{permission_name}`: Description of why this permission is needed
- `{permission_name}`: Description of why this permission is needed

#### Data Access

- **Reads:** {data types accessed}
- **Writes:** {data types modified}
- **External Access:** {yes|no} - {description if yes}

#### Security Recommendations

1. {Security recommendation 1}
2. {Security recommendation 2}
3. {Security recommendation 3}

### Troubleshooting

#### Common Issues

**Issue:** {Common problem description}
- **Symptom:** {What user sees}
- **Cause:** {Why it happens}
- **Solution:** {How to fix it}

**Issue:** {Another common problem}
- **Symptom:** {What user sees}  
- **Cause:** {Why it happens}
- **Solution:** {How to fix it}

#### Debugging Commands

```bash
# Check system status
/hive:status --detailed

# Validate command syntax
/hive:validate "/hive:{command_name} params"

# Get contextual help
/hive:focus {command_name}
```

### Related Commands

| Command | Relationship | Description |
|---------|--------------|-------------|
| `/hive:related_cmd1` | Prerequisite | Must run before this command |
| `/hive:related_cmd2` | Complement | Often used together |
| `/hive:related_cmd3` | Alternative | Alternative approach |

### Workflow Integration

#### Typical Workflows

**Workflow: {Workflow Name}**
1. `/hive:command1` - {Description}
2. `/hive:{command_name}` - {This command's role}
3. `/hive:command3` - {Next step}

#### Automation Hooks

```bash
# Pre-execution hook
before_{command_name}() {
  echo "Preparing for {command_name}"
  # Validation logic
}

# Post-execution hook  
after_{command_name}() {
  echo "{command_name} completed"
  # Cleanup or next steps
}
```

### API Reference

#### HTTP Endpoint

```http
POST /api/hive/execute
Content-Type: application/json

{
  "command": "/hive:{command_name}",
  "mobile_optimized": true,
  "priority": "high"
}
```

#### WebSocket Message

```json
{
  "type": "command",
  "command": "/hive:{command_name}",
  "mobile_optimized": true,
  "request_id": "uuid-here"
}
```

#### Response Schema

```json
{
  "success": boolean,
  "command": string,
  "result": object,
  "execution_time_ms": number,
  "mobile_optimized": boolean,
  "cached": boolean,
  "timestamp": string
}
```

### Testing

#### Unit Tests

```python
async def test_{command_name}_success():
    """Test successful command execution."""
    result = await execute_hive_command("/hive:{command_name}")
    assert result["success"] == True
    
async def test_{command_name}_mobile():
    """Test mobile-optimized execution."""
    result = await execute_hive_command(
        "/hive:{command_name} --mobile"
    )
    assert result["mobile_optimized"] == True
```

#### Integration Tests

```python
async def test_{command_name}_integration():
    """Test command in realistic scenario."""
    # Setup system state
    await execute_hive_command("/hive:start")
    
    # Execute target command
    result = await execute_hive_command("/hive:{command_name}")
    
    # Verify expected state changes
    assert result["success"] == True
```

### Performance Benchmarks

#### Baseline Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Avg Execution Time | {value}ms | {target}ms | {✅|⚠️|❌} |
| Cache Hit Rate | {value}% | {target}% | {✅|⚠️|❌} |
| Success Rate | {value}% | 99% | {✅|⚠️|❌} |
| Mobile Performance Score | {value}/10 | 8/10 | {✅|⚠️|❌} |

#### Load Testing Results

- **Concurrent Users:** {number}
- **Requests/Second:** {number}
- **95th Percentile Response Time:** {value}ms
- **Error Rate:** {value}%

### Changelog

#### Version 2.0 
- Initial implementation
- Mobile optimization support
- Enhanced error handling

#### Version 2.1 (if applicable)
- {Changes made}
- {Bug fixes}
- {Performance improvements}

### Migration Guide

#### From Version 1.x

```bash
# Old format (deprecated)
/old-command params

# New format
/hive:{command_name} params --mobile
```

#### Breaking Changes

1. **Parameter Changes:** {Description of parameter changes}
2. **Response Format:** {Description of response format changes}  
3. **Behavior Changes:** {Description of behavior changes}

### Best Practices

#### Do's ✅

- Always use `--mobile` flag when on mobile devices
- Check system status before running complex commands
- Use appropriate `--priority` levels
- Cache results when possible
- Handle errors gracefully

#### Don'ts ❌

- Don't run without checking prerequisites  
- Don't ignore warning messages
- Don't use deprecated parameter formats
- Don't execute on unstable systems
- Don't bypass security validations

### Support

#### Getting Help

```bash
# Command-specific help
/hive:help {command_name}

# Contextual assistance
/hive:focus {command_name}

# System diagnostics
/hive:status --detailed
```

#### Community Resources

- **Documentation:** [Link to full documentation]
- **Examples Repository:** [Link to examples]
- **Issue Tracker:** [Link to GitHub issues]
- **Discord Community:** [Link to Discord]

---

## Template Usage Instructions

### For Command Authors

1. **Copy this template** for each new command
2. **Replace all placeholders** (text in `{braces}`) with actual values
3. **Remove sections** that don't apply to your command
4. **Add command-specific sections** as needed
5. **Test all examples** before publishing
6. **Review mobile compatibility** thoroughly
7. **Include performance benchmarks** from actual testing

### Template Sections

#### Required Sections
- Quick Summary
- Description  
- Syntax
- Parameters
- Usage Examples
- Error Handling

#### Recommended Sections
- Mobile Optimization
- Integration Examples
- Performance Considerations
- Security Considerations
- Related Commands

#### Optional Sections
- Troubleshooting
- API Reference
- Testing
- Performance Benchmarks
- Migration Guide

### Documentation Standards

#### Writing Style
- Use clear, concise language
- Include practical examples
- Explain the "why" not just the "how"
- Consider mobile users in all examples
- Use consistent terminology

#### Code Examples
- All examples must be tested and working
- Include both basic and advanced usage
- Show expected output
- Include error scenarios
- Demonstrate mobile optimization

#### Mobile Optimization
- Every command should document mobile compatibility
- Include mobile-specific examples
- Document performance implications
- Show mobile-optimized responses

### Quality Checklist

Before publishing command documentation:

- [ ] All placeholders replaced with actual values
- [ ] All code examples tested and working
- [ ] Mobile optimization thoroughly documented
- [ ] Security considerations reviewed
- [ ] Performance benchmarks included
- [ ] Error scenarios documented
- [ ] Integration examples provided
- [ ] Related commands identified
- [ ] Troubleshooting guide complete
- [ ] Best practices documented

---

*This template is part of the LeanVibe Agent Hive 2.0 Command Ecosystem Improvement Plan.*