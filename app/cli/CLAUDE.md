# CLAUDE.md - CLI System Guidelines

## 🎯 **Context: CLI System Enhancement**

You are working in the **CLI system layer** of LeanVibe Agent Hive 2.0. This directory contains Unix-philosophy command implementations that provide professional-grade tooling for power users.

## ✅ **Existing CLI Capabilities (DO NOT REBUILD)**

### **Core Commands Already Implemented**
- `hive status` - System status with --watch capability
- `hive get` - Resource listing (agents, tasks, workflows) with multiple output formats
- `hive logs` - Log viewing with --follow and filtering  
- `hive create/delete/scale` - Resource management operations
- `hive config` - Git-style configuration management
- `hive init/doctor/debug` - System setup and diagnostics

### **Output Formats Already Supported**
- JSON (`--output json`)
- Table format (default)
- YAML support
- Rich terminal formatting with colors and tables

### **Real-time Capabilities Already Implemented**
- `hive status --watch` - Live status monitoring
- `hive logs --follow` - Log streaming
- `hive metrics --watch` - Real-time metrics

## 🔧 **Development Guidelines**

### **Enhancement Strategy (NOT Replacement)**
When adding new CLI functionality:

1. **FIRST**: Check existing commands in `unix_commands.py`
2. **ENHANCE** existing commands rather than create new ones
3. **INTEGRATE** with enhanced command ecosystem from `/app/core/`
4. **MAINTAIN** Unix philosophy: focused, composable, pipeable

### **Integration with Enhanced Systems**
```python
# Pattern for enhancing existing commands
from app.core.command_ecosystem_integration import get_ecosystem_integration
from app.core.enhanced_command_discovery import get_command_discovery

@click.command()
@click.option('--enhanced', is_flag=True, help='Use enhanced AI capabilities')
@click.option('--mobile', is_flag=True, help='Mobile-optimized output')
async def enhanced_existing_command(enhanced, mobile):
    """Enhance existing command with new capabilities."""
    if enhanced:
        ecosystem = await get_ecosystem_integration()
        result = await ecosystem.execute_enhanced_command(
            command=f"/hive:{command_name}",
            mobile_optimized=mobile,
            use_quality_gates=True
        )
        display_enhanced_result(result, mobile=mobile)
    else:
        # Use existing implementation
        existing_command_implementation()
```

### **Code Standards**

#### **Command Structure**
```python
@click.command()
@click.option('--format', type=click.Choice(['json', 'table', 'yaml']), default='table')
@click.option('--mobile', is_flag=True, help='Mobile-optimized output')
@click.argument('resource', required=False)
def hive_command(format, mobile, resource):
    """
    Command description following Unix conventions.
    
    Examples:
        hive command                    # Basic usage
        hive command --format json     # JSON output
        hive command --mobile          # Mobile-optimized
    """
    # Implementation
```

#### **Error Handling**
```python
def safe_api_call(endpoint: str, method: str = "GET", data: dict = None):
    """Standardized API call with consistent error handling."""
    try:
        result = ctx.api_call(endpoint, method, data)
        if not result:
            console.print(f"[red]Failed to connect to API at {ctx.api_base}[/red]")
            console.print("Run 'hive doctor' to diagnose connection issues")
            sys.exit(1)
        return result
    except Exception as e:
        console.print(f"[red]API Error: {e}[/red]")
        sys.exit(1)
```

#### **Output Formatting Standards**
```python
def format_output(data: dict, format_type: str, mobile: bool = False):
    """Standardized output formatting."""
    if format_type == 'json':
        click.echo(json.dumps(data, indent=2))
    elif format_type == 'yaml':
        import yaml
        click.echo(yaml.dump(data, default_flow_style=False))
    elif format_type == 'table':
        display_table(data, mobile=mobile)
```

## 🧪 **Testing Requirements**

### **CLI-Specific Testing**
```python
# tests/cli/test_command_integration.py
def test_command_with_real_api():
    """Test command against real API."""
    runner = CliRunner()
    result = runner.invoke(hive_status)
    assert result.exit_code == 0
    assert "System Status" in result.output

def test_command_output_formats():
    """Test all output format options."""
    for format_type in ['json', 'table', 'yaml']:
        runner = CliRunner()
        result = runner.invoke(hive_get, ['--output', format_type, 'agents'])
        assert result.exit_code == 0
```

### **Integration Testing with Core Systems**
```python
def test_enhanced_command_integration():
    """Test integration with enhanced command ecosystem."""
    runner = CliRunner()
    result = runner.invoke(hive_status, ['--enhanced', '--mobile'])
    assert result.exit_code == 0
    assert "enhanced_execution" in result.output
```

## 🔗 **Integration Points**

### **API Integration** (`app/api/`)
- Uses `ctx.api_call()` for all API communication
- Endpoints: `/health`, `/status`, `/debug-agents`, `/api/tasks/active`
- Error handling for API connectivity issues

### **Core System Integration** (`app/core/`)
- Enhanced command ecosystem: `command_ecosystem_integration.py`
- AI command discovery: `enhanced_command_discovery.py`
- Quality gates: `unified_quality_gates.py`

### **Configuration** (`~/.config/agent-hive/`)
- Shared configuration with API and PWA
- Git-style configuration commands
- Environment-specific settings

## ⚠️ **Critical Guidelines**

### **DO NOT Rebuild Existing Commands**
- All basic CLI functionality exists and works well
- Focus on **enhancement** and **integration**
- Add AI capabilities to existing commands
- Improve mobile optimization for existing commands

### **Maintain Unix Philosophy**
- Each command does one thing well
- Commands are composable and pipeable
- Consistent parameter naming across commands
- Rich help and error messages

### **Performance Requirements**
- Commands respond in <200ms for basic operations
- Real-time streaming maintains <100ms latency
- Memory usage <50MB for CLI process
- Graceful degradation when API is slow

## 📋 **Enhancement Priorities**

### **High Priority**
1. **Integration** with enhanced command ecosystem
2. **AI-powered** command suggestions and discovery
3. **Real-time streaming** improvements
4. **Mobile optimization** for all commands

### **Medium Priority**
5. **Batch operations** and scripting support
6. **Advanced filtering** and query capabilities
7. **Plugin system** for custom commands
8. **Shell integration** improvements (completion, aliases)

### **Low Priority**
9. **Voice command** integration
10. **Advanced configuration** templating
11. **Multi-environment** management
12. **Custom output** formatters

## 🎯 **Success Criteria**

Your CLI enhancements are successful when:
- **Existing functionality** is preserved and enhanced
- **New AI capabilities** integrate seamlessly with existing commands
- **Mobile optimization** works across all commands
- **Performance** meets or exceeds current standards
- **Unix philosophy** is maintained throughout
- **Integration** with core systems is robust and reliable

Focus on **enhancing the excellent foundation** rather than rebuilding from scratch.