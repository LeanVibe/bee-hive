# Context Preserver

The Context Preserver provides sophisticated context preservation capabilities for maintaining execution state and history during agent handoffs in multi-CLI coordination.

## Overview

The `ProductionContextPreserver` implements complete context management for seamless agent handoffs with:

- **Complete State Capture**: Variables, files, execution history, intermediate results
- **Compression Optimization**: Multi-level compression strategies (0-9 levels) 
- **Integrity Validation**: SHA256 checksums and corruption detection
- **Agent Compatibility**: Target-specific context adaptation
- **High Performance**: <1s packaging, <500ms restoration, 50MB+ support

## Key Features

### 🔧 Complete Context Packaging
- Captures execution variables, current state, task history
- Tracks file operations (created, modified, deleted)
- Preserves intermediate results and workflow position
- Includes metadata for debugging and optimization

### 🗜️ Smart Compression
- **Level 0**: No compression (fastest, largest)
- **Level 6**: Balanced compression (default, good ratio)
- **Level 9**: Maximum compression (slowest, smallest)
- Adaptive strategies based on context size

### 🔍 Integrity Validation
- SHA256 hash verification for corruption detection
- Comprehensive validation checks (8 validation points)
- Package format version compatibility
- Expiration and completeness validation

### 🤖 Agent-Specific Optimization
- **Claude Code**: Detailed context, markdown format, full history
- **Cursor**: Minimal overhead, JSON format, reduced history
- **GitHub Copilot**: Code-focused context, code blocks format
- **Universal**: Balanced approach for unknown agents

## Usage

### Basic Context Handoff

```python
from app.core.communication.context_preserver import ProductionContextPreserver
from app.core.agents.universal_agent_interface import AgentType

# Initialize preserver
preserver = ProductionContextPreserver()

# Create execution context
context = {
    "variables": {"project": "my-app", "status": "ready"},
    "current_state": {"phase": "testing"},
    "task_history": [{"task": "implement", "status": "done"}],
    "files_created": ["main.py", "tests.py"],
    "files_modified": ["config.py"]
}

# Package for handoff
package = await preserver.package_context(
    execution_context=context,
    target_agent_type=AgentType.CLAUDE_CODE,
    compression_level=6
)

# Validate integrity
validation = await preserver.validate_context_integrity(package)
if validation["is_valid"]:
    # Restore context in receiving agent
    restored = await preserver.restore_context(package)
    print(f"Handoff successful: {len(restored)} keys restored")
```

### Compression Strategy Selection

```python
# Fast handoffs (small contexts)
package = await preserver.package_context(
    context, AgentType.CURSOR, compression_level=0
)

# Balanced performance (medium contexts) 
package = await preserver.package_context(
    context, AgentType.CLAUDE_CODE, compression_level=6
)

# Network-optimized (large contexts)
package = await preserver.package_context(
    context, AgentType.GITHUB_COPILOT, compression_level=9
)
```

### Error Handling

```python
try:
    restored = await preserver.restore_context(package)
except Exception as e:
    print(f"Context restoration failed: {e}")
    # Implement retry logic or fallback strategy
```

## Performance Characteristics

| Context Size | Compression Level | Package Size | Packaging Time | Restoration Time |
|-------------|------------------|--------------|----------------|------------------|
| Small (1KB) | 0 (None) | 1.0KB | <0.1ms | <0.1ms |
| Small (1KB) | 6 (Balanced) | 0.4KB | <0.2ms | <0.1ms |
| Medium (10KB) | 6 (Balanced) | 1.1KB | <0.5ms | <0.2ms |
| Large (100KB) | 9 (Maximum) | 6.2KB | <5ms | <1ms |
| XLarge (1MB) | 9 (Maximum) | 29KB | <50ms | <10ms |

## Architecture

### Core Components

1. **Context Packaging Engine**
   - Serializes execution state to JSON
   - Applies compression based on level
   - Calculates integrity checksums
   - Adds target agent optimizations

2. **Integrity Validation System**
   - SHA256 hash verification
   - Package structure validation
   - Format version compatibility
   - Content completeness checks

3. **Context Restoration Engine**
   - Validates package integrity
   - Decompresses context data
   - Reconstructs execution state
   - Applies agent adaptations

### Data Flow

```
Agent A Context → Package → Compress → Hash → Transfer
                                               ↓
Agent B Context ← Restore ← Decompress ← Validate ← Receive
```

## Context Package Structure

```python
@dataclass
class ContextPackage:
    # Identification
    package_id: str
    source_agent_id: str
    target_agent_id: str
    
    # Context data
    execution_context: Dict[str, Any]
    task_history: List[Dict[str, Any]]
    intermediate_results: List[Dict[str, Any]]
    files_created: List[str]
    files_modified: List[str]
    
    # State information
    current_state: Dict[str, Any]
    variable_bindings: Dict[str, Any]
    workflow_position: Optional[str]
    
    # Quality and validation
    context_integrity_hash: str  # SHA256
    validation_status: str       # valid/invalid/pending
    package_size_bytes: int
    compression_used: bool
    
    # Metadata (contains compressed data)
    metadata: Dict[str, Any]
```

## Validation Checks

The integrity validation performs 8 comprehensive checks:

1. **Package Structure**: Validates package ID and basic structure
2. **Metadata Presence**: Ensures metadata is present and complete
3. **Compressed Data**: Validates compressed data availability
4. **SHA256 Integrity**: Verifies data integrity with checksums
5. **Format Version**: Checks version compatibility (1.0, 2.0)
6. **Package Size**: Validates size consistency
7. **Expiration**: Checks package expiration time
8. **Content Completeness**: Validates required fields presence

## Agent Optimizations

### Claude Code Agent
- **Format**: Markdown preferred for readability
- **Context Style**: Detailed with full context
- **History**: Include complete task history
- **Use Case**: Complex analysis and implementation

### Cursor Agent  
- **Format**: JSON for lightweight processing
- **Context Style**: Minimal overhead
- **History**: Reduced to essential items only
- **Use Case**: Fast iterations and quick edits

### GitHub Copilot
- **Format**: Code blocks for code focus
- **Context Style**: Code-centric information
- **History**: Include implementation history
- **Use Case**: Code completion and generation

## Error Scenarios

### Common Failures
1. **Data Corruption**: SHA256 mismatch during validation
2. **Missing Data**: Compressed data absent from package
3. **Format Incompatibility**: Unsupported version format
4. **Expiration**: Context package expired before use
5. **Decompression Failure**: Corrupted compressed data

### Recovery Strategies
1. **Retry with Fresh Package**: Re-package context
2. **Fallback Agent**: Use different target agent type
3. **Reduced Context**: Package minimal context subset
4. **Manual Intervention**: Human-assisted recovery

## Best Practices

### Compression Selection
- **Small contexts (<10KB)**: Use level 0 for speed
- **Medium contexts (10KB-100KB)**: Use level 6 for balance
- **Large contexts (>100KB)**: Use level 9 for size
- **Network-constrained**: Always use level 9

### Performance Optimization
- Monitor packaging and restoration times
- Use appropriate compression for context size
- Validate integrity before attempting restoration
- Implement retry logic for failed handoffs

### Security Considerations
- Context packages contain execution state
- Validate source agent identity before restoration
- Implement access controls for sensitive contexts
- Monitor for malicious or corrupted packages

## Testing

Run the comprehensive test suite:

```bash
# Unit tests
python -m pytest tests/unit/test_context_preserver.py -v

# Standalone validation
python test_context_preserver_standalone.py

# Usage examples
python examples/context_preserver_usage.py
```

## Integration

The Context Preserver integrates with:

- **Multi-CLI Protocol**: Core communication framework
- **Agent Orchestrator**: Handles agent selection and routing
- **Workflow Manager**: Manages task execution and handoffs
- **Message Translator**: Converts between agent formats

## Future Enhancements

- **Encryption**: Add encryption for sensitive contexts
- **Delta Compression**: Incremental context updates
- **Streaming**: Support for large context streaming
- **Caching**: Context package caching for efficiency
- **Metrics**: Detailed performance and usage metrics