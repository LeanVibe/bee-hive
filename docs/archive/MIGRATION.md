# Migration Guide - Script Organization Update

This guide helps you migrate from the old scattered script approach to the new standardized Makefile-based development workflow.

## Overview

**Phase 2 Script Consolidation** has reorganized the development tooling to provide:
- ✅ Clean root directory with essential files only
- ✅ Consistent command interface across all operations  
- ✅ Professional project structure following industry standards
- ✅ Backward compatibility during transition period

## Quick Migration Reference

| **Old Command** | **New Command** | **Notes** |
|-----------------|-----------------|-----------|
| `./setup.sh` | `make setup` | Standard setup |
| `./setup-fast.sh` | `make setup` | Same optimized performance |
| `./setup-ultra-fast.sh` | `make setup-minimal` | For CI/CD environments |
| `./start-fast.sh` | `make start` | Start all services |
| `./stop-fast.sh` | `make stop` | Stop all services |
| `./troubleshoot.sh` | `make health` | System health check |
| `./validate-setup.sh` | `make test-smoke` | Quick validation |

## New Command Categories

### Setup & Environment
```bash
make setup              # Standard setup (recommended)
make setup-minimal      # Minimal setup for CI/CD
make setup-full         # Complete development setup
make install            # Install Python dependencies
```

### Development
```bash
make start              # Start all services
make start-minimal      # Start minimal services
make start-bg           # Start services in background
make stop               # Stop all services
make restart            # Restart services
make dev                # Start development server
```

### Testing & Quality
```bash
make test               # Run all tests
make test-unit          # Unit tests only
make test-integration   # Integration tests
make test-performance   # Performance tests
make lint               # Code quality checks
make format             # Auto-format code
```

### Sandbox & Demonstrations
```bash
make sandbox            # Interactive sandbox
make sandbox-demo       # Automated demo
make sandbox-auto       # Autonomous development showcase
```

### Utilities
```bash
make health             # Comprehensive health check
make status             # Quick system status
make logs               # Show service logs
make clean              # Cleanup temporary files
```

## Migration Benefits

### Before (Old Approach)
- 15+ scattered shell scripts in root directory
- Inconsistent naming and interfaces
- Manual dependency on specific script locations
- Mixed concerns (setup, validation, start, stop, etc.)
- Platform-specific implementations

### After (New Approach)
- Single `Makefile` with organized command structure
- Consistent interface: `make <command>`
- Self-documenting: `make help` shows all options
- Cross-platform compatibility
- Professional project organization

## Directory Structure Changes

### Old Structure
```
Root/
├── setup.sh                    # Scattered scripts
├── setup-fast.sh
├── setup-ultra-fast.sh
├── start-fast.sh
├── stop-fast.sh
├── validate-setup.sh
├── test-setup-*.sh
├── troubleshoot.sh
└── [many other scattered scripts]
```

### New Structure
```
Root/
├── Makefile                    # Primary interface
├── setup.sh*                  # Migration wrapper
├── start-fast.sh*             # Migration wrapper  
├── stop-fast.sh*              # Migration wrapper
├── scripts/
│   ├── setup.sh              # New organized scripts
│   ├── start.sh
│   ├── test.sh
│   └── legacy/               # Backward compatibility
│       ├── setup.sh          # Original scripts with deprecation notices
│       ├── setup-fast.sh
│       └── [other legacy scripts]
└── [essential project files only]
```

*Migration wrappers provide smooth transition and usage logging

## Transition Period Support

### Automatic Redirection
Legacy script names in root directory now automatically redirect to new commands:
- Show deprecation warning
- Provide migration guidance  
- Auto-redirect after brief delay
- Log usage for monitoring migration progress

### Legacy Script Access
Original scripts remain available in `scripts/legacy/` with:
- Deprecation notices at startup
- Clear migration instructions
- Continued functionality during transition

## Benefits for Different User Types

### Developers
- **Consistent Interface**: Same `make <command>` pattern for all operations
- **Discoverability**: `make help` shows all available commands with descriptions
- **IDE Integration**: Better integration with development environments
- **Cross-platform**: Works identically on macOS, Linux, Windows

### CI/CD Systems
- **Standardized**: Industry-standard Makefile approach
- **Reliable**: Consistent commands across different environments
- **Optimized**: Separate commands for CI-specific needs (`make setup-minimal`)
- **Maintainable**: Single source of truth for all operations

### New Contributors
- **Professional**: Familiar industry-standard project structure
- **Self-documenting**: Clear command categories and help system
- **Guided**: Deprecation warnings help learn new approach
- **Comprehensive**: All development operations available through unified interface

## Implementation Timeline

### Phase 1: ✅ Complete
- Created new Makefile with comprehensive command structure
- Implemented new organized scripts in `scripts/` directory

### Phase 2: ✅ Complete (Current)
- Root directory cleanup and script consolidation
- Legacy script relocation to `scripts/legacy/`
- Migration wrapper scripts for smooth transition
- Deprecation notices and user guidance

### Phase 3: Future
- Monitor migration usage through log analysis
- Remove migration wrappers after adoption period
- Archive legacy scripts when no longer needed

## Troubleshooting Migration Issues

### Command Not Found
If `make <command>` doesn't work:
1. Ensure you're in the project root directory
2. Check that `Makefile` exists
3. Verify make is installed: `make --version`

### Legacy Script Dependencies
If you have automation depending on old script names:
1. **Immediate**: Scripts still work through migration wrappers
2. **Short-term**: Update automation to use `make` commands
3. **Long-term**: Migration wrappers will be removed

### Performance Concerns
The new approach maintains all performance optimizations:
- Same underlying implementation
- Identical service startup procedures
- Preserved caching and parallel operations

## Getting Help

- **Command Discovery**: `make help`
- **System Status**: `make status`
- **Health Check**: `make health`
- **Documentation**: Comprehensive help text for each command
- **Migration Issues**: Check `.migration_usage.log` for automatic redirection logs

## Feedback and Support

This migration improves the development experience while maintaining full backward compatibility. The transition is designed to be seamless with clear guidance at every step.

For issues or questions about the migration:
1. Check the migration usage logs: `.migration_usage.log`
2. Use `make help` to discover new command equivalents
3. Legacy scripts remain functional in `scripts/legacy/`

---

**Remember**: This migration makes the project more professional, maintainable, and aligned with industry standards while preserving all existing functionality during the transition period.