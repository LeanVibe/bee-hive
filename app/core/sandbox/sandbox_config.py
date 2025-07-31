"""
Sandbox Configuration Management
Automatically detects and enables sandbox mode when API keys are missing
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class SandboxConfig:
    """Configuration for sandbox mode operation."""
    
    # Core sandbox settings
    enabled: bool = False
    auto_detected: bool = False
    reason: str = ""
    
    # Mock service configuration
    mock_anthropic: bool = False
    mock_openai: bool = False
    mock_github: bool = False
    
    # Demo configuration
    demo_scenarios_enabled: bool = True
    realistic_timing: bool = True
    progress_simulation: bool = True
    
    # Performance settings
    response_delay_min: float = 1.0
    response_delay_max: float = 4.0
    chunk_delay_ms: int = 100
    
    # Sandbox indicators
    show_sandbox_banner: bool = True
    show_mock_indicators: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and logging."""
        if self.enabled:
            logger.info("Sandbox mode enabled", 
                       auto_detected=self.auto_detected,
                       reason=self.reason,
                       mock_services={
                           "anthropic": self.mock_anthropic,
                           "openai": self.mock_openai, 
                           "github": self.mock_github
                       })


def detect_sandbox_requirements() -> Dict[str, Any]:
    """
    Detect if sandbox mode should be enabled based on missing API keys.
    
    Returns:
        Dict containing detection results and recommendations
    """
    missing_keys = []
    detection_result = {
        "should_enable_sandbox": False,
        "missing_api_keys": [],
        "mock_services_needed": [],
        "reason": "",
        "confidence": 0.0
    }
    
    # Check for required API keys
    api_key_checks = {
        "ANTHROPIC_API_KEY": {
            "required": True,
            "service": "anthropic",
            "description": "Claude AI API access"
        },
        "OPENAI_API_KEY": {
            "required": False,  # Optional for embeddings
            "service": "openai", 
            "description": "OpenAI embeddings service"
        },
        "GITHUB_TOKEN": {
            "required": False,  # Optional for GitHub integration
            "service": "github",
            "description": "GitHub API integration"
        }
    }
    
    for key_name, config in api_key_checks.items():
        key_value = os.getenv(key_name)
        if not key_value or key_value.strip() == "":
            missing_keys.append({
                "name": key_name,
                "service": config["service"], 
                "required": config["required"],
                "description": config["description"]
            })
    
    # Determine sandbox mode necessity
    required_missing = [key for key in missing_keys if key["required"]]
    optional_missing = [key for key in missing_keys if not key["required"]]
    
    if required_missing:
        # Critical API keys missing - sandbox mode highly recommended
        detection_result.update({
            "should_enable_sandbox": True,
            "missing_api_keys": [key["name"] for key in missing_keys],
            "mock_services_needed": [key["service"] for key in missing_keys],
            "reason": f"Required API keys missing: {', '.join(key['name'] for key in required_missing)}",
            "confidence": 0.95
        })
    elif optional_missing:
        # Only optional keys missing - sandbox mode available but not critical
        detection_result.update({
            "should_enable_sandbox": True,
            "missing_api_keys": [key["name"] for key in missing_keys],
            "mock_services_needed": [key["service"] for key in missing_keys],
            "reason": f"Optional API keys missing, sandbox available: {', '.join(key['name'] for key in optional_missing)}",
            "confidence": 0.7
        })
    else:
        # All keys present - production mode available
        detection_result.update({
            "should_enable_sandbox": False,
            "reason": "All API keys present - production mode available",
            "confidence": 0.9
        })
    
    logger.info("Sandbox detection complete",
               should_enable=detection_result["should_enable_sandbox"],
               missing_keys=len(missing_keys),
               reason=detection_result["reason"])
    
    return detection_result


def create_sandbox_config(
    force_enable: Optional[bool] = None,
    demo_mode: bool = True
) -> SandboxConfig:
    """
    Create sandbox configuration based on environment detection.
    
    Args:
        force_enable: Override auto-detection (True/False/None for auto)
        demo_mode: Enable demo-specific features
        
    Returns:
        SandboxConfig instance
    """
    # Detect sandbox requirements
    detection = detect_sandbox_requirements()
    
    # Determine if sandbox should be enabled
    if force_enable is not None:
        # Manual override
        sandbox_enabled = force_enable
        auto_detected = False
        reason = "Manually enabled" if force_enable else "Manually disabled"
    else:
        # Auto-detection
        sandbox_enabled = detection["should_enable_sandbox"]
        auto_detected = True
        reason = detection["reason"]
    
    # Configure mock services based on missing keys
    mock_services = detection["mock_services_needed"] if sandbox_enabled else []
    
    config = SandboxConfig(
        enabled=sandbox_enabled,
        auto_detected=auto_detected,
        reason=reason,
        
        # Mock service configuration
        mock_anthropic="anthropic" in mock_services,
        mock_openai="openai" in mock_services,
        mock_github="github" in mock_services,
        
        # Demo features (enabled in sandbox mode)
        demo_scenarios_enabled=sandbox_enabled and demo_mode,
        realistic_timing=sandbox_enabled,
        progress_simulation=sandbox_enabled,
        
        # UI indicators
        show_sandbox_banner=sandbox_enabled,
        show_mock_indicators=sandbox_enabled
    )
    
    return config


def is_sandbox_mode() -> bool:
    """Quick check if sandbox mode is enabled."""
    return get_sandbox_config().enabled


def get_sandbox_config() -> SandboxConfig:
    """Get the current sandbox configuration."""
    # Check for cached config
    if hasattr(get_sandbox_config, '_cached_config'):
        return get_sandbox_config._cached_config
    
    # Create and cache config
    config = create_sandbox_config()
    get_sandbox_config._cached_config = config
    
    return config


def reset_sandbox_config():
    """Reset cached sandbox configuration (useful for testing)."""
    if hasattr(get_sandbox_config, '_cached_config'):
        delattr(get_sandbox_config, '_cached_config')


def print_sandbox_banner():
    """Print sandbox mode banner for CLI applications."""
    if not is_sandbox_mode():
        return
    
    config = get_sandbox_config()
    
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                        🏖️  SANDBOX MODE ACTIVE                              ║
║                                                                              ║
║  You're running LeanVibe in demonstration mode with mock AI services.       ║
║  This provides full functionality without requiring API keys.               ║
║                                                                              ║
║  Mock Services: {', '.join(service for service in ['Anthropic', 'OpenAI', 'GitHub'] if getattr(config, f'mock_{service.lower()}', False))}                                      ║
║  Reason: {config.reason:<61} ║
║                                                                              ║
║  To use production mode:                                                     ║
║  1. Set your API keys in .env.local                                         ║  
║  2. Restart the application                                                  ║
║                                                                              ║
║  Documentation: https://github.com/leanvibe/agent-hive                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    
    print(banner)


def get_sandbox_status() -> Dict[str, Any]:
    """Get comprehensive sandbox status for API endpoints."""
    config = get_sandbox_config()
    detection = detect_sandbox_requirements()
    
    return {
        "sandbox_mode": {
            "enabled": config.enabled,
            "auto_detected": config.auto_detected,
            "reason": config.reason
        },
        "mock_services": {
            "anthropic": config.mock_anthropic,
            "openai": config.mock_openai,
            "github": config.mock_github
        },
        "demo_features": {
            "scenarios_enabled": config.demo_scenarios_enabled,
            "realistic_timing": config.realistic_timing,
            "progress_simulation": config.progress_simulation
        },
        "api_keys": {
            "missing": detection["missing_api_keys"],
            "detection_confidence": detection["confidence"]
        },
        "migration": {
            "to_production": "Set missing API keys in environment",
            "required_keys": [key for key in detection["missing_api_keys"] if "ANTHROPIC" in key]
        }
    }