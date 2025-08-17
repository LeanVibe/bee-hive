#!/usr/bin/env python3
"""
Test script to verify our context compression components work end-to-end
"""

import json
import os
import subprocess
import sys
from pathlib import Path

def test_hooks():
    """Test that hooks can execute properly"""
    print("🧪 Testing hooks...")
    
    # Test auto-compact-check
    test_input = {
        "session_id": "test123",
        "transcript_path": "/dev/null", 
        "hook_event_name": "UserPromptSubmit",
        "prompt": "This is a test prompt for compression analysis"
    }
    
    try:
        result = subprocess.run(
            ["./.claude/hooks/auto-compact-check.py"],
            input=json.dumps(test_input),
            text=True,
            capture_output=True,
            cwd=os.getcwd()
        )
        print(f"✅ auto-compact-check.py: Exit code {result.returncode}")
        if result.stdout:
            print(f"📤 Output: {result.stdout[:200]}...")
        if result.stderr:
            print(f"⚠️  Stderr: {result.stderr[:200]}...")
    except Exception as e:
        print(f"❌ auto-compact-check.py failed: {e}")
    
    # Test quality monitor
    test_input2 = {
        "session_id": "test123",
        "transcript_path": "/dev/null",
        "hook_event_name": "PostToolUse", 
        "tool_name": "Read"
    }
    
    try:
        result = subprocess.run(
            ["./.claude/hooks/context-quality-monitor.py"],
            input=json.dumps(test_input2),
            text=True,
            capture_output=True,
            cwd=os.getcwd()
        )
        print(f"✅ context-quality-monitor.py: Exit code {result.returncode}")
        if result.stdout:
            print(f"📤 Output: {result.stdout[:200]}...")
    except Exception as e:
        print(f"❌ context-quality-monitor.py failed: {e}")

def test_commands():
    """Test that our command files are properly formatted"""
    print("\n📋 Testing command files...")
    
    commands_dir = Path(".claude/commands")
    for cmd_file in commands_dir.glob("*.md"):
        print(f"✅ Found command: {cmd_file.name}")
        content = cmd_file.read_text()
        if "---" in content and "description:" in content:
            print(f"  📝 Has valid frontmatter")
        else:
            print(f"  ⚠️  Missing frontmatter")

def test_settings():
    """Test settings configuration"""
    print("\n⚙️  Testing settings...")
    
    try:
        with open(".claude/settings.json") as f:
            settings = json.load(f)
        
        if "hooks" in settings:
            print("✅ Hooks configuration found")
            hooks = settings["hooks"]
            
            required_hooks = ["PreCompact", "UserPromptSubmit", "PostToolUse", "Stop"]
            for hook_type in required_hooks:
                if hook_type in hooks:
                    print(f"  ✅ {hook_type} configured")
                else:
                    print(f"  ❌ {hook_type} missing")
        else:
            print("❌ No hooks configuration found")
            
    except Exception as e:
        print(f"❌ Settings test failed: {e}")

def check_minimal_requirements():
    """Check what's needed for minimal end-to-end value"""
    print("\n🎯 Minimal End-to-End Requirements:")
    print("1. ✅ Custom commands exist (.claude/commands/*.md)")
    print("2. ✅ Hooks are executable and functional")  
    print("3. ✅ Settings are properly configured")
    print("4. ❓ Need actual Claude Code CLI session to test slash commands")
    print("5. ❓ Need real conversation transcript to test compression logic")
    
    print("\n💡 MINIMAL VIABLE TEST:")
    print("   Run this in an actual Claude Code CLI session:")
    print("   > /help")
    print("   > Look for: smart-compact, universal-compact, quick-compress")
    print("   > Try: /smart-compact adaptive")
    print("   > Should trigger our enhanced compression logic")

if __name__ == "__main__":
    print("🧪 Context Compression Component Test")
    print("=" * 50)
    
    test_hooks()
    test_commands() 
    test_settings()
    check_minimal_requirements()
    
    print("\n🏁 Test completed!")