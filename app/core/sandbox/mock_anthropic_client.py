"""
Mock Anthropic API Client for Sandbox Mode
Provides realistic AI responses for autonomous development demonstrations
"""

import asyncio
import json
import random
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger()


class TaskType(Enum):
    """Types of development tasks for context-aware responses."""
    SIMPLE_FUNCTION = "simple_function"
    MODERATE_FEATURE = "moderate_feature" 
    COMPLEX_APPLICATION = "complex_application"
    BUG_FIX = "bug_fix"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    TESTING = "testing"


@dataclass
class MockMessage:
    """Mock message structure matching Anthropic API."""
    role: str
    content: str
    

@dataclass 
class MockResponse:
    """Mock response structure matching Anthropic API."""
    content: List[Dict[str, Any]]
    model: str
    role: str
    stop_reason: str
    stop_sequence: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class MockMessages:
    """Mock messages interface to match Anthropic client structure."""
    
    def __init__(self, client):
        self.client = client
    
    async def create(self, **kwargs):
        """Delegate to client's messages_create method."""
        return await self.client.messages_create(**kwargs)


class MockAnthropicClient:
    """
    Mock Anthropic client that provides realistic responses for sandbox demonstrations.
    Simulates the AsyncAnthropic client interface with context-aware responses.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "sandbox-mock-key"
        self.model = "claude-3-5-sonnet-20241022"
        self.conversation_history: Dict[str, List[MockMessage]] = {}
        
        # Create messages interface to match Anthropic client
        self.messages = MockMessages(self)
        
        # Load pre-defined scenarios and responses
        self.scenarios = self._load_demo_scenarios()
        
        logger.info("MockAnthropicClient initialized for sandbox mode")
    
    def _load_demo_scenarios(self) -> Dict[TaskType, Dict[str, Any]]:
        """Load pre-defined demo scenarios with realistic responses."""
        return {
            TaskType.SIMPLE_FUNCTION: {
                "understanding": {
                    "analysis": "I need to create a Fibonacci number calculation function with input validation and error handling.",
                    "requirements": [
                        "Handle positive integers only",
                        "Include comprehensive input validation", 
                        "Handle edge cases (0, 1)",
                        "Use efficient algorithm (iterative approach)",
                        "Provide clear error messages"
                    ],
                    "complexity_assessment": "Simple - Single function implementation"
                },
                "planning": {
                    "approach": "I'll implement an iterative Fibonacci function with input validation.",
                    "files": ["fibonacci.py", "test_fibonacci.py", "README.md"],
                    "architecture": "Single module with main function and helper utilities"
                },
                "implementation": {
                    "code": '''def fibonacci(n):
    """
    Calculate the nth Fibonacci number using iterative approach.
    
    Args:
        n (int): Position in Fibonacci sequence (non-negative integer)
        
    Returns:
        int: The nth Fibonacci number
        
    Raises:
        TypeError: If n is not an integer
        ValueError: If n is negative
    """
    # Input validation
    if not isinstance(n, int):
        raise TypeError(f"Expected integer, got {type(n).__name__}")
    
    if n < 0:
        raise ValueError("Fibonacci sequence is not defined for negative numbers")
    
    # Handle base cases
    if n == 0:
        return 0
    elif n == 1:
        return 1
    
    # Iterative calculation for efficiency
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b


def fibonacci_sequence(count):
    """
    Generate the first 'count' Fibonacci numbers.
    
    Args:
        count (int): Number of Fibonacci numbers to generate
        
    Returns:
        list: List of Fibonacci numbers
    """
    if not isinstance(count, int) or count < 0:
        raise ValueError("Count must be a non-negative integer")
    
    return [fibonacci(i) for i in range(count)]


if __name__ == "__main__":
    # Example usage
    try:
        print("First 10 Fibonacci numbers:")
        for i in range(10):
            print(f"F({i}) = {fibonacci(i)}")
        
        print("\\nFirst 10 numbers as sequence:")
        print(fibonacci_sequence(10))
        
    except Exception as e:
        print(f"Error: {e}")
''',
                    "explanation": "Implemented iterative Fibonacci function with comprehensive input validation and error handling."
                }
            },
            TaskType.MODERATE_FEATURE: {
                "understanding": {
                    "analysis": "I need to create a temperature converter supporting Celsius, Fahrenheit, and Kelvin with validation.",
                    "requirements": [
                        "Convert between C, F, K",
                        "Validate temperature ranges", 
                        "Handle absolute zero limits",
                        "User-friendly interface",
                        "Comprehensive tests"
                    ],
                    "complexity_assessment": "Moderate - Multiple conversion functions with validation"
                },
                "planning": {
                    "approach": "I'll create a TemperatureConverter class with conversion methods and validation.",
                    "files": ["temperature_converter.py", "test_temperature_converter.py", "cli.py", "README.md"],
                    "architecture": "Object-oriented design with separate CLI interface"
                },
                "implementation": {
                    "code": '''class TemperatureConverter:
    """
    Temperature converter supporting Celsius, Fahrenheit, and Kelvin.
    Includes validation for physical temperature limits.
    """
    
    # Physical constants
    ABSOLUTE_ZERO_C = -273.15
    ABSOLUTE_ZERO_F = -459.67
    ABSOLUTE_ZERO_K = 0.0
    
    def celsius_to_fahrenheit(self, celsius):
        """Convert Celsius to Fahrenheit."""
        self._validate_celsius(celsius)
        return (celsius * 9/5) + 32
    
    def celsius_to_kelvin(self, celsius):
        """Convert Celsius to Kelvin."""
        self._validate_celsius(celsius)
        return celsius + 273.15
    
    def fahrenheit_to_celsius(self, fahrenheit):
        """Convert Fahrenheit to Celsius."""
        self._validate_fahrenheit(fahrenheit)
        return (fahrenheit - 32) * 5/9
    
    def fahrenheit_to_kelvin(self, fahrenheit):
        """Convert Fahrenheit to Kelvin."""
        celsius = self.fahrenheit_to_celsius(fahrenheit)
        return self.celsius_to_kelvin(celsius)
    
    def kelvin_to_celsius(self, kelvin):
        """Convert Kelvin to Celsius."""
        self._validate_kelvin(kelvin)
        return kelvin - 273.15
    
    def kelvin_to_fahrenheit(self, kelvin):
        """Convert Kelvin to Fahrenheit."""
        celsius = self.kelvin_to_celsius(kelvin)
        return self.celsius_to_fahrenheit(celsius)
    
    def _validate_celsius(self, temp):
        """Validate Celsius temperature."""
        if not isinstance(temp, (int, float)):
            raise TypeError("Temperature must be numeric")
        if temp < self.ABSOLUTE_ZERO_C:
            raise ValueError(f"Temperature below absolute zero: {temp}°C < {self.ABSOLUTE_ZERO_C}°C")
    
    def _validate_fahrenheit(self, temp):
        """Validate Fahrenheit temperature.""" 
        if not isinstance(temp, (int, float)):
            raise TypeError("Temperature must be numeric")
        if temp < self.ABSOLUTE_ZERO_F:
            raise ValueError(f"Temperature below absolute zero: {temp}°F < {self.ABSOLUTE_ZERO_F}°F")
    
    def _validate_kelvin(self, temp):
        """Validate Kelvin temperature."""
        if not isinstance(temp, (int, float)):
            raise TypeError("Temperature must be numeric")
        if temp < self.ABSOLUTE_ZERO_K:
            raise ValueError(f"Temperature below absolute zero: {temp}K < {self.ABSOLUTE_ZERO_K}K")


def main():
    """CLI interface for temperature conversion."""
    converter = TemperatureConverter()
    
    print("Temperature Converter")
    print("Supports: Celsius (C), Fahrenheit (F), Kelvin (K)")
    
    while True:
        try:
            temp_input = input("\\nEnter temperature and unit (e.g., 25 C): ").strip()
            if temp_input.lower() in ['quit', 'exit']:
                break
                
            parts = temp_input.split()
            if len(parts) != 2:
                print("Format: [number] [C/F/K]")
                continue
                
            temp_value = float(parts[0])
            unit = parts[1].upper()
            
            if unit == 'C':
                f = converter.celsius_to_fahrenheit(temp_value)
                k = converter.celsius_to_kelvin(temp_value)
                print(f"{temp_value}°C = {f:.2f}°F = {k:.2f}K")
            elif unit == 'F':
                c = converter.fahrenheit_to_celsius(temp_value)
                k = converter.fahrenheit_to_kelvin(temp_value)
                print(f"{temp_value}°F = {c:.2f}°C = {k:.2f}K")
            elif unit == 'K':
                c = converter.kelvin_to_celsius(temp_value)
                f = converter.kelvin_to_fahrenheit(temp_value)
                print(f"{temp_value}K = {c:.2f}°C = {f:.2f}°F")
            else:
                print("Unit must be C, F, or K")
                
        except ValueError as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            break
    
    print("\\nGoodbye!")


if __name__ == "__main__":
    main()
''',
                    "explanation": "Implemented comprehensive temperature converter with validation and CLI interface."
                }
            }
        }
    
    def _detect_task_type(self, prompt: str) -> TaskType:
        """Detect task type from prompt content for context-aware responses."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["fibonacci", "calculate", "simple function"]):
            return TaskType.SIMPLE_FUNCTION
        elif any(word in prompt_lower for word in ["temperature", "converter", "multiple", "feature"]):
            return TaskType.MODERATE_FEATURE
        elif any(word in prompt_lower for word in ["application", "complex", "microservice"]):
            return TaskType.COMPLEX_APPLICATION
        elif any(word in prompt_lower for word in ["bug", "fix", "error", "debug"]):
            return TaskType.BUG_FIX
        elif any(word in prompt_lower for word in ["review", "analyze", "improve"]):
            return TaskType.CODE_REVIEW
        elif any(word in prompt_lower for word in ["document", "readme", "docs"]):
            return TaskType.DOCUMENTATION
        elif any(word in prompt_lower for word in ["test", "pytest", "unittest"]):
            return TaskType.TESTING
        else:
            return TaskType.SIMPLE_FUNCTION  # Default
    
    def _detect_development_phase(self, prompt: str) -> str:
        """Detect current development phase from prompt."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["understand", "analyze", "requirements"]):
            return "understanding"
        elif any(word in prompt_lower for word in ["plan", "design", "architecture"]):
            return "planning" 
        elif any(word in prompt_lower for word in ["implement", "code", "develop"]):
            return "implementation"
        elif any(word in prompt_lower for word in ["test", "verify", "validate"]):
            return "testing"
        elif any(word in prompt_lower for word in ["document", "readme"]):
            return "documentation"
        else:
            return "understanding"  # Default
    
    async def _simulate_processing_delay(self, base_delay: float = 2.0):
        """Simulate realistic API processing time."""
        # Add some randomness to make it feel more realistic
        delay = base_delay + random.uniform(0.5, 2.0)
        await asyncio.sleep(delay)
    
    async def messages_create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        **kwargs
    ) -> MockResponse:
        """
        Mock implementation of Anthropic messages.create API.
        
        Provides context-aware responses based on conversation content.
        """
        await self._simulate_processing_delay()
        
        # Get the latest user message
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            raise ValueError("No user message found")
        
        latest_message = user_messages[-1]["content"]
        
        # Detect task type and phase
        task_type = self._detect_task_type(latest_message)
        phase = self._detect_development_phase(latest_message)
        
        # Generate context-aware response
        response_content = self._generate_response(task_type, phase, latest_message)
        
        # Create mock response matching Anthropic API structure
        response = MockResponse(
            content=[{
                "type": "text",
                "text": response_content
            }],
            model=model,
            role="assistant",
            stop_reason="end_turn",
            usage={
                "input_tokens": len(latest_message.split()) * 2,  # Rough estimate
                "output_tokens": len(response_content.split()) * 2
            }
        )
        
        logger.info("Generated mock response", 
                   task_type=task_type.value, 
                   phase=phase,
                   response_length=len(response_content))
        
        return response
    
    def _generate_response(self, task_type: TaskType, phase: str, prompt: str) -> str:
        """Generate contextually appropriate response."""
        
        if task_type in self.scenarios and phase in self.scenarios[task_type]:
            scenario_data = self.scenarios[task_type][phase]
            
            if phase == "understanding":
                return f"""I'll analyze this development task step by step:

**Task Analysis:**
{scenario_data['analysis']}

**Requirements Identified:**
{chr(10).join(f'• {req}' for req in scenario_data['requirements'])}

**Complexity Assessment:**
{scenario_data['complexity_assessment']}

I'm ready to proceed to the planning phase. Would you like me to create a detailed implementation plan?"""

            elif phase == "planning":
                return f"""Based on my analysis, here's my implementation plan:

**Approach:**
{scenario_data['approach']}

**Files to Create:**
{chr(10).join(f'• {file}' for file in scenario_data['files'])}

**Architecture:**
{scenario_data['architecture']}

**Implementation Strategy:**
1. Create core functionality with proper error handling
2. Add comprehensive input validation  
3. Implement thorough test coverage
4. Add user-friendly interfaces
5. Create clear documentation

Ready to proceed with implementation. Shall I start coding?"""

            elif phase == "implementation":
                return f"""Here's the complete implementation:

**Implementation Explanation:**
{scenario_data['explanation']}

**Generated Code:**

```python
{scenario_data['code']}
```

**Key Features Implemented:**
• Comprehensive input validation
• Proper error handling with descriptive messages
• Efficient algorithms
• Clean, maintainable code structure
• Example usage and CLI interface

The implementation is complete and ready for testing. Shall I proceed to create comprehensive tests?"""

        # Fallback responses for unmatched scenarios
        return self._generate_fallback_response(phase, prompt)
    
    def _generate_fallback_response(self, phase: str, prompt: str) -> str:
        """Generate fallback response for unknown scenarios."""
        
        if phase == "understanding":
            return """I understand you'd like me to work on this development task. Let me analyze the requirements:

**Task Analysis:**
I can help you implement this functionality with proper error handling, validation, and testing.

**Approach:**
I'll break this down into manageable components and implement a clean, maintainable solution.

Ready to proceed with detailed planning. What specific requirements should I focus on?"""

        elif phase == "planning":
            return """Here's my implementation plan:

**Strategy:**
1. Design clean, modular architecture
2. Implement core functionality first
3. Add comprehensive error handling
4. Create thorough test coverage
5. Write clear documentation

**Deliverables:**
• Main implementation file
• Comprehensive test suite  
• User documentation
• Example usage

Ready to start implementation. Should I proceed with coding?"""

        elif phase == "implementation":
            return """I've implemented a solution that includes:

**Key Features:**
• Clean, maintainable code structure
• Comprehensive input validation
• Proper error handling
• Efficient algorithms
• Clear documentation

**Code Structure:**
```python
# Example implementation structure
def main_function(input_param):
    \"\"\"
    Main functionality with proper validation and error handling.
    \"\"\"
    # Input validation
    if not isinstance(input_param, expected_type):
        raise TypeError("Invalid input type")
    
    # Core logic implementation
    result = process_input(input_param)
    
    return result

def process_input(param):
    \"\"\"Helper function for core processing.\"\"\"
    # Implementation details here
    pass
```

The implementation is complete and ready for testing. Shall I create comprehensive tests?"""

        else:
            return f"""I'm ready to help with the {phase} phase of development. 

Please provide more specific details about what you'd like me to focus on, and I'll provide a detailed response tailored to your needs."""


class MockAnthropicStream:
    """Mock streaming response for Anthropic API."""
    
    def __init__(self, response_text: str, chunk_size: int = 50):
        self.response_text = response_text
        self.chunk_size = chunk_size
        self.position = 0
    
    async def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.position >= len(self.response_text):
            raise StopAsyncIteration
        
        # Simulate streaming delay
        await asyncio.sleep(0.1)
        
        # Get next chunk
        chunk = self.response_text[self.position:self.position + self.chunk_size]
        self.position += self.chunk_size
        
        # Return chunk in Anthropic API format
        return {
            "type": "content_block_delta",
            "delta": {"text": chunk}
        }


# Factory function for creating mock client
def create_mock_anthropic_client(api_key: Optional[str] = None) -> MockAnthropicClient:
    """Create mock anthropic client for sandbox mode."""
    return MockAnthropicClient(api_key=api_key)