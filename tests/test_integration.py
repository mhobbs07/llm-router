"""
Integration Test Suite for LLM Router System

This file serves as an index for the comprehensive integration test suite.
The tests have been organized into separate modules for better maintainability:

- test_anthropic_integration.py: Anthropic Claude provider specific tests
- test_provider_routing.py: Provider routing and instantiation logic tests
- test_fallback_logic.py: Fallback mechanism and configuration loading tests
- test_streaming.py: Streaming functionality tests across all providers
- test_monitoring.py: Metrics collection, logging, and monitoring endpoint tests
- test_chat_completions.py: Core chat completion functionality tests
- test_cost_aware_filtering.py: Cost optimization and policy filtering tests

To run all integration tests:
    pytest tests/test_*.py -v

To run specific test modules:
    pytest tests/test_anthropic_integration.py -v
    pytest tests/test_provider_routing.py -v
    pytest tests/test_fallback_logic.py -v
    pytest tests/test_streaming.py -v
    pytest tests/test_monitoring.py -v
    pytest tests/test_chat_completions.py -v
    pytest tests/test_cost_aware_filtering.py -v
"""

import pytest


def test_integration_suite_structure():
    """Test that all integration test modules are properly structured."""
    import importlib
    import os
    
    test_modules = [
        'test_anthropic_integration',
        'test_provider_routing', 
        'test_fallback_logic',
        'test_streaming',
        'test_monitoring',
        'test_chat_completions',
        'test_cost_aware_filtering'
    ]
    
    tests_dir = os.path.dirname(__file__)
    
    for module_name in test_modules:
        module_path = os.path.join(tests_dir, f"{module_name}.py")
        assert os.path.exists(module_path), f"Test module {module_name}.py not found"
        
        # Try to import the module to ensure it's valid Python
        try:
            importlib.import_module(f"tests.{module_name}")
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])