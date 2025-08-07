#!/usr/bin/env python3
"""
Test script for ContextManager integration improvements.
"""

import os
import tempfile
from pathlib import Path

# Test ContextSpec creation
def test_context_spec():
    from autogen_framework.agents.base_agent import ContextSpec
    
    spec = ContextSpec(context_type="plan")
    print(f"✅ ContextSpec created: {spec}")
    assert spec.context_type == "plan"
    print("✅ ContextSpec test passed")

# Test token threshold configuration
def test_token_configuration():
    from autogen_framework.config_manager import ConfigManager
    from autogen_framework.context_manager import ContextManager
    from autogen_framework.memory_manager import MemoryManager
    from autogen_framework.context_compressor import ContextCompressor
    from autogen_framework.models import LLMConfig
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create config manager
        config_manager = ConfigManager(load_env=False)
        
        # Create dependencies
        memory_manager = MemoryManager(temp_dir)
        llm_config = LLMConfig(base_url="http://test", model="test", api_key="test")
        context_compressor = ContextCompressor(llm_config)
        
        # Create ContextManager with config
        context_manager = ContextManager(
            work_dir=temp_dir,
            memory_manager=memory_manager,
            context_compressor=context_compressor,
            config_manager=config_manager
        )
        
        print(f"✅ ContextManager created with token config: {context_manager.token_config}")
        print(f"✅ Token thresholds: {context_manager.token_thresholds}")
        
        # Verify thresholds are reasonable
        assert all(threshold > 0 for threshold in context_manager.token_thresholds.values())
        print("✅ Token configuration test passed")

# Test environment variable override
def test_env_override():
    # Set environment variable
    os.environ['CONTEXT_TOKEN_THRESHOLD_PLAN'] = '5000'
    os.environ['DEFAULT_TOKEN_LIMIT'] = '10000'
    
    try:
        from autogen_framework.config_manager import ConfigManager
        from autogen_framework.context_manager import ContextManager
        from autogen_framework.memory_manager import MemoryManager
        from autogen_framework.context_compressor import ContextCompressor
        from autogen_framework.models import LLMConfig
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(load_env=False)
            memory_manager = MemoryManager(temp_dir)
            llm_config = LLMConfig(base_url="http://test", model="test", api_key="test")
            context_compressor = ContextCompressor(llm_config)
            
            context_manager = ContextManager(
                work_dir=temp_dir,
                memory_manager=memory_manager,
                context_compressor=context_compressor,
                config_manager=config_manager
            )
            
            print(f"✅ Plan threshold with env override: {context_manager.token_thresholds['plan']}")
            assert context_manager.token_thresholds['plan'] == 5000
            print("✅ Environment override test passed")
            
    finally:
        # Clean up environment
        os.environ.pop('CONTEXT_TOKEN_THRESHOLD_PLAN', None)
        os.environ.pop('DEFAULT_TOKEN_LIMIT', None)

if __name__ == "__main__":
    print("🧪 Testing ContextManager integration improvements...")
    
    test_context_spec()
    test_token_configuration()
    test_env_override()
    
    print("🎉 All tests passed!")