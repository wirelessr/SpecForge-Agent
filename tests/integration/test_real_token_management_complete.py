"""
真實Token Management Integration測試 - 完全遵循steering指導原則

這個測試完全遵循.kiro/steering/testing-best-practices.md的指導：
1. 使用真實配置和組件，不使用mock
2. 使用標準fixtures (real_llm_config, integration_config_manager)
3. 遵循標準integration測試模式
4. 優先測試真實功能
"""

import pytest
from typing import Dict, Any

from autogen_framework.agents.base_agent import BaseLLMAgent
from autogen_framework.token_manager import TokenManager
from autogen_framework.context_compressor import ContextCompressor


class RealTestAgent(BaseLLMAgent):
    """真實測試agent - 遵循steering指導"""
    
    async def _process_task_impl(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "result": "Real integration test completed",
            "task_input": task_input,
            "agent_name": self.name
        }
    
    def get_agent_capabilities(self) -> list:
        return [
            "real_integration_test", 
            "token_management", 
            "context_compression",
            "real_llm_calls"
        ]


@pytest.mark.integration
class TestRealTokenManagement:
    """真實Token Management Integration測試 - 遵循steering指導原則"""
    
    @pytest.fixture
    def real_token_manager(self, integration_config_manager):
        """使用真實ConfigManager創建TokenManager - 遵循steering模式"""
        return TokenManager(integration_config_manager)
    
    @pytest.fixture
    def real_context_compressor(self, real_llm_config):
        """使用真實LLM配置創建ContextCompressor - 遵循steering模式"""
        return ContextCompressor(real_llm_config)
    
    @pytest.fixture
    def real_agent(self, real_llm_config, real_token_manager, real_context_compressor, real_managers):
        """創建真實agent - 遵循steering模式"""
        return RealTestAgent(
            name="real_test_agent",
            llm_config=real_llm_config,
            system_message="""
            You are a real test agent for integration testing of token management and context compression.
            Your role is to demonstrate the complete integration of TokenManager and ContextCompressor with BaseLLMAgent.
            You should respond naturally and help test the token tracking and compression capabilities.
            This system message is intentionally detailed to provide sufficient context for testing compression.
            """,
            token_manager=real_token_manager,
            context_manager=real_managers.context_manager)
    
    @pytest.mark.integration
    def test_real_initialization(self, real_agent, real_token_manager, real_context_compressor, real_llm_config, real_managers):
        """測試真實組件初始化 - 遵循steering指導"""
        # 驗證agent正確初始化
        assert real_agent.name == "real_test_agent"
        assert real_agent.token_manager is real_token_manager
        assert real_agent.context_manager is real_managers.context_manager
        
        # 驗證LLM配置使用環境變量
        assert real_agent.llm_config.model == real_llm_config.model
        assert real_agent.llm_config.base_url == real_llm_config.base_url
        
        # 驗證TokenManager配置
        assert real_token_manager.token_config['compression_threshold'] == 0.9
        assert real_token_manager.token_config['compression_enabled'] is True
        
        # 驗證ContextCompressor配置 (through ContextManager)
        assert real_context_compressor.llm_config.model == real_agent.llm_config.model
    
    @pytest.mark.integration
    def test_real_token_tracking(self, real_token_manager):
        """測試真實token tracking功能 - 遵循steering指導"""
        # 初始狀態
        assert real_token_manager.current_context_size == 0
        
        # 模擬真實的token使用
        real_token_manager.update_token_usage("models/gemini-2.0-flash", 500, "test_operation_1")
        assert real_token_manager.current_context_size == 500
        
        real_token_manager.update_token_usage("models/gemini-2.0-flash", 300, "test_operation_2")
        assert real_token_manager.current_context_size == 800
        
        # 檢查統計數據
        stats = real_token_manager.get_usage_statistics()
        assert stats.total_tokens_used == 800
        assert stats.requests_made == 2
        assert stats.average_tokens_per_request == 400.0
        
        # 測試token limit檢查
        token_check = real_token_manager.check_token_limit("models/gemini-2.0-flash")
        assert token_check.current_tokens == 800
        assert token_check.model_limit == 1048576  # Gemini 2.0 Flash limit
        assert not token_check.needs_compression  # 還沒達到90%閾值
    
    @pytest.mark.integration
    def test_real_token_limit_detection(self, real_token_manager):
        """測試真實token limit檢測 - 遵循steering指導"""
        # 設置高token使用量來觸發壓縮閾值
        high_token_count = int(1048576 * 0.95)  # 95% of Gemini limit
        real_token_manager.current_context_size = high_token_count
        
        token_check = real_token_manager.check_token_limit("models/gemini-2.0-flash")
        assert token_check.current_tokens == high_token_count
        assert token_check.percentage_used > 0.9
        assert token_check.needs_compression  # 應該觸發壓縮
    
    @pytest.mark.integration
    async def test_real_context_compression(self, real_context_compressor):
        """測試真實context compression - 遵循steering指導"""
        # 創建真實的大型context用於壓縮
        large_context = {
            'system_info': 'This is a comprehensive system information section with detailed configuration and setup details. ' * 20,
            'conversation_data': {
                'messages': [
                    {'role': 'user', 'content': 'This is a detailed user message with extensive context and information. ' * 10},
                    {'role': 'assistant', 'content': 'This is a comprehensive assistant response with detailed explanations and examples. ' * 15},
                    {'role': 'user', 'content': 'Follow-up user message with additional context and requirements. ' * 8}
                ]
            },
            'technical_details': {
                'architecture': 'Multi-agent system with token management and context compression capabilities. ' * 12,
                'performance': 'Optimized for handling large contexts with automatic compression when needed. ' * 10,
                'features': 'Includes real-time token tracking, intelligent compression, and fallback strategies. ' * 8
            }
        }
        
        # 執行真實的壓縮
        compression_result = await real_context_compressor.compress_context(
            large_context, 
            target_reduction=0.6
        )
        
        # 驗證壓縮結果
        assert compression_result.success, f"Compression failed: {compression_result.error}"
        assert compression_result.original_size > 0
        assert compression_result.compressed_size > 0
        assert compression_result.compression_ratio > 0.3  # 至少30%壓縮率
        assert compression_result.method_used == "llm_compression"
        assert len(compression_result.compressed_content) > 0
        
        # 記錄壓縮效果
        print(f"\n真實壓縮結果:")
        print(f"原始大小: {compression_result.original_size}")
        print(f"壓縮後大小: {compression_result.compressed_size}")
        print(f"壓縮率: {compression_result.compression_ratio:.1%}")
    
    @pytest.mark.integration
    async def test_real_agent_response_generation(self, real_agent):
        """測試真實agent response generation - 遵循steering指導"""
        # 測試正常的response generation
        response = await real_agent.generate_response(
            "Please provide a brief explanation of token management in AI systems."
        )
        
        # 驗證response
        assert isinstance(response, str)
        assert len(response) > 0
        assert "token" in response.lower() or "management" in response.lower()
        
        # 檢查conversation history
        assert len(real_agent.conversation_history) >= 2  # user + assistant
        assert real_agent.conversation_history[-2]['role'] == 'user'
        assert real_agent.conversation_history[-1]['role'] == 'assistant'
        assert real_agent.conversation_history[-1]['content'] == response
    
    @pytest.mark.integration
    async def test_real_token_usage_tracking(self, real_agent, real_token_manager):
        """測試真實token usage tracking - 遵循steering指導"""
        initial_tokens = real_token_manager.current_context_size
        initial_requests = real_token_manager.usage_stats['requests_made']
        
        # 生成response並追蹤token使用
        response = await real_agent.generate_response(
            "Explain the benefits of context compression in AI applications."
        )
        
        # 驗證token使用被正確追蹤
        assert real_token_manager.current_context_size > initial_tokens
        assert real_token_manager.usage_stats['requests_made'] > initial_requests
        
        # 檢查統計數據
        stats = real_token_manager.get_usage_statistics()
        assert stats.total_tokens_used > 0
        assert stats.requests_made > 0
        assert stats.average_tokens_per_request > 0
        
        print(f"\n真實Token使用統計:")
        print(f"總token使用: {stats.total_tokens_used}")
        print(f"請求次數: {stats.requests_made}")
        print(f"平均每次請求token: {stats.average_tokens_per_request:.1f}")
    
    @pytest.mark.integration
    async def test_real_agent_capabilities(self, real_agent):
        """測試真實agent capabilities - 遵循steering指導"""
        capabilities = real_agent.get_agent_capabilities()
        
        assert "real_integration_test" in capabilities
        assert "token_management" in capabilities
        assert "context_compression" in capabilities
        assert "real_llm_calls" in capabilities
        
        # 測試process_task方法
        task_result = await real_agent.process_task({
            "task_type": "integration_test",
            "parameters": {"test_mode": "real"}
        })
        
        assert task_result["result"] == "Real integration test completed"
        assert task_result["agent_name"] == "real_test_agent"
        assert task_result["task_input"]["task_type"] == "integration_test"
    
    @pytest.mark.integration
    def test_real_error_handling(self, real_token_manager):
        """測試真實錯誤處理 - 遵循steering指導"""
        # 測試無效token數量
        initial_size = real_token_manager.current_context_size
        real_token_manager.update_token_usage("test-model", -100, "invalid_operation")
        
        # 應該忽略無效的token數量
        assert real_token_manager.current_context_size == initial_size
        
        # 測試未知模型的token limit
        token_check = real_token_manager.check_token_limit("unknown-model")
        assert token_check.model_limit == 8192  # 應該使用默認limit
    
    @pytest.mark.integration
    def test_real_detailed_reporting(self, real_token_manager):
        """測試真實詳細報告功能 - 遵循steering指導"""
        # 添加一些token使用
        real_token_manager.update_token_usage("models/gemini-2.0-flash", 1000, "test_op_1")
        real_token_manager.update_token_usage("models/gemini-2.0-flash", 500, "test_op_2")
        real_token_manager.increment_compression_count()
        
        # 獲取詳細報告
        detailed_report = real_token_manager.get_detailed_usage_report()
        
        # 驗證報告內容
        assert detailed_report['current_context_size'] >= 0
        assert detailed_report['statistics']['total_tokens_used'] >= 1500
        assert detailed_report['statistics']['requests_made'] >= 2
        assert detailed_report['statistics']['compressions_performed'] >= 1
        assert 'configuration' in detailed_report
        assert 'model_limits' in detailed_report
        assert 'usage_history' in detailed_report
        
        # 驗證配置信息
        config = detailed_report['configuration']
        assert config['compression_threshold'] == 0.9
        assert config['compression_enabled'] is True
        
        print(f"\n真實詳細使用報告:")
        print(f"當前context大小: {detailed_report['current_context_size']}")
        print(f"總token使用: {detailed_report['statistics']['total_tokens_used']}")
        print(f"壓縮次數: {detailed_report['statistics']['compressions_performed']}")