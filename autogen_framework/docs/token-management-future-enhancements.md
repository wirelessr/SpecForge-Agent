# Token Management and Context Optimization

## Overview

This document outlines enhancements to token management and context optimization within the autonomous execution framework. With the introduction of ContextManager and ContextCompressor, token management has become more sophisticated and integrated with the overall system architecture.

## Current Architecture

The framework now uses a multi-layered approach to token management:

1. **ContextManager**: Provides comprehensive project context with automatic compression
2. **ContextCompressor**: Handles intelligent context compression when token limits are approached
3. **TokenManager**: Manages model-specific token limits and counting
4. **Agent-Specific Context**: Each agent receives optimized context for its specific needs

## Enhancement Opportunities

### 1. Dynamic Context Prioritization

**Status**: Planned Enhancement  
**Priority**: High

Implement intelligent context prioritization based on task relevance and execution history.

```python
class ContextPrioritizer:
    """Prioritizes context elements based on relevance and importance."""
    
    async def prioritize_context(self, context: ProjectContext, task: TaskDefinition) -> PrioritizedContext:
        """Ranks context elements by importance for the specific task."""
        priorities = {
            'current_task_requirements': 1.0,
            'related_task_history': 0.8,
            'project_structure': 0.6,
            'memory_patterns': 0.4,
            'general_context': 0.2
        }
        
        return self._apply_priorities(context, priorities, task)
```

### 2. Adaptive Token Budgeting

**Status**: Concept  
**Priority**: Medium

Implement dynamic token budget allocation based on task complexity and agent needs.

```python
class TokenBudgetManager:
    """Manages token budgets across different context types."""
    
    def allocate_budget(self, total_tokens: int, task_complexity: str) -> Dict[str, int]:
        """Allocates token budget based on task complexity."""
        if task_complexity == "high":
            return {
                'requirements': int(total_tokens * 0.3),
                'design': int(total_tokens * 0.3),
                'execution_history': int(total_tokens * 0.25),
                'memory_patterns': int(total_tokens * 0.15)
            }
        # ... other complexity levels
```

### 3. Context Compression Strategies

**Status**: Partially Implemented  
**Priority**: High

Enhance the ContextCompressor with more sophisticated compression strategies.

#### Current Compression Methods
- **Summarization**: LLM-based summarization of large context sections
- **Relevance Filtering**: Remove context elements not relevant to current task
- **Hierarchical Compression**: Compress less important sections more aggressively

#### Planned Enhancements
1. **Semantic Compression**: Preserve semantic meaning while reducing token count
2. **Progressive Compression**: Multi-level compression based on token pressure
3. **Context Reconstruction**: Ability to expand compressed context when needed

### 4. Model-Specific Optimization

**Status**: Planning  
**Priority**: Medium

Optimize context and prompts for specific LLM models and their characteristics.

```python
class ModelOptimizer:
    """Optimizes context and prompts for specific models."""
    
    def optimize_for_model(self, context: ProjectContext, model: str) -> OptimizedContext:
        """Optimizes context based on model characteristics."""
        if model.startswith('gpt-4'):
            return self._optimize_for_gpt4(context)
        elif model.startswith('gemini'):
            return self._optimize_for_gemini(context)
        # ... other models
```

## Integration with Autonomous Execution

### Context-Aware Task Decomposition

The TaskDecomposer can use token budget information to optimize task breakdown:

```python
class TaskDecomposer(BaseLLMAgent):
    async def decompose_task(self, task: TaskDefinition) -> ExecutionPlan:
        # Get available token budget
        token_budget = await self.context_manager.get_token_budget(task)
        
        # Optimize decomposition based on available context space
        if token_budget < 4000:
            return await self._create_simplified_plan(task)
        else:
            return await self._create_detailed_plan(task)
```

### Error Recovery with Context Optimization

ErrorRecovery can optimize context usage during recovery attempts:

```python
class ErrorRecovery(BaseLLMAgent):
    async def recover(self, failed_result: CommandResult, plan: ExecutionPlan) -> RecoveryResult:
        # Compress context for recovery analysis
        compressed_context = await self.context_manager.compress_for_recovery(failed_result)
        
        # Generate strategies with optimized context
        strategies = await self._generate_strategies(compressed_context, plan)
        return await self._execute_strategies(strategies)
```

## Performance Metrics

### Token Efficiency Metrics
- **Context Compression Ratio**: Measure compression effectiveness
- **Token Utilization Rate**: How efficiently tokens are used
- **Context Relevance Score**: Measure relevance of included context
- **Compression Quality Score**: Measure information preservation during compression

### Implementation Timeline

#### Phase 1 (Current)
- ✅ Basic ContextManager and ContextCompressor
- ✅ Agent-specific context optimization
- ✅ Automatic compression when approaching token limits

#### Phase 2 (Next Quarter)
- 🔄 Dynamic context prioritization
- 🔄 Enhanced compression strategies
- 🔄 Token budget management

#### Phase 3 (Future)
- 📋 Model-specific optimization
- 📋 Advanced semantic compression
- 📋 Context reconstruction capabilities

## Best Practices

### For Developers
1. **Use ContextManager**: Always use ContextManager for context retrieval
2. **Monitor Token Usage**: Check token usage in logs and optimize accordingly
3. **Test with Compression**: Test agents with compressed context to ensure robustness
4. **Measure Context Relevance**: Ensure included context is relevant to the task

### For System Administrators
1. **Configure Token Limits**: Set appropriate token limits for your LLM provider
2. **Monitor Compression Rates**: Track context compression effectiveness
3. **Optimize Model Selection**: Choose models with appropriate context windows
4. **Track Performance Metrics**: Monitor token efficiency and context quality

This enhanced token management system provides a solid foundation for efficient context handling in the autonomous execution framework while maintaining flexibility for future enhancements.
- Potential rate limiting

### 2. Configuration File Support

**Concept**: Allow users to define model limits in configuration files.

```python
# model_limits.json
{
    "custom-gpt-4": 8192,
    "company-internal-model": 16384,
    "experimental-model-v2": 32768
}
```

```python
def _load_model_limits_from_config(self) -> Dict[str, int]:
    """Load model limits from configuration file."""
    config_paths = [
        Path("model_limits.json"),
        Path.home() / ".autogen" / "model_limits.json",
        Path("/etc/autogen/model_limits.json")
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load {config_path}: {e}")

    return {}
```

**Benefits**:

- User-customizable without code changes
- Supports organization-specific models
- Easy to maintain and update

### 3. Environment Variable Configuration

**Concept**: Allow model limits to be set via environment variables.

```bash
# Environment variables
MODEL_LIMIT_GPT4=8192
MODEL_LIMIT_CUSTOM_MODEL=16384
MODEL_LIMIT_GEMINI_2_0_FLASH=1048576
```

```python
def _load_model_limits_from_env(self) -> Dict[str, int]:
    """Load model limits from environment variables."""
    limits = {}

    for key, value in os.environ.items():
        if key.startswith('MODEL_LIMIT_'):
            # Convert MODEL_LIMIT_GPT4 -> gpt-4
            model_key = key[12:].lower().replace('_', '-')
            try:
                limits[model_key] = int(value)
            except ValueError:
                self.logger.warning(f"Invalid model limit: {key}={value}")

    return limits
```

**Benefits**:

- Easy deployment configuration
- Container-friendly
- No file system dependencies

### 4. Intelligent Name-Based Inference

**Concept**: Infer token limits from model names using pattern matching.

```python
def _infer_limit_from_name(self, model: str) -> Optional[int]:
    """Infer token limit from model name patterns."""
    model_lower = model.lower()

    # Pattern-based inference
    patterns = [
        (r'gemini-2\.0', 1048576),
        (r'gemini-1\.5-pro', 2097152),
        (r'gemini-1\.5', 1048576),
        (r'gpt-4-turbo', 128000),
        (r'gpt-4', 8192),
        (r'gpt-3\.5', 16385),
        (r'claude-3', 200000),
    ]

    for pattern, limit in patterns:
        if re.search(pattern, model_lower):
            return limit

    return None
```

**Benefits**:

- Handles model variants automatically
- Works with custom model names that follow conventions
- No external dependencies

### 5. Hybrid Strategy Implementation

**Concept**: Combine all approaches in a prioritized fallback chain.

```python
async def get_model_limit_smart(self, model: str) -> int:
    """Get model limit using hybrid strategy."""

    # 1. Check cache first
    if model in self._limit_cache:
        return self._limit_cache[model]

    # 2. Environment variables (highest priority)
    env_limits = self._load_model_limits_from_env()
    if model in env_limits:
        limit = env_limits[model]
        self._limit_cache[model] = limit
        return limit

    # 3. Configuration file
    config_limits = self._load_model_limits_from_config()
    if model in config_limits:
        limit = config_limits[model]
        self._limit_cache[model] = limit
        return limit

    # 4. API query (if enabled)
    if self.token_config.get('enable_api_discovery', False):
        try:
            limit = await self.get_model_limit_from_api(model)
            if limit:
                self._limit_cache[model] = limit
                return limit
        except Exception as e:
            self.logger.debug(f"API discovery failed for {model}: {e}")

    # 5. Name-based inference
    inferred_limit = self._infer_limit_from_name(model)
    if inferred_limit:
        self._limit_cache[model] = inferred_limit
        return inferred_limit

    # 6. Hardcoded known limits
    if model in self.known_limits:
        limit = self.known_limits[model]
        self._limit_cache[model] = limit
        return limit

    # 7. Default fallback
    default_limit = self.token_config.get('default_token_limit', 8192)
    self._limit_cache[model] = default_limit
    return default_limit
```

## Implementation Considerations

### Performance

- **Caching**: Essential to avoid repeated expensive operations
- **Async Support**: API calls and probing should be asynchronous
- **Timeout Handling**: All network operations need proper timeouts

### Configuration

New configuration options would be needed:

```python
# Additional token configuration
{
    'enable_api_discovery': False,
    'model_limit_cache_ttl': 3600,  # 1 hour
    'api_discovery_timeout': 5.0,   # 5 seconds
    'model_limits_config_path': 'model_limits.json'
}
```

### Error Handling

- Graceful degradation when advanced methods fail
- Comprehensive logging for debugging
- Fallback to safe defaults in all error cases

### Testing

- Mock API responses for unit tests
- Integration tests with real providers
- Performance tests for caching behavior
- Error scenario testing

## Migration Strategy

1. **Phase 1**: Add configuration file and environment variable support
2. **Phase 2**: Implement name-based inference
3. **Phase 3**: Add API discovery (optional feature)
4. **Phase 4**: Full hybrid strategy with comprehensive caching

## Benefits of Enhanced Implementation

1. **Flexibility**: Users can customize limits without code changes
2. **Accuracy**: Always up-to-date with provider specifications
3. **Scalability**: Handles new models automatically
4. **Reliability**: Multiple fallback strategies ensure robustness
5. **Performance**: Intelligent caching minimizes overhead

## Conclusion

While the current hardcoded approach works well for known models, these enhancements would provide a more flexible, accurate, and maintainable solution for token limit management. The hybrid strategy approach offers the best of all worlds while maintaining backward compatibility and reliability.

The implementation should be done incrementally, starting with the simpler approaches (configuration files, environment variables) and gradually adding more sophisticated features based on user needs and feedback.
