# Token Management Future Enhancements

## Overview

This document outlines potential improvements to the TokenManager system for more dynamic and intelligent model token limit detection. The current implementation uses hardcoded model limits with fallback to configurable defaults, which works well but could be enhanced for better flexibility and accuracy.

## Current Implementation

The current TokenManager uses a static dictionary approach:

```python
default_limits = {
    'models/gemini-2.0-flash': 1048576,
    'models/gemini-1.5-pro': 2097152,
    'gpt-4': 8192,
    'gpt-4-turbo': 128000,
    # ... more hardcoded values
}
```

For unknown models, it falls back to a configurable default limit (typically 8192 tokens).

## Proposed Enhancements

### 1. API-Based Model Limit Discovery

**Concept**: Query the LLM provider's API to get actual model specifications.

```python
async def get_model_limit_from_api(self, model: str) -> Optional[int]:
    """Query the API for model specifications."""
    try:
        # For OpenAI-compatible APIs
        response = await self.client.get(f"/models/{model}")
        return response.get('context_length')
    except Exception as e:
        self.logger.debug(f"API query failed for {model}: {e}")
        return None
```

**Benefits**:

- Always up-to-date with provider changes
- Supports new models automatically
- Accurate limits for custom/fine-tuned models

**Challenges**:

- Requires API support (not all providers expose this)
- Network dependency
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
