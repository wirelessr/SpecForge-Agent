### Base Agent, Context Manager, and Token Manager Refactor Plan

#### Overview
This plan consolidates token and context responsibilities into the right layers, makes `ContextManager` and `TokenManager` required for `BaseLLMAgent`, removes duplicated compression/truncation logic, and fixes brittle AutoGen wiring. The result is a thinner, deterministic base agent; a single source of truth for context preparation and compression; and lifecycle-driven token tracking.

---

### Problems Identified
- **Duplication and divergence** in `autogen_framework/agents/base_agent.py` vs `autogen_framework/context_manager.py`:
  - Base agent implements compression, truncation, estimation, and formatting that belongs to context layer.
  - Recursive compression paths (compression uses `generate_response` which re-enters token checks).
- **Brittle AutoGen integration**:
  - Mixed `OpenAIChatCompletionClient` with Gemini `ModelInfo`.
  - Referenced `ConversableAgent` type without import while actually constructing `AssistantAgent`.
  - Eager agent reinitialization on any context change.
- **Token logic scattered**:
  - Heuristics for token estimates live in base agent; separate two-phase logic also lives in `ContextManager`.
- **Compression is a no-op** in `ContextManager`:
  - `_dict_to_context` returns original context, so compressed content is not actually used.

---

### Target Architecture
- **BaseLLMAgent (thin orchestrator)**
  - Requires `TokenManager` and `ContextManager` in the constructor.
  - Obtains prepared prompts/messages from `ContextManager`.
  - Sends/receives messages; reports token usage to `TokenManager`.
  - No compression/truncation/estimation logic.

- **ContextManager (single source of truth)**
  - Builds agent-specific context objects.
  - Estimates size, decides compression via `TokenManager` thresholds.
  - Performs actual compression using `ContextCompressor` and reconstructs compressed contexts.
  - Returns a prepared system prompt and message list for the agent.

- **TokenManager (lifecycle-based tracking)**
  - Provides begin/end-turn hooks and a centralized heuristic when actual token counts are unavailable.
  - Supplies target reduction guidance when above threshold.

---

### Module-Level Changes

#### BaseLLMAgent (`autogen_framework/agents/base_agent.py`)
- **Make dependencies mandatory**: `token_manager`, `context_manager` required in `__init__`.
- **Remove compression/truncation/estimation** from base agent:
  - Delete: `_perform_context_compression`, `compress_context`, `compress_and_replace_context`, `truncate_context`, `_apply_additional_truncation`, `_apply_oldest_first_truncation`, `_estimate_static_content_tokens`.
  - Remove token-limit checks inside `generate_response`.
- **Normalize AutoGen wiring**:
  - Introduce a `ModelClientFactory` (new module) to create the correct client based on `LLMConfig` (provider, base_url, model, api_key).
  - Use `AssistantAgent` consistently. Remove stray `ConversableAgent` hints.
  - Avoid destroying the agent on every context update; only rebuild on material client/model changes.
- **Simplify prompt flow**:
  - Accept a prepared `system_prompt` and `messages` from `ContextManager`.
  - Keep `conversation_history` capped; do not embed history into the system prompt.
- **Token lifecycle**:
  - Call `token_manager.begin_turn(prepared.estimated_tokens)` before send.
  - After response, call `token_manager.end_turn(actual_tokens, model, operation)`.
  - Delegate any heuristic-only estimation to `TokenManager` when actual tokens are not available from the client.

Illustrative API surface:

```python
class BaseLLMAgent(ABC):
    async def process_task(self, task_input: dict) -> dict:
        context_spec = self.get_context_requirements(task_input)
        prepared = await self.context_manager.prepare_for_agent(
            agent_name=self.name,
            context_spec=context_spec,
            task_input=task_input,
        )
        self._ensure_agent(prepared.system_prompt)
        self.token_manager.begin_turn(prepared.estimated_tokens)
        response = await self._send(prepared.messages)
        self.token_manager.end_turn(self._extract_actual_tokens(response), self.llm_config.model, "process_task")
        return await self._process_task_impl({**task_input, "agent_response": response})
```

#### ContextManager (`autogen_framework/context_manager.py`)
- **New entrypoint**: `prepare_for_agent(agent_name, context_spec, task_input) -> PreparedContext` with fields:
  - `system_prompt: str`
  - `messages: list[dict]` (excluding the system message)
  - `estimated_tokens: int`
- **Replace `_compress_if_needed` with real compression**:
  - Compute `target_reduction` using model limits from `TokenManager` and current estimated tokens.
  - Call `ContextCompressor.compress_context(dict, target_reduction)` and on success reconstruct compressed context via `_dict_to_context`.
  - Ensure `_dict_to_context` produces a usable compressed context object instead of returning the original.
- **Centralize context utilities**:
  - Move `_context_to_string`, `_context_to_dict`, `_dict_to_context` into `autogen_framework/context_utils.py` for reuse and consistency.
- **Performance and caching**:
  - Keep refresh intervals configurable via `ConfigManager` and avoid redundant project re-walks.

Illustrative API:

```python
@dataclass
class PreparedContext:
    system_prompt: str
    messages: list[dict]
    estimated_tokens: int

class ContextManager:
    async def prepare_for_agent(self, agent_name: str, context_spec: ContextSpec, task_input: dict) -> PreparedContext: ...
```

#### TokenManager (`autogen_framework/token_manager.py`)
- **Lifecycle hooks**:
  - `begin_turn(estimated_tokens: Optional[int]) -> None`
  - `should_compress(current_tokens: int, model: str) -> tuple[bool, Optional[float]]` returning a target reduction when needed.
  - `end_turn(actual_tokens: Optional[int], model: str, operation: str) -> None`
- **Single heuristic location**:
  - If actual counts are unavailable from the client, provide a consistent fallback estimation here (not in base agent).
- **Configuration hardening**:
  - Normalize model limits; move questionable defaults to `ConfigManager` and document them.

---

### Implementation Steps (PR-by-PR)
1) **Client wiring and mandatory deps**
   - Add `ModelClientFactory` and fix AutoGen model/client mismatch.
   - Make `TokenManager` and `ContextManager` required in `BaseLLMAgent.__init__`.
   - Remove stale type hints; stop reinitializing agent on benign context updates.

2) **Context utilities extraction**
   - Create `autogen_framework/context_utils.py` for dict/string conversions and token estimation helpers.
   - Update `ContextManager` to use shared utilities.

3) **Prepared context flow**
   - Implement `ContextManager.prepare_for_agent` returning `PreparedContext`.
   - Replace `_compress_if_needed` with actual compression + reconstruction.

4) **Token lifecycle API**
   - Add `begin_turn`, `should_compress`, `end_turn` in `TokenManager`.
   - Migrate estimations/threshold checks from `BaseLLMAgent` to `ContextManager` + `TokenManager`.

5) **Base agent simplification**
   - Remove all compression/truncation/estimation code from `BaseLLMAgent`.
   - `generate_response`/`process_task` use: prepare → begin_turn → send → end_turn.

6) **Tests and docs**
   - Update unit/integration tests for new flows and mandatory dependencies.
   - Document the new responsibilities and APIs.

---

### Risks and Mitigations
- **Compression fidelity**: Add round-trip tests to ensure `_dict_to_context` reconstructs a usable compressed context.
- **AutoGen API constraints**: If system message cannot be updated in-place, limit re-inits to model/client changes only.
- **Token counts unavailable**: Keep a documented heuristic in `TokenManager` as fallback.

---

### Testing Plan
- Base agent no longer performs compression/truncation; verify via unit tests.
- `ContextManager.prepare_for_agent` produces smaller contexts when thresholds exceeded.
- Token lifecycle events recorded with coherent usage stats and turn boundaries.
- Integration tests for Plan/Design/Implement flows continue to pass with reduced reinitializations.

---

### Acceptance Criteria
- `BaseLLMAgent` contains no compression/truncation code and does not decide about token thresholds.
- `ContextManager` returns actually compressed contexts and usable prompts/messages.
- `TokenManager` lifecycle APIs are used from the base agent path and reflect accurate usage history.
- No unnecessary agent reinitializations during normal context updates.
- All tests pass.

---

### Follow-ups
- Add per-section compression priorities in `ContextCompressor`.
- Support provider capability discovery for model limits.
- Streaming tokens to improve actual usage measurement.


