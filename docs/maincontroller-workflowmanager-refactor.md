# Design Doc: MainController and WorkflowManager Boundary Refinement

## 1) Summary

- Problem: MainController and WorkflowManager have duplicated workflow logic, duplicated state, and mirrored utilities (approvals, error recovery, task-file helpers). This causes leaky boundaries, DRY violations, and raises maintenance and correctness risks.
- Goal: Make WorkflowManager the single owner of workflow orchestration and state; slim MainController to a composition facade that wires components, delegates to WorkflowManager, and exposes a stable public API. Remove duplication, consolidate approval/error-recovery logic, and eliminate state mirroring.
- Non-Goals: Changes to AgentManager internals, large-scale redesign of ContextManager, or reworking agents’ logic.

## 2) Context and Problem Statement

Current architecture:
- MainController: Initializes framework components, exposes `process_request` and other public APIs, but also contains its own copies of phase executors, approval handling, error recovery strategies, and task file parsing/updating. It also stores workflow state and performs two-way state sync with WorkflowManager.
- WorkflowManager: Owns the “real” workflow orchestration and is used by `MainController.process_request`; it already implements the same set of phase executors, approval logic, error recovery stubs, and task file helpers; stores workflow state via `SessionManager`.

Primary issues:
- Responsibility overlap and duplicated code:
  - Phase execution: both MC and WM implement `_execute_requirements_phase`, `_execute_design_phase`, `_execute_tasks_phase`, `_execute_implementation_phase`.
  - Approval and error recovery: `should_auto_approve`, `log_auto_approval`, `handle_error_recovery`, and multiple retry/skip/fallback helpers appear in both.
  - Task-file utilities exist twice: `_parse_tasks_from_file` and `_update_tasks_file_with_completion`.
- Duplicated state and sync churn:
  - MC and WM both hold: `current_workflow`, `user_approval_status`, `phase_results`, `execution_log`, `approval_log`, `error_recovery_attempts`, `workflow_summary`; MC syncs back and forth with WM via `_sync_workflow_state_to_manager`/`_sync_workflow_state_from_manager`.
- Enum duplication:
  - `UserApprovalStatus` exists in both MC and WM.

Consequences:
- Higher maintenance burden; higher risk of drift or subtle bugs; unclear source of truth for workflow state; violates SRP for MainController.

## 3) Goals and Non-Goals

Goals
- Make WorkflowManager the single source of truth (SSOT) for workflow state and workflow logic.
- Trim MainController to a thin facade: initialization, dependency wiring, status/reporting passthrough, and orchestration entry points that delegate to WorkflowManager.
- Remove duplicated logic from MainController.
- Consolidate approval and error-recovery logic into WorkflowManager.
- Unify `UserApprovalStatus` enum into a single definition shared by both.

Non-Goals
- Refactoring AgentManager dependency creation or ContextManager lifecycle. We will only adjust the MC↔WM contract and the WM↔SessionManager interactions.
- Moving task-file logic into a new module right now; we’ll let WorkflowManager own it to keep scope focused. (Optional future improvement is noted.)

## 4) Proposed Design

### 4.1 Single Source of Truth: WorkflowManager owns workflow state
- WorkflowManager remains the sole owner of:
  - `current_workflow` (WorkflowState)
  - `user_approval_status` (UserApprovalStatus)
  - `phase_results`
  - `execution_log`
  - `approval_log`
  - `error_recovery_attempts`
  - `workflow_summary`
- SessionManager persistence is performed only from WorkflowManager, via `_save_session_state`/`_load_session_state_from_manager`.
- MainController does not store these fields redundantly and does not perform sync; it queries WorkflowManager for read-only status when needed.

### 4.2 MainController as a thin facade
Keep in MainController:
- Initialization/wiring: `initialize_framework`, `_initialize_core_components`, `_load_or_create_session`, `_save_session_state` for MC’s own minimal data (e.g., session ID), and component instantiation (`AgentManager`, `WorkflowManager`, `MemoryManager`, `TokenManager`, etc.).
- Public entry points delegating to WorkflowManager:
  - `process_request`, `process_user_request` (legacy), `approve_phase`, `continue_workflow`, `apply_phase_revision`, `get_pending_approval`, `complete_workflow`.
- Status/reporting:
  - `get_framework_status` and `export_workflow_report` remain, but they read state through WorkflowManager getters instead of copied fields.

Remove or deprecate in MainController:
- Phase executors:
  - `_execute_requirements_phase`, `_execute_design_phase`, `_execute_tasks_phase`, `_execute_implementation_phase`.
- Approval and error recovery:
  - `should_auto_approve`, `log_auto_approval`, `handle_error_recovery`, and all retry/skip/fallback helpers, `_get_critical_checkpoints`.
- Task helpers:
  - `_parse_tasks_from_file`, `_update_tasks_file_with_completion`.
- Duplicate state:
  - `current_workflow`, `user_approval_status`, `phase_results`, `execution_log`, `approval_log`, `error_recovery_attempts`, `workflow_summary`.
- Sync helpers:
  - `_sync_workflow_state_to_manager` and `_sync_workflow_state_from_manager`.

### 4.3 Consolidate approval + error recovery in WorkflowManager
- WorkflowManager will continue to own `should_auto_approve`, `log_auto_approval`, `handle_error_recovery`, and the related helpers.
- Implement the real logic in WorkflowManager where MC currently has the more complete logic:
  - Implement meaningful versions of `_retry_with_modified_parameters`, `_skip_non_critical_steps`, `_use_fallback_implementation`, and `_modify_parameters_for_retry` inside WorkflowManager (aligning with MC’s current real behavior).
- This ensures a single, testable logic path for approvals and recovery.

### 4.4 Unify enums
- Move `UserApprovalStatus` to a shared definition (preferably in `models.py`).
- Update both MainController and WorkflowManager to `from .models import UserApprovalStatus`, removing duplicate enum definitions.

### 4.5 Task file helpers
- Keep `_parse_tasks_from_file` and `_update_tasks_file_with_completion` in WorkflowManager as the single implementation for now.
- MainController will not implement or call its own copies.

### 4.6 API surface and compatibility
- MainController public methods remain:
  - `initialize_framework`, `process_request`, `process_user_request`, `approve_phase`, `continue_workflow`, `apply_phase_revision`, `get_pending_approval`, `complete_workflow`, `get_framework_status`, `export_workflow_report`, `execute_specific_task` (will delegate to WM/AM).
- Deprecate private methods removed from MainController (no external API impact).
- `get_framework_status` continues to return a superset view; data regarding workflow is read through WorkflowManager.

## 5) Detailed Implementation Plan

### 5.1 Models: unify UserApprovalStatus
- Move the enum to `models.py`, e.g.:
  - `class UserApprovalStatus(Enum): PENDING, APPROVED, REJECTED, NEEDS_REVISION`
- Update imports in both MainController and WorkflowManager to `from .models import UserApprovalStatus`.
- Remove local enum definitions from MainController and WorkflowManager.

### 5.2 WorkflowManager: solidify ownership
- Keep and complete implementations:
  - `process_request`, `continue_workflow`, `approve_phase`, `apply_phase_revision`, `get_pending_approval`, `complete_workflow`.
  - `_execute_requirements_phase`, `_execute_design_phase`, `_execute_tasks_phase`, `_execute_implementation_phase`.
  - `should_auto_approve`, `_get_critical_checkpoints`, `log_auto_approval`.
  - `handle_error_recovery`, `_get_error_recovery_max_attempts`, and strategy helpers:
    - `_retry_with_modified_parameters`
    - `_skip_non_critical_steps`
    - `_use_fallback_implementation`
    - `_modify_parameters_for_retry`
  - Task helpers: `_parse_tasks_from_file`, `_update_tasks_file_with_completion`.
- Replace placeholder implementations of error-recovery strategies with the concrete logic currently residing in MainController:
  - For example, make `_modify_parameters_for_retry` add timeout/backoff, `simplified_mode` flags, `safe_mode`, etc. based on error patterns.
  - Make `_skip_non_critical_steps` create simplified contexts and re-run coordination accordingly (even if minimal on first pass).
  - Make `_use_fallback_implementation` return to a conservative flow consistent with MC’s current fallback stubs.

### 5.3 MainController: remove duplication and use WM’s state
- Remove from MC:
  - `_execute_requirements_phase`, `_execute_design_phase`, `_execute_tasks_phase`, `_execute_implementation_phase`.
  - `should_auto_approve`, `log_auto_approval`, `handle_error_recovery`, and all retry/skip/fallback helpers, `_get_critical_checkpoints`.
  - `_parse_tasks_from_file`, `_update_tasks_file_with_completion`.
- Remove duplicate state fields and related sync logic in MC:
  - `current_workflow`, `user_approval_status`, `phase_results`, `execution_log`, `approval_log`, `error_recovery_attempts`, `workflow_summary`.
  - `_sync_workflow_state_to_manager`/`_sync_workflow_state_from_manager` are removed.
- Adjust methods to read from WM:
  - `get_framework_status`: build component status as today, but for workflow details (phase, work_directory, approvals), ask WorkflowManager for current state. If needed, expose a `get_status()` or getter methods in WM to produce a JSON-safe status dict.
  - `export_workflow_report`: include data obtained via WM; still export AgentManager’s `export_coordination_report` if available.
  - `process_request`, `approve_phase`, `continue_workflow`, `apply_phase_revision`, `get_pending_approval`, `complete_workflow`: delegate directly to WM and return WM’s results.
  - `execute_specific_task`: ensure it delegates to `AgentManager` with workflow info sourced from WM (or continue as-is if WM exposes current `work_directory`).

### 5.4 SessionManager interactions
- Ensure WorkflowManager remains responsible for saving/loading workflow state via `SessionManager`. MainController no longer persists workflow fields; it only persists its own minimal config if any.
- MainController’s `_load_or_create_session` should still initialize session via `SessionManager`, but not copy/sync workflow fields; instead, WM is fed the `SessionManager` at construction and loads state itself.

### 5.5 Logging/reporting
- Keep WM responsible for workflow execution events. MC can still add a top-level “framework initialization” event as today if needed for audit continuity, but should not log per-phase workflow events; that’s WM’s job.
- Ensure `get_framework_status` and `export_workflow_report` use WM’s logs/summary rather than MC’s copies.

### 5.6 Backward compatibility
- No public API removal expected. MC’s public methods remain, but internals change to delegate fully to WM.
- For internal code that referenced MC’s private phase methods (none expected in public API), we remove them—tests must be updated accordingly.

## 6) Testing Strategy

### 6.1 Unit tests
- MainController unit tests:
  - Verify that `initialize_framework` wires dependencies and WM is created and used.
  - Verify `process_request` delegates to WM and returns results.
  - Verify `get_framework_status` composes data (with WM-provided workflow info) correctly.
  - Verify `export_workflow_report` works and includes WM’s state.
- WorkflowManager unit tests:
  - Test end-to-end phase progression and state persistence via `SessionManager`.
  - Test approval logic (manual and auto-approve) from `should_auto_approve` and `_get_critical_checkpoints`.
  - Test error recovery strategies: ensure the updated (non-placeholder) logic executes and impacts outcomes as expected.
  - Test task-file parsing/updating on realistic `tasks.md` content.

### 6.2 Integration tests
- `test_real_main_controller` and `test_workflow_manager_integration`:
  - Validate that a user request flows through WM; no state duplication in MC.
  - Validate that approvals and `continue_workflow` work with WM state as SSOT.

### 6.3 Regression checks
- Ensure features like `apply_phase_revision` and `get_pending_approval` behave identically (or better) with WM as SSOT.

## 7) Migration/Refactor Plan

Phase 1: Enum Unification and WM completion
- Move `UserApprovalStatus` to models and update imports.
- Port MC’s real error recovery logic into WM’s strategy methods (replace placeholders).
- Ensure WM tests cover the enhanced logic.

Phase 2: Slim MC and remove duplication/state
- Remove duplicated private methods and state from MC.
- Update MC’s `get_framework_status`/`export_workflow_report` to read from WM.
- Remove sync methods and copy state off MC.

Phase 3: Tests and Cleanup
- Update unit tests to align with WM SSOT design.
- Ensure integration tests pass.
- Add deprecation notice in code comments where applicable.

## 8) Risks and Mitigations
- Risk: Hidden test reliance on MC’s private methods or state.
  - Mitigation: Keep public interfaces stable, update tests to go through WM; add targeted integration tests.
- Risk: Logging/reporting changes might alter expected structures.
  - Mitigation: Keep event schemas stable where feasible; add adapters or enrich WM logs to preserve expectations.
- Risk: Error recovery strategy differences could change behavior.
  - Mitigation: Port logic faithfully from MC to WM; add thorough tests with representative error scenarios.

## 9) Acceptance Criteria
- Workflow state exists only in WorkflowManager; MainController does not store or sync workflow fields.
- No duplicated phase executors, approval logic, error recovery, or task helpers in MainController.
- Single `UserApprovalStatus` enum in models; both MC and WM use it.
- All unit and integration tests pass; new tests cover WM’s enhanced recovery logic.
- Public API methods on MainController continue to work and delegate fully to WorkflowManager.

## 10) Open Questions
- Should WorkflowManager expose a structured `get_status()` for MainController instead of MC assembling status piecemeal?
- Should we extract a small `TaskFileService` later to reduce WM’s size? (Defer for now.)
- Do we want MainController to enforce any policy gates (e.g., critical checkpoints) globally, or leave everything to WM?
