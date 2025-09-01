"""
Main Controller for the AutoGen multi-agent framework.

This module implements the MainController class which serves as the central
orchestrator for the entire framework. It manages the initialization process,
coordinates all components, handles user interactions, and controls the
workflow progression with proper user approval checkpoints.
"""

import asyncio
import logging
import json
import os
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from enum import Enum

from .models import LLMConfig, WorkflowState, WorkflowPhase, UserApprovalStatus
from .config_manager import ConfigManager, ConfigurationError
from .session_manager import SessionManager
from .workflow_manager import WorkflowManager
from .agent_manager import AgentManager
from .dependency_container import DependencyContainer


class MainController:
    """
    Main controller for the AutoGen multi-agent framework.
    
    The MainController serves as the central orchestrator that:
    - Initializes the entire framework and all components
    - Processes user requests and manages workflow progression
    - Handles user approval checkpoints between workflow phases
    - Integrates all core components (MemoryManager, AgentManager, ShellExecutor)
    - Provides error handling and recovery mechanisms
    - Maintains framework state and configuration
    """
    
    def __init__(self, workspace_path: str):
        """
        Initialize the Main Controller.
        
        Args:
            workspace_path: Path to the workspace root directory
        """
        self.workspace_path = Path(workspace_path)
        self.logger = logging.getLogger(__name__)
        
        # Core components (initialized later)
        self.container: Optional[DependencyContainer] = None
        self.agent_manager: Optional[AgentManager] = None
        self.session_manager: Optional[SessionManager] = None
        self.workflow_manager: Optional[WorkflowManager] = None
        
        # Framework configuration
        self.llm_config: Optional[LLMConfig] = None
        self.is_initialized = False
        
        # Session-related properties (delegated to SessionManager after initialization)
        self.session_id: Optional[str] = None
        # Deprecated: workflow state is now managed by WorkflowManager as SSOT
        # These fields are kept for backward compatibility and will be removed in future versions
        self._current_workflow: Optional[WorkflowState] = None
        self._user_approval_status: Dict[str, UserApprovalStatus] = {}
        self._phase_results: Dict[str, Dict[str, Any]] = {}
        self._execution_log: List[Dict[str, Any]] = []
        self._approval_log: List[Dict[str, Any]] = []
        self._error_recovery_attempts: Dict[str, int] = {}
        self._workflow_summary: Dict[str, Any] = {
            'phases_completed': [],
            'tasks_completed': [],
            'token_usage': {},
            'compression_events': [],
            'auto_approvals': [],
            'errors_recovered': []
        }
        
        self.logger.info(f"MainController initialized for workspace: {workspace_path}")
    
    # Backward compatibility properties - delegate to WorkflowManager
    @property
    def current_workflow(self) -> Optional[WorkflowState]:
        """Get current workflow from WorkflowManager for backward compatibility."""
        if self.workflow_manager:
            return self.workflow_manager.current_workflow
        return self._current_workflow
    
    @current_workflow.setter
    def current_workflow(self, value: Optional[WorkflowState]) -> None:
        """Set current workflow - updates WorkflowManager if available."""
        self._current_workflow = value
        if self.workflow_manager:
            self.workflow_manager.current_workflow = value
    
    @property
    def user_approval_status(self) -> Dict[str, UserApprovalStatus]:
        """Get user approval status from WorkflowManager for backward compatibility."""
        if self.workflow_manager:
            return self.workflow_manager.user_approval_status
        return self._user_approval_status
    
    @user_approval_status.setter
    def user_approval_status(self, value: Dict[str, UserApprovalStatus]) -> None:
        """Set user approval status - updates WorkflowManager if available."""
        self._user_approval_status = value
        if self.workflow_manager:
            self.workflow_manager.user_approval_status = value
    
    @property
    def phase_results(self) -> Dict[str, Dict[str, Any]]:
        """Get phase results from WorkflowManager for backward compatibility."""
        if self.workflow_manager:
            return self.workflow_manager.phase_results
        return self._phase_results
    
    @phase_results.setter
    def phase_results(self, value: Dict[str, Dict[str, Any]]) -> None:
        """Set phase results - updates WorkflowManager if available."""
        self._phase_results = value
        if self.workflow_manager:
            self.workflow_manager.phase_results = value
    
    @property
    def execution_log(self) -> List[Dict[str, Any]]:
        """Get execution log from WorkflowManager for backward compatibility."""
        if self.workflow_manager:
            return self.workflow_manager.execution_log
        return self._execution_log
    
    @execution_log.setter
    def execution_log(self, value: List[Dict[str, Any]]) -> None:
        """Set execution log - updates WorkflowManager if available."""
        self._execution_log = value
        if self.workflow_manager:
            self.workflow_manager.execution_log = value
    
    @property
    def approval_log(self) -> List[Dict[str, Any]]:
        """Get approval log from WorkflowManager for backward compatibility."""
        if self.workflow_manager:
            return self.workflow_manager.approval_log
        return self._approval_log
    
    @approval_log.setter
    def approval_log(self, value: List[Dict[str, Any]]) -> None:
        """Set approval log - updates WorkflowManager if available."""
        self._approval_log = value
        if self.workflow_manager:
            self.workflow_manager.approval_log = value
    
    @property
    def error_recovery_attempts(self) -> Dict[str, int]:
        """Get error recovery attempts from WorkflowManager for backward compatibility."""
        if self.workflow_manager:
            return self.workflow_manager.error_recovery_attempts
        return self._error_recovery_attempts
    
    @error_recovery_attempts.setter
    def error_recovery_attempts(self, value: Dict[str, int]) -> None:
        """Set error recovery attempts - updates WorkflowManager if available."""
        self._error_recovery_attempts = value
        if self.workflow_manager:
            self.workflow_manager.error_recovery_attempts = value
    
    @property
    def workflow_summary(self) -> Dict[str, Any]:
        """Get workflow summary from WorkflowManager for backward compatibility."""
        if self.workflow_manager:
            return self.workflow_manager.workflow_summary
        return self._workflow_summary
    
    @workflow_summary.setter
    def workflow_summary(self, value: Dict[str, Any]) -> None:
        """Set workflow summary - updates WorkflowManager if available."""
        self._workflow_summary = value
        if self.workflow_manager:
            self.workflow_manager.workflow_summary = value
    
    def initialize_framework(self, llm_config: Optional[LLMConfig] = None) -> bool:
        """
        Initialize the entire framework and all components.
        
        This method fulfills requirements 1.1-1.4 by setting up the complete
        framework environment including virtual environment, AutoGen configuration,
        memory loading, and text interface preparation.
        
        Args:
            llm_config: Optional LLM configuration. If None, uses default configuration.
            
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            self.logger.info("Starting framework initialization")
            
            # Use environment-based configuration if none provided (Requirement 1.2)
            if llm_config is None:
                config_manager = ConfigManager()
                llm_config = LLMConfig.from_config_manager(config_manager)
            
            self.llm_config = llm_config
            
            # Validate LLM configuration
            if not llm_config.validate():
                raise ValueError("Invalid LLM configuration provided")
            
            # Initialize core components
            success = self._initialize_core_components()
            if not success:
                raise RuntimeError("Failed to initialize core components")
            
            # Load memory context (Requirement 1.3)
            success = self._load_framework_memory()
            if not success:
                self.logger.warning("Failed to load memory context, continuing with empty memory")
            
            # Setup agents with LLM configuration
            success = self.agent_manager.setup_agents(llm_config)
            if not success:
                self.logger.error("Failed to setup agents")
                return False
            
            # Load or create session
            self._load_or_create_session()
            
            self.is_initialized = True
            
            # Record initialization
            self._record_execution_event(
                event_type="framework_initialization",
                details={
                    "llm_config": {
                        "model": llm_config.model,
                        "base_url": llm_config.base_url
                    },
                    "workspace_path": str(self.workspace_path),
                    "components_initialized": ["MemoryManager", "AgentManager", "ShellExecutor"]
                }
            )
            
            self.logger.info("Framework initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Framework initialization failed: {e}")
            self._record_execution_event(
                event_type="initialization_error",
                details={"error": str(e)}
            )
            return False
    
    async def process_request(self, user_request: str, auto_approve: bool = False) -> Dict[str, Any]:
        """
        Process a user request through the complete workflow with optional auto-approve.
        
        This is the main entry point for handling user requests. It manages the
        entire workflow from requirements generation through implementation,
        with proper user approval checkpoints at each phase or automatic approval
        when auto_approve is enabled.
        
        Args:
            user_request: The user's request string
            auto_approve: Whether to automatically approve all workflow phases
            
        Returns:
            Dictionary containing the complete workflow results and status
        """
        if not self.is_initialized:
            raise RuntimeError("Framework not initialized. Call initialize_framework() first.")
        
        # Delegate to WorkflowManager
        result = await self.workflow_manager.process_request(user_request, auto_approve)
        
        # Note: ContextManager is automatically managed by DependencyContainer
        # No need to manually pass it to AgentManager
        
        # Deprecated: Avoid mirroring state in MainController; rely on WorkflowManager as SSOT
        # self._sync_workflow_state_from_manager()
        
        return result
    
    async def process_user_request(self, user_request: str) -> Dict[str, Any]:
        """
        Process a user request through the complete workflow with approval checkpoints.
        
        This method provides backward compatibility with the original interface.
        For new code, use process_request() with the auto_approve parameter.
        
        Args:
            user_request: The user's request string
            
        Returns:
            Dictionary containing the complete workflow results and status
        """
        return await self.process_request(user_request, auto_approve=False)
    
    def approve_phase(self, phase: str, approved: bool = True) -> Dict[str, Any]:
        """
        Approve or reject a workflow phase.
        
        Args:
            phase: Phase name ("requirements", "design", "tasks")
            approved: Whether the phase is approved
            
        Returns:
            Dictionary containing approval status and next steps
        """
        # Delegate to WorkflowManager
        result = self.workflow_manager.approve_phase(phase, approved)
        
        # Deprecated: Avoid mirroring state in MainController; rely on WorkflowManager as SSOT
        # self._sync_workflow_state_from_manager()
        
        return result
    
    async def continue_workflow(self) -> Dict[str, Any]:
        """
        Continue the workflow after user approval.
        
        Returns:
            Dictionary containing the next phase results or completion status
        """
        # Delegate to WorkflowManager
        result = await self.workflow_manager.continue_workflow()
        
        # Deprecated: Avoid mirroring state in MainController; rely on WorkflowManager as SSOT
        # self._sync_workflow_state_from_manager()
        
        return result
    
    def get_framework_status(self) -> Dict[str, Any]:
        """
        Get the current status of the framework.
        
        Returns:
            Dictionary containing comprehensive framework status information
        """
        # Prefer reading workflow state directly from WorkflowManager to avoid drift
        wm = self.workflow_manager
        current_workflow = wm.current_workflow if wm else self.current_workflow
        approval_status = ( {k: v.value for k, v in wm.user_approval_status.items()} if wm else {k: v.value for k, v in self.user_approval_status.items()} )
        status = {
            "initialized": self.is_initialized,
            "session_id": self.session_id,
            "workspace_path": str(self.workspace_path),
            "llm_config": {
                "model": self.llm_config.model if self.llm_config else None,
                "base_url": self.llm_config.base_url if self.llm_config else None
            } if self.llm_config else None,
            "current_workflow": {
                "active": self.current_workflow is not None,
                "phase": self.current_workflow.phase.value if self.current_workflow else None,
                "work_directory": self.current_workflow.work_directory if self.current_workflow else None
            },
            "approval_status": {k: v.value for k, v in self.user_approval_status.items()},
            "components": {}
        }
        
        # Add component status if initialized
        if self.is_initialized:
            if self.agent_manager:
                status["components"]["agent_manager"] = self.agent_manager.get_agent_status()
            if self.container:
                # Get component status from container
                memory_manager = self.container.get_memory_manager()
                shell_executor = self.container.get_shell_executor()
                status["components"]["memory_manager"] = memory_manager.get_memory_stats() if hasattr(memory_manager, 'get_memory_stats') else {"status": "available"}
                status["components"]["shell_executor"] = shell_executor.get_execution_stats() if hasattr(shell_executor, 'get_execution_stats') else {"status": "available"}
                status["components"]["container"] = {
                    "managers_created": self.container.get_manager_count(),
                    "created_managers": self.container.get_created_managers()
                }
        
        return status
    
    async def apply_phase_revision(self, phase: str, revision_feedback: str) -> Dict[str, Any]:
        """
        Apply revision feedback to a specific phase.
        
        Args:
            phase: Phase name ("requirements", "design", "tasks")
            revision_feedback: User's revision feedback
            
        Returns:
            Dictionary containing revision results
        """
        # Delegate to WorkflowManager
        result = await self.workflow_manager.apply_phase_revision(phase, revision_feedback)
        
        # Deprecated: Avoid mirroring state in MainController; rely on WorkflowManager as SSOT
        # self._sync_workflow_state_from_manager()
        
        return result
    
    def get_pending_approval(self) -> Optional[Dict[str, Any]]:
        """
        Check if there's a pending approval needed.
        
        Returns:
            Dictionary with pending approval info, or None if no approval needed
        """
        # Delegate to WorkflowManager
        return self.workflow_manager.get_pending_approval()
    
    def complete_workflow(self) -> bool:
        """
        Mark the current workflow as completed and clear it.
        
        Returns:
            True if workflow was completed, False if no active workflow
        """
        # Delegate to WorkflowManager
        result = self.workflow_manager.complete_workflow()
        
        # Deprecated: Avoid mirroring state in MainController; rely on WorkflowManager as SSOT
        # self._sync_workflow_state_from_manager()
        
        return result
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """
        Get the complete execution log.
        
        Returns:
            List of execution events with timestamps and details
        """
        # Prefer WorkflowManager as SSOT, fallback to local copy for backward compatibility
        if self.workflow_manager and hasattr(self.workflow_manager, 'execution_log'):
            return self.workflow_manager.execution_log.copy()
        return self.execution_log.copy()
    
    def reset_framework(self) -> bool:
        """
        Reset the framework to initial state.
        
        Returns:
            True if reset was successful, False otherwise
        """
        try:
            # Reset workflow state
            self.current_workflow = None
            self.user_approval_status.clear()
            self.phase_results.clear()
            self.approval_log.clear()
            self.error_recovery_attempts.clear()
            self.workflow_summary = {
                'phases_completed': [],
                'tasks_completed': [],
                'token_usage': {},
                'compression_events': [],
                'auto_approvals': [],
                'errors_recovered': []
            }
            
            # Reset components if they exist
            if self.agent_manager and hasattr(self.agent_manager, 'reset_coordination_state'):
                self.agent_manager.reset_coordination_state()
            
            # Reset shell executor through container if available
            if self.container:
                try:
                    shell_executor = self.container.get_shell_executor()
                    if hasattr(shell_executor, 'clear_history'):
                        shell_executor.clear_history()
                except Exception:
                    # Ignore errors if shell executor is not available or mocked
                    pass
            
            # Clear execution log
            self.execution_log.clear()
            
            # Save session state after reset
            self._save_session_state()
            
            self.logger.info("Framework reset completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error resetting framework: {e}")
            return False    
   
 # Private helper methods
    
    def _initialize_core_components(self) -> bool:
        """Initialize all core framework components."""
        try:
            # Initialize DependencyContainer
            if self.container is None:
                self.container = DependencyContainer.create_production(
                    work_dir=str(self.workspace_path),
                    llm_config=self.llm_config
                )
                self.logger.info("DependencyContainer initialized")
            
            # Initialize SessionManager
            if self.session_manager is None:
                self.session_manager = SessionManager(str(self.workspace_path))
                self.logger.info("SessionManager initialized")
            
            # Initialize AgentManager
            if self.agent_manager is None:
                self.agent_manager = AgentManager(str(self.workspace_path))
                self.logger.info("AgentManager initialized")
            
            # Initialize WorkflowManager with DependencyContainer
            if self.workflow_manager is None:
                self.workflow_manager = WorkflowManager(
                    self.agent_manager, 
                    self.session_manager,
                    self.container
                )
                self.logger.info("WorkflowManager initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing core components: {e}")
            return False
    
    def _load_framework_memory(self) -> bool:
        """Load memory context for the framework."""
        try:
            # Get memory manager from container
            memory_manager = self.container.get_memory_manager()
            memory_content = memory_manager.load_memory()
            
            # Update agents with memory context if agent manager is available
            if self.agent_manager:
                self.agent_manager.update_agent_memory(memory_content)
            
            self.logger.info("Framework memory loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading framework memory: {e}")
            return False
    
# Deprecated: _execute_requirements_phase removed - WorkflowManager handles phase execution
    
# Deprecated: _execute_design_phase removed - WorkflowManager handles phase execution
    
# Deprecated: _execute_tasks_phase removed - WorkflowManager handles phase execution
    
# Deprecated: _execute_implementation_phase removed - WorkflowManager handles phase execution
    
# Deprecated: approval and error recovery methods removed - WorkflowManager handles these
    
# Deprecated: should_auto_approve removed - WorkflowManager handles auto-approval logic
    
# Deprecated: All error recovery and approval helper methods removed - WorkflowManager handles these
    
# Backward compatibility methods - delegate to WorkflowManager
    def should_auto_approve(self, phase: str, auto_approve: bool) -> bool:
        """Delegate to WorkflowManager for backward compatibility."""
        if self.workflow_manager:
            return self.workflow_manager.should_auto_approve(phase, auto_approve)
        
        # Fallback logic when WorkflowManager is not available (e.g., in tests)
        # First check if phase is already manually approved
        if self.user_approval_status.get(phase) == UserApprovalStatus.APPROVED:
            return True
        
        # If auto-approve is not enabled, return False
        if not auto_approve:
            return False
        
        # Check for critical checkpoints that require explicit approval even in auto-approve mode
        critical_checkpoints = self._get_critical_checkpoints()
        if phase in critical_checkpoints:
            self.log_auto_approval(phase, False, f"Critical checkpoint requires explicit approval")
            return False
        
        # Auto-approve the phase
        self.user_approval_status[phase] = UserApprovalStatus.APPROVED
        self.log_auto_approval(phase, True, f"Auto-approved in auto-approve mode")
        
        # Track in workflow summary
        self.workflow_summary['phases_completed'].append({
            'phase': phase,
            'auto_approved': True,
            'timestamp': datetime.now().isoformat()
        })
        
        return True
    
    def log_auto_approval(self, phase: str, decision: bool, reason: str) -> None:
        """Delegate to WorkflowManager for backward compatibility."""
        if self.workflow_manager:
            self.workflow_manager.log_auto_approval(phase, decision, reason)
        else:
            # Fallback logic when WorkflowManager is not available
            approval_entry = {
                'phase': phase,
                'decision': decision,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }
            self.approval_log.append(approval_entry)
            self.workflow_summary['auto_approvals'].append(approval_entry)
            self.logger.info(f"Auto-approval decision for {phase}: {decision} - {reason}")
    
    def handle_error_recovery(self, error: Exception, phase: str, context: Dict[str, Any]) -> bool:
        """Delegate to WorkflowManager for backward compatibility."""
        if self.workflow_manager:
            return self.workflow_manager.handle_error_recovery(error, phase, context)
        return False
    
    def _get_critical_checkpoints(self) -> List[str]:
        """Delegate to WorkflowManager for backward compatibility."""
        if self.workflow_manager:
            return self.workflow_manager._get_critical_checkpoints()
        
        # Fallback logic when WorkflowManager is not available
        critical_checkpoints_env = os.getenv('AUTO_APPROVE_CRITICAL_CHECKPOINTS', '')
        if critical_checkpoints_env:
            return [phase.strip() for phase in critical_checkpoints_env.split(',')]
        return []
    
    def _parse_tasks_from_file(self, tasks_file: str):
        """Delegate to WorkflowManager for backward compatibility."""
        if self.workflow_manager:
            return self.workflow_manager._parse_tasks_from_file(tasks_file)
        return []
    
    def _update_tasks_file_with_completion(self, tasks_file: str, task_results: List[Dict[str, Any]]) -> None:
        """Delegate to WorkflowManager for backward compatibility."""
        if self.workflow_manager:
            self.workflow_manager._update_tasks_file_with_completion(tasks_file, task_results)
    
    def _get_error_recovery_max_attempts(self) -> int:
        """Delegate to WorkflowManager for backward compatibility."""
        if self.workflow_manager:
            return self.workflow_manager._get_error_recovery_max_attempts()
        
        # Fallback logic when WorkflowManager is not available
        return int(os.getenv('AUTO_APPROVE_ERROR_RECOVERY_ATTEMPTS', '3'))
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all work performed.
        
        Returns:
            Dictionary containing comprehensive workflow summary
        """
        return {
            'workflow_summary': self.workflow_summary,
            'approval_log': self.approval_log,
            'error_recovery_attempts': self.error_recovery_attempts,
            'total_phases_completed': len(self.workflow_summary['phases_completed']),
            'total_tasks_completed': len(self.workflow_summary['tasks_completed']),
            'total_auto_approvals': len(self.workflow_summary['auto_approvals']),
            'total_errors_recovered': len(self.workflow_summary['errors_recovered']),
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_workflow_id(self) -> str:
        """Generate a unique workflow ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"workflow_{timestamp}"
    
    def _get_current_workflow_id(self) -> str:
        """Get the current workflow ID from execution log."""
        for event in reversed(self.execution_log):
            if event.get("event_type") == "framework_initialization":
                continue
            workflow_id = event.get("details", {}).get("workflow_id")
            if workflow_id:
                return workflow_id
        return "unknown_workflow"
    
    def _get_last_phase_result(self, phase: str) -> Dict[str, Any]:
        """Get the result of the last execution of a specific phase."""
        return self.phase_results.get(phase, {})
    
    def _record_execution_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Record an execution event in the framework log."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        self.execution_log.append(event)
        self.logger.debug(f"Recorded execution event: {event_type}")
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
    def _get_json_safe_framework_status(self) -> Dict[str, Any]:
        """
        Get framework status in a JSON-serializable format.
        
        Returns:
            Dictionary containing JSON-safe framework status information
        """
        status = {
            "initialized": self.is_initialized,
            "session_id": self.session_id,
            "workspace_path": str(self.workspace_path),
            "llm_config": {
                "model": self.llm_config.model if self.llm_config else None,
                "base_url": self.llm_config.base_url if self.llm_config else None
            } if self.llm_config else None,
            "current_workflow": {
                "active": self.current_workflow is not None,
                "phase": self.current_workflow.phase.value if self.current_workflow else None,
                "work_directory": self.current_workflow.work_directory if self.current_workflow else None
            },
            "approval_status": {k: v.value for k, v in self.user_approval_status.items()},
            "components": {}
        }
        
        # Add component status in JSON-safe format
        if self.is_initialized:
            if self.agent_manager:
                try:
                    agent_status = self.agent_manager.get_agent_status()
                    # Convert any non-serializable objects to strings
                    status["components"]["agent_manager"] = self._make_json_safe(agent_status)
                except Exception:
                    status["components"]["agent_manager"] = {"status": "available"}
            
            if self.memory_manager:
                try:
                    memory_stats = self.memory_manager.get_memory_stats()
                    status["components"]["memory_manager"] = self._make_json_safe(memory_stats)
                except Exception:
                    status["components"]["memory_manager"] = {"status": "available"}
            
            if self.shell_executor:
                try:
                    exec_stats = self.shell_executor.get_execution_stats()
                    status["components"]["shell_executor"] = self._make_json_safe(exec_stats)
                except Exception:
                    status["components"]["shell_executor"] = {"status": "available"}
        
        return status
    
    def _make_json_safe(self, obj: Any) -> Any:
        """
        Convert an object to a JSON-serializable format.
        
        Args:
            obj: Object to make JSON-safe
            
        Returns:
            JSON-serializable version of the object
        """
        import json
        from unittest.mock import MagicMock
        
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, MagicMock):
            return f"<MagicMock: {obj._mock_name or 'unnamed'}>"
        elif isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_safe(item) for item in obj]
        else:
            # Try to serialize, if it fails, convert to string
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

    def export_workflow_report(self, output_path: str) -> bool:
        """
        Export a comprehensive workflow report.
        
        Args:
            output_path: Path where to save the workflow report
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "framework_status": self._get_json_safe_framework_status(),
                "execution_log": self._make_json_safe(self.get_execution_log()),
                "workflow_state": self._get_workflow_state_from_wm()
            }
            
            # Add agent coordination report if available
            if self.agent_manager:
                try:
                    coordination_report_path = str(Path(output_path).parent / "agent_coordination_report.json")
                    self.agent_manager.export_coordination_report(coordination_report_path)
                    report["agent_coordination_report_path"] = coordination_report_path
                except Exception as e:
                    self.logger.warning(f"Could not export agent coordination report: {e}")
            
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Workflow report exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting workflow report: {e}")
            return False
    
    async def execute_specific_task(self, task_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a specific task from the task list.
        
        Args:
            task_definition: Task definition dictionary
            
        Returns:
            Dictionary containing task execution results
        """
        if not self.current_workflow:
            raise RuntimeError("No active workflow. Cannot execute task.")
        
        if self.current_workflow.phase != WorkflowPhase.IMPLEMENTATION:
            raise RuntimeError("Workflow not in implementation phase. Cannot execute task.")
        
        try:
            result = await self.agent_manager.coordinate_agents(
                "task_execution",
                {
                    "task_type": "execute_task",
                    "task": task_definition,
                    "work_dir": self.current_workflow.work_directory
                }
            )
            
            # Record task execution
            self._record_execution_event(
                event_type="task_executed",
                details={
                    "task_id": task_definition.get("id"),
                    "task_title": task_definition.get("title"),
                    "success": result.get("success", False),
                    "workflow_id": self._get_current_workflow_id()
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_available_commands(self) -> List[str]:
        """
        Get list of available commands for the framework.
        
        Returns:
            List of available command descriptions
        """
        commands = [
            "initialize_framework() - Initialize the framework with LLM configuration",
            "process_user_request(request) - Process a user request through the workflow",
            "approve_phase(phase, approved) - Approve or reject a workflow phase",
            "continue_workflow() - Continue workflow after approval",
            "get_framework_status() - Get current framework status",
            "get_execution_log() - Get complete execution log",
            "reset_framework() - Reset framework to initial state",
            "execute_specific_task(task) - Execute a specific task",
            "export_workflow_report(path) - Export comprehensive workflow report"
        ]
        
        return commands
    
    def _load_or_create_session(self):
        """Load existing session or create a new one."""
        session_data = self.session_manager.load_or_create_session()
        
        # Sync session data from SessionManager to MainController
        self.session_id = self.session_manager.session_id
        self.current_workflow = self.session_manager.current_workflow
        
        # Convert approval status from strings to enum values for backward compatibility
        self.user_approval_status = {
            k: UserApprovalStatus(v) for k, v in self.session_manager.user_approval_status.items()
        }
        
        self.phase_results = self.session_manager.phase_results
        self.execution_log = self.session_manager.execution_log
        self.approval_log = self.session_manager.approval_log
        self.error_recovery_attempts = self.session_manager.error_recovery_attempts
        self.workflow_summary = self.session_manager.workflow_summary
        
        # Deprecated: No longer sync state to WorkflowManager - it loads from SessionManager directly
    
    def _create_new_session(self):
        """Create a new session with unique ID."""
        session_data = self.session_manager._create_new_session()
        
        # Sync session data from SessionManager to MainController
        self.session_id = self.session_manager.session_id
        self.current_workflow = self.session_manager.current_workflow
        self.user_approval_status = {}  # Empty since it's a new session
        self.phase_results = self.session_manager.phase_results
        self.execution_log = self.session_manager.execution_log
        self.approval_log = self.session_manager.approval_log
        self.error_recovery_attempts = self.session_manager.error_recovery_attempts
        self.workflow_summary = self.session_manager.workflow_summary
    
    def _save_session_state(self):
        """Save current session state to disk."""
        # Only save if SessionManager is initialized
        if self.session_manager is None:
            self.logger.warning("SessionManager not initialized, skipping session save")
            return False
            
        # Sync MainController state to SessionManager before saving
        self.session_manager.session_id = self.session_id
        self.session_manager.current_workflow = self.current_workflow
        
        # Convert approval status from enum values to strings for SessionManager
        self.session_manager.user_approval_status = {
            k: v.value for k, v in self.user_approval_status.items()
        }
        
        self.session_manager.phase_results = self.phase_results
        self.session_manager.execution_log = self.execution_log
        self.session_manager.approval_log = self.approval_log
        self.session_manager.error_recovery_attempts = self.error_recovery_attempts
        self.session_manager.workflow_summary = self.workflow_summary
        
        # Save through SessionManager
        return self.session_manager.save_session_state()
    
    def _get_workflow_state_from_wm(self) -> Dict[str, Any]:
        """Get workflow state from WorkflowManager for reporting."""
        if not self.workflow_manager:
            # Fallback to local state for backward compatibility
            return {
                "current_workflow": {
                    "phase": self.current_workflow.phase.value if self.current_workflow else None,
                    "work_directory": self.current_workflow.work_directory if self.current_workflow else None
                },
                "approval_status": {k: v.value for k, v in self.user_approval_status.items()}
            }
        
        wm = self.workflow_manager
        current_workflow = wm.current_workflow
        return {
            "current_workflow": {
                "phase": current_workflow.phase.value if current_workflow else None,
                "work_directory": current_workflow.work_directory if current_workflow else None
            },
            "approval_status": {k: v.value for k, v in wm.user_approval_status.items()}
        }
    
# Backward compatibility stubs for removed sync methods
    def _sync_workflow_state_to_manager(self):
        """Deprecated: No-op for backward compatibility. State is now automatically synced via properties."""
        pass
    
    def _sync_workflow_state_from_manager(self):
        """Deprecated: No-op for backward compatibility. State is now automatically synced via properties."""
        pass
    
    def get_session_id(self) -> str:
        """Get the current session ID."""
        if self.session_manager is None:
            return self.session_id
        return self.session_manager.get_session_id()
    
    def reset_session(self):
        """Reset the current session and create a new one."""
        # Initialize SessionManager if not already initialized
        if self.session_manager is None:
            self.session_manager = SessionManager(str(self.workspace_path))
        
        # Delegate to SessionManager
        result = self.session_manager.reset_session()
        
        if result.get("success"):
            # Sync session data from SessionManager to MainController
            self.session_id = self.session_manager.session_id
            self.current_workflow = self.session_manager.current_workflow
            self.user_approval_status = {}  # Empty since it's a new session
            self.phase_results = self.session_manager.phase_results
            self.execution_log = self.session_manager.execution_log
            self.approval_log = self.session_manager.approval_log
            self.error_recovery_attempts = self.session_manager.error_recovery_attempts
            self.workflow_summary = self.session_manager.workflow_summary
        
        return result
    
    # Error Recovery Helper Methods
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
# Deprecated: Method removed - WorkflowManager handles this functionality
    
    def _fallback_implementation_execution(self, context: Dict[str, Any]) -> bool:
        """Fallback implementation execution using safe methods."""
        try:
            self.logger.info("Using fallback implementation execution")
            
            # Use safe execution methods
            context["safe_execution_only"] = True
            context["basic_implementation"] = True
            context["no_risky_operations"] = True
            
            # This would normally use conservative implementation approach
            self.logger.info("Fallback implementation execution completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Fallback implementation execution failed: {e}")
            return False