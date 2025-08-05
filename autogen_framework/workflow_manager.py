"""
Workflow Manager for the AutoGen multi-agent framework.

This module implements the WorkflowManager class which handles all workflow
orchestration logic in isolation. It manages workflow state, user approvals,
phase transitions, and coordinates agent operations through AgentManager.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from enum import Enum

from .models import WorkflowState, WorkflowPhase
from .session_manager import SessionManager


class UserApprovalStatus(Enum):
    """Status of user approval for workflow phases."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class WorkflowManager:
    """
    Workflow manager for handling all workflow orchestration logic.
    
    The WorkflowManager is responsible for:
    - Owning the business logic of the workflow sequence
    - Managing WorkflowState and UserApprovalStatus
    - Handling phase approvals and revisions
    - Orchestrating agent operations through AgentManager
    - Supporting both auto approve and manual approval modes
    - Coordinating workflow progression and state transitions
    """
    
    def __init__(self, agent_manager, session_manager: SessionManager):
        """
        Initialize the Workflow Manager.
        
        Args:
            agent_manager: AgentManager instance for coordinating agents
            session_manager: SessionManager instance for session persistence
        """
        self.agent_manager = agent_manager
        self.session_manager = session_manager
        self.logger = logging.getLogger(__name__)
        
        # Workflow state (managed through SessionManager)
        self.current_workflow: Optional[WorkflowState] = None
        self.user_approval_status: Dict[str, UserApprovalStatus] = {}
        self.phase_results: Dict[str, Dict[str, Any]] = {}
        self.execution_log: List[Dict[str, Any]] = []
        self.approval_log: List[Dict[str, Any]] = []
        self.error_recovery_attempts: Dict[str, int] = {}
        self.workflow_summary: Dict[str, Any] = {
            'phases_completed': [],
            'tasks_completed': [],
            'token_usage': {},
            'compression_events': [],
            'auto_approvals': [],
            'errors_recovered': []
        }
        
        # Load existing session state from SessionManager
        self._load_session_state_from_manager()
        
        self.logger.info("WorkflowManager initialized")
    
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
        # Check if there's already an active workflow
        if self.current_workflow is not None:
            return {
                "success": False,
                "error": "Active workflow in progress. Please complete current workflow first.",
                "current_workflow": {
                    "phase": self.current_workflow.phase.value,
                    "work_directory": self.current_workflow.work_directory
                }
            }
        
        self.logger.info(f"Processing user request: {user_request[:100]}...")
        
        if auto_approve:
            self.logger.info("Auto-approve mode enabled - proceeding through all phases automatically")
        
        # Initialize workflow state
        workflow_id = self._generate_workflow_id()
        self.current_workflow = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory=""
        )
        
        # Reset workflow summary for new workflow
        self.workflow_summary = {
            'phases_completed': [],
            'tasks_completed': [],
            'token_usage': {},
            'compression_events': [],
            'auto_approvals': [],
            'errors_recovered': []
        }
        
        # Save session state after workflow initialization
        self._save_session_state()
        
        # Note: approval status is managed separately and persists across requests
        # This allows for pre-approval of phases in testing scenarios
        
        workflow_result = {
            "workflow_id": workflow_id,
            "user_request": user_request,
            "phases": {},
            "current_phase": None,
            "success": False,
            "requires_user_approval": False,
            "approval_needed_for": None,
            "auto_approve_enabled": auto_approve
        }
        
        try:
            # Phase 1: Requirements Generation
            requirements_result = await self._execute_requirements_phase(user_request, workflow_id)
            workflow_result["phases"]["requirements"] = requirements_result
            workflow_result["current_phase"] = "requirements"
            
            if not requirements_result.get("success", False):
                error_msg = requirements_result.get("error", "Requirements generation failed")
                raise RuntimeError(error_msg)
            
            # Check if user approval is needed for requirements
            requirements_approved = self.should_auto_approve("requirements", auto_approve)
            if not requirements_approved:
                workflow_result["requires_user_approval"] = True
                workflow_result["approval_needed_for"] = "requirements"
                workflow_result["requirements_path"] = requirements_result.get("requirements_path")
                return workflow_result
            
            # Phase 2: Design Generation (only if requirements approved)
            design_result = await self._execute_design_phase(requirements_result, workflow_id)
            workflow_result["phases"]["design"] = design_result
            workflow_result["current_phase"] = "design"
            
            if not design_result.get("success", False):
                error_msg = design_result.get("error", "Design generation failed")
                raise RuntimeError(error_msg)
            
            # Check if user approval is needed for design
            design_approved = self.should_auto_approve("design", auto_approve)
            if not design_approved:
                workflow_result["requires_user_approval"] = True
                workflow_result["approval_needed_for"] = "design"
                workflow_result["design_path"] = design_result.get("design_path")
                return workflow_result
            
            # Phase 3: Task Generation (only if design approved)
            tasks_result = await self._execute_tasks_phase(design_result, requirements_result, workflow_id)
            workflow_result["phases"]["tasks"] = tasks_result
            workflow_result["current_phase"] = "tasks"
            
            if not tasks_result.get("success", False):
                error_msg = tasks_result.get("error", "Task generation failed")
                raise RuntimeError(error_msg)
            
            # Check if user approval is needed for tasks
            tasks_approved = self.should_auto_approve("tasks", auto_approve)
            if not tasks_approved:
                workflow_result["requires_user_approval"] = True
                workflow_result["approval_needed_for"] = "tasks"
                workflow_result["tasks_path"] = tasks_result.get("tasks_file")
                return workflow_result
            
            # Phase 4: Implementation Preparation (only if tasks approved)
            implementation_result = await self._execute_implementation_phase(tasks_result, workflow_id)
            workflow_result["phases"]["implementation"] = implementation_result
            workflow_result["current_phase"] = "implementation"
            
            # Mark workflow as completed
            self.current_workflow.phase = WorkflowPhase.COMPLETED
            workflow_result["success"] = True
            
            # Complete and clear the workflow
            self.complete_workflow()
            
            self.logger.info(f"User request processed successfully: {workflow_id}")
            return workflow_result
            
        except Exception as e:
            self.logger.error(f"Error processing user request: {e}")
            workflow_result["error"] = str(e)
            
            self._record_execution_event(
                event_type="workflow_error",
                details={
                    "workflow_id": workflow_id,
                    "error": str(e),
                    "failed_phase": workflow_result.get("current_phase")
                }
            )
            
            return workflow_result
    
    async def continue_workflow(self) -> Dict[str, Any]:
        """
        Continue the workflow after user approval.
        
        Returns:
            Dictionary containing the next phase results or completion status
        """
        if not self.current_workflow:
            raise RuntimeError("No active workflow to continue")
        
        workflow_id = self._get_current_workflow_id()
        
        # Determine next phase based on current state and approvals
        if (self.current_workflow.phase == WorkflowPhase.PLANNING and 
            self._is_phase_approved("requirements")):
            
            # Continue to design phase
            requirements_result = self._get_last_phase_result("requirements")
            design_result = await self._execute_design_phase(requirements_result, workflow_id)
            
            # Update workflow phase after successful execution
            if design_result.get("success"):
                self.current_workflow.phase = WorkflowPhase.DESIGN
            
            # Save session state after phase update
            self._save_session_state()
            
            return {
                "phase": "design",
                "result": design_result,
                "requires_approval": not self._is_phase_approved("design")
            }
            
        elif (self.current_workflow.phase == WorkflowPhase.DESIGN and 
              self._is_phase_approved("design")):
            
            # Continue to tasks phase
            design_result = self._get_last_phase_result("design")
            requirements_result = self._get_last_phase_result("requirements")
            tasks_result = await self._execute_tasks_phase(design_result, requirements_result, workflow_id)
            
            # Update workflow phase after successful execution
            if tasks_result.get("success"):
                self.current_workflow.phase = WorkflowPhase.TASK_GENERATION
            
            # Save session state after phase update
            self._save_session_state()
            
            return {
                "phase": "tasks",
                "result": tasks_result,
                "requires_approval": not self._is_phase_approved("tasks")
            }
            
        elif (self.current_workflow.phase == WorkflowPhase.TASK_GENERATION and 
              self._is_phase_approved("tasks")):
            
            # Continue to implementation phase
            tasks_result = self._get_last_phase_result("tasks")
            implementation_result = await self._execute_implementation_phase(tasks_result, workflow_id)
            
            # Mark workflow as completed and clean up
            self.current_workflow.phase = WorkflowPhase.COMPLETED
            self.complete_workflow()
            
            # Save session state after completion
            self._save_session_state()
            
            return {
                "phase": "implementation",
                "result": implementation_result,
                "workflow_completed": True
            }
        
        else:
            return {
                "error": "Cannot continue workflow. Check approval status and current phase.",
                "current_phase": self.current_workflow.phase.value,
                "approval_status": {k: v.value for k, v in self.user_approval_status.items()}
            }
    
    def approve_phase(self, phase: str, approved: bool = True) -> Dict[str, Any]:
        """
        Approve or reject a workflow phase.
        
        Args:
            phase: Phase name ("requirements", "design", "tasks")
            approved: Whether the phase is approved
            
        Returns:
            Dictionary containing approval status and next steps
        """
        # Validate phase name
        valid_phases = ["requirements", "design", "tasks"]
        if phase not in valid_phases:
            return {
                "success": False,
                "error": f"Invalid phase '{phase}'. Valid phases are: {', '.join(valid_phases)}"
            }
        
        # Check if there's an active workflow
        if not self.current_workflow:
            return {
                "success": False,
                "error": "No active workflow found. Please submit a request first using --request."
            }
        
        # Check if the phase exists and can be approved
        phase_file_map = {
            "requirements": "requirements.md",
            "design": "design.md", 
            "tasks": "tasks.md"
        }
        
        if self.current_workflow.work_directory:
            phase_file = Path(self.current_workflow.work_directory) / phase_file_map[phase]
            if not phase_file.exists():
                return {
                    "success": False,
                    "error": f"Cannot approve {phase} phase. The {phase_file_map[phase]} file does not exist. Please ensure the phase has been generated first."
                }
        
        status = UserApprovalStatus.APPROVED if approved else UserApprovalStatus.REJECTED
        self.user_approval_status[phase] = status
        
        self.logger.info(f"Phase '{phase}' {'approved' if approved else 'rejected'} by user")
        
        # Record approval event
        self._record_execution_event(
            event_type="phase_approval",
            details={
                "phase": phase,
                "approved": approved,
                "workflow_id": self._get_current_workflow_id()
            }
        )
        
        # Save session state after approval
        self._save_session_state()
        
        result = {
            "phase": phase,
            "approved": approved,
            "status": status.value,
            "can_proceed": approved
        }
        
        if approved:
            result["message"] = f"Phase '{phase}' approved. Ready to proceed to next phase."
        else:
            result["message"] = f"Phase '{phase}' rejected. Please revise and resubmit."
        
        return result
    
    async def apply_phase_revision(self, phase: str, revision_feedback: str) -> Dict[str, Any]:
        """
        Apply revision feedback to a specific phase.
        
        Args:
            phase: Phase name ("requirements", "design", "tasks")
            revision_feedback: User's revision feedback
            
        Returns:
            Dictionary containing revision results
        """
        if phase not in ["requirements", "design", "tasks"]:
            return {"success": False, "error": f"Invalid phase: {phase}"}
        
        if not self.current_workflow:
            return {"success": False, "error": "No active workflow"}
        
        try:
            self.logger.info(f"Applying revision to {phase} phase: {revision_feedback[:100]}...")
            
            # Get the current phase result
            phase_result = self._get_last_phase_result(phase)
            if not phase_result:
                return {"success": False, "error": f"No {phase} result found"}
            
            # Prepare revision context
            revision_context = {
                "phase": phase,
                "revision_feedback": revision_feedback,
                "current_result": phase_result,
                "work_directory": self.current_workflow.work_directory
            }
            
            # Apply revision through agent coordination
            revision_result = await self.agent_manager.coordinate_agents(
                f"{phase}_revision",
                revision_context
            )
            
            if revision_result.get("success"):
                # Update the phase result
                self.phase_results[phase] = revision_result
                
                # Reset approval status for this phase
                if phase in self.user_approval_status:
                    del self.user_approval_status[phase]
                
                # Record revision event
                self._record_execution_event(
                    event_type="phase_revision",
                    details={
                        "phase": phase,
                        "feedback": revision_feedback,
                        "workflow_id": self._get_current_workflow_id()
                    }
                )
                
                # Save session state after revision
                self._save_session_state()
                
                return {
                    "success": True,
                    "message": f"{phase.title()} phase revised successfully",
                    "updated_path": revision_result.get(f"{phase}_path")
                }
            else:
                return {
                    "success": False,
                    "error": revision_result.get("error", "Revision failed")
                }
                
        except Exception as e:
            self.logger.error(f"Error applying revision to {phase}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_pending_approval(self) -> Optional[Dict[str, Any]]:
        """
        Check if there's a pending approval needed.
        
        Returns:
            Dictionary with pending approval info, or None if no approval needed
        """
        if not self.current_workflow:
            return None
        
        current_phase = self.current_workflow.phase
        
        if current_phase == WorkflowPhase.PLANNING and not self._is_phase_approved("requirements"):
            return {
                "phase": "requirements",
                "phase_name": "Requirements",
                "description": "Requirements document needs approval"
            }
        elif current_phase == WorkflowPhase.DESIGN and not self._is_phase_approved("design"):
            return {
                "phase": "design", 
                "phase_name": "Design",
                "description": "Design document needs approval"
            }
        elif current_phase == WorkflowPhase.TASK_GENERATION and not self._is_phase_approved("tasks"):
            return {
                "phase": "tasks",
                "phase_name": "Tasks", 
                "description": "Task list needs approval"
            }
        
        return None
    
    def complete_workflow(self) -> bool:
        """
        Mark the current workflow as completed and clear it.
        
        Returns:
            True if workflow was completed, False if no active workflow
        """
        if self.current_workflow is None:
            return False
        
        self.logger.info(f"Completing workflow in phase: {self.current_workflow.phase.value}")
        
        # Record workflow completion
        self._record_execution_event(
            event_type="workflow_completed",
            details={
                "final_phase": self.current_workflow.phase.value,
                "work_directory": self.current_workflow.work_directory,
                "workflow_id": self._get_current_workflow_id()
            }
        )
        
        # Clear workflow state
        self.current_workflow = None
        self.user_approval_status.clear()
        
        # Save session state after completion
        self._save_session_state()
        
        return True
    
    def get_workflow_state(self) -> Optional[WorkflowState]:
        """
        Get the current workflow state.
        
        Returns:
            Current WorkflowState or None if no active workflow
        """
        return self.current_workflow   
 
    # Private helper methods
    
    async def _execute_requirements_phase(self, user_request: str, workflow_id: str) -> Dict[str, Any]:
        """Execute the requirements generation phase."""
        self.logger.info("Executing requirements phase")
        self.current_workflow.phase = WorkflowPhase.PLANNING
        
        try:
            result = await self.agent_manager.coordinate_agents(
                "requirements_generation",
                {
                    "user_request": user_request,
                    "workspace_path": str(self.session_manager.workspace_path)
                }
            )
            
            if result.get("success", False):
                # Update workflow state
                self.current_workflow.work_directory = result.get("work_directory", "")
                
                # Store phase result
                self.phase_results["requirements"] = result
                
                # Record phase completion
                self._record_execution_event(
                    event_type="phase_completed",
                    details={
                        "phase": "requirements",
                        "workflow_id": workflow_id,
                        "work_directory": self.current_workflow.work_directory
                    }
                )
                
                # Save session state after phase completion
                self._save_session_state()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Requirements phase failed: {e}")
            
            # Attempt error recovery
            recovery_context = {
                "user_request": user_request,
                "workspace_path": str(self.session_manager.workspace_path),
                "memory_context": {}  # Memory context would be provided by MainController
            }
            
            if self.handle_error_recovery(e, "requirements", recovery_context):
                self.logger.info("Requirements phase recovered from error, retrying...")
                # Retry the phase after successful recovery
                try:
                    result = await self.agent_manager.coordinate_agents(
                        "requirements_generation", recovery_context
                    )
                    if result.get("success", False):
                        self.phase_results["requirements"] = result
                        return result
                except Exception as retry_error:
                    self.logger.error(f"Requirements phase retry failed: {retry_error}")
            
            return {"success": False, "error": str(e)}
    
    async def _execute_design_phase(self, requirements_result: Dict[str, Any], workflow_id: str) -> Dict[str, Any]:
        """Execute the design generation phase."""
        self.logger.info("Executing design phase")
        self.current_workflow.phase = WorkflowPhase.DESIGN
        
        try:
            result = await self.agent_manager.coordinate_agents(
                "design_generation",
                {
                    "requirements_path": requirements_result.get("requirements_path"),
                    "work_directory": requirements_result.get("work_directory"),
                    "memory_context": {}  # Memory context would be provided by MainController
                }
            )
            
            if result.get("success", False):
                # Store phase result
                self.phase_results["design"] = result
                
                # Record phase completion
                self._record_execution_event(
                    event_type="phase_completed",
                    details={
                        "phase": "design",
                        "workflow_id": workflow_id,
                        "design_path": result.get("design_path")
                    }
                )
                
                # Save session state after phase completion
                self._save_session_state()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Design phase failed: {e}")
            
            # Attempt error recovery
            recovery_context = {
                "requirements_path": requirements_result.get("requirements_path"),
                "work_directory": requirements_result.get("work_directory"),
                "memory_context": {}  # Memory context would be provided by MainController
            }
            
            if self.handle_error_recovery(e, "design", recovery_context):
                self.logger.info("Design phase recovered from error, retrying...")
                # Retry the phase after successful recovery
                try:
                    result = await self.agent_manager.coordinate_agents(
                        "design_generation", recovery_context
                    )
                    if result.get("success", False):
                        self.phase_results["design"] = result
                        return result
                except Exception as retry_error:
                    self.logger.error(f"Design phase retry failed: {retry_error}")
            
            return {"success": False, "error": str(e)}
    
    async def _execute_tasks_phase(self, design_result: Dict[str, Any], 
                                  requirements_result: Dict[str, Any], workflow_id: str) -> Dict[str, Any]:
        """Execute the task generation phase."""
        self.logger.info("Executing tasks phase")
        self.current_workflow.phase = WorkflowPhase.TASK_GENERATION
        
        try:
            result = await self.agent_manager.coordinate_agents(
                "task_generation",
                {
                    "task_type": "generate_task_list",
                    "design_path": design_result.get("design_path"),
                    "requirements_path": requirements_result.get("requirements_path"),
                    "work_dir": requirements_result.get("work_directory")
                }
            )
            
            if result.get("success", False):
                # Store phase result
                self.phase_results["tasks"] = result
                
                # Record phase completion
                self._record_execution_event(
                    event_type="phase_completed",
                    details={
                        "phase": "tasks",
                        "workflow_id": workflow_id,
                        "tasks_file": result.get("tasks_file")
                    }
                )
                
                # Save session state after phase completion
                self._save_session_state()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Tasks phase failed: {e}")
            
            # Attempt error recovery
            recovery_context = {
                "design_path": design_result.get("design_path"),
                "requirements_path": requirements_result.get("requirements_path"),
                "work_directory": design_result.get("work_directory"),
                "memory_context": {}  # Memory context would be provided by MainController
            }
            
            if self.handle_error_recovery(e, "tasks", recovery_context):
                self.logger.info("Tasks phase recovered from error, retrying...")
                # Retry the phase after successful recovery
                try:
                    result = await self.agent_manager.coordinate_agents(
                        "task_generation", recovery_context
                    )
                    if result.get("success", False):
                        self.phase_results["tasks"] = result
                        return result
                except Exception as retry_error:
                    self.logger.error(f"Tasks phase retry failed: {retry_error}")
            
            return {"success": False, "error": str(e)}
    
    async def _execute_implementation_phase(self, tasks_result: Dict[str, Any], workflow_id: str) -> Dict[str, Any]:
        """Execute the implementation phase by actually running all tasks."""
        self.logger.info("Executing implementation phase")
        self.current_workflow.phase = WorkflowPhase.IMPLEMENTATION
        
        try:
            # Try both "tasks_file" and "tasks_path" for compatibility
            tasks_file = tasks_result.get("tasks_file") or tasks_result.get("tasks_path")
            work_directory = self.current_workflow.work_directory
            
            if not tasks_file or not Path(tasks_file).exists():
                return {
                    "success": False,
                    "error": f"Tasks file not found: {tasks_file}"
                }
            
            # Parse tasks from tasks.md file
            tasks = self._parse_tasks_from_file(tasks_file)
            
            if not tasks:
                return {
                    "success": False,
                    "error": "No tasks found in tasks.md file"
                }
            
            self.logger.info(f"Found {len(tasks)} tasks to execute")
            
            # Execute all tasks using the ImplementAgent
            execution_result = await self.agent_manager.coordinate_agents(
                "execute_multiple_tasks",
                {
                    "task_type": "execute_multiple_tasks",
                    "tasks": tasks,
                    "work_dir": work_directory,
                    "stop_on_failure": False  # Continue even if some tasks fail
                }
            )
            
            # Update tasks.md file with completion status
            if execution_result.get("success"):
                self._update_tasks_file_with_completion(tasks_file, execution_result.get("task_results", []))
            
            # Record phase completion
            self._record_execution_event(
                event_type="phase_completed",
                details={
                    "phase": "implementation",
                    "workflow_id": workflow_id,
                    "status": "tasks_executed",
                    "total_tasks": len(tasks),
                    "completed_tasks": execution_result.get("completed_count", 0),
                    "success_rate": f"{execution_result.get('completed_count', 0)}/{len(tasks)}"
                }
            )
            
            result = {
                "success": execution_result.get("success", False),
                "message": f"Implementation phase completed. {execution_result.get('completed_count', 0)}/{len(tasks)} tasks executed successfully.",
                "tasks_file": tasks_file,
                "work_directory": work_directory,
                "total_tasks": len(tasks),
                "completed_tasks": execution_result.get("completed_count", 0),
                "task_results": execution_result.get("task_results", []),
                "execution_completed": True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Implementation phase failed: {e}")
            
            # Attempt error recovery
            recovery_context = {
                "tasks_file": tasks_file,
                "work_directory": self.current_workflow.work_directory,
                "memory_context": {}  # Memory context would be provided by MainController
            }
            
            if self.handle_error_recovery(e, "implementation", recovery_context):
                self.logger.info("Implementation phase recovered from error, retrying...")
                # Retry the phase after successful recovery
                try:
                    # Re-parse tasks and retry execution
                    tasks = self._parse_tasks_from_file(tasks_file)
                    uncompleted_tasks = [task for task in tasks if not task.completed]
                    
                    if uncompleted_tasks:
                        task_results = await self.agent_manager.coordinate_agents(
                            "execute_multiple_tasks",
                            {
                                "tasks": [task.to_dict() for task in uncompleted_tasks],
                                "work_directory": self.current_workflow.work_directory,
                                "memory_context": {}  # Memory context would be provided by MainController
                            }
                        )
                        
                        if task_results.get("success", False):
                            self._update_tasks_file_with_completion(tasks_file, task_results.get("task_results", []))
                            result = {
                                "success": True,
                                "execution_completed": True,
                                "tasks_completed": len(task_results.get("task_results", [])),
                                "work_directory": self.current_workflow.work_directory
                            }
                            self.phase_results["implementation"] = result
                            return result
                except Exception as retry_error:
                    self.logger.error(f"Implementation phase retry failed: {retry_error}")
            
            return {"success": False, "error": str(e)}
    
    def _is_phase_approved(self, phase: str) -> bool:
        """Check if a phase has been approved by the user."""
        return self.user_approval_status.get(phase) == UserApprovalStatus.APPROVED
    
    def should_auto_approve(self, phase: str, auto_approve: bool) -> bool:
        """
        Determine if phase should be automatically approved based on --auto-approve flag.
        
        Args:
            phase: Phase name ("requirements", "design", "tasks")
            auto_approve: Whether auto-approve mode is enabled
            
        Returns:
            True if phase should be automatically approved, False otherwise
        """
        # First check if phase is already manually approved
        if self._is_phase_approved(phase):
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
    
    def _get_critical_checkpoints(self) -> List[str]:
        """
        Get list of critical checkpoints that require explicit approval.
        
        Returns:
            List of phase names that are considered critical checkpoints
        """
        # For now, no phases are considered critical checkpoints
        # This can be configured via environment variables or config in the future
        critical_checkpoints_env = os.getenv('AUTO_APPROVE_CRITICAL_CHECKPOINTS', '')
        if critical_checkpoints_env:
            return [phase.strip() for phase in critical_checkpoints_env.split(',')]
        return []
    
    def log_auto_approval(self, phase: str, decision: bool, reason: str) -> None:
        """
        Log automatic approval decisions for audit trail.
        
        Args:
            phase: Phase name
            decision: Whether the phase was approved
            reason: Reason for the decision
        """
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
        """
        Attempt automatic error recovery. Returns True if recovery successful.
        
        Args:
            error: The exception that occurred
            phase: Phase where the error occurred
            context: Context information for recovery
            
        Returns:
            True if recovery was successful, False otherwise
        """
        max_attempts = self._get_error_recovery_max_attempts()
        current_attempts = self.error_recovery_attempts.get(phase, 0)
        
        if current_attempts >= max_attempts:
            self.logger.error(f"Maximum error recovery attempts ({max_attempts}) exceeded for {phase}")
            return False
        
        self.error_recovery_attempts[phase] = current_attempts + 1
        
        # Attempt recovery strategies
        recovery_strategies = [
            self._retry_with_modified_parameters,
            self._skip_non_critical_steps,
            self._use_fallback_implementation
        ]
        
        for strategy in recovery_strategies:
            try:
                if strategy(error, phase, context):
                    self.workflow_summary['errors_recovered'].append({
                        'phase': phase,
                        'error': str(error),
                        'strategy': getattr(strategy, '__name__', str(strategy)),
                        'attempt': current_attempts + 1,
                        'timestamp': datetime.now().isoformat()
                    })
                    self.logger.info(f"Error recovery successful using {getattr(strategy, '__name__', str(strategy))} for {phase}")
                    return True
            except Exception as recovery_error:
                self.logger.warning(f"Recovery strategy {getattr(strategy, '__name__', str(strategy))} failed: {recovery_error}")
        
        return False
    
    def _get_error_recovery_max_attempts(self) -> int:
        """Get maximum error recovery attempts from configuration."""
        return int(os.getenv('AUTO_APPROVE_ERROR_RECOVERY_ATTEMPTS', '3'))
    
    def _retry_with_modified_parameters(self, error: Exception, phase: str, context: Dict[str, Any]) -> bool:
        """
        Recovery strategy: Retry with modified parameters.
        
        This strategy attempts to recover from errors by modifying parameters
        that might have caused the failure and retrying the operation.
        
        Args:
            error: The exception that occurred
            phase: Phase where the error occurred
            context: Context information for recovery
            
        Returns:
            True if recovery was successful, False otherwise
        """
        self.logger.info(f"Attempting retry with modified parameters for {phase}")
        
        try:
            # Modify parameters based on error type and phase
            modified_context = self._modify_parameters_for_retry(error, phase, context)
            
            if not modified_context:
                self.logger.warning(f"Could not determine parameter modifications for {phase}")
                return False
            
            # For now, return False as the actual retry logic would need to be implemented
            # This is a placeholder for the recovery strategy
            return False
                
        except Exception as retry_error:
            self.logger.error(f"Retry with modified parameters failed for {phase}: {retry_error}")
            return False
    
    def _skip_non_critical_steps(self, error: Exception, phase: str, context: Dict[str, Any]) -> bool:
        """
        Recovery strategy: Skip non-critical steps.
        
        This strategy attempts to recover by identifying and skipping
        non-essential steps that might be causing the failure.
        
        Args:
            error: The exception that occurred
            phase: Phase where the error occurred
            context: Context information for recovery
            
        Returns:
            True if recovery was successful, False otherwise
        """
        self.logger.info(f"Attempting to skip non-critical steps for {phase}")
        
        try:
            # For now, return False as the actual skip logic would need to be implemented
            # This is a placeholder for the recovery strategy
            return False
                
        except Exception as skip_error:
            self.logger.error(f"Skip non-critical steps failed for {phase}: {skip_error}")
            return False
    
    def _use_fallback_implementation(self, error: Exception, phase: str, context: Dict[str, Any]) -> bool:
        """
        Recovery strategy: Use fallback implementation.
        
        This strategy uses a simpler, more reliable fallback approach
        when the primary implementation fails.
        
        Args:
            error: The exception that occurred
            phase: Phase where the error occurred
            context: Context information for recovery
            
        Returns:
            True if recovery was successful, False otherwise
        """
        self.logger.info(f"Attempting fallback implementation for {phase}")
        
        try:
            # For now, return False as the actual fallback logic would need to be implemented
            # This is a placeholder for the recovery strategy
            return False
                
        except Exception as fallback_error:
            self.logger.error(f"Fallback implementation failed for {phase}: {fallback_error}")
            return False
    
    def _modify_parameters_for_retry(self, error: Exception, phase: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Modify parameters for retry based on error type and phase.
        
        Args:
            error: The exception that occurred
            phase: Phase where the error occurred
            context: Original context information
            
        Returns:
            Modified context dictionary or None if no modifications possible
        """
        # This is a placeholder for parameter modification logic
        # The actual implementation would analyze the error and modify context accordingly
        return None
    
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
    
    def _parse_tasks_from_file(self, tasks_file: str) -> List:
        """Parse tasks from tasks.md file and return TaskDefinition objects."""
        try:
            from .models import TaskDefinition
            
            with open(tasks_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tasks = []
            lines = content.split('\n')
            current_task_data = None
            
            for i, line in enumerate(lines):
                # Check for task line (starts with - [ ] or - [x])
                if line.strip().startswith('- ['):
                    # Save previous task if exists
                    if current_task_data:
                        task_def = TaskDefinition(
                            id=current_task_data["id"],
                            title=current_task_data["title"],
                            description=current_task_data["description"],
                            steps=current_task_data["steps"],
                            requirements_ref=current_task_data["requirements_ref"],
                            completed=current_task_data["completed"]
                        )
                        tasks.append(task_def)
                    
                    # Extract task title (everything after the checkbox)
                    task_title = line.strip()[6:].strip()  # Remove "- [ ] " or "- [x] "
                    
                    current_task_data = {
                        "id": f"task_{len(tasks) + 1}",
                        "title": task_title,
                        "description": task_title,
                        "steps": [],
                        "requirements_ref": [],
                        "completed": line.strip().startswith('- [x]')
                    }
                
                elif current_task_data and line.strip().startswith('- Step'):
                    # Add step to current task
                    step = line.strip()[2:].strip()  # Remove "- "
                    current_task_data["steps"].append(step)
                
                elif current_task_data and line.strip().startswith('- Requirements:'):
                    # Extract requirements
                    req_text = line.strip()[14:].strip()  # Remove "- Requirements: "
                    current_task_data["requirements_ref"] = [r.strip() for r in req_text.split(',')]
            
            # Add the last task
            if current_task_data:
                task_def = TaskDefinition(
                    id=current_task_data["id"],
                    title=current_task_data["title"],
                    description=current_task_data["description"],
                    steps=current_task_data["steps"],
                    requirements_ref=current_task_data["requirements_ref"],
                    completed=current_task_data["completed"]
                )
                tasks.append(task_def)
            
            # Filter out already completed tasks
            uncompleted_tasks = [task for task in tasks if not task.completed]
            
            self.logger.info(f"Parsed {len(tasks)} total tasks, {len(uncompleted_tasks)} uncompleted")
            return uncompleted_tasks
            
        except Exception as e:
            self.logger.error(f"Error parsing tasks file: {e}")
            return []
    
    def _update_tasks_file_with_completion(self, tasks_file: str, task_results: List[Dict[str, Any]]) -> None:
        """Update tasks.md file to mark completed tasks."""
        try:
            with open(tasks_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            task_index = 0
            
            for i, line in enumerate(lines):
                if line.strip().startswith('- [ ]'):
                    # Check if this task was completed
                    if task_index < len(task_results) and task_results[task_index].get("success", False):
                        # Mark as completed
                        lines[i] = line.replace('- [ ]', '- [x]')
                    task_index += 1
            
            # Write back to file
            with open(tasks_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            self.logger.info(f"Updated tasks file with completion status")
            
        except Exception as e:
            self.logger.error(f"Error updating tasks file: {e}")
    
    def _save_session_state(self):
        """Save current session state through SessionManager."""
        # Sync WorkflowManager state to SessionManager before saving
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
    
    def _load_session_state_from_manager(self):
        """Load session state from SessionManager into WorkflowManager."""
        self.current_workflow = self.session_manager.current_workflow
        
        # Convert approval status from strings to enum values
        self.user_approval_status = {
            k: UserApprovalStatus(v) for k, v in self.session_manager.user_approval_status.items()
        }
        
        self.phase_results = self.session_manager.phase_results
        self.execution_log = self.session_manager.execution_log
        self.approval_log = self.session_manager.approval_log
        self.error_recovery_attempts = self.session_manager.error_recovery_attempts
        self.workflow_summary = self.session_manager.workflow_summary