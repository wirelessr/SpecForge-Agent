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

from .models import WorkflowState, WorkflowPhase, UserApprovalStatus
from .session_manager import SessionManager
from .context_manager import ContextManager
from .dependency_container import DependencyContainer


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
    
    def __init__(self, agent_manager, session_manager: SessionManager, 
                 container: DependencyContainer):
        """
        Initialize the Workflow Manager.
        
        Args:
            agent_manager: AgentManager instance for coordinating agents
            session_manager: SessionManager instance for session persistence
            container: DependencyContainer for accessing framework managers
        """
        self.agent_manager = agent_manager
        self.session_manager = session_manager
        self.container = container
        self.context_manager: Optional[ContextManager] = None
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
        
        # Task completion configuration and tracking
        self.task_completion_config = self._load_task_completion_config()
        self.task_update_stats = {
            'individual_updates_attempted': 0,
            'individual_updates_successful': 0,
            'fallback_updates_used': 0,
            'file_access_errors': 0,
            'task_identification_errors': 0,
            'partial_update_recoveries': 0
        }
        
        # Load existing session state from SessionManager
        self._load_session_state_from_manager()
        
        self.logger.info("WorkflowManager initialized with DependencyContainer")
    
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
                
                # Initialize ContextManager with the work directory
                await self._initialize_context_manager()
                
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
            
            # Provide clear error message for manual revision
            self.logger.error(f"Requirements phase failed: {e}")
            self.logger.info("To fix this issue, use: autogen-framework --revise \"requirements:Please provide more specific requirements\"")
            
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
            
            # Provide clear error message for manual revision
            self.logger.error(f"Design phase failed: {e}")
            self.logger.info("To fix this issue, use: autogen-framework --revise \"design:Please simplify the design and focus on core features\"")
            
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
            
            # Provide clear error message for manual revision
            self.logger.error(f"Tasks phase failed: {e}")
            self.logger.info("To fix this issue, use: autogen-framework --revise \"tasks:Please create simpler tasks with clearer descriptions\"")
            
            return {"success": False, "error": str(e)}
    
    async def _execute_implementation_phase(self, tasks_result: Dict[str, Any], workflow_id: str) -> Dict[str, Any]:
        """Execute the implementation phase by running tasks individually with real-time completion marking."""
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
            
            self.logger.info(f"Found {len(tasks)} tasks to execute individually")
            
            # Execute tasks individually instead of all at once
            aggregated_results = []
            completed_count = 0
            
            for i, task in enumerate(tasks):
                self.logger.info(f"Executing task {i+1}/{len(tasks)}: {task.title}")
                
                # Execute single task via agent_manager
                task_execution_result = await self.agent_manager.coordinate_agents(
                    "task_execution",
                    {
                        "task_type": "execute_task",
                        "task": task,
                        "work_dir": work_directory
                    }
                )
                
                # Determine if task was successful
                task_success = task_execution_result.get("success", False)
                
                # Only attempt individual updates if real-time updates are enabled
                if self.task_completion_config.get('real_time_updates_enabled', True):
                    # Record individual update attempt (before calling the method)
                    self._record_task_update_stats('individual_updates_attempted')
                    
                    # Immediately update tasks.md for this specific task using numerical ID
                    try:
                        update_success = self._update_single_task_completion(tasks_file, task.id, task_success)
                        
                        # Record success/failure statistics
                        if update_success:
                            self._record_task_update_stats('individual_updates_successful')
                        else:
                            self.logger.warning(f"Failed to update completion status for task {task.id}, will use fallback batch update")
                            
                    except PermissionError as e:
                        self.logger.warning(f"File access error updating task {task.id}: {e}")
                        self._record_task_update_stats('file_access_errors')
                        update_success = False
                    except Exception as e:
                        self.logger.warning(f"Error updating task {task.id}: {e}")
                        self._record_task_update_stats('task_identification_errors')
                        update_success = False
                else:
                    # Real-time updates disabled, skip individual updates
                    update_success = False
                
                # Collect results for backward compatibility
                task_result = {
                    "task_id": task.id,
                    "task_title": task.title,
                    "success": task_success,
                    "execution_result": task_execution_result,
                    "completion_updated": update_success
                }
                aggregated_results.append(task_result)
                
                if task_success:
                    completed_count += 1
                else:
                    self.logger.warning(f"Task {task.id} failed: {task_execution_result.get('error', 'Unknown error')}")
                
                # Log individual task completion
                self._record_execution_event(
                    event_type="task_completed",
                    details={
                        "task_id": task.id,
                        "task_title": task.title,
                        "success": task_success,
                        "workflow_id": workflow_id,
                        "task_index": i + 1,
                        "total_tasks": len(tasks)
                    }
                )
            
            # Determine overall success - implementation phase succeeds if at least one task completes
            # This matches the behavior of the original execute_multiple_tasks approach
            overall_success = completed_count > 0
            
            # Comprehensive error recovery and fallback mechanisms
            failed_updates = [r for r in aggregated_results if not r.get("completion_updated", False)]
            fallback_used = False
            recovery_successful = False
            
            if failed_updates and self.task_completion_config.get('fallback_to_batch_enabled', True):
                self.logger.info(f"Performing fallback batch update for {len(failed_updates)} tasks with failed individual updates")
                self._record_task_update_stats('fallback_updates_used')
                
                try:
                    self._update_tasks_file_with_completion(tasks_file, aggregated_results)
                    fallback_used = True
                    self._log_task_completion_debug("Fallback batch update completed successfully")
                except Exception as fallback_error:
                    self.logger.error(f"Fallback batch update failed: {fallback_error}")
                    self._log_task_completion_debug("Fallback batch update failed", {"error": str(fallback_error)})
            
            # Ensure no tasks are lost due to update failures
            if self.task_completion_config.get('recovery_mechanism_enabled', True):
                recovery_successful = self._ensure_no_tasks_lost(tasks_file, tasks, aggregated_results)
                if recovery_successful:
                    self._log_task_completion_debug("Task loss prevention successful")
                else:
                    self.logger.warning("Task loss prevention check failed - some tasks may not be properly marked")
            
            # Log comprehensive task update statistics
            self._log_task_update_statistics(len(tasks), len(failed_updates), fallback_used, recovery_successful)
            
            # Record phase completion
            self._record_execution_event(
                event_type="phase_completed",
                details={
                    "phase": "implementation",
                    "workflow_id": workflow_id,
                    "status": "tasks_executed_individually",
                    "total_tasks": len(tasks),
                    "completed_tasks": completed_count,
                    "success_rate": f"{completed_count}/{len(tasks)}",
                    "individual_updates": len(aggregated_results) - len(failed_updates),
                    "fallback_updates": len(failed_updates)
                }
            )
            
            # Maintain aggregated results collection for backward compatibility
            result = {
                "success": overall_success,
                "message": f"Implementation phase completed. {completed_count}/{len(tasks)} tasks executed successfully.",
                "tasks_file": tasks_file,
                "work_directory": work_directory,
                "total_tasks": len(tasks),
                "completed_tasks": completed_count,
                "task_results": aggregated_results,
                "execution_completed": True,
                "individual_execution": True,  # Flag to indicate new execution method
                "error_recovery": {
                    "individual_updates_attempted": self.task_update_stats['individual_updates_attempted'],
                    "individual_updates_successful": self.task_update_stats['individual_updates_successful'],
                    "fallback_updates_used": self.task_update_stats['fallback_updates_used'],
                    "file_access_errors": self.task_update_stats['file_access_errors'],
                    "task_identification_errors": self.task_update_stats['task_identification_errors'],
                    "partial_update_recoveries": self.task_update_stats['partial_update_recoveries'],
                    "fallback_used": fallback_used,
                    "recovery_successful": recovery_successful,
                    "failed_individual_updates": len(failed_updates)
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Implementation phase failed: {e}")
            
            # Preserve existing error handling and logging behavior
            self.logger.error(f"Implementation phase failed: {e}")
            self.logger.info("Implementation errors are handled by ImplementAgent. If all tasks fail, consider revising tasks: autogen-framework --revise \"tasks:Please create simpler, more specific tasks\"")
            
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
        Simplified error recovery - provides user guidance instead of automatic recovery.
        
        Args:
            error: The exception that occurred
            phase: Phase where the error occurred
            context: Context information for recovery
            
        Returns:
            Always returns False - no automatic recovery, user must use --revise
        """
        # Log the error for debugging
        self.logger.error(f"Phase {phase} failed: {error}")
        
        # Provide user guidance based on phase
        guidance = self._get_phase_error_guidance(phase)
        self.logger.info(guidance)
        
        # Always return False - no automatic recovery
        return False
    
    def _get_phase_error_guidance(self, phase: str) -> str:
        """
        Get user guidance message for phase failures.
        
        Args:
            phase: The phase that failed
            
        Returns:
            User-friendly guidance message with --revise examples
        """
        guidance_messages = {
            "requirements": "To fix this issue, use: autogen-framework --revise \"requirements:Please provide more specific and detailed requirements\"",
            "design": "To fix this issue, use: autogen-framework --revise \"design:Please simplify the design and focus on core features\"",
            "tasks": "To fix this issue, use: autogen-framework --revise \"tasks:Please create simpler tasks with clearer descriptions\"",
            "implementation": "Implementation errors are handled by ImplementAgent. If all tasks fail, consider revising tasks: autogen-framework --revise \"tasks:Please create simpler, more specific tasks\""
        }
        
        return guidance_messages.get(phase, f"To fix this issue, use: autogen-framework --revise \"{phase}:Please provide more specific guidance\"")
    
    # Complex error recovery methods removed - using simple error reporting instead
    # Manual revision via --revise is the preferred approach for phase-level failures
    
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
        """
        Parse tasks from tasks.md file and return TaskDefinition objects.
        
        Enhanced to extract numerical IDs from task titles and capture line positions
        for precise file updates. Supports both numbered and unnumbered tasks for
        backward compatibility.
        """
        try:
            from .models import TaskDefinition, TaskFileMapping
            import re
            
            with open(tasks_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tasks = []
            lines = content.split('\n')
            current_task_data = None
            
            # Store task file mappings for later use
            self.task_file_mappings = {}
            
            for i, line in enumerate(lines):
                # Check for task line (starts with - [ ], - [x], or - [-])
                if line.strip().startswith('- [') and (
                    '- [ ]' in line or '- [x]' in line or '- [-]' in line
                ):
                    # Save previous task if exists
                    if current_task_data:
                        task_def = TaskDefinition(
                            id=current_task_data["id"],
                            title=current_task_data["title"],
                            description=current_task_data["description"],
                            steps=current_task_data["steps"],
                            requirements_ref=current_task_data["requirements_ref"],
                            completed=current_task_data["completed"],
                            line_number=current_task_data["line_number"],
                            file_position=current_task_data["file_position"]
                        )
                        tasks.append(task_def)
                        
                        # Store task file mapping
                        self.task_file_mappings[current_task_data["id"]] = TaskFileMapping(
                            task_id=current_task_data["id"],
                            line_number=current_task_data["line_number"],
                            original_line=current_task_data["original_line"],
                            checkbox_position=current_task_data["checkbox_position"]
                        )
                    
                    # Extract task title (everything after the checkbox)
                    task_title = line.strip()[6:].strip()  # Remove "- [ ] " or "- [x] "
                    
                    # Try to extract numerical ID from task title using regex
                    # Pattern matches: "1. Task Title", "2. Task Title", etc.
                    numerical_id_match = re.match(r'^(\d+)\.\s*(.+)', task_title)
                    
                    if numerical_id_match:
                        # Use numerical ID from task title
                        numerical_id = numerical_id_match.group(1)
                        task_id = numerical_id
                        clean_title = numerical_id_match.group(2).strip()
                    else:
                        # Fallback to sequential ID for backward compatibility
                        task_id = f"task_{len(tasks) + 1}"
                        clean_title = task_title
                    
                    # Find checkbox position for precise updates
                    checkbox_pos = line.find('- [')
                    
                    current_task_data = {
                        "id": task_id,
                        "title": clean_title,
                        "description": clean_title,
                        "steps": [],
                        "requirements_ref": [],
                        "completed": line.strip().startswith('- [x]'),
                        "line_number": i,
                        "file_position": sum(len(l) + 1 for l in lines[:i]),  # Character position
                        "original_line": line,
                        "checkbox_position": checkbox_pos
                    }
                
                elif current_task_data and line.strip().startswith('- Step'):
                    # Add step to current task
                    step = line.strip()[2:].strip()  # Remove "- "
                    current_task_data["steps"].append(step)
                
                elif current_task_data and line.strip().startswith('- Requirements:'):
                    # Extract requirements (old format)
                    # Find the colon and extract everything after it
                    colon_pos = line.find(':')
                    if colon_pos != -1:
                        req_text = line[colon_pos + 1:].strip()
                        current_task_data["requirements_ref"] = [r.strip() for r in req_text.split(',') if r.strip()]
                
                elif current_task_data and line.strip().startswith('- _Requirements:'):
                    # Handle alternative requirements format (new format)
                    # Find the colon and extract everything after it
                    colon_pos = line.find(':')
                    if colon_pos != -1:
                        req_text = line[colon_pos + 1:].strip()
                        # Remove trailing underscore if present
                        if req_text.endswith('_'):
                            req_text = req_text[:-1].strip()
                        current_task_data["requirements_ref"] = [r.strip() for r in req_text.split(',') if r.strip()]
            
            # Add the last task
            if current_task_data:
                task_def = TaskDefinition(
                    id=current_task_data["id"],
                    title=current_task_data["title"],
                    description=current_task_data["description"],
                    steps=current_task_data["steps"],
                    requirements_ref=current_task_data["requirements_ref"],
                    completed=current_task_data["completed"],
                    line_number=current_task_data["line_number"],
                    file_position=current_task_data["file_position"]
                )
                tasks.append(task_def)
                
                # Store task file mapping
                self.task_file_mappings[current_task_data["id"]] = TaskFileMapping(
                    task_id=current_task_data["id"],
                    line_number=current_task_data["line_number"],
                    original_line=current_task_data["original_line"],
                    checkbox_position=current_task_data["checkbox_position"]
                )
            
            # Filter out already completed tasks
            uncompleted_tasks = [task for task in tasks if not task.completed]
            
            # Count numbered vs unnumbered tasks
            numbered_tasks = [t for t in tasks if re.match(r'^\d+$', t.id)]
            unnumbered_tasks = [t for t in tasks if not re.match(r'^\d+$', t.id)]
            self.logger.info(f"Parsed {len(tasks)} total tasks ({len(numbered_tasks)} numbered, {len(unnumbered_tasks)} unnumbered), {len(uncompleted_tasks)} uncompleted")
            return uncompleted_tasks
            
        except Exception as e:
            self.logger.error(f"Error parsing tasks file: {e}")
            return []
    
    def _update_single_task_completion(self, tasks_file: str, task_id: str, success: bool) -> bool:
        """
        Update tasks.md file to mark a specific task as completed with comprehensive error recovery.
        
        Uses numerical IDs and line number mappings for precise file updates.
        Implements atomic file update logic with error handling and retry mechanisms.
        Includes file locking mechanism to prevent corruption during concurrent access.
        Enhanced with comprehensive error recovery and fallback mechanisms.
        
        Args:
            tasks_file: Path to tasks.md file
            task_id: ID of the completed task (numerical or sequential)
            success: Whether the task completed successfully
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self.task_completion_config.get('real_time_updates_enabled', True):
            self._log_task_completion_debug(f"Real-time updates disabled, skipping individual update for task {task_id}")
            return False
        
        self._log_task_completion_debug(f"Starting individual update for task {task_id}", {
            "success": success,
            "tasks_file": tasks_file
        })
        
        import fcntl
        import time
        
        max_retries = self.task_completion_config.get('max_individual_update_retries', 3)
        retry_delay = self.task_completion_config.get('individual_update_retry_delay', 0.1)
        file_lock_timeout = self.task_completion_config.get('file_lock_timeout', 5)
        
        for attempt in range(max_retries):
            try:
                # Check if we have task file mappings
                if not hasattr(self, 'task_file_mappings') or task_id not in self.task_file_mappings:
                    self._log_task_completion_debug(f"No file mapping found for task {task_id}")
                    self._record_task_update_stats('task_identification_errors')
                    
                    # Try to recover by re-parsing the file
                    if self._attempt_task_mapping_recovery(tasks_file, task_id):
                        self._log_task_completion_debug(f"Successfully recovered mapping for task {task_id}")
                    else:
                        self.logger.warning(f"No file mapping found for task {task_id}, falling back to batch update")
                        return False
                
                mapping = self.task_file_mappings[task_id]
                
                # Use file locking to prevent concurrent access
                with open(tasks_file, 'r+', encoding='utf-8') as f:
                    try:
                        # Acquire exclusive lock on the file with timeout
                        start_time = time.time()
                        while time.time() - start_time < file_lock_timeout:
                            try:
                                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                                break
                            except (OSError, IOError):
                                time.sleep(0.1)
                        else:
                            raise OSError("File lock timeout exceeded")
                        
                        # Read entire file into memory for atomic update
                        f.seek(0)
                        lines = f.readlines()
                        
                        # Verify line number is still valid
                        if mapping.line_number >= len(lines):
                            self.logger.error(f"Task {task_id} line number {mapping.line_number} is out of range (file has {len(lines)} lines)")
                            self._record_task_update_stats('task_identification_errors')
                            
                            # Try to recover by finding the task in the file
                            if self._attempt_line_number_recovery(lines, task_id, mapping):
                                self._log_task_completion_debug(f"Successfully recovered line number for task {task_id}")
                            else:
                                return False
                        
                        current_line = lines[mapping.line_number].rstrip('\n')
                        
                        # Verify task identification using numerical ID to find correct task line
                        if not self._verify_task_line(current_line, task_id):
                            self.logger.error(f"Task {task_id} line verification failed. Expected task not found at line {mapping.line_number}")
                            self._record_task_update_stats('task_identification_errors')
                            
                            # Try to recover by searching for the task in nearby lines
                            if self._attempt_task_line_recovery(lines, task_id, mapping):
                                current_line = lines[mapping.line_number].rstrip('\n')
                                self._log_task_completion_debug(f"Successfully recovered task line for task {task_id}")
                            else:
                                return False
                        
                        # Update the specific task line
                        if success:
                            # Mark as completed - handle different checkbox states
                            if '- [ ]' in current_line:
                                updated_line = current_line.replace('- [ ]', '- [x]', 1)
                            elif '- [-]' in current_line:
                                updated_line = current_line.replace('- [-]', '- [x]', 1)
                            else:
                                updated_line = current_line  # Already completed or unknown format
                        else:
                            # Ensure it remains uncompleted (in case of retry)
                            if '- [x]' in current_line:
                                updated_line = current_line.replace('- [x]', '- [ ]', 1)
                            else:
                                updated_line = current_line  # Already uncompleted or unknown format
                        
                        # Only update if there was actually a change
                        if updated_line != current_line:
                            lines[mapping.line_number] = updated_line + '\n'
                            
                            # Write entire file back atomically
                            f.seek(0)
                            f.writelines(lines)
                            f.truncate()
                            f.flush()
                            os.fsync(f.fileno())  # Force write to disk
                            
                            self.logger.info(f"Updated task {task_id} completion status to {'completed' if success else 'uncompleted'}")
                            self._log_task_completion_debug(f"Successfully updated task {task_id}", {
                                "old_line": current_line,
                                "new_line": updated_line
                            })
                            
                            # Update the mapping to reflect the new line content
                            mapping.original_line = updated_line
                            
                            return True
                        else:
                            self.logger.debug(f"Task {task_id} completion status unchanged")
                            return True
                    
                    except (OSError, IOError) as lock_error:
                        self._record_task_update_stats('file_access_errors')
                        if attempt < max_retries - 1:
                            self.logger.warning(f"File lock failed for task {task_id} (attempt {attempt + 1}/{max_retries}): {lock_error}")
                            self._log_task_completion_debug(f"File lock error, retrying", {"attempt": attempt + 1, "error": str(lock_error)})
                            time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                            continue
                        else:
                            self.logger.error(f"Failed to acquire file lock after {max_retries} attempts: {lock_error}")
                            return False
                    
                    finally:
                        # Lock is automatically released when file is closed
                        pass
                        
            except FileNotFoundError:
                self.logger.error(f"Tasks file not found: {tasks_file}")
                self._record_task_update_stats('file_access_errors')
                return False
            except PermissionError as perm_error:
                self._record_task_update_stats('file_access_errors')
                if attempt < max_retries - 1:
                    self.logger.warning(f"Permission denied accessing tasks file (attempt {attempt + 1}/{max_retries}): {perm_error}")
                    self._log_task_completion_debug(f"Permission error, retrying", {"attempt": attempt + 1, "error": str(perm_error)})
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                else:
                    self.logger.error(f"Permission denied accessing tasks file after {max_retries} attempts: {perm_error}")
                    return False
            except Exception as e:
                self._record_task_update_stats('file_access_errors')
                if attempt < max_retries - 1:
                    self.logger.warning(f"Error updating single task completion for {task_id} (attempt {attempt + 1}/{max_retries}): {e}")
                    self._log_task_completion_debug(f"Unexpected error, retrying", {"attempt": attempt + 1, "error": str(e)})
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                else:
                    self.logger.error(f"Error updating single task completion for {task_id} after {max_retries} attempts: {e}")
                    return False
        
        return False
    
    def _attempt_task_mapping_recovery(self, tasks_file: str, task_id: str) -> bool:
        """
        Attempt to recover task file mapping by re-parsing the tasks file.
        
        Args:
            tasks_file: Path to tasks.md file
            task_id: ID of the task to recover mapping for
            
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            self._log_task_completion_debug(f"Attempting task mapping recovery for task {task_id}")
            
            # Clear existing mappings and re-parse
            if hasattr(self, 'task_file_mappings'):
                self.task_file_mappings.clear()
            
            # Re-parse tasks from file to rebuild mappings
            self._parse_tasks_from_file(tasks_file)
            
            # Check if the task mapping was recovered
            if hasattr(self, 'task_file_mappings') and task_id in self.task_file_mappings:
                self._log_task_completion_debug(f"Successfully recovered mapping for task {task_id}")
                return True
            else:
                self._log_task_completion_debug(f"Failed to recover mapping for task {task_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during task mapping recovery: {e}")
            return False
    
    def _attempt_line_number_recovery(self, lines: List[str], task_id: str, mapping) -> bool:
        """
        Attempt to recover correct line number for a task by searching the file.
        
        Args:
            lines: List of file lines
            task_id: ID of the task to recover line number for
            mapping: TaskFileMapping object to update
            
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            self._log_task_completion_debug(f"Attempting line number recovery for task {task_id}")
            
            import re
            
            # Search for the task in all lines
            for i, line in enumerate(lines):
                if line.strip().startswith('- [') and ('- [ ]' in line or '- [x]' in line or '- [-]' in line):
                    task_title = line.strip()[6:].strip()  # Remove "- [ ] " or "- [x] "
                    
                    # Check for numerical ID pattern
                    numerical_id_match = re.match(r'^(\d+)\.\s*(.+)', task_title)
                    
                    if numerical_id_match:
                        line_task_id = numerical_id_match.group(1)
                        if line_task_id == task_id:
                            # Found the task, update the mapping
                            mapping.line_number = i
                            mapping.original_line = line.rstrip('\n')
                            self._log_task_completion_debug(f"Recovered line number {i} for task {task_id}")
                            return True
            
            self._log_task_completion_debug(f"Failed to recover line number for task {task_id}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error during line number recovery: {e}")
            return False
    
    def _attempt_task_line_recovery(self, lines: List[str], task_id: str, mapping) -> bool:
        """
        Attempt to recover correct task line by searching nearby lines.
        
        Args:
            lines: List of file lines
            task_id: ID of the task to recover line for
            mapping: TaskFileMapping object to update
            
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            self._log_task_completion_debug(f"Attempting task line recovery for task {task_id}")
            
            # Search in a window around the expected line number
            search_window = 5
            start_line = max(0, mapping.line_number - search_window)
            end_line = min(len(lines), mapping.line_number + search_window + 1)
            
            for i in range(start_line, end_line):
                if self._verify_task_line(lines[i].rstrip('\n'), task_id):
                    # Found the task, update the mapping
                    mapping.line_number = i
                    mapping.original_line = lines[i].rstrip('\n')
                    self._log_task_completion_debug(f"Recovered task line at line {i} for task {task_id}")
                    return True
            
            self._log_task_completion_debug(f"Failed to recover task line for task {task_id}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error during task line recovery: {e}")
            return False
    
    def _verify_task_line(self, line: str, expected_task_id: str) -> bool:
        """
        Verify that the line contains the expected task by checking the task ID.
        
        Args:
            line: The line content to verify
            expected_task_id: The expected task ID (numerical or sequential)
            
        Returns:
            True if the line contains the expected task, False otherwise
        """
        import re
        
        # Extract task title from line (everything after the checkbox)
        if not (line.strip().startswith('- [') and ('- [ ]' in line or '- [x]' in line or '- [-]' in line)):
            return False
        
        task_title = line.strip()[6:].strip()  # Remove "- [ ] " or "- [x] " or "- [-] "
        
        # Check for numerical ID pattern
        numerical_id_match = re.match(r'^(\d+)\.\s*(.+)', task_title)
        
        if numerical_id_match:
            # Line has numerical ID - check if it matches expected
            line_task_id = numerical_id_match.group(1)
            return line_task_id == expected_task_id
        else:
            # Line doesn't have numerical ID - check if expected is sequential format
            return expected_task_id.startswith('task_')  # Sequential IDs are acceptable for unnumbered tasks

    def _update_tasks_file_with_completion(self, tasks_file: str, task_results: List[Dict[str, Any]]) -> None:
        """
        Update tasks.md file to mark completed tasks (batch update method).
        
        This method is kept for backward compatibility and as a fallback
        when individual task updates fail.
        """
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
            
            self.logger.info(f"Updated tasks file with completion status (batch update)")
            
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
    
    async def _initialize_context_manager(self) -> None:
        """Initialize ContextManager with the current work directory."""
        if not self.current_workflow or not self.current_workflow.work_directory:
            self.logger.warning("Cannot initialize ContextManager: no work directory available")
            return
        
        try:
            # Get ContextManager from container - it will be created with proper dependencies
            self.context_manager = self.container.get_context_manager()
            
            # Initialize the context manager
            await self.context_manager.initialize()
            
            self.logger.info(f"ContextManager initialized for work directory: {self.current_workflow.work_directory}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ContextManager: {e}")
            self.context_manager = None
    
    def get_context_manager(self) -> Optional[ContextManager]:
        """Get the current ContextManager instance."""
        return self.context_manager
    
    def _load_task_completion_config(self) -> Dict[str, Any]:
        """
        Load task completion configuration from environment variables.
        
        Returns:
            Dictionary containing task completion configuration
        """
        return {
            'real_time_updates_enabled': os.getenv('TASK_REAL_TIME_UPDATES_ENABLED', 'true').lower() == 'true',
            'fallback_to_batch_enabled': os.getenv('TASK_FALLBACK_TO_BATCH_ENABLED', 'true').lower() == 'true',
            'max_individual_update_retries': int(os.getenv('TASK_MAX_INDIVIDUAL_UPDATE_RETRIES', '3')),
            'individual_update_retry_delay': float(os.getenv('TASK_INDIVIDUAL_UPDATE_RETRY_DELAY', '0.1')),
            'file_lock_timeout': int(os.getenv('TASK_FILE_LOCK_TIMEOUT', '5')),
            'detailed_logging_enabled': os.getenv('TASK_DETAILED_LOGGING_ENABLED', 'true').lower() == 'true',
            'recovery_mechanism_enabled': os.getenv('TASK_RECOVERY_MECHANISM_ENABLED', 'true').lower() == 'true'
        }
    
    def _log_task_completion_debug(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log detailed debugging information for task completion updates.
        
        Args:
            message: Debug message
            details: Optional additional details to log
        """
        if self.task_completion_config.get('detailed_logging_enabled', True):
            if details:
                self.logger.debug(f"[TaskCompletion] {message}: {details}")
            else:
                self.logger.debug(f"[TaskCompletion] {message}")
    
    def _record_task_update_stats(self, stat_type: str, increment: int = 1) -> None:
        """
        Record task update statistics for monitoring and debugging.
        
        Args:
            stat_type: Type of statistic to record
            increment: Amount to increment the statistic by
        """
        if stat_type in self.task_update_stats:
            self.task_update_stats[stat_type] += increment
            self._log_task_completion_debug(f"Updated {stat_type}", {"new_value": self.task_update_stats[stat_type]})
    
    def _ensure_no_tasks_lost(self, tasks_file: str, expected_tasks: List, task_results: List[Dict[str, Any]]) -> bool:
        """
        Recovery mechanism to ensure no tasks are lost due to update failures.
        
        Verifies that all expected tasks are accounted for and attempts recovery
        if any tasks are missing or incorrectly marked.
        
        Args:
            tasks_file: Path to tasks.md file
            expected_tasks: List of TaskDefinition objects that should be in the file
            task_results: List of task execution results
            
        Returns:
            True if all tasks are properly accounted for, False if recovery failed
        """
        if not self.task_completion_config.get('recovery_mechanism_enabled', True):
            return True
        
        try:
            self._log_task_completion_debug("Starting task loss prevention check", {
                "expected_tasks": len(expected_tasks),
                "task_results": len(task_results)
            })
            
            # Read current file content
            with open(tasks_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            found_tasks = {}
            
            # Parse current file to find all tasks
            for i, line in enumerate(lines):
                if line.strip().startswith('- [') and ('- [ ]' in line or '- [x]' in line or '- [-]' in line):
                    task_title = line.strip()[6:].strip()  # Remove "- [ ] " or "- [x] "
                    
                    # Try to extract numerical ID
                    import re
                    numerical_id_match = re.match(r'^(\d+)\.\s*(.+)', task_title)
                    if numerical_id_match:
                        task_id = numerical_id_match.group(1)
                        clean_title = numerical_id_match.group(2).strip()
                    else:
                        # Use line-based ID for unnumbered tasks
                        task_id = f"task_{len(found_tasks) + 1}"
                        clean_title = task_title
                    
                    found_tasks[task_id] = {
                        'line_number': i,
                        'title': clean_title,
                        'completed': line.strip().startswith('- [x]'),
                        'original_line': line
                    }
            
            # Check for missing tasks
            missing_tasks = []
            for expected_task in expected_tasks:
                if expected_task.id not in found_tasks:
                    missing_tasks.append(expected_task)
            
            if missing_tasks:
                self.logger.error(f"Found {len(missing_tasks)} missing tasks in {tasks_file}")
                self._record_task_update_stats('partial_update_recoveries')
                
                # Attempt to recover missing tasks by re-parsing and updating
                return self._recover_missing_tasks(tasks_file, missing_tasks, task_results)
            
            # Verify completion status matches execution results
            mismatched_tasks = []
            for task_result in task_results:
                task_id = task_result.get('task_id')
                expected_completed = task_result.get('success', False)
                
                if task_id in found_tasks:
                    actual_completed = found_tasks[task_id]['completed']
                    if expected_completed != actual_completed:
                        mismatched_tasks.append({
                            'task_id': task_id,
                            'expected': expected_completed,
                            'actual': actual_completed
                        })
            
            if mismatched_tasks:
                self.logger.warning(f"Found {len(mismatched_tasks)} tasks with mismatched completion status")
                self._log_task_completion_debug("Mismatched tasks found", {"mismatches": mismatched_tasks})
                
                # Attempt to fix mismatched completion status
                return self._fix_mismatched_completion_status(tasks_file, mismatched_tasks, task_results)
            
            self._log_task_completion_debug("Task loss prevention check passed", {
                "found_tasks": len(found_tasks),
                "missing_tasks": len(missing_tasks),
                "mismatched_tasks": len(mismatched_tasks)
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in task loss prevention check: {e}")
            return False
    
    def _recover_missing_tasks(self, tasks_file: str, missing_tasks: List, task_results: List[Dict[str, Any]]) -> bool:
        """
        Attempt to recover missing tasks by re-parsing the original file.
        
        Args:
            tasks_file: Path to tasks.md file
            missing_tasks: List of missing TaskDefinition objects
            task_results: List of task execution results
            
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            self.logger.info(f"Attempting to recover {len(missing_tasks)} missing tasks")
            
            # Force re-parse tasks from file to rebuild mappings
            self.task_file_mappings = {}
            reparsed_tasks = self._parse_tasks_from_file(tasks_file)
            
            if len(reparsed_tasks) >= len(missing_tasks):
                self.logger.info("Successfully recovered missing tasks through re-parsing")
                
                # Apply completion status based on task results
                for task_result in task_results:
                    task_id = task_result.get('task_id')
                    success = task_result.get('success', False)
                    
                    # Try individual update first, then fallback to batch
                    if not self._update_single_task_completion(tasks_file, task_id, success):
                        self.logger.warning(f"Individual update failed for recovered task {task_id}")
                
                return True
            else:
                self.logger.error("Failed to recover all missing tasks through re-parsing")
                return False
                
        except Exception as e:
            self.logger.error(f"Error recovering missing tasks: {e}")
            return False
    
    def _fix_mismatched_completion_status(self, tasks_file: str, mismatched_tasks: List[Dict[str, Any]], 
                                        task_results: List[Dict[str, Any]]) -> bool:
        """
        Fix tasks with mismatched completion status.
        
        Args:
            tasks_file: Path to tasks.md file
            mismatched_tasks: List of tasks with mismatched completion status
            task_results: List of task execution results
            
        Returns:
            True if fixes were successful, False otherwise
        """
        try:
            self.logger.info(f"Attempting to fix {len(mismatched_tasks)} tasks with mismatched completion status")
            
            fixes_successful = 0
            for mismatch in mismatched_tasks:
                task_id = mismatch['task_id']
                expected_completed = mismatch['expected']
                
                # Try to update the task with correct completion status
                if self._update_single_task_completion(tasks_file, task_id, expected_completed):
                    fixes_successful += 1
                    self._log_task_completion_debug(f"Fixed completion status for task {task_id}", {
                        "expected": expected_completed,
                        "was": mismatch['actual']
                    })
                else:
                    self.logger.warning(f"Failed to fix completion status for task {task_id}")
            
            if fixes_successful == len(mismatched_tasks):
                self.logger.info("Successfully fixed all mismatched completion statuses")
                return True
            else:
                self.logger.warning(f"Only fixed {fixes_successful}/{len(mismatched_tasks)} mismatched completion statuses")
                return fixes_successful > 0
                
        except Exception as e:
            self.logger.error(f"Error fixing mismatched completion status: {e}")
            return False
    
    def _log_task_update_statistics(self, total_tasks: int, failed_updates: int, 
                                  fallback_used: bool, recovery_successful: bool) -> None:
        """
        Log comprehensive task update statistics for monitoring and debugging.
        
        Args:
            total_tasks: Total number of tasks processed
            failed_updates: Number of tasks with failed individual updates
            fallback_used: Whether fallback batch update was used
            recovery_successful: Whether recovery mechanisms were successful
        """
        stats = self.task_update_stats.copy()
        stats.update({
            'total_tasks_processed': total_tasks,
            'failed_individual_updates': failed_updates,
            'fallback_batch_update_used': fallback_used,
            'recovery_mechanism_successful': recovery_successful,
            'individual_update_success_rate': (
                stats['individual_updates_successful'] / max(stats['individual_updates_attempted'], 1) * 100
            )
        })
        
        self.logger.info("Task completion update statistics:")
        self.logger.info(f"  Total tasks processed: {stats['total_tasks_processed']}")
        self.logger.info(f"  Individual updates attempted: {stats['individual_updates_attempted']}")
        self.logger.info(f"  Individual updates successful: {stats['individual_updates_successful']}")
        self.logger.info(f"  Individual update success rate: {stats['individual_update_success_rate']:.1f}%")
        self.logger.info(f"  Fallback updates used: {stats['fallback_updates_used']}")
        self.logger.info(f"  File access errors: {stats['file_access_errors']}")
        self.logger.info(f"  Task identification errors: {stats['task_identification_errors']}")
        self.logger.info(f"  Partial update recoveries: {stats['partial_update_recoveries']}")
        self.logger.info(f"  Fallback batch update used: {stats['fallback_batch_update_used']}")
        self.logger.info(f"  Recovery mechanism successful: {stats['recovery_mechanism_successful']}")
        
        # Log detailed debug information if enabled
        if self.task_completion_config.get('detailed_logging_enabled', True):
            self._log_task_completion_debug("Complete task update statistics", stats)