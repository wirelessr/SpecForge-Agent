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
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from enum import Enum

from .models import LLMConfig, WorkflowState, WorkflowPhase
from .memory_manager import MemoryManager
from .agent_manager import AgentManager
from .shell_executor import ShellExecutor
from .config_manager import ConfigManager, ConfigurationError


class UserApprovalStatus(Enum):
    """Status of user approval for workflow phases."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


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
        self.memory_manager: Optional[MemoryManager] = None
        self.agent_manager: Optional[AgentManager] = None
        self.shell_executor: Optional[ShellExecutor] = None
        
        # Framework configuration
        self.llm_config: Optional[LLMConfig] = None
        self.is_initialized = False
        
        # Session management
        self.session_id: Optional[str] = None
        self.session_file = self.workspace_path / "memory" / "session_state.json"
        
        # Workflow state management
        self.current_workflow: Optional[WorkflowState] = None
        self.user_approval_status: Dict[str, UserApprovalStatus] = {}
        
        # Framework execution log
        self.execution_log: List[Dict[str, Any]] = []
        
        # Phase results storage
        self.phase_results: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"MainController initialized for workspace: {workspace_path}")
    
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
    
    async def process_user_request(self, user_request: str) -> Dict[str, Any]:
        """
        Process a user request through the complete workflow with approval checkpoints.
        
        This is the main entry point for handling user requests. It manages the
        entire workflow from requirements generation through implementation,
        with proper user approval checkpoints at each phase.
        
        Args:
            user_request: The user's request string
            
        Returns:
            Dictionary containing the complete workflow results and status
        """
        if not self.is_initialized:
            raise RuntimeError("Framework not initialized. Call initialize_framework() first.")
        
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
        
        # Initialize workflow state
        workflow_id = self._generate_workflow_id()
        self.current_workflow = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory=""
        )
        
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
            "approval_needed_for": None
        }
        
        try:
            # Phase 1: Requirements Generation
            requirements_result = await self._execute_requirements_phase(user_request, workflow_id)
            workflow_result["phases"]["requirements"] = requirements_result
            workflow_result["current_phase"] = "requirements"
            
            if not requirements_result.get("success", False):
                raise RuntimeError("Requirements generation failed")
            
            # Check if user approval is needed for requirements
            if not self._is_phase_approved("requirements"):
                workflow_result["requires_user_approval"] = True
                workflow_result["approval_needed_for"] = "requirements"
                workflow_result["requirements_path"] = requirements_result.get("requirements_path")
                return workflow_result
            
            # Phase 2: Design Generation (only if requirements approved)
            design_result = await self._execute_design_phase(requirements_result, workflow_id)
            workflow_result["phases"]["design"] = design_result
            workflow_result["current_phase"] = "design"
            
            if not design_result.get("success", False):
                raise RuntimeError("Design generation failed")
            
            # Check if user approval is needed for design
            if not self._is_phase_approved("design"):
                workflow_result["requires_user_approval"] = True
                workflow_result["approval_needed_for"] = "design"
                workflow_result["design_path"] = design_result.get("design_path")
                return workflow_result
            
            # Phase 3: Task Generation (only if design approved)
            tasks_result = await self._execute_tasks_phase(design_result, requirements_result, workflow_id)
            workflow_result["phases"]["tasks"] = tasks_result
            workflow_result["current_phase"] = "tasks"
            
            if not tasks_result.get("success", False):
                raise RuntimeError("Task generation failed")
            
            # Check if user approval is needed for tasks
            if not self._is_phase_approved("tasks"):
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
    
    def get_framework_status(self) -> Dict[str, Any]:
        """
        Get the current status of the framework.
        
        Returns:
            Dictionary containing comprehensive framework status information
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
        
        # Add component status if initialized
        if self.is_initialized:
            if self.agent_manager:
                status["components"]["agent_manager"] = self.agent_manager.get_agent_status()
            if self.memory_manager:
                status["components"]["memory_manager"] = self.memory_manager.get_memory_stats()
            if self.shell_executor:
                status["components"]["shell_executor"] = self.shell_executor.get_execution_stats()
        
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
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """
        Get the complete execution log.
        
        Returns:
            List of execution events with timestamps and details
        """
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
            
            # Reset components if they exist
            if self.agent_manager and hasattr(self.agent_manager, 'reset_coordination_state'):
                self.agent_manager.reset_coordination_state()
            
            if self.shell_executor and hasattr(self.shell_executor, 'clear_history'):
                self.shell_executor.clear_history()
            
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
            # Initialize MemoryManager
            if self.memory_manager is None:
                self.memory_manager = MemoryManager(str(self.workspace_path))
                self.logger.info("MemoryManager initialized")
            
            # Initialize ShellExecutor
            if self.shell_executor is None:
                self.shell_executor = ShellExecutor(str(self.workspace_path))
                self.logger.info("ShellExecutor initialized")
            
            # Initialize AgentManager
            if self.agent_manager is None:
                self.agent_manager = AgentManager(str(self.workspace_path))
                self.logger.info("AgentManager initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing core components: {e}")
            return False
    
    def _load_framework_memory(self) -> bool:
        """Load memory context for the framework."""
        try:
            memory_content = self.memory_manager.load_memory()
            
            # Update agents with memory context if agent manager is available
            if self.agent_manager:
                self.agent_manager.update_agent_memory(memory_content)
            
            self.logger.info("Framework memory loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading framework memory: {e}")
            return False
    
    async def _execute_requirements_phase(self, user_request: str, workflow_id: str) -> Dict[str, Any]:
        """Execute the requirements generation phase."""
        self.logger.info("Executing requirements phase")
        self.current_workflow.phase = WorkflowPhase.PLANNING
        
        try:
            result = await self.agent_manager.coordinate_agents(
                "requirements_generation",
                {
                    "user_request": user_request,
                    "workspace_path": str(self.workspace_path)
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
                    "memory_context": self.memory_manager.load_memory()
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
            return {"success": False, "error": str(e)}
    
    async def _execute_tasks_phase(self, design_result: Dict[str, Any], 
                                  requirements_result: Dict[str, Any], workflow_id: str) -> Dict[str, Any]:
        """Execute the task generation phase."""
        self.logger.info("Executing tasks phase")
        self.current_workflow.phase = WorkflowPhase.TASK_GENERATION
        
        try:
            result = await self.agent_manager.coordinate_agents(
                "task_execution",
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
            return {"success": False, "error": str(e)}
    
    def _is_phase_approved(self, phase: str) -> bool:
        """Check if a phase has been approved by the user."""
        return self.user_approval_status.get(phase) == UserApprovalStatus.APPROVED
    
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
    
    def _parse_tasks_from_file(self, tasks_file: str) -> List['TaskDefinition']:
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
                "execution_log": self._make_json_safe(self.execution_log),
                "workflow_state": {
                    "current_workflow": {
                        "phase": self.current_workflow.phase.value if self.current_workflow else None,
                        "work_directory": self.current_workflow.work_directory if self.current_workflow else None
                    },
                    "approval_status": {k: v.value for k, v in self.user_approval_status.items()}
                }
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
        try:
            if self.session_file.exists():
                # Load existing session
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                self.session_id = session_data.get('session_id')
                
                # Reconstruct workflow object if it exists
                workflow_data = session_data.get('current_workflow')
                if workflow_data:
                    self.current_workflow = WorkflowState(
                        phase=WorkflowPhase(workflow_data.get('phase')) if workflow_data.get('phase') else WorkflowPhase.PLANNING,
                        work_directory=workflow_data.get('work_directory', '')
                    )
                else:
                    self.current_workflow = None
                
                # Load approval status
                approval_data = session_data.get('user_approval_status', {})
                self.user_approval_status = {
                    k: UserApprovalStatus(v) for k, v in approval_data.items()
                }
                
                self.phase_results = session_data.get('phase_results', {})
                self.execution_log = session_data.get('execution_log', [])
                
                self.logger.info(f"Loaded existing session: {self.session_id}")
            else:
                # Create new session
                self._create_new_session()
        except Exception as e:
            self.logger.warning(f"Failed to load session, creating new one: {e}")
            self._create_new_session()
    
    def _create_new_session(self):
        """Create a new session with unique ID."""
        self.session_id = str(uuid.uuid4())
        self.current_workflow = None
        self.user_approval_status = {}
        self.phase_results = {}
        self.execution_log = []
        self._save_session_state()
        self.logger.info(f"Created new session: {self.session_id}")
    
    def _save_session_state(self):
        """Save current session state to disk."""
        try:
            # Ensure memory directory exists
            self.session_file.parent.mkdir(parents=True, exist_ok=True)
            
            session_data = {
                'session_id': self.session_id,
                'current_workflow': {
                    'phase': self.current_workflow.phase.value if self.current_workflow else None,
                    'work_directory': self.current_workflow.work_directory if self.current_workflow else None
                } if self.current_workflow else None,
                'user_approval_status': {k: v.value for k, v in self.user_approval_status.items()},
                'phase_results': self.phase_results,
                'execution_log': self.execution_log,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Session state saved: {self.session_id}")
        except Exception as e:
            self.logger.error(f"Failed to save session state: {e}")
    
    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self.session_id
    
    def reset_session(self):
        """Reset the current session and create a new one."""
        try:
            self.logger.info("Starting session reset...")
            
            # Clear in-memory state first (fast)
            self.current_workflow = None
            self.user_approval_status.clear()
            self.phase_results.clear()
            self.execution_log.clear()
            
            # Remove session file if it exists
            if self.session_file.exists():
                self.session_file.unlink()
                self.logger.info("Removed existing session file")
            
            # Create new session
            self._create_new_session()
            self.logger.info(f"Session reset completed, new session: {self.session_id}")
            
            return {
                "success": True,
                "message": f"Session reset successfully. New session ID: {self.session_id}",
                "session_id": self.session_id
            }
            
        except Exception as e:
            error_msg = f"Failed to reset session: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }