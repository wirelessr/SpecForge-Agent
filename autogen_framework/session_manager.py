"""
Session Manager for the AutoGen multi-agent framework.

This module implements the SessionManager class which handles all session
persistence logic in isolation. It manages session state loading, saving,
and resetting operations while maintaining all existing session logic,
file formats, and error handling unchanged.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .models import WorkflowState, WorkflowPhase


# Use shared UserApprovalStatus enum values as plain strings in persisted data


class SessionManager:
    """
    Session manager for handling all session persistence logic.
    
    The SessionManager is responsible for:
    - Loading existing sessions or creating new ones
    - Saving session state to disk
    - Resetting session data
    - Managing session ID generation
    - Handling session file I/O operations
    """
    
    def __init__(self, workspace_path: str):
        """
        Initialize the Session Manager.
        
        Args:
            workspace_path: Path to the workspace root directory
        """
        self.workspace_path = Path(workspace_path)
        self.logger = logging.getLogger(__name__)
        
        # Session management
        self.session_id: Optional[str] = None
        self.session_file = self.workspace_path / "memory" / "session_state.json"
        
        # Session data
        self.current_workflow: Optional[WorkflowState] = None
        self.user_approval_status: Dict[str, str] = {}
        self.phase_results: Dict[str, Dict[str, Any]] = {}
        self.execution_log: list = []
        self.approval_log: list = []
        self.error_recovery_attempts: Dict[str, int] = {}
        self.workflow_summary: Dict[str, Any] = {
            'phases_completed': [],
            'tasks_completed': [],
            'token_usage': {},
            'compression_events': [],
            'auto_approvals': [],
            'errors_recovered': []
        }
        
        self.logger.info(f"SessionManager initialized for workspace: {workspace_path}")
    
    def load_or_create_session(self) -> Dict[str, Any]:
        """
        Load existing session or create a new one.
        
        Returns:
            Dictionary containing session data
        """
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
                self.user_approval_status = approval_data
                
                self.phase_results = session_data.get('phase_results', {})
                self.execution_log = session_data.get('execution_log', [])
                
                # Load auto-approve related data
                self.approval_log = session_data.get('approval_log', [])
                self.error_recovery_attempts = session_data.get('error_recovery_attempts', {})
                self.workflow_summary = session_data.get('workflow_summary', {
                    'phases_completed': [],
                    'tasks_completed': [],
                    'token_usage': {},
                    'compression_events': [],
                    'auto_approvals': [],
                    'errors_recovered': []
                })
                
                self.logger.info(f"Loaded existing session: {self.session_id}")
                return session_data
            else:
                # Create new session
                return self._create_new_session()
        except Exception as e:
            self.logger.warning(f"Failed to load session, creating new one: {e}")
            return self._create_new_session()
    
    def _create_new_session(self) -> Dict[str, Any]:
        """
        Create a new session with unique ID.
        
        Returns:
            Dictionary containing new session data
        """
        self.session_id = str(uuid.uuid4())
        self.current_workflow = None
        self.user_approval_status = {}
        self.phase_results = {}
        self.execution_log = []
        self.approval_log = []
        self.error_recovery_attempts = {}
        self.workflow_summary = {
            'phases_completed': [],
            'tasks_completed': [],
            'token_usage': {},
            'compression_events': [],
            'auto_approvals': [],
            'errors_recovered': []
        }
        
        success = self.save_session_state()
        self.logger.info(f"Created new session: {self.session_id}")
        
        return {
            'session_id': self.session_id,
            'current_workflow': None,
            'user_approval_status': {},
            'phase_results': {},
            'execution_log': [],
            'approval_log': [],
            'error_recovery_attempts': {},
            'workflow_summary': self.workflow_summary,
            'last_updated': datetime.now().isoformat(),
            'save_success': success
        }
    
    def save_session_state(self) -> bool:
        """
        Save current session state to disk.
        
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Ensure memory directory exists
            self.session_file.parent.mkdir(parents=True, exist_ok=True)
            
            session_data = {
                'session_id': self.session_id,
                'current_workflow': {
                    'phase': self.current_workflow.phase.value if self.current_workflow else None,
                    'work_directory': self.current_workflow.work_directory if self.current_workflow else None
                } if self.current_workflow else None,
                'user_approval_status': self.user_approval_status,
                'phase_results': self.phase_results,
                'execution_log': self.execution_log,
                'approval_log': self.approval_log,
                'error_recovery_attempts': self.error_recovery_attempts,
                'workflow_summary': self.workflow_summary,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Session state saved: {self.session_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save session state: {e}")
            return False
    
    def get_session_id(self) -> str:
        """
        Get the current session ID.
        
        Returns:
            Current session ID string
        """
        return self.session_id
    
    def get_session_data(self) -> Dict[str, Any]:
        """
        Get the current session data.
        
        Returns:
            Dictionary containing all session data
        """
        return {
            'session_id': self.session_id,
            'current_workflow': {
                'phase': self.current_workflow.phase.value if self.current_workflow else None,
                'work_directory': self.current_workflow.work_directory if self.current_workflow else None
            } if self.current_workflow else None,
            'user_approval_status': self.user_approval_status,
            'phase_results': self.phase_results,
            'execution_log': self.execution_log,
            'approval_log': self.approval_log,
            'error_recovery_attempts': self.error_recovery_attempts,
            'workflow_summary': self.workflow_summary
        }
    
    def reset_session(self) -> Dict[str, Any]:
        """
        Reset the current session and create a new one.
        
        Returns:
            Dictionary containing reset result and new session info
        """
        try:
            self.logger.info("Starting session reset...")
            
            # Clear in-memory state first (fast)
            self.current_workflow = None
            self.user_approval_status.clear()
            self.phase_results.clear()
            self.execution_log.clear()
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
            
            # Remove session file if it exists
            if self.session_file.exists():
                self.session_file.unlink()
                self.logger.info("Removed existing session file")
            
            # Create new session
            session_data = self._create_new_session()
            self.logger.info(f"Session reset completed, new session: {self.session_id}")
            
            return {
                "success": True,
                "message": f"Session reset successfully. New session ID: {self.session_id}",
                "session_id": self.session_id,
                "session_data": session_data
            }
            
        except Exception as e:
            error_msg = f"Failed to reset session: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }