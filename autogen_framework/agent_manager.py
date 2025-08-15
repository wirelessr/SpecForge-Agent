"""
Agent Manager for the AutoGen multi-agent framework.

This module implements the AgentManager class responsible for coordinating
multiple AI agents (Plan, Design, and Implement agents) to work together
on complex tasks. It handles agent initialization, communication, coordination,
and result integration.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .agents.base_agent import BaseLLMAgent
from .agents.plan_agent import PlanAgent
from .agents.design_agent import DesignAgent
from .agents.implement_agent import ImplementAgent
from .agents.tasks_agent import TasksAgent
from .models import LLMConfig, WorkflowState, WorkflowPhase, AgentContext
from .memory_manager import MemoryManager
from .token_manager import TokenManager
from .context_manager import ContextManager
from .context_compressor import ContextCompressor
from .config_manager import ConfigManager
from .shell_executor import ShellExecutor


class AgentManager:
    """
    Manages and coordinates multiple AI agents in the framework.
    
    The AgentManager is responsible for:
    - Initializing and configuring all agents
    - Coordinating agent interactions and workflows
    - Managing communication between agents
    - Integrating agent outputs into unified results
    - Recording agent interactions and decision processes
    - Maintaining workflow state and progress tracking
    """
    
    def __init__(self, workspace_path: str):
        """
        Initialize the Agent Manager.
        
        Args:
            workspace_path: Path to the workspace root directory
        """
        self.workspace_path = Path(workspace_path)
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.memory_manager = MemoryManager(workspace_path)
        self.context_manager: Optional['ContextManager'] = None
        self.shell_executor = ShellExecutor(str(self.workspace_path))
        
        # Agent instances (initialized later)
        self.plan_agent: Optional[PlanAgent] = None
        self.design_agent: Optional[DesignAgent] = None
        self.tasks_agent: Optional[TasksAgent] = None
        self.implement_agent: Optional[ImplementAgent] = None
        
        # Agent registry for dynamic access
        self.agents: Dict[str, BaseLLMAgent] = {}
        
        # Coordination state
        self.workflow_state: Optional[WorkflowState] = None
        self.agent_interactions: List[Dict[str, Any]] = []
        self.coordination_log: List[Dict[str, Any]] = []
        
        # Configuration
        self.llm_config: Optional[LLMConfig] = None
        self.is_initialized = False
        
        self.logger.info(f"AgentManager initialized for workspace: {workspace_path}")
    
    def setup_agents(self, llm_config: LLMConfig) -> bool:
        """
        Initialize and configure all AI agents.
        
        This method sets up the three core agents (Plan, Design, Implement)
        with the provided LLM configuration and integrates them with the
        memory system and shell executor.
        
        Args:
            llm_config: LLM configuration for all agents
            
        Returns:
            True if all agents were successfully initialized, False otherwise
        """
        try:
            self.llm_config = llm_config
            
            # Validate LLM configuration
            if not llm_config.validate():
                raise ValueError("Invalid LLM configuration provided")
            
            # Load memory context for agents
            memory_context = self.memory_manager.load_memory()

            # Initialize core managers (mandatory)
            config_manager = ConfigManager()
            token_manager = TokenManager(config_manager)
            context_compressor = ContextCompressor(llm_config, token_manager=token_manager)
            self.context_manager = ContextManager(
                work_dir=str(self.workspace_path),
                memory_manager=self.memory_manager,
                context_compressor=context_compressor,
                llm_config=llm_config,
                token_manager=token_manager,
                config_manager=config_manager,
            )
            
            self.logger.info("Initializing agents with LLM configuration")
            
            # Initialize Plan Agent
            self.plan_agent = PlanAgent(
                llm_config=llm_config,
                memory_manager=self.memory_manager,
                token_manager=token_manager,
                context_manager=self.context_manager,
                config_manager=config_manager,
            )
            self.agents["plan"] = self.plan_agent
            
            # Initialize Design Agent
            self.design_agent = DesignAgent(
                llm_config=llm_config,
                memory_context=memory_context,
                token_manager=token_manager,
                context_manager=self.context_manager,
                config_manager=config_manager,
            )
            self.agents["design"] = self.design_agent
            
            # Initialize Tasks Agent
            self.tasks_agent = TasksAgent(
                llm_config=llm_config,
                memory_manager=self.memory_manager,
                token_manager=token_manager,
                context_manager=self.context_manager,
                config_manager=config_manager,
            )
            self.agents["tasks"] = self.tasks_agent
            
            # Initialize Implement Agent
            implement_system_message = self._build_implement_agent_system_message()
            self.implement_agent = ImplementAgent(
                name="ImplementAgent",
                llm_config=llm_config,
                system_message=implement_system_message,
                shell_executor=self.shell_executor,
                token_manager=token_manager,
                context_manager=self.context_manager,
                config_manager=config_manager,
            )
            self.agents["implement"] = self.implement_agent
            
            # Initialize AutoGen agents for all
            initialization_results = []
            for agent_name, agent in self.agents.items():
                try:
                    success = agent.initialize_autogen_agent()
                    initialization_results.append((agent_name, success))
                    if success:
                        self.logger.info(f"Successfully initialized {agent_name} agent")
                    else:
                        self.logger.error(f"Failed to initialize {agent_name} agent")
                except Exception as e:
                    self.logger.error(f"Exception initializing {agent_name} agent: {e}")
                    initialization_results.append((agent_name, False))
            
            # Check if all agents initialized successfully
            all_initialized = all(success for _, success in initialization_results)
            
            if all_initialized:
                self.is_initialized = True
                self.logger.info("All agents initialized successfully")
                
                # Record initialization in coordination log
                self._record_coordination_event(
                    event_type="agent_initialization",
                    details={
                        "agents_initialized": list(self.agents.keys()),
                        "llm_config": {
                            "model": llm_config.model,
                            "base_url": llm_config.base_url
                        },
                        "memory_categories": list(memory_context.keys()) if memory_context else []
                    }
                )
                
                return True
            else:
                failed_agents = [name for name, success in initialization_results if not success]
                self.logger.error(f"Failed to initialize agents: {failed_agents}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error setting up agents: {e}")
            return False
    
    def set_context_manager(self, context_manager: 'ContextManager') -> None:
        """
        Set the ContextManager for all agents.
        
        Args:
            context_manager: ContextManager instance to be used by all agents
        """
        self.context_manager = context_manager
        
        # Update all agents with the context manager
        if self.plan_agent:
            self.plan_agent.set_context_manager(context_manager)
        if self.design_agent:
            self.design_agent.set_context_manager(context_manager)
        if self.tasks_agent:
            self.tasks_agent.set_context_manager(context_manager)
        if self.implement_agent:
            self.implement_agent.set_context_manager(context_manager)
        
        self.logger.info("ContextManager set for all agents")
    
    async def coordinate_agents(self, task_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate multiple agents to work together on a task.
        
        This is the main coordination method that orchestrates agent interactions
        based on the task type and manages the overall workflow.
        
        Args:
            task_type: Type of coordination task (e.g., "full_workflow", "requirements_only")
            context: Context information including user request and parameters
            
        Returns:
            Dictionary containing coordinated results from all relevant agents
        """
        if not self.is_initialized:
            raise RuntimeError("Agents not initialized. Call setup_agents() first.")
        
        self.logger.info(f"Starting agent coordination for task: {task_type}")
        
        # Record coordination start
        coordination_id = self._generate_coordination_id()
        self._record_coordination_event(
            event_type="coordination_start",
            details={
                "coordination_id": coordination_id,
                "task_type": task_type,
                "context_keys": list(context.keys())
            }
        )
        
        try:
            if task_type == "full_workflow":
                return await self._coordinate_full_workflow(context, coordination_id)
            elif task_type == "requirements_generation":
                return await self._coordinate_requirements_generation(context, coordination_id)
            elif task_type == "design_generation":
                return await self._coordinate_design_generation(context, coordination_id)
            elif task_type == "task_generation":
                return await self._coordinate_task_generation(context, coordination_id)
            elif task_type == "task_execution":
                return await self._coordinate_task_execution(context, coordination_id)
            elif task_type == "execute_multiple_tasks":
                return await self._coordinate_multiple_task_execution(context, coordination_id)
            elif task_type.endswith("_revision"):
                return await self._coordinate_phase_revision(context, coordination_id)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except ValueError as ve:
            # Re-raise ValueError for unknown task types
            raise ve
        except Exception as e:
            self.logger.error(f"Error in agent coordination: {e}")
            self._record_coordination_event(
                event_type="coordination_error",
                details={
                    "coordination_id": coordination_id,
                    "error": str(e),
                    "task_type": task_type
                }
            )
            return {
                "success": False,
                "error": str(e),
                "coordination_id": coordination_id
            }
    
    async def _coordinate_full_workflow(self, context: Dict[str, Any], coordination_id: str) -> Dict[str, Any]:
        """
        Coordinate the complete workflow from requirements to implementation.
        
        Args:
            context: Context containing user_request and other parameters
            coordination_id: Unique identifier for this coordination session
            
        Returns:
            Dictionary containing results from all workflow phases
        """
        user_request = context.get("user_request", "")
        if not user_request:
            raise ValueError("user_request is required for full workflow")
        
        workflow_results = {
            "coordination_id": coordination_id,
            "workflow_phases": {},
            "success": False,
            "current_phase": None
        }
        
        # Initialize workflow state
        self.workflow_state = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory=""
        )
        
        try:
            # Phase 1: Requirements Generation (Plan Agent)
            self.logger.info("Phase 1: Requirements generation")
            self.workflow_state.phase = WorkflowPhase.PLANNING
            workflow_results["current_phase"] = "planning"
            
            plan_result = await self._execute_agent_task(
                agent_name="plan",
                task_input={
                    "user_request": user_request,
                    "workspace_path": str(self.workspace_path)
                },
                coordination_id=coordination_id
            )
            
            workflow_results["workflow_phases"]["planning"] = plan_result
            
            if not plan_result.get("success", False):
                raise RuntimeError("Requirements generation failed")
            
            # Update workflow state
            work_directory = plan_result["work_directory"]
            requirements_path = plan_result["requirements_path"]
            self.workflow_state.work_directory = work_directory
            
            # Phase 2: Design Generation (Design Agent)
            self.logger.info("Phase 2: Design generation")
            self.workflow_state.phase = WorkflowPhase.DESIGN
            workflow_results["current_phase"] = "design"
            
            design_result = await self._execute_agent_task(
                agent_name="design",
                task_input={
                    "requirements_path": requirements_path,
                    "work_directory": work_directory,
                    "memory_context": self.memory_manager.load_memory()
                },
                coordination_id=coordination_id
            )
            
            workflow_results["workflow_phases"]["design"] = design_result
            
            if not design_result.get("success", False):
                raise RuntimeError("Design generation failed")
            
            design_path = design_result["design_path"]
            
            # Phase 3: Task Generation (Implement Agent)
            self.logger.info("Phase 3: Task list generation")
            self.workflow_state.phase = WorkflowPhase.TASK_GENERATION
            workflow_results["current_phase"] = "task_generation"
            
            task_generation_result = await self._execute_agent_task(
                agent_name="tasks",
                task_input={
                    "task_type": "generate_task_list",
                    "design_path": design_path,
                    "requirements_path": requirements_path,
                    "work_dir": work_directory
                },
                coordination_id=coordination_id
            )
            
            workflow_results["workflow_phases"]["task_generation"] = task_generation_result
            
            if not task_generation_result.get("success", False):
                raise RuntimeError("Task generation failed")
            
            # Phase 4: Implementation (Implement Agent)
            self.logger.info("Phase 4: Task execution")
            self.workflow_state.phase = WorkflowPhase.IMPLEMENTATION
            workflow_results["current_phase"] = "implementation"
            
            # Note: Actual task execution would be handled separately
            # This phase just prepares for execution
            implementation_result = {
                "success": True,
                "message": "Tasks generated and ready for execution",
                "tasks_file": task_generation_result.get("tasks_file"),
                "work_directory": work_directory
            }
            
            workflow_results["workflow_phases"]["implementation"] = implementation_result
            
            # Mark workflow as completed
            self.workflow_state.phase = WorkflowPhase.COMPLETED
            workflow_results["current_phase"] = "completed"
            workflow_results["success"] = True
            
            # Record successful coordination
            self._record_coordination_event(
                event_type="workflow_completed",
                details={
                    "coordination_id": coordination_id,
                    "work_directory": work_directory,
                    "phases_completed": list(workflow_results["workflow_phases"].keys())
                }
            )
            
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"Workflow coordination failed: {e}")
            workflow_results["error"] = str(e)
            
            self._record_coordination_event(
                event_type="workflow_failed",
                details={
                    "coordination_id": coordination_id,
                    "error": str(e),
                    "failed_phase": workflow_results.get("current_phase")
                }
            )
            
            return workflow_results    

    async def _coordinate_requirements_generation(self, context: Dict[str, Any], coordination_id: str) -> Dict[str, Any]:
        """
        Coordinate requirements generation using the Plan Agent.
        
        Args:
            context: Context containing user_request
            coordination_id: Unique identifier for this coordination session
            
        Returns:
            Dictionary containing requirements generation results
        """
        return await self._execute_agent_task(
            agent_name="plan",
            task_input=context,
            coordination_id=coordination_id
        )
    
    async def _coordinate_design_generation(self, context: Dict[str, Any], coordination_id: str) -> Dict[str, Any]:
        """
        Coordinate design generation using the Design Agent.
        
        Args:
            context: Context containing requirements_path and work_directory
            coordination_id: Unique identifier for this coordination session
            
        Returns:
            Dictionary containing design generation results
        """
        # Add memory context to the task input
        task_input = context.copy()
        task_input["memory_context"] = self.memory_manager.load_memory()
        
        return await self._execute_agent_task(
            agent_name="design",
            task_input=task_input,
            coordination_id=coordination_id
        )
    
    async def _coordinate_task_generation(self, context: Dict[str, Any], coordination_id: str) -> Dict[str, Any]:
        """
        Coordinate task generation using the Tasks Agent.
        
        Args:
            context: Context containing task generation parameters (design_path, requirements_path, work_dir)
            coordination_id: Unique identifier for this coordination session
            
        Returns:
            Dictionary containing task generation results
        """
        return await self._execute_agent_task(
            agent_name="tasks",
            task_input=context,
            coordination_id=coordination_id
        )
    
    async def _coordinate_task_execution(self, context: Dict[str, Any], coordination_id: str) -> Dict[str, Any]:
        """
        Coordinate task execution using the Implement Agent.
        
        Args:
            context: Context containing task execution parameters
            coordination_id: Unique identifier for this coordination session
            
        Returns:
            Dictionary containing task execution results
        """
        return await self._execute_agent_task(
            agent_name="implement",
            task_input=context,
            coordination_id=coordination_id
        )
    
    async def _coordinate_multiple_task_execution(self, context: Dict[str, Any], coordination_id: str) -> Dict[str, Any]:
        """
        Coordinate multiple task execution using the Implement Agent.
        
        Args:
            context: Context containing multiple task execution parameters
            coordination_id: Unique identifier for this coordination session
            
        Returns:
            Dictionary containing multiple task execution results
        """
        self.logger.info("Coordinating multiple task execution with implement agent")
        
        return await self._execute_agent_task(
            agent_name="implement",
            task_input=context,
            coordination_id=coordination_id
        )
    
    async def _coordinate_phase_revision(self, context: Dict[str, Any], coordination_id: str) -> Dict[str, Any]:
        """
        Coordinate phase revision using the appropriate agent.
        
        Args:
            context: Context containing revision parameters including phase and feedback
            coordination_id: Unique identifier for this coordination session
            
        Returns:
            Dictionary containing revision results
        """
        phase = context.get("phase")
        revision_feedback = context.get("revision_feedback")
        
        if not phase or not revision_feedback:
            return {
                "success": False,
                "error": "Missing phase or revision_feedback in context"
            }
        
        # Determine which agent to use based on phase
        agent_mapping = {
            "requirements": "plan",
            "design": "design", 
            "tasks": "tasks"  # Fixed: tasks revision should use TasksAgent, not ImplementAgent
        }
        
        agent_name = agent_mapping.get(phase)
        if not agent_name:
            return {
                "success": False,
                "error": f"No agent available for phase: {phase}"
            }
        
        # Prepare revision task input
        revision_task_input = {
            "task_type": "revision",
            "phase": phase,
            "revision_feedback": revision_feedback,
            "current_result": context.get("current_result"),
            "work_directory": context.get("work_directory")
        }
        
        self.logger.info(f"Coordinating {phase} revision with {agent_name} agent")
        
        return await self._execute_agent_task(
            agent_name=agent_name,
            task_input=revision_task_input,
            coordination_id=coordination_id
        )
    
    async def _execute_agent_task(
        self, 
        agent_name: str, 
        task_input: Dict[str, Any], 
        coordination_id: str
    ) -> Dict[str, Any]:
        """
        Execute a task using a specific agent and record the interaction.
        
        Args:
            agent_name: Name of the agent to use
            task_input: Input parameters for the agent task
            coordination_id: Unique identifier for the coordination session
            
        Returns:
            Dictionary containing task execution results
        """
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        agent = self.agents[agent_name]
        
        # Record agent interaction start
        interaction_start = datetime.now()
        interaction_id = f"{coordination_id}_{agent_name}_{int(interaction_start.timestamp())}"
        
        self._record_agent_interaction(
            interaction_id=interaction_id,
            agent_name=agent_name,
            event_type="task_start",
            details={
                "coordination_id": coordination_id,
                "task_input_keys": list(task_input.keys()),
                "agent_status": agent.get_agent_status()
            }
        )
        
        try:
            # Execute the agent task
            result = await agent.process_task(task_input)
            
            # Calculate execution time
            execution_time = (datetime.now() - interaction_start).total_seconds()
            
            # Record successful interaction
            self._record_agent_interaction(
                interaction_id=interaction_id,
                agent_name=agent_name,
                event_type="task_completed",
                details={
                    "coordination_id": coordination_id,
                    "execution_time": execution_time,
                    "success": result.get("success", False),
                    "result_keys": list(result.keys()) if isinstance(result, dict) else []
                }
            )
            
            self.logger.info(f"Agent {agent_name} completed task in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - interaction_start).total_seconds()
            
            # Record failed interaction
            self._record_agent_interaction(
                interaction_id=interaction_id,
                agent_name=agent_name,
                event_type="task_failed",
                details={
                    "coordination_id": coordination_id,
                    "execution_time": execution_time,
                    "error": str(e)
                }
            )
            
            self.logger.error(f"Agent {agent_name} task failed after {execution_time:.2f}s: {e}")
            raise
    
    def get_agent_status(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status information for one or all agents.
        
        Args:
            agent_name: Optional specific agent name, if None returns all agents
            
        Returns:
            Dictionary containing agent status information
        """
        if agent_name:
            if agent_name not in self.agents:
                raise ValueError(f"Unknown agent: {agent_name}")
            return self.agents[agent_name].get_agent_status()
        else:
            return {
                name: agent.get_agent_status() 
                for name, agent in self.agents.items()
            }
    
    def get_coordination_log(self) -> List[Dict[str, Any]]:
        """
        Get the complete coordination log.
        
        Returns:
            List of coordination events and agent interactions
        """
        return self.coordination_log.copy()
    
    def get_agent_interactions(self) -> List[Dict[str, Any]]:
        """
        Get the complete agent interaction log.
        
        Returns:
            List of agent interaction records
        """
        return self.agent_interactions.copy()
    
    def get_workflow_state(self) -> Optional[WorkflowState]:
        """
        Get the current workflow state.
        
        Returns:
            Current WorkflowState object or None if no workflow is active
        """
        return self.workflow_state
    
    def reset_coordination_state(self) -> None:
        """
        Reset coordination state and clear logs.
        
        This method clears all coordination logs and resets the workflow state
        while preserving agent configurations.
        """
        self.workflow_state = None
        self.agent_interactions.clear()
        self.coordination_log.clear()
        
        # Reset individual agent states
        for agent in self.agents.values():
            agent.reset_agent()
        
        self.logger.info("Coordination state reset")
    
    def export_coordination_report(self, output_path: str) -> bool:
        """
        Export a comprehensive coordination report.
        
        Args:
            output_path: Path where to save the coordination report
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "workspace_path": str(self.workspace_path),
                "agents_initialized": list(self.agents.keys()),
                "workflow_state": {
                    "phase": self.workflow_state.phase.value if self.workflow_state else None,
                    "work_directory": self.workflow_state.work_directory if self.workflow_state else None
                },
                "coordination_events": self.coordination_log,
                "agent_interactions": self.agent_interactions,
                "agent_status": self.get_agent_status()
            }
            
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Coordination report exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting coordination report: {e}")
            return False
    
    # Private helper methods
    
    def _build_implement_agent_system_message(self) -> str:
        """Build the system message for the Implement Agent."""
        return """You are the Implement Agent in an AutoGen multi-agent framework. Your responsibilities are:

1. **Generate Task Lists**: Create comprehensive tasks.md files from design documents
2. **Execute Coding Tasks**: Implement specific coding tasks through shell commands
3. **Manage Execution**: Handle retry mechanisms and error recovery
4. **Record Progress**: Document task completion and learning outcomes

## Key Guidelines:

### Task List Generation
- Break down designs into small, testable coding tasks
- Use markdown checkbox format with detailed steps
- Reference specific requirements for each task
- Ensure tasks build incrementally on each other

### Task Execution
- Use shell commands for all operations (file creation, editing, testing)
- Implement patch-first strategy for file modifications
- Apply retry mechanisms with different approaches
- Record detailed execution logs and learning outcomes

### Shell Integration
- Execute all operations through the provided ShellExecutor
- Support file operations, Python execution, testing, and any command-line tasks
- Handle errors gracefully and try alternative approaches
- Maintain detailed command history and results

### Learning and Memory
- Record task completion details in project directories
- Extract reusable knowledge for global memory
- Document challenges, solutions, and best practices
- Update memory with patterns and techniques learned

You should be thorough, systematic, and focused on producing working, tested code.
"""
    
    def _generate_coordination_id(self) -> str:
        """Generate a unique coordination ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"coord_{timestamp}"
    
    def _record_coordination_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """
        Record a coordination event in the log.
        
        Args:
            event_type: Type of coordination event
            details: Event details and metadata
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        self.coordination_log.append(event)
        self.logger.debug(f"Recorded coordination event: {event_type}")
    
    def _record_agent_interaction(
        self, 
        interaction_id: str, 
        agent_name: str, 
        event_type: str, 
        details: Dict[str, Any]
    ) -> None:
        """
        Record an agent interaction in the log.
        
        Args:
            interaction_id: Unique identifier for this interaction
            agent_name: Name of the agent involved
            event_type: Type of interaction event
            details: Interaction details and metadata
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "interaction_id": interaction_id,
            "agent_name": agent_name,
            "event_type": event_type,
            "details": details
        }
        
        self.agent_interactions.append(interaction)
        self.logger.debug(f"Recorded agent interaction: {agent_name} - {event_type}")
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """
        Get capabilities of all agents.
        
        Returns:
            Dictionary mapping agent names to their capabilities
        """
        return {
            name: agent.get_agent_capabilities() 
            for name, agent in self.agents.items()
        }
    
    def update_agent_memory(self, memory_context: Dict[str, Any]) -> None:
        """
        Update memory context for all agents.
        
        Args:
            memory_context: New memory context to apply to all agents
        """
        for agent in self.agents.values():
            agent.update_memory_context(memory_context)
        
        self.logger.info("Updated memory context for all agents")
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about agent coordination.
        
        Returns:
            Dictionary containing coordination statistics
        """
        total_interactions = len(self.agent_interactions)
        total_events = len(self.coordination_log)
        
        # Count interactions by agent
        agent_interaction_counts = {}
        for interaction in self.agent_interactions:
            agent_name = interaction["agent_name"]
            agent_interaction_counts[agent_name] = agent_interaction_counts.get(agent_name, 0) + 1
        
        # Count events by type
        event_type_counts = {}
        for event in self.coordination_log:
            event_type = event["event_type"]
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        
        return {
            "total_interactions": total_interactions,
            "total_coordination_events": total_events,
            "agent_interaction_counts": agent_interaction_counts,
            "event_type_counts": event_type_counts,
            "agents_initialized": len(self.agents),
            "current_workflow_phase": self.workflow_state.phase.value if self.workflow_state else None
        }