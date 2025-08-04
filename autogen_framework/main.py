#!/usr/bin/env python3
"""
AutoGen Multi-Agent Framework - Main Entry Point

This module serves as the main entry point for the AutoGen multi-agent framework.
It provides a command-line interface for initializing the framework, processing
user requests, and managing the complete workflow from requirements to implementation.

The main program handles:
- Command-line argument parsing
- Framework initialization and startup logic
- Basic error handling and logging
- User interaction through text interface
- Workflow management with approval checkpoints

Requirements fulfilled: 1.1, 1.2, 1.3, 1.4
"""

import click
import asyncio
import logging
import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .main_controller import MainController
from .models import LLMConfig
from .config_manager import ConfigManager, ConfigurationError


# Configure logging
def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration for the framework.
    
    Args:
        verbose: Enable verbose (DEBUG) logging
        log_file: Optional log file path
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def print_banner():
    """Print the framework banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                AutoGen Multi-Agent Framework                 ║
║                                                              ║
║  A framework for multi-agent collaboration using AutoGen    ║
║  for project development, debugging, and code review        ║
╚══════════════════════════════════════════════════════════════╝
    """
    click.echo(click.style(banner, fg='cyan', bold=True))


def print_help():
    """Print usage help and examples."""
    help_text = """
🚀 USAGE EXAMPLES:

1. Process a specific request:
   autogen-framework --request "Create a REST API for user management"

2. Continue existing workflow:
   autogen-framework --continue-workflow

3. Check framework status:
   autogen-framework --status

4. Reset session and start fresh:
   autogen-framework --reset-session

5. Approve a phase:
   autogen-framework --approve requirements

6. Reject a phase:
   autogen-framework --reject design

7. Revise a phase with feedback:
   autogen-framework --revise "requirements:Add more security details"

8. Execute a specific task:
   autogen-framework --execute-task "Create the Data Access Layer"

9. Enable verbose logging:
    autogen-framework --verbose --log-file framework.log

📋 WORKFLOW PHASES:
   1. Requirements Generation - Parse user request and create requirements.md
   2. Design Generation - Create technical design based on requirements
   3. Task Generation - Break down design into actionable tasks
   4. Implementation - Execute tasks with shell commands

⚙️  CONFIGURATION:
   The framework uses environment variables for configuration. Create a .env file:
   
   # Required LLM Configuration
   LLM_BASE_URL=http://your-llm-server:8888/openai/v1
   LLM_MODEL=your-model-name
   LLM_API_KEY=your-api-key
   
   # Optional Framework Configuration
   WORKSPACE_PATH=.
   LOG_FILE=logs/framework.log
   LOG_LEVEL=INFO
   
   Commands:
   - autogen-framework config init     # Create .env from template
   - autogen-framework config validate # Check configuration
   - autogen-framework config show     # Display current config

⚠️  SESSION-BASED WORKFLOW:
   The framework uses session-based workflow management. Each session has a unique ID
   and maintains workflow state across command invocations. Use --reset-session to
   start a new workflow session.



📁 WORKSPACE STRUCTURE:
   workspace/
   ├── memory/              # Framework memory and context
   ├── project-name/        # Generated work directories
   │   ├── requirements.md  # Generated requirements
   │   ├── design.md       # Technical design
   │   └── tasks.md        # Implementation tasks
   └── logs/               # Framework logs
    """
    click.echo(help_text)





@click.command()
@click.option('--workspace', '-w', 
              default=lambda: os.getenv('WORKSPACE_PATH', '.'),
              help='Workspace directory path (overrides WORKSPACE_PATH env var, default: current directory)')
@click.option('--request', '-r', 
              help='Process a specific request')
@click.option('--continue-workflow', is_flag=True,
              help='Continue an existing workflow')
@click.option('--status', is_flag=True,
              help='Show framework status and exit')
@click.option('--verbose', '-v', is_flag=True, 
              help='Enable verbose (DEBUG) logging (can also set LOG_LEVEL=DEBUG in .env)')
@click.option('--log-file', 
              default=lambda: os.getenv('LOG_FILE'),
              help='Log file path (overrides LOG_FILE env var, default: logs/framework.log)')
@click.option('--llm-base-url', 
              default=lambda: os.getenv('LLM_BASE_URL'),
              help='LLM API base URL (overrides LLM_BASE_URL env var)')
@click.option('--llm-model', 
              default=lambda: os.getenv('LLM_MODEL'),
              help='LLM model name (overrides LLM_MODEL env var)')
@click.option('--llm-api-key', 
              default=lambda: os.getenv('LLM_API_KEY'),
              help='LLM API key (overrides LLM_API_KEY env var)')
@click.option('--help-examples', is_flag=True,
              help='Show usage examples and exit')
@click.option('--reset-session', is_flag=True,
              help='Reset current session and start fresh')
@click.option('--approve', 
              help='Approve a specific phase (requirements/design/tasks)')
@click.option('--reject',
              help='Reject a specific phase (requirements/design/tasks)')
@click.option('--revise',
              help='Revise a specific phase with feedback (format: phase:feedback)')
@click.option('--execute-task',
              help='Execute a specific task from tasks.md (format: task_number or task_title)')
@click.option('--auto-approve', is_flag=True,
              help='Automatically approve all workflow phases without manual intervention')
def main(workspace: str, request: Optional[str], continue_workflow: bool,
         status: bool, verbose: bool, log_file: Optional[str],
         llm_base_url: str, llm_model: str, llm_api_key: str,
         help_examples: bool, reset_session: bool, approve: Optional[str], reject: Optional[str], revise: Optional[str], execute_task: Optional[str], auto_approve: bool):
    """
    AutoGen Multi-Agent Framework
    
    A framework for multi-agent collaboration using AutoGen for project development,
    debugging, and code review tasks. The framework processes user requests through
    a complete workflow: Requirements → Design → Tasks → Implementation.
    
    Each phase requires user approval before proceeding to the next phase.
    
    Configuration:
        The framework uses environment variables for configuration. Create a .env file
        in your project root with the required settings. CLI arguments override .env values.
        
        Required: LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
        Optional: WORKSPACE_PATH, LOG_FILE, LOG_LEVEL, and others
        
        Run 'autogen-framework config validate' to check your configuration.
    """
    
    # Show help examples and exit
    if help_examples:
        print_help()
        return
    
    try:
        # Initialize configuration manager first
        config_manager = ConfigManager()
        
        # Get framework configuration
        framework_config = config_manager.get_framework_config()
        
        # Setup workspace path (CLI overrides environment)
        if workspace != os.getenv('WORKSPACE_PATH', '.'):
            # CLI argument provided - show warning
            click.echo("⚠️  Using CLI argument for workspace path. Consider setting WORKSPACE_PATH in .env file.")
        workspace_path = Path(workspace).resolve()
        
        # Setup logging (CLI overrides environment)
        if not log_file:
            # Use environment default or fallback
            log_file = framework_config.get('log_file', 'logs/framework.log')
        elif log_file != os.getenv('LOG_FILE'):
            # CLI argument provided - show warning
            click.echo("⚠️  Using CLI argument for log file. Consider setting LOG_FILE in .env file.")
        
        # Ensure log directory exists
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = workspace_path / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle verbose flag vs LOG_LEVEL environment variable
        env_log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        if verbose and env_log_level != 'DEBUG':
            click.echo("⚠️  Using CLI verbose flag. Consider setting LOG_LEVEL=DEBUG in .env file.")
        
        # Use verbose if specified, otherwise respect environment LOG_LEVEL
        use_verbose = verbose or (env_log_level == 'DEBUG')
        setup_logging(use_verbose, str(log_path))
        logger = logging.getLogger(__name__)
        
        # Print banner
        print_banner()
        
        # Initialize LLM configuration from environment or CLI args
        # Check if CLI arguments were explicitly provided (not just from environment defaults)
        cli_args_provided = any([
            llm_base_url and llm_base_url != os.getenv('LLM_BASE_URL'),
            llm_model and llm_model != os.getenv('LLM_MODEL'),
            llm_api_key and llm_api_key != os.getenv('LLM_API_KEY')
        ])
        
        if cli_args_provided:
            # CLI arguments provided - use them with warning
            click.echo("⚠️  Using CLI arguments for LLM configuration. Consider using .env file instead.")
            
            # Get base config from environment and override with CLI args
            try:
                env_config = config_manager.get_llm_config()
            except ConfigurationError:
                # No environment config available, use CLI args only
                env_config = {}
            
            llm_config = LLMConfig(
                base_url=llm_base_url or env_config.get('base_url', ''),
                model=llm_model or env_config.get('model', ''),
                api_key=llm_api_key or env_config.get('api_key', ''),
                temperature=env_config.get('temperature', 0.7),
                max_output_tokens=env_config.get('max_output_tokens', 8192),
                timeout=env_config.get('timeout', 60)
            )
        else:
            # Use environment-based configuration
            llm_config = LLMConfig.from_config_manager(config_manager)
        
        click.echo(f"📁 Workspace: {workspace_path}")
        click.echo(f"🤖 LLM Model: {llm_config.model}")
        click.echo(f"🔗 LLM Base URL: {llm_config.base_url}")
        if use_verbose:
            click.echo(f"📝 Log File: {log_path}")
            click.echo(f"🔧 Config Source: {'CLI + Environment' if cli_args_provided else 'Environment (.env)'}")
        
        # Initialize MainController
        click.echo("\n🔧 Initializing framework...")
        controller = MainController(str(workspace_path))
        
        # Handle session reset
        if reset_session:
            controller.reset_session()
            click.echo(f"🔄 Session reset. New session ID: {controller.get_session_id()}")
            return  # Exit early for reset-session, no need to initialize framework
        
        # Initialize framework
        if not controller.initialize_framework(llm_config):
            click.echo("❌ Framework initialization failed!")
            logger.error("Framework initialization failed")
            sys.exit(1)
        
        click.echo(f"✅ Framework initialized successfully! Session ID: {controller.get_session_id()}")
        
        # Handle different modes
        if status:
            # Show status and exit
            framework_status = controller.get_framework_status()
            click.echo(f"\n📊 Framework Status (Session: {framework_status.get('session_id', 'Unknown')}):")
            click.echo(json.dumps(framework_status, indent=2))
            return
        
        elif approve:
            # Approve a specific phase
            click.echo(f"\n✅ Approving {approve} phase...")
            try:
                result = controller.approve_phase(approve, True)
                click.echo(f"✅ {result['message']}")
                
                # Try to continue workflow after approval
                async def continue_after_approval():
                    try:
                        continue_result = await controller.continue_workflow()
                        if continue_result.get("workflow_completed"):
                            click.echo("🎉 Workflow completed!")
                        elif continue_result.get("requires_approval"):
                            next_phase = continue_result.get("phase")
                            click.echo(f"⏳ Next phase '{next_phase}' ready for approval")
                        else:
                            click.echo("📋 Workflow continued successfully")
                    except Exception as e:
                        click.echo(f"❌ Error continuing workflow: {e}")
                
                asyncio.run(continue_after_approval())
            except Exception as e:
                click.echo(f"❌ Error approving phase: {e}")
                logger.error(f"Approval error: {e}")
            
        elif reject:
            # Reject a specific phase
            click.echo(f"\n❌ Rejecting {reject} phase...")
            try:
                result = controller.approve_phase(reject, False)
                click.echo(f"❌ {result['message']}")
            except Exception as e:
                click.echo(f"❌ Error rejecting phase: {e}")
                logger.error(f"Rejection error: {e}")
            
        elif revise:
            # Revise a specific phase
            if ':' not in revise:
                click.echo("❌ Revise format should be 'phase:feedback' (e.g., 'requirements:Add more security details')")
                return
            
            parts = revise.split(':', 1)
            if len(parts) != 2:
                click.echo("❌ Invalid revise format. Use 'phase:feedback'")
                return
                
            phase, feedback = parts
            phase = phase.strip()
            feedback = feedback.strip()
            
            # Validate phase
            valid_phases = ["requirements", "design", "tasks"]
            if phase not in valid_phases:
                click.echo(f"❌ Invalid phase '{phase}'. Valid phases are: {', '.join(valid_phases)}")
                return
            
            # Validate feedback is not empty
            if not feedback:
                click.echo("❌ Feedback cannot be empty. Please provide specific revision feedback.")
                return
            
            click.echo(f"\n🔄 Revising {phase} phase with feedback: {feedback}")
            
            try:
                async def apply_revision():
                    result = await controller.apply_phase_revision(phase, feedback)
                    if result.get("success"):
                        click.echo(f"✅ {result['message']}")
                        if result.get("updated_path"):
                            click.echo(f"📄 Updated file: {result['updated_path']}")
                    else:
                        click.echo(f"❌ Revision failed: {result.get('error')}")
                
                asyncio.run(apply_revision())
            except Exception as e:
                click.echo(f"❌ Error applying revision: {e}")
                logger.error(f"Revision error: {e}")
            
        elif execute_task:
            # Execute a specific task
            click.echo(f"\n🔧 Executing task: {execute_task}")
            
            try:
                async def execute_specific_task():
                    # Parse task definition from tasks.md
                    task_definition = {
                        "id": execute_task,
                        "title": execute_task,
                        "description": f"Execute task: {execute_task}"
                    }
                    
                    result = await controller.execute_specific_task(task_definition)
                    if result.get("success"):
                        click.echo(f"✅ Task completed successfully")
                        if result.get("output"):
                            click.echo(f"📄 Output: {result['output']}")
                    else:
                        click.echo(f"❌ Task execution failed: {result.get('error')}")
                
                asyncio.run(execute_specific_task())
            except Exception as e:
                click.echo(f"❌ Error executing task: {e}")
                logger.error(f"Task execution error: {e}")
            
        elif continue_workflow:
            # Continue existing workflow
            click.echo("\n🔄 Continuing existing workflow...")
            
            async def continue_existing():
                try:
                    result = await controller.continue_workflow()
                    
                    if result.get("workflow_completed"):
                        click.echo("🎉 Workflow completed!")
                    elif result.get("requires_approval"):
                        phase = result.get("phase")
                        click.echo(f"⏳ Phase '{phase}' ready for approval")
                    else:
                        error = result.get("error", "Unknown error")
                        click.echo(f"❌ Error: {error}")
                        
                except Exception as e:
                    click.echo(f"❌ Error continuing workflow: {e}")
                    logger.error(f"Continue workflow error: {e}")
            
            asyncio.run(continue_existing())
            
        elif request:
            # Process specific request
            click.echo(f"\n🔄 Processing request: {request}")
            
            async def process_request():
                try:
                    result = await controller.process_request(request, auto_approve=auto_approve)
                    
                    if result.get("success"):
                        click.echo("✅ Request processed successfully!")
                        if result.get("workflow_completed"):
                            click.echo("🎉 Workflow completed!")
                    elif result.get("requires_user_approval"):
                        phase = result.get("approval_needed_for")
                        click.echo(f"⏳ User approval required for: {phase}")
                        
                        # Show file paths
                        if phase == "requirements" and result.get("requirements_path"):
                            click.echo(f"📄 Requirements: {result['requirements_path']}")
                        elif phase == "design" and result.get("design_path"):
                            click.echo(f"📐 Design: {result['design_path']}")
                        elif phase == "tasks" and result.get("tasks_path"):
                            click.echo(f"📋 Tasks: {result['tasks_path']}")
                        
                        click.echo("Use --continue-workflow to proceed after approval")
                    else:
                        error = result.get("error", "Unknown error")
                        click.echo(f"❌ Error: {error}")
                        
                except Exception as e:
                    click.echo(f"❌ Error processing request: {e}")
                    logger.error(f"Request processing error: {e}")
            
            asyncio.run(process_request())
            
        else:
            # No specific command given, show help
            print_help()
    
    except ConfigurationError as e:
        click.echo(f"❌ Configuration Error: {e}")
        click.echo("\n💡 To fix this:")
        click.echo("   1. Create a .env file in your project root")
        click.echo("   2. Copy .env.example and update with your values")
        click.echo("   3. Or run: autogen-framework config init")
        click.echo("   4. Validate with: autogen-framework config validate")
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\n👋 Interrupted by user")
        logger.info("Framework interrupted by user")
    except Exception as e:
        click.echo(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()