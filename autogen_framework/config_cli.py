#!/usr/bin/env python3
"""
Configuration CLI tool for the AutoGen Multi-Agent Framework.

This module provides a command-line interface for validating and managing
configuration settings, including environment variables and .env files.
"""

import click
import sys
from pathlib import Path

from .config_manager import ConfigManager, ConfigurationError


@click.group()
def config_cli():
    """Configuration management CLI for AutoGen Framework."""
    pass


@config_cli.command()
@click.option('--env-file', '-e', help='Path to .env file to validate')
def validate(env_file):
    """Validate configuration and show status."""
    try:
        config_manager = ConfigManager(env_file=env_file)
        config_manager.print_config_status()
        
        errors = config_manager.validate_required_config()
        if errors:
            click.echo("\n❌ Configuration validation failed:", err=True)
            for error in errors:
                click.echo(f"  {error}", err=True)
            sys.exit(1)
        else:
            click.echo("\n✅ Configuration validation passed!")
            
    except Exception as e:
        click.echo(f"❌ Error validating configuration: {e}", err=True)
        sys.exit(1)


@config_cli.command()
@click.option('--output', '-o', default='.env', help='Output file path (default: .env)')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing file')
def init(output, force):
    """Initialize a new .env file from template."""
    output_path = Path(output)
    
    if output_path.exists() and not force:
        click.echo(f"❌ File {output} already exists. Use --force to overwrite.", err=True)
        sys.exit(1)
    
    # Find .env.example template
    template_path = Path('.env.example')
    if not template_path.exists():
        # Look in parent directories
        current_dir = Path.cwd()
        for path in [current_dir] + list(current_dir.parents):
            template_candidate = path / '.env.example'
            if template_candidate.exists():
                template_path = template_candidate
                break
        else:
            click.echo("❌ Could not find .env.example template file.", err=True)
            sys.exit(1)
    
    try:
        # Copy template to output file
        template_content = template_path.read_text()
        output_path.write_text(template_content)
        
        click.echo(f"✅ Created {output} from template")
        click.echo(f"📝 Please edit {output} and update the configuration values")
        click.echo(f"💡 Run 'autogen-config validate' to check your configuration")
        
    except Exception as e:
        click.echo(f"❌ Error creating .env file: {e}", err=True)
        sys.exit(1)


@config_cli.command()
@click.option('--env-file', '-e', help='Path to .env file to check')
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text', 
              help='Output format')
def show(env_file, format):
    """Show current configuration values."""
    try:
        config_manager = ConfigManager(env_file=env_file)
        
        if format == 'json':
            import json
            config = config_manager.get_all_config()
            # Remove sensitive values for JSON output
            safe_config = {}
            sensitive_keys = ['api_key', 'key', 'secret', 'password', 'token']
            
            for key, value in config.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    safe_config[key] = "***HIDDEN***"
                else:
                    safe_config[key] = value
            
            click.echo(json.dumps(safe_config, indent=2))
        else:
            config_manager.print_config_status()
            
    except Exception as e:
        click.echo(f"❌ Error showing configuration: {e}", err=True)
        sys.exit(1)


@config_cli.command()
@click.option('--env-file', '-e', help='Path to .env file to test')
def test(env_file):
    """Test configuration by attempting to load all components."""
    try:
        config_manager = ConfigManager(env_file=env_file)
        
        click.echo("🧪 Testing configuration components...")
        
        # Test LLM config
        try:
            llm_config = config_manager.get_llm_config()
            click.echo("✅ LLM configuration loaded successfully")
            click.echo(f"   Model: {llm_config['model']}")
            click.echo(f"   Base URL: {llm_config['base_url']}")
        except ConfigurationError as e:
            click.echo(f"❌ LLM configuration failed: {e}")
            return
        
        # Test framework config
        try:
            framework_config = config_manager.get_framework_config()
            click.echo("✅ Framework configuration loaded successfully")
            click.echo(f"   Workspace: {framework_config['workspace_path']}")
            click.echo(f"   Log Level: {framework_config['log_level']}")
        except ConfigurationError as e:
            click.echo(f"❌ Framework configuration failed: {e}")
            return
        
        # Test test config
        try:
            test_config = config_manager.get_test_config()
            click.echo("✅ Test configuration loaded successfully")
        except ConfigurationError as e:
            click.echo(f"⚠️  Test configuration warning: {e}")
        
        click.echo("\n🎉 All configuration tests passed!")
        
    except Exception as e:
        click.echo(f"❌ Configuration test failed: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    config_cli()