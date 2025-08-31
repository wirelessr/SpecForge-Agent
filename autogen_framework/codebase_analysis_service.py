"""
Codebase Analysis Service for the AutoGen multi-agent framework.

This module provides a service layer for integrating codebase-agent functionality
into the AutoGen framework, specifically for the DesignAgent to analyze existing
codebases before generating technical designs.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from codebase_agent.agents.manager import AgentManager
from codebase_agent.config.configuration import ConfigurationManager

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class CodebaseAnalysisService:
    """
    Service for analyzing existing codebases using codebase-agent.
    
    This service provides a clean interface for the DesignAgent to analyze
    existing codebases and incorporate the findings into design generation.
    """
    
    def __init__(self):
        """Initialize the codebase analysis service."""
        self.logger = logging.getLogger(__name__)
        self._config_manager = None
        self._agent_manager = None
        self._llm_client = None
    
    def _initialize_llm_client(self) -> bool:
        """Initialize the LLM client for analysis extraction."""
        try:
            if OpenAI is None:
                self.logger.warning("OpenAI not available, skipping LLM extraction")
                return False
            
            # Try to get LLM config from environment
            api_key = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
            base_url = os.getenv('LLM_BASE_URL')
            
            if not api_key:
                self.logger.warning("No LLM API key found, skipping LLM extraction")
                return False
            
            self._llm_client = OpenAI(
                api_key=api_key,
                base_url=base_url if base_url else None
            )
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM client: {e}")
            return False
    
    def is_codebase_present(self, directory_path: str) -> bool:
        """
        Check if the given directory contains a codebase worth analyzing.
        
        Args:
            directory_path: Path to the directory to check
            
        Returns:
            True if directory contains code files, False otherwise
        """
        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                return False
            
            # Look for common code file extensions
            code_extensions = {
                '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
                '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala',
                '.clj', '.ex', '.exs', '.r', '.m', '.mm', '.sql'
            }
            
            # Check for code files in the directory and subdirectories
            for file_path in directory.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in code_extensions:
                    # Skip common non-source directories
                    if any(part in file_path.parts for part in ['.git', '__pycache__', 'node_modules', '.venv', 'venv']):
                        continue
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking codebase presence: {e}")
            return False
    
    def _initialize_codebase_agent(self) -> bool:
        """
        Initialize the codebase agent components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Set up environment variables for codebase agent
            # Map our LLM config to what codebase agent expects
            if os.getenv('LLM_API_KEY') and not os.getenv('OPENAI_API_KEY'):
                os.environ['OPENAI_API_KEY'] = os.getenv('LLM_API_KEY')
            
            if os.getenv('LLM_BASE_URL') and not os.getenv('OPENAI_BASE_URL'):
                os.environ['OPENAI_BASE_URL'] = os.getenv('LLM_BASE_URL')
                
            if os.getenv('LLM_MODEL') and not os.getenv('OPENAI_MODEL'):
                os.environ['OPENAI_MODEL'] = os.getenv('LLM_MODEL')
            
            # Create configuration manager
            self._config_manager = ConfigurationManager()
            
            # Create and initialize agent manager
            self._agent_manager = AgentManager(self._config_manager)
            self._agent_manager.initialize_agents()
            
            self.logger.info("Codebase agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize codebase agent: {e}")
            return False
    
    async def _extract_design_insights(self, analysis_result: str) -> Dict[str, str]:
        """
        Use LLM to extract design-relevant insights from the codebase analysis.
        
        Args:
            analysis_result: Raw analysis result from codebase agent
            
        Returns:
            Dictionary with extracted insights
        """
        if not self._llm_client:
            if not self._initialize_llm_client():
                # Fallback to simple extraction
                return self._simple_fallback_extraction(analysis_result)
        
        try:
            extraction_prompt = f"""You are an expert software architect. Analyze the following codebase analysis report and extract key insights that would be valuable for designing new features or extensions.

Please extract and summarize the following aspects in a concise, actionable format:

1. **Architecture Summary**: Main architectural patterns and overall structure
2. **Key Technologies**: Primary technologies, frameworks, and libraries used
3. **Code Patterns**: Common design patterns, coding conventions, and styles
4. **Data Models**: Important data structures and their relationships
5. **API Design**: Existing API patterns and interface design approaches
6. **Integration Guidelines**: How new code should integrate with existing systems

Format your response as a JSON object with the following keys:
- "architecture_summary"
- "key_technologies" 
- "code_patterns"
- "data_models"
- "api_design"
- "integration_guidelines"

Keep each section concise (2-3 sentences max) and focused on actionable insights for design decisions.

Codebase Analysis Report:
{analysis_result}
"""

            response = self._llm_client.chat.completions.create(
                model=os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
                messages=[
                    {"role": "system", "content": "You are an expert software architect who extracts key design insights from codebase analyses."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse as JSON
            try:
                insights = json.loads(content)
                return insights
            except json.JSONDecodeError:
                # If JSON parsing fails, create structured response from text
                return {
                    "architecture_summary": self._extract_section(content, "architecture"),
                    "key_technologies": self._extract_section(content, "technolog"),
                    "code_patterns": self._extract_section(content, "pattern"),
                    "data_models": self._extract_section(content, "data"),
                    "api_design": self._extract_section(content, "api"),
                    "integration_guidelines": self._extract_section(content, "integration")
                }
            
        except Exception as e:
            self.logger.warning(f"LLM extraction failed: {e}, using fallback")
            return self._simple_fallback_extraction(analysis_result)
    
    def _extract_section(self, text: str, keyword: str) -> str:
        """Extract a section from text based on keyword."""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                # Get next few lines that contain content
                section_lines = []
                for j in range(i, min(i + 5, len(lines))):
                    if lines[j].strip():
                        section_lines.append(lines[j].strip())
                return ' '.join(section_lines) if section_lines else "No specific information found"
        return "No specific information found"
    
    def _simple_fallback_extraction(self, analysis_result: str) -> Dict[str, str]:
        """Simple text-based extraction as fallback."""
        return {
            "architecture_summary": f"Analysis completed with {len(analysis_result)} characters of insights",
            "key_technologies": "Technologies identified in codebase analysis",
            "code_patterns": "Code patterns and conventions documented in analysis",
            "data_models": "Data structures identified in codebase",
            "api_design": "API patterns documented in analysis",
            "integration_guidelines": "Follow existing patterns and maintain consistency with current codebase structure"
        }
    
    async def analyze_codebase(self, codebase_path: str, analysis_query: str = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Analyze the codebase at the given path.
        
        Args:
            codebase_path: Path to the codebase to analyze
            analysis_query: Optional specific query for analysis
            
        Returns:
            Tuple of (success, analysis_result)
            - success: Boolean indicating if analysis was successful
            - analysis_result: Dictionary containing analysis results or None if failed
        """
        try:
            # Check if codebase exists and has code files
            if not self.is_codebase_present(codebase_path):
                self.logger.info(f"No codebase found at {codebase_path}")
                return True, {"has_codebase": False, "message": "No existing codebase found"}
            
            # Initialize codebase agent if not already done
            if not self._agent_manager:
                if not self._initialize_codebase_agent():
                    return False, None
            
            # Default analysis query for design purposes
            if not analysis_query:
                analysis_query = """
                Analyze this codebase to understand:
                1. Overall architecture and design patterns
                2. Main components and their relationships
                3. Code style and conventions used
                4. Key technologies and frameworks
                5. Data models and structures
                6. API interfaces and contracts
                7. Testing approaches and patterns
                8. Configuration and deployment patterns
                
                Focus on providing insights that would help in designing new features
                or extensions that integrate well with the existing codebase.
                """
            
            # Execute the analysis - note: this is a blocking call that internally handles async
            self.logger.info(f"Starting codebase analysis for {codebase_path}")
            
            # Run the codebase agent in a thread to avoid event loop conflicts
            import asyncio
            import concurrent.futures
            
            def run_analysis():
                return self._agent_manager.process_query_with_review_cycle(
                    query=analysis_query,
                    codebase_path=codebase_path
                )
            
            # Use a thread pool to run the blocking operation
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result, stats = await loop.run_in_executor(executor, run_analysis)
            
            # Extract design insights using LLM
            insights = await self._extract_design_insights(result)
            
            # Format the results for design agent consumption
            analysis_result = {
                "has_codebase": True,
                "codebase_path": codebase_path,
                "analysis_result": result,
                "analysis_stats": stats,
                "insights": insights
            }
            
            self.logger.info("Codebase analysis completed successfully")
            return True, analysis_result
            
        except Exception as e:
            self.logger.error(f"Error during codebase analysis: {e}")
            return False, None
