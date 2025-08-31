---
applyTo: '**'
---

# Codebase Agent Quick Reference

Detailed documentation can be found at ./.venv/lib/python3.11/site-packages/codebase_agent

## Initialization and Usage

```python
from codebase_agent.agents.manager import AgentManager
from codebase_agent.config.configuration import ConfigurationManager

# 1. Create configuration manager
config_manager = ConfigurationManager()

# 2. Create and initialize agent manager
agent_manager = AgentManager(config_manager)
agent_manager.initialize_agents()

# 3. Execute code review
result, stats = agent_manager.process_query_with_review_cycle(
    query="Your analysis requirements",
    codebase_path="/path/to/project"
)
```

## Return Format

### result (Analysis Results)
Detailed code analysis report including architecture description and recommendations

### stats (Statistics)
```python
{
    "total_review_cycles": 2,           # Number of review cycles
    "rejections": 1,                    # Number of rejections  
    "final_acceptance_type": "accepted", # "accepted" or "forced"
    "final_confidence": 0.85            # Confidence score (0-1)
}
```

