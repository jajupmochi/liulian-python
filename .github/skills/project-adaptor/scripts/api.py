#!/usr/bin/env python3
"""
Project Adaptor - Main API Entry Point

Parses user invocations (slash commands or natural language) and orchestrates
the adaptation workflow.
"""

import re
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class AdaptationConfig:
    """Configuration for an adaptation run."""
    
    reference_projects: List[str] = field(default_factory=list)
    target_project: Optional[str] = None
    items: List[str] = field(default_factory=list)
    dry_run: bool = False
    mode: str = "minimal"  # or "comprehensive"
    batch_size: int = 3
    auto_test: bool = True
    create_branch: bool = True
    token_budget: int = 50000
    copilot_premium_request_budget: int = 300


def parse_slash_command(command: str) -> Optional[AdaptationConfig]:
    """
    Parse /adapt slash command into AdaptationConfig.
    
    Example:
        /adapt reference=https://github.com/org/repo.git target=. items=task,model
    
    Args:
        command: Slash command string
        
    Returns:
        AdaptationConfig if parsing succeeds, None otherwise
    """
    if not command.strip().startswith("/adapt"):
        return None
    
    config = AdaptationConfig()
    
    # Extract reference projects
    ref_match = re.search(r'reference=([^\s]+)', command)
    if ref_match:
        refs = ref_match.group(1).split(',')
        config.reference_projects = [r.strip() for r in refs]
    
    # Extract target project
    target_match = re.search(r'target=([^\s]+)', command)
    if target_match:
        config.target_project = target_match.group(1)
    
    # Extract items
    items_match = re.search(r'items=\[([^\]]+)\]', command)
    if items_match:
        items = items_match.group(1).split(',')
        config.items = [i.strip() for i in items]
    
    # Extract options if present
    options_match = re.search(r'options=\{([^\}]+)\}', command)
    if options_match:
        options_str = options_match.group(1)
        # Parse key:value pairs
        for opt in options_str.split(','):
            if ':' in opt:
                key, value = opt.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Set config values
                if key == 'dry_run':
                    config.dry_run = value.lower() == 'true'
                elif key == 'mode':
                    config.mode = value
                elif key == 'batch_size':
                    config.batch_size = int(value)
                elif key == 'auto_test':
                    config.auto_test = value.lower() == 'true'
                elif key == 'create_branch':
                    config.create_branch = value.lower() == 'true'
                elif key == 'token_budget':
                    config.token_budget = int(value)
                elif key == 'copilot_premium_request_budget':
                    config.copilot_premium_request_budget = int(value)
    
    return config


def parse_natural_language(request: str) -> Optional[AdaptationConfig]:
    """
    Parse natural language request into AdaptationConfig.
    
    Example:
        "Adapt the SwissRiver dataset from the reference project"
    
    Args:
        request: Natural language request string
        
    Returns:
        AdaptationConfig with detected parameters, or None if not an adapt request
    """
    # Check for adaptation keywords
    adapt_keywords = ['adapt', 'import', 'bring in', 'port', 'migrate']
    if not any(kw in request.lower() for kw in adapt_keywords):
        return None
    
    config = AdaptationConfig()
    
    # Try to extract component types
    component_patterns = {
        'dataset': r'dataset[:\s]+(\w+)',
        'model': r'model[:\s]+(\w+)',
        'task': r'task[:\s]+(\w+)',
        'experiment': r'experiment',
        'tests': r'test',
        'docs': r'doc',
    }
    
    for comp_type, pattern in component_patterns.items():
        match = re.search(pattern, request, re.IGNORECASE)
        if match:
            if match.groups():
                config.items.append(f"{comp_type}:{match.group(1)}")
            else:
                config.items.append(comp_type)
    
    # Try to extract reference project mentions
    ref_patterns = [
        r'from\s+([^\s]+(?:\.git|repository|repo|project))',
        r'reference\s+project[:\s]+([^\s]+)',
    ]
    
    for pattern in ref_patterns:
        match = re.search(pattern, request, re.IGNORECASE)
        if match:
            config.reference_projects.append(match.group(1))
    
    return config


def format_invocation_prompt() -> str:
    """Generate help text for invocation syntax."""
    return """
PROJECT ADAPTOR - INVOCATION HELP

You can invoke the project-adaptor skill using either:

1. Slash command syntax:
   /adapt reference=<url_or_path>[,<url2>...] target=<path> items=[item1,item2,...] options={key:value,...}

2. Natural language:
   "Adapt the SwissRiver dataset from the reference project into my current project"
   "Import the Informer model and tests from agent-lfd repository"

Parameters:
  reference     - Git URL(s) or local path(s) to reference project(s) (required)
  target        - Target project path (default: current directory)
  items         - Components to adapt: task, model, dataset:Name, tests, docs, etc.
  options       - Configuration object:
                  dry_run: bool (default: false)
                  mode: minimal|comprehensive (default: minimal)
                  batch_size: int (default: 3)
                  auto_test: bool (default: true)
                  create_branch: bool (default: true)
                  token_budget: int (default: 50000)
                  copilot_premium_request_budget: int (default: 300)

Examples:
  /adapt reference=https://github.com/org/agent-lfd.git target=. items=dataset:SwissRiver
  /adapt reference=/path/to/project1,/path/to/project2 items=all options={dry_run:true}
"""
