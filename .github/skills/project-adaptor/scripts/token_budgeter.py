#!/usr/bin/env python3
"""
Token Budgeter - Track and manage token/request usage

Implements 6 token-saving strategies:
1. Summary-First Approach
2. Diff-Only Transmission
3. Batch Related Changes
4. Local Template Generation
5. Prompt Memoization
6. Incremental Context Building
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class TokenUsage:
    """Track token usage for a step."""
    step_id: str
    prompt_tokens: int
    response_tokens: int
    total_tokens: int


class TokenBudgeter:
    """Manage token budget and implement token-saving strategies."""
    
    def __init__(self, budget: int = 50000, cache_dir: Optional[Path] = None):
        """
        Initialize token budgeter.
        
        Args:
            budget: Maximum tokens allowed for run
            cache_dir: Directory for prompt memoization cache
        """
        self.budget = budget
        self.used = 0
        self.usage_log: List[TokenUsage] = []
        self.cache_dir = cache_dir or Path(".cache")
        self.cache: Dict[str, Any] = {}
        
        # Load cache if exists
        if self.cache_dir.exists():
            cache_file = self.cache_dir / "prompt_cache.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        self.cache = json.load(f)
                except Exception:
                    self.cache = {}
    
    def track_usage(self, step_id: str, prompt_tokens: int, response_tokens: int) -> None:
        """Record token usage for a step."""
        total = prompt_tokens + response_tokens
        self.used += total
        
        usage = TokenUsage(
            step_id=step_id,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            total_tokens=total
        )
        self.usage_log.append(usage)
    
    def check_budget(self) -> tuple[bool, float]:
        """
        Check if within budget.
        
        Returns:
            Tuple of (within_budget, percentage_used)
        """
        percentage = (self.used / self.budget) * 100
        within_budget = self.used <= self.budget
        return within_budget, percentage
    
    def should_warn(self) -> bool:
        """Return True if usage is above 80% threshold."""
        _, percentage = self.check_budget()
        return percentage >= 80
    
    def format_usage_report(self, step_id: str) -> str:
        """Format token usage report for display."""
        # Find usage for this step
        step_usage = next((u for u in self.usage_log if u.step_id == step_id), None)
        
        if not step_usage:
            return ""
        
        _, percentage = self.check_budget()
        
        return f"""
Token Usage: {step_id}
  - Prompts: {step_usage.prompt_tokens:,} tokens
  - Responses: {step_usage.response_tokens:,} tokens
  - Total this step: {step_usage.total_tokens:,} tokens
  - Cumulative: {self.used:,} / {self.budget:,} ({percentage:.1f}%)
"""
    
    def generate_file_summary(self, file_path: Path, max_lines: int = 3) -> str:
        """
        Strategy 1: Summary-First Approach
        
        Generate concise summary instead of full file content.
        
        Args:
            file_path: Path to file
            max_lines: Maximum lines for summary
            
        Returns:
            Summary string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            
            # Extract key information
            imports = [l.strip() for l in lines if l.strip().startswith('import') or l.strip().startswith('from')]
            classes = [l.strip() for l in lines if l.strip().startswith('class ')]
            functions = [l.strip() for l in lines if l.strip().startswith('def ')]
            
            summary_parts = []
            
            if classes:
                class_names = [c.split('(')[0].replace('class ', '') for c in classes[:3]]
                summary_parts.append(f"Classes: {', '.join(class_names)}")
            
            if functions:
                func_names = [f.split('(')[0].replace('def ', '') for f in functions[:5]]
                summary_parts.append(f"Functions: {', '.join(func_names)}")
            
            if imports:
                key_imports = [i.split()[-1] for i in imports[:3]]
                summary_parts.append(f"Imports: {', '.join(key_imports)}")
            
            summary = f"File: {file_path.name} (~{total_lines} LOC)\n" + " | ".join(summary_parts)
            summary += "\nFull content available on request."
            
            return summary
            
        except Exception as e:
            return f"File: {file_path.name} (could not generate summary: {e})"
    
    def generate_diff_only(
        self,
        original_content: str,
        modified_content: str,
        context_lines: int = 3
    ) -> str:
        """
        Strategy 2: Diff-Only Transmission
        
        Generate unified diff instead of full file.
        
        Args:
            original_content: Original file content
            modified_content: Modified file content
            context_lines: Number of context lines around changes
            
        Returns:
            Unified diff string
        """
        import difflib
        
        original_lines = original_content.split('\n')
        modified_lines = modified_content.split('\n')
        
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            lineterm='',
            n=context_lines
        )
        
        return '\n'.join(diff)
    
    def cache_prompt_response(
        self,
        prompt: str,
        context_files: List[str],
        response: str
    ) -> None:
        """
        Strategy 5: Prompt Memoization
        
        Cache prompt/response pairs for reuse.
        
        Args:
            prompt: Prompt text
            context_files: List of context file paths (for cache key)
            response: Response to cache
        """
        # Generate cache key from prompt + context
        cache_key_input = f"{prompt}|{'|'.join(sorted(context_files))}"
        cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()
        
        self.cache[cache_key] = {
            'prompt': prompt,
            'context_files': context_files,
            'response': response
        }
        
        # Save cache to disk
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "prompt_cache.json"
        with open(cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def get_cached_response(
        self,
        prompt: str,
        context_files: List[str]
    ) -> Optional[str]:
        """
        Retrieve cached response if available.
        
        Args:
            prompt: Prompt text
            context_files: List of context file paths
            
        Returns:
            Cached response or None if not found
        """
        cache_key_input = f"{prompt}|{'|'.join(sorted(context_files))}"
        cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()
        
        if cache_key in self.cache:
            return self.cache[cache_key]['response']
        
        return None


class CopilotRequestBudgeter:
    """Manage Copilot premium request budget."""
    
    def __init__(self, budget: int = 300):
        """
        Initialize request budgeter.
        
        Args:
            budget: Maximum requests allowed for run
        """
        self.budget = budget
        self.used = 0
    
    def track_request(self) -> None:
        """Record a Copilot request."""
        self.used += 1
    
    def check_budget(self) -> tuple[bool, float]:
        """
        Check if within budget.
        
        Returns:
            Tuple of (within_budget, percentage_used)
        """
        percentage = (self.used / self.budget) * 100
        within_budget = self.used <= self.budget
        return within_budget, percentage
    
    def should_warn(self) -> bool:
        """Return True if usage is above 80% threshold."""
        _, percentage = self.check_budget()
        return percentage >= 80
    
    def format_usage_report(self) -> str:
        """Format request usage report."""
        _, percentage = self.check_budget()
        return f"Copilot Requests: {self.used} / {self.budget} ({percentage:.1f}%)"
