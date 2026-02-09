#!/usr/bin/env python3
"""
Project Discovery - Locate and validate reference and target projects

Handles:
- Git repository cloning
- Local path validation
- Project structure analysis
- Configuration confirmation
"""

import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ProjectInfo:
    """Information about a discovered project."""
    
    path: Path
    source: str  # "git", "local", or "cwd"
    original_location: Optional[str] = None
    timestamp: Optional[str] = None
    package_name: Optional[str] = None
    python_version: Optional[str] = None
    test_framework: Optional[str] = None


def clone_git_repository(git_url: str, base_dir: Path) -> Optional[Path]:
    """
    Clone a Git repository to refer_projects/ with timestamp.
    
    Args:
        git_url: Git repository URL
        base_dir: Base directory for cloning (e.g., refer_projects/)
        
    Returns:
        Path to cloned repository, or None if clone failed
    """
    # Extract repo name from URL
    repo_name = git_url.rstrip('/').split('/')[-1]
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]
    
    # Create timestamped directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clone_dir = base_dir / f"{repo_name}_{timestamp}"
    
    # Ensure base directory exists
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Clone repository
    try:
        result = subprocess.run(
            ['git', 'clone', git_url, str(clone_dir)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            return clone_dir
        else:
            print(f"❌ Clone failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"❌ Clone timeout exceeded (5 minutes)")
        return None
    except Exception as e:
        print(f"❌ Clone error: {e}")
        return None


def validate_local_path(path: str) -> Optional[Path]:
    """
    Validate that a local path exists and is accessible.
    
    Args:
        path: Path string (absolute or relative)
        
    Returns:
        Resolved absolute Path if valid, None otherwise
    """
    try:
        p = Path(path).resolve()
        if p.exists() and p.is_dir():
            return p
        else:
            print(f"❌ Path does not exist or is not a directory: {path}")
            return None
    except Exception as e:
        print(f"❌ Path validation error: {e}")
        return None


def analyze_project_structure(project_path: Path) -> ProjectInfo:
    """
    Analyze project directory to extract metadata.
    
    Detects:
    - Python package name (from pyproject.toml or setup.py)
    - Python version requirement
    - Test framework (pytest, unittest)
    - Project type
    
    Args:
        project_path: Path to project directory
        
    Returns:
        ProjectInfo with analyzed metadata
    """
    info = ProjectInfo(
        path=project_path,
        source="unknown"
    )
    
    # Check for pyproject.toml
    pyproject = project_path / "pyproject.toml"
    if pyproject.exists():
        try:
            with open(pyproject, 'r') as f:
                content = f.read()
                
                # Extract package name
                if 'name = "' in content:
                    start = content.find('name = "') + 8
                    end = content.find('"', start)
                    info.package_name = content[start:end]
                
                # Check for pytest
                if 'pytest' in content:
                    info.test_framework = "pytest"
                    
                # Extract Python version
                if 'requires-python' in content:
                    start = content.find('requires-python') + len('requires-python')
                    # Find the value (might be in quotes)
                    line_end = content.find('\n', start)
                    python_version_line = content[start:line_end]
                    if '"' in python_version_line or "'" in python_version_line:
                        quote = '"' if '"' in python_version_line else "'"
                        start_quote = python_version_line.find(quote) + 1
                        end_quote = python_version_line.find(quote, start_quote)
                        info.python_version = python_version_line[start_quote:end_quote]
                        
        except Exception as e:
            print(f"⚠️  Could not parse pyproject.toml: {e}")
    
    # Fallback: check for setup.py
    if not info.package_name:
        setup_py = project_path / "setup.py"
        if setup_py.exists():
            try:
                with open(setup_py, 'r') as f:
                    content = f.read()
                    if 'name=' in content or 'name =' in content:
                        # Try to extract package name (simple heuristic)
                        for line in content.split('\n'):
                            if 'name' in line and '=' in line:
                                parts = line.split('=')
                                if len(parts) == 2:
                                    name = parts[1].strip().strip(',').strip('"').strip("'")
                                    info.package_name = name
                                    break
            except Exception:
                pass
    
    # Check for test directories
    if not info.test_framework:
        if (project_path / "tests").exists():
            # Check if pytest.ini exists
            if (project_path / "pytest.ini").exists():
                info.test_framework = "pytest"
            # Check test files for imports
            else:
                test_files = list((project_path / "tests").glob("test_*.py"))
                if test_files:
                    try:
                        with open(test_files[0], 'r') as f:
                            content = f.read()
                            if 'import pytest' in content:
                                info.test_framework = "pytest"
                            elif 'import unittest' in content:
                                info.test_framework = "unittest"
                    except Exception:
                        pass
    
    return info


def scan_refer_projects_directory(base_dir: Path) -> List[ProjectInfo]:
    """
    Scan refer_projects/ directory for all subdirectories.
    
    Args:
        base_dir: Path to refer_projects/ directory
        
    Returns:
        List of ProjectInfo objects for each discovered project
    """
    if not base_dir.exists():
        return []
    
    projects = []
    for item in base_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            info = analyze_project_structure(item)
            info.source = "refer_projects"
            projects.append(info)
    
    return projects


def format_project_config_summary(
    reference_projects: List[ProjectInfo],
    target_project: ProjectInfo
) -> str:
    """
    Format project configuration summary for user confirmation.
    
    Args:
        reference_projects: List of reference project infos
        target_project: Target project info
        
    Returns:
        Formatted summary string
    """
    lines = ["", "PROJECT CONFIGURATION SUMMARY", "=" * 50, ""]
    
    lines.append("Reference Project(s):")
    for i, proj in enumerate(reference_projects, 1):
        lines.append(f"  C_r{i}: {proj.path}")
        if proj.original_location:
            lines.append(f"       Source: {proj.original_location}")
        if proj.timestamp:
            lines.append(f"       Timestamp: {proj.timestamp}")
        if proj.package_name:
            lines.append(f"       Package: {proj.package_name}")
        lines.append("")
    
    lines.append("Target Project:")
    lines.append(f"  P_t: {target_project.path}")
    if target_project.package_name:
        lines.append(f"       Package: {target_project.package_name}")
    if target_project.python_version:
        lines.append(f"       Python: {target_project.python_version}")
    if target_project.test_framework:
        lines.append(f"       Tests: {target_project.test_framework}")
    
    lines.append("")
    lines.append("Confirm this configuration? (yes/edit)")
    
    return "\n".join(lines)
