#!/usr/bin/env python3
"""
Artifact Manager - Record and manage adaptation artifacts

Creates and maintains the artifacts directory structure:
  artifacts/adaptations/<run-id>/
    ├── plan.yaml
    ├── config.json
    ├── conflict_resolutions.yaml
    ├── changes/
    ├── tests/
    ├── commits/
    ├── report.json
    ├── report.md
    └── adaptation_summary.txt
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ChangeRecord:
    """Record of a single atomic change."""
    step_id: str
    timestamp: str
    file_path: str
    change_type: str  # "create", "modify", "delete"
    lines_added: int
    lines_removed: int
    patch_file: str
    test_results: Optional[Dict[str, Any]] = None
    user_confirmed: bool = False


@dataclass
class AdaptationRun:
    """Complete record of an adaptation run."""
    run_id: str
    timestamp: str
    target_project: str
    reference_projects: List[Dict[str, str]]
    items: List[str]
    config: Dict[str, Any]
    changes: List[ChangeRecord]
    conflicts_resolved: List[Dict[str, Any]]
    status: str  # "in_progress", "completed", "aborted"
    token_usage: int = 0
    copilot_requests: int = 0


def create_run_id() -> str:
    """Generate unique run ID with timestamp."""
    return f"adapt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def initialize_artifact_directory(workspace_path: Path, run_id: str) -> Path:
    """
    Create artifact directory structure for a new adaptation run.
    
    Args:
        workspace_path: Path to workspace root
        run_id: Unique run identifier
        
    Returns:
        Path to the run's artifact directory
    """
    artifact_dir = workspace_path / "artifacts" / "adaptations" / run_id
    
    # Create directory structure
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "changes").mkdir(exist_ok=True)
    (artifact_dir / "tests").mkdir(exist_ok=True)
    (artifact_dir / "commits").mkdir(exist_ok=True)
    
    return artifact_dir


def save_config(artifact_dir: Path, config: Dict[str, Any]) -> None:
    """Save run configuration to config.json."""
    config_file = artifact_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)


def save_plan(artifact_dir: Path, plan: Dict[str, Any]) -> None:
    """Save adaptation plan to plan.yaml."""
    plan_file = artifact_dir / "plan.yaml"
    with open(plan_file, 'w') as f:
        yaml.dump(plan, f, default_flow_style=False, sort_keys=False)


def save_conflict_resolutions(
    artifact_dir: Path,
    resolutions: List[Dict[str, Any]]
) -> None:
    """Save conflict resolutions to conflict_resolutions.yaml."""
    resolution_file = artifact_dir / "conflict_resolutions.yaml"
    with open(resolution_file, 'w') as f:
        yaml.dump(resolutions, f, default_flow_style=False, sort_keys=False)


def record_change(
    artifact_dir: Path,
    change: ChangeRecord
) -> None:
    """
    Record an atomic change.
    
    Creates a change record JSON file in changes/ directory.
    """
    change_file = artifact_dir / "changes" / f"{change.step_id}.json"
    with open(change_file, 'w') as f:
        json.dump(asdict(change), f, indent=2)


def record_test_results(
    artifact_dir: Path,
    step_id: str,
    test_output: str
) -> None:
    """Record test execution results."""
    test_file = artifact_dir / "tests" / f"{step_id}_results.txt"
    with open(test_file, 'w') as f:
        f.write(test_output)


def record_commit(
    artifact_dir: Path,
    step_id: str,
    commit_info: Dict[str, Any]
) -> None:
    """Record Git commit information."""
    commit_file = artifact_dir / "commits" / f"{step_id}.json"
    with open(commit_file, 'w') as f:
        json.dump(commit_info, f, indent=2)


def generate_final_report(
    artifact_dir: Path,
    run: AdaptationRun
) -> Tuple[str, str]:
    """
    Generate final adaptation report in both JSON and Markdown formats.
    
    Returns:
        Tuple of (markdown_content, json_content)
    """
    # Generate JSON report
    json_report = asdict(run)
    
    # Generate Markdown report
    md_lines = [
        "# Adaptation Report",
        "",
        f"**Run ID:** {run.run_id}",
        f"**Timestamp:** {run.timestamp}",
        f"**Status:** {run.status}",
        "",
        "## Configuration",
        "",
        f"**Target Project:** {run.target_project}",
        "",
        "**Reference Projects:**",
    ]
    
    for i, ref in enumerate(run.reference_projects, 1):
        md_lines.append(f"- C_r{i}: {ref.get('path', 'N/A')}")
        if 'source' in ref:
            md_lines.append(f"  - Source: {ref['source']}")
    
    md_lines.extend([
        "",
        "**Items Adapted:**",
    ])
    for item in run.items:
        md_lines.append(f"- {item}")
    
    # Changes summary
    md_lines.extend([
        "",
        f"## Changes Summary",
        "",
        f"**Total Changes:** {len(run.changes)}",
        ""
    ])
    
    files_created = sum(1 for c in run.changes if c.change_type == "create")
    files_modified = sum(1 for c in run.changes if c.change_type == "modify")
    total_lines_added = sum(c.lines_added for c in run.changes)
    total_lines_removed = sum(c.lines_removed for c in run.changes)
    
    md_lines.extend([
        f"- Files created: {files_created}",
        f"- Files modified: {files_modified}",
        f"- Total lines added: {total_lines_added}",
        f"- Total lines removed: {total_lines_removed}",
        ""
    ])
    
    # Conflicts resolved
    if run.conflicts_resolved:
        md_lines.extend([
            "## Conflicts Resolved",
            ""
        ])
        for conflict in run.conflicts_resolved:
            md_lines.append(f"- {conflict.get('description', 'Unknown conflict')}")
            md_lines.append(f"  - Resolution: {conflict.get('resolution', 'N/A')}")
        md_lines.append("")
    
    # Resource usage
    md_lines.extend([
        "## Resource Usage",
        "",
        f"- Token usage: {run.token_usage:,}",
        f"- Copilot requests: {run.copilot_requests}",
        ""
    ])
    
    # Suggested next steps
    md_lines.extend([
        "## Suggested Next Steps",
        "",
        "1. Review all changes in feature branch",
        "2. Run full test suite: `pytest`",
        "3. Update documentation to reflect new components",
        "4. Consider merging feature branch to main",
        ""
    ])
    
    md_content = "\n".join(md_lines)
    json_content = json.dumps(json_report, indent=2)
    
    # Save reports
    (artifact_dir / "report.md").write_text(md_content)
    (artifact_dir / "report.json").write_text(json_content)
    
    return md_content, json_content


def generate_adaptation_summary(run: AdaptationRun) -> str:
    """
    Generate one-line summary for quick review.
    
    Format:
      ADAPT[run_id]: N/M items OK | X commits | Y/Z tests | items adapted | conflicts
    """
    completed_items = len([c for c in run.changes if c.user_confirmed])
    total_items = len(run.items)
    total_commits = len([c for c in run.changes if c.change_type in ["create", "modify"]])
    
    # Count test results
    passed_tests = 0
    total_tests = 0
    for change in run.changes:
        if change.test_results:
            passed_tests += change.test_results.get('passed', 0)
            total_tests += change.test_results.get('total', 0)
    
    conflicts_remaining = len([c for c in run.conflicts_resolved if not c.get('resolved', False)])
    
    items_str = ','.join(run.items[:3])
    if len(run.items) > 3:
        items_str += f",+{len(run.items)-3}"
    
    summary = (
        f"ADAPT[{run.run_id}]: {completed_items}/{total_items} items OK | "
        f"{total_commits} commits | {passed_tests}/{total_tests} tests | "
        f"{items_str} | {conflicts_remaining} conflicts remaining"
    )
    
    return summary
