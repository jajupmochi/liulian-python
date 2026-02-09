#!/usr/bin/env python3
"""
Conflict Detector - Multi-project conflict detection and analysis

Detects conflicts across multiple dimensions:
1. Naming conflicts (same concept, different names)
2. Architectural conflicts (incompatible patterns)
3. Dependency conflicts (version incompatibilities)
4. API signature conflicts (same name, different signatures)
5. Configuration conflicts (different formats)
6. Testing framework conflicts (different test libraries)
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class ConflictSeverity(Enum):
    """Conflict severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class CodeLocation:
    """Location of code in a file."""
    file_path: Path
    start_line: int
    end_line: int
    snippet: str


@dataclass
class Conflict:
    """Represents a detected conflict."""
    conflict_id: int
    severity: ConflictSeverity
    conflict_type: str
    description: str
    affected_projects: List[str]  # e.g., ["C_r1", "C_r2", "P_t"]
    affected_items: List[str]  # e.g., ["model", "dataset"]
    locations: Dict[str, CodeLocation]  # project_id -> location
    resolution_options: List[str]
    impact_assessment: str


def extract_class_definitions(file_path: Path) -> Dict[str, Tuple[int, int]]:
    """
    Extract class names and line ranges from Python file.
    
    Returns:
        Dictionary mapping class name to (start_line, end_line)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        classes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = (node.lineno, node.end_lineno or node.lineno)
        
        return classes
    except Exception as e:
        print(f"⚠️  Could not parse {file_path}: {e}")
        return {}


def extract_function_signatures(file_path: Path) -> Dict[str, str]:
    """
    Extract function names and their signatures.
    
    Returns:
        Dictionary mapping function name to signature string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        functions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = []
                for arg in node.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    args.append(arg_str)
                
                signature = f"{node.name}({', '.join(args)})"
                if node.returns:
                    signature += f" -> {ast.unparse(node.returns)}"
                
                functions[node.name] = signature
        
        return functions
    except Exception:
        return {}


def detect_naming_conflicts(
    projects: Dict[str, Path],
    items: List[str]
) -> List[Conflict]:
    """
    Detect naming conflicts across projects.
    
    Looks for common base class patterns:
    - BaseModel, ExecutableModel, Model
    - BaseTask, Task
    - BaseAdapter, Adapter
    
    Args:
        projects: Dict mapping project_id to project path
        items: List of adaptation items to check
        
    Returns:
        List of detected naming conflicts
    """
    conflicts = []
    conflict_id = 1
    
    # Scan for base model classes
    base_patterns = {
        'model': ['BaseModel', 'ExecutableModel', 'Model', 'AbstractModel'],
        'task': ['BaseTask', 'Task', 'AbstractTask'],
        'adapter': ['BaseAdapter', 'Adapter', 'DataAdapter'],
    }
    
    for item_type, class_patterns in base_patterns.items():
        # Check if this item type is being adapted
        if not any(item_type in item for item in items):
            continue
        
        found_classes = {}
        
        for proj_id, proj_path in projects.items():
            # Search for model/task/adapter files
            search_dirs = [
                proj_path / 'models',
                proj_path / 'tasks',
                proj_path / 'adapters',
                proj_path / 'data',
            ]
            
            for search_dir in search_dirs:
                if not search_dir.exists():
                    continue
                
                for py_file in search_dir.glob('**/*.py'):
                    classes = extract_class_definitions(py_file)
                    
                    for class_name in classes:
                        if class_name in class_patterns:
                            if class_name not in found_classes:
                                found_classes[class_name] = []
                            
                            found_classes[class_name].append((proj_id, py_file, classes[class_name]))
        
        # Check for conflicts (same concept, different names)
        if len(found_classes) > 1:
            # Multiple different class names found - potential conflict
            affected_projects = set()
            locations = {}
            
            for class_name, occurrences in found_classes.items():
                for proj_id, file_path, (start, end) in occurrences:
                    affected_projects.add(proj_id)
                    
                    # Read snippet
                    try:
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                            snippet = ''.join(lines[start-1:min(end, start+10)])
                    except Exception:
                        snippet = f"class {class_name}(...)"
                    
                    locations[f"{proj_id}_{file_path.name}"] = CodeLocation(
                        file_path=file_path,
                        start_line=start,
                        end_line=end,
                        snippet=snippet
                    )
            
            if len(affected_projects) > 1:
                conflict = Conflict(
                    conflict_id=conflict_id,
                    severity=ConflictSeverity.CRITICAL,
                    conflict_type=f"Naming Conflict - Base {item_type.title()} Class",
                    description=f"Multiple base class names found for {item_type}: {', '.join(found_classes.keys())}",
                    affected_projects=list(affected_projects),
                    affected_items=[item_type],
                    locations=locations,
                    resolution_options=[
                        f"A. Standardize to {list(found_classes.keys())[0]}",
                        f"B. Standardize to {list(found_classes.keys())[1]}" if len(found_classes) > 1 else "B. Create adapter pattern",
                        "C. Maintain separate implementations",
                        "D. Manual resolution"
                    ],
                    impact_assessment=f"Affects all {item_type} adaptation items"
                )
                conflicts.append(conflict)
                conflict_id += 1
    
    return conflicts


def detect_dependency_conflicts(projects: Dict[str, Path]) -> List[Conflict]:
    """
    Detect dependency version conflicts.
    
    Args:
        projects: Dict mapping project_id to project path
        
    Returns:
        List of detected dependency conflicts
    """
    conflicts = []
    conflict_id = 100  # Start at 100 to avoid collision with naming conflicts
    
    # Extract dependencies from each project
    dependencies = {}
    
    for proj_id, proj_path in projects.items():
        deps = {}
        
        # Check pyproject.toml
        pyproject = proj_path / "pyproject.toml"
        if pyproject.exists():
            try:
                with open(pyproject, 'r') as f:
                    content = f.read()
                    
                    # Simple extraction (not a full TOML parser)
                    in_dependencies = False
                    for line in content.split('\n'):
                        if 'dependencies' in line and '=' in line:
                            in_dependencies = True
                            continue
                        
                        if in_dependencies:
                            if line.strip().startswith('['):
                                in_dependencies = False
                                continue
                            
                            # Extract package and version
                            if '"' in line:
                                match = re.search(r'"([^"]+)"', line)
                                if match:
                                    dep_spec = match.group(1)
                                    # Parse package name and version constraint
                                    for op in ['>=', '<=', '==', '>', '<', '~=']:
                                        if op in dep_spec:
                                            pkg, ver = dep_spec.split(op, 1)
                                            deps[pkg.strip()] = (op, ver.strip())
                                            break
                                    else:
                                        # No version specified
                                        deps[dep_spec.strip()] = (None, None)
            except Exception as e:
                print(f"⚠️  Could not parse {pyproject}: {e}")
        
        dependencies[proj_id] = deps
    
    # Find conflicting dependencies
    all_packages = set()
    for deps in dependencies.values():
        all_packages.update(deps.keys())
    
    for package in all_packages:
        versions = {}
        for proj_id, deps in dependencies.items():
            if package in deps:
                versions[proj_id] = deps[package]
        
        if len(versions) > 1:
            # Check for actual conflict
            has_conflict = False
            version_strs = []
            
            for proj_id, (op, ver) in versions.items():
                if op and ver:
                    version_strs.append(f"{proj_id}: {package}{op}{ver}")
                    # Simple conflict detection (this would need more sophisticated logic)
                    if op == '==' and len(versions) > 1:
                        has_conflict = True
            
            if has_conflict and len(version_strs) > 1:
                conflict = Conflict(
                    conflict_id=conflict_id,
                    severity=ConflictSeverity.HIGH,
                    conflict_type="Dependency Conflict",
                    description=f"Incompatible version requirements for {package}",
                    affected_projects=list(versions.keys()),
                    affected_items=["dependencies"],
                    locations={},
                    resolution_options=[
                        "A. Use version range that satisfies all constraints",
                        "B. Upgrade to latest compatible version",
                        "C. Make dependency optional with version guards",
                        "D. Manual resolution"
                    ],
                    impact_assessment=f"May affect components that use {package}"
                )
                conflicts.append(conflict)
                conflict_id += 1
    
    return conflicts


def format_conflict_report(conflicts: List[Conflict]) -> str:
    """
    Format conflict detection report for display.
    
    Args:
        conflicts: List of detected conflicts
        
    Returns:
        Formatted report string
    """
    if not conflicts:
        return "\n✅ No conflicts detected - ready to proceed with adaptation.\n"
    
    lines = [
        "",
        "CONFLICT DETECTION REPORT",
        "=" * 70,
        f"\nFound {len(conflicts)} conflict(s) requiring resolution:\n"
    ]
    
    for conflict in conflicts:
        lines.append(f"\nCONFLICT {conflict.conflict_id}: [SEVERITY: {conflict.severity.value}]")
        lines.append(f"Type: {conflict.conflict_type}")
        lines.append(f"Affected Projects: {', '.join(conflict.affected_projects)}")
        if conflict.affected_items:
            lines.append(f"Affected Items: {', '.join(conflict.affected_items)}")
        lines.append(f"\nDescription: {conflict.description}")
        
        if conflict.locations:
            lines.append("\nDetails:")
            for proj_file, location in conflict.locations.items():
                lines.append(f"  {proj_file} ({location.file_path}:{location.start_line}-{location.end_line}):")
                # Show first few lines of snippet
                snippet_lines = location.snippet.split('\n')[:5]
                for sline in snippet_lines:
                    lines.append(f"    {sline}")
        
        lines.append("\nResolution Options:")
        for option in conflict.resolution_options:
            lines.append(f"  {option}")
        
        lines.append(f"\nRequired Action: User must select resolution option.\n")
        lines.append("-" * 70)
    
    # Summary table
    lines.append("\nCONFLICT SUMMARY TABLE:")
    lines.append(f"{'ID':<4} | {'Severity':<8} | {'Type':<25} | {'Projects':<15} | Resolution")
    lines.append("-" * 70)
    
    for conflict in conflicts:
        projects_str = ','.join(conflict.affected_projects)[:15]
        required = "REQUIRED" if conflict.severity in [ConflictSeverity.CRITICAL, ConflictSeverity.HIGH] else "RECOMMENDED"
        lines.append(f"{conflict.conflict_id:<4} | {conflict.severity.value:<8} | {conflict.conflict_type[:25]:<25} | {projects_str:<15} | {required}")
    
    return "\n".join(lines)
