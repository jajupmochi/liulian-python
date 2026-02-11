"""Prompt bank for LLM-based time series models

Adapted from Time-LLM:
    Source: https://github.com/KimMeen/Time-LLM
    File: dataset/prompt_bank/*.txt and utils/tools.py (load_content)

Provides domain-specific text prompts that describe dataset characteristics
for LLM-based time series models like TimeLLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


# Default prompt bank directory
_PROMPT_BANK_DIR = Path(__file__).parent / 'prompt_bank'


# Built-in prompts for common datasets
_BUILTIN_PROMPTS = {
    'ETT': (
        'The Electricity Transformer Temperature (ETT) dataset records '
        'transformer oil temperatures and power load features at regular '
        'intervals. It contains 7 features including HUFL, HULL, MUFL, '
        'MULL, LUFL, LULL, and OT (oil temperature).'
    ),
    'Weather': (
        'Weather is recorded every 10 minutes for the 2020 whole year, '
        'which contains 21 meteorological indicators, such as air '
        'temperature, humidity, etc.'
    ),
    'ECL': (
        'The Electricity Consuming Load (ECL) dataset contains the '
        'electricity consumption recordings from 321 clients at 15-minute '
        'intervals from 2012 to 2014.'
    ),
    'Traffic': (
        'The Traffic dataset describes the road occupancy rates measured '
        'by 862 sensors on San Francisco Bay area freeways from 2015 to '
        '2016, recorded hourly.'
    ),
    'swiss_river': (
        'The Swiss River water temperature dataset contains daily '
        'measurements of water temperature and air temperature at '
        'multiple monitoring stations along Swiss rivers. The data is '
        'used for environmental monitoring and climate impact analysis '
        'on freshwater ecosystems.'
    ),
}


def load_content(
    dataset_name: str,
    prompt_dir: Optional[str] = None,
) -> str:
    """Load dataset description prompt for LLM-based models.

    Looks up the prompt in this order:
    1. Custom prompt file at prompt_dir/<dataset_name>.txt
    2. Built-in prompt bank at liulian/data/prompt_bank/<dataset_name>.txt
    3. Built-in dictionary (_BUILTIN_PROMPTS)
    4. Generic fallback

    Args:
        dataset_name: Name of the dataset (e.g., 'ETT', 'Weather', 'swiss_river')
        prompt_dir: Optional custom directory containing .txt prompt files

    Returns:
        String describing the dataset for LLM context
    """
    # Handle ETT variants
    canonical = dataset_name
    if 'ETT' in dataset_name:
        canonical = 'ETT'

    # 1. Try custom directory
    if prompt_dir is not None:
        custom_path = Path(prompt_dir) / f'{canonical}.txt'
        if custom_path.exists():
            return custom_path.read_text().strip()

    # 2. Try built-in prompt bank directory
    builtin_path = _PROMPT_BANK_DIR / f'{canonical}.txt'
    if builtin_path.exists():
        return builtin_path.read_text().strip()

    # 3. Try built-in dictionary
    if canonical in _BUILTIN_PROMPTS:
        return _BUILTIN_PROMPTS[canonical]

    # 4. Generic fallback
    return (
        f'This is a time series dataset named {dataset_name}. '
        f'It contains numerical measurements recorded at regular intervals.'
    )
