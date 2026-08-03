"""Tests for the hydro-LLM matrix runner's cell enumeration guardrails.

The Time-LLM matrix sweeps modes x datasets. entity_description needs authored
per-station text; only swiss-river-1990 (and ETTh1) has it today. build_cells
must SKIP (dataset, entity_description) for datasets without descriptions rather
than schedule a cell that only raises at run time — otherwise a 3-dataset sweep
looks like it will produce 9 results when 2 of them are guaranteed failures.
"""

from __future__ import annotations

from types import SimpleNamespace

from experiments.hydro_llm.run_matrix import build_cells
from liulian.pipeline import has_entity_descriptions


def _args(**over):
    base = dict(
        datasets=['swiss-river-1990', 'swiss-river-2010', 'swiss-river-zurich'],
        modes=['none', 'entity_description', 'numeric_embedding'],
        a2=['learnable'],
        a1=['default'],
        tuning=['frozen'],
        backbones=['GPT2'],
        seeds=[2026],
        arch='timellm',
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestEntityDescriptionAvailability:
    def test_only_1990_and_etth1_have_descriptions(self) -> None:
        assert has_entity_descriptions('swiss-river-1990') is True
        assert has_entity_descriptions('ETTh1') is True
        assert has_entity_descriptions('swiss-river-2010') is False
        assert has_entity_descriptions('swiss-river-zurich') is False
        assert has_entity_descriptions('does-not-exist') is False


class TestBuildCellsGuardrail:
    def test_entity_description_skipped_where_unavailable(self) -> None:
        cells = build_cells(_args())
        ed = [c for c in cells if c['mode'] == 'entity_description']
        # only swiss-river-1990 keeps its entity_description cell
        assert {c['dataset'] for c in ed} == {'swiss-river-1990'}

    def test_other_modes_run_on_all_datasets(self) -> None:
        cells = build_cells(_args())
        for mode in ('none', 'numeric_embedding'):
            got = {c['dataset'] for c in cells if c['mode'] == mode}
            assert got == {'swiss-river-1990', 'swiss-river-2010', 'swiss-river-zurich'}

    def test_total_cell_count_is_seven_not_nine(self) -> None:
        # 1990: none+entity_description+numeric_embedding (3);
        # 2010/zurich: none+numeric_embedding (2 each) => 3 + 2 + 2 = 7.
        assert len(build_cells(_args())) == 7

    def test_no_skip_when_only_1990_requested(self) -> None:
        cells = build_cells(_args(datasets=['swiss-river-1990']))
        assert {c['mode'] for c in cells} == {'none', 'entity_description', 'numeric_embedding'}
        assert len(cells) == 3
