"""Tests for the upgrade-plan reference fetcher's ID extraction.

WHY THIS MATTERS: `extract_ids()` is the only place where a silent regex miss is
invisible. A dropped arXiv ID or DOI produces a `.bib` that is quietly incomplete,
which surfaces later as a MISSING CITATION in the paper -- exactly the failure this
tooling exists to prevent. The network-facing functions are deliberately not tested
here (they would need live HTTP); this covers the pure, deterministic parser.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MOD_PATH = ROOT / 'tools' / 'fetch_upgrade_plan_refs.py'


def _load():
    spec = importlib.util.spec_from_file_location('fetch_upgrade_plan_refs', MOD_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def mod():
    return _load()


def _write(tmp_path: Path, text: str) -> list[Path]:
    f = tmp_path / 'doc.md'
    f.write_text(text, encoding='utf-8')
    return [f]


def test_extracts_arxiv_from_abs_pdf_html_and_bare_form(mod, tmp_path):
    """All four arXiv spellings used across the plan docs must be found.

    Expected: {2302.04071, 2410.14630, 2310.06625, 1911.08731}. If any is missed the
    corresponding paper silently vanishes from refs.bib.
    """
    md = _write(
        tmp_path,
        'see [arXiv 2302.04071](https://arxiv.org/abs/2302.04071) and\n'
        'https://arxiv.org/pdf/2410.14630 and\n'
        '[iTransformer Table 4](https://arxiv.org/html/2310.06625v4) and\n'
        'plain arXiv:1911.08731 here.\n',
    )
    arxiv, _ = mod.extract_ids(md)
    assert arxiv == {'2302.04071', '2410.14630', '2310.06625', '1911.08731'}


def test_versioned_arxiv_id_is_normalized_without_version(mod, tmp_path):
    """arxiv.org/html/2310.06625v4 must yield 2310.06625, not 2310.06625v4.

    A version suffix would break the bibtex endpoint URL.
    """
    arxiv, _ = mod.extract_ids(_write(tmp_path, 'https://arxiv.org/html/2310.06625v4'))
    assert arxiv == {'2310.06625'}


def test_extracts_doi_in_both_bare_and_url_form_and_dedups(mod, tmp_path):
    """A markdown link contains the DOI twice (label + href); it must dedup to one."""
    md = _write(
        tmp_path,
        '| ✓ [10.5194/hess-23-5089-2019](https://doi.org/10.5194/hess-23-5089-2019) |\n',
    )
    _, dois = mod.extract_ids(md)
    assert dois == {'10.5194/hess-23-5089-2019'}


def test_doi_strips_trailing_markdown_and_punctuation(mod, tmp_path):
    """DOIs sit inside tables/parens/sentences; trailing ) . , must not be captured.

    A trailing character makes the DOI unresolvable at doi.org, so the entry would be
    reported as a failure rather than fetched.
    """
    md = _write(
        tmp_path,
        'a (https://doi.org/10.1029/2021WR031794), and b 10.1088/1748-9326/abd501.\n',
    )
    _, dois = mod.extract_ids(md)
    assert dois == {'10.1029/2021WR031794', '10.1088/1748-9326/abd501'}


def test_no_false_positives_on_prose_without_identifiers(mod, tmp_path):
    """Plain prose with version-like numbers must yield nothing.

    Guards against the regex widening and injecting junk IDs into refs.bib.
    """
    arxiv, dois = mod.extract_ids(_write(tmp_path, 'We ran 3 seeds, v1.2 of the code, C >= 137.\n'))
    assert arxiv == set()
    assert dois == set()


def test_scans_every_file_it_is_given(mod, tmp_path):
    """IDs must be unioned across all plan documents, not just the first."""
    a = tmp_path / 'a.md'
    a.write_text('https://arxiv.org/abs/2211.14730\n', encoding='utf-8')
    b = tmp_path / 'b.md'
    b.write_text('https://arxiv.org/abs/2310.01728\n', encoding='utf-8')
    arxiv, _ = mod.extract_ids([a, b])
    assert arxiv == {'2211.14730', '2310.01728'}
