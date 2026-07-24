#!/usr/bin/env python3
"""Fetch BibTeX + open-access PDFs for every reference in the 2026-07-16 upgrade plan.

Extracts arXiv IDs and DOIs from docs/research/2026-07-16-upgrade-plan/*.md, then:
  * BibTeX  -- arXiv via https://arxiv.org/bibtex/<id>; DOI via Crossref content
               negotiation (Accept: application/x-bibtex). NEVER hand-written.
  * PDFs    -- arXiv always (open access); DOIs only for known open-access hosts
               (Copernicus HESS/ESSD, Nature Scientific Data, MDPI, AGU-open).
               Paywalled publishers are skipped and reported, not faked.

Run:  python tools/fetch_upgrade_plan_refs.py [--no-pdf]
Out:  docs/research/2026-07-16-upgrade-plan/refs/refs.bib
      docs/research/2026-07-16-upgrade-plan/refs/pdf/*.pdf   (gitignored)
      docs/research/2026-07-16-upgrade-plan/refs/FETCH_REPORT.md
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'docs/research/2026-07-16-upgrade-plan'
OUT = PLAN / 'refs'
PDF_DIR = OUT / 'pdf'
UA = 'liulian-refs/1.0 (mailto:jajupmochi@gmail.com)'

# DOI prefixes / hosts that are reliably open access -> safe to fetch a PDF.
OA_DOI_PREFIXES = (
    '10.5194/',   # Copernicus (HESS, ESSD)
    '10.1038/s41597',  # Nature Scientific Data
    '10.3390/',   # MDPI
    '10.5281/',   # Zenodo
)


def get(url: str, accept: str | None = None, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    if accept:
        req.add_header('Accept', accept)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def extract_ids(md_files: list[Path]) -> tuple[set[str], set[str]]:
    """Return (arxiv_ids, dois) found across the docs."""
    arxiv: set[str] = set()
    dois: set[str] = set()
    # arXiv: 4 digits . 4-5 digits, optional version
    ax_re = re.compile(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})|arXiv:(\d{4}\.\d{4,5})', re.I)
    # DOI: doi.org/<doi> or a bare 10.xxxx/... inside a link or backticks.
    # ')' is ALLOWED in the class because legacy Elsevier DOIs embed balanced parens
    # (e.g. 10.1016/0022-1694(70)90255-6, Nash & Sutcliffe). Unbalanced trailing ')'
    # -- the markdown/prose closer -- is stripped afterwards by _clean_doi().
    doi_re = re.compile(r'(?:doi\.org/|\b)(10\.\d{4,9}/[^\s\]<>"\'`,]+)', re.I)
    for f in md_files:
        text = f.read_text(encoding='utf-8')
        for m in ax_re.finditer(text):
            arxiv.add((m.group(1) or m.group(2)))
        for m in doi_re.finditer(text):
            dois.add(_clean_doi(m.group(1)))
    return arxiv, dois


def _clean_doi(d: str) -> str:
    """Trim prose/markdown trailing characters WITHOUT truncating a valid DOI.

    Legacy Elsevier DOIs contain balanced parentheses, so a blanket ')' strip
    truncates them (this really happened: 10.1016/0022-1694(70)90255-6 became
    '10.1016/0022-1694(70' and 404'd). Only drop a trailing ')' when it is
    UNBALANCED, i.e. it closes something that opened outside the DOI.
    """
    # '*' is markdown emphasis, never part of a DOI; strip it alongside prose punctuation
    # so that e.g. '...WR031794)).**' reduces cleanly rather than 404-ing.
    trail = '.,;*'
    d = d.rstrip(trail)
    while d.endswith(')') and d.count(')') > d.count('('):
        d = d[:-1].rstrip(trail)
    return d


def fetch_arxiv(aid: str) -> tuple[str | None, str]:
    try:
        bib = get(f'https://arxiv.org/bibtex/{aid}').decode('utf-8', 'replace').strip()
        if bib.startswith('@'):
            return bib, 'ok'
        return None, f'unexpected body: {bib[:60]!r}'
    except urllib.error.HTTPError as e:
        return None, f'HTTP {e.code}'
    except Exception as e:  # noqa: BLE001 - report, never silently swallow
        return None, f'{type(e).__name__}: {e}'


def fetch_doi(doi: str) -> tuple[str | None, str]:
    try:
        bib = get(f'https://doi.org/{doi}', accept='application/x-bibtex').decode('utf-8', 'replace').strip()
        if bib.startswith('@'):
            return bib, 'ok'
        return None, f'unexpected body: {bib[:60]!r}'
    except urllib.error.HTTPError as e:
        return None, f'HTTP {e.code}'
    except Exception as e:  # noqa: BLE001
        return None, f'{type(e).__name__}: {e}'


def download_pdf(url: str, dest: Path) -> tuple[bool, str]:
    if dest.exists() and dest.stat().st_size > 10_000:
        return True, 'cached'
    try:
        data = get(url, timeout=60)
        if not data[:5].startswith(b'%PDF'):
            return False, 'not a PDF (likely a landing/paywall page)'
        dest.write_bytes(data)
        return True, f'{len(data)//1024} KB'
    except urllib.error.HTTPError as e:
        return False, f'HTTP {e.code}'
    except Exception as e:  # noqa: BLE001
        return False, f'{type(e).__name__}: {e}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-pdf', action='store_true', help='BibTeX only, skip PDF downloads')
    args = ap.parse_args()

    md = sorted(PLAN.glob('*.md'))
    if not md:
        print(f'no markdown found under {PLAN}', file=sys.stderr)
        return 1
    arxiv, dois = extract_ids(md)
    print(f'scanned {len(md)} docs -> {len(arxiv)} arXiv IDs, {len(dois)} DOIs')

    OUT.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    entries: list[str] = []
    rows: list[tuple[str, str, str, str]] = []  # kind, id, bibtex-status, pdf-status

    for aid in sorted(arxiv):
        bib, st = fetch_arxiv(aid)
        pdf_st = 'skipped'
        if bib:
            entries.append(bib)
            if not args.no_pdf:
                ok, pdf_st = download_pdf(f'https://arxiv.org/pdf/{aid}', PDF_DIR / f'arxiv-{aid}.pdf')
                pdf_st = ('OK ' + pdf_st) if ok else ('FAIL ' + pdf_st)
        rows.append(('arXiv', aid, st, pdf_st))
        print(f'  arXiv {aid}: bib={st} pdf={pdf_st}')
        time.sleep(0.4)  # be polite to arxiv.org

    for doi in sorted(dois):
        bib, st = fetch_doi(doi)
        pdf_st = 'skipped (not open-access host)'
        if bib:
            entries.append(bib)
            if not args.no_pdf and doi.startswith(OA_DOI_PREFIXES):
                safe = re.sub(r'[^A-Za-z0-9._-]', '_', doi)
                ok, msg = download_pdf(f'https://doi.org/{doi}', PDF_DIR / f'doi-{safe}.pdf')
                pdf_st = ('OK ' + msg) if ok else ('FAIL ' + msg)
        rows.append(('DOI', doi, st, pdf_st))
        print(f'  DOI {doi}: bib={st} pdf={pdf_st}')
        time.sleep(0.4)  # be polite to doi.org / Crossref

    bib_path = OUT / 'refs.bib'
    header = (
        '% Auto-generated by tools/fetch_upgrade_plan_refs.py -- DO NOT hand-edit.\n'
        '% Every entry was fetched programmatically (arXiv bibtex endpoint / Crossref\n'
        '% content negotiation). Regenerate with:  python tools/fetch_upgrade_plan_refs.py\n\n'
    )
    bib_path.write_text(header + '\n\n'.join(entries) + '\n', encoding='utf-8')

    n_bib_ok = sum(1 for r in rows if r[2] == 'ok')
    n_pdf_ok = sum(1 for r in rows if r[3].startswith('OK'))
    report = [
        '# Reference fetch report',
        '',
        f'Generated by `tools/fetch_upgrade_plan_refs.py` from {len(md)} plan documents.',
        '',
        f'- **BibTeX entries written**: {n_bib_ok} / {len(rows)} -> `refs.bib`',
        f'- **PDFs downloaded**: {n_pdf_ok} (arXiv + open-access DOIs only; `pdf/`, gitignored)',
        '',
        'Paywalled publishers (Wiley/AGU, Elsevier, Springer, IEEE, ACM, AAAI) are **not**',
        'downloaded -- only their BibTeX. Fetch those through the university proxy if needed.',
        '',
        '| Kind | ID | BibTeX | PDF |',
        '|---|---|---|---|',
    ]
    for kind, ident, bst, pst in rows:
        report.append(f'| {kind} | `{ident}` | {bst} | {pst} |')
    (OUT / 'FETCH_REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')

    print(f'\nwrote {bib_path} ({n_bib_ok} entries), PDFs={n_pdf_ok}, report={OUT/"FETCH_REPORT.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
