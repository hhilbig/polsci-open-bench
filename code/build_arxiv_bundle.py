#!/usr/bin/env python3
"""Build a minimal arXiv source bundle for the PDF report."""

from __future__ import annotations

import hashlib
import re
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
SOURCE_TEX = OUTPUT / "report_pdf.tex"
FIGURE_DIR = OUTPUT / "figures"
ARXIV_DIR = OUTPUT / "arxiv"
SOURCE_DIR = ARXIV_DIR / "source"
TARBALL = ARXIV_DIR / "polsci-open-bench-arxiv-source.tar.gz"
MANIFEST = ARXIV_DIR / "manifest.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def referenced_figures(tex: str) -> list[Path]:
    refs = re.findall(r"\{(figures/[^{}]+)\}", tex)
    return sorted({Path(ref) for ref in refs})


def main() -> None:
    if not SOURCE_TEX.exists():
        raise SystemExit(
            "Missing output/report_pdf.tex. Run: quarto render output/report_pdf.qmd --to pdf"
        )
    if not FIGURE_DIR.exists():
        raise SystemExit("Missing output/figures. Run: Rscript code/build_report_assets.R")

    tex = SOURCE_TEX.read_text(encoding="utf-8")
    figures = referenced_figures(tex)
    missing = [str(path) for path in figures if not (OUTPUT / path).exists()]
    if missing:
        raise SystemExit("Missing referenced figure files:\n" + "\n".join(missing))

    if SOURCE_DIR.exists():
        shutil.rmtree(SOURCE_DIR)
    SOURCE_DIR.mkdir(parents=True)
    (SOURCE_DIR / "figures").mkdir()

    (SOURCE_DIR / "main.tex").write_text(tex, encoding="utf-8")
    copied = [SOURCE_DIR / "main.tex"]
    for figure in figures:
        target = SOURCE_DIR / figure
        shutil.copy2(OUTPUT / figure, target)
        copied.append(target)

    if TARBALL.exists():
        TARBALL.unlink()
    with tarfile.open(TARBALL, "w:gz") as archive:
        for path in copied:
            archive.add(path, arcname=path.relative_to(SOURCE_DIR))

    manifest_lines = [
        "arXiv source bundle for polsci-open-bench",
        f"tarball: {TARBALL.relative_to(ROOT)}",
        f"sha256: {sha256(TARBALL)}",
        "",
        "files:",
    ]
    manifest_lines.extend(
        f"- {path.relative_to(SOURCE_DIR)} ({sha256(path)})" for path in copied
    )
    MANIFEST.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    print(f"Wrote {TARBALL.relative_to(ROOT)}")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")
    print(f"Files in source bundle: {len(copied)}")


if __name__ == "__main__":
    main()
