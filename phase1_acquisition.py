"""
EXPEDIA Tier 2 — Phase 1: Fault-Tolerant Data Acquisition
==========================================================
Replaces legacy Biopython Entrez efetch loops with the NCBI Datasets CLI
dehydrated-package protocol.  Three stages:

  1. Manifest ledger  — datasets download … --dehydrated  (~1-5 GB JSONL)
  2. Archive extract  — unzip ncbi_dataset.zip
  3. Atomic rehydrate — datasets rehydrate  (250 GB FASTA, MD5-checkpointed)

The CLI natively skips already-downloaded files on resume, so a plain
re-execution of the rehydrate command is all that is needed after any failure.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from pipeline_config import (
    DEHYDRATED_DIR,
    DEHYDRATED_ZIP,
    DIRS,
    LOG_FILE,
    NCBI_DATASETS_BIN,
    NCBI_EXPECTED_RECORDS,
    NCBI_QUERY_TERMS,
    RAW_FASTA_PATH,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("phase1.acquisition")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess with logging; raise on non-zero exit if check=True."""
    log.info("RUN: %s", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True, **kwargs)
    if result.stdout:
        log.debug("STDOUT: %s", result.stdout[:2000])
    if result.stderr:
        log.debug("STDERR: %s", result.stderr[:2000])
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n"
            f"stderr: {result.stderr[:500]}"
        )
    return result


def _check_cli() -> None:
    """Verify that the NCBI Datasets CLI is available and print its version."""
    try:
        r = _run([NCBI_DATASETS_BIN, "version"], check=False)
        log.info("NCBI Datasets CLI: %s", r.stdout.strip())
    except FileNotFoundError:
        raise EnvironmentError(
            "NCBI Datasets CLI not found.  Install from:\n"
            "  https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/\n"
            "and ensure it is on PATH or set NCBI_DATASETS_BIN env-var."
        )


def _build_ncbi_query() -> str:
    """
    Combine multi-field search terms into a single NCBI nucleotide query string.
    Produces:  Aquatic[filter] OR Marine[filter] OR … OR Planktonic[filter]
    """
    terms = [f"{t}[filter]" for t in NCBI_QUERY_TERMS]
    return " OR ".join(terms)


def _checkpoint_path() -> Path:
    return DIRS.checkpoints / "phase1_acquisition.json"


def _load_checkpoint() -> dict:
    p = _checkpoint_path()
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_checkpoint(state: dict) -> None:
    _checkpoint_path().write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Stage 1 — Dehydrated manifest download
# ---------------------------------------------------------------------------

def stage1_download_manifest(
    query: str,
    output_zip: Path = DEHYDRATED_ZIP,
    force: bool = False,
) -> None:
    """
    Download a dehydrated package: lightweight JSONL metadata + cryptographic
    manifest for every matching sequence.  Typically completes in minutes.

    The --dehydrated flag means ZERO raw FASTA is transferred here — only the
    structural metadata needed to plan Stage 3 rehydration.
    """
    cp = _load_checkpoint()
    if not force and cp.get("stage1_done") and output_zip.exists():
        log.info("Stage 1 already complete (%s).  Skipping.", output_zip)
        return

    log.info("Stage 1 — downloading manifest (dehydrated) for query: %s", query)
    log.info("Expected records: ~%s", f"{NCBI_EXPECTED_RECORDS:,}")

    cmd = [
        NCBI_DATASETS_BIN, "download", "nucleotide",
        "--query",     query,
        "--filename",  str(output_zip),
        "--dehydrated",
    ]

    t0 = time.monotonic()
    _run(cmd)
    elapsed = time.monotonic() - t0
    log.info("Stage 1 complete in %.1f s.  Zip: %s (%.1f MB)",
             elapsed, output_zip, output_zip.stat().st_size / 1e6)

    cp["stage1_done"] = True
    cp["manifest_zip"] = str(output_zip)
    cp["elapsed_s"] = round(elapsed, 1)
    _save_checkpoint(cp)


# ---------------------------------------------------------------------------
# Stage 2 — Archive extraction
# ---------------------------------------------------------------------------

def stage2_extract_archive(
    zip_path: Path = DEHYDRATED_ZIP,
    extract_dir: Path = DEHYDRATED_DIR,
    force: bool = False,
) -> None:
    """
    Extract the dehydrated ZIP to EXPEDIA_Data/01_raw_fasta/ncbi_dataset/.
    Creates the directory structure and the data_report.jsonl metadata ledger
    that Phase 2 will consume for TaxID mapping.
    """
    cp = _load_checkpoint()
    if not force and cp.get("stage2_done") and extract_dir.exists():
        log.info("Stage 2 already complete (%s).  Skipping.", extract_dir)
        return

    log.info("Stage 2 — extracting %s → %s", zip_path, extract_dir)

    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    log.info("Stage 2 complete.  Directory: %s", extract_dir)

    cp["stage2_done"] = True
    cp["extract_dir"] = str(extract_dir)
    _save_checkpoint(cp)


# ---------------------------------------------------------------------------
# Stage 3 — Atomic rehydration (the 250 GB FASTA download)
# ---------------------------------------------------------------------------

def stage3_rehydrate(
    extract_dir: Path = DEHYDRATED_DIR,
    max_workers: int = 4,
    force: bool = False,
) -> None:
    """
    Execute 'datasets rehydrate' against the extracted directory.

    Key behaviours of the CLI that make this bullet-proof:
      • MD5 checksum comparison — files that passed already are SKIPPED.
      • Interrupted byte-streams are RESUMED at the exact byte offset.
      • No custom Python error-handling needed — the CLI handles all of this.

    Simply re-run this function after any failure; the CLI picks up exactly
    where it left off.
    """
    cp = _load_checkpoint()
    if not force and cp.get("stage3_done"):
        log.info("Stage 3 already complete.  Skipping rehydration.")
        return

    log.info("Stage 3 — rehydrating sequences (this will take many hours).")
    log.info("The process is fully restartable: re-run if interrupted.")

    cmd = [
        NCBI_DATASETS_BIN, "rehydrate",
        "--directory",   str(extract_dir),
        "--max-workers", str(max_workers),
    ]

    # Stream output in real time so progress is visible in logs
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        for line in proc.stdout:          # type: ignore[union-attr]
            line = line.rstrip()
            if line:
                log.info("[rehydrate] %s", line)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"datasets rehydrate exited with code {proc.returncode}.\n"
                "Re-run stage3_rehydrate() to resume from last completed file."
            )

    log.info("Stage 3 complete — all FASTA files rehydrated.")

    # Concatenate all per-accession FASTA files into one combined file
    _concatenate_fasta(extract_dir, RAW_FASTA_PATH)

    cp["stage3_done"] = True
    cp["raw_fasta"] = str(RAW_FASTA_PATH)
    _save_checkpoint(cp)


def _concatenate_fasta(extract_dir: Path, output_path: Path) -> None:
    """
    Walk the rehydrated directory tree and concatenate all *.fasta / *.fna
    files into a single stream.  Uses chunked I/O to avoid loading into RAM.
    """
    log.info("Concatenating FASTA files → %s", output_path)
    fasta_files = sorted(extract_dir.rglob("*.fasta")) + sorted(extract_dir.rglob("*.fna"))
    log.info("Found %d FASTA files to merge.", len(fasta_files))

    written = 0
    with output_path.open("wb") as out:
        for fasta in fasta_files:
            with fasta.open("rb") as inp:
                shutil.copyfileobj(inp, out, length=1 << 20)   # 1 MB chunks
            written += 1
            if written % 1000 == 0:
                log.info("  merged %d / %d files …", written, len(fasta_files))

    size_gb = output_path.stat().st_size / 1e9
    log.info("Concatenation complete: %s (%.1f GB)", output_path, size_gb)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_acquisition(
    force: bool = False,
    max_rehydrate_workers: int = 4,
) -> Path:
    """
    Execute all three acquisition stages in sequence.  Safe to call multiple
    times — each stage checks its checkpoint and skips if already done.

    Returns the path to the combined raw FASTA file.
    """
    _check_cli()
    query = _build_ncbi_query()

    stage1_download_manifest(query, force=force)
    stage2_extract_archive(force=force)
    stage3_rehydrate(max_workers=max_rehydrate_workers, force=force)

    log.info("Phase 1 complete.  Raw FASTA: %s", RAW_FASTA_PATH)
    return RAW_FASTA_PATH


if __name__ == "__main__":
    run_acquisition()
