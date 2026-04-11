"""
EXPEDIA Tier 2 — Phase 2: Out-of-Core Genomic Deduplication
=============================================================
Three-stage pipeline that reduces 45.8M raw records to one representative
sequence per species, and then eliminates near-identical homologs — all
without loading the 250 GB dataset into RAM.

Stage 1  DuckDB metadata aggregation
    Reads data_report.jsonl out-of-core; groups by TaxID; emits a flat
    keep_accessions.txt list (one accession per unique TaxID).

Stage 2  SeqKit stream filtering
    Streams the 250 GB FASTA line-by-line; keeps only headers matching the
    accession list; writes deduplicated_marine.fasta.  RAM ≈ 0.

Stage 3  MMseqs2 linclust (optional but recommended)
    Clusters at 95% identity with linear O(N) complexity; eliminates
    functionally identical sub-species / environmental duplicates.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pipeline_config import (
    ACCESSION_KEEP_LIST,
    DATA_REPORT_JSONL,
    DEDUP_CHECKPOINT,
    DEDUP_FASTA_PATH,
    DIRS,
    DUCKDB_MEMORY_LIMIT,
    LOG_FILE,
    MMSEQS2_CLUSTER_DIR,
    MMSEQS2_COVERAGE,
    MMSEQS2_IDENTITY,
    MMSEQS2_REP_FASTA,
    RAW_FASTA_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("phase2.deduplication")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_cp() -> dict:
    return json.loads(DEDUP_CHECKPOINT.read_text()) if DEDUP_CHECKPOINT.exists() else {}

def _save_cp(state: dict) -> None:
    DEDUP_CHECKPOINT.write_text(json.dumps(state, indent=2))

def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    log.info("RUN: %s", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n"
            f"stderr: {result.stderr[:800]}"
        )
    return result


# ---------------------------------------------------------------------------
# Stage 1 — DuckDB: one representative accession per TaxID
# ---------------------------------------------------------------------------

def stage1_duckdb_representative_selection(
    jsonl_path: Path = DATA_REPORT_JSONL,
    output_list: Path = ACCESSION_KEEP_LIST,
    memory_limit: str = DUCKDB_MEMORY_LIMIT,
    force: bool = False,
) -> int:
    """
    Use DuckDB to read the NCBI metadata JSONL (can be many GB) and emit
    exactly one accession per TaxID using an out-of-core GROUP BY.

    DuckDB spills intermediate sort steps to disk automatically when memory
    is exhausted — the memory_limit cap prevents OOM on the edge workstation.

    Returns the number of distinct TaxIDs found.
    """
    cp = _load_cp()
    if not force and cp.get("stage1_done") and output_list.exists():
        n = int(cp.get("distinct_taxa", 0))
        log.info("Stage 1 already complete (%d taxa).  Skipping.", n)
        return n

    try:
        import duckdb
    except ImportError:
        raise EnvironmentError("Install duckdb: pip install duckdb --break-system-packages")

    log.info("Stage 1 — DuckDB representative selection from %s", jsonl_path)
    log.info("Memory cap: %s  (intermediate spill to disk if exceeded)", memory_limit)

    t0 = time.monotonic()

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET temp_directory = '{DIRS.dedup_fasta}'")

    # The NCBI data_report.jsonl schema includes accession and taxid fields.
    # read_json_auto handles nested / inconsistent JSON gracefully.
    sql = f"""
    COPY (
        SELECT
            taxId                           AS tax_id,
            FIRST(accession ORDER BY accession) AS representative_accession
        FROM read_json_auto('{jsonl_path}', format='newline_delimited')
        GROUP BY taxId
        ORDER BY tax_id
    )
    TO '{output_list}'
    (FORMAT CSV, HEADER FALSE, DELIMITER '\\n');
    """

    # Strip the CSV formatting — we only want plain accessions, one per line
    # (DuckDB WITH FORMAT CSV emits them unquoted for simple strings)
    con.execute(sql)

    # Count lines as a proxy for distinct TaxID count
    n_taxa = sum(1 for _ in output_list.open())
    elapsed = time.monotonic() - t0

    log.info(
        "Stage 1 complete: %d distinct taxa in %.1f s.  List: %s",
        n_taxa, elapsed, output_list
    )

    cp["stage1_done"]   = True
    cp["distinct_taxa"] = n_taxa
    cp["elapsed_s"]     = round(elapsed, 1)
    _save_cp(cp)
    return n_taxa


# ---------------------------------------------------------------------------
# Stage 2 — SeqKit: stream-filter the 250 GB FASTA
# ---------------------------------------------------------------------------

def stage2_seqkit_stream_filter(
    raw_fasta: Path = RAW_FASTA_PATH,
    keep_list: Path = ACCESSION_KEEP_LIST,
    output_fasta: Path = DEDUP_FASTA_PATH,
    force: bool = False,
) -> None:
    """
    SeqKit reads the FASTA byte-by-byte, discards any record whose header
    accession does not appear in keep_accessions.txt.  RAM usage ≈ 0.

    The --id-regexp extracts the accession from the NCBI header format:
        >NZ_CP012345.1 Organism name [strain=X]
    """
    cp = _load_cp()
    if not force and cp.get("stage2_done") and output_fasta.exists():
        log.info("Stage 2 already complete (%s).  Skipping.", output_fasta)
        return

    _require_tool("seqkit", "https://bioinf.shenwei.me/seqkit/")

    log.info("Stage 2 — SeqKit stream filter: %s → %s", raw_fasta, output_fasta)
    log.info("Keep list: %s", keep_list)

    t0 = time.monotonic()

    cmd = [
        "seqkit", "grep",
        "--pattern-file", str(keep_list),
        "--by-name",
        "--id-regexp",    r"^(\S+)",   # capture everything up to first whitespace
        "--out-file",     str(output_fasta),
        str(raw_fasta),
    ]

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        for line in proc.stdout:        # type: ignore[union-attr]
            line = line.rstrip()
            if line:
                log.info("[seqkit] %s", line)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"seqkit grep failed (exit {proc.returncode})")

    size_gb = output_fasta.stat().st_size / 1e9
    elapsed = time.monotonic() - t0
    log.info(
        "Stage 2 complete: %.1f GB deduplicated FASTA in %.1f s.",
        size_gb, elapsed
    )

    cp["stage2_done"]       = True
    cp["dedup_fasta_size_gb"] = round(size_gb, 2)
    cp["elapsed_s"]         = round(elapsed, 1)
    _save_cp(cp)


# ---------------------------------------------------------------------------
# Stage 3 — MMseqs2 linclust: eradicate centroid bias
# ---------------------------------------------------------------------------

def stage3_mmseqs2_homology_cluster(
    input_fasta: Path = DEDUP_FASTA_PATH,
    cluster_dir: Path = MMSEQS2_CLUSTER_DIR,
    rep_fasta: Path = MMSEQS2_REP_FASTA,
    identity: float = MMSEQS2_IDENTITY,
    coverage: float = MMSEQS2_COVERAGE,
    threads: int = 0,                  # 0 = auto-detect
    force: bool = False,
) -> Path:
    """
    MMseqs2 easy-linclust clusters the deduplicated FASTA at 95% sequence
    identity with O(N) time complexity.  This eliminates residual centroid
    bias from subspecies and unclassified environmental duplicates.

    Returns the path to the representative sequences FASTA.
    """
    cp = _load_cp()
    if not force and cp.get("stage3_done") and rep_fasta.exists():
        log.info("Stage 3 already complete (%s).  Skipping.", rep_fasta)
        return rep_fasta

    _require_tool("mmseqs", "https://github.com/soedinglab/MMseqs2")

    if threads == 0:
        import multiprocessing
        threads = multiprocessing.cpu_count()

    cluster_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = cluster_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    prefix = cluster_dir / "marine_cluster"
    log.info("Stage 3 — MMseqs2 linclust at %.0f%% identity, %d threads",
             identity * 100, threads)

    t0 = time.monotonic()

    cmd = [
        "mmseqs", "easy-linclust",
        str(input_fasta),
        str(prefix),
        str(tmp_dir),
        "--min-seq-id",    str(identity),
        "--cov-mode",      "1",
        "-c",              str(coverage),
        "--threads",       str(threads),
        "-v",              "2",
    ]
    _run(cmd)

    # easy-linclust emits {prefix}_rep_seq.fasta as the representative set
    generated = cluster_dir / "marine_cluster_rep_seq.fasta"
    if generated.exists():
        shutil.move(str(generated), str(rep_fasta))
    else:
        # fallback — find whatever *_rep_seq.fasta exists
        candidates = list(cluster_dir.glob("*_rep_seq.fasta"))
        if candidates:
            shutil.move(str(candidates[0]), str(rep_fasta))

    size_gb = rep_fasta.stat().st_size / 1e9
    elapsed = time.monotonic() - t0
    log.info(
        "Stage 3 complete: %.1f GB representative FASTA in %.1f s.",
        size_gb, elapsed
    )

    cp["stage3_done"]         = True
    cp["rep_fasta_size_gb"]   = round(size_gb, 2)
    cp["elapsed_s"]           = round(elapsed, 1)
    _save_cp(cp)
    return rep_fasta


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _require_tool(name: str, url: str) -> None:
    if shutil.which(name) is None:
        raise EnvironmentError(
            f"'{name}' not found on PATH.  Install from: {url}"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_deduplication(
    skip_mmseqs: bool = False,
    force: bool = False,
) -> Path:
    """
    Run all three deduplication stages.  Returns path to the final FASTA
    (MMseqs2 representative set if run, otherwise SeqKit-filtered set).

    Args:
        skip_mmseqs: Set True to skip Stage 3 (faster but less thorough).
        force: Re-run all stages even if checkpoints exist.
    """
    n_taxa = stage1_duckdb_representative_selection(force=force)
    log.info("Distinct taxa found: %d", n_taxa)

    stage2_seqkit_stream_filter(force=force)

    if skip_mmseqs:
        log.info("Stage 3 (MMseqs2) skipped by caller.")
        final_fasta = DEDUP_FASTA_PATH
    else:
        final_fasta = stage3_mmseqs2_homology_cluster(force=force)

    log.info("Phase 2 complete.  Final FASTA: %s", final_fasta)
    return final_fasta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mmseqs", action="store_true")
    ap.add_argument("--force",       action="store_true")
    args = ap.parse_args()
    run_deduplication(skip_mmseqs=args.skip_mmseqs, force=args.force)
