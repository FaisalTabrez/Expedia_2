"""
EXPEDIA Tier 2 — Phase 3: High-Throughput Offline Taxonomy Reconstruction
==========================================================================
Reconstructs the full 7-rank evolutionary lineage for every distinct sequence
using TaxonKit operating entirely offline against the local NCBI taxonomy dump.

Pipeline:
  1. Extract TaxIDs from the representative FASTA headers (or from the
     DuckDB-generated metadata table if headers lack TaxIDs).
  2. Chunk TaxIDs into files of TAXON_CHUNK_SIZE (default 50,000).
  3. Dispatch each chunk to a subprocess running 'taxonkit reformat' via
     ProcessPoolExecutor — saturating all CPU cores.
  4. Merge results; repair merged/deleted nodes via taxid-changelog.
  5. Write lineages_full.tsv:
       accession  taxid  superkingdom  phylum  class  order  family  genus  species

TaxonKit performance: ~2-10 s for the entire taxonomy tree — chunked parallel
dispatch reduces this to minutes even for tens of millions of TaxIDs.
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator

from pipeline_config import (
    DATA_REPORT_JSONL,
    DIRS,
    LOG_FILE,
    MMSEQS2_REP_FASTA,
    TAXON_CHUNK_SIZE,
    TAXONKIT_BIN,
    TAXONKIT_DATA_DIR,
    TAXONOMY_CHECKPOINT,
    TAXONOMY_RANKS,
    TAXONOMY_TSV,
    DEDUP_FASTA_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("phase3.taxonomy")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_cp() -> dict:
    return json.loads(TAXONOMY_CHECKPOINT.read_text()) if TAXONOMY_CHECKPOINT.exists() else {}

def _save_cp(state: dict) -> None:
    TAXONOMY_CHECKPOINT.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Step 1 — Extract TaxIDs from FASTA headers or metadata JSONL
# ---------------------------------------------------------------------------

_TAXID_RE = re.compile(r"\|taxid\|(\d+)\|")

def _taxid_from_header(header: str) -> str | None:
    """
    Try to parse TaxID from an NCBI FASTA header.
    Handles the common pipe-delimited format:
        >NZ_CP012345.1 |taxid|12345| Organism name
    Returns None if no TaxID is embedded.
    """
    m = _TAXID_RE.search(header)
    return m.group(1) if m else None


def extract_taxids_from_fasta(
    fasta_path: Path,
) -> dict[str, str]:
    """
    Stream the FASTA file; collect {accession: taxid} for every record whose
    header contains a TaxID.  Returns a dict.
    """
    log.info("Extracting TaxIDs from FASTA headers: %s", fasta_path)
    acc_taxid: dict[str, str] = {}
    header_re = re.compile(r"^>(\S+)")

    with fasta_path.open("r", encoding="utf-8", errors="replace") as fh:
        current_acc: str | None = None
        for line in fh:
            if line.startswith(">"):
                m = header_re.match(line)
                current_acc = m.group(1) if m else None
                taxid = _taxid_from_header(line)
                if current_acc and taxid:
                    acc_taxid[current_acc] = taxid
    log.info("TaxIDs from headers: %d", len(acc_taxid))
    return acc_taxid


def extract_taxids_from_jsonl(
    jsonl_path: Path,
    accession_filter: set[str] | None = None,
) -> dict[str, str]:
    """
    Fall-back: read accession→taxId mapping from data_report.jsonl.
    Faster for the full dataset than header parsing when the metadata exists.
    """
    log.info("Extracting TaxIDs from metadata JSONL: %s", jsonl_path)
    acc_taxid: dict[str, str] = {}
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            acc = record.get("accession")
            tid = str(record.get("taxId", ""))
            if acc and tid and tid != "None":
                if accession_filter is None or acc in accession_filter:
                    acc_taxid[acc] = tid

    log.info("TaxIDs from JSONL: %d", len(acc_taxid))
    return acc_taxid


def build_accession_taxid_map(
    fasta_path: Path,
    jsonl_path: Path | None = None,
) -> dict[str, str]:
    """
    Build the definitive {accession: taxid} map.
    Prioritises JSONL metadata (complete) over header parsing (partial).
    """
    # Try loading from file header first for a quick sample
    header_map = extract_taxids_from_fasta(fasta_path)

    if jsonl_path and jsonl_path.exists():
        # Use JSONL for full coverage; fall back to header where JSONL is absent
        jsonl_map = extract_taxids_from_jsonl(jsonl_path)
        merged = {**header_map, **jsonl_map}
        log.info("Merged map: %d accessions with TaxIDs", len(merged))
        return merged

    return header_map


# ---------------------------------------------------------------------------
# Step 2 & 3 — Chunked parallel TaxonKit reformat
# ---------------------------------------------------------------------------

def _ensure_taxonkit_data() -> None:
    """
    Check that the NCBI taxonomy dump is present at ~/.taxonkit/.
    If not, download it automatically via taxonkit download.
    """
    nodes = TAXONKIT_DATA_DIR / "nodes.dmp"
    if nodes.exists():
        log.info("TaxonKit data found: %s", TAXONKIT_DATA_DIR)
        return

    log.info("TaxonKit data not found.  Downloading NCBI taxonomy dump …")
    TAXONKIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [TAXONKIT_BIN, "download", "--data-dir", str(TAXONKIT_DATA_DIR)],
        check=True,
    )
    log.info("Taxonomy dump downloaded.")


def _taxonkit_reformat_chunk(args: tuple[Path, Path]) -> Path:
    """
    Worker function: call 'taxonkit reformat' on one chunk file.
    Returns the path to the output TSV for this chunk.

    TaxonKit output columns (with --show-lineage-taxids -P):
        taxid  formatted_lineage  status

    We request the 7-rank canonical format:
        {k};{p};{c};{o};{f};{g};{s}
    """
    chunk_path, out_path = args
    cmd = [
        TAXONKIT_BIN,
        "reformat",
        "--data-dir",   str(TAXONKIT_DATA_DIR),
        "--lineage-field", "1",
        "--format",     "{k}\\t{p}\\t{c}\\t{o}\\t{f}\\t{g}\\t{s}",
        "--fill-miss-rank",
        "-j",           str(max(1, os.cpu_count() or 1)),   # goroutine threads
        "-o",           str(out_path),
        str(chunk_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"taxonkit reformat failed on {chunk_path}:\n{result.stderr[:400]}"
        )
    return out_path


def reconstruct_lineages_parallel(
    taxid_list: list[str],
    max_workers: int | None = None,
    chunk_size: int = TAXON_CHUNK_SIZE,
) -> list[dict]:
    """
    Split taxid_list into chunks, run TaxonKit on each chunk in parallel,
    and collect all lineage dicts.

    Returns list of dicts with keys:
        taxid, superkingdom, phylum, class, order, family, genus, species
    """
    _ensure_taxonkit_data()

    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 16)

    log.info(
        "Reconstructing lineages for %d TaxIDs in %d-ID chunks (%d workers)",
        len(taxid_list), chunk_size, max_workers
    )

    # Write chunks to temp files
    with tempfile.TemporaryDirectory(dir=DIRS.taxonomy, prefix="taxonkit_") as tmpdir:
        tmpdir_p = Path(tmpdir)
        chunk_args: list[tuple[Path, Path]] = []

        for i in range(0, len(taxid_list), chunk_size):
            chunk = taxid_list[i: i + chunk_size]
            chunk_in  = tmpdir_p / f"chunk_{i:010d}.txt"
            chunk_out = tmpdir_p / f"chunk_{i:010d}.tsv"
            chunk_in.write_text("\n".join(chunk))
            chunk_args.append((chunk_in, chunk_out))

        log.info("Dispatching %d chunks to ProcessPoolExecutor …", len(chunk_args))
        t0 = time.monotonic()

        results: list[dict] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_taxonkit_reformat_chunk, arg): arg for arg in chunk_args}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                out_path = fut.result()   # raises if chunk failed
                done += 1
                if done % 50 == 0 or done == len(chunk_args):
                    log.info("  chunks complete: %d / %d", done, len(chunk_args))

                # Parse TSV output from this chunk
                for row in _parse_taxonkit_tsv(out_path):
                    results.append(row)

        elapsed = time.monotonic() - t0
        log.info(
            "Taxonomy reconstruction: %d lineages in %.1f s", len(results), elapsed
        )
        return results


def _parse_taxonkit_tsv(tsv_path: Path) -> Iterator[dict]:
    """
    Parse a taxonkit reformat output TSV.
    Lines with empty lineage (merged/deleted nodes) are skipped.
    """
    with tsv_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            taxid, sk, ph, cl, od, fa, ge, sp = parts[:8]
            if not taxid.isdigit():
                continue
            yield {
                "taxid":        taxid,
                "superkingdom": sk.strip(),
                "phylum":       ph.strip(),
                "class":        cl.strip(),
                "order":        od.strip(),
                "family":       fa.strip(),
                "genus":        ge.strip(),
                "species":      sp.strip(),
            }


# ---------------------------------------------------------------------------
# Step 4 — Merge and write output TSV
# ---------------------------------------------------------------------------

def write_taxonomy_tsv(
    acc_taxid: dict[str, str],
    lineages: list[dict],
    output_path: Path = TAXONOMY_TSV,
) -> int:
    """
    Join the accession→taxid map with the lineage records; write the final
    annotation TSV.  Returns number of annotated sequences.
    """
    # Index lineages by taxid for O(1) lookup
    lineage_index: dict[str, dict] = {r["taxid"]: r for r in lineages}

    log.info("Writing taxonomy TSV: %s", output_path)
    written = 0
    missing = 0

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["accession", "taxid"] + TAXONOMY_RANKS)

        for acc, taxid in acc_taxid.items():
            row = lineage_index.get(taxid)
            if row is None:
                missing += 1
                continue
            writer.writerow([
                acc, taxid,
                row["superkingdom"], row["phylum"], row["class"],
                row["order"],        row["family"], row["genus"],
                row["species"],
            ])
            written += 1

    log.info(
        "Taxonomy TSV: %d annotated, %d without lineage → %s",
        written, missing, output_path
    )
    return written


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_taxonomy(
    fasta_path: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Execute the full taxonomy reconstruction pipeline.

    Args:
        fasta_path: Path to the representative FASTA (Phase 2 output).
                    Defaults to MMSEQS2_REP_FASTA, falling back to DEDUP_FASTA_PATH.
        force:      Re-run even if checkpoint exists.

    Returns path to lineages_full.tsv.
    """
    cp = _load_cp()
    if not force and cp.get("done") and TAXONOMY_TSV.exists():
        log.info("Phase 3 already complete (%s).  Skipping.", TAXONOMY_TSV)
        return TAXONOMY_TSV

    # Resolve input FASTA
    if fasta_path is None:
        fasta_path = MMSEQS2_REP_FASTA if MMSEQS2_REP_FASTA.exists() else DEDUP_FASTA_PATH
    if not fasta_path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {fasta_path}")

    log.info("Phase 3 — taxonomy reconstruction.  Input: %s", fasta_path)

    # Build accession→taxid map
    jsonl_path = DATA_REPORT_JSONL if DATA_REPORT_JSONL.exists() else None
    acc_taxid = build_accession_taxid_map(fasta_path, jsonl_path)

    # Run parallel TaxonKit
    unique_taxids = list(set(acc_taxid.values()))
    lineages = reconstruct_lineages_parallel(unique_taxids)

    # Write output
    n_annotated = write_taxonomy_tsv(acc_taxid, lineages)

    cp["done"]         = True
    cp["n_annotated"]  = n_annotated
    cp["n_taxids"]     = len(unique_taxids)
    _save_cp(cp)

    log.info("Phase 3 complete.  Taxonomy TSV: %s", TAXONOMY_TSV)
    return TAXONOMY_TSV


if __name__ == "__main__":
    run_taxonomy()
