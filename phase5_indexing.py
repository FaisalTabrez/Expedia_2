"""
EXPEDIA Tier 2 — Phase 5: LanceDB IVF-RQ Vector Indexing
==========================================================
Ingests GenomeOcean embeddings into LanceDB with a radically recalibrated
IVF-RQ (RaBitQ) index tuned for 45.8M high-dimensional vectors.

Tier 1 → Tier 2 parameter migration:
  Index type:    IVF_PQ (8-bit codebook)    → IVF_RQ (1-bit sign pattern)
  Partitions:    256                         → 8192  (sqrt(45.8M) ≈ 6767 → 2^13)
  Quantisation:  96 sub-vectors @ 8-bit      → 1-bit per dimension
  Memory/vec:    4096 bytes (float32 1024D)  → 136 bytes
  Latency goal:  <10 ms @ 313k records       → <10 ms @ 45.8M records

RaBitQ's O(1/sqrt(D)) error bound means accuracy *improves* as GenomeOcean's
embedding dimensionality increases — the opposite of PQ's degradation curve.

The full 45.8M × 136-byte index fits in the OS page cache (~6 GB), eliminating
random-read latency against USB/NVMe hardware entirely.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline_config import (
    DIRS,
    EMBED_DIMENSION,
    EMBED_IDS_TXT,
    EMBED_OUTPUT_NPY,
    IVF_NUM_PARTITIONS,
    IVF_NUM_SUB_VECTORS,
    LANCEDB_CHECKPOINT,
    LANCEDB_PATH,
    LANCEDB_TABLE_NAME,
    LOG_FILE,
    RABITQ_NUM_BITS,
    TAXONOMY_TSV,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("phase5.indexing")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_cp() -> dict:
    return json.loads(LANCEDB_CHECKPOINT.read_text()) if LANCEDB_CHECKPOINT.exists() else {}

def _save_cp(state: dict) -> None:
    LANCEDB_CHECKPOINT.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Taxonomy annotation loader
# ---------------------------------------------------------------------------

def _load_taxonomy_index(tsv_path: Path) -> dict[str, dict]:
    """
    Read lineages_full.tsv into an {accession: {rank: value}} dict.
    Used to annotate each vector row with taxonomic metadata.
    """
    import csv
    index: dict[str, dict] = {}
    with tsv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            index[row["accession"]] = dict(row)
    log.info("Loaded taxonomy annotations: %d entries", len(index))
    return index


# ---------------------------------------------------------------------------
# Batch ingestion helpers
# ---------------------------------------------------------------------------

def _iter_batches(
    embeddings: np.ndarray,
    accession_ids: list[str],
    taxonomy_index: dict[str, dict],
    batch_size: int = 50_000,
):
    """
    Yield PyArrow-compatible dicts in batches for LanceDB ingest.
    Each row: {vector, accession, taxid, superkingdom, phylum, …, species}
    """
    n = len(accession_ids)
    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch_accs = accession_ids[start:end]
        batch_vecs = embeddings[start:end].tolist()   # PyArrow requires Python lists

        rows = []
        for acc, vec in zip(batch_accs, batch_vecs):
            tax = taxonomy_index.get(acc, {})
            rows.append({
                "vector":       vec,
                "accession":    acc,
                "taxid":        tax.get("taxid",       ""),
                "superkingdom": tax.get("superkingdom",""),
                "phylum":       tax.get("phylum",      ""),
                "class":        tax.get("class",       ""),
                "order":        tax.get("order",       ""),
                "family":       tax.get("family",      ""),
                "genus":        tax.get("genus",       ""),
                "species":      tax.get("species",     ""),
            })
        yield rows
        log.debug("Yielded batch rows %d–%d", start, end)


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------

def build_lancedb_table(
    embeddings_npy: Path = EMBED_OUTPUT_NPY,
    ids_txt:        Path = EMBED_IDS_TXT,
    taxonomy_tsv:   Path = TAXONOMY_TSV,
    db_path:        Path = LANCEDB_PATH,
    table_name:     str  = LANCEDB_TABLE_NAME,
    ingest_batch:   int  = 50_000,
    force:          bool = False,
) -> None:
    """
    Ingest embeddings into LanceDB, building the IVF-RQ index.

    The function streams embeddings in batches to avoid loading all 45.8M
    float32 vectors (~180 GB) into RAM simultaneously — only `ingest_batch`
    rows are in memory at a time.
    """
    cp = _load_cp()
    if not force and cp.get("table_built"):
        log.info("LanceDB table already built.  Skipping ingest.")
        return

    try:
        import lancedb
        import pyarrow as pa
    except ImportError:
        raise EnvironmentError(
            "Install: pip install lancedb pyarrow --break-system-packages"
        )

    # Load accession IDs
    log.info("Loading accession IDs: %s", ids_txt)
    accession_ids = ids_txt.read_text(encoding="utf-8").splitlines()
    accession_ids = [a.strip() for a in accession_ids if a.strip()]
    log.info("Accession IDs loaded: %d", len(accession_ids))

    # Memory-map embeddings (avoids loading all ~180 GB at once)
    log.info("Memory-mapping embeddings: %s", embeddings_npy)
    embeddings = np.load(embeddings_npy, mmap_mode="r")
    assert embeddings.shape[0] == len(accession_ids), (
        f"Embedding count mismatch: {embeddings.shape[0]} vectors vs "
        f"{len(accession_ids)} IDs"
    )
    log.info("Embeddings shape: %s  dtype: %s", embeddings.shape, embeddings.dtype)

    # Load taxonomy
    taxonomy_index = _load_taxonomy_index(taxonomy_tsv) if taxonomy_tsv.exists() else {}

    # Connect (creates dir if absent)
    db = lancedb.connect(str(db_path))

    # Drop existing table if rebuilding
    if table_name in db.table_names() and not force:
        log.info("Table '%s' exists.  Set force=True to rebuild.", table_name)
        cp["table_built"] = True
        _save_cp(cp)
        return
    if table_name in db.table_names():
        db.drop_table(table_name)
        log.info("Dropped existing table '%s'.", table_name)

    # Build PyArrow schema
    schema = pa.schema([
        pa.field("vector",       pa.list_(pa.float32(), EMBED_DIMENSION)),
        pa.field("accession",    pa.string()),
        pa.field("taxid",        pa.string()),
        pa.field("superkingdom", pa.string()),
        pa.field("phylum",       pa.string()),
        pa.field("class",        pa.string()),
        pa.field("order",        pa.string()),
        pa.field("family",       pa.string()),
        pa.field("genus",        pa.string()),
        pa.field("species",      pa.string()),
    ])

    log.info("Creating LanceDB table '%s' …", table_name)
    t0 = time.monotonic()
    table = None
    total_ingested = 0

    for batch_rows in _iter_batches(
        embeddings, accession_ids, taxonomy_index, ingest_batch
    ):
        if table is None:
            table = db.create_table(
                table_name,
                data=batch_rows,
                schema=schema,
                mode="create",
            )
        else:
            table.add(batch_rows)

        total_ingested += len(batch_rows)
        if total_ingested % 500_000 == 0 or total_ingested == len(accession_ids):
            elapsed = time.monotonic() - t0
            rate    = total_ingested / max(elapsed, 1)
            log.info(
                "Ingested %d / %d  (%.0f rows/s)",
                total_ingested, len(accession_ids), rate
            )

    elapsed = time.monotonic() - t0
    log.info(
        "Ingest complete: %d rows in %.1f min.  Building IVF-RQ index …",
        total_ingested, elapsed / 60
    )

    cp["table_built"]    = True
    cp["n_rows"]         = total_ingested
    cp["ingest_time_min"] = round(elapsed / 60, 1)
    _save_cp(cp)


def build_ivf_rq_index(
    db_path:         Path  = LANCEDB_PATH,
    table_name:      str   = LANCEDB_TABLE_NAME,
    num_partitions:  int   = IVF_NUM_PARTITIONS,
    num_sub_vectors: int   = IVF_NUM_SUB_VECTORS,
    num_bits:        int   = RABITQ_NUM_BITS,
    force:           bool  = False,
) -> None:
    """
    Build (or rebuild) the IVF-RQ index on the vector column.

    Parameters:
        num_partitions:  8192  — sqrt(45.8M) ≈ 6767, rounded to 2^13
        num_sub_vectors: 128   — dim / 8 = 1024 / 8
        num_bits:        1     — RaBitQ 1-bit per dimension
    """
    cp = _load_cp()
    if not force and cp.get("index_built"):
        log.info("IVF-RQ index already built.  Skipping.")
        return

    try:
        import lancedb
    except ImportError:
        raise EnvironmentError(
            "Install: pip install lancedb --break-system-packages"
        )

    log.info(
        "Building IVF-RQ index: %d partitions, %d sub-vectors, %d-bit",
        num_partitions, num_sub_vectors, num_bits
    )
    log.info("Estimated index size: ~%.1f GB (1-bit per dimension per vector)",
             # 45.8M * 1024 bits / 8 / 1e9
             (45_800_000 * 1024 / 8) / 1e9)

    db    = lancedb.connect(str(db_path))
    table = db.open_table(table_name)

    t0 = time.monotonic()

    # LanceDB API: create_index with IVF_RQ type
    # Ref: https://lancedb.github.io/lancedb/ann_indexes/
    table.create_index(
        metric="cosine",
        vector_column_name="vector",
        index_type="IVF_RQ",
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        num_bits=num_bits,
        replace=force,
    )

    elapsed = time.monotonic() - t0
    log.info("IVF-RQ index built in %.1f min.", elapsed / 60)

    cp["index_built"]    = True
    cp["num_partitions"] = num_partitions
    cp["num_sub_vectors"]= num_sub_vectors
    cp["num_bits"]       = num_bits
    cp["index_time_min"] = round(elapsed / 60, 1)
    _save_cp(cp)


# ---------------------------------------------------------------------------
# Query helper (used by Phase 6 PROTAX)
# ---------------------------------------------------------------------------

def query_knn(
    query_vector: np.ndarray,
    k:            int  = 15,
    db_path:      Path = LANCEDB_PATH,
    table_name:   str  = LANCEDB_TABLE_NAME,
) -> list[dict]:
    """
    Return the k nearest neighbours for a query vector.
    Result rows include accession, taxid, and full lineage fields.
    """
    import lancedb

    db    = lancedb.connect(str(db_path))
    table = db.open_table(table_name)

    results = (
        table.search(query_vector.tolist())
        .limit(k)
        .select(["accession", "taxid",
                 "superkingdom", "phylum", "class",
                 "order", "family", "genus", "species",
                 "_distance"])
        .to_list()
    )
    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_indexing(force: bool = False) -> None:
    """Build the LanceDB table and IVF-RQ index end-to-end."""
    cp = _load_cp()
    if not force and cp.get("table_built") and cp.get("index_built"):
        log.info("Phase 5 already complete.  Skipping.")
        return

    build_lancedb_table(force=force)
    build_ivf_rq_index(force=force)
    log.info("Phase 5 complete.  LanceDB at: %s", LANCEDB_PATH)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_indexing(force=args.force)
