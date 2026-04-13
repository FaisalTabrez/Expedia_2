"""
EXPEDIA Demo Configuration
===========================
Overrides production pipeline_config.py for a scoped demo run.

  Scale:    ~2M sequences  (Marine + Benthic subset)
  Storage:  ~35 GB — all inside the project directory (EXPEDIA_Data/)
  Compute:  Google Colab T4/A100 for Phase 4 embedding

Usage
-----
Set the env-var once before any imports, then call apply_demo_config():

    import os
    os.environ["EXPEDIA_DEMO"] = "1"
    # Optional — defaults to EXPEDIA_Data/ next to the code files:
    os.environ["EXPEDIA_ROOT"] = "/absolute/path/to/EXPEDIA_Data"

    import pipeline_config          # must come first
    from demo_config import apply_demo_config
    apply_demo_config()

Or just run a phase directly:

    EXPEDIA_DEMO=1 python phase2_deduplication.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Guard: must be imported after pipeline_config has already run
# ---------------------------------------------------------------------------

def apply_demo_config() -> None:
    """
    Inject all demo overrides into the pipeline_config module namespace.

    Every phase module that imports from pipeline_config will automatically
    see the demo values because we mutate the live module object.
    """
    import pipeline_config as cfg

    # -----------------------------------------------------------------------
    # Single project root — raw data, embeddings, index all live here
    # -----------------------------------------------------------------------
    project_root = Path(r"C:\Volume D\Expedia_2\Expedia_Data").resolve()

    # Re-initialise the directory layout against the project root so every
    # sub-path is consistent. DirectoryLayout.__post_init__ creates all dirs.
    cfg.EXPEDIA_ROOT = project_root
    cfg.DIRS         = cfg.DirectoryLayout(root=project_root)

    # -----------------------------------------------------------------------
    # Phase 1 — Scoped NCBI query  (Marine + Benthic → ~2M records, ~12 GB)
    # -----------------------------------------------------------------------
    cfg.NCBI_DATASETS_BIN     = r"C:\Volume D\Expedia_2\NCBI_Exe\datasets.exe"
    cfg.NCBI_QUERY_TERMS      = ["Marine", "Benthic"]
    cfg.NCBI_EXPECTED_RECORDS = 2_000_000
    cfg.NCBI_EXPECTED_SIZE_GB = 12

    cfg.DEHYDRATED_ZIP    = cfg.DIRS.raw_fasta / "ncbi_dataset_dehydrated.zip"
    cfg.DEHYDRATED_DIR    = cfg.DIRS.raw_fasta / "ncbi_dataset"
    cfg.DATA_REPORT_JSONL = (
        cfg.DIRS.raw_fasta
        / "ncbi_dataset" / "ncbi_dataset" / "data" / "data_report.jsonl"
    )
    cfg.RAW_FASTA_PATH    = cfg.DIRS.raw_fasta / "marine_sequences_raw.fasta"

    # -----------------------------------------------------------------------
    # Phase 2 — Deduplication
    # -----------------------------------------------------------------------
    cfg.DUCKDB_MEMORY_LIMIT = "2GB"
    cfg.ACCESSION_KEEP_LIST = cfg.DIRS.dedup_fasta / "keep_accessions.txt"
    cfg.DEDUP_FASTA_PATH    = cfg.DIRS.dedup_fasta / "deduplicated_marine_demo.fasta"
    cfg.MMSEQS2_CLUSTER_DIR = cfg.DIRS.dedup_fasta / "mmseqs2_cluster"
    cfg.MMSEQS2_REP_FASTA   = cfg.DIRS.dedup_fasta / "representative_marine_demo.fasta"
    cfg.DEDUP_CHECKPOINT    = cfg.DIRS.checkpoints  / "phase2_dedup.json"

    # -----------------------------------------------------------------------
    # Phase 3 — Taxonomy
    # -----------------------------------------------------------------------
    cfg.TAXON_CHUNK_SIZE    = 20_000
    cfg.TAXONOMY_TSV        = cfg.DIRS.taxonomy    / "lineages_demo.tsv"
    cfg.TAXONOMY_CHECKPOINT = cfg.DIRS.checkpoints / "phase3_taxonomy.json"

    # -----------------------------------------------------------------------
    # Phase 4 — Embedding  (Colab GPU; outputs stay in project dir)
    # -----------------------------------------------------------------------
    cfg.EMBED_BATCH_SIZE      = _detect_colab_batch_size()
    cfg.EMBED_MAX_LENGTH      = 512
    cfg.EMBED_CHECKPOINT_N    = 5_000

    cfg.EMBED_OUTPUT_NPY = cfg.DIRS.embeddings / "embeddings_demo.npy"
    cfg.EMBED_IDS_TXT    = cfg.DIRS.embeddings / "embedding_accession_ids_demo.txt"
    cfg.EMBED_CHECKPOINT = cfg.DIRS.checkpoints / "phase4_embedding.json"

    # Model weights cache — inside the project so they persist across sessions
    cfg.GENOME_OCEAN_LOCAL_DIR = cfg.DIRS.models / "GenomeOcean-4B"

    # -----------------------------------------------------------------------
    # Phase 5 — LanceDB  (sqrt(2M) ≈ 1414 → 512 partitions for demo scale)
    # -----------------------------------------------------------------------
    cfg.IVF_NUM_PARTITIONS = 512
    cfg.LANCEDB_CHECKPOINT = cfg.DIRS.checkpoints / "phase5_lancedb.json"

    # -----------------------------------------------------------------------
    # Phase 6 — Clustering
    # -----------------------------------------------------------------------
    cfg.HDBSCAN_MIN_CLUSTER_SIZE = 15
    cfg.HDBSCAN_MIN_SAMPLES      = 3
    cfg.UMAP_LOW_MEMORY          = False   # 2M points fits in RAM
    cfg.CLUSTER_CHECKPOINT       = cfg.DIRS.checkpoints / "phase6_clustering.json"

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    cfg.LOG_FILE  = cfg.DIRS.logs / "expedia_demo.log"
    cfg.LOG_LEVEL = "DEBUG"

    # Demo-only markers
    cfg.IS_DEMO       = True
    cfg.DEMO_DATA_DIR = project_root

    _print_demo_summary(cfg)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_colab_batch_size() -> int:
    """
    Probe available GPU VRAM and return a safe batch size for GenomeOcean.
    Falls back to 32 if no GPU is found (will be slow but correct).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 32
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 38:    # A100 40GB
            return 128
        if vram_gb >= 14:    # T4 16GB
            return 64
        if vram_gb >= 10:    # V100 or similar
            return 48
        return 32            # smaller GPU
    except Exception:
        return 32


def _print_demo_summary(cfg) -> None:
    """Print a concise summary of the active demo configuration."""
    sep = "─" * 60
    lines = [
        "",
        sep,
        "  EXPEDIA DEMO CONFIGURATION ACTIVE",
        sep,
        f"  Runtime env   : {cfg.RUNTIME_ENV}",
        f"  Data root     : {cfg.DEMO_DATA_DIR}",
        f"  NCBI query    : {' OR '.join(cfg.NCBI_QUERY_TERMS)}",
        f"  Expected seqs : ~{cfg.NCBI_EXPECTED_RECORDS:,}",
        f"  Embed batch   : {cfg.EMBED_BATCH_SIZE}",
        f"  Checkpoint N  : {cfg.EMBED_CHECKPOINT_N:,}",
        f"  IVF partitions: {cfg.IVF_NUM_PARTITIONS}",
        f"  Model cache   : {cfg.GENOME_OCEAN_LOCAL_DIR}",
        f"  Embed output  : {cfg.EMBED_OUTPUT_NPY}",
        sep,
        "",
    ]
    for line in lines:
        print(line, file=sys.stderr)


# ---------------------------------------------------------------------------
# Convenience: apply on import if EXPEDIA_DEMO env-var is set
# ---------------------------------------------------------------------------

if os.environ.get("EXPEDIA_DEMO"):
    apply_demo_config()
