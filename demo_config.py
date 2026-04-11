"""
EXPEDIA Demo Configuration
===========================
Overrides production pipeline_config.py for a scoped demo run:

  Hardware:   64 GB USB pen drive  (local phases 1–3, 5–7)
  Compute:    Google Colab T4/A100  (Phase 4 embedding only)
  Scale:      ~2M sequences  (Marine + Bacteria subset — fits on USB)
  Storage:    ~35 GB total across all phases

Import pattern
--------------
Every phase already imports its constants from pipeline_config.  To use demo
settings, set the environment variable before the process starts:

    # In a terminal or at the top of a notebook:
    import os
    os.environ["EXPEDIA_DEMO"] = "1"
    os.environ["EXPEDIA_ROOT"] = "/path/to/usb/EXPEDIA_Data"   # optional override

Then in each phase file, add ONE line after the pipeline_config import:

    from pipeline_config import *        # existing line
    if os.environ.get("EXPEDIA_DEMO"):
        from demo_config import *        # overlay demo values

This keeps production code untouched — the demo overlay is purely additive.

Alternatively, run any phase directly with demo settings:

    python -c "
    import os; os.environ['EXPEDIA_DEMO']='1'
    from demo_config import apply_demo_config; apply_demo_config()
    from phase2_deduplication import run_deduplication; run_deduplication()
    "
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
    Call this once (before importing any phase module) to inject all demo
    overrides into the pipeline_config module namespace.

    This approach means every phase module that does:
        from pipeline_config import EMBED_BATCH_SIZE
    will see the demo value, because we mutate the module object in-place.
    """
    import pipeline_config as cfg

    # -----------------------------------------------------------------------
    # Resolve roots
    # -----------------------------------------------------------------------

    # USB drive root — honour env-var, otherwise use the auto-detected root
    # (which may already be the USB if it has an EXPEDIA_Data folder on it)
    usb_root = Path(os.environ.get("EXPEDIA_ROOT", str(cfg.EXPEDIA_ROOT)))

    # Colab Drive root — where Phase 4 outputs land during embedding
    # On Colab this resolves to /content/drive/MyDrive/EXPEDIA_Data
    # Locally it falls back to the same usb_root
    drive_root_str = os.environ.get(
        "EXPEDIA_DRIVE_ROOT",
        "/content/drive/MyDrive/EXPEDIA_Data"
    )
    drive_root = Path(drive_root_str)
    if not drive_root.exists():
        # Not on Colab — use USB for everything
        drive_root = usb_root

    # -----------------------------------------------------------------------
    # Override DirectoryLayout  (USB root stays, embeddings go to Drive)
    # -----------------------------------------------------------------------

    # Re-initialise DIRS against the USB root to get correct sub-paths
    cfg.EXPEDIA_ROOT = usb_root
    cfg.DIRS         = cfg.DirectoryLayout(root=usb_root)

    # Embeddings sub-dir lives on Drive (large, needs persistence across
    # Colab disconnects) — override just the embeddings sub-paths
    demo_embed_dir = drive_root / "04_embeddings"
    demo_embed_dir.mkdir(parents=True, exist_ok=True)
    cfg.DIRS.embeddings = demo_embed_dir

    # Checkpoints also live on Drive so Colab resumes survive disconnects
    demo_ckpt_dir = drive_root / "checkpoints"
    demo_ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.DIRS.checkpoints = demo_ckpt_dir

    # -----------------------------------------------------------------------
    # Phase 1 — Scoped NCBI query  (Marine + Bacteria only → ~2M records)
    # -----------------------------------------------------------------------

    cfg.NCBI_QUERY_TERMS      = ["Marine", "Benthic"]    # narrowed from 6 terms
    cfg.NCBI_EXPECTED_RECORDS = 2_000_000                # ~2M estimated
    cfg.NCBI_EXPECTED_SIZE_GB = 12                       # ~12 GB raw FASTA

    # Re-derive dependent paths from the new DIRS
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

    cfg.DUCKDB_MEMORY_LIMIT = "2GB"        # conservative for USB-backed temp files
    cfg.ACCESSION_KEEP_LIST = cfg.DIRS.dedup_fasta / "keep_accessions.txt"
    cfg.DEDUP_FASTA_PATH    = cfg.DIRS.dedup_fasta / "deduplicated_marine_demo.fasta"
    cfg.MMSEQS2_CLUSTER_DIR = cfg.DIRS.dedup_fasta / "mmseqs2_cluster"
    cfg.MMSEQS2_REP_FASTA   = cfg.DIRS.dedup_fasta / "representative_marine_demo.fasta"
    cfg.DEDUP_CHECKPOINT    = cfg.DIRS.checkpoints / "phase2_dedup.json"

    # -----------------------------------------------------------------------
    # Phase 3 — Taxonomy
    # -----------------------------------------------------------------------

    cfg.TAXON_CHUNK_SIZE       = 20_000    # smaller chunks — fewer sequences
    cfg.TAXONOMY_TSV           = cfg.DIRS.taxonomy / "lineages_demo.tsv"
    cfg.TAXONOMY_CHECKPOINT    = cfg.DIRS.checkpoints / "phase3_taxonomy.json"

    # -----------------------------------------------------------------------
    # Phase 4 — Embedding  (Colab T4/A100)
    # -----------------------------------------------------------------------

    # Colab T4  = 16 GB VRAM  →  batch 64  safely  (128 if A100)
    # Auto-detect: if an A100 is available push to 128, else stay at 64
    _colab_batch = _detect_colab_batch_size()
    cfg.EMBED_BATCH_SIZE   = _colab_batch
    cfg.EMBED_MAX_LENGTH   = 512           # shorter context for speed on demo seqs
    cfg.EMBED_CHECKPOINT_N = 5_000         # checkpoint every 5k (shorter sessions)

    # All embedding outputs go to Google Drive for persistence
    cfg.EMBED_OUTPUT_NPY = drive_root / "04_embeddings" / "embeddings_demo.npy"
    cfg.EMBED_IDS_TXT    = drive_root / "04_embeddings" / "embedding_accession_ids_demo.txt"
    cfg.EMBED_CHECKPOINT = drive_root / "checkpoints"   / "phase4_embedding.json"

    # Model cache on Colab's local /content  (faster I/O, re-downloaded each session
    # if not already on Drive — see notebook for Drive model caching logic)
    colab_model_cache = Path("/content/models/GenomeOcean-4B")
    drive_model_cache = drive_root / "resources" / "models" / "GenomeOcean-4B"
    if colab_model_cache.exists():
        cfg.GENOME_OCEAN_LOCAL_DIR = colab_model_cache
    elif drive_model_cache.exists():
        cfg.GENOME_OCEAN_LOCAL_DIR = drive_model_cache
    else:
        # Will pull from HuggingFace and cache to Drive on first run
        cfg.GENOME_OCEAN_LOCAL_DIR = drive_model_cache

    # -----------------------------------------------------------------------
    # Phase 5 — LanceDB  (USB, smaller index)
    # -----------------------------------------------------------------------

    # sqrt(2_000_000) ≈ 1414 → nearest power of 2 = 2048
    cfg.IVF_NUM_PARTITIONS = 512           # conservative for 2M records on USB
    cfg.LANCEDB_CHECKPOINT = cfg.DIRS.checkpoints / "phase5_lancedb.json"

    # -----------------------------------------------------------------------
    # Phase 6 — Clustering
    # -----------------------------------------------------------------------

    cfg.HDBSCAN_MIN_CLUSTER_SIZE = 15      # smaller — fewer total sequences
    cfg.HDBSCAN_MIN_SAMPLES      = 3
    cfg.UMAP_LOW_MEMORY          = False   # 2M points fits in RAM; skip disk-backed ANN
    cfg.CLUSTER_CHECKPOINT       = cfg.DIRS.checkpoints / "phase6_clustering.json"

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    cfg.LOG_FILE  = cfg.DIRS.logs / "expedia_demo.log"
    cfg.LOG_LEVEL = "DEBUG"                # verbose for demo / debugging

    # -----------------------------------------------------------------------
    # Demo-only constants (not in production config)
    # -----------------------------------------------------------------------

    cfg.IS_DEMO          = True
    cfg.DEMO_DRIVE_ROOT  = drive_root
    cfg.DEMO_USB_ROOT    = usb_root

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
        f"  Runtime env  : {cfg.RUNTIME_ENV}",
        f"  USB root     : {cfg.DEMO_USB_ROOT}",
        f"  Drive root   : {cfg.DEMO_DRIVE_ROOT}",
        f"  NCBI query   : {' OR '.join(cfg.NCBI_QUERY_TERMS)}",
        f"  Expected seqs: ~{cfg.NCBI_EXPECTED_RECORDS:,}",
        f"  Embed batch  : {cfg.EMBED_BATCH_SIZE}",
        f"  Checkpoint N : {cfg.EMBED_CHECKPOINT_N:,}",
        f"  IVF partitions: {cfg.IVF_NUM_PARTITIONS}",
        f"  Model cache  : {cfg.GENOME_OCEAN_LOCAL_DIR}",
        f"  Embed output : {cfg.EMBED_OUTPUT_NPY}",
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
