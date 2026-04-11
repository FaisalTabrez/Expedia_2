"""
EXPEDIA Tier 2 — Pipeline Configuration
========================================
Centralised configuration for the 45.8M-record aquatic genomics pipeline.
All phases import from here so tuning one value propagates everywhere.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Volume / directory discovery
# ---------------------------------------------------------------------------

def find_expedia_root() -> Path:
    """
    Scan all available drive letters (Windows) or /mnt/* (Linux/WSL) for the
    canonical EXPEDIA_Data folder.  Falls back to a local ./EXPEDIA_Data.
    """
    candidates: list[Path] = []

    if platform.system() == "Windows":
        import string
        for letter in string.ascii_uppercase:
            p = Path(f"{letter}:/EXPEDIA_Data")
            candidates.append(p)
    else:
        # WSL or native Linux
        for mnt in Path("/mnt").iterdir() if Path("/mnt").exists() else []:
            candidates.append(mnt / "EXPEDIA_Data")
        candidates.append(Path.home() / "EXPEDIA_Data")

    for c in candidates:
        if c.exists():
            return c

    # Auto-create under cwd as fallback
    fallback = Path.cwd() / "EXPEDIA_Data"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


EXPEDIA_ROOT: Path = find_expedia_root()


# ---------------------------------------------------------------------------
# Sub-directory layout (mirrors existing Tier 1 structure)
# ---------------------------------------------------------------------------

@dataclass
class DirectoryLayout:
    root:            Path
    raw_fasta:       Path = field(init=False)
    dedup_fasta:     Path = field(init=False)
    taxonomy:        Path = field(init=False)
    embeddings:      Path = field(init=False)
    lancedb:         Path = field(init=False)
    clustering:      Path = field(init=False)
    checkpoints:     Path = field(init=False)
    models:          Path = field(init=False)
    logs:            Path = field(init=False)
    ipc_buffers:     Path = field(init=False)

    def __post_init__(self):
        self.raw_fasta   = self.root / "01_raw_fasta"
        self.dedup_fasta = self.root / "02_dedup_fasta"
        self.taxonomy    = self.root / "03_taxonomy"
        self.embeddings  = self.root / "04_embeddings"
        self.lancedb     = self.root / "05_lancedb"
        self.clustering  = self.root / "06_clustering"
        self.checkpoints = self.root / "checkpoints"
        self.models      = self.root / "resources" / "models"
        self.logs        = self.root / "logs"
        self.ipc_buffers = self.root / "ipc_buffers"
        for attr in vars(self).values():
            if isinstance(attr, Path):
                attr.mkdir(parents=True, exist_ok=True)


DIRS = DirectoryLayout(root=EXPEDIA_ROOT)


# ---------------------------------------------------------------------------
# Phase 1 — NCBI Datasets CLI acquisition
# ---------------------------------------------------------------------------

NCBI_QUERY_TERMS = [
    "Aquatic", "Marine", "Freshwater",
    "Benthic", "Pelagic", "Planktonic",
]

# Estimated total records and compressed size
NCBI_EXPECTED_RECORDS  = 45_819_927
NCBI_EXPECTED_SIZE_GB  = 250

# Path to NCBI datasets binary (must be on PATH or set here)
NCBI_DATASETS_BIN = os.environ.get("NCBI_DATASETS_BIN", "datasets")
NCBI_DATAFORMAT_BIN = os.environ.get("NCBI_DATAFORMAT_BIN", "dataformat")

# Where dehydrated ZIP lands
DEHYDRATED_ZIP   = DIRS.raw_fasta / "ncbi_dataset_dehydrated.zip"
DEHYDRATED_DIR   = DIRS.raw_fasta / "ncbi_dataset"
DATA_REPORT_JSONL = DEHYDRATED_DIR / "ncbi_dataset" / "data" / "data_report.jsonl"
RAW_FASTA_PATH   = DIRS.raw_fasta / "aquatic_sequences_raw.fasta"


# ---------------------------------------------------------------------------
# Phase 2 — Out-of-core deduplication
# ---------------------------------------------------------------------------

DUCKDB_MEMORY_LIMIT   = "4GB"
ACCESSION_KEEP_LIST   = DIRS.dedup_fasta / "keep_accessions.txt"
DEDUP_FASTA_PATH      = DIRS.dedup_fasta / "deduplicated_marine.fasta"
MMSEQS2_CLUSTER_DIR   = DIRS.dedup_fasta / "mmseqs2_cluster"
MMSEQS2_REP_FASTA     = DIRS.dedup_fasta / "representative_marine.fasta"
MMSEQS2_IDENTITY      = 0.95          # 95% identity threshold
MMSEQS2_COVERAGE      = 0.80          # 80% coverage

# Phase 2 checkpoint
DEDUP_CHECKPOINT = DIRS.checkpoints / "phase2_dedup.json"


# ---------------------------------------------------------------------------
# Phase 3 — TaxonKit taxonomy reconstruction
# ---------------------------------------------------------------------------

TAXONKIT_BIN       = os.environ.get("TAXONKIT_BIN", "taxonkit")
TAXONKIT_DATA_DIR  = Path.home() / ".taxonkit"   # nodes.dmp / names.dmp live here
TAXON_CHUNK_SIZE   = 50_000           # IDs per TaxonKit subprocess
TAXONOMY_TSV       = DIRS.taxonomy / "lineages_full.tsv"
TAXONOMY_CHECKPOINT = DIRS.checkpoints / "phase3_taxonomy.json"

# Seven canonical ranks output by taxonkit reformat
TAXONOMY_RANKS = ["superkingdom", "phylum", "class", "order", "family", "genus", "species"]


# ---------------------------------------------------------------------------
# Phase 4 — GenomeOcean neural embedding
# ---------------------------------------------------------------------------

# GenomeOcean HuggingFace model ID (freeze weights locally after first pull)
GENOME_OCEAN_MODEL_ID  = "jgi-doe/GenomeOcean-4B"
GENOME_OCEAN_LOCAL_DIR = DIRS.models / "GenomeOcean-4B"

EMBED_BATCH_SIZE      = 32            # sequences per forward pass (tune to VRAM)
EMBED_MAX_LENGTH      = 2048          # BPE token context window
EMBED_DIMENSION       = 1024          # GenomeOcean output dimensionality
EMBED_CHECKPOINT_N    = 10_000        # write embeddings to disk every N sequences
EMBED_OUTPUT_NPY      = DIRS.embeddings / "embeddings.npy"
EMBED_IDS_TXT         = DIRS.embeddings / "embedding_accession_ids.txt"
EMBED_CHECKPOINT      = DIRS.checkpoints / "phase4_embedding.json"


# ---------------------------------------------------------------------------
# Phase 5 — LanceDB IVF-RQ indexing
# ---------------------------------------------------------------------------

LANCEDB_PATH          = DIRS.lancedb
LANCEDB_TABLE_NAME    = "expedia_vectors"

# IVF partition count: sqrt(45_800_000) ≈ 6767 → nearest power-of-2 = 8192
IVF_NUM_PARTITIONS    = 8192
# RaBitQ 1-bit quantization: 4096-byte float32 vector → 136 bytes
RABITQ_NUM_BITS       = 1
IVF_NUM_SUB_VECTORS   = 128           # dim / 8 = 1024/8 = 128

LANCEDB_CHECKPOINT    = DIRS.checkpoints / "phase5_lancedb.json"


# ---------------------------------------------------------------------------
# Phase 6 — Avalanche clustering (UMAP + HDBSCAN) + PROTAX
# ---------------------------------------------------------------------------

# UMAP
UMAP_N_COMPONENTS     = 10
UMAP_N_NEIGHBORS      = 30
UMAP_MIN_DIST         = 0.0
UMAP_METRIC           = "cosine"
UMAP_LOW_MEMORY       = True          # disk-backed approximate NN graph

# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE = 50
HDBSCAN_MIN_SAMPLES      = 5
HDBSCAN_PREDICTION_DATA  = True       # required for soft clustering
HDBSCAN_SOFT_THRESHOLD   = 0.85       # below this → flag as ambiguous clade

# PROTAX k-NN
PROTAX_K_NEIGHBORS    = 15            # number of reference neighbours to query
PROTAX_CONF_THRESHOLD = 0.95          # high-confidence family-level assignment

CLUSTER_OUTPUT_DIR    = DIRS.clustering
CLUSTER_CHECKPOINT    = DIRS.checkpoints / "phase6_clustering.json"


# ---------------------------------------------------------------------------
# Phase 7 — Apache Arrow zero-copy IPC bridge
# ---------------------------------------------------------------------------

# Shared memory block name prefix (Windows / POSIX shared memory)
SHM_NAME_PREFIX       = "EXPEDIA_SHM_"
# Maximum shared memory allocation for one manifold transfer (bytes)
SHM_MAX_BYTES         = 4 * 1024 ** 3   # 4 GB
# IPC socket / pipe path
IPC_SOCKET_PATH       = str(DIRS.ipc_buffers / "expedia_ipc.sock")
# Arrow IPC stream magic bytes for framing validation
ARROW_IPC_MAGIC       = b"ARROW1\x00\x00"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE = DIRS.logs / "expedia_tier2.log"
LOG_LEVEL = os.environ.get("EXPEDIA_LOG_LEVEL", "INFO")
