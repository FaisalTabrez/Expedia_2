# EXPEDIA Tier 2 — Scalable Genomic Surveillance Engine

Scaling from **313,574 sequences (Tier 1)** to **45.8M records / 250 GB (Tier 2)**  
Platform: Windows 11 · Python 3.13 · Edge Workstation · NVMe recommended

---

## Architecture Overview

| Component | Tier 1 (baseline) | Tier 2 (this code) |
|---|---|---|
| Data acquisition | Biopython Entrez efetch loops | NCBI Datasets CLI dehydrated + rehydrate |
| Deduplication | Polars in-memory hash | DuckDB out-of-core + SeqKit stream + MMseqs2 |
| Taxonomy | TaxonKit sequential | TaxonKit parallel (ProcessPoolExecutor, 50k chunks) |
| Neural core | Nucleotide Transformer v2-50M (6-mer) | GenomeOcean 4B (BPE, metagenomic pretrain) |
| Vector index | LanceDB IVF-PQ 256 parts, 8-bit | LanceDB IVF-RQ 8192 parts, 1-bit RaBitQ |
| Clustering | UMAP + HDBSCAN deterministic | UMAP + HDBSCAN soft (λ probabilities) + PROTAX |
| IPC | JSON-RPC + NTFS disk handshake | Apache Arrow zero-copy shared memory |

---

## Prerequisites

### System tools (must be on PATH)

```bash
# NCBI Datasets CLI
# Download from: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/

# SeqKit
# Download from: https://bioinf.shenwei.me/seqkit/

# TaxonKit
# Download from: https://bioinf.shenwei.me/taxonkit/

# MMseqs2
# Download from: https://github.com/soedinglab/MMseqs2
```

### Python dependencies

```bash
pip install duckdb pyarrow lancedb numpy umap-learn hdbscan \
            transformers torch accelerate \
            --break-system-packages
```

> **GPU recommended for Phase 4.** GenomeOcean is a 4B-parameter model.  
> On CPU alone, expect ~24h for 45M sequences.  A single A100/H100 reduces this to ~8h.

---

## Quick Start

```bash
cd expedia_tier2

# Full pipeline (automatically resumes on restart)
python orchestrator.py

# Resume after a failure
python orchestrator.py --resume

# Run phases selectively
python orchestrator.py --phases 4 5 6       # embed + index + cluster only
python orchestrator.py --phases 2 --force   # re-deduplicate

# Skip MMseqs2 for a faster first pass
python orchestrator.py --skip-mmseqs
```

---

## Phase Reference

### Phase 1 — Data Acquisition (`phase1_acquisition.py`)
Downloads 45.8M aquatic sequences from NCBI using the Datasets CLI dehydrated-package protocol.  
- Stage 1: `datasets download --dehydrated` → small metadata ZIP (~1-5 GB)  
- Stage 2: `unzip` → populates `data_report.jsonl` metadata ledger  
- Stage 3: `datasets rehydrate` → 250 GB FASTA, MD5-checked, fully restartable  

### Phase 2 — Deduplication (`phase2_deduplication.py`)
- **DuckDB** `GROUP BY TaxID` out-of-core SQL → one accession per species  
- **SeqKit** `grep --pattern-file` → stream-filters 250 GB without loading into RAM  
- **MMseqs2** `easy-linclust` at 95% identity → eradicates centroid bias  

### Phase 3 — Taxonomy (`phase3_taxonomy.py`)
- Parallel TaxonKit `reformat` in 50k-chunk sub-processes → 7-rank lineage for every TaxID  
- Handles merged/deleted nodes via TaxonKit's taxid-changelog repair  
- Output: `lineages_full.tsv` (accession, taxid, superkingdom → species)  

### Phase 4 — Embedding (`phase4_embedding.py`)
- Loads GenomeOcean (4B params, BPE tokenisation) from HuggingFace or local cache  
- Streams FASTA in mini-batches; mean-pools final attention layer → 1024D vectors  
- Checkpoints every 10,000 sequences; safe to interrupt and resume  

### Phase 5 — Vector Indexing (`phase5_indexing.py`)
- Ingests embeddings + taxonomy metadata into LanceDB in 50k-row batches  
- Builds **IVF-RQ** index: 8192 partitions, 1-bit RaBitQ quantisation  
- Full index (~6 GB) fits in OS page cache → sub-10ms query latency at 45M scale  

### Phase 6 — Clustering + PROTAX (`phase6_clustering.py`)
- L2 normalise → UMAP 10D (low-memory mode for 45M points) → HDBSCAN soft clustering  
- Extracts `probabilities_` attribute (λ-based) for every sequence  
- PROTAX k-NN: queries 15 nearest reference neighbours; plurality-votes each rank  
- Outputs JSONL with structured probabilistic taxonomy per sequence  

### Phase 7 — IPC Bridge (`phase7_ipc_bridge.py`)
- `ScienceBridge` (Science Kernel): serialises arrays to Arrow IPC → writes to `SharedMemory`  
- `DisplayBridge` (Display Kernel): maps `SharedMemory` → materialises numpy arrays, zero copy  
- Transfer time < 2ms for 500MB manifold (vs 4-8s for disk handshake in Tier 1)  

---

## Directory Layout

```
EXPEDIA_Data/
├── 01_raw_fasta/
│   ├── ncbi_dataset_dehydrated.zip
│   ├── ncbi_dataset/               ← rehydrated structure
│   └── aquatic_sequences_raw.fasta ← concatenated 250 GB
├── 02_dedup_fasta/
│   ├── keep_accessions.txt
│   ├── deduplicated_marine.fasta
│   └── representative_marine.fasta ← MMseqs2 output
├── 03_taxonomy/
│   └── lineages_full.tsv
├── 04_embeddings/
│   ├── embeddings.npy              ← (N, 1024) float32
│   └── embedding_accession_ids.txt
├── 05_lancedb/                     ← LanceDB database files
├── 06_clustering/
│   ├── umap_manifold.npy
│   ├── hdbscan_labels.npy
│   ├── hdbscan_probs.npy
│   └── ntu_taxonomy_assignments.jsonl
├── checkpoints/                    ← JSON state files per phase
├── resources/models/GenomeOcean-4B/
├── ipc_buffers/
└── logs/expedia_tier2.log
```

---

## Tuning Parameters

All parameters live in `pipeline_config.py`:

| Parameter | Default | Notes |
|---|---|---|
| `IVF_NUM_PARTITIONS` | 8192 | sqrt(45.8M) → 2^13 |
| `RABITQ_NUM_BITS` | 1 | 4096 B/vec → 136 B/vec |
| `EMBED_BATCH_SIZE` | 32 | Tune up if VRAM allows |
| `TAXON_CHUNK_SIZE` | 50,000 | IDs per TaxonKit subprocess |
| `HDBSCAN_SOFT_THRESHOLD` | 0.85 | Below = ambiguous clade flag |
| `PROTAX_K_NEIGHBORS` | 15 | Reference neighbours per query |
| `DUCKDB_MEMORY_LIMIT` | 4GB | Cap for DuckDB sort spill |

---

## JSON NumPy serialisation (legacy fix preserved)

The Tier 1 issue with NumPy arrays over JSON pipes is handled in Phase 7:  
all arrays are serialised as Apache Arrow IPC (binary, not JSON).  
For any residual JSON-RPC payloads, always call `.tolist()` before serialisation.
