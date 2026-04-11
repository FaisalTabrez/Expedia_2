"""
EXPEDIA Tier 2 — Phase 4: GenomeOcean Neural Embedding
========================================================
Replaces the Nucleotide Transformer v2-50M (6-mer tokenisation, 512D output)
with GenomeOcean (4B parameters, BPE tokenisation, metagenomic pretrain).

Key improvements over Tier 1:
  • 220 TB metagenomic ocean/lake/soil pretrain — weights natively model dark taxa
  • BPE tokenisation: up to 150× throughput over k-mer sliding window
  • Adjusted Rand Index 0.86 species-level (vs ~0.4 for generic foundation models)
  • Mean-pooled final-layer embeddings → 1024-D vectors

The pipeline streams sequences in mini-batches and appends embeddings to a
memory-mapped NumPy array, checkpointing every EMBED_CHECKPOINT_N records.
This ensures safe resumption after any failure without re-processing sequences.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from pipeline_config import (
    DEDUP_FASTA_PATH,
    DIRS,
    EMBED_BATCH_SIZE,
    EMBED_CHECKPOINT,
    EMBED_CHECKPOINT_N,
    EMBED_DIMENSION,
    EMBED_IDS_TXT,
    EMBED_MAX_LENGTH,
    EMBED_OUTPUT_NPY,
    GENOME_OCEAN_LOCAL_DIR,
    GENOME_OCEAN_MODEL_ID,
    LOG_FILE,
    MMSEQS2_REP_FASTA,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("phase4.embedding")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_cp() -> dict:
    return json.loads(EMBED_CHECKPOINT.read_text()) if EMBED_CHECKPOINT.exists() else {}

def _save_cp(state: dict) -> None:
    EMBED_CHECKPOINT.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# FASTA streaming iterator
# ---------------------------------------------------------------------------

def _iter_fasta(fasta_path: Path, start_idx: int = 0) -> Iterator[tuple[str, str]]:
    """
    Yield (accession, sequence) tuples from a FASTA file.
    Skips the first `start_idx` records for checkpoint resumption.
    """
    acc: str | None  = None
    seq_parts: list[str] = []
    idx = 0

    with fasta_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if acc is not None:
                    if idx >= start_idx:
                        yield acc, "".join(seq_parts)
                    idx += 1
                    seq_parts = []
                acc = line[1:].split()[0]   # accession = first token after '>'
            else:
                seq_parts.append(line)

    # Flush last record
    if acc is not None and idx >= start_idx:
        yield acc, "".join(seq_parts)


def _count_fasta_records(fasta_path: Path) -> int:
    """Fast record count by counting '>' lines."""
    n = 0
    with fasta_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            n += chunk.count(b">")
    return n


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_genome_ocean(local_dir: Path = GENOME_OCEAN_LOCAL_DIR):
    """
    Load GenomeOcean from a local frozen copy or pull from HuggingFace once.
    Returns (model, tokenizer) on the best available device.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise EnvironmentError(
            "Install: pip install torch transformers --break-system-packages"
        )

    if local_dir.exists() and any(local_dir.iterdir()):
        log.info("Loading GenomeOcean from local: %s", local_dir)
        source = str(local_dir)
    else:
        log.info("Pulling GenomeOcean from HuggingFace: %s", GENOME_OCEAN_MODEL_ID)
        log.info("This is a 4B-parameter model — first download may take significant time.")
        source = GENOME_OCEAN_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    model     = AutoModel.from_pretrained(
        source,
        trust_remote_code=True,
        torch_dtype=torch.float16,   # half-precision to fit on edge GPU/CPU
    )

    # Choose device: CUDA > DirectML (Windows) > MPS (Apple) > CPU
    device = _best_device()
    model = model.to(device)
    model.eval()

    # Freeze weights — we only do inference, never fine-tune in this phase
    for param in model.parameters():
        param.requires_grad_(False)

    if local_dir != source:
        log.info("Caching model weights to: %s", local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(local_dir))
        model.save_pretrained(str(local_dir))

    log.info("GenomeOcean loaded on %s", device)
    return model, tokenizer, device


def _best_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Embedding loop
# ---------------------------------------------------------------------------

def _embed_batch(
    sequences: list[str],
    model,
    tokenizer,
    device: str,
    max_length: int = EMBED_MAX_LENGTH,
) -> np.ndarray:
    """
    Tokenise a mini-batch and extract mean-pooled last-layer embeddings.
    Returns float32 array of shape (batch_size, EMBED_DIMENSION).
    """
    import torch

    enc = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc, output_hidden_states=False)
        # last_hidden_state: (batch, seq_len, hidden_dim)
        hidden = out.last_hidden_state
        # Mask padding tokens before mean pooling
        mask   = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

    return pooled.cpu().float().numpy()


def stream_embed_fasta(
    fasta_path: Path,
    model,
    tokenizer,
    device: str,
    batch_size: int = EMBED_BATCH_SIZE,
    checkpoint_n: int = EMBED_CHECKPOINT_N,
    output_npy: Path = EMBED_OUTPUT_NPY,
    output_ids: Path = EMBED_IDS_TXT,
    start_from: int = 0,
) -> int:
    """
    Stream all sequences from fasta_path through GenomeOcean in mini-batches.
    Appends embeddings to a memory-mapped NumPy file and accession IDs to a
    text file.  Checkpoints every `checkpoint_n` records.

    Returns total number of sequences embedded.
    """
    log.info("Streaming embeddings from: %s", fasta_path)
    log.info("Output arrays: %s  /  %s", output_npy, output_ids)

    n_total = _count_fasta_records(fasta_path)
    log.info("Total records to embed: %d (starting at index %d)", n_total, start_from)

    # Open ID file in append mode for resumption
    ids_mode = "a" if start_from > 0 else "w"

    batch_accs:  list[str] = []
    batch_seqs:  list[str] = []
    all_embeddings: list[np.ndarray] = []
    processed = start_from

    t0 = time.monotonic()

    with output_ids.open(ids_mode, encoding="utf-8") as id_fh:
        for acc, seq in _iter_fasta(fasta_path, start_idx=start_from):
            batch_accs.append(acc)
            batch_seqs.append(seq)

            if len(batch_seqs) >= batch_size:
                vecs = _embed_batch(batch_seqs, model, tokenizer, device)
                all_embeddings.append(vecs)
                id_fh.write("\n".join(batch_accs) + "\n")
                processed += len(batch_accs)
                batch_accs = []
                batch_seqs = []

                if processed % checkpoint_n < batch_size:
                    _flush_embeddings(all_embeddings, output_npy, processed, start_from)
                    all_embeddings = []
                    elapsed = time.monotonic() - t0
                    rate = (processed - start_from) / max(elapsed, 1)
                    eta  = (n_total - processed) / max(rate, 1)
                    log.info(
                        "Embedded %d / %d  (%.0f seq/s  ETA %.0f min)",
                        processed, n_total, rate, eta / 60
                    )
                    cp = _load_cp()
                    cp["last_index"] = processed
                    _save_cp(cp)

        # Flush final partial batch
        if batch_seqs:
            vecs = _embed_batch(batch_seqs, model, tokenizer, device)
            all_embeddings.append(vecs)
            id_fh.write("\n".join(batch_accs) + "\n")
            processed += len(batch_accs)

    # Final flush
    if all_embeddings:
        _flush_embeddings(all_embeddings, output_npy, processed, start_from)

    elapsed = time.monotonic() - t0
    log.info(
        "Embedding complete: %d sequences in %.1f min (%.0f seq/s)",
        processed, elapsed / 60, processed / max(elapsed, 1)
    )
    return processed


def _flush_embeddings(
    chunks: list[np.ndarray],
    output_npy: Path,
    n_written: int,
    start_from: int,
) -> None:
    """
    Append accumulated embedding chunks to the output .npy file.
    Uses numpy save/concatenate; for production consider HDF5 or Zarr
    to support true random-access appends.
    """
    if not chunks:
        return
    new_vecs = np.concatenate(chunks, axis=0)

    if output_npy.exists() and start_from > 0:
        existing = np.load(output_npy, mmap_mode="r")
        combined = np.concatenate([existing, new_vecs], axis=0)
    else:
        combined = new_vecs

    # Atomic write: save to temp then rename
    tmp = output_npy.with_suffix(".tmp")
    np.save(tmp, combined)
    tmp.rename(output_npy)
    log.debug("Flushed embeddings → %s  shape=%s", output_npy, combined.shape)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_embedding(
    fasta_path: Path | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """
    Run Phase 4: embed all sequences with GenomeOcean.

    Args:
        fasta_path: Path to representative FASTA (Phase 2/3 output).
        force:      Re-embed from scratch even if checkpoint exists.

    Returns (embeddings_npy_path, ids_txt_path).
    """
    cp = _load_cp()
    if not force and cp.get("done"):
        log.info("Phase 4 already complete.  Skipping.")
        return EMBED_OUTPUT_NPY, EMBED_IDS_TXT

    # Resolve input FASTA
    if fasta_path is None:
        fasta_path = MMSEQS2_REP_FASTA if MMSEQS2_REP_FASTA.exists() else DEDUP_FASTA_PATH
    if not fasta_path.exists():
        raise FileNotFoundError(f"Representative FASTA not found: {fasta_path}")

    # Resume from checkpoint
    start_from = cp.get("last_index", 0) if not force else 0
    if start_from > 0:
        log.info("Resuming from sequence index %d", start_from)

    model, tokenizer, device = load_genome_ocean()

    n_embedded = stream_embed_fasta(
        fasta_path=fasta_path,
        model=model,
        tokenizer=tokenizer,
        device=device,
        start_from=start_from,
    )

    cp["done"]      = True
    cp["n_embedded"] = n_embedded
    cp["embed_dim"]  = EMBED_DIMENSION
    _save_cp(cp)

    log.info(
        "Phase 4 complete: %d embeddings  →  %s",
        n_embedded, EMBED_OUTPUT_NPY
    )
    return EMBED_OUTPUT_NPY, EMBED_IDS_TXT


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_embedding(force=args.force)
