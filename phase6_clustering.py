"""
EXPEDIA Tier 2 — Phase 6: Avalanche Clustering + PROTAX Probabilistic Taxonomy
===============================================================================
Implements the "Avalanche Standard" pipeline:

  1. L2-normalise all embedding vectors (ensures cosine geometry)
  2. UMAP dimensionality reduction  (1024D → 10D manifold)
  3. HDBSCAN soft clustering         (NTU discovery + probability scores)
  4. PROTAX-inspired probabilistic   (k-NN genealogy → Bayesian tax. assignment)
     taxonomy assignment

Tier 1 limitation resolved:
  • Tier 1 used DETERMINISTIC cluster labelling → accurate only at Phylum/Class
  • Tier 2 HDBSCAN soft clustering exposes per-sequence cluster membership
    probabilities (P(p∈C) formula from the condensed tree λ values)
  • PROTAX multinomial regression then resolves fine-grained genus/species
    level with explicit uncertainty quantification

Every sequence receives a full structured taxonomy output:
  {
    "accession": "NZ_CP012345",
    "cluster_id": 42,
    "cluster_probability": 0.96,
    "taxonomy": {
      "superkingdom": {"name": "Bacteria",     "confidence": 0.99},
      "phylum":       {"name": "Proteobacteria","confidence": 0.97},
      "class":        {"name": "Gammaproteobacteria","confidence":0.93},
      "order":        {"name": "Vibrionales",  "confidence": 0.88},
      "family":       {"name": "Vibrionaceae", "confidence": 0.84},
      "genus":        {"name": "Unknown",      "confidence": 0.72},
      "species":      {"name": "Unknown",      "confidence": 0.0},
    }
  }
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline_config import (
    CLUSTER_CHECKPOINT,
    CLUSTER_OUTPUT_DIR,
    DIRS,
    EMBED_IDS_TXT,
    EMBED_OUTPUT_NPY,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    HDBSCAN_PREDICTION_DATA,
    HDBSCAN_SOFT_THRESHOLD,
    LANCEDB_PATH,
    LANCEDB_TABLE_NAME,
    LOG_FILE,
    PROTAX_CONF_THRESHOLD,
    PROTAX_K_NEIGHBORS,
    TAXONOMY_RANKS,
    TAXONOMY_TSV,
    UMAP_LOW_MEMORY,
    UMAP_METRIC,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("phase6.clustering")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_cp() -> dict:
    return json.loads(CLUSTER_CHECKPOINT.read_text()) if CLUSTER_CHECKPOINT.exists() else {}

def _save_cp(state: dict) -> None:
    CLUSTER_CHECKPOINT.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Step 1 — L2 normalisation
# ---------------------------------------------------------------------------

def l2_normalise(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalise each row vector to unit sphere.
    Ensures Euclidean distance in the normalised space equals cosine distance.
    Operates in-place on a copy to avoid mutating the memory-mapped source.
    """
    log.info("L2-normalising %s embeddings …", embeddings.shape)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid division by zero
    return (embeddings / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 2 — UMAP dimensionality reduction
# ---------------------------------------------------------------------------

def fit_umap(
    vectors: np.ndarray,
    n_components: int = UMAP_N_COMPONENTS,
    n_neighbors:  int = UMAP_N_NEIGHBORS,
    min_dist:    float = UMAP_MIN_DIST,
    metric:       str = UMAP_METRIC,
    low_memory:  bool = UMAP_LOW_MEMORY,
    output_path: Path | None = None,
) -> tuple[object, np.ndarray]:
    """
    Fit UMAP on vectors and return (reducer, manifold).

    For 45.8M points, low_memory=True uses approximate nearest-neighbour
    graph construction (pynndescent) with a disk-backed index — dramatically
    reducing peak RAM usage at the cost of ~5% recall.

    Returns the fitted UMAP reducer (for transform on new points) and the
    10D manifold coordinates.
    """
    try:
        import umap
    except ImportError:
        raise EnvironmentError(
            "Install: pip install umap-learn --break-system-packages"
        )

    log.info(
        "Fitting UMAP: %d points  %dD → %dD  (metric=%s  low_memory=%s)",
        vectors.shape[0], vectors.shape[1], n_components, metric, low_memory
    )
    t0 = time.monotonic()

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        low_memory=low_memory,
        verbose=True,
        random_state=42,
    )
    manifold = reducer.fit_transform(vectors)
    elapsed  = time.monotonic() - t0
    log.info(
        "UMAP complete: shape %s  in %.1f min", manifold.shape, elapsed / 60
    )

    if output_path is not None:
        np.save(output_path, manifold)
        log.info("Manifold saved: %s", output_path)

    return reducer, manifold


def transform_umap(reducer, new_vectors: np.ndarray) -> np.ndarray:
    """Embed new sequences into an existing UMAP manifold (inference only)."""
    return reducer.transform(l2_normalise(new_vectors))


# ---------------------------------------------------------------------------
# Step 3 — HDBSCAN soft clustering
# ---------------------------------------------------------------------------

def fit_hdbscan(
    manifold: np.ndarray,
    min_cluster_size: int   = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples:      int   = HDBSCAN_MIN_SAMPLES,
    prediction_data:  bool  = HDBSCAN_PREDICTION_DATA,
    output_dir:       Path  = CLUSTER_OUTPUT_DIR,
) -> tuple[object, np.ndarray, np.ndarray]:
    """
    Fit HDBSCAN on the 10D UMAP manifold.

    Returns (clusterer, labels, soft_probabilities).

    soft_probabilities[i] = P(p_i ∈ C_i) computed from the λ values of the
    condensed cluster tree:
        P(p ∈ C) = (λ_p - λ_birth) / (λ_death - λ_birth)

    Sequences with P < HDBSCAN_SOFT_THRESHOLD are flagged as ambiguous clades.
    """
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        raise EnvironmentError(
            "Install: pip install hdbscan --break-system-packages"
        )

    log.info(
        "Fitting HDBSCAN: min_cluster_size=%d  min_samples=%d  prediction_data=%s",
        min_cluster_size, min_samples, prediction_data
    )
    t0 = time.monotonic()

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=prediction_data,
        core_dist_n_jobs=-1,         # use all CPUs
        gen_min_span_tree=False,     # save memory at scale
    )
    labels = clusterer.fit_predict(manifold)
    elapsed = time.monotonic() - t0

    # Soft clustering: extract per-point cluster membership probabilities
    soft_probs = clusterer.probabilities_.copy()

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    log.info(
        "HDBSCAN complete in %.1f min: %d NTU clusters, %d noise points",
        elapsed / 60, n_clusters, n_noise
    )

    # Persist
    np.save(output_dir / "hdbscan_labels.npy",    labels)
    np.save(output_dir / "hdbscan_probs.npy",     soft_probs)
    np.save(output_dir / "umap_manifold.npy",     manifold)
    log.info("Cluster arrays saved to %s", output_dir)

    return clusterer, labels, soft_probs


# ---------------------------------------------------------------------------
# Step 4 — PROTAX probabilistic taxonomy assignment
# ---------------------------------------------------------------------------

def _majority_vote(
    neighbor_taxonomies: list[dict],
    ranks: list[str] = TAXONOMY_RANKS,
) -> dict[str, dict]:
    """
    Given the taxonomic annotations of k nearest reference neighbours,
    compute a plurality-vote confidence for each rank.

    Returns dict: {rank: {"name": str, "confidence": float}}
    """
    result: dict[str, dict] = {}
    for rank in ranks:
        values  = [n.get(rank, "") for n in neighbor_taxonomies if n.get(rank)]
        if not values:
            result[rank] = {"name": "Unknown", "confidence": 0.0}
            continue
        counts  = Counter(values)
        best    = counts.most_common(1)[0]
        name, cnt = best
        conf    = cnt / len(values)
        result[rank] = {"name": name, "confidence": round(conf, 4)}
    return result


def _load_taxonomy_lookup(tsv_path: Path) -> dict[str, dict]:
    """Read lineages_full.tsv into {accession: {rank: value}}."""
    import csv
    index: dict[str, dict] = {}
    with tsv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            index[row["accession"]] = dict(row)
    return index


def protax_assign(
    query_acc:        str,
    query_vector:     np.ndarray,
    cluster_id:       int,
    cluster_prob:     float,
    taxonomy_lookup:  dict[str, dict],
    k:                int  = PROTAX_K_NEIGHBORS,
    conf_threshold:   float = PROTAX_CONF_THRESHOLD,
    db_path:          Path  = LANCEDB_PATH,
    table_name:       str   = LANCEDB_TABLE_NAME,
) -> dict:
    """
    PROTAX-inspired taxonomic classification for one sequence.

    Algorithm:
      1. Query LanceDB for k nearest reference neighbours that have ground-truth
         TaxIDs (populated from TaxonKit in Phase 3).
      2. Compute plurality-vote taxonomy at each rank.
      3. If cluster_prob < HDBSCAN_SOFT_THRESHOLD, flag the whole assignment
         as AMBIGUOUS and cap genus/species confidence at 0.
      4. Apply the PROTAX confidence floor: if family-level agreement ≥
         conf_threshold but genus-level splits, report genus as Unknown.
    """
    from phase5_indexing import query_knn

    # Get k nearest neighbours from vector DB
    neighbours = query_knn(query_vector, k=k, db_path=db_path, table_name=table_name)

    # Resolve full taxonomy for each neighbour
    neighbour_tax: list[dict] = []
    for nb in neighbours:
        acc = nb.get("accession", "")
        t   = taxonomy_lookup.get(acc, {})
        if t:
            neighbour_tax.append(t)
        else:
            # Fallback: use LanceDB row's inline taxonomy fields
            neighbour_tax.append({r: nb.get(r, "") for r in TAXONOMY_RANKS})

    ranked = _majority_vote(neighbour_tax)

    # Apply PROTAX family-confirmation rule
    family_conf = ranked.get("family", {}).get("confidence", 0.0)
    genus_conf  = ranked.get("genus",  {}).get("confidence", 0.0)
    if family_conf >= conf_threshold and genus_conf < 0.5:
        ranked["genus"]   = {"name": "Unknown", "confidence": genus_conf}
        ranked["species"] = {"name": "Unknown", "confidence": 0.0}

    # Cap confidences if cluster membership is weak
    is_ambiguous = cluster_prob < HDBSCAN_SOFT_THRESHOLD
    if is_ambiguous:
        for rank in ["genus", "species"]:
            ranked[rank] = {"name": "Unknown (ambiguous clade)", "confidence": 0.0}

    return {
        "accession":           query_acc,
        "cluster_id":          int(cluster_id),
        "cluster_probability": round(float(cluster_prob), 4),
        "is_ambiguous_clade":  is_ambiguous,
        "taxonomy":            ranked,
        "n_reference_neighbours": len(neighbour_tax),
    }


# ---------------------------------------------------------------------------
# Batch taxonomy assignment
# ---------------------------------------------------------------------------

def assign_taxonomy_batch(
    accession_ids: list[str],
    embeddings:    np.ndarray,
    labels:        np.ndarray,
    soft_probs:    np.ndarray,
    taxonomy_lookup: dict[str, dict],
    output_path:   Path,
    log_every:     int = 50_000,
) -> int:
    """
    Assign PROTAX-style probabilistic taxonomy to all sequences.
    Writes results to a JSON-Lines file (one record per line).
    Returns number of records written.
    """
    log.info("Assigning probabilistic taxonomy to %d sequences …", len(accession_ids))
    t0 = time.monotonic()
    written = 0

    with output_path.open("w", encoding="utf-8") as fh:
        for i, (acc, label, prob) in enumerate(
            zip(accession_ids, labels, soft_probs)
        ):
            result = protax_assign(
                query_acc=acc,
                query_vector=embeddings[i],
                cluster_id=int(label),
                cluster_prob=float(prob),
                taxonomy_lookup=taxonomy_lookup,
            )
            fh.write(json.dumps(result) + "\n")
            written += 1

            if written % log_every == 0:
                elapsed = time.monotonic() - t0
                rate    = written / max(elapsed, 1)
                log.info(
                    "PROTAX: %d / %d assigned  (%.0f seq/s)",
                    written, len(accession_ids), rate
                )

    log.info("Taxonomy assignment complete: %d records → %s", written, output_path)
    return written


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_clustering(
    embeddings_npy: Path = EMBED_OUTPUT_NPY,
    ids_txt:        Path = EMBED_IDS_TXT,
    taxonomy_tsv:   Path = TAXONOMY_TSV,
    force: bool = False,
) -> Path:
    """
    Execute the full Avalanche + PROTAX pipeline.
    Returns path to the JSONL taxonomy assignments file.
    """
    cp = _load_cp()
    output_jsonl = CLUSTER_OUTPUT_DIR / "ntu_taxonomy_assignments.jsonl"

    if not force and cp.get("done") and output_jsonl.exists():
        log.info("Phase 6 already complete.  Skipping.")
        return output_jsonl

    # Load embeddings (memory-mapped)
    log.info("Loading embeddings: %s", embeddings_npy)
    embeddings = np.load(embeddings_npy, mmap_mode="r")

    accession_ids = ids_txt.read_text(encoding="utf-8").splitlines()
    accession_ids = [a.strip() for a in accession_ids if a.strip()]
    log.info("Sequences: %d", len(accession_ids))

    # Step 1: L2 normalise
    normalised = l2_normalise(embeddings[:])   # copy from mmap for UMAP

    # Step 2: UMAP
    umap_path = CLUSTER_OUTPUT_DIR / "umap_manifold.npy"
    if not force and umap_path.exists():
        log.info("Loading existing UMAP manifold: %s", umap_path)
        manifold = np.load(umap_path)
        reducer  = None
    else:
        reducer, manifold = fit_umap(normalised, output_path=umap_path)

    cp["umap_done"] = True
    _save_cp(cp)

    # Step 3: HDBSCAN soft clustering
    label_path = CLUSTER_OUTPUT_DIR / "hdbscan_labels.npy"
    prob_path  = CLUSTER_OUTPUT_DIR / "hdbscan_probs.npy"
    if not force and label_path.exists() and prob_path.exists():
        log.info("Loading existing HDBSCAN results.")
        labels     = np.load(label_path)
        soft_probs = np.load(prob_path)
        clusterer  = None
    else:
        clusterer, labels, soft_probs = fit_hdbscan(manifold)

    n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    cp["hdbscan_done"]  = True
    cp["n_ntus"]        = n_clusters
    _save_cp(cp)

    # Step 4: PROTAX taxonomy assignment
    taxonomy_lookup = _load_taxonomy_lookup(taxonomy_tsv) if taxonomy_tsv.exists() else {}
    written = assign_taxonomy_batch(
        accession_ids=accession_ids,
        embeddings=embeddings,
        labels=labels,
        soft_probs=soft_probs,
        taxonomy_lookup=taxonomy_lookup,
        output_path=output_jsonl,
    )

    cp["done"]          = True
    cp["n_assignments"] = written
    _save_cp(cp)

    log.info(
        "Phase 6 complete.  %d NTU clusters, %d assignments → %s",
        n_clusters, written, output_jsonl
    )
    return output_jsonl


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_clustering(force=args.force)
