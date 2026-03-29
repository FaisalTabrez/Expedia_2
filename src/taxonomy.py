from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


RANKS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]


def calculate_consensus_lineage(neighbors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute weighted consensus lineage from nearest neighbors.

    Expected neighbor item format:
    {
        "distance": float,
        "taxonomy": {
            "Kingdom": "...",
            "Phylum": "...",
            ...
        }
    }
    """
    if not neighbors:
        return {"lineage": {}, "summary": {"classification": "NOVEL", "reason": "no_neighbors"}}

    rank_scores: Dict[str, Dict[str, float]] = {rank: defaultdict(float) for rank in RANKS}

    for item in neighbors:
        distance = float(item.get("distance", 1.0))
        weight = 1.0 / (distance + 1e-6)
        taxonomy = item.get("taxonomy", {})
        for rank in RANKS:
            taxon = taxonomy.get(rank)
            if taxon:
                rank_scores[rank][taxon] += weight

    lineage: Dict[str, Any] = {}
    novel_flags = 0
    for rank in RANKS:
        candidates = rank_scores[rank]
        if not candidates:
            lineage[rank] = {
                "taxon": "Unknown",
                "confidence": 0.0,
                "status": "NOVEL",
            }
            novel_flags += 1
            continue

        total = sum(candidates.values())
        top_taxon, top_weight = max(candidates.items(), key=lambda x: x[1])
        confidence = top_weight / total if total > 0 else 0.0

        if confidence > 0.95:
            status = "CONFIRMED"
        elif confidence >= 0.85:
            status = "DIVERGENT"
        else:
            status = "NOVEL"
            novel_flags += 1

        lineage[rank] = {
            "taxon": top_taxon,
            "confidence": confidence,
            "status": status,
        }

    is_unsequenced_abyssal_species = lineage["Species"]["status"] == "NOVEL" or novel_flags >= 3
    summary = {
        "classification": "POTENTIAL_UNSEQUENCED_ABYSSAL_SPECIES" if is_unsequenced_abyssal_species else "KNOWN_LINEAGE",
        "novel_rank_count": novel_flags,
    }
    return {"lineage": lineage, "summary": summary}
