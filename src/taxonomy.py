from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

RANKS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

def parse_taxonomy_path(tax_path: str) -> Dict[str, str]:
    if not tax_path or tax_path == "Unknown":
        return {}
    parts = [p.strip() for p in tax_path.split(";")]
    # Assuming standard 7 ranks from the refiner matching our RANKS list. 
    # The taxonomy refiner goes superkingdom->species. Let's just zip.
    # To be safe, we pad or just map based on length. Usually length is 7 or 8.
    res = {}
    for i, rank in enumerate(RANKS):
        if i < len(parts):
            res[rank] = parts[i]
    return res

def calculate_consensus_lineage(neighbors: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not neighbors:
        return {"lineage": {}, "summary": {"classification": "NOVEL", "reason": "no_neighbors", "anomaly": "None"}}

    rank_scores: Dict[str, Dict[str, float]] = {rank: defaultdict(float) for rank in RANKS}

    phyla_detected = set()
    distances = []

    for item in neighbors:
        distance = float(item.get("_distance", 1.0))
        distances.append(distance)
        weight = 1.0 / (distance + 1e-6)
        
        tax_path = item.get("taxonomy_path", "")
        # fallback for old taxonomy dict if someone passes it
        if "taxonomy" in item and isinstance(item["taxonomy"], dict):
            taxonomy = item["taxonomy"]
        else:
            taxonomy = parse_taxonomy_path(tax_path)
            
        phylum = taxonomy.get("Phylum")
        if phylum:
            phyla_detected.add(phylum)

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
    classification = "POTENTIAL_UNSEQUENCED_ABYSSAL_SPECIES" if is_unsequenced_abyssal_species else "KNOWN_LINEAGE"

    # Anomaly checks
    anomaly = "None"
    
    # Taxonomic Conflict Metric
    if len(phyla_detected) > 3:
        anomaly = "CHIMERIC/HIGHLY DIVERGENT"
    
    # Isolation Metric
    # top 10 average
    top_10_dist = distances[:10]
    avg_top_10 = sum(top_10_dist) / len(top_10_dist) if top_10_dist else 0
    if avg_top_10 > 0.45:
        anomaly = "EXTREME NOVELTY"

    summary = {
        "classification": classification,
        "novel_rank_count": novel_flags,
        "anomaly": anomaly,
        "avg_top_10_distance": avg_top_10,
        "phyla_count": len(phyla_detected)
    }
    return {"lineage": lineage, "summary": summary}
