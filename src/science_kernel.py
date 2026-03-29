import os
import lancedb
import numpy as np
import umap
import hdbscan
from sklearn.preprocessing import normalize
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ScienceKernel")

class ScienceKernel:
    """
    Encapsulates the high-parameter Nucleotide Transformer model (embeddings managed externally),
    LanceDB interface, and manifold learning libraries (UMAP/HDBSCAN).
    """
    import asyncio
    import logging
    from pathlib import Path
    from typing import Any, Dict, List

    import hdbscan
    import numpy as np
    import polars as pl
    import umap
    from sklearn.preprocessing import normalize

    from src.bridge import BridgeServer
    from src.taxonomy import calculate_consensus_lineage

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("science_kernel")

    E_TEMP_DIR = Path("E:/EXPEDIA_Data/temp")
    MANIFOLD_OUTPUT = E_TEMP_DIR / "manifold_latest.parquet"


    class AvalancheAnalyzer:
        """Avalanche analytics pipeline for 768D latent vectors."""

        def __init__(self, random_state: int = 42) -> None:
            self.random_state = random_state

        def run(self, vectors: np.ndarray, accessions: List[str]) -> pl.DataFrame:
            if vectors.ndim != 2 or vectors.shape[1] != 768:
                raise ValueError("Expected latent vectors with shape (N, 768)")
            if len(accessions) != vectors.shape[0]:
                raise ValueError("accessions length must match vectors row count")

            logger.info("Avalanche Step 1/4: L2 normalization")
            norm_vectors = normalize(vectors, norm="l2")

            logger.info("Avalanche Step 2/4: UMAP to 10D")
            reducer_10d = umap.UMAP(
                n_components=10,
                low_memory=True,
                random_state=self.random_state,
                n_neighbors=30,
                min_dist=0.0,
                metric="cosine",
            )
            embedding_10d = reducer_10d.fit_transform(norm_vectors)

            logger.info("Avalanche Step 3/4: UMAP to 3D")
            reducer_3d = umap.UMAP(
                n_components=3,
                low_memory=True,
                random_state=self.random_state,
                n_neighbors=30,
                min_dist=0.0,
                metric="cosine",
            )
            embedding_3d = reducer_3d.fit_transform(norm_vectors)

            logger.info("Avalanche Step 4/4: HDBSCAN on 10D")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=30,
                min_samples=10,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )
            cluster_labels = clusterer.fit_predict(embedding_10d)
            outlier_scores = clusterer.outlier_scores_

            manifold_df = pl.DataFrame(
                {
                    "accession": accessions,
                    "x": embedding_3d[:, 0],
                    "y": embedding_3d[:, 1],
                    "z": embedding_3d[:, 2],
                    "cluster_id": cluster_labels,
                    "outlier_score": outlier_scores,
                }
            )
            return manifold_df


    class ScienceKernelService:
        """Science-side service exposed via BridgeServer JSON-RPC."""

        def __init__(self) -> None:
            self.analyzer = AvalancheAnalyzer()
            E_TEMP_DIR.mkdir(parents=True, exist_ok=True)

        async def run_avalanche(self, params: Dict[str, Any]) -> Dict[str, Any]:
            payload_df: pl.DataFrame
            if "payload_df" in params and isinstance(params["payload_df"], pl.DataFrame):
                payload_df = params["payload_df"]
            else:
                vectors = np.asarray(params.get("vectors", []), dtype=np.float32)
                accessions = params.get("accessions", [])
                payload_df = pl.DataFrame({"accession": accessions, "vector": vectors.tolist()})

            if payload_df.is_empty():
                raise ValueError("No vectors received for avalanche pipeline")

            accessions = payload_df["accession"].to_list()
            vectors = np.asarray(payload_df["vector"].to_list(), dtype=np.float32)

            manifold_df = self.analyzer.run(vectors=vectors, accessions=accessions)
            manifold_df.write_parquet(MANIFOLD_OUTPUT)
            logger.info("Saved manifold output to %s", MANIFOLD_OUTPUT)

            return {
                "status": "ok",
                "payload_path": str(MANIFOLD_OUTPUT),
                "row_count": manifold_df.height,
                "signal": "manifold_ready",
            }

        async def ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
            return {"status": "alive", "echo": params}

        async def consensus_lineage(self, params: Dict[str, Any]) -> Dict[str, Any]:
            neighbors = params.get("neighbors", [])
            return calculate_consensus_lineage(neighbors)


    async def main() -> None:
        service = ScienceKernelService()
        server = BridgeServer(host="127.0.0.1", port=8899)
        server.register_method("ping", service.ping)
        server.register_method("run_avalanche", service.run_avalanche)
        server.register_method("consensus_lineage", service.consensus_lineage)
        await server.serve_forever()


    if __name__ == "__main__":
        asyncio.run(main())
