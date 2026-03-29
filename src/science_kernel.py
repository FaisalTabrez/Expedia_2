import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

import hdbscan
import numpy as np
import polars as pl
import umap
from sklearn.preprocessing import normalize
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from src.bridge import BridgeServer
from src.taxonomy import calculate_consensus_lineage
from src.librarian import Librarian

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("science_kernel")

E_TEMP_DIR = Path("E:/EXPEDIA_Data/temp")
MANIFOLD_OUTPUT = E_TEMP_DIR / "manifold_latest.parquet"


class LocalEmbedder:
    """Locally frozen Nucleotide Transformer for inference."""
    def __init__(self):
        logger.info("Initializing LocalEmbedder...")
        model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        # Apply SwiGLU patch
        for layer in self.model.esm.encoder.layer:
            if hasattr(layer.intermediate, 'dense'):
                layer.intermediate.dense.out_features = 4096
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def embed(self, sequence: str, k: int = 6) -> List[float]:
        # 6-mer sliding window
        kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
        kmer_seq = " ".join(kmers)
        inputs = self.tokenizer([kmer_seq], return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask, 1)
            sum_mask = mask.sum(1)
            mean_pooled = sum_embeddings / torch.clamp(sum_mask, min=1e-9)
            
            pad_size = 768 - mean_pooled.size(1)
            padded = torch.nn.functional.pad(mean_pooled, (0, pad_size))
            return padded[0].cpu().numpy().tolist()


class AvalancheAnalyzer:
    """Avalanche analytics pipeline for 768D latent vectors."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def run(self, vectors: np.ndarray, accessions: List[str], kingdoms: List[str], outlier_scores_raw: List[float] = None) -> pl.DataFrame:
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
                "kingdom": kingdoms,
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
        # Initialize the LanceDB connection on Volume E:
        self.librarian = Librarian()
        # Initialize Local Embedder for Surveillance
        self.embedder = LocalEmbedder()

    async def fetch_manifold_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Samples 100,000 points from HNSW index, runs Avalanche, and writes to disk buffer.
        """
        logger.info("Fetching manifold data for UI...")
        sample_df = self.librarian.sample_data(limit=100000)
        
        if len(sample_df) == 0:
            return {"status": "error", "message": "No data found in LanceDB"}
            
        accessions = sample_df['accession'].tolist()
        
        # Parse kingdoms from taxonomy_path
        if 'taxonomy_path' in sample_df.columns:
            kingdoms = [tax.split(";")[0].strip() if tax and tax != "Unknown" else "Unknown" for tax in sample_df['taxonomy_path'].to_list()]
        else:
            kingdoms = ["Unknown"] * len(accessions)

        # Convert list of vectors to numpy array
        vectors = np.stack(sample_df['vector'].values).astype(np.float32)

        manifold_df = self.analyzer.run(vectors=vectors, accessions=accessions, kingdoms=kingdoms)
        manifold_df.write_parquet(MANIFOLD_OUTPUT)
        logger.info("Manifold output written to disk handshake buffer.")

        return {
            "status": "ok",
            "payload_path": str(MANIFOLD_OUTPUT),
            "row_count": manifold_df.height,
            "signal": "manifold_ready",
        }

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
        
        kingdoms = ["Unknown"] * len(accessions)
        if "taxonomy_path" in payload_df.columns:
            kingdoms = [tax.split(";")[0].strip() if tax and tax != "Unknown" else "Unknown" for tax in payload_df['taxonomy_path'].to_list()]
            
        vectors = np.stack(payload_df["vector"].to_list()).astype(np.float32)   

        manifold_df = self.analyzer.run(vectors=vectors, accessions=accessions, kingdoms=kingdoms)
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
        """
        Takes a query vector, calls Librarian.find_neighbors, and uses taxonomy math.
        """
        query_vector = params.get("query_vector")
        if query_vector is None:
            return {"status": "error", "message": "query_vector required"}
            
        # call Librarian to get neighbors
        neighbors = self.librarian.find_neighbors(query_vector, k=50)
        
        # calculate taxonomy status
        consensus = calculate_consensus_lineage(neighbors)
        return {"status": "ok", "consensus": consensus}

    async def analyze_surveillance_sequence(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        End-to-end local inference: sequence -> vector -> hnsw -> consensus.
        """
        sequence = params.get("sequence")
        if not sequence:
            return {"status": "error", "message": "sequence required"}
            
        logger.info("Tokenizing and embedding surveillance sequence locally...")
        try:
            vector = self.embedder.embed(sequence)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return {"status": "error", "message": f"Embedding failed: {e}"}
            
        logger.info("Searching LanceDB for nearest neighbors...")
        neighbors = self.librarian.find_neighbors(vector, k=50)
        
        logger.info("Computing taxonomic consensus...")
        consensus = calculate_consensus_lineage(neighbors)
        
        # Extract neighbor accessions for UI highlighting
        neighbor_accessions = [n.get("accession") for n in neighbors if "accession" in n]
        
        return {
            "status": "ok",
            "consensus": consensus,
            "neighbors": neighbor_accessions
        }

async def main() -> None:
    service = ScienceKernelService()
    server = BridgeServer(host="127.0.0.1", port=8899)
    server.register_method("ping", service.ping)
    server.register_method("fetch_manifold_data", service.fetch_manifold_data)  
    server.register_method("run_avalanche", service.run_avalanche)
    server.register_method("consensus_lineage", service.consensus_lineage)
    server.register_method("analyze_surveillance_sequence", service.analyze_surveillance_sequence)
