import logging
from pathlib import Path
from typing import Any, Dict, List
import lancedb
import pyarrow as pa
import polars as pl
import numpy as np

logger = logging.getLogger("librarian")

DB_URI = "E:/EXPEDIA_Data/lancedb"

class Librarian:
    def __init__(self, db_uri: str = DB_URI):
        self.db_uri = db_uri
        # Make sure directory exists
        Path(self.db_uri).parent.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(self.db_uri)
        self.table_name = "genomic_embeddings"
        
        # Define Schema
        self.schema = pa.schema([
            pa.field("accession", pa.string()),
            pa.field("taxonomy_id", pa.int64()),
            pa.field("taxonomy_path", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 768))
        ])
        
        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=self.schema)
        else:
            self.table = self.db.open_table(self.table_name)
            
    def ingest_colab_vectors(self, parquet_path: str):
        """Reads embedded Parquet files and appends them to the table."""
        logger.info(f"Ingesting vectors from {parquet_path}")
        # LanceDB directly supports ingesting pyarrow/polars/pandas. 
        # Using polars to read parquet
        df = pl.read_parquet(parquet_path)
        self.table.add(df.to_arrow())
        logger.info(f"Successfully ingested from {parquet_path}. Total rows: {self.table.count_rows()}")
        
    def create_hnsw_index(self):
        """Creates an HNSW index optimally scaled for ~3.8M vectors."""
        logger.info("Creating HNSW index...")
        self.table.create_index(
            metric="cosine",
            vector_column_name="vector",
            num_partitions=1024,
            num_sub_vectors=96
        )
        logger.info("HNSW index created successfully.")
        
    def find_neighbors(self, query_vector: List[float], k: int = 50) -> List[Dict[str, Any]]:
        """Finds k nearest neighbors for a query vector."""
        # query_vector should be 768D
        results = self.table.search(query_vector).metric("cosine").limit(k).to_list()
        return results
        
    def sample_data(self, limit: int = 100000) -> pl.DataFrame:
        """Helper to sample data for the Avalanche pipeline to keep UI fluid."""
        # For a massive database, random sampling might require specific strategies,
        # but in LanceDB we can fetch a chunk or use search if we had random vectors.
        # Alternatively, just fetch the first `limit` rows (or random sequence).
        # We'll do a simple select limit:
        logger.info(f"Sampling {limit} rows from LanceDB...")
        if self.table.count_rows() == 0:
            logger.warning("Table is empty. Returning empty DataFrame.")
            return pl.DataFrame({"accession": [], "vector": []})
        
        # simple limit
        data = self.table.head(limit).to_pandas()
        return pl.DataFrame(data)

