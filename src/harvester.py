import hashlib
import logging
from typing import List, Generator, Dict, Any
from Bio import Entrez, SeqIO
import polars as pl
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Harvester:
    """
    Agent 01: The Harvester
    Goal: Query NCBI for abyssal/deep-sea genomic sequences and perform cryptographic deduplication.
    """
    def __init__(self, email: str, db_path: str = "raw_sequences.parquet"):
        self.email = email
        self.db_path = db_path
        Entrez.email = self.email

    def query_ncbi(self, terms: List[str], retmax: int = 1000) -> Generator[Dict[str, Any], None, None]:
        """
        Query NCBI for given terms and yield records.
        Using ESearch and EFetch.
        """
        search_term = " OR ".join([f"{term}[All Fields]" for term in terms])
        logger.info(f"Searching NCBI for: {search_term}")
        
        try:
            handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=retmax, usehistory="y")
            search_results = Entrez.read(handle)
            handle.close()
            
            count = int(search_results["Count"])
            webenv = search_results["WebEnv"]
            query_key = search_results["QueryKey"]
            
            logger.info(f"Found {count} records. Fetching...")
            
            batch_size = 100
            for start in range(0, min(count, retmax), batch_size):
                end = min(count, start + batch_size)
                logger.info(f"Fetching records {start} to {end}...")
                
                fetch_handle = Entrez.efetch(
                    db="nucleotide",
                    rettype="fasta",
                    retmode="text",
                    retstart=start,
                    retmax=batch_size,
                    webenv=webenv,
                    query_key=query_key
                )
                
                for record in SeqIO.parse(fetch_handle, "fasta"):
                    yield {
                        "id": record.id,
                        "description": record.description,
                        "sequence": str(record.seq).upper()
                    }
                fetch_handle.close()
                
        except Exception as e:
            logger.error(f"Error querying NCBI: {e}")

    def deduplicate_and_save(self, records: Generator[Dict[str, Any], None, None]):
        """
        Consume the generator, compute SHA-256 hashes, deduplicate, and save to Parquet.
        """
        data = []
        for record in records:
            # Compute SHA-256 hash of the sequence for deduplication
            seq_hash = hashlib.sha256(record["sequence"].encode('utf-8')).hexdigest()
            record["seq_hash"] = seq_hash
            data.append(record)
            
        if not data:
            logger.warning("No data found to save.")
            return

        logger.info(f"Processing {len(data)} records for deduplication...")
        
        df = pl.DataFrame(data)
        
        # Deduplicate based on sequence hash
        unique_df = df.unique(subset=["seq_hash"])
        duplicate_count = len(df) - len(unique_df)
        
        logger.info(f"Removed {duplicate_count} duplicates. Saving {len(unique_df)} unique records.")
        
        # Save to Parquet
        unique_df.write_parquet(self.db_path)
        logger.info(f"Data saved to {self.db_path}")

if __name__ == "__main__":
    # Example usage
    agent = Harvester(email="your_email@example.com") # User should replace email
    keywords = ["Abyssal", "Hadopelagic", "Hydrothermal Vent"]
    
    # In a real scenario, retmax might be much higher or we'd stream to disk directly
    records_gen = agent.query_ncbi(keywords, retmax=500) 
    agent.deduplicate_and_save(records_gen)
