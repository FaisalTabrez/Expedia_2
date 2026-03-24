import os
import re
import time
import logging
import concurrent.futures
import polars as pl
from Bio import Entrez, SeqIO
from io import StringIO
from pathlib import Path
from threading import Lock
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
from urllib.error import HTTPError
import http.client

# Configuration
API_KEY = "6e6e31ff83e545b01c27d3ca85876e515908"
EMAIL = "faisaltabrez01@gmail.com"
Entrez.email = EMAIL
Entrez.api_key = API_KEY

# Directories
PRIMARY_DATA_DIR = Path("E:/EXPEDIA_Data/chunks")
FALLBACK_DATA_DIR = Path("data/chunks")

# Architecture Constants
BATCH_SIZE = 500         # Records per HTTP request
CHUNK_SIZE = 10000       # Records per parquet file
MAX_WORKERS = 10         # Thread pool size

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("harvester_extreme.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Harvester")

class Harvester:
    def __init__(self):
        # Setup Data Directory
        try:
            if not os.path.exists(PRIMARY_DATA_DIR.parent):
                os.makedirs(PRIMARY_DATA_DIR.parent, exist_ok=True)
            os.makedirs(PRIMARY_DATA_DIR, exist_ok=True)
            self.data_dir = PRIMARY_DATA_DIR
            logger.info(f"Using Primary Data Directory: {self.data_dir}")
        except OSError:
            logger.warning(f"Could not access {PRIMARY_DATA_DIR}. Switching to fallback: {FALLBACK_DATA_DIR}")
            self.data_dir = FALLBACK_DATA_DIR
            os.makedirs(self.data_dir, exist_ok=True)
            
        self.checkpoint_file = self.data_dir.parent / "checkpoint.txt"
        self.webenv = None
        self.query_key = None
        self.total_count = 0

    def get_start_index(self):
        if self.checkpoint_file.exists():
            try:
                return int(self.checkpoint_file.read_text().strip())
            except ValueError:
                return 0
        return 0

    def update_checkpoint(self, index):
        self.checkpoint_file.write_text(str(index))

    def search(self, query):
        """Perform initial search with history to get WebEnv."""
        logger.info(f"Executing search: {query}")
        try:
            handle = Entrez.esearch(
                db="nucleotide",
                term=query,
                usehistory="y",
                retmax=0
            )
            results = Entrez.read(handle)
            handle.close()
            self.webenv = results["WebEnv"]
            self.query_key = results["QueryKey"]
            self.total_count = int(results["Count"])
            logger.info(f"Search complete. Found {self.total_count} records. WebEnv: {self.webenv}")
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    @retry(
        retry=retry_if_exception_type((http.client.IncompleteRead, HTTPError, RuntimeError, ConnectionError)),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def fetch_batch_data(self, start_idx, count):
        """Fetch raw FASTA data with advanced retry logic."""
        handle = Entrez.efetch(
            db="nuccore",
            query_key=self.query_key,
            WebEnv=self.webenv,
            retstart=start_idx,
            retmax=count,
            rettype="fasta",
            retmode="text"
        )
        return handle.read()

    def process_batch(self, start_index):
        """Worker function to process a batch of records."""
        batch_records = []
        try:
            raw_fasta = self.fetch_batch_data(start_index, BATCH_SIZE)
            
            if not raw_fasta:
                return batch_records

            # Parse FASTA
            for record in SeqIO.parse(StringIO(raw_fasta), "fasta"):
                # 1. Undefined / None check
                if record.seq is None:
                    continue
                
                seq_str = str(record.seq).upper()
                
                # 2. Length Check (< 100bp)
                if len(seq_str) < 100:
                    continue
                
                # 3. 'N' Content Check (> 10%)
                n_count = seq_str.count('N')
                if (n_count / len(seq_str)) > 0.10:
                    continue

                # Header Parsing Logic
                # Expected format: >Accession [Organism] [TaxID]
                # Default SeqIO puts everything after Accession into record.description
                
                description = record.description
                
                # Extract Organism
                # Look for [Organism Name]
                organism_match = re.search(r'\[([^\]]+)\]', description)
                organism = organism_match.group(1) if organism_match else "Unknown"
                
                # Extract TaxID if present in header (often as taxon:123 or [TaxID=123])
                # We try both common NCBI formats if not explicitly bracketed as [TaxID]
                taxid_match = re.search(r'taxon:(\d+)|TaxID=(\d+)|\[TaxID=(\d+)\]', description)
                taxid = "Unknown"
                if taxid_match:
                    # extensive check which group matched
                    items = [g for g in taxid_match.groups() if g]
                    if items:
                        taxid = items[0]

                batch_records.append({
                    "Accession": record.id,
                    "Organism": organism,
                    "TaxID": taxid,
                    "Sequence": seq_str
                })

        except Exception as e:
            logger.error(f"Error processing batch {start_index}: {e}")
            # Tenacity handles retries, so if we are here, it's a critical failure or parse error.
            # We log and skip to keep the pipeline moving.
            pass

        return batch_records

    def save_chunk(self, data, index_marker):
        """Save buffer as a single parquet file."""
        if not data:
            return
            
        filename = f"chunk_{index_marker}.parquet"
        filepath = self.data_dir / filename
        
        try:
            df = pl.DataFrame(data)
            df.write_parquet(filepath)
            # logger.info(f"Saved chunk {filename}") # Reduce log spam
        except Exception as e:
            logger.error(f"Failed to save chunk {filename}: {e}")

    def run(self):
        # Query for 38.7M records
        query = '(Aquatic[All Fields] OR Marine[All Fields] OR Freshwater[All Fields]) AND "is_nuccore"[Filter] AND 100:1000000[SLEN] NOT "master"[All Fields]'
        
        try:
            self.search(query)
        except Exception:
            return

        start_index = self.get_start_index()
        logger.info(f"Resuming from index {start_index}...")

        # Progress Bar
        pbar = tqdm(total=self.total_count, initial=start_index, unit="rec", desc="Harvesting")
        
        buffer = []
        current_processed_index = start_index

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # We use map to ensure we process results in order, which is crucial for
            # simple checkpointing (we only save "up to index X").
            batch_indices = range(start_index, self.total_count, BATCH_SIZE)
            
            # Submit all tasks (lazily evaluated)
            results_generator = executor.map(self.process_batch, batch_indices)
            
            for batch_result in results_generator:
                if batch_result:
                    buffer.extend(batch_result)
                
                # Advance progress by BATCH_SIZE (fetch attempt size)
                current_processed_index += BATCH_SIZE
                pbar.update(BATCH_SIZE)

                # Flush buffer if full
                if len(buffer) >= CHUNK_SIZE:
                    self.save_chunk(buffer, current_processed_index)
                    self.update_checkpoint(current_processed_index)
                    buffer = [] # Clear memory immediately

            # Final flush
            if buffer:
                self.save_chunk(buffer, current_processed_index)
                self.update_checkpoint(current_processed_index)

        pbar.close()
        logger.info("Harvest complete.")

if __name__ == "__main__":
    harvester = Harvester()
    harvester.run()
