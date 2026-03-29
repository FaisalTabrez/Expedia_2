import urllib.request
import tarfile
import logging
from pathlib import Path
from typing import Dict, Tuple

import polars as pl

logger = logging.getLogger("taxonomy_refiner")

class TaxonomyRefiner:
    """Reconstructs full NCBI lineages using parsed taxdump."""
    def __init__(self, data_dir: str = "E:/EXPEDIA_Data/taxdump"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[int, str] = {}
        self.nodes_dict: Dict[int, Tuple[int, str]] = {}
        self.names_dict: Dict[int, str] = {}
        self._load_taxdump()

    def _download_and_extract(self):
        url = "https://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz"
        tar_path = self.data_dir / "taxdump.tar.gz"
        logger.info(f"Downloading {url} to {tar_path}...")
        urllib.request.urlretrieve(url, tar_path)
        logger.info("Extracting nodes.dmp and names.dmp...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extract("nodes.dmp", path=self.data_dir)
            tar.extract("names.dmp", path=self.data_dir)

    def _load_taxdump(self):
        nodes_path = self.data_dir / "nodes.dmp"
        names_path = self.data_dir / "names.dmp"
        if not nodes_path.exists() or not names_path.exists():
            self._download_and_extract()

        logger.info("Loading taxonomy nodes and names into memory...")
        with open(nodes_path, "r") as f:
            for line in f:
                parts = line.split("\t|\t")
                tax_id = int(parts[0].strip())
                parent_id = int(parts[1].strip())
                rank = parts[2].strip()
                self.nodes_dict[tax_id] = (parent_id, rank)

        with open(names_path, "r") as f:
            for line in f:
                parts = line.split("\t|\t")
                if len(parts) >= 4 and "scientific name" in parts[3]:
                    tax_id = int(parts[0].strip())
                    name = parts[1].strip()
                    self.names_dict[tax_id] = name
        logger.info("Taxonomy data loaded.")

    def get_lineage(self, tax_id: int) -> str:
        if tax_id in self.cache:
            return self.cache[tax_id]
        
        lineage = []
        current = tax_id
        target_ranks = {"superkingdom", "kingdom", "phylum", "class", "order", "family", "genus", "species"}
        
        seen = set()
        while current != 1 and current not in seen:
            seen.add(current)
            node = self.nodes_dict.get(current)
            if not node:
                break
            parent_id, rank = node
            
            if rank in target_ranks or current == tax_id:
                name = self.names_dict.get(current, f"Unknown_{current}")
                if not lineage or lineage[-1] != name:
                    lineage.append(name)
            
            current = parent_id
            
        lineage.reverse()
        result = "; ".join(lineage) if lineage else "Unknown"
        self.cache[tax_id] = result
        return result

    def enrich_parquet_chunk(self, input_path: str, output_path: str):
        logger.info(f"Enriching {input_path} with lineages...")
        df = pl.read_parquet(input_path)
        
        if "taxonomy_id" in df.columns:
            tax_ids = df["taxonomy_id"].to_list()
            # Resolve caching sequentially
            lineages = [self.get_lineage(tid) for tid in tax_ids]
            df = df.with_columns(pl.Series("taxonomy_path", lineages))
            df.write_parquet(output_path)
            logger.info(f"Saved enriched chunk to {output_path}")
        else:
            logger.warning("No 'taxonomy_id' column found in chunk!")
