import polars as pl
import time

def main():
    input_chunks = 'C:/Volume D/Expedia_2/data/chunks/*.parquet'
    
    print(f"Initializing lazy scan for {input_chunks}...")
    start_time = time.time()
    
    lf = pl.scan_parquet(input_chunks)
    
    print("Building execution graph for approx_n_unique metrics...")
    
    try:
        metrics_lf = lf.select([
            pl.len().alias("total_count"),
            pl.col("Sequence").approx_n_unique().alias("unique_sequences"),
            pl.col("TaxID").cast(pl.Utf8).approx_n_unique().alias("unique_taxids")
        ])
    except AttributeError:
        # Fallback for Polars versions < 0.20.5 
        metrics_lf = lf.select([
            pl.col("Sequence").count().alias("total_count"),
            pl.col("Sequence").approx_n_unique().alias("unique_sequences"),
            pl.col("TaxID").cast(pl.Utf8).approx_n_unique().alias("unique_taxids")
        ])
    
    print("Executing... (This will run efficiently natively in Rust)")
    
    # Use engine='streaming' instead of deprecated streaming=True
    try:
        results = metrics_lf.collect(engine='streaming')
    except Exception as e:
        # Fallback if streaming is not supported on this query
        print(f"Streaming failed ({e}), falling back to default engine...")
        results = metrics_lf.collect()
    
    total_count = results["total_count"][0]
    unique_seqs = results["unique_sequences"][0]
    unique_taxids = results["unique_taxids"][0]
    duration = time.time() - start_time
    
    print("\n" + "="*40)
    print("        DIAGNOSTIC RESULTS      ")
    print("="*40)
    print(f"Total Rows:                 {total_count:,}")
    print(f"Est. Unique Sequences:      {unique_seqs:,}")
    print(f"Est. Unique TaxIDs:         {unique_taxids:,}")
    print(f"Execution Time:             {duration:.2f} seconds")
    print("="*40)

if __name__ == "__main__":
    main()
