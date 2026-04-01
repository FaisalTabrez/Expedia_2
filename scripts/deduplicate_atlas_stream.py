import polars as pl
import time
import os

def main():
    input_chunks = 'C:/Volume D/Expedia_2/data/chunks/*.parquet'
    output_file = 'C:/Volume D/Expedia_2/data/cleaned_atlas_final.parquet'

    start_time = time.time()
    
    print(f"Lazy scanning {input_chunks}...")
    lf = pl.scan_parquet(input_chunks)
    
    # 1. Fingerprinting and Length Calculation
    print("Setting up logic: Sequence Hashing + Length Calculation...")
    lf = lf.with_columns([
        pl.col("Sequence").str.len_bytes().alias("length"),
        pl.col("Sequence").hash().alias("seq_hash")
    ])
    
    # 2. Selection: Sort by Length (Descending) and keep first occurrence of each Hash
    print("Setting up logic: Sorting by Length and Filtering Unique Fingerprints...")
    lf = (
        lf.sort("length", descending=True)
        .unique(subset=["seq_hash"], keep="first")
        .drop(["seq_hash", "length"]) # Clean up temporary sorting/hashing columns
    )
    
    # 3. Streaming Execution
    print(f"Executing and sinking directly to {output_file} (This may take a few minutes)...")
    # sink_parquet evaluates the graph streamingly to avoid blowing up the RAM
    lf.sink_parquet(output_file)
    
    duration = time.time() - start_time
    print(f"\n--- Deduplication Stream Complete ---")
    print(f"Time Elapsed: {duration:.2f} seconds")
    
    # 4. Validation
    print(f"\nValidating the final output shape...")
    try:
        final_count = pl.scan_parquet(output_file).select(pl.len()).collect().item()
    except AttributeError:
        # Fallback for polars < 0.20
        final_count = pl.scan_parquet(output_file).select(pl.count()).collect().item()
        
    print("="*45)
    print("           VALIDATION SUMMARY")
    print("="*45)
    print(f"Target Output:   ~16,898,218")
    print(f"Actual Count:     {final_count:,}")
    if abs(final_count - 16898218) < 100000:
        print("Status:           SUCCESS (Count matches expected range)")
    else:
        print("Status:           WARNING (Count deviated from expectation)")
    print("="*45)


if __name__ == "__main__":
    main()
