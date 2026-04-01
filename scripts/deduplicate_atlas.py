from pathlib import Path
import time

import polars as pl


CHUNKS_DIR = Path("C:/Volume D/Expedia_2/data/chunks/")
OUTPUT_FILE = Path("C:/Volume D/Expedia_2/data/cleaned_atlas.parquet")
WINNERS_FILE = Path("C:/Volume D/Expedia_2/data/winner_accessions.parquet")
TEMP_BATCH_DIR = Path("C:/Volume D/Expedia_2/data/temp_dedup_batches")
LOG_FILE = Path("C:/Volume D/Expedia_2/data/dedup_scan_failures.log")
BATCH_SIZE = 128


def chunked(items: list[str], size: int):
    for index in range(0, len(items), size):
        yield items[index:index + size]


def get_validated_files() -> list[str]:
    return [str(path) for path in sorted(CHUNKS_DIR.glob("*.parquet"))]


def get_length_expr(schema: pl.Schema) -> pl.Expr:
    for candidate in ("length", "Length", "seq_len", "SeqLen"):
        if candidate in schema.names():
            return pl.col(candidate).cast(pl.Int64).alias("seq_len")
    return pl.col("Sequence").str.len_bytes().alias("seq_len")


def log_scan_failure(phase: str, file_path: str, err: Exception) -> None:
    message = f"[{phase}] {file_path} :: {type(err).__name__}: {err}"
    print(message)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(message + "\n")


def collect_batch_winners(files: list[str]) -> pl.DataFrame:
    schema = pl.scan_parquet(files).collect_schema()
    length_expr = get_length_expr(schema)
    return (
        pl.scan_parquet(files)
        .select([pl.col("Accession"), pl.col("TaxID"), length_expr])
        .sort(["TaxID", "seq_len", "Accession"], descending=[False, True, False])
        .unique(subset=["TaxID"], keep="first")
        .select(["TaxID", "Accession", "seq_len"])
        .collect(engine="streaming")
    )


def main():
    start_time = time.time()
    validated_files = get_validated_files()

    if not validated_files:
        raise FileNotFoundError(f"No parquet files found in {CHUNKS_DIR}")

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    print(f"Found {len(validated_files):,} validated parquet files in {CHUNKS_DIR}.")
    print("Phase 1: collecting TaxID winners with streaming collect...")

    winner_parts: list[pl.DataFrame] = []
    scan_safe_files: list[str] = []

    for batch_number, batch_files in enumerate(chunked(validated_files, BATCH_SIZE), start=1):
        try:
            winner_parts.append(collect_batch_winners(batch_files))
            scan_safe_files.extend(batch_files)
        except Exception as batch_err:
            print(f"Batch {batch_number} failed in Phase 1, checking files one-by-one: {batch_err}")
            for file_path in batch_files:
                try:
                    winner_parts.append(collect_batch_winners([file_path]))
                    scan_safe_files.append(file_path)
                except Exception as file_err:
                    log_scan_failure("phase1", file_path, file_err)

    if not winner_parts:
        raise RuntimeError("No winners could be computed from the available parquet files.")

    combined_winners = pl.concat(winner_parts, how="vertical", rechunk=True)
    winners_df = (
        combined_winners
        .sort(["TaxID", "seq_len", "Accession"], descending=[False, True, False])
        .unique(subset=["TaxID"], keep="first")
        .select(["Accession"])
        .unique()
    )
    winners_df.write_parquet(str(WINNERS_FILE))

    print("Phase 2: filtering full records by winners and streaming to disk...")
    TEMP_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for old_batch in TEMP_BATCH_DIR.glob("*.parquet"):
        old_batch.unlink()

    winners_lf = winners_df.lazy()
    dedup_batch_outputs: list[str] = []

    for batch_number, batch_files in enumerate(chunked(scan_safe_files, BATCH_SIZE), start=1):
        batch_output = TEMP_BATCH_DIR / f"dedup_batch_{batch_number:05d}.parquet"
        try:
            (
                pl.scan_parquet(batch_files)
                .join(winners_lf, on="Accession", how="semi")
                .sink_parquet(str(batch_output))
            )
            dedup_batch_outputs.append(str(batch_output))
        except Exception as batch_err:
            print(f"Batch {batch_number} failed in Phase 2, checking files one-by-one: {batch_err}")
            for file_path in batch_files:
                single_output = TEMP_BATCH_DIR / f"dedup_{Path(file_path).stem}.parquet"
                try:
                    (
                        pl.scan_parquet(file_path)
                        .join(winners_lf, on="Accession", how="semi")
                        .sink_parquet(str(single_output))
                    )
                    dedup_batch_outputs.append(str(single_output))
                except Exception as file_err:
                    log_scan_failure("phase2", file_path, file_err)

    if not dedup_batch_outputs:
        raise RuntimeError("No deduplicated batch output was produced.")

    pl.scan_parquet(dedup_batch_outputs).sink_parquet(str(OUTPUT_FILE))

    try:
        input_count = pl.scan_parquet(scan_safe_files).select(pl.len()).collect().item()
        cleaned_count = pl.scan_parquet(str(OUTPUT_FILE)).select(pl.len()).collect().item()
    except AttributeError:
        input_count = pl.scan_parquet(scan_safe_files).select(pl.count()).collect().item()
        cleaned_count = pl.scan_parquet(str(OUTPUT_FILE)).select(pl.count()).collect().item()

    duration = time.time() - start_time
    print("\n--- Deduplication Summary ---")
    print(f"Time Elapsed:      {duration:.2f} seconds")
    print(f"Validated Files:   {len(validated_files):,}")
    print(f"Scannable Files:   {len(scan_safe_files):,}")
    print(f"Winner Accessions: {winners_df.height:,}")
    print(f"Input Rows Used:   {input_count:,}")
    print(f"Output Rows:       {cleaned_count:,}")
    print(f"Rows Reduced:      {input_count - cleaned_count:,}")
    print(f"Failure Log:       {LOG_FILE}")

if __name__ == "__main__":
    main()
