from pathlib import Path
import shutil

import polars as pl


CHUNKS_DIR = Path("C:/Volume D/Expedia_2/data/chunks/")
CORRUPTED_DIR = Path("C:/Volume D/Expedia_2/data/corrupted_chunks/")


def _unique_destination_path(destination_dir: Path, filename: str) -> Path:
    target = destination_dir / filename
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    counter = 1
    while True:
        candidate = destination_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def validate_chunks() -> None:
    if not CHUNKS_DIR.exists() or not CHUNKS_DIR.is_dir():
        raise FileNotFoundError(f"Chunks directory not found: {CHUNKS_DIR}")

    CORRUPTED_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(CHUNKS_DIR.glob("*.parquet"))
    corrupted_files: list[str] = []

    print(f"Scanning {len(parquet_files)} parquet files in {CHUNKS_DIR}...")

    for file_path in parquet_files:
        try:
            pl.read_parquet(str(file_path), n_rows=1)
        except (pl.exceptions.ComputeError, Exception):
            destination = _unique_destination_path(CORRUPTED_DIR, file_path.name)
            shutil.move(str(file_path), str(destination))
            corrupted_files.append(file_path.name)
            print(f"Corrupted file moved: {file_path.name}")

    print("\nValidation complete.")
    print(f"Total files checked: {len(parquet_files)}")
    print(f"Corrupted files found: {len(corrupted_files)}")

    if corrupted_files:
        print("Corrupted filenames:")
        for name in corrupted_files:
            print(name)


if __name__ == "__main__":
    validate_chunks()
