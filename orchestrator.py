"""
EXPEDIA Tier 2 — Pipeline Orchestrator
=======================================
Master controller that runs all 7 phases in sequence with:
  • Phase-level checkpointing (skip already-completed phases on resume)
  • Structured timing and progress reporting
  • Configurable phase selection (run subsets for testing/debugging)
  • Graceful error handling with actionable messages

Usage examples:
  # Full pipeline from scratch
  python orchestrator.py

  # Resume from last checkpoint
  python orchestrator.py --resume

  # Run only phases 4 and 5 (embedding + indexing)
  python orchestrator.py --phases 4 5

  # Force re-run of phase 2
  python orchestrator.py --phases 2 --force

  # Skip MMseqs2 in phase 2 (faster for testing)
  python orchestrator.py --skip-mmseqs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from pipeline_config import DIRS, LOG_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("orchestrator")

ORCHESTRATOR_STATE = DIRS.checkpoints / "orchestrator_state.json"


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def _phase1(force: bool, **kwargs) -> None:
    from phase1_acquisition import run_acquisition
    run_acquisition(force=force, max_rehydrate_workers=kwargs.get("rehydrate_workers", 4))


def _phase2(force: bool, **kwargs) -> None:
    from phase2_deduplication import run_deduplication
    run_deduplication(skip_mmseqs=kwargs.get("skip_mmseqs", False), force=force)


def _phase3(force: bool, **kwargs) -> None:
    from phase3_taxonomy import run_taxonomy
    run_taxonomy(force=force)


def _phase4(force: bool, **kwargs) -> None:
    from phase4_embedding import run_embedding
    run_embedding(force=force)


def _phase5(force: bool, **kwargs) -> None:
    from phase5_indexing import run_indexing
    run_indexing(force=force)


def _phase6(force: bool, **kwargs) -> None:
    from phase6_clustering import run_clustering
    run_clustering(force=force)


def _phase7_test(force: bool, **kwargs) -> None:
    """Phase 7 is a runtime service, not a batch job.  Run a self-test here."""
    from phase7_ipc_bridge import (
        arrays_to_arrow_bytes,
        arrow_bytes_to_arrays,
    )
    import numpy as np

    log.info("Phase 7 — running Arrow IPC self-test …")
    manifold   = np.random.randn(500, 10).astype(np.float32)
    labels     = np.random.randint(0, 20, 500).astype(np.int32)
    probs      = np.random.rand(500).astype(np.float32)
    accs       = [f"ACC_{i:06d}" for i in range(500)]

    arrow_bytes = arrays_to_arrow_bytes(manifold, labels, probs, accs)
    recovered   = arrow_bytes_to_arrays(arrow_bytes)

    assert recovered["manifold"].shape  == (500, 10),   "manifold shape mismatch"
    assert recovered["labels"].shape    == (500,),       "labels shape mismatch"
    assert len(recovered["accessions"]) == 500,          "accession count mismatch"
    log.info(
        "Phase 7 self-test PASSED.  %.1f KB serialised → %.1f KB for 500 rows.",
        len(arrow_bytes) / 1024, len(arrow_bytes) / 1024
    )


PHASE_RUNNERS = {
    1: (_phase1,      "Fault-tolerant NCBI data acquisition"),
    2: (_phase2,      "Out-of-core genomic deduplication"),
    3: (_phase3,      "Parallel TaxonKit taxonomy reconstruction"),
    4: (_phase4,      "GenomeOcean neural embedding"),
    5: (_phase5,      "LanceDB IVF-RQ vector indexing"),
    6: (_phase6,      "Avalanche clustering + PROTAX taxonomy"),
    7: (_phase7_test, "Apache Arrow IPC bridge (self-test)"),
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def load_state() -> dict:
    return json.loads(ORCHESTRATOR_STATE.read_text()) if ORCHESTRATOR_STATE.exists() else {}

def save_state(state: dict) -> None:
    ORCHESTRATOR_STATE.write_text(json.dumps(state, indent=2))


def run_pipeline(
    phases:             list[int],
    force:              bool = False,
    skip_mmseqs:        bool = False,
    rehydrate_workers:  int  = 4,
) -> None:
    """
    Run the selected phases in order, respecting global checkpoints.

    Args:
        phases:            Ordered list of phase numbers to run (1-7).
        force:             If True, re-run phases even if completed.
        skip_mmseqs:       Pass to Phase 2 to skip MMseqs2 clustering.
        rehydrate_workers: Parallel download threads for Phase 1.
    """
    state = load_state()
    pipeline_start = time.monotonic()

    log.info("=" * 70)
    log.info("EXPEDIA Tier 2 — Pipeline Orchestrator")
    log.info("Phases to run: %s", phases)
    log.info("Output root:   %s", DIRS.root)
    log.info("=" * 70)

    kwargs = {
        "skip_mmseqs":       skip_mmseqs,
        "rehydrate_workers": rehydrate_workers,
    }

    for phase_num in phases:
        runner, description = PHASE_RUNNERS[phase_num]
        phase_key = f"phase{phase_num}_done"

        if not force and state.get(phase_key):
            log.info(
                "Phase %d (%s) — already complete.  Skipping.  "
                "(use --force to re-run)",
                phase_num, description
            )
            continue

        log.info("-" * 70)
        log.info("Phase %d — %s", phase_num, description)
        log.info("-" * 70)

        t0 = time.monotonic()
        try:
            runner(force=force, **kwargs)
        except Exception as e:
            log.error(
                "Phase %d FAILED: %s\n"
                "The pipeline is fully resumable.  Fix the error above and "
                "re-run with --resume to continue from this phase.",
                phase_num, e,
                exc_info=True,
            )
            sys.exit(1)

        elapsed = time.monotonic() - t0
        log.info(
            "Phase %d complete in %.1f min.", phase_num, elapsed / 60
        )
        state[phase_key]              = True
        state[f"phase{phase_num}_elapsed_min"] = round(elapsed / 60, 1)
        save_state(state)

    total = time.monotonic() - pipeline_start
    log.info("=" * 70)
    log.info("All %d phases complete in %.1f min.", len(phases), total / 60)
    log.info("=" * 70)
    _print_summary(state)


def _print_summary(state: dict) -> None:
    log.info("Pipeline summary:")
    for i in range(1, 8):
        done    = state.get(f"phase{i}_done", False)
        elapsed = state.get(f"phase{i}_elapsed_min", "—")
        status  = f"✓  {elapsed} min" if done else "—  not run"
        _, desc = PHASE_RUNNERS[i]
        log.info("  Phase %d %-46s %s", i, desc, status)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="EXPEDIA Tier 2 — Scalable Genomic Surveillance Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestrator.py                  # Full pipeline
  python orchestrator.py --resume         # Resume from last failure
  python orchestrator.py --phases 4 5 6  # Only embed + index + cluster
  python orchestrator.py --phases 2 --force --skip-mmseqs
        """
    )
    ap.add_argument(
        "--phases", type=int, nargs="+",
        default=list(range(1, 8)),
        metavar="N",
        help="Phase numbers to run (default: 1 2 3 4 5 6 7)",
    )
    ap.add_argument(
        "--force",  action="store_true",
        help="Re-run selected phases even if checkpoints exist",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="Skip phases that have completed successfully (default behaviour; "
             "explicit flag for clarity)",
    )
    ap.add_argument(
        "--skip-mmseqs", action="store_true",
        help="Skip MMseqs2 homology clustering in Phase 2 (faster, less thorough)",
    )
    ap.add_argument(
        "--rehydrate-workers", type=int, default=4,
        help="Parallel download threads for Phase 1 rehydration (default: 4)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        phases=sorted(set(args.phases)),
        force=args.force,
        skip_mmseqs=args.skip_mmseqs,
        rehydrate_workers=args.rehydrate_workers,
    )
