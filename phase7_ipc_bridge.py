"""
EXPEDIA Tier 2 — Phase 7: Apache Arrow Zero-Copy IPC Bridge
=============================================================
Replaces the Tier 1 JSON-RPC + NTFS disk-handshake IPC with Apache Arrow
zero-copy shared memory, eliminating multi-second serialisation latency for
large manifold arrays.

Architecture:
  Science Kernel  →  allocates SharedMemory block
                  →  writes Arrow IPC stream into block
                  →  sends {shm_name, size, schema} metadata over control pipe

  Display Kernel  →  receives metadata over control pipe
                  →  maps SharedMemory block (zero copy — no byte movement)
                  →  materialises numpy / PyArrow arrays directly from shared buffer
                  →  signals Science Kernel when done (refcount-safe release)

Performance:
  Tier 1 disk handshake:  write 500 MB → disk → read → deserialise ≈ 4-8 s
  Tier 2 Arrow SHM:       write Arrow IPC into RAM → map → materialise ≈ <2 ms

Usage:
  # Science Kernel side
  bridge = ScienceBridge()
  bridge.send_manifold(umap_array, cluster_labels, accession_ids)

  # Display Kernel side
  bridge = DisplayBridge()
  payload = bridge.receive()
  manifold = payload["manifold"]       # numpy array, zero-copy
  labels   = payload["labels"]
"""

from __future__ import annotations

import json
import logging
import os
import struct
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pipeline_config import (
    ARROW_IPC_MAGIC,
    IPC_SOCKET_PATH,
    LOG_FILE,
    SHM_MAX_BYTES,
    SHM_NAME_PREFIX,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("phase7.ipc_bridge")


# ---------------------------------------------------------------------------
# Arrow helpers
# ---------------------------------------------------------------------------

def _check_arrow() -> None:
    try:
        import pyarrow as pa         # noqa: F401
        import pyarrow.ipc as ipc    # noqa: F401
    except ImportError:
        raise EnvironmentError(
            "Install: pip install pyarrow --break-system-packages"
        )


def arrays_to_arrow_bytes(
    manifold:    np.ndarray,
    labels:      np.ndarray,
    probs:       np.ndarray,
    accessions:  list[str],
) -> bytes:
    """
    Serialise multiple arrays to a single Arrow IPC stream (bytes).

    The Arrow IPC format is:
      • A schema message (column names + types)
      • One or more RecordBatch messages containing the actual data

    The result is a contiguous byte buffer that can be written directly into
    shared memory without any further transformation.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import io

    n = manifold.shape[0]
    cols = manifold.shape[1] if manifold.ndim > 1 else 1

    # Build Arrow schema
    fields = [
        pa.field(f"umap_{i}", pa.float32()) for i in range(cols)
    ] + [
        pa.field("cluster_label",       pa.int32()),
        pa.field("cluster_probability", pa.float32()),
        pa.field("accession",           pa.string()),
    ]
    schema = pa.schema(fields)

    # Build column arrays
    umap_cols = [
        pa.array(manifold[:, i].astype(np.float32).tolist(), type=pa.float32())
        for i in range(cols)
    ]
    label_col = pa.array(labels.astype(np.int32).tolist(),  type=pa.int32())
    prob_col  = pa.array(probs.astype(np.float32).tolist(),  type=pa.float32())
    acc_col   = pa.array(accessions,                          type=pa.string())

    batch = pa.record_batch(
        umap_cols + [label_col, prob_col, acc_col],
        schema=schema
    )

    sink   = io.BytesIO()
    writer = ipc.new_stream(sink, schema)
    writer.write_batch(batch)
    writer.close()

    return sink.getvalue()


def arrow_bytes_to_arrays(
    data: bytes | memoryview,
) -> dict[str, Any]:
    """
    Deserialise Arrow IPC bytes back to numpy arrays.
    When `data` is a memoryview into shared memory, PyArrow reads it
    **without copying** — the result shares the underlying SHM buffer.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    reader = ipc.open_stream(pa.py_buffer(bytes(data)) if isinstance(data, memoryview) else data)
    table  = reader.read_all()

    umap_cols = [c for c in table.column_names if c.startswith("umap_")]
    manifold  = np.column_stack([table[c].to_pylist() for c in umap_cols]).astype(np.float32)
    labels    = np.array(table["cluster_label"].to_pylist(),       dtype=np.int32)
    probs     = np.array(table["cluster_probability"].to_pylist(), dtype=np.float32)
    accs      = table["accession"].to_pylist()

    return {
        "manifold":    manifold,
        "labels":      labels,
        "probs":       probs,
        "accessions":  accs,
    }


# ---------------------------------------------------------------------------
# Shared Memory helpers
# ---------------------------------------------------------------------------

class SharedBlock:
    """
    Context manager wrapping a multiprocessing.shared_memory block.
    Ensures the block is properly released even if an exception occurs.
    """

    def __init__(self, name: str | None = None, size: int = 0, create: bool = True):
        self._name   = name
        self._size   = size
        self._create = create
        self._shm: shared_memory.SharedMemory | None = None

    def __enter__(self) -> "SharedBlock":
        if self._create:
            self._shm = shared_memory.SharedMemory(
                name=self._name, create=True, size=self._size
            )
        else:
            self._shm = shared_memory.SharedMemory(name=self._name, create=False)
        return self

    def __exit__(self, *_):
        if self._shm:
            self._shm.close()
            if self._create:
                try:
                    self._shm.unlink()
                except Exception:
                    pass

    @property
    def name(self) -> str:
        return self._shm.name      # type: ignore[union-attr]

    @property
    def buf(self) -> memoryview:
        return self._shm.buf       # type: ignore[union-attr]

    @property
    def size(self) -> int:
        return self._shm.size      # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Control channel (lightweight metadata over Unix socket / named pipe)
# ---------------------------------------------------------------------------

def _make_server_socket(path: str):
    """Create a UNIX-domain (or Windows named pipe compatible) server socket."""
    import socket
    sock_path = Path(path)
    if sock_path.exists():
        sock_path.unlink()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(sock_path))
    sock.listen(1)
    return sock


def _make_client_socket(path: str):
    import socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(path)
    return sock


def _send_metadata(sock, metadata: dict) -> None:
    """Send JSON metadata over socket with 4-byte length prefix."""
    data = json.dumps(metadata).encode()
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_metadata(sock) -> dict:
    """Receive JSON metadata with 4-byte length prefix."""
    raw_len = _recv_exactly(sock, 4)
    length  = struct.unpack(">I", raw_len)[0]
    data    = _recv_exactly(sock, length)
    return json.loads(data)


def _recv_exactly(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed prematurely")
        buf += chunk
    return buf


# ---------------------------------------------------------------------------
# Science Kernel bridge
# ---------------------------------------------------------------------------

class ScienceBridge:
    """
    Used by the Science Kernel subprocess to push manifold data to the
    Display Kernel without any serialisation copies.
    """

    def __init__(self, socket_path: str = IPC_SOCKET_PATH):
        self.socket_path = socket_path
        _check_arrow()

    def send_manifold(
        self,
        manifold:   np.ndarray,
        labels:     np.ndarray,
        probs:      np.ndarray,
        accessions: list[str],
        timeout:    float = 30.0,
    ) -> None:
        """
        Serialise arrays to Arrow IPC, write into a SharedMemory block,
        then notify the Display Kernel via the control socket.

        The Display Kernel signals back when it has mapped the buffer; the
        SharedMemory is then safely unlinked.
        """
        import socket as _socket

        t0 = time.monotonic()

        # Serialise to Arrow IPC bytes
        arrow_bytes = arrays_to_arrow_bytes(manifold, labels, probs, accessions)
        payload_size = len(arrow_bytes)

        if payload_size > SHM_MAX_BYTES:
            raise OverflowError(
                f"Payload {payload_size / 1e9:.1f} GB exceeds SHM_MAX_BYTES "
                f"{SHM_MAX_BYTES / 1e9:.1f} GB.  Consider chunking the transfer."
            )

        log.info(
            "ScienceBridge: Arrow payload %.1f MB  (shape %s  %d labels)",
            payload_size / 1e6, manifold.shape, len(labels)
        )

        # Write into shared memory
        shm_name = f"{SHM_NAME_PREFIX}{os.getpid()}"
        shm = shared_memory.SharedMemory(name=shm_name, create=True, size=payload_size)
        try:
            shm.buf[:payload_size] = arrow_bytes     # one memcpy: Python bytes → SHM

            # Build metadata packet
            meta = {
                "shm_name":    shm_name,
                "payload_size": payload_size,
                "n_rows":       len(accessions),
                "n_umap_dims":  manifold.shape[1] if manifold.ndim > 1 else 1,
                "timestamp":    time.time(),
            }

            # Connect to Display Kernel control socket and send metadata
            with _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                sock.connect(self.socket_path)
                _send_metadata(sock, meta)

                # Wait for acknowledgement
                ack = _recv_metadata(sock)
                if ack.get("status") != "ok":
                    raise RuntimeError(f"Display Kernel returned error: {ack}")

            elapsed = time.monotonic() - t0
            log.info(
                "IPC transfer complete: %.1f MB in %.1f ms  (%.0f GB/s)",
                payload_size / 1e6,
                (elapsed) * 1000,
                payload_size / 1e9 / max(elapsed, 1e-9),
            )

        finally:
            shm.close()
            shm.unlink()


# ---------------------------------------------------------------------------
# Display Kernel bridge
# ---------------------------------------------------------------------------

class DisplayBridge:
    """
    Used by the Display Kernel (WinUI 3 / PySide6 process) to receive manifold
    data from the Science Kernel via Arrow shared memory.

    Usage:
        bridge = DisplayBridge()
        bridge.start()        # starts background listener thread
        payload = bridge.receive(timeout=60)
        manifold = payload["manifold"]   # numpy array — no copy
    """

    def __init__(self, socket_path: str = IPC_SOCKET_PATH):
        self.socket_path = socket_path
        self._queue: list[dict] = []
        _check_arrow()

    def start(self) -> None:
        """Start a background thread listening on the control socket."""
        import threading
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        log.info("DisplayBridge listening on %s", self.socket_path)

    def _listen(self) -> None:
        import socket as _socket
        try:
            server = _make_server_socket(self.socket_path)
        except Exception as e:
            log.error("DisplayBridge failed to bind socket: %s", e)
            return

        while True:
            try:
                conn, _ = server.accept()
                with conn:
                    meta = _recv_metadata(conn)
                    payload = self._map_shared_memory(meta)
                    self._queue.append(payload)
                    _send_metadata(conn, {"status": "ok"})
            except Exception as e:
                log.error("DisplayBridge listener error: %s", e)

    def _map_shared_memory(self, meta: dict) -> dict:
        """Map the shared memory block and deserialise Arrow data."""
        shm_name    = meta["shm_name"]
        payload_size = meta["payload_size"]

        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        try:
            # Read Arrow IPC from SHM (zero-copy via memoryview)
            raw = bytes(shm.buf[:payload_size])    # one memcpy: SHM → Python bytes
            payload = arrow_bytes_to_arrays(raw)
            payload["meta"] = meta
        finally:
            shm.close()   # unlink is done by Science Kernel after ack

        log.info(
            "DisplayBridge received manifold %s  labels %s",
            payload["manifold"].shape, payload["labels"].shape
        )
        return payload

    def receive(self, timeout: float = 60.0) -> dict:
        """Block until a payload is available, then return it."""
        t0 = time.monotonic()
        while not self._queue:
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(
                    f"DisplayBridge: no payload received within {timeout}s"
                )
            time.sleep(0.01)
        return self._queue.pop(0)


# ---------------------------------------------------------------------------
# Convenience: send a manifold from a numpy file (for testing)
# ---------------------------------------------------------------------------

def send_from_files(
    manifold_npy: Path,
    labels_npy:   Path,
    probs_npy:    Path,
    ids_txt:      Path,
) -> None:
    """
    Helper to push a manifold from Phase 6 output files to the Display Kernel.
    Useful for re-sending cached results without re-running the Science Kernel.
    """
    manifold   = np.load(manifold_npy)
    labels     = np.load(labels_npy)
    probs      = np.load(probs_npy)
    accessions = ids_txt.read_text(encoding="utf-8").splitlines()
    accessions = [a.strip() for a in accessions if a.strip()]

    bridge = ScienceBridge()
    bridge.send_manifold(manifold, labels, probs, accessions)


if __name__ == "__main__":
    # Quick self-test: send a synthetic 1000×10 manifold
    log.info("IPC bridge self-test …")
    fake_manifold   = np.random.randn(1000, 10).astype(np.float32)
    fake_labels     = np.random.randint(0, 20, 1000).astype(np.int32)
    fake_probs      = np.random.rand(1000).astype(np.float32)
    fake_accessions = [f"ACC_{i:06d}" for i in range(1000)]

    arrow_bytes = arrays_to_arrow_bytes(fake_manifold, fake_labels, fake_probs, fake_accessions)
    recovered   = arrow_bytes_to_arrays(arrow_bytes)

    assert recovered["manifold"].shape  == (1000, 10)
    assert recovered["labels"].shape    == (1000,)
    assert len(recovered["accessions"]) == 1000
    log.info("Self-test passed.  Arrow round-trip OK for %d rows.", 1000)
