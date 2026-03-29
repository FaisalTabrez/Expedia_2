import asyncio
from pathlib import Path

import polars as pl

from src.bridge import BridgeClient


class UIKernel:
    """UI-side orchestrator that launches ScienceKernel and sends RPC commands."""

    def __init__(self) -> None:
        self.bridge = BridgeClient()
        self.science_process = None

    async def start(self) -> None:
        await self.bridge.start_callback_server()
        self.science_process = self.bridge.launch_science_kernel("src/science_kernel.py")
        await asyncio.sleep(1.0)

    async def run_avalanche_from_parquet(self, parquet_path: str) -> dict:
        result = await self.bridge.call(
            "run_avalanche",
            {
                "payload_path": parquet_path,
                "origin": "ui_kernel",
            },
        )
        return result

    async def stop(self) -> None:
        if self.science_process is not None:
            self.science_process.terminate()


async def main() -> None:
    ui = UIKernel()
    await ui.start()

    # Example: send a disk-handshake request if a temp payload exists.
    sample_payload = Path("E:/EXPEDIA_Data/temp/vectors_latest.parquet")
    if sample_payload.exists():
        res = await ui.run_avalanche_from_parquet(str(sample_payload))
        print(res)

        if isinstance(res, dict) and "payload_path" in res and Path(res["payload_path"]).exists():
            manifold_df = pl.read_parquet(res["payload_path"])
            print(f"Loaded manifold rows: {manifold_df.height}")

    await ui.stop()


if __name__ == "__main__":
    asyncio.run(main())
