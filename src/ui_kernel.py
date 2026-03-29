import sys
import asyncio
import threading
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
import polars as pl

from src.bridge import BridgeClient
from src.expedia_map import ExpediaMap
from src.surveillance_panel import SurveillancePanel

class ExpediaMainWindow(QMainWindow):
    def __init__(self, bridge_client: BridgeClient, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.setWindowTitle("Project EXPEDIA - Deep-Sea Taxonomy")
        self.resize(1400, 900)
        
        self.bridge_client = bridge_client
        
        central_widget = QWidget()
        layout = QHBoxLayout(central_widget)
        
        # Map view (3D Manifold)
        self.map_view = ExpediaMap(bridge_client=self.bridge_client, loop=loop, parent=self)
        layout.addWidget(self.map_view, stretch=3)
        
        # Surveillance Input panel
        self.surveillance_panel = SurveillancePanel(
            bridge_client=self.bridge_client, 
            map_widget=self.map_view, 
            loop=loop,
            parent=self
        )
        layout.addWidget(self.surveillance_panel, stretch=1)
        
        self.setCentralWidget(central_widget)

class UIKernel:
    """UI-side orchestrator that launches ScienceKernel and sends RPC commands."""

    def __init__(self) -> None:
        self.bridge = BridgeClient()
        self.science_process = None

    async def start_async(self) -> None:
        await self.bridge.start_callback_server()
        self.science_process = self.bridge.launch_science_kernel("src/science_kernel.py")
        await asyncio.sleep(1.0)
        
        # Initial data fetch to populate the map
        response = await self.bridge.request("fetch_manifold_data", {})
        print("Initial Fetch:", response)

    def stop(self) -> None:
        if self.science_process is not None:
            self.science_process.terminate()


def run_async_loop(loop: asyncio.AbstractEventLoop, ui: UIKernel):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ui.start_async())
    loop.run_forever()

def main():
    app = QApplication(sys.argv)
    
    # Setup Asyncio loop in background thread to handle RPC
    loop = asyncio.new_event_loop()
    ui = UIKernel()
    
    t = threading.Thread(target=run_async_loop, args=(loop, ui), daemon=True)
    t.start()
    
    # Wait slightly to ensure science kernel launch
    import time
    time.sleep(2)
    
    main_window = ExpediaMainWindow(ui.bridge, loop)
    main_window.show()
    
    # Note: For production, we'd fetch the manifold parquet here and render it bounds
    # e.g., main_window.map_view.render_points(...)
    
    exit_code = app.exec()
    
    # Cleanup
    ui.stop()
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=1.0)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
