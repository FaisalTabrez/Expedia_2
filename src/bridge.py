import asyncio
import json
import sys
import os
import numpy as np
import logging
from typing import Any, Dict
from src.science_kernel import ScienceKernel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - Bridge - %(levelname)s - %(message)s')
logger = logging.getLogger("Bridge")

# Define buffer path
TEMP_DIR = "EXPEDIA_Data\\temp"
BUFFER_FILE = "bridge.tmp"

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for NumPy data types """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class IPCBridge:
    def __init__(self):
        self.kernel = None  # Will be initialized in run()
        self.data_root = None

    async def initialize_kernel(self):
        try:
            self.kernel = ScienceKernel()
            self.data_root = self.kernel.db_path
            logger.info("Science Kernel Initialized.")
            return {"status": "initialized", "root": self.data_root}
        except Exception as e:
            logger.error(f"Failed to initialize kernel: {e}")
            return {"status": "error", "message": str(e)}

    def _get_temp_path(self):
        if self.data_root:
            temp_path = os.path.join(self.data_root, "temp")
            os.makedirs(temp_path, exist_ok=True)
            return os.path.join(temp_path, BUFFER_FILE)
        return "bridge.tmp" # Fallback

    def _serialize_response(self, result: Any, req_id: Any) -> str:
        # 1. Serialize using NumpyEncoder
        try:
            json_str = json.dumps(result, cls=NumpyEncoder)
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            return json.dumps({"jsonrpc": "2.0", "error": {"code": -32603, "message": "Serialization Error"}, "id": req_id})

        # 2. Check Size > 100KB (102400 bytes)
        if len(json_str) > 102400:
            logger.info(f"Payload size {len(json_str)} bytes exceeds 100KB. Initiating Disk-Based Handshake.")
            buffer_path = self._get_temp_path()
            try:
                with open(buffer_path, "w", encoding='utf-8') as f:
                    f.write(json_str)
                
                # Create Handshake Response
                handshake_response = {
                    "jsonrpc": "2.0",
                    "result": {
                        "__handshake__": True,
                        "path": buffer_path,
                        "size": len(json_str)
                    },
                    "id": req_id
                }
                return json.dumps(handshake_response)
            except Exception as e:
                logger.error(f"Disk Handshake failed: {e}")
                return json.dumps({"jsonrpc": "2.0", "error": {"code": -32603, "message": "Disk Handshake Error"}, "id": req_id})
        
        # 3. Standard Response
        response = {
            "jsonrpc": "2.0",
            "result": result,
            "id": req_id
        }
        return json.dumps(response, cls=NumpyEncoder)

    async def handle_request(self, line: str):
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            logger.error("Invalid JSON")
            return

        method = req.get("method")
        params = req.get("params", {})
        req_id = req.get("id")
        result = None

        logger.info(f"Received method: {method}")

        try:
            if method == "initialize":
                result = await self.initialize_kernel()
            
            elif method == "search":
                if not self.kernel:
                    raise Exception("Kernel not initialized")
                vector = params.get("vector")
                k = params.get("k", 50)
                result = self.kernel.search(vector, k)
            
            elif method == "pipeline":
                if not self.kernel:
                    raise Exception("Kernel not initialized")
                # Assume params contains 'vectors' or we use data in memory? 
                # For robustness, we might pass vectors here or a query ID.
                # Assuming passing vectors for now (might trigger handshake on REQUEST too, but prompt only detailed RESPONSE handshake)
                vectors = np.array(params.get("vectors")) 
                result = self.kernel.run_avalanche_pipeline(vectors)

            else:
                result = {"error": "Method not found"}

            # Send Response
            response_str = self._serialize_response(result, req_id)
            sys.stdout.write(response_str + "\n")
            sys.stdout.flush()

        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            err_resp = {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}, "id": req_id}
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()

    async def run(self):
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            line = await reader.readline()
            if not line:
                break
            await self.handle_request(line.decode().strip())

if __name__ == "__main__":
    # Windows SelectorEventLoop policy might be needed for pipes?
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    bridge = IPCBridge()
    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        pass
