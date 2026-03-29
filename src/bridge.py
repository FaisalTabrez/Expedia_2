import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

import numpy as np
import polars as pl

JSONRPC_VERSION = "2.0"
E_TEMP_DIR = Path("E:/EXPEDIA_Data/temp")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("bridge")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts NumPy scalars/arrays into plain Python types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


RpcHandler = Callable[[Dict[str, Any]], Awaitable[Any]]


class BridgeServer:
    """Science-side JSON-RPC server implemented with asyncio.start_server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8899) -> None:
        self.host = host
        self.port = port
        self._server: Optional[asyncio.base_events.Server] = None
        self._methods: Dict[str, RpcHandler] = {}

    def register_method(self, name: str, handler: RpcHandler) -> None:
        self._methods[name] = handler

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)
        sockets = ", ".join(str(sock.getsockname()) for sock in (self._server.sockets or []))
        logger.info("BridgeServer listening on %s", sockets)

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        logger.info("Client connected: %s", peer)
        try:
            while not reader.at_eof():
                raw = await reader.readline()
                if not raw:
                    break
                raw = raw.strip()
                if not raw:
                    continue
                response = await self._dispatch(raw)
                if response is not None:
                    writer.write((json.dumps(response, cls=NumpyEncoder) + "\n").encode("utf-8"))
                    await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            logger.info("Client disconnected: %s", peer)

    async def _dispatch(self, raw: bytes) -> Optional[Dict[str, Any]]:
        try:
            request = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return self._error_response(None, -32700, "Parse error")

        if request.get("jsonrpc") != JSONRPC_VERSION:
            return self._error_response(request.get("id"), -32600, "Invalid Request")

        method = request.get("method")
        params = request.get("params", {})
        req_id = request.get("id")

        if not isinstance(method, str) or method not in self._methods:
            return self._error_response(req_id, -32601, "Method not found")

        try:
            prepared = await self._hydrate_payload_path(params)
            result = await self._methods[method](prepared)
            if req_id is None:
                return None
            return {"jsonrpc": JSONRPC_VERSION, "id": req_id, "result": result}
        except Exception as exc:
            logger.exception("Method %s failed", method)
            return self._error_response(req_id, -32603, str(exc))

    async def _hydrate_payload_path(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"value": params}
        payload_path = params.get("payload_path")
        if not payload_path:
            return params

        parquet_path = Path(payload_path)
        if not parquet_path.is_absolute():
            parquet_path = E_TEMP_DIR / parquet_path
        if parquet_path.drive.upper() != "E:":
            parquet_path = E_TEMP_DIR / parquet_path.name

        df = pl.read_parquet(parquet_path)
        enriched = dict(params)
        enriched["payload_df"] = df
        enriched["payload_path"] = str(parquet_path)
        return enriched

    @staticmethod
    def _error_response(req_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": req_id,
            "error": {"code": code, "message": message},
        }


class BridgeClient:
    """UI-side client with callback server; uses JSON-RPC over asyncio streams."""

    def __init__(
        self,
        science_host: str = "127.0.0.1",
        science_port: int = 8899,
        ui_host: str = "127.0.0.1",
        ui_port: int = 8900,
    ) -> None:
        self.science_host = science_host
        self.science_port = science_port
        self.ui_host = ui_host
        self.ui_port = ui_port
        self._request_id = 0
        self._callbacks: Dict[str, RpcHandler] = {}
        self._callback_server: Optional[asyncio.base_events.Server] = None

    async def start_callback_server(self) -> None:
        self._callback_server = await asyncio.start_server(self._handle_callback, self.ui_host, self.ui_port)
        sockets = ", ".join(str(sock.getsockname()) for sock in (self._callback_server.sockets or []))
        logger.info("BridgeClient callback server listening on %s", sockets)

    def register_callback(self, method: str, handler: RpcHandler) -> None:
        self._callbacks[method] = handler

    def launch_science_kernel(self, script_path: str = "src/science_kernel.py") -> subprocess.Popen:
        """Launch the science kernel process from the UI kernel."""
        logger.info("Launching Science Kernel via subprocess: %s", script_path)
        return subprocess.Popen([sys.executable, script_path], cwd=str(Path.cwd()))

    async def call(self, method: str, params: Dict[str, Any]) -> Any:
        self._request_id += 1
        request = {
            "jsonrpc": JSONRPC_VERSION,
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        reader, writer = await asyncio.open_connection(self.science_host, self.science_port)
        try:
            writer.write((json.dumps(request, cls=NumpyEncoder) + "\n").encode("utf-8"))
            await writer.drain()
            raw = await reader.readline()
            if not raw:
                raise RuntimeError("No response received from BridgeServer")
            response = json.loads(raw.decode("utf-8"))
            if "error" in response:
                raise RuntimeError(response["error"]["message"])
            result = response.get("result")
            if isinstance(result, dict) and "payload_path" in result:
                result = await self._hydrate_payload_path(result)
            return result
        finally:
            writer.close()
            await writer.wait_closed()

    async def notify(self, method: str, params: Dict[str, Any]) -> None:
        request = {"jsonrpc": JSONRPC_VERSION, "method": method, "params": params}
        reader, writer = await asyncio.open_connection(self.science_host, self.science_port)
        try:
            writer.write((json.dumps(request, cls=NumpyEncoder) + "\n").encode("utf-8"))
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_callback(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await reader.readline()
            if not raw:
                return
            request = json.loads(raw.decode("utf-8"))
            method = request.get("method")
            params = await self._hydrate_payload_path(request.get("params", {}))
            handler = self._callbacks.get(method)
            if handler:
                result = await handler(params)
            else:
                result = {"status": "ignored", "reason": f"No callback for {method}"}
            if request.get("id") is not None:
                response = {"jsonrpc": JSONRPC_VERSION, "id": request["id"], "result": result}
                writer.write((json.dumps(response, cls=NumpyEncoder) + "\n").encode("utf-8"))
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def _hydrate_payload_path(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"value": params}
        payload_path = params.get("payload_path")
        if not payload_path:
            return params
        parquet_path = Path(payload_path)
        if not parquet_path.is_absolute():
            parquet_path = E_TEMP_DIR / parquet_path
        if parquet_path.drive.upper() != "E:":
            parquet_path = E_TEMP_DIR / parquet_path.name
        df = pl.read_parquet(parquet_path)
        enriched = dict(params)
        enriched["payload_df"] = df
        enriched["payload_path"] = str(parquet_path)
        return enriched
