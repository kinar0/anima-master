from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HELPER = Path(__file__).resolve().parents[1] / "agent_tools" / "comfyui_agent.py"


def _run_helper(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    # Tests import plugin modules in-process. The real helper gets no such path.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONSAFEPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(HELPER), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=45,
        check=False,
    )


def test_cli_bootstraps_plugin_imports_outside_plugin_directory(tmp_path: Path):
    result = _run_helper(tmp_path, "--help")

    assert result.returncode == 0, result.stderr
    assert "status" in result.stdout
    assert "generate" in result.stdout


def test_cli_status_reads_runtime_config_and_reaches_api(tmp_path: Path):
    requested_paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requested_paths.append(self.path)
            if self.path == "/system_stats":
                payload = {"system": {"comfyui_version": "test"}, "devices": []}
            elif self.path == "/object_info":
                payload = {
                    node: {"input": {"required": {field: [[model]]}}}
                    for node, field, model in (
                        ("UNETLoader", "unet_name", "test-unet"),
                        ("CLIPLoader", "clip_name", "test-clip"),
                        ("VAELoader", "vae_name", "test-vae"),
                    )
                }
            else:
                self.send_error(404)
                return
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    (tmp_path / "astrbot").mkdir()
    config_dir = tmp_path / "data" / "config"
    config_dir.mkdir(parents=True)
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        base_url = f"http://127.0.0.1:{server.server_port}"
        (config_dir / "astrbot_plugin_anima_master_config.json").write_text(
            json.dumps(
                {
                    "anima_master_comfyui_connection": {"comfyui_base_url": base_url},
                    "anima_master_models": {
                        "unet_name": "test-unet",
                        "clip_name": "test-clip",
                        "vae_name": "test-vae",
                    },
                }
            ),
            encoding="utf-8-sig",
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            quick = _run_helper(tmp_path, "status", "--quick")
            assert quick.returncode == 0, quick.stderr
            quick_status = json.loads(quick.stdout)
            assert quick_status["ok"] is True
            assert quick_status["comfyui_api_reachable"] is True
            assert quick_status["capabilities_checked"] is False
            assert requested_paths == ["/system_stats"]

            full = _run_helper(tmp_path, "status")
            assert full.returncode == 0, full.stderr
            full_status = json.loads(full.stdout)
            assert full_status["base_url"] == base_url
            assert full_status["ok"] is True
            assert all(full_status[key] for key in (
                "unet_available", "clip_available", "vae_available"
            ))
            assert requested_paths == ["/system_stats", "/system_stats", "/object_info"]
        finally:
            server.shutdown()
            thread.join(timeout=5)
