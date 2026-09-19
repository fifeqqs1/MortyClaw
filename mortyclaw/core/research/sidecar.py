from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import warnings
from urllib.request import Request, urlopen
import zipfile

from .settings import ResearchSettings


QDRANT_VERSION = "1.19.1"
QDRANT_WINDOWS_URL = (
    "https://github.com/qdrant/qdrant/releases/download/"
    f"v{QDRANT_VERSION}/qdrant-x86_64-pc-windows-msvc.zip"
)
QDRANT_WINDOWS_SHA256 = "9b6f69bd85f6abed4bc13f943099f55c6ffd55f5dd90388635320d8fbb569eb0"


class QdrantSidecar:
    def __init__(self, settings: ResearchSettings | None = None):
        self.settings = settings or ResearchSettings.from_env()
        self.runtime_dir = Path(self.settings.qdrant_storage_path).parent / "research"
        self.pid_path = self.runtime_dir / "qdrant.pid"
        self.log_path = self.runtime_dir / "qdrant.log"

    def executable(self) -> Path | None:
        configured = self.settings.qdrant_command.strip()
        if configured:
            path = Path(configured).expanduser()
            if path.is_file():
                return path.resolve()
            located = shutil.which(configured)
            return Path(located).resolve() if located else None
        bundled = Path(self.settings.qdrant_binary_dir) / "qdrant.exe"
        if bundled.is_file():
            return bundled.resolve()
        located = shutil.which("qdrant")
        return Path(located).resolve() if located else None

    def install(self) -> Path:
        if os.name != "nt":
            raise RuntimeError("自动安装目前只支持 Windows x64；请设置 QDRANT_COMMAND。")
        target_dir = Path(self.settings.qdrant_binary_dir)
        target = target_dir / "qdrant.exe"
        if target.is_file():
            return target.resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mortyclaw-qdrant-") as temp_dir:
            archive = Path(temp_dir) / "qdrant.zip"
            verified = False
            for attempt in range(3):
                archive.unlink(missing_ok=True)
                digest = hashlib.sha256()
                request = Request(QDRANT_WINDOWS_URL, headers={"User-Agent": "MortyClaw/1.0"})
                try:
                    with urlopen(request, timeout=120) as response, archive.open("wb") as output:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            digest.update(chunk)
                            output.write(chunk)
                    verified = digest.hexdigest().lower() == QDRANT_WINDOWS_SHA256
                    if verified:
                        break
                except OSError:
                    if attempt == 2:
                        raise
            if not verified:
                raise RuntimeError("Qdrant 下载文件 SHA-256 校验失败")
            with zipfile.ZipFile(archive) as package:
                member = package.getinfo("qdrant.exe")
                with package.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        return target.resolve()

    def start(self, *, timeout: float = 20.0) -> dict[str, object]:
        if self.is_running():
            return {"status": "already_running", "pid": self._read_pid()}
        executable = self.executable()
        if executable is None:
            raise RuntimeError("未找到 Qdrant，请先执行 mortyclaw research install。")
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        Path(self.settings.qdrant_storage_path).mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(
            {
                "QDRANT__SERVICE__HOST": "127.0.0.1",
                "QDRANT__SERVICE__HTTP_PORT": "6333",
                "QDRANT__SERVICE__GRPC_PORT": "6334",
                "QDRANT__STORAGE__STORAGE_PATH": str(self.settings.qdrant_storage_path),
                "QDRANT__TELEMETRY_DISABLED": "true",
            }
        )
        if self.settings.qdrant_api_key:
            environment["QDRANT__SERVICE__API_KEY"] = self.settings.qdrant_api_key
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        log = self.log_path.open("ab")
        try:
            process = subprocess.Popen(
                [str(executable)],
                cwd=str(executable.parent),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        finally:
            log.close()
        self.pid_path.write_text(str(process.pid), encoding="ascii")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Qdrant 启动失败，请查看 workspace/research/qdrant.log。")
            if self.is_healthy():
                return {"status": "started", "pid": process.pid}
            time.sleep(0.25)
        raise RuntimeError("Qdrant 启动超时")

    def stop(self, *, timeout: float = 10.0) -> dict[str, object]:
        pid = self._read_pid()
        if not pid:
            return {"status": "not_running"}
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            self.pid_path.unlink(missing_ok=True)
            return {"status": "not_running"}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._pid_exists(pid):
                self.pid_path.unlink(missing_ok=True)
                return {"status": "stopped", "pid": pid}
            time.sleep(0.2)
        raise RuntimeError("Qdrant 未能在超时时间内停止")

    def is_healthy(self) -> bool:
        try:
            from qdrant_client import QdrantClient

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Api key is used with an insecure connection.*")
                QdrantClient(
                    url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_api_key or None,
                    timeout=2,
                    check_compatibility=False,
                ).get_collections()
            return True
        except Exception:
            return False

    def is_running(self) -> bool:
        pid = self._read_pid()
        return bool(pid and self._pid_exists(pid) and self.is_healthy())

    def status(self) -> dict[str, object]:
        pid = self._read_pid()
        return {
            "installed": self.executable() is not None,
            "running": bool(pid and self._pid_exists(pid)),
            "connected": self.is_healthy(),
            "pid": pid,
            "version": QDRANT_VERSION,
        }

    def _read_pid(self) -> int | None:
        try:
            value = int(self.pid_path.read_text(encoding="ascii").strip())
            return value if value > 0 else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        if os.name == "nt":
            import ctypes

            process_query_limited_information = 0x1000
            still_active = 259
            handle = ctypes.windll.kernel32.OpenProcess(
                process_query_limited_information, False, pid
            )
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return exit_code.value == still_active
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False
