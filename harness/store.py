"""Run directories, provenance, hashing, locks, and resource guards (no torch)."""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

from . import REPO

DEFAULT_ROOT = REPO / "results" / "harness"


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_new_json(path, payload):
    """Create a new JSON file; never overwrite."""
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def replace_json(path, payload):
    """Atomically rewrite a file this process owns (a run's own manifest)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


@contextlib.contextmanager
def flocked(path, blocking=True):
    """Exclusive advisory lock; raises BlockingIOError when non-blocking and held."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def lock_path(root, name):
    return Path(root) / "locks" / f"{name}.lock"


def lock_held(root, name):
    """Probe a lock without keeping it (contention reporting only)."""
    try:
        with flocked(lock_path(root, name), blocking=False):
            return False
    except BlockingIOError:
        return True


class HashCache:
    """sha256 keyed by (resolved path, size, mtime_ns); an in-place edit keeping both is missed."""

    method = "sha256_cached_by_path_size_mtime"

    def __init__(self, root=DEFAULT_ROOT):
        self.path = Path(root) / "cache" / "sha256.json"

    def __call__(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        key = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
        with flocked(self.path.with_suffix(".lock")):
            table = json.loads(self.path.read_text()) if self.path.exists() else {}
            if key not in table:
                table[key] = sha256_path(path)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
                tmp.write_text(json.dumps(table, indent=0, sort_keys=True))
                os.replace(tmp, self.path)
            return table[key]


def git_provenance(out_dir=None):
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              check=True).stdout
    head = git("rev-parse", "HEAD").strip()
    porcelain = git("status", "--porcelain")
    diff = git("diff", "HEAD")
    info = dict(head=head, dirty=bool(porcelain.strip()), porcelain=porcelain.splitlines(),
                diff_sha256=hashlib.sha256(diff.encode()).hexdigest() if diff else None)
    if out_dir is not None and diff:
        with (Path(out_dir) / "git_diff.patch").open("x") as handle:
            handle.write(diff)
        info["diff_file"] = "git_diff.patch"
    return info


def env_info():
    info = dict(python=sys.version.split()[0], platform=platform.platform())
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    return info


def mem_available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def ram_guard(expected_gb, what, margin_gb=2.0):
    """Refuse to start a process that would push the machine into swap (swap is ignored)."""
    available = mem_available_gb()
    if available < expected_gb + margin_gb:
        raise MemoryError(f"{what} needs ~{expected_gb:.1f} GB + {margin_gb:.1f} GB margin; "
                          f"only {available:.1f} GB available. Unload the live LM or wait for "
                          "other harness jobs.")
    return available


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", text).strip("-")[:80] or "run"


def new_run_dir(root, pipeline_name, tier):
    return fresh_dir(Path(root) / "runs" / f"{utc_stamp()}_{safe_name(pipeline_name)}_{tier}")


def fresh_dir(base):
    """Create a new directory at `base`, or `base-2`, `base-3`, ... on a same-second collision."""
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 100):
        path = base if attempt == 1 else base.with_name(f"{base.name}-{attempt}")
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create a fresh run directory near {base}")


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def run_status(manifest):
    status = manifest.get("status")
    if status == "running" and not pid_alive(manifest.get("pid")):
        return "interrupted"
    return status


def list_runs(root=DEFAULT_ROOT):
    """Manifest index of all harness runs, newest first."""
    runs = []
    for path in sorted((Path(root) / "runs").glob("*/manifest.json"), reverse=True):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        manifest["status"] = run_status(manifest)
        manifest["run_dir"] = str(path.parent)
        runs.append(manifest)
    return runs


def resolve_run(root, run_id):
    """Map a user/API run id to its directory without trusting it as a path."""
    names = {p.name: p for p in (Path(root) / "runs").iterdir() if p.is_dir()} \
        if (Path(root) / "runs").exists() else {}
    if run_id in names:
        return names[run_id]
    matches = [name for name in names if name.startswith(run_id)]
    if len(matches) != 1:
        raise KeyError(f"Unknown or ambiguous run id: {run_id}")
    return names[matches[0]]
