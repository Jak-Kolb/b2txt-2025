"""Job status for the viewer's Jobs page and agent tools: what is running, waiting and finished (no torch).

Sources, merged by `list_jobs`:
- queue scripts that report through `report()` (each folder results/harness/queue/<stamp>_<name>/ gets a
  job.json; every state change is appended to results/harness/queue/jobs.jsonl);
- training queues that keep their own status.json ({"started", "jobs": {name: {"state", "train_dir", "runs"}},
  "finished"}), shown with one stage per queued model;
- older queue folders with only a queue.log: its stage lines are parsed;
  the last two count as live while a process runs a matching results/harness/queue/ script;
- queue scripts running with no folder yet, found from /proc;
- `harness train` supervisors (train.status()).

Logs exposed through `job_log_tail` are restricted to files under the results root and pass a filter that
drops lines that look like sentence text (quoted multi-word strings, transcript/reference/hypothesis fields),
so the API never carries B2T sentences even if a script printed one.
"""
from __future__ import annotations

import atexit
import contextlib
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from . import REPO
from .store import DEFAULT_ROOT, lock_path, replace_json

SCHEMA = 1
JOB_FILE = "job.json"   # status.json is taken: training queues write their own
STAMP = re.compile(r"^(\d{8}T\d{6}Z)_(.+)$")
LINE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) (.*)$")
STAGE = re.compile(r"^stage (\S+): (start|done|FAILED)\b ?(.*)$")
WAITING = re.compile(r"waiting for PID (\d+)")
EXITED = re.compile(r"^PID (\d+) exited")
RUN_ID = re.compile(r"\b\d{8}T\d{6}Z_[A-Za-z0-9._+-]+")
SCRIPT_DATE = re.compile(r"^(.+?)_(\d{8})$")
LIVE = ("running", "waiting", "queued")
LOCKS = ("job", "paced")
MAX_TAIL = 2000


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_time(text):
    if not text:
        return None
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            value = datetime.strptime(text, "%Y%m%dT%H%M%SZ")
        except ValueError:
            return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def iso(value):
    return value.astimezone(timezone.utc).isoformat(timespec="seconds") if value else None


# ---------------------------------------------------------------- privacy filter

_QUOTED = re.compile(r"""(["'])((?:(?!\1).){0,400}?)\1""")
_FIELDS = re.compile(r"""\b(sentence|sentences|transcript|transcription|reference_text|ref_text|hyp_text|"""
                     r"""reference|hypothesis|ref|hyp|text|true_sentence|pred_sentence|prediction)\b["']?\s*[:=]\s*(.*)""",
                     re.IGNORECASE)
_WORD = re.compile(r"(?<![\w/.])[A-Za-z']+(?![\w/.])")


def looks_like_sentence(line):
    """True for lines that may carry sentence text: a quoted run of 4+ words, or a text field followed by words."""
    for m in _FIELDS.finditer(line):
        if len(_WORD.findall(m.group(2))) >= 3:
            return True
    for _, inner in _QUOTED.findall(line):
        if len(_WORD.findall(inner)) >= 4:
            return True
    return False


def redact(lines):
    out, dropped = [], 0
    for line in lines:
        if looks_like_sentence(line):
            dropped += 1
            out.append("[line hidden: may contain sentence text]")
        else:
            out.append(line)
    return out, dropped


# ---------------------------------------------------------------- reporting (queue scripts)

_TRACKED = {}


def _queue_root(folder):
    folder = Path(folder)
    return folder.parent if folder.parent.name == "queue" else None


def _rel(path):
    try:
        path = Path(path).resolve()
        return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
    except OSError:
        return str(path)


def _stdout_path():
    try:
        target = os.readlink("/proc/self/fd/1")
    except OSError:
        return None
    return target if target.startswith("/") and Path(target).is_file() else None


def _event(folder, status, event, **extra):
    queue = _queue_root(folder)
    if queue is None:
        return
    record = dict(t=now_iso(), job=status["id"], event=event, state=status["state"], stage=status.get("stage"),
                  pid=status.get("pid"), **extra)
    with (queue / "jobs.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def _save(folder, status):
    status["updated"] = now_iso()
    replace_json(Path(folder) / JOB_FILE, status)


def _load_or_new(folder):
    folder = Path(folder)
    key = str(folder.resolve())
    if key in _TRACKED:
        return _TRACKED[key]
    match = STAMP.match(folder.name)
    status = dict(schema=SCHEMA, id=folder.name, name=match.group(2) if match else folder.name,
                  script=_rel(sys.argv[0]) if sys.argv and sys.argv[0] else None, args=sys.argv[1:],
                  pid=os.getpid(), after_pid=None, state="running", stage=None, stages=[], started=now_iso(),
                  ended=None, runs=[], stdout=_stdout_path(), exit=None)
    _TRACKED[key] = status
    _install_exit_hooks(folder, status)
    _save(folder, status)
    _event(folder, status, "start", script=status["script"])
    return status


def _install_exit_hooks(folder, status):
    previous = sys.excepthook

    def hook(kind, value, tb):
        with contextlib.suppress(Exception):
            status["exit"] = dict(error=kind.__name__, message=redact([str(value)[:300]])[0][0])
        previous(kind, value, tb)

    sys.excepthook = hook
    atexit.register(_finish, folder, status)


def _finish(folder, status):
    with contextlib.suppress(Exception):
        if status.get("ended"):
            return
        failed = bool(status.get("exit"))
        status["state"] = "failed" if failed else "done"
        status["ended"] = now_iso()
        for stage in status["stages"]:
            if stage["state"] == "running":
                stage.update(state="failed" if failed else "unfinished", ended=status["ended"])
        status["stage"] = None
        _save(folder, status)
        _event(folder, status, "end", exit=status.get("exit"))


def report(folder, msg):
    """Record one queue-log message in FOLDER/job.json (stage starts/ends, PID waits, run ids).

    Call it from a queue script's log() with the same text it writes to queue.log. It never raises, so a
    reporting problem cannot break a run; the job is finalized (done, or failed on an uncaught exception)
    when the process exits.
    """
    with contextlib.suppress(Exception):
        _report(Path(folder), str(msg))


def _report(folder, msg):
    status = _load_or_new(folder)
    stamp = now_iso()
    event = None
    if m := WAITING.search(msg):
        status.update(state="waiting", after_pid=int(m.group(1)))
        event = ("waiting", dict(after_pid=int(m.group(1))))
    elif EXITED.match(msg):
        status["state"] = "running"
        event = ("resumed", {})
    elif m := STAGE.match(msg):
        name, what, rest = m.groups()
        status["state"] = "running"
        if what == "start":
            status["stages"].append(dict(name=name, state="running", started=stamp, ended=None))
            status["stage"] = name
        else:
            stage = next((s for s in reversed(status["stages"]) if s["name"] == name and s["ended"] is None), None)
            if stage is None:
                stage = dict(name=name, started=None)
                status["stages"].append(stage)
            stage.update(state="done" if what == "done" else "failed", ended=stamp)
            if what == "FAILED":
                stage["error"] = redact([rest.strip(" ;")[:300]])[0][0]
            status["stage"] = None
        event = (f"stage_{what.lower()}", dict(name=name))
    for run_id in RUN_ID.findall(msg):
        if run_id not in status["runs"] and run_id != status["id"]:
            status["runs"].append(run_id)
    _save(folder, status)
    if event:
        _event(folder, status, event[0], **event[1])


def queue_logger(folder):
    """A ready-made log(msg) for new queue scripts: timestamped line to stdout and FOLDER/queue.log, plus status."""
    folder = Path(folder)

    def log(msg):
        line = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {msg}"
        print(line, flush=True)
        with (folder / "queue.log").open("a") as handle:
            handle.write(line + "\n")
        report(folder, msg)
    return log


# ---------------------------------------------------------------- GPU lock

def lock_holders(path):
    """Processes holding or waiting on a flock-ed file, from /proc/locks: (holders, waiters) PID lists."""
    try:
        inode = os.stat(path).st_ino
        text = Path("/proc/locks").read_text()
    except OSError:
        return [], []
    holders, waiters = [], []
    for line in text.splitlines():
        parts = line.split()
        waiting = len(parts) > 1 and parts[1] == "->"
        fields = parts[2:] if waiting else parts[1:]
        if len(fields) < 5 or fields[0] != "FLOCK":
            continue
        try:
            pid, ino = int(fields[3]), int(fields[4].split(":")[-1])
        except ValueError:
            continue
        if ino == inode:
            (waiters if waiting else holders).append(pid)
    return holders, waiters


@contextlib.contextmanager
def gpu_lock(root=DEFAULT_ROOT, log=None, on_wait=None):
    """Hold results/harness/locks/job.lock (the lock benchmarks take) for a GPU step; wait while another holds it.

    Do not call `harness bench`, `sweep` or `train` inside it: they take the same lock themselves (benchmarks
    refuse to start while it is held). Shell form: `flock results/harness/locks/job.lock CMD ...`.
    """
    path = lock_path(root, "job")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holders = lock_holders(path)[0]
            if log:
                log(f"waiting for the GPU job lock (held by PID {', '.join(map(str, holders)) or '?'})")
            if on_wait:
                on_wait(holders)
            fcntl.flock(handle, fcntl.LOCK_EX)
            if log:
                log("got the GPU job lock")
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# ---------------------------------------------------------------- processes

_BTIME = []


def _boot_time():
    if not _BTIME:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                _BTIME.append(int(line.split()[1]))
                break
        else:
            _BTIME.append(0)
    return _BTIME[0]


def _proc_stat(pid):
    """(ppid, start datetime) of a process, or None."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = text[text.rindex(")") + 2:].split()
    start = _boot_time() + int(fields[19]) / os.sysconf("SC_CLK_TCK")
    return int(fields[1]), datetime.fromtimestamp(start, timezone.utc)


def _cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")[:-1]
    except OSError:
        return []


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def process_table():
    """{pid: (ppid, start, argv)} for every visible process."""
    table = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        stat = _proc_stat(entry.name)
        if stat:
            table[int(entry.name)] = (stat[0], stat[1], _cmdline(entry.name))
    return table


def _script_of(argv):
    for arg in argv[:3]:
        if "results/harness/queue/" in arg and arg.endswith((".py", ".sh")) and not any(c.isspace() for c in arg):
            return arg[arg.index("results/harness/queue/"):]
    return None


def queue_processes(table):
    """Top-level processes running a results/harness/queue/ script (worker re-invocations are skipped)."""
    found = {pid: _script_of(argv) for pid, (_, _, argv) in table.items()}
    found = {pid: script for pid, script in found.items() if script}
    out = []
    for pid, script in found.items():
        parent, seen = table[pid][0], set()
        while parent in table and parent not in seen and parent > 1:
            if parent in found:
                break
            seen.add(parent)
            parent = table[parent][0]
        else:
            out.append(dict(pid=pid, script=script, start=table[pid][1], argv=table[pid][2]))
    return out


def _ancestors(pid, table):
    chain, seen = [], set()
    while pid in table and pid not in seen and pid > 1:
        chain.append(pid)
        seen.add(pid)
        pid = table[pid][0]
    return chain


def _script_base(script):
    stem = Path(script).stem
    m = SCRIPT_DATE.match(stem)
    return m.group(1) if m else stem


# ---------------------------------------------------------------- legacy queue folders

_PARSE_CACHE = {}


def parse_queue_log(path):
    """Stages, waits, run ids and times from a queue.log written by the scripts' log() helpers."""
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key in _PARSE_CACHE:
        return _PARSE_CACHE[key]
    stages, runs, first, last, after, waiting, last_msg = [], [], None, None, None, False, None
    with path.open(errors="replace") as handle:
        for raw in handle:
            m = LINE.match(raw.rstrip("\n"))
            if not m:
                continue
            stamp, msg = iso(parse_time(m.group(1))), m.group(2)
            first, last, last_msg = first or stamp, stamp, msg
            if w := WAITING.search(msg):
                after, waiting = int(w.group(1)), True
            elif EXITED.match(msg):
                waiting = False
            elif s := STAGE.match(msg):
                name, what, rest = s.groups()
                waiting = False
                if what == "start":
                    stages.append(dict(name=name, state="running", started=stamp, ended=None))
                else:
                    stage = next((x for x in reversed(stages) if x["name"] == name and x["ended"] is None), None)
                    if stage is None:
                        stage = dict(name=name, started=None)
                        stages.append(stage)
                    stage.update(state="done" if what == "done" else "failed", ended=stamp)
                    if what == "FAILED":
                        stage["error"] = redact([rest.strip(" ;")[:300]])[0][0]
            runs.extend(r for r in RUN_ID.findall(msg) if r not in runs)
    open_stages = [x for x in stages if x["ended"] is None]
    for stage in open_stages[:-1] if stages and stages[-1]["ended"] is None else open_stages:
        stage["state"] = "unfinished"   # a later stage line followed it: rerun, or a parallel stage that died
    result = dict(stages=stages, runs=runs, first=first, last=last, after_pid=after, waiting=waiting,
                  finished=last_msg == "done")
    if len(_PARSE_CACHE) > 2000:
        _PARSE_CACHE.clear()
    _PARSE_CACHE[key] = result
    return result


def _stdout_for_folder(queue, base, started, last):
    """The detached runner's stdout log (queue/<script>_<date>*.log) whose mtime is nearest the folder's end."""
    best, best_gap = None, None
    target = (parse_time(last) or parse_time(started) or datetime.now(timezone.utc)).timestamp()
    start = (parse_time(started) or datetime.fromtimestamp(0, timezone.utc)).timestamp()
    for path in queue.glob(f"{base}_*.log"):
        rest = path.name[len(base) + 1:]
        if not rest[:8].isdigit():
            continue
        mtime = path.stat().st_mtime
        if mtime < start - 120:
            continue
        gap = abs(mtime - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = path, gap
    return best


# ---------------------------------------------------------------- the merged list

def _elapsed(started, ended):
    a = parse_time(started)
    if not a:
        return None
    b = parse_time(ended) or datetime.now(timezone.utc)
    return max(0, int((b - a).total_seconds()))


def _match_proc(base, stamp, procs):
    """The live queue process whose script matches a folder name and started within 3 minutes of its stamp."""
    for candidate in procs:
        cbase = _script_base(candidate["script"])
        if (base == cbase or base.startswith(cbase + "_")) and stamp and abs(
                (candidate["start"] - stamp).total_seconds()) <= 180:
            return candidate
    return None


def _proc_stdout(proc):
    with contextlib.suppress(OSError):
        out = Path(os.readlink(f"/proc/{proc['pid']}/fd/1"))
        if out.is_file():
            return out
    return None


def _str_ids(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str)]
    return []


def _train_queue_stage(name, sub):
    raw = str(sub.get("state") or "")
    low = raw.lower()
    if low in ("done", "pilot done") or re.fullmatch(r"(\w+ )?exit 0", low):
        state = "done"
    elif any(w in low for w in ("fail", "gave up", "missing", "error")) or re.fullmatch(r"(\w+ )?exit [1-9]\d*", low):
        state = "failed"
    else:
        state = "running"
    return dict(name=name, state=state, started=None, ended=sub.get("updated") if state != "running" else None,
                detail=raw[:80])


def _reporter_job(folder, match, status, table):
    pid = status.get("pid")
    alive = pid in table and Path(status.get("script") or "?").name in " ".join(table[pid][2])
    state = status.get("state")
    if state in LIVE and not alive:
        state = "interrupted"
    logs = {}
    if (folder / "queue.log").exists():
        logs["queue"] = folder / "queue.log"
    stdout = status.get("stdout")
    if stdout and Path(stdout).is_file():
        logs["stdout"] = Path(stdout)
    job = dict(id=folder.name, kind="queue", name=status.get("name") or match.group(2), state=state,
               stage=status.get("stage") if state in LIVE else None, stages=status.get("stages", []),
               started=status.get("started"),
               ended=status.get("ended") or (status.get("updated") if state == "interrupted" else None),
               updated=status.get("updated"), pid=pid,
               after_pid=status.get("after_pid") if state == "waiting" else None, script=status.get("script"),
               runs=list(status.get("runs", [])), trains=[], source=JOB_FILE, exit=status.get("exit"))
    return job, logs, pid if alive else None


def _train_queue_job(folder, match, status, procs, queue):
    stamp = parse_time(match.group(1))
    proc = _match_proc(match.group(2), stamp, procs)
    stages = [_train_queue_stage(name, sub) for name, sub in status["jobs"].items() if isinstance(sub, dict)]
    runs, trains = [], []
    for sub in status["jobs"].values():
        if not isinstance(sub, dict):
            continue
        for key in ("runs", "run", "run_dir"):
            runs.extend(Path(r).name for r in _str_ids(sub.get(key)) if Path(r).name not in runs)
        if isinstance(sub.get("train_dir"), str):
            trains.append(Path(sub["train_dir"]).name)
    finished = status.get("finished")
    if proc:
        state = "running"
    elif finished:
        state = "done"
    else:
        state = "interrupted"
    current = next((s for s in stages if s["state"] == "running"), None) if state == "running" else None
    if state != "running":
        for stage in stages:
            if stage["state"] == "running":
                stage["state"] = "unfinished"
    updated = max((s.get("ended") or "" for s in stages), default="") or None
    logs = {}
    if (folder / "commands.log").exists():
        logs["commands"] = folder / "commands.log"
    stdout = _proc_stdout(proc) if proc else _stdout_for_folder(queue, _script_base(match.group(2)),
                                                                iso(stamp), finished or updated)
    if stdout:
        logs["stdout"] = stdout
    job = dict(id=folder.name, kind="queue", name=match.group(2), state=state,
               stage=f"{current['name']}: {current['detail']}" if current else None, stages=stages,
               started=iso(parse_time(status.get("started"))) or iso(stamp),
               ended=iso(parse_time(finished if isinstance(finished, str) else None)) or (
                   None if state == "running" else updated),
               updated=updated, pid=proc["pid"] if proc else None, after_pid=None,
               script=proc["script"] if proc else None, runs=runs, trains=trains, source="status.json", exit=None)
    return job, logs, proc["pid"] if proc else None


def _log_job(folder, match, procs, queue):
    parsed = parse_queue_log(folder / "queue.log")
    base, stamp = match.group(2), parse_time(match.group(1))
    proc = _match_proc(base, stamp, procs)
    if proc:
        state = "waiting" if parsed["waiting"] else "running"
    elif parsed["finished"] or all(s["ended"] or s["state"] == "unfinished" for s in parsed["stages"]):
        state = "done"
    else:
        state = "interrupted"
    stages = [dict(s) for s in parsed["stages"]]
    if state not in LIVE:
        for stage in stages:
            if stage["ended"] is None:
                stage["state"] = "unfinished"
    open_stage = next((s["name"] for s in reversed(stages) if s["ended"] is None and s["state"] == "running"), None)
    logs = {"queue": folder / "queue.log"}
    stdout = _proc_stdout(proc) if proc else _stdout_for_folder(queue, _script_base(base), iso(stamp),
                                                                parsed["last"])
    if stdout:
        logs["stdout"] = stdout
    job = dict(id=folder.name, kind="queue", name=base, state=state, stage=open_stage if state in LIVE else None,
               stages=stages, started=iso(stamp), ended=None if state in LIVE else parsed["last"],
               updated=parsed["last"], pid=proc["pid"] if proc else None,
               after_pid=parsed["after_pid"] if state == "waiting" else None,
               script=proc["script"] if proc else None, runs=list(parsed["runs"]), trains=[], source="queue.log",
               exit=None)
    return job, logs, proc["pid"] if proc else None


def _queue_jobs(root, table, procs):
    queue = Path(root) / "queue"
    jobs, claimed = [], set()
    if not queue.is_dir():
        return jobs, claimed
    runs_dir, train_dir, base = Path(root) / "runs", Path(root) / "train", Path(root).resolve()
    for folder in sorted(queue.iterdir()):
        match = STAMP.match(folder.name)
        if not match or not folder.is_dir():
            continue
        try:
            if (folder / JOB_FILE).exists():
                job, logs, pid = _reporter_job(folder, match, json.loads((folder / JOB_FILE).read_text()), table)
            elif (folder / "status.json").exists() and isinstance(
                    (status := json.loads((folder / "status.json").read_text())).get("jobs"), dict):
                job, logs, pid = _train_queue_job(folder, match, status, procs, queue)
            elif (folder / "queue.log").exists():
                job, logs, pid = _log_job(folder, match, procs, queue)
            else:
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue   # a half-written file: skip this round
        if pid:
            claimed.add(pid)
        logs = {k: v for k, v in logs.items() if Path(v).resolve().is_relative_to(base)}
        job["links"] = dict(runs=[r for r in job["runs"] if (runs_dir / r).is_dir()],
                            jobs=[r for r in job["runs"] if r != folder.name and (queue / r).is_dir()],
                            train=[t for t in job.pop("trains") if (train_dir / t).is_dir()])
        job["failed_stages"] = sum(s.get("state") == "failed" for s in job["stages"])
        job["logs"] = sorted(logs)
        job["_logs"] = logs
        del job["runs"]
        jobs.append(job)
    return jobs, claimed


def _orphan_jobs(root, procs, claimed):
    jobs = []
    for proc in procs:
        if proc["pid"] in claimed:
            continue
        logs = {}
        out = _proc_stdout(proc)
        if out and out.resolve().is_relative_to(Path(root).resolve()):
            logs["stdout"] = out
        after = None
        if "--after" in proc["argv"]:
            with contextlib.suppress(ValueError, IndexError):
                after = int(proc["argv"][proc["argv"].index("--after") + 1])
        jobs.append(dict(id=f"pid-{proc['pid']}", kind="queue", name=_script_base(proc["script"]),
                         state="running", stage=None, stages=[], started=iso(proc["start"]), ended=None,
                         updated=None, pid=proc["pid"], after_pid=after if after and pid_alive(after) else None,
                         script=proc["script"], source="process", exit=None, failed_stages=0,
                         links=dict(runs=[], jobs=[], train=[]), logs=sorted(logs), _logs=logs))
        if jobs[-1]["after_pid"]:
            jobs[-1]["state"] = "waiting"
    return jobs


TRAIN_STATES = {"forked": "done", "smoke_running": "running", "smoke_passed": "done", "smoke_failed": "failed",
                "launched": "running", "scheduled": "waiting", "gpu_waiting": "waiting", "training": "running",
                "registered": "done", "benchmark_waiting": "waiting", "benchmarking": "running",
                "benchmarked": "done", "failed": "failed", "interrupted": "interrupted"}


def _train_jobs(root):
    from .train import status as train_status
    jobs = []
    try:
        rows = train_status(root)
    except Exception:   # a half-written train dir must not take the page down
        return jobs
    for row in rows:
        raw = row.get("state") or "unknown"
        state = TRAIN_STATES.get(raw, raw)
        updated = row.get("updated")
        if raw == "smoke_running" and _elapsed(updated, None) and _elapsed(updated, None) > 3 * 3600:
            state = "interrupted"
        if raw == "registered" and row.get("then_bench"):
            state = "running"
        train_dir = Path(root) / "train" / row["id"]
        logs = {name: train_dir / f"{name}.log" for name in ("train", "supervise", "smoke")
                if (train_dir / f"{name}.log").exists()}
        ended = updated if state not in LIVE else None
        if state == "interrupted" and logs:   # state.json stops at the launch; the last log write is when it died
            ended = iso(datetime.fromtimestamp(max(p.stat().st_mtime for p in logs.values()), timezone.utc))
        log = row.get("log") or {}
        waiting = None
        if raw == "scheduled":
            waiting = dict(label=f"start time {row.get('not_before')}")
        elif raw in ("gpu_waiting", "benchmark_waiting"):
            waiting = dict(label="GPU job lock (results/harness/locks/job.lock)", lock="job")
        pid = row.get("trainer_pid") if raw == "training" else row.get("supervisor_pid")
        jobs.append(dict(id=f"train-{row['id']}", kind="train", name=row["name"], state=state, stage=raw,
                         stages=[], started=row.get("created"), ended=ended,
                         updated=updated, pid=pid if state in LIVE else None, after_pid=None, waiting_on=waiting,
                         script=None, source="train", exit=None, failed_stages=0,
                         detail=dict(forked_from=row.get("forked_from"), overrides=row.get("overrides"),
                                     question=row.get("question"), last_val_per=log.get("last_val_per"),
                                     last_loss=log.get("last_loss"), registry_entry=row.get("registry_entry")),
                         links=dict(runs=[row["benchmark_run"]] if row.get("benchmark_run") else [], jobs=[],
                                    train=[row["id"]]),
                         logs=sorted(logs), _logs=logs))
    return jobs


ORDER = {"running": 0, "waiting": 1, "queued": 1}


def _collect(root):
    root = Path(root)
    table = process_table()
    procs = queue_processes(table)
    queue_jobs, claimed = _queue_jobs(root, table, procs)
    jobs = queue_jobs + _orphan_jobs(root, procs, claimed) + _train_jobs(root)
    by_pid = {}
    for job in jobs:
        if job.get("pid") and job["state"] in LIVE:
            by_pid[job["pid"]] = job["id"]
    for job in jobs:
        job["elapsed_s"] = _elapsed(job.get("started"), job.get("ended"))
        if job.get("after_pid") and not job.get("waiting_on"):
            job["waiting_on"] = dict(pid=job["after_pid"], job=by_pid.get(job["after_pid"]),
                                     label=f"PID {job['after_pid']}")
        job.setdefault("waiting_on", None)
        job.pop("after_pid", None)
    return jobs, table


def _public(job):
    return {k: v for k, v in job.items() if not k.startswith("_")}


def list_jobs(root=DEFAULT_ROOT, limit=50):
    """All live jobs (running first, then waiting) and the LIMIT most recently finished ones."""
    jobs, _ = _collect(root)
    live = sorted((j for j in jobs if j["state"] in LIVE),
                  key=lambda j: (ORDER.get(j["state"], 2), -(parse_time(j["started"]) or
                                                             datetime.fromtimestamp(0, timezone.utc)).timestamp()))
    done = sorted((j for j in jobs if j["state"] not in LIVE),
                  key=lambda j: (parse_time(j.get("ended") or j.get("updated") or j.get("started"))
                                 or datetime.fromtimestamp(0, timezone.utc)), reverse=True)
    counts = {}
    for job in jobs:
        counts[job["state"]] = counts.get(job["state"], 0) + 1
    return dict(checked=now_iso(), counts=counts, total_finished=len(done),
                jobs=[_public(j) for j in live + done[:max(0, int(limit))]])


def get_job(job_id, root=DEFAULT_ROOT):
    jobs, _ = _collect(root)
    for job in jobs:
        if job["id"] == job_id:
            return job
    raise KeyError(f"Unknown job id: {job_id}")


def job_log_tail(job_id, n=100, which=None, root=DEFAULT_ROOT):
    """Last N lines of one of a job's logs (sentence-like lines hidden). Paths come from the job, never the caller."""
    job = get_job(job_id, root)
    logs = job["_logs"]
    if not logs:
        raise FileNotFoundError("This job has no log")
    name = which or next((k for k in ("queue", "train", "commands", "supervise", "smoke", "stdout") if k in logs),
                         sorted(logs)[0])
    if name not in logs:
        raise FileNotFoundError(f"No log {name!r} for this job (has: {', '.join(sorted(logs))})")
    path = Path(logs[name]).resolve()
    base = Path(root).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise PermissionError("Log is outside the results folder")
    n = max(1, min(int(n), MAX_TAIL))
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - 512 * 1024))
        data = handle.read().decode(errors="replace")
    lines = data.splitlines()
    truncated = len(lines) > n or size > 512 * 1024
    lines, hidden = redact(lines[-n:])
    return dict(id=job_id, log=name, file=str(path.relative_to(base)), size=size, lines=lines,
                hidden_lines=hidden, truncated=truncated, available=sorted(logs))


# ---------------------------------------------------------------- GPU

_GPU_CACHE = {}


def _nvidia_smi():
    return shutil.which("nvidia-smi") or next((p for p in ("/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi")
                                               if Path(p).exists()), None)


def _smi(args):
    exe = _nvidia_smi()
    if not exe:
        raise FileNotFoundError("nvidia-smi not found")
    out = subprocess.run([exe, *args, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
    if out.returncode:
        raise RuntimeError(out.stderr.strip() or f"nvidia-smi exit {out.returncode}")
    return [[c.strip() for c in line.split(",")] for line in out.stdout.strip().splitlines() if line.strip()]


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def gpu_status(root=DEFAULT_ROOT, max_age_s=3.0):
    """GPU use (nvidia-smi), compute processes mapped to job ids, and who holds or waits on the GPU locks."""
    cached = _GPU_CACHE.get(str(root))
    if cached and time.monotonic() - cached[0] < max_age_s:
        return cached[1]
    jobs, table = _collect(root)
    job_pids = {}
    for job in jobs:
        if job.get("pid") and job["state"] in LIVE:
            job_pids[job["pid"]] = job["id"]

    def job_of(pid):
        for ancestor in _ancestors(pid, table):
            if ancestor in job_pids:
                return job_pids[ancestor]
        return None

    def describe(pid):
        argv = table.get(pid, (None, None, []))[2]
        return dict(pid=pid, job=job_of(pid), command=" ".join(Path(a).name if i == 0 else a for i, a in
                                                                   enumerate(argv[:3]))[:160] or None)

    result = dict(checked=now_iso(), available=False, error=None, gpus=[], apps=[], apps_visible=None, locks={})
    try:
        for row in _smi(["--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"]):
            result["gpus"].append(dict(index=int(row[0]), name=row[1], util_pct=_num(row[2]),
                                       mem_used_mib=_num(row[3]), mem_total_mib=_num(row[4]), temp_c=_num(row[5])))
        result["available"] = True
        try:
            apps = _smi(["--query-compute-apps=pid,used_memory,process_name"])
        except RuntimeError:
            apps = []
        for row in apps:
            with contextlib.suppress(ValueError):
                pid = int(row[0])
                result["apps"].append(dict(describe(pid), mem_mib=_num(row[1]), process=Path(row[2]).name))
        result["apps_visible"] = bool(result["apps"]) or not any(
            (g["mem_used_mib"] or 0) > 1500 for g in result["gpus"])
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        result["error"] = str(exc)[:200]
    for name in LOCKS:
        path = lock_path(root, name)
        holders, waiters = lock_holders(path) if path.exists() else ([], [])
        result["locks"][name] = dict(file=str(path.relative_to(Path(root))) if path.is_relative_to(Path(root))
                                     else str(path), held=bool(holders), holders=[describe(p) for p in holders],
                                     waiters=[describe(p) for p in waiters])
    _GPU_CACHE[str(root)] = (time.monotonic(), result)
    return result
