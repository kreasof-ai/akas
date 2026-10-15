"""Durable run provenance, searchable summaries and qualified schedule exports."""

from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import distributions, distribution, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tarfile
import uuid

from akas.benchmarking.workload import STANDARD, STANDARD_SHA256

SCHEMA = "akas.benchmark-run.v1"


def utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity(root):
    """Hash the working source, including uncommitted and untracked source files."""
    root = Path(root).resolve()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, check=True
        ).stdout

    try:
        revision = git("rev-parse", "HEAD").decode().strip()
        paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
        names = paths.decode().strip("\0").split("\0")
        dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
        names = [str(path.relative_to(root)) for path in root.rglob("*.py")]
    files = {}
    for name in sorted(set(names)):
        # Results must not recursively become part of the measured source.
        if not name or name.startswith("benchmarks/") or ".ipynb_checkpoints" in name:
            continue
        if revision is not None and not (
            name.startswith(("src/", "tools/", "tests/", "examples/"))
            or name in ("pyproject.toml", "uv.lock", "MANIFEST.in")
            or ("/" not in name and name.endswith(".py"))
        ):
            continue
        path = root / name
        if path.is_symlink():
            files[name] = hashlib.sha256(
                path.readlink().as_posix().encode()
            ).hexdigest()
        elif path.is_file():
            files[name] = digest(path)
    tree = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(
        revision=revision,
        dirty=dirty,
        tree_sha256=tree,
        files=files,
    )


def shapes(report):
    """Keep total-output and legacy additional-decode accounting distinguishable."""
    protocol = report["metadata"].get("protocol")
    result = []
    for row in report["results"]:
        case = row["case"]
        output = case["generated"] + (protocol == "native-token-step-v1")
        result.append(
            dict(
                batch=case["batch"],
                prefix_tokens=case["prompt"],
                output_tokens=output,
                cached_prefix_tokens=case.get("depth", 0),
                ragged=case.get("mixed", False),
            )
        )
    return result


class Recorder:
    """Capture source before work, update one record, then seal its outcome.

    Raw outputs and source snapshots live beside the record. A run with source
    changes during measurement is retained as invalidated and cannot seed search.
    """

    def __init__(
        self,
        root,
        metadata,
        *,
        source_root=None,
        change=None,
        parent=None,
        kind="configuration",
    ):
        self.root = Path(root).resolve()
        checkout = Path(__file__).resolve().parents[3]
        default = (
            checkout
            if (checkout / "src/akas").is_dir()
            else Path(__file__).resolve().parents[1]
        )
        self.source_root = Path(source_root or default)
        started = utc()
        identity = source_identity(self.source_root)
        self.id = re.sub(r"[^0-9]", "", started) + "-" + uuid.uuid4().hex[:8]
        self.directory = self.root / "runs" / self.id
        self.directory.mkdir(parents=True)
        archive = self.directory / "source.tar.gz"
        with tarfile.open(archive, "w:gz", dereference=False) as stream:
            for name in identity["files"]:
                stream.add(self.source_root / name, arcname=name, recursive=False)
        self.record = dict(
            schema=SCHEMA,
            id=self.id,
            started_at=started,
            completed_at=None,
            status="running",
            source=identity,
            metadata=dict(metadata),
            change=dict(kind=kind, description=change, parent_run=parent),
            dependencies=metadata.get("dependencies")
            or {d.metadata["Name"]: d.version for d in distributions()},
            environment=dict(
                platform=platform.platform(),
                python=platform.python_version(),
                cpu_count=os.cpu_count(),
            ),
            workload=dict(id=None, shapes=[]),
            results=[],
            search=[],
            artifacts=[
                dict(
                    path="source.tar.gz",
                    sha256=digest(archive),
                    bytes=archive.stat().st_size,
                )
            ],
        )
        self.record["source"]["captured_at"] = started
        packages = {}
        for name in ("tensor-workspace", "wgpu", "tilelang", "apache-tvm-ffi", "numpy"):
            try:
                package = distribution(name)
                contents = package.read_text("RECORD")
                direct = package.read_text("direct_url.json")
                packages[name] = dict(
                    version=package.version,
                    record_sha256=hashlib.sha256(contents.encode()).hexdigest()
                    if contents
                    else None,
                    vcs=json.loads(direct).get("vcs_info") if direct else None,
                )
            except PackageNotFoundError:
                continue
        self.record["environment"]["packages"] = packages
        try:
            gpu = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.memory",
                    "--format=csv",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            self.record["environment"]["nvidia_smi"] = gpu.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            patch = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.source_root),
                    "diff",
                    "HEAD",
                    "--binary",
                    "--",
                    "src",
                    "tools",
                    "tests",
                    "examples",
                    "pyproject.toml",
                    "uv.lock",
                    "MANIFEST.in",
                ],
                capture_output=True,
                check=True,
            ).stdout
            if patch:
                (self.directory / "source.patch").write_bytes(patch)
                self.artifact(self.directory / "source.patch", "source.patch")
        except (OSError, subprocess.CalledProcessError):
            pass
        write(self.directory / "run.json", self.record)

    def artifact(self, source, name):
        source, destination = Path(source), self.directory / name
        if not destination.resolve().is_relative_to(self.directory):
            raise ValueError("artifact must stay inside the run directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        self.record["artifacts"] = [
            a for a in self.record["artifacts"] if a["path"] != name
        ]
        self.record["artifacts"].append(
            dict(
                path=name, sha256=digest(destination), bytes=destination.stat().st_size
            )
        )

    def program(self, directory):
        """Retain exact phase winners and their qualification trace, not binaries."""
        directory = Path(directory).resolve()
        manifest = json.loads((directory / "program.json").read_text())
        kernel_evidence = getattr(self, "_kernel_evidence", {})
        identity = digest(directory / "program.json")
        self.artifact(directory / "program.json", f"programs/{identity}/program.json")
        phases = list(manifest["phases"])
        for phase in manifest["phases"]:
            if phase.get("commit"):
                phases.append(
                    dict(
                        phase["commit"],
                        phase=phase["phase"] + ".commit",
                        rows=phase["rows"],
                        chunk=phase["chunk"],
                    )
                )
            for label, entry in phase.get("journal", {}).items():
                phases.append(
                    dict(
                        entry,
                        phase="journal." + label,
                        rows=phase["rows"],
                        chunk=phase["chunk"],
                    )
                )
        if manifest.get("reset"):
            phases.append(dict(manifest["reset"], phase="reset", rows=1, chunk=1))
        for entry in manifest.get("rounds", ()):
            phases.append(dict(entry, phase="round", chunk=entry["window"]))
        for phase in phases:
            path = (directory / phase["path"]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("program phase escapes its directory")
            if phase.get("sha256") and digest(path) != phase["sha256"]:
                raise ValueError("program phase checksum mismatch")
            data = json.loads(path.read_text())
            for kernel in data.get("kernels", {}).values():
                if kernel["sha256"] in kernel_evidence:
                    continue
                binary = (directory / kernel["path"]).resolve()
                if (
                    not binary.is_relative_to(directory)
                    or digest(binary) != kernel["sha256"]
                ):
                    raise ValueError("compiled kernel checksum mismatch")
                from tensor.artifacts.format import read_artifact

                metadata, files = read_artifact(binary)
                sources = {
                    name: content.decode()
                    for name, content in files.items()
                    if name.endswith((".tirx.json", ".wgsl", ".cu", ".ptx"))
                }
                kernel_evidence[kernel["sha256"]] = dict(
                    metadata=metadata, sources=sources
                )
            tuning = data.get("tuning", {})
            if tuning.get("inherited_from") or not tuning.get("profile"):
                continue
            name = f"programs/{identity}/{phase['phase']}-b{phase['rows']}-t{phase['chunk']}.json"
            self.artifact(path, name)
            entry = dict(
                program_sha256=identity,
                phase=phase["phase"],
                batch=phase["rows"],
                chunk=phase["chunk"],
                provider=manifest["provider"],
                target=manifest["target"],
                contract=data["forward"]["contract"],
                qualification=name,
                profile=tuning["profile"],
                winners=[
                    dict(
                        kind=op["kind"],
                        parameters=op["parameters"],
                        selected=op["selected"],
                    )
                    for op in tuning.get("operations", [])
                ],
            )
            if entry not in self.record["search"]:
                self.record["search"].append(entry)
        if kernel_evidence:
            self._kernel_evidence = kernel_evidence
            evidence = self.directory / "kernels.json.gz"
            evidence.write_bytes(
                gzip.compress(
                    json.dumps(kernel_evidence, sort_keys=True).encode(), mtime=0
                )
            )
            self.artifact(evidence, "kernels.json.gz")

    def update(self, report, *, final=False, status=None):
        if self.record["completed_at"] is not None:
            raise ValueError("completed benchmark runs are immutable")
        self.record["metadata"] = dict(report["metadata"])
        self.record["results"] = report["results"]
        observed = shapes(report)
        standard = bool(observed) and all(
            row["batch"] in STANDARD["batches"]
            and row["prefix_tokens"] in STANDARD["prefix_tokens"]
            and row["output_tokens"] == STANDARD["output_tokens"]
            and row["cached_prefix_tokens"] == 0
            and not row["ragged"]
            for row in observed
        )
        self.record["workload"] = dict(
            id="standard-v1" if standard else "custom", shapes=observed
        )
        if standard:
            self.record["workload"]["definition"] = STANDARD
            self.record["workload"]["sha256"] = STANDARD_SHA256
        write(self.directory / "report.json", report)
        if final:
            unchanged = (
                source_identity(self.source_root)["tree_sha256"]
                == self.record["source"]["tree_sha256"]
            )
            self.record["source"]["changed_during_run"] = not unchanged
            self.record["completed_at"] = utc()
            self.record["status"] = (
                "invalidated"
                if not unchanged
                else status
                or (
                    "passed"
                    if report["results"]
                    and all(r["status"] == "ok" for r in report["results"])
                    else "failed"
                )
            )
            self.record["artifacts"].append(
                dict(
                    path="report.json",
                    sha256=digest(self.directory / "report.json"),
                    bytes=(self.directory / "report.json").stat().st_size,
                )
            )
        write(self.directory / "run.json", self.record)
        if final:
            rebuild_index(self.root)
        return self.record


def records(root, **filters):
    for path in sorted((Path(root) / "runs").glob("*/run.json")):
        record = json.loads(path.read_text())
        if record.get("schema") != SCHEMA:
            raise ValueError(f"unsupported benchmark record: {path}")
        meta = record["metadata"]
        values = dict(
            status=record["status"],
            hardware=str(meta.get("device", {})),
            model=meta.get("model", meta.get("definition", "")),
            runtime=meta.get("runtime", "akas"),
            workload=record["workload"]["id"],
            provider=meta.get("provider", ""),
            mode=meta.get("mode", "ar"),
            purpose=meta.get("purpose", "benchmark"),
        )
        if all(
            value is None or str(value).lower() in str(values[key]).lower()
            for key, value in filters.items()
        ):
            yield record


def rebuild_index(root):
    root = Path(root)
    entries = []
    for record in records(root):
        meta = record["metadata"]
        device = meta.get("device", {})
        entries.append(
            dict(
                id=record["id"],
                started_at=record["started_at"],
                status=record["status"],
                revision=record["source"]["revision"],
                dirty=record["source"]["dirty"],
                hardware=device.get("name", str(device))
                if isinstance(device, dict)
                else str(device),
                model=meta.get("model", meta.get("definition", "")),
                runtime=meta.get("runtime", "akas"),
                workload=record["workload"]["id"],
                provider=meta.get("provider"),
                mode=meta.get("mode", "ar"),
                kv_dtype=meta.get("kv_dtype"),
                weight_format=meta.get("weight_format", meta.get("weight_dtype")),
                change=record["change"],
                purpose=meta.get("purpose", "benchmark"),
                shapes=record["workload"]["shapes"],
                search_phases=len(record["search"]),
                path=f"runs/{record['id']}/run.json",
            )
        )
    write(root / "index.json", dict(schema="akas.benchmark-index.v1", runs=entries))
    guide = Path(__file__).resolve().parents[3] / "docs/benchmarking.md"
    guide_link = (
        os.path.relpath(guide, root)
        if guide.is_file()
        else "https://github.com/kreasof-ai/akas/blob/main/docs/benchmarking.md"
    )
    lines = [
        "# Benchmark results",
        "",
        f"Generated from run records. See [the benchmark guide]({guide_link}).",
        "",
        "| UTC | Hardware | Runtime / model | Backend / weights / mode / KV | Shapes (B/P/O) | Purpose | Status | Source | Record |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in reversed(entries):
        clean = lambda text: str(text).replace("|", "\\|").replace("\n", " ")
        revision = (entry["revision"][:10] if entry["revision"] else "unavailable") + (
            " + snapshot" if entry["dirty"] else ""
        )
        configuration = " / ".join(
            str(entry[key] or "unspecified")
            for key in ("provider", "weight_format", "mode", "kv_dtype")
        )
        geometry = ", ".join(
            f"{shape['batch']}/{shape['prefix_tokens']}/{shape['output_tokens']}"
            + (
                f" cached={shape['cached_prefix_tokens']}"
                if shape["cached_prefix_tokens"]
                else ""
            )
            + (" ragged" if shape["ragged"] else "")
            for shape in entry["shapes"]
        )
        lines.append(
            f"| {entry['started_at']} | {clean(entry['hardware'])} | {clean(entry['runtime'])} / {clean(entry['model'])} | {clean(configuration)} | {geometry} | {entry['purpose']} | {entry['status']} | {revision} | [run]({entry['path']}) |"
        )
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")


def export_profile(root, run_id, *, phase, batch, chunk, provider, target, out):
    """Export a qualified incumbent; future compilation must qualify it again."""
    matching = [row for row in records(root) if row["id"] == run_id]
    if len(matching) != 1 or matching[0]["status"] != "passed":
        raise ValueError("schedule export requires a passed, unchanged run")
    record = matching[0]
    selected = [
        row
        for row in record["search"]
        if all(
            row[key] == value
            for key, value in dict(
                phase=phase, batch=batch, chunk=chunk, provider=provider, target=target
            ).items()
        )
    ]
    if len(selected) != 1:
        raise ValueError(
            "select exactly one matching phase/batch/chunk/provider/target"
        )
    entry = selected[0]
    directory = Path(root) / "runs" / run_id
    artifact = next(
        a for a in record["artifacts"] if a["path"] == entry["qualification"]
    )
    path = directory / artifact["path"]
    if (
        not path.resolve().is_relative_to(directory.resolve())
        or digest(path) != artifact["sha256"]
    ):
        raise ValueError("qualification artifact checksum mismatch")
    data = json.loads(path.read_text())
    tuning = data["tuning"]
    if tuning["profile"] != entry["profile"] or not tuning.get("operations"):
        raise ValueError("schedule has no retained qualification evidence")
    write(out, entry["profile"])
    return entry["profile"]


def arguments(parser):
    parser.add_argument("--root", type=Path, default=Path("benchmarks"))
    sub = parser.add_subparsers(dest="action", required=True)
    listing = sub.add_parser("list", help="filter durable benchmark runs")
    for key in (
        "hardware",
        "model",
        "runtime",
        "status",
        "workload",
        "provider",
        "mode",
        "purpose",
    ):
        listing.add_argument("--" + key)
    listing.add_argument(
        "--json", action="store_true", help="include complete structured records"
    )
    showing = sub.add_parser("show", help="show one complete run record")
    showing.add_argument("run_id")
    seed = sub.add_parser(
        "seed", help="export a qualified phase schedule for requalification"
    )
    seed.add_argument("run_id")
    for key in ("phase", "provider", "target"):
        seed.add_argument("--" + key, required=True)
    for key in ("batch", "chunk"):
        seed.add_argument("--" + key, required=True, type=int)
    seed.add_argument("--out", required=True, type=Path)
    sub.add_parser("index", help="rebuild the history index")
    parser.set_defaults(run=command)


def command(args):
    if args.action == "list":
        found = list(
            records(
                args.root,
                **{
                    k: getattr(args, k)
                    for k in (
                        "hardware",
                        "model",
                        "runtime",
                        "status",
                        "workload",
                        "provider",
                        "mode",
                        "purpose",
                    )
                },
            )
        )
        if args.json:
            print(json.dumps(found, indent=2))
        else:
            for row in found:
                meta = row["metadata"]
                print(
                    row["id"],
                    row["status"],
                    meta.get("runtime", "akas"),
                    meta.get("model", meta.get("definition", "")),
                    meta.get("purpose", "benchmark"),
                    len(row["workload"]["shapes"]),
                    "shapes",
                )
    elif args.action == "show":
        found = [row for row in records(args.root) if row["id"] == args.run_id]
        if len(found) != 1:
            raise ValueError("benchmark run not found")
        print(json.dumps(found[0], indent=2))
    elif args.action == "seed":
        export_profile(
            args.root,
            args.run_id,
            **{
                k: getattr(args, k)
                for k in ("phase", "batch", "chunk", "provider", "target", "out")
            },
        )
    else:
        rebuild_index(args.root)
