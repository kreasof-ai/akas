"""User intent and identities; no inference schedules are user requirements."""

from dataclasses import dataclass, asdict
import hashlib
from importlib import import_module
import json
import math
from pathlib import Path
import re

from akas.benchmarking.workload import load


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def duration(value):
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(s|m|h|d)?", str(value))
    if not match:
        raise ValueError("budget must be a positive duration such as 30m or 4h")
    seconds = (
        float(match[1]) * {None: 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[match[2]]
    )
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("budget must be positive")
    return seconds


def measurement_limits():
    """Freeze measurement controls in both matrix and configuration identities."""
    import os

    return dict(
        shape_timeout_seconds=float(os.environ.get("AKAS_SHAPE_TIMEOUT", "0")),
        http_timeout_seconds=os.environ.get("AKAS_HTTP_TIMEOUT"),
    )


def symbol(name):
    module, separator, attribute = name.partition(":")
    if not separator:
        raise ValueError("custom definition must be module:class")
    return getattr(import_module(module), attribute)


def discover(checkpoint, definition=None):
    """Families declare their checkpoint types; the coordinator has no model table."""
    if definition:
        return (
            definition
            if isinstance(definition, str)
            else definition.__module__ + ":" + definition.__name__
        )
    from pkgutil import iter_modules
    import akas.models
    from akas.models.definition import Definition

    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        config = json.loads((checkpoint / "config.json").read_text())
        architecture = config.get("text_config", config).get("model_type")
    else:
        from akas.checkpoints.gguf import GGUF

        architecture = GGUF(checkpoint).metadata.get("general.architecture")
    found = set()
    for entry in iter_modules(akas.models.__path__):
        if entry.ispkg and not entry.name.startswith("_"):
            family = import_module(f"akas.models.{entry.name}")
            for name in getattr(family, "__all__", ()):
                candidate = getattr(family, name)
                if isinstance(candidate, type) and issubclass(candidate, Definition):
                    if architecture in getattr(
                        candidate, "checkpoint_architectures", ()
                    ):
                        found.add(candidate)
    if len(found) != 1:
        raise ValueError(
            f"no unambiguous model declaration for {architecture!r}; supply --definition for a custom model"
        )
    candidate = found.pop()
    return candidate.__module__ + ":" + candidate.__name__


def workload(value="standard-v1"):
    data = dict(value) if isinstance(value, dict) else load(value)
    for key in ("batches", "prefix_tokens"):
        values = data.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(type(v) is not int or v < 1 for v in values)
        ):
            raise ValueError(f"workload {key} must contain positive integers")
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate workload {key}")
    if type(data.get("output_tokens")) is not int or data["output_tokens"] < 2:
        raise ValueError("workload needs at least two output tokens")
    for name in ("early_eos", "prefix_cache", "cpu_offload"):
        if data.get(name, False) is not False:
            raise ValueError(f"this workload requires {name}=false")
    if (
        data.get("cached_prefix_tokens", 0) != 0
        or data.get("sampling", {}).get("temperature", 0) != 0
    ):
        raise ValueError("optimization currently uses empty-cache greedy requests")
    return data


@dataclass(frozen=True)
class Context:
    checkpoint: str
    definition: str
    workload: dict
    backend: str = "auto"
    device: int = 0
    memory_bytes: int | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    kv_dtype: str | None = None
    mode: str = "auto"

    def __post_init__(self):
        workload(self.workload)
        if self.mode not in ("auto", "ar", "mtp") or self.kv_dtype not in (
            None,
            "float16",
            "bfloat16",
            "fp8",
        ):
            raise ValueError("invalid generation mode or cache precision")
        if (
            self.backend not in ("auto", "cuda", "webgpu")
            or type(self.device) is not int
            or self.device < 0
        ):
            raise ValueError("invalid backend or device")
        for name in ("memory_bytes", "ttft_ms", "tpot_ms"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be positive")

    def metadata(self):
        return asdict(self)


def source_root():
    checkout = Path(__file__).resolve().parents[3]
    return (
        checkout
        if (checkout / "src/akas").is_dir()
        else Path(__file__).resolve().parents[1]
    )


def arguments(parser):
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="local safetensors directory or GGUF checkpoint",
    )
    parser.add_argument(
        "--workload",
        default="standard-v1",
        help="packaged workload ID or a workload JSON file",
    )
    parser.add_argument(
        "--budget",
        required=True,
        help="additional wall-clock allowance, e.g. 30m or 4h; includes compilation",
    )
    parser.add_argument("--backend", choices=("auto", "cuda", "webgpu"), default="auto")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("auto", "ar", "mtp"),
        default="auto",
        help="required generation mode, or evaluate supported modes",
    )
    parser.add_argument(
        "--kv-cache",
        choices=("float16", "bfloat16", "fp8"),
        help="required cache precision; defaults to the model declaration",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--definition", help="optional module:class for a custom model")
    parser.add_argument("--memory-gib", type=float, help="optional GPU memory ceiling")
    parser.add_argument("--ttft-ms", type=float, help="optional maximum serving TTFT")
    parser.add_argument(
        "--tpot-ms",
        type=float,
        help="optional maximum serving time per subsequent output token",
    )
    parser.add_argument(
        "--shape-timeout",
        help="maximum duration per measurement repetition, including diagnostics, e.g. 2m",
    )
    parser.add_argument("--history", type=Path, default=Path("benchmarks"))
    parser.set_defaults(run=command)


def command(args):
    from akas.optimization.supervisor import optimize

    if args.shape_timeout is not None:
        import os

        os.environ["AKAS_SHAPE_TIMEOUT"] = str(duration(args.shape_timeout))
    checkpoint = args.model.resolve()
    context = Context(
        str(checkpoint),
        discover(checkpoint, args.definition),
        workload(args.workload),
        args.backend,
        args.device,
        None if args.memory_gib is None else args.memory_gib * 1024**3,
        args.ttft_ms,
        args.tpot_ms,
        args.kv_cache,
        args.mode,
    )
    result = optimize(
        context, args.out, budget=duration(args.budget), history=args.history
    )
    print(json.dumps(result, indent=2))
    if not result.get("deployment"):
        raise SystemExit(1)
