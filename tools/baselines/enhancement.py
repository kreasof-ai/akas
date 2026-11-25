"""Measure complete offline voice-enhancement requests on AKAS and upstream."""

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
import os
from statistics import median
from queue import Empty, Queue
from threading import Thread
import time

import numpy as np


def upstream_worker():
    import torch
    from inworld import voice_isolation

    configuration = json.loads(sys.argv[2])
    torch.set_num_threads(configuration["threads"])
    model = voice_isolation.VoiceIsolator(
        configuration["checkpoint"], device="cpu", precision="fp32", mode="offline"
    )
    print(
        "ENHANCE "
        + json.dumps(
            dict(torch_version=torch.__version__, threads=configuration["threads"])
        ),
        flush=True,
    )
    for line in sys.stdin:
        payload = json.loads(line)
        try:
            started = time.perf_counter()
            samples, rate = voice_isolation.load_audio(payload["path"])
            voice = model.isolate(samples, rate, output_sample_rate=48000)
            background = samples - voice
            elapsed = time.perf_counter() - started
            np.savez(
                payload["out"],
                voice=voice.numpy(),
                background=background.numpy(),
                input=samples.numpy(),
                elapsed=np.array(elapsed),
            )
            result = dict(ok=True, elapsed=elapsed)
        except (OSError, RuntimeError, ValueError) as error:
            result = dict(ok=False, error=f"{type(error).__name__}: {error}")
        print("ENHANCE " + json.dumps(result), flush=True)


def akas_worker():
    configuration = json.loads(sys.argv[2])
    module, name = configuration["factory"].split(":")
    from importlib import import_module

    context = getattr(import_module(module), name)(**configuration["options"])
    with context as model:
        print("ENHANCE " + json.dumps(model.metadata), flush=True)
        for line in sys.stdin:
            payload = json.loads(line)
            try:
                started = time.perf_counter()
                result = model.isolate(Path(payload["path"]).read_bytes())
                elapsed = time.perf_counter() - started
                np.savez(
                    payload["out"],
                    voice=result.voice,
                    background=result.background,
                    chunks=np.array(result.chunk_count),
                    elapsed=np.array(elapsed),
                )
                response = dict(ok=True, elapsed=elapsed)
            except (OSError, RuntimeError, ValueError) as error:
                response = dict(ok=False, error=f"{type(error).__name__}: {error}")
            print("ENHANCE " + json.dumps(response), flush=True)


@contextmanager
def worker(command, log, deadline, *, env, cwd):
    with (
        log.open("wb") as output,
        subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=output,
            env=env,
            cwd=cwd,
        ) as process,
    ):
        responses = Queue()

        def receive():
            for line in process.stdout:
                if line.startswith(b"ENHANCE "):
                    responses.put(line)
                else:
                    output.write(line)
                    output.flush()
            responses.put(None)

        reader = Thread(target=receive, daemon=True)
        reader.start()

        def next_response():
            try:
                line = responses.get(timeout=max(0, deadline - time.monotonic()))
            except Empty as error:
                raise TimeoutError(
                    "enhancement worker response deadline reached"
                ) from error
            if line is None:
                raise RuntimeError(f"enhancement worker stopped responding; see {log}")
            return json.loads(line[8:])

        def invoke(path, repeat, case):
            result_path = log.parent / f"audio-{case}-{repeat}.npz"
            request = json.dumps(dict(path=str(path), out=str(result_path))) + "\n"
            process.stdin.write(request.encode())
            process.stdin.flush()
            response = next_response()
            if not response.get("ok"):
                raise ValueError(response.get("error", "enhancement failed"))
            if time.monotonic() > deadline:
                raise TimeoutError("enhancement request deadline reached")
            return result_path, response["elapsed"]

        try:
            yield invoke, next_response()
        finally:
            if process.poll() is None:
                process.stdin.close()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            reader.join(timeout=10)


def measure(session, recorder, report, workload, root, meta, reference):
    from akas.benchmarking import enhancement

    for index, case in enumerate(workload["clips"], 1):
        source = (root / case["path"]).resolve()
        audio = source.read_bytes()
        if sha256(audio).hexdigest() != case["sha256"]:
            raise ValueError(f"enhancement input checksum mismatch: {source}")
        from akas.models.voice_isolator.audio import wav_samples

        samples, rate = wav_samples(audio)
        seconds = len(samples) / rate
        recorder.artifact(source, f"inputs/{case['id']}.wav")
        row = dict(case=case, status="error", samples=[])
        report["results"].append(row)
        golden = (
            np.load(root / case["golden"]) if case.get("golden") is not None else None
        )
        print(f"CHECK AND MEASURE {meta['runtime']} {case['id']}", flush=True)
        failures = []
        for iteration in range(meta["warmup"] + meta["repeats"]):
            try:
                result, elapsed = session(source, iteration, index)
                with np.load(result) as output:
                    voice, background = output["voice"], output["background"]
                    if "input" in output and not np.array_equal(
                        output["input"], samples
                    ):
                        raise ValueError("upstream decoded PCM differs from AKAS")
                    if (
                        voice.shape != samples.shape
                        or background.shape != samples.shape
                    ):
                        raise ValueError("enhancement output sample count differs")
                    if (
                        not np.isfinite(voice).all()
                        or not np.isfinite(background).all()
                    ):
                        raise ValueError("enhancement output contains NaN or infinity")
                    if not np.allclose(voice + background, samples, rtol=0, atol=1e-6):
                        raise ValueError(
                            "voice and background do not reconstruct input"
                        )
                    from akas.models.voice_isolator.audio import float_wav

                    measured = iteration >= meta["warmup"]
                    voice_path = (
                        recorder.directory
                        / f"audio/{index:02d}-{iteration + 1:02d}-voice.wav"
                    )
                    background_path = (
                        recorder.directory
                        / f"audio/{index:02d}-{iteration + 1:02d}-background.wav"
                    )
                    voice_path.parent.mkdir(exist_ok=True)
                    voice_path.write_bytes(float_wav(voice))
                    background_path.write_bytes(float_wav(background))
                    if measured:
                        recorder.artifact(
                            voice_path,
                            voice_path.relative_to(recorder.directory).as_posix(),
                        )
                        recorder.artifact(
                            background_path,
                            background_path.relative_to(recorder.directory).as_posix(),
                        )
                    voice_sha = sha256(voice.tobytes()).hexdigest()
                    sample = dict(
                        repeat=iteration - meta["warmup"] + 1 if measured else 0,
                        wall_seconds=elapsed,
                        audio_seconds=seconds,
                        real_time_factor=elapsed / seconds,
                        voice_sha256=voice_sha,
                        background_sha256=sha256(background.tobytes()).hexdigest(),
                        voice_wav=voice_path.relative_to(recorder.directory).as_posix(),
                        background_wav=background_path.relative_to(
                            recorder.directory
                        ).as_posix(),
                        chunk_count=int(output["chunks"])
                        if "chunks" in output
                        else None,
                    )
                    if measured and reference is not None and case["id"] in reference:
                        expected = reference[case["id"]][0]["voice"]
                        sample["upstream_comparison"] = enhancement.errors(
                            voice, expected
                        )
                    if golden is not None:
                        sample["golden"] = enhancement.errors(voice, golden)
                        sample["golden_tolerance"] = 1e-4
                        sample["golden_pass"] = (
                            sample["golden"]["finite"]
                            and sample["golden"]["max_abs_error"] <= 1e-4
                        )
                        if not sample["golden_pass"]:
                            failures.append(
                                f"request {iteration + 1}: golden waveform differs"
                            )
                if measured:
                    row["samples"].append(sample)
                else:
                    voice_path.unlink(missing_ok=True)
                    background_path.unlink(missing_ok=True)
            except (OSError, RuntimeError, ValueError) as error:
                failures.append(f"request {iteration + 1}: {error}")
        if row["samples"]:
            row["summary"] = {
                key: median(sample[key] for sample in row["samples"])
                for key in ("wall_seconds", "audio_seconds", "real_time_factor")
            }
            if reference is not None and case["id"] in reference:
                row["comparison"] = dict(
                    wall_speedup=meta["reference"]["wall_seconds"][case["id"]]
                    / row["summary"]["wall_seconds"],
                    rtf_speedup=meta["reference"]["real_time_factor"][case["id"]]
                    / row["summary"]["real_time_factor"],
                )
        complete = len(row["samples"]) == meta["repeats"]
        row["correctness"] = dict(
            complete=complete,
            deterministic=enhancement.deterministic(row["samples"]),
            reconstruction=True,
            full_model_reference=case["id"] == "golden-3s"
            and all(sample.get("golden_pass", False) for sample in row["samples"]),
        )
        if failures or not complete or not row["correctness"]["deterministic"]:
            row["error"] = " | ".join(
                failures or ["enhanced PCM differed across repetitions"]
            )
        else:
            row["status"] = "ok"
        recorder.update(report)
        print(f"{case['id']}: {row['status']}", flush=True)
        if golden is not None and row["status"] != "ok":
            break


def run(args, definition_path, model, enhancement_options, weight_format):
    from akas.benchmarking import enhancement
    from akas.benchmarking.history import Recorder
    from akas.benchmarking.provenance import digest
    from akas.benchmarking.workload import load
    from tools.baselines.speech import checked_file

    root = Path(__file__).resolve().parents[2]
    workload = load(args.workload or enhancement_options["workload"])
    enhancement.validate(workload)
    if weight_format != "f32":
        raise ValueError(
            "enhancement shares the released F32 checkpoint bytes; select F16 upload separately"
        )
    weight = checked_file(root, model["weights"][weight_format])
    if args.runtime not in ("inworld", "akas"):
        raise ValueError("unsupported enhancement runtime")
    native = args.runtime == "akas"
    if not native and args.repeats is not None:
        raise ValueError(
            "freeze the complete upstream reference before changing repeat counts"
        )
    metadata = dict(
        protocol=workload["protocol"],
        speech_workload=dict(id=workload["id"], definition=workload),
        runtime=args.runtime,
        provider="webgpu" if native else "cpu",
        model=model["model"],
        mode="enhance",
        purpose="speech-demo",
        weight_format=weight_format,
        checkpoint=weight["path"],
        checkpoint_sha256=weight["sha256"],
        checkpoint_source=weight["source"],
        model_definition_sha256=digest(definition_path),
        device=dict(name=platform.processor() or platform.machine()),
        sample_rate_hz=enhancement_options["sample_rate_hz"],
        warmup=workload["warmup"],
        repeats=args.repeats or workload["repeats"],
        deadline_seconds=args.seconds,
        weight_dtype="float32",
        environment={"WGPU_BACKEND_TYPE": "Vulkan"} if native else {},
    )
    if native:
        import tensor

        factory = model["runtime_options"]["akas"]["enhancement_factory"]
        bundle = args.bundle.resolve(strict=True)
        configuration = dict(
            factory=factory, options=dict(checkpoint=weight["path"], bundle=str(bundle))
        )
        command = [
            sys.executable,
            "-u",
            "-m",
            "tools.baselines.enhancement",
            "akas",
            json.dumps(configuration),
        ]
        metadata["configuration"] = command
        metadata["weight_dtype"] = json.loads((bundle / "speech.json").read_text())[
            "weight_dtype"
        ]
        dependency_sources = None
        tensor_source = root.parent / "tensor"
        if Path(tensor.__file__).resolve().is_relative_to(tensor_source):
            dependency_sources = {"tensor": tensor_source}
    else:
        threads = 6
        configuration = dict(checkpoint=weight["path"], threads=threads)
        command = [
            str(root / enhancement_options["python"]),
            "-u",
            str(Path(__file__).resolve()),
            "inworld",
            json.dumps(configuration),
        ]
        metadata["configuration"] = command
        metadata["cpu_threads"] = threads
        metadata["upstream_revision"] = enhancement_options["revision"]
        packages = subprocess.check_output(
            [
                str(root / enhancement_options["python"]),
                "-c",
                "import importlib.metadata as m,json;print(json.dumps({name:m.version(name) for name in ('inworld-voice-isolation','torch')}))",
            ],
            text=True,
            cwd=root,
        )
        metadata.update(json.loads(packages))
        dependency_sources = None
    recorder = Recorder(
        args.history,
        metadata,
        kind="configuration" if native else "baseline",
        change="Measure warm offline voice enhancement on AKAS WebGPU"
        if native
        else "Measure upstream offline voice enhancement on CPU",
        dependency_sources=dependency_sources,
    )
    recorder.artifact(definition_path, "model.json")
    report = dict(schema="akas.benchmark.v1", metadata=metadata, results=[])
    log = recorder.directory / "runtime.log"
    deadline = time.monotonic() + args.seconds
    status = "failed"
    try:
        environment = dict(os.environ, PYTHONUTF8="1")
        if native:
            environment["WGPU_BACKEND_TYPE"] = "Vulkan"
        with worker(command, log, deadline, env=environment, cwd=root) as (
            session,
            properties,
        ):
            metadata.update(properties)
            if native:
                metadata["program_sha256"] = digest(bundle / "speech.json")
                recorder.artifact(bundle / "speech.json", "speech.json")
            if native and args.reference_run is not None:
                from akas.benchmarking.archive import read_file

                raw, reference = enhancement.reference_report(
                    args.history, args.reference_run, workload, weight["sha256"],
                    golden_only=True,
                )
                metadata["reference"] = dict(
                    run_id=args.reference_run, report_sha256=sha256(raw).hexdigest()
                )
                metadata["reference"].update(
                    wall_seconds={
                        row["case"]["id"]: row["summary"]["wall_seconds"]
                        for row in reference["results"]
                    },
                    real_time_factor={
                        row["case"]["id"]: row["summary"]["real_time_factor"]
                        for row in reference["results"]
                    },
                )
                cached = {}
                for row in reference["results"]:
                    cached[row["case"]["id"]] = []
                    for sample in row["samples"]:
                        raw_audio = read_file(
                            args.history,
                            "runs",
                            args.reference_run,
                            sample["voice_wav"],
                        )
                        from akas.models.voice_isolator.audio import wav_samples

                        values, _ = wav_samples(raw_audio)
                        cached[row["case"]["id"]].append(dict(voice=values))
            else:
                cached = None
            measure(session, recorder, report, workload, root, metadata, cached)
            status = (
                "passed"
                if all(row["status"] == "ok" for row in report["results"])
                else "failed"
            )
    except (TimeoutError, subprocess.TimeoutExpired) as error:
        metadata["error"] = str(error)
        status = "budget_exhausted"
    except KeyboardInterrupt:
        metadata["error"] = "interrupted"
        status = "interrupted"
    except Exception as error:
        metadata["error"] = f"{type(error).__name__}: {error}"
    finally:
        if log.is_file():
            recorder.artifact(log, log.name)
        record = recorder.update(report, final=True, status=status)
    if metadata.get("error"):
        print(metadata["error"], file=sys.stderr, flush=True)
    return record


if __name__ == "__main__":
    if sys.argv[1] == "inworld":
        upstream_worker()
    elif sys.argv[1] == "akas":
        akas_worker()
    else:
        raise ValueError("enhancement worker must be inworld or akas")
