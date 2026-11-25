"""Measure complete speech requests on AKAS and the pinned llama.cpp server."""

from array import array
from contextlib import contextmanager
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import platform
from queue import Empty, Queue
from socket import socket
from statistics import median
import subprocess
import sys
from threading import Thread
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import wave

from akas.benchmarking.history import Recorder
from akas.benchmarking.provenance import digest
from akas.benchmarking import transcription
from akas.benchmarking.workload import load


def checked_file(root, declaration):
    path = (root / declaration["path"]).resolve()
    actual = digest(path)
    if actual != declaration["sha256"]:
        raise ValueError(f"checkpoint checksum mismatch: {path}")
    return dict(path=str(path), sha256=actual, source=declaration["source"])


def request(url, deadline, payload=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("speech probe deadline reached")
    req = Request(
        url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=remaining) as response:
        return response.headers, response.read()


def wait_for_server(process, url, deadline):
    while process.poll() is None:
        try:
            _, data = request(url + "/health", deadline)
            if json.loads(data)["status"] == "ok":
                return
        except HTTPError as error:
            if error.code != 503:
                raise
            error.close()
        except URLError as error:
            if time.monotonic() >= deadline:
                raise TimeoutError("speech server startup deadline reached") from error
        time.sleep(min(0.2, max(0, deadline - time.monotonic())))
    raise RuntimeError(f"llama-server exited with code {process.returncode}")


def wav_sample(headers, pcm, sample_rate):
    if (
        headers.get_content_type() != "audio/pcm"
        or headers.get_param("encoding") != "signed-int"
        or headers.get_param("bits") != "16"
        or headers.get_param("rate") != str(sample_rate)
        or not pcm
        or len(pcm) % 2
    ):
        raise ValueError("expected nonempty mono PCM16 at the declared sample rate")
    frames = array("h", pcm)
    if sys.byteorder != "little":
        frames.byteswap()
    peak = max(abs(min(frames)), abs(max(frames)))
    if not peak:
        raise ValueError("speech server returned silence")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        wav.writeframes(pcm)
    return output.getvalue(), dict(
        samples=len(frames), audio_seconds=len(frames) / sample_rate, peak_pcm16=peak
    )


def measure(url, deadline, recorder, report, workload, speech):
    meta = report["metadata"]
    for index, item in enumerate(workload["utterances"]):
        row = dict(case=item, status="error", samples=[])
        report["results"].append(row)
        payload = dict(
            workload["request"],
            input=item["text"],
            voice=item.get("voice", speech["voice"]),
        )
        if "lang" in item:
            payload["lang"] = item["lang"]
        print(f"MEASURE {meta['provider']} {item['id']}", flush=True)
        errors = []
        for iteration in range(meta["warmup"] + meta["repeats"]):
            try:
                start = time.perf_counter()
                headers, pcm = request(url + "/tts", deadline, payload)
                elapsed = time.perf_counter() - start
                audio, sample = wav_sample(headers, pcm, speech["sample_rate_hz"])
                if meta["runtime"] == "akas":
                    sample["terminated"] = headers.get("X-AKAS-Terminated") == "true"
                    sample["audio_frames"] = int(
                        headers.get("X-AKAS-Audio-Frames", "0")
                    )
                    if not sample["terminated"]:
                        errors.append(
                            f"Request {iteration + 1}: reached the frame limit"
                        )
            except HTTPError as error:
                with error:
                    message = error.read(1024).decode("utf-8", errors="replace").strip()
                errors.append(f"Request {iteration + 1}: HTTP {error.code} {message}")
                continue
            except ValueError as error:
                errors.append(f"Request {iteration + 1}: {error}")
                continue
            if iteration < meta["warmup"]:
                continue
            repeat = iteration - meta["warmup"] + 1
            name = f"audio/{index + 1:02d}-{repeat:02d}.wav"
            path = recorder.directory / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(audio)
            recorder.artifact(path, name)
            sample.update(
                wall_seconds=elapsed,
                real_time_factor=elapsed / sample["audio_seconds"],
                wav=name,
                repeat=repeat,
                wav_sha256=recorder.record["artifacts"][-1]["sha256"],
            )
            row["samples"].append(sample)
        if row["samples"]:
            row["summary"] = {
                key: median(sample[key] for sample in row["samples"])
                for key in ("wall_seconds", "audio_seconds", "real_time_factor")
            }
        row["correctness"] = dict(
            non_silent_pcm=len(row["samples"]) == meta["repeats"],
            deterministic_pcm=len(row["samples"]) == meta["repeats"]
            and len({s["wav_sha256"] for s in row["samples"]}) == 1,
            full_model_reference=False,
        )
        if errors:
            row["error"] = " | ".join(errors)
        else:
            row["status"] = "ok"
        recorder.update(report)
        print(
            json.dumps(row.get("error", row.get("summary")), ensure_ascii=True),
            flush=True,
        )


@contextmanager
def speech_server(command, env, log, url, deadline):
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        wait_for_server(process, url, deadline)
        yield None
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@contextmanager
def transcription_process(
    command, env, log, transcript_log, deadline, *, native, limit
):
    with (
        transcript_log.open("wb") as output,
        subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env
        ) as process,
    ):
        chunks = Queue()

        def read_stdout():
            try:
                while data := process.stdout.read1(8192):
                    output.write(data)
                    output.flush()
                    chunks.put(data)
            finally:
                chunks.put(None)

        def receive(marker):
            data = bytearray()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("transcription process deadline reached")
                try:
                    chunk = chunks.get(timeout=remaining)
                except Empty as error:
                    raise TimeoutError(
                        "transcription process deadline reached"
                    ) from error
                if chunk is None:
                    raise RuntimeError(
                        f"transcription process closed its output ({process.poll()})"
                    )
                data.extend(chunk)
                normalized = bytes(data).replace(b"\r\n", b"\n")
                if normalized.endswith(marker):
                    return normalized

        def send(text):
            process.stdin.write(text.encode("utf-8"))
            process.stdin.flush()

        def native_response():
            for line in reversed(receive(b"\n").splitlines()):
                if line.startswith(b"ASR "):
                    return json.loads(line[4:])
            raise ValueError("invalid native transcription response")

        def invoke(path, language):
            if native:
                payload = json.dumps(
                    dict(path=str(path), language=language, limit=limit)
                )
                start = time.perf_counter()
                send(payload + "\n")
                response = native_response()
                return response, time.perf_counter() - start
            # Start each request with an empty text and audio context.
            send("/clear\n")
            receive(b"\n> ")
            start = time.perf_counter()
            send(f"/audio {path}\n")
            if b" loaded\n" not in receive(b"\n> "):
                raise ValueError("llama.cpp did not load the WAV")
            send(f"lang:{language}\n")
            line = receive(b"\n> ")[:-3]
            elapsed = time.perf_counter() - start
            # b12521's mtmd CLI emits a newline on EOG, before its next prompt.
            return dict(
                text=line.decode("utf-8").strip(),
                finish_reason="stop" if line.endswith(b"\n") else "length",
            ), elapsed

        reader = Thread(target=read_stdout, daemon=True)
        reader.start()
        try:
            if native:
                props = native_response()
            else:
                receive(b"\n> ")
                props = {}
            yield invoke, props
        finally:
            if process.poll() is None:
                process.stdin.close()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            reader.join()


def transcription_worker(factory, configuration):
    from importlib import import_module

    module, name = factory.split(":")
    with getattr(import_module(module), name)(**json.loads(configuration)) as service:
        print("ASR " + json.dumps(service.metadata), flush=True)
        for line in sys.stdin:
            payload = json.loads(line)
            try:
                result = service.transcribe(
                    Path(payload["path"]).read_bytes(),
                    language=payload["language"],
                    limit=payload["limit"],
                    **(
                        {"profile": True}
                        if os.environ.get("AKAS_ASR_PROFILE") == "1"
                        else {}
                    ),
                )
                response = dict(
                    text=result.text,
                    token_ids=result.token_ids,
                    prompt_tokens=result.prompt_tokens,
                    finish_reason="stop" if result.terminated else "length",
                )
                if (timings := getattr(result, "timings", None)) is not None:
                    response["phase_seconds"] = timings
            except (OSError, ValueError) as error:
                response = dict(error=str(error))
            print("ASR " + json.dumps(response), flush=True)


def measure_transcription(invoke, recorder, report, workload, root):
    meta = report["metadata"]
    for index, item in enumerate(workload["utterances"], 1):
        path = (root / item["path"]).resolve()
        audio = path.read_bytes()
        if sha256(audio).hexdigest() != item["sha256"]:
            raise ValueError(f"ASR input checksum mismatch: {path}")
        with wave.open(io.BytesIO(audio), "rb") as wav:
            audio_seconds = wav.getnframes() / wav.getframerate()
        if audio_seconds <= 0:
            raise ValueError(f"ASR input is empty: {path}")
        recorder.artifact(path, f"audio/{index:02d}.wav")
        row = dict(case=item, status="error", samples=[])
        report["results"].append(row)
        print(f"MEASURE {meta['provider']} {item['id']}", flush=True)
        errors = []
        for iteration in range(1, meta["warmup"] + meta["repeats"] + 1):
            try:
                response, elapsed = invoke(path, item["language"])
                if "error" in response:
                    raise ValueError(response["error"])
                transcript = response["text"].strip()
                finish = response["finish_reason"]
                if not transcript or finish != "stop":
                    raise ValueError(
                        f"empty or incomplete transcript (finish reason {finish!r})"
                    )
            except (AttributeError, IndexError, KeyError, ValueError) as error:
                errors.append(f"request {iteration}: {error}")
                continue
            if iteration <= meta["warmup"]:
                continue
            sample = dict(
                repeat=iteration - meta["warmup"],
                transcript=transcript,
                finish_reason=finish,
                prompt_tokens=response.get("prompt_tokens"),
                wall_seconds=elapsed,
                audio_seconds=audio_seconds,
                real_time_factor=elapsed / audio_seconds,
                error_rate=transcription.error_rate(
                    item["text"], transcript, item["metric"]
                ),
            )
            if "token_ids" in response:
                sample["token_ids"] = response["token_ids"]
            if "phase_seconds" in response:
                sample["phase_seconds"] = response["phase_seconds"]
            row["samples"].append(sample)
        samples = row["samples"]
        complete = len(samples) == meta["repeats"]
        deterministic = complete and all(
            (s["transcript"], s.get("token_ids"))
            == (samples[0]["transcript"], samples[0].get("token_ids"))
            for s in samples
        )
        row["correctness"] = dict(
            complete=complete, deterministic=deterministic, full_model_reference=False
        )
        if samples:
            row["summary"] = {
                key: median(sample[key] for sample in samples)
                for key in (
                    "wall_seconds",
                    "audio_seconds",
                    "real_time_factor",
                    "error_rate",
                )
            }
        if errors or not deterministic:
            row["error"] = " | ".join(
                errors or ["greedy transcripts differed across repetitions"]
            )
        else:
            row["status"] = "ok"
        recorder.update(report)
        print(
            json.dumps(row.get("error", row.get("summary")), ensure_ascii=True),
            flush=True,
        )


def reference_report(args, metadata):
    from akas.benchmarking.archive import read_file
    from akas.benchmarking.history import load_record

    run_id = args.reference_run
    left = load_record(args.history, run_id)
    raw = read_file(args.history, "runs", run_id, "report.json")
    checksum = sha256(raw).hexdigest()
    report = json.loads(raw)
    reference = report["metadata"]
    match = (
        "protocol",
        "speech_workload",
        "model",
        "weight_format",
        "checkpoint_sha256",
        "sample_rate_hz",
        "warmup",
        "repeats",
    )
    match += ("context",) if metadata["mode"] == "asr" else ("voice",)
    if (
        left["status"] not in ("passed", "failed")
        or reference["runtime"] != "llama"
        or reference["provider"] != "vulkan"
        or checksum
        != next(a["sha256"] for a in left["artifacts"] if a["path"] == "report.json")
        or any(reference[key] != metadata[key] for key in match)
        or reference["projector"]["sha256"] != metadata["projector"]["sha256"]
        or reference["device"]["name"].split(" (", 1)[0] != metadata["device"]["name"]
        or [row["case"] for row in report["results"]]
        != metadata["speech_workload"]["definition"]["utterances"]
    ):
        raise ValueError("the retained llama.cpp Vulkan speech run does not match")
    metadata["reference"] = dict(run_id=run_id, report_sha256=checksum)
    return {row["case"]["id"]: row for row in report["results"]}


def llama_runtime(binary, speech, *, vulkan, device):
    version = subprocess.check_output(
        [str(binary), "--version"], text=True, stderr=subprocess.STDOUT, timeout=10
    )
    if speech["revision"][:9] not in version:
        raise ValueError("llama.cpp executable does not match the declared revision")
    metadata = dict(
        runtime_revision=speech["revision"],
        runtime_version=version.strip(),
        binary_sha256=digest(binary),
        runtime_files={
            path.name: digest(path)
            for pattern in ("*.dll", "*.so*")
            for path in sorted(binary.parent.glob(pattern))
            if path.is_file()
        },
    )
    if vulkan:
        listing = subprocess.check_output(
            [str(binary), "--list-devices"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        adapter = next(
            (
                line.strip()
                for line in listing.splitlines()
                if line.strip().startswith(device + ":")
            ),
            "",
        )
        if not adapter or any(
            name in adapter.lower() for name in ("llvmpipe", "swiftshader", "lavapipe")
        ):
            raise ValueError("hardware Vulkan0 is absent")
        metadata["device"] = dict(name=adapter.split(": ", 1)[1], listing=listing)
    elif sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as cpu:
                name, _ = winreg.QueryValueEx(cpu, "ProcessorNameString")
        except OSError:
            pass
        else:
            metadata["device"] = dict(name=name.strip())
    return metadata


def run(args, definition_path, model, speech, weight_format):
    root = Path(__file__).resolve().parents[2]
    workload = load(args.workload or speech["workload"])
    mode = model["modes"][0]
    asr = mode == "asr"
    if asr:
        transcription.validate(workload)
    elif (
        workload["schema"] != "akas.speech-workload.v1"
        or workload["protocol"] != "client-speech-pcm-v1"
        or workload["request"]["response_format"] != "pcm16"
        or workload["request"]["stream"] is not False
        or workload["warmup"] < 1
        or workload["repeats"] < 1
        or not workload["utterances"]
    ):
        raise ValueError(
            "speech requires a non-streaming PCM16 workload with warmup and repeats"
        )
    weight = checked_file(root, model["weights"][weight_format])
    projector = checked_file(root, speech["projector"])
    native = args.runtime == "akas"
    if not asr:
        with socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
    vulkan = args.provider == "vulkan"
    device = "Vulkan0" if vulkan else "none"
    if native:
        bundle = args.bundle.resolve(strict=True)
        factory = model["runtime_options"]["akas"]["speech_factory"]
        configuration = json.dumps(
            dict(
                checkpoint=weight["path"],
                projector=projector["path"],
                bundle=str(bundle),
            )
        )
        command = (
            [
                sys.executable,
                "-u",
                "-m",
                "tools.baselines.speech",
                factory,
                configuration,
            ]
            if asr
            else [
                sys.executable,
                "-u",
                "-m",
                "akas.cli",
                "serve",
                "--speech",
                "--factory",
                factory,
                "--config",
                configuration,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ]
        )
    else:
        binary = (args.binary or root / speech["binary"]).resolve(strict=True)
        command = [
            str(binary),
            "-m",
            weight["path"],
            "-mm",
            projector["path"],
            "--device",
            device,
            "--mmproj-device",
            device,
            "-ngl",
            "all" if vulkan else "0",
            "--mmproj-offload" if vulkan else "--no-mmproj-offload",
            "-c",
            str(speech.get("context", 1024)),
            "-t",
            "6",
            "-tb",
            "6",
            "--offline",
        ]
        if asr:
            command.extend(
                (
                    "-n",
                    str(workload["request"]["max_tokens"]),
                    "--temp",
                    "0",
                    "--jinja",
                    "--log-colors",
                    "off",
                )
            )
        else:
            command.extend(
                (
                    "-np",
                    "1",
                    "--no-cache-prompt",
                    "--cache-ram",
                    "0",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--no-ui",
                    "--verbosity",
                    "4",
                )
            )
    metadata = dict(
        protocol=workload["protocol"],
        speech_workload=dict(id=workload["id"], definition=workload),
        runtime=args.runtime,
        provider="webgpu" if native else args.provider,
        model=model["model"],
        mode=mode,
        purpose="speech-demo",
        weight_format=weight_format,
        checkpoint=weight["path"],
        checkpoint_sha256=weight["sha256"],
        checkpoint_source=weight["source"],
        projector=projector,
        model_definition_sha256=digest(definition_path),
        device=dict(name=platform.processor() or platform.machine()),
        sample_rate_hz=speech["sample_rate_hz"],
        warmup=workload["warmup"],
        repeats=args.repeats or workload["repeats"],
        configuration=command,
        deadline_seconds=args.seconds,
        environment={
            name: value
            for name, value in os.environ.items()
            if name.startswith(("GGML_VK_", "VK_", "LLAMA_ARG_", "MTMD_"))
            and not any(secret in name for secret in ("KEY", "TOKEN", "PASSWORD"))
        },
    )
    if asr:
        metadata["context"] = speech["context"]
    else:
        metadata["voice"] = speech["voice"]
    dependency_sources = None
    if native:
        import tensor

        tensor_source = root.parent / "tensor"
        if Path(tensor.__file__).resolve().is_relative_to(tensor_source):
            dependency_sources = {"tensor": tensor_source}
    recorder = Recorder(
        args.history,
        metadata,
        kind="configuration" if native else "baseline",
        change="Measure warm speech transcription"
        if asr
        else "Measure warm speech synthesis",
        dependency_sources=dependency_sources,
    )
    recorder.artifact(definition_path, "model.json")
    report = dict(schema="akas.benchmark.v1", metadata=metadata, results=[])
    log_path = recorder.directory / ("runtime.log" if asr else "server.log")
    transcript_log = recorder.directory / "transcripts.log"
    deadline = time.monotonic() + args.seconds
    status = "failed"
    try:
        env = dict(os.environ, PYTHONUTF8="1")
        if native:
            env["WGPU_BACKEND_TYPE"] = "Vulkan"
            metadata["environment"]["WGPU_BACKEND_TYPE"] = "Vulkan"
            if asr and env.get("AKAS_ASR_PROFILE") == "1":
                metadata["environment"]["AKAS_ASR_PROFILE"] = "1"
            metadata["runtime_revision"] = recorder.record["source"]["revision"]
            recorder.artifact(bundle / "speech.json", "speech.json")
        else:
            metadata.update(llama_runtime(binary, speech, vulkan=vulkan, device=device))
        with log_path.open("wb") as log:
            client = (
                transcription_process(
                    command,
                    env,
                    log,
                    transcript_log,
                    deadline,
                    native=native,
                    limit=workload["request"]["max_tokens"],
                )
                if asr
                else speech_server(command, env, log, url, deadline)
            )
            with client as session:
                reference = None
                if native:
                    props = (
                        session[1]
                        if asr
                        else json.loads(request(url + "/props", deadline)[1])
                    )
                    metadata.update(props)
                    if metadata["program_sha256"] != digest(bundle / "speech.json"):
                        raise ValueError(
                            "speech server loaded a different compiled bundle"
                        )
                    program = json.loads(
                        (bundle / "speech.json").read_text(encoding="utf-8")
                    )
                    if "codec" in program["phases"]:
                        metadata["compiled_codec_frames"] = sorted(
                            map(int, program["phases"]["codec"])
                        )
                    for phases in program["phases"].values():
                        for entry in phases.values():
                            recorder.artifact(bundle / entry["path"], entry["path"])
                    if args.reference_run:
                        reference = reference_report(args, metadata)
                if asr:
                    measure_transcription(session[0], recorder, report, workload, root)
                else:
                    measure(url, deadline, recorder, report, workload, speech)
                if reference:
                    for row in report["results"]:
                        baseline_row = reference[row["case"]["id"]]
                        if row["status"] != "ok" or baseline_row["status"] != "ok":
                            continue
                        baseline = baseline_row["summary"]
                        row["comparison"] = dict(
                            llama_vulkan_real_time_factor=baseline["real_time_factor"],
                            speedup=baseline["real_time_factor"]
                            / row["summary"]["real_time_factor"],
                            wall_speedup=baseline["wall_seconds"]
                            / row["summary"]["wall_seconds"],
                        )
                        if asr:
                            row["comparison"]["transcript_match"] = (
                                row["samples"][0]["transcript"]
                                == baseline_row["samples"][0]["transcript"]
                            )
                        print(
                            row["case"]["id"], json.dumps(row["comparison"]), flush=True
                        )
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
        for path in (log_path, transcript_log):
            if path.is_file():
                recorder.artifact(path, path.name)
        record = recorder.update(report, final=True, status=status)
    if metadata.get("error"):
        print(metadata["error"], file=sys.stderr, flush=True)
    return record


if __name__ == "__main__":
    transcription_worker(*sys.argv[1:])
