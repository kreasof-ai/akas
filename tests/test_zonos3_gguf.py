"""Exercise the published ZONOS3 weights in AKAS's compiled WebGPU path."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from akas.models.zonos3 import Zonos3MiniCheckpoint, compose
from akas.models.zonos3.model import codebook_logits, feedback_embedding, feedback_ids


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    os.environ.get("AKAS_ZONOS3_VULKAN") != "1",
    reason="requires both published ZONOS3 GGUF files and a hardware Vulkan adapter",
)
def test_zonos3_checkpoint_operations_on_vulkan(tmp_path):
    from akas.benchmarking.history import Recorder
    from akas.benchmarking.provenance import digest
    from akas.compilation.compile import produce
    from akas.compilation.tuning import SearchSettings
    from akas.execution.forward import Forward
    from akas.benchmarking.workload import load
    from tools.baselines.__main__ import model_definition
    from tools.baselines.speech import checked_file
    from tools.context_sweep import vulkan_device

    model_path, model = model_definition(ROOT / "benchmarks/models", "zonos3-mini")
    speech = model["runtime_options"]["llama"]["speech"]
    weight = checked_file(ROOT, model["weights"]["q8_0"])
    projector = checked_file(ROOT, speech["projector"])
    checkpoint = Zonos3MiniCheckpoint(weight["path"], projector["path"])
    config = checkpoint.config
    assert checkpoint.voice_rows().shape == (180, config.width)
    inputs = {
        "text_embedding": np.array([checkpoint.tokenizer.bos], dtype="int32"),
        "gdn_input": np.random.default_rng(17)
        .normal(0, 0.25, config.width)
        .astype("float32"),
        "audio_feedback": feedback_ids(config, [0, 1, 4095, 4096, 4097, 5, 6, 7, 8, 9]),
    }
    inputs["audio_head"] = inputs["gdn_input"]
    assert inputs["audio_feedback"].tolist() == [
        code + index * 4098
        for index, code in enumerate([0, 1, 4095, 4096, 4097, 5, 6, 7, 8, 9])
    ]
    metadata = dict(
        model=model["model"],
        runtime="akas",
        mode="tts",
        purpose="operation-qualification",
        provider="webgpu",
        protocol="zonos3-checkpoint-operation-probe-v1",
        weight_format="q8_0",
        checkpoint=weight["path"],
        checkpoint_sha256=weight["sha256"],
        checkpoint_source=weight["source"],
        projector=projector,
        model_definition_sha256=digest(model_path),
        checks=[],
    )
    recorder = Recorder(
        ROOT / "benchmarks",
        metadata,
        kind="qualification",
        change="Qualify ZONOS3 checkpoint operations on the native AKAS Vulkan path",
    )
    recorder.artifact(model_path, "model.json")
    report = dict(schema="akas.benchmark.v1", metadata=metadata, results=[])
    status = "failed"
    try:
        binary = ROOT / speech["binary"]
        executable = binary.with_name("llama-tokenize" + binary.suffix)
        metadata["tokenizer_binary_sha256"] = digest(executable)
        metadata["prompt_ids"] = {}
        for item in load(speech["workload"])["utterances"]:
            command = [
                str(executable),
                "-m",
                weight["path"],
                "--device",
                "none",
                "--ids",
                "--no-parse-special",
                "-p",
                item["text"],
            ]
            ids = json.loads(subprocess.check_output(command, text=True, timeout=20))
            prompt = checkpoint.prompt_ids(item["text"])
            assert prompt[3:] == ids
            metadata["prompt_ids"][item["id"]] = prompt

        with vulkan_device() as device:
            metadata["device"] = dict(
                device.info, name=device.info["adapter"]["device"]
            )
            metadata["kernels"] = {}
            for phase, value in inputs.items():
                graph = compose(checkpoint, phase=phase, provider="webgpu")
                directory = tmp_path / phase
                manifest = produce(
                    graph,
                    directory,
                    provider="webgpu",
                    device=device,
                    search_settings=SearchSettings(budget=2, samples=3, repeats=3),
                )
                recorder.artifact(directory / "forward.json", f"{phase}/forward.json")
                metadata["kernels"][phase] = manifest["kernels"]
                weight_name = next(
                    n for n, b in graph.buffers.items() if b.role == "weight"
                )
                source = checkpoint.source(weight_name)
                expected = (
                    np.concatenate([source.row(weight_name, int(i)) for i in value])
                    if phase in ("text_embedding", "audio_feedback")
                    else source.array(weight_name, dtype=np.float64)
                    @ value.astype(np.float64)
                )
                with ExitStack() as resources:
                    supplied = checkpoint.upload(device, (weight_name,))
                    resources.callback(supplied[weight_name].release)
                    output = next(
                        b for b in graph.buffers.values() if b.role == "output"
                    )
                    input_name = next(
                        n for n, b in graph.buffers.items() if b.role == "input"
                    )
                    supplied[input_name] = device.from_numpy(value)
                    resources.callback(supplied[input_name].release)
                    supplied[output.name] = device.empty(output.shape, output.dtype)
                    resources.callback(supplied[output.name].release)
                    with Forward(directory, device, supplied, graphs=True) as forward:
                        forward.run()
                        device.synchronize()
                        actual = supplied[output.name].to_numpy()

                bitwise = np.array_equal(actual, expected.astype(np.float32))
                error = float(np.max(np.abs(actual - expected)))
                metadata["checks"].append(
                    dict(
                        output=phase,
                        elements=actual.size,
                        max_abs_error=error,
                        bitwise=bitwise,
                    )
                )
                if phase in ("text_embedding", "audio_feedback"):
                    np.testing.assert_array_equal(actual, expected)
                else:
                    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=4e-5)
                if phase == "audio_feedback":
                    np.testing.assert_array_equal(
                        feedback_embedding(config, actual)[0],
                        expected.reshape(10, config.width).sum(axis=0),
                    )
                if phase == "audio_head":
                    np.testing.assert_allclose(
                        codebook_logits(config, actual).reshape(-1),
                        15 * np.tanh(expected / 15),
                        rtol=3e-5,
                        atol=4e-5,
                    )
        status = "passed"
    except Exception as error:
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record = recorder.update(report, final=True, status=status)
        print(
            f"ZONOS3 operation evidence {record['id']} {record['status']}", flush=True
        )
    assert record["status"] == "passed"


@pytest.mark.skipif(
    os.environ.get("AKAS_ZONOS3_VULKAN") != "1",
    reason="requires the published ZONOS3 GGUF pair and a hardware Vulkan adapter",
)
def test_zonos3_backbone_matches_reference_across_phases():
    from akas.benchmarking.history import Recorder
    from akas.benchmarking.provenance import digest
    from akas.benchmarking.workload import load
    from akas.compilation.compile import produce
    from akas.compilation.tuning import SearchSettings
    from akas.models.zonos3 import Zonos3Mini
    from tests.support.zonos3_reference import reference
    from tools.baselines.__main__ import model_definition
    from tools.baselines.speech import checked_file
    from tools.context_sweep import vulkan_device

    model_path, model = model_definition(ROOT / "benchmarks/models", "zonos3-mini")
    speech = model["runtime_options"]["llama"]["speech"]
    weight = checked_file(ROOT, model["weights"]["q8_0"])
    projector = checked_file(ROOT, speech["projector"])
    checkpoint = Zonos3MiniCheckpoint(weight["path"], projector["path"])
    utterance = load(speech["workload"])["utterances"][0]
    tokens = checkpoint.prompt_ids(utterance["text"])
    embeddings = np.stack(
        [checkpoint.backbone.row("token_embd.weight", i) for i in tokens]
    )
    metadata = dict(
        runtime="akas",
        mode="tts",
        purpose="backbone-qualification",
        provider="webgpu",
        protocol="zonos3-backbone-reference-v1",
        weight_format="q8_0",
        model=model["model"],
        checkpoint_sha256=weight["sha256"],
        projector=projector,
        model_definition_sha256=digest(model_path),
        checks=[],
        tokens=tokens,
    )
    recorder = Recorder(
        ROOT / "benchmarks",
        metadata,
        kind="qualification",
        change="Qualify ZONOS3 backbone and shared state against the host reference",
    )
    recorder.artifact(model_path, "model.json")
    report = dict(schema="akas.benchmark.v1", metadata=metadata, results=[])
    status = "failed"
    try:
        expected = reference(checkpoint, embeddings)
        directory = ROOT / "build/zonos3-mini/backbone-qualification"
        context = 256
        phases = {1: "decode.json", len(tokens) - 1: "prefill.json"}
        with vulkan_device() as device:
            metadata["device"] = dict(
                device.info, name=device.info["adapter"]["device"]
            )
            metadata["context"] = context
            for rows, filename in phases.items():
                print(f"Compiling ZONOS3 {rows}-row backbone", flush=True)
                graph = compose(
                    checkpoint, phase="backbone", rows=rows, context=context
                )
                produce(
                    graph,
                    directory,
                    filename=filename,
                    provider="webgpu",
                    device=device,
                    search_settings=SearchSettings(budget=1, samples=2, repeats=1),
                )
                recorder.artifact(directory / filename, filename)
            with Zonos3Mini(checkpoint, directory, device, phases=phases) as backbone:
                chunked = np.concatenate(
                    [
                        backbone.advance(embeddings[:-1]),
                        backbone.advance(embeddings[-1:]),
                    ]
                )
                backbone.reset()
                stepped = np.concatenate(
                    [backbone.advance(row[None]) for row in embeddings]
                )
        for kind, output, target in (
            ("host", chunked, expected),
            ("incremental", stepped, chunked),
        ):
            diff = output - target
            check = dict(
                name=kind,
                bitwise=bool(np.array_equal(output, target.astype("float32"))),
                rms=float(np.sqrt(np.mean(diff**2))),
                max_abs=float(np.max(np.abs(diff))),
            )
            metadata["checks"].append(check)
            print(check, flush=True)
            assert check["rms"] <= 0.002 and check["max_abs"] <= 0.02
        status = "passed"
    except BaseException as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record = recorder.update(report, final=True, status=status)
        print(f"ZONOS3 backbone evidence {record['id']} {record['status']}", flush=True)
    assert record["status"] == "passed"


@pytest.mark.skipif(
    os.environ.get("AKAS_ZONOS3_SPEECH") != "1",
    reason="compiles and runs both ZONOS3 speech prompts on a hardware Vulkan adapter",
)
def test_zonos3_synthesizes_speech_workload(tmp_path):
    from time import perf_counter
    import wave

    from akas.benchmarking.history import Recorder
    from akas.benchmarking.provenance import digest
    from akas.benchmarking.workload import load
    from akas.compilation import compile_checkpoint
    from akas.compilation.tuning import SearchSettings
    from akas.models.zonos3 import Zonos3MiniSynthesis
    from tools.baselines.__main__ import model_definition
    from tools.baselines.speech import checked_file
    from tools.context_sweep import vulkan_device

    model_path, model = model_definition(ROOT / "benchmarks/models", "zonos3-mini")
    speech = model["runtime_options"]["llama"]["speech"]
    weight = checked_file(ROOT, model["weights"]["q8_0"])
    projector = checked_file(ROOT, speech["projector"])
    workload = load(speech["workload"])
    metadata = dict(
        runtime="akas",
        mode="tts",
        purpose="speech-preflight",
        model=model["model"],
        provider="webgpu",
        weight_format="q8_0",
        protocol="zonos3-speech-preflight-v1",
        checkpoint_sha256=weight["sha256"],
        projector=projector,
        workload=workload,
        model_definition_sha256=digest(model_path),
        checks=[],
    )
    recorder = Recorder(
        ROOT / "benchmarks",
        metadata,
        kind="qualification",
        change="Compile all ZONOS3 speech stages and verify the generated PCM",
    )
    recorder.artifact(model_path, "model.json")
    report = dict(schema="akas.benchmark.v1", metadata=metadata, results=[])
    status = "failed"
    directory = tmp_path / "speech"
    try:
        with vulkan_device() as device:
            metadata["device"] = dict(
                device.info, name=device.info["adapter"]["device"]
            )
            program = compile_checkpoint(
                Zonos3MiniSynthesis,
                weight["path"],
                directory,
                profile="speech",
                checkpoint_options={"projector": projector["path"]},
                provider="webgpu",
                context=1024,
                prefill_rows=64,
                codec_frames=(32, 64),
                device=device,
                search_settings=SearchSettings(budget=1, samples=2, repeats=1),
            )
            recorder.artifact(directory / "speech.json", "speech.json")
            for profiles in program["phases"].values():
                for entry in profiles.values():
                    recorder.artifact(directory / entry["path"], entry["path"])
            checkpoint = Zonos3MiniCheckpoint(weight["path"], projector["path"])
            with Zonos3MiniSynthesis(checkpoint, directory, device) as synthesis:
                for item in workload["utterances"]:
                    print(f"Synthesizing {item['id']}", flush=True)
                    start = perf_counter()
                    output = synthesis.synthesize(
                        item["text"],
                        voice=speech["voice"],
                        seed=workload["request"]["seed"],
                        temperature=workload["request"]["temperature"],
                        limit=workload["request"]["n_predict"],
                    )
                    elapsed = perf_counter() - start
                    peak = float(np.max(np.abs(output.samples)))
                    check = dict(
                        utterance=item["id"],
                        samples=len(output.samples),
                        frames=output.frames,
                        sample_rate=output.sample_rate,
                        terminated=output.terminated,
                        peak=peak,
                        synthesis_seconds=elapsed,
                    )
                    metadata["checks"].append(check)
                    print(check, flush=True)
                    assert output.terminated and 0.001 < peak <= 1
                    assert output.sample_rate == 44100
                    assert (
                        len(output.samples)
                        == output.frames * checkpoint.config.samples_per_frame
                    )
                    name = f"{item['id']}.wav"
                    file = recorder.directory / name
                    with wave.open(str(file), "wb") as wav:
                        wav.setparams(
                            (1, 2, output.sample_rate, 0, "NONE", "not compressed")
                        )
                        wav.writeframes(
                            np.rint(output.samples * 32767).astype("<i2").tobytes()
                        )
                    recorder.artifact(file, name)
        status = "passed"
    except BaseException as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record = recorder.update(report, final=True, status=status)
        print(f"ZONOS3 speech evidence {record['id']} {record['status']}", flush=True)
    assert record["status"] == "passed"


def test_speech_request_uses_language_preset_and_respects_explicit_voice():
    import asyncio
    from contextlib import contextmanager

    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    from akas.serving.speech import SpeechService, create_app

    checkpoint = object.__new__(Zonos3MiniCheckpoint)
    checkpoint.config = SimpleNamespace(width=8)
    english, french = "Savannah_en-American_F_1", "Gaspard_fr_M_1"
    tensor = SimpleNamespace(type=0, shape=(1, 8))
    checkpoint.projector = SimpleNamespace(
        metadata={"clip.gen.audio.default_voice": english},
        tensors={
            f"voice.{name}": tensor
            for name in ("Cillian_en-Irish_M_1", english, french)
        },
        array=lambda name: name,
    )
    selected = []

    def synthesize(text, *, lang, voice=None):
        selected.append(checkpoint.voice_rows(voice, lang=lang))
        return SimpleNamespace(
            samples=np.ones(1), sample_rate=44100, frames=1, terminated=True
        )

    @contextmanager
    def factory():
        yield SpeechService(synthesize, {}, ("en", "fr", "ja"))

    async def requests():
        async with TestClient(TestServer(create_app(factory))) as client:
            for options in (
                {},
                {"lang": "fr"},
                {"lang": "ja"},
                {"lang": "fr", "voice": english},
            ):
                async with client.post(
                    "/tts",
                    json={"input": "Example.", "response_format": "pcm16", **options},
                ) as response:
                    assert response.status == 200
                    await response.read()

    asyncio.run(requests())
    assert selected == [
        f"voice.{voice}" for voice in (english, french, english, english)
    ]
