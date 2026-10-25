"""Phase linking retains qualification bytes and rejects incompatible pools."""

import hashlib
import json

import pytest

from akas.benchmarking.history import write
from tools.vulkan_link import link


def program(root, phase, generation):
    path = f"programs/{generation}/phases/{phase}.json"
    manifest = dict(schema="akas.forward.v1", provider="webgpu", target="webgpu-portable-v1",
        forward=dict(buffers=[dict(name=n,shape=s,dtype=t,role=r) for n,s,t,r in (
            ("tokens", [1], "int32", "input"), ("active", [1], "int32", "input"),
            ("predicted", [1], "int32", "output"), ("state", [1,4], "float32", "state"),
            ("weight", [4], "float32", "weight"))]), kernels={}, tuning=dict(operations=[{}]))
    write(root/path, manifest)
    data=dict(schema="akas.program.v1", provider="webgpu", target="webgpu-portable-v1",
        config=dict(width=4), max_requests=1,max_context=64, model_options=dict(kv_dtype="float16"),
        upload_options={}, policy={}, reset=None, rounds=[], phases=[dict(phase=phase,rows=1,
            chunk=1 if phase=="decode" else 32, path=path,
            sha256=hashlib.sha256((root/path).read_bytes()).hexdigest(),
            inputs=dict(tokens="tokens",lengths="active"),prediction="predicted")])
    write(root/"program.json", data)
    return data


@pytest.mark.parametrize("mismatch", [None, "config", "state", "weight", "qualification"])
def test_link_keeps_manifests_and_requires_identical_shared_abi(tmp_path, mismatch):
    a, b, out = tmp_path/"prefill", tmp_path/"decode", tmp_path/"linked"
    left = program(a, "prefill", "a"*32)
    right = program(b, "decode", "b"*32)
    if mismatch == "config":
        right["config"]["width"] = 8
        write(b/"program.json", right)
    elif mismatch:
        path=b/right["phases"][0]["path"]
        data=json.loads(path.read_text())
        if mismatch == "qualification":
            data["tuning"]={}
        else:
            next(s for s in data["forward"]["buffers"] if s["name"] == mismatch)["shape"]=[8]
        write(path,data)
        right["phases"][0]["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        write(b/"program.json",right)
    if mismatch:
        with pytest.raises(ValueError):
            link(a,b,out)
        assert not out.exists()
    else:
        result=link(a,b,out)
        assert [p["phase"] for p in result["phases"]] == ["prefill","decode"]
        for root,data in ((a,left),(b,right)):
            path=data["phases"][0]["path"]
            assert (root/path).read_bytes() == (out/path).read_bytes()
        assert len(result["linked_from"]) == 2
        with pytest.raises(ValueError, match="outside"):
            link(a,b,a/"nested")
