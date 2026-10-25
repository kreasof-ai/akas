from pathlib import Path
import json
from akas.benchmarking.history import Recorder, write, digest
root=Path("build/vulkan-tune")
meta=dict(runtime="akas",purpose="tuning-evidence",model="LiquidAI/LFM2.5-2.6B",provider="webgpu",
    target="webgpu-portable-v1",weight_format="q4_0",
    scope="Artifact collection for offline kernel searches, exact rounding/layout tests, archived Tensor schedule replay, synthetic decode screening, and intermediate real-text pilots. Kernel hot-buffer/reset timings are diagnostic, not additive model throughput. Rejected numerical shortcuts and slower layouts/policies are retained. The selected full program is independently qualified again and measured on retained C4 token IDs.",
    intermediate_pilot_run_ids=["20261024202805694336-73203cb4", "20261024211804854903-fc9747b0", "20261024212834544783-2c994836"],
    selected_pilot_run_id="20261024215128041404-5569d5c7",
    repeated_baseline_run_id="20261024215220032242-9ad91991",
    historical_transfer="Tensor d5ee508: eight-chain F32 GEMV, Q6 floating dots and subgroup RMS. Padded prefill search retained as diagnostic; no clear transferred win. Short F16 chains and Q8 activation approximations not selected. Historical QAD weights/context-512 timings differ from this standard Q4_0/context-131072 campaign.",
    checkpoint_sha256="e1a61bf937bc60726e18626e97f7ee9bfd2574d95744c2ed909de98b78006fbe")
r=Recorder("benchmarks",meta,kind="preparation",parent="20261024182401411794-847a6f68",
    change="Retain Vulkan tuning evidence; this collection creates no new inference samples")
for folder in ["decode","prefill","vector","branch-prefill","attention16","attention64","attention128","channel16","channel64","split-prefill","staged-prefill","padded-prefill"]:
    for name in ["qualification.json","profile.json"]:
        path=root/folder/name
        if path.exists():r.artifact(path,"search/"+folder+"/"+name)
for path in (root/"attention-prefill"/"search-cache").glob("*.progress.json"):
    r.artifact(path,"partial-attention-search/"+path.name)
for path in root.glob("*.json"):
    r.artifact(path,"diagnostics/"+path.name)
for path in Path("build").glob("vulkan-*.log"):
    r.artifact(path,"logs/"+path.name)
r.artifact("tools/vulkan_tune.py","tools/vulkan_tune.py")
r.artifact("tools/vulkan_profile.py","tools/vulkan_profile.py")
r.artifact("tools/vulkan_decode_probe.py","tools/vulkan_decode_probe.py")
r.artifact("tools/vulkan_link.py","tools/vulkan_link.py")
for path in (root/"tensor-archive").glob("*"):
    r.artifact(path,"tensor-archive/"+path.name)
r.artifact("build/retain_tensor_archive.py","tools/retain_tensor_archive.py")
r.artifact(__file__,"archive.py")
r.update(dict(metadata=meta,results=[]),final=True,status="passed")
write(root/"evidence.json",dict(run_id=r.id,status="passed"))
print(r.id,flush=True)
