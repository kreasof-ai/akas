# Campaign benchmark report

UTC: 2026-10-18T13:08:43.563125Z → 2026-10-18T15:32:17.496835Z. Status: `complete`. Active time: 2.39 hours; per-configuration ceiling: 1.00 hours.

Source: `5259580d29f1d0182f1346f43cd8ced15913a4e2`; measured source tree: `ac41a32ba517441617a40976c02053d25233bb477b99e4c57a34d5599a262bea`.

Qualified AKAS deployments: **48/48**. Qualified baseline profiles: **10/18**.

- [Side-by-side tables for every requested shape](COMPARISON.md)
- [All AKAS configurations and per-shape measurements](AKAS.md)
- [All baseline profiles and per-shape measurements](BASELINES.md)
- [Every candidate attempt](CANDIDATES.md)
- [Full measurement CSV](measurements.csv) · [Candidate measurement CSV](candidate-measurements.csv) · [Candidate outcomes CSV](candidates.csv)
- [Selected kernel implementations and schedules](SCHEDULES.json) · [Portable report data](report.json) · [Campaign snapshot](campaign.json)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

## Hardware and measurement provenance

| Hardware | Driver | Baseline/runtime | Version or revision | Source | Evidence |
| --- | --- | --- | --- | --- | --- |
| NVIDIA L4 |  | akas / lfm2.5-230m / bf16 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018130852567320-14df8906](../../runs/20261018130852567320-14df8906/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / bf16 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018131006740252-43da420a](../../runs/20261018131006740252-43da420a/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / bf16 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018131122206190-b71485b2](../../runs/20261018131122206190-b71485b2/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / bf16 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018131330864843-039fb80c](../../runs/20261018131330864843-039fb80c/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / f16 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018131544975070-4cbc5243](../../runs/20261018131544975070-4cbc5243/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / f16 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018131656581346-a1667c54](../../runs/20261018131656581346-a1667c54/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / f16 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018131810905834-8f8a2aba](../../runs/20261018131810905834-8f8a2aba/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / f16 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018132021967816-affcd720](../../runs/20261018132021967816-affcd720/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / q4_0 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018132230646310-93a139e8](../../runs/20261018132230646310-93a139e8/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / q4_0 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018132342934575-aca09c85](../../runs/20261018132342934575-aca09c85/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / q4_0 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018132454970187-edf9a076](../../runs/20261018132454970187-edf9a076/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / q4_0 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018132713680721-e2e2fd23](../../runs/20261018132713680721-e2e2fd23/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / q4_k_m / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018132935164249-1ec690bc](../../runs/20261018132935164249-1ec690bc/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / q4_k_m / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018133054198297-56a5e435](../../runs/20261018133054198297-56a5e435/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / q4_k_m / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018133213394457-6422e80a](../../runs/20261018133213394457-6422e80a/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / q4_k_m / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018133440998595-e60f24f2](../../runs/20261018133440998595-e60f24f2/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / bf16 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018133711475982-66e2695e](../../runs/20261018133711475982-66e2695e/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / bf16 / mtp | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018133859638625-0eef8c58](../../runs/20261018133859638625-0eef8c58/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / bf16 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018134119261822-597c8bda](../../runs/20261018134119261822-597c8bda/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / bf16 / mtp | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018134325570980-469e5ae3](../../runs/20261018134325570980-469e5ae3/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / bf16 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018134601703707-d59fcdfa](../../runs/20261018134601703707-d59fcdfa/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / bf16 / mtp | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018134847189946-c5bec17a](../../runs/20261018134847189946-c5bec17a/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / bf16 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018135254888257-4722a5b7](../../runs/20261018135254888257-4722a5b7/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / bf16 / mtp | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018135534665929-102701fc](../../runs/20261018135534665929-102701fc/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / f16 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018135933378298-423b61d0](../../runs/20261018135933378298-423b61d0/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / f16 / mtp | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018140456927334-1cf13a3a](../../runs/20261018140456927334-1cf13a3a/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / f16 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018141112684266-3eb5b891](../../runs/20261018141112684266-3eb5b891/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / f16 / mtp | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018141518118717-e1387382](../../runs/20261018141518118717-e1387382/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / f16 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018142118202879-02440bd9](../../runs/20261018142118202879-02440bd9/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / f16 / mtp | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018142401278788-e1b20739](../../runs/20261018142401278788-e1b20739/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / f16 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018142738044876-56415fb5](../../runs/20261018142738044876-56415fb5/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / f16 / mtp | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018143024873596-fc39ef42](../../runs/20261018143024873596-fc39ef42/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / q4_0 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018143407909404-6fe39872](../../runs/20261018143407909404-6fe39872/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / q4_0 / mtp | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018143824470622-2fc170e3](../../runs/20261018143824470622-2fc170e3/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / q4_0 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018144310958228-59226851](../../runs/20261018144310958228-59226851/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / q4_0 / mtp | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018144546940645-b9caf87b](../../runs/20261018144546940645-b9caf87b/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / q4_0 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018144907492049-2f845f93](../../runs/20261018144907492049-2f845f93/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / q4_0 / mtp | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018145209029629-85607bc4](../../runs/20261018145209029629-85607bc4/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / q4_0 / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018145559662873-743978c8](../../runs/20261018145559662873-743978c8/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / q4_0 / mtp | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018145840104186-24d4d259](../../runs/20261018145840104186-24d4d259/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / q4_k_m / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018150214725744-e6b064ad](../../runs/20261018150214725744-e6b064ad/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / q4_k_m / mtp | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018150729812014-9634f119](../../runs/20261018150729812014-9634f119/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / q4_k_m / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018151237223272-af03ee15](../../runs/20261018151237223272-af03ee15/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / q4_k_m / mtp | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018151512909614-9b31261c](../../runs/20261018151512909614-9b31261c/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / q4_k_m / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018151836127904-b7ab440e](../../runs/20261018151836127904-b7ab440e/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / q4_k_m / mtp | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018152156367296-42537605](../../runs/20261018152156367296-42537605/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / q4_k_m / ar | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018152555702528-5964692d](../../runs/20261018152555702528-5964692d/run.json) |
| NVIDIA L4 |  | akas / qwen3.5-0.8b / q4_k_m / mtp | 0.1.0 | `5259580d29f1d0182f1346f43cd8ced15913a4e2` | [20261018152842513121-97464fc5](../../runs/20261018152842513121-97464fc5/run.json) |
| NVIDIA L4 | 595.91.07 | vllm / lfm2.5-230m / bf16 / ar | 0.31.0 | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015181815641257-391d76bc](../../runs/20261015181815641257-391d76bc/run.json) |
| NVIDIA L4 | 595.91.07 | sglang / lfm2.5-230m / bf16 / ar | 0.5.21 | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015190359153016-bcd95971](../../runs/20261015190359153016-bcd95971/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-230m / f16 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015192832378421-704bee0c](../../runs/20261015192832378421-704bee0c/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-230m / q4_0 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015193656299664-fe2eba27](../../runs/20261015193656299664-fe2eba27/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-230m / q4_k_m / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015194346054741-23180dda](../../runs/20261015194346054741-23180dda/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-230m / f16 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015195038132061-53344b1f](../../runs/20261015195038132061-53344b1f/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-230m / q4_0 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015201742219174-9ab41fef](../../runs/20261015201742219174-9ab41fef/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-230m / q4_k_m / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015202523676529-ef8b1d74](../../runs/20261015202523676529-ef8b1d74/run.json) |
| NVIDIA L4 | 595.91.07 | vllm / qwen3.5-0.8b / bf16 / ar | 0.31.0 | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015191912117158-bf3b040b](../../runs/20261015191912117158-bf3b040b/run.json) |
| NVIDIA L4 | 595.91.07 | vllm / qwen3.5-0.8b / bf16 / mtp | 0.31.0 | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015205432983160-cebe91d1](../../runs/20261015205432983160-cebe91d1/run.json) |
| NVIDIA L4 | 595.91.07 | sglang / qwen3.5-0.8b / bf16 / ar | 0.5.21 | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015190536066752-9e49f8e2](../../runs/20261015190536066752-9e49f8e2/run.json) |
| NVIDIA L4 | 595.91.07 | sglang / qwen3.5-0.8b / bf16 / mtp | 0.5.21 | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015190958176695-723ea8a5](../../runs/20261015190958176695-723ea8a5/run.json) |
| NVIDIA L4 | 595.91.07 | llama / qwen3.5-0.8b / f16 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015210200344123-b240f1f0](../../runs/20261015210200344123-b240f1f0/run.json) |
| NVIDIA L4 | 595.91.07 | llama / qwen3.5-0.8b / q4_0 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015211120204768-ca9ae9c9](../../runs/20261015211120204768-ca9ae9c9/run.json) |
| NVIDIA L4 | 595.91.07 | llama / qwen3.5-0.8b / q4_k_m / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015212004566075-2f3bc973](../../runs/20261015212004566075-2f3bc973/run.json) |
| NVIDIA L4 | 595.91.07 | llama / qwen3.5-0.8b / f16 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015212857450400-a5e9f4cb](../../runs/20261015212857450400-a5e9f4cb/run.json) |
| NVIDIA L4 | 595.91.07 | llama / qwen3.5-0.8b / q4_0 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015214155673053-00188f08](../../runs/20261015214155673053-00188f08/run.json) |
| NVIDIA L4 | 595.91.07 | llama / qwen3.5-0.8b / q4_k_m / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015215423483123-d676f19f](../../runs/20261015215423483123-d676f19f/run.json) |

Captured host: 4 logical CPUs; Linux-6.12.103-129.197.amzn2023.x86_64-x86_64-with-glibc2.39.

```text
index, uuid, name, driver_version, memory.total [MiB], power.limit [W], clocks.max.sm [MHz], clocks.max.memory [MHz]
0, GPU-b03cd627-998c-2476-6169-f62b483e2c27, NVIDIA L4, 595.91.07, 23034 MiB, 72.00 W, 2040 MHz, 6251 MHz
```

Full dependency versions, runtime hyperparameters, checkpoint hashes, UTC timestamps, sample values and artifact checksums are retained in the linked run records and report.json. Selected schedules include each operation’s parameters, implementation, geometry and qualification evidence.

## Selected configurations

HTTP speedup below is against the first qualified AKAS candidate in the same configuration. External baselines are shown separately in the side-by-side tables. Peak VRAM is sampled whole-device usage.

| Configuration | KV dtype | Chunk | Prefill group | Replay | Trials allowance | HTTP speedup over own reference | Peak VRAM GiB | Evidence |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| lfm2.5-230m-bf16-cuda-default-ar | float16 | 32 | 32 | True | 1 | 1.000x | 7.5 | [20261018130852567320-14df8906](../../runs/20261018130852567320-14df8906/run.json) |
| lfm2.5-230m-bf16-cuda-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.000x | 4.3 | [20261018131006740252-43da420a](../../runs/20261018131006740252-43da420a/run.json) |
| lfm2.5-230m-bf16-webgpu-default-ar | float16 | 32 | 32 | True | 1 | 1.000x | 14.0 | [20261018131122206190-b71485b2](../../runs/20261018131122206190-b71485b2/run.json) |
| lfm2.5-230m-bf16-webgpu-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.000x | 4.4 | [20261018131330864843-039fb80c](../../runs/20261018131330864843-039fb80c/run.json) |
| lfm2.5-230m-f16-cuda-default-ar | float16 | 32 | 32 | True | 1 | 1.000x | 7.5 | [20261018131544975070-4cbc5243](../../runs/20261018131544975070-4cbc5243/run.json) |
| lfm2.5-230m-f16-cuda-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.000x | 4.3 | [20261018131656581346-a1667c54](../../runs/20261018131656581346-a1667c54/run.json) |
| lfm2.5-230m-f16-webgpu-default-ar | float16 | 32 | 32 | True | 1 | 1.000x | 14.0 | [20261018131810905834-8f8a2aba](../../runs/20261018131810905834-8f8a2aba/run.json) |
| lfm2.5-230m-f16-webgpu-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.000x | 4.4 | [20261018132021967816-affcd720](../../runs/20261018132021967816-affcd720/run.json) |
| lfm2.5-230m-q4_0-cuda-default-ar | float16 | 32 | 32 | True | 1 | 1.000x | 7.2 | [20261018132230646310-93a139e8](../../runs/20261018132230646310-93a139e8/run.json) |
| lfm2.5-230m-q4_0-cuda-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.000x | 4.0 | [20261018132342934575-aca09c85](../../runs/20261018132342934575-aca09c85/run.json) |
| lfm2.5-230m-q4_0-webgpu-default-ar | float16 | 32 | 32 | True | 1 | 1.000x | 13.6 | [20261018132454970187-edf9a076](../../runs/20261018132454970187-edf9a076/run.json) |
| lfm2.5-230m-q4_0-webgpu-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.000x | 4.1 | [20261018132713680721-e2e2fd23](../../runs/20261018132713680721-e2e2fd23/run.json) |
| lfm2.5-230m-q4_k_m-cuda-default-ar | float16 | 32 | 32 | True | 1 | 1.000x | 7.2 | [20261018132935164249-1ec690bc](../../runs/20261018132935164249-1ec690bc/run.json) |
| lfm2.5-230m-q4_k_m-cuda-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.000x | 4.0 | [20261018133054198297-56a5e435](../../runs/20261018133054198297-56a5e435/run.json) |
| lfm2.5-230m-q4_k_m-webgpu-default-ar | float16 | 32 | 32 | True | 1 | 1.000x | 13.6 | [20261018133213394457-6422e80a](../../runs/20261018133213394457-6422e80a/run.json) |
| lfm2.5-230m-q4_k_m-webgpu-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.000x | 4.1 | [20261018133440998595-e60f24f2](../../runs/20261018133440998595-e60f24f2/run.json) |
| qwen3.5-0.8b-bf16-cuda-default-ar | bfloat16 | 32 | 32 | True | 1 | 1.000x | 13.5 | [20261018133711475982-66e2695e](../../runs/20261018133711475982-66e2695e/run.json) |
| qwen3.5-0.8b-bf16-cuda-default-mtp | bfloat16 | 32 | 32 | True | 1 | 1.000x | 16.3 | [20261018133859638625-0eef8c58](../../runs/20261018133859638625-0eef8c58/run.json) |
| qwen3.5-0.8b-bf16-cuda-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.000x | 10.4 | [20261018134119261822-597c8bda](../../runs/20261018134119261822-597c8bda/run.json) |
| qwen3.5-0.8b-bf16-cuda-fp8-mtp | fp8 | 32 | 32 | True | 1 | 1.000x | 12.7 | [20261018134325570980-469e5ae3](../../runs/20261018134325570980-469e5ae3/run.json) |
| qwen3.5-0.8b-bf16-webgpu-default-ar | float16 | 32 | 32 | True | 1 | 1.000x | 13.4 | [20261018134601703707-d59fcdfa](../../runs/20261018134601703707-d59fcdfa/run.json) |
| qwen3.5-0.8b-bf16-webgpu-default-mtp | float16 | 32 | 32 | True | 1 | 1.000x | 15.1 | [20261018134847189946-c5bec17a](../../runs/20261018134847189946-c5bec17a/run.json) |
| qwen3.5-0.8b-bf16-webgpu-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.000x | 10.4 | [20261018135254888257-4722a5b7](../../runs/20261018135254888257-4722a5b7/run.json) |
| qwen3.5-0.8b-bf16-webgpu-fp8-mtp | fp8 | 32 | 32 | True | 1 | 1.000x | 11.5 | [20261018135534665929-102701fc](../../runs/20261018135534665929-102701fc/run.json) |
| qwen3.5-0.8b-f16-cuda-default-ar | bfloat16 | 32 | 32 | True | 1 | 1.000x | 13.5 | [20261018135933378298-423b61d0](../../runs/20261018135933378298-423b61d0/run.json) |
| qwen3.5-0.8b-f16-cuda-default-mtp | bfloat16 | 32 | 32 | True | 1 | 1.000x | 16.2 | [20261018140456927334-1cf13a3a](../../runs/20261018140456927334-1cf13a3a/run.json) |
| qwen3.5-0.8b-f16-cuda-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.000x | 10.4 | [20261018141112684266-3eb5b891](../../runs/20261018141112684266-3eb5b891/run.json) |
| qwen3.5-0.8b-f16-cuda-fp8-mtp | fp8 | 32 | 32 | True | 1 | 1.000x | 12.6 | [20261018141518118717-e1387382](../../runs/20261018141518118717-e1387382/run.json) |
| qwen3.5-0.8b-f16-webgpu-default-ar | float16 | 32 | 32 | True | 1 | 1.000x | 13.4 | [20261018142118202879-02440bd9](../../runs/20261018142118202879-02440bd9/run.json) |
| qwen3.5-0.8b-f16-webgpu-default-mtp | float16 | 32 | 32 | True | 1 | 1.000x | 15.1 | [20261018142401278788-e1b20739](../../runs/20261018142401278788-e1b20739/run.json) |
| qwen3.5-0.8b-f16-webgpu-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.000x | 10.4 | [20261018142738044876-56415fb5](../../runs/20261018142738044876-56415fb5/run.json) |
| qwen3.5-0.8b-f16-webgpu-fp8-mtp | fp8 | 32 | 32 | True | 1 | 1.000x | 11.5 | [20261018143024873596-fc39ef42](../../runs/20261018143024873596-fc39ef42/run.json) |
| qwen3.5-0.8b-q4_0-cuda-default-ar | bfloat16 | 32 | 32 | True | 1 | 1.000x | 12.5 | [20261018143407909404-6fe39872](../../runs/20261018143407909404-6fe39872/run.json) |
| qwen3.5-0.8b-q4_0-cuda-default-mtp | bfloat16 | 32 | 32 | True | 1 | 1.000x | 15.2 | [20261018143824470622-2fc170e3](../../runs/20261018143824470622-2fc170e3/run.json) |
| qwen3.5-0.8b-q4_0-cuda-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.000x | 9.4 | [20261018144310958228-59226851](../../runs/20261018144310958228-59226851/run.json) |
| qwen3.5-0.8b-q4_0-cuda-fp8-mtp | fp8 | 32 | 32 | True | 1 | 1.000x | 11.6 | [20261018144546940645-b9caf87b](../../runs/20261018144546940645-b9caf87b/run.json) |
| qwen3.5-0.8b-q4_0-webgpu-default-ar | float16 | 32 | 32 | True | 1 | 1.000x | 12.3 | [20261018144907492049-2f845f93](../../runs/20261018144907492049-2f845f93/run.json) |
| qwen3.5-0.8b-q4_0-webgpu-default-mtp | float16 | 32 | 32 | True | 1 | 1.000x | 14.0 | [20261018145209029629-85607bc4](../../runs/20261018145209029629-85607bc4/run.json) |
| qwen3.5-0.8b-q4_0-webgpu-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.000x | 9.3 | [20261018145559662873-743978c8](../../runs/20261018145559662873-743978c8/run.json) |
| qwen3.5-0.8b-q4_0-webgpu-fp8-mtp | fp8 | 32 | 32 | True | 1 | 1.000x | 10.5 | [20261018145840104186-24d4d259](../../runs/20261018145840104186-24d4d259/run.json) |
| qwen3.5-0.8b-q4_k_m-cuda-default-ar | bfloat16 | 32 | 32 | True | 1 | 1.000x | 12.5 | [20261018150214725744-e6b064ad](../../runs/20261018150214725744-e6b064ad/run.json) |
| qwen3.5-0.8b-q4_k_m-cuda-default-mtp | bfloat16 | 32 | 32 | True | 1 | 1.000x | 15.2 | [20261018150729812014-9634f119](../../runs/20261018150729812014-9634f119/run.json) |
| qwen3.5-0.8b-q4_k_m-cuda-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.000x | 9.5 | [20261018151237223272-af03ee15](../../runs/20261018151237223272-af03ee15/run.json) |
| qwen3.5-0.8b-q4_k_m-cuda-fp8-mtp | fp8 | 32 | 32 | True | 1 | 1.000x | 11.7 | [20261018151512909614-9b31261c](../../runs/20261018151512909614-9b31261c/run.json) |
| qwen3.5-0.8b-q4_k_m-webgpu-default-ar | float16 | 32 | 32 | True | 1 | 1.000x | 12.4 | [20261018151836127904-b7ab440e](../../runs/20261018151836127904-b7ab440e/run.json) |
| qwen3.5-0.8b-q4_k_m-webgpu-default-mtp | float16 | 32 | 32 | True | 1 | 1.000x | 14.1 | [20261018152156367296-42537605](../../runs/20261018152156367296-42537605/run.json) |
| qwen3.5-0.8b-q4_k_m-webgpu-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.000x | 9.4 | [20261018152555702528-5964692d](../../runs/20261018152555702528-5964692d/run.json) |
| qwen3.5-0.8b-q4_k_m-webgpu-fp8-mtp | fp8 | 32 | 32 | True | 1 | 1.000x | 10.6 | [20261018152842513121-97464fc5](../../runs/20261018152842513121-97464fc5/run.json) |

## Complete requested AKAS coverage

| Model | Weights | Backend | KV request | Mode | Outcome | Time min | Selected run | Reason |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| lfm2.5-230m | bf16 | cuda | default | ar | [qualified; complete](AKAS.md#lfm2.5-230m-bf16-cuda-default-ar) | 1.2 | [20261018130852567320-14df8906](../../runs/20261018130852567320-14df8906/run.json) |  |
| lfm2.5-230m | bf16 | cuda | fp8 | ar | [qualified; complete](AKAS.md#lfm2.5-230m-bf16-cuda-fp8-ar) | 1.2 | [20261018131006740252-43da420a](../../runs/20261018131006740252-43da420a/run.json) |  |
| lfm2.5-230m | bf16 | webgpu | default | ar | [qualified; complete](AKAS.md#lfm2.5-230m-bf16-webgpu-default-ar) | 2.1 | [20261018131122206190-b71485b2](../../runs/20261018131122206190-b71485b2/run.json) |  |
| lfm2.5-230m | bf16 | webgpu | fp8 | ar | [qualified; complete](AKAS.md#lfm2.5-230m-bf16-webgpu-fp8-ar) | 2.2 | [20261018131330864843-039fb80c](../../runs/20261018131330864843-039fb80c/run.json) |  |
| lfm2.5-230m | f16 | cuda | default | ar | [qualified; complete](AKAS.md#lfm2.5-230m-f16-cuda-default-ar) | 1.2 | [20261018131544975070-4cbc5243](../../runs/20261018131544975070-4cbc5243/run.json) |  |
| lfm2.5-230m | f16 | cuda | fp8 | ar | [qualified; complete](AKAS.md#lfm2.5-230m-f16-cuda-fp8-ar) | 1.2 | [20261018131656581346-a1667c54](../../runs/20261018131656581346-a1667c54/run.json) |  |
| lfm2.5-230m | f16 | webgpu | default | ar | [qualified; complete](AKAS.md#lfm2.5-230m-f16-webgpu-default-ar) | 2.2 | [20261018131810905834-8f8a2aba](../../runs/20261018131810905834-8f8a2aba/run.json) |  |
| lfm2.5-230m | f16 | webgpu | fp8 | ar | [qualified; complete](AKAS.md#lfm2.5-230m-f16-webgpu-fp8-ar) | 2.1 | [20261018132021967816-affcd720](../../runs/20261018132021967816-affcd720/run.json) |  |
| lfm2.5-230m | q4_0 | cuda | default | ar | [qualified; complete](AKAS.md#lfm2.5-230m-q4_0-cuda-default-ar) | 1.2 | [20261018132230646310-93a139e8](../../runs/20261018132230646310-93a139e8/run.json) |  |
| lfm2.5-230m | q4_0 | cuda | fp8 | ar | [qualified; complete](AKAS.md#lfm2.5-230m-q4_0-cuda-fp8-ar) | 1.2 | [20261018132342934575-aca09c85](../../runs/20261018132342934575-aca09c85/run.json) |  |
| lfm2.5-230m | q4_0 | webgpu | default | ar | [qualified; complete](AKAS.md#lfm2.5-230m-q4_0-webgpu-default-ar) | 2.3 | [20261018132454970187-edf9a076](../../runs/20261018132454970187-edf9a076/run.json) |  |
| lfm2.5-230m | q4_0 | webgpu | fp8 | ar | [qualified; complete](AKAS.md#lfm2.5-230m-q4_0-webgpu-fp8-ar) | 2.4 | [20261018132713680721-e2e2fd23](../../runs/20261018132713680721-e2e2fd23/run.json) |  |
| lfm2.5-230m | q4_k_m | cuda | default | ar | [qualified; complete](AKAS.md#lfm2.5-230m-q4_k_m-cuda-default-ar) | 1.3 | [20261018132935164249-1ec690bc](../../runs/20261018132935164249-1ec690bc/run.json) |  |
| lfm2.5-230m | q4_k_m | cuda | fp8 | ar | [qualified; complete](AKAS.md#lfm2.5-230m-q4_k_m-cuda-fp8-ar) | 1.3 | [20261018133054198297-56a5e435](../../runs/20261018133054198297-56a5e435/run.json) |  |
| lfm2.5-230m | q4_k_m | webgpu | default | ar | [qualified; complete](AKAS.md#lfm2.5-230m-q4_k_m-webgpu-default-ar) | 2.5 | [20261018133213394457-6422e80a](../../runs/20261018133213394457-6422e80a/run.json) |  |
| lfm2.5-230m | q4_k_m | webgpu | fp8 | ar | [qualified; complete](AKAS.md#lfm2.5-230m-q4_k_m-webgpu-fp8-ar) | 2.5 | [20261018133440998595-e60f24f2](../../runs/20261018133440998595-e60f24f2/run.json) |  |
| qwen3.5-0.8b | bf16 | cuda | default | ar | [qualified; complete](AKAS.md#qwen3.5-0.8b-bf16-cuda-default-ar) | 1.8 | [20261018133711475982-66e2695e](../../runs/20261018133711475982-66e2695e/run.json) |  |
| qwen3.5-0.8b | bf16 | cuda | default | mtp | [qualified; complete](AKAS.md#qwen3.5-0.8b-bf16-cuda-default-mtp) | 2.3 | [20261018133859638625-0eef8c58](../../runs/20261018133859638625-0eef8c58/run.json) |  |
| qwen3.5-0.8b | bf16 | cuda | fp8 | ar | [qualified; complete](AKAS.md#qwen3.5-0.8b-bf16-cuda-fp8-ar) | 2.1 | [20261018134119261822-597c8bda](../../runs/20261018134119261822-597c8bda/run.json) |  |
| qwen3.5-0.8b | bf16 | cuda | fp8 | mtp | [qualified; complete](AKAS.md#qwen3.5-0.8b-bf16-cuda-fp8-mtp) | 2.6 | [20261018134325570980-469e5ae3](../../runs/20261018134325570980-469e5ae3/run.json) |  |
| qwen3.5-0.8b | bf16 | webgpu | default | ar | [qualified; complete](AKAS.md#qwen3.5-0.8b-bf16-webgpu-default-ar) | 2.7 | [20261018134601703707-d59fcdfa](../../runs/20261018134601703707-d59fcdfa/run.json) |  |
| qwen3.5-0.8b | bf16 | webgpu | default | mtp | [qualified; complete](AKAS.md#qwen3.5-0.8b-bf16-webgpu-default-mtp) | 4.1 | [20261018134847189946-c5bec17a](../../runs/20261018134847189946-c5bec17a/run.json) |  |
| qwen3.5-0.8b | bf16 | webgpu | fp8 | ar | [qualified; complete](AKAS.md#qwen3.5-0.8b-bf16-webgpu-fp8-ar) | 2.7 | [20261018135254888257-4722a5b7](../../runs/20261018135254888257-4722a5b7/run.json) |  |
| qwen3.5-0.8b | bf16 | webgpu | fp8 | mtp | [qualified; complete](AKAS.md#qwen3.5-0.8b-bf16-webgpu-fp8-mtp) | 3.8 | [20261018135534665929-102701fc](../../runs/20261018135534665929-102701fc/run.json) |  |
| qwen3.5-0.8b | f16 | cuda | default | ar | [qualified; complete](AKAS.md#qwen3.5-0.8b-f16-cuda-default-ar) | 5.6 | [20261018135933378298-423b61d0](../../runs/20261018135933378298-423b61d0/run.json) |  |
| qwen3.5-0.8b | f16 | cuda | default | mtp | [qualified; complete](AKAS.md#qwen3.5-0.8b-f16-cuda-default-mtp) | 6.3 | [20261018140456927334-1cf13a3a](../../runs/20261018140456927334-1cf13a3a/run.json) |  |
| qwen3.5-0.8b | f16 | cuda | fp8 | ar | [qualified; complete](AKAS.md#qwen3.5-0.8b-f16-cuda-fp8-ar) | 4.1 | [20261018141112684266-3eb5b891](../../runs/20261018141112684266-3eb5b891/run.json) |  |
| qwen3.5-0.8b | f16 | cuda | fp8 | mtp | [qualified; complete](AKAS.md#qwen3.5-0.8b-f16-cuda-fp8-mtp) | 6.0 | [20261018141518118717-e1387382](../../runs/20261018141518118717-e1387382/run.json) |  |
| qwen3.5-0.8b | f16 | webgpu | default | ar | [qualified; complete](AKAS.md#qwen3.5-0.8b-f16-webgpu-default-ar) | 2.7 | [20261018142118202879-02440bd9](../../runs/20261018142118202879-02440bd9/run.json) |  |
| qwen3.5-0.8b | f16 | webgpu | default | mtp | [qualified; complete](AKAS.md#qwen3.5-0.8b-f16-webgpu-default-mtp) | 3.6 | [20261018142401278788-e1b20739](../../runs/20261018142401278788-e1b20739/run.json) |  |
| qwen3.5-0.8b | f16 | webgpu | fp8 | ar | [qualified; complete](AKAS.md#qwen3.5-0.8b-f16-webgpu-fp8-ar) | 2.8 | [20261018142738044876-56415fb5](../../runs/20261018142738044876-56415fb5/run.json) |  |
| qwen3.5-0.8b | f16 | webgpu | fp8 | mtp | [qualified; complete](AKAS.md#qwen3.5-0.8b-f16-webgpu-fp8-mtp) | 3.7 | [20261018143024873596-fc39ef42](../../runs/20261018143024873596-fc39ef42/run.json) |  |
| qwen3.5-0.8b | q4_0 | cuda | default | ar | [qualified; complete](AKAS.md#qwen3.5-0.8b-q4_0-cuda-default-ar) | 4.3 | [20261018143407909404-6fe39872](../../runs/20261018143407909404-6fe39872/run.json) |  |
| qwen3.5-0.8b | q4_0 | cuda | default | mtp | [qualified; complete](AKAS.md#qwen3.5-0.8b-q4_0-cuda-default-mtp) | 4.8 | [20261018143824470622-2fc170e3](../../runs/20261018143824470622-2fc170e3/run.json) |  |
| qwen3.5-0.8b | q4_0 | cuda | fp8 | ar | [qualified; complete](AKAS.md#qwen3.5-0.8b-q4_0-cuda-fp8-ar) | 2.6 | [20261018144310958228-59226851](../../runs/20261018144310958228-59226851/run.json) |  |
| qwen3.5-0.8b | q4_0 | cuda | fp8 | mtp | [qualified; complete](AKAS.md#qwen3.5-0.8b-q4_0-cuda-fp8-mtp) | 3.3 | [20261018144546940645-b9caf87b](../../runs/20261018144546940645-b9caf87b/run.json) |  |
| qwen3.5-0.8b | q4_0 | webgpu | default | ar | [qualified; complete](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-default-ar) | 3.0 | [20261018144907492049-2f845f93](../../runs/20261018144907492049-2f845f93/run.json) |  |
| qwen3.5-0.8b | q4_0 | webgpu | default | mtp | [qualified; complete](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-default-mtp) | 3.8 | [20261018145209029629-85607bc4](../../runs/20261018145209029629-85607bc4/run.json) |  |
| qwen3.5-0.8b | q4_0 | webgpu | fp8 | ar | [qualified; complete](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-fp8-ar) | 2.7 | [20261018145559662873-743978c8](../../runs/20261018145559662873-743978c8/run.json) |  |
| qwen3.5-0.8b | q4_0 | webgpu | fp8 | mtp | [qualified; complete](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-fp8-mtp) | 3.5 | [20261018145840104186-24d4d259](../../runs/20261018145840104186-24d4d259/run.json) |  |
| qwen3.5-0.8b | q4_k_m | cuda | default | ar | [qualified; complete](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-default-ar) | 5.3 | [20261018150214725744-e6b064ad](../../runs/20261018150214725744-e6b064ad/run.json) |  |
| qwen3.5-0.8b | q4_k_m | cuda | default | mtp | [qualified; complete](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-default-mtp) | 5.1 | [20261018150729812014-9634f119](../../runs/20261018150729812014-9634f119/run.json) |  |
| qwen3.5-0.8b | q4_k_m | cuda | fp8 | ar | [qualified; complete](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-fp8-ar) | 2.6 | [20261018151237223272-af03ee15](../../runs/20261018151237223272-af03ee15/run.json) |  |
| qwen3.5-0.8b | q4_k_m | cuda | fp8 | mtp | [qualified; complete](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-fp8-mtp) | 3.4 | [20261018151512909614-9b31261c](../../runs/20261018151512909614-9b31261c/run.json) |  |
| qwen3.5-0.8b | q4_k_m | webgpu | default | ar | [qualified; complete](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-default-ar) | 3.3 | [20261018151836127904-b7ab440e](../../runs/20261018151836127904-b7ab440e/run.json) |  |
| qwen3.5-0.8b | q4_k_m | webgpu | default | mtp | [qualified; complete](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-default-mtp) | 4.0 | [20261018152156367296-42537605](../../runs/20261018152156367296-42537605/run.json) |  |
| qwen3.5-0.8b | q4_k_m | webgpu | fp8 | ar | [qualified; complete](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-fp8-ar) | 2.8 | [20261018152555702528-5964692d](../../runs/20261018152555702528-5964692d/run.json) |  |
| qwen3.5-0.8b | q4_k_m | webgpu | fp8 | mtp | [qualified; complete](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-fp8-mtp) | 3.7 | [20261018152842513121-97464fc5](../../runs/20261018152842513121-97464fc5/run.json) |  |

## Candidate outcomes

48 attempts: 48 passed.

| Rejection or deadline | Attempts |
| --- | ---: |
