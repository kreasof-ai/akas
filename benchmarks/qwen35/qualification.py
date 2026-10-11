"""Qualify completed serving measurements independently of cloud tooling."""


def assess(result):
    """Only the unchanged, completed load can enter the experimental beam."""
    import math

    report = result.get("client_report", {})
    if (
        report.get("workload_sha256")
        != "6e012fe014f8fc86d58d0065862c62e77e1374fccba612a7ab5d54c1679db44a"
        or result.get("device", {}).get("name") != "NVIDIA H200"
    ):
        return None
    servers = report.get("servers", [])
    if result.get("status") != "measured-experimental" or len(servers) != 1:
        return None
    points = servers[0].get("points", [])
    if len(points) != 1 or points[0].get("concurrency") != 8:
        return None
    metrics = points[0]["summary"]
    if (
        metrics["completed"],
        metrics["failed"],
        metrics["prompt_tokens"],
        metrics["output_tokens"],
    ) != (8, 0, 256000, 128000):
        return None
    checks = result.get("same_state_quality", [])
    if (
        result.get("kernel_tests_exit_code") != 0
        or len(checks) != 2
        or not all(c["passed"] for c in checks)
    ):
        return None
    elapsed = metrics["elapsed_seconds"]
    if not math.isfinite(elapsed) or elapsed <= 0:
        return None
    if not math.isclose(
        metrics["output_tokens_per_second"], 128000 / elapsed, rel_tol=1e-10
    ):
        return None
    return elapsed
