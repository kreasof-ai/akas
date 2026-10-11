"""Benchmark entry points for shared projection transforms and runtime binding."""

from akas.compilation.transforms.projections import mma_bundle
from akas.compilation.transforms.projections import selected_bundle
from akas.execution.bindings.block_fp8.optimizations.linear_projections import replace
