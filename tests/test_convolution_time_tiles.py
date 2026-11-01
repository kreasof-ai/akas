"""Time tiles carry convolution history across full, short and inactive chunks."""
import os
import pytest
from test_lfm_moe_native import test_native_operations_match_independent_references as check_operation


@pytest.mark.skipif(os.getenv("AKAS_LFM_FP8_CUDA") != "1", reason="CUDA convolution qualification")
@pytest.mark.parametrize("parameters", [
    dict(r=3, c=513, chunk=65, length=5, time_tile=32, columns=32, threads=128),
    dict(r=4, c=2048, chunk=256, length=3, time_tile=64, columns=64, threads=256),
    dict(r=3, c=129, chunk=33, length=5, time_tile=2, columns=128, threads=128),
])
def test_convolution_time_tiles_match_independent_recurrent_equations(tmp_path, parameters):
    check_operation(tmp_path, "conv", parameters)
