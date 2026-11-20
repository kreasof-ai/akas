# Mamba operations

The shared SiLU and columnwise-softplus activation prepare the convolution and
step projection.
`selective_scan` keeps all sixteen recurrent states for one channel in local
registers, loops over sequence frames in F32, and applies the skip and SiLU
output gate. `mamba_reverse` and `mamba_join` express bidirectional branches
without an in-place transpose or hidden host synchronization.

The WebGPU scan resets at every independent sequence boundary and maps several
sequences to one workgroup and 128 channels per tile. `rows_per_block` is a
schedule-only occupancy choice and never changes recurrence or state layout.
All Mamba projections use the shared linear family; the short causal FIR uses
the shared convolution family with the same sequence geometry.

CUDA has no registered implementation yet. NumPy validation runs the complete
frame recurrence independently and checks final outputs, not a simplified
linear-time invariant.
