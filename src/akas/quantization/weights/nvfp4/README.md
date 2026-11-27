# NVFP4 expert weights

Each byte stores two consecutive E2M1 coefficients, with the first coefficient
in the low nibble. The checkpoint stores one E4M3 scale per row and 16 columns
and the outer scale as each expert's amax divided by a shared denominator.
The CUDA reader decodes eight coefficients per aligned word and uses the
native E4M3 conversion for the block scale. Aligned tiles within one expert
reuse the outer scale. The shared reader handles unaligned tiles and tiles
crossing an expert boundary.

The CUDA warp projection also consumes one aligned word per lane. It rounds
each decoded coefficient to BF16 before adding its product to the FP32 sum,
matching the matrix operand precision.

The lane MMA candidate builds a 2 KiB table for one expert in shared memory.
It tabulates the eight rounded magnitudes for each absolute E4M3 scale. Each
lane loads two entries, restores the signed nibbles and scale sign, and packs
its BF16 operand register. Both scales are applied before BF16 rounding.
