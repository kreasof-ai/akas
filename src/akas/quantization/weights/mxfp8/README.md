# MXFP8 expert weights

Each coefficient is E4M3. The checkpoint stores unsigned E8M0 exponents
expanded across the 32 rows of each 32-by-32 weight block. The CUDA tile reader
converts four E4M3 coefficients per aligned word and uses one exponent to scale
them. The shared reader handles unaligned tiles.

The CUDA warp projection uses two aligned words per lane and shares a scale
across those eight coefficients. It rounds coefficients to BF16 before
accumulating their products in FP32.

The lane MMA candidate converts two E4M3 values at a time using Ada's packed
conversion instruction. It applies the block scale and rounds both values
into the BF16 pair consumed by the MMA instruction.
