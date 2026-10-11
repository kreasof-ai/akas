"""BF16 projection schedule parameters shared by producer and consumer."""


def schedule(rows, k, o, width):
    # Narrow output projections need extra CTAs at low concurrency. Keep the
    # same partitioning at every capacity to preserve request arithmetic.
    m = 64 if rows >= 128 else 32 if rows >= 32 and o > width else 16
    # Measured across five row counts; K partitioning remains unchanged.
    return dict(
        r=rows, k=k, o=o, m=m, n=64, bk=128, parts=8 if o == width and k > width else 1
    )
