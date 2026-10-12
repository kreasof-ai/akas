"""Reuse scratch allocations whose operation lifetimes do not overlap."""


def scratch_slots(forward):
    """Return buffer-name groups, with identical shapes and disjoint lifetimes.

    Only scratch is eligible. Inputs, weights, outputs and persistent state keep
    their ownership, including when two forward plans share the same state.
    A buffer stays live through its last call, so an input cannot alias that
    call's output. Graph replay sees the same allocation mapping every time.
    """
    uses = {}
    for index, call in enumerate(forward.calls):
        for name in (*call.arguments, *call.bindings.values()):
            uses.setdefault(name, []).append(index)
    groups, ends = [], []
    scratch = sorted(
        (name for name, spec in forward.buffers.items() if spec.role == "scratch"),
        key=lambda name: uses.get(name, [-1])[0],
    )
    for name in scratch:
        spec = forward.buffers[name]
        begin, end = uses.get(name, [-1])[0], uses.get(name, [-1])[-1]
        for slot, names in enumerate(groups):
            previous = forward.buffers[names[0]]
            if ends[slot] < begin and (previous.shape, previous.dtype) == (
                spec.shape,
                spec.dtype,
            ):
                names.append(name)
                ends[slot] = end
                break
        else:
            groups.append([name])
            ends.append(end)
    return groups
