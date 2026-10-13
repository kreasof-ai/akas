"""Benchmark token-step protocol over the common compiled-program executor."""


class Session:
    def __init__(self, executor, *, batch, chunk=32):
        if type(batch) is not int or not 1 <= batch <= executor.capacity:
            raise ValueError("benchmark batch exceeds request capacity")
        if type(chunk) is not int or not 1 <= chunk <= executor.context:
            raise ValueError("invalid benchmark prefill chunk")
        self.executor, self.batch, self.prefill_chunk = executor, batch, chunk
        self.capacity, self.device, self.config = executor.context, executor.device, executor.config
        self.kv_dtype = executor.program.get("model_options", {}).get("kv_dtype")
        self.slots, self.tokens, self.feedback_ready = [], None, False

    @property
    def allocated_bytes(self):
        buffers = [*self.executor.weights.values(), *self.executor.batches.states.values()]
        for frame in self.executor.batches.frames.values():
            buffers.extend(frame.buffers.values())
        storage = {}
        for buffer in buffers:
            while hasattr(buffer, "_base"):
                buffer = buffer._base
            storage[id(buffer)] = buffer
        return sum(buffer.nbytes for buffer in storage.values())

    def reset(self):
        self.close()
        try:
            for _ in range(self.batch):
                self.slots.append(self.executor.admit())
        except BaseException:
            self.close()
            raise
        self.tokens, self.feedback_ready = None, False

    def prefill(self, prompts, *, read=True):
        if len(prompts) != self.batch or len(self.slots) != self.batch:
            raise ValueError("one prompt per admitted benchmark request is required")
        self.tokens = self.executor.prefill(self.slots, [[int(token) for token in prompt] for prompt in prompts],
                                            chunk=self.prefill_chunk)
        self.feedback_ready = False
        return self.tokens if read else None

    def decode(self, steps, *, read=True):
        if self.tokens is None:
            raise ValueError("benchmark decode requires prefill")
        result = self.executor.decode_steps(self.slots, None if self.feedback_ready else self.tokens,
                                            steps, read=read)
        self.feedback_ready = True
        return result

    def close(self):
        while self.slots:
            self.executor.release(self.slots[-1])
            self.slots.pop()

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()
