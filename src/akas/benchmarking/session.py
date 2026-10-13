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
        self.last_chunk = None

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
        self.last_chunk = None

    def prefill(self, prompts, *, read=True):
        if len(prompts) != self.batch or len(self.slots) != self.batch:
            raise ValueError("one prompt per admitted benchmark request is required")
        self.tokens = self.executor.prefill(self.slots, [[int(token) for token in prompt] for prompt in prompts],
                                            chunk=self.prefill_chunk)
        self.feedback_ready = False
        self.last_chunk = getattr(self.executor,"last_prefill_chunk",None) or self.prefill_chunk
        return self.tokens if read else None

    def decode(self, steps, *, read=True):
        if self.tokens is None:
            raise ValueError("benchmark decode requires prefill")
        result = self.executor.decode_steps(self.slots, None if self.feedback_ready else self.tokens,
                                            steps, read=read)
        self.feedback_ready = True
        self.last_chunk = 1
        return result

    def step(self, tokens, *, read=True):
        """Execute a supplied token per request through the common decode phase."""
        if len(self.slots) != self.batch or len(tokens) != self.batch:
            raise ValueError("one token per admitted benchmark request is required")
        supplied = [int(token) for token in tokens]
        result = self.executor.decode_steps(self.slots, supplied, 1, read=read)
        self.tokens = result if read else supplied
        self.feedback_ready, self.last_chunk = True, 1
        return result

    def read_feature(self, name):
        """Read a model-declared diagnostic feature from the last token phase."""
        if self.last_chunk is None or len(self.slots) != self.batch:
            raise ValueError("feature read requires a completed token phase")
        frame = getattr(self.executor,"last_frame",None)
        if frame is None:
            frame = self.executor.batches.select(self.batch,self.last_chunk)
        live = tuple(getattr(frame,"slots",self.slots))
        self.last_feature_slots = live
        if name not in frame.features:
            raise ValueError(f"phase does not declare feature {name!r}")
        values = frame.feature(name).to_numpy()
        if values.size % frame.rows:
            raise ValueError("feature does not have a request-row layout")
        return values.reshape(frame.rows,-1)[:len(live)].copy()

    def close(self):
        while self.slots:
            self.executor.release(self.slots[-1])
            self.slots.pop()
        self.tokens, self.feedback_ready, self.last_chunk = None, False, None

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()
