"""Request-row inputs for any compiled token phase."""

import numpy as np
from akas.memory.transfer import write, write_many


class Batch:
    def __init__(self, forward, *, rows, chunk, inputs, prediction, owner=None, phase=None, features=None):
        if type(rows) is not int or rows < 1 or type(chunk) is not int or chunk < 1:
            raise ValueError("batch geometry must contain positive integers")
        self.forward, self.rows, self.chunk = forward, rows, chunk
        self.inputs, self.prediction = dict(inputs), prediction
        self.owner, self.phase = owner, phase
        self.features = dict(features or {})
        self.commit_forward = None
        if set(self.inputs) - {"tokens","mapping","positions","lengths","head_positions","head_active"}:
            raise ValueError("unknown request input field")
        if "tokens" not in self.inputs or "lengths" not in self.inputs:
            raise ValueError("token phases require token and length bindings")

    def prepare(self, slots, sequences, positions):
        if not slots or len(slots) != len(sequences) or len(slots) > self.rows or len(set(slots)) != len(slots):
            raise ValueError("invalid request-row batch")
        if any(not 1 <= len(sequence) <= self.chunk for sequence in sequences):
            raise ValueError("invalid request chunk length")
        tokens = np.zeros((self.rows,self.chunk),dtype="int32")
        mapping = np.full(self.rows,-1,dtype="int32")
        starts = np.zeros(self.rows,dtype="int32")
        lengths = np.zeros(self.rows,dtype="int32")
        for row,(slot,sequence) in enumerate(zip(slots,sequences)):
            if type(slot) is not int or not 0 <= slot < len(positions):
                raise ValueError("request mapping is outside state pool")
            mapping[row], starts[row], lengths[row] = slot, positions[slot], len(sequence)
            tokens[row,:len(sequence)] = sequence
        fields = dict(tokens=tokens.reshape(-1),mapping=mapping,positions=starts,
                      lengths=lengths,head_positions=starts+np.maximum(lengths-1,0),
                      head_active=(lengths!=0).astype("int32"))
        updates = []
        for field,name in self.inputs.items():
            buffer = self.forward.buffers[name]
            values = fields[field]
            if self.chunk > 1 and buffer.shape == (self.rows*self.chunk,):
                if field == "head_positions": values = (starts[:,None]+np.arange(self.chunk)[None,:]).reshape(-1)
                elif field == "head_active": values = (np.arange(self.chunk)[None,:]<lengths[:,None]).astype("int32").reshape(-1)
            updates.append((buffer, values))
        write_many(self.forward.device, updates)
        self.slots, self.lengths = tuple(slots), lengths[:len(slots)].copy()

    @property
    def buffers(self):
        return self.forward.buffers

    @property
    def kernels(self):
        return self.forward.kernels

    @property
    def calls(self):
        return [(entry[0], entry[1]) for entry in self.forward.plan]

    @property
    def commit_calls(self):
        if self.commit_forward is None:
            return []
        return [(entry[0], entry[1]) for entry in self.commit_forward.plan]

    @property
    def graph(self):
        return self.forward.graph

    def write(self, name, values):
        write(self.forward.device, self.buffers[name], values)

    def feature(self, name):
        return self.buffers[self.features[name]]

    def commit(self, accepted):
        if self.owner is None:
            raise ValueError("batch has no request owner")
        self.owner.commit(self.slots, accepted)

    def discard(self):
        if self.owner is None:
            raise ValueError("batch has no request owner")
        self.owner.discard(self.slots)

    def run(self, slots, sequences, positions=None):
        if positions is None:
            if self.owner is None:
                raise ValueError("batch has no request owner")
            if self.phase == "verify":
                return self.owner.verify(slots, sequences, chunk=self.chunk)
            return self.owner.run_batch(slots, sequences, chunk=self.chunk)
        self.prepare(slots,sequences,positions)
        self.forward.run()
        result = self.forward.buffers[self.prediction].to_numpy()
        return [int(value) for value in result[:len(slots)]]


class BatchBank:
    """Select compiled phase geometries while sharing weights and request state."""

    def __init__(self, directory, device, records, weights, *, graphs=True, reset=None):
        self.directory, self.device = directory, device
        self.owner = None
        self.records, self.weights = tuple(records), dict(weights)
        self.graphs, self.frames, self.states = graphs, {}, {}
        self.reset_record, self.reset_forward = reset, None
        self.owned_controls = []
        self.owned_states = []
        if not self.records:
            raise ValueError("program needs at least one phase geometry")
        identities = [(row["phase"],row["rows"],row["chunk"]) for row in self.records]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate compiled phase geometry")

    def select(self, count, chunk, *, phase=None):
        phase = ("decode" if chunk == 1 else "prefill") if phase is None else phase
        choices = [row for row in self.records if row["phase"] == phase
                   and row["rows"] >= count and row["chunk"] == chunk]
        if not choices:
            raise ValueError("program has no qualified phase geometry for this batch")
        record = min(choices,key=lambda row:row["rows"])
        key = (record["phase"],record["rows"],record["chunk"])
        if key not in self.frames:
            self.frames[key] = self._load(record)
        return self.frames[key]

    def select_window(self, count, tokens, *, phase="verify", limit=None, chunks=None):
        """Use the smallest qualified window that fits the live token budget."""
        if type(count) is not int or count < 1 or type(tokens) is not int or tokens < 1:
            raise ValueError("invalid window request")
        choices = [row for row in self.records if row["phase"] == phase
                   and row["rows"] >= count and row["chunk"] >= tokens
                   and (limit is None or row["chunk"] <= limit)
                   and (chunks is None or row["chunk"] in chunks)]
        if not choices:
            raise ValueError("program has no qualified window for this batch")
        record = min(choices, key=lambda row: (row["chunk"], row["rows"]))
        return self.select(count, record["chunk"], phase=phase)

    def _load(self, record):
        import json
        from pathlib import Path
        from akas.graph import Graph
        from akas.execution.forward import Forward

        root = Path(self.directory).resolve()
        path = (root/record["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("phase path escapes program")
        manifest = json.loads(path.read_text())
        graph = Graph.from_metadata(manifest["forward"])
        supplied = {name:value for name,value in self.weights.items() if name in graph.buffers}
        supplied.update((name,value) for name,value in self.states.items() if name in graph.buffers)
        controls, forward = [], None
        try:
            for name,spec in graph.buffers.items():
                if spec.role in ("input","output") and spec.alias_of is None:
                    host = np.zeros(spec.shape,dtype="float32" if spec.dtype=="bfloat16" else spec.dtype)
                    value = (self.device.from_numpy(host,dtype=spec.dtype) if spec.dtype=="bfloat16" else self.device.from_numpy(host))
                    controls.append(value)
                    supplied[name] = value
            forward = Forward(root,self.device,supplied,filename=record["path"],graphs=self.graphs)
            frame = Batch(forward,rows=record["rows"],chunk=record["chunk"],
                          inputs=record["inputs"],prediction=record["prediction"],
                          owner=self.owner, phase=record["phase"], features=record.get("features"))
            frame.feedback = record.get("feedback", False)
            if record.get("commit") is not None:
                import json
                from akas.graph import Graph
                commit_manifest = json.loads((root/record["commit"]["path"]).read_text())
                commit_graph = Graph.from_metadata(commit_manifest["forward"])
                frame.commit_forward = Forward(root,self.device,
                    {name:forward.buffers[name] for name in commit_graph.buffers},
                    filename=record["commit"]["path"],graphs=self.graphs)
            self._adopt_states(forward)
            self.owned_controls.extend(controls)
            return frame
        except BaseException:
            if forward is not None:
                forward.close()
            for value in reversed(controls):
                value.release()
            raise

    def _adopt_states(self, forward):
        states = {name:forward.buffers[name] for name,spec in forward.definition.buffers.items()
                  if spec.role=="state" and spec.alias_of is None}
        owned_ids = {id(value) for value in states.values()}
        self.owned_states.extend(value for value in forward.workspace.owned if id(value) in owned_ids)
        forward.workspace.owned[:] = [value for value in forward.workspace.owned if id(value) not in owned_ids]
        self.states.update(states)

    def reset(self, slot):
        from pathlib import Path
        import json
        from akas.graph import Graph
        from akas.execution.forward import Forward

        if self.reset_record is None:
            raise ValueError("program has no state reset phase")
        if self.reset_forward is None:
            manifest = json.loads((Path(self.directory)/self.reset_record["path"]).read_text())
            graph = Graph.from_metadata(manifest["forward"])
            supplied = {name:value for name,value in self.states.items() if name in graph.buffers}
            mapping = self.device.from_numpy(np.zeros((1,),dtype="int32"))
            supplied[self.reset_record["mapping"]] = mapping
            try:
                self.reset_forward = Forward(self.directory,self.device,supplied,
                                             filename=self.reset_record["path"],graphs=self.graphs)
            except BaseException:
                mapping.release()
                raise
            self.owned_controls.append(mapping)
            self._adopt_states(self.reset_forward)
        write(self.device,self.reset_forward.buffers[self.reset_record["mapping"]],np.array([slot],dtype="int32"))
        self.reset_forward.run()

    def close(self):
        for frame in reversed(tuple(self.frames.values())):
            if frame.commit_forward is not None:
                frame.commit_forward.close()
            frame.forward.close()
        if self.reset_forward is not None:
            self.reset_forward.close()
            self.reset_forward = None
        for value in reversed((*self.owned_controls,*self.owned_states)):
            value.release()
        self.frames.clear()
        self.owned_controls.clear()
        self.owned_states.clear()
        self.states.clear()
