"""Capture composed phase plans with artifact-defined device controls."""

import numpy as np

from akas.execution.forward import Forward
from akas.execution.plan import ExecutionPlan
from akas.memory.transfer import write


class CapturedRound:
    def __init__(self, engine, count, *, window=None):
        self.engine, self.device = engine, engine.target.device
        self.window = engine.window if window is None else window
        self.verifier = engine.target.batch(count, self.window, verify=True)
        self.rows = self.verifier.rows
        self.repair = engine.proposer.frame(self.rows, self.window)
        self.draft = engine.proposer.frame(self.rows, 1) if self.window > 2 else None
        self.journal = None
        self.verification_forward = self.verifier.forward
        self.commit_forward = self.verifier.commit_forward
        if engine.deferred:
            from akas.execution.state import Journal
            key = (self.rows,self.window)
            if key not in engine.target.journals:
                record = next(row for row in engine.target.batches.records
                              if (row["phase"],row["rows"],row["chunk"]) == ("verify",self.rows,self.window))
                if "journal" not in record:
                    raise ValueError("program has no compiled deferred-state variant")
                engine.target.journals[key] = Journal(engine.target,self.verifier,record["journal"])
            self.journal = engine.target.journals[key]
            self.verification_forward = self.journal.forwards["verify"]
            self.commit_forward = self.journal.forwards["commit"]
        if any(frame.rows != self.rows for frame in (self.repair,self.draft) if frame is not None):
            raise ValueError("captured phases need matching request-row geometries")
        records = [row for row in engine.target.program.get("rounds", ())
                   if (row["rows"],row["window"]) == (self.rows,self.window)]
        if len(records) != 1:
            raise ValueError("program has no compiled controls for this round geometry")
        self.control = self.device.from_numpy(np.zeros((self.rows,5),dtype="int32"))
        self.result = self.device.from_numpy(np.zeros((self.rows,self.window+2),dtype="int32"))
        self.controls, self.graph = None, None
        try:
            import json
            from akas.graph import Graph
            definition = Graph.from_metadata(json.loads((engine.target.bundle/records[0]["path"]).read_text())["forward"])
            supplied = {"control":self.control,"result":self.result}
            frames = {"verify":self.verifier,"repair":self.repair,"draft":self.draft}
            for name in definition.buffers:
                if name in supplied:
                    continue
                prefix, field = name.split(".",1)
                frame = frames[prefix]
                if field == "hidden":
                    supplied[name] = frame.feature("hidden" if prefix=="verify" else "hidden_input")
                elif field == "mode":
                    supplied[name] = frame.feature("mode")
                elif field == "predicted":
                    supplied[name] = frame.buffers[frame.prediction]
                else:
                    supplied[name] = frame.buffers[frame.inputs[field]]
            self.controls = Forward(engine.target.bundle,self.device,supplied,
                                    filename=records[0]["path"],graphs=False)
            controls = {entry[2]:entry for entry in self.controls.plan}
            calls = [controls["setup"]]
            for depth in range(1,self.window-1):
                calls.extend((controls[f"draft.{depth}"],*self.draft.forward.plan,
                              controls[f"proposal.{depth}"]))
            calls.extend((*self.verification_forward.plan,controls["accept"],
                          *self.commit_forward.plan,controls["repair"],
                          *self.repair.forward.plan,controls["result"]))
            resources = [*self.controls.buffers.values(),*self.controls.kernels.values()]
            for frame in frames.values():
                if frame is not None:
                    resources.extend((*frame.buffers.values(),*frame.kernels.values()))
            resources.extend((*self.verification_forward.buffers.values(),*self.verification_forward.kernels.values(),
                              *self.commit_forward.buffers.values(),*self.commit_forward.kernels.values()))
            self.plan = ExecutionPlan(self.device,calls,labeled=True)
            self.graph = self.plan.capture(resources=tuple(dict.fromkeys(resources)))
        except BaseException:
            self.close()
            raise

    def run(self, requests):
        target, proposer = self.engine.target, self.engine.proposer
        target._check()
        slots = [request.slot for request in requests]
        if not 1 <= len(slots) <= self.rows or len(set(slots)) != len(slots):
            raise ValueError("invalid speculative batch")
        if getattr(self.verifier,"verification",None) is not None:
            raise ValueError("verification journal has pending requests")
        control = np.zeros((self.rows,5),dtype="int32")
        control[:,0] = -1
        control[:,3:] = -1
        for row,request in enumerate(requests):
            target._slot(request.slot)
            if request.slot in target.pending:
                raise ValueError("request has a pending verification")
            remaining = min(self.window,request.limit-len(request.output),
                            target.context-int(target.positions[request.slot]))
            if remaining < 1 or not request.output:
                raise ValueError("request cannot decode")
            control[row] = (request.slot,target.positions[request.slot],remaining,
                            request.output[-1],proposer.cached[request.slot])
        target._materialize(slots, window=None if self.journal is None else self.window)
        write(self.device,self.control,control)
        self.graph.launch()
        result = self.result.to_numpy()[:len(requests)]
        emitted = []
        for request,row,limit in zip(requests,result,control[:,2]):
            count = int(row[0])
            if not 1 <= count <= limit:
                raise RuntimeError("invalid device acceptance count")
            target.positions[request.slot] += count
            proposer.positions[request.slot] = target.positions[request.slot]
            proposer.cached[request.slot] = int(row[-1])
            if self.journal is not None:
                self.journal.dirty[request.slot] = count > 1
            emitted.append(row[1:count+1].tolist())
        stats = self.engine.stats
        stats["rounds"] += 1
        stats["request_rounds"] += len(slots)
        stats["proposed"] += int(np.sum(control[:len(slots),2]-1))
        stats["accepted"] += sum(len(tokens)-1 for tokens in emitted)
        stats["emitted"] += sum(map(len,emitted))
        return emitted

    def close(self):
        if self.graph is not None:
            self.graph.close()
            self.graph = None
        if self.controls is not None:
            self.controls.close()
            self.controls = None
        for name in ("control","result"):
            buffer = getattr(self,name,None)
            if buffer is not None:
                buffer.release()
                setattr(self,name,None)
