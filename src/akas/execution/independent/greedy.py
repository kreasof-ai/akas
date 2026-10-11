"""Submit one greedy decode step and advance request state."""


def advance_greedy(self):
    if self.provider == "webgpu":
        next_token = self.greedy[1].launch(readback=self.workspaces[1]["tokens"])
    else:
        long = bool(self.long_plan) and self.position + 1 >= self.grouped_threshold
        if self.graphs_enabled:
            (self.long_greedy_graph if long else self.greedy[1]).launch()
        else:
            self._submit(self.long_greedy_plan if long else self.greedy_plans[1])
        next_token = self.workspaces[1]["tokens"].to_numpy()
    self.position += 1
    return next_token
