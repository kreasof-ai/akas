# Native baseline provenance

Each linked run retains its exact source and dependency snapshots under `dependencies/<dependency>/source.tar.gz`, plus per-request audio hashes and server logs. Use those artifacts, not the mutable checkout, when inspecting the baseline source.
