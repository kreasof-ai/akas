"""Install optional GPU-event hooks in baseline worker subprocesses."""

import os

if os.environ.get("AKAS_GPU_EVENT_DIR"):
    from tools.baselines.native_events import install

    install()
