"""Run Tensor's NVRTC bootstrap from the pinned compiler dependency revision."""

import runpy
from pathlib import Path
import hashlib
import urllib.request

REVISION = "d2bba09a0aca0ab6252ece4d983e57172e59c874"
URL = f"https://raw.githubusercontent.com/kreasof-ai/tensor/{REVISION}/tools/bootstrap/bootstrap_nvrtc.py"


def main():
    path = Path("build/tools/tensor-bootstrap-nvrtc.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as response:
        source = response.read()
    if (
        hashlib.sha256(source).hexdigest()
        != "7b71438f12b7b0c4b79e63606760d926c7852325428f162666cb55e20c207684"
    ):
        raise RuntimeError("Tensor NVRTC bootstrap checksum mismatch")
    path.write_bytes(source)
    runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    main()
