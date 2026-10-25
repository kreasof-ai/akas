import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.campaign_report import REPORT_FILES, check, render, report_directory

history = Path('build/vulkan-retuning-report-audit-5569d5c7').resolve()
history.mkdir(parents=True, exist_ok=True)
for filename in sys.argv[1:]:
    source = Path(filename).resolve()
    data = json.loads(source.read_text(encoding='utf-8'))
    destination = report_directory(data['campaign'], history)
    render(data, destination, history)
    for name in REPORT_FILES:
        assert (source.parent / name).read_bytes() == (destination / name).read_bytes(), name
    print(source.parent.name, 'regenerates byte-for-byte')
check(history)
print('Isolated canonical report check passed')
