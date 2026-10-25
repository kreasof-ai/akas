import io,tarfile,urllib.request,hashlib,json
from pathlib import Path
root=Path('build/vulkan-next/queue-sources');root.mkdir(parents=True,exist_ok=True)
url='https://crates.io/api/v1/crates/wgpu-hal/27.0.2/download'
data=urllib.request.urlopen(url).read()
sha=hashlib.sha256(data).hexdigest();assert sha=='2618a2d6b8a5964ecc1ac32a5db56cb3b1e518725fcd773fd9a782e023453f2b'
(root/'wgpu-hal-27.0.2.crate').write_bytes(data)
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 source=archive.extractfile('wgpu-hal-27.0.2/src/vulkan/adapter.rs').read().decode()
(root/'adapter.rs').write_text(source)
lines=source.splitlines();matches=[dict(line=i+1,text=line.strip()) for i,line in enumerate(lines) if 'let family_index = 0' in line or 'let queue_family_index = 0' in line]
(root/'source.json').write_text(json.dumps(dict(url=url,archive_sha256=sha,matches=matches),indent=2));print(matches)
