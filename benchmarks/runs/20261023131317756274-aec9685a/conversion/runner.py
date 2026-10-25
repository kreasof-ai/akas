"""Convert the official F16 sidecar with llama.cpp's bit-exact Q8_0 encoder."""
import hashlib,json,mmap,resource,time
from pathlib import Path
import numpy as np
import gguf
from akas.benchmarking.history import Recorder,write,digest
root=Path('/home/sagemaker-user/akas');source=root/'build/checkpoints/lfm2.5-24b-a2b/dspark-f16.gguf';output=source.with_name('dspark-q8_0.gguf')
meta=dict(runtime='akas',purpose='conversion',draft_source_sha256=digest(source),quantizer_revision='620d2225cc436ab0b001e3200e590e3b286ef9ab',quantization='Q8_0',qualification='Use the pinned llama.cpp gguf-py Q8_0 encoder, documented bit-exact with ggml-quants.c. Quantize native floating matrices only; retain original F32 vectors, metadata and shapes. Sequential temporary-file output and mapped-page discard bound host memory. This conversion alone establishes no speed or acceptance claim.')
rec=Recorder(root/'benchmarks',meta,kind='configuration',parent='20261023124514712767-7ea28db7',change='Prepare a local Q8_0 draft from the original F16 sidecar for a short-context acceptance and speed comparison')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False;print('RUN',rec.id,flush=True)
try:
 reader=gguf.GGUFReader(source);writer=gguf.GGUFWriter(output,'dflash',use_temp_file=True)
 for name,field in reader.fields.items():
  if name.startswith('GGUF.') or name in ('general.file_type','general.quantization_version'):continue
  writer.add_key_value(name,field.contents(),field.types[0],field.types[-1] if field.types[0]==gguf.GGUFValueType.ARRAY else None)
 writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0);writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
 counts={}
 for tensor in reader.tensors:
  value=tensor.data
  if len(tensor.shape)>1:
   if tensor.tensor_type not in (gguf.GGMLQuantizationType.F16,gguf.GGMLQuantizationType.F32):raise ValueError('Refuse requantization')
   encoded=gguf.quantize(value.astype('float32'),gguf.GGMLQuantizationType.Q8_0)
   writer.add_tensor(tensor.name,encoded,raw_dtype=gguf.GGMLQuantizationType.Q8_0);del encoded
  else:writer.add_tensor(tensor.name,value)
  counts[tensor.name]=dict(shape=list(map(int,tensor.shape)),source_bytes=int(value.nbytes))
  reader.data._mmap.madvise(mmap.MADV_DONTNEED)
 writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
 result=gguf.GGUFReader(output)
 if len(result.tensors)!=len(reader.tensors):raise ValueError('Changed tensor inventory')
 for old,new in zip(reader.tensors,result.tensors):
  if old.name!=new.name or not np.array_equal(old.shape,new.shape):raise ValueError('Changed tensor geometry')
  if len(old.shape)==1 and (old.tensor_type!=new.tensor_type or not np.array_equal(old.data,new.data)):raise ValueError('Changed norm/bias')
 meta.update(output_sha256=digest(output),output_bytes=output.stat().st_size,source_bytes=source.stat().st_size,tensor_count=len(result.tensors),matrices=sum(len(t.shape)>1 for t in result.tensors));passed=True
except BaseException as error:
 import traceback
 meta['error']=f'{type(error).__name__}: {error}';meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;rec.artifact(__file__,'conversion/runner.py');rec.artifact(root/'build/baselines/llama.cpp/gguf-py/gguf/quants.py','conversion/quantizer.py');rec.update(report,final=True,status='passed' if passed else 'failed');write(source.with_name('dspark-q8_0-conversion.json'),dict(run_id=rec.id,**meta));print('RETAINED',rec.id,passed,json.dumps(meta),flush=True)
if not passed:raise SystemExit(1)
