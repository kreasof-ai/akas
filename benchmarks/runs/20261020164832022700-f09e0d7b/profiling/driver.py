"""Launch one unmodified artifact through the CUDA driver for counter collection."""
import ctypes as c,json,sys
from pathlib import Path
root=Path(sys.argv[1]);spec=json.loads((root/'target.json').read_text());lib=c.CDLL('libcuda.so.1');u=c.c_uint;v=c.c_void_p
for name,types in dict(cuInit=[u],cuDeviceGet=[c.POINTER(c.c_int),c.c_int],cuDevicePrimaryCtxRetain=[c.POINTER(v),c.c_int],cuCtxSetCurrent=[v],cuMemAlloc_v2=[c.POINTER(c.c_uint64),c.c_size_t],cuMemcpyHtoD_v2=[c.c_uint64,v,c.c_size_t],cuModuleLoadData=[c.POINTER(v),v],cuModuleGetFunction=[c.POINTER(v),v,c.c_char_p],cuFuncSetAttribute=[v,c.c_int,c.c_int],cuLaunchKernel=[v,u,u,u,u,u,u,u,v,c.POINTER(v),v],cuCtxSynchronize=[],cuMemFree_v2=[c.c_uint64],cuModuleUnload=[v],cuDevicePrimaryCtxRelease_v2=[c.c_int]).items():
 fn=getattr(lib,name);fn.argtypes=types;fn.restype=c.c_int

def call(name,*args):
 code=getattr(lib,name)(*args)
 if code:raise RuntimeError(f'{name} failed: {code}')
call('cuInit',0);device=c.c_int();call('cuDeviceGet',c.byref(device),0);ctx=v();call('cuDevicePrimaryCtxRetain',c.byref(ctx),device);call('cuCtxSetCurrent',ctx);module=v();allocated=[]
try:
 payload=c.create_string_buffer((root/'kernel.cubin').read_bytes());call('cuModuleLoadData',c.byref(module),payload);function=v();call('cuModuleGetFunction',c.byref(function),module,spec['entrypoint'].encode());pointers={}
 for name in spec['arguments']:
  data=bytearray((root/f'{name}.bin').read_bytes());host=(c.c_ubyte*len(data)).from_buffer(data);pointer=c.c_uint64();call('cuMemAlloc_v2',c.byref(pointer),len(data));allocated.append(pointer);call('cuMemcpyHtoD_v2',pointer,host,len(data));pointers[name]=pointer;del host,data
 launch=spec['launch'];shared=launch.get('shared_memory_bytes',0)
 if shared:call('cuFuncSetAttribute',function,8,shared)
 arguments=(v*len(spec['abi']))(*(c.cast(c.byref(pointers[a['name']]),v) for a in spec['abi']));call('cuLaunchKernel',function,*launch['grid'],*launch['block'],shared,None,arguments,None);call('cuCtxSynchronize');print('PROFILED_UNMODIFIED_CUBIN',spec['sha256'],flush=True)
finally:
 for pointer in reversed(allocated):call('cuMemFree_v2',pointer)
 if module:call('cuModuleUnload',module)
 call('cuDevicePrimaryCtxRelease_v2',device)
