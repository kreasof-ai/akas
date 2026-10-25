import ctypes as c,json
vk=c.WinDLL('vulkan-1.dll')
class App(c.Structure):_fields_=[('sType',c.c_uint32),('pNext',c.c_void_p),('name',c.c_char_p),('version',c.c_uint32),('engine',c.c_char_p),('engineVersion',c.c_uint32),('apiVersion',c.c_uint32)]
class Info(c.Structure):_fields_=[('sType',c.c_uint32),('pNext',c.c_void_p),('flags',c.c_uint32),('app',c.POINTER(App)),('layerCount',c.c_uint32),('layers',c.c_void_p),('extCount',c.c_uint32),('exts',c.c_void_p)]
class Queue(c.Structure):_fields_=[('flags',c.c_uint32),('count',c.c_uint32),('timestamp_bits',c.c_uint32),('granularity',c.c_uint32*3)]
app=App(0,None,b'AKAS queue inventory',0,b'',0,1<<22);info=Info(1,None,0,c.pointer(app),0,None,0,None);inst=c.c_void_p()
vk.vkCreateInstance.argtypes=[c.POINTER(Info),c.c_void_p,c.POINTER(c.c_void_p)];assert vk.vkCreateInstance(c.byref(info),None,c.byref(inst))==0
vk.vkEnumeratePhysicalDevices.argtypes=[c.c_void_p,c.POINTER(c.c_uint32),c.c_void_p]
vk.vkGetPhysicalDeviceProperties.argtypes=[c.c_void_p,c.c_void_p]
vk.vkGetPhysicalDeviceQueueFamilyProperties.argtypes=[c.c_void_p,c.POINTER(c.c_uint32),c.c_void_p]
vk.vkDestroyInstance.argtypes=[c.c_void_p,c.c_void_p]
count=c.c_uint32();vk.vkEnumeratePhysicalDevices(inst,c.byref(count),None);devices=(c.c_void_p*count.value)();vk.vkEnumeratePhysicalDevices(inst,c.byref(count),devices);rows=[]
for device in devices:
 props=c.create_string_buffer(4096);vk.vkGetPhysicalDeviceProperties(device,props);name=props.raw[20:276].split(b'\0')[0].decode()
 n=c.c_uint32();vk.vkGetPhysicalDeviceQueueFamilyProperties(device,c.byref(n),None);queues=(Queue*n.value)();vk.vkGetPhysicalDeviceQueueFamilyProperties(device,c.byref(n),queues)
 row=dict(device=name,families=[dict(index=i,count=q.count,flags=q.flags,graphics=bool(q.flags&1),compute=bool(q.flags&2),transfer=bool(q.flags&4)) for i,q in enumerate(queues)]);rows.append(row)
vk.vkDestroyInstance(inst,None)
from pathlib import Path
Path('build/vulkan-next/queue-families.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
