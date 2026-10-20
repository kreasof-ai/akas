from pathlib import Path
from akas.compilation.emit import build_artifact
s=Path('build/lfm-moe/fused_expert_activation.py').read_text()
s=s.replace('capacity=(rows*top+3)//4+experts;groups=(rows*top+bm-1)//bm+experts;layout=bm.bit_length()-3','capacity=(rows*top+3)//4+experts;map_block=min(bm,64);pairs=bm//map_block;groups=(rows*top+map_block-1)//map_block+experts;layout=map_block.bit_length()-3')
s=s.replace("x:Tensor((rows,depth),'float32')","x:Tensor((rows*top,depth),'float16')").replace("counts:Tensor((experts,),'int32'),","counts:Tensor((experts,),'int32'),offsets:Tensor((experts+1,),'int32'),")
s=s.replace('Tensor((rows,top,columns),dtype)','Tensor((rows*top,columns),dtype)').replace('output((rows,top,columns),dtype)','output((rows*top,columns),dtype)')
s=s.replace('by=tile_rows[layout,group]','mapped_row=tile_rows[layout,group];by=mapped_row//pairs')
s=s.replace('if expert>=0 and by*bm<counts[expert]:','if expert>=0 and mapped_row%pairs==0 and by*bm<counts[expert]:')
a=s.index('     for i,j in T.Parallel(bm,bk):');b=s.index('     load((w,)',a)
s=s[:a]+'     T.copy(x[offsets[expert]+by*bm,tile*bk],lhs)\n'+s[b:]
s=s.replace('num_stages=2',"num_stages=p['stages']")
s=s.replace('out[route//top,route%top,bx*bn+j]','out[offsets[expert]+index,bx*bn+j]')
s=s.replace("routes:Tensor((experts,rows),'int32'),",'').replace('      route=routes[expert,index]\n','')
Path('build/lfm-moe/packed_expert_activation.py').write_text(s)
root=Path('build/lfm-moe/packed-expert-activation');root.mkdir(exist_ok=True)
for bm,bn,bk,threads,stages in ((64,64,64,256,1),(128,64,64,256,1),(128,64,64,256,2),(64,128,64,256,1)):
 p=dict(r=4096,top=4,experts=64,o=1536,k=2048,block_m=bm,columns=bn,block_k=bk,threads=threads,stages=stages,output_dtype='float16',weight_parts=({'rows':64*1536,'format':'ggml','options':{'encoding':2}},))
 path=root/f'{bm}-{bn}-{bk}-{threads}-{stages}.py';path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom packed_expert_activation import factory\ndef tensor_export():\n return {"kernel":factory('+repr(p)+')}\n');build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)
