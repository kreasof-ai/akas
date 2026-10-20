from akas.kernels.kernel import Tensor,kernel,output

def factory(p):
 import tilelang.language as T
 rows,top,experts=(p[n] for n in ('r','top','experts'))
 threads=128
 @kernel
 def compute(ids:Tensor((rows,top),'int32'),active:Tensor(rows,'int32'))->tuple[Tensor(experts,'int32'),Tensor((experts,rows),'int32')]:
  counts=output(experts,'int32');routes=output((experts,rows),'int32')
  with T.Kernel(experts,threads=threads) as expert:
   tx=T.get_thread_binding();lane=tx%32;warp=tx//32
   totals=T.alloc_shared((4,),'int32');cursor=T.alloc_shared((1,),'int32')
   for row in T.Parallel(rows):routes[expert,row]=-1
   if tx==0:cursor[0]=0
   T.sync_threads()
   for tile in T.serial(T.ceildiv(rows,threads)):
    row=tile*threads+tx
    count=T.alloc_var('int32');inclusive=T.alloc_var('int32');prefix=T.alloc_var('int32');local=T.alloc_var('int32')
    count=0
    for rank in T.unroll(top):
     count+=T.if_then_else(row<rows and active[T.min(row,rows-1)]!=0 and ids[T.min(row,rows-1),rank]==expert,1,0)
    inclusive=count
    for step in T.unroll(5):
     previous=T.shfl_sync(inclusive,T.max(lane-(1<<step),0))
     inclusive+=T.if_then_else(lane>=(1<<step),previous,0)
    if lane==31:totals[warp]=inclusive
    T.sync_threads()
    prefix=cursor[0]+inclusive-count
    for w in T.unroll(4):prefix+=T.if_then_else(w<warp,totals[w],0)
    local=0
    for rank in T.unroll(top):
     if row<rows and active[T.min(row,rows-1)]!=0 and ids[T.min(row,rows-1),rank]==expert:
      routes[expert,prefix+local]=row*top+rank
      local+=1
    # Every thread has consumed the previous cursor before it is updated.
    T.sync_threads()
    if tx==0:cursor[0]+=totals[0]+totals[1]+totals[2]+totals[3]
    T.sync_threads()
   if tx==0:counts[expert]=cursor[0]
  return counts,routes
 return compute
