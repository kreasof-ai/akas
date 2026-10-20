"""Independent NumPy equation for per-position FP8 scale factoring."""
import numpy as np

def decode(bits):
 bits=bits.astype('uint32');exponent=(bits>>3)&15;mantissa=bits&7
 values=np.where(exponent==0,mantissa*.125*2**-6,(1+mantissa*.125)*np.exp2(exponent.astype('float64')-7))
 values=np.where(bits&128,-values,values)
 return np.where((bits&127)==127,np.nan,values).astype('float32')

def inputs(capacity=131072):
 rng=np.random.default_rng(936)
 q=rng.normal(0,.15,(1,2048)).astype('float32')
 shape=(1,8,capacity,64);metadata=(1,8,capacity,1)
 k=rng.integers(0,256,shape,dtype='uint8');v=rng.integers(0,256,shape,dtype='uint8')
 k[(k&127)==127]=0;v[(v&127)==127]=0
 ks=rng.uniform(.0003,.003,metadata).astype('float32');vs=rng.uniform(.0003,.003,metadata).astype('float32')
 return dict(q=q,kc=k,vc=v,ks=ks,vs=vs,positions=np.array([capacity-18],dtype='int32'),enabled=np.array([1],dtype='int32'),out=np.zeros(32*64*66,dtype='float32'))

def expected(arrays,bk,probability_scale=128):
 length=int(arrays['positions'][0])+1 if arrays['enabled'][0] else 0;splits=64
 out=np.zeros((32,splits,66),dtype='float32');out[:,:,64]=-np.inf
 if not length:return out.ravel()
 span=((length+bk-1)//bk+splits-1)//splits
 for head in range(8):
  key=decode(arrays['kc'][0,head,:length]);value=decode(arrays['vc'][0,head,:length])
  query=arrays['q'].reshape(32,64)[head*4:head*4+4].astype('float16').astype('float32')
  score=(query@key.T*.125)*arrays['ks'][0,head,:length,0][None,:]
  for split in range(splits):
   begin=split*span*bk;end=min(length,begin+span*bk)
   if begin>=end:continue
   maximum=np.full(4,-np.inf,dtype='float32');denominator=np.zeros(4,dtype='float32');result=np.zeros((4,64),dtype='float32')
   for start in range(begin,end,bk):
    stop=min(start+bk,end);tile=score[:,start:stop]
    updated=np.maximum(maximum,tile.max(-1));factor=np.where(np.isfinite(maximum),np.exp(maximum-updated),0).astype('float32')
    probability=np.exp(tile-updated[:,None]).astype('float32');denominator=denominator*factor+probability.sum(-1)
    staged=(probability*arrays['vs'][0,head,start:stop,0][None,:]*probability_scale).astype('float16').astype('float32')
    result=result*factor[:,None]+staged@value[start:stop];maximum=updated
   out[head*4:head*4+4,split,:64]=result/probability_scale
   out[head*4:head*4+4,split,64]=maximum;out[head*4:head*4+4,split,65]=denominator
 return out.ravel()

def merge(values):
 values=values.reshape(32,64,66).astype('float64');maximum=values[:,:,64].max(1)
 factors=np.zeros((32,64),dtype='float64');active=np.isfinite(maximum)
 factors[active]=np.exp(values[active,:,64]-maximum[active,None])
 numerator=(values[:,:,:64]*factors[:,:,None]).sum(1);denominator=(values[:,:,65]*factors).sum(1)
 return numerator/np.maximum(denominator[:,None],1e-30)
