"""Independent bounded NumPy forward for the native F16 GGUF DSpark sidecar."""
import numpy as np
from akas.checkpoints.gguf import dequantize
from akas.quantization.weights.ggml.definition import ENCODINGS


def qualify(target, draft):
    config=draft.config
    weights=draft.checkpoint.gguf
    target_weights=target.checkpoint.gguf
    size, width, block=32,config.width,config.block_size
    rng=np.random.default_rng(293)
    features=rng.normal(0,.15,(size,config.feature_width)).astype('float32')

    def matrix(x,name,*,source=weights,half_input=None):
        info=source.tensors[name]
        rows,depth=info.shape
        assert x.shape[1]==depth
        if half_input is None: half_input=len(x)>1
        operands=x.astype('float16').astype('float32') if half_input else x
        out=np.empty((len(x),rows),'float32')
        _,values_per_block,bytes_per_block=ENCODINGS[info.type]
        stride=depth//values_per_block*bytes_per_block
        raw=source.packed(name)
        for start in range(0,rows,512):
            stop=min(start+512,rows)
            weight=dequantize(raw[start*stride:stop*stride],info.type).reshape(stop-start,depth)
            if half_input: weight=weight.astype('float16').astype('float32')
            out[:,start:stop]=operands @ weight.astype('float32').T
        source.discard(name)
        return out

    def norm(x,name,channels=width):
        shape=x.shape
        v=x.reshape(-1,channels).astype('float64')
        gamma=weights.array(name).astype('float64')
        result=(v/np.sqrt(np.mean(v*v,axis=-1,keepdims=True)+config.epsilon)*gamma).astype('float32')
        return result.reshape(shape)

    def rotate(x,positions,heads):
        values=x.reshape(len(x),heads,config.head_dim).astype('float64')
        half=config.head_dim//2
        angle=np.asarray(positions)[:,None,None]*config.theta**(-2*np.arange(half)[None,None,:]/config.head_dim)
        left,right=values[...,:half],values[...,half:]
        return np.concatenate((left*np.cos(angle)-right*np.sin(angle),
                               left*np.sin(angle)+right*np.cos(angle)),axis=-1).astype('float32')

    context=norm(matrix(features,'fc.weight'),'enc.output_norm.weight')
    caches=[]
    for i in range(config.depth):
        p=f'blk.{i}.'
        k=norm(matrix(context,p+'attn_k.weight'),p+'attn_k_norm.weight',config.head_dim)
        k=rotate(k,np.arange(size),config.kv_heads).astype('float16')
        v=matrix(context,p+'attn_v.weight').reshape(size,config.kv_heads,config.head_dim).astype('float16')
        caches.append((k,v))
    slot=draft.admit()
    frame=draft.batches.select(1,size,phase='context')
    frame.write(frame.features['hidden_input'],features.reshape(-1))
    draft.run_batch([slot],[[0]*size],chunk=size,phase='context',read=False)
    context_errors=[]
    for i,(k,v) in enumerate(caches):
        for name,expected in (('kc',k),('vc',v)):
            got=draft.batches.states[f'blk.{i}.{name}'].to_numpy()[slot,:,:size].transpose(1,0,2)
            context_errors.append(float(np.max(np.abs(got.astype('float32')-expected))))
            np.testing.assert_allclose(got,expected,atol=.003,rtol=.005)
    anchor=target.checkpoint.tokenizer.encode('A useful answer')[0]
    noise=[anchor,*([config.mask_token]*(block-1))]
    hidden=np.stack([target_weights.row('token_embd.weight',token) for token in noise])
    for i,(past_k,past_v) in enumerate(caches):
        p=f'blk.{i}.'
        normal=norm(hidden,p+'attn_norm.weight')
        q=norm(matrix(normal,p+'attn_q.weight'),p+'attn_q_norm.weight',config.head_dim)
        q=rotate(q,np.arange(size,size+block),config.heads).astype('float16').astype('float64')
        k=norm(matrix(normal,p+'attn_k.weight'),p+'attn_k_norm.weight',config.head_dim)
        k=rotate(k,np.arange(size,size+block),config.kv_heads).astype('float16')
        v=matrix(normal,p+'attn_v.weight').reshape(block,config.kv_heads,config.head_dim).astype('float16')
        keys=np.concatenate((past_k,k),axis=0).astype('float64')
        vals=np.concatenate((past_v,v),axis=0).astype('float64')
        attended=np.zeros((block,config.heads,config.head_dim),'float64')
        # All queries see all live noise positions as well as true context.
        for head in range(config.heads):
            kh=head//(config.heads//config.kv_heads)
            scores=q[:,head] @ keys[:,kh].T / np.sqrt(config.head_dim)
            probabilities=np.exp(scores-scores.max(axis=1,keepdims=True))
            attended[:,head]=(probabilities.astype('float16').astype('float64') @ vals[:,kh])/probabilities.sum(axis=1,keepdims=True)
        hidden=hidden+matrix(attended.reshape(block,width).astype('float32'),p+'attn_output.weight')
        normal=norm(hidden,p+'ffn_norm.weight')
        gate,up=[matrix(normal,p+name) for name in ('ffn_gate.weight','ffn_up.weight')]
        hidden=hidden+matrix((gate/(1+np.exp(-gate)))*up,p+'ffn_down.weight')
    normalized=norm(hidden,'output_norm.weight')
    base=matrix(normalized,draft.checkpoint.shared['head'],source=target_weights)
    expected_tokens=[]
    expected_conf=[]
    prev=anchor
    for i in range(block):
        latent=weights.row('markov_w1.weight',prev)[None,:]
        bias=matrix(latent,'markov_w2.weight',half_input=False)
        prev=int(np.argmax(base[i]+bias[0]))
        expected_tokens.append(prev)
        joined=np.concatenate((normalized[i:i+1],latent),axis=1)
        score=matrix(joined,'conf_proj.weight',half_input=False)[0,0]+weights.array('conf_proj.bias')[0]
        expected_conf.append(float(1/(1+np.exp(-score))))
    proposal=draft.batches.select(1,block,phase='proposal')
    proposal.prepare([slot],[noise],draft.positions)
    proposal.write(proposal.features['anchor_input'],np.array([anchor],dtype='int32'))
    proposal.forward.run()
    actual_base=proposal.feature('logits').to_numpy().reshape(block,config.vocab)
    actual_tokens=proposal.feature('tokens').to_numpy().reshape(block).astype(int).tolist()
    actual_conf=proposal.feature('confidence').to_numpy().reshape(block)
    error=(actual_base-base).astype('float64')
    relative_rmse=float(np.sqrt(np.mean(error**2)/np.mean(base.astype('float64')**2)))
    maximum_error=float(np.max(np.abs(error)))
    if relative_rmse>.005 or maximum_error>.05:
        raise AssertionError(f'Draft NumPy projection mismatch: relative RMSE {relative_rmse}, max {maximum_error}')
    if actual_tokens!=expected_tokens:
        raise AssertionError(f'Draft Markov tokens differ: actual {actual_tokens}, reference {expected_tokens}')
    np.testing.assert_allclose(actual_conf,expected_conf,rtol=.005,atol=.003)
    if draft.positions[slot]!=size:
        raise AssertionError('Proposal changed logical draft prefix')
    draft.release(slot)
    return dict(correct=True,context_tokens=size,noise_tokens=block,noncausal=True,
        reference='Independent NumPy projection/RMS/NeoX-RoPE/attention/SwiGLU/Markov/confidence; bounded 512-row matrix decoding',
        full_base_logits=base.size,relative_rmse=relative_rmse,max_abs_logit_error=maximum_error,
        context_max_abs_errors=context_errors,markov_tokens=actual_tokens,
        confidence=actual_conf.astype(float).tolist())
