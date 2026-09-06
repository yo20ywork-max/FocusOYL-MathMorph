"""Protect a declared vocabulary contrast subspace, not task answers or whole behavior."""
from pathlib import Path
import numpy as np

ANCHORS=(['</think>','<|im_end|>','<|endoftext|>','FINAL','Final','A','B','C','D','E']
    +[str(i) for i in range(21)]
    +['print','sum','len','range','set','append','return','def','for','if','else','True','False','None',
      'But','So','Therefore','First','Finally','Wait','answer','write','output','I','We',
      '+','-','*','/','=','(',')','[',']','{','}',';',':',',','.'])

def protect_delta(delta,q):
    a=np.asarray(delta,dtype=np.float32);q=np.asarray(q,dtype=np.float32)
    if a.ndim!=2 or q.ndim!=2 or a.shape[0]!=q.shape[0]:raise ValueError('Protection shape mismatch')
    if not np.isfinite(a).all() or not np.isfinite(q).all():raise ValueError('Finite matrices required')
    return a-q@(q.T@a)

def contrast_basis(g,max_anchors=128):
    vocab=g.metadata.get('tokenizer.ggml.tokens')
    if not isinstance(vocab,list):
        import gguf
        reader=gguf.GGUFReader(str(g.path))
        field=reader.fields.get('tokenizer.ggml.tokens')
        if field is None:raise ValueError('Readable tokenizer vocabulary required')
        vocab=list(field.contents());del reader
    lookup={t:i for i,t in enumerate(vocab)}
    eos=g.metadata.get('tokenizer.ggml.eos_token_id')
    if not isinstance(eos,int) or not 0<=eos<len(vocab):raise ValueError('Valid EOS token required')
    ids=[eos]
    for s in ANCHORS:
        for prefix in ('','Ġ','▁'):
            i=lookup.get(prefix+s)
            if i is not None and i not in ids and len(ids)<max_anchors:ids.append(i)
    name='output.weight' if 'output.weight' in g.tensors else 'token_embd.weight'
    t=g.tensors[name]
    if t.qtype not in (0,1) or len(t.shape)!=2:raise ValueError('Native dense output head required')
    m=np.memmap(g.path,dtype='<f4' if t.qtype==0 else '<f2',mode='r',offset=t.offset,shape=t.shape)
    rows=np.array(m[ids],dtype=np.float64);reference=np.array(m[eos],dtype=np.float64);del m
    r=(rows-reference)*g.matrix('output_norm.weight').reshape(-1)
    u,s,_=np.linalg.svd(r.T,full_matrices=False)
    rank=int(np.sum(s>max(s[0]*1e-7,1e-12))) if len(s) else 0
    q=u[:,:rank].astype(np.float32)
    if rank>=r.shape[1]:raise ValueError('Protection spans whole output; no safe complement')
    return q,{'reference_token_id':eos,'selected_ids':ids,'rank':rank,
      'anchor_source':'fixed lexical design plus source GGUF vocabulary; no examples or answer labels',
      'condition':'ideal arithmetic, identical token prefix, only final FFN down changed, scalar RMSNorm followed by linear head',
      'protected_property':'pairwise ordering of selected-token logits for the same prefix before quantization',
      'does_not_guarantee_identical_generation_or_termination':True}
