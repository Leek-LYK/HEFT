"""Explicit inference conversion and audit helpers for incremental flow refinement."""
import copy
import hashlib
import numpy as np
import torch


def load_checkpoint_safe(path, *, mmap=True):
    """Restricted torch loader: allow only the fixed legacy NumPy RNG representation.

    No arbitrary globals from the checkpoint are imported or allowlisted. Unknown
    types fail closed; there is deliberately no weights_only=False fallback.
    """
    reconstruct=np.core.multiarray._reconstruct
    allowed=[(reconstruct,'numpy.core.multiarray._reconstruct'),
             (reconstruct,'numpy._core.multiarray._reconstruct'),
             np.ndarray,np.dtype,type(np.dtype('uint32'))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path,map_location='cpu',weights_only=True,mmap=mmap)


def tensor_digest(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def eight_step_origin(state,source_path,source_sha256):
    """Only change solver metadata; never claim this inference snapshot resumed training."""
    if state.get('stage')!='E' or state.get('update')!=2300:
        raise ValueError('Origin must be the registered E2300 endpoint')
    decoder=state['constructor'].get('action_decoder',{})
    if decoder.get('kind')!='flow' or decoder.get('steps')!=4:
        raise ValueError('Expected the original four-step flow decoder')
    if state['config'].get('action_decoder')!=decoder or not state['constructor'].get('student_fsq',{}).get('enabled'):
        raise ValueError('Origin configuration or FSQ mismatch')
    result={k:v for k,v in state.items() if k not in ('optimizer','replay','decoder_rng')}
    result['constructor']=copy.deepcopy(state['constructor']);result['constructor']['action_decoder']['steps']=8
    result['config']=copy.deepcopy(state['config']);result['config']['action_decoder']['steps']=8
    result.update(optimizer=None,replay=None,decoder_rng={},restart_semantics='inference-only solver conversion; new optimizer/replay required for incremental learning')
    result['solver_conversion']=dict(source_checkpoint=str(source_path),source_sha256=source_sha256,
        from_steps=4,to_steps=8,model_tensors_sha256=tensor_digest(state['model']),weights_changed=False)
    return result
