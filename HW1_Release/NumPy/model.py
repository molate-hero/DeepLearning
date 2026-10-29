"""Expert-module forward and backward propagation for Part Two.

Complete forward and backward using NumPy without automatic differentiation.

x: (B,D), y: integer labels (B,), k: number of selected experts.
Parameter shapes follow the column-vector convention:
W1 (E,H,D), b1 (E,H), W2 (E,D,H), b2 (E,D),
G (E,D), C (K,D), bc (K,).
Gradient checks use float64 arithmetic.
"""
import numpy as np

def softmax(scores):
    shifted = scores - scores.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)

def selection_mask(scores, k):
    """Select the k highest-scoring experts in each row.

    Return a Boolean mask with the same shape as scores. Treat the selected
    set as fixed when differentiating within a local selection region.
    """
    ids = np.argsort(-scores, axis=-1, kind='stable')[:, :k]
    mask = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(mask, ids, True, axis=-1)
    return mask

def forward(x, y, params, k, mask=None):
    """Return (mean cross-entropy, cache).

    Use mask when it is not None; otherwise compute it with selection_mask.
    Store 'probs' (B,K) in the cache dictionary, together with any intermediate
    values needed for the backward pass.
    """
    scores = x @ params['G'].T
    mask = selection_mask(scores, k) if mask is None else mask
    gates = softmax(np.where(mask, scores, -np.inf))
    v = np.einsum('bd,ehd->beh', x, params['W1']) + params['b1']
    u = np.maximum(v, 0)
    f = np.einsum('beh,edh->bed', u, params['W2']) + params['b2']
    r = x + np.einsum('be,bed->bd', gates, f)
    logits = r @ params['C'].T + params['bc']
    probs = softmax(logits)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    loss = -log_probs[np.arange(len(x)), y].mean()
    return loss, dict(probs=probs, gates=gates, v=v, u=u, f=f, r=r)

def backward(x, y, params, cache):
    """Return (parameter_grads, input_grad).

    parameter_grads has the same keys/shapes as params; input_grad is (B,D).
    Include expert, router and residual contributions to input_grad.
    """
    dz = cache['probs'].copy()
    dz[np.arange(len(x)), y] -= 1
    dz /= len(x)
    dr = dz @ params['C']
    df = cache['gates'][:, :, None] * dr[:, None, :]
    du = np.einsum('bed,edh->beh', df, params['W2'])
    dv = du * (cache['v'] > 0)
    dg = np.einsum('bed,bd->be', cache['f'], dr)
    ds = cache['gates'] * (dg - (cache['gates'] * dg).sum(axis=-1, keepdims=True))
    grads = dict(
        W1=np.einsum('beh,bd->ehd', dv, x), b1=dv.sum(axis=0),
        W2=np.einsum('bed,beh->edh', df, cache['u']), b2=df.sum(axis=0),
        G=ds.T @ x, C=dz.T @ cache['r'], bc=dz.sum(axis=0))
    dx = dr + np.einsum('beh,ehd->bd', dv, params['W1']) + ds @ params['G']
    return grads, dx
