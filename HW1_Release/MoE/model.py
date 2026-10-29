"""Feedforward experts and sparse routing for Part Three.

Complete the TODO methods using PyTorch. Normalization and residual operations
are defined in train.py.
"""
import torch
from torch import nn
from torch.nn import functional as F

def permute_routes(gates,ids,batch,length):
    """Exchange expert IDs and gate weights across sequences at each position.

    This diagnostic preserves expert counts at each position and never moves
    routes across time. Perturbed predictions depend on other sequences in
    the batch and are not independent language-model predictions.
    """
    gates=gates.reshape(batch,length,-1);ids=ids.reshape(batch,length,-1)
    orders=torch.stack([torch.randperm(batch,device=ids.device)
                        for _ in range(length)],dim=1)
    times=torch.arange(length,device=ids.device)[None,:]
    return gates[orders,times].reshape(batch*length,-1),ids[orders,times].reshape(batch*length,-1)

class Expert(nn.Module):
    def __init__(self,d,width):
        super().__init__()
        self.w1=nn.Linear(d,width)
        self.w2=nn.Linear(width,d)
    def forward(self,x):
        """x (...,D) -> (...,D), two FC layers with ReLU between them."""
        return self.w2(F.relu(self.w1(x)))

class DenseFFN(nn.Module):
    """Dense two-layer feedforward network with a ReLU hidden layer."""
    def __init__(self,d,width):
        super().__init__()
        self.w1=nn.Linear(d,width);self.w2=nn.Linear(width,d)
    def forward(self,x):
        return self.w2(F.relu(self.w1(x)))

class FFN(nn.Module):
    def __init__(self,d,width,moe,experts=16,topk=2):
        super().__init__()
        self.moe,self.topk,self.d,self.width=moe,topk,d,width
        if moe:
            self.router=nn.Linear(d,experts,bias=False)
            nn.init.normal_(self.router.weight,std=.02)
            self.experts=nn.ModuleList(Expert(d,width) for _ in range(experts))
        else:self.dense=DenseFFN(d,width)
        self.last_counts=None

    def route(self,flat):
        """TODO: return (all_probs, gates, ids).

        all_probs (N,E): Softmax over all logits for the load-balancing loss.
        gates (N,k): selected Softmax weights, renormalized to sum to one.
        ids (N,k): selected expert indices, descending router score.
        Only gates, rather than all_probs, weight the expert outputs.
        """
        logits = self.router(flat)
        selected, ids = logits.topk(self.topk, dim=-1)
        return logits.softmax(dim=-1), selected.softmax(dim=-1), ids

    def dispatch(self,flat,gates,ids):
        """TODO: return (N,D), executing only selected token/expert pairs.

        For each expert, gather the selected rows, apply that expert, multiply
        by the corresponding gates and accumulate into the output rows.
        Do not evaluate every expert on every token and mask afterward.
        """
        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            rows, slots = torch.where(ids == expert_id)
            if rows.numel() == 0:
                continue
            weighted = expert(flat.index_select(0, rows)) * gates[rows, slots, None]
            out = out.index_add(0, rows, weighted)
        return out

    def forward(self,x,shuffle=False):
        if not self.moe:return self.dense(x),x.new_zeros(())
        shape=x.shape;flat=x.reshape(-1,self.d)
        probs,gates,ids=self.route(flat)
        if shuffle:
            if x.ndim!=3:raise ValueError('Route perturbation requires (B,T,D).')
            gates,ids=permute_routes(gates,ids,shape[0],shape[1])
        counts=torch.bincount(ids.flatten(),minlength=len(self.experts))
        self.last_counts=counts.detach()
        out=self.dispatch(flat,gates,ids)
        fraction=counts.to(probs.dtype)/(len(flat)*self.topk)
        aux=len(self.experts)*(fraction.detach()*probs.mean(0)).sum()
        return out.reshape(shape),aux
