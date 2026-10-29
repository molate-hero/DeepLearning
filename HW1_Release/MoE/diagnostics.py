"""Sparse-dispatch checks and last-layer routing analysis.

Expert IDs and gate weights move together across sequences at each position,
never across time. Interpret the perturbed cross-entropy as a batch-dependent
diagnostic rather than independent language-model performance.
"""
import argparse
import copy
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from model import FFN,permute_routes
from train import Corpus,Transformer,evaluate_full


def check_dispatch():
    """Check sparse forward values, gradients, and token-expert counts."""
    torch.manual_seed(77)
    layer=FFN(4,5,True,experts=3,topk=2).double()
    x=torch.randn(7,4,dtype=torch.float64,requires_grad=True)
    executed=[]
    hooks=[e.register_forward_pre_hook(lambda m,a:executed.append(len(a[0])))
           for e in layer.experts]
    sparse,_=layer(x)
    for hook in hooks:hook.remove()
    assert sum(executed)==14,'Dispatch must execute exactly N*k token-expert pairs.'
    probs,gates,ids=layer.route(x)
    logits=layer.router(x)
    torch.testing.assert_close(probs,logits.softmax(-1),rtol=1e-12,atol=1e-12)
    assert torch.equal(ids.sort(-1).values,logits.topk(2,-1).indices.sort(-1).values)
    expected_gates=logits.gather(1,ids).softmax(-1)
    torch.testing.assert_close(gates,expected_gates,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(gates.sum(-1),torch.ones(7,dtype=x.dtype))
    assert bool(((ids>=0)&(ids<3)).all()) and bool((ids[:,0]!=ids[:,1]).all())
    dense_gates=torch.zeros(7,3,dtype=x.dtype).scatter(1,ids,expected_gates)
    # Use an all-expert calculation as the numerical reference for this check.
    oracle=(torch.stack([e(x) for e in layer.experts],dim=1)*dense_gates[:,:,None]).sum(1)
    torch.testing.assert_close(sparse,oracle,rtol=1e-12,atol=1e-12)
    parameters=[x]+list(layer.parameters())
    g1=torch.autograd.grad(sparse.square().sum(),parameters,retain_graph=True,allow_unused=True)
    g2=torch.autograd.grad(oracle.square().sum(),parameters,allow_unused=True)
    errors=[]
    for parameter,a,b in zip(parameters,g1,g2):
        a=torch.zeros_like(parameter) if a is None else a
        b=torch.zeros_like(parameter) if b is None else b
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
        errors.append(float((a-b).abs().max()))
    # Each position's multiset of paired records and expert counts is preserved.
    torch.manual_seed(91)
    pair_ids=torch.randint(0,3,(4,8,2))
    pair_gates=torch.arange(64,dtype=torch.float64).reshape(4,8,2)/64
    shuffled_gates,shuffled_ids=permute_routes(pair_gates,pair_ids,4,8)
    shuffled_gates=shuffled_gates.reshape(4,8,2);shuffled_ids=shuffled_ids.reshape(4,8,2)
    torch.testing.assert_close(F.one_hot(pair_ids,3).sum((0,2)),
                               F.one_hot(shuffled_ids,3).sum((0,2)),rtol=0,atol=0)
    for t in range(8):
        before=sorted(zip(pair_ids[:,t].tolist(),pair_gates[:,t].tolist()))
        after=sorted(zip(shuffled_ids[:,t].tolist(),shuffled_gates[:,t].tolist()))
        assert before==after,'Expert IDs and gates must move together at the same position.'
    return {'passed':True,'dtype':'float64','example_shapes':{'input':[7,4],'gates':[7,2],'ids':[7,2]},
            'selected_token_expert_pairs':14,'executed_token_expert_pairs':sum(executed),
            'all_expert_pairs_if_dense':21,'max_output_error':float((sparse-oracle).abs().max()),
            'max_gradient_error':max(errors),'per_position_route_records_preserved':True}


@torch.no_grad()
def check_prefix(model):
    """Check that suffix changes leave prefix predictions unchanged.

    Test both normal routing and the position-wise perturbation. This checks
    time causality conditional on the batch prefixes, not independence from
    other sequences in the batch.
    """
    probe=copy.deepcopy(model).cpu().double().eval()
    length=probe.position.num_embeddings;cut=length//2
    torch.manual_seed(73)
    inputs=torch.randint(0,256,(4,length))
    changed=inputs.clone();changed[:,cut:]=(changed[:,cut:]+127)%256
    errors={}
    for perturb in [False,True]:
        last=probe.blocks[-1];original=last.forward
        if perturb:
            last.forward=lambda x,shuffle=False:original(x,shuffle=True)
        try:
            torch.manual_seed(20261004);a,_=probe(inputs)
            torch.manual_seed(20261004);b,_=probe(changed)
            error=float((a[:,:cut]-b[:,:cut]).abs().max())
            assert error<1e-10,'Changing future positions changed prefix predictions.'
            errors['position_wise_perturbation' if perturb else 'normal']=error
        finally:last.forward=original
    return {'passed':True,'dtype':'float64 CPU','prefix_length':cut,
            'max_prefix_logit_errors':errors,'tolerance':1e-10,
            'scope':'invariance to suffix changes in all batch rows, with the same routing RNG'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    p.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--check',action='store_true',help='Check sparse dispatch and prefix invariance on CPU.')
    args=p.parse_args();torch.set_num_threads(4)
    verification=check_dispatch()
    if args.check:
        torch.manual_seed(416)
        probe=Transformer(16,8,12,True,experts=3)
        print(json.dumps({'dispatch':verification,'prefix':check_prefix(probe)},indent=2))
        return
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    torch.backends.cuda.matmul.allow_tf32=True
    log_path=args.results/'language.json'
    log=json.loads(log_path.read_text());metadata=log['metadata']
    corpus=Corpus(args.root/'corpus.npz',metadata['context'],'transformer',metadata['domains'].split(','),device)
    records=[];routing_seeds=[20261004,20261005,20261006]
    for seed in map(int,metadata['seeds'].split(',')):
        checkpoint=torch.load(args.results/f'moe_seed{seed}.pt',map_location='cpu',weights_only=False)
        model=Transformer(metadata['context'],metadata['d'],metadata['width'],True,experts=metadata['experts']).to(device)
        model.load_state_dict(checkpoint['model'])
        prefix=check_prefix(model)
        normal=evaluate_full(model,corpus,'test',metadata['batch'])
        last=model.blocks[-1];original=last.forward
        trials=[]
        last.forward=lambda x,shuffle=False:original(x,shuffle=True)
        try:
            for routing_seed in routing_seeds:
                result=evaluate_full(model,corpus,'test',metadata['batch'],routing_seed=routing_seed)
                assert normal['route_fraction']==result['route_fraction']
                # A cross-sequence intervention has no standalone LM perplexity.
                result.pop('perplexity')
                trials.append({'routing_seed':routing_seed,'diagnostic':result})
                print(json.dumps({'model_seed':seed,'routing_seed':routing_seed,
                                  'diagnostic_ce':result['ce']}),flush=True)
        finally:last.forward=original
        ces=[trial['diagnostic']['ce'] for trial in trials]
        record={'seed':seed,'normal_test':normal,'position_wise_trials':trials,
                'diagnostic_ce_mean':sum(ces)/len(ces),'diagnostic_ce_range':[min(ces),max(ces)],
                'all_layers_identical_expert_token_counts':True,'prefix_verification':prefix,
                'batch_size':metadata['batch'],'routing_seeds':routing_seeds,
                'interpretation':'paired routes exchanged at equal positions across fixed batch rows; '
                    'batch-dependent checkpoint perturbation, not independent LM performance; '
                    'changes expert identity and gate weights jointly; no causal claim about training benefits'}
        records.append(record);print(json.dumps(record),flush=True)
        del model
        if device=='cuda':torch.cuda.empty_cache()
    log['dispatch_verification']=verification
    log['routing_control']=records
    log['routing_control_revision']='position-wise paired-route perturbation'
    log_path.write_text(json.dumps(log,indent=2))


if __name__=='__main__':main()
