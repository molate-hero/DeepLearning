"""Causal language modeling: corpus sampling, training, and evaluation."""
import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import FFN

def synchronize(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize()

def matrix_macs(d,context,width,moe,experts=16,layers=2,k=2):
    attention=4*d*d+2*context*d
    ffn=experts*d+2*k*d*width if moe else 2*d*width
    return layers*(attention+ffn)+d*256

class Block(nn.Module):
    def __init__(self,d,width,moe,experts):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d,3*d)
        self.proj = nn.Linear(d,d)
        self.ffn = FFN(d,width,moe,experts)
        self.heads = 4
    def forward(self,x,shuffle=False):
        b,t,d = x.shape
        q,k,v = self.qkv(self.norm1(x)).chunk(3,-1)
        def reshape(a): return a.view(b,t,self.heads,d//self.heads).transpose(1,2)
        att = F.scaled_dot_product_attention(reshape(q),reshape(k),reshape(v),is_causal=True)
        x = x+self.proj(att.transpose(1,2).contiguous().view(b,t,d))
        f,aux = self.ffn(self.norm2(x),shuffle)
        return x+f,aux


class Transformer(nn.Module):
    def __init__(self,context,d,width,moe,layers=2,experts=8):
        super().__init__()
        self.embed, self.position = nn.Embedding(256,d), nn.Embedding(context,d)
        self.blocks = nn.ModuleList(Block(d,width,moe,experts) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d,256,bias=False)
    def forward(self,ids,shuffle=False):
        x = self.embed(ids)+self.position(torch.arange(ids.shape[1],device=ids.device))
        aux = x.new_zeros(())
        for block in self.blocks:
            x, a = block(x,shuffle); aux = aux+a
        return self.head(self.norm(x)), aux/len(self.blocks)


class Corpus:
    def __init__(self,path,context,architecture,domains,device):
        npz = np.load(path)
        self.parts={}
        self.segment_starts={}
        self.context, self.architecture = context,architecture
        for split in ['train','validation','test']:
            self.parts[split]=[]
            self.segment_starts[split]=[]
            for domain in domains:
                a=npz[f'{domain}_{split}']
                bounds=npz[f'{domain}_{split}_bounds']
                starts=np.concatenate([np.arange(lo,hi-context,dtype=np.int64) for lo,hi in zip(bounds[:-1],bounds[1:]) if hi-lo>context])
                self.parts[split].append((torch.as_tensor(a.astype(np.int64),device=device),torch.as_tensor(starts,device=device)))
                segment_starts=np.concatenate([np.arange(lo,hi-context,context,dtype=np.int64) for lo,hi in zip(bounds[:-1],bounds[1:]) if hi-lo>context])
                self.segment_starts[split].append(torch.as_tensor(segment_starts,device=device))
        self.offsets=torch.arange(context+1,device=device)
        self.device=device
    def batch(self,split,batch_size,generator,domain=None):
        domains = list(range(len(self.parts[split]))) if domain is None else [domain]
        xs,ys = [],[]
        for i in domains:
            a,starts=self.parts[split][i]
            size=batch_size//len(domains)
            positions=starts[torch.randint(len(starts),(size,),generator=generator,device=self.device)]
            windows=a[positions[:,None]+self.offsets[None,:]]
            xs.append(windows[:,:-1]); ys.append(windows[:,1:] if self.architecture=='transformer' else windows[:,-1])
        return torch.cat(xs),torch.cat(ys)


@torch.no_grad()
def evaluate(model,corpus,split,batch,eval_batches,seed=99,shuffle=False):
    model.eval()
    generator=torch.Generator(device=corpus.device).manual_seed(seed)
    torch.manual_seed(seed+170)
    losses, accuracies,counts=[],[],[]
    for domain in range(len(corpus.parts[split])):
        total_loss=total_correct=total_tokens=0
        counter=None
        for _ in range(eval_batches):
            x,y=corpus.batch(split,batch,generator,domain)
            pred,_=model(x,shuffle)
            total_loss+=F.cross_entropy(pred.reshape(-1,256),y.reshape(-1),reduction='sum').item()
            total_correct+=(pred.argmax(-1)==y).sum().item();total_tokens+=y.numel()
            modules=[m for m in model.modules() if isinstance(m,FFN) and m.moe]
            if modules:
                current=torch.stack([m.last_counts for m in modules]).cpu().numpy()
                counter=current if counter is None else counter+current
        losses.append(total_loss/total_tokens);accuracies.append(total_correct/total_tokens)
        counts.append(None if counter is None else (counter/counter.sum(axis=-1,keepdims=True)).tolist())
    return {'ce':float(np.mean(losses)),'perplexity':float(np.exp(np.mean(losses))),
            'accuracy':float(np.mean(accuracies)), 'domain_ce':losses,'domain_accuracy':accuracies,
            'route_fraction':counts,'evaluated_targets_per_domain':total_tokens}


@torch.no_grad()
def evaluate_full(model,corpus,split,batch,shuffle=False,routing_seed=20261004):
    """Evaluate next-byte prediction and expert usage over the selected split.

    Transformer inputs are nonoverlapping full-length sequences within
    document boundaries. Metrics give equal weight to the two domains.
    """
    model.eval();torch.manual_seed(routing_seed)
    losses,accuracies,counts,target_counts=[],[],[],[]
    for domain,(a,starts) in enumerate(corpus.parts[split]):
        if corpus.architecture=='transformer':starts=corpus.segment_starts[split][domain]
        total_loss=total_correct=total_tokens=0;counter=None
        for off in range(0,len(starts),batch):
            positions=starts[off:off+batch]
            windows=a[positions[:,None]+corpus.offsets[None,:]]
            x=windows[:,:-1]
            y=windows[:,1:] if corpus.architecture=='transformer' else windows[:,-1]
            pred,_=model(x,shuffle)
            total_loss+=F.cross_entropy(pred.reshape(-1,256),y.reshape(-1),reduction='sum').item()
            total_correct+=(pred.argmax(-1)==y).sum().item();total_tokens+=y.numel()
            modules=[m for m in model.modules() if isinstance(m,FFN) and m.moe]
            if modules:
                current=torch.stack([m.last_counts for m in modules]).cpu().numpy()
                counter=current if counter is None else counter+current
        losses.append(total_loss/total_tokens);accuracies.append(total_correct/total_tokens)
        counts.append(None if counter is None else (counter/counter.sum(axis=-1,keepdims=True)).tolist())
        target_counts.append(total_tokens)
    return {'ce':float(np.mean(losses)),'perplexity':float(np.exp(np.mean(losses))),
            'accuracy':float(np.mean(accuracies)),'domain_ce':losses,'domain_accuracy':accuracies,
            'route_fraction':counts,'evaluated_targets_per_domain':target_counts,
            'evaluation':('disjoint full-length packs; document boundaries respected; equal domain weighting' if corpus.architecture=='transformer' else 'all valid next-byte targets; equal domain weighting')}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    p.add_argument('--architecture',choices=['transformer'],default='transformer')
    p.add_argument('--mode',choices=['all','dense_compute','moe','dense_parameter'],default='all')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--seeds',default='416')
    p.add_argument('--domains',default='shakespeare,python')
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--context',type=int,default=64)
    p.add_argument('--d',type=int,default=64)
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--experts',type=int,default=16)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--balance',type=float,default=.01)
    p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--eval-batches',type=int,default=20)
    p.add_argument('--test',action='store_true',default=True)
    p.add_argument('--full-test',action='store_true',default=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    args.models='dense_compute,moe,dense_parameter' if args.mode=='all' else args.mode
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    domains=args.domains.split(',')
    corpus=Corpus(args.root/'corpus.npz',args.context,args.architecture,domains,device)
    all_results=[]
    metadata={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    metadata.update({'torch':torch.__version__,'python':platform.python_version(),'device':torch.cuda.get_device_name() if device=='cuda' else platform.processor() or 'CPU',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'corpus_sha256':hashlib.sha256((args.root/'corpus.npz').read_bytes()).hexdigest(),
        'dense_widths':{'compute':2*args.width+args.experts//2,'parameter':args.experts*args.width+args.experts//2},
        'topk':2,'test_was_used_for_selection':False})
    log_path=args.out/'language.json'
    previous=json.loads(log_path.read_text()) if log_path.exists() and args.mode!='all' else {}
    all_results=previous.get('runs',[])
    for seed in map(int,args.seeds.split(',')):
        # Prepare non-FFN parameters for the dense and MoE comparison models.
        torch.manual_seed(seed)
        reference=Transformer(args.context,args.d,2*args.width+args.experts//2,False)
        common={k:v.clone() for k,v in reference.state_dict().items() if '.ffn.' not in k and not k.startswith('ffn.')}
        del reference
        for mode in args.models.split(','):
            torch.manual_seed(seed)
            width=args.width if mode=='moe' else (2*args.width+args.experts//2 if mode=='dense_compute' else args.experts*args.width+args.experts//2)
            cls=Transformer
            model=cls(args.context,args.d,width,mode=='moe',experts=args.experts).to(device)
            state=model.state_dict();state.update(common);model.load_state_dict(state)
            optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
            g=torch.Generator(device=device).manual_seed(seed+5000)
            best_ce=math.inf;best_state=None;history=[]
            if device=='cuda':torch.cuda.reset_peak_memory_stats()
            synchronize(device);start=time.perf_counter()
            for step in range(1,args.steps+1):
                model.train()
                lr=args.lr*(.15+.85*.5*(1+math.cos(math.pi*step/args.steps)))
                for group in optimizer.param_groups:group['lr']=lr
                x,y=corpus.batch('train',args.batch,g)
                pred,aux=model(x)
                ce=F.cross_entropy(pred.reshape(-1,256),y.reshape(-1))
                loss=ce+args.balance*aux
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                if step%args.eval_every==0 or step==args.steps:
                    ev=evaluate(model,corpus,'validation',args.batch,args.eval_batches)
                    history.append({'step':step,'train_ce':ce.item(),'aux':aux.item(),'validation':ev})
                    if ev['ce']<best_ce:
                        best_ce=ev['ce'];best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    print(json.dumps({'architecture':args.architecture,'seed':seed,'mode':mode,'step':step,'train_ce':ce.item(),'val_ce':ev['ce'],'val_ppl':ev['perplexity']}),flush=True)
            synchronize(device);elapsed=time.perf_counter()-start
            peak=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None
            model.load_state_dict(best_state)
            result={'seed':seed,'mode':mode,'matrix_macs_per_token':matrix_macs(args.d,args.context,width,mode=='moe',args.experts),'parameters':sum(p.numel() for p in model.parameters()),'history':history,
                    'best_validation_ce':best_ce,'train_validation_seconds':elapsed,'peak_allocated_mib':peak}
            if args.test:
                if args.full_test:
                    result['test']=evaluate_full(model,corpus,'test',args.batch)
                else:
                    result['test']=evaluate(model,corpus,'test',args.batch,args.eval_batches*2,seed=190)
            # Save the MoE checkpoint for routing analysis and open-ended investigations.
            if mode=='moe':
                torch.save({'model':best_state,'args':metadata},args.out/f'moe_seed{seed}.pt')
            # Measure full-batch forward latency, including expert dispatch.
            xx,_=corpus.batch('validation',args.batch,torch.Generator(device=device).manual_seed(150))
            with torch.no_grad():
                for _ in range(10):model(xx)
                ts=[]
                for _ in range(30):
                    synchronize(device);t=time.perf_counter();model(xx);synchronize(device);ts.append((time.perf_counter()-t)*1000)
            result['forward_batch_median_ms']=float(np.median(ts))
            all_results=[r for r in all_results if (r['seed'],r['mode'])!=(seed,mode)]
            all_results.append(result)
            log_path.write_text(json.dumps({'metadata':metadata,'runs':all_results},indent=2))
            print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
            del model,optimizer
            if device=='cuda':torch.cuda.empty_cache()


if __name__=='__main__':main()
