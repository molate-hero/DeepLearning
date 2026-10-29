"""Central-difference gradient checks and SGD training for the expert module."""
import argparse
import json
from pathlib import Path
import numpy as np
import model

ROOT = Path(__file__).resolve().parents[1]

def toy():
    x = np.array([[2., 2.]])
    y = np.array([1])
    w = dict(W1=np.array([[[1,0],[0,-1]],[[0,1],[-1,0]]], dtype=float),
             b1=np.array([[0,.5],[0,1.]]),
             W2=np.array([[[.2,0],[0,.2]],[[0,.2],[.2,0]]]),
             b2=np.array([[0,.5],[.5,0]]), G=np.array([[.25,-.25],[0.,0.]]),
             C=np.array([[.2,0],[0,.2],[.2,.2]]),bc=np.array([-.49,-.49,-.98]))
    return x,y,w,2

def sparse_case():
    rng=np.random.default_rng(1125)
    while True:
        x=rng.normal(size=(2,2));y=np.array([0,2])
        # Store weight matrices according to the column-vector convention.
        w=dict(W1=rng.normal(0,.5,(3,2,2)).transpose(0,2,1),
               b1=rng.normal(0,.3,(3,2)),
               W2=rng.normal(0,.5,(3,2,2)).transpose(0,2,1),
               b2=rng.normal(0,.2,(3,2)),G=rng.normal(0,.5,(2,3)).T,
               C=rng.normal(0,.5,(2,3)).T,bc=np.zeros(3))
        scores=np.sort(x@w['G'].T,axis=-1)
        pre=np.einsum('bd,ehd->beh',x,w['W1'])+w['b1']
        if np.min(np.abs(pre))>.05 and np.min(scores[:,-2]-scores[:,-3])>.05:
            return x,y,w,2

def normalized_error(a,n):
    return float(np.max(np.abs(a-n)/np.maximum(1,np.abs(a)+np.abs(n))))

def check_case(case):
    x,y,w,k=case
    mask=model.selection_mask(x@w['G'].T,k)
    loss,cache=model.forward(x,y,w,k,mask)
    grads,dx=model.backward(x,y,w,cache)
    errors={}
    for name,arr in {**w,'input':x}.items():
        numerical=np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            old=arr[idx]
            try:
                arr[idx]=old+1e-6;plus=model.forward(x,y,w,k,mask)[0]
                arr[idx]=old-1e-6;minus=model.forward(x,y,w,k,mask)[0]
            finally:
                arr[idx]=old
            numerical[idx]=(plus-minus)/2e-6
        errors[name]=normalized_error(dx if name=='input' else grads[name],numerical)
        if name=='input':
            # Compare a deliberately altered input gradient with central differences.
            dz=cache['probs'].copy();dz[np.arange(len(x)),y]-=1;dz/=len(x)
            faulty_dx=dx-dz@w['C']
            faulty_error=normalized_error(faulty_dx,numerical)
    return {'loss':float(loss),'max_normalized_errors':errors,
            'faulty_input_error':faulty_error, 'passed':max(errors.values())<1e-5}

def train(out):
    batch=np.load(ROOT/'data'/'numpy_batch.npz')
    x,y=batch['x'],batch['y']
    w={key:batch['init_'+key].copy() for key in ['W1','b1','W2','b2','G','C','bc']}
    history=[]
    for step in range(401):
        loss,cache=model.forward(x,y,w,2)
        history.append({'step':step,'ce':float(loss)})
        if step<400:
            grads,_=model.backward(x,y,w,cache)
            for key in w:w[key]-=.2*grads[key]
    # Inspect representative contexts from the text and code domains.
    examples=[0,64,1]
    records=[{'context_bytes':batch['ids'][i].tolist(),'target_byte':int(y[i]),
              'predicted_byte':int(cache['probs'][i].argmax())} for i in examples]
    result={'history':history,'predictions':records,'updates':400,'lr':.2}
    import matplotlib.pyplot as plt
    plt.plot([r['step'] for r in history],[r['ce'] for r in history])
    plt.xlabel('SGD update');plt.ylabel('Training cross-entropy');plt.tight_layout()
    plt.savefig(out/'numpy_loss.pdf');plt.close()
    print(json.dumps({'initial_ce':history[0]['ce'],'final_ce':history[-1]['ce'],
                      'predictions':records},indent=2))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--check',action='store_true');p.add_argument('--train',action='store_true')
    p.add_argument('--out',type=Path,default=ROOT/'results')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if not (args.check or args.train):p.error('Select --check and/or --train.')
    path=args.out/'numpy.json'
    log=json.loads(path.read_text()) if path.exists() else {}
    if args.check:
        results={'handwritten_toy':check_case(toy()),'sparse_top2':check_case(sparse_case())}
        log['checks']=results
        print(json.dumps(results,indent=2))
    if args.train:log['training']=train(args.out)
    path.write_text(json.dumps(log,indent=2))

if __name__=='__main__':main()
