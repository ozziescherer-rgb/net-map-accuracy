"""PROBE 2b (the serious one): the block detector assigns migrated groups to the
'nearest EMPTY transformer'. In the sim, the ONLY empty transformers are ones emptied
by the corruption itself -> 'empty' is a nearly perfect tell. Real GIS contains spare,
unmetered, streetlight and decommissioned transformers that would act as decoys.
Test: inject phantom always-empty transformers and re-run."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import numpy as np
src=open(f"{OUT}/audit.py").read(); exec(src[:src.index("results = {}")])
from lib import corr_matrix, degrade, OUT
V0,true_g,XY0,nearest,sp = load_feeder("ckt5")
C = corr_matrix(degrade(V0)); n=len(true_g); nx0=true_g.max()+1
seeds=[(7,5),(17,15),(27,25)]

def run(decoy_frac, rng_seed=3):
    rd=np.random.default_rng(rng_seed)
    nd=int(decoy_frac*nx0)
    # phantom transformers scattered among the real ones (jittered real positions)
    idx=rd.choice(nx0,nd); XY=np.vstack([XY0, XY0[idx]+rd.normal(0,0.8*sp,(nd,2))])
    nx=nx0+nd
    out=[]
    for cseed,jseed in seeds:
        rc=np.random.default_rng(cseed); rec=true_g.copy(); bad=[]
        gs=np.bincount(true_g,minlength=nx0)
        for g in rc.permutation(np.where(gs>=2)[0]):
            if len(bad)>=int(0.10*n): break
            mem=np.where(true_g==g)[0]; rec[mem]=rc.choice(nearest[g][:5]); bad.extend(mem.tolist())
        bad=np.array(bad); is_bad=np.zeros(n,bool); is_bad[bad]=True
        rj=np.random.default_rng(jseed); mxy=XY0[true_g]+rj.normal(0,0.35*sp,(n,2))
        occ=np.zeros(nx,bool); occ[:nx0]=np.bincount(rec,minlength=nx0)>0
        props=[]
        for g in range(nx0):
            mem=np.where(rec==g)[0]
            if len(mem)<4: continue
            sub=C[np.ix_(mem,mem)]; A=sub-sub.mean()
            lab=(np.linalg.eigh(A)[1][:,-1]>0)
            c1,c2=mem[lab],mem[~lab]
            if len(c1)<2 or len(c2)<2: continue
            w1=sub[np.ix_(lab,lab)]; w2=sub[np.ix_(~lab,~lab)]
            within=(w1.sum()-len(c1))/max(len(c1)*(len(c1)-1),1); within2=(w2.sum()-len(c2))/max(len(c2)*(len(c2)-1),1)
            if min(within,within2)-sub[np.ix_(lab,~lab)].mean()<0.03: continue
            d1=np.linalg.norm(mxy[c1]-XY[g],axis=1).mean(); d2=np.linalg.norm(mxy[c2]-XY[g],axis=1).mean()
            mig=c1 if d1>d2 else c2; cen=mxy[mig].mean(0)
            dd=np.linalg.norm(XY-cen,axis=1); dd[occ]=np.inf
            t=int(np.argmin(dd))
            if dd[t]<3*sp:
                for i in mig: props.append((int(i),t))
        fl=np.array([i for i,_ in props],dtype=int)
        prec=is_bad[fl].mean() if len(fl) else float('nan')
        corr=sum(1 for i,t in props if t==true_g[i])
        lab2=rec.copy()
        for i,t in props: lab2[i]=t
        out.append((len(props),prec,corr,np.mean(rec==true_g),np.mean(lab2==true_g)))
    return out

for f in (0.0,0.10,0.25,0.50):
    print(f"  decoy empties = {int(f*100):3d}% of real transformer count:")
    for (np_,prec,corr,b,a) in run(f):
        print(f"     {np_:3d} props | flag precision {prec:.2f} | correct target {corr:3d}/{np_:3d} "
              f"| net {b:.3f} -> {a:.3f}  ({'+' if a>=b else ''}{(a-b)*100:.1f} pts)")
