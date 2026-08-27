import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import numpy as np, pandas as pd
from lib import corr_matrix, degrade, OUT
src=open(f"{OUT}/audit.py").read(); exec(src[:src.index("results = {}")])
V0,true_g,XY,nearest,sp = load_feeder("ckt5")
n=len(true_g); nx=true_g.max()+1
C0 = corr_matrix(degrade(V0))
seeds=[(7,5),(17,15),(27,25)]

print("="*70)
print("PROBE 1: GPS prior centered on truth. What if premises are biased toward a neighbor?")
def pipe_biased(C, cseed, jseed, bias_frac, pull=0.6):
    rj=np.random.default_rng(jseed)
    mxy=XY[true_g]+rj.normal(0,0.35*sp,(n,2))
    # a fraction of premises sit `pull` of the way toward a random neighboring transformer
    nb = rj.integers(0,3,n)
    tgt = XY[nearest[true_g, nb]]
    sel = rj.random(n) < bias_frac
    mxy[sel] = mxy[sel] + pull*(tgt[sel]-XY[true_g][sel])
    Dm=np.nan_to_num(np.sqrt(((mxy[:,None,:]-XY[None,:,:])**2).sum(-1)),nan=np.inf)
    gps=np.argsort(Dm,axis=1)[:,:5]
    contain=np.mean([true_g[i] in gps[i] for i in range(n)])
    nearest1=np.mean(np.argmin(Dm,axis=1)==true_g)
    rc=np.random.default_rng(cseed); rec=true_g.copy()
    bad=rc.choice(n,size=int(.10*n),replace=False)
    for i in bad: rec[i]=rc.choice(nearest[true_g[i]])
    is_bad=np.zeros(n,bool); is_bad[bad]=True
    within=np.empty(n)
    for i in range(n):
        mem=np.where(rec==rec[i])[0]; mem=mem[mem!=i]
        within[i]=C[i,mem].mean() if len(mem) else np.nan
    ok=~np.isnan(within); med=np.median(within[ok]); mad=np.median(np.abs(within[ok]-med))*1.4826
    props=[];flagged=np.zeros(n,bool);corrected=np.zeros(n,int)-1
    for i in range(n):
        cands=set(gps[i].tolist())|{rec[i]}; sc={}
        for g in cands:
            mem=np.where(rec==g)[0]; mem=mem[mem!=i]
            if len(mem): sc[g]=C[i,mem].mean()
        alt={g:s for g,s in sc.items() if g!=rec[i]}
        if not alt: continue
        gb=max(alt,key=alt.get); sb=alt[gb]; w=sc.get(rec[i],np.nan)
        if (not np.isnan(w) and w<med-3*mad and sb>w) or (sb>med-1*mad and (np.isnan(w) or sb>w)):
            flagged[i]=True; corrected[i]=gb
            props.append((sb-(w if not np.isnan(w) else med-3*mad),i,gb))
    fixable=np.array([np.sum((rec==true_g[i])&(np.arange(n)!=i)&(true_g==true_g[i]))>0 for i in bad])
    det=flagged[bad].mean(); fpr=flagged[~is_bad].mean()
    cfix=(corrected[bad][fixable]==true_g[bad][fixable]).mean() if fixable.any() else np.nan
    props.sort(reverse=True); lab=rec.copy(); base=np.mean(lab==true_g); accs=[]
    for (cf,i,gb) in props: lab[i]=gb; accs.append(np.mean(lab==true_g))
    blind=accs[-1] if accs else base
    rv=np.random.default_rng(cseed+1000)
    ver=set(rv.choice(len(props),size=min(40,len(props)),replace=False).tolist())
    fl=[(true_g[i]!=rec[i]) and (gb==true_g[i]) for (cf,i,gb) in props]
    stop=len(props); seen=[]
    for k in range(len(props)):
        if k in ver: seen.append((k,fl[k]))
        rec_=[c for (kk,c) in seen if kk>k-60]
        if len(rec_)>=8 and np.mean(rec_)<0.5: stop=k; break
    return dict(contain=contain,near1=nearest1,det=det,fpr=fpr,corr_fix=cfix,base=base,
                obs_gate=accs[stop-1] if stop>0 and accs else base, blind=blind)
for bf in (0.0,0.10,0.25,0.50):
    d=pd.DataFrame([pipe_biased(C0,cs,js,bf) for cs,js in seeds])
    print(f"  {int(bf*100):3d}% premises pulled toward a neighbor: contain(K=5)={d.contain.mean():.3f} "
          f"nearest-1 correct={d.near1.mean():.3f} det={d.det.mean():.3f} corr_fix={d.corr_fix.mean():.3f} "
          f"obs_gate={d.obs_gate.mean():.3f} base={d.base.mean():.3f}")

print("="*70)
print("PROBE 4: is the coupling 'z' really a z? effective sample size check")
P=np.load(f"{OUT}/P15_90d.npy"); V=degrade(V0); T=V.shape[0]
X=np.column_stack([np.ones(T),P.sum(1),V.mean(1)])
rV=V-X@np.linalg.lstsq(X,V,rcond=None)[0]; rP=P-X@np.linalg.lstsq(X,P,rcond=None)[0]
rVz=(rV-rV.mean(0))/(rV.std(0)+1e-12); rPz=(rP-rP.mean(0))/(rP.std(0)+1e-12)
def ac1(Z):
    a=Z[:-1]-Z[:-1].mean(0); b=Z[1:]-Z[1:].mean(0)
    return float(np.mean((a*b).mean(0)/(a.std(0)*b.std(0)+1e-12)))
rho_v, rho_p = ac1(rVz), ac1(rPz)
neff_factor_v=(1-rho_v)/(1+rho_v); neff_factor_p=(1-rho_p)/(1+rho_p)
fac=np.sqrt(max(min(neff_factor_v,neff_factor_p),1e-6))
print(f"  lag-1 autocorr: residual voltage rho={rho_v:.3f}, residual load rho={rho_p:.3f}")
print(f"  T={T}; naive sqrt(T)={np.sqrt(T):.1f}")
print(f"  Bartlett-style effective T (voltage) ~ {T*neff_factor_v:.0f}, (load) ~ {T*neff_factor_p:.0f}")
print(f"  => a reported 'z' of -8 is really about {8*fac:.2f} sigma if you use the more")
print(f"     conservative effective sample size. THETA=-8 is a threshold, not 8 sigma.")
