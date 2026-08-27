"""Adversarial review probes. Each item is a criticism to be confirmed or refuted with numbers."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import numpy as np, pandas as pd, time
from lib import corr_matrix, degrade, OUT
src = open(f"{OUT}/audit.py").read()
exec(src[:src.index("results = {}")])          # load_feeder, pipeline

V0, true_g, XY, nearest, sp = load_feeder("ckt5")
n = len(true_g); nx = true_g.max()+1
seeds = [(7,5),(17,15),(27,25),(37,35),(47,45)]

print("="*70)
print("PROBE 2: how many transformers in the index have zero meters?")
cnt = np.bincount(true_g, minlength=nx)
print(f"  transformers in index: {nx}; with >=1 meter in truth: {(cnt>0).sum()}; "
      f"with zero: {(cnt==0).sum()}")
print("  -> index is built from meta.xfmr.unique(), so by construction every indexed")
print("     transformer serves >=1 meter. Genuinely unmetered transformers do not exist here.")

print("="*70)
print("PROBE 5: is measurement noise frozen across the multi-seed table?")
t0=time.time()
rows_frozen = [pipeline(corr_matrix(degrade(V0)), true_g, XY, nearest, sp, .10, cs, js) for cs,js in seeds]
df_f = pd.DataFrame(rows_frozen)
print(f"  FROZEN noise (published setup, seed=11 fixed):")
for k in ("det","fpr","corr_fix","obs_gate"):
    print(f"    {k:9s} {df_f[k].mean():.4f} ± {df_f[k].std():.4f}")
rows_var = []
for idx,(cs,js) in enumerate(seeds):
    C = corr_matrix(degrade(V0, seed=1000+idx))
    rows_var.append(pipeline(C, true_g, XY, nearest, sp, .10, cs, js))
df_v = pd.DataFrame(rows_var)
print(f"  VARIED noise (one AMI noise draw per seed):")
for k in ("det","fpr","corr_fix","obs_gate"):
    print(f"    {k:9s} {df_v[k].mean():.4f} ± {df_v[k].std():.4f}")
print(f"  [{time.time()-t0:.0f}s]")

print("="*70)
print("PROBE 8: is the corruption model too easy? (uniform nearest-15 vs nearest-3)")
def pipeline_knear(C, cseed, jseed, kn):
    """identical to pipeline() but corruption draws only from the kn nearest transformers."""
    rj = np.random.default_rng(jseed)
    mxy = XY[true_g] + rj.normal(0, 0.35*sp, (n,2))
    Dm = np.nan_to_num(np.sqrt(((mxy[:,None,:]-XY[None,:,:])**2).sum(-1)), nan=np.inf)
    gps = np.argsort(Dm, axis=1)[:,:5]
    rc = np.random.default_rng(cseed)
    rec = true_g.copy()
    bad = rc.choice(n, size=int(.10*n), replace=False)
    for i in bad: rec[i] = rc.choice(nearest[true_g[i]][:kn])
    is_bad = np.zeros(n,bool); is_bad[bad]=True
    within = np.empty(n)
    for i in range(n):
        mem = np.where(rec==rec[i])[0]; mem=mem[mem!=i]
        within[i] = C[i,mem].mean() if len(mem) else np.nan
    ok=~np.isnan(within); med=np.median(within[ok]); mad=np.median(np.abs(within[ok]-med))*1.4826
    props=[]; flagged=np.zeros(n,bool); corrected=np.zeros(n,int)-1
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
    props.sort(reverse=True)
    lab=rec.copy(); base=np.mean(lab==true_g); accs=[]
    for (cf,i,gb) in props: lab[i]=gb; accs.append(np.mean(lab==true_g))
    blind=accs[-1] if accs else base
    rv=np.random.default_rng(cseed+1000)
    ver=set(rv.choice(len(props),size=min(40,len(props)),replace=False).tolist())
    cf_flags=[(true_g[i]!=rec[i]) and (gb==true_g[i]) for (cf,i,gb) in props]
    stop=len(props); seen=[]
    for k in range(len(props)):
        if k in ver: seen.append((k,cf_flags[k]))
        recent=[c for (kk,c) in seen if kk>k-60]
        if len(recent)>=8 and np.mean(recent)<0.5: stop=k; break
    obs=accs[stop-1] if stop>0 and accs else base
    return dict(det=det,fpr=fpr,corr_fix=cfix,base=base,obs_gate=obs,blind=blind)
C0 = corr_matrix(degrade(V0))
for kn in (15,5,3,1):
    d=pd.DataFrame([pipeline_knear(C0,cs,js,kn) for cs,js in seeds])
    print(f"  corruption -> nearest-{kn:2d}: det={d.det.mean():.3f} fpr={d.fpr.mean():.3f} "
          f"corr_fix={d.corr_fix.mean():.3f} base={d.base.mean():.3f} "
          f"obs_gate={d.obs_gate.mean():.3f} blind={d.blind.mean():.3f}")

print("="*70)
print("PROBE 3: singleton false-positive rate, measured directly")
for cs,js in seeds[:3]:
    rj=np.random.default_rng(js)
    mxy=XY[true_g]+rj.normal(0,0.35*sp,(n,2))
    Dm=np.nan_to_num(np.sqrt(((mxy[:,None,:]-XY[None,:,:])**2).sum(-1)),nan=np.inf)
    gps=np.argsort(Dm,axis=1)[:,:5]
    rc=np.random.default_rng(cs); rec=true_g.copy()
    bad=rc.choice(n,size=int(.10*n),replace=False)
    for i in bad: rec[i]=rc.choice(nearest[true_g[i]])
    is_bad=np.zeros(n,bool); is_bad[bad]=True
    within=np.empty(n)
    for i in range(n):
        mem=np.where(rec==rec[i])[0]; mem=mem[mem!=i]
        within[i]=C0[i,mem].mean() if len(mem) else np.nan
    single = np.isnan(within)          # recorded as sole member of its group
    ok=~np.isnan(within); med=np.median(within[ok]); mad=np.median(np.abs(within[ok]-med))*1.4826
    flagged=np.zeros(n,bool)
    for i in range(n):
        cands=set(gps[i].tolist())|{rec[i]}; sc={}
        for g in cands:
            mem=np.where(rec==g)[0]; mem=mem[mem!=i]
            if len(mem): sc[g]=C0[i,mem].mean()
        alt={g:s for g,s in sc.items() if g!=rec[i]}
        if not alt: continue
        gb=max(alt,key=alt.get); sb=alt[gb]; w=sc.get(rec[i],np.nan)
        if (not np.isnan(w) and w<med-3*mad and sb>w) or (sb>med-1*mad and (np.isnan(w) or sb>w)):
            flagged[i]=True
    clean=~is_bad
    print(f"  seed {cs}: recorded-singletons {single.sum():4d} ({single.mean()*100:.1f}%) | "
          f"FPR singles {flagged[clean&single].mean():.3f} | FPR non-singles {flagged[clean&~single].mean():.3f} | "
          f"share of all false positives that are singletons "
          f"{(flagged&clean&single).sum()/max((flagged&clean).sum(),1):.2f}")
