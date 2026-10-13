import numpy as np, warnings, itertools, sys
warnings.filterwarnings('ignore')
from multiprocessing import Pool
from sim import make_panel, standardise_impute, peer_mean
from harness import split
import models as M
from sklearn.metrics import roc_auc_score

def one(a):
    seed, l2, H, lr, pat, l2v = a
    d = make_panel(seed, N=500, K=15)
    rng = np.random.default_rng(10_000+seed); tr,va,te = split(rng,d['y_true'])
    Xs,_,_ = standardise_impute(d['X'], tr); sec=d['sec']; y=d['y_obs']
    out={}
    m = M.PCGNN(hidden=H,l2=l2,l2v=l2v,lr=lr,patience=pat,seed=seed).fit(Xs,sec,y,tr,va)
    s = m.score(Xs,sec)
    out['all']=roc_auc_score(d['y_true'][te], s[te])
    for t in (1,2,3,4):
        mask = (d['atype'][te]==t)|(d['y_true'][te]==0)
        out[t]=roc_auc_score(d['y_true'][te][mask], s[te][mask])
    P=peer_mean(Xs,sec); dX=Xs-P
    z=np.sqrt(dX[:,0]**2+dX[:,1]**2+dX[:,2]**2)
    out['pz_all']=roc_auc_score(d['y_true'][te], z[te])
    for t in (1,2,3,4):
        mask = (d['atype'][te]==t)|(d['y_true'][te]==0)
        out['pz%d'%t]=roc_auc_score(d['y_true'][te][mask], z[te][mask])
    return out

if __name__=='__main__':
    grid=[(1e-1,4,0.02,60,1e-3),(1e-1,4,0.02,60,1e-2),(3e-1,4,0.02,60,1e-3),(3e-1,16,0.02,60,1e-3),(1.0,16,0.02,60,1e-3),(1e-1,16,0.02,60,1e-4)]
    with Pool(2) as p:
        for g in grid:
            r = p.map(one,[(s,)+g for s in range(1,21)])
            keys=r[0].keys()
            print(g, {k: round(float(np.nanmean([x[k] for x in r])),3) for k in keys if not str(k).startswith('pz')})
