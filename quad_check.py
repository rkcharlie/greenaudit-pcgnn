import numpy as np, json, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
from sim import make_panel, standardise_impute, peer_mean
from harness import split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score
import models as M
def run(a):
    seed,cfg,C=a
    d=make_panel(seed,**cfg); rng=np.random.default_rng(10_000+seed); tr,va,te=split(rng,d['y_true'])
    Xs,_,_=standardise_impute(d['X'],tr); sec=d['sec']; y=d['y_obs']
    P=peer_mean(Xs,sec); dX=Xs-P
    F=np.hstack([Xs,PolynomialFeatures(2,include_bias=False).fit_transform(dX)])
    mu,sd=F[tr].mean(0),F[tr].std(0)+1e-9; F=(F-mu)/sd
    c=LogisticRegression(C=C,max_iter=5000,class_weight='balanced').fit(F[tr],y[tr])
    s=c.decision_function(F)
    return roc_auc_score(d['y_true'][te],s[te])
if __name__=='__main__':
    cfg=dict(N=500,K=15)
    with Pool(2) as p:
        for C in (0.003,0.01,0.03,0.1,0.3,1.0):
            r=p.map(run,[(s,cfg,C) for s in range(1,21)]); print('dev C',C,round(np.mean(r),3),flush=True)

def final():
    out={}
    R=json.load(open('res_all.json'))
    rng=np.random.default_rng(4)
    for name,cfg,key in (('main',dict(N=500,K=15),'main'),('N250',dict(N=250,K=15,prev=0.30),'N250'),('N100',dict(N=100,K=15,prev=0.30),'N100')):
        with Pool(2) as p:
            q=np.array(p.map(run,[(5000+s,cfg,0.1) for s in range(30)]))
        pc=np.array(R[key]['tab']['PC-GNN']['roc_auc'])
        d=pc-q; b=[d[rng.integers(0,30,30)].mean() for _ in range(4000)]
        from scipy.stats import wilcoxon
        out[name]=dict(quad=float(q.mean()),pc=float(pc.mean()),diff=float(d.mean()),ci=list(map(float,np.percentile(b,[2.5,97.5]))),p=float(wilcoxon(d).pvalue))
        print(name,out[name],flush=True)
    json.dump(out,open('res_quad.json','w'))
