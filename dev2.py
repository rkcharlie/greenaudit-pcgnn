import numpy as np, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
from sim import make_panel, standardise_impute, peer_mean
from harness import split
import models as M
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HG
from sklearn.neural_network import MLPClassifier

def prep(seed):
    d = make_panel(seed, N=500, K=15)
    rng = np.random.default_rng(10_000+seed); tr,va,te = split(rng,d['y_true'])
    Xs,_,_ = standardise_impute(d['X'], tr)
    return d,tr,va,te,Xs

def one(a):
    seed, kind, hp = a
    d,tr,va,te,Xs = prep(seed); sec=d['sec']; y=d['y_obs']; yt=d['y_true']
    P=peer_mean(Xs,sec); R=np.hstack([Xs,Xs-P])
    if kind=='sage':
        m=M.PCGNN(hidden=hp[1],l2=hp[0],l2v=1e-3,use_align=False,seed=seed).fit(Xs,sec,y,tr,va); s=m.score(Xs,sec)
    elif kind=='nopeer':
        m=M.PCGNN(hidden=4,l2=hp[0],l2v=1e-3,use_peers=False,seed=seed).fit(Xs,sec,y,tr,va); s=m.score(Xs,sec)
    elif kind=='gbm_res':
        s=HG(max_depth=hp[0],max_iter=hp[1],learning_rate=hp[2],class_weight='balanced',random_state=seed).fit(R[tr],y[tr]).decision_function(R)
    elif kind=='gbm_own':
        s=HG(max_depth=hp[0],max_iter=hp[1],learning_rate=hp[2],class_weight='balanced',random_state=seed).fit(Xs[tr],y[tr]).decision_function(Xs)
    elif kind=='gbm_peer':
        F=np.hstack([Xs,P]); s=HG(max_depth=hp[0],max_iter=hp[1],learning_rate=hp[2],class_weight='balanced',random_state=seed).fit(F[tr],y[tr]).decision_function(F)
    elif kind=='mlp':
        F=np.hstack([Xs,P]); s=MLPClassifier(hidden_layer_sizes=(hp[1],),alpha=hp[0],max_iter=800,random_state=seed).fit(F[tr],y[tr]).predict_proba(F)[:,1]
    elif kind=='lr_res':
        F=np.hstack([R,R[:,12:]**2]); s=LogisticRegression(C=hp[0],max_iter=3000,class_weight='balanced').fit(F[tr],y[tr]).decision_function(F)
    return roc_auc_score(yt[te],s[te])

if __name__=='__main__':
    jobs={'sage':[(0.1,4),(0.3,4),(0.3,16),(1.0,16)],'nopeer':[(0.1,),(0.3,),(1.0,)],
          'gbm_res':[(2,100,0.05),(3,150,0.08),(2,300,0.03)],'gbm_own':[(2,100,0.05),(3,150,0.08)],'gbm_peer':[(2,100,0.05),(3,150,0.08)],
          'mlp':[(1e-3,16),(1e-1,16),(1.0,16),(1e-1,4)],'lr_res':[(0.01,),(0.1,),(1.0,)]}
    with Pool(2) as p:
        for k,hps in jobs.items():
            for hp in hps:
                r=p.map(one,[(s,k,hp) for s in range(1,21)])
                print(k,hp,round(float(np.mean(r)),3),flush=True)
