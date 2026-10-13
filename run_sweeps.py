import json, time, sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
from run_main import run_cfg
SUB = ['PC-GNN','PC-GNN (no peers)','PC-GNN (no hidden layer)','GraphSAGE (no align head)','Fixed-weight SAGE + LR (pilot)',
       'LR (peer-residual + squares)','GBM (peer-residual feats)','MLP (own+peer)','Peer z-score (unsup.)','LOF','IsolationForest']
import harness
def job(a):
    seed, cfg, models = a
    return harness.run_rep(seed, cfg, models=models, timing=True)
def run_cfg2(cfg, nseed, base, pool, models):
    outs = pool.map(job, [(base+s, cfg, models) for s in range(nseed)])
    tab = {}
    for o in outs:
        for m, r in o['res'].items():
            for k, v in r.items(): tab.setdefault(m, {}).setdefault(k, []).append(v)
        for m, t in o['times'].items(): tab.setdefault(m, {}).setdefault('time', []).append(t)
    return tab
CONFIGS = {}
CONFIGS['main'] = (dict(N=500,K=15), 30, None)
for N in (38,60,100,250,1000):
    CONFIGS[f'N{N}'] = (dict(N=N,K=min(15,max(6,N//5)),prev=0.30), 30, SUB)
CONFIGS['N500_prev30'] = (dict(N=500,K=15,prev=0.30), 30, SUB)
for e in (0.1,0.2,0.3): CONFIGS[f'labelnoise{e}'] = (dict(N=500,K=15,label_mode='noisy',label_noise=e), 20, SUB)
CONFIGS['heuristic_label'] = (dict(N=500,K=15,label_mode='heuristic'), 20, SUB)
for t in (0.6,1.2): CONFIGS[f'textnoise{t}'] = (dict(N=500,K=15,text_noise=t), 20, SUB)
for mb in (0.2,0.4): CONFIGS[f'miss{mb}'] = (dict(N=500,K=15,miss_base=mb), 20, SUB)
for sm in (0.0,0.2): CONFIGS[f'shock{sm}'] = (dict(N=500,K=15,sigma_m=sm), 20, SUB)
for pv in (0.05,): CONFIGS[f'prev{pv}'] = (dict(N=500,K=15,prev=pv), 20, SUB)
for we in (0.2,0.5): CONFIGS[f'wrongedge{we}'] = (dict(N=500,K=15,wrong_edge=we), 20, SUB)
CONFIGS['delta0.6'] = (dict(N=500,K=15,delta_scale=0.6), 20, SUB)
CONFIGS['rho0.4'] = (dict(N=500,K=15,rho=0.4), 20, SUB)
if __name__=='__main__':
    out = {}
    t0=time.time()
    with Pool(2) as pool:
        for name,(cfg,ns,models) in CONFIGS.items():
            out[name] = dict(cfg=cfg, tab=run_cfg2(cfg, ns, 5000, pool, models))
            json.dump(out, open('res_all.json','w'))
            print(name, 'done', round(time.time()-t0), flush=True)
