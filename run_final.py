import json, time, warnings, numpy as np
warnings.filterwarnings('ignore')
from multiprocessing import Pool
import harness
SUB = ['PC-GNN','PC-GNN-beta (learned shrinkage)','PC-GNN (no peers)','PC-GNN (no hidden layer)','GraphSAGE (no align head)','Fixed-weight SAGE + LR (pilot)',
       'LR (peer-residual + squares)','GBM (peer-residual feats)','MLP (own+peer)','Peer z-score (unsup.)','LOF','IsolationForest']
def job(a):
    seed, cfg, models = a
    o = harness.run_rep(seed, cfg, models=models, timing=False, return_objs=True)
    d, te = o['d'], o['te']
    from sklearn.metrics import roc_auc_score
    bytype = {}
    for m, s in o['scores'].items():
        for t in (1, 2, 3, 4):
            mask = (d['atype'][te] == t) | (d['y_true'][te] == 0)
            if (d['atype'][te] == t).sum() >= 1:
                bytype.setdefault(m, {})[t] = roc_auc_score(d['y_true'][te][mask], s[te][mask])
    beta = o['objs']['beta'].p['bp'][[1, 0]].tolist() if 'beta' in o['objs'] else None
    return dict(res=o['res'], bytype=bytype, beta=beta)
CONFIGS = {}
for sm in (0.0, 0.02, 0.05, 0.10, 0.20): CONFIGS[f'shock{sm}'] = dict(N=500, K=15, sigma_m=sm)
CONFIGS['calib_N55'] = dict(N=55, K=8, sigma_m=0.02, sigma_e=0.10, tail_nu=3, prev=0.25)
CONFIGS['calib_N150'] = dict(N=150, K=10, sigma_m=0.02, sigma_e=0.10, tail_nu=3, prev=0.25)
CONFIGS['calib_N500'] = dict(N=500, K=15, sigma_m=0.02, sigma_e=0.10, tail_nu=3, prev=0.25)
CONFIGS['default_types'] = dict(N=500, K=15)
if __name__ == '__main__':
    out = {}; t0 = time.time()
    with Pool(2) as pool:
        for name, cfg in CONFIGS.items():
            rs = pool.map(job, [(7000 + s, cfg, SUB) for s in range(30)])
            tab = {}; bt = {}; betas = []
            for r in rs:
                for m, v in r['res'].items():
                    for k, x in v.items(): tab.setdefault(m, {}).setdefault(k, []).append(x)
                for m, v in r['bytype'].items():
                    for t, x in v.items(): bt.setdefault(m, {}).setdefault(str(t), []).append(x)
                if r['beta']: betas.append(r['beta'])
            out[name] = dict(cfg=cfg, tab=tab, bytype=bt, beta=betas)
            json.dump(out, open('res_final.json', 'w'))
            print(name, 'done', round(time.time() - t0), flush=True)
