import numpy as np, json, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import harness
def one(seed):
    o = harness.run_rep(seed, dict(N=500,K=15,label_mode='heuristic'), models=['PC-GNN','Peer z-score (unsup.)'], return_objs=True)
    d, te, va = o['d'], o['te'], o['va']
    yt = d['y_true'][te]; rule = d['y_obs'][te]
    s = o['scores']['PC-GNN'][te]
    k = int(rule.sum()) or 1
    top = np.argsort(-s)[:k]
    # among PC-GNN's top-k (same budget as the rule), how many are true anomalies the rule did NOT flag?
    new_true = int(((yt[top] == 1) & (rule[top] == 0)).sum())
    return dict(rule_auc=roc_auc_score(yt, rule), rule_prec=precision_score(yt, rule), rule_rec=recall_score(yt, rule),
                pc_auc=roc_auc_score(yt, s), pc_prec_atk=float(yt[top].mean()), pc_new_true=new_true, rule_n=int(rule.sum()),
                label_precision=float(d['y_true'][d['y_obs']==1].mean()), label_recall=float(d['y_obs'][d['y_true']==1].mean()))
if __name__ == '__main__':
    with Pool(2) as p: r = p.map(one, [5000+s for s in range(20)])
    out = {k: [x[k] for x in r] for k in r[0]}
    json.dump(out, open('res_heur.json','w'))
    for k,v in out.items(): print(f'{k:16s} {np.mean(v):.3f} (sd {np.std(v,ddof=1):.3f})')
