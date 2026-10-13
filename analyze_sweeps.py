import json, numpy as np
R = json.load(open('res_all.json'))
show = ['PC-GNN','PC-GNN (no peers)','PC-GNN (no hidden layer)','GraphSAGE (no align head)','Fixed-weight SAGE + LR (pilot)','LR (peer-residual + squares)','GBM (peer-residual feats)','MLP (own+peer)','Peer z-score (unsup.)','LOF','IsolationForest']
short = ['PC','noPeer','noHid','SAGE','Pilot','LRres','GBMres','MLP','PeerZ','LOF','IF']
print(f"{'config':18s}" + ''.join(f'{s:>8s}' for s in short) + '   PC-minus-bestOther [CI]')
for name, v in R.items():
    tab = v['tab']; row = f'{name:18s}'
    means = {}
    for m in show:
        if m in tab:
            means[m] = np.nanmean(tab[m]['roc_auc']); row += f'{means[m]:8.3f}'
        else: row += f'{"-":>8s}'
    others = [m for m in show if m in tab and m not in ('PC-GNN','PC-GNN (no hidden layer)')]
    best = max(others, key=lambda m: np.nanmean(tab[m]['roc_auc']))
    a = np.array(tab['PC-GNN']['roc_auc']); b = np.array(tab[best]['roc_auc']); ok = ~np.isnan(a) & ~np.isnan(b)
    diff = a[ok] - b[ok]
    rng = np.random.default_rng(0); bs = [rng.choice(diff, len(diff)).mean() for _ in range(4000)]
    row += f'   {diff.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] vs {best.split(" ")[0]} n={ok.sum()}'
    print(row)
