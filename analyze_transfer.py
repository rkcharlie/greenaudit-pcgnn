"""Summaries for the out-of-panel transfer experiment (Section 7.9).
For each scenario: mean test ROC-AUC per model; PC-GNN minus the strongest other model (chosen by mean in that
scenario, so the choice favours the baselines), paired bootstrap 95% interval, Wilcoxon p, Holm over scenarios."""
import json, numpy as np
from scipy.stats import wilcoxon
R = json.load(open('res_transfer.json'))
res = R['res']; rng = np.random.default_rng(7)
OTHER = ['LR (peer-residual + squares)', 'GBM (peer-residual feats)', 'MLP (own+peer)', 'Peer z-score (unsup.)', 'IsolationForest', 'GraphSAGE (no align head)']
rows = []; out = {}
for sc, tab in res.items():
    auc = {m: np.array(v['roc_auc']) for m, v in tab.items()}
    best = max(OTHER, key=lambda m: auc[m].mean())
    d = auc['PC-GNN'] - auc[best]
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    p = wilcoxon(auc['PC-GNN'], auc[best]).pvalue
    out[sc] = dict(mean={m: float(a.mean()) for m, a in auc.items()}, sd={m: float(a.std(ddof=1)) for m, a in auc.items()},
                   best=best, diff=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), p=float(p),
                   nohidden=float((auc['PC-GNN'] - auc['PC-GNN (no hidden layer)']).mean()),
                   nopeers=float((auc['PC-GNN'] - auc['PC-GNN (no peers)']).mean()), n=len(d))
ps = np.array([out[s]['p'] for s in out]); order = np.argsort(ps); m = len(ps); holm = np.empty(m); run = 0
for rank, i in enumerate(order):
    run = max(run, min(1, (m - rank) * ps[i])); holm[i] = run
for s, h in zip(out, holm): out[s]['holm'] = float(h)
json.dump(out, open('res_transfer_summary.json', 'w'), indent=1)
for s, o in out.items():
    print('%-26s PC-GNN %.3f | best other %-28s %.3f | diff %+.3f [%+.3f,%+.3f] holm %.3f | no-hidden %+.3f no-peers %+.3f' %
          (s, o['mean']['PC-GNN'], o['best'], o['mean'][o['best']], o['diff'], o['lo'], o['hi'], o['holm'], o['nohidden'], o['nopeers']))
