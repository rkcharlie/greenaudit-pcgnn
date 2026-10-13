import json, numpy as np
from scipy.stats import wilcoxon
d = json.load(open('res_all.json'))['main']['tab']
ref = 'PC-GNN'
rng = np.random.default_rng(1)
names = [m for m in d if m != ref]
rows = []
for m in names:
    for metric in ('roc_auc', 'pr_auc'):
        a = np.array(d[ref][metric]); b = np.array(d[m][metric]); diff = a - b
        bs = [rng.choice(diff, len(diff)).mean() for _ in range(10000)]
        try: p = wilcoxon(a, b).pvalue
        except Exception: p = np.nan
        rows.append((m, metric, diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), p, (diff > 0).mean()))
# Holm over AUC comparisons
auc = [r for r in rows if r[1] == 'roc_auc']
order = np.argsort([r[5] for r in auc]); holm = {}
k = len(auc); run = 0
for rank, i in enumerate(order):
    adj = min(1.0, (k - rank) * auc[i][5]); run = max(run, adj); holm[auc[i][0]] = run
print(f"{'model':36s} {'AUC':>6s} {'sd':>6s} {'PC-GNN - model (95% CI)':>28s} {'p':>9s} {'Holm p':>9s} {'win%':>5s}")
for m in [ref] + names:
    a = np.array(d[m]['roc_auc'])
    if m == ref:
        print(f"{m:36s} {a.mean():6.3f} {a.std(ddof=1):6.3f}"); continue
    r = [x for x in auc if x[0] == m][0]
    print(f"{m:36s} {a.mean():6.3f} {a.std(ddof=1):6.3f} {r[2]:+.3f} [{r[3]:+.3f}, {r[4]:+.3f}] {r[5]:9.2e} {holm[m]:9.2e} {r[6]*100:5.0f}")
print()
for m in [ref] + names:
    print(f"{m:36s} PR {np.mean(d[m]['pr_auc']):.3f}  P@k {np.mean(d[m]['prec_at_k']):.3f}  F1 {np.nanmean(d[m]['f1']):.3f}  time {np.mean(d[m]['time'])*1000:.0f} ms")
