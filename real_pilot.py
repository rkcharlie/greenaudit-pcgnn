import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
df = pd.read_csv('modeling_dataset.csv'); lab = pd.read_csv('company_weak_labels.csv')
c = df.groupby('company').agg(sector=('sector','first'), s=('scope12_tco2e','mean'), p=('scope12_prior','mean')).reset_index().merge(lab, on='company')
c['y'] = (c.weak_label == 'NEGATIVE').astype(int)
c['lvl'] = np.log1p(c.s.clip(lower=0)); c['pct'] = (c.s - c.p) / c.p.replace(0, np.nan)
c['miss'] = c.lvl.isna().astype(float)
N = len(c); y = c.y.values
sizes = c.groupby('sector').company.transform('count').values
print('N', N, 'positives', y.sum(), 'missing emissions', int(c.miss.sum()))
print('sector groups (raw labels):', c.sector.nunique(), ' singletons:', int((c.groupby('sector').company.count()==1).sum()))
k = sizes - 1
print('peers per company: mean %.2f median %.1f; companies with >=3 peers: %d of %d; with 0 peers: %d' % (k.mean(), np.median(k), (k>=3).sum(), N, (k==0).sum()))
# leakage-free LOOCV: imputation, scaling and peer means computed inside each fold using training labels-free stats
def feats(train_idx, idx, mode):
    lv = c.lvl.values.copy(); pc = c.pct.values.copy()
    med_l = np.nanmedian(lv[train_idx]); med_p = np.nanmedian(pc[train_idx])
    lv = np.where(np.isnan(lv), med_l, lv); pc = np.where(np.isnan(pc), med_p, pc)
    X = np.column_stack([lv, pc, c.miss.values])
    mu = X[train_idx].mean(0); sd = X[train_idx].std(0) + 1e-9; mu[2] = 0; sd[2] = 1
    X = (X - mu) / sd
    secs = c.sector.values
    P = np.zeros_like(X)
    for s in np.unique(secs):
        ids = np.where(secs == s)[0]
        if len(ids) > 1: P[ids] = (X[ids].sum(0) - X[ids]) / (len(ids) - 1)
    if mode == 'own': return X
    if mode == 'resid': R = X - P; return np.hstack([R, R[:, :2] ** 2])
    if mode == 'pilot':
        C = np.hstack([X[:, :2], P[:, :2]]); rg = np.random.default_rng(42)
        lim = np.sqrt(6 / (C.shape[1] + 2)); W = rg.uniform(-lim, lim, (C.shape[1], 2)); return np.tanh(C @ W)
out = {}
for mode in ('own', 'resid', 'pilot'):
    pr = np.zeros(N)
    for i in range(N):
        tr = np.array([j for j in range(N) if j != i]); F = feats(tr, [i], mode)
        clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(F[tr], y[tr]); pr[i] = clf.predict_proba(F[i:i+1])[0, 1]
    rng = np.random.default_rng(43)
    bs = []
    for _ in range(2000):
        ii = rng.integers(0, N, N)
        if len(np.unique(y[ii])) == 2: bs.append(roc_auc_score(y[ii], pr[ii]))
    out[mode] = dict(auc=roc_auc_score(y, pr), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), pr_auc=average_precision_score(y, pr))
    print(f'{mode:6s} AUC {out[mode]["auc"]:.3f} [{out[mode]["lo"]:.3f}, {out[mode]["hi"]:.3f}]  PR-AUC {out[mode]["pr_auc"]:.3f}')
# unsupervised peer z-score (no training at all)
tr = np.arange(N); F = feats(tr, None, 'own'); secs = c.sector.values
P = np.zeros_like(F)
for s in np.unique(secs):
    ids = np.where(secs == s)[0]
    if len(ids) > 1: P[ids] = (F[ids].sum(0) - F[ids]) / (len(ids) - 1)
z = np.sqrt(((F - P)[:, :2] ** 2).sum(1))
bs = []; rng = np.random.default_rng(43)
for _ in range(2000):
    ii = rng.integers(0, N, N)
    if len(np.unique(y[ii])) == 2: bs.append(roc_auc_score(y[ii], z[ii]))
out['peerz'] = dict(auc=roc_auc_score(y, z), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
print('peerz  AUC %.3f [%.3f, %.3f]' % (out['peerz']['auc'], out['peerz']['lo'], out['peerz']['hi']))
out['panel'] = dict(N=N, pos=int(y.sum()), n_missing=int(c.miss.sum()), groups=int(c.sector.nunique()), mean_peers=float(k.mean()), ge3=int((k>=3).sum()), zero_peers=int((k==0).sum()))
json.dump(out, open('res_real.json', 'w'))
import json as J
R = J.load(open('res_all.json'))['N38']['tab']
a = np.array(R['PC-GNN']['roc_auc']); print('simulated N=38 PC-GNN AUC mean %.3f sd %.3f; 2.5-97.5 pct [%.3f, %.3f]' % (a.mean(), a.std(ddof=1), *np.percentile(a, [2.5, 97.5])))
