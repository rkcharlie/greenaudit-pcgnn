import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, t as tdist
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})
R = json.load(open('res_all.json')); TH = json.load(open('theory_check.json')); EX = json.load(open('res_explain.json'))
C_OURS, C_ABL, C_BASE, C_UN = '#0072B2', '#56B4E9', '#999999', '#E69F00'
def ci(a):
    a = np.array(a, float); a = a[~np.isnan(a)]; se = a.std(ddof=1) / np.sqrt(len(a)); h = tdist.ppf(0.975, len(a) - 1) * se
    return a.mean(), h
# ---------- Fig A: main
main = R['main']['tab']
order = ['PC-GNN', 'PC-GNN (no hidden layer)', 'PC-GNN (no peers)', 'GraphSAGE (no align head)', 'Fixed-weight SAGE + LR (pilot)',
         'LR (peer-residual + squares)', 'GBM (peer-residual feats)', 'MLP (own+peer)', 'GBM (own+peer)', 'GBM (own)', 'LR (own+peer)', 'LR (own)',
         'Peer z-score (unsup.)', 'LOF', 'IsolationForest', 'OneClassSVM', 'Autoencoder']
labels = {'PC-GNN': 'PC-GNN (proposed)', 'PC-GNN (no hidden layer)': 'ablation: no hidden layer', 'PC-GNN (no peers)': 'ablation: no peers',
          'GraphSAGE (no align head)': 'ablation: GraphSAGE, no alignment head', 'Fixed-weight SAGE + LR (pilot)': 'fixed-weight SAGE + LR (earlier pilot)',
          'LR (peer-residual + squares)': 'LR, peer-residual features', 'GBM (peer-residual feats)': 'GBM, peer-residual features', 'MLP (own+peer)': 'MLP, own + peer',
          'GBM (own+peer)': 'GBM, own + peer', 'GBM (own)': 'GBM, own', 'LR (own+peer)': 'LR, own + peer', 'LR (own)': 'LR, own',
          'Peer z-score (unsup.)': 'peer z-score (no training)', 'LOF': 'Local Outlier Factor', 'IsolationForest': 'Isolation Forest', 'OneClassSVM': 'One-Class SVM', 'Autoencoder': 'Autoencoder'}
col = {m: (C_OURS if m == 'PC-GNN' else C_ABL if m in ('PC-GNN (no hidden layer)', 'PC-GNN (no peers)', 'GraphSAGE (no align head)', 'Fixed-weight SAGE + LR (pilot)') else C_UN if m in ('Peer z-score (unsup.)', 'LOF', 'IsolationForest', 'OneClassSVM', 'Autoencoder') else C_BASE) for m in order}
fig, ax = plt.subplots(figsize=(6.3, 4.3))
for i, m in enumerate(order):
    mu, h = ci(main[m]['roc_auc']); ax.barh(i, mu, xerr=h, color=col[m], height=0.7, capsize=2, error_kw=dict(lw=0.8))
    ax.text(mu + h + 0.006, i, f'{mu:.3f}', va='center', fontsize=7.5)
ax.set_yticks(range(len(order))); ax.set_yticklabels([labels[m] for m in order], fontsize=8); ax.invert_yaxis()
ax.axvline(0.5, color='k', lw=0.6, ls=':'); ax.set_xlim(0.45, 0.92); ax.set_xlabel('Test ROC-AUC (mean over 30 independent panels, 95% CI)')
fig.tight_layout(); fig.savefig('figs/fig_main.png', dpi=300); plt.close(fig)
# ---------- Fig B: sample size
Ns = [38, 60, 100, 250, 500, 1000]; keyN = {38: 'N38', 60: 'N60', 100: 'N100', 250: 'N250', 500: 'N500_prev30', 1000: 'N1000'}
show = [('PC-GNN', C_OURS, '-', 'PC-GNN'), ('PC-GNN (no hidden layer)', C_OURS, '--', 'PC-GNN, no hidden layer'), ('Peer z-score (unsup.)', C_UN, '-', 'peer z-score (no training)'),
        ('LR (peer-residual + squares)', C_BASE, '-', 'LR, peer-residual features'), ('MLP (own+peer)', '#009E73', '-', 'MLP, own + peer'), ('Fixed-weight SAGE + LR (pilot)', '#CC79A7', '-', 'fixed-weight SAGE (pilot)')]
fig, ax = plt.subplots(figsize=(5.2, 3.5))
for m, c, ls, lab in show:
    mu = []; hh = []
    for n in Ns:
        a, h = ci(R[keyN[n]]['tab'][m]['roc_auc']); mu.append(a); hh.append(h)
    mu = np.array(mu); hh = np.array(hh); ax.plot(Ns, mu, color=c, ls=ls, marker='o', ms=3, label=lab); ax.fill_between(Ns, mu - hh, mu + hh, color=c, alpha=0.10)
ax.set_xscale('log'); ax.set_xticks(Ns); ax.set_xticklabels(Ns); ax.axhline(0.5, color='k', lw=0.6, ls=':')
ax.set_xlabel('Number of companies (30% anomalous)'); ax.set_ylabel('Test ROC-AUC'); ax.legend(fontsize=7, frameon=False, loc='center right', bbox_to_anchor=(1.0, 0.33))
ax.axvline(38, color='grey', lw=0.5, ls='--'); ax.text(41, 0.445, 'earlier pilot size (N=38)', fontsize=7, color='grey')
fig.tight_layout(); fig.savefig('figs/fig_sample_size.png', dpi=300); plt.close(fig)
# ---------- Fig C: robustness
ext = ['LR (peer-residual + squares)', 'GBM (peer-residual feats)', 'MLP (own+peer)', 'Peer z-score (unsup.)', 'LOF', 'IsolationForest', 'GraphSAGE (no align head)', 'Fixed-weight SAGE + LR (pilot)']
cfgs = [('main', 'default'), ('labelnoise0.1', 'label noise 10%'), ('labelnoise0.2', 'label noise 20%'), ('labelnoise0.3', 'label noise 30%'), ('heuristic_label', 'rule-based (heuristic) labels'),
        ('textnoise0.6', 'text noise x2'), ('textnoise1.2', 'text noise x4'), ('miss0.2', 'missing KPIs 20%'), ('miss0.4', 'missing KPIs 40%'), ('shock0.0', 'no sector shock (control)'),
        ('shock0.2', 'strong sector shock'), ('prev0.05', 'prevalence 5%'), ('wrongedge0.2', 'wrong peers 20%'), ('wrongedge0.5', 'wrong peers 50%'), ('delta0.6', 'weaker anomalies'), ('rho0.4', 'weaker claim adherence')]
rows = []
for k, lab in cfgs:
    tab = R[k]['tab']; best = max([m for m in ext if m in tab], key=lambda m: np.nanmean(tab[m]['roc_auc']))
    a = np.array(tab['PC-GNN']['roc_auc']); b = np.array(tab[best]['roc_auc']); ok = ~np.isnan(a) & ~np.isnan(b); d = a[ok] - b[ok]
    rng = np.random.default_rng(0); bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    rows.append(dict(cfg=k, label=lab, pc=float(np.nanmean(a)), best=best, best_auc=float(np.nanmean(b)), diff=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=int(ok.sum()),
                     nopeer=float(np.nanmean(tab['PC-GNN (no peers)']['roc_auc'])), nohid=float(np.nanmean(tab['PC-GNN (no hidden layer)']['roc_auc']))))
fig, ax = plt.subplots(figsize=(6.3, 4.6))
for i, r in enumerate(rows):
    sig = r['lo'] > 0 or r['hi'] < 0
    ax.errorbar(r['diff'], i, xerr=[[r['diff'] - r['lo']], [r['hi'] - r['diff']]], fmt='o', color=(C_OURS if r['lo'] > 0 else '#D55E00' if r['hi'] < 0 else C_BASE), capsize=2, ms=4)
ax.axvline(0, color='k', lw=0.7); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r['label'] for r in rows], fontsize=8); ax.invert_yaxis()
ax.set_xlabel('PC-GNN minus strongest independent baseline, ROC-AUC\n(95% bootstrap CI over independent panels)')
fig.tight_layout(); fig.savefig('figs/fig_robust.png', dpi=300); plt.close(fig)
json.dump(rows, open('res_robust_rows.json', 'w'), indent=1)
# ---------- Fig D: theory
import pandas as pd
df = pd.DataFrame(TH); fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0))
ax = axs[0]; ax.scatter(df.raw_formula, df.raw_emp, s=10, c=C_BASE, label='no peers'); ax.scatter(df.peer_formula, df.peer_emp, s=10, c=C_OURS, label='peer-centred')
ax.plot([0.5, 1], [0.5, 1], 'k', lw=0.6); ax.set_xlabel('Closed-form AUC (Prop. 2)'); ax.set_ylabel('Monte Carlo AUC'); ax.legend(frameon=False, fontsize=7)
ax = axs[1]
for k, c in ((3, '#D55E00'), (10, C_OURS), (30, '#009E73')):
    s = df[(df.k == k) & (df.pi == 0.15) & (df.delta == 0.25)]
    ax.plot(s.sm, s.peer_emp - s.raw_emp, 'o-', color=c, ms=3, label=f'k = {k} peers')
ax.axhline(0, color='k', lw=0.6); ax.set_xlabel('Sector-shock sd  (sigma_m)'); ax.set_ylabel('AUC gain from peer centring'); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig('figs/fig_theory.png', dpi=300); plt.close(fig)
# ---------- Fig E: explanation
mods = ['PC-GNN', 'PC-GNN (no hidden layer)', 'GraphSAGE (no align head)', 'Fixed-weight SAGE + LR (pilot)']
short = ['PC-GNN', 'no hidden layer', 'GraphSAGE\n(no align head)', 'fixed-weight\n(pilot)']
metr = [('Shapley_top1', 'Top-ranked group is the\ninjected cause'), ('agree_top1_Shap_vs_Grad', 'Shapley and gradient\nagree on top group'), ('stability_top1', 'Top group stable under\nsmall input noise')]
fig, ax = plt.subplots(figsize=(6.3, 3.1)); w = 0.24
for j, (k, lab) in enumerate(metr):
    for i, m in enumerate(mods):
        mu, h = ci(EX[m][k]); ax.bar(i + (j - 1) * w, mu, w, yerr=h, color=['#0072B2', '#E69F00', '#009E73'][j], capsize=1.5, error_kw=dict(lw=0.7), label=lab if i == 0 else None)
ax.axhline(np.mean(EX['PC-GNN']['chance_top1']), color='k', ls=':', lw=0.7); ax.text(3.45, np.mean(EX['PC-GNN']['chance_top1']) + 0.01, 'chance for first metric', fontsize=6.5, ha='right')
ax.set_xticks(range(4)); ax.set_xticklabels(short, fontsize=8); ax.set_ylim(0, 1.12); ax.set_ylabel('Proportion (mean, 95% CI)'); ax.legend(frameon=False, fontsize=6.8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.2))
fig.tight_layout(); fig.savefig('figs/fig_explain.png', dpi=300); plt.close(fig)
for r in rows: print(f"{r['label']:30s} PC {r['pc']:.3f} vs {r['best'][:22]:22s} {r['best_auc']:.3f}  diff {r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]  noPeer {r['nopeer']:.3f} noHid {r['nohid']:.3f}")
