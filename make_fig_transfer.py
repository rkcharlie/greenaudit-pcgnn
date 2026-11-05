import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
S = json.load(open('res_transfer_summary.json'))
names = list(S)
lab = {'same generator': 'Same generator', 'heavy-tailed noise (t3)': 'Heavy-tailed noise', 'idiosyncratic noise x2': 'Noise x2',
       'claim text noise x2': 'Claim-text noise x2', 'weaker claim adherence': 'Weaker claim adherence',
       'more level/unsupported': 'More level / unsupported', 'prevalence 5%': 'Prevalence 5%', 'prevalence 30%': 'Prevalence 30%',
       'small panel (N=100)': 'New panel, N = 100', 'real-size panel (N=55)': 'New panel, N = 55 (real size)'}
y = np.arange(len(names))[::-1]
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)
a = ax[0]
for key, c, mk, l in (('PC-GNN', '#0072B2', 'o', 'PC-GNN'), ('PC-GNN (no peers)', '#56B4E9', 's', 'PC-GNN without peers'), ('Peer z-score (unsup.)', '#E69F00', '^', 'Peer z-score (no training)')):
    a.scatter([S[n]['mean'][key] for n in names], y, c=c, marker=mk, s=22, label=l, zorder=3)
a.scatter([S[n]['mean'][S[n]['best']] for n in names], y, facecolors='none', edgecolors='#444', marker='D', s=26, label='Strongest other model', zorder=3)
a.set_yticks(y); a.set_yticklabels([lab[n] for n in names], fontsize=8); a.set_xlabel('Test ROC-AUC on the unseen panel'); a.set_xlim(0.6, 0.9)
a.legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2); a.spines[['top', 'right']].set_visible(False); a.set_title('(a) Mean AUC over 30 repeats', fontsize=9)
for yy in y: a.axhline(yy, color='#eee', lw=0.6, zorder=0)
b = ax[1]
d = [S[n]['diff'] for n in names]; lo = [S[n]['diff'] - S[n]['lo'] for n in names]; hi = [S[n]['hi'] - S[n]['diff'] for n in names]
b.errorbar(d, y, xerr=[lo, hi], fmt='o', color='#0072B2', ms=4, capsize=2, lw=1.2)
b.axvline(0, color='k', lw=0.7)
b.set_xlabel('PC-GNN minus strongest other model (AUC)'); b.set_xlim(-0.02, 0.11); b.spines[['top', 'right']].set_visible(False)
b.set_title('(b) Paired difference, 95% bootstrap interval', fontsize=9)
for yy in y: b.axhline(yy, color='#eee', lw=0.6, zorder=0)
plt.tight_layout(); plt.savefig('figs/fig_transfer.png', dpi=220); plt.close(); print('ok')
