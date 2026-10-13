import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'serif','font.size':9})
S=json.load(open('res_final_summary.json'))
R=json.load(open('res_final.json'))
OUT='../paper2_final/'
# ---- fig_shock: (a) AUC vs sector shock sd, (b) calibrated sizes
fig,ax=plt.subplots(1,2,figsize=(8.6,3.4))
sm=[0.0,0.02,0.05,0.10,0.20]; keys=[f'shock{s}' for s in sm]
def ci(k,m):
    x=np.array(R[k]['tab'][m]['roc_auc']); b=np.random.default_rng(1).integers(0,len(x),(2000,len(x)))
    mm=x[b].mean(1); return x.mean(), np.percentile(mm,2.5), np.percentile(mm,97.5)
series=[('PC-GNN','PC-GNN','#0072B2','o'),('PC-GNN (no peers)','PC-GNN without peers','#56B4E9','s'),('Peer z-score (unsup.)','Peer z-score (no training)','#E69F00','^'),('LR (peer-residual + squares)','LR on peer-residual features','#7f7f7f','D')]
for m,l,c,mk in series:
    v=[ci(k,m) for k in keys]
    ax[0].errorbar(sm,[a[0] for a in v],yerr=[[a[0]-a[1] for a in v],[a[2]-a[0] for a in v]],label=l,color=c,marker=mk,ms=4,capsize=2,lw=1.2)
ax[0].set_xlabel('Sector-shock standard deviation $\\sigma_m$'); ax[0].set_ylabel('Test ROC-AUC (30 panels, 95% CI)'); ax[0].legend(frameon=False,fontsize=7.5,loc='lower right'); ax[0].set_title('(a) Peer information vs. common shock',fontsize=9)
ax[0].spines[['top','right']].set_visible(False)
cal=['calib_N55','calib_N150','calib_N500']; Ns=[55,150,500]
series2=[('PC-GNN','PC-GNN','#0072B2','o'),('PC-GNN (no peers)','PC-GNN without peers','#56B4E9','s'),('Peer z-score (unsup.)','Peer z-score (no training)','#E69F00','^'),('LR (peer-residual + squares)','LR on peer-residual features','#7f7f7f','D'),('IsolationForest','Isolation Forest','#CC79A7','v')]
for m,l,c,mk in series2:
    v=[ci(k,m) for k in cal]
    ax[1].errorbar(np.array(Ns)*(1+0.0*np.arange(3)),[a[0] for a in v],yerr=[[a[0]-a[1] for a in v],[a[2]-a[0] for a in v]],label=l,color=c,marker=mk,ms=4,capsize=2,lw=1.2)
ax[1].set_xscale('log'); ax[1].set_xticks(Ns); ax[1].set_xticklabels(Ns); ax[1].set_xlabel('Number of companies $N$'); ax[1].set_title('(b) Real-panel-calibrated scenario',fontsize=9)
ax[1].legend(frameon=False,fontsize=7.5,loc='upper left'); ax[1].spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig(OUT+'fig_shock.png',dpi=220); plt.close()
# ---- fig_types
fig,ax=plt.subplots(figsize=(5.2,3.2))
T=R['default_types']['bytype']; names=['Direction\ncontradiction','Implausible\nmagnitude','Level\ninconsistency','Unsupported\nclaim']
ms=[('PC-GNN','PC-GNN','#0072B2'),('LR (peer-residual + squares)','LR, peer-residual','#7f7f7f'),('Peer z-score (unsup.)','Peer z-score','#E69F00'),('IsolationForest','Isolation Forest','#CC79A7')]
w=0.2
for i,(m,l,c) in enumerate(ms):
    v=[np.mean(T[m][str(t)]) for t in (1,2,3,4)]
    ax.bar(np.arange(4)+(i-1.5)*w,v,w,label=l,color=c)
ax.axhline(0.5,color='k',lw=0.6,ls=':'); ax.set_xticks(range(4)); ax.set_xticklabels(names,fontsize=8); ax.set_ylim(0.3,1.0); ax.set_ylabel('ROC-AUC, this type vs. normal')
ax.legend(frameon=False,fontsize=7.5,ncol=2,loc='upper right'); ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig(OUT+'fig_types.png',dpi=220); plt.close()
# ---- architecture figure
fig,ax=plt.subplots(figsize=(9.0,3.6)); ax.set_xlim(0,116); ax.set_ylim(0,44); ax.axis('off')
def box(x,y,w,h,t,fc='#EAF2FA',ec='#0072B2',fs=8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=1.2',fc=fc,ec=ec,lw=1)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs)
def arr(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,lw=1,color='#333'))
box(1,31,19,9,'Claim text of $i$\n(8-dim embedding $e_i$)',fs=7.5)
box(1,17,19,9,'Reported KPIs of $i$:\nchange $g_i$, lag $g^{\\prime}_i$,\nlevel $\\ell_i$, missing flag $m_i$',fc='#FFF4DE',ec='#E69F00',fs=7.5)
box(1,3,19,9,'Sector peers $\\mathcal{N}(i)$\n(leave-self-out means\n$\\bar g,\\bar e,\\bar\\ell$)',fc='#EEEEEE',ec='#666',fs=7.5)
box(28,25,30,15,'One-layer GraphSAGE block\n$h_i=\\tanh(W_s x_i+W_n\\bar x_i+b)$\n4 units; $x_i=[\\ell,g,g^{\\prime},m,e]$',fs=7)
box(28,3,30,17,'Learned alignment residuals\n$r_i=(g_i-\\bar g)-(e_i-\\bar e)^{\\top}w_c$\n$l_i=\\ell_i-\\bar\\ell,\\quad q_i=g^{\\prime}_i-\\bar g^{\\prime}$\n($w_c$: claim decoder, trained jointly)',fs=7)
box(65,13,23,17,'Head features\n$z_i=[h_i,\\,r_i,\\,r_i^2,\\,l_i,\\,l_i^2,$\n$q_i,\\,m_i]$',fs=8)
box(94,22,21,15,'Anomaly score\n$s_i=v^{\\top}z_i+b_0$\nclass-balanced BCE,\nAdam, early stopping',fc='#E6F4EA',ec='#009E73',fs=7.5)
box(94,3,21,15,'Explanation\nexact group Shapley,\ngradient$\\times$input,\nocclusion',fc='#F7E8F1',ec='#CC79A7',fs=7.5)
arr(20,35,28,34); arr(20,22,28,29); arr(20,8,28,28.5)
arr(20,33,28,17); arr(20,21,28,13); arr(20,8,28,9)
arr(58,32,65,26); arr(58,11,65,17); arr(88,24,94,29); arr(88,19,94,12); arr(104.5,22,104.5,18.4)
ax.text(43,41.5,'graph aggregation (peers)',fontsize=7.5,ha='center',style='italic'); ax.text(43,21,'explicit cross-modal alignment',fontsize=7.5,ha='center',style='italic')
plt.savefig(OUT+'fig_arch.png',dpi=220,bbox_inches='tight'); plt.close()
print('ok')
