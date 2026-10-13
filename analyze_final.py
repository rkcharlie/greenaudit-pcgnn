import json, numpy as np
from scipy.stats import wilcoxon
r=json.load(open('res_final.json'))
rng=np.random.default_rng(0)
def boot(d,B=4000):
    d=np.asarray(d); m=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(B)]
    return np.percentile(m,[2.5,97.5])
EXT=['PC-GNN (no peers)','GraphSAGE (no align head)','Fixed-weight SAGE + LR (pilot)','LR (peer-residual + squares)','GBM (peer-residual feats)','MLP (own+peer)','Peer z-score (unsup.)','LOF','IsolationForest']
out={}
for k,v in r.items():
    t=v['tab']; a=np.array(t['PC-GNN']['roc_auc'])
    row={'cfg':v['cfg'],'n':len(a),'PC-GNN':float(a.mean())}
    for m in t:
        x=np.array(t[m]['roc_auc']); row[m]=float(x.mean())
    ext={m:np.mean(t[m]['roc_auc']) for m in EXT if m in t and m!='PC-GNN (no peers)'}
    best=max(ext,key=ext.get)
    d=a-np.array(t[best]['roc_auc']); lo,hi=boot(d)
    row['best_other']=best; row['diff_best']=float(d.mean()); row['ci']=[float(lo),float(hi)]
    row['p_best']=float(wilcoxon(d).pvalue)
    dp=a-np.array(t['PC-GNN (no peers)']['roc_auc']); row['diff_nopeer']=float(dp.mean()); row['ci_nopeer']=list(map(float,boot(dp))); row['p_nopeer']=float(wilcoxon(dp).pvalue)
    b=np.array(t['PC-GNN-beta (learned shrinkage)']['roc_auc']); db=b-a; row['beta_minus_pc']=float(db.mean()); row['beta_ci']=list(map(float,boot(db)))
    row['pr']=float(np.mean(t['PC-GNN']['pr_auc'])); row['pk']=float(np.mean(t['PC-GNN']['prec_at_k']))
    row['bytype']={m:{tt:float(np.mean(vv)) for tt,vv in v['bytype'][m].items()} for m in ('PC-GNN','Peer z-score (unsup.)') if m in v['bytype']}
    out[k]=row
json.dump(out,open('res_final_summary.json','w'),indent=1)
for k,row in out.items():
    print(k,'PC',round(row['PC-GNN'],3),'noPeer',round(row['PC-GNN (no peers)'],3),'best',row['best_other'],round(row[row['best_other']],3),'diff',round(row['diff_best'],3),[round(x,3) for x in row['ci']],'p',round(row['p_best'],4),
          '| vs nopeer',round(row['diff_nopeer'],3),[round(x,3) for x in row['ci_nopeer']],'| beta-pc',round(row['beta_minus_pc'],4),[round(x,4) for x in row['beta_ci']], '| PR',round(row['pr'],3))
    print('   bytype',{m:{t:round(x,2) for t,x in d.items()} for m,d in row['bytype'].items()})
