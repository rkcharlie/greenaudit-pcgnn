import sys, json, time, numpy as np, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
from harness import run_rep, MODEL_ORDER

def job(a):
    seed, cfg = a
    return run_rep(seed, cfg, timing=True)

def run_cfg(cfg, nseed=30, base=5000, pool=None):
    args = [(base + s, cfg) for s in range(nseed)]
    outs = pool.map(job, args) if pool else [job(a) for a in args]
    tab = {}
    for o in outs:
        for m, r in o['res'].items():
            for k, v in r.items():
                tab.setdefault(m, {}).setdefault(k, []).append(v)
        for m, t in o['times'].items():
            tab.setdefault(m, {}).setdefault('time', []).append(t)
    return tab

if __name__ == '__main__':
    t0 = time.time()
    with Pool(2) as pool:
        tab = run_cfg(dict(N=500, K=15), 30, pool=pool)
    json.dump(tab, open('res_main_clean.json', 'w'))
    print('elapsed', round(time.time() - t0))
    for m in MODEL_ORDER:
        if m in tab:
            a = np.array(tab[m]['roc_auc']); p = np.array(tab[m]['pr_auc'])
            print(f"{m:34s} AUC {a.mean():.3f} (sd {a.std(ddof=1):.3f})  PR {p.mean():.3f}  P@k {np.mean(tab[m]['prec_at_k']):.3f}  F1 {np.nanmean(tab[m]['f1']):.3f}")
