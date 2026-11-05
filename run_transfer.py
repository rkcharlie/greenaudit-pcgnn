"""Out-of-panel transfer: train and validate on panel A, score unseen companies in panel B.
Panel B has new companies and new sectors, and may differ from A in noise, anomaly mix or size.
The claim encoder (text -> 8-dim embedding) is fixed across panels (wm_seed), as a real encoder would be.
Standardisation and imputation statistics come from the training companies of A only.
"""
import json, sys, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
from multiprocessing import Pool
from sim import make_panel, standardise_impute, peer_mean
import models as M
from harness import split

WM = 424242
TRAIN_CFG = dict(N=500, K=15, wm_seed=WM)
SCEN = {
    'same generator':            dict(N=500, K=15),
    'heavy-tailed noise (t3)':   dict(N=500, K=15, tail_nu=3),
    'idiosyncratic noise x2':    dict(N=500, K=15, sigma_e=0.24),
    'claim text noise x2':       dict(N=500, K=15, text_noise=0.6),
    'weaker claim adherence':    dict(N=500, K=15, rho=0.4),
    'more level/unsupported':    dict(N=500, K=15, mix=(0.2, 0.2, 0.4, 0.2)),
    'prevalence 5%':             dict(N=500, K=15, prev=0.05),
    'prevalence 30%':            dict(N=500, K=15, prev=0.30),
    'small panel (N=100)':       dict(N=100, K=8),
    'real-size panel (N=55)':    dict(N=55, K=8, sigma_m=0.02, sigma_e=0.10, tail_nu=3, prev=0.25),
}
MODS = ['PC-GNN', 'PC-GNN (no hidden layer)', 'PC-GNN (no peers)', 'GraphSAGE (no align head)',
        'LR (peer-residual + squares)', 'GBM (peer-residual feats)', 'MLP (own+peer)',
        'Peer z-score (unsup.)', 'IsolationForest']


def one(args):
    seed, name = args
    dA = make_panel(seed, **TRAIN_CFG)
    cfgB = dict(SCEN[name]); cfgB['wm_seed'] = WM
    dB = make_panel(seed + 100000, **cfgB)
    NA = len(dA['y_true']); NB = len(dB['y_true'])
    X = np.vstack([dA['X'], dB['X']])
    sec = np.concatenate([dA['sec'], dB['sec'] + 1000])
    y = np.concatenate([dA['y_obs'], np.zeros(NB, int)])        # labels of B are never used
    ytrue = np.concatenate([dA['y_true'], dB['y_true']])
    rng = np.random.default_rng(10_000 + seed)
    trA, vaA, _ = split(rng, dA['y_true'], frac=(0.6, 0.4, 0.0))
    tr, va = trA, vaA
    te = np.arange(NA, NA + NB)
    Xs, _, _ = standardise_impute(X, tr)
    out = {}

    def pc(up, ua, hidden=4):
        hp = dict(hidden=hidden, l2=0.3, l2v=1e-3) if ua else dict(hidden=hidden, l2=0.1, l2v=1e-3)
        m = M.PCGNN(use_peers=up, use_align=ua, seed=seed, **hp).fit(Xs, sec, y, tr, va)
        return m.score(Xs, sec)
    sc = {}
    sc['PC-GNN'] = pc(True, True)
    sc['PC-GNN (no hidden layer)'] = pc(True, True, hidden=0)
    sc['PC-GNN (no peers)'] = pc(False, True)
    sc['GraphSAGE (no align head)'] = pc(True, False)
    P = peer_mean(Xs, sec)
    F = np.hstack([Xs, Xs - P]); F2 = np.hstack([F, F[:, 12:] ** 2])
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.ensemble import HistGradientBoostingClassifier as HG
    sc['LR (peer-residual + squares)'] = LR(C=1.0, max_iter=3000, class_weight='balanced').fit(F2[tr], y[tr]).decision_function(F2)
    sc['GBM (peer-residual feats)'] = HG(max_depth=2, max_iter=100, learning_rate=0.05, class_weight='balanced',
                                         random_state=seed).fit(F[tr], y[tr]).decision_function(F)
    sc['MLP (own+peer)'] = M.mlp_model(Xs, sec, y, tr, True, seed)
    dX = Xs - P
    sc['Peer z-score (unsup.)'] = np.sqrt(dX[:, 0] ** 2 + dX[:, 1] ** 2 + dX[:, 2] ** 2)
    sc['IsolationForest'] = np.concatenate([np.zeros(NA), M.unsup_scores(Xs[te], seed)['IsolationForest']])
    for m, s in sc.items():
        r = M.metrics(ytrue[te], s[te])
        if r is not None:
            out[m] = r
    return out


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    base = int(sys.argv[2]) if len(sys.argv) > 2 else 9000
    res = {}; t0 = time.time()
    with Pool(2) as pool:
        for name in SCEN:
            rs = pool.map(one, [(base + s, name) for s in range(reps)])
            tab = {}
            for r in rs:
                for m, v in r.items():
                    for k, x in v.items():
                        tab.setdefault(m, {}).setdefault(k, []).append(float(x))
            res[name] = tab
            json.dump(dict(cfg=SCEN, reps=reps, base=base, res=res), open('res_transfer.json', 'w'))
            print(name, 'done', round(time.time() - t0), flush=True)
