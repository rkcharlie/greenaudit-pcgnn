import time, numpy as np
from sim import make_panel, standardise_impute, peer_mean
import models as M


def split(rng, y_true, frac=(0.5, 0.2, 0.3)):
    N = len(y_true)
    idx = np.arange(N)
    tr, va, te = [], [], []
    for cls in (0, 1):
        ids = rng.permutation(idx[y_true == cls])
        a = int(round(frac[0] * len(ids))); b = int(round((frac[0] + frac[1]) * len(ids)))
        tr += list(ids[:a]); va += list(ids[a:b]); te += list(ids[b:])
    return np.array(tr), np.array(va), np.array(te)


MODEL_ORDER = ['PC-GNN', 'PC-GNN-beta (learned shrinkage)', 'PC-GNN (no peers)', 'PC-GNN (no hidden layer)', 'GraphSAGE (no align head)', 'Fixed-weight SAGE + LR (pilot)',
               'LR (own)', 'LR (own+peer)', 'GBM (own)', 'GBM (own+peer)', 'MLP (own+peer)',
               'GBM (peer-residual feats)', 'LR (peer-residual + squares)',
               'Peer z-score (unsup.)', 'IsolationForest', 'OneClassSVM', 'LOF', 'Autoencoder']


def run_rep(seed, cfg, models=None, timing=False, return_objs=False):
    d = make_panel(seed, **cfg)
    rng = np.random.default_rng(10_000 + seed)
    tr, va, te = split(rng, d['y_true'])
    Xs, _, _ = standardise_impute(d['X'], tr)
    sec, y = d['sec'], d['y_obs']
    ytrue = d['y_true']
    scores, times, objs = {}, {}, {}

    def run(name, fn):
        if models is not None and name not in models:
            return
        t0 = time.perf_counter()
        out = fn()
        times[name] = time.perf_counter() - t0
        scores[name] = out

    def pc(use_peers, use_align):
        # hyper-parameters frozen on development seeds 1-20 (never on evaluation seeds)
        hp = dict(hidden=4, l2=0.3, l2v=1e-3) if use_align else dict(hidden=4, l2=0.1, l2v=1e-3)
        m = M.PCGNN(use_peers=use_peers, use_align=use_align, seed=seed, **hp).fit(Xs, sec, y, tr, va)
        objs[(use_peers, use_align)] = m
        return m.score(Xs, sec)

    run('PC-GNN', lambda: pc(True, True))
    run('PC-GNN (no peers)', lambda: pc(False, True))
    def pcb(beta_init=0.5):
        m = M.PCGNNBeta(hidden=4, l2=0.3, l2v=1e-3, beta_init=beta_init, seed=seed).fit(Xs, sec, y, tr, va)
        objs['beta'] = m
        return m.score(Xs, sec)
    run('PC-GNN-beta (learned shrinkage)', pcb)
    def pc0():
        m = M.PCGNN(hidden=0, l2=0.3, l2v=1e-3, use_peers=True, use_align=True, seed=seed).fit(Xs, sec, y, tr, va)
        objs['nohidden'] = m
        return m.score(Xs, sec)
    run('PC-GNN (no hidden layer)', pc0)
    run('GraphSAGE (no align head)', lambda: pc(True, False))
    run('Fixed-weight SAGE + LR (pilot)', lambda: M.fixed_sage_lr(Xs, sec, y, tr, seed)[0])
    run('LR (own)', lambda: M.lr_model(Xs, sec, y, tr, False))
    run('LR (own+peer)', lambda: M.lr_model(Xs, sec, y, tr, True))
    run('GBM (own)', lambda: M.gb_model(Xs, sec, y, tr, False, seed))
    run('GBM (own+peer)', lambda: M.gb_model(Xs, sec, y, tr, True, seed))
    run('MLP (own+peer)', lambda: M.mlp_model(Xs, sec, y, tr, True, seed))

    def resid_feats():
        P = peer_mean(Xs, sec)
        return np.hstack([Xs, Xs - P])
    def gbm_res():
        F = resid_feats()
        from sklearn.ensemble import HistGradientBoostingClassifier as HG
        c = HG(max_depth=2, max_iter=100, learning_rate=0.05, class_weight='balanced', random_state=seed).fit(F[tr], y[tr])
        return c.decision_function(F)
    def lr_res():
        F = resid_feats()
        from sklearn.linear_model import LogisticRegression as LR
        F = np.hstack([F, F[:, 12:] ** 2])
        c = LR(C=1.0, max_iter=3000, class_weight='balanced').fit(F[tr], y[tr])
        return c.decision_function(F)
    run('GBM (peer-residual feats)', gbm_res)
    run('LR (peer-residual + squares)', lr_res)

    def peerz():
        P = peer_mean(Xs, sec)
        dX = Xs - P
        return np.sqrt(dX[:, 0] ** 2 + dX[:, 1] ** 2 + dX[:, 2] ** 2)
    run('Peer z-score (unsup.)', peerz)
    if models is None or any(n in models for n in ('IsolationForest', 'OneClassSVM', 'LOF', 'Autoencoder')):
        t0 = time.perf_counter()
        u = M.unsup_scores(Xs, seed)
        tt = (time.perf_counter() - t0) / 4
        for k, v in u.items():
            if models is None or k in models:
                scores[k] = v; times[k] = tt

    res = {}
    for name, s in scores.items():
        r = M.metrics(ytrue[te], s[te], ytrue[va] if False else y[va], s[va])
        if r is not None:
            res[name] = r
    out = dict(res=res)
    if timing:
        out['times'] = times
    if return_objs:
        out.update(dict(objs=objs, d=d, Xs=Xs, tr=tr, va=va, te=te, scores=scores))
    return out


if __name__ == '__main__':
    cfg = dict(N=500, K=15)
    t0 = time.time()
    r = run_rep(1, cfg, timing=True)
    print('elapsed', round(time.time() - t0, 1))
    for k in MODEL_ORDER:
        if k in r['res']:
            print(f"{k:34s}", {a: round(b, 3) for a, b in r['res'][k].items()}, round(r['times'][k], 2), 's')
