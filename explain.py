import json, itertools, math, time, numpy as np, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
from scipy.stats import spearmanr
from sim import make_panel, standardise_impute, peer_mean, GROUPS, CAUSE_SETS
from harness import split
import models as M
from sklearn.linear_model import LogisticRegression

GN = ['trend', 'claim_text', 'level', 'missing']
GI = [GROUPS[g] for g in GN]
FACT = [math.factorial(i) for i in range(5)]

def coalition_input(X, P, S):
    keep = np.zeros(X.shape[1], bool)
    for g in S: keep[GI[g]] = True
    return X * keep, P * keep

def shapley(f, X, P):
    n = 4
    v = {}
    for r in range(n + 1):
        for S in itertools.combinations(range(n), r):
            Xs, Ps = coalition_input(X, P, S)
            v[S] = f(Xs, Ps)
    phi = np.zeros((len(X), n))
    for g in range(n):
        for r in range(n):
            for S in itertools.combinations([h for h in range(n) if h != g], r):
                w = FACT[len(S)] * FACT[n - len(S) - 1] / FACT[n]
                phi[:, g] += w * (v[tuple(sorted(S + (g,)))] - v[S])
    return phi, v

def grad_x_input(f, X, P, eps=1e-4):
    gx = np.zeros_like(X); gp = np.zeros_like(P)
    for j in range(X.shape[1]):
        Xp = X.copy(); Xp[:, j] += eps; Xm = X.copy(); Xm[:, j] -= eps
        gx[:, j] = (f(Xp, P) - f(Xm, P)) / (2 * eps)
        Pp = P.copy(); Pp[:, j] += eps; Pm = P.copy(); Pm[:, j] -= eps
        gp[:, j] = (f(X, Pp) - f(X, Pm)) / (2 * eps)
    a = gx * X + gp * P
    return np.stack([a[:, g].sum(1) for g in GI], 1)

def occlusion(f, X, P):
    full = f(X, P)
    out = np.zeros((len(X), 4))
    for g in range(4):
        keep = [h for h in range(4) if h != g]
        Xs, Ps = coalition_input(X, P, keep)
        out[:, g] = full - f(Xs, Ps)
    return out

def one(seed):
    cfg = dict(N=500, K=15)
    d = make_panel(seed, **cfg)
    rng = np.random.default_rng(10_000 + seed)
    tr, va, te = split(rng, d['y_true'])
    Xs, _, _ = standardise_impute(d['X'], tr)
    sec, y = d['sec'], d['y_obs']
    P = peer_mean(Xs, sec)
    # models
    pc = M.PCGNN(hidden=4, l2=0.3, l2v=1e-3, seed=seed).fit(Xs, sec, y, tr, va)
    pc0 = M.PCGNN(hidden=0, l2=0.3, l2v=1e-3, seed=seed).fit(Xs, sec, y, tr, va)
    sg = M.PCGNN(hidden=4, l2=0.1, l2v=1e-3, use_align=False, seed=seed).fit(Xs, sec, y, tr, va)
    # fixed pilot-style model
    C = np.hstack([Xs, P]); rg = np.random.default_rng(seed)
    lim = np.sqrt(6 / (C.shape[1] + Xs.shape[1])); W = rg.uniform(-lim, lim, (C.shape[1], Xs.shape[1]))
    clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(np.tanh(C @ W)[tr], y[tr])
    fns = {
        'PC-GNN': lambda X, P_: pc._forward(X, P_, X - P_)[0],
        'PC-GNN (no hidden layer)': lambda X, P_: pc0._forward(X, P_, X - P_)[0],
        'GraphSAGE (no align head)': lambda X, P_: sg._forward(X, P_, X - P_)[0],
        'Fixed-weight SAGE + LR (pilot)': lambda X, P_: np.tanh(np.hstack([X, P_]) @ W) @ clf.coef_[0] + clf.intercept_[0],
    }
    # explain true anomalies in the test split that carry a type with a defined cause
    te_a = te[(d['y_true'][te] == 1)]
    Xa, Pa, at = Xs[te_a], P[te_a], d['atype'][te_a]
    res = {}
    nrng = np.random.default_rng(seed + 7)
    cont = [0, 1, 2] + list(range(4, 12))
    for name, f in fns.items():
        sh, vtab = shapley(f, Xa, Pa)
        gi = grad_x_input(f, Xa, Pa)
        oc = occlusion(f, Xa, Pa)
        r = {}
        for mname, A in (('Shapley', sh), ('GradxInput', gi), ('Occlusion', oc)):
            top = A.argmax(1)
            hit = np.array([GN[t] in CAUSE_SETS[a] for t, a in zip(top, at)])
            r[f'{mname}_top1'] = float(hit.mean())
        r['chance_top1'] = float(np.mean([len(CAUSE_SETS[a]) / 4 for a in at]))
        # per-anomaly-type top-1 hit (Shapley)
        for a in (1, 2, 3, 4):
            m_ = at == a
            if m_.sum():
                r[f'Shapley_top1_A{a}'] = float(np.mean([GN[t] in CAUSE_SETS[a] for t in sh[m_].argmax(1)]))
        # agreement between methods
        r['agree_top1_Shap_vs_Grad'] = float((sh.argmax(1) == gi.argmax(1)).mean())
        r['agree_top1_Shap_vs_Occl'] = float((sh.argmax(1) == oc.argmax(1)).mean())
        r['sign_agree_Shap_vs_Grad'] = float((np.sign(sh) == np.sign(gi)).mean())
        sp = [spearmanr(sh[i], gi[i])[0] for i in range(len(sh)) if np.std(sh[i]) > 0 and np.std(gi[i]) > 0]
        r['spearman_Shap_vs_Grad'] = float(np.nanmean(sp)) if sp else np.nan
        # stability under small input noise (Shapley top-1)
        stab = []
        for _ in range(10):
            Xn = Xa.copy(); Xn[:, cont] += nrng.normal(0, 0.05, (len(Xa), len(cont)))
            Pn = Pa.copy(); Pn[:, cont] += nrng.normal(0, 0.05 / 3, (len(Pa), len(cont)))
            sn, _ = shapley(f, Xn, Pn)
            stab.append((sn.argmax(1) == sh.argmax(1)).mean())
        r['stability_top1'] = float(np.mean(stab))
        # faithfulness: logit drop from deleting top-1 group vs a random other group
        full = f(Xa, Pa)
        drop_top, drop_rand = [], []
        for i in range(len(Xa)):
            g = sh[i].argmax()
            keep = [h for h in range(4) if h != g]
            Xs_, Ps_ = coalition_input(Xa[i:i+1], Pa[i:i+1], keep)
            drop_top.append(full[i] - f(Xs_, Ps_)[0])
            others = [h for h in range(4) if h != g]
            rr = nrng.choice(others)
            keep = [h for h in range(4) if h != rr]
            Xs_, Ps_ = coalition_input(Xa[i:i+1], Pa[i:i+1], keep)
            drop_rand.append(full[i] - f(Xs_, Ps_)[0])
        r['deletion_drop_top'] = float(np.mean(drop_top)); r['deletion_drop_random'] = float(np.mean(drop_rand))
        # Shapley efficiency check (sum of phi = f(full) - f(empty))
        r['efficiency_err'] = float(np.abs(sh.sum(1) - (full - vtab[()])).max())
        res[name] = r
    # own-vs-peer split for PC-GNN (gradient x input, all features)
    gx, gp = pc.grad_input(Xs, sec)
    own = np.abs(gx[te_a] * Xa).sum(1); peer = np.abs(gp[te_a] * Pa).sum(1)
    res['PC-GNN']['peer_share_of_attribution'] = float((peer / (own + peer + 1e-12)).mean())
    # analytic vs numeric gradient sanity
    num = grad_x_input(fns['PC-GNN'], Xa, Pa)
    ana_all = gx[te_a] * Xa + gp[te_a] * Pa
    ana = np.stack([ana_all[:, g].sum(1) for g in GI], 1)
    res['PC-GNN']['grad_analytic_vs_numeric_maxerr'] = float(np.abs(num - ana).max())
    return res

if __name__ == '__main__':
    t0 = time.time()
    seeds = [5000 + s for s in range(30)]
    with Pool(2) as p:
        outs = p.map(one, seeds)
    tab = {}
    for o in outs:
        for m, r in o.items():
            for k, v in r.items():
                tab.setdefault(m, {}).setdefault(k, []).append(v)
    json.dump(tab, open('res_explain.json', 'w'))
    print('elapsed', round(time.time() - t0))
    for m, r in tab.items():
        print('\n' + m)
        for k, v in r.items():
            print(f'  {k:36s} {np.nanmean(v):.3f}  (sd {np.nanstd(v, ddof=1):.3f})')
