"""
Models for the benchmark.  PCGNN is a one-layer GraphSAGE-style network with an explicit,
learnable cross-modal alignment residual, trained end to end by back-propagation (Adam).
Gradients are derived by hand and verified against finite differences (see gradcheck()).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
from sim import peer_mean

TXT = slice(4, 12)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class PCGNN:
    """use_peers: peer aggregation + peer-centred residuals; use_align: explicit residual head."""

    def __init__(self, hidden=16, use_peers=True, use_align=True, lr=0.02, epochs=400,
                 l2=1e-3, patience=60, seed=0, l2v=None):
        self.H, self.use_peers, self.use_align = hidden, use_peers, use_align
        self.lr, self.epochs, self.l2, self.patience, self.seed = lr, epochs, l2, patience, seed
        self.l2v = l2 if l2v is None else l2v

    # ---- parameter handling -------------------------------------------------------
    def _init(self, d):
        rng = np.random.default_rng(self.seed)
        H = self.H
        nz = 6 if self.use_align else 1               # [r, r^2, l, l^2, q, miss] or [miss]
        self.p = {
            'Ws': rng.normal(0, 1 / np.sqrt(d), (d, H)),
            'Wn': rng.normal(0, 1 / np.sqrt(d), (d, H)) if self.use_peers else np.zeros((d, H)),
            'b': np.zeros(H),
            'wc': np.zeros(8),
            'v': rng.normal(0, 0.1, H + nz),
            'b0': np.zeros(1),
        }

    def _inputs(self, X, sec):
        P = peer_mean(X, sec) if self.use_peers else np.zeros_like(X)
        return X, P, X - P

    def _forward(self, X, P, dX, p=None):
        p = p or self.p
        A = X @ p['Ws'] + P @ p['Wn'] + p['b']
        h = np.tanh(A)
        if self.use_align:
            r = dX[:, 1] - dX[:, TXT] @ p['wc']
            l = dX[:, 0]
            q = dX[:, 2]
            miss = X[:, 3]
            z = np.column_stack([h, r, r ** 2, l, l ** 2, q, miss])
        else:
            r = l = q = None
            z = np.column_stack([h, X[:, 3]])
        logit = z @ p['v'] + p['b0'][0]
        return logit, (A, h, z, r)

    def _loss_grad(self, X, P, dX, y, w, p=None):
        p = p or self.p
        logit, (A, h, z, r) = self._forward(X, P, dX, p)
        pr = sigmoid(logit)
        sw = w.sum()
        eps = 1e-12
        loss = -(w * (y * np.log(pr + eps) + (1 - y) * np.log(1 - pr + eps))).sum() / sw
        reg = self.l2 * sum((p[k] ** 2).sum() for k in ('Ws', 'Wn', 'wc')) + self.l2v * (p['v'] ** 2).sum()
        dlogit = w * (pr - y) / sw
        g = {}
        g['v'] = z.T @ dlogit + 2 * self.l2v * p['v']
        g['b0'] = np.array([dlogit.sum()])
        dz = np.outer(dlogit, p['v'])
        dh = dz[:, :self.H]
        dA = dh * (1 - h ** 2)
        g['Ws'] = X.T @ dA + 2 * self.l2 * p['Ws']
        g['Wn'] = (P.T @ dA + 2 * self.l2 * p['Wn']) if self.use_peers else np.zeros_like(p['Wn'])
        g['b'] = dA.sum(0)
        if self.use_align:
            dr = dz[:, self.H] + 2 * r * dz[:, self.H + 1]
            g['wc'] = -(dX[:, TXT].T @ dr) + 2 * self.l2 * p['wc']
        else:
            g['wc'] = np.zeros(8)
        return loss + reg, g

    # ---- training -----------------------------------------------------------------
    def fit(self, X, sec, y, train_idx, val_idx):
        X, P, dX = self._inputs(X, sec)
        self._init(X.shape[1])
        yt = y[train_idx].astype(float)
        pos = max(yt.sum(), 1.0); neg = max(len(yt) - yt.sum(), 1.0)
        wcls = np.where(yt == 1, 0.5 / pos, 0.5 / neg) * len(yt)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        s = {k: np.zeros_like(v) for k, v in self.p.items()}
        best, best_p, bad = -1, None, 0
        b1, b2 = 0.9, 0.999
        for ep in range(1, self.epochs + 1):
            loss, g = self._loss_grad(X[train_idx], P[train_idx], dX[train_idx], yt, wcls)
            for k in self.p:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                s[k] = b2 * s[k] + (1 - b2) * g[k] ** 2
                self.p[k] = self.p[k] - self.lr * (m[k] / (1 - b1 ** ep)) / (np.sqrt(s[k] / (1 - b2 ** ep)) + 1e-8)
            if ep % 5 == 0 and len(np.unique(y[val_idx])) == 2:
                sc = self.score_arrays(X, P, dX)[val_idx]
                auc = roc_auc_score(y[val_idx], sc)
                if auc > best + 1e-4:
                    best, bad = auc, 0
                    best_p = {k: v.copy() for k, v in self.p.items()}
                else:
                    bad += 5
                    if bad >= self.patience:
                        break
        if best_p is not None:
            self.p = best_p
        self.epochs_run = ep
        return self

    def score_arrays(self, X, P, dX):
        return self._forward(X, P, dX)[0]

    def score(self, X, sec):
        X, P, dX = self._inputs(X, sec)
        return self.score_arrays(X, P, dX)

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    # ---- explanation helpers ------------------------------------------------------
    def grad_input(self, X, sec):
        """Exact d logit / d (own x) and d logit / d (peer-mean x), via the chain rule."""
        X, P, dX = self._inputs(X, sec)
        p = self.p
        A = X @ p['Ws'] + P @ p['Wn'] + p['b']
        h = np.tanh(A)
        dh = 1 - h ** 2
        v = p['v']
        gh = (dh * v[:self.H]) # N x H
        gx = gh @ p['Ws'].T
        gp = gh @ p['Wn'].T
        if self.use_align:
            r = dX[:, 1] - dX[:, TXT] @ p['wc']
            dl_dr = v[self.H] + 2 * r * v[self.H + 1]
            dl_dl = v[self.H + 2] + 2 * dX[:, 0] * v[self.H + 3]
            dl_dq = v[self.H + 4]
            gx[:, 1] += dl_dr
            gx[:, TXT] += -np.outer(dl_dr, p['wc'])
            gx[:, 0] += dl_dl
            gx[:, 2] += dl_dq
            gx[:, 3] += v[self.H + 5]
            gp[:, 1] += -dl_dr
            gp[:, TXT] += np.outer(dl_dr, p['wc'])
            gp[:, 0] += -dl_dl
            gp[:, 2] += -dl_dq
        else:
            gx[:, 3] += v[self.H]
        return gx, gp


class PCGNNBeta(PCGNN):
    """PC-GNN with learned shrinkage peer-centring:  residual = own - beta (.) peer_mean, beta learned per
    feature (beta = 1: full peer-centring; beta = 0: no peers).  Initialised at beta_init."""

    def __init__(self, *a, beta_init=0.5, **k):
        super().__init__(*a, **k)
        self.beta_init = beta_init

    def _init(self, d):
        super()._init(d)
        self.p['bp'] = np.full(d, self.beta_init)

    def _forward(self, X, P, dX=None, p=None):
        p = p or self.p
        return super()._forward(X, P, X - p['bp'] * P, p)

    def score_arrays(self, X, P, dX=None):
        return self._forward(X, P, None)[0]

    def _loss_grad(self, X, P, dX, y, w, p=None):
        p = p or self.p
        dXb = X - p['bp'] * P
        loss, g = super()._loss_grad(X, P, dXb, y, w, p)
        if self.use_align and self.use_peers:
            logit, (A, h, z, r) = PCGNN._forward(self, X, P, dXb, p)
            pr = sigmoid(logit)
            dlogit = w * (pr - y) / w.sum()
            dz = np.outer(dlogit, p['v'])
            H = self.H
            dr = dz[:, H] + 2 * r * dz[:, H + 1]
            dl = dz[:, H + 2] + 2 * dXb[:, 0] * dz[:, H + 3]
            dq = dz[:, H + 4]
            G = np.zeros_like(X)
            G[:, 1] += dr
            G[:, TXT] += -np.outer(dr, p['wc'])
            G[:, 0] += dl
            G[:, 2] += dq
            g['bp'] = -(P * G).sum(0)
        else:
            g['bp'] = np.zeros_like(p['bp'])
        return loss, g

    def grad_input(self, X, sec):
        X, P, _ = self._inputs(X, sec)
        p = self.p; b = p['bp']
        dX = X - b * P
        A = X @ p['Ws'] + P @ p['Wn'] + p['b']
        h = np.tanh(A); v = p['v']; H = self.H
        gh = (1 - h ** 2) * v[:H]
        gx = gh @ p['Ws'].T; gp = gh @ p['Wn'].T
        r = dX[:, 1] - dX[:, TXT] @ p['wc']
        dl_dr = v[H] + 2 * r * v[H + 1]
        dl_dl = v[H + 2] + 2 * dX[:, 0] * v[H + 3]
        dl_dq = v[H + 4]
        gx[:, 1] += dl_dr; gx[:, TXT] += -np.outer(dl_dr, p['wc']); gx[:, 0] += dl_dl; gx[:, 2] += dl_dq; gx[:, 3] += v[H + 5]
        gp[:, 1] += -b[1] * dl_dr; gp[:, TXT] += np.outer(dl_dr, b[TXT] * p['wc']); gp[:, 0] += -b[0] * dl_dl; gp[:, 2] += -b[2] * dl_dq
        return gx, gp


def gradcheck(seed=0):
    from sim import make_panel, standardise_impute
    d = make_panel(seed, N=60, K=5)
    Xs, _, _ = standardise_impute(d['X'], np.arange(d['N']))
    m = PCGNN(seed=1)
    X, P, dX = m._inputs(Xs, d['sec'])
    m._init(X.shape[1])
    m.p['wc'] = np.random.default_rng(3).normal(0, 0.3, 8)
    y = d['y_true'].astype(float); w = np.ones(len(y))
    loss, g = m._loss_grad(X, P, dX, y, w)
    worst = 0.0
    rng = np.random.default_rng(5)
    for k in m.p:
        flat = m.p[k].ravel()
        for _ in range(4):
            j = rng.integers(len(flat))
            old = flat[j]; e = 1e-5
            flat[j] = old + e; lp, _ = m._loss_grad(X, P, dX, y, w)
            flat[j] = old - e; lm, _ = m._loss_grad(X, P, dX, y, w)
            flat[j] = old
            num = (lp - lm) / (2 * e)
            ana = g[k].ravel()[j]
            worst = max(worst, abs(num - ana) / (abs(num) + abs(ana) + 1e-8))
    # input-gradient check
    gx, gp = m.grad_input(Xs, d['sec'])
    base = m.score(Xs, d['sec'])
    i = 7; j = 1; e = 1e-5
    X2 = Xs.copy(); X2[i, j] += e
    # own-feature perturbation also moves peers' peer-means; isolate by using the node-wise partial
    Xp, Pp, dXp = m._inputs(Xs, d['sec'])
    Xp2 = Xp.copy(); Xp2[i, j] += e
    num = (m._forward(Xp2[i:i+1], Pp[i:i+1], (Xp2 - Pp)[i:i+1])[0][0] - m._forward(Xp[i:i+1], Pp[i:i+1], dXp[i:i+1])[0][0]) / e
    return worst, abs(num - gx[i, j])


# ---------------------------------------------------------------------------------
# Baselines.  Each returns a real-valued anomaly score for ALL nodes (transductive).
# ---------------------------------------------------------------------------------
def feats(Xs, sec, peers):
    return np.hstack([Xs, peer_mean(Xs, sec)]) if peers else Xs


def lr_model(Xs, sec, y, tr, peers):
    F = feats(Xs, sec, peers)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0).fit(F[tr], y[tr])
    return clf.decision_function(F)


def gb_model(Xs, sec, y, tr, peers, seed):
    F = feats(Xs, sec, peers)
    clf = HistGradientBoostingClassifier(max_depth=(2 if not peers else 3), max_iter=(100 if not peers else 150), learning_rate=(0.05 if not peers else 0.08),
                                         class_weight='balanced', random_state=seed).fit(F[tr], y[tr])
    return clf.decision_function(F)


def mlp_model(Xs, sec, y, tr, peers, seed):
    F = feats(Xs, sec, peers)
    clf = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', alpha=1.0, max_iter=800,
                        random_state=seed).fit(F[tr], y[tr])
    return clf.predict_proba(F)[:, 1]


def fixed_sage_lr(Xs, sec, y, tr, seed):
    """Replica of the original pilot: untrained Xavier GraphSAGE + tanh, then a trained LR head."""
    P = peer_mean(Xs, sec)
    C = np.hstack([Xs, P])
    rng = np.random.default_rng(seed)
    F = Xs.shape[1]
    lim = np.sqrt(6 / (C.shape[1] + F))
    W = rng.uniform(-lim, lim, (C.shape[1], F))
    H1 = np.tanh(C @ W)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(H1[tr], y[tr])
    return clf.decision_function(H1), (W, H1)


def unsup_scores(Xs, seed):
    out = {}
    out['IsolationForest'] = -IsolationForest(random_state=seed).fit(Xs).score_samples(Xs)
    out['OneClassSVM'] = -OneClassSVM(nu=0.2, gamma='scale').fit(Xs).decision_function(Xs)
    lof = LocalOutlierFactor(n_neighbors=10).fit(Xs)
    out['LOF'] = -lof.negative_outlier_factor_
    ae = MLPRegressor(hidden_layer_sizes=(4,), activation='tanh', max_iter=1500, random_state=seed,
                      solver='lbfgs').fit(Xs, Xs)
    out['Autoencoder'] = np.linalg.norm(Xs - ae.predict(Xs), axis=1)
    return out


def metrics(y, s, val_y=None, val_s=None):
    out = {}
    if len(np.unique(y)) < 2:
        return None
    out['roc_auc'] = roc_auc_score(y, s)
    out['pr_auc'] = average_precision_score(y, s)
    k = int(y.sum())
    top = np.argsort(-s)[:k]
    out['prec_at_k'] = float(y[top].mean())
    # threshold from validation (max F1), applied to test
    if val_y is not None and len(np.unique(val_y)) == 2:
        cand = np.quantile(val_s, np.linspace(0.05, 0.95, 37))
        f1s = [f1_score(val_y, val_s >= t, zero_division=0) for t in cand]
        t = cand[int(np.argmax(f1s))]
        out['f1'] = f1_score(y, s >= t, zero_division=0)
    else:
        out['f1'] = np.nan
    return out


def gradcheck_beta(seed=0):
    from sim import make_panel, standardise_impute
    d = make_panel(seed, N=60, K=5)
    Xs, _, _ = standardise_impute(d['X'], np.arange(d['N']))
    m = PCGNNBeta(hidden=4, seed=1, beta_init=0.6)
    X, P, _ = m._inputs(Xs, d['sec'])
    m._init(X.shape[1]); m.p['wc'] = np.random.default_rng(3).normal(0, 0.3, 8)
    for k in ('v', 'Ws', 'Wn'):
        m.p[k] = m.p[k] + np.random.default_rng(4).normal(0, 0.2, m.p[k].shape)
    y = d['y_true'].astype(float); w = np.ones(len(y))
    loss, g = m._loss_grad(X, P, None, y, w)
    worst = 0.0; rng = np.random.default_rng(5)
    for k in m.p:
        flat = m.p[k].ravel()
        for _ in range(6):
            j = rng.integers(len(flat)); old = flat[j]; e = 1e-5
            flat[j] = old + e; lp, _ = m._loss_grad(X, P, None, y, w)
            flat[j] = old - e; lm, _ = m._loss_grad(X, P, None, y, w)
            flat[j] = old
            num = (lp - lm) / (2 * e); ana = g[k].ravel()[j]
            worst = max(worst, abs(num - ana) / (abs(num) + abs(ana) + 1e-8))
    # input-gradient: perturb node i's own feature only (peers held fixed)
    gx, gp = m.grad_input(Xs, d['sec'])
    i, j, e = 7, 1, 1e-5
    Xp2 = X.copy(); Xp2[i, j] += e
    num = (m._forward(Xp2[i:i+1], P[i:i+1])[0][0] - m._forward(X[i:i+1], P[i:i+1])[0][0]) / e
    Pp2 = P.copy(); Pp2[i, j] += e
    numP = (m._forward(X[i:i+1], Pp2[i:i+1])[0][0] - m._forward(X[i:i+1], P[i:i+1])[0][0]) / e
    return worst, abs(num - gx[i, j]), abs(numP - gp[i, j])


if __name__ == '__main__':
    print('gradcheck (max rel err params, input-grad abs err):', gradcheck())
    print('gradcheck beta model (param rel err, d/dx abs err, d/dpeer abs err):', gradcheck_beta())
