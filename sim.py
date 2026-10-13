"""
Injected-anomaly benchmark for peer-conditioned ESG disclosure checking.

Each company i belongs to a sector s(i).  It states a claimed annual emissions-change
rate c_i (reduction / maintain / growth) in text, and reports a year-on-year change g_it
in Scope 1+2 emissions plus a log-level L_i.  Honest companies follow
    g_it = rho*c_i + m_{s(i),t} + eps_it
where m_{s,t} is a sector-wide (common-mode) shock shared by all peers and eps is
idiosyncratic noise.  Anomalies of four documented types are injected with known cause:
  A1 direction contradiction   (reported change moves against the claim)
  A2 implausible magnitude     (very large swing, e.g. a transcription or boundary error)
  A3 level/boundary inconsistency (level far from sector peers)
  A4 unsupported claim         (KPI evidence absent although a claim is made)
Parameters are calibrated to the audited 38-company panel (YoY sd ~0.19, log10 level sd ~1.2).
"""
import numpy as np

FEATURES = ['level', 'g_cur', 'g_prev', 'miss'] + [f'e{k}' for k in range(8)]
GROUPS = {            # feature index sets for group-level explanation
    'trend': [1, 2],
    'claim_text': list(range(4, 12)),
    'level': [0],
    'missing': [3],
}
CAUSE_SETS = {         # which groups are a correct explanation for each anomaly type
    1: {'trend', 'claim_text'},
    2: {'trend'},
    3: {'level'},
    4: {'missing'},
}


def make_panel(seed, N=500, K=15, prev=0.15, sigma_m=0.10, sigma_e=0.12, rho=0.8,
               text_noise=0.30, miss_base=0.08, mix=(0.5, 0.2, 0.2, 0.1),
               label_mode='clean', label_noise=0.0, wrong_edge=0.0, delta_scale=1.0, tail_nu=None):
    rng = np.random.default_rng(seed)
    # sector sizes: uneven, at least 3 members
    w = rng.dirichlet(np.ones(K) * 1.5)
    sizes = np.maximum(3, np.round(w * N).astype(int))
    sec = np.repeat(np.arange(K), sizes)
    if len(sec) > N:
        sec = rng.permutation(sec)[:N]
    elif len(sec) < N:
        sec = np.concatenate([sec, rng.integers(0, K, N - len(sec))])
    sec = rng.permutation(sec)
    N = len(sec)

    mu_s = rng.normal(5.9, 1.0, K)
    level = mu_s[sec] + rng.normal(0, 0.5, N)
    # claimed rate
    u = rng.random(N)
    claim = np.where(u < 0.70, rng.uniform(-0.12, -0.02, N),
                     np.where(u < 0.85, 0.0, rng.uniform(0.02, 0.08, N)))
    ctype = np.where(claim < -0.005, 0, np.where(claim > 0.005, 2, 1))
    m_cur = rng.normal(0, sigma_m, K)
    m_prev = rng.normal(0, sigma_m, K)
    if tail_nu is None:
        e_cur, e_prev = rng.normal(0, sigma_e, N), rng.normal(0, sigma_e, N)
    else:   # Student-t noise whose robust (MAD-based) sd equals sigma_e
        from scipy.stats import t as _t
        sc = sigma_e / (1.4826 * _t.ppf(0.75, tail_nu))
        e_cur, e_prev = sc * rng.standard_t(tail_nu, N), sc * rng.standard_t(tail_nu, N)
    g_cur = rho * claim + m_cur[sec] + e_cur
    g_prev = rho * claim + m_prev[sec] + e_prev

    # anomalies
    anom = rng.random(N) < prev
    atype = np.zeros(N, int)
    idx = np.where(anom)[0]
    atype[idx] = rng.choice([1, 2, 3, 4], size=len(idx), p=np.array(mix) / np.sum(mix))
    miss = rng.random(N) < miss_base
    for i in idx:
        a = atype[i]
        if a == 1:
            direction = -np.sign(claim[i]) if abs(claim[i]) > 0.005 else rng.choice([-1, 1])
            d = delta_scale * rng.uniform(0.15, 0.35)
            g_cur[i] += direction * d
            if rng.random() < 0.5:
                g_prev[i] += direction * d * 0.6
        elif a == 2:
            g_cur[i] += rng.choice([-1, 1]) * delta_scale * rng.uniform(0.5, 0.9)
        elif a == 3:
            level[i] += rng.choice([-1, 1]) * delta_scale * rng.uniform(0.6, 1.2)
        elif a == 4:
            miss[i] = True
    g_cur = np.clip(g_cur, -0.95, 3.0)

    # text embedding of the claim
    Wm = rng.normal(0, 1, (4, 8))
    Wm, _ = np.linalg.qr(Wm.T)           # 8x4 orthonormal columns
    sig = np.stack([claim / 0.1, ctype == 0, ctype == 1, ctype == 2], 1).astype(float)
    E = sig @ Wm.T + rng.normal(0, text_noise, (N, 8))

    # observed features, missing KPI -> NaN then imputed
    G_cur, G_prev, Lv = g_cur.copy(), g_prev.copy(), level.copy()
    G_cur[miss] = np.nan; G_prev[miss] = np.nan; Lv[miss] = np.nan
    X = np.column_stack([Lv, G_cur, G_prev, miss.astype(float), E])

    # graph: sector cliques with optional wrong edges (peer misassignment)
    sec_obs = sec.copy()
    if wrong_edge > 0:
        flip = rng.random(N) < wrong_edge
        sec_obs[flip] = rng.integers(0, K, flip.sum())

    # observed (possibly weak) labels
    y_true = anom.astype(int)
    if label_mode == 'clean':
        y_obs = y_true.copy()
    elif label_mode == 'noisy':
        flip = rng.random(N) < label_noise
        y_obs = np.where(flip, 1 - y_true, y_true)
    elif label_mode == 'heuristic':
        # rubric-style rule on the RAW trend: no peer adjustment (mirrors Dir scored from raw YoY)
        raw_contra = ((claim < -0.005) & (g_cur > 0.08)) | ((claim > 0.005) & (g_cur < -0.05)) | \
                     (np.abs(g_cur) > 0.5)
        raw_contra = np.where(miss, False, raw_contra)
        y_obs = raw_contra.astype(int)
    else:
        raise ValueError(label_mode)
    return dict(X=X, sec=sec_obs, sec_true=sec, y_true=y_true, y_obs=y_obs, atype=atype,
                claim=claim, N=N, K=K)


def standardise_impute(X, fit_idx):
    """Median-impute KPI columns and z-score using statistics from fit_idx only."""
    X = X.copy()
    for j in (0, 1, 2):
        med = np.nanmedian(X[fit_idx, j])
        col = X[:, j]
        col[np.isnan(col)] = med
        X[:, j] = col
    mu = X[fit_idx].mean(0)
    sd = X[fit_idx].std(0) + 1e-9
    sd[3] = 1.0; mu[3] = 0.0
    return (X - mu) / sd, mu, sd


def peer_mean(X, sec):
    """Mean over same-sector peers excluding self (leave-self-out). Singletons get zeros."""
    N = len(sec)
    P = np.zeros_like(X)
    for s in np.unique(sec):
        ids = np.where(sec == s)[0]
        if len(ids) < 2:
            continue
        tot = X[ids].sum(0)
        P[ids] = (tot - X[ids]) / (len(ids) - 1)
    return P
