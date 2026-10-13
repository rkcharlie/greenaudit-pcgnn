import numpy as np, json
from scipy.stats import norm
rng = np.random.default_rng(2026)

def auc_emp(a, n):
    # P(a > n) with ties 1/2, via ranks
    from scipy.stats import rankdata
    x = np.concatenate([a, n]); r = rankdata(x)
    return (r[:len(a)].sum() - len(a)*(len(a)+1)/2) / (len(a)*len(n))

def sim(sig, sm, k, pi, delta, M=200_000):
    # normal company i and k peers; each peer anomalous w.p. pi (shift +delta)
    def residual(i_anom):
        m = rng.normal(0, sm, M)
        e_i = rng.normal(0, sig, M) + (delta if i_anom else 0.0)
        peers = rng.normal(0, sig, (M, k)) + (rng.random((M, k)) < pi) * delta
        s_i = m + e_i
        r = e_i - peers.mean(1)           # shock cancels exactly
        return s_i, r
    s_n, r_n = residual(False); s_a, r_a = residual(True)
    return auc_emp(s_a, s_n), auc_emp(r_a, r_n)

def formula(sig, sm, k, pi, delta):
    raw = norm.cdf(delta / np.sqrt(2 * (sig**2 + sm**2)))
    peer = norm.cdf(delta / np.sqrt(2 * (sig**2 * (1 + 1/k) + pi*(1-pi)*delta**2 / k)))
    return raw, peer

rows = []
sig = 0.12
for sm in (0.0, 0.05, 0.10, 0.20):
    for k in (3, 5, 10, 30):
        for pi, delta in ((0.15, 0.25), (0.3, 0.25), (0.15, 0.15)):
            er, ep = sim(sig, sm, k, pi, delta)
            fr, fp = formula(sig, sm, k, pi, delta)
            thr = (sig**2 + pi*(1-pi)*delta**2) / k
            rows.append(dict(sm=sm, k=k, pi=pi, delta=delta, raw_emp=er, raw_formula=fr, peer_emp=ep, peer_formula=fp,
                             helps_pred=bool(sm**2 > thr), helps_emp=bool(ep > er), thr=thr))
json.dump(rows, open('theory_check.json', 'w'))
import pandas as pd
df = pd.DataFrame(rows)
print('max |raw emp-formula|  =', (df.raw_emp-df.raw_formula).abs().max().round(4))
print('max |peer emp-formula| =', (df.peer_emp-df.peer_formula).abs().max().round(4), ' mean', (df.peer_emp-df.peer_formula).abs().mean().round(4))
print('threshold rule agrees with simulation in', int((df.helps_pred==df.helps_emp).sum()), 'of', len(df), 'cells')
print(df[df.helps_pred!=df.helps_emp].round(4).to_string())
