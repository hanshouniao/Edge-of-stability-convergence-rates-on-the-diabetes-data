import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

L, SEEDS, T, T_CRIT, EVERY, SIGMA = 5, 20, 200_000, 2_000_000, 10, 0.2
CS = [1.975, 2.000, 2.025]
BATCHES = [32, 64, 128, 221]
TOL = 0.015

d = load_diabetes(scaled=False)
x, y = d.data[:, 2], d.target
x, y = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
n = len(x)
r = np.mean(x * y)
ws = r ** (1 / L)
lam = L * r ** (2 - 2 / L)


def init():
    return np.stack([ws * np.exp(SIGMA * np.random.default_rng(s).standard_normal(L)) for s in range(SEEDS)])


def dist(W):
    return np.linalg.norm(np.abs(W) - ws, axis=-1)


def step(W, eta, a=1.0, b=r):
    p = W.prod(-1)
    return W - eta * ((a * p - b) * p)[..., None] / W


def gd(c, steps, record):
    W, eta, out = init(), c / lam, {}
    for t in range(1, steps + 1):
        W = step(W, eta)
        if t in record:
            out[t] = W.copy()
    return np.array(sorted(out)), np.array([out[k] for k in sorted(out)]), W


def sgd(c, B):
    W, eta, nb = init(), c / lam, n // B
    rngs = [np.random.default_rng(1000 + s) for s in range(SEEDS)]
    dead = np.zeros(SEEDS, bool)
    ds, t = [], 0
    while t < T:
        idx = np.stack([g.permutation(n)[: nb * B].reshape(nb, B) for g in rngs])
        A, Bm = (x[idx] ** 2).mean(2), (x[idx] * y[idx]).mean(2)
        for k in range(nb):
            if t == T:
                break
            Wn = step(W, eta, A[:, k], Bm[:, k])
            dead |= ~np.isfinite(Wn).all(1) | (np.abs(Wn).max(1) > 1e3)
            W = np.where(dead[:, None], np.nan, Wn)
            t += 1
            if t % EVERY == 0:
                ds.append(dist(W))
    stalled = ~dead & (np.abs(W.prod(1)) < 0.1) & (np.abs(W).max(1) < 0.3)
    near = ~dead & ~stalled
    ds = np.array(ds)
    return ds, dict(diverged=int(dead.sum()), stalled_at_origin=int(stalled.sum()), near_solution_set=int(near.sum()),
                    final_distance=summ(ds[-len(ds) // 10:, near].mean(0)) if near.sum() > 1 else None)


def summ(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), se=float(v.std(ddof=1) / np.sqrt(len(v))))


def boot(v, k=10_000):
    m = np.random.default_rng(0).choice(v, (k, len(v))).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def rate(tt, e, lo, hi):
    m = (tt >= lo) & (tt <= hi) & (e > 1e-11)
    return -np.polyfit(tt[m], np.log(e[m]), 1)[0]


res = {"n": n, "r": float(r), "w_star": float(ws), "lambda_star": float(lam), "tolerance": TOL}
ps = np.linspace(-2, 2, 401)
res["reduction_max_error"] = float(max(abs(0.5 * np.mean((y - p * x) ** 2) - 0.5 * (p - r) ** 2 - 0.5 * (1 - r ** 2)) for p in ps))
Wa = Wb = init()[0]
gap = 0.0
for _ in range(5000):
    Wa = Wa + (2 / lam) * np.mean((y - Wa.prod() * x) * x) * Wa.prod() / Wa
    Wb = step(Wb, 2 / lam)
    gap = max(gap, np.abs(Wa - Wb).max())
res["real_vs_scalar_max_gap"] = float(gap)

grid = set(range(EVERY, T + 1, EVERY))
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

c = 1.975
tt, P, W = gd(c, T, grid)
D = dist(P)
S_inf = ((W.prod(1)[:, None] / W) ** 2).sum(1)
pred = -np.log(np.abs(1 - (c / lam) * S_inf))
fit = []
for s in range(SEEDS):
    e = np.abs(D[:, s] - D[-1, s])
    end = tt[e > 1e-11][-1]
    fit.append(rate(tt, e, end // 2, end))
rel = np.abs(np.array(fit) - pred) / pred
res["subcritical"] = dict(rel_error=summ(rel), rel_error_median=float(np.median(rel)), rel_error_boot95=boot(rel),
                          seeds_within_tol=int((rel <= TOL).sum()), agrees=bool(rel.mean() <= TOL),
                          sharpness_gap=summ(S_inf - lam), gap_bound=float(2 * lam / c - lam),
                          all_below_bound=bool((S_inf <= 2 * lam / c).all()))
ax[0, 0].plot(tt, D, color="0.75", lw=0.7)
ax[0, 0].plot(tt, np.median(D, 1), color="C0", lw=2)
ax[0, 0].set(xscale="log", yscale="log", title="subcritical 1.975", ylabel="distance to flattest solution")
for s in range(SEEDS):
    e = np.abs(D[:, s] - D[-1, s])
    end = tt[e > 1e-11][-1]
    i = np.searchsorted(tt, end // 2)
    ax[1, 0].plot(pred[s] * (tt[i:] - tt[i]), e[i:] / e[i], color="C1" if rel[s] > TOL else "C0", lw=0.8)
z = np.linspace(0, 15, 50)
ax[1, 0].plot(z, np.exp(-z), "k--", lw=2)
ax[1, 0].set(yscale="log", xlim=(0, 15), ylim=(1e-7, 2), xlabel="(t - window start) x predicted rate",
             title=f"final rate: mean error {100 * rel.mean():.2f}%")

c = 2.0
rec = set(np.unique(np.logspace(3, np.log10(T_CRIT), 200).astype(int)))
tl, P, _ = gd(c, T_CRIT, rec)
D = dist(P)
m = tl >= 2e5
ex = np.array([np.polyfit(np.log(tl[m]), np.log(D[m, s]), 1)[0] for s in range(SEEDS)])
res["critical"] = dict(exponent=summ(ex), prediction=-0.5, rel_error=float(abs(ex.mean() + 0.5) / 0.5),
                       agrees=bool(abs(ex.mean() + 0.5) / 0.5 <= TOL), window=[2e5, 2e6])
med = np.median(D, 1)
C = np.exp(np.mean(np.log(med[m]) + 0.5 * np.log(tl[m])))
ax[0, 1].plot(tl, D, color="0.75", lw=0.7)
ax[0, 1].plot(tl, med, color="C0", lw=2)
ax[0, 1].plot(tl, C / np.sqrt(tl), "k--", lw=2)
ax[0, 1].set(xscale="log", yscale="log", title="critical 2.000")
lt = np.log(tl)
cen, sl = [], []
for k in range(len(tl)):
    w = np.abs(lt - lt[k]) <= 0.575
    if lt[k] - 0.575 >= lt[0] and lt[k] + 0.575 <= lt[-1]:
        cen.append(tl[k])
        sl.append(np.polyfit(lt[w], np.log(med[w]), 1)[0])
ax[1, 1].fill_between(cen, -0.5 * (1 + TOL), -0.5 * (1 - TOL), color="C1", alpha=0.2)
ax[1, 1].plot(cen, sl, color="C0", lw=2)
ax[1, 1].axhline(-0.5, color="k", ls="--")
ax[1, 1].set(xscale="log", ylim=(-0.56, -0.44), xlabel="iteration", ylabel="local exponent",
             title=f"exponent on [2e5, 2e6]: {ex.mean():.4f}")

c = 2.025
eps = c - 2
A_pred, rho_pred = np.sqrt(L) * ws * np.sqrt(eps / 112), eps / 28
tt, P, W = gd(c, T, grid)
D = dist(P)
half = 0.5 * (dist(W) + dist(step(W, c / lam)))
imb = np.linalg.norm(np.abs(P) - np.abs(P).mean(-1, keepdims=True), axis=-1)
rr = np.array([rate(tt, imb[:, s], 2000, T) for s in range(SEEDS)])
res["supercritical"] = dict(amplitude=summ(half), amplitude_prediction=float(A_pred),
                            amplitude_rel_error=float(abs(half.mean() - A_pred) / A_pred),
                            rate=summ(rr), rate_prediction=rho_pred, rate_rel_error=float(abs(rr.mean() - rho_pred) / rho_pred),
                            agrees=bool(abs(half.mean() - A_pred) / A_pred <= TOL and abs(rr.mean() - rho_pred) / rho_pred <= TOL))
ax[0, 2].plot(tt, D, color="0.75", lw=0.7)
ax[0, 2].axhline(A_pred, color="k", ls="--", lw=2)
ax[0, 2].set(xscale="log", yscale="log", title="supercritical 2.025")
ax[1, 2].plot(tt, imb, color="0.75", lw=0.7)
i0, i1 = np.searchsorted(tt, 2000), np.searchsorted(tt, 30000)
mi = np.median(imb, 1)
ax[1, 2].plot(tt[i0:i1], mi[i0] * np.exp(-rho_pred * (tt[i0:i1] - tt[i0])), "k--", lw=2)
ax[1, 2].set(yscale="log", xlim=(0, 45000), ylim=(1e-15, 3), xlabel="iteration", ylabel="distance to balanced line",
             title=f"rate {rr.mean():.4e} vs {rho_pred:.4e}")
fig.tight_layout()
fig.savefig("fig_fullbatch.png", dpi=150)

res["sgd"] = {}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
for j, c in enumerate(CS):
    res["sgd"][str(c)] = {}
    for B in BATCHES:
        ds, out = sgd(c, B)
        res["sgd"][str(c)][str(B)] = out
        if B == 32:
            ax[j].plot(np.arange(1, len(ds) + 1) * EVERY, ds, color="C1", lw=0.7)
    ax[j].axhline(np.sqrt(L) * ws, color="k", ls=":")
    ax[j].set(xscale="log", yscale="log", ylim=(1e-2, 20), xlabel="iteration", title=f"batch 32, {c}")
ax[0].set_ylabel("distance to flattest solution")
fig.tight_layout()
fig.savefig("fig_batch32.png", dpi=150)

json.dump(res, open("results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
