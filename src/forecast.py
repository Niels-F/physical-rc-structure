"""
Étape 11 — Le banc de prévision, et le test de solubilité linéaire.

Tâche : à l'instant t, le réservoir a reçu la série s jusqu'à s(t) ; il doit
prédire s(t + h), h pas plus loin.

Deux séries chaotiques connues :
  Mackey-Glass (tau = 17), échantillonnée toutes les 1 unité de temps
  Lorenz (composante x), échantillonnée toutes les 0.05 unité de temps

Règle du banc : on ne garde que les tâches (série, horizon h) qu'un modèle
LINÉAIRE ne sait pas résoudre, mais qu'un modèle NON LINÉAIRE simple sait
améliorer. Sinon, un réservoir qui réussit ne prouverait rien sur la non-linéarité
(leçon d'ENSO). Les références :
  persistance   s(t+h) = s(t)                    (ne rien savoir)
  AR linéaire   combinaison des 50 dernières valeurs (même nombre de canaux que le réservoir)
  NG-RC         8 dernières valeurs + tous leurs produits deux à deux (44 canaux)

Même honnêteté que le banc précédent : apprentissage sur un morceau de la
série, test sur un AUTRE morceau, depuis un autre départ ; bruit de mesure de 1 %.

Notebook : exécuter d'abord les cellules des étapes 1 à 10.
La cellule exécute les milestones F1 à F3, puis : fc = forecast_bench()
"""

import time

import numpy as np

# Notebook : pas d'import entre fichiers.

HORIZONS = [1, 5, 10, 20]
AR_ORDER, NG_DELAYS = 50, 8
SEG_TRAIN, SEG_TEST = (0, 3500), (4500, 8000)          # deux morceaux disjoints
AR_FAIL, NL_GAIN = 0.2, 0.7      # gardée si AR > 0.2 ET NG-RC < 0.7 x AR (fixés avant tout réservoir)

_T0 = time.perf_counter()


def milestone(tag, ok, msg):
    status = "OK  " if ok else "FAIL"
    print(f"[{tag}] [{status}] {msg:<62} ({time.perf_counter() - _T0:5.2f} s)")
    return ok


def result(msg):
    print(f"[RES] {msg}")


# ------------------------------------------------------------------ les séries
def mackey_glass(n=8000, tau=17.0, dt=0.05, sample=1.0, discard=1000, seed=0):
    """dx/dt = 0.2 x(t-tau) / (1 + x(t-tau)^10) - 0.1 x   (Euler, pas dt)."""
    rng = np.random.default_rng(seed)
    d, k = int(round(tau / dt)), int(round(sample / dt))
    total = (n + discard) * k + d
    x = np.empty(total)
    x[: d + 1] = 1.2 + 0.01 * rng.standard_normal(d + 1)          # histoire initiale
    for t in range(d, total - 1):
        xd = x[t - d]
        x[t + 1] = x[t] + dt * (0.2 * xd / (1 + xd ** 10) - 0.1 * x[t])
    return x[d::k][discard: discard + n]


def lorenz(n=8000, dt=0.01, sample=0.05, discard=1000, x0=(1.0, 1.0, 1.0), full=False):
    """Lorenz (sigma=10, rho=28, beta=8/3), Runge-Kutta 4. Renvoie la composante x."""
    def f(v):
        x, y, z = v
        return np.array([10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z])
    k = int(round(sample / dt))
    v, out = np.array(x0, float), []
    for i in range((n + discard) * k):
        k1 = f(v); k2 = f(v + dt / 2 * k1); k3 = f(v + dt / 2 * k2); k4 = f(v + dt * k3)
        v = v + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i % k == k - 1:
            out.append(v.copy())
    out = np.array(out)[discard:]
    return out if full else out[:, 0]


def normalize(s):
    """Ramène la série dans [-1, 1] avec les bornes du morceau d'apprentissage."""
    a, b = s[SEG_TRAIN[0]: SEG_TRAIN[1]].min(), s[SEG_TRAIN[0]: SEG_TRAIN[1]].max()
    return 2 * (s - a) / (b - a) - 1


# ------------------------------------------------------------- les références
def _nrmse(p, y):
    return float(min(np.sqrt(np.mean((p - y) ** 2) / np.var(y)), 1.0))


def _fit_predict(F_tr, y_tr, F_te, noise=NOISE, seed=0):
    rng = np.random.default_rng(seed)
    A = np.hstack([_noisy(F_tr, noise, rng), np.ones((len(F_tr), 1))])
    B = np.hstack([_noisy(F_te, noise, rng), np.ones((len(F_te), 1))])
    w = np.linalg.lstsq(A, y_tr, rcond=None)[0]
    return B @ w


def _delays(s, start, stop, p):
    """Lignes t = start..stop-1 : [s(t), s(t-1), ..., s(t-p+1)]."""
    return np.stack([s[start - j: stop - j] for j in range(p)], axis=1)


def baselines(s, h, warm=WARMUP):
    """NRMSE de la persistance, de l'AR linéaire et du NG-RC quadratique."""
    out = {}
    (a0, a1), (b0, b1) = SEG_TRAIN, SEG_TEST
    t_tr, t_te = (a0 + warm, a1 - h), (b0 + warm, b1 - h)
    y_tr, y_te = s[t_tr[0] + h: t_tr[1] + h], s[t_te[0] + h: t_te[1] + h]
    out["persistance"] = _nrmse(s[t_te[0]: t_te[1]], y_te)
    out["AR linéaire"] = _nrmse(_fit_predict(_delays(s, *t_tr, AR_ORDER), y_tr,
                                             _delays(s, *t_te, AR_ORDER)), y_te)
    def ng(t0, t1):
        L = _delays(s, t0, t1, NG_DELAYS)
        iu = np.triu_indices(NG_DELAYS)
        return np.hstack([L, (L[:, :, None] * L[:, None, :])[:, iu[0], iu[1]]])
    out["NG-RC"] = _nrmse(_fit_predict(ng(*t_tr), y_tr, ng(*t_te)), y_te)
    return out


def forecast_nrmse(sub, s, h, seed=0, warm=WARMUP):
    """Le réservoir piloté par la série : appris sur un morceau depuis le départ
    standard, testé sur un autre morceau depuis un autre départ."""
    (a0, a1), (b0, b1) = SEG_TRAIN, SEG_TEST
    X_tr, _ = run(sub, s[a0:a1])
    X_te, _ = run(sub, s[b0:b1], sub.random_state(np.random.default_rng(2000 + seed)))
    F_tr, y_tr = X_tr[warm: len(X_tr) - h], s[a0 + warm + h: a1]
    F_te, y_te = X_te[warm: len(X_te) - h], s[b0 + warm + h: b1]
    return _nrmse(_fit_predict(F_tr, y_tr, F_te, seed=seed), y_te)


# ================================================================= milestones
def self_test():
    print("=" * 84)
    print("Étape 11 — validation des séries chaotiques")
    print("=" * 84)
    mg = mackey_glass()
    ok = 0.2 < mg.min() and mg.max() < 1.5 and np.std(mg) > 0.1
    milestone("F1", ok, f"Mackey-Glass : valeurs entre {mg.min():.2f} et {mg.max():.2f} (attendu ~0.4-1.3)")

    # F2 : l'exposant de Lyapunov de Lorenz est connu, ~0.906 par unité de temps.
    # Méthode de Benettin : on renormalise l'écart à chaque pas et on moyenne sa croissance
    # sur une longue trajectoire. (Une seule pente sur une trajectoire courte le sous-estime :
    # 0.32 sur 12 unités de temps, car l'étirement de Lorenz est très inégal sur l'attracteur.
    # Limite à garder en tête pour notre conditional_lyapunov, qui mesure une seule pente.)
    def f(v):
        x, y, z = v
        return np.array([10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z])
    def rk4(v, dt=0.01):
        k1 = f(v); k2 = f(v + dt / 2 * k1); k3 = f(v + dt / 2 * k2); k4 = f(v + dt * k3)
        return v + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    v = np.array([1.0, 1.0, 1.0])
    for _ in range(2000):
        v = rk4(v)
    w, acc, n_steps = v + np.array([1e-8, 0, 0]), 0.0, 20000
    for _ in range(n_steps):
        v, w = rk4(v), rk4(w)
        d = np.linalg.norm(w - v)
        acc += np.log(d / 1e-8)
        w = v + (w - v) * 1e-8 / d
    lam = acc / (n_steps * 0.01)
    milestone("F2", abs(lam - 0.906) < 0.1, f"Lorenz : lambda = {lam:.3f} par unité de temps (théorie 0.906)")

    # F3 : les deux morceaux de série sont bien disjoints et de même nature
    s = normalize(mg)
    same = abs(s[SEG_TRAIN[0]:SEG_TRAIN[1]].std() - s[SEG_TEST[0]:SEG_TEST[1]].std()) < 0.1
    milestone("F3", SEG_TRAIN[1] < SEG_TEST[0] and same,
              "apprentissage et test : morceaux disjoints, même statistique")
    print("=" * 84)


def forecast_bench(show=True):
    print("=" * 84)
    print("Le banc de prévision : quelles tâches sont vraiment non linéaires ?")
    print("=" * 84)
    series = {"Mackey-Glass": normalize(mackey_glass()), "Lorenz x": normalize(lorenz())}
    table, kept = {}, []
    for name, s in series.items():
        print(f"   {name}")
        print(f"      {'h':>3} {'persistance':>12} {'AR linéaire':>12} {'NG-RC':>8} {'ESN':>8}   gardée ?")
        for h in HORIZONS:
            b = baselines(s, h)
            esn = np.mean([forecast_nrmse(ESNSubstrate(seed=g), s, h, g) for g in range(3)])
            keep = b["AR linéaire"] > AR_FAIL and b["NG-RC"] < NL_GAIN * b["AR linéaire"]
            table[(name, h)] = b | {"ESN": esn, "gardée": keep}
            if keep:
                kept.append((name, h))
            print(f"      {h:>3} {b['persistance']:12.3f} {b['AR linéaire']:12.3f} "
                  f"{b['NG-RC']:8.3f} {esn:8.3f}   {'OUI' if keep else 'non'}")
    milestone("F4", len(kept) > 0 and all(np.isfinite(v["ESN"]) for v in table.values()),
              f"{len(kept)} tâches gardées sur {len(table)}")
    result("tâches retenues : " + ", ".join(f"{n} h={h}" for n, h in kept))

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, name in zip(axs, series):
        for key, style in (("persistance", ":"), ("AR linéaire", "--"), ("NG-RC", "-."), ("ESN", "-")):
            ax.plot(HORIZONS, [table[(name, h)][key] for h in HORIZONS], style, marker="o", label=key)
        for h in HORIZONS:
            if table[(name, h)]["gardée"]:
                ax.axvspan(h - 0.6, h + 0.6, color="tab:green", alpha=0.12)
        ax.axhline(AR_FAIL, color="0.6", lw=0.8)
        ax.set(xlabel="horizon h (pas)", ylabel="NRMSE (plafonnée à 1)", title=f"{name} (vert = tâche gardée)",
               xticks=HORIZONS, ylim=(0, 1.05))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("forecast_bench.png", dpi=110)
    milestone("F5", True, "figure enregistrée : forecast_bench.png")
    print("=" * 84)
    if show:
        plt.show()
    return {"series": series, "table": table, "kept": kept}


if __name__ == "__main__":
    self_test()
