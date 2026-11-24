"""
Étape 5 — Balayage de l'ESN : le vecteur suit-il la performance ?
(capacité mémoire et NARMA-10)

Grille rho x alpha, 3 graines par point, même entrée de référence (bruit i.i.d.).
Pour chaque réservoir : le vecteur de complexité (avec la consistance), la
capacité mémoire (MC) et l'erreur sur NARMA-10 (NRMSE). Banc honnête : la lecture
est apprise sur un départ et testée sur un autre départ, avec une autre entrée.

    [W..] milestones : le code marche (OK / FAIL)
    [RES] résultats  : ce que disent les données (ne peuvent pas « échouer »)

Notebook : exécuter d'abord les cellules des étapes 1, 2a, 3 et 4.
Puis : results = sweep()        (environ 30 s)
"""

import time

import numpy as np

# Notebook : pas d'import entre fichiers (complexity_vector, ESNSubstrate,
# run_pair, normalize_vector, AXES, draw_axes viennent des cellules précédentes).

RHOS = [0.2, 0.5, 0.8, 0.95, 1.1, 1.3]
ALPHAS = [0.1, 0.3, 0.6, 1.0]
SEEDS = [0, 1, 2]
T, WARMUP, K_MAX, NOISE = 3000, 500, 80, 0.01     # NOISE : même plancher que C_N

_T0 = time.perf_counter()


def milestone(tag, ok, msg):
    status = "OK  " if ok else "FAIL"
    print(f"[{tag}] [{status}] {msg:<62} ({time.perf_counter() - _T0:5.2f} s)")
    return ok


def result(msg):
    print(f"[RES] {msg}")


def _noisy(X, noise, rng):
    """Bruit de mesure : noise x écart-type de chaque canal."""
    return X + noise * X.std(axis=0) * rng.standard_normal(X.shape)


def memory_capacity(X, u, X_test=None, u_test=None, k_max=K_MAX, noise=NOISE, seed=0):
    """MC de Jaeger : somme sur k des r² pour retrouver u(t-k) avec une lecture linéaire.

    Les états reçoivent d'abord le bruit de mesure : sans lui, la lecture exploiterait
    des directions sous le plancher, celles que C_N ignore déjà.
    Sans X_test : apprentissage sur 70 % d'une même simulation, test sur les 30 %
    restants (ancien banc). Avec X_test : apprentissage sur X, test sur une AUTRE
    simulation (autre départ, autre entrée). C'est le banc honnête : une lecture qui
    n'a appris qu'une trajectoire échoue si le système n'oublie pas son départ.
    """
    rng = np.random.default_rng(seed)
    def design(Xs, us):
        A = np.hstack([_noisy(Xs, noise, rng)[k_max:], np.ones((len(Xs) - k_max, 1))])
        Y = np.stack([us[k_max - k: len(us) - k] for k in range(1, k_max + 1)], axis=1)
        return A, Y
    A, Y = design(X, u)
    if X_test is None:
        n_tr = int(0.7 * len(A))
        A, At, Y, Yt = A[:n_tr], A[n_tr:], Y[:n_tr], Y[n_tr:]
    else:
        At, Yt = design(X_test, u_test)
    W = np.linalg.lstsq(A, Y, rcond=None)[0]                          # tous les k d'un coup
    P = At @ W
    r2 = [np.corrcoef(P[:, k], Yt[:, k])[0, 1] ** 2 for k in range(k_max)]
    return float(np.sum(r2)), np.array(r2)


def narma10(u01):
    """Cible NARMA-10 (Atiya et Parlos 2000), entrée dans [0, 0.5] :
    y(t+1) = 0.3 y(t) + 0.05 y(t) Σ_{i=0..9} y(t-i) + 1.5 u(t-9) u(t) + 0.1
    Il faut à la fois de la mémoire (10 pas) et de la non-linéarité (produits)."""
    y = np.zeros(len(u01))
    for t in range(9, len(u01) - 1):
        y[t + 1] = (0.3 * y[t] + 0.05 * y[t] * y[t - 9: t + 1].sum()
                    + 1.5 * u01[t - 9] * u01[t] + 0.1)
    return y


def narma_nrmse(X, u, X_test=None, u_test=None, noise=NOISE, seed=0, skip=100):
    """NRMSE de la lecture linéaire sur NARMA-10 (0 = parfait, 1 = pas mieux qu'une constante).

    Plafonnée à 1 : au-delà, c'est un échec, et la valeur ne mesure plus que la force
    avec laquelle la lecture extrapole hors de sa zone d'apprentissage (le Coulomb
    donnait 28 à 141). Une régularisation ridge n'y change rien (testé : 52 à 141).

    Le réservoir reçoit u dans [-1, 1] ; la cible est calculée sur (u + 1) / 4, dans
    [0, 0.5] : même information, donc MC et NARMA se mesurent sur la même simulation.
    L'état X[t] a vu u(t) : il doit prédire y(t+1). X_test : comme memory_capacity.
    """
    rng = np.random.default_rng(seed)
    def design(Xs, us):
        y = narma10((us + 1) / 4)
        A = np.hstack([_noisy(Xs, noise, rng)[skip:-1], np.ones((len(Xs) - skip - 1, 1))])
        return A, y[skip + 1:]
    A, y = design(X, u)
    if X_test is None:
        n_tr = int(0.7 * len(A))
        A, At, y, yt = A[:n_tr], A[n_tr:], y[:n_tr], y[n_tr:]
    else:
        At, yt = design(X_test, u_test)
    w = np.linalg.lstsq(A, y, rcond=None)[0]
    err = At @ w - yt
    return float(min(np.sqrt(np.mean(err ** 2) / np.var(yt)), 1.0))


def evaluate(sub, seed=0, n_starts=3):
    """Le banc complet d'un substrat :
      A  départ standard, entrée de référence          -> vecteur, apprentissage
      B  A décalé de 1e-6, même entrée                 -> lambda
      C  n_starts autres départs, même entrée          -> consistance (moyenne)
      D  encore un autre départ, AUTRE entrée          -> test de la lecture
    Plusieurs départs pour la consistance : un seul peut tomber par chance dans la
    même configuration qu'A et cacher une multistabilité (vu sur un réseau de ressorts).
    Renvoie (vecteur brut, MC, NRMSE NARMA-10)."""
    u = np.random.default_rng(42).uniform(-1, 1, WARMUP + T)           # entrée de référence
    u_test = np.random.default_rng(7).uniform(-1, 1, WARMUP + T)       # entrée de test
    rng = np.random.default_rng(1000 + seed)
    Xa, Xb = run_pair(sub, u, warmup=WARMUP)
    C = complexity_vector(Xa, Xb, D=4, warn=False)
    C["consist"] = float(np.mean([consistency(Xa, run(sub, u, sub.random_state(rng))[0][WARMUP:])
                                  for _ in range(n_starts)]))
    Xd, _ = run(sub, u_test, sub.random_state(rng))
    mc = memory_capacity(Xa, u[WARMUP:], Xd[WARMUP:], u_test[WARMUP:])[0]
    nr = narma_nrmse(Xa, u[WARMUP:], Xd[WARMUP:], u_test[WARMUP:])
    return C, mc, nr


def _ranks(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return np.corrcoef(_ranks(x[ok]), _ranks(y[ok]))[0, 1]


def loco_knn_r2(F, y, groups, k=3):
    """R² de prédiction quand on retire TOUTES les graines d'un réglage à la fois :
    on prédit chaque réglage à partir des autres réglages seulement (k plus proches voisins)."""
    F = (F - np.nanmean(F, 0)) / (np.nanstd(F, 0) + 1e-12)
    F = np.nan_to_num(F)
    pred = np.empty_like(y)
    for g in np.unique(groups):
        test, train = groups == g, groups != g
        for i in np.where(test)[0]:
            d = np.linalg.norm(F[train] - F[i], axis=1)
            pred[i] = y[train][np.argsort(d)[:k]].mean()
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def analyze(name, perf, V, HP, groups, higher_is_better=True):
    """Corrélation de chaque axe avec la performance, puis prédiction d'un réglage jamais vu."""
    result(f"{name} : corrélation de rang (Spearman) entre chaque axe et la performance")
    sign = 1 if higher_is_better else -1                  # signe : + veut dire « aide »
    for j, (key, _, _) in enumerate(AXES):
        print(f"      {key:<9} {sign * spearman(V[:, j], perf):+.2f}")
    # Le résultat ne doit pas dépendre du nombre de voisins k : on en montre plusieurs.
    result(f"{name} d'un réglage jamais vu, R² selon le nombre de voisins k :")
    out = []
    for k in (1, 3, 5, 8):
        r2_vec, r2_hp = loco_knn_r2(V, perf, groups, k), loco_knn_r2(HP, perf, groups, k)
        out.append((r2_vec, r2_hp))
        print(f"      k={k}   vecteur {r2_vec:5.2f}   (rho, alpha) {r2_hp:5.2f}")
    return out


def sweep(show=True):
    print("=" * 84)
    print("Étape 5 — balayage de l'ESN : vecteur, capacité mémoire et NARMA-10")
    print("=" * 84)
    u = np.random.default_rng(42).uniform(-1, 1, WARMUP + T)            # entrée de référence
    y_check = np.concatenate([narma10((u + 1) / 4),
                              narma10((np.random.default_rng(7).uniform(-1, 1, WARMUP + T) + 1) / 4)])
    milestone("W0", np.all(np.isfinite(y_check)) and y_check.max() < 2,
              f"cible NARMA-10 stable (max {y_check.max():.2f})")
    rows = []                                    # (rho, alpha, seed, vecteur brut, MC, NRMSE)
    for rho in RHOS:
        for alpha in ALPHAS:
            for seed in SEEDS:
                C, mc, nr = evaluate(ESNSubstrate(rho=rho, alpha=alpha, seed=seed), seed)
                rows.append((rho, alpha, seed, C, mc, nr))
    n_cfg = len(RHOS) * len(ALPHAS)
    V = np.array([normalize_vector(r[3]) for r in rows])
    MC = np.array([r[4] for r in rows])
    NR = np.array([r[5] for r in rows])
    HP = np.array([[r[0], r[1]] for r in rows])
    groups = np.repeat(np.arange(n_cfg), len(SEEDS))
    milestone("W1", len(rows) == n_cfg * len(SEEDS) and np.all(np.isfinite(MC)),
              f"{len(rows)} réservoirs simulés ({n_cfg} réglages x {len(SEEDS)} graines)")
    N = 50                                       # canaux lus par l'ESN
    milestone("W2", np.all(MC <= N + 1e-9) and MC.max() > 2 * MC.min(),
              f"MC entre {MC.min():.1f} et {MC.max():.1f}, toujours <= N = {N}")
    milestone("W3", np.all(np.isfinite(NR)) and NR.min() > 0 and NR.max() < 1.5,
              f"NARMA-10 : NRMSE entre {NR.min():.2f} et {NR.max():.2f} (0 = parfait)")

    # Question 1 : l'écart entre graines est-il petit devant l'effet du réglage ?
    Vg = V.reshape(n_cfg, len(SEEDS), -1)
    within = np.nanmean(np.nanstd(Vg, axis=1), axis=0)
    between = np.nanstd(np.nanmean(Vg, axis=1), axis=0)
    result("signal/bruit par axe = écart entre réglages / écart entre graines")
    for (key, _, _), w, b in zip(AXES, within, between):
        print(f"      {key:<9} entre réglages {b:.3f}   entre graines {w:.3f}   rapport {b / w:6.1f}")
    for name, P in (("MC", MC), ("NRMSE", NR)):
        g = P.reshape(n_cfg, len(SEEDS))
        result(f"{name} : entre réglages {g.mean(1).std():.3f}, entre graines {g.std(1).mean():.3f}")

    # Question 2 : quels axes suivent chaque tâche, et le vecteur prédit-il ?
    analyze("MC", MC, V, HP, groups)
    analyze("NARMA-10", NR, V, HP, groups, higher_is_better=False)

    # Question 3 : les deux tâches veulent-elles la même dynamique ?
    mcg, nrg = MC.reshape(n_cfg, -1).mean(1), NR.reshape(n_cfg, -1).mean(1)
    cfg = HP[:: len(SEEDS)]
    b_mc, b_nr = cfg[np.argmax(mcg)], cfg[np.argmin(nrg)]
    result(f"meilleur réglage MC : rho={b_mc[0]}, alpha={b_mc[1]}   "
           f"meilleur NARMA : rho={b_nr[0]}, alpha={b_nr[1]}")
    result(f"accord entre les deux tâches (Spearman MC / -NRMSE par réglage) : "
           f"{spearman(mcg, -nrg):+.2f}")

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.05, 1.05])
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    for ax, data, title, cmap in ((ax1, mcg, "MC (plus = mieux)", "viridis"),
                                  (ax2, nrg, "NARMA-10 NRMSE (moins = mieux)", "viridis_r")):
        im = ax.imshow(data.reshape(len(RHOS), len(ALPHAS)), origin="lower", aspect="auto", cmap=cmap)
        ax.set(xticks=range(len(ALPHAS)), xticklabels=ALPHAS, yticks=range(len(RHOS)),
               yticklabels=RHOS, xlabel="alpha", ylabel="rho", title=title)
        fig.colorbar(im, ax=ax)
    sc = ax3.scatter(MC, NR, c=HP[:, 1], cmap="plasma", s=18)
    ax3.set(xlabel="MC", ylabel="NRMSE NARMA-10", title="Les deux tâches (couleur = alpha)")
    fig.colorbar(sc, ax=ax3, label="alpha")

    for row, perf, label, cmap in ((1, MC, "MC", "viridis"), (2, NR, "NRMSE NARMA-10", "viridis_r")):
        axP = fig.add_subplot(gs[row, :])
        draw_axes(axP)
        norm = plt.Normalize(perf.min(), perf.max())
        order = np.argsort(perf if cmap == "viridis" else -perf)        # meilleurs au-dessus
        for i in order:
            axP.plot(range(len(AXES)), V[i], lw=1.2, alpha=0.6, color=plt.get_cmap(cmap)(norm(perf[i])))
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axP, label=label)
        axP.set_title(f"Les 72 réservoirs dans l'espace des vecteurs, colorés par {label} (clair = meilleur)")
    fig.tight_layout()
    fig.savefig("sweep.png", dpi=110)
    milestone("W4", True, "figure enregistrée : sweep.png")
    print("=" * 84)
    if show:
        plt.show()
    return {"rows": rows, "V": V, "MC": MC, "NR": NR, "HP": HP}


if __name__ == "__main__":
    results = sweep()
