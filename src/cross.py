"""
Étape 7 — Le test décisif : le vecteur prédit-il un substrat qu'on n'a jamais réglé ?

1. Balayage du Coulomb : frottement gamma x champ E0 x 3 graines (environ 6 min).
   On lit seulement les positions : 50 canaux, comme les 50 nœuds de l'ESN.
2. On apprend « vecteur -> performance » sur les 72 ESN de l'étape 5, puis on
   prédit les 72 Coulomb, et inversement. Les hyperparamètres ne peuvent pas
   faire ce test : rho et gamma n'ont rien en commun.

    [X..] milestones : le code marche (OK / FAIL)
    [RES] résultats  : ce que disent les données

Notebook : exécuter d'abord les cellules des étapes 1 à 6.
Puis : cross = cross_test()          (refait aussi le balayage ESN, ~10 s)
"""

import time

import numpy as np

# Notebook : pas d'import entre fichiers (CoulombSubstrate, run_pair, sweep,
# complexity_vector, normalize_vector, memory_capacity, narma_nrmse, AXES,
# T, WARMUP, SEEDS, spearman viennent des cellules précédentes).

GAMMAS = [0.05, 0.2, 0.5, 1.0, 2.0, 4.0]
E0S = [0.25, 0.5, 1.0, 2.0]

_T0 = time.perf_counter()


def milestone(tag, ok, msg):
    status = "OK  " if ok else "FAIL"
    print(f"[{tag}] [{status}] {msg:<62} ({time.perf_counter() - _T0:5.2f} s)")
    return ok


def result(msg):
    print(f"[RES] {msg}")


class CoulombPositions(CoulombSubstrate):
    """Même physique, mais on ne lit que les positions (2N = 50 canaux)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.n_obs = 2 * self.n

    def observe(self, s):
        return s[: 2 * self.n]


def coulomb_sweep():
    rows = []
    for gamma in GAMMAS:
        for E0 in E0S:
            for seed in SEEDS:
                C, mc, nr = evaluate(CoulombPositions(damping=gamma, E0=E0, seed=seed), seed)
                rows.append((gamma, E0, seed, C, mc, nr))
        print(f"   gamma={gamma:<5} fait")
    return {"rows": rows, "V": np.array([normalize_vector(r[3]) for r in rows]),
            "MC": np.array([r[4] for r in rows]), "NR": np.array([r[5] for r in rows]),
            "HP": np.array([[r[0], r[1]] for r in rows])}


def _standardize(F_train, F_test):
    mu, sd = F_train.mean(0), F_train.std(0) + 1e-9
    return (F_train - mu) / sd, (F_test - mu) / sd


def knn_predict(F_train, y_train, F_test, k=5):
    A, B = _standardize(F_train, F_test)
    d = np.linalg.norm(B[:, None, :] - A[None, :, :], axis=2)
    return y_train[np.argsort(d, axis=1)[:, :k]].mean(axis=1)


def ridge_predict(F_train, y_train, F_test, alpha=1.0):
    A, B = _standardize(F_train, F_test)
    A1, B1 = np.hstack([A, np.ones((len(A), 1))]), np.hstack([B, np.ones((len(B), 1))])
    reg = alpha * np.eye(A1.shape[1]); reg[-1, -1] = 0
    w = np.linalg.solve(A1.T @ A1 + reg, A1.T @ y_train)
    return B1 @ w


def r2(y, p):
    """R² mesuré par rapport à la moyenne du substrat PRÉDIT : 0 = pas mieux que deviner
    sa moyenne (qu'on ne connaît pas), négatif = pire. Test sévère."""
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def in_domain(F_train, F_test):
    """Un point est « dans le domaine » s'il est aussi près des points d'entraînement
    que ceux-ci le sont entre eux (95e centile des distances au plus proche voisin)."""
    A, B = _standardize(F_train, F_test)
    dAA = np.linalg.norm(A[:, None] - A[None], axis=2); np.fill_diagonal(dAA, np.inf)
    seuil = np.percentile(dAA.min(1), 95)
    return np.linalg.norm(B[:, None] - A[None], axis=2).min(1) <= seuil


def transfer(name_a, A, name_b, B):
    """Apprendre sur A, prédire B, pour MC et NARMA, avec plusieurs prédicteurs."""
    inside = in_domain(A["V"], B["V"])
    result(f"{name_a} -> {name_b} : {inside.sum()}/{len(inside)} points de {name_b} "
           f"dans le domaine couvert par {name_a}")
    lam = [j for j, (k, _, _) in enumerate(AXES) if k == "lam"]
    out = {}
    for task, key in (("MC", "MC"), ("NARMA", "NR")):
        ya, yb = A[key], B[key]
        preds = {"vecteur, k-NN": knn_predict(A["V"], ya, B["V"]),
                 "vecteur, linéaire": ridge_predict(A["V"], ya, B["V"]),
                 "lambda seul, k-NN": knn_predict(A["V"][:, lam], ya, B["V"][:, lam]),
                 "moyenne de A": np.full_like(yb, ya.mean())}
        print(f"      {task:<6} R² sur tout {name_b} | sur la partie dans le domaine")
        for pname, p in preds.items():
            inner = r2(yb[inside], p[inside]) if inside.sum() > 3 else np.nan
            print(f"        {pname:<20} {r2(yb, p):6.2f} | {inner:6.2f}")
        out[task] = (preds, inside)
    return out


def cross_test(esn=None, coul=None, show=True):
    print("=" * 84)
    print("Étape 7 — ESN et Coulomb : prédire un substrat jamais réglé")
    print("=" * 84)
    if esn is None:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):                # étape 5, sans son texte
            esn = sweep(show=False)
    milestone("X1", len(esn["MC"]) == 72, "balayage ESN repris de l'étape 5 (72 réservoirs)")
    if coul is None:
        coul = coulomb_sweep()
    ok = np.all(np.isfinite(coul["MC"])) and np.all(np.isfinite(coul["NR"]))
    milestone("X2", ok and np.all(np.isfinite(coul["V"])),
              f"balayage Coulomb : {len(coul['MC'])} réservoirs, tout est fini")

    lam_c = np.array([r[3]["lam"] for r in coul["rows"]])
    result(f"Coulomb : {np.sum(lam_c > 0)}/{len(lam_c)} réservoirs chaotiques (lambda > 0)")
    result(f"Coulomb : MC de {coul['MC'].min():.1f} à {coul['MC'].max():.1f}, "
           f"NRMSE de {coul['NR'].min():.2f} à {coul['NR'].max():.2f}")
    result(f"ESN     : MC de {esn['MC'].min():.1f} à {esn['MC'].max():.1f}, "
           f"NRMSE de {esn['NR'].min():.2f} à {esn['NR'].max():.2f}")

    t_ec = transfer("ESN", esn, "Coulomb", coul)
    t_ce = transfer("Coulomb", coul, "ESN", esn)
    milestone("X3", True, "transferts calculés dans les deux sens")

    import matplotlib.pyplot as plt
    V = np.vstack([esn["V"], coul["V"]])
    Vc = V - V.mean(0)
    _, _, Wt = np.linalg.svd(Vc, full_matrices=False)                   # ACP commune
    P = Vc @ Wt[:2].T
    n_e = len(esn["V"])
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    for ax, key, label, cmap in ((axs[0, 0], "MC", "MC", "viridis"),
                                 (axs[0, 1], "NR", "NRMSE NARMA (moins = mieux)", "viridis_r")):
        y = np.concatenate([esn[key], coul[key]])
        norm = plt.Normalize(y.min(), y.max())
        ax.scatter(P[:n_e, 0], P[:n_e, 1], c=y[:n_e], cmap=cmap, norm=norm, marker="o", s=30,
                   label="ESN", edgecolors="k", linewidths=0.3)
        sc = ax.scatter(P[n_e:, 0], P[n_e:, 1], c=y[n_e:], cmap=cmap, norm=norm, marker="^", s=40,
                        label="Coulomb", edgecolors="k", linewidths=0.3)
        fig.colorbar(sc, ax=ax, label=label)
        ax.set(xlabel="composante principale 1", ylabel="composante principale 2",
               title=f"Les deux substrats dans le même espace, couleur = {label}")
        ax.legend(fontsize=8)
    for ax, task, key in ((axs[1, 0], "MC", "MC"), (axs[1, 1], "NARMA", "NR")):
        preds, inside = t_ec[task]
        p, y = preds["vecteur, k-NN"], coul[key]
        ax.scatter(y[inside], p[inside], c="tab:green", s=25, label="dans le domaine de l'ESN")
        ax.scatter(y[~inside], p[~inside], c="tab:red", s=25, label="hors domaine")
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set(xlabel=f"{task} réelle du Coulomb", ylabel=f"{task} prédite (appris sur l'ESN)",
               title=f"{task} : prédire le Coulomb après avoir appris sur l'ESN")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("cross_test.png", dpi=110)
    milestone("X4", True, "figure enregistrée : cross_test.png")
    print("=" * 84)
    if show:
        plt.show()
    return {"esn": esn, "coulomb": coul, "esn_to_coulomb": t_ec, "coulomb_to_esn": t_ce}


if __name__ == "__main__":
    cross = cross_test()
