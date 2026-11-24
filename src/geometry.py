"""
Étape 10 — Les géométries comme paramètres cachés de fabrication.

Chaque graine tire un réseau masses-ressorts différent (positions, ressorts,
masses qui reçoivent l'entrée), comme deux puces « identiques » d'un même lot.
Question : SANS connaître la géométrie, le vecteur dit-il quels réseaux marchent ?

Grille : 2 amplitudes d'entrée x 2 amortissements (le point de fonctionnement,
connu et réglable) x 12 géométries (cachées) = 48 réseaux, environ 6 minutes.

Test : on retire une géométrie entière (ses 4 points de fonctionnement), on la
prédit à partir des 11 autres. Prédicteurs comparés :
  - le point de fonctionnement seul (amp, amortissement) : ne voit pas la géométrie
  - la consistance seule
  - le vecteur sans la consistance (la « richesse » seule)
  - le vecteur complet

Notebook : exécuter d'abord les cellules des étapes 1 à 9.
Puis : geo = geometry_test()
"""

import time

import numpy as np

# Notebook : pas d'import entre fichiers.

AMPS = [0.15, 0.3]
DAMPINGS = [0.5, 1.0]
GEOMETRIES = list(range(12))

_T0 = time.perf_counter()


def milestone(tag, ok, msg):
    status = "OK  " if ok else "FAIL"
    print(f"[{tag}] [{status}] {msg:<62} ({time.perf_counter() - _T0:5.2f} s)")
    return ok


def result(msg):
    print(f"[RES] {msg}")


def geometry_sweep():
    rows = []                                    # (amp, damping, géométrie, vecteur, MC, NRMSE)
    for g in GEOMETRIES:
        for amp in AMPS:
            for c in DAMPINGS:
                C, mc, nr = evaluate(MassSpringSubstrate(amp=amp, damping=c, seed=g), g)
                rows.append((amp, c, g, C, mc, nr))
        print(f"   géométrie {g:2d} faite   (MC aux 4 points : "
              + " ".join(f"{r[4]:4.1f}" for r in rows[-4:]) + ")")
    return {"rows": rows,
            "V": np.array([normalize_vector(r[3]) for r in rows]),
            "MC": np.array([r[4] for r in rows]),
            "NR": np.array([r[5] for r in rows]),
            "OP": np.array([[r[0], r[1]] for r in rows]),
            "G": np.array([r[2] for r in rows])}


def geometry_test(data=None, show=True):
    print("=" * 84)
    print("Étape 10 — le vecteur reconnaît-il les bonnes géométries ?")
    print("=" * 84)
    if data is None:
        data = geometry_sweep()
    V, MC, NR, OP, G = data["V"], data["MC"], data["NR"], data["OP"], data["G"]
    milestone("G1", len(MC) == len(GEOMETRIES) * len(AMPS) * len(DAMPINGS)
              and np.all(np.isfinite(V)), f"{len(MC)} réseaux mesurés, vecteurs complets")

    keys = [k for k, _, _ in AXES]
    ic = keys.index("consist")
    result("par point de fonctionnement (les 12 géométries) :")
    for amp in AMPS:
        for c in DAMPINGS:
            m = (OP[:, 0] == amp) & (OP[:, 1] == c)
            print(f"      amp={amp:<5} amort.={c:<4} MC {MC[m].mean():5.2f} ± {MC[m].std():.2f}   "
                  f"NARMA {NR[m].mean():.2f} ± {NR[m].std():.2f}   "
                  f"consistants (>0.95) : {np.sum(V[m, ic] > 0.95)}/{m.sum()}")

    # Combien de la variation vient de la géométrie, combien du point de fonctionnement ?
    for name, y in (("MC", MC), ("NARMA", NR)):
        op_means = np.array([y[(OP[:, 0] == a) & (OP[:, 1] == c)].mean() for a, c in OP])
        share = 1 - np.var(y - op_means) / np.var(y)
        result(f"{name} : le point de fonctionnement explique {share:.0%} de la variation, "
               f"la géométrie cachée le reste ({1 - share:.0%})")

    result("corrélation de rang avec la performance (+ = aide), sur les 48 réseaux")
    print("      " + " ".join(f"{k[:8]:>9}" for k in keys))
    print("  MC  " + " ".join(f"{spearman(V[:, j], MC):+9.2f}" for j in range(len(keys))))
    print("NARMA " + " ".join(f"{-spearman(V[:, j], NR):+9.2f}" for j in range(len(keys))))

    feats = {"point de fonctionnement": OP,
             "consistance seule": V[:, [ic]],
             "vecteur sans consistance": np.delete(V, ic, axis=1),
             "vecteur complet": V}
    result("prédire une géométrie jamais vue (R², selon le nombre de voisins k)")
    scores = {}
    for task, y in (("MC", MC), ("NARMA", NR)):
        print(f"      {task:<6}" + "".join(f"{'k=' + str(k):>8}" for k in (3, 5, 8)))
        for name, F in feats.items():
            r = [loco_knn_r2(F, y, G, k) for k in (3, 5, 8)]
            scores[(task, name)] = r
            print(f"        {name:<25}" + "".join(f"{x:8.2f}" for x in r))
    milestone("G2", all(np.all(np.isfinite(v)) for v in scores.values()), "prédictions calculées")

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    labels = {(a, c): f"amp {a}, amort. {c}" for a in AMPS for c in DAMPINGS}
    for (a, c), lab in labels.items():
        m = (OP[:, 0] == a) & (OP[:, 1] == c)
        ax1.scatter(V[m, ic], MC[m], s=25, label=lab)
        ax2.scatter(V[m, ic], NR[m], s=25, label=lab)
    ax1.set(xlabel="consistance", ylabel="MC", title="MC contre consistance")
    ax2.set(xlabel="consistance", ylabel="NRMSE NARMA (plafonnée à 1)", title="NARMA contre consistance")
    ax1.legend(fontsize=7); ax2.legend(fontsize=7)

    F = V
    pred = np.empty_like(MC)
    for g in np.unique(G):
        te, tr = G == g, G != g
        A, B = (F[tr] - F[tr].mean(0)) / (F[tr].std(0) + 1e-9), (F[te] - F[tr].mean(0)) / (F[tr].std(0) + 1e-9)
        d = np.linalg.norm(B[:, None] - A[None], axis=2)
        pred[te] = MC[tr][np.argsort(d, axis=1)[:, :5]].mean(1)
    ax3.scatter(MC, pred, c=V[:, ic], cmap="RdYlGn", s=25, vmin=0, vmax=1)
    lo, hi = min(MC.min(), pred.min()), max(MC.max(), pred.max())
    ax3.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax3.set(xlabel="MC réelle", ylabel="MC prédite (géométrie jamais vue, k=5)",
            title="Prédiction (vecteur, k=5)\ncouleur = consistance")

    axP = fig.add_subplot(gs[1, :])
    draw_axes(axP)
    norm = plt.Normalize(MC.min(), MC.max())
    for i in np.argsort(MC):
        axP.plot(range(len(AXES)), V[i], lw=1.2, alpha=0.6, color=plt.cm.viridis(norm(MC[i])))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axP, label="MC")
    axP.set_title("48 réseaux masses-ressorts (12 géométries x 4 points), colorés par MC")
    fig.tight_layout()
    fig.savefig("geometry_test.png", dpi=110)
    milestone("G3", True, "figure enregistrée : geometry_test.png")
    print("=" * 84)
    if show:
        plt.show()
    return data | {"scores": scores}


if __name__ == "__main__":
    geo = geometry_test()
