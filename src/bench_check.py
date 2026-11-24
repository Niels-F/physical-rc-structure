"""
Étape 8 — Vérifier le banc corrigé et l'axe de consistance.

Deux corrections, découvertes en comparant ESN et Coulomb :
  1. la lecture est apprise sur un départ et testée sur un AUTRE départ, avec une
     autre entrée (avant : 70 % / 30 % de la même simulation) ;
  2. le vecteur reçoit un 8e axe, la consistance : la même entrée, depuis deux
     départs vraiment différents, donne-t-elle la même réponse ?

Ce qu'on attend : rien ne change pour l'ESN (il oublie son départ), tout change
pour un système qui ne l'oublie pas (le Coulomb).

Notebook : exécuter d'abord les cellules des étapes 1 à 6 (sweep et coulomb
contiennent les nouvelles versions). La cellule exécute les milestones B1 à B6.
"""

import time

import numpy as np

# Notebook : pas d'import entre fichiers.

_T0 = time.perf_counter()


def milestone(tag, ok, msg):
    status = "OK  " if ok else "FAIL"
    print(f"[{tag}] [{status}] {msg:<62} ({time.perf_counter() - _T0:5.2f} s)")
    return ok


def result(msg):
    print(f"[RES] {msg}")


def old_bench(sub):
    """L'ancien banc, pour comparer : une seule simulation, 70 % / 30 %."""
    u = np.random.default_rng(42).uniform(-1, 1, WARMUP + T)
    Xa, _ = run_pair(sub, u, warmup=WARMUP)
    return memory_capacity(Xa, u[WARMUP:])[0], narma_nrmse(Xa, u[WARMUP:])


def self_test():
    print("=" * 84)
    print("Étape 8 — validation du banc corrigé et de l'axe de consistance")
    print("=" * 84)
    rng = np.random.default_rng(0)

    subs = {"ESN": ESNSubstrate(), "Coulomb": CoulombSubstrate(damping=1.0, E0=0.25)}
    ok = True
    for name, sub in subs.items():
        s0, s1 = sub.init_state(rng), sub.random_state(rng)
        ok &= np.shape(s1) == np.shape(s0) and np.linalg.norm(s1 - s0) > 0.5
        ok &= not check_substrate(sub)
    milestone("B1", ok, "random_state : forme correcte, départ vraiment différent")

    # B2 : la consistance sur des cas dont on connaît la réponse
    t = np.arange(3000)
    lin_a = np.outer(np.sin(t / 7), np.ones(5))
    lin_b = lin_a + 0.01 * rng.standard_normal(lin_a.shape)              # même réponse
    indep = rng.standard_normal(lin_a.shape)                            # rien en commun
    c_same, c_ind = consistency(lin_a, lin_b), consistency(lin_a, indep)
    milestone("B2", c_same > 0.99 and abs(c_ind) < 0.05,
              f"consistance : même réponse {c_same:.3f} (~1), indépendantes {c_ind:+.3f} (~0)")

    # B3 : ESN et Coulomb sous le banc complet
    res = {}
    for name, sub in subs.items():
        C, mc, nr = evaluate(sub)
        mc_old, nr_old = old_bench(sub)
        res[name] = (C, mc, nr, mc_old, nr_old)
        result(f"{name:<8} consistance {C['consist']:+.3f}   MC ancien {mc_old:5.2f} -> "
               f"nouveau {mc:5.2f}   NARMA ancien {nr_old:.3f} -> nouveau {nr:.3f}")
    milestone("B3", res["ESN"][0]["consist"] > 0.99,
              f"ESN : consistance {res['ESN'][0]['consist']:.4f} (> 0.99)")

    # B4 : pour un système consistant, le nouveau banc ne change presque rien
    C, mc, nr, mc_old, nr_old = res["ESN"]
    milestone("B4", abs(mc - mc_old) / mc_old < 0.15 and abs(nr - nr_old) < 0.05,
              f"ESN : MC {mc_old:.2f} -> {mc:.2f}, NARMA {nr_old:.3f} -> {nr:.3f} (inchangés)")

    # B5 : le vecteur a bien 8 axes, tous dans [0, 1]
    v = normalize_vector(res["Coulomb"][0])
    milestone("B5", len(v) == len(AXES) == 8 and np.all((v >= 0) & (v <= 1)),
              f"vecteur à {len(v)} axes, tous dans [0, 1]")
    milestone("B5b", res["Coulomb"][2] == 1.0,
              f"NRMSE plafonnée : échec du Coulomb affiché {res['Coulomb'][2]:.2f} (au lieu de ~28)")

    # B6 : la figure, ESN et Coulomb avec le nouvel axe
    import matplotlib.pyplot as plt
    ax = plot_parallel([res["ESN"][0], res["Coulomb"][0]],
                       [f"ESN  (MC {res['ESN'][1]:.1f}, NARMA {res['ESN'][2]:.2f})",
                        f"Coulomb gamma=1  (MC {res['Coulomb'][1]:.1f}, NARMA {res['Coulomb'][2]:.2f})"],
                       title="Le vecteur à 8 axes, banc corrigé")
    ax.figure.savefig("bench_check.png", dpi=110)
    milestone("B6", True, "figure enregistrée : bench_check.png")
    print("=" * 84)
    plt.show()


if __name__ == "__main__":
    self_test()
