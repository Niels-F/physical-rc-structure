"""
Étape 14 — La capacité mémoire prédite par les équations, sans paramètre ajustable.

Autour de son équilibre, un système piloté se comporte comme sa linéarisation :
    s(t+1) = J s(t) + b u(t),      x(t) = C s(t)   (canaux lus)
donc x(t) = Σ_j g_j u(t-j) avec g_j = C J^j b. Pour une entrée i.i.d., la part de
u(t-k) qu'une lecture linéaire retrouve dans les canaux bruités vaut
    MC_k = g_k^T (G + eps² diag G)^{-1} g_k,      G = Σ_j g_j g_j^T,
où eps est le bruit de mesure relatif (1 %). La capacité est MC = Σ_{k=1..80} MC_k.
Tout vient de J, b, C : aucune simulation pilotée, aucun paramètre libre.

Ce fichier définit aussi la NON-LINÉARITÉ STATIQUE : on applique une entrée
constante u = 1 puis u = 0.5 et on compare les déplacements d'équilibre,
    nl = |dx(1) - 2 dx(0.5)| / |dx(1)|      (0 pour un système linéaire).

Critère fixé AVANT l'exécution de linear_mc_test() :
  P1  pour au moins 80 % des systèmes presque linéaires (nl < 0.02), la capacité
      mesurée est à moins de 20 % de la capacité prédite.

Notebook : exécuter d'abord les cellules des étapes 1 à 13.
La cellule exécute les milestones T1 à T4, puis : lin = linear_mc_test()   (~5 min)
"""

import time

import numpy as np

# Notebook : pas d'import entre fichiers.

N_LONG = 20000            # pas de mesure pour la comparaison à la théorie (voir measure_mc)

_T0 = time.perf_counter()


def milestone(tag, ok, msg):
    status = "OK  " if ok else "FAIL"
    print(f"[{tag}] [{status}] {msg:<62} ({time.perf_counter() - _T0:5.2f} s)")
    return ok


def result(msg):
    print(f"[RES] {msg}")


# ------------------------------------------------------------- la théorie
def linear_memory_capacity(J, b, Cm, k_max=K_MAX, noise=NOISE, horizon=3000):
    """Capacité mémoire du système linéarisé. Renvoie (MC, profil MC_k pour k = 1..k_max)."""
    if np.max(np.abs(np.linalg.eigvals(J))) > 1 + 1e-6:     # équilibre instable : la
        return float("nan"), np.full(k_max, np.nan)          # linéarisation n'a pas de sens
    g, v = [], b.copy()
    for _ in range(horizon):                      # g_j = C J^j b
        g.append(Cm @ v)
        v = J @ v
    g = np.array(g)
    G = g.T @ g                                   # covariance des canaux (entrée de variance 1)
    R = G + noise ** 2 * np.diag(np.diag(G)) + 1e-300 * np.eye(len(G))
    live = np.diag(G) > 1e-24 * np.diag(G).max()  # canaux qui ne reçoivent rien : retirés
    Rinv = np.linalg.pinv(R[np.ix_(live, live)], hermitian=True)
    gk = g[1: k_max + 1][:, live]
    mck = np.einsum("ki,ij,kj->k", gk, Rinv, gk)
    return float(mck.sum()), mck


def linearize(sub, s_eq=None):
    """(s_eq, J, b, C) autour de l'équilibre sans entrée."""
    if s_eq is None:
        s_eq, _ = relax(sub, sub.init_state())
    return s_eq, step_jacobian(sub, s_eq), input_vector(sub, s_eq), observe_matrix(sub, s_eq)


# ------------------------------------------------------------- non-linéarité statique
def static_response(sub, s0, u, max_steps=8000, tol=1e-9):
    """État atteint sous une entrée CONSTANTE u, en partant de s0."""
    s = s0
    for _ in range(max_steps):
        s_new = sub.step(s, u)
        if np.linalg.norm(s_new - s) < tol * (1 + np.linalg.norm(s)):
            return s_new
        s = s_new
    return s


def static_nonlinearity(sub, s_eq, u=1.0):
    """Écart à la proportionnalité de la réponse statique, vu par les canaux lus."""
    x0 = sub.observe(s_eq)
    d1 = sub.observe(static_response(sub, s_eq, u)) - x0
    dh = sub.observe(static_response(sub, s_eq, u / 2)) - x0
    n1 = np.linalg.norm(d1)
    return float(np.linalg.norm(d1 - 2 * dh) / n1) if n1 > 1e-12 else 0.0


# ------------------------------------------------------------- la mesure
def measure_mc(sub, seed=0, n=None):
    """Capacité mémoire mesurée (banc honnête : lecture testée sur un autre départ, autre entrée).

    n : nombre de pas. La formule théorique donne la limite d'une lecture apprise sur
    une infinité de données. Avec n = 3000 (banc habituel) et 50 canaux, la lecture est
    un peu sur-ajustée et la mesure lit ~10 % trop bas (ESN linéaire : 8.95 pour 10.05 ;
    9.70 à n = 10 000 ; 9.93 à n = 100 000). On compare donc à la théorie avec N_LONG pas.
    """
    n = T if n is None else n
    u = np.random.default_rng(42).uniform(-1, 1, WARMUP + n)
    u_test = np.random.default_rng(7).uniform(-1, 1, WARMUP + n)
    X, _ = run(sub, u)
    Xd, _ = run(sub, u_test, sub.random_state(np.random.default_rng(1000 + seed)))
    return memory_capacity(X[WARMUP:], u[WARMUP:], Xd[WARMUP:], u_test[WARMUP:])[0]


class LinearESN(ESNSubstrate):
    """ESN dont on a retiré le tanh : exactement linéaire, pour tester la formule."""

    def step(self, s, u):
        e = self.esn
        return (1 - e.alpha) * s + e.alpha * (e.W @ s + e.W_in * u + e.b)


class _Scalar(Substrate):
    """x(t+1) = a x(t) + u(t) : un seul canal, capacité connue exactement."""
    n_obs = 1

    def __init__(self, a=0.5):
        self.a = a

    def init_state(self, rng=None):
        return np.zeros(1)

    def step(self, s, u):
        return self.a * s + u

    def observe(self, s):
        return s


# ================================================================= milestones
def self_test():
    print("=" * 84)
    print("Étape 14 — validation de la capacité mémoire théorique")
    print("=" * 84)
    # T1 : un canal, x(t+1) = a x + u. r²_k = a^{2k} (1 - a²) / (1 + eps²), somme = a² / (1 + eps²)
    sc = _Scalar(0.5)
    _, J, b, Cm = linearize(sc)
    mc, mck = linear_memory_capacity(J, b, Cm)
    exact = 0.25 / (1 + NOISE ** 2)
    milestone("T1", abs(mc - exact) < 1e-9 and abs(mck[0] - 0.1875 / (1 + NOISE ** 2)) < 1e-9,
              f"système scalaire : MC = {mc:.6f} (exact {exact:.6f})")

    # T2 : ESN exactement linéaire, la formule doit donner la capacité mesurée
    lin = LinearESN(rho=0.9, alpha=0.3)
    _, J, b, Cm = linearize(lin)
    mc_th, mc_me = linear_memory_capacity(J, b, Cm)[0], measure_mc(lin, n=30000)
    mc_short = measure_mc(lin)
    milestone("T2", abs(mc_me - mc_th) / mc_th < 0.05 and mc_short < mc_me,
              f"ESN linéaire : théorie {mc_th:.2f}, mesure {mc_me:.2f} (30 000 pas), {mc_short:.2f} (3000)")

    # T3 : la non-linéarité statique vaut 0 pour un système linéaire, > 0 sinon
    s_l, _ = relax(lin, lin.init_state())
    esn = ESNSubstrate(rho=0.9, alpha=0.3)
    s_e, _ = relax(esn, esn.init_state())
    nl_l, nl_e = static_nonlinearity(lin, s_l), static_nonlinearity(esn, s_e)
    milestone("T3", nl_l < 1e-6 and nl_e > 0.01,
              f"non-linéarité statique : ESN linéaire {nl_l:.1e}, ESN tanh {nl_e:.3f}")

    # T4 : ESN tanh à très faible entrée : presque linéaire, la théorie doit tenir
    small = ESNSubstrate(rho=0.9, alpha=0.3, input_scale=0.01)
    s_s, J, b, Cm = linearize(small)
    mc_th, mc_me = linear_memory_capacity(J, b, Cm)[0], measure_mc(small, n=30000)
    milestone("T4", abs(mc_me - mc_th) / mc_th < 0.10,
              f"ESN tanh, entrée x0.01 : théorie {mc_th:.2f}, mesure {mc_me:.2f} "
              f"(nl = {static_nonlinearity(small, s_s):.4f})")
    print("=" * 84)


def linear_mc_test(show=True):
    """Théorie contre mesure, sur trois familles et plusieurs amplitudes d'entrée."""
    print("=" * 84)
    print("La capacité mémoire prédite par la linéarisation tient-elle ? (critère P1)")
    print("=" * 84)
    systems = []
    for g in range(3):
        for sc in (0.01, 0.1, 0.3, 1.0, 3.0):
            systems.append(("ESN", g, sc, ESNSubstrate(seed=g, input_scale=sc)))
    for g in range(2):
        for amp in (0.03, 0.3, 1.0, 3.0):
            systems.append(("Chaîne (a)", g, amp, ChainSubstrate(seed=g, amp=amp, a=0.5, b=0.5)))
    for g in (0, 3, 7):                               # géométries admises par le filtre
        for amp in (0.03, 0.3):
            systems.append(("Ressorts", g, amp, MassSpringSubstrate(seed=g, amp=amp, damping=1.0)))
    rows = []
    print(f"   {'système':<12}{'g':>3}{'ampl.':>7}{'nl':>8}{'MC théorie':>12}{'MC mesure':>11}{'écart':>8}")
    for name, g, amp, sub in systems:
        s_eq, J, b, Cm = linearize(sub)
        mc_th = linear_memory_capacity(J, b, Cm)[0]
        nl = static_nonlinearity(sub, s_eq)
        mc_me = measure_mc(sub, g, n=N_LONG)
        rows.append((name, g, amp, nl, mc_th, mc_me))
        print(f"   {name:<12}{g:>3}{amp:>7}{nl:>8.4f}{mc_th:>12.2f}{mc_me:>11.2f}"
              f"{(mc_me - mc_th) / mc_th:>+8.0%}")
    milestone("T5", all(np.isfinite(r[4]) and np.isfinite(r[5]) for r in rows),
              f"{len(rows)} systèmes : théorie et mesure calculées")

    nl = np.array([r[3] for r in rows]); th = np.array([r[4] for r in rows]); me = np.array([r[5] for r in rows])
    err = np.abs(me - th) / th
    near = nl < 0.02
    frac = np.mean(err[near] < 0.20) if near.any() else np.nan
    result(f"P1 : {np.sum(err[near] < 0.20)}/{near.sum()} systèmes presque linéaires à moins de 20 % "
           f"de la théorie ({frac:.0%}) -> {'RÉUSSI' if frac >= 0.8 else 'ÉCHOUÉ'}")
    result(f"écart médian : {np.median(err[near]):.0%} si nl < 0.02, "
           f"{np.median(err[~near]):.0%} si nl >= 0.02")
    result(f"sens de l'écart quand le système devient non linéaire : la mesure est "
           f"{'inférieure' if np.median((me - th)[~near]) < 0 else 'supérieure'} à la théorie "
           f"dans {np.mean((me - th)[~near] < 0):.0%} des cas")

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    marks = {"ESN": "o", "Chaîne (a)": "s", "Ressorts": "^"}
    for fam, mk in marks.items():
        m = np.array([r[0] == fam for r in rows])
        axs[0].scatter(th[m], me[m], c=np.log10(nl[m] + 1e-4), cmap="viridis", vmin=-4, vmax=0, marker=mk,
                       s=40, edgecolors="k", linewidths=0.3, label=fam)
        sc = axs[1].scatter(nl[m] + 1e-4, me[m] / th[m], c=np.log10(nl[m] + 1e-4), cmap="viridis", vmin=-4,
                            vmax=0, marker=mk, s=40, edgecolors="k", linewidths=0.3, label=fam)
    lim = max(th.max(), me.max()) * 1.05
    axs[0].plot([0, lim], [0, lim], "k--", lw=1)
    axs[0].set(xlabel="capacité mémoire prédite (équations seules)", ylabel="capacité mémoire mesurée",
               title="Théorie contre mesure")
    axs[0].legend(fontsize=8)
    axs[1].axhline(1, color="k", ls="--", lw=1); axs[1].axvline(0.02, color="0.6", lw=0.8)
    axs[1].set(xscale="log", xlabel="non-linéarité statique nl", ylabel="mesure / théorie",
               title="L'accord se dégrade quand nl augmente")
    fig.colorbar(sc, ax=axs[1], label="log10(nl)")
    fig.tight_layout()
    fig.savefig("linear_mc.png", dpi=110)
    milestone("T6", True, "figure enregistrée : linear_mc.png")
    print("=" * 84)
    if show:
        plt.show()
    return rows


if __name__ == "__main__":
    self_test()
