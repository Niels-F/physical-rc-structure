"""
Étape 6 — Deuxième substrat : N charges dans un puits harmonique (Coulomb).

Physique reprise du réservoir N-corps de Léo (github.com/lele394/
N-Body-electromagnetic-reservoir-computer), mêmes équations et mêmes valeurs :

    m a_i = Σ_j k q_i q_j (r_i - r_j) / (|r_i - r_j|² + ε²)^{3/2}
            - k_well r_i - γ v_i + q_i E0 u(t) ex

Différences voulues avec l'original (fait pour MNIST, donc statique) :
  - l'entrée arrive au fil du temps, par un champ électrique extérieur E0 u(t) ;
    les charges aléatoires (+ et -) font que le même champ pousse chaque particule
    différemment : c'est l'équivalent physique de W_in ;
  - on lit seulement positions et vitesses (ce qu'un capteur mesure), pas les
    énergies par particule, qui ajouteraient de la non-linéarité côté lecture ;
  - un pas d'entrée = n_sub pas d'intégration (Verlet-vitesse, dt = 0.04).

Notebook : exécuter d'abord les cellules des étapes 1, 2a, 3, 4 et 5.
La cellule exécute les milestones C1 à C5, puis : comp = compare()
"""

import time

import numpy as np

# Notebook : pas d'import entre fichiers (Substrate, check_substrate, run, run_pair,
# ESNSubstrate, complexity_vector, normalize_vector, draw_axes, AXES,
# memory_capacity, narma_nrmse, T, WARMUP viennent des cellules précédentes).

_T0 = time.perf_counter()


def milestone(tag, ok, msg):
    status = "OK  " if ok else "FAIL"
    print(f"[{tag}] [{status}] {msg:<62} ({time.perf_counter() - _T0:5.2f} s)")
    return ok


class CoulombSubstrate(Substrate):
    """État : vecteur plat [positions (N, 2), vitesses (N, 2)] de taille 4N."""

    def __init__(self, n=25, E0=0.5, k_coulomb=1.5, epsilon=0.4, k_well=0.2,
                 damping=0.05, dt=0.04, n_sub=5, seed=0):
        rng = np.random.default_rng(seed)
        self.n, self.E0, self.k, self.eps = n, E0, k_coulomb, epsilon
        self.k_well, self.gamma, self.dt, self.n_sub = k_well, damping, dt, n_sub
        self.q = rng.uniform(-1.0, 1.0, (n, 1))                 # charges fixes
        self.qq = self.q * self.q.T
        np.fill_diagonal(self.qq, 0.0)                           # pas d'auto-interaction
        self.pos0 = rng.uniform(-2, 2, (n, 2))                   # départ dans le puits
        self.n_obs = 4 * n

    def coulomb_force(self, pos):
        """Force de Coulomb adoucie sur chaque particule, (N, 2)."""
        diff = pos[:, None, :] - pos[None, :, :]                 # r_i - r_j
        inv_r3 = (np.sum(diff ** 2, axis=-1) + self.eps ** 2) ** -1.5
        return self.k * np.sum((self.qq * inv_r3)[:, :, None] * diff, axis=1)

    def accel(self, pos, vel, u):
        a = self.coulomb_force(pos) - self.k_well * pos - self.gamma * vel
        a[:, 0] += self.q[:, 0] * self.E0 * u                    # champ extérieur selon x
        return a                                                  # masses = 1

    def energy(self, s):
        """Énergie totale (cinétique + puits + Coulomb, paires comptées une fois)."""
        pos, vel = s[: 2 * self.n].reshape(-1, 2), s[2 * self.n:].reshape(-1, 2)
        diff = pos[:, None, :] - pos[None, :, :]
        r = np.sqrt(np.sum(diff ** 2, axis=-1) + self.eps ** 2)
        return (0.5 * np.sum(vel ** 2) + 0.5 * self.k_well * np.sum(pos ** 2)
                + 0.5 * np.sum(self.k * self.qq / r))

    def init_state(self, rng=None):
        return np.concatenate([self.pos0.ravel(), np.zeros(2 * self.n)])

    def random_state(self, rng):
        """Autres positions tirées dans le puits, au repos."""
        return np.concatenate([rng.uniform(-2, 2, 2 * self.n), np.zeros(2 * self.n)])

    def step(self, s, u):
        pos, vel = s[: 2 * self.n].reshape(-1, 2), s[2 * self.n:].reshape(-1, 2)
        acc = self.accel(pos, vel, u)
        for _ in range(self.n_sub):                              # Verlet-vitesse (comme Léo)
            v_half = vel + 0.5 * acc * self.dt
            pos = pos + v_half * self.dt
            acc = self.accel(pos, v_half, u)
            vel = v_half + 0.5 * acc * self.dt
        return np.concatenate([pos.ravel(), vel.ravel()])        # nouvel état, s intact

    def observe(self, s):
        return s


def self_test():
    print("=" * 84)
    print("Étape 6 — validation du substrat de Coulomb")
    print("=" * 84)
    sub = CoulombSubstrate()
    problems = check_substrate(sub)
    milestone("C1", not problems, f"contrôle technique ({sub.n} particules, {sub.n_obs} canaux)")
    for p in problems:
        print("   -", p)

    # C2 : mêmes accélérations que le code de Léo, si sa classe est chargée
    pos = np.random.default_rng(3).uniform(-2, 2, (sub.n, 2))
    vel = np.random.default_rng(4).normal(0, 0.3, (sub.n, 2))
    if "NBodyReservoir" in globals():
        ref = NBodyReservoir(n_particles=sub.n, dt=sub.dt)       # noqa: F821
        ref.charges, ref.damping = sub.q, sub.gamma
        err = np.abs(ref._accel(pos[None], vel[None])[0] - sub.accel(pos, vel, 0.0)).max()
        milestone("C2", err < 1e-12, f"accélérations identiques au code de Léo (écart max {err:.1e})")
    else:
        print("[C2] [----] code de Léo non chargé : comparaison sautée (optionnelle)")

    # C3 : 3e loi de Newton, les forces internes s'annulent deux à deux
    total = sub.coulomb_force(pos).sum(axis=0)
    milestone("C3", np.abs(total).max() < 1e-10, f"somme des forces de Coulomb = {np.abs(total).max():.1e} (~0)")

    # C4 : sans frottement ni entrée, l'énergie se conserve, avec une erreur en dt².
    # L'énergie totale (~1.5) est une petite différence de grands termes (cinétique ~10,
    # Coulomb négatif entre charges opposées) : on la compare à l'énergie cinétique.
    # Verlet est d'ordre 2 : diviser dt par 2 doit diviser l'erreur par ~4.
    spans = []
    for dt, n_sub in ((0.04, 5), (0.02, 10)):                     # même durée physique
        free = CoulombSubstrate(damping=0.0, E0=0.0, dt=dt, n_sub=n_sub)
        s, E, K = free.init_state(), [], []
        for _ in range(2000):
            s = free.step(s, 0.0)
            E.append(free.energy(s)); K.append(0.5 * np.sum(s[2 * free.n:] ** 2))
        spans.append((max(E) - min(E)) / np.mean(K))
    ratio = spans[0] / spans[1]
    milestone("C4", spans[0] < 1e-2 and 3 < ratio < 5,
              f"énergie : erreur {spans[0]:.1e} (relative à E_cin), /{ratio:.1f} si dt/2 (attendu /4)")

    # C5 : piloté par l'entrée de référence, le vecteur se calcule
    u = np.random.default_rng(42).uniform(-1, 1, WARMUP + T)
    Xa, Xb = run_pair(sub, u, warmup=WARMUP)
    C = complexity_vector(Xa, Xb, D=4, warn=False)
    finite = all(np.isfinite(v) for k, v in C.items() if k != "consist")   # pas de 3e trajectoire ici
    milestone("C5", finite, f"vecteur calculé, lambda = {C['lam']:+.4f}")
    print("   C =", {k: round(float(v), 3) for k, v in C.items()})
    print("=" * 84)


def compare(seeds=(0, 1, 2), show=True):
    """ESN et Coulomb sous la même entrée : vecteur, MC et NARMA-10, côte à côte."""
    print("=" * 84)
    print("Comparaison ESN / Coulomb (deux frottements) sous la même entrée de référence")
    print("=" * 84)
    u = np.random.default_rng(42).uniform(-1, 1, WARMUP + T)
    makers = {"ESN (rho=0.9, alpha=0.3)": lambda s: ESNSubstrate(seed=s),
              "Coulomb, gamma=0.05 (Léo)": lambda s: CoulombSubstrate(seed=s),
              "Coulomb, gamma=1.0": lambda s: CoulombSubstrate(damping=1.0, seed=s)}
    out = {}
    for name, make in makers.items():
        res = []
        for s in seeds:
            Xa, Xb = run_pair(make(s), u, warmup=WARMUP)
            C = complexity_vector(Xa, Xb, D=4, warn=False)
            res.append((C, memory_capacity(Xa, u[WARMUP:])[0], narma_nrmse(Xa, u[WARMUP:])))
        out[name] = res
        mc = np.array([r[1] for r in res]); nr = np.array([r[2] for r in res])
        print(f"{name:<26} MC = {mc.mean():5.2f} ± {mc.std():.2f}   "
              f"NARMA NRMSE = {nr.mean():.3f} ± {nr.std():.3f}")
    milestone("C6", all(np.isfinite(r[1]) and np.isfinite(r[2]) for v in out.values() for r in v),
              "les deux substrats mesurés sur les mêmes tâches")

    print("   vecteurs normalisés (moyenne des graines) :")
    print("   " + " " * 24 + " ".join(f"{k[:8]:>8}" for k, _, _ in AXES))
    for name, res in out.items():
        m = np.nanmean([normalize_vector(r[0]) for r in res], axis=0)
        print(f"   {name[:24]:<24}" + " ".join(f"{x:8.2f}" for x in m))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4.8))
    draw_axes(ax)
    for (name, res), col in zip(out.items(), ("tab:blue", "tab:red", "tab:orange")):
        for i, r in enumerate(res):
            ax.plot(range(len(AXES)), normalize_vector(r[0]), "o-", color=col, lw=2, alpha=0.7,
                    label=f"{name}  MC={np.mean([x[1] for x in res]):.1f}, "
                          f"NARMA={np.mean([x[2] for x in res]):.2f}" if i == 0 else None)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("ESN et Coulomb à deux frottements, dans l'espace des vecteurs (une ligne par graine)")
    fig.tight_layout()
    fig.savefig("compare_esn_coulomb.png", dpi=110)
    milestone("C7", True, "figure enregistrée : compare_esn_coulomb.png")
    print("=" * 84)
    if show:
        plt.show()
    return out


if __name__ == "__main__":
    self_test()
