"""
Étape 9 — Troisième substrat : un réseau masses-ressorts (calcul morphologique).

Conception de Hauser et al. (2011) : des masses reliées à leurs voisines par des
ressorts amortis, quelques masses du bord fixées (ancrages), l'entrée appliquée
comme une force sur une partie des masses. Force d'un ressort d'allongement x :

    F = k1 x + k3 x³ + c dx/dt        (le long du ressort)

Deux versions qui ne diffèrent que par k3 :
    linéaire  : k3 = 0        cubique : k3 > 0
Attention : même « linéaire », le réseau n'est pas un système linéaire. La longueur
d'un ressort |r_i - r_j| dépend des positions de façon non linéaire (géométrie).
Il ne devient linéaire que pour de petites déformations.

Comme pour le Coulomb : 25 masses libres, on lit leurs positions (50 canaux),
un pas d'entrée = 0.2 unité de temps (4 pas de Verlet de 0.05).

Notebook : exécuter d'abord les cellules des étapes 1 à 8.
La cellule exécute les milestones K1 à K6, puis : prc = compare_prc()
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


class MassSpringSubstrate(Substrate):
    """État : vecteur plat [positions des masses libres (F, 2), vitesses (F, 2)]."""

    def __init__(self, n_free=25, n_anchor=6, k_neigh=4, k3=0.0, damping=1.0,
                 amp=0.3, input_frac=0.4, dt=0.05, n_sub=4, seed=0):
        # amp et damping : avec amp=1 et damping=0.3, le réseau se replie (ressorts écrasés
        # à 6 % de leur longueur) et devient multistable : consistance 0.32. Avec amp=0.3
        # et damping=1, il reste déformé sans se replier : consistance 1.000.
        rng = np.random.default_rng(seed)
        n = n_free + n_anchor
        P = rng.uniform(-2, 2, (n, 2))
        order = np.argsort(-np.linalg.norm(P, axis=1))                  # les plus extérieures
        self.anchor, self.free = order[:n_anchor], np.sort(order[n_anchor:])
        self.P0 = P                                                      # équilibre = départ
        D = np.linalg.norm(P[:, None] - P[None], axis=2)
        np.fill_diagonal(D, np.inf)
        edges = {tuple(sorted((i, j))) for i in range(n) for j in np.argsort(D[i])[:k_neigh]}
        self.ei, self.ej = np.array(sorted(edges)).T                     # ressorts (i, j)
        self.L0 = D[self.ei, self.ej]                                    # longueurs au repos
        self.k1 = rng.uniform(0.5, 1.5, len(self.L0))
        self.k3 = k3 * self.k1                                           # même rapport partout
        self.c = damping
        self.n, self.F, self.dt, self.n_sub = n, n_free, dt, n_sub
        w = np.zeros((n, 2))                                             # force d'entrée
        inp = rng.choice(self.free, int(input_frac * n_free), replace=False)
        w[inp] = rng.uniform(-1, 1, (len(inp), 2)) * amp
        self.w_in = w[self.free]
        self.n_obs = 2 * n_free

    # ------------------------------------------------------------- physique
    def _full(self, pos_free):
        P = self.P0.copy()
        P[self.free] = pos_free
        return P

    def spring_forces(self, P, V):
        """Force de chaque ressort sur toutes les masses (ancrages compris), (n, 2)."""
        d = P[self.ej] - P[self.ei]
        l = np.linalg.norm(d, axis=1)
        e = d / l[:, None]
        x = l - self.L0
        rate = np.sum((V[self.ej] - V[self.ei]) * e, axis=1)
        f = (self.k1 * x + self.k3 * x ** 3 + self.c * rate)[:, None] * e   # tire i vers j si étiré
        out = np.zeros_like(P)
        np.add.at(out, self.ei, f)
        np.add.at(out, self.ej, -f)
        return out

    def accel(self, pos, vel, u):
        V = np.zeros((self.n, 2)); V[self.free] = vel
        return self.spring_forces(self._full(pos), V)[self.free] + self.w_in * u   # masses = 1

    def energy(self, s):
        pos, vel = s[: 2 * self.F].reshape(-1, 2), s[2 * self.F:].reshape(-1, 2)
        P = self._full(pos)
        x = np.linalg.norm(P[self.ej] - P[self.ei], axis=1) - self.L0
        return 0.5 * np.sum(vel ** 2) + np.sum(0.5 * self.k1 * x ** 2 + 0.25 * self.k3 * x ** 4)

    # ------------------------------------------------------------- interface
    def init_state(self, rng=None):
        return np.concatenate([self.P0[self.free].ravel(), np.zeros(2 * self.F)])

    def random_state(self, rng):
        """Le même réseau, déformé : chaque masse libre déplacée jusqu'à ±0.3."""
        pos = self.P0[self.free] + rng.uniform(-0.3, 0.3, (self.F, 2))
        return np.concatenate([pos.ravel(), np.zeros(2 * self.F)])

    def step(self, s, u):
        pos, vel = s[: 2 * self.F].reshape(-1, 2), s[2 * self.F:].reshape(-1, 2)
        acc = self.accel(pos, vel, u)
        for _ in range(self.n_sub):                                      # Verlet-vitesse
            v_half = vel + 0.5 * acc * self.dt
            pos = pos + v_half * self.dt
            acc = self.accel(pos, v_half, u)
            vel = v_half + 0.5 * acc * self.dt
        return np.concatenate([pos.ravel(), vel.ravel()])

    def observe(self, s):
        return s[: 2 * self.F]


def self_test():
    print("=" * 84)
    print("Étape 9 — validation du réseau masses-ressorts")
    print("=" * 84)
    lin, cub = MassSpringSubstrate(k3=0.0), MassSpringSubstrate(k3=20.0)
    problems = check_substrate(lin) + check_substrate(cub)
    milestone("K1", not problems, f"contrôle technique ({lin.F} masses libres, "
                                  f"{len(lin.anchor)} ancrages, {len(lin.L0)} ressorts)")
    for p in problems:
        print("   -", p)

    # K2 : au repos et sans entrée, le réseau est à l'équilibre
    s0 = lin.init_state()
    a0 = np.abs(lin.accel(s0[:50].reshape(-1, 2), np.zeros((lin.F, 2)), 0.0)).max()
    milestone("K2", a0 < 1e-12, f"équilibre au départ : accélération max {a0:.1e} (~0)")

    # K3 : 3e loi de Newton (ancrages compris, les ressorts ne créent pas de force nette)
    rng = np.random.default_rng(1)
    P = lin.P0 + rng.normal(0, 0.2, lin.P0.shape)
    net = np.abs(lin.spring_forces(P, rng.normal(0, 0.3, P.shape)).sum(axis=0)).max()
    milestone("K3", net < 1e-12, f"somme des forces des ressorts = {net:.1e} (~0)")

    # K4 : sans frottement ni entrée, l'énergie se conserve : erreur en dt², pas de dérive.
    # Cas sévère exprès : ressorts cubiques (k3=20) et masses déplacées jusqu'à 0.3,
    # donc des ressorts jusqu'à ~6 fois plus raides qu'au repos.
    spans, drifts = [], []
    for dt, n_sub in ((0.05, 4), (0.025, 8)):
        free = MassSpringSubstrate(k3=20.0, damping=0.0, dt=dt, n_sub=n_sub)
        s = free.random_state(np.random.default_rng(2))
        E = []
        for _ in range(2000):
            s = free.step(s, 0.0)
            E.append(free.energy(s))
        E = np.array(E)
        spans.append((E.max() - E.min()) / E.mean())
        drifts.append(abs(E[-200:].mean() - E[:200].mean()) / E.mean())
    ratio = spans[0] / spans[1]
    milestone("K4", 3 < ratio < 5 and drifts[0] < spans[0],
              f"énergie : oscille de {spans[0]:.1e}, /{ratio:.1f} si dt/2 (attendu /4), "
              f"dérive {drifts[0]:.0e}")

    # K5 : le terme cubique n'agit que pour de grandes déformations
    u = np.random.default_rng(42).uniform(-1, 1, 1500)
    gaps = {}
    for amp in (0.01, 0.3):
        Xl, _ = run(MassSpringSubstrate(k3=0.0, amp=amp), u)
        Xc, _ = run(MassSpringSubstrate(k3=20.0, amp=amp), u)
        gaps[amp] = np.linalg.norm(Xl - Xc) / np.linalg.norm(Xl - Xl.mean(0))
    milestone("K5", gaps[0.01] < 0.01 and gaps[0.3] > 10 * gaps[0.01],
              f"écart linéaire/cubique : entrée 0.01 -> {gaps[0.01]:.4f}, entrée 0.3 -> {gaps[0.3]:.3f}")

    # K6 : banc complet sur les deux versions
    for name, sub in (("linéaire", lin), ("cubique", cub)):
        C, mc, nr = evaluate(sub)
        result(f"{name:<9} consistance {C['consist']:.3f}  lambda {C['lam']:+.4f}  "
               f"MC {mc:5.2f}  NARMA {nr:.3f}")
    milestone("K6", np.isfinite(mc) and np.isfinite(nr), "banc complet calculé sur les deux versions")
    print("=" * 84)


def compare_prc(seeds=(0, 1, 2), show=True):
    """Les deux PRC sous le même banc : masses-ressorts (2 versions) et Coulomb."""
    print("=" * 84)
    print("Comparaison des substrats physiques (banc honnête)")
    print("=" * 84)
    makers = {"Ressorts linéaires": lambda s: MassSpringSubstrate(k3=0.0, seed=s),
              "Ressorts cubiques": lambda s: MassSpringSubstrate(k3=20.0, seed=s),
              "Coulomb (gamma=1, E0=0.25)": lambda s: CoulombPositions(damping=1.0, E0=0.25, seed=s)}
    out = {}
    for name, make in makers.items():
        res = [evaluate(make(s), s) for s in seeds]
        out[name] = res
        mc = np.array([r[1] for r in res]); nr = np.array([r[2] for r in res])
        cs = np.array([r[0]["consist"] for r in res])
        print(f"{name:<27} consistance {cs.mean():.3f}   MC {mc.mean():5.2f} ± {mc.std():.2f}   "
              f"NARMA {np.median(nr):.3f} (médiane)")
    milestone("K7", all(np.isfinite(r[1]) for v in out.values() for r in v),
              "les trois substrats mesurés sous le même banc")

    print("   vecteurs normalisés (moyenne des graines) :")
    print("   " + " " * 27 + " ".join(f"{k[:8]:>8}" for k, _, _ in AXES))
    for name, res in out.items():
        m = np.nanmean([normalize_vector(r[0]) for r in res], axis=0)
        print(f"   {name[:27]:<27}" + " ".join(f"{x:8.2f}" for x in m))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 5))
    draw_axes(ax)
    for (name, res), col in zip(out.items(), ("tab:green", "tab:purple", "tab:orange")):
        for i, r in enumerate(res):
            lab = (f"{name}  MC={np.mean([x[1] for x in res]):.1f}, "
                   f"NARMA={np.median([x[2] for x in res]):.2f}") if i == 0 else None
            ax.plot(range(len(AXES)), normalize_vector(r[0]), "o-", color=col, lw=2, alpha=0.7, label=lab)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Trois substrats physiques dans l'espace des vecteurs (une ligne par graine)")
    fig.tight_layout()
    fig.savefig("compare_prc.png", dpi=110)
    milestone("K8", True, "figure enregistrée : compare_prc.png")
    print("=" * 84)
    if show:
        plt.show()
    return out


if __name__ == "__main__":
    self_test()
