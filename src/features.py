"""
Étape 15 — Le profil structurel gradué, et deux nouvelles familles de substrats.

Le filtre de l'étape 12 répond par oui ou non. Le profil ajoute des DEGRÉS, toujours
calculés sans piloter le système par une série (seules sont autorisées : l'évolution
sans entrée, une perturbation infinitésimale, une entrée constante) :

  mc_lin   capacité mémoire prédite par la linéarisation (étape 14)
  nl       non-linéarité statique à l'amplitude de travail (étape 14)
  d2       version graduée de Q2 : distance moyenne entre les états finaux obtenus
           depuis 6 départs et l'équilibre de référence, rapportée à l'amplitude du
           déplacement statique sous entrée, |x(u=+1) - x(u=-1)| / 2.
           d2 = 0 : le système revient toujours au même état ; d2 << 1 : les états
           finaux diffèrent moins que ce que l'entrée fait bouger ; d2 >> 1 : ils
           diffèrent plus.
  + les grandeurs du filtre : modes neutres, nombre d'états finaux, temps le plus lent.

Nouveaux substrats :
  DelayNodeSubstrate   un seul nœud non linéaire et une boucle de retard, 50 nœuds
                       virtuels le long du retard (le « graphe dans le temps »)
  CoulombRepulsive     charges toutes de même signe dans un puits elliptique

Notebook : exécuter d'abord les cellules des étapes 1 à 14.
La cellule exécute les milestones U1 à U5.
"""

import time

import numpy as np

# Notebook : pas d'import entre fichiers.

_T0 = time.perf_counter()


def milestone(tag, ok, msg):
    status = "OK  " if ok else "FAIL"
    print(f"[{tag}] [{status}] {msg:<62} ({time.perf_counter() - _T0:5.2f} s)")
    return ok


# ------------------------------------------------------------- nouveaux substrats
class DelayNodeSubstrate(Substrate):
    """Nœud unique à retard (idée d'Appeltant et al. 2011), version à temps discret.
    Les n nœuds virtuels sont parcourus dans l'ordre le long de la boucle :
        s_i(t) = d s_{i-1}(t) + (1 - d) tanh(eta s_i(t-1) + gamma m_i u(t) + phi),
    avec s_0(t) = s_n(t-1), d = exp(-theta), et m_i = ±1 un masque fixe."""

    def __init__(self, n=50, eta=0.8, gamma=0.5, theta=0.5, phi=0.2, seed=0):
        rng = np.random.default_rng(seed)
        self.n, self.eta, self.gamma, self.d, self.phi = n, eta, gamma, np.exp(-theta), phi
        self.mask = rng.choice([-1.0, 1.0], n)
        self.n_obs = n

    def init_state(self, rng=None):
        return np.zeros(self.n)

    def random_state(self, rng):
        return rng.uniform(-1, 1, self.n)

    def step(self, s, u):
        drive = (1 - self.d) * np.tanh(self.eta * s + self.gamma * self.mask * u + self.phi)
        new, prev = np.empty(self.n), s[-1]
        for i in range(self.n):                       # inertie le long de la boucle
            prev = self.d * prev + drive[i]
            new[i] = prev
        return new

    def observe(self, s):
        return s


class CoulombRepulsive(CoulombBroken):
    """Charges toutes positives (0.3 à 1) dans un puits elliptique : plus d'attraction
    entre charges opposées, donc a priori moins d'arrangements d'équilibre."""

    def __init__(self, **kw):
        super().__init__(**kw)
        rng = np.random.default_rng(kw.get("seed", 0) + 900)
        self.q = rng.uniform(0.3, 1.0, (self.n, 1))
        self.qq = self.q * self.q.T
        np.fill_diagonal(self.qq, 0.0)


# ------------------------------------------------------------- le profil
def structural_profile(sub, seed=0, horizon=200):
    """Le filtre de l'étape 12 (mêmes tirages, donc mêmes verdicts) et les degrés."""
    rng = np.random.default_rng(seed)
    s_eq, conv = relax(sub, sub.init_state(rng))
    J = step_jacobian(sub, s_eq)
    mu = np.abs(np.linalg.eigvals(J))
    n_neutral = int(np.sum(mu > 1 - NEUTRAL_TOL))
    starts = [sub.random_state(rng) for _ in range(EQ_STARTS)]
    ends = [relax(sub, st)[0] for st in starts]
    scale = max(np.linalg.norm(s_eq), np.mean([np.linalg.norm(st) for st in starts]))
    distinct = [s_eq]
    for e in ends:
        if all(np.linalg.norm(e - d) > 1e-3 * scale for d in distinct):
            distinct.append(e)
    b, Cm = input_vector(sub, s_eq), observe_matrix(sub, s_eq)

    # Degrés : réponses statiques à une entrée constante
    x0 = sub.observe(s_eq)
    xp = sub.observe(static_response(sub, s_eq, 1.0))
    xh = sub.observe(static_response(sub, s_eq, 0.5))
    xm = sub.observe(static_response(sub, s_eq, -1.0))
    n1 = np.linalg.norm(xp - x0)
    nl = float(np.linalg.norm((xp - x0) - 2 * (xh - x0)) / n1) if n1 > 1e-12 else 0.0
    amp = 0.5 * np.linalg.norm(xp - xm)
    spread = float(np.mean([np.linalg.norm(sub.observe(e) - x0) for e in ends]))
    d2 = spread / amp if amp > 1e-12 else np.inf

    p = {"converged": conv, "rho": float(mu.max()), "n_neutral": n_neutral,
         "n_equilibria": len(distinct),
         "slowest": float(-1 / np.log(min(mu.max(), 1 - 1e-12))),
         "mc_lin": linear_memory_capacity(J, b, Cm)[0], "nl": nl, "d2": float(d2),
         "static_amp": float(amp), "spread": spread}
    p["Q1"], p["Q2"] = n_neutral == 0, len(distinct) == 1
    p["Q3"], p["Q4"] = bool(np.linalg.norm(b) > 1e-12), bool(np.linalg.norm(Cm) > 1e-12)
    p["admis"] = bool(p["Q1"] and p["Q2"] and p["Q3"] and p["Q4"])
    return p


# ================================================================= milestones
def self_test():
    print("=" * 84)
    print("Étape 15 — validation du profil structurel et des nouveaux substrats")
    print("=" * 84)
    new = [DelayNodeSubstrate(), DelayNodeSubstrate(eta=1.5, phi=0.0), CoulombRepulsive(damping=1.0, E0=0.25)]
    probs = sum((check_substrate(s) for s in new), [])
    milestone("U1", not probs, "nœud à retard et Coulomb répulsif : contrôle technique")
    for pb in probs:
        print("   -", pb)

    # U2 : le profil reproduit exactement les verdicts du filtre de l'étape 12
    cases = [("ESN", ESNSubstrate(seed=0))] + [(k, ChainSubstrate(seed=0, **kw)) for k, kw in CHAINS.items()]
    profs = {name: structural_profile(sub) for name, sub in cases}
    same = all(profs[name]["admis"] == structural_report(sub)["admis"]
               and profs[name]["n_equilibria"] == structural_report(sub)["n_equilibria"]
               for name, sub in cases)
    milestone("U2", same, "profil et filtre de l'étape 12 : mêmes verdicts sur 4 systèmes")

    # U3 : les degrés sur des cas connus
    lin = structural_profile(LinearESN(rho=0.9, alpha=0.3))
    d2_esn, d2_dw = profs["ESN"]["d2"], profs["chaîne b) double puits"]["d2"]
    milestone("U3", lin["nl"] < 1e-6 and d2_esn < 1e-6 and d2_dw > 1,
              f"nl (ESN linéaire) = {lin['nl']:.0e} ; d2 : ESN {d2_esn:.0e}, double puits {d2_dw:.1f}")

    # U4 : le nœud à retard avec eta > 1 est bistable, avec eta < 1 il ne l'est pas
    mono, bi = structural_profile(new[0]), structural_profile(new[1])
    milestone("U4", mono["admis"] and bi["n_equilibria"] >= 2,
              f"nœud à retard : eta=0.8 -> {mono['n_equilibria']} état (admis), "
              f"eta=1.5 -> {bi['n_equilibria']} états")

    # U5 : la capacité linéaire du profil est celle de l'étape 14
    esn = ESNSubstrate(seed=0)
    _, J, b, Cm = linearize(esn)
    milestone("U5", abs(profs["ESN"]["mc_lin"] - linear_memory_capacity(J, b, Cm)[0]) < 1e-9,
              f"mc_lin du profil = étape 14 ({profs['ESN']['mc_lin']:.2f})")
    for name, pr in list(profs.items()) + [("nœud à retard", mono), ("Coulomb répulsif", structural_profile(new[2]))]:
        print(f"   {name:<24} neutres {pr['n_neutral']}  états {pr['n_equilibria']}  mc_lin {pr['mc_lin']:6.2f}  "
              f"nl {pr['nl']:.3f}  d2 {pr['d2']:8.3f}  {'ADMIS' if pr['admis'] else 'rejeté'}")
    print("=" * 84)


if __name__ == "__main__":
    self_test()
