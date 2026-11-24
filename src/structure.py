"""
Étape 12 — Le filtre structurel : lire dans les équations si un système PEUT prévoir.

Règle du jeu : uniquement les équations et leurs paramètres, SANS entrée pilotée
et sans tâche. On a le droit de laisser le système se relâcher sans entrée (u = 0)
et de le linéariser autour de son équilibre.

Quatre conditions nécessaires à une mémoire évanescente (oui / non) :
  Q1  pas de mode neutre       : aucune direction où le système n'oublie jamais
                                 (valeur propre de la linéarisation |mu| ~ 1)
  Q2  retour au même état      : depuis des départs différents, il revient au
                                 même état en un temps raisonnable (sinon : plusieurs
                                 équilibres, ou des directions si molles qu'il y glisse)
  Q3  l'entrée agit            : l'entrée a un effet sur l'état (oui / non)
  Q4  la lecture voit          : les canaux lus dépendent de l'état (oui / non)

Q3 et Q4 sont volontairement qualitatives. Une première version demandait « au moins
2 directions effectives » : seuil arbitraire, qui rejetait à tort deux ESN excellents
(NRMSE ~0.1). Les quantités `reach` et `see` restent calculées : elles serviront à
l'analyse QUANTITATIVE, pas au filtre.

Un système admis peut prévoir ; un système rejeté ne le peut pas (hypothèse à
valider). Le filtre juge un système À UN POINT DE FONCTIONNEMENT.

Notebook : exécuter d'abord les cellules des étapes 1 à 11.
La cellule exécute les milestones Z1 à Z4, puis : val = validate_filter()
"""

import time

import numpy as np

# Notebook : pas d'import entre fichiers.

NEUTRAL_TOL = 1e-5        # |mu| > 1 - 1e-5 : direction neutre (une lente dissipation donne ~1e-3)
EQ_STARTS = 6             # départs pour compter les équilibres

_T0 = time.perf_counter()


def milestone(tag, ok, msg):
    status = "OK  " if ok else "FAIL"
    print(f"[{tag}] [{status}] {msg:<62} ({time.perf_counter() - _T0:5.2f} s)")
    return ok


def result(msg):
    print(f"[RES] {msg}")


# ------------------------------------------------------------- outils
def relax(sub, s, max_steps=8000, tol=1e-9):
    """Laisse le système évoluer sans entrée jusqu'à l'arrêt. Renvoie (état, convergé ?)."""
    for _ in range(max_steps):
        s_new = sub.step(s, 0.0)
        if np.linalg.norm(s_new - s) < tol * (1 + np.linalg.norm(s)):
            return s_new, True
        s = s_new
    return s, False


def step_jacobian(sub, s, eps=1e-6):
    """Jacobienne de l'application « un pas d'entrée » (u = 0), différences centrées."""
    n = len(s)
    J = np.empty((n, n))
    for i in range(n):
        d = np.zeros(n); d[i] = eps
        J[:, i] = (sub.step(s + d, 0.0) - sub.step(s - d, 0.0)) / (2 * eps)
    return J


def input_vector(sub, s, eps=1e-6):
    """b = d(étape)/du : comment l'entrée pousse l'état à l'équilibre."""
    return (sub.step(s, eps) - sub.step(s, -eps)) / (2 * eps)


def observe_matrix(sub, s, eps=1e-6):
    n = len(s)
    return np.stack([(sub.observe(s + eps * np.eye(n)[i]) - sub.observe(s - eps * np.eye(n)[i])) / (2 * eps)
                     for i in range(n)], axis=1)


def _eff_rank(M):
    """Rang effectif (rapport de participation des valeurs singulières au carré)."""
    sv = np.linalg.svd(M, compute_uv=False) ** 2
    return float(sv.sum() ** 2 / (sv ** 2).sum()) if sv.sum() > 0 else 0.0


def structural_report(sub, seed=0, horizon=200):
    """La fiche du filtre, calculée sans jamais piloter le système."""
    rng = np.random.default_rng(seed)
    s_eq, conv = relax(sub, sub.init_state(rng))
    J = step_jacobian(sub, s_eq)
    mu = np.abs(np.linalg.eigvals(J))
    n_neutral = int(np.sum(mu > 1 - NEUTRAL_TOL))

    # Q2 : d'autres départs mènent-ils au même équilibre ?
    starts = [sub.random_state(rng) for _ in range(EQ_STARTS)]
    ends = [relax(sub, st)[0] for st in starts]
    # Échelle = taille de la région explorée. (Correction d'un bug : l'échelle était
    # |s_eq| seule ; pour un équilibre à l'origine, le seuil devenait ~1e-12 et le
    # moindre résidu numérique comptait comme un « autre état ».)
    scale = max(np.linalg.norm(s_eq), np.mean([np.linalg.norm(st) for st in starts]))
    distinct = [s_eq]
    for e in ends:
        if all(np.linalg.norm(e - d) > 1e-3 * scale for d in distinct):
            distinct.append(e)

    # Q3 / Q4 : l'entrée atteint-elle, la lecture voit-elle ? (sur `horizon` pas)
    b, Cm = input_vector(sub, s_eq), observe_matrix(sub, s_eq)
    K, v = [], b.copy()
    for _ in range(horizon):
        K.append(v); v = J @ v
    reach = _eff_rank(np.array(K).T)
    O, w = [], Cm.copy()
    for _ in range(horizon):
        O.append(w); w = w @ J
    see = _eff_rank(np.vstack(O))

    rep = {"converged": conv, "rho": float(mu.max()), "n_neutral": n_neutral,
           "n_equilibria": len(distinct), "reach": reach, "see": see,
           "slowest": float(-1 / np.log(min(mu.max(), 1 - 1e-12)))}       # temps le plus long
    rep["Q1"], rep["Q2"] = n_neutral == 0, len(distinct) == 1
    rep["Q3"], rep["Q4"] = np.linalg.norm(b) > 1e-12, np.linalg.norm(Cm) > 1e-12
    rep["admis"] = rep["Q1"] and rep["Q2"] and rep["Q3"] and rep["Q4"]
    return rep


# ------------------------------------------------- Coulomb à symétrie brisée
class CoulombBroken(CoulombPositions):
    """Puits elliptique (k_x != k_y) et quelques charges fixes (ancrages) :
    la symétrie de rotation du puits isotrope disparaît."""

    def __init__(self, k_ratio=1.75, n_pins=0, pin_charge=1.0, **kw):
        super().__init__(**kw)
        rng = np.random.default_rng(kw.get("seed", 0) + 500)
        self.kxy = np.array([self.k_well, self.k_well * k_ratio])
        self.pins = rng.uniform(-2, 2, (n_pins, 2))
        self.qp = rng.choice([-1, 1], n_pins) * pin_charge

    def accel(self, pos, vel, u):
        a = self.coulomb_force(pos) - self.kxy * pos - self.gamma * vel
        if len(self.pins):
            d = pos[:, None, :] - self.pins[None]                       # (N, P, 2)
            inv = (np.sum(d ** 2, -1) + self.eps ** 2) ** -1.5
            a += self.k * np.sum((self.q * self.qp[None, :] * inv)[:, :, None] * d, axis=1)
        a[:, 0] += self.q[:, 0] * self.E0 * u
        return a


# ================================================================= milestones
def self_test():
    print("=" * 84)
    print("Étape 12 — validation des outils du filtre")
    print("=" * 84)
    # Z1 : la jacobienne numérique de l'ESN = la formule à la main
    sub = ESNSubstrate(rho=0.9)
    s_eq, conv = relax(sub, sub.init_state())
    e = sub.esn
    pre = e.W @ s_eq + e.b
    J_true = (1 - e.alpha) * np.eye(e.N) + e.alpha * (1 - np.tanh(pre) ** 2)[:, None] * e.W
    err = np.abs(step_jacobian(sub, s_eq) - J_true).max()
    milestone("Z1", conv and err < 1e-8, f"ESN : jacobienne numérique = analytique (écart {err:.1e})")

    # Z2 : le Coulomb isotrope SANS champ a un mode neutre, et c'est la rotation
    cou = CoulombPositions(damping=1.0, E0=0.0)
    s_c, _ = relax(cou, cou.init_state())
    w, V = np.linalg.eig(step_jacobian(cou, s_c))
    i = np.argmax(np.abs(w))
    pos = s_c[: 2 * cou.n].reshape(-1, 2)
    rot = np.concatenate([np.stack([-pos[:, 1], pos[:, 0]], 1).ravel(), np.zeros(2 * cou.n)])
    overlap = abs(np.real(V[:, i]) @ rot) / (np.linalg.norm(np.real(V[:, i])) * np.linalg.norm(rot))
    milestone("Z2", abs(abs(w[i]) - 1) < NEUTRAL_TOL and overlap > 0.99,
              f"Coulomb isotrope : |mu| = {abs(w[i]):.7f}, et c'est la rotation (recouvrement {overlap:.3f})")

    # Z3 : briser la symétrie fait disparaître ce mode neutre
    br = CoulombBroken(damping=1.0, E0=0.0)
    rb = structural_report(br)
    milestone("Z3", rb["n_neutral"] == 0, f"puits elliptique : {rb['n_neutral']} mode neutre (0 attendu), "
                                         f"|mu| max {rb['rho']:.5f}")

    # Z4 : l'ESN passe le filtre (un seul équilibre, pas de mode neutre)
    r = structural_report(ESNSubstrate(rho=0.9))
    milestone("Z4", r["admis"], f"ESN admis : {r['n_equilibria']} équilibre, {r['n_neutral']} mode neutre, "
                                f"atteinte {r['reach']:.1f}, vision {r['see']:.1f}")
    print("=" * 84)


def validate_filter(show=True):
    """Le filtre prédit-il la prévision ? Verdict calculé AVANT, performance APRÈS."""
    print("=" * 84)
    print("Validation du filtre : verdict structurel contre prévision réelle (Mackey-Glass)")
    print("=" * 84)
    s = normalize(mackey_glass())
    systems = [("ESN", g, ESNSubstrate(seed=g)) for g in range(3)]
    systems += [("Ressorts", g, MassSpringSubstrate(amp=0.3, damping=1.0, seed=g)) for g in range(12)]
    systems += [("Coulomb isotrope", g, CoulombPositions(damping=1.0, E0=0.25, seed=g)) for g in range(2)]
    systems += [("Coulomb elliptique", g, CoulombBroken(damping=1.0, E0=0.25, seed=g)) for g in range(2)]
    systems += [("Coulomb ellip.+ancrages", g, CoulombBroken(damping=1.0, E0=0.25, n_pins=6, seed=g))
                for g in range(2)]
    rows = []
    print(f"   {'système':<24}{'g':>3}  {'neutres':>7} {'équil.':>6} {'atteinte':>8} {'vision':>6}  "
          f"{'verdict':<8}{'h=10':>6}{'h=20':>6}")
    for name, g, sub in systems:
        rep = structural_report(sub, seed=g)                     # 1) le verdict, sans entrée
        nr10, nr20 = forecast_nrmse(sub, s, 10, g), forecast_nrmse(sub, s, 20, g)   # 2) la réalité
        rows.append((name, g, rep, nr10, nr20))
        print(f"   {name:<24}{g:>3}  {rep['n_neutral']:>7} {rep['n_equilibria']:>6} {rep['reach']:>8.1f} "
              f"{rep['see']:>6.1f}  {'ADMIS' if rep['admis'] else 'rejeté':<8}{nr10:6.2f}{nr20:6.2f}")
    milestone("Z5", all(np.isfinite(r[3]) for r in rows), f"{len(rows)} systèmes jugés puis testés")

    adm = np.array([r[2]["admis"] for r in rows])
    perf = np.array([(r[3] + r[4]) / 2 for r in rows])
    if adm.any() and (~adm).any():
        result(f"admis  : NRMSE moyenne {perf[adm].mean():.2f} (médiane {np.median(perf[adm]):.2f}, n={adm.sum()})")
        result(f"rejetés: NRMSE moyenne {perf[~adm].mean():.2f} (médiane {np.median(perf[~adm]):.2f}, n={(~adm).sum()})")
        worst_admitted, best_rejected = perf[adm].max(), perf[~adm].min()
        result(f"pire admis {worst_admitted:.2f}, meilleur rejeté {best_rejected:.2f} : "
               + ("séparation parfaite" if worst_admitted < best_rejected else "les deux groupes se chevauchent"))
        k = int(np.sum(np.cumsum(adm[np.argsort(perf)]) == np.arange(1, len(perf) + 1)))
        result(f"les {k} meilleurs systèmes sont tous admis (condition nécessaire, pas suffisante)")
    for q in ("Q1", "Q2", "Q3", "Q4"):
        fail = np.array([not r[2][q] for r in rows])
        if fail.any():
            result(f"{q} échoue pour {fail.sum()} systèmes : NRMSE moyenne {perf[fail].mean():.2f}")
    print("=" * 84)
    return rows


if __name__ == "__main__":
    self_test()
