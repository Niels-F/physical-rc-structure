"""
Étape 13 — Confirmation : le filtre FIGÉ, sur des systèmes jamais vus.

Le filtre de l'étape 12 n'est plus modifié. On l'applique à :
  - 24 nouvelles géométries de ressorts (graines 12 à 35), mêmes réglages ;
  - 3 nouveaux ESN (graines 3 à 5) ;
  - un NOUVEAU type de substrat : une chaîne de 50 masses sur une ligne,
      m x_i'' = k_c (x_{i+1} - 2 x_i + x_{i-1}) - V'(x_i) - gamma x_i' + w_i u
    en trois variantes construites pour tester chaque condition :
      a) puits simple, extrémités fixées      -> PRÉDIT ADMIS
      b) double puits (V bistable), fixées     -> PRÉDIT REJETÉ par Q2 (multistable)
      c) sans potentiel, extrémités libres     -> PRÉDIT REJETÉ par Q1 (translation neutre)

Critères de réussite, fixés AVANT l'exécution :
  K1  NRMSE médiane des admis < NRMSE médiane des rejetés
  K2  parmi le quart des systèmes qui prévoient le mieux, au moins 75 % sont admis
  K3  les trois variantes de chaîne reçoivent le verdict prédit

Notebook : exécuter d'abord les cellules des étapes 1 à 12.
La cellule exécute les milestones H1 et H2, puis : conf = confirm()   (~25 min)
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


class ChainSubstrate(Substrate):
    """Chaîne de n masses (déplacement 1D). V(x) = a x²/2 + b x⁴/4 :
    a > 0 puits simple, a < 0 double puits. fixed=False : extrémités libres."""

    def __init__(self, n=50, a=0.5, b=0.5, k_c=1.0, damping=0.5, amp=0.3, fixed=True,
                 dt=0.05, n_sub=4, seed=0):
        rng = np.random.default_rng(seed)
        self.n, self.a, self.b, self.k_c, self.gamma = n, a, b, k_c, damping
        self.fixed, self.dt, self.n_sub = fixed, dt, n_sub
        self.w_in = rng.uniform(-1, 1, n) * amp
        self.n_obs = n

    def accel(self, x, v, u):
        if self.fixed:
            xp = np.concatenate([[0.0], x, [0.0]])                       # murs à 0
        else:
            xp = np.concatenate([[x[0]], x, [x[-1]]])                    # bords libres
        lap = xp[2:] - 2 * x + xp[:-2]
        return self.k_c * lap - self.a * x - self.b * x ** 3 - self.gamma * v + self.w_in * u

    def init_state(self, rng=None):
        x = 0.01 * np.sin(np.arange(self.n))                             # pas exactement 0 :
        return np.concatenate([x, np.zeros(self.n)])                     # 0 est instable si a < 0

    def random_state(self, rng):
        return np.concatenate([rng.uniform(-1, 1, self.n), np.zeros(self.n)])

    def step(self, s, u):
        x, v = s[: self.n], s[self.n:]
        acc = self.accel(x, v, u)
        for _ in range(self.n_sub):
            v_half = v + 0.5 * acc * self.dt
            x = x + v_half * self.dt
            acc = self.accel(x, v_half, u)
            v = v_half + 0.5 * acc * self.dt
        return np.concatenate([x, v])

    def observe(self, s):
        return s[: self.n]


CHAINS = {"chaîne a) puits simple": dict(a=0.5, b=0.5, fixed=True),
          "chaîne b) double puits": dict(a=-0.5, b=1.0, fixed=True),
          "chaîne c) libre": dict(a=0.0, b=0.0, fixed=False)}
PREDICTED = {"chaîne a) puits simple": True, "chaîne b) double puits": False, "chaîne c) libre": False}


def self_test():
    print("=" * 84)
    print("Étape 13 — validation de la chaîne")
    print("=" * 84)
    probs = sum((check_substrate(ChainSubstrate(**kw)) for kw in CHAINS.values()), [])
    milestone("H1", not probs, "les trois variantes de chaîne passent le contrôle technique")
    for p in probs:
        print("   -", p)
    # H2 : la physique de base. Double puits : minima à ±sqrt(-a/b) pour une masse isolée.
    ch = ChainSubstrate(k_c=0.0, **{k: v for k, v in CHAINS["chaîne b) double puits"].items()})
    x_eq, _ = relax(ch, np.concatenate([np.full(ch.n, 0.3), np.zeros(ch.n)]))
    target = np.sqrt(0.5 / 1.0)
    milestone("H2", np.allclose(x_eq[: ch.n], target, atol=1e-4),
              f"double puits sans couplage : équilibre à {x_eq[0]:.4f} (théorie {target:.4f})")
    print("=" * 84)


def confirm():
    print("=" * 84)
    print("Confirmation du filtre figé sur des systèmes jamais vus")
    print("=" * 84)
    print("   Prédictions écrites avant l'exécution :")
    for name, adm in PREDICTED.items():
        print(f"      {name:<26} -> {'ADMIS' if adm else 'REJETÉ'}")
    s = normalize(mackey_glass())
    systems = [("ESN", g, ESNSubstrate(seed=g)) for g in range(3, 6)]
    systems += [("Ressorts", g, MassSpringSubstrate(amp=0.3, damping=1.0, seed=g)) for g in range(12, 36)]
    systems += [(name, g, ChainSubstrate(seed=g, **kw)) for name, kw in CHAINS.items() for g in range(2)]
    rows = []
    print(f"\n   {'système':<26}{'g':>3}  {'neutres':>7} {'états':>6}  {'verdict':<8}{'h=10':>6}{'h=20':>6}")
    for name, g, sub in systems:
        rep = structural_report(sub, seed=g)
        nr = (forecast_nrmse(sub, s, 10, g), forecast_nrmse(sub, s, 20, g))
        rows.append((name, g, rep, np.mean(nr)))
        print(f"   {name:<26}{g:>3}  {rep['n_neutral']:>7} {rep['n_equilibria']:>6}  "
              f"{'ADMIS' if rep['admis'] else 'rejeté':<8}{nr[0]:6.2f}{nr[1]:6.2f}")
    milestone("H3", all(np.isfinite(r[3]) for r in rows), f"{len(rows)} systèmes jugés puis testés")

    adm = np.array([r[2]["admis"] for r in rows])
    perf = np.array([r[3] for r in rows])
    print()
    k1 = adm.any() and (~adm).any() and np.median(perf[adm]) < np.median(perf[~adm])
    result(f"K1 : médiane admis {np.median(perf[adm]):.2f} contre rejetés {np.median(perf[~adm]):.2f} "
           f"-> {'RÉUSSI' if k1 else 'ÉCHOUÉ'}")
    top = np.argsort(perf)[: max(1, len(perf) // 4)]
    frac = adm[top].mean()
    result(f"K2 : {adm[top].sum()}/{len(top)} des meilleurs sont admis ({frac:.0%}) "
           f"-> {'RÉUSSI' if frac >= 0.75 else 'ÉCHOUÉ'}")
    k3 = all(r[2]["admis"] == PREDICTED[r[0]] for r in rows if r[0] in PREDICTED)
    result(f"K3 : verdicts des chaînes conformes aux prédictions -> {'RÉUSSI' if k3 else 'ÉCHOUÉ'}")
    for name in PREDICTED:
        sel = [r for r in rows if r[0] == name]
        why = ", ".join(q for q in ("Q1", "Q2", "Q3", "Q4") if not sel[0][2][q]) or "aucune condition échouée"
        result(f"    {name:<26} {'ADMIS' if sel[0][2]['admis'] else 'rejeté'} ({why}), "
               f"NRMSE {np.mean([r[3] for r in sel]):.2f}")
    ms = np.array([r[0] == "Ressorts" for r in rows])
    if (adm & ms).any() and (~adm & ms).any():
        result(f"nouvelles géométries : {np.sum(adm & ms)} admises (médiane {np.median(perf[adm & ms]):.2f}), "
               f"{np.sum(~adm & ms)} rejetées (médiane {np.median(perf[~adm & ms]):.2f})")
    print("=" * 84)
    return rows


if __name__ == "__main__":
    self_test()
