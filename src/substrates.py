"""
Étape 3 — L'usine à réservoirs : une interface commune à tous les substrats.

Un substrat sait faire quatre choses (et une cinquième, optionnelle) :
    init_state(rng)          un état de départ
    step(s, u)               un pas de temps sous l'entrée u (sans modifier s)
    observe(s)               les canaux LUS, vecteur de taille n_obs (pas tout l'état)
    perturb(s, delta, rng)   une copie de s décalée d'une distance delta
    random_state(rng)        un départ vraiment différent (par défaut : décalé de 1)

Tout le reste est générique : run, run_pair, le vecteur de complexité, live_time.
`check_substrate` est le contrôle technique que tout nouveau substrat doit passer.

La cellule exécute les milestones S1 à S5 à la fin.
"""

import time

import numpy as np

# Notebook : pas d'import entre fichiers. Exécuter d'abord les cellules des étapes 1
# (complexity_vector, conditional_lyapunov) et 2a (ESN, lyapunov_pair).

_T0 = time.perf_counter()


def milestone(tag, ok, msg):
    status = "OK  " if ok else "FAIL"
    print(f"[{tag}] [{status}] {msg:<62} ({time.perf_counter() - _T0:5.2f} s)")
    return ok


# ------------------------------------------------------------- l'interface
class Substrate:
    n_obs = None                                       # nombre de canaux lus

    def init_state(self, rng=None):
        raise NotImplementedError

    def step(self, s, u):
        raise NotImplementedError

    def observe(self, s):
        raise NotImplementedError

    def random_state(self, rng):
        """Un départ VRAIMENT différent (pas une petite perturbation) : sert à tester
        la consistance et à évaluer la lecture sur un autre départ. Par défaut :
        l'état initial décalé d'une distance 1 ; chaque substrat peut faire mieux."""
        return self.perturb(self.init_state(rng), 1.0, rng)

    def perturb(self, s, delta, rng):
        """Par défaut : direction aléatoire, norme exactement delta."""
        d = rng.standard_normal(np.shape(s))
        return s + delta * d / np.linalg.norm(d)


def run(sub, u, s0=None, rng=None):
    """Fait tourner le substrat sur la séquence u. Renvoie (X (T, n_obs), état final)."""
    s = sub.init_state(rng) if s0 is None else s0
    X = np.empty((len(u), sub.n_obs))
    for t, ut in enumerate(u):
        s = sub.step(s, ut)
        X[t] = sub.observe(s)
    return X, s


def run_pair(sub, u, delta=1e-6, warmup=500, seed=1):
    """Deux copies, même entrée : on chauffe, on décale de delta, on fait tourner."""
    _, s = run(sub, u[:warmup])                        # fin du transitoire
    sb = sub.perturb(s, delta, np.random.default_rng(seed))
    Xa, _ = run(sub, u[warmup:], s)
    Xb, _ = run(sub, u[warmup:], sb)
    return Xa, Xb


# ------------------------------------------------------ le contrôle technique
def check_substrate(sub, T=300, seed=0):
    """Vérifie le contrat de l'interface. Renvoie la liste des problèmes (vide = OK)."""
    problems = []
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, T)
    try:
        s0 = sub.init_state(np.random.default_rng(seed))
        o = np.asarray(sub.observe(s0))
        if o.shape != (sub.n_obs,):
            problems.append(f"observe renvoie {o.shape}, attendu ({sub.n_obs},)")
        s_copy = np.copy(s0)
        s1 = sub.step(s0, u[0])
        if not np.array_equal(s0, s_copy):
            problems.append("step modifie l'état reçu (il doit en renvoyer un nouveau)")
        if np.shape(s1) != np.shape(s0):
            problems.append(f"step change la forme de l'état {np.shape(s0)} -> {np.shape(s1)}")
        X1, _ = run(sub, u, rng=np.random.default_rng(seed))
        X2, _ = run(sub, u, rng=np.random.default_rng(seed))
        if not np.all(np.isfinite(X1)):
            problems.append(f"valeurs non finies dès le pas {np.argmax(~np.isfinite(X1).all(1))}")
        elif not np.array_equal(X1, X2):
            problems.append("non déterministe : même graine, même entrée, résultats différents")
        sr = np.copy(s_copy)                           # état neuf : step a pu abîmer s0
        sp = sub.perturb(sr, 1e-6, np.random.default_rng(seed))
        dist = np.linalg.norm(np.ravel(sp) - np.ravel(s_copy))
        if not np.isclose(dist, 1e-6, rtol=1e-3):
            problems.append(f"perturb décale de {dist:.2e} au lieu de 1e-6")
        if not np.array_equal(sr, s_copy):
            problems.append("perturb modifie l'état reçu")
    except Exception as e:                             # une erreur est aussi un problème
        problems.append(f"exception : {type(e).__name__}: {e}")
    return problems


# --------------------------------------------------------- premier substrat
class ESNSubstrate(Substrate):
    """L'ESN de reservoir.py derrière l'interface. obs_idx : nœuds lus (tous par défaut)."""

    def __init__(self, obs_idx=None, **kw):
        self.esn = ESN(**kw)
        self.obs_idx = np.arange(self.esn.N) if obs_idx is None else np.asarray(obs_idx)
        self.n_obs = len(self.obs_idx)

    def init_state(self, rng=None):
        return np.zeros(self.esn.N)

    def random_state(self, rng):
        return rng.uniform(-1, 1, self.esn.N)                   # tout l'intervalle de tanh

    def step(self, s, u):
        return self.esn.step(s, u)

    def observe(self, s):
        return s[self.obs_idx]


class _BrokenSubstrate(ESNSubstrate):
    """Volontairement cassé : sert à vérifier que le contrôle technique détecte les fautes."""

    def step(self, s, u):
        s[0] += 1e-3                                   # faute 1 : modifie l'état reçu
        return np.full_like(s, np.nan) if u > 0.9 else self.esn.step(s, u)    # faute 2 : NaN


# ================================================================= milestones
def self_test():
    print("=" * 84)
    print("Étape 3 — validation de l'interface des substrats")
    print("=" * 84)
    u = np.random.default_rng(0).uniform(-1, 1, 3000)
    sub = ESNSubstrate(rho=0.9)

    problems = check_substrate(sub)
    milestone("S1", not problems, "ESNSubstrate passe le contrôle technique")
    for p in problems:
        print("   -", p)

    X_old = sub.esn.run(u)
    X_new, _ = run(sub, u)
    milestone("S2", np.array_equal(X_old, X_new), "run() identique bit à bit à l'ancien ESN.run()")

    lam_old, Xa_old, Xb_old = lyapunov_pair(sub.esn, u)
    Xa, Xb = run_pair(sub, u)
    lam_new = conditional_lyapunov(Xa, Xb)
    same = np.array_equal(Xa, Xa_old) and np.array_equal(Xb, Xb_old) and lam_new == lam_old
    milestone("S3", same, f"run_pair() identique à lyapunov_pair() (lambda = {lam_new:+.4f})")

    part = ESNSubstrate(obs_idx=range(10), rho=0.9)
    Xp, _ = run(part, u)
    C = complexity_vector(Xp[500:], D=4, warn=False)
    ok = Xp.shape == (3000, 10) and np.array_equal(Xp, X_new[:, :10]) and np.isfinite(C["PR"])
    milestone("S4", ok, f"lecture partielle : 10 nœuds sur 50 lus, PR = {C['PR']:.3f}")

    problems = check_substrate(_BrokenSubstrate(rho=0.9))
    found = (any("step modifie" in p for p in problems) and any("non finies" in p for p in problems)
             and not any("perturb" in p for p in problems))
    milestone("S5", found, "substrat cassé : les 2 fautes trouvées, aucune fausse accusation")
    for p in problems:
        print("   -", p)
    print("=" * 84)


if __name__ == "__main__":
    self_test()
