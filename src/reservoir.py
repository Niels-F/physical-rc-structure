"""
Étape 2a — Un réservoir minimal (ESN à fuite), la formule du notebook :

    r(t+1) = (1 - alpha) r(t) + alpha * tanh(W r(t) + W_in u(t) + b)

`step(r, u)` ne garde pas d'état interne : on peut donc faire avancer deux
copies (r_a et r_b) avec les MÊMES matrices et la MÊME entrée, ce qu'exige le
Lyapunov conditionnel.

Lancer `python reservoir.py` exécute les milestones R1 à R5.
"""

import time

import numpy as np

from complexity_vector import complexity_vector, conditional_lyapunov

_T0 = time.perf_counter()


def milestone(tag, ok, msg):
    status = "OK  " if ok else "FAIL"
    print(f"[{tag}] [{status}] {msg:<62} ({time.perf_counter() - _T0:5.2f} s)")
    return ok


class ESN:
    def __init__(self, N=50, rho=0.9, alpha=0.3, input_scale=1.0, bias=0.1, seed=0):
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((N, N))
        self.W = W * rho / np.max(np.abs(np.linalg.eigvals(W)))   # rayon spectral = rho
        self.W_in = rng.uniform(-1, 1, N) * input_scale
        self.b = rng.uniform(-1, 1, N) * bias
        self.N, self.rho, self.alpha = N, rho, alpha

    def step(self, r, u):
        """Un pas de temps : nouvel état à partir de l'état r et de l'entrée u."""
        return (1 - self.alpha) * r + self.alpha * np.tanh(self.W @ r + self.W_in * u + self.b)

    def run(self, u, r0=None):
        """Fait tourner le réservoir sur toute la séquence u. Renvoie X (T, N)."""
        r = np.zeros(self.N) if r0 is None else r0.copy()
        X = np.empty((len(u), self.N))
        for t, ut in enumerate(u):
            r = self.step(r, ut)
            X[t] = r
        return X


def lyapunov_pair(esn, u, delta=1e-6, warmup=500, seed=1):
    """Protocole du lambda conditionnel : on chauffe le réservoir, on crée une
    copie décalée de delta, puis on pilote les deux avec la même entrée."""
    r = esn.run(u[:warmup])[-1]                                   # fin du transitoire
    d0 = np.random.default_rng(seed).standard_normal(esn.N)
    Xa = esn.run(u[warmup:], r)
    Xb = esn.run(u[warmup:], r + delta * d0 / np.linalg.norm(d0))
    return conditional_lyapunov(Xa, Xb), Xa, Xb


def self_test():
    print("=" * 84)
    print("Étape 2a — validation du réservoir")
    print("=" * 84)
    rng = np.random.default_rng(0)
    u = rng.uniform(-1, 1, 3000)                                  # entrée i.i.d.

    esn = ESN(rho=0.9)
    rho_mes = np.max(np.abs(np.linalg.eigvals(esn.W)))
    milestone("R1", abs(rho_mes - 0.9) < 1e-9, f"rayon spectral mesuré = {rho_mes:.4f} (cible 0.9)")

    X = esn.run(u)
    milestone("R2", np.all(np.isfinite(X)) and np.abs(X).max() <= 1,
              f"3000 pas, états finis et dans [-1, 1] (max |r| = {np.abs(X).max():.2f})")

    # R3 : propriété d'écho, deux départs très différents finissent confondus
    Xa = esn.run(u, rng.uniform(-1, 1, esn.N))
    Xb = esn.run(u, rng.uniform(-1, 1, esn.N))
    gap = np.linalg.norm(Xa[-1] - Xb[-1])
    milestone("R3", gap < 1e-8, f"écho (rho=0.9) : écart final = {gap:.1e} (départs aléatoires)")

    # R4 : lambda conditionnel négatif en régime d'écho, positif en régime chaotique
    lam_echo, _, _ = lyapunov_pair(esn, u)
    chaos = ESN(rho=2.0, alpha=1.0, input_scale=0.05)
    lam_chaos, _, _ = lyapunov_pair(chaos, u)
    milestone("R4", lam_echo < 0 < lam_chaos,
              f"lambda : écho={lam_echo:+.3f}, rho=2 faible entrée={lam_chaos:+.3f}")

    # R5 : le vecteur de complexité sur les états du réservoir
    C = complexity_vector(X[500:], D=4)                           # transitoire retiré
    finite = all(np.isfinite(v) for k, v in C.items() if k not in ("lam", "consist"))
    milestone("R5", finite, "vecteur de complexité calculé sur les états")
    print("   C =", {k: round(float(v), 3) for k, v in C.items()})
    print("=" * 84)


if __name__ == "__main__":
    self_test()
