"""
Étape 1 — Le vecteur de complexité C d'une dynamique.

Entrée : une série d'états X de forme (T, N) : T pas de temps, N canaux lus.
Sortie : un dictionnaire avec les composantes de C.

    H_perm   entropie de permutation normalisée (Bandt-Pompe)      [0, 1]
    C_JS     complexité statistique de Jensen-Shannon (Rosso)       [0, ~0.5]
    PR       dimension effective (rapport de participation) / N     [1/N, 1]
    C_N      complexité neurale gaussienne (Tononi-Sporns-Edelman)  >= 0
    C_N_form C_N rapporté à son maximum possible (forme de l'intégration) [0, 1]
    tau_med  temps de relaxation médian (en pas)                    > 0
    tau_sprd dispersion des temps de relaxation (écart-type de log tau)
    lam      exposant de Lyapunov conditionnel (si 2e trajectoire)  <0 : écho, >0 : chaos
    consist  consistance (si 3e trajectoire partie d'ailleurs)      1 : oublie son départ

Lancer `python complexity_vector.py` exécute les tests de validation (milestones).
"""

import time
from math import factorial

import numpy as np


# ---------------------------------------------------------------- milestones
_T0 = time.perf_counter()


def milestone(tag, ok, msg):
    """Affiche une étape de debug : [M1] [OK]  message   (temps écoulé)."""
    status = "OK  " if ok else "FAIL"
    print(f"[{tag}] [{status}] {msg:<62} ({time.perf_counter() - _T0:5.2f} s)")
    return ok


# ------------------------------------------------- 1. Bandt-Pompe : H et C_JS
def permutation_probs(x, D=5, tau=1, warn=True, rng=None):
    """Probabilités des motifs ordinaux de longueur D dans la série x.

    Égalités (données quantifiées, ex. 6-10 bits) : elles sont départagées au
    hasard. Sinon argsort les classe toujours dans le même ordre, ce qui crée
    de faux motifs favoris et gonfle C_JS (bruit sur 4 bits : C x48).
    """
    n = len(x) - (D - 1) * tau
    if warn and n < 10 * factorial(D):
        print(f"   ! attention : {n} fenêtres pour {factorial(D)} motifs, statistique faible")
    ties = np.mean(np.diff(x) == 0)
    if ties > 0:
        if warn and ties > 0.01:
            print(f"   ! attention : {ties:.1%} de valeurs consécutives égales, départagées au hasard")
        rng = np.random.default_rng(0) if rng is None else rng
        steps = np.diff(np.unique(x))
        x = x + rng.uniform(-0.25, 0.25, len(x)) * (steps.min() if len(steps) else 1.0)
    windows = np.stack([x[i * tau: i * tau + n] for i in range(D)], axis=1)
    patterns = np.argsort(windows, axis=1, kind="stable")
    codes = patterns @ (D ** np.arange(D))          # un entier par motif
    _, counts = np.unique(codes, return_counts=True)
    p = np.zeros(factorial(D))
    p[: len(counts)] = counts / counts.sum()        # motifs absents = 0
    return p


def _shannon(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def entropy_complexity(x, D=5, tau=1, warn=True):
    """Retourne (H normalisée, C_JS) pour une série 1D (Rosso et al. 2007)."""
    p = permutation_probs(x, D, tau, warn)
    M = len(p)
    pe = np.full(M, 1.0 / M)
    H = _shannon(p) / np.log(M)
    J = _shannon((p + pe) / 2) - _shannon(p) / 2 - _shannon(pe) / 2
    Q0 = -2.0 / (((M + 1) / M) * np.log(M + 1) - 2 * np.log(2 * M) + np.log(M))
    return H, Q0 * J * H


# ------------------------------------------------- 2. dimension effective
def participation_ratio(X, standardize=True):
    """(Σλ)² / Σλ² des valeurs propres, divisé par N.

    standardize=True : matrice de corrélation, insensible aux unités des canaux
    (sinon un canal en mV et un autre en V ne pèsent pas pareil).
    standardize=False : covariance brute, utile si tous les canaux ont la même
    unité et qu'un canal presque muet ne doit pas compter comme une direction.
    """
    M = np.corrcoef(X, rowvar=False) if standardize else np.cov(X, rowvar=False)
    lam = np.clip(np.linalg.eigvalsh(M), 0, None)
    return (lam.sum() ** 2 / (lam ** 2).sum()) / X.shape[1]


# ------------------------------------------------- 3. temps de relaxation
def timescales(X, max_lag=500, warn=True):
    """Pour chaque canal : retard après lequel |autocorrélation| ne remonte plus
    au-dessus de 1/e.

    On prend le DERNIER passage et la valeur absolue : un canal qui oscille
    (ex. mode de giration d'un skyrmion) passe vite sous 1/e mais y remonte à
    chaque période. Le premier passage le prendrait à tort pour un canal sans
    mémoire. Pour une décroissance monotone, les deux définitions coïncident.
    """
    taus, capped = [], 0
    for x in X.T:
        x = x - x.mean()
        f = np.fft.rfft(x, n=2 * len(x))
        acf = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
        acf = acf / acf[0]
        above = np.where(np.abs(acf) >= 1 / np.e)[0]
        tau = above[-1] + 1
        capped += tau > max_lag
        taus.append(min(tau, max_lag))
    if warn and capped:
        print(f"   ! attention : {capped} canal(aux) gardent leur mémoire au-delà de "
              f"max_lag={max_lag}, tau sous-estimé")
    taus = np.array(taus, float)
    return np.median(taus), np.std(np.log(taus))


# ------------------------------------------------- 4. complexité neurale
def _integration(R):
    """I(X) = -1/2 log det R pour des variables gaussiennes."""
    sign, logdet = np.linalg.slogdet(R)
    return -0.5 * logdet


def neural_complexity(X, n_max=24, n_samples=30, noise_floor=0.01, rng=None):
    """C_N = Σ_k [ (k/n) I(X) - <I(X_k)> ], sous-ensembles tirés au hasard.

    noise_floor : bruit de mesure, en fraction de l'écart-type de chaque canal
    (0.01 = 1 %, environ 7 bits). La matrice utilisée est exactement celle des
    canaux + ce bruit : les directions sous le plancher ne comptent plus.

    Renvoie (C_N brut, forme). La forme = C_N / ((n-1)/2 · I(X)) est dans [0, 1] :
    C_N brut grandit aussi avec la synchronie (49 pour 20 canaux synchronisés,
    contre 4.7 pour X_mix), la forme non (0.06 contre 0.16). C'est la forme qui
    est « basse aux deux extrêmes ».
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n = min(X.shape[1], n_max)
    cols = rng.choice(X.shape[1], n, replace=False)
    v = noise_floor ** 2
    R = (np.corrcoef(X[:, cols], rowvar=False) + v * np.eye(n)) / (1 + v)
    I_total = _integration(R)
    cn = 0.0
    for k in range(1, n):                           # k = n donne 0 : inutile
        Ik = np.mean([_integration(R[np.ix_(s, s)])
                      for s in (rng.choice(n, k, replace=False) for _ in range(n_samples))])
        cn += (k / n) * I_total - Ik
    # Avec T pas, des canaux indépendants montrent quand même une intégration
    # accidentelle d'environ n(n-1)/(4T). En dessous de 5 fois ce biais, il n'y a
    # rien à mettre en forme : le rapport de deux quasi-zéros serait du bruit.
    bias = n * (n - 1) / (4 * X.shape[0])
    form = cn / ((n - 1) / 2 * I_total) if I_total > 5 * bias else 0.0
    return cn, float(np.clip(form, 0, 1))


# ------------------------------------------------- 5. Lyapunov conditionnel
def conditional_lyapunov(Xa, Xb, floor=1e-10, ceil_frac=0.1):
    """Pente de log||Xa - Xb|| : deux trajectoires, même entrée, départs différents.

    Unité : par pas d'échantillonnage. Fenêtre d'ajustement : entre `floor`
    (erreurs d'arrondi ; pour des données physiques, mettre le niveau de bruit)
    et `ceil_frac` x la taille des FLUCTUATIONS de Xa (pas sa valeur absolue :
    un décalage constant, ex. une aimantation moyenne, ne doit pas élargir la
    fenêtre). Protocole conseillé : écart initial ~1e-6 x cette taille, pour
    laisser plusieurs décades de mesure dans les deux sens.
    """
    d = np.linalg.norm(Xa - Xb, axis=1)
    ceil = ceil_frac * np.linalg.norm(Xa - Xa.mean(axis=0), axis=1).mean() + floor
    ok = (d > floor) & (d < ceil)                   # zone exponentielle
    idx = np.where(ok)[0]
    if len(idx) < 5:
        return np.nan
    idx = idx[: np.argmax(np.diff(idx) > 1) + 1] if np.any(np.diff(idx) > 1) else idx
    return np.polyfit(idx, np.log(d[idx]), 1)[0] if len(idx) >= 5 else np.nan


# ------------------------------------------------- 6. consistance
def consistency(Xa, Xc):
    """Corrélation moyenne, canal par canal, entre deux réponses à la MÊME entrée
    parties de départs VRAIMENT différents (transitoire déjà retiré).

    lambda ne teste qu'une perturbation infinitésimale. Un système multistable
    peut avoir lambda ~ 0 et ne jamais oublier son départ : seule la consistance
    le voit (ESN : 1.000, Coulomb à 25 charges : 0.0 à 0.35).
    """
    c = [np.corrcoef(a, b)[0, 1] for a, b in zip(Xa.T, Xc.T) if a.std() > 0 and b.std() > 0]
    return float(np.mean(c)) if c else np.nan


# ------------------------------------------------- le vecteur complet
def complexity_vector(X, Xb=None, D=5, warn=True, Xc=None):
    """Calcule le vecteur C sur des états X (T, N). Xb : 2e trajectoire optionnelle
    (petite perturbation, pour lambda). Xc : 3e trajectoire optionnelle (même entrée,
    départ vraiment différent, pour la consistance).

    Les états doivent être pris APRÈS le transitoire (le code ne le retire pas).
    """
    X = np.atleast_2d(X.T).T                         # accepte aussi une série 1D
    if Xb is not None:
        Xb = np.atleast_2d(Xb.T).T                   # même forme que X, sinon
        if Xb.shape != X.shape:                      # X - Xb « broadcast » en (T, T)
            raise ValueError(f"X {X.shape} et Xb {Xb.shape} doivent avoir la même forme")
    live = X.std(axis=0) > 0                         # canaux constants (nœud mort,
    if not live.all():                               # tanh saturé) : NaN sinon
        if warn:
            print(f"   ! attention : {np.sum(~live)} canal(aux) constant(s) retiré(s)")
        X = X[:, live]
        Xb = Xb[:, live] if Xb is not None else None
        Xc = Xc[:, live] if Xc is not None else None
    HC = np.array([entropy_complexity(x, D, warn=warn and i == 0) for i, x in enumerate(X.T)])
    tau_med, tau_sprd = timescales(X, warn=warn)
    cn, cn_form = neural_complexity(X) if X.shape[1] > 1 else (0.0, 0.0)
    return {
        "H_perm": HC[:, 0].mean(),
        "C_JS": HC[:, 1].mean(),
        "PR": participation_ratio(X) if X.shape[1] > 1 else 1.0,
        "C_N": cn,
        "C_N_form": cn_form,
        "tau_med": tau_med,
        "tau_sprd": tau_sprd,
        "lam": conditional_lyapunov(X, Xb) if Xb is not None else np.nan,
        "consist": consistency(X, np.atleast_2d(Xc.T).T) if Xc is not None else np.nan,
    }


# ================================================= validation par milestones
def self_test(T=20000, seed=1):
    rng = np.random.default_rng(seed)
    print("=" * 84)
    print("Étape 1 — validation du vecteur de complexité sur des signaux connus")
    print("=" * 84)

    # Signaux de référence
    noise = rng.standard_normal(T)
    sine = np.sin(2 * np.pi * np.arange(T) / 50)
    logi = np.empty(T); logi[0] = 0.4
    for t in range(T - 1):
        logi[t + 1] = 4 * logi[t] * (1 - logi[t])
    milestone("M1", True, "signaux générés : bruit blanc, sinus, carte logistique r=4")

    # M2-M3 : plan complexité-entropie (Rosso 2007)
    Hn, Cn = entropy_complexity(noise)
    Hs, Cs = entropy_complexity(sine)
    Hl, Cl = entropy_complexity(logi)
    print(f"   bruit    H={Hn:.3f}  C={Cn:.3f}")
    print(f"   sinus    H={Hs:.3f}  C={Cs:.3f}")
    print(f"   logist.  H={Hl:.3f}  C={Cl:.3f}")
    milestone("M2", Hn > 0.99 and Cn < 0.01, "bruit : entropie maximale, complexité ~0")
    milestone("M3", Cl > 5 * Cn and Hl < Hn, "chaos : plus complexe et moins entropique que le bruit")

    # M4 : dimension effective
    X_indep = rng.standard_normal((T, 20))
    X_sync = np.outer(sine, np.ones(20)) + 1e-3 * rng.standard_normal((T, 20))
    pr_i, pr_s = participation_ratio(X_indep), participation_ratio(X_sync)
    milestone("M4", pr_i > 0.9 and pr_s < 0.1,
              f"PR : indépendants={pr_i:.2f} (~1), synchronisés={pr_s:.2f} (~1/N)")

    # M5 : complexité neurale aux trois régimes (le régime synchronisé manquait)
    common = rng.standard_normal((T, 1))
    X_mix = 0.6 * common + 0.8 * rng.standard_normal((T, 20))
    (cn_i, f_i), (cn_m, f_m), (cn_s, f_s) = (neural_complexity(X) for X in (X_indep, X_mix, X_sync))
    print(f"   C_N brut : indépendants={cn_i:.3f}  intégrés+ségrégés={cn_m:.2f}  synchronisés={cn_s:.2f}")
    print(f"   forme    : indépendants={f_i:.3f}  intégrés+ségrégés={f_m:.3f}  synchronisés={f_s:.3f}")
    milestone("M5", cn_i < 0.05 and f_m > 2 * f_s and f_m > 2 * f_i,
              "forme de C_N : basse aux deux extrêmes, haute entre les deux")

    # M6 : temps de relaxation sur un AR(1), tau théorique = -1/ln(a)
    a, ar = 0.95, np.zeros((T, 1))
    for t in range(T - 1):
        ar[t + 1] = a * ar[t] + rng.standard_normal()
    tau_th = -1 / np.log(a)
    tau_med, _ = timescales(ar)
    milestone("M6", abs(tau_med - tau_th) < 0.25 * tau_th,
              f"tau AR(1) : mesuré={tau_med:.1f}, théorique={tau_th:.1f}")

    # M7-M8 : Lyapunov conditionnel, valeurs exactes connues
    xa, xb = np.empty(60), np.empty(60)
    xa[0], xb[0] = 0.4, 0.4 + 1e-9
    for t in range(59):
        xa[t + 1], xb[t + 1] = 4 * xa[t] * (1 - xa[t]), 4 * xb[t] * (1 - xb[t])
    lam_chaos = conditional_lyapunov(xa[:, None], xb[:, None], ceil_frac=1e-2)
    milestone("M7", abs(lam_chaos - np.log(2)) < 0.25,
              f"lambda logistique : {lam_chaos:.3f} (théorie ln2 = 0.693)")

    t = np.arange(60)
    ca, cb = (0.5 ** t)[:, None] * 1.0, (0.5 ** t)[:, None] * 2.0
    lam_stable = conditional_lyapunov(np.ones((60, 1)) + ca, np.ones((60, 1)) + cb)
    milestone("M8", abs(lam_stable - np.log(0.5)) < 0.05,
              f"lambda contractant : {lam_stable:.3f} (théorie ln0.5 = -0.693)")

    # M9 : le vecteur complet sur un cas multicanal
    C = complexity_vector(X_mix)
    finite = all(np.isfinite(v) for k, v in C.items() if k not in ("lam", "consist"))
    milestone("M9", finite, "vecteur complet calculé sans NaN")
    print("   C =", {k: round(float(v), 3) for k, v in C.items()})
    print("=" * 84)


if __name__ == "__main__":
    self_test()
