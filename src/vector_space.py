"""
Étape 4 — L'espace des vecteurs : normalisation fixe et coordonnées parallèles.

Les axes sont normalisés par des bornes connues à l'avance (pas par une
population) : ils ne bougent pas pendant le direct, et deux systèmes mesurés à
des moments différents restent comparables.

Notebook : exécuter d'abord les cellules des étapes 1, 2a et 3.
La cellule exécute les milestones N1 à N5 à la fin.
"""

import time

import numpy as np

# Notebook : pas d'import entre fichiers (complexity_vector, ESNSubstrate, run
# viennent des cellules précédentes).

MAX_LAG = 500        # même borne que timescales()
TAU_SPRD_MAX = 2.0   # écart-type de log(tau) = 2 : échelles de temps étalées sur ~x50
LAM_SCALE = 0.1      # |lambda| = 0.1 / pas : déjà loin de la frontière écho/chaos

# (clé du vecteur, étiquette d'axe, ce que veulent dire 0 et 1)
AXES = [
    ("H_perm", "H\nentropie", "ordre → hasard"),
    ("C_JS", "C_JS\ncomplexité", "0 → 0.5"),
    ("PR", "PR\ndimension", "1 direction → toutes"),
    ("C_N_form", "C_N forme\nintrications", "aucune → max"),
    ("tau_med", "tau médian\n(log)", f"1 → {MAX_LAG} pas"),
    ("tau_sprd", "tau étalement", f"0 → {TAU_SPRD_MAX}"),
    ("lam", "lambda\n(milieu = 0)", "écho ← | → chaos"),
    ("consist", "consistance", "garde son départ → l'oublie"),
]

_T0 = time.perf_counter()


def milestone(tag, ok, msg):
    status = "OK  " if ok else "FAIL"
    print(f"[{tag}] [{status}] {msg:<62} ({time.perf_counter() - _T0:5.2f} s)")
    return ok


def normalize_vector(C):
    """Vecteur brut (dict) -> tableau de len(AXES) valeurs dans [0, 1] (NaN si absent)."""
    tau = max(C["tau_med"], 1.0)
    vals = {
        "H_perm": C["H_perm"],
        "C_JS": C["C_JS"] / 0.5,
        "PR": C["PR"],
        "C_N_form": C["C_N_form"],
        "tau_med": np.log(tau) / np.log(MAX_LAG),
        "tau_sprd": C["tau_sprd"] / TAU_SPRD_MAX,
        "lam": 0.5 + 0.5 * np.tanh(C["lam"] / LAM_SCALE),   # NaN reste NaN
        "consist": C.get("consist", np.nan),                 # corrélation : < 0 ramené à 0
    }
    return np.clip(np.array([vals[k] for k, _, _ in AXES], float), 0, 1)


def draw_axes(ax):
    """Dessine le cadre des coordonnées parallèles (axes verticaux, étiquettes)."""
    n = len(AXES)
    for i, (key, label, meaning) in enumerate(AXES):
        ax.axvline(i, color="0.75", lw=1, zorder=0)
        ax.annotate(meaning, xy=(i, 0), xycoords=("data", "axes fraction"),     # sous le nom
                    xytext=(0, -30), textcoords="offset points", ha="center", va="top",
                    fontsize=7, color="0.4")
    i_lam = [k for k, _, _ in AXES].index("lam")
    ax.plot([i_lam - 0.15, i_lam + 0.15], [0.5, 0.5], color="tab:red", lw=2)  # frontière écho/chaos
    ax.set_xticks(range(n))
    ax.set_xticklabels([label for _, label, _ in AXES], fontsize=8)
    ax.set(xlim=(-0.3, n - 0.7), ylim=(-0.02, 1.1), ylabel="valeur normalisée")


def plot_parallel(vectors, labels, ax=None, title="Coordonnées parallèles"):
    """Version statique : un dict brut par système, une ligne par système."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.5))
    draw_axes(ax)
    for C, lab in zip(vectors, labels):
        ax.plot(range(len(AXES)), normalize_vector(C), "o-", lw=2, label=lab)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(title)
    ax.figure.tight_layout()
    return ax


def self_test():
    print("=" * 84)
    print("Étape 4 — validation de la normalisation fixe")
    print("=" * 84)
    rng, T = np.random.default_rng(2), 20000

    X_noise = rng.standard_normal((T, 20))
    v = normalize_vector(complexity_vector(X_noise, warn=False) | {"lam": np.nan})
    milestone("N1", v[0] > 0.99 and v[1] < 0.01 and v[2] > 0.95,
              f"bruit blanc : H={v[0]:.2f} (1), C={v[1]:.2f} (0), PR={v[2]:.2f} (1)")

    lams = np.array([-1, -0.1, -0.01, 0, 0.01, 0.1, 1])
    base = complexity_vector(X_noise, warn=False)
    i_lam = [k for k, _, _ in AXES].index("lam")
    yl = np.array([normalize_vector(base | {"lam": l})[i_lam] for l in lams])
    milestone("N2", yl[3] == 0.5 and np.all(np.diff(yl) > 0),
              f"lambda=0 -> {yl[3]:.2f} (milieu), croissant de {yl[0]:.3f} à {yl[-1]:.3f}")

    t1 = normalize_vector(base | {"tau_med": 1.0})[4]
    tM = normalize_vector(base | {"tau_med": float(MAX_LAG)})[4]
    milestone("N3", t1 == 0 and tM == 1, f"tau : 1 pas -> {t1:.0f}, {MAX_LAG} pas -> {tM:.0f}")

    u = rng.uniform(-1, 1, 3000)
    systems = {"bruit blanc (20 canaux)": X_noise[:3000]}
    for rho in (0.5, 0.9, 1.2):
        Xa, Xb = run_pair(ESNSubstrate(rho=rho), u)
        systems[f"ESN rho={rho}"] = (Xa, Xb)
    vectors, labels = [], []
    for name, X in systems.items():
        C = (complexity_vector(X[0], X[1], D=4, warn=False) if isinstance(X, tuple)
             else complexity_vector(X, D=4, warn=False) | {"lam": np.nan})
        vectors.append(C); labels.append(name)
    norm = np.array([normalize_vector(C) for C in vectors])
    inside = np.all((norm >= 0) & (norm <= 1) | np.isnan(norm))
    milestone("N4", inside and norm[0, 3] < 0.05,
              "4 systèmes dans [0, 1], et bruit court : forme de C_N ~0")
    for name, row in zip(labels, norm):
        print(f"   {name:<24}" + " ".join(f"{x:5.2f}" for x in row))

    import matplotlib.pyplot as plt
    ax = plot_parallel(vectors, labels, title="Quatre systèmes dans l'espace des vecteurs")
    ax.figure.savefig("parallel_static.png", dpi=110)
    milestone("N5", True, "figure statique enregistrée : parallel_static.png")
    print("=" * 84)
    plt.show()


if __name__ == "__main__":
    self_test()
