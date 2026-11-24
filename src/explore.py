"""
Étape 16 — Exploration : que disent les degrés structurels sur les systèmes DÉJÀ VUS ?

Pour chaque système : le profil structurel (étape 15), calculé d'abord et sans série,
puis l'erreur de prévision de Mackey-Glass à 10 et 20 pas (étape 11).

L'ensemble d'exploration ne contient que des systèmes dont la performance avait déjà
été regardée (validation et confirmation du filtre, essai sur l'amplitude de la chaîne).
On a donc le droit d'y chercher librement ; ce qu'on y trouve est ensuite GELÉ, puis
testé à l'étape 17 sur des systèmes jamais vus.

Deux choses sont choisies ici :
  1. le seuil theta de la règle graduée « admis si aucun mode neutre et d2 < theta » ;
  2. un prédicteur de l'erreur à partir du profil, parmi deux candidats fixés d'avance
     (régression ridge ou 5 plus proches voisins), jugés en retirant un système à la fois.

Notebook : exécuter d'abord les cellules des étapes 1 à 15.
Puis : rows = run_catalog(EXPLORATION, "exploration.jsonl")     (~45 min)
       frozen = explore_analysis(rows)
"""

import json
import os
import time

import numpy as np

# Notebook : pas d'import entre fichiers.

_T0 = time.perf_counter()
GOOD = 0.25                       # « bonne prévision » : NRMSE moyenne < 0.25
FEATURES = ["log_mc", "log_nl", "log_nl2", "log_d2", "neutral"]


def milestone(tag, ok, msg):
    status = "OK  " if ok else "FAIL"
    print(f"[{tag}] [{status}] {msg:<62} ({time.perf_counter() - _T0:5.2f} s)")
    return ok


def result(msg):
    print(f"[RES] {msg}")


# ------------------------------------------------------------- le catalogue
def make_system(family, seed, **kw):
    if family == "ESN":
        return ESNSubstrate(seed=seed, **kw)
    if family == "Ressorts":
        return MassSpringSubstrate(seed=seed, **kw)
    if family == "Coulomb":
        kw = dict(kw)
        kind = kw.pop("kind")
        if kind == "isotrope":
            return CoulombPositions(seed=seed, **kw)
        return CoulombBroken(seed=seed, n_pins=6 if kind == "ancrages" else 0, **kw)
    if family == "Chaîne":
        kw = dict(kw)
        return ChainSubstrate(seed=seed, **CHAINS[kw.pop("kind")], **kw)
    if family == "Retard":
        return DelayNodeSubstrate(seed=seed, **kw)
    if family == "Coulomb répulsif":
        return CoulombRepulsive(seed=seed, **kw)
    raise ValueError(family)


_CH = {"a": "chaîne a) puits simple", "b": "chaîne b) double puits", "c": "chaîne c) libre"}

EXPLORATION = (
    [("ESN", g, {}) for g in range(6)]
    + [("Ressorts", g, {"amp": 0.3, "damping": 1.0}) for g in range(36)]
    + [("Coulomb", g, {"kind": k, "damping": 1.0, "E0": 0.25})
       for k in ("isotrope", "elliptique", "ancrages") for g in range(2)]
    + [("Chaîne", g, {"kind": _CH[k], "amp": 0.3}) for k in "abc" for g in range(2)]
    + [("Chaîne", g, {"kind": _CH["a"], "amp": a}) for a in (0.03, 0.1, 1.0, 3.0, 10.0) for g in range(2)]
)


def evaluate_system(family, seed, kw, series):
    """Profil d'abord (sans série), prévision ensuite."""
    sub = make_system(family, seed, **kw)
    prof = structural_profile(sub, seed=seed)
    n10, n20 = forecast_nrmse(sub, series, 10, seed), forecast_nrmse(sub, series, 20, seed)
    return {"family": family, "seed": seed, "params": kw, **prof,
            "nrmse10": n10, "nrmse20": n20, "perf": (n10 + n20) / 2}


def run_catalog(specs, path, part=None):
    """Évalue les systèmes de `specs` et ajoute une ligne JSON par système à `path`.
    Reprend là où il s'était arrêté. part=(i, n) : ne traite que les systèmes d'indice
    congru à i modulo n (pour lancer n processus en parallèle, un fichier chacun)."""
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["key"] for l in open(path)}
    series = normalize(mackey_glass())
    for idx, (family, seed, kw) in enumerate(specs):
        key = f"{family}|{seed}|{json.dumps(kw, sort_keys=True, ensure_ascii=False)}"
        if key in done or (part is not None and idx % part[1] != part[0]):
            continue
        row = evaluate_system(family, seed, kw, series)
        row["key"] = key
        with open(path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
        print(f"   {family:<17}{seed:>3} {str(kw)[:44]:<44} neutres {row['n_neutral']} états {row['n_equilibria']} "
              f"nl {row['nl']:.3f} d2 {row['d2']:9.3f} -> {row['nrmse10']:.2f} {row['nrmse20']:.2f}", flush=True)
    return load_rows(path)


def load_rows(*paths):
    rows = [json.loads(l) for p in paths if os.path.exists(p) for l in open(p)]
    return sorted(rows, key=lambda r: r["key"])


# ------------------------------------------------------------- descripteurs et modèles
def feature_matrix(rows):
    mc = np.array([r["mc_lin"] if r["mc_lin"] is not None and np.isfinite(r["mc_lin"]) else 0.0 for r in rows])
    nl = np.array([r["nl"] for r in rows])
    d2 = np.array([min(r["d2"], 1e3) if np.isfinite(r["d2"]) else 1e3 for r in rows])
    ln = np.log10(nl + 1e-3)
    return np.column_stack([np.log10(mc + 0.1), ln, ln ** 2, np.log10(d2 + 1e-3),
                            [float(r["n_neutral"] > 0) for r in rows]])


def _zs(F_tr, F_te):
    mu, sd = F_tr.mean(0), F_tr.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (F_tr - mu) / sd, (F_te - mu) / sd


def predict_ridge(F_tr, y_tr, F_te, alpha=1.0):
    A, B = _zs(F_tr, F_te)
    A1, B1 = np.hstack([A, np.ones((len(A), 1))]), np.hstack([B, np.ones((len(B), 1))])
    reg = alpha * np.eye(A1.shape[1]); reg[-1, -1] = 0
    return np.clip(B1 @ np.linalg.solve(A1.T @ A1 + reg, A1.T @ y_tr), 0, 1)


def predict_knn(F_tr, y_tr, F_te, k=5):
    A, B = _zs(F_tr, F_te)
    d = np.linalg.norm(B[:, None] - A[None], axis=2)
    return y_tr[np.argsort(d, axis=1)[:, :k]].mean(1)


MODELS = {"ridge": predict_ridge, "5 plus proches voisins": predict_knn}


def r2_score(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def graded_admit(rows, theta):
    return np.array([r["n_neutral"] == 0 and r["d2"] < theta for r in rows])


def separation(adm, perf):
    """(sensibilité + spécificité - 1) pour reconnaître les bonnes prévisions (perf < GOOD)."""
    good = perf < GOOD
    if good.sum() == 0 or (~good).sum() == 0:
        return np.nan
    return float(adm[good].mean() + (~adm)[~good].mean() - 1)


# ================================================================= l'exploration
def explore_analysis(rows, show=True):
    print("=" * 84)
    print("Étape 16 — exploration sur les systèmes déjà vus")
    print("=" * 84)
    perf = np.array([r["perf"] for r in rows]); fam = np.array([r["family"] for r in rows])
    binary = np.array([r["admis"] for r in rows])
    F = feature_matrix(rows)
    milestone("E1", len(rows) == len(EXPLORATION) and np.all(np.isfinite(F)) and np.all(np.isfinite(perf)),
              f"{len(rows)} systèmes, profils et prévisions complets")

    result("par famille : effectif, admis par le filtre binaire, NRMSE médiane")
    for f in sorted(set(fam)):
        m = fam == f
        print(f"      {f:<10} n = {m.sum():>2}   admis {binary[m].sum():>2}   médiane {np.median(perf[m]):.2f}")

    result("corrélation de rang entre chaque degré et l'erreur (+ : l'erreur augmente avec le degré)")
    names = ["mc_lin", "nl", "d2", "slowest"]
    cols = {n: np.array([r[n] if r[n] is not None and np.isfinite(r[n]) else np.nan for r in rows]) for n in names}
    groups = {"tous": np.ones(len(rows), bool), "admis (binaire)": binary, "ressorts": fam == "Ressorts"}
    print("      " + " " * 18 + "".join(f"{n:>10}" for n in names))
    for gname, m in groups.items():
        print(f"      {gname:<18}" + "".join(f"{spearman(cols[n][m], perf[m]):>+10.2f}" for n in names))

    # 1. Seuil de la règle graduée
    result("règle graduée « aucun mode neutre et d2 < theta » : séparation des bonnes prévisions")
    thetas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    seps = [separation(graded_admit(rows, th), perf) for th in thetas]
    for th, sp in zip(thetas, seps):
        adm = graded_admit(rows, th)
        print(f"      theta = {th:<5} admis {adm.sum():>2}   séparation {sp:+.2f}   "
              f"médiane admis {np.median(perf[adm]) if adm.any() else np.nan:.2f} / "
              f"rejetés {np.median(perf[~adm]) if (~adm).any() else np.nan:.2f}")
    print(f"      filtre binaire      admis {binary.sum():>2}   séparation {separation(binary, perf):+.2f}")
    theta = thetas[int(np.nanargmax(seps))]
    result(f"seuil retenu : theta = {theta}")

    # 2. Prédicteur : deux candidats, un système retiré à la fois
    result("prédire l'erreur à partir du profil (R², un système retiré à la fois)")
    loo = {}
    for name, fn in MODELS.items():
        p = np.array([fn(np.delete(F, i, 0), np.delete(perf, i), F[i:i + 1])[0] for i in range(len(rows))])
        loo[name] = p
        print(f"      {name:<24} tous {r2_score(perf, p):5.2f}   ressorts seuls "
              f"{r2_score(perf[fam == 'Ressorts'], p[fam == 'Ressorts']):5.2f}")
    base = np.array([np.mean(np.delete(perf, i)[np.delete(fam, i) == fam[i]]) for i in range(len(rows))])
    print(f"      {'moyenne de la famille':<24} tous {r2_score(perf, base):5.2f}   ressorts seuls "
          f"{r2_score(perf[fam == 'Ressorts'], base[fam == 'Ressorts']):5.2f}")
    model = max(loo, key=lambda n: r2_score(perf, loo[n]))
    result(f"modèle retenu : {model}")
    milestone("E2", True, "seuil et modèle choisis sur l'exploration seule")

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    marks = {"ESN": "o", "Ressorts": "^", "Coulomb": "X", "Chaîne": "s"}
    for f, mk in marks.items():
        m = fam == f
        axs[0].scatter(np.clip(cols["d2"][m], 1e-9, 1e3) + 1e-9, perf[m], marker=mk, s=30, label=f)
        axs[1].scatter(cols["nl"][m] + 1e-4, perf[m], marker=mk, s=30, label=f)
        axs[2].scatter(perf[m], loo[model][m], marker=mk, s=30, label=f)
    axs[0].axvline(theta, color="tab:red", lw=1)
    axs[0].set(xscale="log", xlabel="d2 (états finaux / amplitude statique)", ylabel="NRMSE de prévision",
               title=f"Q2 graduée (seuil theta = {theta})")
    axs[1].set(xscale="log", xlabel="non-linéarité statique nl", ylabel="NRMSE de prévision",
               title="Non-linéarité au point de fonctionnement")
    axs[2].plot([0, 1], [0, 1], "k--", lw=1)
    axs[2].set(xlabel="NRMSE réelle", ylabel="NRMSE prédite (système retiré)", title=f"Prédicteur : {model}")
    for ax in axs:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("exploration.png", dpi=110)
    milestone("E3", True, "figure enregistrée : exploration.png")
    print("=" * 84)
    if show:
        plt.show()
    return {"theta": theta, "model": model, "F": F, "perf": perf, "family": fam}


if __name__ == "__main__":
    print("Étape 16 chargée : run_catalog(EXPLORATION, 'exploration.jsonl') puis explore_analysis(rows)")
