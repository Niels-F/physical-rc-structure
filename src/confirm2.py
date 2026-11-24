"""
Étape 17 — Confirmation sur des systèmes JAMAIS VUS, dont deux familles nouvelles.

Tout ce qui suit a été écrit AVANT d'exécuter le moindre système de confirmation :
le catalogue, les valeurs gelées à l'étape 16 (seuil theta, modèle), les hypothèses
P2 à P5 et leurs critères chiffrés. Rien n'est ajusté après coup ; si un critère
échoue, il est rapporté comme échoué.

  P2  Un système admis mais presque linéaire (nl < 0.01) ne prévoit pas mieux qu'un
      modèle linéaire. Critère : au moins 80 % de ces systèmes ont une NRMSE >= 0.35,
      et aucun n'est sous 0.25.
  P3  Parmi les réseaux de ressorts, c'est la capacité mémoire linéarisée mc_lin qui
      ordonne les géométries (exploration : corrélation de rang -0.73 avec l'erreur,
      contre +0.13 pour d2 ; l'hypothèse initiale portait sur d2, elle a été remplacée
      ici, avant toute confirmation). Critères, sur les 40 nouvelles géométries :
      (a) corrélation de rang entre mc_lin et l'erreur <= -0.4 ;
      (b) [gardé tel que prévu, bien que l'exploration ne le favorise pas : séparation
      0.21 contre 0.28 pour le filtre binaire] avec la règle graduée « aucun mode neutre et d2 < theta », médiane des admis
      < médiane des rejetés, et au moins 75 % du meilleur quart est admis.
  P4  Le profil prédit l'erreur mieux que la simple moyenne de la famille. Critères :
      (a) erreur quadratique du modèle gelé < celle de la référence, sur l'ensemble ;
      (b) parmi les 40 ressorts, R² du modèle > 0.1.
      Référence : moyenne de la famille dans l'exploration ; pour une famille nouvelle,
      moyenne de toute l'exploration.
  P5  Le filtre binaire de l'étape 12 tient sur l'ensemble de confirmation. Critères :
      (a) médiane des admis inférieure d'au moins 0.10 à celle des rejetés ;
      (b) au moins 75 % du meilleur quart est admis.

Notebook : exécuter d'abord les cellules des étapes 1 à 16.
Puis : conf = run_catalog(CONFIRMATION, "confirmation.jsonl")     (~1 h)
       confirm2_analysis(load_rows("exploration.jsonl"), conf)
"""

import time

import numpy as np

# Notebook : pas d'import entre fichiers.

_T0 = time.perf_counter()

# ---- Valeurs gelées à l'issue de l'étape 16 (ne pas modifier) ----------------
FROZEN_THETA = 0.01          # choisis à l'étape 16, sur les 64 systèmes d'exploration
FROZEN_MODEL = "5 plus proches voisins"
NL_LINEAR = 0.01             # P2 : « presque linéaire »
# ------------------------------------------------------------------------------


def milestone(tag, ok, msg):
    status = "OK  " if ok else "FAIL"
    print(f"[{tag}] [{status}] {msg:<62} ({time.perf_counter() - _T0:5.2f} s)")
    return ok


def result(msg):
    print(f"[RES] {msg}")


CONFIRMATION = (
    [("Ressorts", g, {"amp": 0.3, "damping": 1.0}) for g in range(36, 76)]
    + [("ESN", g, {"input_scale": a}) for g in (6, 7, 8) for a in (0.01, 0.1, 1.0, 3.0)]
    + [("Chaîne", g, {"kind": _CH["a"], "amp": a}) for g in (2, 3, 4) for a in (0.03, 0.3, 1.0, 3.0, 10.0)]
    + [("Chaîne", g, {"kind": _CH[k], "amp": 0.3}) for k in "bc" for g in (2, 3)]
    + [("Retard", g, {"gamma": a}) for g in (0, 1, 2) for a in (0.05, 0.5, 2.0)]
    + [("Retard", g, {"eta": 1.5, "phi": 0.0}) for g in (0, 1)]
    + [("Coulomb répulsif", g, {"damping": d, "E0": 0.25}) for g in range(4) for d in (1.0, 2.0)]
)


def _verdict(ok):
    return "RÉUSSI" if ok else "ÉCHOUÉ"


def _quartile_admitted(adm, perf):
    best = np.argsort(perf)[:max(1, len(perf) // 4)]
    return int(adm[best].sum()), len(best)


def confirm2_analysis(explo, conf, show=True):
    print("=" * 84)
    print("Étape 17 — confirmation sur des systèmes jamais vus")
    print("=" * 84)
    assert FROZEN_THETA is not None and FROZEN_MODEL in MODELS, "geler theta et le modèle d'abord"
    perf = np.array([r["perf"] for r in conf]); fam = np.array([r["family"] for r in conf])
    binary = np.array([r["admis"] for r in conf]); nl = np.array([r["nl"] for r in conf])
    d2 = np.array([min(r["d2"], 1e3) for r in conf])
    graded = graded_admit(conf, FROZEN_THETA)
    keys_e = {r["key"] for r in explo}
    milestone("V1", len(conf) == len(CONFIRMATION) and not any(r["key"] in keys_e for r in conf),
              f"{len(conf)} systèmes, aucun en commun avec l'exploration")
    milestone("V2", bool(np.all(np.isfinite(perf))), "prévisions complètes")

    result("par famille : effectif, admis (binaire), admis (gradué), NRMSE médiane")
    for f in sorted(set(fam)):
        m = fam == f
        print(f"      {f:<17} n = {m.sum():>2}   binaire {binary[m].sum():>2}   gradué {graded[m].sum():>2}   "
              f"médiane {np.median(perf[m]):.2f}")
    out = {}

    # ---- P2
    m = binary & (nl < NL_LINEAR)
    if m.sum():
        frac, low = float(np.mean(perf[m] >= 0.35)), int(np.sum(perf[m] < 0.25))
        out["P2"] = frac >= 0.8 and low == 0
        result(f"P2 : {m.sum()} systèmes admis presque linéaires ; {frac:.0%} ont une NRMSE >= 0.35, "
               f"{low} sous 0.25 (min {perf[m].min():.2f}) -> {_verdict(out['P2'])}")
        m2 = binary & (nl >= NL_LINEAR)
        result(f"     pour comparaison, admis non linéaires : médiane {np.median(perf[m2]):.2f} (n = {m2.sum()})")
    else:
        out["P2"] = None
        result("P2 : aucun système concerné -> NON TESTÉ")

    # ---- P3
    s = fam == "Ressorts"
    mc = np.array([r["mc_lin"] for r in conf])
    rho = spearman(mc[s], perf[s])
    out["P3a"] = rho <= -0.4
    result(f"P3a : ressorts, corrélation de rang entre mc_lin et l'erreur = {rho:+.2f} (critère <= -0.40) "
           f"-> {_verdict(out['P3a'])}")
    result(f"     pour comparaison, corrélation de rang entre d2 et l'erreur : {spearman(d2[s], perf[s]):+.2f}")
    ga = graded[s]
    if ga.any() and (~ga).any():
        ma, mr = np.median(perf[s][ga]), np.median(perf[s][~ga])
        k, n = _quartile_admitted(ga, perf[s])
        out["P3b"] = ma < mr and k / n >= 0.75
        result(f"P3b : règle graduée, {ga.sum()} admis / {(~ga).sum()} rejetés ; médianes {ma:.2f} / {mr:.2f} ; "
               f"meilleur quart admis {k}/{n} -> {_verdict(out['P3b'])}")
    else:
        out["P3b"] = False
        result(f"P3b : la règle graduée ne sépare rien ({ga.sum()} admis sur {s.sum()}) -> ÉCHOUÉ")
    ba = binary[s]
    if ba.any() and (~ba).any():
        k, n = _quartile_admitted(ba, perf[s])
        result(f"     pour comparaison, filtre binaire sur les ressorts : {ba.sum()} admis ; médianes "
               f"{np.median(perf[s][ba]):.2f} / {np.median(perf[s][~ba]):.2f} ; meilleur quart admis {k}/{n}")

    # ---- P4
    Fe, ye = feature_matrix(explo), np.array([r["perf"] for r in explo])
    fe = np.array([r["family"] for r in explo])
    pred = MODELS[FROZEN_MODEL](Fe, ye, feature_matrix(conf))
    base = np.array([ye[fe == f].mean() if (fe == f).any() else ye.mean() for f in fam])
    mse_m, mse_b = float(np.mean((perf - pred) ** 2)), float(np.mean((perf - base) ** 2))
    out["P4a"] = mse_m < mse_b
    result(f"P4a : erreur quadratique, modèle gelé ({FROZEN_MODEL}) {mse_m:.4f} contre référence {mse_b:.4f} "
           f"-> {_verdict(out['P4a'])}")
    r2s = r2_score(perf[s], pred[s])
    out["P4b"] = r2s > 0.1
    result(f"P4b : parmi les ressorts, R² du modèle = {r2s:+.2f} (critère > 0.10) -> {_verdict(out['P4b'])}")
    result(f"     R² sur l'ensemble : modèle {r2_score(perf, pred):+.2f}, référence {r2_score(perf, base):+.2f} ; "
           f"corrélation de rang prédit/réel {spearman(pred, perf):+.2f}")
    for f in sorted(set(fam)):
        m = fam == f
        print(f"      {f:<17} erreur quadratique : modèle {np.mean((perf[m] - pred[m]) ** 2):.4f}   "
              f"référence {np.mean((perf[m] - base[m]) ** 2):.4f}")

    # ---- P5
    ma, mr = np.median(perf[binary]), np.median(perf[~binary])
    k, n = _quartile_admitted(binary, perf)
    out["P5a"], out["P5b"] = (mr - ma) >= 0.10, k / n >= 0.75
    result(f"P5a : filtre binaire, {binary.sum()} admis / {(~binary).sum()} rejetés ; médianes {ma:.2f} / {mr:.2f} "
           f"-> {_verdict(out['P5a'])}")
    result(f"P5b : meilleur quart admis {k}/{n} -> {_verdict(out['P5b'])}")
    for f in ("Retard", "Coulomb répulsif"):
        m = fam == f
        a, r = m & binary, m & ~binary
        print(f"      famille nouvelle {f:<17} admis {a.sum()} (médiane {np.median(perf[a]) if a.any() else np.nan:.2f})"
              f"   rejetés {r.sum()} (médiane {np.median(perf[r]) if r.any() else np.nan:.2f})")

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    marks = {"ESN": "o", "Ressorts": "^", "Chaîne": "s", "Retard": "D", "Coulomb répulsif": "X"}
    for f, mk in marks.items():
        m = fam == f
        axs[0].scatter(perf[m & binary], pred[m & binary], marker=mk, s=34, label=f)
        axs[0].scatter(perf[m & ~binary], pred[m & ~binary], marker=mk, s=34, facecolors="none", edgecolors="0.4")
        axs[1].scatter(nl[m & binary] + 1e-4, perf[m & binary], marker=mk, s=34, label=f)
    axs[0].plot([0, 1], [0, 1], "k--", lw=1)
    axs[0].set(xlabel="NRMSE réelle", ylabel="NRMSE prédite par le modèle gelé",
               title="P4 : prédiction (creux : rejetés par le filtre)")
    axs[1].axvline(NL_LINEAR, color="tab:red", lw=1)
    axs[1].set(xscale="log", xlabel="non-linéarité statique nl", ylabel="NRMSE de prévision",
               title="P2 : systèmes admis")
    axs[2].scatter(mc[s], perf[s], marker="^", s=34, c=np.where(binary[s], "tab:blue", "0.6"))
    axs[2].set(xlabel="capacité mémoire linéarisée mc_lin", ylabel="NRMSE de prévision",
               title="P3 : 40 nouveaux réseaux de ressorts (gris : rejetés binaire)")
    for ax in axs[:2]:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("confirmation2.png", dpi=110)
    milestone("V3", True, "figure enregistrée : confirmation2.png")
    n_t = sum(v is not None for v in out.values()); n_ok = sum(bool(v) for v in out.values() if v is not None)
    result(f"bilan : {n_ok} critères réussis sur {n_t} testés")
    print("=" * 84)
    if show:
        plt.show()
    return out


if __name__ == "__main__":
    print("Étape 17 chargée : run_catalog(CONFIRMATION, 'confirmation.jsonl') puis confirm2_analysis(explo, conf)")
