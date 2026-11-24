"""
Étape 2b — Le réservoir en direct, dans le temps.

L'entrée change de classe toutes les SEG étapes (bruit -> sinus -> logistique,
les trois signaux de M1). Le vecteur de complexité est recalculé sur une
fenêtre glissante des W derniers pas.

Notebook : exécuter d'abord les cellules des étapes 1, 2a, 3 et 4. Puis, dans une
cellule séparée, choisir un backend interactif AVANT de lancer :

    %matplotlib widget        # dans VS Code (pip install ipympl), ou :
    %matplotlib qt            # fenêtre séparée (pip install pyqt5)

    anim = live()                              # animation (espace = pause)
    anim = live(rho=1.3, alpha=0.5)            # autre régime du réservoir
    live(headless=True, frames=200)            # sans animation : texte + image finale

Garder `anim = ...` : sans variable, Python efface l'animation et elle se fige.
"""

import time

import numpy as np

# Notebook : pas d'import entre fichiers. complexity_vector, conditional_lyapunov,
# entropy_complexity (étape 1), ESNSubstrate (étape 3), normalize_vector, AXES et
# draw_axes (étape 4) viennent des cellules précédentes.

N, W, K = 50, 2000, 50       # nœuds, fenêtre glissante (pas), pas simulés par image
SEG = 3000                   # durée de chaque classe d'entrée (pas)
EPISODE, DELTA = 400, 1e-6   # lambda : durée d'un épisode, écart initial
D = 4                        # motifs ordinaux (24 motifs : assez de fenêtres en direct)
TRACE, TRAIL, PRINT_EVERY = 300, 40, 10
CLASSES = ["bruit", "sinus", "logistique"]

_T0 = time.perf_counter()


def milestone(tag, ok, msg):
    status = "OK  " if ok else "FAIL"
    print(f"[{tag}] [{status}] {msg:<62} ({time.perf_counter() - _T0:5.2f} s)")


class InputSource:
    """Entrée scalaire dans [-1, 1] qui change de classe toutes les SEG étapes."""

    def __init__(self, seed=0):
        self.rng, self.t, self.x = np.random.default_rng(seed), 0, 0.4

    def next(self):
        cls = CLASSES[(self.t // SEG) % len(CLASSES)]
        if cls == "bruit":
            u = self.rng.uniform(-1, 1)
        elif cls == "sinus":
            u = np.sin(2 * np.pi * self.t / 50)
        else:
            self.x = 4 * self.x * (1 - self.x)
            if not 0 < self.x < 1:                    # garde-fou : point fixe 0
                self.x = self.rng.uniform(0.1, 0.9)
            u = 2 * self.x - 1
        self.t += 1
        return u, cls


class LiveSystem:
    """Un substrat et sa copie décalée, avancés ensemble avec la même entrée.
    N'utilise que l'interface : marche avec n'importe quel substrat de substrates.py."""

    def __init__(self, sub, seed=0):
        self.sub, self.src = sub, InputSource(seed)
        self.rng = np.random.default_rng(seed + 1)
        self.sa = sub.init_state(self.rng)
        self._new_episode()
        self.X, self.U = np.empty((0, sub.n_obs)), np.empty(0)
        self.cls, self.lams, self.last_ep_d = CLASSES[0], [], None

    def _new_episode(self):
        self.sb = self.sub.perturb(self.sa, DELTA, self.rng)
        self.ep_a, self.ep_b = [], []

    def advance(self, k):
        xs, us = [], []
        for _ in range(k):
            u, self.cls = self.src.next()
            self.sa, self.sb = self.sub.step(self.sa, u), self.sub.step(self.sb, u)
            xa, xb = self.sub.observe(self.sa), self.sub.observe(self.sb)
            xs.append(xa); us.append(u)
            self.ep_a.append(xa); self.ep_b.append(xb)
            if len(self.ep_a) == EPISODE:             # fin d'épisode : on mesure lambda
                Xa, Xb = np.array(self.ep_a), np.array(self.ep_b)
                self.lams.append(conditional_lyapunov(Xa, Xb))
                self.last_ep_d = np.linalg.norm(Xa - Xb, axis=1)
                self._new_episode()
        self.X = np.vstack([self.X, xs])[-W:]
        self.U = np.concatenate([self.U, us])[-W:]

    def measure(self):
        """Vecteur sur la fenêtre + position (H, C) de l'entrée. None tant que la fenêtre n'est pas pleine."""
        if len(self.X) < W:
            return None
        return complexity_vector(self.X, D=D, warn=False), entropy_complexity(self.U, D, warn=False)


def live(rho=0.9, alpha=0.3, scale=1.0, headless=False, frames=200):
    """Lance le direct. Renvoie l'animation (à garder dans une variable)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    sub = ESNSubstrate(N=N, rho=rho, alpha=alpha, input_scale=scale)
    sysm = LiveSystem(sub)
    milestone("L1", True, f"réservoir créé : N={N}, rho={rho}, alpha={alpha}, entrée x{scale}")

    # Repères du plan de Rosso : les trois signaux de M1
    rng, T = np.random.default_rng(1), 20000
    logi = np.empty(T); logi[0] = 0.4
    for t in range(T - 1):
        logi[t + 1] = 4 * logi[t] * (1 - logi[t])
    refs = {"bruit": rng.standard_normal(T), "sinus": np.sin(2 * np.pi * np.arange(T) / 50),
            "logistique": logi}
    refs = {k: entropy_complexity(v, D, warn=False) for k, v in refs.items()}
    milestone("L2", True, "repères : " + ", ".join(f"{k} ({h:.2f}, {c:.2f})" for k, (h, c) in refs.items()))

    # ------------------------------------------------------------- la figure
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1])
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    axP = fig.add_subplot(gs[2, :])
    lnU, = axA.plot([], [], "k", lw=1.5, label="entrée u")
    lnX = [axA.plot([], [], lw=0.8, label=f"nœud {i}")[0] for i in range(4)]
    axA.set(xlim=(0, TRACE), ylim=(-1.1, 1.1), xlabel="derniers pas", title="Traces")
    axA.legend(loc="upper right", fontsize=7)

    for k, (h, c) in refs.items():
        axB.plot(h, c, "s", color="grey"); axB.annotate(k, (h, c), fontsize=8, xytext=(4, 4),
                                                        textcoords="offset points")
    trail, = axB.plot([], [], "-", color="tab:blue", alpha=0.4)
    ptR, = axB.plot([], [], "o", color="tab:blue", ms=9, label="réservoir")
    ptU, = axB.plot([], [], "x", color="k", ms=9, mew=2, label="entrée")
    axB.set(xlim=(0, 1.02), ylim=(0, 0.5), xlabel="H (entropie)", ylabel="C_JS (complexité)",
            title="Plan complexité–entropie")
    axB.legend(loc="upper left", fontsize=8)

    lnD, = axC.plot([], [], "tab:red")
    axC.set(xlim=(0, EPISODE), ylim=(-16, 1), xlabel="pas dans l'épisode",
            ylabel="log10 écart", title="Divergence des deux copies (dernier épisode)")
    txtC = axC.text(0.02, 0.05, "", transform=axC.transAxes, fontsize=9)

    hist = {"H_perm": [], "C_JS": [], "PR": [], "C_N_form": []}
    lnH = {k: axD.plot([], [], label=k)[0] for k in hist}
    steps = []
    axD.set(ylim=(0, 1.05), xlabel="pas", title="Évolution du vecteur")
    axD.legend(loc="upper left", fontsize=8)
    txtD = axD.text(0.98, 0.95, "", transform=axD.transAxes, ha="right", va="top",
                    fontsize=8, family="monospace")

    # Coordonnées parallèles (étape 4) : ligne actuelle épaisse, passé estompé
    PAST = 30
    COLORS = {"bruit": "tab:blue", "sinus": "tab:orange", "logistique": "tab:green"}
    draw_axes(axP)
    i_lam = [k for k, _, _ in AXES].index("lam")
    axP.text(i_lam + 0.15, 0.52, "lambda = 0", fontsize=7, color="tab:red")
    past_lines = [axP.plot([], [], lw=1)[0] for _ in range(PAST)]
    cur_line, = axP.plot([], [], "o-", lw=3, ms=7, color="k")
    cur_txt = [axP.text(i + 0.06, 0, "", fontsize=8, va="center") for i in range(len(AXES))]
    axP.set_title("Coordonnées parallèles — couleur = entrée (bleu : bruit, orange : sinus, "
                  "vert : logistique), passé estompé")
    past = []                                   # (vecteur normalisé, classe d'entrée)

    state = {"paused": False, "frame": 0, "full": False, "lam": False, "cls": CLASSES[0],
             "t_last": time.perf_counter()}
    fig.canvas.mpl_connect("key_press_event",
                           lambda e: state.update(paused=not state["paused"]) if e.key == " " else None)

    def update(_):
        if state["paused"]:
            return
        sysm.advance(K)
        state["frame"] += 1
        t = sysm.src.t

        n = min(TRACE, len(sysm.U))
        lnU.set_data(np.arange(n), sysm.U[-n:])
        for i, ln in enumerate(lnX):
            ln.set_data(np.arange(n), sysm.X[-n:, i])

        if sysm.last_ep_d is not None:
            lnD.set_data(np.arange(EPISODE), np.log10(np.maximum(sysm.last_ep_d, 1e-16)))
            lam = sysm.lams[-1]
            txtC.set_text(f"lambda = {lam:+.4f} / pas   ({'écho' if lam < 0 else 'chaos'})")
            if not state["lam"]:
                state["lam"] = True
                milestone("L4", np.isfinite(lam), f"premier lambda mesuré : {lam:+.4f}")

        if sysm.cls != state["cls"]:
            print(f"   --> t={t}: l'entrée passe de « {state['cls']} » à « {sysm.cls} »")
            state["cls"] = sysm.cls

        res = sysm.measure()
        if res is None:
            return
        C, (Hu, Cu) = res
        if not state["full"]:
            state["full"] = True
            milestone("L3", True, f"fenêtre pleine ({W} pas) : premières mesures")
        if not all(np.isfinite(v) for k, v in C.items() if k not in ("lam", "consist")):
            milestone("L5", False, f"t={t} : NaN dans le vecteur {C}")

        steps.append(t)
        for k in hist:
            hist[k].append(C[k]); lnH[k].set_data(steps, hist[k])
        axD.set_xlim(max(0, t - 30 * SEG), t + 1)
        trail.set_data(hist["H_perm"][-TRAIL:], hist["C_JS"][-TRAIL:])
        ptR.set_data([C["H_perm"]], [C["C_JS"]]); ptU.set_data([Hu], [Cu])
        lam = sysm.lams[-1] if sysm.lams else np.nan
        tau = f"{C['tau_med']:8.1f}" if C["tau_med"] < 500 else "  >= 500"     # 500 = max_lag : plafonné
        txtD.set_text(f"C_N      {C['C_N']:8.2f}\ntau_med  {tau}\n"
                      f"tau_sprd {C['tau_sprd']:8.2f}\nlambda   {lam:+8.4f}")
        fig.suptitle(f"t = {t}   entrée : {sysm.cls}   (espace = pause)")

        v = normalize_vector(C | {"lam": lam})
        past.append((v, sysm.cls)); del past[:-PAST]
        for j, ln in enumerate(past_lines):
            if j < len(past) - 1:                   # la dernière est la ligne actuelle
                pv, pc = past[j]
                ln.set_data(range(len(AXES)), pv)
                ln.set_color(COLORS[pc]); ln.set_alpha(0.05 + 0.35 * j / PAST)
            else:
                ln.set_data([], [])
        cur_line.set_data(range(len(AXES)), v); cur_line.set_color(COLORS[sysm.cls])
        raw = [f"{C['H_perm']:.2f}", f"{C['C_JS']:.3f}", f"{C['PR']:.3f}", f"{C['C_N_form']:.2f}",
               tau.strip(), f"{C['tau_sprd']:.2f}", f"{lam:+.3f}",
               "non mesurée"]                       # le direct ne lance pas de 3e trajectoire
        for i, (txt, r) in enumerate(zip(cur_txt, raw)):
            txt.set_position((i + 0.06, v[i] if np.isfinite(v[i]) else 0.5)); txt.set_text(r)

        if state["frame"] % PRINT_EVERY == 0:
            now = time.perf_counter()
            fps, state["t_last"] = PRINT_EVERY / (now - state["t_last"]), now
            print(f"[t={t:6d}] {sysm.cls:<10} H={C['H_perm']:.3f} C={C['C_JS']:.3f} "
                  f"PR={C['PR']:.3f} C_N={C['C_N']:6.1f} forme={C['C_N_form']:.2f} tau={tau.strip():>6} "
                  f"lam={lam:+.4f}  ({fps:4.1f} img/s)")

    fig.tight_layout(rect=(0, 0, 1, 0.96))    # place pour le titre général
    if headless:
        for i in range(frames):
            update(i)
        fig.savefig("live_snapshot.png", dpi=110)
        plt.close(fig)                         # évite un double affichage dans le notebook
        milestone("L6", True, f"{frames} images simulées, capture : live_snapshot.png")
        return None
    anim = FuncAnimation(fig, update, interval=1, cache_frame_data=False)
    plt.show()
    return anim
