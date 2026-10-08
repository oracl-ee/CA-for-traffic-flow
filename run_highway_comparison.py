"""
Highway (ring road) comparison of NaSch vs. the custom ADSD rules.
Produces, in ./figures/:
  - space_time_comparison.png : NaSch and ADSD from the SAME initial configuration
  - fundamental_diagram_ablation.png : q(rho) for NaSch, +anticipation only,
        +startup delay only, and full ADSD, so each difference can be traced to one rule
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from custom_rules import CustomRingCA

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Shared parameters (edit these to explore, Stage 2 of the proposal)
L, V_MAX, P, P0 = 400, 5, 0.2, 0.5

VARIANTS = {
    "NaSch (alpha=0, p0=p)":      dict(alpha=0, p0=P),
    "anticipation only":          dict(alpha=1, p0=P),
    "startup delay only":         dict(alpha=0, p0=P0),
    "ADSD (both)":                dict(alpha=1, p0=P0),
}


def occupancy_grid(pos_hist, L):
    grid = np.zeros((pos_hist.shape[0], L), dtype=int)
    for t, row in enumerate(pos_hist):
        grid[t, row] = 1
    return grid


def space_time_comparison(density=0.15, steps=300, burn_in=200, seed=1):
    N = round(density * L)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    for ax, name in zip(axes, ["NaSch (alpha=0, p0=p)", "ADSD (both)"]):
        ca = CustomRingCA(L, N, V_MAX, P, rng=np.random.default_rng(seed), **VARIANTS[name])
        res = ca.run(steps=steps, burn_in=burn_in)
        ax.imshow(occupancy_grid(res["positions_history"], L), cmap="Greys",
                  aspect="auto", interpolation="nearest")
        ax.set_title(f"{name}\nmean flow q = {res['mean_flow']:.3f}")
        ax.set_xlabel("cell")
    axes[0].set_ylabel("time step (down = later)")
    fig.suptitle(f"Space-time diagrams, rho={density}, L={L}, v_max={V_MAX}, p={P}, p0={P0}")
    fig.tight_layout()
    out = FIG_DIR / "space_time_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def fundamental_diagram_ablation(n_rho=30, steps=1500, burn_in=1000, n_seeds=3):
    densities = np.linspace(0.02, 0.9, n_rho)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, kw in VARIANTS.items():
        flows = []
        for rho in densities:
            N = max(1, round(rho * L))
            q = [CustomRingCA(L, N, V_MAX, P, rng=np.random.default_rng(s), **kw)
                 .run(steps=steps, burn_in=burn_in)["mean_flow"] for s in range(n_seeds)]
            flows.append(np.mean(q))
        flows = np.array(flows)
        k = flows.argmax()
        print(f"{name:24s} max q = {flows[k]:.3f} at rho = {densities[k]:.3f}")
        ax.plot(densities, flows, "o-", ms=3, label=name)
    ax.set_xlabel(r"density $\rho = N/L$")
    ax.set_ylabel(r"flow $q$")
    ax.set_title(f"Fundamental diagram, rule ablation (L={L}, v_max={V_MAX}, p={P}, p0={P0})")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "fundamental_diagram_ablation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    space_time_comparison()
    fundamental_diagram_ablation()