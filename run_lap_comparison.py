"""
One car, one full run: NaSch vs. ADSD from a single driver's point of view.

The fundamental diagram averages over every car on the road. This script
instead follows ONE car (by its permanent id) and records its speed until it
has driven a full run of L cells (one lap of the ring), the same measurement
the live sim on oracl-ee.dev shows.

For each finished run we record
    steps    time steps needed to drive L cells
    stopped  steps spent at speed 0
    jams     number of separate stops (speed drops from >0 to 0)
    speed    average speed over the run = cells driven / steps

Consistency check: over many runs a tagged car's average speed is the road's
mean speed v_bar = q / rho, so the mean run time should be close to
L / v_bar = L * rho / q. The script prints both.

Produces, in ./figures/:
  - lap_example.png        speed during one typical run (the median-length run), NaSch vs ADSD
  - lap_distributions.png  run time and time stopped over many runs
  - lap_time_vs_density.png  mean run time vs density for the four rule variants (ablation)
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from custom_rules import CustomRingCA

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Same road as the website: 120 cells, v_max 5, p 0.2, p0 0.5
L, V_MAX, P, P0 = 120, 5, 0.2, 0.5
DENSITY = 0.22

VARIANTS = {
    "NaSch (alpha=0, p0=p)": dict(alpha=0, p0=P),
    "anticipation only":     dict(alpha=1, p0=P),
    "startup delay only":    dict(alpha=0, p0=P0),
    "ADSD (both)":           dict(alpha=1, p0=P0),
}
MAIN = ["NaSch (alpha=0, p0=p)", "ADSD (both)"]
SHORT = {"NaSch (alpha=0, p0=p)": "NaSch", "ADSD (both)": "ADSD"}
COLOR = {"NaSch (alpha=0, p0=p)": "tab:blue", "anticipation only": "tab:green",
         "startup delay only": "tab:purple", "ADSD (both)": "tab:red"}


def follow_runs(density, n_runs, seed, burn_in=300, car_id=0, **rule_kw):
    """Follow one car through n_runs consecutive full runs of L cells.

    Returns a list of dicts (one per run) plus the road's mean flow over the
    same steps, for the L*rho/q consistency check.
    """
    N = max(2, round(density * L))
    ca = CustomRingCA(L, N, V_MAX, P, rng=np.random.default_rng(seed), **rule_kw)
    for _ in range(burn_in):
        ca.step()

    runs, flows = [], []
    speeds, dist = [], 0
    while len(runs) < n_runs:
        ca.step()
        v = int(ca.speeds[ca.ids == car_id][0])
        flows.append(ca.speeds.sum() / L)
        speeds.append(v)
        dist += v
        if dist >= L:
            s = np.array(speeds)
            # jams = separate stretches at speed 0 (a run that starts inside a jam counts it too)
            jams = int(np.sum((s == 0) & np.r_[True, s[:-1] != 0]))
            runs.append(dict(speeds=s, steps=len(s), stopped=int(np.sum(s == 0)),
                             jams=jams, cells=dist, speed=dist / len(s)))
            speeds, dist = [], dist - L  # overshoot past L counts toward the next run
    return runs, float(np.mean(flows))


def collect(density, rule_kw, n_seeds, runs_per_seed):
    """Runs from several independent roads. Consecutive runs of one car are
    correlated, so uncertainty is estimated from the per-road (per-seed) means."""
    all_runs, seed_means, flows = [], [], []
    for s in range(n_seeds):
        runs, q = follow_runs(density, runs_per_seed, seed=s, **rule_kw)
        all_runs += runs
        seed_means.append(np.mean([r["steps"] for r in runs]))
        flows.append(q)
    seed_means = np.array(seed_means)
    ci = 1.96 * seed_means.std(ddof=1) / np.sqrt(n_seeds)
    return all_runs, float(seed_means.mean()), float(ci), float(np.mean(flows))


def summarize(name, runs, mean, ci, q, density):
    steps = np.array([r["steps"] for r in runs])
    stopped = np.array([r["stopped"] for r in runs])
    jams = np.array([r["jams"] for r in runs])
    speed = np.array([r["speed"] for r in runs])
    print(f"{name:24s} runs={len(runs):4d}  steps: mean {mean:6.1f} ± {ci:4.1f} (95% CI), "
          f"median {np.median(steps):5.1f} | stopped {stopped.mean():5.1f} | "
          f"jams {jams.mean():4.2f} | speed {speed.mean():4.2f} | "
          f"check L*rho/q = {L * density / q:6.1f}")


def typical(runs):
    """The run whose length is closest to the median: representative, not cherry-picked."""
    med = np.median([r["steps"] for r in runs])
    return min(runs, key=lambda r: (abs(r["steps"] - med), r["steps"]))


def plot_example(results):
    ex = {name: typical(results[name][0]) for name in MAIN}
    longest = max(r["steps"] for r in ex.values())
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, name in zip(axes, MAIN):
        r = ex[name]
        t = np.arange(r["steps"])
        ax.step(t, r["speeds"], where="post", color=COLOR[name], lw=1.8)
        ax.fill_between(t, 0, V_MAX, where=r["speeds"] == 0, step="post", color="tab:red", alpha=0.12, lw=0)
        ax.axhline(V_MAX, ls="--", lw=0.8, color="grey")
        ax.axvline(r["steps"] - 1, lw=0.8, color="k")
        ax.set_xlim(0, longest)
        ax.set_ylim(-0.2, V_MAX + 0.4)
        ax.set_xlabel("time step within the run")
        ax.set_title(f"{SHORT[name]}: {r['steps']} steps, stopped {r['stopped']}, "
                     f"{r['jams']} jam(s), avg speed {r['speed']:.2f}")
    axes[0].set_ylabel("speed of the followed car")
    fig.suptitle(f"One typical run of {L} cells (median-length run), rho={DENSITY}, "
                 f"v_max={V_MAX}, p={P}, p0={P0}.  Shaded = stopped.")
    fig.tight_layout()
    out = FIG_DIR / "lap_example.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_distributions(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for key, ax, label in [("steps", axes[0], "steps to drive one full run"),
                           ("stopped", axes[1], "steps spent stopped during the run")]:
        data = {name: np.array([r[key] for r in results[name][0]]) for name in MAIN}
        hi = max(d.max() for d in data.values())
        bins = np.arange(0, hi + 3, 2 if key == "steps" else 1) - 0.5
        for name in MAIN:
            d = data[name]
            ax.hist(d, bins=bins, alpha=0.5, color=COLOR[name], density=True,
                    label=f"{SHORT[name]} (mean {d.mean():.1f})")
            ax.axvline(d.mean(), color=COLOR[name], lw=1.5)
        ax.set_xlabel(label)
        ax.set_ylabel("fraction of runs")
        ax.legend()
    n = len(results[MAIN[0]][0])
    fig.suptitle(f"{n} runs per model (followed car, rho={DENSITY}, L={L}); vertical lines = means")
    fig.tight_layout()
    out = FIG_DIR / "lap_distributions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_vs_density(densities, n_seeds=6, runs_per_seed=25):
    fig, ax = plt.subplots(figsize=(8, 5))
    print("\nMean run time vs density (followed car):")
    for name, kw in VARIANTS.items():
        means, cis = [], []
        for rho in densities:
            _, m, ci, _ = collect(rho, kw, n_seeds, runs_per_seed)
            means.append(m)
            cis.append(ci)
        means, cis = np.array(means), np.array(cis)
        print(f"  {name:24s} " + "  ".join(f"{m:6.1f}" for m in means))
        ax.errorbar(densities, means, yerr=cis, fmt="o-", ms=3, capsize=2, color=COLOR[name], label=name)
    ax.axhline(L / V_MAX, ls=":", color="grey", label=f"free road, no hesitation (L/v_max = {L // V_MAX})")
    ax.set_xlabel(r"density $\rho = N/L$")
    ax.set_ylabel(f"mean steps to drive {L} cells")
    ax.set_title(f"Run time of one car vs density, rule ablation (L={L}, v_max={V_MAX}, p={P}, p0={P0})")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "lap_time_vs_density.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    print(f"Followed-car runs at rho = {DENSITY} (20 independent roads x 25 runs each):")
    results = {}
    for name in MAIN:
        results[name] = collect(DENSITY, VARIANTS[name], n_seeds=20, runs_per_seed=25)
        summarize(name, *results[name], DENSITY)
    plot_example(results)
    plot_distributions(results)
    plot_vs_density(np.round(np.arange(0.06, 0.61, 0.06), 2))
