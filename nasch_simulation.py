"""
nasch_simulation.py

A minimal, well-documented NaSch (Nagel-Schreckenberg) cellular automaton
simulation on a periodic (ring) 1D highway. This is starting-point code for
the "set up Python environment" Preliminary Objective in the capstone
proposal ("Designing Cellular Automaton Rules for Traffic Flow Across
Multiple Road Geometries").

It demonstrates the three things the proposal's preliminary objective asks
for:
    1. Ability to animate/track CA evolution over time.
    2. Ability to produce a space-time diagram.
    3. Ability to plot a flow-density (fundamental diagram) curve.

The implementation follows the four NaSch rules exactly as stated in the
proposal (Section 1.3):
    1. Accelerate:  v <- min(v + 1, v_max)
    2. Brake:       v <- min(v, d - 1)
    3. Randomize:   with probability p, v <- max(v - 1, 0)
    4. Move:        x <- (x + v) mod L

All position arithmetic uses the ring-road (mod L) convention discussed in
the prerequisites notes.

Run this file directly to generate:
    - space_time_diagram.png   (one run, a single density)
    - fundamental_diagram.png  (flow vs. density, simulation vs. theory)
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt


class RingRoadCA:
    """NaSch cellular automaton on a periodic 1D road of length L."""

    def __init__(self, L: int, N: int, v_max: int, p: float, rng: np.random.Generator | None = None):
        """
        Parameters
        ----------
        L : number of cells on the ring
        N : number of cars (N <= L)
        v_max : maximum allowed speed
        p : randomization (hesitation) probability, in [0, 1]
        rng : optional numpy random Generator for reproducibility
        """
        if N > L:
            raise ValueError("Cannot place more cars than cells (N > L).")
        self.L = L
        self.N = N
        self.v_max = v_max
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng()

        # Random initial placement of N cars on L cells, all starting at speed 0.
        positions = self.rng.choice(L, size=N, replace=False)
        self.positions = np.sort(positions).astype(int)
        self.speeds = np.zeros(N, dtype=int)

    def _gaps_ahead(self) -> np.ndarray:
        """Number of empty cells strictly between each car and the next car ahead (mod L)."""
        # positions is sorted ascending; "next car ahead" of the last car wraps to the first.
        next_positions = np.roll(self.positions, -1)
        gap = (next_positions - self.positions - 1) % self.L
        return gap

    def step(self) -> None:
        """
        Advance the system by one NaSch time step (rules 1-4, applied simultaneously).

        Note on the brake rule / gap convention (see prerequisites_notes.md, Sec 2.3):
        `_gaps_ahead()` returns d = the number of EMPTY cells strictly between a car
        and the next car ahead. Moving by exactly d cells lands the car on the last
        empty cell before the next car (safe); moving by d+1 collides. So the
        collision-avoiding cap on speed is v <- min(v, d).
        The proposal's Section 1.3 states the rule as v <- min(v, d-1), using a
        "d = distance to next car" convention (i.e. proposal's d = this d + 1).
        Both describe the identical physical rule; only the bookkeeping differs.
        """
        d = self._gaps_ahead()

        # Rule 1: Accelerate
        v = np.minimum(self.speeds + 1, self.v_max)
        # Rule 2: Brake to avoid collision (v <- min(v, d), per this module's gap convention)
        v = np.minimum(v, d)

        # Rule 3: Randomize
        hesitate = self.rng.random(self.N) < self.p
        v = np.where(hesitate, np.maximum(v - 1, 0), v)

        # Rule 4: Move (mod L, ring road)
        self.positions = (self.positions + v) % self.L
        self.speeds = v

        # Re-sort by position so "next car ahead" bookkeeping stays valid.
        order = np.argsort(self.positions)
        self.positions = self.positions[order]
        self.speeds = self.speeds[order]

    def run(self, steps: int, burn_in: int = 0) -> dict:
        """
        Run the simulation for `steps` steps (after an optional burn-in).

        Returns a dict with:
            'positions_history' : (steps, N) array of positions after burn-in
            'speeds_history'    : (steps, N) array of speeds after burn-in
            'mean_flow'         : time-averaged flow q = density * mean_speed
        """
        for _ in range(burn_in):
            self.step()

        pos_hist = np.zeros((steps, self.N), dtype=int)
        speed_hist = np.zeros((steps, self.N), dtype=int)
        for t in range(steps):
            pos_hist[t] = self.positions
            speed_hist[t] = self.speeds
            self.step()

        density = self.N / self.L
        mean_speed = speed_hist.mean()
        mean_flow = density * mean_speed

        return {
            "positions_history": pos_hist,
            "speeds_history": speed_hist,
            "mean_flow": mean_flow,
            "density": density,
        }


def space_time_diagram(L: int, N: int, v_max: int, p: float, steps: int = 100,
                        burn_in: int = 50, seed: int = 0, savepath: str = "space_time_diagram.png"):
    """Produce and save a space-time diagram for one run of the CA."""
    rng = np.random.default_rng(seed)
    ca = RingRoadCA(L=L, N=N, v_max=v_max, p=p, rng=rng)
    result = ca.run(steps=steps, burn_in=burn_in)
    pos_hist = result["positions_history"]

    # Build a binary occupancy grid (time x cell) for a clean plot.
    grid = np.zeros((steps, L), dtype=int)
    for t in range(steps):
        grid[t, pos_hist[t]] = 1

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(grid, cmap="Greys", aspect="auto", interpolation="nearest", origin="upper")
    ax.set_xlabel("cell (position on ring)")
    ax.set_ylabel("time step")
    ax.set_title(f"Space-time diagram (NaSch)\nL={L}, N={N}, v_max={v_max}, p={p}, "
                 f"density={result['density']:.2f}")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"Saved {savepath}  (mean flow over run = {result['mean_flow']:.4f})")


def fundamental_diagram(L: int = 200, v_max: int = 1, p: float = 0.3,
                         steps: int = 300, burn_in: int = 200, seed: int = 0,
                         savepath: str = "fundamental_diagram.png"):
    """
    Sweep density rho = N/L, measure simulated flow q, and compare against
    the theoretical NaSch mean-field result for v_max = 1 (Theorem 1 in the
    proposal): q(rho) = (1 - p) * rho * (1 - rho).
    """
    rng = np.random.default_rng(seed)
    densities = np.linspace(0.02, 0.98, 25)
    sim_flows = []

    for rho in densities:
        N = max(1, round(rho * L))
        ca = RingRoadCA(L=L, N=N, v_max=v_max, p=p, rng=rng)
        result = ca.run(steps=steps, burn_in=burn_in)
        sim_flows.append(result["mean_flow"])

    sim_flows = np.array(sim_flows)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(densities, sim_flows, "o", label="simulation", color="tab:blue")

    if v_max == 1:
        rho_theory = np.linspace(0, 1, 200)
        q_theory = (1 - p) * rho_theory * (1 - rho_theory)
        ax.plot(rho_theory, q_theory, "-", label="theory (mean field, $v_{max}=1$)",
                color="tab:orange")

    ax.set_xlabel(r"density $\rho = N/L$")
    ax.set_ylabel(r"flow $q$")
    ax.set_title(f"NaSch fundamental diagram (L={L}, v_max={v_max}, p={p})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"Saved {savepath}")


if __name__ == "__main__":
    # 1. Sanity check: animate/track evolution + produce a space-time diagram.
    space_time_diagram(L=60, N=15, v_max=2, p=0.3, steps=100, burn_in=50, seed=1)

    # 2. Fundamental diagram at v_max = 1, checked against Theorem 1 in the proposal.
    fundamental_diagram(L=200, v_max=1, p=0.3, steps=300, burn_in=200, seed=1)