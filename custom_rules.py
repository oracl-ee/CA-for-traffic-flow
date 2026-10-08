"""

Gap convention (same as nasch_simulation.py): d = number of EMPTY cells
strictly between a car and the car ahead. Moving exactly d cells is safe.

Local information each car uses at time t:
    v   = own speed
    d   = empty cells ahead
    v_f = speed of the car ahead
    d_f = empty cells ahead of the car ahead

Parameters:
    v_max  maximum speed
    p      base hesitation probability (moving cars)
    p0     startup hesitation probability (stopped cars, v = 0), p0 >= p
    alpha  anticipation switch, 0 or 1

Update (applied simultaneously to every car):
    R1  Accelerate:  u = min(v + 1, v_max)
    R2  Anticipate:  g_f   = max(min(v_f, d_f) - 1, 0)   (guaranteed motion of car ahead)
                     d_eff = d + alpha * g_f
    R3  Brake:       u = min(u, d_eff)
    R4  Hesitate:    with probability h(v), u = max(u - 1, 0),
                     where h(v) = p0 if v == 0 else p
    R5  Move:        x = (x + u) mod L

Special case: alpha = 0 and p0 = p gives exactly NaSch. Every difference from
NaSch can therefore be traced to one switch (alpha or p0), which is the
structure the A-level comparison objective needs.

Collision-freeness (sketch, full proof goes in the paper):
    The car ahead ends with u_f >= min(v_f, d_f) - 1 and u_f >= 0, so u_f >= g_f.
    This car has u <= d + g_f. New gap = d + u_f - u >= 0.
    The code also checks this numerically every step (check_collisions=True).
"""

from __future__ import annotations
import numpy as np


class CustomRingCA:
    """ADSD rules on a periodic 1D road of L cells."""

    def __init__(self, L: int, N: int, v_max: int, p: float, p0: float | None = None,
                 alpha: int = 1, rng: np.random.Generator | None = None,
                 check_collisions: bool = True):
        if N > L:
            raise ValueError("Cannot place more cars than cells (N > L).")
        if v_max >= L:
            raise ValueError("v_max must be smaller than L.")
        self.L, self.N, self.v_max, self.p = L, N, v_max, p
        self.p0 = p if p0 is None else p0
        self.alpha = alpha
        self.check_collisions = check_collisions
        self.rng = rng if rng is not None else np.random.default_rng()

        # Same initialization as RingRoadCA so the two can be compared seed-for-seed.
        positions = self.rng.choice(L, size=N, replace=False)
        self.positions = np.sort(positions).astype(int)
        self.speeds = np.zeros(N, dtype=int)
        self.ids = np.arange(N)  # permanent car labels (positions get re-sorted on wraparound)

    @classmethod
    def from_state(cls, L, positions, speeds, v_max, p, p0=None, alpha=1, rng=None):
        """Build a CA from an explicit configuration (for hand-traced examples)."""
        positions = np.asarray(positions, dtype=int)
        speeds = np.asarray(speeds, dtype=int)
        ca = cls(L, len(positions), v_max, p, p0, alpha, rng=np.random.default_rng(0))
        order = np.argsort(positions)
        ca.positions, ca.speeds = positions[order], speeds[order]
        ca.ids = np.arange(len(positions))
        if rng is not None:
            ca.rng = rng
        return ca

    def _gaps_ahead(self) -> np.ndarray:
        return (np.roll(self.positions, -1) - self.positions - 1) % self.L

    def step(self) -> dict:
        """One simultaneous update. Returns intermediate values (useful for tracing)."""
        v = self.speeds
        d = self._gaps_ahead()

        # R1 accelerate
        u = np.minimum(v + 1, self.v_max)

        # R2 anticipate (disabled for a single car: the "car ahead" would be itself)
        if self.N > 1 and self.alpha:
            v_f, d_f = np.roll(v, -1), np.roll(d, -1)
            g_f = np.maximum(np.minimum(v_f, d_f) - 1, 0)
        else:
            g_f = np.zeros(self.N, dtype=int)
        d_eff = d + g_f

        # R3 brake
        u = np.minimum(u, d_eff)
        u_before_hesitation = u.copy()

        # R4 hesitate, probability depends on whether the car was stopped
        h = np.where(v == 0, self.p0, self.p)
        hesitate = self.rng.random(self.N) < h
        u = np.where(hesitate, np.maximum(u - 1, 0), u)

        if self.check_collisions and self.N > 1:
            new_gap = d + np.roll(u, -1) - u
            if np.any(new_gap < 0):
                raise RuntimeError(f"Collision detected! new gaps = {new_gap}")

        # R5 move
        self.positions = (self.positions + u) % self.L
        self.speeds = u
        order = np.argsort(self.positions)
        self.positions, self.speeds = self.positions[order], self.speeds[order]
        self.ids = self.ids[order]

        return {"d": d, "d_eff": d_eff, "u_braked": u_before_hesitation,
                "hesitate": hesitate}

    def run(self, steps: int, burn_in: int = 0) -> dict:
        """Same interface and outputs as RingRoadCA.run."""
        for _ in range(burn_in):
            self.step()
        pos_hist = np.zeros((steps, self.N), dtype=int)
        speed_hist = np.zeros((steps, self.N), dtype=int)
        for t in range(steps):
            pos_hist[t] = self.positions
            speed_hist[t] = self.speeds
            self.step()
        density = self.N / self.L
        return {"positions_history": pos_hist, "speeds_history": speed_hist,
                "mean_flow": density * speed_hist.mean(), "density": density}