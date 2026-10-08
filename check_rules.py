"""
to checks three things:
  1. With alpha=0 and p0=p, the custom rules reproduce NaSch EXACTLY (same seed, same trajectory).
  2. No collisions across many random parameter combinations (stress test).
  3. Prints a small configuration table you can compare to your Week 4 hand trace.
"""

import numpy as np
from nasch_simulation import RingRoadCA
from custom_rules import CustomRingCA


def test_reduces_to_nasch():
    for seed in range(20):
        a = RingRoadCA(L=50, N=15, v_max=3, p=0.3, rng=np.random.default_rng(seed))
        b = CustomRingCA(L=50, N=15, v_max=3, p=0.3, p0=0.3, alpha=0,
                         rng=np.random.default_rng(seed))
        for _ in range(200):
            a.step(); b.step()
            assert np.array_equal(a.positions, b.positions)
            assert np.array_equal(a.speeds, b.speeds)
    print("[pass] alpha=0, p0=p reproduces NaSch exactly (20 seeds x 200 steps)")


def test_no_collisions():
    rng = np.random.default_rng(123)
    runs = 0
    for _ in range(300):
        L = int(rng.integers(10, 80))
        N = int(rng.integers(1, L + 1))
        v_max = int(rng.integers(1, min(8, L)))
        p, p0 = sorted(rng.random(2))
        ca = CustomRingCA(L, N, v_max, p, p0, alpha=1, rng=rng)
        for _ in range(100):
            ca.step()  # raises RuntimeError on any collision
            assert len(np.unique(ca.positions)) == ca.N
        runs += 1
    print(f"[pass] no collisions in {runs} random runs x 100 steps")


def trace_example():
    """Small hand-traceable example: L=10, 4 cars, v_max=2, 6 steps."""
    ca = CustomRingCA.from_state(L=10, positions=[0, 2, 3, 7], speeds=[1, 0, 0, 1],
                                 v_max=2, p=0.2, p0=0.5, alpha=1,
                                 rng=np.random.default_rng(7))
    print("\nTrace: L=10, v_max=2, p=0.2, p0=0.5, alpha=1, seed=7")
    print(" t | car | x | v | d | d_eff | u after R1-R3 | hesitated | new v")
    for t in range(6):
        x_old, v_old, ids = ca.positions.copy(), ca.speeds.copy(), ca.ids.copy()
        info = ca.step()
        for i in np.argsort(ids):
            u = info["u_braked"][i] - int(info["hesitate"][i] and info["u_braked"][i] > 0)
            print(f"{t:2d} | {ids[i]:3d} | {x_old[i]} | {v_old[i]} | {info['d'][i]} | "
                  f"{info['d_eff'][i]:5d} | {info['u_braked'][i]:13d} | "
                  f"{str(bool(info['hesitate'][i])):9s} | {u}")
        road = ["." for _ in range(ca.L)]
        for x, v in zip(ca.positions, ca.speeds):
            road[x] = str(v)
        print("   road after step (digits = speeds):", "".join(road))


if __name__ == "__main__":
    test_reduces_to_nasch()
    test_no_collisions()
    trace_example()