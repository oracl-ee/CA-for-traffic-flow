# Cellular Automaton Models of Traffic Flow

Senior capstone project (Mathematics, Fall 2026).

This project designs new cellular automaton (CA) rules for vehicular traffic, compares them against the classic **Nagel–Schreckenberg (NaSch)** model, and studies how traffic behavior (free flow, phantom jams, phase transitions) changes as the rules and the road geometry change.

![Space-time diagrams: NaSch vs. ADSD](figures/space_time_comparison.png)

**Live demo:** the ring road runs in the browser on [oracl-ee.dev](https://oracl-ee.dev), following one car through a jam, with a NaSch vs. ADSD run comparison.

## The model

A road is a ring of `L` cells. Each cell is empty or holds one car with an integer speed `0 … v_max`. At every time step, all cars update **simultaneously** using only local information.

### Baseline: NaSch (1992)
1. Accelerate: `v = min(v + 1, v_max)`
2. Brake: `v = min(v, d)` (where `d` = empty cells ahead)
3. Randomize: with probability `p`, `v = max(v − 1, 0)`
4. Move: `x = (x + v) mod L`

### New rules: ADSD (Anticipating Driver with Startup Delay)

Each car also looks at the car ahead: its speed `v_f` and its gap `d_f`.

| Rule | Update | Driver behavior it models |
|---|---|---|
| R1 Accelerate | `u = min(v + 1, v_max)` | Speed up when possible |
| R2 Anticipate | `g_f = max(min(v_f, d_f) − 1, 0)`, `d_eff = d + α·g_f` | Brake for where the car ahead *will be*, not where it is now |
| R3 Brake | `u = min(u, d_eff)` | Never hit the car ahead |
| R4 Hesitate | with prob. `p0` if stopped, `p` if moving: `u = max(u − 1, 0)` | Stopped drivers are slower to start (like at a green light) |
| R5 Move | `x = (x + u) mod L` | |

**Key design property:** setting `α = 0` and `p0 = p` recovers NaSch *exactly*. Every difference between the models can therefore be traced to a single switch (anticipation or startup delay).

**Collision-freeness:** the car ahead always moves at least `g_f` cells, so moving at most `d + g_f` cells is safe. The code also verifies this numerically at every step.

## Results so far

The fundamental diagram (flow vs. density) for four rule sets, turning each new rule on separately:

![Fundamental diagram ablation](figures/fundamental_diagram_ablation.png)

- **Anticipation** lets cars follow more closely, since each driver accounts for the car ahead also moving.
- **Startup delay** makes jams "stickier": stopped cars restart slowly, so congestion forms large, long-lived jams that move backward along the road (visible as dark diagonal bands in the space-time diagram above).

### One car's point of view

The fundamental diagram averages over the whole road. `run_lap_comparison.py` instead follows **one car** (by its permanent id) for full runs of `L = 120` cells, on the same road as the live demo (`v_max = 5`, `p = 0.2`, `p0 = 0.5`). A typical run at density 0.22:

![One typical run: NaSch vs. ADSD](figures/lap_example.png)

Over 500 runs per model at ρ = 0.22 (20 independent roads × 25 runs):

| | NaSch | ADSD |
|---|---|---|
| steps to drive 120 cells | 50.2 ± 0.3 | 69.1 ± 0.7 |
| steps stopped | 11.7 | 38.0 |
| separate jams per run | 4.9 | 3.3 |

(± is a 95% confidence interval computed from the 20 independent roads, since consecutive runs of the same car are correlated.)

- ADSD runs take about **38% longer**. ADSD drivers hit *fewer* jams than NaSch drivers, but each stop lasts much longer.
- Mean run time agrees with `L·ρ/q` computed from the road's flow (51.0 for NaSch, 70.5 for ADSD), connecting the single-car view back to the fundamental diagram.

Repeating this across densities and turning each rule on separately:

![Run time vs. density, rule ablation](figures/lap_time_vs_density.png)

- Below ρ ≈ 0.12 all four rule sets agree: there are no jams to get stuck in.
- Above that, **startup delay** accounts for essentially all of ADSD's slowdown: "startup delay only" and full ADSD overlap, while "anticipation only" stays with (slightly below) NaSch.

Distributions of run time and time stopped are in [`figures/lap_distributions.png`](figures/lap_distributions.png).

## Repository structure

| File | Purpose |
|---|---|
| `nasch_simulation.py` | NaSch baseline on a ring road (`RingRoadCA`) |
| `custom_rules.py` | ADSD rules (`CustomRingCA`), with collision checking |
| `check_rules.py` | Verification: ADSD reduces to NaSch, collision stress test, hand-trace table |
| `run_highway_comparison.py` | Generates the space-time and fundamental diagrams in `figures/` |
| `run_lap_comparison.py` | Follows one car for full 120-cell runs: NaSch vs. ADSD run time, time stopped, and the density ablation |

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python check_rules.py            # tests + small trace table
python run_highway_comparison.py # writes figures/ (about 1 minute)
python run_lap_comparison.py     # followed-car runs, writes figures/lap_*.png (about 40 seconds)
```

Parameters (`L`, `v_max`, `p`, `p0`) are set near the top of each `run_*.py` script.

**Reproducibility:** every simulation uses a seeded random generator (`np.random.default_rng(seed)`), so rerunning a script reproduces the same numbers and figures exactly. To check that a result is not an artifact of particular seeds, change the seeds (e.g. `seed=s + 100` in `collect()`); the means should move by less than their confidence intervals.

## Roadmap

- [x] NaSch baseline and ADSD rules on a ring highway
- [x] Verification and rule-by-rule comparison
- [x] Followed-car run comparison (NaSch vs. ADSD, with ablation)
- [ ] Parameter sweeps and phase-transition analysis
- [ ] T-intersection geometry (yielding rule)
- [ ] Merge / on-ramp geometry (gap-acceptance rule)
- [ ] Mean-field model for flow vs. density under ADSD
- [ ] Formal collision-freeness proof for every geometry

## References

1. K. Nagel and M. Schreckenberg, "A cellular automaton model for freeway traffic," *J. Phys. I France* 2 (1992), 2221–2229.
2. D. Chowdhury, L. Santen, and A. Schadschneider, "Statistical physics of vehicular traffic and some related systems," *Physics Reports* 329 (2000), 199–329.
3. R. Barlovic, L. Santen, A. Schadschneider, and M. Schreckenberg, "Metastable states in cellular automata for traffic flow," *Eur. Phys. J. B* 5 (1998), 793–800.
