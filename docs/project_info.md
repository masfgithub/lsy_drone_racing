# MPCC++ for Autonomous Drone Racing

This document details the MPCC++ extension of the LSY drone-racing pipeline: the
method, the module layout, the main tunable parameters, and the evaluation
results. For the project overview, see the [README](../README.md).

---

## 1. Overview

MPCC++ follows a geometric racing line while deciding its own speed along that
line. Instead of tracking a time-stamped trajectory, the controller tracks a
**path** parameterized by arc length and maximizes progress along it, subject to
a **tunnel** around the path and to gate/obstacle constraints. All environment
constraints are formulated as *smooth soft penalties*, so the optimal control
problem (OCP) stays feasible even when the drone is pushed off the reference —
hence *always-feasible*.

The pipeline has two decoupled parts:

- **Planner** → produces the geometric reference path through the remaining gates
  (see [Planning](#4-planning)).
- **MPCC++ controller** → builds a tunnel around that path and solves a
  receding-horizon OCP that trades progress against contour/lag error and
  constraint penalties.

Because the controller consumes only the path *geometry* (not the planner's
timing), planner and controller can be tuned independently.

---

## 2. Project Structure

```
lsy_drone_racing/
├── control/
│   ├── planner/                          # Path-planning module
│   │   ├── lightweight_planner.py        # Single-pass planner (SP-2, hardware)
│   │   ├── smart_planner.py              # Recursive branch-and-bound planner (sim)
│   │   ├── smart_planner_base.py         # Base class for the recursive planner
│   │   └── spline_planner_base.py        # Shared spline / waypoint utilities
│   ├── mpcc/                             # MPCC / MPCC++ controllers
│   │   ├── mpccpp.py                     # MPCC++ controller
│   │   ├── mpccpp_setup.py               # MPCC++ acados OCP setup
│   │   ├── mpccpp_reference.py           # MPCC++ tunnel reference path
│   │   ├── mpcc.py                       # Baseline MPCC controller
│   │   ├── mpcc_setup.py                 # MPCC acados OCP setup
│   │   └── mpcc_reference.py             # MPCC reference path
│   ├── controller_interface.py           # Abstract controller base class
│   ├── drone_racing_pipeline.py          # Pipeline entry point (run this)
│   ├── drone_racing_pipeline_config.py   # Pipeline / controller-selection config
│   └── env_obs.py                        # Observation → EnvState_t extraction
├── envs/
│   └── mpccpp_visualizer.py              # Renders MPCC++ tunnel / predictions
├── utils/
│   └── environment_constraints/
│       └── utils/
│           ├── env_constraints.py        # Hard constraint expressions
│           ├── env_soft_constraints.py   # Soft (penalty) constraint expressions
│           ├── obstacle.py               # CylinderObstacle keep-out model
│           └── wedge_window.py           # WedgeWindow gate model
└── config/
    ├── level0.toml                       # Track / randomization configs
    ├── level1.toml
    ├── level2.toml
    ├── level3.toml
    └── final.toml
```

---

## 3. Method

### 3.1 Contouring formulation

A progress state `θ` indexes the reference path `p_d(θ)`. At each node the
position error is split into two components in the path frame:

- **Contour error** `e_c` — lateral deviation from the path (perpendicular).
- **Lag error** `e_l` — deviation along the path (how far ahead/behind `p_d(θ)`).

The stage cost penalizes `e_c` and `e_l` and **rewards progress** through a term
`-μ · v_θ`, where `v_θ = dθ/dt` is a control input (the progress speed). Larger
`μ` ⇒ more aggressive racing; contour/lag weights keep the drone on the line.

### 3.2 Tunnel reference

The tunnel is built once per plan from the planner's path:

1. Resample the planner path to **arc length** to get a clean centerline
   `p_d(θ)`, `θ ∈ [0, L]`.
2. **Project** each remaining gate center onto the centerline to get its
   arc-length position `θ_gate`.
3. Define a rectangular cross-section `(W(θ), H(θ))` around `p_d(θ)` in the path
   frame `(n, b)`. The cross-section is wide between gates and **pinches** to the
   gate opening at each `θ_gate` via a Gaussian bump.

The tunnel is enforced as a soft constraint, so momentary excursions are
penalized rather than infeasible.

### 3.3 Always-feasible soft constraints

- **Gate frames** are modeled as `WedgeWindow` prisms (four wedges); the keep-out
  is a smooth penalty that vanishes inside the opening.
- **Obstacles** are `CylinderObstacle` keep-outs — a smooth radial penalty.

All penalties are differentiable CasADi expressions added to the cost (a
`NONLINEAR_LS` / `GAUSS_NEWTON` residual), so the QP subproblems never become
infeasible from the environment constraints.

### 3.4 Online replanning

The reference tube is rebuilt and progress reset to `θ = 0` (drone = new start)
whenever a gate or obstacle is observed to move beyond a small tolerance. Between
replans the geometry is fixed and only `θ` advances.

---

## 4. Planning

Two interchangeable planners share a common spline base and expose the same path
interface to the controller:

| Planner | Module | Strategy | Use |
| :-- | :-- | :-- | :-- |
| **Recursive (smart)** | `control/planner/smart_planner.py` | Recursive branch-and-bound over waypoint detours; resolves gate re-crossings and obstacle conflicts | Simulation (highest success rate) |
| **Lightweight (SP-2)** | `control/planner/lightweight_planner.py` | Single-pass spline with bracket detours around obstacles and passed-gate frames | Hardware (cheap to re-plan) |

Both generate waypoints from gate orientations, detect backtracking and insert
detours, avoid obstacles with a configurable margin, and reparameterize by arc
length before handing the path to MPCC++.

---

## 5. Pipeline & Configuration

The stack is launched from `control/drone_racing_pipeline.py`. The active planner
is selected in `control/drone_racing_pipeline_config.py`.

```bash
python scripts/sim.py --config <tomlfile_name>.toml --controller drone_racing_pipeline.py
```

---

## 6. Key Tunable Parameters

The main knobs live as named constants at the top of the controller/reference
files. Representative roles (verify the current defaults in your source before
citing them):

| Parameter | Role | Effect |
| :-- | :-- | :-- |
| `N` | Horizon length (nodes) | Longer horizon = smoother/faster but heavier solve |
| `mu` | Progress reward weight | Higher = more aggressive racing |
| `v_theta_max` | Progress-speed cap | Upper bound on effective path speed |
| `tunnel_sigma` | Gaussian width of the gate pinch | Larger = pinch spreads further along the path |
| `qc_gate`, `gate_sigma` | Speed shaping near gates | Slow-down profile through gate openings |
| `w_c`, `w_l` | Contour / lag weights | Tracking tightness vs. progress freedom |
| gate/obstacle penalty weights | Soft-constraint stiffness | Higher = harder keep-out |

> These are the primary parameters that trade **speed against success rate**; the
> "generalized" configuration in Table I balances the two.

---

## 7. Results

The MPCC++ pipeline was evaluated with the *generalized* configuration (tuned to
balance speed and success rate), across both planners and two horizon lengths `N`.

| Planner | N | Level 2 SR [%] | Level 2 Time [s] | Level 3 SR [%] | Level 3 Time [s] | Final SR [%] | Final Time [s] |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **Recursive (smart)** | 40 | **80** | 6.80 | **75** | 6.84 | **80** | 6.87 |
|                       | 20 | 65 | 6.48 | 60 | 6.66 | 60 | 6.51 |
| **Lightweight (SP-2)**| 40 | 65 | 7.51 | 75 | 7.57 | 75 | 7.63 |
|                       | 20 | 60 | 6.89 | 65 | 6.85 | 75 | 6.77 |

**TABLE I:** Simulation results for the generalized MPCC++ configuration
(balancing speed and success rate): success rate (SR) and mean lap time over 20
runs on the Level-2 track, a Level-3 track, and the final-evaluation track
(Final), for two horizon lengths `N`.

**Takeaway.** The recursive planner at `N = 40` gives the best and most
consistent success rate across all three tracks, while the lightweight planner
trades a little success rate and lap time for much cheaper replanning suited to
hardware.

### Demo videos

**Level 2 track** — MPCC++ on the Level-2 track (randomized gates and obstacles, online re-planning).

<video controls playsinline preload="metadata" style="width:100%;max-width:720px;height:auto;">
  <source src="videos/Level_2_track.mp4" type="video/mp4">
  <a href="videos/Level_2_track.mp4">Level 2 track</a>
</video>

**Final track — simulation** — full lap on the final-evaluation track in simulation.

<video controls playsinline preload="metadata" style="width:100%;max-width:720px;height:auto;">
  <source src="videos/Sim_final_track.mp4" type="video/mp4">
  <a href="videos/Sim_final_track.mp4">Final track (simulation)</a>
</video>

**Final track — fast** — final track with the speed-tuned configuration (minimum lap time).

<video controls playsinline preload="metadata" style="width:100%;max-width:720px;height:auto;">
  <source src="videos/Fast_final_track.mp4" type="video/mp4">
  <a href="videos/Fast_final_track.mp4">Final track (fast)</a>
</video>

**Final track — slow** — final track with the conservative configuration (maximum success rate).

<video controls playsinline preload="metadata" style="width:100%;max-width:720px;height:auto;">
  <source src="videos/Slow_final_track.mp4" type="video/mp4">
  <a href="videos/Slow_final_track.mp4">Final track (slow)</a>
</video>
