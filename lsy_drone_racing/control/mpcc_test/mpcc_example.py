"""mpcc_example.py
---------------
Closed-loop simulation of the plain MPCC controller on a 7-gate loop, with
plots: a 3D drone trajectory (colored by speed) over the reference path and the
gates, plus speed / progress profiles.

The geometric gate loop stands in for the (non-feasible) planner output. The
"plant" integrates the same nominal dynamics with RK4 (no model mismatch).

Requires: numpy, scipy, casadi, acados, matplotlib.
"""

import numpy as np

from lsy_drone_racing.control.mpcc_test.mpcc_controller import MPCCController
from lsy_drone_racing.control.mpcc_test.mpcc_model import (
    IDX_THETA,
    IDX_VTHETA,
    MPCCConfig,
    make_dynamics_fn,
)
from lsy_drone_racing.control.mpcc_test.mpcc_reference import ReferencePath


def rk4_step(f_dyn, x, u, dt):
    k1 = f_dyn(x, u)
    k2 = f_dyn(x + dt / 2 * k1, u)
    k3 = f_dyn(x + dt / 2 * k2, u)
    k4 = f_dyn(x + dt * k3, u)
    return np.array(x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)).flatten()


def simulate(n_steps=400):
    gates = np.array([
        [1.0, -1.0, 1.5],
        [6.0, -6.0, 1.5],
        [9.0, -2.0, 2.5],
        [5.0,  3.0, 1.5],
        [-1.0, 5.0, 1.5],
        [-4.0, 0.0, 3.0],
        [-2.0, -5.0, 1.5],
    ])
    ref = ReferencePath(gates, closed=True, gate_indices=list(range(len(gates))),
                        qc_nom=1.0, qc_gate=120.0, gate_sigma=0.8)
    print(f"path length = {ref.length:.2f} m")

    cfg = MPCCConfig()
    ctrl = MPCCController(cfg, ref)
    ctrl.mu = 1.0   # progress weight (quadratic): raise to go faster

    f_dyn = make_dynamics_fn(cfg)
    p0 = ref.eval(0.0)
    t0 = ref.tangent(0.0)
    x = ctrl.initial_state(p=p0, q=[1, 0, 0, 0], v=1.0 * t0, w=[0, 0, 0], vtheta=1.0)

    log = {k: [] for k in ("t", "pos", "speed", "theta", "vtheta", "thrust", "status")}
    for i in range(n_steps):
        res = ctrl.solve(x)
        log["t"].append(i * cfg.dt)
        log["pos"].append(x[0:3].copy())
        log["speed"].append(float(np.linalg.norm(x[7:10])))
        log["theta"].append(float(x[IDX_THETA]))
        log["vtheta"].append(float(x[IDX_VTHETA]))
        log["thrust"].append(res["collective_thrust"])
        log["status"].append(res["status"])
        if res["status"] not in (0, 1, 2):
            print(f"[{i}] solver status {res['status']}")
        x = rk4_step(f_dyn, x, res["u0"], cfg.dt)
        if i % 20 == 0:
            print(f"t={i*cfg.dt:5.2f}s  theta={x[IDX_THETA]:6.2f}m "
                  f"vtheta={x[IDX_VTHETA]:5.2f}m/s  |v|={np.linalg.norm(x[7:10]):5.2f}m/s")

    log = {k: np.array(v) for k, v in log.items()}
    print(f"\ncovered {x[IDX_THETA]:.1f} m (~{x[IDX_THETA]/ref.length:.2f} laps) "
          f"in {n_steps*cfg.dt:.1f} s")
    return ref, gates, log


def plot(ref, gates, log, save_prefix="mpcc"):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # reference centerline samples
    ths = np.linspace(0, ref.length, 600)
    ctr = np.array([ref.eval(t) for t in ths])
    pos = log["pos"]
    spd = log["speed"]

    # ---- 3D trajectory --------------------------------------------------
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ctr[:, 0], ctr[:, 1], ctr[:, 2], "k--", lw=1.0, alpha=0.5, label="reference path")
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=spd, cmap="viridis", s=8)
    ax.scatter(gates[:, 0], gates[:, 1], gates[:, 2], c="red", marker="s", s=80,
               depthshade=False, label="gates")
    for j, g in enumerate(gates):
        ax.text(g[0], g[1], g[2] + 0.3, f"G{j+1}", color="red", fontsize=8)
    fig.colorbar(sc, ax=ax, label="speed [m/s]", shrink=0.6, pad=0.1)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("MPCC: drone trajectory (colored by speed)")
    ax.legend(loc="upper left")
    _set_equal_3d(ax, pos)
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_trajectory.png", dpi=140)

    # ---- top view + profiles -------------------------------------------
    fig2, axs = plt.subplots(2, 2, figsize=(11, 7))
    a = axs[0, 0]
    a.plot(ctr[:, 0], ctr[:, 1], "k--", lw=1, alpha=0.5)
    a.scatter(pos[:, 0], pos[:, 1], c=spd, cmap="viridis", s=8)
    a.scatter(gates[:, 0], gates[:, 1], c="red", marker="s", s=60)
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.set_title("top view (XY)")
    a.set_aspect("equal", adjustable="datalim")

    axs[0, 1].plot(log["t"], log["speed"]); axs[0, 1].set_title("speed |v| [m/s]")
    axs[0, 1].set_xlabel("t [s]")
    axs[1, 0].plot(log["t"], log["vtheta"]); axs[1, 0].set_title("progress speed vtheta [m/s]")
    axs[1, 0].set_xlabel("t [s]")
    axs[1, 1].plot(log["t"], log["theta"]); axs[1, 1].set_title("progress theta [m]")
    axs[1, 1].set_xlabel("t [s]")
    for a in axs.flat:
        a.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{save_prefix}_profiles.png", dpi=140)
    print(f"saved {save_prefix}_trajectory.png and {save_prefix}_profiles.png")
    plt.show()


def _set_equal_3d(ax, pts):
    mins = pts.min(0); maxs = pts.max(0)
    c = (mins + maxs) / 2.0
    r = (maxs - mins).max() / 2.0 + 1.0
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(max(0, c[2] - r), c[2] + r)


def main():
    ref, gates, log = simulate(n_steps=400)
    try:
        plot(ref, gates, log)
    except ImportError:
        print("matplotlib not available -- skipping plots")


if __name__ == "__main__":
    main()
