"""
mpccpp_example.py
-----------------
Closed-loop MPCC++ on a 7-gate loop. The tunnel is GATE-ALIGNED: at every gate
its cross section equals the gate opening (drawn as a red rectangle), so staying
inside the tunnel guarantees flying through the gate. Between gates the tube
widens to W_nom for corner-cutting freedom.

Plots: a translucent 3D tube + gate rectangles + drone path colored by speed,
a top view with the tube rails, and speed / progress / tunnel-violation /
thrust profiles.

Feasibility
-----------
* Soft tunnel (acados slacks) -> the QP is always feasible.
* Gate openings are generous (gate_w_half / gate_h_half below); enlarge them,
  or lower mu / vtheta_max, if you ever see violations.
* Stage 0 is unconstrained by the tunnel (x0 is fixed).

Requires: numpy, scipy, casadi, acados, matplotlib.
"""

import numpy as np

from lsy_drone_racing.control.mpcc_test.mpcc_model import (
    make_dynamics_fn, IDX_THETA, IDX_VTHETA,
)
from lsy_drone_racing.control.mpcc_test.mpccpp_model import MPCCppConfig
from lsy_drone_racing.control.mpcc_test.mpccpp_reference import TunnelReferencePath
from lsy_drone_racing.control.mpcc_test.mpccpp_controller import MPCCppController


def rk4_step(f_dyn, x, u, dt):
    k1 = f_dyn(x, u)
    k2 = f_dyn(x + dt / 2 * k1, u)
    k3 = f_dyn(x + dt / 2 * k2, u)
    k4 = f_dyn(x + dt * k3, u)
    return np.array(x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)).flatten()


def simulate(n_steps=400):
    gate_centers = np.array([
        [1.0, -1.0, 1.5],
        [6.0, -6.0, 1.5],
        [9.0, -2.0, 2.5],
        [5.0,  3.0, 1.5],
        [-1.0, 5.0, 1.5],
        [-4.0, 0.0, 3.0],
    ])
    ref = TunnelReferencePath(
        gate_centers=gate_centers, closed=True,
        # gate opening -> the tube cross section AT each gate (2.0 x 2.0 m).
        # Enlarge these to make the gates bigger / the squeeze easier.
        gate_w_half=1.0, gate_h_half=1.0,
        qc_nom=1.0, qc_gate=120.0, gate_sigma=0.8,
        W_nom=3.0, tunnel_sigma=1.2,          # wide between gates
    )
    print(f"path length = {ref.length:.2f} m")

    cfg = MPCCppConfig(use_tunnel=True, tunnel_soft=True)
    ctrl = MPCCppController(cfg, ref)
    ctrl.mu = 1.0
    print(f"tunnel: {'ON' if cfg.use_tunnel else 'OFF'} "
          f"({'soft' if cfg.tunnel_soft else 'hard'})  "
          f"gate opening = {2*ref.gate_hw[0]:.1f} x {2*ref.gate_hh[0]:.1f} m  "
          f"W_nom = {ref.W_nom:.1f} m")

    f_dyn = make_dynamics_fn(cfg)
    p0 = ref.eval(0.0)
    t0 = ref.tangent(0.0)
    x = ctrl.initial_state(p=p0, q=[1, 0, 0, 0], v=1.0 * t0, w=[0, 0, 0], vtheta=1.0)

    log = {k: [] for k in ("t", "pos", "speed", "theta", "vtheta", "thrust", "viol", "status")}
    for i in range(n_steps):
        res = ctrl.solve(x)
        th = res["theta"]
        n, b = ref.frame(th)
        d = x[0:3] - ref.eval(th)
        W, H = ref.width(th)
        viol = max(abs(d @ n) - W, abs(d @ b) - H, 0.0)

        log["t"].append(i * cfg.dt)
        log["pos"].append(x[0:3].copy())
        log["speed"].append(float(np.linalg.norm(x[7:10])))
        log["theta"].append(float(x[IDX_THETA]))
        log["vtheta"].append(float(x[IDX_VTHETA]))
        log["thrust"].append(res["collective_thrust"])
        log["viol"].append(viol)
        log["status"].append(res["status"])
        if res["status"] not in (0, 1, 2):
            print(f"[{i}] solver status {res['status']}")
        x = rk4_step(f_dyn, x, res["u0"], cfg.dt)
        if i % 20 == 0:
            print(f"t={i*cfg.dt:5.2f}s  theta={x[IDX_THETA]:6.2f}m "
                  f"vtheta={x[IDX_VTHETA]:5.2f}m/s  |v|={np.linalg.norm(x[7:10]):5.2f}m/s "
                  f"tunnel_viol={viol:.3f}m")

    log = {k: np.array(v) for k, v in log.items()}
    print(f"\ncovered {x[IDX_THETA]:.1f} m (~{x[IDX_THETA]/ref.length:.2f} laps) "
          f"in {n_steps*cfg.dt:.1f} s")
    print(f"max tunnel violation = {log['viol'].max():.3f} m  "
          f"(~0 means the drone stayed inside the tube / through the gates)")
    return ref, log


def tunnel_corners(ref, n_theta=220):
    ths = np.linspace(0.0, ref.length, n_theta)
    C = np.zeros((n_theta, 4, 3))
    for i, th in enumerate(ths):
        pd = ref.eval(th); n, b = ref.frame(th); W, H = ref.width(th)
        C[i, 0] = pd + W * n + H * b
        C[i, 1] = pd + W * n - H * b
        C[i, 2] = pd - W * n - H * b
        C[i, 3] = pd - W * n + H * b
    return C


def plot(ref, log, save_prefix="mpccpp"):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D                       # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ths = np.linspace(0, ref.length, 600)
    ctr = np.array([ref.eval(t) for t in ths])
    C = tunnel_corners(ref)
    pos = log["pos"]; spd = log["speed"]
    M = len(ref.gate_centers)

    faces = []
    for (a, bb) in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        for i in range(len(C) - 1):
            faces.append([C[i, a], C[i, bb], C[i + 1, bb], C[i + 1, a]])

    # ---- 3D trajectory + tube + gate rectangles ------------------------
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(faces, facecolor="tab:blue",
                                         alpha=0.10, edgecolor="none"))
    ax.plot(ctr[:, 0], ctr[:, 1], ctr[:, 2], "k--", lw=1.0, alpha=0.6, label="reference path")
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=spd, cmap="viridis", s=8)
    for j in range(M):
        r = ref.gate_rect(j)
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color="red", lw=2)
        gc = ref.gate_centers[j]
        ax.text(gc[0], gc[1], gc[2] + 0.3, f"G{j+1}", color="red", fontsize=8)
    fig.colorbar(sc, ax=ax, label="speed [m/s]", shrink=0.6, pad=0.1)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("MPCC++: tube goes through each gate opening (red)")
    ax.legend(loc="upper left")
    _set_equal_3d(ax, C.reshape(-1, 3))
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_trajectory.png", dpi=140)

    # ---- top view + profiles -------------------------------------------
    fig2, axs = plt.subplots(2, 3, figsize=(15, 8))
    a = axs[0, 0]
    for k in range(4):
        a.plot(C[:, k, 0], C[:, k, 1], color="tab:blue", lw=0.8, alpha=0.4)
    a.plot(ctr[:, 0], ctr[:, 1], "k--", lw=1, alpha=0.6)
    a.scatter(pos[:, 0], pos[:, 1], c=spd, cmap="viridis", s=8)
    for j in range(M):
        r = ref.gate_rect(j)
        a.plot(r[:, 0], r[:, 1], color="red", lw=2)
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.set_title("top view (XY) + tube + gates")
    a.set_aspect("equal", adjustable="datalim")

    axs[0, 1].plot(log["t"], log["speed"]); axs[0, 1].set_title("speed |v| [m/s]")
    axs[0, 2].plot(log["t"], log["viol"]); axs[0, 2].set_title("tunnel violation [m]")
    axs[1, 0].plot(log["t"], log["vtheta"]); axs[1, 0].set_title("progress speed vtheta [m/s]")
    axs[1, 1].plot(log["t"], log["theta"]); axs[1, 1].set_title("progress theta [m]")
    axs[1, 2].plot(log["t"], log["thrust"]); axs[1, 2].set_title("collective thrust [N]")
    for ax2 in axs.flat:
        ax2.set_xlabel("t [s]"); ax2.grid(alpha=0.3)
    axs[0, 0].set_xlabel("x [m]")
    fig2.tight_layout()
    fig2.savefig(f"{save_prefix}_profiles.png", dpi=140)
    print(f"saved {save_prefix}_trajectory.png and {save_prefix}_profiles.png")
    plt.show()


def _set_equal_3d(ax, pts):
    mins = pts.min(0); maxs = pts.max(0)
    c = (mins + maxs) / 2.0
    r = (maxs - mins).max() / 2.0 + 0.5
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(max(0, c[2] - r), c[2] + r)


def main():
    ref, log = simulate(n_steps=50)
    try:
        plot(ref, log)
    except ImportError:
        print("matplotlib not available -- skipping plots")


if __name__ == "__main__":
    main()
