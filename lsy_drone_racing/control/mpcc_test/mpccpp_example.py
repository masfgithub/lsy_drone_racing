"""Closed-loop MPCC++ example on a 7-gate loop with a tunnel and stick obstacles.

Four stick obstacles (3 cm poles, shared keep-out radius) are avoided via soft
constraints, on top of the gate-aligned tunnel.

Obstacles are specified just like the gates -- an editable (M,2) array of
(x, y) centers -- and, like the gates, their positions are treated as runtime
data: loaded at startup with set_obstacles() and refreshed every step with
set_obstacles() (here the "measured" positions equal the nominal ones; swap in
your perception estimate).

Plots: 3D tube + gate rectangles + stick keep-out cylinders + drone path
(colored by speed); top view with tube rails, gates and keep-out circles; and
profiles incl. tunnel violation and distance-to-nearest-stick.

Requires: numpy, scipy, casadi, acados, matplotlib.
"""

from typing import Callable

import numpy as np

from lsy_drone_racing.control.mpcc_test.mpcc_model import IDX_THETA, IDX_VTHETA, make_dynamics_fn
from lsy_drone_racing.control.mpcc_test.mpccpp_controller import MPCCppController
from lsy_drone_racing.control.mpcc_test.mpccpp_model import MPCCppConfig
from lsy_drone_racing.control.mpcc_test.mpccpp_reference import TunnelReferencePath

# === scenario definition (edit these like a track file) =====================
GATE_CENTERS = np.array(
    [
        [1.0, -1.0, 1.5],
        [6.0, -6.0, 1.5],
        [9.0, -2.0, 2.5],
        [5.0, 3.0, 1.5],
        [-1.0, 5.0, 1.5],
        [-4.0, 0.0, 3.0],
    ]
)

# four sticks: (x, y) centers -- move them wherever you like.
# obs 0,1 sit on the reference centerline; obs 2,3 sit just off it.
OBSTACLE_CENTERS = np.array(
    [
        [3.60, -3.70],  # on path
        [1.29, 5.00],  # on path
        [9.0, -1.43],  # near path
        [-3.78, -1.49],  # near path
    ]
)
OBSTACLE_RADIUS = 0.5  # shared keep-out radius [m] (3 cm pole + safety margin)
# ============================================================================


def rk4_step(
    f_dyn: Callable[[np.ndarray, np.ndarray], np.ndarray], x: np.ndarray, u: np.ndarray, dt: float
) -> np.ndarray:
    """Advance the state by one RK4 step of the dynamics f_dyn."""
    k1 = f_dyn(x, u)
    k2 = f_dyn(x + dt / 2 * k1, u)
    k3 = f_dyn(x + dt / 2 * k2, u)
    k4 = f_dyn(x + dt * k3, u)
    return np.array(x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)).flatten()


def simulate(n_steps: int = 50) -> tuple[TunnelReferencePath, tuple[np.ndarray, float], dict]:
    """Closed-loop simulation of the MPCC++ controller on the example scenario."""
    ref = TunnelReferencePath(
        gate_centers=GATE_CENTERS,
        closed=True,
        gate_w_half=1.0,
        gate_h_half=1.0,
        qc_nom=1.0,
        qc_gate=120.0,
        gate_sigma=0.8,
        w_nom=3.0,
        tunnel_sigma=1.2,
    )
    print(f"path length = {ref.length:.2f} m")

    n_obst = len(OBSTACLE_CENTERS)
    cfg = MPCCppConfig(
        use_tunnel=True,
        tunnel_soft=True,
        use_obstacles=True,
        n_obstacles=n_obst,
        obstacle_soft=True,
    )
    ctrl = MPCCppController(cfg, ref)
    ctrl.mu = 1.0

    radii = np.full(n_obst, OBSTACLE_RADIUS)
    ctrl.set_obstacles(OBSTACLE_CENTERS, radii)  # nominal positions at startup
    print(f"{n_obst} obstacles, shared keep-out r = {OBSTACLE_RADIUS:.2f} m")

    f_dyn = make_dynamics_fn(cfg)
    p0 = ref.eval(0.0)
    t0 = ref.tangent(0.0)
    x = ctrl.initial_state(p=p0, q=[1, 0, 0, 0], v=1.0 * t0, w=[0, 0, 0], vtheta=1.0)

    log = {k: [] for k in ("t", "pos", "speed", "theta", "vtheta", "viol", "obs_dist", "status")}
    for i in range(n_steps):
        # --- online obstacle update (here measured == nominal) --------------
        measured_centers = OBSTACLE_CENTERS  # replace with live estimates
        ctrl.set_obstacles(measured_centers, radii)

        res = ctrl.solve(x)

        th = res["theta"]
        nfr, bfr = ref.frame(th)
        d = x[0:3] - ref.eval(th)
        W, H = ref.width(th)
        viol = max(abs(d @ nfr) - W, abs(d @ bfr) - H, 0.0)
        obs_dist = float(np.min(np.linalg.norm(measured_centers - x[0:2], axis=1)))

        log["t"].append(i * cfg.dt)
        log["pos"].append(x[0:3].copy())
        log["speed"].append(float(np.linalg.norm(x[7:10])))
        log["theta"].append(float(x[IDX_THETA]))
        log["vtheta"].append(float(x[IDX_VTHETA]))
        log["viol"].append(viol)
        log["obs_dist"].append(obs_dist)
        log["status"].append(res["status"])
        if res["status"] not in (0, 1, 2):
            print(f"[{i}] solver status {res['status']}")
        x = rk4_step(f_dyn, x, res["u0"], cfg.dt)
        if i % 20 == 0:
            print(
                f"t={i * cfg.dt:5.2f}s  theta={x[IDX_THETA]:6.2f}m "
                f"|v|={np.linalg.norm(x[7:10]):5.2f}m/s  "
                f"tunnel_viol={viol:.3f}m  min_obs_dist={obs_dist:5.2f}m"
            )

    log = {k: np.array(v) for k, v in log.items()}
    print(f"\ncovered {x[IDX_THETA]:.1f} m (~{x[IDX_THETA] / ref.length:.2f} laps)")
    print(f"max tunnel violation   = {log['viol'].max():.3f} m")
    print(
        f"min distance to a stick = {log['obs_dist'].min():.3f} m "
        f"(keep-out radius = {OBSTACLE_RADIUS:.2f} m)"
    )
    return ref, (OBSTACLE_CENTERS, OBSTACLE_RADIUS), log


def tunnel_corners(ref: TunnelReferencePath, n_theta: int = 220) -> np.ndarray:
    """Sample the four tunnel-corner rails along the path, shape (n_theta, 4, 3)."""
    ths = np.linspace(0.0, ref.length, n_theta)
    C = np.zeros((n_theta, 4, 3))
    for i, th in enumerate(ths):
        pd = ref.eval(th)
        n, b = ref.frame(th)
        W, H = ref.width(th)
        C[i, 0] = pd + W * n + H * b
        C[i, 1] = pd + W * n - H * b
        C[i, 2] = pd - W * n - H * b
        C[i, 3] = pd - W * n + H * b
    return C


def plot(
    ref: TunnelReferencePath,
    obstacles: tuple[np.ndarray, float],
    log: dict,
    save_prefix: str = "mpccpp",
) -> None:
    """Plot the 3D trajectory + tunnel + obstacles, and the top view + time profiles."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    obs_centers, obs_r = obstacles
    ths = np.linspace(0, ref.length, 600)
    ctr = np.array([ref.eval(t) for t in ths])
    C = tunnel_corners(ref)
    pos = log["pos"]
    spd = log["speed"]
    M = len(ref.gate_centers)

    faces = []
    for a, bb in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        for i in range(len(C) - 1):
            faces.append([C[i, a], C[i, bb], C[i + 1, bb], C[i + 1, a]])

    # ---- 3D trajectory + tube + gates + stick keep-outs ----------------
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(faces, facecolor="tab:blue", alpha=0.10, edgecolor="none"))
    ax.plot(ctr[:, 0], ctr[:, 1], ctr[:, 2], "k--", lw=1.0, alpha=0.6, label="reference path")
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=spd, cmap="viridis", s=8)
    for j in range(M):
        r = ref.gate_rect(j)
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color="red", lw=2)
        gc = ref.gate_centers[j]
        ax.text(gc[0], gc[1], gc[2] + 0.3, f"G{j + 1}", color="red", fontsize=8)
    for oc in obs_centers:
        _draw_stick(ax, oc, obs_r, z0=0.0, z1=3.0)
    fig.colorbar(sc, ax=ax, label="speed [m/s]", shrink=0.6, pad=0.1)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("MPCC++: tunnel + soft avoidance of 4 sticks")
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
    for oc in obs_centers:
        a.add_patch(plt.Circle(oc, obs_r, color="darkorange", alpha=0.6))
        a.plot(oc[0], oc[1], "x", color="black", ms=6)
    a.set_xlabel("x [m]")
    a.set_ylabel("y [m]")
    a.set_title("top view + tube + sticks")
    a.set_aspect("equal", adjustable="datalim")

    axs[0, 1].plot(log["t"], log["speed"])
    axs[0, 1].set_title("speed |v| [m/s]")
    axs[0, 2].plot(log["t"], log["viol"])
    axs[0, 2].set_title("tunnel violation [m]")
    axs[1, 0].plot(log["t"], log["vtheta"])
    axs[1, 0].set_title("progress speed vtheta [m/s]")
    axs[1, 1].plot(log["t"], log["theta"])
    axs[1, 1].set_title("progress theta [m]")
    axs[1, 2].plot(log["t"], log["obs_dist"])
    axs[1, 2].axhline(obs_r, color="darkorange", ls="--")
    axs[1, 2].set_title("distance to nearest stick [m] (dashed = keep-out)")
    for ax2 in axs.flat:
        ax2.set_xlabel("t [s]")
        ax2.grid(alpha=0.3)
    axs[0, 0].set_xlabel("x [m]")
    fig2.tight_layout()
    fig2.savefig(f"{save_prefix}_profiles.png", dpi=140)
    print(f"saved {save_prefix}_trajectory.png and {save_prefix}_profiles.png")
    plt.show()


def _draw_stick(
    ax: object, xy: np.ndarray, r: float, z0: float, z1: float, color: str = "darkorange"
) -> None:
    """Draw a cylindrical obstacle stick from z0 to z1 at xy on a 3D axis."""
    ax.plot([xy[0], xy[0]], [xy[1], xy[1]], [z0, z1], color="black", lw=2)
    phi = np.linspace(0, 2 * np.pi, 24)
    zc = np.linspace(z0, z1, 2)
    Phi, Zc = np.meshgrid(phi, zc)
    Xc = xy[0] + r * np.cos(Phi)
    Yc = xy[1] + r * np.sin(Phi)
    ax.plot_surface(Xc, Yc, Zc, color=color, alpha=0.2, linewidth=0)


def _set_equal_3d(ax: object, pts: np.ndarray) -> None:
    """Set equal-aspect 3D axis limits covering all points, on a 3D axis."""
    mins = pts.min(0)
    maxs = pts.max(0)
    c = (mins + maxs) / 2.0
    r = (maxs - mins).max() / 2.0 + 0.5
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(max(0, c[2] - r), c[2] + r)


def main() -> None:
    """Run the closed-loop simulation and plot the results."""
    ref, obstacles, log = simulate(n_steps=800)
    try:
        plot(ref, obstacles, log)
    except ImportError:
        print("matplotlib not available -- skipping plots")


if __name__ == "__main__":
    main()
