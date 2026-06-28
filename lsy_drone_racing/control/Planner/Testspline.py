"""Interactive 3D viewer for SplinePlanner — no simulator required.

Runs the planner on a fake EnvState_t (start + 2 gates + obstacles) and writes
an interactive Plotly HTML you open in a browser: rotate, zoom, and hover any
waypoint to read its index and coordinates. Trajectory, gates, obstacle pillars
(with keep-out shells) and gate-post keep-outs are all drawn.

The SCIPY_ARRAY_API line MUST stay first (crazyflow, pulled in by
lsy_drone_racing, refuses to load otherwise).

Run:  python view_spline_3d.py        # writes scene.html, open it in a browser
"""
import os
os.environ.setdefault("SCIPY_ARRAY_API", "1")

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt   # NOTE: don't force a backend here, so plt.show() can open a window

from lsy_drone_racing.control.Planner.smart_planner import SplinePlanner
from lsy_drone_racing.control.Planner.planner import FRAME_WIDTH, FRAME_OPENING, CLEARANCE

try:
    from lsy_drone_racing.control.env_obs import EnvState_t
except Exception:                       # crazyflow chain unavailable -> local stand-in
    @dataclass
    class EnvState_t:
        pBLL: np.ndarray = field(default_factory=lambda: np.zeros(3))
        vBLL: np.ndarray = field(default_factory=lambda: np.zeros(3))
        wBLL: np.ndarray = field(default_factory=lambda: np.zeros(3))
        qBLB: np.ndarray = field(default_factory=lambda: np.zeros(4))
        pTLL_array: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))
        pTLL_index: int = 0
        qTLT_array: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
        pOLL_array: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))
        hT: float = 0.3
        lT: float = 0.3
        wT: float = 0.02

from types import SimpleNamespace

POST_R = 0.10 + 0.05
HOLE_H = 0.23
MARGIN = 0.05


def make_obs(start, gates, obstacles) -> "EnvState_t":
    obs = EnvState_t()
    obs.pBLL = np.asarray(start, float)
    obs.vBLL = np.zeros(3)
    obs.wBLL = np.zeros(3)
    obs.qBLB = np.array([0.0, 0.0, 0.0, 1.0])
    obs.pTLL_array = np.array([g[0] for g in gates], float)
    obs.qTLT_array = np.array([R.from_euler("Z", g[1]).as_quat() for g in gates])
    obs.pTLL_index = 0
    obs.pOLL_array = (np.asarray(obstacles, float).reshape(-1, 3)
                      if len(obstacles) else np.zeros((0, 3)))
    return obs


def get_set_waypoints(planner, obs):
    """Exact waypoints if the planner exposes self._waypoints, else reconstruct
    the start + prev/gate/next set the way _build_waypoints does."""
    wps = getattr(planner, "_waypoints", None)
    if wps is not None and len(wps):
        return np.asarray(wps, float)
    pDLL = obs.pBLL
    pG, yaws = planner._gate(obs)
    D = 0.6
    out = [np.asarray(pDLL, float)]
    if np.linalg.norm(pDLL - pG[0]) > 1.2 * D:
        out.append(pG[0] - D * np.array([np.cos(yaws[0]), np.sin(yaws[0]), 0.0]))
    for i in range(len(pG)):
        out.append(pG[i])
        out.append(pG[i] + D * np.array([np.cos(yaws[i]), np.sin(yaws[i]), 0.0]))
    return np.array(out, float)


# ---- plotly trace builders --------------------------------------------------
def line_trace(pts, name, color, width=5, dash="solid"):
    pts = np.asarray(pts)
    return go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
                        line=dict(color=color, width=width, dash=dash), name=name)


def gate_frame_trace(c, yaw, half, color, name):
    w = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    zz = np.array([0.0, 0.0, 1.0])
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
    pts = np.array([c + a * half * w + b * half * zz for a, b in corners])
    return go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
                        line=dict(color=color, width=6), name=name, showlegend=False)


def cylinder_trace(c, radius, z0, z1, color, opacity, name):
    th = np.linspace(0, 2 * np.pi, 30)
    TH, Z = np.meshgrid(th, np.array([z0, z1]))
    X = c[0] + radius * np.cos(TH)
    Y = c[1] + radius * np.sin(TH)
    return go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=opacity,
                      colorscale=[[0, color], [1, color]], name=name, showlegend=False)


def build_figure(start, gates, obstacles, traj, traj0, waypoints):
    fig = go.Figure()
    fig.add_trace(line_trace(traj0, "nominal (no obstacles)", "lightgray", 3, "dash"))
    fig.add_trace(line_trace(traj, "planned (with obstacles)", "green", 6))

    # waypoints: labelled, hover shows index + coords
    fig.add_trace(go.Scatter3d(
        x=waypoints[:, 0], y=waypoints[:, 1], z=waypoints[:, 2],
        mode="markers+text",
        marker=dict(size=5, color="orange", symbol="diamond", line=dict(color="black", width=1)),
        text=[str(i) for i in range(len(waypoints))],
        textposition="top center",
        hovertemplate="wp %{text}<br>x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra></extra>",
        name="waypoints"))

    fig.add_trace(go.Scatter3d(x=[start[0]], y=[start[1]], z=[start[2]], mode="markers",
                  marker=dict(size=7, color="black"), name="start"))

    for k, (c, yaw) in enumerate(gates):
        fig.add_trace(gate_frame_trace(c, yaw, FRAME_WIDTH / 2, "royalblue", f"gate{k} frame"))
        fig.add_trace(gate_frame_trace(c, yaw, FRAME_OPENING / 2, "deepskyblue", f"gate{k} hole"))
        z_top = c[2] - HOLE_H / 2 - MARGIN - POST_R
        fig.add_trace(cylinder_trace(c, POST_R, 0.0, max(z_top, 0.01), "orange", 0.25, f"post{k}"))

    r_keep = 0.15 + CLEARANCE
    for j, o in enumerate(obstacles):
        fig.add_trace(cylinder_trace(o, 0.15, 0.0, 2.0, "firebrick", 0.6, f"pillar{j}"))
        fig.add_trace(cylinder_trace(o, r_keep, 0.0, 2.0, "orange", 0.12, f"keepout{j}"))

    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
        title="SplinePlanner trajectory + waypoints", margin=dict(l=0, r=0, t=40, b=0))
    return fig


def main():
    start = [-1.5, 0.75, 0.01]
    gates = [(np.array([0.5,  0.25, 0.7]), -0.78),
            (np.array([1.05, 0.75, 1.2]),  2.35),
            (np.array([-1.0, -0.25, 0.7]), 3.14),
            (np.array([0.0, -0.75, 1.2]),  0.0)]
    obstacles = [[0.0,  0.75, 1.55],
                [1.0,  0.25, 1.55],
                [-1.27, -0.4, 1.55],
                [-0.7, -0.75, 1.55]]
    
    start = [-0.3, 0.0, 0.01]
    gates = [(np.array([-1.0, -0.25, 0.7]), 3.14),
            (np.array([0.0, -0.75, 1.2]),  0.0)]
    obstacles = [[-1.27, -0.4, 1.55]]

    cfg = SimpleNamespace(env=SimpleNamespace(freq=50))
    planner = SplinePlanner(make_obs(start, gates, obstacles), {}, cfg, t_total=12)
    obs = make_obs(start, gates, obstacles)
    traj = planner.plan(obs, 0.0).positions
    waypoints = get_set_waypoints(planner, obs)

    obs0 = make_obs(start, gates, [])
    traj0 = SplinePlanner(obs0, {}, cfg, t_total=12).plan(obs0, 0.0).positions

    fig = build_figure(start, gates, obstacles, traj, traj0, waypoints)
    fig.write_html("scene.html", include_plotlyjs="cdn", auto_open=False)
    print(f"wrote scene.html  ({len(waypoints)} waypoints, traj {traj.shape[0]} pts)")
    print("open scene.html in a browser; rotate/zoom and hover the orange diamonds")
    fig = build_figure(start, gates, obstacles, traj, traj0, waypoints)
    fig.write_html("scene.html", include_plotlyjs="cdn", auto_open=False)

    plot_matplotlib(start, gates, obstacles, traj, traj0, waypoints,
                    save_path="scene.png", show=True)   # <-- interactive window + PNG

# ---- matplotlib (static / interactive 3D, no browser) -----------------------
def _mpl_gate(ax, c, yaw, half, color):
    w  = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    zz = np.array([0.0, 0.0, 1.0])
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
    pts = np.array([np.asarray(c) + a * half * w + b * half * zz for a, b in corners])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=2)


def _mpl_cylinder(ax, c, radius, z0, z1, color, alpha):
    th = np.linspace(0, 2 * np.pi, 28)
    T, Z = np.meshgrid(th, np.array([z0, z1]))
    X = c[0] + radius * np.cos(T)
    Y = c[1] + radius * np.sin(T)
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, shade=False)


def _set_equal_3d(ax):
    xl, yl, zl = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    ax.set_box_aspect((xl[1] - xl[0], yl[1] - yl[0], zl[1] - zl[0]))  # true scale


def plot_matplotlib(start, gates, obstacles, traj, traj0, waypoints,
                    save_path="scene.png", show=True):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(traj0[:, 0], traj0[:, 1], traj0[:, 2], "--", color="0.6", lw=1.5, label="nominal")
    ax.plot(traj[:, 0],  traj[:, 1],  traj[:, 2],  "-",  color="g",   lw=2.5, label="planned")

    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
               c="orange", marker="D", s=55, edgecolor="k", depthshade=False, label="waypoints")
    for i, p in enumerate(waypoints):
        ax.text(p[0], p[1], p[2] + 0.05, str(i), fontsize=8)

    ax.scatter(start[0], start[1], start[2], c="k", s=90, label="start")

    for c, yaw in gates:
        _mpl_gate(ax, c, yaw, FRAME_WIDTH / 2, "tab:blue")     # outer frame
        _mpl_gate(ax, c, yaw, FRAME_OPENING / 2, "tab:cyan")   # hole
        z_top = c[2] - HOLE_H / 2 - MARGIN - POST_R
        _mpl_cylinder(ax, c, POST_R, 0.0, max(z_top, 0.01), "tab:orange", 0.3)

    r_keep = 0.15 + CLEARANCE
    for o in obstacles:
        _mpl_cylinder(ax, o, 0.15,   0.0, 2.0, "tab:red",    0.5)   # pillar
        _mpl_cylinder(ax, o, r_keep, 0.0, 2.0, "tab:orange", 0.12)  # keep-out shell

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("SplinePlanner trajectory + waypoints")
    ax.legend(loc="upper left")
    _set_equal_3d(ax)

    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        print(f"wrote {save_path}")
    if show:
        plt.show()
    return fig

    
if __name__ == "__main__":
    main()