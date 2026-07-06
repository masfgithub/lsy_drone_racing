"""Visualize the gate collision region as defined by _check_gate3.

Samples a 3D grid of points around each gate, runs each through the
collision check, and plots the points classified as "inside the frame"
in red. Use this to verify the check's geometry matches your physical gate.

Run:  python gate_area_debug.py
"""

import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")

from dataclasses import dataclass, field
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from lsy_drone_racing.control.planner.planner import FRAME_OPENING, FRAME_THICK, FRAME_WIDTH
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.planner.smart_planner import SplinePlanner

try:
    from lsy_drone_racing.control.env_obs import EnvState
except Exception:

    @dataclass
    class EnvState:
        """Fallback observation state, used when the real EnvState is unavailable."""

        p_bll: np.ndarray = field(default_factory=lambda: np.zeros(3))
        v_bll: np.ndarray = field(default_factory=lambda: np.zeros(3))
        w_bll: np.ndarray = field(default_factory=lambda: np.zeros(3))
        q_blb: np.ndarray = field(default_factory=lambda: np.zeros(4))
        p_tll_array: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))
        p_tll_index: int = 0
        q_tlt_array: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
        p_oll_array: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))
        h_t: float = 0.3
        l_t: float = 0.3
        w_t: float = 0.02


# ---------- helpers ----------------------------------------------------------


def make_obs(start: np.ndarray, gates: list, obstacles: list) -> EnvState:
    """Build a minimal EnvState from a start position, gate list, and obstacle list."""
    obs = EnvState()
    obs.p_bll = np.asarray(start, float)
    obs.v_bll = np.zeros(3)
    obs.w_bll = np.zeros(3)
    obs.q_blb = np.array([0.0, 0.0, 0.0, 1.0])
    obs.p_tll_array = np.array([g[0] for g in gates], float)
    obs.q_tlt_array = np.array([R.from_euler("Z", g[1]).as_quat() for g in gates])
    obs.p_tll_index = 0
    obs.p_oll_array = (
        np.asarray(obstacles, float).reshape(-1, 3) if len(obstacles) else np.zeros((0, 3))
    )
    return obs


def sample_grid_around_gate(
    gate_pos: np.ndarray, gate_yaw: float, resolution: int = 35, margin: float = 0.2
) -> np.ndarray:
    """Build a 3D grid of world-frame points around the gate."""
    half = max(FRAME_WIDTH / 2, FRAME_THICK / 2) + margin

    body_x = np.linspace(-half, half, resolution)
    body_y = np.linspace(-half, half, resolution)
    body_z = np.linspace(-half, half, resolution)
    BX, BY, BZ = np.meshgrid(body_x, body_y, body_z, indexing="ij")

    cos_y, sin_y = np.cos(gate_yaw), np.sin(gate_yaw)
    WX = gate_pos[0] + cos_y * BX - sin_y * BY
    WY = gate_pos[1] + sin_y * BX + cos_y * BY
    WZ = gate_pos[2] + BZ

    return np.stack([WX.ravel(), WY.ravel(), WZ.ravel()], axis=1)


def classify_points(
    planner: object, pts_world: np.ndarray, gate_pos: np.ndarray, gate_yaw: float
) -> np.ndarray:
    """Run each point through _check_gate3, return boolean mask of 'inside'."""
    pGLL_single = np.array([gate_pos])
    yaw_single = np.array([gate_yaw])

    inside = np.zeros(len(pts_world), dtype=bool)
    for k, p in enumerate(pts_world):
        is_inside, _, _ = planner._check_gate3(p, pGLL_single, yaw_single)
        inside[k] = is_inside
    return inside


def draw_gate_outlines(ax: object, gate_pos: np.ndarray, gate_yaw: float) -> None:
    """Add the nominal frame and opening outlines to the plot."""
    w = np.array([-np.sin(gate_yaw), np.cos(gate_yaw), 0.0])
    zz = np.array([0.0, 0.0, 1.0])

    for half_size, color, label in [
        (FRAME_WIDTH / 2, "blue", "outer frame"),
        (FRAME_OPENING / 2, "cyan", "opening"),
    ]:
        corners = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
        outline = np.array([gate_pos + a * half_size * w + b * half_size * zz for a, b in corners])
        ax.plot(outline[:, 0], outline[:, 1], outline[:, 2], color=color, linewidth=2, label=label)

    # Through-direction arrow
    forward = np.array([np.cos(gate_yaw), np.sin(gate_yaw), 0.0])
    ax.quiver(
        gate_pos[0],
        gate_pos[1],
        gate_pos[2],
        forward[0],
        forward[1],
        forward[2],
        length=0.3,
        color="green",
        label="through-axis",
    )


def plot_single_gate(
    planner: object,
    gate_pos: np.ndarray,
    gate_yaw: float,
    gate_idx: int,
    resolution: int = 35,
    margin: float = 0.2,
    save_dir: str = ".",
) -> plt.Figure:
    """Generate a single-gate visualization."""
    pts_world = sample_grid_around_gate(gate_pos, gate_yaw, resolution, margin)
    inside = classify_points(planner, pts_world, gate_pos, gate_yaw)
    n_inside = inside.sum()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    in_pts = pts_world[inside]
    if len(in_pts) > 0:
        ax.scatter(
            in_pts[:, 0],
            in_pts[:, 1],
            in_pts[:, 2],
            c="red",
            s=8,
            alpha=0.6,
            label=f"inside ({n_inside} pts)",
        )

    out_pts = pts_world[~inside][::25]
    if len(out_pts) > 0:
        ax.scatter(
            out_pts[:, 0],
            out_pts[:, 1],
            out_pts[:, 2],
            c="lightgray",
            s=2,
            alpha=0.15,
            label="outside (sparse)",
        )

    draw_gate_outlines(ax, gate_pos, gate_yaw)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(
        f"Gate {gate_idx}: pos={tuple(np.round(gate_pos, 2))}, "
        f"yaw={np.rad2deg(gate_yaw):.1f}°\n"
        f"{n_inside}/{len(pts_world)} points flagged as inside frame"
    )
    ax.legend(loc="upper right", fontsize=8)

    # Equal aspect
    max_range = (max(FRAME_WIDTH / 2, FRAME_THICK / 2) + margin) * 2
    ax.set_box_aspect([max_range, max_range, max_range])

    path = os.path.join(save_dir, f"gate_check_{gate_idx}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"wrote {path}  ({n_inside} inside)")
    return fig


def plot_slice_views(
    planner: object,
    gate_pos: np.ndarray,
    gate_yaw: float,
    gate_idx: int,
    resolution: int = 80,
    margin: float = 0.2,
    save_dir: str = ".",
) -> plt.Figure:
    """Generate 2D slices through the gate for easier inspection.

    Three slices: through the gate plane (xy at gate z),
    through the through-axis vertically (xz at gate y),
    and frontal (yz at gate x).
    """
    half = max(FRAME_WIDTH / 2, FRAME_THICK / 2) + margin

    cos_y, sin_y = np.cos(gate_yaw), np.sin(gate_yaw)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ---- Slice 1: frontal view (body yz plane, body_x = 0) ----
    ax = axes[0]
    by = np.linspace(-half, half, resolution)
    bz = np.linspace(-half, half, resolution)
    BY, BZ = np.meshgrid(by, bz, indexing="ij")
    # Body x = 0 means we're exactly in the gate's plane
    WX = gate_pos[0] - sin_y * BY
    WY = gate_pos[1] + cos_y * BY
    WZ = gate_pos[2] + BZ
    pts = np.stack([WX.ravel(), WY.ravel(), WZ.ravel()], axis=1)
    inside = classify_points(planner, pts, gate_pos, gate_yaw).reshape(BY.shape)

    ax.imshow(inside.T, extent=[-half, half, -half, half], origin="lower", cmap="Reds", alpha=0.7)
    # Frame outline
    for s, color in [(FRAME_WIDTH / 2, "blue"), (FRAME_OPENING / 2, "cyan")]:
        ax.plot([-s, s, s, -s, -s], [-s, -s, s, s, -s], color=color, lw=2)
    ax.set_xlabel("body y (left)")
    ax.set_ylabel("body z (up)")
    ax.set_title('Frontal slice (body_x = 0)\nthe "window" view')
    ax.set_aspect("equal")

    # ---- Slice 2: top-down view (body xy plane, body_z = 0) ----
    ax = axes[1]
    bx = np.linspace(-half, half, resolution)
    by = np.linspace(-half, half, resolution)
    BX, BY = np.meshgrid(bx, by, indexing="ij")
    WX = gate_pos[0] + cos_y * BX - sin_y * BY
    WY = gate_pos[1] + sin_y * BX + cos_y * BY
    WZ = np.full_like(BX, gate_pos[2])
    pts = np.stack([WX.ravel(), WY.ravel(), WZ.ravel()], axis=1)
    inside = classify_points(planner, pts, gate_pos, gate_yaw).reshape(BX.shape)

    ax.imshow(inside.T, extent=[-half, half, -half, half], origin="lower", cmap="Reds", alpha=0.7)
    ax.axvline(-FRAME_THICK / 2, color="blue", lw=1, linestyle="--")
    ax.axvline(FRAME_THICK / 2, color="blue", lw=1, linestyle="--")
    ax.axhline(-FRAME_WIDTH / 2, color="blue", lw=1)
    ax.axhline(FRAME_WIDTH / 2, color="blue", lw=1)
    ax.axhline(-FRAME_OPENING / 2, color="cyan", lw=1)
    ax.axhline(FRAME_OPENING / 2, color="cyan", lw=1)
    ax.set_xlabel("body x (through)")
    ax.set_ylabel("body y (left)")
    ax.set_title("Top-down slice (body_z = 0)\nshows frame thickness profile")
    ax.set_aspect("equal")

    # ---- Slice 3: side view (body xz plane, body_y = 0) ----
    ax = axes[2]
    bx = np.linspace(-half, half, resolution)
    bz = np.linspace(-half, half, resolution)
    BX, BZ = np.meshgrid(bx, bz, indexing="ij")
    WX = gate_pos[0] + cos_y * BX
    WY = gate_pos[1] + sin_y * BX
    WZ = gate_pos[2] + BZ
    pts = np.stack([WX.ravel(), WY.ravel(), WZ.ravel()], axis=1)
    inside = classify_points(planner, pts, gate_pos, gate_yaw).reshape(BX.shape)

    ax.imshow(inside.T, extent=[-half, half, -half, half], origin="lower", cmap="Reds", alpha=0.7)
    ax.axvline(-FRAME_THICK / 2, color="blue", lw=1, linestyle="--")
    ax.axvline(FRAME_THICK / 2, color="blue", lw=1, linestyle="--")
    ax.axhline(-FRAME_WIDTH / 2, color="blue", lw=1)
    ax.axhline(FRAME_WIDTH / 2, color="blue", lw=1)
    ax.axhline(-FRAME_OPENING / 2, color="cyan", lw=1)
    ax.axhline(FRAME_OPENING / 2, color="cyan", lw=1)
    ax.set_xlabel("body x (through)")
    ax.set_ylabel("body z (up)")
    ax.set_title("Side slice (body_y = 0)")
    ax.set_aspect("equal")

    fig.suptitle(f'Gate {gate_idx} slices: red = "inside" per _check_gate3')
    fig.tight_layout()

    path = os.path.join(save_dir, f"gate_check_slices_{gate_idx}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"wrote {path}")
    return fig


# ---------- main -------------------------------------------------------------


def main() -> None:
    """Generate gate-check visualizations for a fixed test track."""
    # Same gates as your level 2 nominal track
    start = [-1.5, 0.75, 0.01]
    gates = [
        (np.array([0.5, 0.25, 0.7]), -0.78),
        (np.array([1.05, 0.75, 1.2]), 2.35),
        (np.array([-1.0, -0.25, 0.7]), 3.14),
        (np.array([0.0, -0.75, 1.2]), 0.0),
    ]
    obstacles = [[0.0, 0.75, 1.55], [1.0, 0.25, 1.55], [-1.5, -0.25, 1.55], [-0.5, -0.75, 1.55]]

    cfg = SimpleNamespace(env=SimpleNamespace(freq=50))
    obs = make_obs(start, gates, obstacles)
    planner = SplinePlanner(obs, {}, cfg, t_total=12)

    print(f"FRAME_WIDTH   = {FRAME_WIDTH}")
    print(f"FRAME_OPENING = {FRAME_OPENING}")
    print(f"FRAME_THICK   = {FRAME_THICK}")
    print()

    # Visualize each gate
    for k, (gpos, gyaw) in enumerate(gates):
        plot_single_gate(planner, gpos, gyaw, k, resolution=30)
        plot_slice_views(planner, gpos, gyaw, k, resolution=80)

    plt.show()


if __name__ == "__main__":
    main()
