"""Visualization helpers for the MPCC++ controller and planner."""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_capsule, draw_line, draw_points

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

    from lsy_drone_racing.control.drone_racing_pipeline_config import PLANNER_TYPE
    from lsy_drone_racing.control.mpcc.mpccpp import MPCCpp

    if PLANNER_TYPE == "Smart":
        from lsy_drone_racing.control.planner.smart_planner import SplinePlanner
    elif PLANNER_TYPE == "Lightweight":
        from lsy_drone_racing.control.planner.lightweight_planner import SplinePlanner


def render_mpccpp(sim: Sim, planner: SplinePlanner, controller: MPCCpp):
    """Visualize the planned path, MPC predictions, gates, and obstacles."""
    # Planned path (green)
    trajectory = planner.get_pos_traj()
    draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

    # Tunnel centerline for MPCC++ (yellow)
    # _draw_tunnel_centerline(sim, self._controller._ref)

    # MPC predicted trajectory (purple dots)
    pred_trajectory = controller.get_predicted_traj()
    for p in pred_trajectory:
        draw_points(sim, p.reshape(1, -1), rgba=(0.58, 0.0, 0.83, 0.5), size=0.01)

    # MPCC++ prediction tunnel (cyan edges + yellow corners)
    _draw_mpccpp_tunnel(sim, controller)
    # Reference trajectory (red dots)
    # ref_trajectory = self._controller.get_ref_traj()
    # for p in ref_trajectory:
    #    draw_points(sim, p.reshape(1, -1), rgba=(1.0, 0.0, 0.0, 0.5), size=0.01)

    # Gates
    #for gate in controller._gates:
    #    _draw_wedge_gate(
    #        sim,
    #        position=gate.position,
    #        quaternion=gate.quaternion,
    #        total_length=gate.total_length,
    #        total_height=gate.total_height,
    #        hole_width=gate.hole_width,
    #        hole_height=gate.hole_height,
    #        thickness=gate.thickness,
    #        rgba=np.array([0.0, 0.5, 1.0, 1.0]),
    #    )

    ## Obstacles
    #for obs in controller._obstacles:
    #    _draw_cylinder_obstacle(
    #        sim,
    #        position=obs.position,
    #        height=obs.total_height,
    #        radius=obs.d_min,
    #        rgba=np.array([1.0, 0.2, 0.2, 0.7]),
    #    )


def _draw_wedge_gate(
    sim: Sim,
    position: NDArray,
    quaternion: NDArray,
    total_length: float,
    total_height: float,
    hole_width: float,
    hole_height: float,
    thickness: float = 0.05,
    rgba: NDArray | None = None,
    radius: float = 0.015,
):
    """Draw the four wedge prisms of a WedgeWindow gate as capsule edges.

    Each wedge has 6 vertices (4 base corners + 2 tip corners) and 9 edges.
    The edges are drawn as capsules, matching the WedgeWindow geometry exactly.

    Gate-local frame: x = gate normal, y = width, z = height.

        Left / Right:  base at y = ±hl  (x∈[-a_x,+a_x], z∈[-hh,+hh])
                       tip  at y = ±hw  (x=0, z∈[-hho,+hho])
        Top  / Bottom: base at z = ±hh  (x∈[-a_x,+a_x], y∈[-hl,+hl])
                       tip  at z = ±hho (x=0, y∈[-hw,+hw])
    """
    if sim.viewer is None:
        return

    if rgba is None:
        rgba = np.array([0.0, 0.5, 1.0, 1.0])

    qw, qx, qy, qz = quaternion / np.linalg.norm(quaternion)
    r_mat = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )

    def to_world(lv: np.ndarray) -> np.ndarray:
        return np.asarray(position) + r_mat @ lv

    def edge(a: np.ndarray, b: np.ndarray):
        draw_capsule(sim, to_world(a), to_world(b), radius=radius, rgba=rgba, cylinder=True)

    a_x = thickness / 2.0
    hl = total_length / 2.0
    hh = total_height / 2.0
    hw = hole_width / 2.0
    hho = hole_height / 2.0

    def draw_prism(
        base_d: float,
        tip_d: float,
        depth_idx: int,
        ax_idx: int,
        perp_idx: int,
        h_perp_base: float,
        h_perp_tip: float,
    ) -> None:
        """Draw one wedge prism as 9 capsule edges.

        depth_idx  : axis of tapering (1=y for L/R, 2=z for T/B)
        ax_idx     : gate normal axis (always 0 = x)
        perp_idx   : remaining axis   (2=z for L/R, 1=y for T/B)
        base_d     : depth coordinate at base
        tip_d      : depth coordinate at tip
        h_perp_base: perp half-extent at base
        h_perp_tip : perp half-extent at tip
        """

        def pt(d: float, xv: float, perpv: float) -> np.ndarray:
            v = np.zeros(3)
            v[depth_idx] = d
            v[ax_idx] = xv
            v[perp_idx] = perpv
            return v

        # 4 base corners
        b0 = pt(base_d, -a_x, h_perp_base)
        b1 = pt(base_d, a_x, h_perp_base)
        b2 = pt(base_d, a_x, -h_perp_base)
        b3 = pt(base_d, -a_x, -h_perp_base)
        # 2 tip corners
        t0 = pt(tip_d, 0.0, h_perp_tip)
        t1 = pt(tip_d, 0.0, -h_perp_tip)

        # Base rectangle (4 edges)
        edge(b0, b1)
        edge(b1, b2)
        edge(b2, b3)
        edge(b3, b0)
        # Slant edges base → tip (4 edges)
        edge(b0, t0)
        edge(b1, t0)
        edge(b2, t1)
        edge(b3, t1)
        # Tip edge (1 edge)
        edge(t0, t1)

    # Left:   depth=y(1), tip toward +y; base at -hl, tip at -hw
    draw_prism(-hl, -hw, depth_idx=1, ax_idx=0, perp_idx=2, h_perp_base=hh, h_perp_tip=hho)
    # Right:  base at +hl, tip at +hw
    draw_prism(hl, hw, depth_idx=1, ax_idx=0, perp_idx=2, h_perp_base=hh, h_perp_tip=hho)
    # Top:    depth=z(2); base at +hh, tip at +hho
    draw_prism(hh, hho, depth_idx=2, ax_idx=0, perp_idx=1, h_perp_base=hl, h_perp_tip=hw)
    # Bottom: base at -hh, tip at -hho
    draw_prism(-hh, -hho, depth_idx=2, ax_idx=0, perp_idx=1, h_perp_base=hl, h_perp_tip=hw)


def _draw_mpccpp_tunnel(
    sim: Sim,
    controller: object,
    ring_rgba: NDArray | None = None,
    corner_rgba: NDArray | None = None,
):
    """Draw the MPCC++ prediction tunnel.

    Draws the rectangular cross-section at every predicted horizon node
    (4 corners + 4 edges) plus the longitudinal rails.

    Cross-section at progress theta: centre ref.eval(theta), spanned by the
    tunnel frame (n, b) = ref.frame(theta) with half-extents (W, H) =
    ref.width(theta) -- exactly the prism enforced by the tunnel constraint.
    """
    if sim.viewer is None:
        return
    ref = getattr(controller, "_ref", None)
    if ref is None:
        return

    # per-node progress: solved theta state (index 14), aligned with the
    # predicted trajectory; fall back to the controller's theta_pred guess.
    if getattr(controller, "_x_warm", None):
        thetas = [float(x[14]) for x in controller._x_warm]
    elif getattr(controller, "_theta_pred", None) is not None:
        thetas = [float(t) for t in controller._theta_pred]
    else:
        return

    if ring_rgba is None:
        ring_rgba = np.array([0.1, 0.85, 0.95, 0.6])  # cyan edges
    if corner_rgba is None:
        corner_rgba = np.array([1.0, 0.9, 0.0, 0.9])  # yellow corners

    # corners[k] = 4x3, order (+n+b, -n+b, -n-b, +n-b)
    corners = []
    for th in thetas:
        pd = np.asarray(ref.eval(th), dtype=float)
        n, b = ref.frame(th)
        W, H = ref.width(th)
        n = np.asarray(n, float)
        b = np.asarray(b, float)
        corners.append(
            np.array(
                [pd + W * n + H * b, pd - W * n + H * b, pd - W * n - H * b, pd + W * n - H * b]
            )
        )
    corners = np.array(corners)  # (K, 4, 3)

    # cross-section rectangle (4 edges) at every prediction node
    for k in range(len(corners)):
        ring = np.vstack([corners[k], corners[k, 0]])  # closed (5,3)
        draw_line(sim, ring, rgba=ring_rgba)

    # longitudinal rails connecting node k -> k+1 along each corner
    for cidx in range(4):
        draw_line(sim, corners[:, cidx, :], rgba=ring_rgba)

    # corner markers
    # draw_points(sim, corners.reshape(-1, 3), rgba=corner_rgba, size=0.02)


def _draw_cylinder_obstacle(
    sim: Sim, position: NDArray, height: float, radius: float, rgba: NDArray | None = None
):
    """Draw a vertical cylinder obstacle as a single capsule in the simulation."""
    if sim.viewer is None:
        return

    if rgba is None:
        rgba = np.array([1.0, 0.2, 0.2, 0.8])

    pos = np.asarray(position)
    z_base = float(pos[2]) if len(pos) == 3 else 0.0
    cx, cy = float(pos[0]), float(pos[1])
    draw_capsule(
        sim,
        np.array([cx, cy, z_base]),
        np.array([cx, cy, z_base + height]),
        radius=radius,
        rgba=rgba,
        cylinder=True,
    )


def _draw_post(
    sim: Sim,
    gate_position: NDArray,
    r_post: float,
    hole_height: float,
    margin: float,
    z_floor: float = 0.0,
    rgba: NDArray | None = None,
):
    """Draw the gate-post keep-out as a vertical capsule below the opening.

    Mirrors post_penalty_sym: a capsule of radius r_post around the segment
    from the floor up to z_top = gate_z - hole_height/2 - margin - r_post.
    The rounded top therefore reaches gate_z - hole_height/2 - margin, just
    below the opening, exactly like the penalty's keep-out.
    """
    if sim.viewer is None:
        return
    if rgba is None:
        rgba = np.array([1.0, 0.6, 0.0, 0.5])  # orange: distinct from blue gates / red obstacles

    cx, cy, cz = float(gate_position[0]), float(gate_position[1]), float(gate_position[2])
    z_top = cz - hole_height / 2.0 - margin - r_post
    draw_capsule(
        sim,
        np.array([cx, cy, z_floor]),
        np.array([cx, cy, z_top]),
        radius=r_post,
        rgba=rgba,
        cylinder=False,  # rounded caps -> matches distance-to-segment keep-out
    )


def _draw_tunnel_centerline(sim: Sim, ref: object, n: int = 150, rgba: NDArray | None = None):
    """Draw the MPCC++ tunnel centerline as a sequence of line segments."""
    if sim.viewer is None:
        return
    if rgba is None:
        rgba = np.array([1.0, 0.8, 0.0, 0.8])
    s_vals = np.linspace(0.0, ref.length, n)
    pts = np.array([ref.eval(float(s)) for s in s_vals])
    draw_line(sim, pts, rgba=rgba)
