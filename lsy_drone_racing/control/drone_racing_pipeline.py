"""This module implements the pipeline for the drone racing.

TBD specify more in detail.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_capsule, draw_line

from lsy_drone_racing.control.basic_planner import BasicPlanner
from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.control.env_obs import extract_env_states
from lsy_drone_racing.control.nmpc.nmpc import NMPC

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


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
    R_mat = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )

    def to_world(lv: np.ndarray) -> np.ndarray:
        return np.asarray(position) + R_mat @ lv

    def edge(a: np.ndarray, b: np.ndarray):
        draw_capsule(sim, to_world(a), to_world(b), radius=radius, rgba=rgba, cylinder=True)

    a_x = thickness / 2.0
    hl = total_length / 2.0
    hh = total_height / 2.0
    hw = hole_width / 2.0
    hho = hole_height / 2.0

    def draw_prism(base_d, tip_d, depth_idx, ax_idx, perp_idx, h_perp_base, h_perp_tip):
        """Draw one wedge prism as 9 capsule edges.

        depth_idx  : axis of tapering (1=y for L/R, 2=z for T/B)
        ax_idx     : gate normal axis (always 0 = x)
        perp_idx   : remaining axis   (2=z for L/R, 1=y for T/B)
        base_d     : depth coordinate at base
        tip_d      : depth coordinate at tip
        h_perp_base: perp half-extent at base
        h_perp_tip : perp half-extent at tip
        """

        def pt(d, xv, perpv):
            v = np.zeros(3)
            v[depth_idx] = d
            v[ax_idx] = xv
            v[perp_idx] = perpv
            return v

        # 4 base corners
        B0 = pt(base_d, -a_x, h_perp_base)
        B1 = pt(base_d, a_x, h_perp_base)
        B2 = pt(base_d, a_x, -h_perp_base)
        B3 = pt(base_d, -a_x, -h_perp_base)
        # 2 tip corners
        T0 = pt(tip_d, 0.0, h_perp_tip)
        T1 = pt(tip_d, 0.0, -h_perp_tip)

        # Base rectangle (4 edges)
        edge(B0, B1)
        edge(B1, B2)
        edge(B2, B3)
        edge(B3, B0)
        # Slant edges base → tip (4 edges)
        edge(B0, T0)
        edge(B1, T0)
        edge(B2, T1)
        edge(B3, T1)
        # Tip edge (1 edge)
        edge(T0, T1)

    # Left:   depth=y(1), tip toward +y; base at -hl, tip at -hw
    draw_prism(-hl, -hw, depth_idx=1, ax_idx=0, perp_idx=2, h_perp_base=hh, h_perp_tip=hho)
    # Right:  base at +hl, tip at +hw
    draw_prism(hl, hw, depth_idx=1, ax_idx=0, perp_idx=2, h_perp_base=hh, h_perp_tip=hho)
    # Top:    depth=z(2); base at +hh, tip at +hho
    draw_prism(hh, hho, depth_idx=2, ax_idx=0, perp_idx=1, h_perp_base=hl, h_perp_tip=hw)
    # Bottom: base at -hh, tip at -hho
    draw_prism(-hh, -hho, depth_idx=2, ax_idx=0, perp_idx=1, h_perp_base=hl, h_perp_tip=hw)


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


class DroneRacingPipeline(Controller):
    """This class handles the pipeline for the drone racing. It includes planning and control."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the pipeline."""
        super().__init__(obs, info, config)

        t_total = 8
        env_states = extract_env_states(obs)
        self._tick = 0
        self._finished = False

        self._planner = BasicPlanner(config, t_total)
        planner_dict = self._planner.plan()
        self._controller = NMPC(env_states, planner_dict, info, config, t_total, use_soft=True)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone."""
        env_states = extract_env_states(obs)
        self._planner.replan()
        return self._controller.control(env_states, info)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1
        self._controller.set_tick(self._tick)
        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0
        self._controller.set_tick(self._tick)

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory, setpoint, gates and obstacles."""
        trajectory = self._planner.get_pos_traj()
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
        trajectory = self._controller.get_predicted_traj()
        draw_line(sim, trajectory, rgba=np.array([0.58, 0.0, 0.83, 1.0]))

        for gate in self._controller._gates:
            _draw_wedge_gate(
                sim,
                position=gate.position,
                quaternion=gate.quaternion,
                total_length=gate.total_length,
                total_height=gate.total_height,
                hole_width=gate.hole_width,
                hole_height=gate.hole_height,
                thickness=gate.thickness,
                rgba=np.array([0.0, 0.5, 1.0, 1.0]),
            )

        for obs in self._controller._obstacles:
            _draw_cylinder_obstacle(
                sim,
                position=obs.position,
                height=obs.total_height,
                radius=obs.d_min,
                rgba=np.array([1.0, 0.2, 0.2, 0.7]),
            )
