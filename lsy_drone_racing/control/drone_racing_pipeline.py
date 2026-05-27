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


def _draw_gate(
    sim: Sim,
    position: NDArray,
    quaternion: NDArray,
    total_length: float,
    total_height: float,
    hole_width: float,
    hole_height: float,
    thickness: float = 0.05,
    rgba: NDArray | None = None,
    radius: float = 0.02,
):
    """Draw a gate (window frame) as capsules accounting for wall thickness."""
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

    def to_world(local_vec: list) -> np.ndarray:
        return position + R_mat @ np.asarray(local_vec)

    hl = total_length / 2.0
    hh = total_height / 2.0
    hw = hole_width / 2.0
    hho = hole_height / 2.0
    hx = thickness / 2.0  # half-thickness along local x

    # Corner points at front face (+hx) and back face (-hx)
    # Outer corners
    otl_f = [+hx, -hl, hh]
    otl_b = [-hx, -hl, hh]
    otr_f = [+hx, hl, hh]
    otr_b = [-hx, hl, hh]
    obl_f = [+hx, -hl, -hh]
    obl_b = [-hx, -hl, -hh]
    obr_f = [+hx, hl, -hh]
    obr_b = [-hx, hl, -hh]

    # Inner (hole) corners
    tl_f = [+hx, -hw, hho]
    tl_b = [-hx, -hw, hho]
    tr_f = [+hx, hw, hho]
    tr_b = [-hx, hw, hho]
    bl_f = [+hx, -hw, -hho]
    bl_b = [-hx, -hw, -hho]
    br_f = [+hx, hw, -hho]
    br_b = [-hx, hw, -hho]

    def bar(a: list, b: list):
        draw_capsule(sim, to_world(a), to_world(b), radius=radius, rgba=rgba, cylinder=True)

    # ── Front face bars ───────────────────────────────────────────────────────
    bar(otl_f, obl_f)  # outer left vertical
    bar(otr_f, obr_f)  # outer right vertical
    bar(otl_f, otr_f)  # outer top horizontal
    bar(obl_f, obr_f)  # outer bottom horizontal
    bar(tl_f, tr_f)  # inner top horizontal
    bar(bl_f, br_f)  # inner bottom horizontal
    bar(tl_f, bl_f)  # inner left vertical
    bar(tr_f, br_f)  # inner right vertical

    # ── Back face bars ────────────────────────────────────────────────────────
    bar(otl_b, obl_b)
    bar(otr_b, obr_b)
    bar(otl_b, otr_b)
    bar(obl_b, obr_b)
    bar(tl_b, tr_b)
    bar(bl_b, br_b)
    bar(tl_b, bl_b)
    bar(tr_b, br_b)

    # ── Depth bars connecting front to back (along local x) ──────────────────
    # Outer corners
    bar(otl_f, otl_b)
    bar(otr_f, otr_b)
    bar(obl_f, obl_b)
    bar(obr_f, obr_b)
    # Inner (hole) corners
    bar(tl_f, tl_b)
    bar(tr_f, tr_b)
    bar(bl_f, bl_b)
    bar(br_f, br_b)


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
        """Initialize the pipeline.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)

        # variable setup
        t_total = 8
        env_states = extract_env_states(obs)  # align information with naming convention
        self._tick = 0
        self._finished = False

        # setup for planner
        self._planner = BasicPlanner(config, t_total)
        planner_dict = self._planner.plan()

        # setup for controller
        self._controller = NMPC(env_states, planner_dict, info, config, t_total, use_soft=True)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        env_states = extract_env_states(obs)  # align information with naming convention

        self._planner.replan()
        u0 = self._controller.control(env_states, info)
        return u0

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
            _draw_gate(
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
