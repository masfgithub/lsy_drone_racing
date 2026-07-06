"""Planner interface: abstract base class for all trajectory planners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from lsy_drone_racing.control.env_obs import EnvState

__all__ = ["Trajectory", "Planner", "DEFAULT_MAX_SPEED"]

DEFAULT_MAX_SPEED = 12.0  # m/s
FRAME_WIDTH = 0.7
FRAME_OPENING = 0.4
FRAME_THICK = 0.15
POST_WIDTH = 0.2
CLEARANCE = 0.1


@dataclass
class Trajectory:
    """Class to represent trajectory from planner with positions, velocities and timestamps."""

    positions: np.ndarray
    velocities: np.ndarray
    timestamps: np.ndarray

    # p_ref_LL: np.ndarray
    # v_ref_LL: np.ndarray
    # t_ref: np.ndarray


class Planner(ABC):
    """Abstract base class for drone trajectory planners."""

    def __init__(self, obs: EnvState, info: dict, config: dict):
        """Initialize Planner Class.

        Arg:
            obs:                Environment state observation.
            info:               Additional environment information.
            config:             Environment configuration.
        """
        self.freq = config.env.freq
        self._gates_information = {
            "total_length": 0.72,
            "total_height": 0.72,
            "hole_width": 0.25,
            "hole_height": 0.25,
            "thickness": 0.3,
            "margin": 0.05,
        }

    @abstractmethod
    def plan(self, obs: EnvState, info: dict, config: dict) -> Trajectory:
        """Compute a trajectory through the gates. Subclasses must implement.

        Args:
            obs:                Environment state observation.
            info:               Additional environment information.
            config:             Environment configuration.

        Returns:
            trajectory:         pos, vel, time in a trajectory class.
        """

    def _gate(self, obs: EnvState) -> tuple[np.ndarray, np.ndarray]:
        """Returns gate yaw and gate centre position from environment observation.

        Args:
            obs:                Environment state observation.

        Returns:
            y_GBL_array:        Gate orientation yaw of body relative to Local frame.
            pGLL_array:         Gate center position of Target relative to Local frame in Local
                                coordinates.
        """
        # Quaternion of gate frames
        qTLT = obs.q_tlt_array
        p_tll_index = obs.p_tll_index

        # Extracted rotation matrix/Euler angles from the quaternion
        y_GBL_array = R.from_quat(qTLT[p_tll_index:]).as_euler("ZYX")[:, 0]

        # Centre position of gate frames
        pGLL_array = obs.p_tll_array[p_tll_index:]

        return pGLL_array, y_GBL_array

    def _check_gate(
        self, p_ref_LL: np.array, pGLL_array: np.ndarray, y_GBL_array: np.ndarray
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None, float | None, float | None]:
        """Checks if a trajectory point is colliding with any gate frame.

        Args:
            p_ref_LL:           Trajectory point to check.
            pGLL_array:         Center positions of all gates.
            y_GBL_array:        Yaws of all the gates.

        Returns:
            is_inside_frame:    True if the point is colliding with a gate's solid frame structure.
            centre:             Centre of the gate that's been violated.
            local:              Distance from the point to the centre of the violated gate.
            yaw:                Yaw of the violated gate.
            half_outer:         Outer distance of the gate frame.

        """
        # Half-dimensions including clearance margins
        half_outer = FRAME_WIDTH / 2
        half_open = FRAME_OPENING / 2
        half_thick = FRAME_THICK
        half_post_width = POST_WIDTH / 2 + CLEARANCE

        # Difference trajectory point to all gate centers
        diff = p_ref_LL - pGLL_array

        # Rotate differences into each individual gate's local frame using its yaw
        lx = diff[:, 0] * np.cos(y_GBL_array) + diff[:, 1] * np.sin(y_GBL_array)
        ly = -diff[:, 0] * np.sin(y_GBL_array) + diff[:, 1] * np.cos(y_GBL_array)
        lz = diff[:, 2]

        # Calculate bounding box overlaps for all gates simultaneously
        in_depth = np.abs(lx) < half_thick
        in_outer = (np.abs(ly) < half_outer) & (np.abs(lz) < half_outer)
        in_open = (np.abs(ly) < half_open) & (np.abs(lz) < half_open)

        # Gate Post
        # point is in the post zone when it sits BELOW the opening (local frame, consistent units)
        lz_post = lz < -half_open  # not: diff[:,2] < post_height
        in_post = np.abs(ly) < half_post_width
        post_collisions = lz_post & in_post & in_depth

        # Collision happens if it is inside gate frame but not the airhole
        gate_collisions = in_depth & in_outer & ~in_open

        is_inside_frame = bool(np.any(gate_collisions))
        post_collisions = lz_post & in_post & in_depth
        is_post_collisions = bool(np.any(post_collisions))

        if not is_inside_frame | is_post_collisions:
            return False, None, None, None, None

        if is_post_collisions:
            lz = np.abs(lz * 10)

        # print(is_post_collisions, is_inside_frame)

        # if is_inside_frame:
        # print(in_depth, in_outer, ~in_open)
        # breakpoint()
        g = int(np.argmax(gate_collisions))
        centre = pGLL_array[g]
        local = np.array([lx[g], ly[g], lz[g]])
        yaw = y_GBL_array[g]
        return True, centre, local, yaw, half_outer

    def _check_obsticle(
        self, p_ref_LL: np.array, p_oll_array: np.ndarray
    ) -> tuple[bool, np.ndarray | None, float | None]:
        """Checks if a trajectory point is inside an obsticle.

        Args:
            p_ref_LL:       Trajectory point to be checked.
            p_oll_array:     Centre point of obsticles.

        Returns:
            is_inside_obsticle:     Boolian value if point is inside true if its outside false.
            d_O:                    Distance the point needs to clear the obsticle.
        """
        # Hardcoded physical pillar radius + safety margin (match to your env configuration)
        r_obstacle = 0.15 + CLEARANCE

        # consider only x and y coordinates because it is a pillar
        p_ref_xy = p_ref_LL[0:2]
        pOLL_xy = p_oll_array[:, 0:2]

        # xy diff to all obsticles
        diff_xy = p_ref_xy - pOLL_xy

        # xy norm distance to the obsticle
        distances_xy = np.linalg.norm(diff_xy, axis=1)

        # A collision occurs if the distance is less than the obstacle radius threshold
        obstacle_collisions = distances_xy < r_obstacle
        is_inside_obstacle = bool(np.any(obstacle_collisions))

        if not is_inside_obstacle:
            return False, None, None

        # Return obsticle centre of violation and push for the detour point
        o = int(np.argmin(distances_xy))
        return True, p_oll_array[o, :2], r_obstacle

    def get_pos_traj(self) -> np.ndarray:
        """Return the planned position samples (call plan() first)."""
        return self.trajectory.positions

    def setpoint_at(self, t: float, lookahead_t: float = 0.15) -> np.ndarray:
        """Setpoint for sim."""
        ts = self.trajectory.timestamps
        tq = min(t + lookahead_t, ts[-1])
        return np.array([np.interp(tq, ts, self.trajectory.positions[:, k]) for k in range(3)])
