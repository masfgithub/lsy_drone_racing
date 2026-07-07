"""Planner interface: abstract base class for all trajectory planners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from lsy_drone_racing.control.env_obs import EnvState

__all__ = ["Trajectory", "Planner"]
FRAME_WIDTH = 0.72
FRAME_OPENING = 0.2
FRAME_THICK = 0.4
R_OBSTACLE = 0.15
DRONE_RADIUS = 0.05


@dataclass
class Trajectory:
    """Class to represent trajectory from planner with positions, velocities and timestamps."""

    positions: np.ndarray
    velocities: np.ndarray
    timestamps: np.ndarray


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

    def get_pos_traj(self) -> np.ndarray:
        """Return the planned position samples (call plan() first)."""
        return self.trajectory.positions

    def setpoint_at(self, t: float, lookahead_t: float = 0.15) -> np.ndarray:
        """Setpoint for sim."""
        ts = self.trajectory.timestamps
        tq = min(t + lookahead_t, ts[-1])
        return np.array([np.interp(tq, ts, self.trajectory.positions[:, k]) for k in range(3)])

    def _check_obsticle(self, p_ref_LL: np.array, p_oll_array: np.ndarray) -> tuple[bool, np.array]:
        """Checks if a trajectory point is inside an obsticle.

        Args:
            p_ref_LL:       Trajectory point to be checked.
            p_oll_array:     Obstacle center positions.

        Returns:
            is_inside_obsticle:     Boolian value if point is inside true if its outside
            obsticle:               Centre point of obsticle that is violated.
        """
        if len(p_oll_array) == 0:
            return False, None
        r_obsticle = R_OBSTACLE

        distance = np.linalg.norm(p_ref_LL[0:2] - p_oll_array[:, 0:2], axis=1)

        is_inside_obsticle = np.any(distance < r_obsticle)
        obsticle = p_oll_array[np.argmin(distance)]

        return is_inside_obsticle, obsticle

    def _get_obsticle_push(
        self, p_ref_LL: np.array, obsticle: np.array, push_vector: np.array
    ) -> float:
        """Get the push vector to avoid the obsticle.

        Args:
            p_ref_LL:       Trajectory point from which to push away from the obsticle.
            obsticle:       Centre point of obsticle that is violated.
            push_vector:    Push vector to avoid the obsticle.

        Returns:
            push:           Length of push vector to avoid the obsticle.
        """
        r_obsticle = R_OBSTACLE
        push_steps = 0.01
        i = 0
        p_ref_LL = p_ref_LL.copy()
        while np.linalg.norm(p_ref_LL[:2] - obsticle[:2]) < r_obsticle:
            p_ref_LL[0:2] += push_steps * push_vector[0:2]
            i += 1

        push = i * push_steps + 0.03
        return push

    def _check_gate(
        self, p_ref_LL: np.array, pGLL_array: np.ndarray, y_GBL_array: np.ndarray
    ) -> tuple[bool, np.array, float]:
        """Checks if a trajectory point is colliding with any gate frame.

        Args:
            p_ref_LL:           Trajectory point to check.
            pGLL_array:         Center positions of all gates.
            y_GBL_array:        Yaws of all the gates.

        Returns:
            is_inside_frame:    True if the point is colliding with a gate's solid frame
                                structure.
            centre:             Centre of the gate that's been violated.
            yaw:                Yaw of the violated gate.
        """
        # Transform ref point into each gate's local frame
        diff = p_ref_LL - pGLL_array
        x = diff[:, 0] * np.cos(y_GBL_array) + diff[:, 1] * np.sin(y_GBL_array)
        y = -diff[:, 0] * np.sin(y_GBL_array) + diff[:, 1] * np.cos(y_GBL_array)
        z = diff[:, 2]

        z0 = FRAME_OPENING / 2 - DRONE_RADIUS
        y0 = FRAME_OPENING / 2 - DRONE_RADIUS

        z1 = FRAME_WIDTH / 2 + DRONE_RADIUS
        y1 = FRAME_WIDTH / 2 + DRONE_RADIUS

        xmax = FRAME_THICK / 2
        m = xmax / (z1 - z0)

        check_z_x_axis = (
            (abs(z) > z0) & (abs(z) < z1) & (abs(x) < m * (abs(z) - z0)) & (abs(y) < y1)
        )
        check_y_x_axis = (
            (abs(y) > y0) & (abs(y) < y1) & (abs(x) < m * (abs(y) - y0)) & (abs(z) < z1)
        )

        is_inside_frame = np.any(check_z_x_axis | check_y_x_axis)

        if not is_inside_frame:
            return False, None, None

        g = int(np.argmax(check_z_x_axis | check_y_x_axis))
        centre = pGLL_array[g]
        yaw = y_GBL_array[g]
        return True, centre, yaw

    def _get_gate_push(
        self, p_ref_LL: np.array, centre: np.array, yaw: float, push_vector: np.array
    ) -> np.array:
        """Get the push length to avoid the gate frame.

        Args:
            p_ref_LL:       Trajectory point from which to push away from the gate frame.
            centre:         Centre of the gate that's been violated.
            yaw:            Yaw of the violated gate.
            push_vector:    Push vector to avoid the gate frame.

        Returns:
            p_ref_LL:       New trajectory point after pushing away from the gate frame.
        """
        inside_gate = True
        push_steps = 0.01
        i = 0
        p_ref_LL = p_ref_LL.copy()

        while inside_gate:
            p_ref_LL += push_steps * push_vector
            inside_gate, _, _ = self._check_gate(p_ref_LL, np.array([centre]), np.array([yaw]))
            i += 1
        push = i * push_steps + 0.1
        return push
