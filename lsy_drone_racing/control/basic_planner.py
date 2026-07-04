"""Fixed-waypoint cubic-spline planner used as the reference for the MPC/NMPC controllers."""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
from scipy.interpolate import CubicSpline


class BasicPlanner:
    """Plans a time-parameterized cubic-spline trajectory through a hand-tuned waypoint set."""

    def __init__(self, config: dict, t_total: int):
        """Store the race configuration and precompute the time grid for the waypoints.

        Args:
            config: The race configuration; only `config.env.freq` is used, to determine
                how densely the spline is sampled.
            t_total: Total assumed trajectory duration in seconds, spread evenly across
                the waypoints to build the spline's time knots.
        """
        # Same waypoints as in the trajectory controller. Determined by trial and error.
        self._waypoints = np.array(
            [
                [-1.5, 0.75, 0.05],
                [-1.0, 0.55, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [-0.0, -0.7, 1.2],
                [0.5, -0.75, 1.2],
            ]
        )
        self._t_total = t_total  # s
        self._freq = config.env.freq
        self._t = np.linspace(0, self._t_total, len(self._waypoints))

    def replan(self) -> dict:
        """No-op: the waypoints are fixed, so replanning just returns the existing trajectory."""
        return self.get_trajectories()

    def plan(self) -> dict:
        """Fit the position/velocity splines through the waypoints and densely sample them.

        Returns:
            The planner dict from `get_trajectories()` (splines + dense position/velocity
            samples).
        """
        self._des_pos_spline = CubicSpline(self._t, self._waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()
        self._waypoints_pos = self._des_pos_spline(
            np.linspace(0, self._t_total, int(self._freq * self._t_total))
        )
        self._waypoints_vel = self._des_vel_spline(
            np.linspace(0, self._t_total, int(self._freq * self._t_total))
        )
        self._waypoints_yaw = self._waypoints_pos[:, 0] * 0
        self._finished = False

        return self.get_trajectories()

    def get_trajectories(self) -> dict:
        """Bundle the fitted splines and their dense samples into the planner output dict.

        Returns:
            Dict with keys "des_pos_spline", "des_vel_spline", "waypoints_pos",
            "waypoints_vel", consumed by the MPC/NMPC controllers.
        """
        planner_dict = {
            "des_pos_spline": self._des_pos_spline,
            "des_vel_spline": self._des_vel_spline,
            "waypoints_pos": self._waypoints_pos,
            "waypoints_vel": self._waypoints_vel,
        }

        return planner_dict

    def get_pos_traj(self) -> np.ndarray:
        """Return 100 position samples of the planned spline, evenly spaced over its duration."""
        return self._des_pos_spline(np.linspace(0, self._t_total, 100))
