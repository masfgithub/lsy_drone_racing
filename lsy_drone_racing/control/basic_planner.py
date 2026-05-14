"""<For RUFF: Brief description of what this module does>."""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
from scipy.interpolate import CubicSpline


class BasicPlanner:
    """MPC using the collective thrust and attitude interface."""

    def __init__(self, config: dict, t_total: int):
        """TBD: for Ruff.

        Args:
            config: TBD for rust.
            t_total: TBD for ruff.
            TBD: for Ruff.

        Returns:
            TBD: for Ruff.
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
        """TBD: do nothing."""
        return self.get_trajectories()

    def plan(self) -> dict:
        """TBD: for Ruff.

        Args:
            TBD: for Ruff.

        Returns:
            TBD: for Ruff.
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
        """TBD: for Ruff.

        Args:
            TBD: for Ruff.

        Returns:
            TBD: for Ruff.
        """
        planner_dict = {
            "des_pos_spline": self._des_pos_spline,
            "des_vel_spline": self._des_vel_spline,
            "waypoints_pos": self._waypoints_pos,
            "waypoints_vel": self._waypoints_vel
        }
        
        return planner_dict

    def get_pos_traj(self) -> np.ndarray:
        """TBD: for Ruff.

        Args:
            TBD: for Ruff.

        Returns:
            TBD: for Ruff.
        """
        return self._des_pos_spline(np.linspace(0, self._t_total, 100))