"""Gate-aware planner for the MPCC++ controller.

Unlike BasicPlanner (which uses hardcoded waypoints), this planner derives its
path directly from the gate positions in the current observation so that the
visualised reference trajectory always matches the actual race track.

The planner builds a cubic spline from the drone's initial position through all
gate centres in order, using chord-length arc-length for timing. The resulting
spline serves purely for visualisation and warm-start; the MPCC++ controller
builds its own TunnelReferencePath from gate geometry and does NOT consume the
spline for contouring.

Planner dict keys (compatible with BasicPlanner):
    "des_pos_spline"  – scipy CubicSpline over time (position)
    "des_vel_spline"  – its first derivative (velocity)
    "waypoints_pos"   – densely sampled position array  (n_ticks, 3)
    "waypoints_vel"   – densely sampled velocity array  (n_ticks, 3)
    "gate_positions"  – raw gate centre array           (n_gates, 3)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from lsy_drone_racing.control.env_obs import EnvState_t


class BasicPlannerMPCCpp:
    """Minimal planner for MPCC++: smooth spline through observed gate centres."""

    def __init__(self, obs: EnvState_t, config, t_total: int):
        """Initialise the planner.

        Args:
            obs:     Initial environment observation (gate positions are read here).
            config:  Race configuration (config.env.freq used for dense sampling).
            t_total: Total trajectory duration in seconds.
        """
        self._freq    = config.env.freq
        self._t_total = t_total
        self._gate_positions = obs.pTLL_array.copy()   # (n_gates, 3)
        self._start_pos      = obs.pBLL.copy()          # drone initial position

        self._des_pos_spline: CubicSpline | None = None
        self._des_vel_spline: CubicSpline | None = None
        self._waypoints_pos:  np.ndarray | None  = None
        self._waypoints_vel:  np.ndarray | None  = None
        self._waypoints = np.ndarray | None  # for debugging only; not included in the planner dict
    # ──────────────────────────────────────────────────────────────────────────

    def plan(self) -> dict:
        """Build the reference spline and return the planner dict."""
        # Waypoints: drone start → gate centres (in order)
        waypoints = np.vstack([self._start_pos, self._gate_positions])

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

        # Chord-length times: scale so the whole trajectory fits in t_total
        segs = np.linalg.norm(np.diff(self._waypoints, axis=0), axis=1)
        segs = np.maximum(segs, 1e-3)          # avoid zero-length segments
        cum  = np.concatenate([[0.0], np.cumsum(segs)])
        t    = cum / cum[-1] * self._t_total   # normalise to [0, t_total]

        self._des_pos_spline = CubicSpline(t, self._waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()

        t_dense = np.linspace(0.0, self._t_total, int(self._freq * self._t_total))
        self._waypoints_pos = self._des_pos_spline(t_dense)
        self._waypoints_vel = self._des_vel_spline(t_dense)

        return self._get_dict()

    def replan(self) -> dict:
        """No-op replan: MPCC++ tracks gates via theta, not a replanned spline."""
        return self._get_dict()

    # ──────────────────────────────────────────────────────────────────────────

    def _get_dict(self) -> dict:
        if self._des_pos_spline is None:
            raise RuntimeError("Call plan() before accessing the planner dict.")
        return {
            "des_pos_spline":     self._des_pos_spline,
            "des_vel_spline":     self._des_vel_spline,
            "waypoints_pos":      self._waypoints_pos,
            "waypoints_vel":      self._waypoints_vel,
            "gate_positions":     self._gate_positions,
            "approach_waypoints": self._waypoints,  # sparse knots for the tunnel ref path
        }

    def get_pos_traj(self, n: int = 200) -> np.ndarray:
        """Return n evenly-spaced positions along the reference spline.

        Used by render_callback to draw the planned path.
        """
        if self._des_pos_spline is None:
            raise RuntimeError("Call plan() before get_pos_traj().")
        return self._des_pos_spline(np.linspace(0.0, self._t_total, n))
