"""Gate-aware planner for the MPCC++ controller.

The planner owns a hand-tuned racing line (an ordered waypoint list that already
threads every gate). That line is fitted to a cubic spline and consumed by the
MPCC++ controller as the tunnel CENTERLINE.

Online updates
--------------
Gate poses are observed with uncertainty and refined (or moved) at runtime.
``replan(obs)`` follows those updates *without throwing away the hand-tuned
shape*: each gate's displacement from its nominal (initial) position is applied
to the racing line as a smooth, gate-localized WARP. Near gate ``i`` the line
shifts by that gate's displacement; between gates the shifts blend; far from any
moved gate the line is unchanged. The warped line therefore still passes through
the gates at their current poses while preserving the authored racing line.

The rebuild is skipped (cached dict returned) when no gate has moved more than
``gate_move_tol``, so steady-state steps cost nothing.

Planner dict keys:
    "des_pos_spline"  - scipy CubicSpline over time (position), possibly warped
    "des_vel_spline"  - its first derivative (velocity)
    "waypoints_pos"   - densely sampled position array  (n_ticks, 3)
    "waypoints_vel"   - densely sampled velocity array  (n_ticks, 3)
    "gate_positions"  - current gate centre array       (n_gates, 3)
    "approach_waypoints" - the (warped) racing-line waypoints
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from lsy_drone_racing.control.env_obs import EnvState_t


# Hand-tuned racing line: ordered waypoints that already thread every gate.
_RACING_LINE = np.array(
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
        [1.5, -0.75, 1.2],
    ]
)


class BasicPlannerMPCCpp:
    """Planner for MPCC++: hand-tuned racing line, warped to follow gate updates."""

    def __init__(
        self,
        obs: "EnvState_t",
        config: dict,
        t_total: int,
        warp_sigma: float = 0.8,
        gate_move_tol: float = 0.02,
    ):
        """Initialise the planner.

        Args:
            obs:           Initial environment observation (defines nominal gates).
            config:        Race configuration (config.env.freq for dense sampling).
            t_total:       Total trajectory duration in seconds.
            warp_sigma:    Arc-length width (m) over which a gate's displacement is
                           blended into the racing line. Keep below ~1/3 of the gate
                           spacing so a moved gate barely disturbs its neighbours;
                           larger = smoother but bleeds into adjacent gates.
            gate_move_tol: A gate must move more than this (m) to trigger a re-warp.
        """
        self._freq = config.env.freq
        self._t_total = t_total
        self._nominal_gates = obs.pTLL_array.copy()  # warp anchor (initial poses)
        self._start_pos = obs.pBLL.copy()
        self._warp_sigma = float(warp_sigma)
        self._gate_move_tol = float(gate_move_tol)

        self._des_pos_spline: CubicSpline | None = None
        self._des_vel_spline: CubicSpline | None = None
        self._waypoints_pos: np.ndarray | None = None
        self._waypoints_vel: np.ndarray | None = None
        self._waypoints: np.ndarray | None = None  # current (warped) racing line
        self._wp_s: np.ndarray | None = None  # arc-length of each waypoint
        self._gate_s_nom: np.ndarray | None = None  # arc-length of each nominal gate
        self._last_gates: np.ndarray | None = None  # gates used for current spline

    # ----------------------------------------------------------------------

    def plan(self) -> dict:
        """Build the nominal reference spline and return the planner dict."""
        self._waypoints = _RACING_LINE.copy()
        self._build_spline(self._waypoints)

        # Arc-length positions (on the nominal centerline) of every waypoint and
        # every nominal gate -- the common axis the warp blends along.
        t_fine = np.linspace(0.0, self._t_total, 4000)
        pts = self._des_pos_spline(t_fine)
        s_fine = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        self._wp_s = np.array(
            [s_fine[int(np.argmin(np.sum((pts - w) ** 2, axis=1)))] for w in self._waypoints]
        )
        self._gate_s_nom = np.array(
            [s_fine[int(np.argmin(np.sum((pts - g) ** 2, axis=1)))] for g in self._nominal_gates]
        )
        self._last_gates = self._nominal_gates.copy()
        return self._get_dict()

    def replan(self, obs: "EnvState_t | None" = None) -> dict:
        """Re-fit the racing line to the current gate poses (warp), if they moved.

        Args:
            obs: Current observation. If None, the cached dict is returned (no-op).

        Returns:
            The planner dict (with a possibly re-warped des_pos_spline).
        """
        if obs is None or self._waypoints is None:
            return self._get_dict()

        gates = np.asarray(obs.pTLL_array, dtype=float)
        if (
            self._last_gates is not None
            and gates.shape == self._last_gates.shape
            and np.max(np.abs(gates - self._last_gates)) < self._gate_move_tol
        ):
            return self._get_dict()  # unchanged -> cached spline

        self._build_spline(self._warp_waypoints(gates))
        self._last_gates = gates.copy()
        return self._get_dict()

    # ----------------------------------------------------------------------

    def _warp_waypoints(self, gates: np.ndarray) -> np.ndarray:
        """Warp the hand-tuned line by per-gate displacement from nominal.

        waypoint_j += sum_i  w_ij * (gate_i - nominal_gate_i),
        w_ij = exp(-0.5 * ((wp_s_j - gate_s_nom_i) / warp_sigma)^2).
        """
        m = min(len(gates), len(self._nominal_gates))
        delta = gates[:m] - self._nominal_gates[:m]  # (m, 3)
        warped = _RACING_LINE.copy()
        for j, s_w in enumerate(self._wp_s):
            w = np.exp(-0.5 * ((s_w - self._gate_s_nom[:m]) / self._warp_sigma) ** 2)
            warped[j] += (w[:, None] * delta).sum(axis=0)
        return warped

    def _build_spline(self, waypoints: np.ndarray) -> None:
        """Fit the time-parameterized position/velocity splines + dense samples."""
        segs = np.maximum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1), 1e-3)
        cum = np.concatenate([[0.0], np.cumsum(segs)])
        t = cum / cum[-1] * self._t_total

        self._waypoints = np.asarray(waypoints, dtype=float)
        self._des_pos_spline = CubicSpline(t, self._waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()

        t_dense = np.linspace(0.0, self._t_total, int(self._freq * self._t_total))
        self._waypoints_pos = self._des_pos_spline(t_dense)
        self._waypoints_vel = self._des_vel_spline(t_dense)

    def _get_dict(self) -> dict:
        if self._des_pos_spline is None:
            raise RuntimeError("Call plan() before accessing the planner dict.")
        return {
            "des_pos_spline": self._des_pos_spline,
            "des_vel_spline": self._des_vel_spline,
            "waypoints_pos": self._waypoints_pos,
            "waypoints_vel": self._waypoints_vel,
            "gate_positions": self._last_gates,
            "approach_waypoints": self._waypoints,
        }

    def get_pos_traj(self, n: int = 200) -> np.ndarray:
        """Return n evenly-spaced positions along the (current) reference spline."""
        if self._des_pos_spline is None:
            raise RuntimeError("Call plan() before get_pos_traj().")
        return self._des_pos_spline(np.linspace(0.0, self._t_total, n))
