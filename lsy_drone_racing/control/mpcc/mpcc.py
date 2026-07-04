"""MPCC controller for drone racing, compatible with drone_racing_pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controller_interface import ControllerInterface
from lsy_drone_racing.control.nmpc.env_soft_constraints import (
    get_gate_objects,
    get_obstacle_objects,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.env_obs import EnvState_t


class MPCC(ControllerInterface):
    """Model Predictive Contouring Control using acados.

    State vector (15):  [px, py, pz, vx, vy, vz, roll, pitch, yaw,
                         f_col, f_cmd, r_cmd, p_cmd, y_cmd, theta]
    Input vector  (5):  [df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta]
    Output         :    [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]
    """

    def __init__(
        self,
        obs: EnvState_t,
        planner: dict,
        info: dict,
        config: dict,
        t_total: int,
        N_horizon: int = 40,
        T_horizon: float = 0.7,
        model_arc_step: float = 0.05,
        model_traj_length: float = 15.0,
        q_lag: float = 80.0,
        q_lag_peak: float = 500.0,
        q_contour: float = 120.0,
        q_contour_peak: float = 700.0,
        q_attitude: float = 1.0,
        r_thrust: float = 0.2,
        r_roll: float = 0.3,
        r_pitch: float = 0.3,
        r_yaw: float = 0.5,
        mu_speed: float = 10.0,
        w_speed_gate: float = 9.0,
    ):
        """Initialize the MPCC controller.

        Args:
            obs:               Initial environment observation.
            planner:           Planner dict with 'des_pos_spline' (CubicSpline).
            info:              Initial environment information.
            config:            Race configuration (config.env.freq, config.sim.drone_model).
            t_total:           Total trajectory duration in seconds.
            N_horizon:         MPC prediction horizon (steps).
            T_horizon:         MPC prediction horizon (seconds).
            model_arc_step:    Arc-length discretization for trajectory encoding (m).
            model_traj_length: Total arc-length encoded in the OCP parameters (m).
            q_lag:             Lag-error tracking weight.
            q_lag_peak:        Extra lag-error weight near gates.
            q_contour:         Contouring-error weight.
            q_contour_peak:    Extra contouring weight near gates.
            q_attitude:        Attitude regularisation weight.
            r_thrust:          Thrust-increment smoothness weight.
            r_roll:            Roll-increment smoothness weight.
            r_pitch:           Pitch-increment smoothness weight.
            r_yaw:             Yaw-increment smoothness weight.
            mu_speed:          Progress reward coefficient.
            w_speed_gate:      Speed penalty coefficient near gates.
        """
        super().__init__(obs, planner, info, config, t_total)

        self._N = N_horizon
        self._T = T_horizon
        self._model_arc_step = model_arc_step
        self._model_traj_length = model_traj_length
        self._t_total = t_total
        self._freq = config.env.freq
        self._tick = 0
        self._finished = False

        self._cost_cfg = {
            "q_lag": q_lag,
            "q_lag_peak": q_lag_peak,
            "q_contour": q_contour,
            "q_contour_peak": q_contour_peak,
            "q_attitude": q_attitude,
            "r_thrust": r_thrust,
            "r_roll": r_roll,
            "r_pitch": r_pitch,
            "r_yaw": r_yaw,
            "mu_speed": mu_speed,
            "w_speed_gate": w_speed_gate,
        }

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._mass = float(self.drone_params["mass"])
        self._gravity = -float(self.drone_params["gravity_vec"][-1])
        self._hover_thrust = self._mass * self._gravity

        self._gates_information = {
            "total_length": 0.8,
            "total_height": 0.8,
            "hole_width": 0.23,
            "hole_height": 0.23,
            "thickness": 0.35,
            "margin": 0.05,
        }
        self._obstacles_information = {"d_min": 0.15, "total_height": 2.0}

        # Gate/obstacle objects are updated each control step for render_callback
        gates_quat_wxyz = np.roll(obs.qTLT_array, 1, axis=-1)
        self._gates = get_gate_objects(obs.pTLL_array, gates_quat_wxyz, self._gates_information)
        self._obstacles = get_obstacle_objects(obs.pOLL_array, self._obstacles_information)

        # Build arc-length spline from the planner's time-parameterised spline.
        # _model_arc_step and _model_traj_length must be set before this call.
        self.set_ref_traj(planner)

        # Build the acados OCP solver (code-generates C code on first call).
        from lsy_drone_racing.control.mpcc.mpcc_setup import create_ocp_solver_mpcc

        self._acados_ocp_solver, self._ocp, self._n_samples = create_ocp_solver_mpcc(
            N=self._N,
            Tf=self._T,
            parameters=self.drone_params,
            model_arc_step=model_arc_step,
            model_traj_length=model_traj_length,
            cost_cfg=self._cost_cfg,
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()

        # Encode the trajectory into solver parameters and set them for every stage.
        param_vec = self._encode_trajectory_params(obs.pTLL_array)
        for k in range(self._N + 1):
            self._acados_ocp_solver.set(k, "p", param_vec)

        # Internal control state (unknown from obs; bootstrapped from hover)
        self.last_theta = 0.0
        self.last_f_col = self._hover_thrust
        self.last_f_cmd = self._hover_thrust
        self.last_rpy_cmd = np.zeros(3)
        self._infeas_counter = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Trajectory helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_arc_spline(self, pos_spline: CubicSpline) -> tuple[CubicSpline, float]:
        """Convert a time-parameterised CubicSpline to arc-length parameterisation.

        The spline is extended beyond the actual trajectory by _model_traj_length
        using a straight-line extrapolation so that 'theta' can always look ahead.

        Returns:
            (arc_spline, traj_arc_length) where traj_arc_length is the arc length
            of the original trajectory (before the extension).
        """
        t_end = float(pos_spline.x[-1])
        n_fine = 10_000
        t_fine = np.linspace(0.0, t_end, n_fine)
        pts = pos_spline(t_fine)

        diffs = np.diff(pts, axis=0)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])

        # CubicSpline requires strictly increasing x — deduplicate arc values.
        _, idx = np.unique(arc, return_index=True)
        arc_u, pts_u = arc[idx], pts[idx]
        traj_arc_length = float(arc_u[-1])

        # Extend with a straight line so theta can advance to model_traj_length.
        if traj_arc_length < self._model_traj_length:
            arc_spline_tmp = CubicSpline(arc_u, pts_u)
            tangent = arc_spline_tmp.derivative()(traj_arc_length)
            t_unit = tangent / (np.linalg.norm(tangent) + 1e-9)
            n_ext = 300
            s_ext = np.linspace(traj_arc_length, traj_arc_length + self._model_traj_length, n_ext)
            pos_ext = pts_u[-1] + (s_ext - traj_arc_length)[:, None] * t_unit
            arc_spline = CubicSpline(
                np.concatenate([arc_u, s_ext[1:]]), np.concatenate([pts_u, pos_ext[1:]])
            )
        else:
            arc_spline = CubicSpline(arc_u, pts_u)

        return arc_spline, traj_arc_length

    def _encode_trajectory_params(self, gate_positions: np.ndarray | None = None) -> np.ndarray:
        """Encode the arc-length trajectory as the OCP parameter vector.

        Parameter layout: [pd_flat (3*n), tp_flat (3*n), qc (n)]
        where n = n_samples = int(model_traj_length / model_arc_step).

        Args:
            gate_positions: Array of shape (n_gates, 3) for dynamic weight computation.

        Returns:
            1-D parameter vector of length 7 * n_samples.
        """
        theta_s = np.arange(0.0, self._model_traj_length, self._model_arc_step)[: self._n_samples]
        s_clip = np.clip(theta_s, 0.0, float(self._arc_spline.x[-1]))

        pd_vals = self._arc_spline(s_clip)  # (n, 3) positions
        tp_vals = self._arc_spline.derivative()(s_clip)  # (n, 3) ≈ unit tangents

        qc = np.zeros(self._n_samples)
        if gate_positions is not None:
            for gp in gate_positions:
                d = np.linalg.norm(pd_vals - gp, axis=-1)
                qc = np.maximum(qc, 0.4 * np.exp(-8.0 * d**2))

        return np.concatenate([pd_vals.flatten(), tp_vals.flatten(), qc])

    # ──────────────────────────────────────────────────────────────────────────
    # ControllerInterface implementation
    # ──────────────────────────────────────────────────────────────────────────

    def control(self, obs: EnvState_t, info: dict | None = None) -> NDArray[np.floating]:
        """Compute attitude + collective-thrust command via MPCC.

        Returns:
            np.ndarray of shape (4,): [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd].
        """
        # Update gate/obstacle objects each step for render_callback visualisation.
        gates_quat_wxyz = np.roll(obs.qTLT_array, 1, axis=-1)
        self._gates = get_gate_objects(obs.pTLL_array, gates_quat_wxyz, self._gates_information)
        self._obstacles = get_obstacle_objects(obs.pOLL_array, self._obstacles_information)

        # Convert quaternion to Euler RPY.
        rpy = R.from_quat(obs.qBLB).as_euler("xyz")

        # Build full state vector from observable state + internal controller state.
        x0 = np.concatenate(
            [
                obs.pBLL,
                obs.vBLL,
                rpy,
                [self.last_f_col, self.last_f_cmd],
                self.last_rpy_cmd,
                [self.last_theta],
            ]
        )

        # Shift warm start by one step; duplicate last entry.
        if not hasattr(self, "_x_warm"):
            self._x_warm = [x0.copy() for _ in range(self._N + 1)]
            self._u_warm = [np.zeros(self._nu) for _ in range(self._N)]
        else:
            self._x_warm = self._x_warm[1:] + [self._x_warm[-1].copy()]
            self._u_warm = self._u_warm[1:] + [self._u_warm[-1].copy()]

        for i in range(self._N):
            self._acados_ocp_solver.set(i, "x", self._x_warm[i])
            self._acados_ocp_solver.set(i, "u", self._u_warm[i])
        self._acados_ocp_solver.set(self._N, "x", self._x_warm[self._N])

        # Pin the initial state.
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        if self.last_theta >= self._traj_arc_length:
            self._finished = True

        # Solve.
        status = self._acados_ocp_solver.solve()

        if status not in (0, 1, 3):
            self._infeas_counter = min(self._infeas_counter + 1, self._N - 1)
            print(
                f"[MPCC] Solver status {status} — open-loop fallback step {self._infeas_counter}."
            )
            return np.array(
                [self.last_rpy_cmd[0], self.last_rpy_cmd[1], self.last_rpy_cmd[2], self.last_f_cmd],
                dtype=np.float32,
            )

        self._infeas_counter = 0
        self._x_warm = [self._acados_ocp_solver.get(i, "x") for i in range(self._N + 1)]
        self._u_warm = [self._acados_ocp_solver.get(i, "u") for i in range(self._N)]

        x_next = self._acados_ocp_solver.get(1, "x")
        self.last_f_col = float(x_next[9])
        self.last_f_cmd = float(x_next[10])
        self.last_rpy_cmd = np.array(x_next[11:14])
        self.last_theta = float(x_next[14])

        return np.array(
            [self.last_rpy_cmd[0], self.last_rpy_cmd[1], self.last_rpy_cmd[2], self.last_f_cmd],
            dtype=np.float32,
        )

    def set_ref_traj(self, planner_traj: dict) -> None:
        """Compute arc-length parameterisation from the planner's time-based spline.

        Args:
            planner_traj: Dict with key 'des_pos_spline' (scipy CubicSpline over time).
        """
        pos_spline: CubicSpline = planner_traj["des_pos_spline"]
        self._arc_spline, self._traj_arc_length = self._compute_arc_spline(pos_spline)

    def reset(self) -> None:
        """Reset internal control state for a new episode."""
        self._tick = 0
        self._finished = False
        self.last_theta = 0.0
        self.last_f_col = self._hover_thrust
        self.last_f_cmd = self._hover_thrust
        self.last_rpy_cmd = np.zeros(3)
        self._infeas_counter = 0
        for attr in ("_x_warm", "_u_warm"):
            if hasattr(self, attr):
                delattr(self, attr)

    def set_tick(self, tick: int) -> None:
        """Set tick counter (kept for interface compatibility; MPCC uses theta)."""
        self._tick = tick

    def get_states(self) -> None:
        """Return internal states (unused)."""
        return None

    def get_predicted_traj(self) -> np.ndarray:
        """Return predicted (x, y, z) positions over the MPC horizon."""
        if not hasattr(self, "_x_warm"):
            return np.zeros((self._N + 1, 3))
        return np.array([x[:3] for x in self._x_warm])

    def get_ref_traj(self) -> np.ndarray:
        """Return reference (x, y, z) positions along the arc-spline for the horizon."""
        s_end = min(self.last_theta + 3.0, float(self._arc_spline.x[-1]))
        s_max = float(self._arc_spline.x[-1])
        s_q = np.clip(np.linspace(self.last_theta, s_end, self._N + 1), 0.0, s_max)
        return self._arc_spline(s_q)
