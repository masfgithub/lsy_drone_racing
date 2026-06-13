"""MPCC++ controller for drone racing.

Extends the production MPCC with:
  - Gate-aligned prismatic tunnel constraints (4 halfspaces per horizon node)
  - Soft obstacle (cylinder) avoidance constraints

The tunnel is built from the gate positions/quaternions supplied by the
environment. At every gate the tunnel cross-section matches the gate opening;
between gates it widens to a generous nominal size so the drone is essentially
unconstrained. Staying inside the tunnel therefore guarantees gate passage.

Obstacle positions are updated online at every control step so the OCP always
sees the latest perception data.

State (15): [px, py, pz, vx, vy, vz, roll, pitch, yaw,
              f_col, f_cmd, r_cmd, p_cmd, y_cmd, theta]
Input  (5): [df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta]
Output (4): [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controller_interface import ControllerInterface
from lsy_drone_racing.control.mpcc.mpccpp_setup import (
    OBST_DIM,
    _BNM,
    _HIDX,
    _MU,
    _NRM,
    _OBST_START,
    _PD,
    _PDD,
    _QC,
    _TD,
    _THETA_BAR,
    _WIDX,
    num_params,
)
from lsy_drone_racing.control.mpcc_test.mpcc_reference import ReferencePath
from lsy_drone_racing.control.mpcc_test.mpccpp_reference import TunnelReferencePath, _gate_axes
from lsy_drone_racing.control.nmpc.env_soft_constraints import (
    get_gate_objects,
    get_obstacle_objects,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.env_obs import EnvState_t


def _gate_normals_from_quats(quats_wxyz: np.ndarray) -> np.ndarray:
    """Derive gate through-normals (x-axis of gate frame) from wxyz quaternions."""
    quats_xyzw = np.roll(quats_wxyz, -1, axis=-1)
    return R.from_quat(quats_xyzw).apply([1.0, 0.0, 0.0])


def _build_extended_tunnel_ref(
    approach_wps: np.ndarray,
    gate_positions: np.ndarray,
    gate_normals: np.ndarray,
    gate_w_half: float,
    gate_h_half: float,
    W_nom: float,
    H_nom: float,
    tunnel_sigma: float,
    gate_tangent_len: float = 0.5,
    qc_nom: float = 1.0,
    qc_gate: float = 120.0,
    gate_sigma: float = 0.8,
    frame_up: tuple = (0.0, 0.0, 1.0),
) -> TunnelReferencePath:
    """Build a TunnelReferencePath whose spline begins at the approach waypoints.

    Approach waypoints define the pre-gate portion of the path but are NOT
    added to gate_indices, so they produce no qc spike and no tunnel pinch.
    Only real gate centers trigger the qc bump and tunnel narrowing.

    Args:
        approach_wps:     (K, 3) waypoints from drone start to first gate approach.
        gate_positions:   (M, 3) gate center positions.
        gate_normals:     (M, 3) gate through-normals (x-axis of gate frame).
        gate_w_half:      Gate half-width (m).
        gate_h_half:      Gate half-height (m).
        W_nom:            Nominal tunnel half-width between gates (m).
        H_nom:            Nominal tunnel half-height between gates (m).
        tunnel_sigma:     Gaussian sigma for tunnel pinch at each gate (arc-length, m).
        gate_tangent_len: Distance of the pre/post-gate helper points from gate center.
        qc_nom:           Baseline contouring weight multiplier (outside gates).
        qc_gate:          Peak contouring weight multiplier (at gate centers).
        gate_sigma:       Gaussian sigma for the qc bump at each gate (arc-length, m).
        frame_up:         World up-vector used to compute gate lateral/vertical axes.

    Returns:
        A TunnelReferencePath instance whose spline starts at approach_wps[0].
    """
    up = np.asarray(frame_up, dtype=float)
    delta = float(gate_tangent_len)
    M = len(gate_positions)

    # Augmented knot list:  approach points (no gate tag) + gate triples (tagged)
    aug: list[np.ndarray] = list(np.asarray(approach_wps, dtype=float))
    gidx: list[int] = []
    for i in range(M):
        c, n = gate_positions[i], gate_normals[i]
        aug.append(c - delta * n)   # pre-gate helper (not a gate)
        aug.append(c)               # gate center → gidx
        gidx.append(len(aug) - 1)
        aug.append(c + delta * n)   # post-gate helper (not a gate)
    aug_arr = np.array(aug)

    gate_hw = np.broadcast_to(gate_w_half, (M,)).astype(float).copy()
    gate_hh = np.broadcast_to(gate_h_half, (M,)).astype(float).copy()
    gw = np.zeros((M, 3))
    gh = np.zeros((M, 3))
    for i in range(M):
        gw[i], gh[i] = _gate_axes(gate_normals[i], up)

    # Bypass TunnelReferencePath.__init__ so we can pass a hand-built gidx
    # that excludes the approach waypoints from the gate set.
    ref = object.__new__(TunnelReferencePath)
    ReferencePath.__init__(
        ref, aug_arr, closed=False, gate_indices=gidx,
        qc_nom=qc_nom, qc_gate=qc_gate, gate_sigma=gate_sigma,
    )
    ref.gate_centers = np.asarray(gate_positions, dtype=float)
    ref.gate_n  = gate_normals
    ref.gate_w  = gw
    ref.gate_h  = gh
    ref.gate_hw = gate_hw
    ref.gate_hh = gate_hh
    ref.W_nom        = float(W_nom)
    ref.H_nom        = float(W_nom if H_nom is None else H_nom)
    ref.tunnel_sigma = float(tunnel_sigma)
    ref._up = up
    return ref


class MPCCpp(ControllerInterface):
    """MPCC++ controller: MPCC contouring + gate tunnel + obstacle avoidance.

    Identical interface to NMPC / MPCC; drop-in replacement in drone_racing_pipeline.
    """

    def __init__(
        self,
        obs: EnvState_t,
        planner: dict,
        info: dict,
        config: dict,
        t_total: int,
        N_horizon: int = 20,
        T_horizon: float = 0.5,
        mu: float = 55.0,
        q_lag: float = 80.0,
        q_lag_peak: float = 500.0,
        q_contour: float = 120.0,
        q_contour_peak: float = 700.0,
        q_attitude: float = 1.0,
        r_thrust: float = 0.2,
        r_roll: float = 0.3,
        r_pitch: float = 0.3,
        r_yaw: float = 0.5,
        w_speed_gate: float = 9.0,
        W_nom: float = 0.3,
        H_nom: float = 0.3,
        tunnel_sigma: float = 1.0,
        tunnel_soft: bool = True,
        tunnel_slack_lin: float = 1e3,
        tunnel_slack_quad: float = 1e3,
        obstacle_soft: bool = True,
        obstacle_slack_lin: float = 1e4,
        obstacle_slack_quad: float = 1e4,
        n_obstacles: int | None = None,
    ):
        """Initialize the MPCC++ controller.

        Args:
            obs:                Initial environment observation.
            planner:            Planner dict (not used; reference built from gates).
            info:               Initial environment information.
            config:             Race configuration (config.env.freq, config.sim.drone_model).
            t_total:            Total trajectory duration in seconds.
            N_horizon:          MPC prediction horizon (steps).
            T_horizon:          Horizon duration (seconds).
            mu:                 Progress incentive weight (larger = faster).
            q_lag:              Lag-error tracking weight.
            q_lag_peak:         Extra lag-error weight near gates.
            q_contour:          Contouring-error weight.
            q_contour_peak:     Extra contouring weight near gates.
            q_attitude:         Attitude regularisation weight.
            r_thrust:           Thrust-increment smoothness weight.
            r_roll:             Roll-increment smoothness weight.
            r_pitch:            Pitch-increment smoothness weight.
            r_yaw:              Yaw-increment smoothness weight.
            w_speed_gate:       Speed penalty coefficient near gates.
            W_nom:              Nominal tunnel half-width between gates (m).
            H_nom:              Nominal tunnel half-height between gates (m).
            tunnel_sigma:       Gaussian sigma for tunnel pinch at gates (m arc-length).
            tunnel_soft:        If True, soften tunnel constraints via slacks.
            tunnel_slack_lin:   Linear slack penalty for tunnel.
            tunnel_slack_quad:  Quadratic slack penalty for tunnel.
            obstacle_soft:      If True, soften obstacle constraints via slacks.
            obstacle_slack_lin: Linear slack penalty for obstacles.
            obstacle_slack_quad:Quadratic slack penalty for obstacles.
            n_obstacles:        OCP obstacle slots. Defaults to len(obs.pOLL_array).
        """
        super().__init__(obs, planner, info, config, t_total)

        self._N  = N_horizon
        self._Tf = T_horizon
        self._dt = T_horizon / N_horizon
        self._mu = float(mu)
        self._finished = False
        self._tick = 0

        self._gates_information = {
            "total_length": 0.8, "total_height": 0.8,
            "hole_width": 0.23,  "hole_height": 0.23,
            "thickness": 0.35,   "margin": 0.05,
        }
        self._obstacles_information = {"d_min": 0.15, "total_height": 2.0}

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._mass    = float(self.drone_params["mass"])
        self._gravity = -float(self.drone_params["gravity_vec"][-1])

        # Gate / obstacle objects for rendering (updated each control step)
        gates_quat_wxyz = np.roll(obs.qTLT_array, 1, axis=-1)
        self._gates     = get_gate_objects(obs.pTLL_array, gates_quat_wxyz, self._gates_information)
        self._obstacles = get_obstacle_objects(obs.pOLL_array, self._obstacles_information)

        n_obs = len(obs.pOLL_array) if n_obstacles is None else n_obstacles
        self._n_obstacles = n_obs
        self._npar = num_params(n_obs)

        # Obstacle parameter slots [xo, yo, ro] (updated online each step)
        self._obst_params = np.zeros((n_obs, OBST_DIM))
        self._update_obst_params(obs.pOLL_array)

        # Build tunnel reference path.
        # If the planner provides approach waypoints, prepend them so the path
        # starts near the drone's actual position (not at the first gate helper).
        # Approach waypoints are excluded from gate_indices → no qc spike, no
        # tunnel pinch there; only real gate centers get those.
        self._W_nom = float(W_nom)
        self._H_nom = float(H_nom)
        gate_normals = _gate_normals_from_quats(gates_quat_wxyz)
        approach_wps = planner.get("approach_waypoints", None)
        if approach_wps is not None and len(approach_wps) > 0:
            self._ref = _build_extended_tunnel_ref(
                approach_wps=np.asarray(approach_wps),
                gate_positions=obs.pTLL_array,
                gate_normals=gate_normals,
                gate_w_half=self._gates_information["hole_width"] / 2.0,
                gate_h_half=self._gates_information["hole_height"] / 2.0,
                W_nom=W_nom, H_nom=H_nom, tunnel_sigma=tunnel_sigma,
            )
        else:
            self._ref = self._build_tunnel_ref(
                obs.pTLL_array,
                gates_quat_wxyz,
                self._gates_information,
                W_nom=W_nom, H_nom=H_nom, tunnel_sigma=tunnel_sigma,
            )

        cost_cfg = {
            "q_lag": q_lag, "q_lag_peak": q_lag_peak,
            "q_contour": q_contour, "q_contour_peak": q_contour_peak,
            "q_attitude": q_attitude,
            "r_thrust": r_thrust, "r_roll": r_roll,
            "r_pitch": r_pitch, "r_yaw": r_yaw,
            "w_speed_gate": w_speed_gate,
        }

        from lsy_drone_racing.control.mpcc.mpccpp_setup import create_ocp_solver_mpccpp

        self._solver, self._ocp = create_ocp_solver_mpccpp(
            N=N_horizon, Tf=T_horizon,
            parameters=self.drone_params,
            n_obstacles=n_obs,
            cost_cfg=cost_cfg,
            tunnel_soft=tunnel_soft,
            tunnel_slack_lin=tunnel_slack_lin,
            tunnel_slack_quad=tunnel_slack_quad,
            obstacle_soft=obstacle_soft,
            obstacle_slack_lin=obstacle_slack_lin,
            obstacle_slack_quad=obstacle_slack_quad,
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()

        # Always start at theta=0 (beginning of the tunnel path).
        # Projecting obs.pBLL is unreliable: if the drone starts near the last
        # gate (geometrically closest), projection returns a theta near the end.
        theta_0 = 0.0
        self._theta_pred = np.arange(self._N + 1) * self._dt * 1.0
        self._init_warmstart(obs)

        # Internal controller state (bootstrapped from hover)
        self._last_theta  = theta_0
        self._last_f_col  = self._mass * self._gravity
        self._last_f_cmd  = self._mass * self._gravity
        self._last_rpy_cmd = np.zeros(3)
        self._infeas_counter = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Reference path construction
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_tunnel_ref(
        gate_positions: np.ndarray,
        gate_quats_wxyz: np.ndarray,
        gates_info: dict,
        W_nom: float,
        H_nom: float,
        tunnel_sigma: float,
    ) -> TunnelReferencePath:
        """Build a TunnelReferencePath from gate poses and geometry."""
        gate_normals = _gate_normals_from_quats(gate_quats_wxyz)
        gate_w_half  = gates_info["hole_width"]  / 2.0
        gate_h_half  = gates_info["hole_height"] / 2.0
        return TunnelReferencePath(
            gate_centers=gate_positions,
            gate_normals=gate_normals,
            gate_w_half=gate_w_half,
            gate_h_half=gate_h_half,
            closed=False,
            W_nom=W_nom, H_nom=H_nom,
            tunnel_sigma=tunnel_sigma,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Obstacle management
    # ──────────────────────────────────────────────────────────────────────────

    def _update_obst_params(self, positions: np.ndarray) -> None:
        """Refresh obstacle parameter slots from current XY positions."""
        m = min(len(positions), self._n_obstacles)
        self._obst_params[:m, 0] = positions[:m, 0]
        self._obst_params[:m, 1] = positions[:m, 1]
        self._obst_params[:m, 2] = self._obstacles_information["d_min"]
        # unused slots keep ro=0 → constraint trivially satisfied

    # ──────────────────────────────────────────────────────────────────────────
    # Per-node parameter vector
    # ──────────────────────────────────────────────────────────────────────────

    def _param_vector(self, theta: float) -> np.ndarray:
        """Compute the per-node parameter vector for the given path parameter."""
        pd   = self._ref.eval(theta)
        td   = self._ref.deriv1(theta)
        pdd  = self._ref.deriv2(theta)
        qc   = self._ref.qc(theta)
        n, b = self._ref.frame(theta)
        W, H = self._ref.width(theta)

        pvec = np.zeros(self._npar)
        pvec[_PD]        = pd
        pvec[_TD]        = td
        pvec[_PDD]       = pdd
        pvec[_THETA_BAR] = theta
        pvec[_QC]        = qc
        pvec[_MU]        = self._mu
        pvec[_NRM]       = n
        pvec[_BNM]       = b
        pvec[_WIDX]      = W
        pvec[_HIDX]      = H
        if self._n_obstacles > 0:
            pvec[_OBST_START: _OBST_START + OBST_DIM * self._n_obstacles] = (
                self._obst_params.reshape(-1)
            )
        return pvec

    # ──────────────────────────────────────────────────────────────────────────
    # Warm start
    # ──────────────────────────────────────────────────────────────────────────

    def _nominal_state(self, theta: float) -> np.ndarray:
        """Hover state along the path at the given arc-length coordinate."""
        pos          = self._ref.eval(theta)
        hover_thrust = self._mass * self._gravity
        x = np.zeros(self._nx)
        x[0:3]  = pos
        x[9]    = hover_thrust   # f_col
        x[10]   = hover_thrust   # f_cmd
        x[14]   = theta
        return x

    def _init_warmstart(self, obs: EnvState_t) -> None:
        """Set up solver warm start and initialise _x_warm / _u_warm caches."""
        self._x_warm = []
        self._u_warm = []
        for k in range(self._N + 1):
            theta_k = float(self._theta_pred[k])
            x_k = self._nominal_state(theta_k)
            self._solver.set(k, "x", x_k)
            self._solver.set(k, "p", self._param_vector(theta_k))
            self._x_warm.append(x_k)
        u_zero = np.zeros(self._nu)
        for k in range(self._N):
            self._solver.set(k, "u", u_zero)
            self._u_warm.append(u_zero.copy())

    # ──────────────────────────────────────────────────────────────────────────
    # ControllerInterface implementation
    # ──────────────────────────────────────────────────────────────────────────

    def control(self, obs: EnvState_t, info: dict | None = None) -> NDArray[np.floating]:
        """Compute attitude + collective-thrust command via MPCC++.

        Returns:
            np.ndarray of shape (4,): [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd].
        """
        # Update gate/obstacle objects for rendering
        gates_quat_wxyz = np.roll(obs.qTLT_array, 1, axis=-1)
        self._gates     = get_gate_objects(obs.pTLL_array, gates_quat_wxyz, self._gates_information)
        self._obstacles = get_obstacle_objects(obs.pOLL_array, self._obstacles_information)
        self._update_obst_params(obs.pOLL_array)

        if self._last_theta >= self._ref.length:
            self._finished = True

        # Assemble initial state from obs + internal controller state
        rpy = self._rpy_from_quat(obs.qBLB)
        x0 = np.array([
            obs.pBLL[0], obs.pBLL[1], obs.pBLL[2],
            obs.vBLL[0], obs.vBLL[1], obs.vBLL[2],
            rpy[0], rpy[1], rpy[2],
            self._last_f_col, self._last_f_cmd,
            self._last_rpy_cmd[0], self._last_rpy_cmd[1], self._last_rpy_cmd[2],
            self._last_theta,
        ])

        # Set per-node parameters and warm start (_x_warm/_u_warm always exist after __init__)
        for k in range(self._N + 1):
            self._solver.set(k, "p", self._param_vector(float(self._theta_pred[k])))
            self._solver.set(k, "x", self._x_warm[k])
        for k in range(self._N):
            self._solver.set(k, "u", self._u_warm[k])

        # Pin initial state
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)
        self._solver.set(0, "x",   x0)

        status = self._solver.solve()

        if status not in (0, 1, 3):
            self._infeas_counter = min(self._infeas_counter + 1, self._N - 1)
            print(
                f"[MPCC++] Solver status {status} — "
                f"open-loop fallback step {self._infeas_counter}."
            )
            return np.array(
                [self._last_rpy_cmd[0], self._last_rpy_cmd[1],
                 self._last_rpy_cmd[2], self._last_f_cmd],
                dtype=np.float32,
            )

        self._infeas_counter = 0

        # Extract solution for warm start
        self._x_warm = [self._solver.get(k, "x") for k in range(self._N + 1)]
        self._u_warm = [self._solver.get(k, "u") for k in range(self._N)]

        # Update internal controller state from stage-1 prediction
        x1 = self._x_warm[1]
        self._last_f_col   = float(x1[9])
        self._last_f_cmd   = float(x1[10])
        self._last_rpy_cmd = np.array(x1[11:14])
        self._last_theta   = float(x1[14])

        # Update theta_pred: shift solution, extrapolate last entry
        sol_theta = np.array([float(self._x_warm[k][14]) for k in range(self._N + 1)])
        u_last    = self._u_warm[-1]
        v_theta_last = float(u_last[4])
        self._theta_pred = np.concatenate([
            sol_theta[1:],
            [sol_theta[-1] + self._dt * v_theta_last],
        ])

        return np.array(
            [self._last_rpy_cmd[0], self._last_rpy_cmd[1],
             self._last_rpy_cmd[2], self._last_f_cmd],
            dtype=np.float32,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # ControllerInterface helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _rpy_from_quat(quat_xyzw: np.ndarray) -> np.ndarray:
        return R.from_quat(quat_xyzw).as_euler("xyz")

    def set_ref_traj(self, planner_traj: dict) -> None:
        """No-op: MPCC++ reference is built from gate geometry, not the planner spline."""
        pass

    def reset(self) -> None:
        """Reset internal controller state for a new episode."""
        self._tick           = 0
        self._finished       = False
        self._last_theta     = 0.0
        self._last_f_col     = self._mass * self._gravity
        self._last_f_cmd     = self._mass * self._gravity
        self._last_rpy_cmd   = np.zeros(3)
        self._infeas_counter = 0
        # Re-initialise theta_pred and warm-start caches at path start
        self._theta_pred = np.arange(self._N + 1) * self._dt * 1.0
        x_zero = np.zeros(self._nx)
        u_zero = np.zeros(self._nu)
        self._x_warm = [x_zero.copy() for _ in range(self._N + 1)]
        self._u_warm = [u_zero.copy() for _ in range(self._N)]

    def set_tick(self, tick: int) -> None:
        """Set tick counter (for interface compatibility; MPCC++ uses theta)."""
        self._tick = tick

    def get_states(self) -> None:
        return None

    def get_predicted_traj(self) -> np.ndarray:
        """Return predicted (x, y, z) positions over the MPC horizon."""
        return np.array([x[:3] for x in self._x_warm])

    def get_ref_traj(self) -> np.ndarray:
        """Return reference (x, y, z) positions along the tunnel path for the horizon."""
        s_end = min(self._last_theta + 3.0, self._ref.length)
        s_q   = np.linspace(self._last_theta, s_end, self._N + 1)
        return np.array([self._ref.eval(float(s)) for s in s_q])
