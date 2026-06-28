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
from scipy.interpolate import CubicSpline
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
    WEDGE_NP,
    num_params,
)
from lsy_drone_racing.control.mpcc_test.mpccpp_reference import (
    TunnelReferencePath,
    _gate_axes,
    _perp,
)
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


def _project_point(spline, d1, length, pos, s_lo=0.0, s_hi=None,
                   n_coarse=2000, n_newton=8):
    """Arc length of the point on `spline` nearest to `pos`, searched within
    [s_lo, s_hi]. Coarse nearest-sample seed + a few Gauss-Newton steps."""
    if s_hi is None:
        s_hi = length
    grid = np.linspace(s_lo, s_hi, n_coarse)
    pts = np.asarray(spline(grid))
    s = float(grid[int(np.argmin(np.sum((pts - pos) ** 2, axis=1)))])
    for _ in range(n_newton):
        e = np.asarray(spline(s)) - pos
        t = np.asarray(d1(s))
        h = float(t @ t)
        if h < 1e-12:
            break
        s -= float(e @ t) / h
        s = min(max(s, s_lo), s_hi)
    return s


class SplineTunnelReference:
    """MPCC++ tunnel reference whose centerline IS the planner's racing line.

    The planner's position spline (``des_pos_spline`` -- a time-parameterized
    curve that already threads every gate in order) is reparameterized to arc
    length and used directly as the contouring centerline ``pd(theta)``. Gates
    are NOT inserted as spline knots: each gate center is PROJECTED onto the
    centerline to obtain its progress value ``theta_gate``, and the tunnel pinch
    and ``qc`` bump are keyed to those projected values. Consequently ``theta``
    runs ``0 .. length`` over the true racing line and matches the drone's
    physical progress, and ``ref.length`` equals the racing-line length.

    Exposes the same interface the controller/renderer/plotter rely on:
    ``eval``/``deriv1``/``deriv2``/``tangent``/``frame``/``width``/``qc`` and the
    attributes ``length``/``gate_s``/``gate_centers``/``W_nom``/``H_nom``/
    ``gate_hw``/``gate_hh``/``tunnel_sigma``/``closed``.
    """

    def __init__(self, arc_spline, length, gate_s, gate_centers, gate_n,
                 gate_w, gate_h, gate_hw, gate_hh, W_nom, H_nom, tunnel_sigma,
                 qc_nom=1.0, qc_gate=120.0, gate_sigma=0.8,
                 frame_up=(0.0, 0.0, 1.0), width_floor=0.05):
        self._spline = arc_spline
        self._d1 = arc_spline.derivative(1)
        self._d2 = arc_spline.derivative(2)
        self.length = float(length)
        self.closed = False
        self.gate_s = np.asarray(gate_s, dtype=float)
        self.gate_centers = np.asarray(gate_centers, dtype=float)
        self.gate_n = np.asarray(gate_n, dtype=float)
        self.gate_w = np.asarray(gate_w, dtype=float)
        self.gate_h = np.asarray(gate_h, dtype=float)
        self.gate_hw = np.asarray(gate_hw, dtype=float)
        self.gate_hh = np.asarray(gate_hh, dtype=float)
        self.W_nom = float(W_nom)
        self.H_nom = float(H_nom)
        self.tunnel_sigma = float(tunnel_sigma)
        self.qc_nom = float(qc_nom)
        self.qc_gate = float(qc_gate)
        self.gate_sigma = float(gate_sigma)
        self._up = np.asarray(frame_up, dtype=float)
        self._floor = float(width_floor)

    # ---- centerline (arc-length parameterized) ----
    def _wrap(self, theta):
        return float(np.clip(theta, 0.0, self.length))

    def eval(self, theta):
        return np.asarray(self._spline(self._wrap(theta)), dtype=float)

    def deriv1(self, theta):
        return np.asarray(self._d1(self._wrap(theta)), dtype=float)

    def deriv2(self, theta):
        return np.asarray(self._d2(self._wrap(theta)), dtype=float)

    def tangent(self, theta):
        t = self.deriv1(theta)
        nrm = np.linalg.norm(t)
        return t / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0])

    def project(self, pos, s_lo=0.0, s_hi=None):
        return _project_point(self._spline, self._d1, self.length,
                              np.asarray(pos, dtype=float), s_lo=s_lo, s_hi=s_hi)

    # ---- gate-keyed tunnel (same math as TunnelReferencePath, open path) ----
    def _segment(self, th):
        gs = self.gate_s
        M = len(gs)
        if M == 1:
            return 0, 0, 0.0
        th = min(max(th, gs[0]), gs[-1])
        i = int(np.searchsorted(gs, th, side="right")) - 1
        i = max(0, min(M - 2, i))
        return i, i + 1, (th - gs[i]) / (gs[i + 1] - gs[i] + 1e-12)

    def frame(self, theta):
        t = self.tangent(theta)
        if self.gate_s.size == 0:
            n = _perp(t)
            return n, np.cross(t, n)
        i, j, a = self._segment(self._wrap(theta))
        ew = (1 - a) * self.gate_w[i] + a * self.gate_w[j]
        eh = (1 - a) * self.gate_h[i] + a * self.gate_h[j]
        n = ew - np.dot(ew, t) * t
        nn = np.linalg.norm(n)
        n = n / nn if nn > 1e-9 else _perp(t)
        b = np.cross(t, n)
        if np.dot(b, eh) < 0:
            n, b = -n, -b
        return n, b

    def width(self, theta):
        th = self._wrap(theta)
        d = th - self.gate_s
        g = np.exp(-0.5 * (d / self.tunnel_sigma) ** 2)
        W = self.W_nom - float(np.sum((self.W_nom - self.gate_hw) * g))
        H = self.H_nom - float(np.sum((self.H_nom - self.gate_hh) * g))
        return max(W, self._floor), max(H, self._floor)

    def qc(self, theta):
        if self.gate_s.size == 0:
            return self.qc_nom
        th = self._wrap(theta)
        d = th - self.gate_s
        bump = float(np.exp(-0.5 * (d / self.gate_sigma) ** 2).sum())
        bump = min(bump, 1.0)
        return self.qc_nom + (self.qc_gate - self.qc_nom) * bump


def _centerline_points(centerline, n_arc: int = 4000) -> np.ndarray:
    """Return dense (n, 3) centerline samples from a spline OR a points array.

    Accepts a scipy spline (has ``.x``; sampled over its parameter range) or an
    already-dense (n, 3) array (e.g. the online planner's Trajectory.positions).
    """
    if hasattr(centerline, "x"):                       # scipy CubicSpline over a parameter
        t0, t1 = float(centerline.x[0]), float(centerline.x[-1])
        return np.asarray(centerline(np.linspace(t0, t1, n_arc)), dtype=float)
    return np.asarray(centerline, dtype=float)         # already dense path samples


def _planner_centerline(planner):
    """Extract the centerline source from whatever the planner handed us.

    - Online planner: a Trajectory dataclass -> use its .positions (n, 3).
    - Warp / basic planner: a dict -> use its 'des_pos_spline' (scipy spline).
    """
    if hasattr(planner, "positions"):                  # Trajectory(positions, velocities, ts)
        return np.asarray(planner.positions, dtype=float)
    if isinstance(planner, dict) and "des_pos_spline" in planner:
        return planner["des_pos_spline"]
    raise TypeError("planner must be a Trajectory (with .positions) or a dict "
                    "with 'des_pos_spline'.")


def _gate_anchored_centerline(
    start_pos: np.ndarray,
    gate_centers: np.ndarray,
    gate_normals: np.ndarray,
    gate_tangent_len: float = 0.3,
    n_dense: int = 4000,
) -> np.ndarray:
    """Build a GATE-anchored centerline (Krinner et al., RSS 2024, Sec. IV-A).

    The curve passes through every gate center and, AT each gate, has its tangent
    aligned with the gate normal -- achieved with two collinear helper knots
    ``c +/- delta * n`` around each center. It begins at ``start_pos`` (the drone),
    so theta = 0 maps to the drone and runs to the final gate. Unlike the
    planner-anchored centerline, this curve is fixed by the gate geometry alone,
    so the tunnel pinch sits ON the real gate opening rather than wherever the
    planner happened to route.

    The knots are fitted to a cubic and densely resampled, so the downstream
    arc-length reparameterisation in ``_build_spline_tunnel_ref`` is accurate
    (theta == arc length, |centerline'| == 1).

    Args:
        start_pos:        (3,) current drone position -- first centerline knot.
        gate_centers:     (M, 3) gate center positions (in pass order).
        gate_normals:     (M, 3) gate through-normals (unit; from gate quats).
        gate_tangent_len: Helper-knot offset delta (m) along the normal. Larger =
                          straighter approach/exit through the gate; keep below the
                          gate spacing. Assumes the drone start is upstream of gate 0.
        n_dense:          Dense resample count for the fitted centerline.

    Returns:
        (n_dense, 3) densely sampled gate-threading centerline.
    """
    delta = float(gate_tangent_len)
    centers = np.asarray(gate_centers, dtype=float)
    normals = np.asarray(gate_normals, dtype=float)

    knots = [np.asarray(start_pos, dtype=float)]
    for c, n in zip(centers, normals):
        nn = n / (np.linalg.norm(n) + 1e-12)
        knots.append(c - delta * nn)   # pre-gate helper: tangent -> gate normal
        knots.append(c)                # gate center (curve passes through it)
        knots.append(c + delta * nn)   # post-gate helper
    knots = np.asarray(knots, dtype=float)

    # Fit a cubic through the (sparse) knots and resample densely, so that the
    # arc-length reparameterisation downstream is accurate.
    seg = np.linalg.norm(np.diff(knots, axis=0), axis=1)
    u   = np.concatenate([[0.0], np.cumsum(seg)])
    u_u, idx = np.unique(u, return_index=True)   # drop coincident knots
    spline = CubicSpline(u_u, knots[idx])
    return spline(np.linspace(0.0, float(u_u[-1]), n_dense))


def _build_spline_tunnel_ref(
    centerline,
    gate_positions: np.ndarray,
    gate_quats_wxyz: np.ndarray,
    gate_w_half: float,
    gate_h_half: float,
    W_nom: float,
    H_nom: float,
    tunnel_sigma: float,
    frame_up: tuple = (0.0, 0.0, 1.0),
    qc_nom: float = 0.0,
    qc_gate: float = 1.0,
    gate_sigma: float = 0.4,
    n_arc: int = 10000,
) -> SplineTunnelReference:
    """Build a SplineTunnelReference from the centerline path and the gate poses.

    Args:
        centerline:      The racing-line geometry, either a scipy spline (warp
                         planner) or a dense (n, 3) points array (online planner
                         Trajectory.positions). It already threads the gates.
        gate_positions:  (M, 3) gate center positions (the gates to pinch at).
        gate_quats_wxyz: (M, 4) gate orientation quaternions (wxyz).
        gate_w_half:     Gate cross-section half-width target at the gates (m).
        gate_h_half:     Gate cross-section half-height target at the gates (m).
        W_nom:           Nominal tunnel half-width between gates (m).
        H_nom:           Nominal tunnel half-height between gates (m).
        tunnel_sigma:    Gaussian sigma for the tunnel pinch (arc-length, m).
        frame_up:        World up-vector for gate lateral/vertical axes.
        qc_nom/qc_gate:  Gate-proximity bump that scales q_*_peak / w_speed_gate.
                         Must be a 0..1 multiplier (qc_nom=0 away, qc_gate~1 at a
                         gate) -- NOT a raw contour weight. The cost already holds
                         the magnitudes in q_lag_peak / q_contour_peak / w_speed_gate.
        gate_sigma:      Gaussian sigma for the qc bump (arc-length, m).
        n_arc:           Resample count when `centerline` is a spline (ignored for
                         an already-dense points array).

    Returns:
        A SplineTunnelReference whose centerline is the arc-length racing line.
    """
    # 1) reparameterize the centerline path to arc length.
    pts = _centerline_points(centerline, n_arc=n_arc)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    s_u, idx = np.unique(s, return_index=True)   # strictly increasing knots
    arc_spline = CubicSpline(s_u, pts[idx])
    length = float(s_u[-1])
    d1 = arc_spline.derivative(1)

    # 2) project each gate center onto the centerline, in path order so that
    #    gate_s is monotonically increasing (avoids snapping to a later pass if
    #    the racing line loops back near an earlier gate).
    gates = np.asarray(gate_positions, dtype=float)
    M = len(gates)
    gate_s = np.zeros(M)
    s_lo = 0.0
    for i in range(M):
        gate_s[i] = _project_point(arc_spline, d1, length, gates[i], s_lo=s_lo)
        s_lo = min(gate_s[i] + 1e-3, length)
    gate_s = np.clip(gate_s, 0.0, length)

    # 3) gate cross-section axes from the gate normals.
    up = np.asarray(frame_up, dtype=float)
    normals = _gate_normals_from_quats(gate_quats_wxyz)
    gw = np.zeros((M, 3))
    gh = np.zeros((M, 3))
    for i in range(M):
        gw[i], gh[i] = _gate_axes(normals[i], up)
    gate_hw = np.broadcast_to(gate_w_half, (M,)).astype(float).copy()
    gate_hh = np.broadcast_to(gate_h_half, (M,)).astype(float).copy()

    return SplineTunnelReference(
        arc_spline=arc_spline, length=length, gate_s=gate_s,
        gate_centers=gates, gate_n=normals, gate_w=gw, gate_h=gh,
        gate_hw=gate_hw, gate_hh=gate_hh,
        W_nom=W_nom, H_nom=(W_nom if H_nom is None else H_nom),
        tunnel_sigma=tunnel_sigma, qc_nom=qc_nom, qc_gate=qc_gate,
        gate_sigma=gate_sigma, frame_up=frame_up,
    )


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
        N_horizon: int = 40,
        T_horizon: float = 0.7,
        mu: float = 8.0,
        q_lag: float = 80.0,
        q_lag_peak: float = 50.0,
        q_contour: float = 120.0,
        q_contour_peak: float = 100.0,
        q_attitude: float = 1.0,
        r_thrust: float = 0.2,
        r_roll: float = 0.3,
        r_pitch: float = 0.3,
        r_yaw: float = 0.5,
        w_speed_gate: float = 5.0,
        W_nom: float = 0.3,
        H_nom: float = 0.3,
        tunnel_sigma: float = 0.4,
        v_theta_max: float = 3.0,
        df_cmd_rate_max: float | None = 5.0,
        dr_cmd_rate_max: float | None = None,
        dp_cmd_rate_max: float | None = None,
        dy_cmd_rate_max: float | None = None,
        qc_gate: float = 1.0,
        gate_sigma: float = 0.4,
        tunnel_mode: str = "planner",
        gate_tangent_len: float = 0.3,
        tunnel_soft: bool = True,
        tunnel_slack_lin: float = 1e3,
        tunnel_slack_quad: float = 1e3,
        obstacle_soft: bool = True,
        obstacle_slack_lin: float = 1e4,
        obstacle_slack_quad: float = 1e4,
        gate_soft: bool = True,
        gate_weight: float = 3*1e3,
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
            v_theta_max:        Max progress speed v_theta (m/s of arc length). Must
                                exceed the drone's along-track speed or theta lags.
            df_cmd_rate_max:    Slew-rate limit on the collective-thrust command
                                (|df_cmd| <= value, N/s). Finite value activates it;
                                None => inactive. Per step the command moves at most
                                df_cmd_rate_max * (T_horizon / N_horizon).
            dr_cmd_rate_max:    Slew-rate limit on the roll command (rad/s); None off.
            dp_cmd_rate_max:    Slew-rate limit on the pitch command (rad/s); None off.
            dy_cmd_rate_max:    Slew-rate limit on the yaw command (rad/s); None off.
            qc_gate:            Peak of the gate-proximity bump (0..1) that scales the
                                near-gate tracking weights and the gate speed penalty.
                                Raise to slow/tighten more at gates, lower to fly faster.
            gate_sigma:         Arc-length width (m) of that bump. Keep below ~half the
                                gate spacing so bumps don't overlap and brake everywhere.
            tunnel_mode:        Tunnel centerline source. "gate" (default) anchors the
                                centerline to the gate centers with tangents along the
                                gate normals, so the tube/pinch sit ON the real gate
                                openings (Krinner et al. Sec. IV-A). "planner" uses the
                                planner's racing line as the centerline (legacy).
            gate_tangent_len:   Helper-knot offset delta (m) along each gate normal for
                                the "gate" centerline; larger = straighter through-gate
                                approach. Unused in "planner" mode.
            tunnel_soft:        If True, soften tunnel constraints via slacks.
            tunnel_slack_lin:   Linear slack penalty for tunnel.
            tunnel_slack_quad:  Quadratic slack penalty for tunnel.
            obstacle_soft:      If True, soften obstacle constraints via slacks.
            obstacle_slack_lin: Linear slack penalty for obstacles.
            obstacle_slack_quad:Quadratic slack penalty for obstacles.
            gate_soft:          If True, add a soft WedgeWindow gate-frame penalty
                                to the cost (one per gate, using the REAL gate
                                opening, independent of the tunnel pinch). This is
                                the backup that keeps the drone off the physical
                                frame if the soft tunnel is violated.
            gate_weight:        Weight of the soft gate-frame penalty.
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
            "total_length": 0.9, "total_height": 0.9,
            "hole_width": 0.18,  "hole_height": 0.18,
            "thickness": 0.35,   "margin": 0.05,
        }
        self._obstacles_information = {"d_min": 0.15, "total_height": 2.0}
        # Real gate geometry for the soft frame penalty, captured BEFORE the
        # tunnel-pinch override below shrinks hole_width/hole_height. The frame
        # bars must span the true 0.23 opening, not the 0.1 tunnel pinch.
        self._gate_frame_info = dict(self._gates_information)

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._mass    = float(self.drone_params["mass"])
        self._gravity = -float(self.drone_params["gravity_vec"][-1])

        # Gate / obstacle objects for rendering (updated each control step)
        gates_quat_wxyz = np.roll(obs.qTLT_array, 1, axis=-1)
        self._gates     = get_gate_objects(obs.pTLL_array, gates_quat_wxyz, self._gates_information)
        self._obstacles = get_obstacle_objects(obs.pOLL_array, self._obstacles_information)

        n_obs = len(obs.pOLL_array) if n_obstacles is None else n_obstacles
        self._n_obstacles = n_obs

        # Gate-frame penalty slots: one per gate (all gates, fixed count), so the
        # parameter size stays constant; passed gates contribute ~0 penalty.
        self._gate_soft   = bool(gate_soft)
        self._gate_weight = float(gate_weight)
        self._n_gates     = len(obs.pTLL_array) if self._gate_soft else 0
        self._gate_frame_params = np.zeros((self._n_gates, WEDGE_NP))
        self._update_gate_frame_params(obs.pTLL_array, gates_quat_wxyz)

        self._npar = num_params(n_obs, self._n_gates)

        # Obstacle parameter slots [xo, yo, ro] (updated online each step)
        self._obst_params = np.zeros((n_obs, OBST_DIM))
        self._update_obst_params(obs.pOLL_array)

        # Build the tunnel reference path.
        # The CENTERLINE is the planner's racing line -- either a Trajectory
        # (online planner, re-rooted at the drone) or a dict with des_pos_spline
        # (warp/basic planner). It already threads the gates; we reparameterize it
        # to arc length and use it as pd(theta). Gates are PROJECTED onto it to
        # locate the pinches, so theta runs 0..ref.length over the racing line.
        # Only the *remaining* gates (pTLL_index:) are pinched, matching the
        # online planner, which plans from the current target gate onward.
        self._W_nom = float(W_nom)
        self._H_nom = float(H_nom)
        # Tunnel cross-section target at the gates (half the hole opening).
        #self._gates_information["hole_width"] = 0.1
        #self._gates_information["hole_height"] = 0.1
        self._gate_w_half  = self._gates_information["hole_width"] / 2.0
        self._gate_h_half  = self._gates_information["hole_height"] / 2.0
        self._tunnel_sigma = float(tunnel_sigma)
        self._qc_gate      = float(qc_gate)
        self._gate_sigma   = float(gate_sigma)
        self._v_theta_max  = float(v_theta_max)
        self._tunnel_mode  = str(tunnel_mode)
        self._gate_tangent_len = float(gate_tangent_len)

        gi = int(getattr(obs, "pTLL_index", 0))
        gate_pos  = np.asarray(obs.pTLL_array, dtype=float)[gi:]
        gate_quat = gates_quat_wxyz[gi:]
        self._ref = _build_spline_tunnel_ref(
            centerline=self._centerline_source(planner, obs, gate_pos, gate_quat),
            gate_positions=gate_pos,
            gate_quats_wxyz=gate_quat,
            gate_w_half=self._gate_w_half,
            gate_h_half=self._gate_h_half,
            W_nom=W_nom, H_nom=H_nom, tunnel_sigma=tunnel_sigma,
            qc_gate=qc_gate, gate_sigma=gate_sigma,
        )
        # Gate poses behind the current ref (for the warp planner's change-detect).
        self._gate_update_tol = 0.02
        self._ref_gate_pos    = gate_pos.copy()
        self._ref_gate_quat   = np.asarray(gate_quat, dtype=float).copy()

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
            v_theta_max=v_theta_max,
            df_cmd_rate_max=df_cmd_rate_max,
            dr_cmd_rate_max=dr_cmd_rate_max,
            dp_cmd_rate_max=dp_cmd_rate_max,
            dy_cmd_rate_max=dy_cmd_rate_max,
            n_gates=self._n_gates,
            gate_weight=self._gate_weight,
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

    def _centerline_source(self, planner, obs, gate_pos, gate_quat):
        """Pick the tunnel centerline according to ``self._tunnel_mode``.

        "gate":    a gate-anchored centerline through the gate centers with
                   tangents along the gate normals, rooted at the current drone
                   position (the tube sits ON the real gate openings).
        "planner": the planner's racing line (legacy behaviour).
        """
        if self._tunnel_mode == "gate":
            normals = _gate_normals_from_quats(gate_quat)
            return _gate_anchored_centerline(
                obs.pBLL, gate_pos, normals, self._gate_tangent_len
            )
        return _planner_centerline(planner)

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

    def _update_gate_frame_params(self, positions: np.ndarray, quats_wxyz: np.ndarray) -> None:
        """Refresh gate-frame WedgeWindow params from current gate poses.

        Uses the REAL gate opening (``self._gate_frame_info``), independent of the
        tunnel pinch. Builds one WedgeWindow per gate and stores its 17-param
        vector; these are written into every node's parameter block.
        """
        if self._n_gates == 0:
            return
        frames = get_gate_objects(positions, quats_wxyz, self._gate_frame_info)
        m = min(len(frames), self._n_gates)
        for i in range(m):
            self._gate_frame_params[i] = frames[i].param_vector()

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
        if self._n_gates > 0:
            gstart = _OBST_START + OBST_DIM * self._n_obstacles
            pvec[gstart: gstart + WEDGE_NP * self._n_gates] = (
                self._gate_frame_params.reshape(-1)
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
        x[0:3]  = self._ref.eval(theta)#pos
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
        self._update_gate_frame_params(obs.pTLL_array, gates_quat_wxyz)

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

    def replan_reference(self, trajectory, obs: EnvState_t) -> None:
        """Adopt a freshly planned trajectory as the new tunnel centerline.

        The online planner re-roots its trajectory at the *current drone
        position*, so adopting it means:
          1. rebuild the arc-length centerline + tube from trajectory.positions
             and the remaining gates (pTLL_index:),
          2. reset progress theta to 0 -- the drone IS the new start,
          3. re-seed theta_pred and the warm start along the new centerline.

        Call this only when the planner has actually replanned (e.g. on the
        pipeline's gate/obstacle-moved trigger), not every control step.

        Args:
            trajectory: The planner's Trajectory (uses .positions), or a dict
                        with 'des_pos_spline'.
            obs:        Current observation (drone pose + remaining gate poses).
        """
        gi = int(getattr(obs, "pTLL_index", 0))
        gate_pos  = np.asarray(obs.pTLL_array, dtype=float)[gi:]
        gate_quat = np.roll(obs.qTLT_array, 1, axis=-1)[gi:]

        self._ref = _build_spline_tunnel_ref(
            centerline=self._centerline_source(trajectory, obs, gate_pos, gate_quat),
            gate_positions=gate_pos,
            gate_quats_wxyz=gate_quat,
            gate_w_half=self._gate_w_half,
            gate_h_half=self._gate_h_half,
            W_nom=self._W_nom, H_nom=self._H_nom,
            tunnel_sigma=self._tunnel_sigma,
            qc_gate=self._qc_gate, gate_sigma=self._gate_sigma,
        )
        self._ref_gate_pos  = gate_pos.copy()
        self._ref_gate_quat = np.asarray(gate_quat, dtype=float).copy()

        # theta resets to the start: the new centerline begins at the drone.
        self._last_theta = 0.0
        self._finished   = False
        v_guess = float(np.clip(np.linalg.norm(obs.vBLL), 0.5, self._v_theta_max))
        self._theta_pred = np.clip(
            np.arange(self._N + 1) * self._dt * v_guess, 0.0, self._ref.length
        )
        self._reinit_warmstart()

    def _reinit_warmstart(self) -> None:
        """Re-seed _x_warm / _u_warm with hover states along the current centerline.

        Solver stage values are (re)applied from these caches at the next
        control() call, which also pins node 0 to the measured state.
        """
        self._x_warm = [self._nominal_state(float(self._theta_pred[k]))
                        for k in range(self._N + 1)]
        self._u_warm = [np.zeros(self._nu) for _ in range(self._N)]

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