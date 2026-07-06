"""MPCC++ acados OCP setup: RPY drone model + gate-tunnel constraints + obstacle avoidance.

Per-node parameter layout (20 + 3*n_obstacles + 17*n_gates entries):
    p[0:3]   = pd          (reference position at theta_bar)
    p[3:6]   = td          (reference tangent)
    p[6:9]   = pdd         (reference 2nd derivative)
    p[9]     = theta_bar   (reference progress value)
    p[10]    = qc          (gate proximity weight, dynamic)
    p[11]    = mu          (progress incentive weight, dynamic)
    p[12:15] = n           (tunnel lateral axis, unit vector)
    p[15:18] = b           (tunnel vertical axis, unit vector)
    p[18]    = W           (tunnel half-width)
    p[19]    = H           (tunnel half-height)
    p[20  : 20 + 3*n_obstacles]                 = obstacles ([xo, yo, ro] each)
    p[20 + 3*n_obstacles : .. + 17*n_gates]     = gate frames (WedgeWindow params)

Nonlinear constraints (all >= 0):
    Tunnel (4):    [W + n·d, W − n·d, H + b·d, H − b·d]  where d = pos − pd
    Obstacles (k): (px − xo_i)^2 + (py − yo_i)^2 − ro_i^2  for i = 0..k−1

Soft gate-frame penalty (added to the EXTERNAL cost, not a constraint):
    gate_weight * sum_g WedgeWindow.casadi_penalty_sym(pos, p_g)
    The penalty is 0 when the drone is clear of the gate frame and grows
    quadratically as it penetrates any of the four wedge bars.
"""

from __future__ import annotations

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import DM, MX, cos, dot, norm_2, sin, vertcat
from lsy_drone_racing.envs.environment_constraints.wedge_window import WedgeWindow

# ── Parameter layout constants (mirror mpcc_test/mpccpp_model.py) ─────────────
NP_BASE = 20  # 12 cost params + 8 tunnel-frame params
N_TUNNEL = 4  # four tunnel halfspace constraints
OBST_DIM = 3  # [xo, yo, ro] per obstacle
WEDGE_NP = WedgeWindow.N_PARAMS  # 17 params per gate frame (soft penalty)

_PD = slice(0, 3)
_TD = slice(3, 6)
_PDD = slice(6, 9)
_THETA_BAR = 9
_QC = 10
_MU = 11
_NRM = slice(12, 15)
_BNM = slice(15, 18)
_WIDX = 18
_HIDX = 19
_OBST_START = 20


def num_params(n_obstacles: int, n_gates: int = 0) -> int:
    """Total parameter vector length for given obstacle and gate-frame counts."""
    return NP_BASE + OBST_DIM * n_obstacles + WEDGE_NP * n_gates


def _build_mpccpp_model(
    parameters: dict, n_obstacles: int, cost_cfg: dict, n_gates: int = 0, gate_weight: float = 1e3
) -> AcadosModel:
    """Build the MPCC++ acados model (RPY drone + MPCC cost + tunnel/obstacle h).

    State (15):  [px, py, pz, vx, vy, vz, roll, pitch, yaw,
                  f_col, f_cmd, r_cmd, p_cmd, y_cmd, theta]
    Input  (5):  [df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta]

    When n_gates > 0, a soft gate-frame penalty (WedgeWindow) is added to the
    EXTERNAL cost with weight gate_weight (one block of WEDGE_NP params per gate,
    appended after the obstacle params).
    """
    mass = float(parameters["mass"])
    gravity = -float(parameters["gravity_vec"][-1])

    k = np.array(parameters["rpy_coef"], dtype=float)
    d = np.array(parameters["rpy_rates_coef"], dtype=float)
    b = np.array(parameters["cmd_rpy_coef"], dtype=float)
    eps = 1e-9
    a_rpy = -k / (d + eps)
    beta_rpy = -b / (d + eps)

    # ── State symbols ─────────────────────────────────────────────────────────
    px = MX.sym("px")
    py = MX.sym("py")
    pz = MX.sym("pz")
    vx = MX.sym("vx")
    vy = MX.sym("vy")
    vz = MX.sym("vz")
    roll = MX.sym("roll")
    pitch = MX.sym("pitch")
    yaw = MX.sym("yaw")
    f_col = MX.sym("f_col")
    f_cmd = MX.sym("f_cmd")
    r_cmd = MX.sym("r_cmd")
    p_cmd = MX.sym("p_cmd")
    y_cmd = MX.sym("y_cmd")
    theta = MX.sym("theta")

    states = vertcat(
        px, py, pz, vx, vy, vz, roll, pitch, yaw, f_col, f_cmd, r_cmd, p_cmd, y_cmd, theta
    )

    # ── Input symbols ─────────────────────────────────────────────────────────
    df_cmd = MX.sym("df_cmd")
    dr_cmd = MX.sym("dr_cmd")
    dp_cmd = MX.sym("dp_cmd")
    dy_cmd = MX.sym("dy_cmd")
    v_theta = MX.sym("v_theta")
    inputs = vertcat(df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta)

    # ── Per-node parameters ───────────────────────────────────────────────────
    npar = num_params(n_obstacles, n_gates)
    p = MX.sym("p", npar)

    # ── Dynamics (RPY model identical to mpcc_setup.py) ───────────────────────
    inv_m = 1.0 / mass
    ax = inv_m * f_col * (cos(roll) * sin(pitch) * cos(yaw) + sin(roll) * sin(yaw))
    ay = inv_m * f_col * (cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw))
    az = inv_m * f_col * cos(roll) * cos(pitch) - gravity

    f_dyn = vertcat(
        vx,
        vy,
        vz,
        ax,
        ay,
        az,
        float(a_rpy[0]) * roll + float(beta_rpy[0]) * r_cmd,
        float(a_rpy[1]) * pitch + float(beta_rpy[1]) * p_cmd,
        float(a_rpy[2]) * yaw + float(beta_rpy[2]) * y_cmd,
        10.0 * (f_cmd - f_col),
        df_cmd,
        dr_cmd,
        dp_cmd,
        dy_cmd,
        v_theta,
    )

    # ── MPCC cost (EXTERNAL, per-node params) ─────────────────────────────────
    pos = vertcat(px, py, pz)
    attitude = vertcat(roll, pitch, yaw)
    du = vertcat(df_cmd, dr_cmd, dp_cmd, dy_cmd)

    pd = p[_PD]
    td = p[_TD]
    pdd = p[_PDD]
    theta_bar = p[_THETA_BAR]
    qc = p[_QC]
    mu = p[_MU]

    s = theta - theta_bar
    pd_theta = pd + td * s + 0.5 * pdd * s * s  # local quadratic approximation
    t_raw = td + pdd * s
    tp_unit = t_raw / (norm_2(t_raw) + 1e-6)

    err = pos - pd_theta
    e_lag = dot(tp_unit, err)
    e_cnt = err - e_lag * tp_unit

    r_du = DM(
        np.diag([cost_cfg["r_thrust"], cost_cfg["r_roll"], cost_cfg["r_pitch"], cost_cfg["r_yaw"]])
    )
    # qc in [0, 1] is the gate-proximity bump (from the tunnel reference): 0 away from
    # gates, ~1 at a gate. It scales the *_peak lag/contour weights below and the
    # speed-vs-progress trade-off in `speed`, so tracking tightens and the drone is
    # discouraged from overspeeding right around each gate.
    track = (
        (cost_cfg["q_lag"] + cost_cfg["q_lag_peak"] * qc) * e_lag * e_lag
        + (cost_cfg["q_contour"] + cost_cfg["q_contour_peak"] * qc) * dot(e_cnt, e_cnt)
        + cost_cfg["q_attitude"] * dot(attitude, attitude)
    )
    smooth = du.T @ r_du @ du
    # mu (p[11]) controls the progress incentive at runtime
    speed = -mu * v_theta + cost_cfg["w_speed_gate"] * qc * v_theta**2

    cost_expr = track + smooth + speed

    # ── Soft gate-frame penalty (WedgeWindow), appended to the EXTERNAL cost ──
    # One block of WEDGE_NP params per gate, located right after the obstacle
    # params. The penalty is >= 0, zero when the drone is clear of the frame, and
    # grows as it penetrates any of the four wedge bars. It applies at every node
    # (stage and terminal), independent of the tunnel: the tunnel is the primary
    # soft guide, this is the backup that keeps the drone off the physical frame.
    gate_cost_e = MX(0)
    if n_gates > 0:
        gate_start = _OBST_START + OBST_DIM * n_obstacles
        gate_pen = MX(0)
        for i in range(n_gates):
            pg = p[gate_start + WEDGE_NP * i : gate_start + WEDGE_NP * (i + 1)]
            gate_pen = gate_pen + WedgeWindow.casadi_penalty_sym(pos, pg)
        cost_expr = cost_expr + gate_weight * gate_pen
        gate_cost_e = gate_weight * gate_pen

    # ── Tunnel constraints: four halfspaces, h >= 0 when inside prism ─────────
    n_tun = p[_NRM]
    b_tun = p[_BNM]
    W = p[_WIDX]
    H = p[_HIDX]
    d_pos = pos - pd  # deviation from reference point (not pd_theta)
    h_tunnel = vertcat(
        W + dot(d_pos, n_tun), W - dot(d_pos, n_tun), H + dot(d_pos, b_tun), H - dot(d_pos, b_tun)
    )

    # ── Obstacle keep-out constraints: squared XY distance >= r^2 ────────────
    h_parts: list[MX] = [h_tunnel]
    for i in range(n_obstacles):
        xo = p[_OBST_START + OBST_DIM * i + 0]
        yo = p[_OBST_START + OBST_DIM * i + 1]
        ro = p[_OBST_START + OBST_DIM * i + 2]
        h_parts.append((px - xo) ** 2 + (py - yo) ** 2 - ro**2)
    h = vertcat(*h_parts)

    model = AcadosModel()
    model.name = "mpccpp_drone_racing"
    model.x = states
    model.u = inputs
    model.p = p
    model.f_expl_expr = f_dyn
    model.cost_expr_ext_cost = cost_expr
    model.cost_expr_ext_cost_e = gate_cost_e  # gate penalty only (no tracking)
    model.con_h_expr = h
    model.con_h_expr_e = h
    return model


def create_ocp_solver_mpccpp(
    N: int,
    Tf: float,
    parameters: dict,
    n_obstacles: int = 0,
    cost_cfg: dict | None = None,
    tunnel_soft: bool = True,
    tunnel_slack_lin: float = 1e3,
    tunnel_slack_quad: float = 1e3,
    obstacle_soft: bool = True,
    obstacle_slack_lin: float = 1e4,
    obstacle_slack_quad: float = 1e4,
    v_theta_max: float = 5.0,
    df_cmd_rate_max: float | None = 5.0,
    dr_cmd_rate_max: float | None = None,
    dp_cmd_rate_max: float | None = None,
    dy_cmd_rate_max: float | None = None,
    rate_limit_default: float = 10.0,
    n_gates: int = 0,
    gate_weight: float = 1e3,
    verbose: bool = False,
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Build the MPCC++ acados solver.

    Args:
        N:                  MPC prediction horizon (steps).
        Tf:                 Horizon duration (seconds).
        parameters:         Drone model parameter dict from drone_models.core.load_params.
        n_obstacles:        Number of obstacle slots baked into the OCP.
                            Unused slots are disabled by setting ro=0 at runtime.
        cost_cfg:           MPCC cost weights (defaults provided).
        tunnel_soft:        If True, soften tunnel constraints via acados slacks.
        tunnel_slack_lin:   Linear slack penalty for tunnel.
        tunnel_slack_quad:  Quadratic slack penalty for tunnel.
        obstacle_soft:      If True, soften obstacle constraints via acados slacks.
        obstacle_slack_lin: Linear slack penalty for obstacles.
        obstacle_slack_quad:Quadratic slack penalty for obstacles.
        v_theta_max:        Upper bound on the progress speed v_theta (m/s of arc
                            length). The previous value of 2.0 throttled theta
                            below the drone's along-track speed.
        df_cmd_rate_max:    Slew-rate limit on the collective-thrust command
                            (|df_cmd| <= value, N/s). A finite value ACTIVATES the
                            limit; None falls back to rate_limit_default (inactive).
                            Per control step the command can change by at most
                            df_cmd_rate_max * dt, dt = Tf / N. Default 5.0 N/s is a
                            starting point keyed to the thrust-lag timescale
                            (tau ~ 0.1 s from f_col dynamics); tune to taste.
        dr_cmd_rate_max:    Slew-rate limit on the roll command (|dr_cmd| <= value,
                            rad/s). None => inactive (rate_limit_default).
        dp_cmd_rate_max:    Slew-rate limit on the pitch command (rad/s). None =>
                            inactive.
        dy_cmd_rate_max:    Slew-rate limit on the yaw command (rad/s). None =>
                            inactive.
        rate_limit_default: Wide symmetric bound applied to any command-rate input
                            whose *_cmd_rate_max is None (matches the previous
                            hard-coded +/-10, i.e. effectively unconstrained).
        n_gates:            Number of gate-frame slots baked into the OCP. When > 0,
                            a soft WedgeWindow penalty for each gate is added to the
                            EXTERNAL cost (one WEDGE_NP param block per gate). 0 =>
                            no gate-frame penalty (baseline behaviour).
        gate_weight:        Weight of the soft gate-frame penalty in the cost.
        verbose:            Pass to AcadosOcpSolver.

    Returns:
        (solver, ocp)
    """
    if cost_cfg is None:
        cost_cfg = {
            "q_lag": 80.0,
            "q_lag_peak": 500.0,
            "q_contour": 120.0,
            "q_contour_peak": 700.0,
            "q_attitude": 1.0,
            "r_thrust": 0.2,
            "r_roll": 0.3,
            "r_pitch": 0.3,
            "r_yaw": 0.5,
            "w_speed_gate": 9.0,
        }

    model = _build_mpccpp_model(
        parameters, n_obstacles, cost_cfg, n_gates=n_gates, gate_weight=gate_weight
    )
    nh = N_TUNNEL + n_obstacles
    npar = num_params(n_obstacles, n_gates)
    nx = model.x.rows()

    ocp = AcadosOcp()
    ocp.model = model

    ocp.solver_options.N_horizon = N
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # ── State box constraints: f_col, f_cmd, r_cmd, p_cmd, y_cmd ─────────────
    thrust_min = float(parameters["thrust_min"]) * 4.0
    thrust_max = float(parameters["thrust_max"]) * 4.0
    ocp.constraints.lbx = np.array([thrust_min, thrust_min, -1.57, -1.57, -1.57])
    ocp.constraints.ubx = np.array([thrust_max, thrust_max, 1.57, 1.57, 1.57])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

    # ── Input box constraints: command-rate inputs + progress speed ──────────
    # The four command-rate inputs [df_cmd, dr_cmd, dp_cmd, dy_cmd] are the time
    # derivatives of the command STATES [f_cmd, r_cmd, p_cmd, y_cmd]. Node 0 pins
    # each command state to the last applied value, so bounding a rate input is a
    # true slew-rate limit on the corresponding command -- enforced within the
    # horizon AND across the MPC-step boundary. A finite *_cmd_rate_max activates
    # that input's limit (physical units: N/s for thrust, rad/s for attitude);
    # None leaves it at rate_limit_default (wide => effectively unconstrained).
    # v_theta is the virtual progress input, not a drone command: it keeps its
    # own [0, v_theta_max] box and is excluded from the slew mechanism.
    def _rate_bound(val: float | None) -> float:
        return float(rate_limit_default) if val is None else float(val)

    du_rate = np.array(
        [
            _rate_bound(df_cmd_rate_max),  # collective-thrust command rate
            _rate_bound(dr_cmd_rate_max),  # roll command rate
            _rate_bound(dp_cmd_rate_max),  # pitch command rate
            _rate_bound(dy_cmd_rate_max),  # yaw command rate
        ]
    )
    ocp.constraints.lbu = np.concatenate([-du_rate, [0.0]])
    ocp.constraints.ubu = np.concatenate([du_rate, [v_theta_max]])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    # ── Nonlinear constraints (h >= 0) ────────────────────────────────────────
    big = 1e9
    ocp.constraints.lh = np.zeros(nh)
    ocp.constraints.uh = big * np.ones(nh)
    ocp.constraints.lh_e = np.zeros(nh)
    ocp.constraints.uh_e = big * np.ones(nh)

    # ── Soft slacks (optional per group) ─────────────────────────────────────
    soft_idx: list[int] = []
    zl_vals: list[float] = []
    zl_quad_vals: list[float] = []

    if tunnel_soft:
        soft_idx += list(range(0, N_TUNNEL))
        zl_vals += [tunnel_slack_lin] * N_TUNNEL
        zl_quad_vals += [tunnel_slack_quad] * N_TUNNEL

    if obstacle_soft and n_obstacles > 0:
        soft_idx += list(range(N_TUNNEL, N_TUNNEL + n_obstacles))
        zl_vals += [obstacle_slack_lin] * n_obstacles
        zl_quad_vals += [obstacle_slack_quad] * n_obstacles

    if soft_idx:
        idx = np.array(soft_idx)
        zl = np.array(zl_vals, dtype=float)
        zl_quad = np.array(zl_quad_vals, dtype=float)
        ocp.constraints.idxsh = idx
        ocp.constraints.idxsh_e = idx
        ocp.cost.zl = zl
        ocp.cost.zu = zl
        ocp.cost.Zl = zl_quad
        ocp.cost.Zu = zl_quad
        ocp.cost.zl_e = zl
        ocp.cost.zu_e = zl
        ocp.cost.Zl_e = zl_quad
        ocp.cost.Zu_e = zl_quad

    ocp.constraints.x0 = np.zeros(nx)
    ocp.parameter_values = np.zeros(npar)

    # ── Solver options ────────────────────────────────────────────────────────
    so = ocp.solver_options
    so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    so.hessian_approx = "GAUSS_NEWTON"
    so.integrator_type = "ERK"
    so.sim_method_num_stages = 4
    so.sim_method_num_steps = 2
    so.nlp_solver_type = "SQP_RTI"
    so.qp_solver_iter_max = 50
    so.levenberg_marquardt = 1e-2
    # The WedgeWindow penalty in the EXTERNAL cost can make the exact Hessian
    # locally indefinite; convexify it (as the NMPC soft setup does) so the QP
    # stays well-posed. Only enabled when the gate penalty is active, to leave
    # baseline behaviour untouched.
    if n_gates > 0:
        so.regularize_method = "CONVEXIFY"
    so.qp_solver_warm_start = 1
    so.tf = Tf

    solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/mpccpp_drone_racing.json",
        verbose=verbose,
        build=True,
        generate=True,
    )
    return solver, ocp
