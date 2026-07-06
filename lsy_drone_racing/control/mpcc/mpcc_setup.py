"""MPCC acados OCP setup for drone racing."""

from __future__ import annotations

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import DM, MX, cos, dot, floor, if_else, norm_2, sin, vertcat


def _piecewise_linear_interp(
    sym_theta: MX, theta_grid: np.ndarray, flat_sym: MX, dim: int = 3
) -> MX:
    """CasADi-compatible piecewise linear interpolation along the arc-length axis."""
    M = len(theta_grid)
    t0 = float(theta_grid[0])
    t1 = float(theta_grid[-1])
    idx_f = (sym_theta - t0) / (t1 - t0) * (M - 1)
    idx_l = floor(idx_f)
    idx_h = idx_l + 1
    alpha = idx_f - idx_l
    idx_l = if_else(idx_l < 0, MX(0), idx_l)
    idx_h = if_else(idx_h >= M, MX(M - 1), idx_h)
    p_l = vertcat(*[flat_sym[dim * idx_l + k] for k in range(dim)])
    p_h = vertcat(*[flat_sym[dim * idx_h + k] for k in range(dim)])
    return (1.0 - alpha) * p_l + alpha * p_h


def _build_mpcc_model(
    parameters: dict, model_arc_step: float, model_traj_length: float
) -> tuple[AcadosModel, dict]:
    """Build MPCC acados model.

    State vector (15):  [px, py, pz, vx, vy, vz, roll, pitch, yaw,
                         f_col, f_cmd, r_cmd, p_cmd, y_cmd, theta]
    Input vector  (5):  [df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta]

    Returns (model, sym_vars) where sym_vars is a dict of named CasADi symbols.
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

    # ── Input symbols ─────────────────────────────────────────────────────────
    df_cmd = MX.sym("df_cmd")
    dr_cmd = MX.sym("dr_cmd")
    dp_cmd = MX.sym("dp_cmd")
    dy_cmd = MX.sym("dy_cmd")
    v_theta = MX.sym("v_theta")

    states = vertcat(
        px, py, pz, vx, vy, vz, roll, pitch, yaw, f_col, f_cmd, r_cmd, p_cmd, y_cmd, theta
    )
    inputs = vertcat(df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta)

    # ── Continuous dynamics ───────────────────────────────────────────────────
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

    # ── Trajectory parameters ─────────────────────────────────────────────────
    n_samples = int(model_traj_length / model_arc_step)
    pd_list = MX.sym("pd_list", 3 * n_samples)
    tp_list = MX.sym("tp_list", 3 * n_samples)
    qc_list = MX.sym("qc_list", n_samples)
    params = vertcat(pd_list, tp_list, qc_list)

    model = AcadosModel()
    model.name = "mpcc_drone_racing"
    model.f_expl_expr = f_dyn
    model.x = states
    model.u = inputs
    model.p = params

    sym = {
        "px": px,
        "py": py,
        "pz": pz,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "f_col": f_col,
        "f_cmd": f_cmd,
        "r_cmd": r_cmd,
        "p_cmd": p_cmd,
        "y_cmd": y_cmd,
        "theta": theta,
        "df_cmd": df_cmd,
        "dr_cmd": dr_cmd,
        "dp_cmd": dp_cmd,
        "dy_cmd": dy_cmd,
        "v_theta": v_theta,
        "pd_list": pd_list,
        "tp_list": tp_list,
        "qc_list": qc_list,
        "n_samples": n_samples,
    }
    return model, sym


def _build_cost_expr(
    sym: dict, cost_cfg: dict, model_arc_step: float, model_traj_length: float
) -> MX:
    """Build the MPCC external stage cost expression (lag + contouring + speed)."""
    n_samples = sym["n_samples"]
    theta_grid = np.arange(0.0, model_traj_length, model_arc_step)[:n_samples]

    position = vertcat(sym["px"], sym["py"], sym["pz"])
    attitude = vertcat(sym["roll"], sym["pitch"], sym["yaw"])
    du = vertcat(sym["df_cmd"], sym["dr_cmd"], sym["dp_cmd"], sym["dy_cmd"])
    theta = sym["theta"]

    # Interpolate trajectory at current theta
    pd_theta = _piecewise_linear_interp(theta, theta_grid, sym["pd_list"], dim=3)
    tp_theta = _piecewise_linear_interp(theta, theta_grid, sym["tp_list"], dim=3)
    qc_theta = _piecewise_linear_interp(theta, theta_grid, sym["qc_list"], dim=1)

    # Lag / contouring decomposition
    tp_unit = tp_theta / (norm_2(tp_theta) + 1e-6)
    err = position - pd_theta
    e_lag = dot(tp_unit, err) * tp_unit
    e_cnt = err - e_lag

    # Tracking cost
    q_att = cost_cfg["q_attitude"] * DM(np.eye(3))
    track = (
        (cost_cfg["q_lag"] + cost_cfg["q_lag_peak"] * qc_theta) * dot(e_lag, e_lag)
        + (cost_cfg["q_contour"] + cost_cfg["q_contour_peak"] * qc_theta) * dot(e_cnt, e_cnt)
        + attitude.T @ q_att @ attitude
    )

    # Input smoothness cost
    r_du = DM(
        np.diag([cost_cfg["r_thrust"], cost_cfg["r_roll"], cost_cfg["r_pitch"], cost_cfg["r_yaw"]])
    )
    smooth = du.T @ r_du @ du

    # Speed incentive: reward progress, penalise speed near gates
    v_theta = sym["v_theta"]
    speed = -cost_cfg["mu_speed"] * v_theta + cost_cfg["w_speed_gate"] * qc_theta * v_theta**2

    return track + smooth + speed


def create_ocp_solver_mpcc(
    N: int,
    Tf: float,
    parameters: dict,
    model_arc_step: float = 0.05,
    model_traj_length: float = 15.0,
    cost_cfg: dict | None = None,
    verbose: bool = False,
) -> tuple[AcadosOcpSolver, AcadosOcp, int]:
    """Build the MPCC acados solver.

    Returns:
        (solver, ocp, n_samples) where n_samples is the trajectory sample count.
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
            "mu_speed": 10.0,
            "w_speed_gate": 9.0,
        }

    model, sym = _build_mpcc_model(parameters, model_arc_step, model_traj_length)
    model.cost_expr_ext_cost = _build_cost_expr(sym, cost_cfg, model_arc_step, model_traj_length)
    model.cost_expr_ext_cost_e = MX(0)  # no terminal cost

    ocp = AcadosOcp()
    ocp.model = model

    nx = model.x.rows()
    n_samples = sym["n_samples"]

    ocp.solver_options.N_horizon = N
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # State box constraints on idx 9-13: f_col, f_cmd, r_cmd, p_cmd, y_cmd
    thrust_min = float(parameters["thrust_min"]) * 4.0
    thrust_max = float(parameters["thrust_max"]) * 4.0
    ocp.constraints.lbx = np.array([thrust_min, thrust_min, -1.57, -1.57, -1.57])
    ocp.constraints.ubx = np.array([thrust_max, thrust_max, 1.57, 1.57, 1.57])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

    # Input box constraints: df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta
    ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0, -10.0, 0.0])
    ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0, 2.0])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    ocp.constraints.x0 = np.zeros(nx)
    ocp.parameter_values = np.zeros(7 * n_samples)  # pd(3) + tp(3) + qc(1) per sample

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.tf = Tf

    solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/mpcc_drone_racing.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return solver, ocp, n_samples
