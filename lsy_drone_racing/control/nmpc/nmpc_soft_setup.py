"""TBD: for Ruff."""

import casadi as cs
import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import vertcat
from drone_models.so_rpy import symbolic_dynamics_euler

from lsy_drone_racing.control.nmpc.env_soft_constraints import create_soft_env_constraints
from lsy_drone_racing.control.nmpc.obstacle import CylinderObstacle
from lsy_drone_racing.control.nmpc.window import Window


def create_acados_model(parameters: dict, use_input_rate: bool = False) -> AcadosModel:
    """Creates an acados model from a symbolic drone model.

    When ``use_input_rate`` is True the four commands [r_cmd, p_cmd, y_cmd, f_cmd]
    are promoted to states and their rates [dr_cmd, dp_cmd, dy_cmd, df_cmd] become
    the inputs (see create_ocp_solver_soft).
    """
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    model = AcadosModel()
    model.f_impl_expr = None

    if use_input_rate:
        sym = cs.SX.sym if isinstance(U, cs.SX) else cs.MX.sym
        nu0 = U.shape[0]
        u_cmd = sym("u_cmd", nu0)  # [r_cmd, p_cmd, y_cmd, f_cmd]  (states)
        du_cmd = sym("du_cmd", nu0)  # [dr_cmd, dp_cmd, dy_cmd, df_cmd]  (inputs)
        X_dot_sub = cs.substitute(X_dot, U, u_cmd)
        model.name = "soft_rate_aug_mpc"
        model.x = cs.vertcat(X, u_cmd)
        model.u = du_cmd
        model.f_expl_expr = cs.vertcat(X_dot_sub, du_cmd)
    else:
        model.name = "soft_example_mpc"
        model.x = X
        model.u = U
        model.f_expl_expr = X_dot

    return model


def create_ocp_solver_soft(
    Tf: float,
    N: int,
    parameters: dict,
    gates: list[Window],
    obstacles: list[CylinderObstacle],
    gate_weight: float = 1000.0,
    obstacle_weight: float = 1000.0,
    post_weight: float = 1000.0,
    use_input_rate: bool = False,
    df_cmd_rate_max: float | None = 5.0,
    dr_cmd_rate_max: float | None = None,
    dp_cmd_rate_max: float | None = None,
    dy_cmd_rate_max: float | None = None,
    rate_limit_default: float = 10.0,
    r_rate: float = 0.01,
    verbose: bool = False,
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados OCP with soft environment constraints in the cost.

    ``use_input_rate`` and the ``*_cmd_rate_max`` / ``r_rate`` args behave exactly
    as in the hard ``create_ocp_solver`` (commands become states, rates the inputs,
    input box -> per-command slew-rate limit).
    """
    ocp = AcadosOcp()
    ocp.model = create_acados_model(parameters, use_input_rate=use_input_rate)

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu

    ocp.solver_options.N_horizon = N

    # ── Soft env constraints ──────────────────────────────────────────────────
    # Must be done before cost setup so penalty_expr is available.
    pBLL = ocp.model.x[:3]
    env = create_soft_env_constraints(
        model=ocp.model,
        pBLL=pBLL,
        gates=gates,
        obstacles=obstacles,
        gate_weight=gate_weight,
        obstacle_weight=obstacle_weight,
        post_weight=post_weight,
    )

    # ── Cost: NONLINEAR_LS with penalty appended to residual ──────────────────
    # We keep LINEAR_LS for the tracking part and add the penalty as an
    # extra nonlinear residual term so GAUSS_NEWTON still applies.
    #
    # Stage cost:    ||Vx*x + Vu*u - yref||_W^2  +  penalty(x, p)
    # Terminal cost: ||Vx_e*x - yref_e||_W_e^2    +  penalty(x, p)
    #
    # We use NONLINEAR_LS and build the full residual vector manually.

    x_sym = ocp.model.x
    u_sym = ocp.model.u

    # Tracking residual (nx + nu terms)
    tracking_residual = vertcat(x_sym, u_sym)  # yref subtracts at solve time

    # Penalty residual: sqrt(penalty) so that squaring it recovers the penalty.
    # GAUSS_NEWTON squares the residual, so we need sqrt here.
    # We add a small eps for differentiability at zero.
    eps_soft = 1e-6
    from casadi import sqrt as casadi_sqrt

    penalty_residual = casadi_sqrt(env["penalty_expr"] + eps_soft)
    penalty_residual_e = casadi_sqrt(env["penalty_expr_e"] + eps_soft)

    # Full residual vectors
    stage_residual = vertcat(tracking_residual, penalty_residual)
    terminal_residual = vertcat(x_sym, penalty_residual_e)

    ocp.model.cost_y_expr = stage_residual
    ocp.model.cost_y_expr_e = terminal_residual

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    # ── Weight matrices ───────────────────────────────────────────────────────
    Q_diag = np.array([100.0, 100.0, 300.0, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
    Q_e_diag = np.array([50.0, 50.0, 50.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    cmd_diag = np.array([0.1, 0.1, 0.1, 0.1])  # command penalty (rpy_cmd + thrust)

    if use_input_rate:
        # Commands are states -> penalty joins Q; inputs are rates -> small reg.
        Q = np.diag(np.concatenate([Q_diag, cmd_diag]))
        Q_e = np.diag(np.concatenate([Q_e_diag, cmd_diag]))
        R = np.diag([r_rate, r_rate, r_rate, r_rate])
    else:
        Q = np.diag(Q_diag)
        Q_e = np.diag(Q_e_diag)
        R = np.diag(cmd_diag)

    # Penalty weight in the LS sense: weight = 1 because the weight is already
    # baked into gate_weight/obstacle_weight inside penalty_expr.
    W_penalty = np.array([[1.0]])
    W_penalty_e = np.array([[1.0]])

    ocp.cost.W = scipy.linalg.block_diag(Q, R, W_penalty)
    ocp.cost.W_e = scipy.linalg.block_diag(Q_e, W_penalty_e)

    # yref: tracking part is zeros (overwritten at runtime), penalty part = 0
    # (we want the penalty residual to be driven to 0)
    ocp.cost.yref = np.zeros(ny + 1)  # nx + nu + 1 penalty term
    ocp.cost.yref_e = np.zeros(nx + 1)  # nx + 1 penalty term

    # ── State and input constraints ───────────────────────────────────────────
    thrust_min = parameters["thrust_min"] * 4
    thrust_max = parameters["thrust_max"] * 4
    if use_input_rate:
        # Commands are states: magnitude limits -> state box (idx 12..15); the
        # input box becomes per-command slew-rate limits on the rate inputs.
        ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, thrust_min])
        ocp.constraints.ubx = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, thrust_max])
        ocp.constraints.idxbx = np.array([3, 4, 5, 12, 13, 14, 15])

        def _rate_bound(val: float | None) -> float:
            return float(rate_limit_default) if val is None else float(val)

        du_rate = np.array(
            [
                _rate_bound(dr_cmd_rate_max),
                _rate_bound(dp_cmd_rate_max),
                _rate_bound(dy_cmd_rate_max),
                _rate_bound(df_cmd_rate_max),
            ]
        )
        ocp.constraints.lbu = -du_rate
        ocp.constraints.ubu = du_rate
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    else:
        ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
        ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
        ocp.constraints.idxbx = np.array([3, 4, 5])

        ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, thrust_min])
        ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, thrust_max])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    ocp.constraints.x0 = np.zeros(nx)

    # No lh/uh — soft constraints live in the cost
    ocp.parameter_values = env["p0"]

    # ── Solver options ────────────────────────────────────────────────────────
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.levenberg_marquardt = 1e-2
    ocp.solver_options.nlp_solver_tol_stat = 1e-3
    ocp.solver_options.nlp_solver_tol_eq = 1e-3
    ocp.solver_options.nlp_solver_tol_ineq = 1e-3
    ocp.solver_options.nlp_solver_tol_comp = 1e-3
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_example_mpc_soft.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp, env
