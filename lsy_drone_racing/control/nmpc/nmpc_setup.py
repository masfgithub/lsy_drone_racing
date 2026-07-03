"""TBD: for Ruff."""

import casadi as cs
import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.so_rpy import symbolic_dynamics_euler

from lsy_drone_racing.control.nmpc.env_constraints import create_env_constraints
from lsy_drone_racing.control.nmpc.obstacle import CylinderObstacle
from lsy_drone_racing.control.nmpc.window import Window


def create_acados_model(parameters: dict, use_input_rate: bool = False) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model.

    Args:
        parameters:     Drone model parameters.
        use_input_rate: If True, augment the model so the four commands
                        [r_cmd, p_cmd, y_cmd, f_cmd] become STATES and their time
                        rates [dr_cmd, dp_cmd, dy_cmd, df_cmd] become the inputs.
                        This is what lets the input box act as a per-command
                        slew-rate limit (see create_ocp_solver). When False the
                        baseline 12-state / 4-input model is returned unchanged.
    """
    # For more info on the models, check out https://github.com/utiasDSL/drone-models
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

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.f_impl_expr = None

    if use_input_rate:
        # Promote the commands U = [r_cmd, p_cmd, y_cmd, f_cmd] to states and make
        # their rates the new inputs:  x_aug = [X; u_cmd],  u_aug = du_cmd,
        # with  d/dt(u_cmd) = du_cmd  and the original dynamics evaluated at the
        # command STATE (U -> u_cmd) instead of a free input.
        sym = cs.SX.sym if isinstance(U, cs.SX) else cs.MX.sym
        nu0 = U.shape[0]
        u_cmd  = sym("u_cmd", nu0)    # [r_cmd, p_cmd, y_cmd, f_cmd]  (now states)
        du_cmd = sym("du_cmd", nu0)   # [dr_cmd, dp_cmd, dy_cmd, df_cmd]  (now inputs)
        X_dot_sub = cs.substitute(X_dot, U, u_cmd)
        model.name = "rate_aug_mpc"
        model.x = cs.vertcat(X, u_cmd)
        model.u = du_cmd
        model.f_expl_expr = cs.vertcat(X_dot_sub, du_cmd)
    else:
        model.name = "basic_example_mpc"
        model.x = X
        model.u = U
        model.f_expl_expr = X_dot

    return model


def create_ocp_solver(
    Tf: float,
    N: int,
    parameters: dict,
    gates: list[Window],
    obstacles: list[CylinderObstacle],
    use_input_rate: bool = False,
    df_cmd_rate_max: float | None = 5.0,
    dr_cmd_rate_max: float | None = None,
    dp_cmd_rate_max: float | None = None,
    dy_cmd_rate_max: float | None = None,
    rate_limit_default: float = 10.0,
    r_rate: float = 0.01,
    verbose: bool = False,
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver.

    Args:
        Tf, N, parameters, gates, obstacles: as before.
        use_input_rate:     If True, build the rate-augmented model (commands are
                            states, rates are inputs) and apply per-command
                            slew-rate limits via the input box.
        df_cmd_rate_max:    Slew limit on the thrust command (|df_cmd| <= value,
                            N/s). Finite activates; None => rate_limit_default.
        dr/dp/dy_cmd_rate_max: Slew limits on roll/pitch/yaw commands (rad/s).
                            None => inactive (rate_limit_default).
        rate_limit_default: Wide bound for inactive rate inputs.
        r_rate:             Small LS weight on the rate inputs (conditioning /
                            smoothing; only used when use_input_rate=True).
    """
    ocp = AcadosOcp()

    # Set model
    ocp.model = create_acados_model(parameters, use_input_rate=use_input_rate)

    # Get Dimensions
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    #

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    # Base physical-state weights (pos, rpy, vel, drpy) -- identical to baseline.
    Q_diag = np.array(
        [50.0, 50.0, 400.0,  1.0, 1.0, 1.0,  5.0, 5.0, 5.0,  5.0, 5.0, 5.0]
    )
    Q_e_diag = np.array(
        [100.0, 100.0, 100.0,  0.1, 0.1, 0.1,  0.1, 0.1, 0.1,  0.1, 0.1, 0.1]
    )
    # Command penalty (reference = upright orientation + hover thrust). In the
    # baseline these weight the INPUTS; under augmentation the commands are states
    # so the same weights move onto the command-state block of Q / Q_e instead.
    cmd_diag = np.array([0.1, 0.1, 0.1, 0.1])  # [r_cmd, p_cmd, y_cmd, f_cmd]

    if use_input_rate:
        # Commands are states -> their penalty joins Q; the new inputs are the
        # rates, which get a small LS weight for conditioning / smoothing.
        Q = np.diag(np.concatenate([Q_diag, cmd_diag]))
        Q_e = np.diag(np.concatenate([Q_e_diag, cmd_diag]))
        R = np.diag([r_rate, r_rate, r_rate, r_rate])
    else:
        Q = np.diag(Q_diag)
        Q_e = np.diag(Q_e_diag)
        R = np.diag(cmd_diag)

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx_e = Vx_e

    # Set initial references. We will overwrite these later to track the trajectory
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    thrust_min = parameters["thrust_min"] * 4
    thrust_max = parameters["thrust_max"] * 4
    if use_input_rate:
        # Commands are now states: their MAGNITUDE limits (formerly the input box)
        # move to the STATE box on indices 12..15; the input box becomes the
        # per-command SLEW-RATE limit on [dr_cmd, dp_cmd, dy_cmd, df_cmd].
        ocp.constraints.lbx = np.array(
            [-0.5, -0.5, -0.5,  -0.5, -0.5, -0.5,  thrust_min]
        )
        ocp.constraints.ubx = np.array(
            [0.5, 0.5, 0.5,  0.5, 0.5, 0.5,  thrust_max]
        )
        ocp.constraints.idxbx = np.array([3, 4, 5, 12, 13, 14, 15])

        def _rate_bound(val: float | None) -> float:
            return float(rate_limit_default) if val is None else float(val)

        du_rate = np.array([
            _rate_bound(dr_cmd_rate_max),   # roll-command rate
            _rate_bound(dp_cmd_rate_max),   # pitch-command rate
            _rate_bound(dy_cmd_rate_max),   # yaw-command rate
            _rate_bound(df_cmd_rate_max),   # thrust-command rate
        ])
        ocp.constraints.lbu = -du_rate
        ocp.constraints.ubu = du_rate
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    else:
        # Set State Constraints (rpy < ~30°)
        ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
        ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
        ocp.constraints.idxbx = np.array([3, 4, 5])

        # Set Input Constraints (rpy command + thrust magnitude)
        ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, thrust_min])
        ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, thrust_max])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # Set environmental constraints
    pBLL = ocp.model.x[:3]
    env = create_env_constraints(model=ocp.model, pBLL=pBLL, gates=gates, obstacles=obstacles)
    ocp.constraints.lh = env["lh"]
    ocp.constraints.uh = env["uh"]
    ocp.constraints.lh_e = env["lh"]
    ocp.constraints.uh_e = env["uh"]
    ocp.parameter_values = env["p0"]

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_, PARTIAL_ ,_HPIPM, _QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP, SQP_RTI
    ocp.solver_options.nlp_solver_tol_stat = 1e-3
    ocp.solver_options.nlp_solver_tol_eq = 1e-3
    ocp.solver_options.nlp_solver_tol_ineq = 1e-3
    ocp.solver_options.nlp_solver_tol_comp = 1e-3

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_example_mpc.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp