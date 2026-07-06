"""NMPC obstacle avoidance — scenario entry point."""

import matplotlib.pyplot as plt
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, vertcat
from env_constraints import (
    create_env_constraints,
    get_gate_objects,
    get_obstacle_objects,
    set_env_params,
    verify_env_constraints,
)
from obstacle import CylinderObstacle
from window import Window


def build_ocp(
    x0: np.ndarray,
    x_ref: np.ndarray,
    gates: list[Window],
    obstacles: list[CylinderObstacle],
    N: int = 200,
    Tf: float = 10.0,
) -> AcadosOcp:
    """Assemble and return the AcadosOcp for gate and obstacle avoidance."""
    model = AcadosModel()
    model.name = "obstacle_avoidance"

    n_x, n_u = 6, 3
    x = MX.sym("x", n_x)
    u = MX.sym("u", n_u)
    model.x = x
    model.u = u
    model.f_expl_expr = vertcat(x[3], x[4], x[5], u[0], u[1], u[2])

    pos = vertcat(x[0], x[1], x[2])

    # Attach all environment constraints and parameters to the model
    env = create_env_constraints(model=model, p_bll=pos, gates=gates, obstacles=obstacles)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = N
    ocp.solver_options.tf = Tf

    n_y = n_x + n_u
    Q = np.diag([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    R = np.diag([0.1, 0.1, 0.1])

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = np.block([[Q, np.zeros((n_x, n_u))], [np.zeros((n_u, n_x)), R]])
    ocp.cost.W_e = Q
    ocp.cost.Vx = np.vstack([np.eye(n_x), np.zeros((n_u, n_x))])
    ocp.cost.Vu = np.vstack([np.zeros((n_x, n_u)), np.eye(n_u)])
    ocp.cost.Vx_e = np.eye(n_x)
    ocp.cost.yref = np.zeros(n_y)
    ocp.cost.yref_e = np.zeros(n_x)

    ocp.constraints.lh = env["lh"]
    ocp.constraints.uh = env["uh"]
    ocp.constraints.lh_e = env["lh"]
    ocp.constraints.uh_e = env["uh"]
    ocp.parameter_values = env["p0"]

    ocp.constraints.lbx = np.array([-20.0, -20.0, -20.0, -5.0, -5.0, -5.0])
    ocp.constraints.ubx = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 5.0])
    ocp.constraints.idxbx = np.arange(n_x)
    ocp.constraints.lbu = np.array([-2.0, -2.0, -2.0])
    ocp.constraints.ubu = np.array([2.0, 2.0, 2.0])
    ocp.constraints.idxbu = np.arange(n_u)
    ocp.constraints.x0 = x0

    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 400
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_iter_max = 400
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_tol_stat = 1e-3
    ocp.solver_options.nlp_solver_tol_eq = 1e-3
    ocp.solver_options.nlp_solver_tol_ineq = 1e-3
    ocp.solver_options.nlp_solver_tol_comp = 1e-3

    return ocp, env


def warm_start(x0: np.ndarray, x_ref: np.ndarray, gates: list[Window], N: int) -> np.ndarray:
    """Return a straight-line warm-start trajectory that steers through gate holes."""
    traj = np.zeros((N + 1, len(x0)))
    for k in range(N + 1):
        beta = k / N
        p = (1.0 - beta) * x0[:3] + beta * x_ref[:3]
        for gate in gates:
            dp = p - gate.position
            p_local = gate.R @ dp
            hx = gate.thickness / 2.0
            hl = gate.total_length / 2.0
            hh = gate.total_height / 2.0
            hw = gate.hole_width / 2.0
            hho = gate.hole_height / 2.0
            in_slab = abs(p_local[0]) <= hx
            in_total = abs(p_local[1]) <= hl and abs(p_local[2]) <= hh
            in_hole = abs(p_local[1]) <= hw and abs(p_local[2]) <= hho
            if in_slab and in_total and not in_hole:
                p_local[1] = 0.0
                p_local[2] = 0.0
                p = gate.position + gate.R.T @ p_local
        traj[k] = np.array([*p, 0.0, 0.0, 0.0])
    return traj


def plot_trajectory(
    x_traj: np.ndarray,
    x0: np.ndarray,
    x_ref: np.ndarray,
    gates: list[Window],
    obstacles: list[CylinderObstacle],
):
    """Plot the optimised 3-D trajectory together with gates and cylinder obstacles."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    fig.suptitle("NMPC Obstacle Avoidance — 3-D Trajectory", fontsize=13, fontweight="bold")

    for obs in obstacles:
        obs.draw(ax)
    for gate in gates:
        gate.draw(ax)

    cmap = plt.cm.Blues
    n_seg = len(x_traj) - 1
    for k in range(n_seg):
        c = cmap(0.35 + 0.65 * k / n_seg)
        ax.plot(
            x_traj[k : k + 2, 0], x_traj[k : k + 2, 1], x_traj[k : k + 2, 2], color=c, linewidth=2.0
        )
    ax.scatter(
        x_traj[:, 0],
        x_traj[:, 1],
        x_traj[:, 2],
        color="steelblue",
        s=8,
        zorder=6,
        label="trajectory",
    )
    ax.scatter(*x0[:3], color="green", marker="s", s=80, label="start", zorder=7)
    ax.scatter(*x_ref[:3], color="green", marker="*", s=160, label="goal", zorder=7)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N = 200
    Tf = 10.0

    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_ref = np.array([7.0, 4.5, 1.5, 0.0, 0.0, 0.0])

    def quat_z(deg: float) -> list:
        """Return a [qw, qx, qy, qz] quaternion for a rotation of deg degrees about Z."""
        a = np.deg2rad(deg) / 2.0
        return [np.cos(a), 0.0, 0.0, np.sin(a)]

    # ── Gates (Windows) ───────────────────────────────────────────────────────
    gate_info = {
        "total_length": 3.0,
        "total_height": 3.0,
        "hole_width": 1.0,
        "hole_height": 1.0,
        "thickness": 0.3,
        "margin": 0.10,
    }

    gate_positions = np.array([[2.0, 3.5, 1.5], [5.0, 0.0, 1.5], [6.0, 4.0, 1.5]])
    gate_quaternions = np.array([quat_z(0.0), quat_z(0.0), quat_z(0.0)])

    gates = get_gate_objects(gate_positions, gate_quaternions, gate_info)

    # ── Cylinder obstacles ────────────────────────────────────────────────────
    obstacle_positions = np.array([[2.0, 0.5], [4.0, 3.0]])
    obstacles_info = {"d_min": 0.6, "total_height": 1.55}
    obstacles = get_obstacle_objects(obstacle_positions, obstacles_info)

    # ── Build OCP ─────────────────────────────────────────────────────────────
    ocp, env = build_ocp(x0=x0, x_ref=x_ref, gates=gates, obstacles=obstacles, N=N, Tf=Tf)
    solver = AcadosOcpSolver(ocp, json_file="ocp_obstacle.json")

    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)
    for k in range(N):
        solver.set(k, "yref", np.concatenate([x_ref, np.zeros(3)]))
    solver.set(N, "yref", x_ref)

    set_env_params(solver, gates, obstacles, N)

    x_init = warm_start(x0, x_ref, gates, N)
    for k in range(N + 1):
        solver.set(k, "x", x_init[k])
    for k in range(N):
        solver.set(k, "u", np.zeros(3))

    status = solver.solve()
    status_meanings = {
        0: "SUCCESS",
        1: "NLP_ITERATION_MAXIMUM",
        2: "INFEASIBLE",
        3: "MINIMUM_STEP_SIZE",
        4: "QP_FAILURE",
        5: "READY",
    }
    print(f"Solver status: {status} → {status_meanings.get(status, 'UNKNOWN')}")
    print(f"Objective value:     {solver.get_cost():.4f}")
    res = solver.get_residuals()
    print(f"Stationarity:        {res[0]:.2e}")
    print(f"Equality (dynamics): {res[1]:.2e}")
    print(f"Inequality:          {res[2]:.2e}")
    print(f"Complementarity:     {res[3]:.2e}")
    print(f"SQP iterations:      {solver.get_stats('sqp_iter')}")
    print(f"Time (total):        {solver.get_stats('time_tot'):.4f} s")

    x_traj = np.array([solver.get(k, "x") for k in range(N + 1)])

    print("\n── Environment constraint verification ──")
    verify_env_constraints(x_traj, gates, obstacles)

    # ── Runtime update example (no rebuild) ───────────────────────────────────
    # gates[0].update(position=[2.0, 2.0, 1.5])
    # obstacles[0].update(position=[3.0, 1.0])
    # set_env_params(solver, gates, obstacles, N)
    # solver.solve()

    plot_trajectory(x_traj, x0, x_ref, gates, obstacles)
