"""Obstacle-avoidance (3D) NMPC example utilities.

Provides functions to build an acados OCP with obstacle avoidance
constraints and to plot results.
"""

import matplotlib.pyplot as plt
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, sumsqr, vertcat


def create_obstacle_avoidance_mpc(pos_indices: list, obst: np.ndarray, d_min: float) -> tuple:
    """Adds obstacle avoidance inequality constraints to an acados OCP.

    Constraint form (acados 'nh' nonlinear constraints):
        lh <= h(x,u) <= uh
        d_min <= ||pos - obst_i||_2 <= +inf   for each obstacle i
    """
    model = AcadosModel()
    model.name = "obstacle_avoidance_example"

    n_x = len(pos_indices) + 3  # [px, py, pz, vx, vy, vz]
    n_u = 3  # [ax, ay, az]
    x = SX.sym("x", n_x)
    u = SX.sym("u", n_u)

    vx, vy, vz = x[3], x[4], x[5]
    ax, ay, az = u[0], u[1], u[2]
    f_expl = vertcat(vx, vy, vz, ax, ay, az)

    model.x = x
    model.u = u
    model.f_expl_expr = f_expl

    n_obst = obst.shape[0]
    pos = vertcat(*[x[i] for i in pos_indices])

    h_list = []
    for i in range(n_obst):
        obst_i = SX(obst[i])
        diff = pos[:2] - obst_i
        dist = np.sqrt(sumsqr(diff))
        h_list.append(dist)

    h_expr = vertcat(*h_list)
    model.con_h_expr = h_expr
    model.con_h_expr_e = h_expr

    return model, n_obst


def build_ocp(obst: np.ndarray, d_min: float, N: int = 20, Tf: float = 2.0) -> AcadosOcp:
    """Build an acados OCP for obstacle-avoidance NMPC.

    Args:
        obst: Array of obstacle positions.
        d_min: Minimum allowed distance to each obstacle.
        N: Prediction horizon length.
        Tf: Time horizon length.

    Returns:
        Configured AcadosOcp instance.
    """
    pos_indices = [0, 1, 2]
    model, n_obst = create_obstacle_avoidance_mpc(pos_indices, obst, d_min)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = N
    ocp.solver_options.tf = Tf

    n_x = model.x.shape[0]
    n_u = model.u.shape[0]
    n_y = n_x + n_u

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    Q = np.diag([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    R = np.diag([0.1, 0.1, 0.1])

    ocp.cost.W = np.block([[Q, np.zeros((n_x, n_u))], [np.zeros((n_u, n_x)), R]])
    ocp.cost.W_e = Q

    # Vx: (n_y x n_x),  Vu: (n_y x n_u)  — fixed shape
    ocp.cost.Vx = np.vstack([np.eye(n_x), np.zeros((n_u, n_x))])
    ocp.cost.Vu = np.vstack([np.zeros((n_x, n_u)), np.eye(n_u)])
    ocp.cost.Vx_e = np.eye(n_x)

    ocp.cost.yref = np.zeros(n_y)
    ocp.cost.yref_e = np.zeros(n_x)

    # ---------------------------------------------------------------
    # OBSTACLE AVOIDANCE INEQUALITY CONSTRAINTS
    #   d_min <= ||pos - obst_i||_2 <= +inf
    # ---------------------------------------------------------------
    ocp.constraints.lh = np.full(n_obst, d_min)
    ocp.constraints.uh = np.full(n_obst, 1e9)
    ocp.constraints.lh_e = np.full(n_obst, d_min)
    ocp.constraints.uh_e = np.full(n_obst, 1e9)

    ocp.constraints.lbx = np.array([-20.0, -20.0, -20.0, -5.0, -5.0, -5.0])
    ocp.constraints.ubx = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 5.0])
    ocp.constraints.idxbx = np.arange(n_x)

    ocp.constraints.lbu = np.array([-2.0, -2.0, -2.0])
    ocp.constraints.ubu = np.array([2.0, 2.0, 2.0])
    ocp.constraints.idxbu = np.arange(n_u)

    ocp.constraints.x0 = np.zeros(n_x)

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.nlp_solver_max_iter = 5000

    return ocp


def plot_results(
    x_traj: np.ndarray,
    u_traj: np.ndarray,
    obstacles: np.ndarray,
    d_min: float,
    x0: np.ndarray,
    x_ref: np.ndarray,
    Tf: float,
    obs_height: float = 5.0,
):
    """Three-panel plot: 3-D trajectory | velocity time-series | control inputs."""
    N = u_traj.shape[0]
    t = np.linspace(0, Tf, N + 1)
    tc = np.linspace(0, Tf, N)

    fig = plt.figure(figsize=(17, 5))
    fig.suptitle("NMPC Obstacle Avoidance", fontsize=14, fontweight="bold")

    # ── Panel 1: 3-D trajectory ──────────────────────────────────────────────
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.set_title("3-D Trajectory")

    # Draw obstacles as vertical cylinders
    theta = np.linspace(0, 2 * np.pi, 60)
    z_cyl = np.linspace(0, obs_height, 2)
    for i, obs in enumerate(obstacles):
        xc = obs[0] + d_min * np.cos(theta)
        yc = obs[1] + d_min * np.sin(theta)
        Theta, Z = np.meshgrid(theta, z_cyl)
        Xc = obs[0] + d_min * np.cos(Theta)
        Yc = obs[1] + d_min * np.sin(Theta)
        ax.plot_surface(Xc, Yc, Z, alpha=0.20, color="tab:red", zorder=2)
        # Top/bottom rings
        for z_ring in [0, obs_height]:
            ax.plot(xc, yc, z_ring, "r--", linewidth=1.0, alpha=0.7)
        ax.plot([obs[0]], [obs[1]], [obs_height / 2], "r+", markersize=8, markeredgewidth=2)
        ax.text(obs[0], obs[1], obs_height + 0.2, f"obs {i}", fontsize=7, color="tab:red")

    # Trajectory with colour gradient (early = light, late = dark)
    n_seg = len(x_traj) - 1
    cmap = plt.cm.Blues
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
        s=10,
        zorder=6,
        label="trajectory",
    )
    ax.scatter(*x0[:3], color="green", marker="s", s=80, label="start", zorder=7)
    ax.scatter(*x_ref[:3], color="green", marker="*", s=160, label="goal", zorder=7)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(fontsize=8)

    # ── Panel 2: velocities ──────────────────────────────────────────────────
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title("Velocity")
    ax.plot(t, x_traj[:, 3], label="vx", linewidth=1.8)
    ax.plot(t, x_traj[:, 4], label="vy", linewidth=1.8)
    ax.plot(t, x_traj[:, 5], label="vz", linewidth=1.8)
    speed = np.linalg.norm(x_traj[:, 3:6], axis=1)
    ax.plot(t, speed, "k--", label="‖v‖", linewidth=1.4)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("velocity [m/s]")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    # ── Panel 3: control inputs ──────────────────────────────────────────────
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("Control Inputs")
    ax.step(tc, u_traj[:, 0], where="post", label="ax", linewidth=1.8)
    ax.step(tc, u_traj[:, 1], where="post", label="ay", linewidth=1.8)
    ax.step(tc, u_traj[:, 2], where="post", label="az", linewidth=1.8)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("acceleration [m/s²]")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    obstacles = np.array([[2.0, 1.5], [4.0, 3.0], [4.0, 1.5], [4.0, 0.0], [6.0, 1.0], [7.0, 2.5]])
    d_min = 0.6
    N = 200
    Tf = 20.0

    ocp = build_ocp(obst=obstacles, d_min=d_min, N=N, Tf=Tf)
    solver = AcadosOcpSolver(ocp, json_file="ocp_obstacle.json")

    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_ref = np.array([8.0, 2.0, 0.0, 0.0, 0.0, 0.0])

    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)

    for k in range(N):
        solver.set(k, "yref", np.concatenate([x_ref, np.zeros(3)]))
    solver.set(N, "yref", x_ref)

    for k in range(N + 1):
        alpha = k / N
        x_init = (1 - alpha) * x0 + alpha * x_ref
        solver.set(k, "x", x_init)

    status = solver.solve()
    print(f"Solver status: {status}")  # 0 = success

    # Solver diagnostics
    status_meanings = {
        0: "SUCCESS",
        1: "NLP_ITERATION_MAXIMUM",
        2: "INFEASIBLE",
        3: "MINIMUM_STEP_SIZE",
        4: "QP_FAILURE",
        5: "READY",
    }

    print(f"  → {status_meanings.get(status, 'UNKNOWN')}")

    # Cost and residuals
    cost = solver.get_cost()
    print(f"\nObjective value:     {cost:.4f}")

    residuals = solver.get_residuals()  # [res_stat, res_eq, res_ineq, res_comp]
    print(f"Stationarity:        {residuals[0]:.2e}")
    print(f"Equality (dynamics): {residuals[1]:.2e}")
    print(f"Inequality:          {residuals[2]:.2e}")
    print(f"Complementarity:     {residuals[3]:.2e}")

    # Constraint violations per stage
    print("\n── Constraint violations per stage ──")
    print(f"{'stage':>6}  {'lam_lbx_min':>12}  {'lam_ubx_min':>12}  {'h_min_dist':>12}")
    for k in range(N + 1):
        x_k = solver.get(k, "x")

        # Distance to each obstacle (should be >= d_min)
        dists = np.linalg.norm(x_k[:2] - obstacles, axis=1)
        h_min = dists.min()

        if k < N:
            lam = solver.get(k, "lam")
            # lam layout: [lbx, ubx, lbu, ubu, lh, uh]  (acados ordering)
            n_lbx = ocp.constraints.idxbx.size
            n_lbu = ocp.constraints.idxbu.size
            lam_lbx = lam[:n_lbx]
            lam_ubx = lam[n_lbx : 2 * n_lbx]
            flag = " ← VIOLATION" if h_min < d_min - 1e-3 else ""
            print(f"{k:>6}  {lam_lbx.min():>12.3e}  {lam_ubx.min():>12.3e}  {h_min:>12.4f}{flag}")
        else:
            flag = " ← VIOLATION" if h_min < d_min - 1e-3 else ""
            print(f"{k:>6}  {'(terminal)':>12}  {'':>12}  {h_min:>12.4f}{flag}")

    # QP iteration count per SQP step (useful when max_iter is low)
    print(f"\nSQP iterations used: {solver.get_stats('sqp_iter')}")
    print(f"QP iterations:       {solver.get_stats('qp_iter')}")
    print(f"Time (total):        {solver.get_stats('time_tot'):.4f} s")
    print(f"Time (lin):          {solver.get_stats('time_lin'):.4f} s")
    print(f"Time (QP):           {solver.get_stats('time_qp'):.4f} s")

    print(f"Solver status: {status}")  # 0 = success

    x_traj = np.array([solver.get(k, "x") for k in range(N + 1)])
    u_traj = np.array([solver.get(k, "u") for k in range(N)])

    print("Position trajectory (px, py):")

    plot_results(x_traj, u_traj, obstacles, d_min, x0, x_ref, Tf)
