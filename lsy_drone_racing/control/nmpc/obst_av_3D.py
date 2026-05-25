"""Obstacle-avoidance (3D) NMPC example utilities."""

import matplotlib.pyplot as plt
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, fabs, fmax, sumsqr, vertcat


def create_obstacle_avoidance_mpc(
    pos_indices: list, obst: np.ndarray, d_min: float, wall: dict | None = None
) -> tuple:
    """Build the acados model with cylinder and optional wall constraints."""
    model = AcadosModel()
    model.name = "obstacle_avoidance_example"

    n_x = 6
    n_u = 3
    x = MX.sym("x", n_x)
    u = MX.sym("u", n_u)

    vx, vy, vz = x[3], x[4], x[5]
    ax, ay, az = u[0], u[1], u[2]
    model.x = x
    model.u = u
    model.f_expl_expr = vertcat(vx, vy, vz, ax, ay, az)

    n_obst = obst.shape[0]
    pos = vertcat(*[x[i] for i in pos_indices])
    h_list = []

    # ── Cylinder constraints (XY plane) ──────────────────────────────────────
    for i in range(n_obst):
        diff = pos[:2] - MX(obst[i, :2])
        h_list.append(np.sqrt(sumsqr(diff)))

    # ── Wall signed-distance constraint ──────────────────────────────────────
    # Signed distance to each axis-aligned face (negative inside the box).
    # Outside: sdf_ext = ||max(s,0)||  (standard exterior SDF, >= 0)
    # Inside:  sdf_int = max(sx,sy,sz) (least-negative face, < 0)
    # Unified: fmax(sdf_ext - sqrt(eps), sdf_int)
    #   → positive and smooth outside
    #   → negative with non-zero gradient inside (points toward nearest face)
    # This allows a feasible-but-minimal warm-start to work without biasing
    # the solver toward any particular avoidance direction.
    if wall is not None:
        cx, cy, cz = wall["center"]
        bx, by, bz = wall["half_extents"]
        eps = 1e-4

        sx = fabs(pos[0] - cx) - bx
        sy = fabs(pos[1] - cy) - by
        sz = fabs(pos[2] - cz) - bz

        sdf_ext = np.sqrt(sumsqr(vertcat(fmax(sx, 0.0), fmax(sy, 0.0), fmax(sz, 0.0))) + eps)
        sdf_int = fmax(fmax(sx, sy), sz)
        sdf = fmax(sdf_ext - np.sqrt(eps), sdf_int)

        h_list.append(sdf)

    h_expr = vertcat(*h_list)
    model.con_h_expr = h_expr
    model.con_h_expr_e = h_expr

    return model, n_obst, len(h_list)


def build_ocp(
    obst: np.ndarray,
    d_min: float,
    wall: dict | None = None,
    wall_margin: float = 0.05,
    N: int = 20,
    Tf: float = 2.0,
) -> AcadosOcp:
    """Assemble and return the AcadosOcp for the obstacle avoidance problem."""
    pos_indices = [0, 1, 2]
    model, n_obst, n_h = create_obstacle_avoidance_mpc(pos_indices, obst, d_min, wall=wall)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = N
    ocp.solver_options.tf = Tf

    n_x = model.x.shape[0]
    n_u = model.u.shape[0]
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

    lh = np.array([d_min] * n_obst + [wall_margin] * (n_h - n_obst))
    uh = np.full(n_h, 1e9)
    ocp.constraints.lh = lh
    ocp.constraints.uh = uh
    ocp.constraints.lh_e = lh
    ocp.constraints.uh_e = uh

    ocp.constraints.lbx = np.array([-20.0, -20.0, -20.0, -5.0, -5.0, -5.0])
    ocp.constraints.ubx = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 5.0])
    ocp.constraints.idxbx = np.arange(n_x)
    ocp.constraints.lbu = np.array([-2.0, -2.0, -2.0])
    ocp.constraints.ubu = np.array([2.0, 2.0, 2.0])
    ocp.constraints.idxbu = np.arange(n_u)
    ocp.constraints.x0 = np.zeros(n_x)

    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_iter_max = 200
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_tol_stat = 1e-3
    ocp.solver_options.nlp_solver_tol_eq = 1e-3
    ocp.solver_options.nlp_solver_tol_ineq = 1e-3
    ocp.solver_options.nlp_solver_tol_comp = 1e-3

    return ocp


def warm_start(x0: np.ndarray, x_ref: np.ndarray, wall: dict, N: int) -> np.ndarray:
    """Straight line in x/y, lifted just enough in z to clear the wall top.

    Feasible from the start (no node inside the solid) but minimally biased —
    the lift is only as high as needed to clear the wall, leaving the SQP
    free to move nodes sideways or lower to find the true optimum.
    """
    cz = wall["center"][2]
    bz = wall["half_extents"][2]
    z_clear = cz + bz + 0.1  # 0.1 m above wall top

    traj = np.zeros((N + 1, len(x0)))
    for k in range(N + 1):
        beta = k / N
        px = (1.0 - beta) * x0[0] + beta * x_ref[0]
        py = (1.0 - beta) * x0[1] + beta * x_ref[1]
        pz_straight = (1.0 - beta) * x0[2] + beta * x_ref[2]
        pz = max(pz_straight, z_clear)
        traj[k] = np.array([px, py, pz, 0.0, 0.0, 0.0])
    return traj


def _draw_wall(ax: object, wall: dict, color: str = "saddlebrown", alpha: float = 0.35):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cx, cy, cz = wall["center"]
    bx, by, bz = wall["half_extents"]

    xs = [cx - bx, cx + bx]
    ys = [cy - by, cy + by]
    zs = [cz - bz, cz + bz]

    for x_val in xs:
        Y, Z = np.meshgrid(ys, zs)
        ax.plot_surface(np.full_like(Y, x_val), Y, Z, color=color, alpha=alpha, zorder=3)
    for y_val in ys:
        X, Z = np.meshgrid(xs, zs)
        ax.plot_surface(X, np.full_like(X, y_val), Z, color=color, alpha=alpha, zorder=3)
    for z_val in zs:
        X, Y = np.meshgrid(xs, ys)
        ax.plot_surface(X, Y, np.full_like(X, z_val), color=color, alpha=alpha, zorder=3)

    verts = [
        [
            (cx - bx, cy - by, cz - bz),
            (cx + bx, cy - by, cz - bz),
            (cx + bx, cy + by, cz - bz),
            (cx - bx, cy + by, cz - bz),
        ],
        [
            (cx - bx, cy - by, cz + bz),
            (cx + bx, cy - by, cz + bz),
            (cx + bx, cy + by, cz + bz),
            (cx - bx, cy + by, cz + bz),
        ],
        [
            (cx - bx, cy - by, cz - bz),
            (cx + bx, cy - by, cz - bz),
            (cx + bx, cy - by, cz + bz),
            (cx - bx, cy - by, cz + bz),
        ],
        [
            (cx - bx, cy + by, cz - bz),
            (cx + bx, cy + by, cz - bz),
            (cx + bx, cy + by, cz + bz),
            (cx - bx, cy + by, cz + bz),
        ],
        [
            (cx - bx, cy - by, cz - bz),
            (cx - bx, cy + by, cz - bz),
            (cx - bx, cy + by, cz + bz),
            (cx - bx, cy - by, cz + bz),
        ],
        [
            (cx + bx, cy - by, cz - bz),
            (cx + bx, cy + by, cz - bz),
            (cx + bx, cy + by, cz + bz),
            (cx + bx, cy - by, cz + bz),
        ],
    ]
    ax.add_collection3d(
        Poly3DCollection(verts, edgecolor="sienna", linewidth=0.8, facecolor=(0, 0, 0, 0))
    )


def plot_results(
    x_traj: np.ndarray,
    obstacles: np.ndarray,
    d_min: float,
    x0: np.ndarray,
    x_ref: np.ndarray,
    obs_height: float = 5.0,
    wall: dict | None = None,
):
    """Plot the optimised 3-D trajectory together with obstacles and an optional wall."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    fig.suptitle("NMPC Obstacle Avoidance — 3-D Trajectory", fontsize=13, fontweight="bold")

    theta = np.linspace(0, 2 * np.pi, 60)
    for i, obs in enumerate(obstacles):
        Theta, Z = np.meshgrid(theta, np.linspace(0, obs_height, 2))
        ax.plot_surface(
            obs[0] + d_min * np.cos(Theta),
            obs[1] + d_min * np.sin(Theta),
            Z,
            alpha=0.18,
            color="tab:red",
            zorder=2,
        )
        for z_ring in [0, obs_height]:
            ax.plot(
                obs[0] + d_min * np.cos(theta),
                obs[1] + d_min * np.sin(theta),
                z_ring,
                "r--",
                linewidth=0.9,
                alpha=0.6,
            )
        ax.text(obs[0], obs[1], obs_height + 0.15, f"obs {i}", fontsize=7, color="tab:red")

    if wall is not None:
        _draw_wall(ax, wall)

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
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    obstacles = np.empty((0, 3))
    d_min = 0.6
    N = 200
    Tf = 20.0

    length = 1.5
    height = 3.0
    thickness = 0.3
    wall = {
        "center": [5.0, 1.5, height / 2],
        "half_extents": [thickness / 2, length / 2, height / 2],
    }

    ocp = build_ocp(obst=obstacles, d_min=d_min, wall=wall, wall_margin=0.05, N=N, Tf=Tf)
    solver = AcadosOcpSolver(ocp, json_file="ocp_obstacle.json")

    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_ref = np.array([8.0, 2.0, 2.0, 0.0, 0.0, 0.0])

    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)
    for k in range(N):
        solver.set(k, "yref", np.concatenate([x_ref, np.zeros(3)]))
    solver.set(N, "yref", x_ref)

    x_init = warm_start(x0, x_ref, wall, N)
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
    print(f"QP iterations:       {solver.get_stats('qp_iter')}")
    print(f"Time (total):        {solver.get_stats('time_tot'):.4f} s")
    print(f"Time (lin):          {solver.get_stats('time_lin'):.4f} s")
    print(f"Time (QP):           {solver.get_stats('time_qp'):.4f} s")

    x_traj = np.array([solver.get(k, "x") for k in range(N + 1)])

    plot_results(x_traj, obstacles, d_min, x0, x_ref, wall=wall)
