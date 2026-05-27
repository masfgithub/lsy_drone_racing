"""Environment constraints as soft penalties in the cost function.

Soft constraints via quadratic penalty added to the cost:
    penalty = w * max(0, margin - sdf)^2

No con_h_expr is set — the problem is always feasible.
No QP failures from infeasible warm-starts.

Usage in nmpc_soft_setup.py:
    from env_soft_constraints import create_soft_env_constraints, set_env_params

    env = create_soft_env_constraints(
        model           = ocp.model,
        pBLL            = pBLL,
        gates           = gates,
        obstacles       = obstacles,
        gate_weight     = 1000.0,
        obstacle_weight = 1000.0,
    )
    # Then attach env["penalty_expr"] to your cost expression.
"""

import numpy as np
from casadi import MX, fmax

try:
    from obstacle import CylinderObstacle
    from window import Window
except ImportError:
    from lsy_drone_racing.control.nmpc.obstacle import CylinderObstacle
    from lsy_drone_racing.control.nmpc.window import Window


def get_obstacle_objects(
    positions: np.ndarray, obstacles_information: dict
) -> list[CylinderObstacle]:
    """Construct CylinderObstacle objects from position array."""
    return [
        CylinderObstacle(position=positions[i], obstacles_information=obstacles_information)
        for i in range(len(positions))
    ]


def get_gate_objects(
    positions: np.ndarray, quaternions: np.ndarray, gates_information: dict
) -> list[Window]:
    """Construct Window objects from pose arrays and a shared info dict."""
    gates = []
    for pos, quat in zip(positions, quaternions):
        gates.append(
            Window(
                position=pos,
                quaternion=quat,
                total_length=gates_information["total_length"],
                total_height=gates_information["total_height"],
                hole_width=gates_information["hole_width"],
                hole_height=gates_information["hole_height"],
                thickness=gates_information["thickness"],
                margin=gates_information["margin"],
            )
        )
    return gates


def build_param_vector(gates: list[Window], obstacles: list[CylinderObstacle]) -> np.ndarray:
    """Concatenate all gate and obstacle parameter vectors."""
    parts = [g.param_vector() for g in gates] + [o.param_vector() for o in obstacles]
    return np.concatenate(parts) if parts else np.array([])


def create_soft_env_constraints(
    model: object,
    pBLL: MX,
    gates: list[Window],
    obstacles: list[CylinderObstacle] | None = None,
    gate_weight: float = 1000.0,
    obstacle_weight: float = 1000.0,
) -> dict:
    """Attach soft environment constraints to an AcadosModel as runtime parameters.

    Builds the parameter symbol, attaches it to model.p, and returns a penalty
    expression to be added to the stage and terminal cost. No con_h_expr is set.

    Args:
        model:           AcadosModel — must not yet have model.p set.
        pBLL:            CasADi MX of shape (3,) — position symbol from model.x.
        gates:           List of Window objects.
        obstacles:       List of CylinderObstacle objects (may be None).
        gate_weight:     Quadratic penalty weight for gate violations.
        obstacle_weight: Quadratic penalty weight for obstacle violations.

    Returns:
        Dict with keys:
            "gates"          – list[Window]
            "obstacles"      – list[CylinderObstacle]
            "p"              – flat CasADi MX parameter symbol
            "p0"             – np.ndarray of initial parameter values
            "n_gates"        – int
            "n_obs"          – int
            "penalty_expr"   – scalar MX to add to stage cost
            "penalty_expr_e" – scalar MX to add to terminal cost (identical)
    """
    if obstacles is None:
        obstacles = []

    n_gates = len(gates)
    n_obs = len(obstacles)
    n_p = Window.N_PARAMS * n_gates + CylinderObstacle.N_PARAMS * n_obs

    p = MX.sym("p", n_p)
    model.p = p

    penalty = MX(0)
    offset = 0

    # ── Gate soft penalties ───────────────────────────────────────────────────
    for gate in gates:
        p_win = p[offset : offset + Window.N_PARAMS]
        sdf = Window.casadi_constraints_sym(pBLL, p_win)
        violation = fmax(MX(0), MX(gate.margin) - sdf)
        penalty = penalty + gate_weight * violation * violation
        offset += Window.N_PARAMS

    # ── Obstacle soft penalties ───────────────────────────────────────────────
    for obs in obstacles:
        p_obs = p[offset : offset + CylinderObstacle.N_PARAMS]
        sdf = CylinderObstacle.casadi_constraint_sym(pBLL, p_obs)
        violation = fmax(MX(0), MX(obs.d_min) - sdf)
        penalty = penalty + obstacle_weight * violation * violation
        offset += CylinderObstacle.N_PARAMS

    p0 = build_param_vector(gates, obstacles)

    return {
        "gates": gates,
        "obstacles": obstacles,
        "p": p,
        "p0": p0,
        "n_gates": n_gates,
        "n_obs": n_obs,
        "penalty_expr": penalty,
        "penalty_expr_e": penalty,
    }


def set_env_params(solver: object, gates: list[Window], obstacles: list[CylinderObstacle], N: int):
    """Push current gate and obstacle parameters to every shooting node."""
    p_vec = build_param_vector(gates, obstacles)
    for k in range(N + 1):
        solver.set(k, "p", p_vec)


def verify_env_constraints(
    x_traj: np.ndarray, gates: list[Window], obstacles: list[CylinderObstacle]
) -> bool:
    """Post-solve geometric verification for all gates and obstacles."""
    all_ok = True
    for g_idx, gate in enumerate(gates):
        clean = gate.verify(x_traj)
        print(f"  gate {g_idx}: {'OK' if clean else 'VIOLATIONS FOUND'}")
        all_ok = all_ok and clean
    for o_idx, obs in enumerate(obstacles):
        clean = obs.verify(x_traj)
        print(f"  obstacle {o_idx}: {'OK' if clean else 'VIOLATIONS FOUND'}")
        all_ok = all_ok and clean
    return all_ok
