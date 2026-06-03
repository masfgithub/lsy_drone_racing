"""Environment constraints as soft penalties in the cost function.

Gates are modelled as four wedge-shaped prisms (WedgeWindow).
Obstacles remain circular cylinders (CylinderObstacle).

Soft constraints via quadratic penalty — no con_h_expr is set so the
problem is always feasible.
"""

import numpy as np
from casadi import MX, fmax

try:
    from obstacle import CylinderObstacle
    from wedge_window import WedgeWindow
except ImportError:
    from lsy_drone_racing.control.nmpc.obstacle import CylinderObstacle
    from lsy_drone_racing.control.nmpc.wedge_window import WedgeWindow


def get_obstacle_objects(
    positions: np.ndarray, obstacles_information: dict
) -> list[CylinderObstacle]:
    """Construct CylinderObstacle objects from a position array."""
    return [
        CylinderObstacle(position=positions[i], obstacles_information=obstacles_information)
        for i in range(len(positions))
    ]


def get_gate_objects(
    positions: np.ndarray, quaternions: np.ndarray, gates_information: dict
) -> list[WedgeWindow]:
    """Construct WedgeWindow objects from pose arrays and a shared info dict.

    Args:
        positions:         Array of shape (n, 3).
        quaternions:       Array of shape (n, 4) — [qw, qx, qy, qz].
        gates_information: Dict with keys: total_length, total_height,
                           hole_width, hole_height, thickness, margin.

    Returns:
        List of n WedgeWindow objects.
    """
    return [
        WedgeWindow(
            position=pos,
            quaternion=quat,
            total_length=gates_information["total_length"],
            total_height=gates_information["total_height"],
            hole_width=gates_information["hole_width"],
            hole_height=gates_information["hole_height"],
            thickness=gates_information["thickness"],
            margin=gates_information["margin"],
        )
        for pos, quat in zip(positions, quaternions)
    ]


def build_param_vector(gates: list[WedgeWindow], obstacles: list[CylinderObstacle]) -> np.ndarray:
    """Concatenate all gate and obstacle parameter vectors into one flat array."""
    parts = [g.param_vector() for g in gates] + [o.param_vector() for o in obstacles]
    return np.concatenate(parts) if parts else np.array([])


def create_soft_env_constraints(
    model: object,
    pBLL: MX,
    gates: list[WedgeWindow],
    obstacles: list[CylinderObstacle] | None = None,
    gate_weight: float = 1000.0,
    obstacle_weight: float = 1000.0,
) -> dict:
    """Attach soft environment constraints to an AcadosModel as runtime parameters.

    Gate penalty
    ~~~~~~~~~~~~
    ``WedgeWindow.casadi_penalty_sym`` returns the unweighted sum of
    quadratic wedge penetration penalties for all four bars, multiplied by
    ``gate_weight``.

    Obstacle penalty
    ~~~~~~~~~~~~~~~~
    ``CylinderObstacle.casadi_constraint_sym`` returns the XY distance.
    Penalty fires when that distance drops below ``d_min``.

    Args:
        model:           AcadosModel — must not yet have model.p set.
        pBLL:            CasADi MX (3,) — position symbol from model.x.
        gates:           List of WedgeWindow objects.
        obstacles:       List of CylinderObstacle objects (may be None).
        gate_weight:     Quadratic penalty weight for gate violations.
        obstacle_weight: Quadratic penalty weight for obstacle violations.

    Returns:
        Dict with keys: gates, obstacles, p, p0, n_gates, n_obs,
                        penalty_expr, penalty_expr_e.
    """
    if obstacles is None:
        obstacles = []

    n_gates = len(gates)
    n_obs = len(obstacles)
    n_p = WedgeWindow.N_PARAMS * n_gates + CylinderObstacle.N_PARAMS * n_obs

    p = MX.sym("p", n_p)
    model.p = p

    penalty = MX(0)
    offset = 0

    for gate in gates:
        p_win = p[offset : offset + WedgeWindow.N_PARAMS]
        penalty = penalty + gate_weight * WedgeWindow.casadi_penalty_sym(pBLL, p_win)
        offset += WedgeWindow.N_PARAMS

    for obs in obstacles:
        p_obs = p[offset : offset + CylinderObstacle.N_PARAMS]
        dist = CylinderObstacle.casadi_constraint_sym(pBLL, p_obs)
        viol = fmax(MX(0), MX(obs.d_min) - dist)
        penalty = penalty + obstacle_weight * viol * viol
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


def set_env_params(
    solver: object, gates: list[WedgeWindow], obstacles: list[CylinderObstacle], N: int
):
    """Push current gate and obstacle parameters to every shooting node."""
    p_vec = build_param_vector(gates, obstacles)
    for k in range(N + 1):
        solver.set(k, "p", p_vec)


def verify_env_constraints(
    x_traj: np.ndarray, gates: list[WedgeWindow], obstacles: list[CylinderObstacle]
) -> bool:
    """Geometric verification for all gates and obstacles after solving."""
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
