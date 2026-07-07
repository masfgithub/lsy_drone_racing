"""Environment constraints — gates (Windows) and cylinder obstacles.

Provides create_env_constraints() which attaches all obstacle and gate
SDF constraints to an existing AcadosModel as runtime parameters, and
returns everything needed to set parameters and verify solutions.

Usage in MPC_main.py:
    from env_constraints import create_env_constraints, set_env_params

    env = create_env_constraints(
        model         = model,
        p_bll       = p_bll,
        gates         = gates,
        obstacles     = obstacles,
    )

    # In the OCP build, after env = create_env_constraints(...):
    ocp.constraints.lh   = env["lh"]
    ocp.constraints.uh   = env["uh"]
    ocp.constraints.lh_e = env["lh"]
    ocp.constraints.uh_e = env["uh"]
    ocp.parameter_values = env["p0"]

    # At runtime, after updating gate/obstacle positions:
    set_env_params(solver, env["gates"], env["obstacles"], N)
"""

import numpy as np
from casadi import MX, vertcat
from lsy_drone_racing.envs.environment_constraints.window import Window

from lsy_drone_racing.envs.environment_constraints.obstacle import CylinderObstacle


def get_obstacle_objects(
    positions: np.ndarray, obstacles_information: dict
) -> list[CylinderObstacle]:
    """Construct a list of CylinderObstacle objects from an array of positions.

    Args:
        positions: Array of shape (n, 2) or (n, 3) — obstacle XY centres
                   (z is ignored for infinite-height cylinders).
        obstacles_information: Dict with keys:
                               d_min, total_height.

    Returns:
        List of n CylinderObstacle objects.
    """
    n = len(positions)

    return [
        CylinderObstacle(position=positions[i], obstacles_information=obstacles_information)
        for i in range(n)
    ]


def get_gate_objects(
    positions: np.ndarray, quaternions: np.ndarray, gates_information: dict
) -> list[Window]:
    """Construct a list of Window objects from arrays of poses.

    Args:
        positions:         Array of shape (n, 3) — gate centre positions.
        quaternions:       Array of shape (n, 4) — gate orientations [qw,qx,qy,qz].
        gates_information: Dict with keys:
                               total_length, total_height,
                               hole_width,   hole_height,
                               thickness,    margin.

    Returns:
        List of n Window objects.
    """
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
    """Concatenate all gate and obstacle parameter vectors into one flat array.

    Layout: [gate_0 params | gate_1 params | ... | obs_0 params | obs_1 params | ...]
    """
    parts = [g.param_vector() for g in gates] + [o.param_vector() for o in obstacles]
    return np.concatenate(parts) if parts else np.array([])


def create_env_constraints(
    model: object, p_bll: MX, gates: list[Window], obstacles: list[CylinderObstacle] | None = None
) -> dict:
    """Attach environment constraints to an AcadosModel as runtime parameters.

    Builds the full CasADi parameter symbol, attaches it to model.p,
    and returns a dict with everything the OCP builder and runtime updater need.

    Args:
        model:     AcadosModel — must not yet have model.p set.
        p_bll:   CasADi MX of shape (3,) — the position symbol from model.x.
        gates:     List of Window objects.
        obstacles: List of CylinderObstacle objects (may be empty or None).

    Returns:
        Dict with keys:
            "gates"     – list[Window]
            "obstacles" – list[CylinderObstacle]
            "p"         – flat CasADi MX parameter symbol
            "h_expr"    – vertcat of all constraint expressions
            "lh"        – np.ndarray of lower bounds (one per constraint)
            "uh"        – np.ndarray of upper bounds (all 1e9)
            "p0"        – np.ndarray of initial parameter values
            "n_gates"   – int
            "n_obs"     – int
    """
    if obstacles is None:
        obstacles = []

    n_gates = len(gates)
    n_obs = len(obstacles)
    n_p = Window.N_PARAMS * n_gates + CylinderObstacle.N_PARAMS * n_obs

    p = MX.sym("p", n_p)
    model.p = p

    h_list = []
    lh_list = []

    # ── Gate constraints (one signed-distance per gate) ───────────────────────
    offset = 0
    for gate in gates:
        p_win = p[offset : offset + Window.N_PARAMS]
        h_list.append(Window.casadi_constraints_sym(p_bll, p_win))
        lh_list.append(gate.margin)
        offset += Window.N_PARAMS

    # ── Cylinder obstacle constraints ─────────────────────────────────────────
    for obs in obstacles:
        p_obs = p[offset : offset + CylinderObstacle.N_PARAMS]
        h_list.append(CylinderObstacle.casadi_constraint_sym(p_bll, p_obs))
        lh_list.append(obs.d_min)
        offset += CylinderObstacle.N_PARAMS

    h_expr = vertcat(*h_list) if h_list else MX()

    model.con_h_expr = h_expr
    model.con_h_expr_e = h_expr

    lh = np.array(lh_list)
    uh = np.full(len(lh_list), 1e9)
    p0 = build_param_vector(gates, obstacles)

    return {
        "gates": gates,
        "obstacles": obstacles,
        "p": p,
        "h_expr": h_expr,
        "lh": lh,
        "uh": uh,
        "p0": p0,
        "n_gates": n_gates,
        "n_obs": n_obs,
    }


def set_env_params(solver: object, gates: list[Window], obstacles: list[CylinderObstacle], N: int):
    """Push current gate and obstacle parameters to every shooting node.

    Call this after any gate.update() or obstacle.update() call,
    before solver.solve().

    Args:
        solver:    AcadosOcpSolver instance.
        gates:     List of Window objects (current geometry).
        obstacles: List of CylinderObstacle objects (current geometry).
        N:         Number of shooting nodes.
    """
    p_vec = build_param_vector(gates, obstacles)
    for k in range(N + 1):
        solver.set(k, "p", p_vec)


def verify_env_constraints(
    x_traj: np.ndarray, gates: list[Window], obstacles: list[CylinderObstacle]
) -> bool:
    """Run post-solve verification for all gates and obstacles.

    Args:
        x_traj:    Array of shape (N+1, n_x).
        gates:     List of Window objects.
        obstacles: List of CylinderObstacle objects.

    Returns:
        True if all constraints satisfied, False otherwise.
    """
    all_ok = True

    for g_idx, gate in enumerate(gates):
        clean = gate.verify(x_traj)
        if not clean:
            print(f"  gate {g_idx}: VIOLATIONS FOUND")
            all_ok = False
        else:
            print(f"  gate {g_idx}: OK")

    for o_idx, obs in enumerate(obstacles):
        clean = obs.verify(x_traj)
        if not clean:
            print(f"  obstacle {o_idx}: VIOLATIONS FOUND")
            all_ok = False
        else:
            print(f"  obstacle {o_idx}: OK")

    return all_ok
