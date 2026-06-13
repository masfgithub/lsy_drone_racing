"""
mpccpp_model.py
---------------
MPCC++ model: plain MPCC + the prismatic gate/track tunnel (eq.7) + soft
obstacle (stick) avoidance.

Dynamics and the least-squares cost residual are reused unchanged from
mpcc_model.py. Two constraint groups are exposed via con_h_expr:

  1. Tunnel (4 halfspaces, eq.7): keeps the drone inside the gate-aligned prism.
  2. Obstacles (n_obstacles entries): each a vertical stick with a keep-out
     radius r. The keep-out is a SQUARED-distance inequality in the (x,y) plane
        h_obs = (x - x_o)^2 + (y - y_o)^2 - r^2 >= 0
     i.e. the drone center must stay at least r away from the stick center.
     Squared distance (not ||.||) is used so the gradient is smooth everywhere
     and the SQP/Gauss-Newton linearization is well-behaved; r already folds in
     the stick radius + the drone's safety margin (e.g. r = 0.12 m).
     z is ignored -> the stick is treated as infinitely tall (a pole).

Both groups are softened in the controller via acados slacks (independent
penalty weights), so the OCP stays feasible. Stage 0 is left unconstrained.

Parameter vector p (length num_params(cfg)):
    [ pd(3), td(3), pdd(3), theta_bar(1), qc(1), mu(1),   # 0..11  MPCC cost
      n(3), b(3), W(1), H(1),                             # 12..19 tunnel
      x_o0,y_o0,r0,  x_o1,y_o1,r1, ... ]                  # 20..    obstacles
Obstacle params are the SAME at every horizon node (static stick within a
solve) and are refreshed online each control step by the controller.
"""

from dataclasses import dataclass
import casadi as ca
from acados_template import AcadosModel

from lsy_drone_racing.control.mpcc_test.mpcc_model import (
    MPCCConfig, quadrotor_dynamics, mpcc_residual, NX, NU, IDX_P,
)

# ---- parameter / constraint layout ----------------------------------------
N_TUNNEL = 4                 # tunnel halfspaces
NP_BASE_TUNNEL = 20          # 12 cost params + 8 tunnel params (always reserved)
NRM = slice(12, 15)          # tunnel frame n
BNM = slice(15, 18)          # tunnel frame b
WIDX = 18                    # tunnel half-width
HIDX = 19                    # tunnel half-height
OBST_START = 20              # first obstacle param index
OBST_DIM = 3                 # [x, y, r] per obstacle


def num_params(cfg) -> int:
    n = NP_BASE_TUNNEL
    if getattr(cfg, "use_obstacles", False):
        n += OBST_DIM * cfg.n_obstacles
    return n


def num_h(cfg) -> int:
    n = N_TUNNEL if cfg.use_tunnel else 0
    if getattr(cfg, "use_obstacles", False):
        n += cfg.n_obstacles
    return n


@dataclass
class MPCCppConfig(MPCCConfig):
    # tunnel
    use_tunnel: bool = True
    tunnel_soft: bool = True
    tunnel_slack_lin: float = 1e3
    tunnel_slack_quad: float = 1e3
    # obstacles (sticks)
    use_obstacles: bool = True
    n_obstacles: int = 1                 # max number of sticks baked into the OCP
    obstacle_soft: bool = True
    obstacle_slack_lin: float = 1e4      # higher than the tunnel -> near-hard keep-out
    obstacle_slack_quad: float = 1e4


def tunnel_constraint(x, p):
    """Four halfspaces of the MPCC++ prism, eq.(7); all >= 0 when inside."""
    pos = x[IDX_P]
    pd = p[0:3]; n = p[NRM]; b = p[BNM]; W = p[WIDX]; H = p[HIDX]
    d = pos - pd
    dn = ca.dot(d, n)
    db = ca.dot(d, b)
    return ca.vertcat(W + dn, W - dn, H + db, H - db)


def obstacle_constraint(x, p, n_obstacles):
    """One squared-distance keep-out per stick; all >= 0 when clear."""
    px = x[0]; py = x[1]
    h = []
    for i in range(n_obstacles):
        xo = p[OBST_START + OBST_DIM * i + 0]
        yo = p[OBST_START + OBST_DIM * i + 1]
        ro = p[OBST_START + OBST_DIM * i + 2]
        h.append((px - xo) ** 2 + (py - yo) ** 2 - ro ** 2)
    return ca.vertcat(*h)


def export_mpccpp_model(cfg: MPCCppConfig) -> AcadosModel:
    npar = num_params(cfg)
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    p = ca.SX.sym("p", npar)
    xdot = ca.SX.sym("xdot", NX)

    model = AcadosModel()
    model.name = "quadrotor_mpccpp"
    model.x = x
    model.u = u
    model.p = p
    model.xdot = xdot
    model.f_expl_expr = quadrotor_dynamics(x, u, cfg)
    model.f_impl_expr = xdot - model.f_expl_expr
    model.cost_y_expr = mpcc_residual(x, u, p, cfg, terminal=False)
    model.cost_y_expr_e = mpcc_residual(x, u, p, cfg, terminal=True)

    # constraint stack: tunnel first, then obstacles (order matters for slacks)
    h_parts = []
    if cfg.use_tunnel:
        h_parts.append(tunnel_constraint(x, p))
    if cfg.use_obstacles:
        h_parts.append(obstacle_constraint(x, p, cfg.n_obstacles))
    if h_parts:
        h = ca.vertcat(*h_parts)
        model.con_h_expr = h            # stages 1 .. N-1
        model.con_h_expr_e = h          # terminal stage N (stage 0 left free)
    return model
