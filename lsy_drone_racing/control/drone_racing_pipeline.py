"""This module implements the pipeline for the drone racing.

TBD specify more in detail.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_capsule, draw_line, draw_points

from lsy_drone_racing.control.basic_planner import BasicPlanner
from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.control.env_obs import extract_env_states
from lsy_drone_racing.control.mpcc.mpcc import MPCC
from lsy_drone_racing.control.mpcc.mpccpp import MPCCpp
from lsy_drone_racing.control.nmpc.nmpc import NMPC

# Active controller: "mpccpp" | "mpcc" | "nmpc"
CONTROLLER_TYPE = "mpccpp"

# Active planner: "Smart" | "Simple"
PLANNER_TYPE = "Simple"

if PLANNER_TYPE == "Smart":
    from lsy_drone_racing.control.Planner.smart_planner import SplinePlanner
elif PLANNER_TYPE == "Simple":
    from lsy_drone_racing.control.Planner.SplinePlanner_2 import SplinePlanner

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

    from lsy_drone_racing.control.env_obs import EnvState_t


def _draw_wedge_gate(
    sim: Sim,
    position: NDArray,
    quaternion: NDArray,
    total_length: float,
    total_height: float,
    hole_width: float,
    hole_height: float,
    thickness: float = 0.05,
    rgba: NDArray | None = None,
    radius: float = 0.015,
):
    """Draw the four wedge prisms of a WedgeWindow gate as capsule edges.

    Each wedge has 6 vertices (4 base corners + 2 tip corners) and 9 edges.
    The edges are drawn as capsules, matching the WedgeWindow geometry exactly.

    Gate-local frame: x = gate normal, y = width, z = height.

        Left / Right:  base at y = ±hl  (x∈[-a_x,+a_x], z∈[-hh,+hh])
                       tip  at y = ±hw  (x=0, z∈[-hho,+hho])
        Top  / Bottom: base at z = ±hh  (x∈[-a_x,+a_x], y∈[-hl,+hl])
                       tip  at z = ±hho (x=0, y∈[-hw,+hw])
    """
    if sim.viewer is None:
        return

    if rgba is None:
        rgba = np.array([0.0, 0.5, 1.0, 1.0])

    qw, qx, qy, qz = quaternion / np.linalg.norm(quaternion)
    R_mat = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )

    def to_world(lv: np.ndarray) -> np.ndarray:
        return np.asarray(position) + R_mat @ lv

    def edge(a: np.ndarray, b: np.ndarray):
        draw_capsule(sim, to_world(a), to_world(b), radius=radius, rgba=rgba, cylinder=True)

    a_x = thickness / 2.0
    hl = total_length / 2.0
    hh = total_height / 2.0
    hw = hole_width / 2.0
    hho = hole_height / 2.0

    def draw_prism(
        base_d: float,
        tip_d: float,
        depth_idx: int,
        ax_idx: int,
        perp_idx: int,
        h_perp_base: float,
        h_perp_tip: float,
    ) -> None:
        """Draw one wedge prism as 9 capsule edges.

        depth_idx  : axis of tapering (1=y for L/R, 2=z for T/B)
        ax_idx     : gate normal axis (always 0 = x)
        perp_idx   : remaining axis   (2=z for L/R, 1=y for T/B)
        base_d     : depth coordinate at base
        tip_d      : depth coordinate at tip
        h_perp_base: perp half-extent at base
        h_perp_tip : perp half-extent at tip
        """

        def pt(d: float, xv: float, perpv: float) -> np.ndarray:
            v = np.zeros(3)
            v[depth_idx] = d
            v[ax_idx] = xv
            v[perp_idx] = perpv
            return v

        # 4 base corners
        B0 = pt(base_d, -a_x, h_perp_base)
        B1 = pt(base_d, a_x, h_perp_base)
        B2 = pt(base_d, a_x, -h_perp_base)
        B3 = pt(base_d, -a_x, -h_perp_base)
        # 2 tip corners
        T0 = pt(tip_d, 0.0, h_perp_tip)
        T1 = pt(tip_d, 0.0, -h_perp_tip)

        # Base rectangle (4 edges)
        edge(B0, B1)
        edge(B1, B2)
        edge(B2, B3)
        edge(B3, B0)
        # Slant edges base → tip (4 edges)
        edge(B0, T0)
        edge(B1, T0)
        edge(B2, T1)
        edge(B3, T1)
        # Tip edge (1 edge)
        edge(T0, T1)

    # Left:   depth=y(1), tip toward +y; base at -hl, tip at -hw
    draw_prism(-hl, -hw, depth_idx=1, ax_idx=0, perp_idx=2, h_perp_base=hh, h_perp_tip=hho)
    # Right:  base at +hl, tip at +hw
    draw_prism(hl, hw, depth_idx=1, ax_idx=0, perp_idx=2, h_perp_base=hh, h_perp_tip=hho)
    # Top:    depth=z(2); base at +hh, tip at +hho
    draw_prism(hh, hho, depth_idx=2, ax_idx=0, perp_idx=1, h_perp_base=hl, h_perp_tip=hw)
    # Bottom: base at -hh, tip at -hho
    draw_prism(-hh, -hho, depth_idx=2, ax_idx=0, perp_idx=1, h_perp_base=hl, h_perp_tip=hw)


def _draw_mpccpp_tunnel(
    sim: Sim,
    controller: object,
    ring_rgba: NDArray | None = None,
    corner_rgba: NDArray | None = None,
):
    """Draw the MPCC++ prediction tunnel.

    Draws the rectangular cross-section at every predicted horizon node
    (4 corners + 4 edges) plus the longitudinal rails.

    Cross-section at progress theta: centre ref.eval(theta), spanned by the
    tunnel frame (n, b) = ref.frame(theta) with half-extents (W, H) =
    ref.width(theta) -- exactly the prism enforced by the tunnel constraint.
    """
    if sim.viewer is None:
        return
    ref = getattr(controller, "_ref", None)
    if ref is None:
        return

    # per-node progress: solved theta state (index 14), aligned with the
    # predicted trajectory; fall back to the controller's theta_pred guess.
    if getattr(controller, "_x_warm", None):
        thetas = [float(x[14]) for x in controller._x_warm]
    elif getattr(controller, "_theta_pred", None) is not None:
        thetas = [float(t) for t in controller._theta_pred]
    else:
        return

    if ring_rgba is None:
        ring_rgba = np.array([0.1, 0.85, 0.95, 0.6])  # cyan edges
    if corner_rgba is None:
        corner_rgba = np.array([1.0, 0.9, 0.0, 0.9])  # yellow corners

    # corners[k] = 4x3, order (+n+b, -n+b, -n-b, +n-b)
    corners = []
    for th in thetas:
        pd = np.asarray(ref.eval(th), dtype=float)
        n, b = ref.frame(th)
        W, H = ref.width(th)
        n = np.asarray(n, float)
        b = np.asarray(b, float)
        corners.append(
            np.array(
                [pd + W * n + H * b, pd - W * n + H * b, pd - W * n - H * b, pd + W * n - H * b]
            )
        )
    corners = np.array(corners)  # (K, 4, 3)

    # cross-section rectangle (4 edges) at every prediction node
    for k in range(len(corners)):
        ring = np.vstack([corners[k], corners[k, 0]])  # closed (5,3)
        draw_line(sim, ring, rgba=ring_rgba)

    # longitudinal rails connecting node k -> k+1 along each corner
    for cidx in range(4):
        draw_line(sim, corners[:, cidx, :], rgba=ring_rgba)

    # corner markers
    # draw_points(sim, corners.reshape(-1, 3), rgba=corner_rgba, size=0.02)


def _draw_cylinder_obstacle(
    sim: Sim, position: NDArray, height: float, radius: float, rgba: NDArray | None = None
):
    """Draw a vertical cylinder obstacle as a single capsule in the simulation."""
    if sim.viewer is None:
        return

    if rgba is None:
        rgba = np.array([1.0, 0.2, 0.2, 0.8])

    pos = np.asarray(position)
    z_base = float(pos[2]) if len(pos) == 3 else 0.0
    cx, cy = float(pos[0]), float(pos[1])
    draw_capsule(
        sim,
        np.array([cx, cy, z_base]),
        np.array([cx, cy, z_base + height]),
        radius=radius,
        rgba=rgba,
        cylinder=True,
    )


def _draw_post(
    sim: Sim,
    gate_position: NDArray,
    r_post: float,
    hole_height: float,
    margin: float,
    z_floor: float = 0.0,
    rgba: NDArray | None = None,
):
    """Draw the gate-post keep-out as a vertical capsule below the opening.

    Mirrors post_penalty_sym: a capsule of radius r_post around the segment
    from the floor up to z_top = gate_z - hole_height/2 - margin - r_post.
    The rounded top therefore reaches gate_z - hole_height/2 - margin, just
    below the opening, exactly like the penalty's keep-out.
    """
    if sim.viewer is None:
        return
    if rgba is None:
        rgba = np.array([1.0, 0.6, 0.0, 0.5])  # orange: distinct from blue gates / red obstacles

    cx, cy, cz = float(gate_position[0]), float(gate_position[1]), float(gate_position[2])
    z_top = cz - hole_height / 2.0 - margin - r_post
    draw_capsule(
        sim,
        np.array([cx, cy, z_floor]),
        np.array([cx, cy, z_top]),
        radius=r_post,
        rgba=rgba,
        cylinder=False,  # rounded caps -> matches distance-to-segment keep-out
    )


def _draw_tunnel_centerline(sim: Sim, ref: object, n: int = 150, rgba: NDArray | None = None):
    """Draw the MPCC++ tunnel centerline as a sequence of line segments."""
    if sim.viewer is None:
        return
    if rgba is None:
        rgba = np.array([1.0, 0.8, 0.0, 0.8])
    s_vals = np.linspace(0.0, ref.length, n)
    pts = np.array([ref.eval(float(s)) for s in s_vals])
    draw_line(sim, pts, rgba=rgba)


def plot_tube_width_profile(
    ref: object,
    save_path: str = "tube_width_profile.png",
    n: int = 1500,
    floor: float = 0.05,
    show_activation_panel: bool = True,
) -> str | None:
    """Dump a PNG of the MPCC++ tunnel cross-section size along the path.

    Use this to debug tube shrink/expand: it shows the W(theta)/H(theta) the
    solver actually sees, plus a diagnostic of the per-gate Gaussian bumps.

    Top panel:    W(theta), H(theta) straight from ref.width(theta), with the
                  nominal size, the per-gate target (ref.gate_hw/hh) and the
                  width() floor drawn for reference, and a vertical marker at
                  every gate arc-length.
    Bottom panel: the per-gate Gaussian activations exp(-1/2 (d/sigma)^2), shown
                  as BOTH their sum (what the current width() subtracts) and
                  their max (nearest-gate). sum >> max, or sum > 1, means the
                  gate bumps overlap and the tube is being over-pinched.

    Args:
        ref:                   MPCC++ TunnelReferencePath (e.g. controller._ref).
        save_path:             Output PNG path (relative paths land in CWD).
        n:                     Number of theta samples.
        floor:                 The clamp value used inside width() (drawn for ref).
        show_activation_panel: Add the Gaussian-activation diagnostic subplot.

    Returns:
        The saved path, or None if plotting was not possible.
    """
    if ref is None:
        return None
    try:
        import sys

        import matplotlib

        if "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")  # headless: crazyflow viewer is not matplotlib
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[tube-profile] matplotlib unavailable: {exc}")
        return None

    L = float(ref.length)
    thetas = np.linspace(0.0, L, n)
    WH = np.array([ref.width(float(t)) for t in thetas])
    W, H = WH[:, 0], WH[:, 1]

    gate_s = np.atleast_1d(np.asarray(getattr(ref, "gate_s", []), dtype=float))
    W_nom = float(getattr(ref, "W_nom", np.nan))
    gate_hw = np.atleast_1d(np.asarray(getattr(ref, "gate_hw", []), dtype=float))
    sigma = float(getattr(ref, "tunnel_sigma", np.nan))
    closed = bool(getattr(ref, "closed", False))

    two = show_activation_panel and gate_s.size > 0 and np.isfinite(sigma)
    npan = 2 if two else 1
    fig, axs = plt.subplots(npan, 1, figsize=(11, 3.7 * npan), squeeze=False)

    # ---- top panel: the cross-section the solver sees ----------------------
    ax = axs[0, 0]
    top = max(W.max(), H.max(), W_nom if np.isfinite(W_nom) else 0.0) * 1.18 + 1e-3
    ax.set_ylim(0, top)
    ax.plot(thetas, W, color="tab:blue", lw=2.0, label=r"$W(\theta)$ half-width")
    if not np.allclose(H, W):
        ax.plot(thetas, H, color="tab:purple", lw=1.8, label=r"$H(\theta)$ half-height")
    if np.isfinite(W_nom):
        ax.axhline(W_nom, color="0.45", ls="--", lw=1.1)
        ax.text(
            L * 0.01, W_nom, f" W_nom={W_nom:.3g}", color="0.35", fontsize=9, va="bottom", ha="left"
        )
    if gate_hw.size:
        gt = float(np.min(gate_hw))
        ax.axhline(gt, color="tab:red", ls=":", lw=1.3)
        ax.text(
            L * 0.01,
            gt,
            f" gate target={gt:.3g}",
            color="tab:red",
            fontsize=9,
            va="bottom",
            ha="left",
        )
    ax.axhline(floor, color="k", ls="-.", lw=0.8)
    ax.text(
        L * 0.55,
        floor,
        f" width() floor={floor:.3g}",
        color="k",
        fontsize=8,
        va="bottom",
        ha="left",
    )
    for k, gs in enumerate(gate_s):
        ax.axvline(gs, color="tab:red", lw=1.0, alpha=0.5)
        ax.text(gs, top * 0.04, f"G{k + 1}", color="tab:red", ha="center", fontsize=8)
    ax.set_xlim(0, L)
    ax.set_xlabel(r"progress $\theta$ [m]")
    ax.set_ylabel("tunnel half-size [m]")
    ax.set_title(
        f"MPCC++ tube cross-section vs progress   "
        f"(sigma={sigma:.3g} m, L={L:.2f} m, {gate_s.size} gates)",
        fontsize=10,
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    # ---- bottom panel: Gaussian activation diagnostic ----------------------
    if two:
        ax2 = axs[1, 0]
        d = thetas[:, None] - gate_s[None, :]
        if closed:
            d = (d + L / 2.0) % L - L / 2.0
        g = np.exp(-0.5 * (d / sigma) ** 2)  # (n, M)
        g_sum, g_max = g.sum(axis=1), g.max(axis=1)
        ax2.plot(
            thetas,
            g_sum,
            color="tab:red",
            lw=2.0,
            label=r"$\sum_j g_j$  (current width() subtracts this)",
        )
        ax2.plot(thetas, g_max, color="tab:green", lw=2.0, label=r"$\max_j g_j$  (nearest-gate)")
        ax2.axhline(1.0, color="0.4", ls="--", lw=1.0)
        ax2.text(L * 0.01, 1.0, " over-pinch threshold = 1", color="0.35", fontsize=9, va="bottom")
        for gs in gate_s:
            ax2.axvline(gs, color="tab:red", lw=1.0, alpha=0.5)
        ax2.set_xlim(0, L)
        ax2.set_ylim(0, max(1.1, float(g_sum.max()) * 1.1))
        ax2.set_xlabel(r"progress $\theta$ [m]")
        ax2.set_ylabel("Gaussian activation")
        ax2.set_title(
            "Per-gate bump activation: sum >> max (or sum > 1) "
            "means bumps overlap and the tube is over-pinched",
            fontsize=10,
        )
        ax2.grid(alpha=0.25)
        ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[tube-profile] saved {save_path}")
    return save_path


class DroneRacingPipeline(Controller):
    """Pipeline for drone racing: planner + controller, supporting MPCC++/MPCC/NMPC."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the pipeline."""
        super().__init__(obs, info, config)

        t_total = 12
        env_states = extract_env_states(obs)
        self._tick = 0
        self._freq = config.env.freq
        self._finished = False
        self._force_replan = False

        if CONTROLLER_TYPE == "mpccpp":
            # Online adaptive planner: re-roots its trajectory at the current
            # drone position and is replanned when a gate/obstacle moves. MPCC++
            # consumes only the path geometry (its own v_theta sets the speed).
            # The planner is time-parameterized and crashes if t_elapsed >= its
            # horizon (t_remaining <= 0), so keep that horizon comfortably above
            # the real flight time. Since MPCC++ ignores the timing, a large
            # value only makes the sampled path denser -- the geometry is identical.
            self._planner = SplinePlanner(env_states, info, config, t_total)
            trajectory = self._planner.plan(env_states, 0.0)
            self._controller = MPCCpp(env_states, trajectory, info, config, t_total)
            # Replan trigger bookkeeping.
            self.nominal_gates_position = env_states.pTLL_array
            self.nominal_obstacles_position = env_states.pOLL_array
            self._t_replan = 0.0

        elif CONTROLLER_TYPE == "mpcc":
            self._planner = BasicPlanner(config, t_total)
            planner_dict = self._planner.plan()
            self._controller = MPCC(env_states, planner_dict, info, config, t_total)

        elif CONTROLLER_TYPE == "nmpc":
            self._planner = BasicPlanner(config, t_total)
            planner_dict = self._planner.plan()
            self._controller = NMPC(env_states, planner_dict, info, config, t_total, use_soft=True)

        else:
            raise ValueError(
                f"Unknown CONTROLLER_TYPE '{CONTROLLER_TYPE}'. Choose 'mpccpp', 'mpcc', or 'nmpc'."
            )

        # Debug: dump the MPCC++ tube cross-section profile to a PNG on startup.
        # if CONTROLLER_TYPE == "mpccpp" and hasattr(self._controller, "_ref"):
        #    try:
        #        plot_tube_width_profile(self._controller._ref, "tube_width_profile.png")
        #    except Exception as exc:  # noqa: BLE001 -- never let debug plotting break a run
        #        print(f"[pipeline] tube profile plot failed: {exc}")

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone."""
        env_states = extract_env_states(obs)
        if CONTROLLER_TYPE == "mpccpp":
            # Replan when a gate/obstacle has moved (or after an episode reset).
            # The planner re-roots at the current drone position; the controller
            # then rebuilds its tube and resets theta to 0 (drone = new start).
            if self._force_replan or self._get_replan_reason(env_states):
                self.nominal_gates_position = env_states.pTLL_array
                self.nominal_obstacles_position = env_states.pOLL_array
                trajectory = self._planner.plan(env_states, self._tick / self._freq)
                self._t_replan = self._tick / self._freq
                self._controller.replan_reference(trajectory, env_states)
                self._force_replan = False
                print("[MPCC++] Replanned -> tube + theta reset")
            return self._controller.control(env_states, info)

        # MPCC / NMPC: original flow.
        self._planner.replan()
        return self._controller.control(env_states, info)

    def _get_replan_reason(self, env_states: EnvState_t) -> bool:
        """True if the active gate or any obstacle moved more than 1 cm."""
        idx = int(getattr(env_states, "pTLL_index", 0))
        if np.linalg.norm(self.nominal_gates_position[idx] - env_states.pTLL_array[idx]) > 0.01:
            return True
        n = min(len(self.nominal_obstacles_position), len(env_states.pOLL_array))
        for i in range(n):
            if np.linalg.norm(self.nominal_obstacles_position[i] - env_states.pOLL_array[i]) > 0.01:
                return True
        return False

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1
        self._controller.set_tick(self._tick)
        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0
        self._controller.reset()
        self._controller.set_tick(self._tick)
        # Force a fresh replan (re-root + tube rebuild) on the next control step.
        self._force_replan = True
        self._t_replan = 0.0

    def dump_tube_profile(self, save_path: str = "tube_width_profile.png") -> str | None:
        """Regenerate the MPCC++ tube cross-section profile PNG on demand.

        Call this any time after construction (e.g. from a debugger or
        episode_callback) to inspect the current tube shrink/expand behaviour.
        Returns the saved path, or None if it could not be produced.
        """
        ref = getattr(self._controller, "_ref", None)
        return plot_tube_width_profile(ref, save_path)

    def render_callback(self, sim: Sim):
        """Visualize the planned path, MPC predictions, gates, and obstacles."""
        # Planned path (green)
        trajectory = self._planner.get_pos_traj()
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

        # Tunnel centerline for MPCC++ (yellow)
        # if CONTROLLER_TYPE == "mpccpp":
        #    _draw_tunnel_centerline(sim, self._controller._ref)

        # MPC predicted trajectory (purple dots)
        pred_trajectory = self._controller.get_predicted_traj()
        for p in pred_trajectory:
            draw_points(sim, p.reshape(1, -1), rgba=(0.58, 0.0, 0.83, 0.5), size=0.01)

        # MPCC++ prediction tunnel (cyan edges + yellow corners)
        # if CONTROLLER_TYPE == "mpccpp" and hasattr(self._controller, "_ref"):
        #    _draw_mpccpp_tunnel(sim, self._controller)
        # Reference trajectory (red dots)
        # ref_trajectory = self._controller.get_ref_traj()
        # for p in ref_trajectory:
        #    draw_points(sim, p.reshape(1, -1), rgba=(1.0, 0.0, 0.0, 0.5), size=0.01)

        # Gates
        for gate in self._controller._gates:
            _draw_wedge_gate(
                sim,
                position=gate.position,
                quaternion=gate.quaternion,
                total_length=gate.total_length,
                total_height=gate.total_height,
                hole_width=gate.hole_width,
                hole_height=gate.hole_height,
                thickness=gate.thickness,
                rgba=np.array([0.0, 0.5, 1.0, 1.0]),
            )

        # Obstacles
        for obs in self._controller._obstacles:
            _draw_cylinder_obstacle(
                sim,
                position=obs.position,
                height=obs.total_height,
                radius=obs.d_min,
                rgba=np.array([1.0, 0.2, 0.2, 0.7]),
            )
