"""Interactive 2-candidate obstacle-detour debug plot with vector-PDF export.

Companion to the gate-candidate plot, for the obstacle branch of `_explore`.
The obstacle search forks two ways at each pillar (+push / -push); this draws
both resulting trajectories in a single 3D view zoomed on the pillar, highlights
the chosen side in colour and greys out the rejected one, keeps the winner's
detour waypoints, and saves a vector PDF at whatever camera angle you leave the
interactive window.

Workflow:
    - a popup opens with both candidate paths around the obstacle;
    - rotate to a good angle;
    - press 's' to snapshot the current angle to a numbered PDF (optional);
    - close the window -> the final angle is saved and the search continues.

Note: needs an interactive backend for the popup (e.g. TkAgg/QtAgg). Set it
before the first pyplot import:  import matplotlib; matplotlib.use("TkAgg").

Because `_explore` is recursive, this fires once per obstacle *decision node* --
so a track with several pillars (or deep branching) produces several popups. To
throttle, guard the call site, e.g. `if depth == 0:` for only the first/outermost
decision, or cap `_COUNTER` below.

Usage (in SplinePlanner._explore, replacing the final return):

    from lsy_drone_racing.control.Planner.plot.ObstacleCanidateplot import (
        plot_obstacle_candidates,
    )
    result_wps, result_clear = self._pick_better(
        wps_A, clear_A, wps_B, clear_B, t_elapsed, depth
    )
    winner = "A" if result_wps is wps_A else "B"
    plot_obstacle_candidates(
        self, wps_A, wps_B, winner, entry_obst_c, t_elapsed, depth,
        pOLL_array, getattr(self, "_pGLL_array", None),
        getattr(self, "_y_GBL_array", None),
    )
    return result_wps, result_clear
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

try:
    from lsy_drone_racing.control.Planner.planner import (
        FRAME_WIDTH, R_OBSTACLE, CLEARANCE,
    )
except Exception:  # pragma: no cover - fallback for standalone use
    FRAME_WIDTH, R_OBSTACLE, CLEARANCE = 0.72, 0.15, 0.10

_WIN_COLOR = "tab:green"
_LOSE_COLOR = "0.6"     # grey
_COUNTER = {"n": 0}     # makes filenames unique across recursion nodes


def _draw_cylinder(ax, c, radius, z0, z1, color, alpha):
    """Vertical pillar as a translucent surface strip."""
    th = np.linspace(0, 2 * np.pi, 40)
    T = np.vstack([th, th])
    Z = np.vstack([np.full_like(th, z0), np.full_like(th, z1)])
    X = c[0] + radius * np.cos(T)
    Y = c[1] + radius * np.sin(T)
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, shade=False)


def _draw_frame_3d(ax, c, yaw, half, color, lw=2.0, alpha=1.0):
    """Square gate frame in 3D (context only)."""
    w = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    zz = np.array([0.0, 0.0, 1.0])
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
    pts = np.array([np.asarray(c, float) + s * half * w + t * half * zz for s, t in corners])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=lw, alpha=alpha)


def plot_obstacle_candidates(
    planner,
    wps_A,
    wps_B,
    winner: str,
    obst_c,
    t_elapsed: float,
    depth: int,
    pOLL_array: np.ndarray,
    pGLL_array: np.ndarray | None = None,
    y_GBL_array: np.ndarray | None = None,
    save_dir: str = "obstacle_candidate_debug",
    pad: float = 1.0,
    interactive: bool = True,
) -> str:
    """Plot both obstacle-detour candidates for one pillar and save a PDF.

    Args:
        planner:      SplinePlanner (used for `_create_spline`).
        wps_A:        Waypoints of the +push branch.
        wps_B:        Waypoints of the -push branch.
        winner:       "A" or "B" -- which branch was chosen by `_pick_better`.
        obst_c:       2D (xy) center of the target obstacle.
        t_elapsed:    Current race time (for spline timing).
        depth:        Recursion depth (used in the title/filename).
        pOLL_array:   Obstacle centers (target + context pillars).
        pGLL_array:   Optional gate centers for faded context.
        y_GBL_array:  Optional gate yaws.
        save_dir:     Output directory for the PDF.
        pad:          Half-size [m] of the local view box around the pillar.
        interactive:  If True, open a popup and save the angle you close it at.

    Returns:
        Path to the saved PDF.
    """
    os.makedirs(save_dir, exist_ok=True)
    obst_c = np.asarray(obst_c, float)[:2]

    def sample(wps):
        try:
            wps = np.asarray(wps, float)
            sp, t_sample = planner._create_spline(wps, t_elapsed)
            td = np.linspace(0, t_sample[-1], len(t_sample) * 4)
            return sp(td), wps
        except Exception:
            return None, None

    ptsA, wpA = sample(wps_A)
    ptsB, wpB = sample(wps_B)

    win_is_A = winner == "A"
    win_pts, win_wps = (ptsA, wpA) if win_is_A else (ptsB, wpB)
    lose_pts = ptsB if win_is_A else ptsA
    win_label = "A (+push)" if win_is_A else "B (-push)"
    lose_label = "B (-push)" if win_is_A else "A (+push)"

    # Zoom z: the winner trajectory's height where it passes the pillar.
    z_center = 1.0
    if win_pts is not None:
        d = np.linalg.norm(win_pts[:, :2] - obst_c, axis=1)
        z_center = float(win_pts[int(np.argmin(d)), 2])
    z0, z1 = z_center - pad, z_center + pad

    def near(p) -> bool:
        return np.linalg.norm(np.asarray(p, float)[:2] - obst_c) < pad + 0.6

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # rejected side (grey) first, chosen side (green) on top
    if lose_pts is not None:
        ax.plot(lose_pts[:, 0], lose_pts[:, 1], lose_pts[:, 2], "-",
                color=_LOSE_COLOR, lw=1.6, zorder=3, label=f"rejected: {lose_label}")
    if win_pts is not None:
        ax.plot(win_pts[:, 0], win_pts[:, 1], win_pts[:, 2], "-",
                color=_WIN_COLOR, lw=2.6, zorder=6, label=f"chosen: {win_label}")

    # winner detour waypoints near the pillar
    if win_wps is not None:
        mask = np.array([near(w) for w in win_wps])
        wpn = win_wps[mask] if mask.any() else win_wps
        ax.scatter(wpn[:, 0], wpn[:, 1], wpn[:, 2], c="orange", marker="D", s=55,
                   edgecolor="k", depthshade=False, label="chosen waypoints")

    # target pillar: solid keep-out + clearance shell
    _draw_cylinder(ax, obst_c, R_OBSTACLE, z0, z1, "firebrick", 0.35)
    _draw_cylinder(ax, obst_c, R_OBSTACLE + CLEARANCE, z0, z1, "orange", 0.12)

    # nearby context pillars (faded)
    for o in np.asarray(pOLL_array, float).reshape(-1, 3):
        if np.allclose(o[:2], obst_c) or not near(o):
            continue
        _draw_cylinder(ax, o[:2], R_OBSTACLE, z0, z1, "firebrick", 0.15)

    # nearby context gates (faded)
    if pGLL_array is not None and y_GBL_array is not None:
        for gc, gy in zip(np.asarray(pGLL_array, float), np.asarray(y_GBL_array, float)):
            if not near(gc):
                continue
            _draw_frame_3d(ax, gc, gy, FRAME_WIDTH / 2, "tab:blue", lw=1.2, alpha=0.5)

    ax.set_xlim(obst_c[0] - pad, obst_c[0] + pad)
    ax.set_ylim(obst_c[1] - pad, obst_c[1] + pad)
    ax.set_zlim(z0, z1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(f"obstacle detour (depth {depth}) — chosen {win_label}")
    ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        f"Obstacle at ({obst_c[0]:.2f}, {obst_c[1]:.2f}) — "
        f"green = chosen, grey = rejected", fontsize=11)
    fig.tight_layout()

    _COUNTER["n"] += 1
    pdf_path = os.path.join(save_dir, f"obstacle_candidates_{_COUNTER['n']:03d}_d{depth}.pdf")

    if interactive:
        snap = {"k": 0}

        def _on_key(event):
            if event.key == "s":
                p = os.path.join(save_dir, f"obstacle_candidates_{_COUNTER['n']:03d}_d{depth}_snap{snap['k']}.pdf")
                fig.savefig(p)
                snap["k"] += 1
                print(f"[obstacle candidates] snapshot -> {p}")

        cid = fig.canvas.mpl_connect("key_press_event", _on_key)
        print("[obstacle candidates] rotate the 3D view; press 's' to snapshot an angle; "
              "close the window to save the final angle and continue.")
        plt.show(block=True)
        try:
            fig.canvas.mpl_disconnect(cid)
        except Exception:
            pass

    fig.savefig(pdf_path)
    print(f"[obstacle candidates] saved {pdf_path}")
    plt.close(fig)
    return pdf_path