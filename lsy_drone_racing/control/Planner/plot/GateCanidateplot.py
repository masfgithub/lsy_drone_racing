"""Interactive 3-candidate gate-detour debug plot with vector-PDF export.

Replaces the per-branch `_plot_gate_branch` calls: instead of three separate
figures, this draws all three candidate trajectories (Left / Right / Top) for a
single gate violation in one figure, highlights the chosen one in colour and
greys out the other two, keeps the winner's waypoints and a local view of the
environment around the gate, and saves a vector PDF at whatever camera angle you
leave the interactive window.

Workflow:
    - a popup window opens with all three candidates;
    - rotate the 3D panel to a good angle;
    - press 's' to snapshot the current angle to a numbered PDF (optional);
    - close the window -> the final angle is saved to the main PDF and the
      planner continues to the next gate.

Note: for the popup you need an interactive backend (e.g. TkAgg/QtAgg). If you
import this after something already selected 'Agg', set the backend first:
    import matplotlib; matplotlib.use("TkAgg")

Usage (from SplinePlanner._avoid_gates_tree, right after the winner is picked):

    from lsy_drone_racing.control.Planner.plot.gate_candidate_plot import (
        plot_gate_candidates,
    )
    plot_gate_candidates(
        self,
        {"Left": branch_L, "Right": branch_R, "Top": branch_T},
        winner_name, gate_c, gate_yaw,
        pGLL_array, y_GBL_array, pOLL_array,
        t_elapsed, outer_iter,
    )
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

# Frame / obstacle sizes -- taken from the planner, with a fallback so the module
# also runs on its own for testing.
try:
    from lsy_drone_racing.control.Planner.planner import (
        FRAME_WIDTH, FRAME_OPENING, R_OBSTACLE, CLEARANCE,
    )
except Exception:  # pragma: no cover - fallback for standalone use
    FRAME_WIDTH, FRAME_OPENING, R_OBSTACLE, CLEARANCE = 0.72, 0.10, 0.15, 0.10

_ORDER = ("Left", "Right", "Top")
_WIN_COLOR = "tab:green"
_LOSE_COLOR = "0.6"   # grey


def _draw_frame_3d(ax, c, yaw, half, color, lw=2.0, alpha=1.0):
    """Square gate frame in 3D (side dir = gate width, up = world z)."""
    w = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    zz = np.array([0.0, 0.0, 1.0])
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
    pts = np.array([np.asarray(c, float) + s * half * w + t * half * zz for s, t in corners])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=lw, alpha=alpha)


def _draw_frame_top(ax, c, yaw, half, color, lw=2.0, alpha=1.0):
    """Gate frame as a line across the width in the top-down (xy) view."""
    w = np.array([-np.sin(yaw), np.cos(yaw)])
    a = np.asarray(c, float)[:2] - half * w
    b = np.asarray(c, float)[:2] + half * w
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, alpha=alpha)


def plot_gate_candidates(
    planner,
    branches: dict,
    winner_name: str,
    gate_c,
    gate_yaw: float,
    pGLL_array: np.ndarray,
    y_GBL_array: np.ndarray,
    pOLL_array: np.ndarray,
    t_elapsed: float,
    outer_iter: int,
    save_dir: str = "gate_candidate_debug",
    pad: float = 1.0,
    interactive: bool = True,
) -> str:
    """Plot the three gate-detour candidates for one violation and save a PDF.

    Args:
        planner:      The SplinePlanner (used for `_create_spline`).
        branches:     {"Left": branch_L, "Right": branch_R, "Top": branch_T},
                      each a dict from `_evaluate_gate_branch` (needs 'wps').
        winner_name:  Which branch was chosen ("Left"/"Right"/"Top").
        gate_c:       3D center of the target gate.
        gate_yaw:     Yaw of the target gate.
        pGLL_array:   All gate centers (for faded context gates).
        y_GBL_array:  All gate yaws.
        pOLL_array:   Obstacle centers.
        t_elapsed:    Current race time (for spline timing).
        outer_iter:   Gate-tree iteration index (used in the filename).
        save_dir:     Output directory for the PDF.
        pad:          Half-size [m] of the local view box around the gate.
        interactive:  If True, open a popup, save the angle you close it at.

    Returns:
        Path to the saved PDF.
    """
    os.makedirs(save_dir, exist_ok=True)
    gate_c = np.asarray(gate_c, float)

    # --- sample each candidate trajectory ------------------------------------
    trajs = {}
    for name in _ORDER:
        br = branches.get(name)
        if br is None:
            continue
        try:
            wps = np.asarray(br["wps"], float)
            spline, t_sample = planner._create_spline(wps, t_elapsed)
            t_dense = np.linspace(0, t_sample[-1], len(t_sample) * 4)
            trajs[name] = (spline(t_dense), wps)
        except Exception:
            trajs[name] = None

    winner_wps = None
    if trajs.get(winner_name) is not None:
        winner_wps = trajs[winner_name][1]

    def near(p) -> bool:
        return np.linalg.norm(np.asarray(p, float)[:2] - gate_c[:2]) < pad + 0.6

    # --- figure --------------------------------------------------------------
    fig = plt.figure(figsize=(15, 7))
    ax3d = fig.add_subplot(121, projection="3d")
    axtop = fig.add_subplot(122)

    # trajectories: losers (grey) first, winner (colour) on top
    for name in _ORDER:
        if trajs.get(name) is None:
            continue
        pts = trajs[name][0]
        win = name == winner_name
        col, lw, z = (_WIN_COLOR, 2.6, 6) if win else (_LOSE_COLOR, 1.4, 3)
        lbl = f"{name} (chosen)" if win else name
        ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-", color=col, lw=lw, zorder=z, label=lbl)
        axtop.plot(pts[:, 0], pts[:, 1], "-", color=col, lw=lw, zorder=z, label=lbl)

    # winner waypoints (numbered), only the ones near the gate
    if winner_wps is not None:
        mask = np.array([near(w) for w in winner_wps])
        wp_near = winner_wps[mask] if mask.any() else winner_wps
        ax3d.scatter(wp_near[:, 0], wp_near[:, 1], wp_near[:, 2], c="orange", marker="D",
                     s=55, edgecolor="k", depthshade=False,
                     label=f"chosen waypoints ({len(winner_wps)})")
        axtop.scatter(wp_near[:, 0], wp_near[:, 1], c="orange", marker="D", s=60,
                      edgecolor="k", zorder=7)
        for i, w in enumerate(winner_wps):
            if near(w):
                ax3d.text(w[0], w[1], w[2] + 0.03, str(i), fontsize=8)
                axtop.annotate(str(i), (w[0], w[1]), textcoords="offset points",
                               xytext=(5, 5), fontsize=8)

    # target gate frame: outer + opening
    _draw_frame_3d(ax3d, gate_c, gate_yaw, FRAME_WIDTH / 2, "tab:blue", lw=2.5)
    _draw_frame_3d(ax3d, gate_c, gate_yaw, FRAME_OPENING / 2, "tab:cyan", lw=1.5)
    _draw_frame_top(axtop, gate_c, gate_yaw, FRAME_WIDTH / 2, "tab:blue", lw=3)
    _draw_frame_top(axtop, gate_c, gate_yaw, FRAME_OPENING / 2, "tab:cyan", lw=4)
    axtop.scatter([gate_c[0]], [gate_c[1]], c="blue", s=70, marker="o", edgecolor="k", zorder=8)

    # nearby context gates (faded)
    for gc, gy in zip(np.asarray(pGLL_array, float), np.asarray(y_GBL_array, float)):
        if np.allclose(gc, gate_c) or not near(gc):
            continue
        _draw_frame_3d(ax3d, gc, gy, FRAME_WIDTH / 2, "gray", lw=1.0, alpha=0.4)
        _draw_frame_top(axtop, gc, gy, FRAME_WIDTH / 2, "gray", lw=1.0, alpha=0.4)

    # nearby obstacles: keep-out shells (top-down) + light pillar rings (3D)
    th = np.linspace(0, 2 * np.pi, 40)
    for o in np.asarray(pOLL_array, float).reshape(-1, 3):
        if not near(o):
            continue
        axtop.add_patch(plt.Circle((o[0], o[1]), R_OBSTACLE + CLEARANCE, color="orange", alpha=0.20))
        axtop.add_patch(plt.Circle((o[0], o[1]), R_OBSTACLE, color="firebrick", alpha=0.55))
        for zz in (gate_c[2] - pad, gate_c[2] + pad):
            ax3d.plot(o[0] + R_OBSTACLE * np.cos(th), o[1] + R_OBSTACLE * np.sin(th),
                      np.full_like(th, zz), color="firebrick", lw=1.0, alpha=0.6)
        ax3d.plot([o[0], o[0]], [o[1], o[1]], [gate_c[2] - pad, gate_c[2] + pad],
                  color="firebrick", lw=1.2, alpha=0.6)

    # local zoom around the gate
    ax3d.set_xlim(gate_c[0] - pad, gate_c[0] + pad)
    ax3d.set_ylim(gate_c[1] - pad, gate_c[1] + pad)
    ax3d.set_zlim(gate_c[2] - pad, gate_c[2] + pad)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.set_title(f"3-way gate detour — iter {outer_iter}  (chosen: {winner_name})")
    ax3d.legend(loc="upper left", fontsize=8)

    axtop.set_xlim(gate_c[0] - pad, gate_c[0] + pad)
    axtop.set_ylim(gate_c[1] - pad, gate_c[1] + pad)
    axtop.set_aspect("equal"); axtop.grid(alpha=0.3)
    axtop.set_xlabel("x [m]"); axtop.set_ylabel("y [m]")
    axtop.set_title("top-down")
    axtop.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"Gate at ({gate_c[0]:.2f}, {gate_c[1]:.2f}, {gate_c[2]:.2f}) — "
        f"chosen '{winner_name}', grey = rejected candidates", fontsize=12)
    fig.tight_layout()

    pdf_path = os.path.join(save_dir, f"gate_candidates_iter{outer_iter:02d}.pdf")

    if interactive:
        snap = {"k": 0}

        def _on_key(event):
            if event.key == "s":
                p = os.path.join(save_dir, f"gate_candidates_iter{outer_iter:02d}_snap{snap['k']}.pdf")
                fig.savefig(p)
                snap["k"] += 1
                print(f"[gate candidates] snapshot -> {p}")

        cid = fig.canvas.mpl_connect("key_press_event", _on_key)
        print("[gate candidates] rotate the 3D view; press 's' to snapshot an angle; "
              "close the window to save the final angle and continue.")
        plt.show(block=True)          # blocks until you close the window
        try:
            fig.canvas.mpl_disconnect(cid)
        except Exception:
            pass

    fig.savefig(pdf_path)             # vector PDF at the current (closed) angle
    print(f"[gate candidates] saved {pdf_path}")
    plt.close(fig)
    return pdf_path