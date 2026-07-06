"""Interactive 3-candidate gate-detour debug plot (top-down) with vector-PDF export.

Companion to the obstacle-candidate plot, for the gate branch of the planner.
For a single gate violation the search forks three ways (Left / Right / Top);
this shows, in a TOP-DOWN (x-y) view:

  * all three candidate trajectories -- chosen in colour, rejected in grey;
  * the default gate waypoints (grey circles, shared by every branch) and each
    branch's own detour waypoints (diamonds: chosen yellow, rejected grey);
  * obstacle-collision hits along each branch (red x);
  * the target gate frame (outer + opening) and nearby context gates;
  * nearby obstacle keep-out shells.

Saves a vector PDF at whatever pan/zoom you leave the interactive window.

Workflow: a popup opens; pan/zoom to frame it; press 's' to snapshot the current
view to a numbered PDF (optional); close the window -> the current view is saved
and the planner continues to the next gate.

Note: needs an interactive backend for the popup (e.g. TkAgg/QtAgg); set it
before the first pyplot import.

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
_LOSE_COLOR = "0.55"   # grey


def _draw_frame_top(ax, c, yaw, half, color, lw=2.0, alpha=1.0):
    """Gate frame as a line across the width in the top-down (x-y) view."""
    w = np.array([-np.sin(yaw), np.cos(yaw)])
    a = np.asarray(c, float)[:2] - half * w
    b = np.asarray(c, float)[:2] + half * w
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, alpha=alpha)


def _shared_and_own(branch_wps, tol=1e-6):
    """Split branch waypoints into shared-by-all (default) vs. branch-specific.

    Args:
        branch_wps: {name: wps_array_or_None} for each candidate branch.

    Returns:
        (shared_xyz, {name: own_xyz}) -- 'shared' points appear in every branch
        (the default gate-centre waypoints); each 'own' set is that branch's
        newly inserted detour waypoints.
    """
    names = [n for n, W in branch_wps.items() if W is not None and len(W)]
    if not names:
        return np.empty((0, 3)), {}

    def has(W, p):
        W = np.asarray(W, float)
        return len(W) > 0 and np.min(np.linalg.norm(W - p, axis=1)) < tol

    ref = np.asarray(branch_wps[names[0]], float)
    shared = np.array([p for p in ref if all(has(branch_wps[n], p) for n in names)])
    if not len(shared):
        shared = np.empty((0, 3))

    own = {}
    for n in names:
        W = np.asarray(branch_wps[n], float)
        pts = [p for p in W if not has(shared, p)]
        own[n] = np.array(pts) if pts else np.empty((0, 3))
    return shared, own


def _obstacle_violations(pts, pOLL_array):
    """xy of dense samples that lie inside an obstacle (a branch hitting one)."""
    if pts is None or pOLL_array is None:
        return np.empty((0, 2))
    obs = np.asarray(pOLL_array, float).reshape(-1, 3)
    if not len(obs):
        return np.empty((0, 2))
    hits = [p[:2] for p in pts
            if np.any(np.linalg.norm(obs[:, :2] - p[:2], axis=1) < R_OBSTACLE)]
    return np.array(hits) if hits else np.empty((0, 2))


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
    pad: float = 0.9,     # half-size [m] of the top-down view box around the gate
    interactive: bool = True,
) -> str:
    """Top-down plot of the three gate-detour candidates for one violation; save a PDF.

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
        interactive:  If True, open a popup, save the view you close it at.

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

    # split waypoints into shared-by-all (default gate-centre) vs. each branch's
    # own newly inserted detour waypoints
    shared, own = _shared_and_own(
        {n: (trajs[n][1] if trajs.get(n) is not None else None) for n in _ORDER}
    )

    def near(p) -> bool:
        return np.linalg.norm(np.asarray(p, float)[:2] - gate_c[:2]) < pad + 0.6

    # --- figure --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))

    # trajectories: losers (grey) first, winner (colour) on top
    for name in _ORDER:
        if trajs.get(name) is None:
            continue
        pts = trajs[name][0]
        win = name == winner_name
        col, lw, z = (_WIN_COLOR, 2.4, 5) if win else (_LOSE_COLOR, 1.7, 3)
        lbl = f"{name} (chosen)" if win else f"{name} (rejected)"
        ax.plot(pts[:, 0], pts[:, 1], "-", color=col, lw=lw, zorder=z, label=lbl)

    # default gate-centre waypoints, shared by all branches (grey circles)
    if len(shared):
        sh = shared[np.array([near(w) for w in shared])]
        if len(sh):
            ax.scatter(sh[:, 0], sh[:, 1], s=45, c="0.75", marker="o",
                       zorder=4, label="gate waypoints")

    # rejected branches' own detour waypoints (grey diamonds)
    lose_wp_labeled = False
    for name in _ORDER:
        if name == winner_name or name not in own or not len(own[name]):
            continue
        W = own[name][np.array([near(w) for w in own[name]])]
        if not len(W):
            continue
        ax.scatter(W[:, 0], W[:, 1], s=55, marker="D",
                   facecolor=_LOSE_COLOR, edgecolor="k", zorder=9,
                   label=None if lose_wp_labeled else "rejected detour wp")
        lose_wp_labeled = True

    # chosen branch's own detour waypoints (yellow/orange diamonds)
    if winner_name in own and len(own[winner_name]):
        W = own[winner_name][np.array([near(w) for w in own[winner_name]])]
        if len(W):
            ax.scatter(W[:, 0], W[:, 1], s=60, marker="D",
                       facecolor="orange", edgecolor="k", zorder=10,
                       label="chosen detour wp")

    # obstacle-collision violations along each branch (red x)
    viol_labeled = False
    for name in _ORDER:
        if trajs.get(name) is None:
            continue
        viol = _obstacle_violations(trajs[name][0], pOLL_array)
        if len(viol):
            ax.scatter(viol[:, 0], viol[:, 1], s=42, c="red", marker="x",
                       linewidths=1.6, zorder=8,
                       label=None if viol_labeled else "obstacle violation")
            viol_labeled = True

    # target gate frame: outer + opening
    _draw_frame_top(ax, gate_c, gate_yaw, FRAME_WIDTH / 2, "tab:blue", lw=3.0)
    _draw_frame_top(ax, gate_c, gate_yaw, FRAME_OPENING / 2, "tab:cyan", lw=4.0)
    ax.scatter([gate_c[0]], [gate_c[1]], c="tab:blue", s=45, marker="o",
               edgecolor="k", zorder=6)

    # nearby context gates (faded)
    for gc, gy in zip(np.asarray(pGLL_array, float), np.asarray(y_GBL_array, float)):
        if np.allclose(gc, gate_c) or not near(gc):
            continue
        _draw_frame_top(ax, gc, gy, FRAME_WIDTH / 2, "gray", lw=1.0, alpha=0.4)

    # nearby obstacles: keep-out shells (top-down)
    obst_labeled = False
    for o in np.asarray(pOLL_array, float).reshape(-1, 3):
        if not near(o):
            continue
        ax.add_patch(plt.Circle((o[0], o[1]), R_OBSTACLE + CLEARANCE,
                                color="orange", alpha=0.18, zorder=1))
        ax.add_patch(plt.Circle((o[0], o[1]), R_OBSTACLE,
                                color="firebrick", alpha=0.55, zorder=2,
                                label=None if obst_labeled else "obstacle keep-out"))
        obst_labeled = True

    # local zoom around the gate (x:y = 5:4)
    ax.set_xlim(gate_c[0] - pad * 1.25, gate_c[0] + pad * 1.25)
    ax.set_ylim(gate_c[1] - pad, gate_c[1] + pad)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]", fontsize=20)
    ax.set_ylabel("y [m]", fontsize=20)
    ax.tick_params(axis="both", labelsize=17)
    ax.locator_params(axis="both", nbins=4)   # <= 5 ticks per axis
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              fontsize=14, frameon=False, columnspacing=1.2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26)   # room for the legend band below the axes

    pdf_path = os.path.join(save_dir, f"gate_candidates_iter{outer_iter:02d}.pdf")

    if interactive:
        snap = {"k": 0}

        def _on_key(event):
            if event.key == "s":
                p = os.path.join(save_dir,
                                 f"gate_candidates_iter{outer_iter:02d}_snap{snap['k']}.pdf")
                fig.savefig(p, bbox_inches="tight")
                snap["k"] += 1
                print(f"[gate candidates] snapshot -> {p}")

        cid = fig.canvas.mpl_connect("key_press_event", _on_key)
        print("[gate candidates] pan/zoom to frame it; press 's' to snapshot; "
              "close the window to save the current view and continue.")
        plt.show(block=True)          # blocks until you close the window
        try:
            fig.canvas.mpl_disconnect(cid)
        except Exception:
            pass

    fig.savefig(pdf_path, bbox_inches="tight")   # vector PDF at the current (closed) view
    print(f"[gate candidates] saved {pdf_path}")
    plt.close(fig)
    return pdf_path
