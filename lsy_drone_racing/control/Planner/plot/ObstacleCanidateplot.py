"""Interactive obstacle-detour SEARCH plot (top-down), one frame per push step.

Shows the recursive push search for ONE pillar in action, in a TOP-DOWN (x-y)
view (the relevant projection: obstacles are vertical pillars, only horizontal
distance matters). It writes a NUMBERED SEQUENCE of PDF frames that add ONE push
at a time, in the order the search actually runs them, so you can step through
the whole process:

  * frame 00: the initial trajectory that runs straight through the obstacle
    (grey dashed) -- no pushes yet;
  * then ONE detour per frame, in execution order: branch A (+push) is built up
    to completion first (its initial detour, then refinements until the pillar
    clears), then branch B (-push) starting from the same first p_mid;
  * every step is an arrow from its p_mid (star marker, on the trajectory) to the
    new detour waypoint (diamond);
  * the last frame is the full picture (also saved under the un-suffixed name).

Intermediate and rejected trajectories / waypoints stay grey; the chosen final
trajectory (green) and its waypoints (orange) are highlighted.

You get ONE interactive popup per obstacle node showing the full picture: pan/
zoom to frame it, press 's' to snapshot, close it -- every step frame is then
rendered and saved at that same view.

Note: needs an interactive backend for the popup (e.g. TkAgg/QtAgg); set it
before the first pyplot import. Because `_explore` is recursive this fires once
per obstacle decision node -- guard the call site (e.g. `if depth == 0:`) if that
is too many frames.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from lsy_drone_racing.control.Planner.planner import (
        FRAME_WIDTH, FRAME_OPENING, R_OBSTACLE, CLEARANCE,
    )
except Exception:  # pragma: no cover - fallback for standalone use
    FRAME_WIDTH, FRAME_OPENING, R_OBSTACLE, CLEARANCE = 0.72, 0.10, 0.15, 0.10

_WIN_COLOR = "tab:green"
_LOSE_COLOR = "0.55"      # grey (rejected final)
_INTER_COLOR = "0.7"      # lighter grey (intermediate iterations)
_COUNTER = {"n": 0}       # keeps filenames unique across recursion nodes


def _draw_frame_top(ax, c, yaw, half, color, lw, alpha):
    """Gate frame as a line across the width in the top-down (x-y) view."""
    w = np.array([-np.sin(yaw), np.cos(yaw)])
    a = np.asarray(c, float)[:2] - half * w
    b = np.asarray(c, float)[:2] + half * w
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, alpha=alpha)


def _frame_hit_points(planner, pts, pGLL_array, y_GBL_array):
    """xy points of a sampled trajectory that collide with a gate frame.

    Uses the planner's own `_check_gate3` -- the tight, physically-shaped frame
    test that `_count_gate_hits`/`_pick_better` use to actually decide the
    winner -- so the marked collisions are exactly the ones the search counts.
    (NOT `_check_gate`, whose ~1 m-deep keep-out slab is for pushing waypoints
    and flags points nowhere near the physical frame.)

    Marks every sampled point that is inside a frame. Returns (N, 2) or None.
    """
    if pts is None or pGLL_array is None or y_GBL_array is None:
        return None
    pG = np.asarray(pGLL_array, float)
    yG = np.asarray(y_GBL_array, float)
    hits = []
    for p in np.asarray(pts, float):
        try:
            inside = bool(planner._check_gate3(p, pG, yG)[0])
        except Exception:
            inside = False
        if inside:
            hits.append(p[:2])
    return np.asarray(hits) if hits else None


def _draw_branch(ax, sample, steps, reveal, *, final_color, final_lw, wp_face,
                 arrow_color, pmid_color, final_zorder, wp_zorder):
    """Draw one branch's iterations, revealing only the first `reveal` steps.

    The last revealed trajectory is `final_color`; earlier ones are faint grey.
    Each revealed step contributes a p_mid star, an arrow p_mid -> new waypoint,
    and the new detour waypoint (diamond).
    """
    shown = steps[:reveal]
    if not shown:
        return
    n = len(shown)

    # trajectory after each revealed iteration: intermediates faint, latest bold
    for k, st in enumerate(shown):
        pts = sample(st["wps_after"])
        if pts is None:
            continue
        if k == n - 1:
            ax.plot(pts[:, 0], pts[:, 1], "-", color=final_color, lw=final_lw,
                    zorder=final_zorder)
        else:
            ax.plot(pts[:, 0], pts[:, 1], "-", color=_INTER_COLOR, lw=1.2,
                    alpha=0.55, zorder=4)

    # push vectors: arrow from each revealed p_mid to the waypoint it created
    for st in shown:
        p_mid = np.asarray(st["p_mid"], float)
        new_wp = np.asarray(st["new_wp"], float)
        ax.annotate("", xy=(new_wp[0], new_wp[1]), xytext=(p_mid[0], p_mid[1]),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=2.0),
                    zorder=11)
        ax.scatter([new_wp[0]], [new_wp[1]], s=60, marker="D",
                   facecolor=wp_face, edgecolor="k", zorder=wp_zorder)
        ax.scatter([p_mid[0]], [p_mid[1]], s=120, marker="*",
                   facecolor=pmid_color, edgecolor="k", zorder=12)


def plot_obstacle_candidates(
    planner,
    wps_A,
    wps_B,
    winner: str,
    obst_c,
    t_elapsed: float,
    depth: int,
    pOLL_array: np.ndarray = None,          # kept for signature compatibility; not drawn
    pGLL_array: np.ndarray | None = None,
    y_GBL_array: np.ndarray | None = None,
    save_dir: str = "obstacle_candidate_debug",
    pad: float = 0.6,                        # half-size [m] of the top-down view box around the pillar
    interactive: bool = True,
    history: dict | None = None,
) -> str:
    """Top-down step-by-step plot of the push search for one pillar; save PDFs.

    Args:
        planner:      SplinePlanner (uses `_create_spline`).
        wps_A, wps_B: Final waypoints of the +push / -push branches (context).
        winner:       "A" or "B" -- which branch `_pick_better` chose.
        obst_c:       2D (xy) center of the obstacle being resolved.
        t_elapsed:    Current race time (for spline timing).
        depth:        Recursion depth (filename).
        pGLL_array:   Gate centers (for faded context frames).
        y_GBL_array:  Gate yaws.
        history:      {"initial_wps", "A": steps, "B": steps, "winner"} where each
                      steps entry is {"p_mid", "new_wp", "wps_after"}.
        interactive:  If True, open one popup on the full picture and save every
                      step frame at the view you close it at.

    Returns:
        Path to the saved full-picture PDF.
    """
    os.makedirs(save_dir, exist_ok=True)
    obst_c = np.asarray(obst_c, float)[:2]

    def sample(wps):
        try:
            wps = np.asarray(wps, float)
            sp, t_sample = planner._create_spline(wps, t_elapsed)
            td = np.linspace(0, t_sample[-1], len(t_sample) * 4)
            return sp(td)
        except Exception:
            return None

    win_is_A = winner == "A"
    win_label = "A (+push)" if win_is_A else "B (-push)"
    lose_label = "B (-push)" if win_is_A else "A (+push)"

    history = history or {}
    steps_A = history.get("A") or []
    steps_B = history.get("B") or []
    init_wps = history.get("initial_wps")
    len_A, len_B = len(steps_A), len(steps_B)

    # winner branch drawn green/orange, loser grey
    win_style = dict(final_color=_WIN_COLOR, final_lw=2.6, wp_face="orange",
                     arrow_color="darkorange", pmid_color="gold",
                     final_zorder=6, wp_zorder=10)
    lose_style = dict(final_color=_LOSE_COLOR, final_lw=1.9, wp_face=_LOSE_COLOR,
                      arrow_color="0.4", pmid_color="0.55",
                      final_zorder=5, wp_zorder=9)
    style_A = win_style if win_is_A else lose_style
    style_B = lose_style if win_is_A else win_style

    handles = [
        Patch(facecolor="firebrick", alpha=0.55, label="obstacle keep-out"),
        Line2D([0], [0], color="0.35", ls="--", lw=1.6, label="initial (hits obstacle)"),
        Line2D([0], [0], color=_WIN_COLOR, lw=2.6, label=f"chosen: {win_label}"),
        Line2D([0], [0], color=_LOSE_COLOR, lw=1.9, label=f"rejected: {lose_label}"),
        Line2D([0], [0], color=_INTER_COLOR, lw=1.2, label="intermediate"),
        Line2D([0], [0], marker="*", ls="", markerfacecolor="gold",
               markeredgecolor="k", markersize=15, color="w", label="p_mid"),
        Line2D([0], [0], marker="D", ls="", markerfacecolor="orange",
               markeredgecolor="k", markersize=9, color="w", label="detour wp"),
        Line2D([0], [0], color="darkorange", lw=2.0, label="push vector"),
        Line2D([0], [0], marker="x", ls="", markeredgecolor="red",
               markeredgewidth=2.5, markersize=10, color="w", label="gate-frame hit"),
    ]

    def render(ax, reveal_A, reveal_B):
        """Draw the scene revealing reveal_A pushes of branch A and reveal_B of B."""
        # obstacle keep-out (behind everything)
        ax.add_patch(plt.Circle((obst_c[0], obst_c[1]), R_OBSTACLE,
                                color="firebrick", alpha=0.55, zorder=2))

        # initial trajectory (runs through the obstacle)
        if init_wps is not None:
            pts0 = sample(init_wps)
            if pts0 is not None:
                ax.plot(pts0[:, 0], pts0[:, 1], "--", color="0.35", lw=1.6,
                        alpha=0.85, zorder=3)

        # each branch, revealed up to its own count (zorder keeps winner on top)
        _draw_branch(ax, sample, steps_A, reveal_A, **style_A)
        _draw_branch(ax, sample, steps_B, reveal_B, **style_B)

        # mark gate-frame collisions on the latest revealed trajectory of each
        # branch with a red x -- makes it obvious which side hits a frame (and
        # therefore why the other side is picked).
        for steps, reveal in ((steps_A, reveal_A), (steps_B, reveal_B)):
            shown = steps[:reveal]
            if not shown:
                continue
            hits = _frame_hit_points(planner, sample(shown[-1]["wps_after"]),
                                     pGLL_array, y_GBL_array)
            if hits is not None:
                ax.scatter(hits[:, 0], hits[:, 1], s=90, marker="x", color="red",
                           linewidths=2.5, zorder=13)

        # faded context gate frames for orientation
        if pGLL_array is not None and y_GBL_array is not None:
            for gc, gy in zip(np.asarray(pGLL_array, float), np.asarray(y_GBL_array, float)):
                _draw_frame_top(ax, gc, gy, FRAME_WIDTH / 2, "tab:blue", 2.0, 0.35)
                _draw_frame_top(ax, gc, gy, FRAME_OPENING / 2, "tab:cyan", 3.0, 0.45)

        ax.set_xlim(obst_c[0] - pad * 1.25, obst_c[0] + pad * 1.25)
        ax.set_ylim(obst_c[1] - pad, obst_c[1] + pad)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x [m]", fontsize=25)
        ax.set_ylabel("y [m]", fontsize=25)
        ax.tick_params(axis="both", labelsize=22)
        ax.locator_params(axis="both", nbins=4)   # <= 5 ticks per axis
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=3, fontsize=16, frameon=False, columnspacing=1.2)

    def finish(fig, ax, view):
        if view is not None:
            ax.set_xlim(view[0])
            ax.set_ylim(view[1])
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.28)   # room for the legend band below the axes

    _COUNTER["n"] += 1
    base = f"obstacle_candidates_{_COUNTER['n']:03d}_d{depth}"

    # One interactive popup on the FULL picture; capture the view for the frames.
    view = None
    if interactive:
        fig, ax = plt.subplots(figsize=(9, 7))
        render(ax, len_A, len_B)
        finish(fig, ax, None)
        snap = {"k": 0}

        def _on_key(event):
            if event.key == "s":
                p = os.path.join(save_dir, f"{base}_snap{snap['k']}.pdf")
                fig.savefig(p, bbox_inches="tight")
                snap["k"] += 1
                print(f"[obstacle candidates] snapshot -> {p}")

        cid = fig.canvas.mpl_connect("key_press_event", _on_key)
        print("[obstacle candidates] pan/zoom to frame the full picture; press 's' "
              "to snapshot; close the window to render every step frame at that view.")
        plt.show(block=True)
        view = (ax.get_xlim(), ax.get_ylim())
        try:
            fig.canvas.mpl_disconnect(cid)
        except Exception:
            pass
        plt.close(fig)

    # Render + save one frame per push step, in execution order: A first, then B.
    full_path = os.path.join(save_dir, f"{base}.pdf")
    n_frames = len_A + len_B + 1
    for f in range(n_frames):
        reveal_A = min(f, len_A)
        reveal_B = min(max(0, f - len_A), len_B)
        fig, ax = plt.subplots(figsize=(9, 7))
        render(ax, reveal_A, reveal_B)
        finish(fig, ax, view)
        fig.savefig(os.path.join(save_dir, f"{base}_iter{f:02d}.pdf"),
                    bbox_inches="tight")
        if f == n_frames - 1:                  # last frame == full picture
            fig.savefig(full_path, bbox_inches="tight")
        plt.close(fig)

    print(f"[obstacle candidates] saved {n_frames} step frames -> "
          f"{base}_iter00..{n_frames - 1:02d}.pdf (+ {os.path.basename(full_path)})")
    return full_path
