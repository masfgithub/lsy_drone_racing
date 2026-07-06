"""Standalone illustration: the 3 gate-detour push directions on one gate.

Not tied to the planner search -- a self-contained figure that draws a single
gate FACE-ON (as seen from behind, looking along the gate normal) and marks the
three detour push vectors (Left / Right / Top) from the gate center out to the
three new detour waypoints, exactly as the gate branch places them:

    push_L = [-sin(yaw),  cos(yaw), 0]      (gate's left)
    push_R = -push_L                         (gate's right)
    push_T = [0, 0, 1]                       (up)
    new_wp = gate_center + (FRAME_WIDTH / 2 + margin) * push

Run it directly to save a PDF:

    python GatePushIllustration.py

(pass interactive=True to also pop up a window).
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import PathPatch

try:  # keep the file standalone; fall back if the planner isn't importable
    from lsy_drone_racing.control.Planner.planner import FRAME_WIDTH, FRAME_OPENING
except Exception:  # pragma: no cover
    FRAME_WIDTH, FRAME_OPENING = 0.72, 0.10


def _square(half: float) -> np.ndarray:
    """Closed CCW square of half-size `half`, centered at the origin."""
    return np.array([[-half, -half], [half, -half], [half, half],
                     [-half, half], [-half, -half]])


def plot_gate_push_directions(
    save_dir: str = "gate_push_debug",
    push_margin: float = 0.2,
    collision_pt=(0.15, 0.15),
    filename: str = "gate_push_directions.pdf",
    interactive: bool = False,
) -> str:
    """Draw one gate face-on with its 3 detour push vectors + new waypoints.

    Args:
        save_dir:     Output directory for the PDF.
        push_margin:  Extra push beyond the frame half-width (matches the planner's
                      FRAME_WIDTH/2 + 0.2 detour placement).
        collision_pt: (lateral, z) point inside the solid frame band -- a simulated
                      spline / gate-frame collision.
        filename:     Output PDF name.
        interactive:  If True, open a popup before saving.

    Returns:
        Path to the saved PDF.
    """
    os.makedirs(save_dir, exist_ok=True)
    hw = FRAME_WIDTH / 2                 # frame half-width
    ho = FRAME_OPENING / 2              # opening half-width
    push_len = hw + push_margin        # == _compute_gate_detour_waypoint length

    # Face-on plane: horizontal = lateral (gate width), vertical = z (up).
    # The three push directions from the gate center (origin) in this plane.
    dirs = {
        "Left":  np.array([-1.0, 0.0]),
        "Right": np.array([+1.0, 0.0]),
        "Top":   np.array([0.0, +1.0]),
    }

    fig, ax = plt.subplots(figsize=(9, 7))

    # Solid gate frame: outer square with the opening punched out as a hole.
    outer = _square(hw)
    inner = _square(ho)[::-1]           # reversed winding -> even-odd hole
    verts = np.vstack([outer, inner])
    codes = ([Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]) * 2
    ax.add_patch(PathPatch(Path(verts, codes), facecolor="tab:blue", alpha=0.18,
                           edgecolor="none", zorder=1))

    # Frame outlines (outer + opening).
    ax.plot(outer[:, 0], outer[:, 1], color="tab:blue", lw=3.0, zorder=3)
    op = _square(ho)
    ax.plot(op[:, 0], op[:, 1], color="tab:cyan", lw=3.5, zorder=3)

    # Gate center.
    ax.scatter([0], [0], c="tab:blue", s=60, marker="o", edgecolor="k", zorder=6)

    # 3 push vectors (center -> new waypoint) with labelled diamonds.
    for name, d in dirs.items():
        tip = push_len * d
        ax.annotate("", xy=(tip[0], tip[1]), xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="-|>", color="darkorange", lw=2.6),
                    zorder=8)
        ax.scatter([tip[0]], [tip[1]], s=90, marker="D", facecolor="orange",
                   edgecolor="k", zorder=10)
        ax.annotate(name, (tip[0], tip[1]), textcoords="offset points",
                    xytext=(9, 9), fontsize=16, zorder=11)

    # Simulated spline / gate-frame collision inside the solid frame band.
    cp = np.asarray(collision_pt, float)
    ax.scatter([cp[0]], [cp[1]], s=140, c="red", marker="x", linewidths=2.8,
               zorder=12)
    ax.annotate("collision", (cp[0], cp[1]), textcoords="offset points",
                xytext=(8, -15), fontsize=14, color="red", zorder=12)

    span = push_len + 0.25
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span * 1.08)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("lateral [m]", fontsize=20)
    ax.set_ylabel("vertical z [m]", fontsize=20)
    ax.tick_params(axis="both", labelsize=17)
    ax.locator_params(axis="both", nbins=4)

    handles = [
        Line2D([0], [0], color="tab:blue", lw=3.0, label="gate frame"),
        Line2D([0], [0], color="tab:cyan", lw=3.5, label="gate opening"),
        Line2D([0], [0], marker="o", ls="", markerfacecolor="tab:blue",
               markeredgecolor="k", markersize=9, color="w", label="gate center"),
        Line2D([0], [0], color="darkorange", lw=2.6, label="push vector"),
        Line2D([0], [0], marker="D", ls="", markerfacecolor="orange",
               markeredgecolor="k", markersize=9, color="w", label="new waypoint"),
        Line2D([0], [0], marker="x", ls="", markeredgecolor="red",
               markerfacecolor="red", markersize=10, markeredgewidth=2.5,
               color="w", label="frame collision"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, fontsize=14, frameon=False, columnspacing=1.2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)

    if interactive:
        print("[gate push] close the window to save the current view.")
        plt.show(block=True)

    pdf_path = os.path.join(save_dir, filename)
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[gate push] saved {pdf_path}")
    plt.close(fig)
    return pdf_path


if __name__ == "__main__":
    plot_gate_push_directions()
