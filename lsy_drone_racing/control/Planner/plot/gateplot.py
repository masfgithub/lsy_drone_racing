#!/usr/bin/env python3
"""Standalone 3D visualization of the planner's gate-frame model (`_check_gate3`).

The planner treats a gate frame as four tapered "bars": each bar is sharp (zero
thickness along the gate normal) at the opening and grows to the full frame
thickness at the outer edge. This reproduces that exact solid as four wedge
prisms and saves a vector PDF.

Run:
    python plot_gate_model.py                 # writes gate_frame_model.pdf + shows
    (set SHOW_POINTS = True below to overlay samples straight from check_gate3
     as a proof that the polygons match the inequality)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------------------------------------------------------------------
# Gate constants -- mirror lsy_drone_racing/.../planner.py
#
# NOTE: planner.py sets FRAME_OPENING = 0.1, which together with DRONE_RADIUS =
# 0.05 gives z0 = FRAME_OPENING/2 - DRONE_RADIUS = 0.0: the modeled opening
# collapses to a point and there is no visible hole. The gate metadata dict in
# the same file uses hole_width = 0.25, which yields a sensible opening. We
# default to 0.25 here so the model is legible -- set this to your real opening.
# ---------------------------------------------------------------------------
FRAME_WIDTH   = 0.72   # outer edge length of the (square) frame   [m]
FRAME_OPENING = 0.25   # flyable opening edge length               [m]  (planner.py: 0.10)
FRAME_THICK   = 0.40   # frame depth along the gate normal         [m]
DRONE_RADIUS  = 0.05   # inflation applied inside _check_gate3      [m]

SHOW_POINTS = False    # overlay interior samples from check_gate3 (verification)
SAVE_PATH   = "gate_frame_model.pdf"

# Derived half-extents, exactly as in _check_gate3 -----------------------------
z0 = FRAME_OPENING / 2 - DRONE_RADIUS   # inner (opening) half-extent, height
y0 = FRAME_OPENING / 2 - DRONE_RADIUS   # inner (opening) half-extent, width
z1 = FRAME_WIDTH   / 2 + DRONE_RADIUS   # outer half-extent, height
y1 = FRAME_WIDTH   / 2 + DRONE_RADIUS   # outer half-extent, width
a  = FRAME_THICK   / 2                  # max half-thickness along the normal
m  = a / (z1 - z0)                      # taper slope (thickness per unit offset)

if z0 <= 0 or y0 <= 0:
    print(f"[warning] modeled opening collapsed (z0={z0:.3f}, y0={y0:.3f}); "
          f"set FRAME_OPENING > 2*DRONE_RADIUS for a visible hole.")


def check_gate3(p_local):
    """Standalone copy of the planner's frame test, in gate-local coordinates.

    p_local : (..., 3) points already in the gate frame (x=normal, y=width,
              z=height). Returns a boolean array, True inside the solid frame.
    """
    x, y, z = p_local[..., 0], p_local[..., 1], p_local[..., 2]
    check_z = (np.abs(z) > z0) & (np.abs(z) < z1) & (np.abs(x) < m * (np.abs(z) - z0)) & (np.abs(y) < y1)
    check_y = (np.abs(y) > y0) & (np.abs(y) < y1) & (np.abs(x) < m * (np.abs(y) - y0)) & (np.abs(z) < z1)
    return check_z | check_y


def wedge_faces(depth_axis, depth_inner, depth_outer, perp_axis, perp_half):
    """Polygon faces of one tapered bar (a triangular prism).

    Sharp (zero thickness in the normal/x direction) at depth_inner, full
    thickness (+/- a) at depth_outer, spanning +/- perp_half along perp_axis --
    exactly one branch of check_gate3.

    depth_axis / perp_axis : 1 (width/y) or 2 (height/z); the normal is x (0).
    """
    x_ax = 0

    def vtx(depth, xv, perp):
        p = np.zeros(3)
        p[depth_axis] = depth
        p[x_ax] = xv
        p[perp_axis] = perp
        return p

    A0 = vtx(depth_inner, 0.0, -perp_half)   # apex edge (sharp, inner)
    A1 = vtx(depth_inner, 0.0, +perp_half)
    B0 = vtx(depth_outer, -a, -perp_half)    # base rectangle (outer)
    B1 = vtx(depth_outer, +a, -perp_half)
    B2 = vtx(depth_outer, +a, +perp_half)
    B3 = vtx(depth_outer, -a, +perp_half)

    return [
        np.array([A0, B0, B1]),          # triangular end cap (perp = -)
        np.array([A1, B3, B2]),          # triangular end cap (perp = +)
        np.array([A0, A1, B2, B1]),      # slanted +x face
        np.array([A0, A1, B3, B0]),      # slanted -x face
        np.array([B0, B1, B2, B3]),      # outer base face
    ]


def build_gate():
    """All four bars (top, bottom, right, left) as one list of faces."""
    faces = []
    faces += wedge_faces(2,  z0,  z1, 1, y1)   # top    (tapers in +z)
    faces += wedge_faces(2, -z0, -z1, 1, y1)   # bottom (tapers in -z)
    faces += wedge_faces(1,  y0,  y1, 2, z1)   # right  (tapers in +y)
    faces += wedge_faces(1, -y0, -y1, 2, z1)   # left   (tapers in -y)
    return faces


def main():
    faces = build_gate()

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    coll = Poly3DCollection(
        faces,
        facecolor=(0.30, 0.55, 0.95, 0.42),
        edgecolor=(0.10, 0.20, 0.45, 0.9),
        linewidths=0.5,
    )
    ax.add_collection3d(coll)

    # Opening outline (the sharp apex edges lie at x = 0) for orientation.
    op = np.array([[0, -y0, -z0], [0, y0, -z0], [0, y0, z0], [0, -y0, z0], [0, -y0, -z0]])
    ax.plot(op[:, 0], op[:, 1], op[:, 2], color="crimson", lw=1.6, label="opening (x = 0)")

    # Optional proof that the polygons equal the check_gate3 inequality.
    if SHOW_POINTS:
        g = np.linspace(-1, 1, 41)
        X, Y, Z = np.meshgrid(a * g, y1 * g, z1 * g, indexing="ij")
        P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        pin = P[check_gate3(P)][::7]
        ax.scatter(pin[:, 0], pin[:, 1], pin[:, 2], s=2, c="k", alpha=0.15)

    # Gate-normal arrow.
    ax.quiver(0, 0, 0, 1.15 * a, 0, 0, color="k", arrow_length_ratio=0.15, lw=1.2)
    ax.text(1.25 * a, 0, 0, "gate normal (x)", fontsize=9)

    ax.set_xlabel("normal  x  [m]")
    ax.set_ylabel("width   y  [m]")
    ax.set_zlabel("height  z  [m]")
    ax.set_xlim(-a, a)
    ax.set_ylim(-y1, y1)
    ax.set_zlim(-z1, z1)
    ax.set_box_aspect((2 * a, 2 * y1, 2 * z1))   # undistorted geometry
    ax.view_init(elev=22, azim=-58)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    fig.savefig(SAVE_PATH)   # PDF backend -> vector output
    print(f"saved {SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    main()