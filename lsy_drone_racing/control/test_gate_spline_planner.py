"""Standalone test for GateCenterSplinePlanner.

Builds a mock observation (a few gates with varied yaw), runs the planner,
prints the labelled waypoints, and shows + saves a 3D plot of the gates,
waypoints and the fitted spline. Run with:  python test_gate_spline_planner.py
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import gate_spline_planner as gsp
from gate_spline_planner import GateCenterSplinePlanner, plot_plan

# Distance (m) between each gate's approach and exit waypoint (they sit at
# +/- half this along the gate normal). Set to 10 cm.
THREAD_SEPARATION = 0.10


def yaw_quat_xyzw(psi: float) -> np.ndarray:
    """Quaternion (xyzw) for a yaw rotation about world z."""
    return np.array([0.0, 0.0, np.sin(psi / 2.0), np.cos(psi / 2.0)])


def make_obs() -> SimpleNamespace:
    """Mock observation: 4 gates with assorted yaw + a drone start pose."""
    drone = np.array([-1.5, 0.75, 0.01])
    centers = np.array(
        [
            [0.5, 0.25, 0.7],
            [1.05, 0.75, 1.2],
            [-1.0, -0.25, 0.7],
            [0.0, -0.75, 1.2],
        ]
    )
    yaws = [-0.78, 2.35, 3.14, 0.0]
    quats = np.array([yaw_quat_xyzw(p) for p in yaws])
    # Obstacles (env.track.obstacles). The planner ignores these for now; they are
    # only drawn so we can see them relative to the gates and the spline.
    obstacles = np.array(
        [
            [0.0, 0.5, 1.55],
            [0.75, 0.5, 1.55],
            [-1.5, -0.25, 1.55],
            [-0.5, -0.75, 1.55],
        ]
    )

    # Real-track config (overrides the standard layout above).
    drone = np.array([1, -0.75056289434432983, 0.6750795841217041])
    centers = np.array(
        [
            #[0.9006273746490479, -0.45056289434432983, 0.6750795841217041],
            [0.7419164776802063, 1.0919069051742554, 1.2226346731185913],
            [-0.45536500215530396, -0.16117151081562042, 0.6655994057655334],
            [-0.8262401223182678, -1.460203766822815, 1.2101666927337646],
        ]
    )
    obstacles = np.array(
        [
            [0.40722018480300903, 1.5295193195343018, 1.552402377128601],
            [1.7187120914459229, -0.03745437040925026, 1.5517929792404175],
            [-1.3613214492797852, -1.5580755472183228, 1.5421255826950073],
            [-0.6277318477630615, -0.7380406260490417, 1.5428614616394043],
        ]
    )
    yaws = [-1.2669805970326333, -2.194743180674716, 0.1733892323819487]
    quats = np.array([yaw_quat_xyzw(p) for p in yaws])

    return SimpleNamespace(
        pBLL=drone,
        qBLB=Rot.from_euler("xyz", [0.0, 0.3, 0.0]).as_quat(),  # slight forward pitch (xyzw)
        pTLL_array=centers,
        qTLT_array=quats,           # xyzw
        pTLL_index=0,
        pOLL_array=obstacles,       # ignored by the planner (drawn only)
        vBLL=np.zeros(3),
    )


def main() -> None:
    obs = make_obs()
    config = SimpleNamespace(env=SimpleNamespace(freq=50))

    # plan() auto-plots after each plan when this flag is on; the standalone test
    # does its own blocking show + save at the end, so turn the auto-plot off here.
    gsp.SHOW_PLAN_PLOT = False

    planner = GateCenterSplinePlanner(obs, info={}, config=config, t_total=8.0,
                                      thread_separation=THREAD_SEPARATION)
    traj = planner.plan(obs, t_elapsed=0.0)
    wps, labels = planner._waypoints, planner._waypoint_labels

    print("Waypoints (label, x, y, z):")
    for p, lab in zip(wps, labels):
        print(f"  {lab:8s} [{p[0]:7.3f} {p[1]:7.3f} {p[2]:7.3f}]")
    print(f"\nTrajectory: positions {traj.positions.shape}, "
          f"velocities {traj.velocities.shape}, ts {traj.ts.shape}")
    print(f"approach<->exit separation: {2 * planner._thread_offset:.3f} m")
    print(f"speed range: {np.linalg.norm(traj.velocities, axis=1).min():.2f} "
          f".. {np.linalg.norm(traj.velocities, axis=1).max():.2f} m/s")

    if planner._avoid_log:
        print("\nObstacle avoidance decisions:")
        for d in planner._avoid_log:
            print(f"  obstacle {d['obstacle']}: chose '{d['chosen']}'  "
                  f"left[viol {d['left_nviol']}, curv {d['left_curv']:.2f}] "
                  f"right[viol {d['right_nviol']}, curv {d['right_curv']:.2f}] "
                  f"(dropped {d['dropped_detours']} detour)")
    else:
        print("\nNo obstacle violations to resolve.")

    obs_xy = np.asarray(obs.pOLL_array, float)[:, :2]
    final_viol = planner._count_violations(traj.positions[:, :2], obs_xy,
                                           planner._obstacle_d_min)
    final_frame = planner._frame_clips(traj.positions)
    if planner._frame_log:
        print("\nGate-frame re-crossing pushes:")
        for d in planner._frame_log:
            print(f"  gate {d['gate']}: re-crossing at offset {d['from_offset']} m "
                  f"-> wrapped '{d['side']}' side at {d['wrapped_to']} m "
                  f"({d['points']} wps; near {d['near_obst']} obst viol, "
                  f"far {d['far_obst']} obst viol)")
    print(f"\nFinal trajectory obstacle violations: {final_viol}")
    print(f"Final trajectory gate-frame clips:    {final_frame}")

    plot_plan(planner, obs, traj, save_path="planner_test.png")
    print("\nSaved plot to planner_test.png")


if __name__ == "__main__":
    main()