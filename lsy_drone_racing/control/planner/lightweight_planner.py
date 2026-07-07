"""Subclass definition of Splineplanner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control.planner.spline_planner_base import (
    CLEARANCE,
    FRAME_WIDTH,
    Planner,
    Trajectory,
)

_MAX_AVOID_ITER = 6  # re-check loop cap
_OBST_RADIUS = 0.15  # pillar radius (m), matches _check_obsticle
_SAMPLE_DS = 0.03  # collision-check spacing (m)
_LEAD = 0
_MAX_GATE_ITER = 2
_GATE_OFFSET = 0.5
PUSH_GAIN = 1.1

if TYPE_CHECKING:
    from lsy_drone_racing.control.env_obs import EnvState


class SplinePlanner(Planner):
    """Class to generate smooth Drone Trajectory for MPC."""

    def __init__(
        self, obs: EnvState, info: dict, config: dict, t_total: float, max_speed: float = 2.0
    ):
        """Initialize SplinePlanner.

        Args:
            obs:        Initial observation.
            info:       Additional environment information.
            config:     Environment configuration.
            t_total:    Assumed total time for the trajectory.
            max_speed:  Maximum assumed speed the drone can reach.
        """
        super().__init__(obs, info, config)
        self._t_total = t_total
        self.max_speed = max_speed

    def plan(self, obs: EnvState, t_elapsed: float) -> Trajectory:
        """Function called at the initilazion of the drone racing pipline.

        Args:
            obs:                Current Observed environment.
            t_elapsed:          Time passed so far.

        Returns:
            trajectory:         pos, vel, time in a trajectory class.
        """
        # Create Waypoints with designated function
        p_WLL_array = self._build_waypoints(obs, t_elapsed)

        # Cubic Spline
        spline_ref_array, t_sample = self._create_spline(p_WLL_array, t_elapsed)

        # Call Trajectory Class
        p_ref_array = spline_ref_array(t_sample)
        v_ref_array = spline_ref_array(t_sample, nu=1)
        self.trajectory = Trajectory(p_ref_array, v_ref_array, t_sample)

        return self.trajectory

    def _build_waypoints(self, obs: EnvState, t_elapsed: float) -> np.ndarray:
        """Creates waypoints to avoid hindrances and complete gates.

        Args:
            obs:                Observed environment states.
            t_elapsed:          Time passed in the race so far.

        Returns:
            p_WLL_array:        N-dim array of waypoints for the cubic spline.
        """
        # Current dron position
        pDLL = obs.p_bll

        # Read out gates
        pGLL_array, y_GBL_array = self._gate(obs)

        # Parameter defined to set helping points in front and behind the gates
        distance = 0.3
        lead_dist = 0.1

        # Create waypoint matrix
        p_WLL_array = pDLL

        x, y, z, w = obs.q_blb
        fwd = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 0.0])
        nf = np.linalg.norm(fwd)
        if nf > 1e-6:
            lead = pDLL + lead_dist * (fwd / nf)
            p_WLL_array = np.vstack([p_WLL_array, lead])

        pPrevLL = np.zeros(3)
        pNextLL = np.zeros(3)

        bool_prev_waypoint = np.linalg.norm(pDLL - pGLL_array[0]) > 1.2 * distance

        for i in range(len(pGLL_array)):
            if bool_prev_waypoint:
                pPrevLL[0] = pGLL_array[i, 0] - distance * np.cos(y_GBL_array[i])
                pPrevLL[1] = pGLL_array[i, 1] - distance * np.sin(y_GBL_array[i])
                pPrevLL[2] = pGLL_array[i, 2]

                p_WLL_array = np.vstack([p_WLL_array, pPrevLL])

            p_WLL_array = np.vstack([p_WLL_array, pGLL_array[i]])
            if i == len(pGLL_array):
                distance = 2 * distance
            pNextLL[0] = pGLL_array[i, 0] + distance * np.cos(y_GBL_array[i])
            pNextLL[1] = pGLL_array[i, 1] + distance * np.sin(y_GBL_array[i])
            pNextLL[2] = pGLL_array[i, 2]

            p_WLL_array = np.vstack([p_WLL_array, pNextLL])

        # print('waypoints before detour:', p_WLL_array)
        p_WLL_array = self._detour_gates1(obs, p_WLL_array, pGLL_array, y_GBL_array, t_elapsed)

        p_WLL_array = self._avoid_hindrance(obs, pGLL_array, y_GBL_array, p_WLL_array, t_elapsed)

        # p_WLL_array = self._detour(obs, p_WLL_array, pGLL_array, y_GBL_array, t_elapsed)

        self._waypoints = np.asarray(p_WLL_array, dtype=float).copy()

        first_distance = np.linalg.norm(pDLL - p_WLL_array)

        if not (first_distance < 1e-6):
            p_WLL_array = np.vstack([pDLL, p_WLL_array])

        return p_WLL_array

    def _create_spline1(
        self, p_WLL_array: np.ndarray, t_elapsed: float
    ) -> tuple[CubicSpline, np.ndarray]:
        """Creates a Cubic spline.

        Arg:
            p_WLL_array:            Waypoints the Spline has to bend around.
            t_elapsed:              Time elapsed in the race.
        """
        # Calculate total distance estimate
        segments = np.diff(p_WLL_array, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        cumulative_distances = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        d_total = cumulative_distances[-1]

        # Compute initial times at gates and time samples needed for the remaining time
        t_remaining = self._t_total - t_elapsed
        t_gates = (cumulative_distances / d_total) * t_remaining
        t_sample = np.linspace(0, t_remaining, int(np.round(t_remaining * self.freq)))

        # Cubic Spline
        spline_ref_array = CubicSpline(t_gates, p_WLL_array, axis=0)

        return spline_ref_array, t_sample

    def _create_spline(
        self, p_WLL_array: np.ndarray, t_elapsed: float
    ) -> tuple[CubicSpline, np.ndarray]:
        """Creates a cubic spline through waypoints, dropping coincident points."""
        p_WLL_array = np.asarray(p_WLL_array, dtype=float)

        segment_lengths = np.linalg.norm(np.diff(p_WLL_array, axis=0), axis=1)
        cumulative_distances = np.concatenate([[0.0], np.cumsum(segment_lengths)])

        # drop points that don't advance arc-length (coincident waypoints)
        keep = np.concatenate([[True], segment_lengths > 1e-6])
        p_WLL_array = p_WLL_array[keep]
        cumulative_distances = cumulative_distances[keep]

        d_total = cumulative_distances[-1]
        t_remaining = self._t_total - t_elapsed
        t_gates = (cumulative_distances / d_total) * t_remaining
        t_sample = np.linspace(0, t_remaining, int(np.round(t_remaining * self.freq)))

        spline_ref_array = CubicSpline(t_gates, p_WLL_array, axis=0)
        return spline_ref_array, t_sample

    def _avoid_hindrance(
        self,
        obs: EnvState,
        pGLL_array: np.ndarray,
        y_GBL_array: np.array,
        p_WLL_array: np.ndarray,
        t_elapsed: float,
    ) -> np.ndarray:
        """Check if current waypoints hit obsticles or gate frames and replan.

        Args:
            obs:                Environment state observation.
            pGLL_array:         N dim array with gate centres.
            y_GBL_array:        Array with gate yaws.
            p_WLL_array:        N-dim array of waypoints for the cubic spline.
            t_elapsed:          Time passed during the race.

        Returns:
            p_WLL_array:        N-dim array of waypoints for the cubic spline.
        """
        # Read out obsticles
        p_oll_array = obs.p_oll_array

        # Make a dense Spline
        spline_test_array, t_sample = self._create_spline(p_WLL_array, t_elapsed)
        t_dense = np.linspace(0, t_sample[-1], len(t_sample) * 4)
        pts = spline_test_array(t_dense)

        # Init helping variables
        kept = []
        inside = False
        entry_i = None
        entry_c = None
        entry_push = None

        # Check each point from dense Spline for collision with obsticle and gateframe and reroute
        for i, p in enumerate(pts):
            hit_obst, c_xy_obst, push_obst = self._check_obsticle(p, p_oll_array)
            # hit_gate, c_gate, local_gate, yaw_gate, push_gate = self._check_gate(
            #     p, pGLL_array, y_GBL_array
            # )

            if hit_obst:  # or hit_gate:
                if not inside:
                    inside = True
                    entry_i = i
                    if hit_obst:
                        entry_c = c_xy_obst
                        entry_push = push_obst
                    # else:
                    # entry_is_gate = True
                    # entry_c = c_gate
                    # entry_local = local_gate
                    # entry_yaw = yaw_gate
                    # entry_push = push_gate
                continue

            if inside:
                inside = False
                p_in, p_out = pts[entry_i], p

                # 2D radial push around obsticle
                bis = (p_in[:2] - entry_c) + (p_out[:2] - entry_c)
                nb = np.linalg.norm(bis)
                if nb < 1e-6:  # tangent pass
                    tv = p_out[:2] - p_in[:2]
                    bis = np.array([-tv[1], tv[0]])
                    nb = np.linalg.norm(bis) + 1e-9
                new_xy = entry_c + entry_push * bis / nb
                new_wp = [new_xy[0], new_xy[1], 0.5 * (p_in[2] + p_out[2])]
                hit_gate, c_gate, local_gate, yaw_gate, push_gate = self._check_gate(
                    new_wp, pGLL_array, y_GBL_array
                )
                if hit_gate:
                    new_xy = entry_c - entry_push * bis / nb
                kept.append([new_xy[0], new_xy[1], 0.5 * (p_in[2] + p_out[2])])
            kept.append(p)

        if inside:
            kept.append(pts[-1])

        return np.asarray(kept)

    def _detour_gates1(
        self,
        obs: EnvState,
        p_WLL_array: np.ndarray,
        pGLL_array: np.ndarray,
        y_GBL_array: np.ndarray,
        t_elapsed: float,
    ) -> np.ndarray:
        """Samples detours around gates in case the nominal trajectory hits a gate fram.

        Arg:
            obs:                    Environment state estimation.
            p_WLL_array:            Nominal Waypoints.
            pGLL_array:             Centre of gates.
            y_GBL_array:            Yaw of gates.
            t_elapsed:              Time passed in the race so far.

        Returns:
            new_p_WLL_array:        Total waypoints.
        """
        detours = []
        base_push = FRAME_WIDTH / 2 + CLEARANCE * 2

        for _ in range(_MAX_GATE_ITER):
            wps = self._assemble_with_detours(p_WLL_array, detours)
            spline, t_sample = self._create_spline(wps, t_elapsed)
            t_dense = np.linspace(0, t_sample[-1], len(t_sample) * 4)
            pts = spline(t_dense)

            # Check how far each waypoint is so that a new waypoint is set right
            seg = np.linalg.norm(np.diff(wps, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            t_wps = cum / cum[-1] * t_dense[-1]

            hit = False
            for i, p in enumerate(pts):
                is_hit, c, local, yaw, push = self._check_gate(p, pGLL_array, y_GBL_array)
                if is_hit:
                    n_passed = int(np.searchsorted(t_wps, t_dense[i]))
                    d = self._push_dir_frame1(local, yaw)
                    detour = c + 1.5 * base_push * d
                    detours.append((n_passed, detour))
                    hit = True
                    break

            if not hit:
                return wps

            new_p_WLL_array = self._assemble_with_detours1(p_WLL_array, detours)

        return new_p_WLL_array

    def _assemble_with_detours1(self, p_WLL_array: np.ndarray, detours: list) -> np.array:
        """Insert each detour at the proper waypoint-count position.

        Args:
            p_WLL_array:            Current waypoints.
            detours:                Detour waypoint to add.

        Returns:
            new_p_WLL_array:        New waypoints with detour inserterted.
        """
        wps = list(p_WLL_array)
        for idx, pt in sorted(detours, key=lambda x: x[0], reverse=True):
            wps.insert(idx, pt)
        new_p_WLL_array = np.array(wps)
        return new_p_WLL_array

    def _push_dir_frame1(self, local: np.ndarray, yaw: np.array) -> np.array:
        """Gives the direction the sampled point has to be pushed in order to clear the gate.

        Args:
            local:              contains the distances lx, ly, lz to the frame boarders provided
                                by _check_gate.
            yaw:                Yaw of the gate.

        Returns:
            push_dir:           Direction the point needs to be pushed to clear the gate frame.
        """
        lx, ly, lz = local
        half = FRAME_WIDTH / 2
        d = [half - ly, half + ly, half - lz, half + lz]
        k = int(np.argmin(d))
        if k == 0:
            ld = np.array([0.0, 1.0, 0.0])
        elif k == 1:
            ld = np.array([0.0, -1.0, 0.0])
        elif k == 2:
            ld = np.array([0.0, 0.0, 1.0])
        else:
            ld = np.array([0.0, 0.0, -1.0])

        cos, sin = np.cos(yaw), np.sin(yaw)

        push_dir = np.array([ld[0] * cos - ld[1] * sin, ld[0] * sin + ld[1] * cos, ld[2]])
        return push_dir

    def _assemble_with_detours(self, p_WLL_array: np.ndarray, detours: list) -> np.array:
        """Insert each detour at the proper waypoint-count position.

        Args:
            p_WLL_array:            Current waypoints.
            detours:                Detour waypoint to add.

        Returns:
            new_p_WLL_array:        New waypoints with detour inserterted.
        """
        wps = list(p_WLL_array)
        for idx, pt in sorted(detours, key=lambda x: x[0], reverse=True):
            wps.insert(idx, pt)
        new_p_WLL_array = np.array(wps)
        return new_p_WLL_array
