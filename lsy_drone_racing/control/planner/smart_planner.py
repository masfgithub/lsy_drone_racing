"""Subclass definition of new planner."""

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control.env_obs import EnvState
from lsy_drone_racing.control.planner.smart_planner_base import (
    FRAME_WIDTH,
    R_OBSTACLE,
    Planner,
    Trajectory,
)

_MAX_AVOID_ITER = 20  # maximum number of iterations to avoid obstacles


class SplinePlanner(Planner):
    """Class to generate smooth Drone Trajectory for MPC."""

    def __init__(self, obs: EnvState, info: dict, config: dict, t_total: float):
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

        self._waypoints = p_WLL_array

        # Cubic Spline
        spline_ref_array, t_sample = self._create_spline(p_WLL_array, t_elapsed)

        # Call Trajectory Class
        p_ref_array = spline_ref_array(t_sample)
        v_ref_array = spline_ref_array(t_sample, nu=1)
        self.trajectory = Trajectory(p_ref_array, v_ref_array, t_sample)

        return self.trajectory

    def _create_spline(
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

    def _build_waypoints(self, obs: EnvState, t_elapsed: float) -> np.ndarray:
        """Creates waypoints to avoid hindrances and complete gates.

        Args:
            obs:                Observed environment states.
            t_elapsed:          Time passed in the race so far.

        Returns:
            p_WLL_array:        N-dim array of waypoints for the cubic spline.
        """
        # Current drone position
        pDLL = obs.p_bll

        # Read out gates
        pGLL_array, y_GBL_array = self._gate(obs)

        # Read out obstacles
        p_oll_array = obs.p_oll_array
        # print(p_oll_array)
        # Parameter defined to set helping points in front and behind the gates
        distance = 0.05

        # Create waypoint matrix
        p_WLL_array = pDLL

        pPrevLL = np.zeros(3)
        pNextLL = np.zeros(3)

        for i in range(len(pGLL_array)):
            pPrevLL[0] = pGLL_array[i, 0] - distance * np.cos(y_GBL_array[i])
            pPrevLL[1] = pGLL_array[i, 1] - distance * np.sin(y_GBL_array[i])
            pPrevLL[2] = pGLL_array[i, 2]

            p_WLL_array = np.vstack([p_WLL_array, pPrevLL])

            p_WLL_array = np.vstack([p_WLL_array, pGLL_array[i]])

            if i == len(pGLL_array) - 1:
                distance = 1

            pNextLL[0] = pGLL_array[i, 0] + distance * np.cos(y_GBL_array[i])
            pNextLL[1] = pGLL_array[i, 1] + distance * np.sin(y_GBL_array[i])
            pNextLL[2] = pGLL_array[i, 2]

            p_WLL_array = np.vstack([p_WLL_array, pNextLL])
        # p_WLL_array = self._180_degree_turn(p_WLL_array,
        #                                    p_oll_array, pGLL_array, y_GBL_array, t_elapsed, obs)
        # p_WLL_array = self._avoid_collisions(
        #     p_WLL_array, p_oll_array, pGLL_array, y_GBL_array, t_elapsed
        # )

        p_WLL_array = self._avoidance_tree(
            p_WLL_array, p_oll_array, pGLL_array, y_GBL_array, t_elapsed
        )

        return p_WLL_array

    def _avoidance_tree(
        self,
        p_WLL_array: np.ndarray,
        p_oll_array: np.ndarray,
        pGLL_array: np.ndarray,
        y_GBL_array: np.array,
        t_elapsed: float,
    ) -> np.ndarray:
        # p_WLL_array = self._avoid_gates(p_WLL_array, pGLL_array, y_GBL_array, t_elapsed)
        p_WLL_array = self._avoid_gates_tree(
            p_WLL_array, pGLL_array, y_GBL_array, p_oll_array, t_elapsed
        )
        p_WLL_array = self._avoid_obsticles(
            p_WLL_array, p_oll_array, pGLL_array, y_GBL_array, t_elapsed
        )

        return p_WLL_array

    def _score_branch(self, wps: np.ndarray, t_elapsed: float) -> float:
        """Compute the arc length of the spline through these waypoints.

        Args:
            wps:        Waypoint list to score.
            t_elapsed:  Current race time.

        Returns:
            arc_length: Total arc length of the spline.
        """
        spline, t_sample = self._create_spline(wps, t_elapsed)
        t_dense = np.linspace(0, t_sample[-1], len(t_sample) * 4)
        pts = spline(t_dense)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return float(np.sum(seg))

    def _avoid_obsticles(
        self,
        p_WLL_array: np.ndarray,
        p_oll_array: np.ndarray,
        pGLL_array: np.ndarray,
        y_GBL_array: np.ndarray,
        t_elapsed: float,
    ) -> np.ndarray:
        wps = p_WLL_array.copy()
        # Stash for the tree to use in scoring
        self._pGLL_array = pGLL_array
        self._y_GBL_array = y_GBL_array

        wps_final, is_clear = self._explore(wps, t_elapsed, p_oll_array, max_depth=6)
        print(f"final: clear={is_clear}")
        return wps_final

    def _explore(
        self,
        wps: np.ndarray,
        t_elapsed: float,
        p_oll_array: np.ndarray,
        max_depth: int = 6,
        depth: int = 0,
    ) -> tuple[np.ndarray, bool]:
        """Recursively explore: at each NEW obstacle, branch into both directions.

        Iteratively resolve that one obstacle inside each branch.
        """
        if depth >= max_depth:
            return wps, False

        # Build spline, find first violation
        spline, t_sample = self._create_spline(wps, t_elapsed)
        t_dense = np.linspace(0, t_sample[-1], len(t_sample) * 4)
        pts = spline(t_dense)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])

        entry_i, exit_i, entry_obst_c = self._find_first_obstacle_violation(pts, p_oll_array)
        if entry_i is None:
            return wps, True  # No violations remaining — done

        # Branch A: explore the +push side. Iteratively resolve THIS obstacle
        # inside the branch, then recurse for the NEXT obstacle.
        wps_A, ok_A = self._explore_branch(
            wps,
            entry_i,
            exit_i,
            entry_obst_c,
            pts,
            cum,
            initial_push_sign=+1,
            t_elapsed=t_elapsed,
            p_oll_array=p_oll_array,
            max_iter=10,
            target_obst_c=entry_obst_c,
        )
        if ok_A:
            wps_A, clear_A = self._explore(
                wps_A, t_elapsed, p_oll_array, max_depth=max_depth, depth=depth + 1
            )
        else:
            clear_A = False

        # Branch B: same with -push side
        wps_B, ok_B = self._explore_branch(
            wps,
            entry_i,
            exit_i,
            entry_obst_c,
            pts,
            cum,
            initial_push_sign=-1,
            t_elapsed=t_elapsed,
            p_oll_array=p_oll_array,
            max_iter=10,
            target_obst_c=entry_obst_c,
        )
        if ok_B:
            wps_B, clear_B = self._explore(
                wps_B, t_elapsed, p_oll_array, max_depth=max_depth, depth=depth + 1
            )
        else:
            clear_B = False

        return self._pick_better(wps_A, clear_A, wps_B, clear_B, t_elapsed, depth)

    def _pick_better(
        self,
        wps_A: np.ndarray,
        clear_A: bool,
        wps_B: np.ndarray,
        clear_B: bool,
        t_elapsed: float,
        depth: int = 0,
    ) -> tuple[np.ndarray, bool]:
        """Pick the better of two candidate branches.

        Priority:
        1. Obstacle-clear status (clear > not clear)
        2. Gate-hit count (fewer is better)
        3. Arc length (shorter is better)
        """
        # Need gates and yaws here; access them from instance state set up by the call site
        pGLL_array = self._pGLL_array
        y_GBL_array = self._y_GBL_array

        len_A = self._score_branch(wps_A, t_elapsed)
        len_B = self._score_branch(wps_B, t_elapsed)
        gates_A = self._count_gate_hits(wps_A, t_elapsed, pGLL_array, y_GBL_array)
        gates_B = self._count_gate_hits(wps_B, t_elapsed, pGLL_array, y_GBL_array)

        indent = "  " * depth
        print(
            f"{indent}A: clear={clear_A}, gates={gates_A}, len={len_A:.2f}   "
            f"B: clear={clear_B}, gates={gates_B}, len={len_B:.2f}"
        )

        # Priority 1: obstacle-clear
        if clear_A and not clear_B:
            print(f"{indent}-> picked A (only clear)")
            return wps_A, True
        if clear_B and not clear_A:
            print(f"{indent}-> picked B (only clear)")
            return wps_B, True

        # Both obstacle-clear or both obstacle-not-clear
        # Priority 2: fewer gate hits
        if gates_A < gates_B:
            print(f"{indent}-> picked A (fewer gate hits)")
            return wps_A, clear_A
        if gates_B < gates_A:
            print(f"{indent}-> picked B (fewer gate hits)")
            return wps_B, clear_B

        # Priority 3: shorter arc length
        if len_A <= len_B:
            print(f"{indent}-> picked A (shorter)")
            return wps_A, clear_A
        else:
            print(f"{indent}-> picked B (shorter)")
            return wps_B, clear_B

    def _count_gate_hits(
        self, wps: np.ndarray, t_elapsed: float, pGLL_array: np.ndarray, y_GBL_array: np.ndarray
    ) -> int:
        """Count distinct gate-frame violation segments along the spline.

        Args:
            wps:          Waypoint list.
            t_elapsed:    Race time.
            pGLL_array:   Gate centers.
            y_GBL_array:  Gate yaws.

        Returns:
            n:            Number of distinct gate-frame violation segments.
        """
        spline, t_sample = self._create_spline(wps, t_elapsed)
        t_dense = np.linspace(0, t_sample[-1], len(t_sample) * 4)
        pts = spline(t_dense)

        n = 0
        inside_gate = False
        for p in pts:
            hit, _, _ = self._check_gate(p, pGLL_array, y_GBL_array)
            if hit:
                if not inside_gate:
                    n += 1
                    inside_gate = True
            else:
                inside_gate = False
        return n

    def _explore_branch(
        self,
        wps_initial: np.ndarray,
        entry_i: int,
        exit_i: int,
        entry_obst_c: np.ndarray,
        pts: np.ndarray,
        cum: np.ndarray,
        initial_push_sign: int,
        t_elapsed: float,
        p_oll_array: np.ndarray,
        target_obst_c: np.ndarray,
        max_iter: int = 10,
    ) -> tuple[np.ndarray, bool]:
        """Insert initial detour with chosen sign, then iteratively add waypoints.

        Waypoints are added until the SPECIFIC target obstacle is no longer hit.

        Returns:
            wps:        Waypoints after this obstacle is resolved.
            ok:         True if the target obstacle was successfully cleared.
        """
        # 1. Initial detour
        p_in, p_out = pts[entry_i], pts[exit_i]
        push_vector = self._compute_initial_push_vector(p_in, p_out, entry_obst_c)
        push_vector = initial_push_sign * push_vector
        initial_wp = self._compute_detour_waypoint(p_in, p_out, entry_obst_c, push_vector)
        s_detour = 0.5 * (cum[entry_i] + cum[exit_i])
        wps = self._insert_detour(wps_initial, initial_wp, s_detour, t_elapsed)

        # 2. Iteratively resolve THIS obstacle only
        for _ in range(max_iter):
            spline, t_sample = self._create_spline(wps, t_elapsed)
            t_dense = np.linspace(0, t_sample[-1], len(t_sample) * 4)
            branch_pts = spline(t_dense)
            branch_seg = np.linalg.norm(np.diff(branch_pts, axis=0), axis=1)
            branch_cum = np.concatenate([[0.0], np.cumsum(branch_seg)])

            # Find ANY violation on the target obstacle
            target_violation = self._find_specific_obstacle_violation(branch_pts, target_obst_c)
            if target_violation is None:
                # Target obstacle is clear — done
                return wps, True

            e_i, x_i = target_violation
            p_in_b, p_out_b = branch_pts[e_i], branch_pts[x_i]
            pv = self._compute_initial_push_vector(p_in_b, p_out_b, target_obst_c)
            wp = self._compute_detour_waypoint(p_in_b, p_out_b, target_obst_c, pv)
            s_new = 0.5 * (branch_cum[e_i] + branch_cum[x_i])
            wps = self._insert_detour(wps, wp, s_new, t_elapsed)

        # Failed to clear after max_iter
        return wps, False

    def _find_specific_obstacle_violation(
        self, pts: np.ndarray, target_obst_c: np.ndarray, r_obstacle: float | None = None
    ) -> tuple[int, int] | None:
        """Find the first violation of a SPECIFIC obstacle in pts.

        Args:
            pts:            Dense samples (N, 3).
            target_obst_c:  2D center of the target obstacle.
            r_obstacle:     Radius (defaults to R_OBSTACLE).

        Returns:
            (entry_i, exit_i) for the first violation, or None.
        """
        if r_obstacle is None:
            r_obstacle = R_OBSTACLE

        inside = False
        entry_i = None

        for i, p in enumerate(pts):
            d = np.linalg.norm(p[:2] - target_obst_c)
            if d < r_obstacle:
                if not inside:
                    inside = True
                    entry_i = i
                continue
            if inside:
                return entry_i, i

        if inside:
            return entry_i, len(pts) - 1
        return None

    def _insert_detour(
        self, wps: np.ndarray, new_wp: np.ndarray, s_target: float, t_elapsed: float
    ) -> np.ndarray:
        """Insert a detour waypoint into wps at the right arc-length position."""
        # Build current spline to get knot arc-lengths
        spline, t_sample = self._create_spline(wps, t_elapsed)
        t_dense = np.linspace(0, t_sample[-1], len(t_sample) * 4)
        pts = spline(t_dense)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        t_knots = spline.x
        kept_wps = spline(t_knots)
        s_wp = np.interp(t_knots, t_dense, cum)

        items = [(s_wp[k], kept_wps[k]) for k in range(len(t_knots))]
        items.append((s_target, new_wp))
        items.sort(key=lambda it: it[0])
        return np.array([pt for _, pt in items])

    def _compute_initial_push_vector(
        self, p_in: np.ndarray, p_out: np.ndarray, obst_c: np.ndarray
    ) -> np.ndarray:
        """Compute the radial bisector push direction (xy unit vector).

        Args:
            p_in:    Entry point of the violation.
            p_out:   Exit point of the violation.
            obst_c:  2D obstacle center.

        Returns:
            push_vector:  2D unit vector pointing away from obstacle through midpoint.
        """
        bis = (p_in[:2] - obst_c) + (p_out[:2] - obst_c)
        nb = np.linalg.norm(bis)
        if nb < 1e-9:
            # Degenerate; pick perpendicular to chord
            tv = (p_out - p_in)[:2]
            bis = np.array([-tv[1], tv[0]])
            nb = np.linalg.norm(bis) + 1e-9
        return bis / nb

    def _find_first_obstacle_violation(
        self, pts: np.ndarray, p_oll_array: np.ndarray
    ) -> tuple[int, int, np.ndarray] | tuple[None, None, None]:
        """Find the first obstacle-violation segment in a dense pts array.

        Args:
            pts:        Dense spline samples (N, 3).
            p_oll_array: Obstacle centers.

        Returns:
            (entry_i, exit_i, obstacle_center) for the first violation, or
            (None, None, None) if no violation.
        """
        inside = False
        entry_i = None
        entry_obst_c = None

        for i, p in enumerate(pts):
            hit, obst_c = self._check_obsticle(p, p_oll_array)
            if hit:
                if not inside:
                    inside = True
                    entry_i = i
                    entry_obst_c = obst_c[:2].copy()
                continue
            if inside:
                return entry_i, i, entry_obst_c

        # If still inside at end of pts, treat last sample as exit
        if inside:
            return entry_i, len(pts) - 1, entry_obst_c
        return None, None, None

    def _compute_detour_waypoint(
        self, p_in: np.ndarray, p_out: np.ndarray, obst_c: np.ndarray, push_vector: np.ndarray
    ) -> np.ndarray:
        """Compute a single detour waypoint outside the obstacle.

        Args:
            p_in:         3D position where trajectory entered the obstacle.
            p_out:        3D position where trajectory exited.
            obst_c:       2D obstacle center (xy only).
            push_vector:  2D unit push direction.

        Returns:
            new_wp:       3D detour waypoint position.
        """
        p_mid = (p_in + p_out) / 2
        push_length = self._get_obsticle_push(p_mid.copy(), obst_c, push_vector)
        new_xy = p_mid[:2] + push_length * push_vector
        new_wp = np.array([new_xy[0], new_xy[1], 0.5 * (p_in[2] + p_out[2])])
        return new_wp

    def _local_spline_hits_gate(
        self,
        wps: np.ndarray,
        new_wp: np.ndarray,
        s_detour: float,
        t_elapsed: float,
        pGLL_array: np.ndarray,
        y_GBL_array: np.ndarray,
        local_radius: float = 0.5,
        n_samples: int = 30,
    ) -> bool:
        """Insert new_wp into wps and check for a nearby gate violation.

        Checks if the resulting spline violates a gate in a window of arc-length
        around the insertion point.

        Args:
            wps:            Current waypoint list (without new_wp).
            new_wp:         Proposed detour waypoint to test.
            s_detour:       Arc-length where new_wp will be inserted (used to pick
                            the local window).
            t_elapsed:      Current race time (for spline timing).
            pGLL_array:     Gate centers.
            y_GBL_array:    Gate yaws.
            local_radius:   Half-width (in meters of arc-length) of the local
                            window around new_wp to check.
            n_samples:      Number of samples within the local window.

        Returns:
            True if any local sample is inside a gate frame, False otherwise.
        """
        # 1. Build temporary waypoint list with new_wp inserted at the right arc-length
        seg = np.linalg.norm(np.diff(wps, axis=0), axis=1)
        cum_wps = np.concatenate([[0.0], np.cumsum(seg)])

        # Find insertion index: first waypoint whose cumulative arc-length is >= s_detour
        insert_idx = int(np.searchsorted(cum_wps, s_detour))
        test_wps = np.insert(wps, insert_idx, new_wp, axis=0)

        # 2. Build a spline through the temporary list
        try:
            test_spline, test_t_sample = self._create_spline(test_wps, t_elapsed)
        except Exception:
            # If spline construction fails (e.g. duplicate points), treat as unsafe
            return True

        # 3. Sample only in the local window in arc-length, then map to t for the spline
        test_t_dense = np.linspace(0, test_t_sample[-1], int(test_t_sample[-1] * self.freq * 4))
        test_pts = test_spline(test_t_dense)
        test_seg = np.linalg.norm(np.diff(test_pts, axis=0), axis=1)
        test_cum = np.concatenate([[0.0], np.cumsum(test_seg)])

        # Find the new waypoint's arc-length in the dense sampling
        # (it's the closest dense sample to new_wp)
        dists = np.linalg.norm(test_pts - new_wp, axis=1)
        idx_wp = int(np.argmin(dists))
        s_wp_dense = test_cum[idx_wp]

        # Window: from s_wp_dense - local_radius to s_wp_dense + local_radius
        s_min = s_wp_dense - local_radius
        s_max = s_wp_dense + local_radius

        # Get dense samples within the window
        in_window = (test_cum >= s_min) & (test_cum <= s_max)
        window_pts = test_pts[in_window]

        # Down-sample to n_samples if window has more
        if len(window_pts) > n_samples:
            step = len(window_pts) // n_samples
            window_pts = window_pts[::step]

        # 4. Check each window sample for gate violation
        for p in window_pts:
            hit, _, _ = self._check_gate(p, pGLL_array, y_GBL_array)
            if hit:
                return True

        return False

    def _avoid_gates(
        self,
        p_WLL_array: np.ndarray,
        pGLL_array: np.ndarray,
        y_GBL_array: np.array,
        t_elapsed: float,
    ) -> np.ndarray:
        wps = p_WLL_array.copy()

        for _ in range(_MAX_AVOID_ITER):
            spline, t_sample = self._create_spline(wps, t_elapsed)
            t_dense = np.linspace(0, t_sample[-1], len(t_sample) * 4)  # change later
            pts = spline(t_dense)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            t_gates = spline.x
            kept_wps = spline(t_gates)
            s_wp = np.interp(t_gates, t_dense, cum)

            # Init helping variables
            detours = []
            inside_gate = False
            entry_i = None

            entry_gate_c = None
            entry_gate_yaw = None

            # Check each point from dense Spline for collision with obsticle
            for i, p in enumerate(pts):
                hit_gate, gate_centre, gate_yaw = self._check_gate(p, pGLL_array, y_GBL_array)

                if hit_gate:
                    if not inside_gate:
                        inside_gate = True
                        entry_i = i
                        entry_gate_c = gate_centre
                        entry_gate_yaw = gate_yaw
                        # print('this gate was hit', entry_gate_c, entry_gate_yaw)
                    continue

                if inside_gate:
                    inside_gate = False
                    p_in, p_out = pts[entry_i], p
                    p_mid = (p_out + p_in) / 2
                    # breakpoint()
                    # 2D radial push around obsticle
                    push_vector = (p_mid - entry_gate_c) / np.linalg.norm(p_mid - entry_gate_c)
                    # push_vector = self._get_gate_push_vector(
                    #     entry_gate_c, entry_gate_yaw, p_in, p_out, p_mid
                    # )
                    push_length = self._get_gate_push(
                        p_mid.copy(), entry_gate_c, entry_gate_yaw, push_vector
                    )
                    # push_length = 0.72
                    # print(push_length)
                    new_xy = p_mid + push_length * push_vector
                    new_wp = np.array([new_xy[0], new_xy[1], 0.5 * (p_in[2] + p_out[2])])
                    # breakpoint()
                    detours.append((0.5 * (cum[entry_i] + cum[i]), new_wp))

            if not detours:
                return wps
            # print(detours)
            items = [(s_wp[k], kept_wps[k]) for k in range(len(t_gates))] + detours
            items.sort(key=lambda it: it[0])
            wps = np.array([pt for _, pt in items])

        return wps

    def _compute_gate_detour_waypoint(
        self, gate_c: np.ndarray, gate_yaw: float, push_vector: np.ndarray
    ) -> np.ndarray:
        """Compute a detour waypoint outside the gate frame in the given direction.

        Starts from the gate CENTER, pushes outward in push_vector direction until
        clear of the modeled frame. This gives symmetric, geometrically grounded
        detour waypoints regardless of where the trajectory clipped the frame.

        Args:
            gate_c:       3D center of the violated gate.
            gate_yaw:     Yaw of the violated gate.
            push_vector:  3D unit vector for push direction.

        Returns:
            new_wp:       3D detour waypoint position.
        """
        push_length = (
            FRAME_WIDTH / 2 + 0.2
        )  # self._get_gate_push(gate_c.copy(), gate_c, gate_yaw, push_vector)
        new_wp = gate_c + push_length * push_vector
        return new_wp

    def _find_first_gate_violation(
        self,
        pts: np.ndarray,
        cum: np.ndarray,
        pGLL_array: np.ndarray,
        y_GBL_array: np.ndarray,
        skip_approach: bool = True,
    ) -> tuple[int, int, np.ndarray, float] | tuple[None, None, None, None]:
        """Find first gate-frame violation along the spline.

        Skips "approach-side" violations: when the spline curves toward a gate and
        grazes its frame on the way in. These are not real problems — the spline
        is about to thread through the gate. We catch exit-side and post-gate
        violations.

        Args:
            pts:            Dense spline samples (N, 3).
            cum:            Cumulative arc-length array (N,).
            pGLL_array:     Gate centers (remaining gates after p_tll_index).
            y_GBL_array:    Gate yaws.
            skip_approach:  If True, skip approach-side violations.

        Returns:
            (entry_i, exit_i, gate_center, gate_yaw) or all None.
        """
        if len(pGLL_array) == 0:
            return None, None, None, None

        # Compute the arc-length where each gate is threaded by the trajectory.
        # This is the spline sample closest to the gate center.
        gate_arclens = []
        for gc in pGLL_array:
            dists = np.linalg.norm(pts - gc, axis=1)
            gate_arclens.append(cum[int(np.argmin(dists))])
        gate_arclens = np.array(gate_arclens)

        inside = False
        entry_i = None
        entry_c = None
        entry_yaw = None

        for i, p in enumerate(pts):
            hit, gate_c, gate_yaw = self._check_gate(p, pGLL_array, y_GBL_array)
            if hit:
                if skip_approach:
                    # Identify which gate was hit
                    d_to_gates = np.linalg.norm(pGLL_array - gate_c, axis=1)
                    hit_gate_idx = int(np.argmin(d_to_gates))
                    s_hit_gate = gate_arclens[hit_gate_idx]

                    # If we're BEFORE the gate threading arc-length, it's approach-side
                    if cum[i] < s_hit_gate:
                        continue  # Skip the approach grazing

                if not inside:
                    inside = True
                    entry_i = i
                    entry_c = gate_c
                    entry_yaw = gate_yaw
                continue
            if inside:
                return entry_i, i, entry_c, entry_yaw

        if inside:
            return entry_i, len(pts) - 1, entry_c, entry_yaw
        return None, None, None, None

    def _evaluate_gate_branch(
        self,
        wps: np.ndarray,
        new_wp: np.ndarray,
        s_detour: float,
        push_vector: np.ndarray,
        target_gate_c: np.ndarray,
        target_gate_yaw: float,
        t_elapsed: float,
        pGLL_array: np.ndarray,
        y_GBL_array: np.ndarray,
        p_oll_array: np.ndarray,
        local_radius: float = 0.8,
        max_iter: int = 5,
        extra_push: float = 0.15,
    ) -> dict:
        """Insert detour, iteratively clear the target gate, then score."""
        # Insert the initial detour
        try:
            test_wps = self._insert_detour(wps, new_wp, s_detour, t_elapsed)
        except Exception:
            return {
                "wps": wps,
                "gate_hits": 999,
                "obst_hits": 999,
                "local_len": float("inf"),
                "success": False,
            }

        # Iterative cleanup: find new violation midpoints, push them out
        for it in range(max_iter):
            try:
                spline, t_sample = self._create_spline(test_wps, t_elapsed)
            except Exception:
                return {
                    "wps": test_wps,
                    "gate_hits": 999,
                    "obst_hits": 999,
                    "local_len": float("inf"),
                    "success": False,
                }

            t_dense = np.linspace(0, t_sample[-1], len(t_sample) * 4)
            pts = spline(t_dense)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])

            # Find next violation of THIS gate only
            entry_i, exit_i = self._find_target_gate_violation(pts, target_gate_c, target_gate_yaw)
            if entry_i is None:
                break  # Target gate is clear

            # Midpoint of the new violation
            p_mid_new = (pts[entry_i] + pts[exit_i]) / 2
            push_vector_iter = (p_mid_new - target_gate_c) / np.linalg.norm(
                p_mid_new - target_gate_c
            )
            # Push the midpoint outward in the chosen direction
            # The distance needed is roughly the gap from p_mid_new to the outer frame
            # in the push direction. For simplicity, use a fixed extra_push.
            extra_push = self._get_gate_push(
                p_mid_new, target_gate_c, target_gate_yaw, push_vector_iter
            )
            push_wp = p_mid_new + extra_push * push_vector_iter
            s_new = 0.5 * (cum[entry_i] + cum[exit_i])
            # NEW DIAGNOSTIC:
            print(
                f"    [iter {it}] cleanup wp = {push_wp}  "
                f"push_dir = {push_vector_iter}  push_len = {extra_push:.3f}"
            )

            try:
                test_wps = self._insert_detour(test_wps, push_wp, s_new, t_elapsed)
            except Exception:
                return {
                    "wps": test_wps,
                    "gate_hits": 999,
                    "obst_hits": 999,
                    "local_len": float("inf"),
                    "success": False,
                }

        # Score the final result
        try:
            spline, t_sample = self._create_spline(test_wps, t_elapsed)
        except Exception:
            return {
                "wps": test_wps,
                "gate_hits": 999,
                "obst_hits": 999,
                "local_len": float("inf"),
                "success": False,
            }

        t_dense = np.linspace(0, t_sample[-1], len(t_sample) * 4)
        pts = spline(t_dense)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])

        # Use the original detour position as the local window center
        dists = np.linalg.norm(pts - new_wp, axis=1)
        idx_wp = int(np.argmin(dists))
        s_wp_dense = cum[idx_wp]

        s_min = s_wp_dense - local_radius
        s_max = s_wp_dense + local_radius
        in_window = (cum >= s_min) & (cum <= s_max)
        window_pts = pts[in_window]
        window_cum = cum[in_window]

        # Count gate-frame violations
        gate_hits = 0
        inside_gate = False
        for p in window_pts:
            hit, _, _ = self._check_gate(p, pGLL_array, y_GBL_array)
            if hit:
                if not inside_gate:
                    gate_hits += 1
                    inside_gate = True
            else:
                inside_gate = False

        # Count obstacle violations
        obst_hits = 0
        inside_obst = False
        if len(p_oll_array) > 0:
            for p in window_pts:
                hit, _ = self._check_obsticle(p, p_oll_array)
                if hit:
                    if not inside_obst:
                        obst_hits += 1
                        inside_obst = True
                else:
                    inside_obst = False

        # 6. Count gate-frame violations in the window
        gate_hits = 0
        inside_gate = False
        violation_arclens = []  # NEW: track where violations happen
        for j, p in enumerate(window_pts):
            hit, _, _ = self._check_gate(p, pGLL_array, y_GBL_array)
            if hit:
                violation_arclens.append(window_cum[j])  # NEW
                if not inside_gate:
                    gate_hits += 1
                    inside_gate = True
            else:
                inside_gate = False

        # NEW: print where the violations are
        if violation_arclens:
            print(
                f"    [diag] violations at arc-length "
                f"{min(violation_arclens):.2f}..{max(violation_arclens):.2f}, "
                f"n={len(violation_arclens)}, "
                f"detour anchor at {s_wp_dense:.2f}"
            )

        local_len = float(window_cum[-1] - window_cum[0]) if len(window_cum) > 1 else 0.0

        # DEBUG: plot this branch's result
        # self._plot_gate_branch(
        #    wps=test_wps,
        #    spline=spline,
        #    t_dense=t_dense,
        #    target_gate_c=target_gate_c,
        #    target_gate_yaw=target_gate_yaw,
        #    pGLL_array=pGLL_array,
        #    y_GBL_array=y_GBL_array,
        #    p_oll_array=p_oll_array,
        #    branch_name=branch_name,
        # )

        return {
            "wps": test_wps,
            "gate_hits": gate_hits,
            "obst_hits": obst_hits,
            "local_len": local_len,
            "success": True,
        }

    def _insert_detour(
        self, wps: np.ndarray, new_wp: np.ndarray, s_target: float, t_elapsed: float
    ) -> np.ndarray:
        """Insert new_wp into wps at the position matching s_target arc-length."""
        spline, t_sample = self._create_spline(wps, t_elapsed)
        t_dense = np.linspace(0, t_sample[-1], len(t_sample) * 4)
        pts = spline(t_dense)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        t_knots = spline.x
        kept_wps = spline(t_knots)
        s_wp = np.interp(t_knots, t_dense, cum)

        items = [(s_wp[k], kept_wps[k]) for k in range(len(t_knots))]
        items.append((s_target, new_wp))
        items.sort(key=lambda it: it[0])
        return np.array([pt for _, pt in items])

    def _avoid_gates_tree(
        self,
        p_WLL_array: np.ndarray,
        pGLL_array: np.ndarray,
        y_GBL_array: np.ndarray,
        p_oll_array: np.ndarray,
        t_elapsed: float,
    ) -> np.ndarray:
        """Resolve all gate frame violations using 3-way branching per violation.

        For each gate violation:
        1. Compute three candidate detour waypoints (Left, Right, Top of gate)
        2. For each candidate, insert and iteratively clear the target gate
        3. Pick the best branch by (gate_hits, obst_hits, local_len)
        4. Apply the best branch's waypoints and continue

        Loops until no gate violations remain or _MAX_AVOID_ITER hit.

        Args:
            p_WLL_array:  Current waypoint list (output of _build_waypoints).
            pGLL_array:   Gate centers.
            y_GBL_array:  Gate yaws.
            p_oll_array:   Obstacle centers (used in scoring, not avoided here).
            t_elapsed:    Current race time.

        Returns:
            Modified waypoint list with gate detours inserted.
        """
        wps = p_WLL_array.copy()

        for outer_iter in range(_MAX_AVOID_ITER):
            # Build spline, find first gate violation
            spline, t_sample = self._create_spline(wps, t_elapsed)
            t_dense = np.linspace(0, t_sample[-1], len(t_sample) * 4)
            pts = spline(t_dense)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])

            # self._plot_full_trajectory(
            #    wps=wps,
            #    spline=spline,
            #    t_dense=t_dense,
            #    pts=pts,
            #    pGLL_array=pGLL_array,
            #    y_GBL_array=y_GBL_array,
            #    p_oll_array=p_oll_array,
            #    tag=f"iter {outer_iter}",
            #    block=True,    # script pauses until you close the window
            # )

            entry_i, exit_i, gate_c, gate_yaw = self._find_first_gate_violation(
                pts, cum, pGLL_array, y_GBL_array, skip_approach=True
            )
            if entry_i is None:
                print(f"[gate tree] no more gate violations after {outer_iter} iter(s)")
                return wps

            # Arc-length position for the initial detour insertion
            s_detour = 0.5 * (cum[entry_i] + cum[exit_i])

            # Three candidate detour waypoints, anchored to gate center
            push_L, push_R, push_T = self._compute_3_gate_push_directions(gate_yaw)
            wp_L = self._compute_gate_detour_waypoint(gate_c, gate_yaw, push_L)
            wp_R = self._compute_gate_detour_waypoint(gate_c, gate_yaw, push_R)
            wp_T = self._compute_gate_detour_waypoint(gate_c, gate_yaw, push_T)

            # Evaluate each branch (insert + iterative cleanup + score)
            branch_L = self._evaluate_gate_branch(
                wps,
                wp_L,
                s_detour,
                push_L,
                gate_c,
                gate_yaw,
                t_elapsed,
                pGLL_array,
                y_GBL_array,
                p_oll_array,
            )
            branch_R = self._evaluate_gate_branch(
                wps,
                wp_R,
                s_detour,
                push_R,
                gate_c,
                gate_yaw,
                t_elapsed,
                pGLL_array,
                y_GBL_array,
                p_oll_array,
            )
            branch_T = self._evaluate_gate_branch(
                wps,
                wp_T,
                s_detour,
                push_T,
                gate_c,
                gate_yaw,
                t_elapsed,
                pGLL_array,
                y_GBL_array,
                p_oll_array,
            )

            # Print scores
            print(
                f"\n[gate tree iter {outer_iter}] gate at {gate_c}, "
                f"violation samples {entry_i}..{exit_i}"
            )
            print(
                f"  Left  wps={len(branch_L['wps'])}: "
                f"gate_hits={branch_L['gate_hits']}, "
                f"obst_hits={branch_L['obst_hits']}, "
                f"local_len={branch_L['local_len']:.2f}"
            )
            print(
                f"  Right wps={len(branch_R['wps'])}: "
                f"gate_hits={branch_R['gate_hits']}, "
                f"obst_hits={branch_R['obst_hits']}, "
                f"local_len={branch_R['local_len']:.2f}"
            )
            print(
                f"  Top   wps={len(branch_T['wps'])}: "
                f"gate_hits={branch_T['gate_hits']}, "
                f"obst_hits={branch_T['obst_hits']}, "
                f"local_len={branch_T['local_len']:.2f}"
            )

            # Pick the best branch
            winner, winner_name = self._pick_best_gate_branch(branch_L, branch_R, branch_T)
            print(f"  -> picked {winner_name}")

            # Apply the winning branch's waypoints and continue
            wps = winner["wps"]

        print(f"[gate tree] hit max iterations ({_MAX_AVOID_ITER})")
        return wps

    def _pick_best_gate_branch(
        self, branch_L: dict, branch_R: dict, branch_T: dict
    ) -> tuple[dict, str]:
        """Pick the best of three gate-detour branches.

        Priority:
        1. Fewest gate hits (most important — actually clears the gate frame)
        2. Fewest obstacle hits (avoids picking a branch that flies into a pillar)
        3. Shortest local arc length (tiebreaker — pick the most efficient path)

        Args:
            branch_L:  Dict from _evaluate_gate_branch for the left detour.
            branch_R:  Dict from _evaluate_gate_branch for the right detour.
            branch_T:  Dict from _evaluate_gate_branch for the top detour.

        Returns:
            (winning_branch, name):  The winning branch dict and its name string.
        """
        candidates = [("Left", branch_L), ("Right", branch_R), ("Top", branch_T)]
        candidates.sort(key=lambda nb: (nb[1]["gate_hits"], nb[1]["obst_hits"], nb[1]["local_len"]))
        winner_name, winner = candidates[0]
        return winner, winner_name

    def _compute_3_gate_push_directions(
        self, gate_yaw: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return three unit push vectors in world frame: (left, right, top).

        These are the three sensible directions to displace a detour waypoint
        away from a gate-frame violation. Bottom is omitted because gate posts
        typically extend below the opening.

        Args:
            gate_yaw:   Yaw of the gate (rotation about world z).

        Returns:
            push_L:     3D unit vector pointing to the gate's left side.
            push_R:     3D unit vector pointing to the gate's right side.
            push_T:     3D unit vector pointing up (world z).
        """
        push_L = np.array([-np.sin(gate_yaw), np.cos(gate_yaw), 0.0])
        push_R = -push_L
        push_T = np.array([0.0, 0.0, 1.0])
        return push_L, push_R, push_T

    def _find_target_gate_violation(
        self, pts: np.ndarray, target_gate_c: np.ndarray, target_gate_yaw: float
    ) -> tuple[int, int] | tuple[None, None]:
        """Find the first violation of a SPECIFIC gate in pts.

        Args:
            pts:              Dense samples (N, 3).
            target_gate_c:    3D center of the target gate.
            target_gate_yaw:  Yaw of the target gate.

        Returns:
            (entry_i, exit_i) for the first violation, or (None, None).
        """
        target_arr = np.array([target_gate_c])
        yaw_arr = np.array([target_gate_yaw])

        inside = False
        entry_i = None

        for i, p in enumerate(pts):
            hit, _, _ = self._check_gate(p, target_arr, yaw_arr)
            if hit:
                if not inside:
                    inside = True
                    entry_i = i
                continue
            if inside:
                return entry_i, i

        if inside:
            return entry_i, len(pts) - 1
        return None, None

    def _pick_best_gate_branch(
        self, branch_L: dict, branch_R: dict, branch_T: dict
    ) -> tuple[dict, str]:
        """Pick the best of three gate-detour branches.

        Priority:
        1. Fewest gate hits
        2. Fewest obstacle hits
        3. Shortest local arc length

        Args:
            branch_L:  Dict from _evaluate_gate_branch for the left detour.
            branch_R:  Dict from _evaluate_gate_branch for the right detour.
            branch_T:  Dict from _evaluate_gate_branch for the top detour.

        Returns:
            (winning_branch_dict, branch_name)
        """
        candidates = [("Left", branch_L), ("Right", branch_R), ("Top", branch_T)]
        # Sort by (gate_hits, obst_hits, local_len) ascending
        candidates.sort(key=lambda nb: (nb[1]["gate_hits"], nb[1]["obst_hits"], nb[1]["local_len"]))
        winner_name, winner = candidates[0]
        return winner, winner_name

    def _avoid_collisions(
        self,
        p_WLL_array: np.ndarray,
        p_oll_array: np.ndarray,
        pGLL_array: np.ndarray,
        y_GBL_array: np.ndarray,
        t_elapsed: float,
    ) -> np.ndarray:
        """Avoids obsticles or gate frames by setting waypoints around them.

        Args:
            p_WLL_array:            Waypoints to be passed through.
            p_oll_array:             Obsticle positions.
            pGLL_array:             Gate positions.
            y_GBL_array:            Gate orientations.
            t_elapsed:              Time elapsed in the race so far.

        Returns:
            p_WLL_array:            Waypoints to be passed through, with added waypoints
                                    to avoid obsticles and gate frames.
        """
        wps = p_WLL_array.copy()

        for _ in range(_MAX_AVOID_ITER):
            spline, t_sample = self._create_spline(wps, t_elapsed)
            t_dense = np.linspace(0, t_sample[-1], len(t_sample) * 4)  # change later
            pts = spline(t_dense)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            t_gates = spline.x
            kept_wps = spline(t_gates)
            s_wp = np.interp(t_gates, t_dense, cum)

            # Init helping variables
            detours = []
            inside_obst = False
            inside_gate = False
            entry_i = None
            entry_obst_c = None

            entry_gate_c = None
            entry_gate_yaw = None

            # Check each point from dense Spline for collision with obsticle
            for i, p in enumerate(pts):
                hit_obsticle, obsticle_centre = self._check_obsticle(p, p_oll_array)
                hit_gate, gate_centre, gate_yaw = self._check_gate(p, pGLL_array, y_GBL_array)

                if hit_obsticle:
                    if not inside_obst:
                        inside_obst = True
                        entry_i = i
                        entry_obst_c = obsticle_centre[:2]
                    continue

                if False:  # hit_gate:
                    if not inside_gate:
                        inside_gate = True
                        entry_i = i
                        entry_gate_c = gate_centre
                        entry_gate_yaw = gate_yaw
                        # print('this gate was hit', entry_gate_c, entry_gate_yaw)
                    continue

                if inside_obst:
                    inside_obst = False
                    p_in, p_out = pts[entry_i], p
                    p_mid = (p_out + p_in) / 2
                    # p_mid = pts[(entry_i + i) // 2]
                    # breakpoint()
                    # 2D radial push around obsticle
                    bis = (p_in[:2] - entry_obst_c) + (p_out[:2] - entry_obst_c)
                    nb = np.linalg.norm(bis)

                    push_vector = bis / nb
                    # breakpoint()
                    push_length = self._get_obsticle_push(p_mid.copy(), entry_obst_c, push_vector)
                    # print(push_length)
                    # breakpoint()
                    new_xy = p_mid[:2] + push_length * push_vector
                    # new_xy = p_mid[:2] + 0.2 * push_vector
                    new_wp = [new_xy[0], new_xy[1], 0.5 * (p_in[2] + p_out[2])]

                    # ---- Local spline check: would the resulting trajectory hit a gate? ----
                    s_detour = 0.5 * (cum[entry_i] + cum[i])
                    local_hits = self._local_spline_hits_gate(
                        wps, new_wp, s_detour, t_elapsed, pGLL_array, y_GBL_array, local_radius=0.5
                    )

                    if local_hits:
                        # breakpoint()
                        # Flip push to the opposite side of the obstacle
                        push_vector = -push_vector
                        push_length = self._get_obsticle_push(
                            p_mid.copy(), entry_obst_c, push_vector
                        )
                        new_xy = p_mid[:2] + push_length * push_vector
                        new_wp = np.array([new_xy[0], new_xy[1], 0.5 * (p_in[2] + p_out[2])])
                        # breakpoint()
                    detours.append((0.5 * (cum[entry_i] + cum[i]), new_wp))

                if inside_gate:
                    inside_gate = False
                    p_in, p_out = pts[entry_i], p
                    p_mid = (p_out + p_in) / 2
                    # breakpoint()
                    # 2D radial push around obsticle
                    push_vector = (p_mid - entry_gate_c) / np.linalg.norm(p_mid - entry_gate_c)
                    # push_vector = self._get_gate_push_vector(
                    #     entry_gate_c, entry_gate_yaw, p_in, p_out, p_mid
                    # )
                    push_length = self._get_gate_push(
                        p_mid.copy(), entry_gate_c, entry_gate_yaw, push_vector
                    )
                    # push_length = 0.72
                    # print(push_length)
                    new_xy = p_mid + push_length * push_vector
                    new_wp = np.array([new_xy[0], new_xy[1], 0.5 * (p_in[2] + p_out[2])])
                    detours.append((0.5 * (cum[entry_i] + cum[i]), new_wp))

            if not detours:
                return wps
            # print(detours)
            items = [(s_wp[k], kept_wps[k]) for k in range(len(t_gates))] + detours
            items.sort(key=lambda it: it[0])
            wps = np.array([pt for _, pt in items])

        return wps

    def _plot_gate_branch(
        self,
        wps: np.ndarray,
        spline: CubicSpline,
        t_dense: np.ndarray,
        target_gate_c: np.ndarray,
        target_gate_yaw: float,
        pGLL_array: np.ndarray,
        y_GBL_array: np.ndarray,
        p_oll_array: np.ndarray,
        branch_name: str,
        save_dir: str = "gate_branch_debug",
    ) -> str:
        """Plot the gate-branch result for visual debugging.

        Shows the target gate, all waypoints, the resulting trajectory, and any
        violation samples. Saves two views (3D and top-down) as one PNG.

        Args:
            wps:             Final waypoint list for this branch.
            spline:          The cubic spline (returned by _create_spline).
            t_dense:         Dense time sampling.
            target_gate_c:   Center of the target gate.
            target_gate_yaw: Yaw of the target gate.
            pGLL_array:      All gate centers.
            y_GBL_array:     All gate yaws.
            p_oll_array:      Obstacle centers.
            branch_name:     Label for the plot title (e.g., "Left", "Right", "Top").
            save_dir:        Output directory.

        Returns:
            Path to the saved PNG.
        """
        import os

        import matplotlib.pyplot as plt

        os.makedirs(save_dir, exist_ok=True)

        # Constants — match planner.py
        from lsy_drone_racing.control.planner.planner import (
            CLEARANCE,
            FRAME_OPENING,
            FRAME_WIDTH,
            R_OBSTACLE,
        )

        # Sample the trajectory
        pts = spline(t_dense)

        # Find which samples violate the target gate
        target_arr = np.array([target_gate_c])
        yaw_arr = np.array([target_gate_yaw])
        violation_mask = np.zeros(len(pts), dtype=bool)
        for i, p in enumerate(pts):
            hit, _, _ = self._check_gate(p, target_arr, yaw_arr)
            violation_mask[i] = hit
        violation_pts = pts[violation_mask]

        # ----- Set up figure with 2 subplots: 3D and top-down -----
        fig = plt.figure(figsize=(16, 8))
        ax3d = fig.add_subplot(121, projection="3d")
        ax_top = fig.add_subplot(122)

        # ----- 3D view -----
        # Trajectory
        ax3d.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            "-",
            color="tab:green",
            lw=2.0,
            label=f"trajectory ({len(pts)} pts)",
        )

        # Violation samples
        if len(violation_pts):
            ax3d.scatter(
                violation_pts[:, 0],
                violation_pts[:, 1],
                violation_pts[:, 2],
                c="red",
                s=15,
                marker="x",
                label=f"violations ({len(violation_pts)})",
            )

        # Waypoints (numbered)
        ax3d.scatter(
            wps[:, 0],
            wps[:, 1],
            wps[:, 2],
            c="orange",
            marker="D",
            s=60,
            edgecolor="k",
            depthshade=False,
            label=f"waypoints ({len(wps)})",
        )
        for i, p in enumerate(wps):
            ax3d.text(p[0], p[1], p[2] + 0.03, str(i), fontsize=8)

        # Target gate frame: outer + opening
        self._plot_gate_frame_3d(
            ax3d, target_gate_c, target_gate_yaw, FRAME_WIDTH / 2, "tab:blue", lw=2.5
        )
        self._plot_gate_frame_3d(
            ax3d, target_gate_c, target_gate_yaw, FRAME_OPENING / 2, "tab:cyan", lw=1.5
        )

        # Other gates (faded)
        for k, (gc, gy) in enumerate(zip(pGLL_array, y_GBL_array)):
            if np.allclose(gc, target_gate_c):
                continue
            self._plot_gate_frame_3d(ax3d, gc, gy, FRAME_WIDTH / 2, "gray", lw=1.0, alpha=0.4)

        # Obstacles (just markers in 3D for clarity)
        if len(p_oll_array):
            ax3d.scatter(
                p_oll_array[:, 0],
                p_oll_array[:, 1],
                p_oll_array[:, 2],
                c="firebrick",
                s=60,
                marker="^",
                alpha=0.7,
                label="obstacles",
            )

        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        ax3d.set_title(f"3D view — branch {branch_name}")
        ax3d.legend(loc="upper left", fontsize=8)

        # Zoom to the target gate area
        pad = 1.0
        ax3d.set_xlim(target_gate_c[0] - pad, target_gate_c[0] + pad)
        ax3d.set_ylim(target_gate_c[1] - pad, target_gate_c[1] + pad)
        ax3d.set_zlim(target_gate_c[2] - pad, target_gate_c[2] + pad)

        # ----- Top-down view (xy) -----
        # Trajectory
        ax_top.plot(pts[:, 0], pts[:, 1], "-", color="tab:green", lw=2.0, label="trajectory")

        # Violations
        if len(violation_pts):
            ax_top.scatter(
                violation_pts[:, 0],
                violation_pts[:, 1],
                c="red",
                s=20,
                marker="x",
                label="violations",
            )

        # Waypoints
        ax_top.scatter(
            wps[:, 0],
            wps[:, 1],
            c="orange",
            marker="D",
            s=70,
            edgecolor="k",
            label="waypoints",
            zorder=5,
        )
        for i, p in enumerate(wps):
            ax_top.annotate(
                str(i), (p[0], p[1]), textcoords="offset points", xytext=(5, 5), fontsize=8
            )

        # Target gate frame in top-down (just a line segment along the gate width)
        w_dir = np.array([-np.sin(target_gate_yaw), np.cos(target_gate_yaw)])
        g_xy = target_gate_c[:2]
        outer_a = g_xy - (FRAME_WIDTH / 2) * w_dir
        outer_b = g_xy + (FRAME_WIDTH / 2) * w_dir
        open_a = g_xy - (FRAME_OPENING / 2) * w_dir
        open_b = g_xy + (FRAME_OPENING / 2) * w_dir

        # Outer frame: blue line through the full width
        ax_top.plot(
            [outer_a[0], outer_b[0]],
            [outer_a[1], outer_b[1]],
            color="tab:blue",
            lw=3,
            label="gate outer frame",
        )
        # Opening: cyan line through the opening width
        ax_top.plot(
            [open_a[0], open_b[0]],
            [open_a[1], open_b[1]],
            color="tab:cyan",
            lw=4,
            label="gate opening",
        )
        # Gate center
        ax_top.scatter(
            [g_xy[0]],
            [g_xy[1]],
            c="blue",
            s=80,
            marker="o",
            edgecolor="k",
            zorder=6,
            label="gate center",
        )

        # Obstacles (keep-out shells)
        for o in p_oll_array:
            ax_top.add_patch(
                plt.Circle((o[0], o[1]), R_OBSTACLE + CLEARANCE, color="orange", alpha=0.2)
            )
            ax_top.add_patch(plt.Circle((o[0], o[1]), 0.15, color="firebrick", alpha=0.6))

        # Other gates (faded lines)
        for gc, gy in zip(pGLL_array, y_GBL_array):
            if np.allclose(gc, target_gate_c):
                continue
            ow_dir = np.array([-np.sin(gy), np.cos(gy)])
            gxy = gc[:2]
            oa = gxy - (FRAME_WIDTH / 2) * ow_dir
            ob = gxy + (FRAME_WIDTH / 2) * ow_dir
            ax_top.plot([oa[0], ob[0]], [oa[1], ob[1]], color="gray", lw=1.0, alpha=0.4)

        ax_top.set_xlabel("x")
        ax_top.set_ylabel("y")
        ax_top.set_title(f"Top-down (xy) — branch {branch_name}")
        ax_top.set_xlim(target_gate_c[0] - pad, target_gate_c[0] + pad)
        ax_top.set_ylim(target_gate_c[1] - pad, target_gate_c[1] + pad)
        ax_top.set_aspect("equal")
        ax_top.grid(alpha=0.3)
        ax_top.legend(loc="upper left", fontsize=8)

        fig.suptitle(
            f"Gate branch '{branch_name}': "
            f"{len(violation_pts)} violation samples, {len(wps)} waypoints",
            fontsize=12,
        )

        save_path = os.path.join(save_dir, f"branch_{branch_name}.png")
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        print(f"    [plot] wrote {save_path}")
        plt.show(block=False)
        plt.pause(0.001)
        return save_path

    def _plot_gate_frame_3d(
        self,
        ax: object,
        c: np.ndarray,
        yaw: float,
        half: float,
        color: str,
        lw: float = 2.0,
        alpha: float = 1.0,
    ) -> None:
        """Helper: draw a square gate frame in 3D matplotlib axes."""
        w = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        zz = np.array([0.0, 0.0, 1.0])
        corners = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
        pts = np.array([np.asarray(c) + a * half * w + b * half * zz for a, b in corners])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=lw, alpha=alpha)

    def _plot_full_trajectory(
        self,
        wps: np.ndarray,
        spline: CubicSpline,
        t_dense: np.ndarray,
        pts: np.ndarray,
        pGLL_array: np.ndarray,
        y_GBL_array: np.ndarray,
        p_oll_array: np.ndarray,
        tag: str = "",
        block: bool = True,
    ) -> None:
        """Plot the full trajectory + waypoints + gates + obstacles in a popup window.

        Used for debugging _avoid_gates_tree — shows the global state of the planner
        at each outer iteration.

        Args:
            wps:          Current waypoint list.
            spline:       Cubic spline through wps.
            t_dense:      Dense time sampling array.
            pts:          Pre-sampled trajectory points (spline(t_dense)).
            pGLL_array:   Gate centers.
            y_GBL_array:  Gate yaws.
            p_oll_array:   Obstacle centers.
            tag:          Title suffix (e.g. "iter 0").
            block:        If True, pause script until window closed.
        """
        import matplotlib.pyplot as plt
        from lsy_drone_racing.control.planner.planner import (
            CLEARANCE,
            FRAME_OPENING,
            FRAME_WIDTH,
            R_OBSTACLE,
        )

        # Find violation samples (any gate frame)
        violation_mask = np.zeros(len(pts), dtype=bool)
        for i, p in enumerate(pts):
            hit, _, _ = self._check_gate(p, pGLL_array, y_GBL_array)
            violation_mask[i] = hit
        violation_pts = pts[violation_mask]

        # ----- Figure with 3D and top-down -----
        fig = plt.figure(figsize=(16, 8))
        ax3d = fig.add_subplot(121, projection="3d")
        ax_top = fig.add_subplot(122)

        # ----- 3D view -----
        ax3d.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            "-",
            color="tab:green",
            lw=2.0,
            label=f"trajectory ({len(pts)} pts)",
        )

        if len(violation_pts):
            ax3d.scatter(
                violation_pts[:, 0],
                violation_pts[:, 1],
                violation_pts[:, 2],
                c="red",
                s=18,
                marker="x",
                label=f"violations ({len(violation_pts)})",
            )

        ax3d.scatter(
            wps[:, 0],
            wps[:, 1],
            wps[:, 2],
            c="orange",
            marker="D",
            s=55,
            edgecolor="k",
            depthshade=False,
            label=f"waypoints ({len(wps)})",
        )
        for i, p in enumerate(wps):
            ax3d.text(p[0], p[1], p[2] + 0.04, str(i), fontsize=7)

        # Drone start (assume first waypoint)
        if len(wps):
            ax3d.scatter(
                wps[0, 0], wps[0, 1], wps[0, 2], c="black", s=100, marker="o", label="start"
            )

        # All gates
        for k, (gc, gy) in enumerate(zip(pGLL_array, y_GBL_array)):
            self._plot_gate_frame_3d(ax3d, gc, gy, FRAME_WIDTH / 2, "tab:blue", lw=2.0)
            self._plot_gate_frame_3d(ax3d, gc, gy, FRAME_OPENING / 2, "tab:cyan", lw=1.3)
            ax3d.text(
                gc[0], gc[1], gc[2] + 0.15, f"G{k}", fontsize=9, color="tab:blue", weight="bold"
            )

        # Obstacles
        if len(p_oll_array):
            for j, o in enumerate(p_oll_array):
                ax3d.scatter(o[0], o[1], o[2], c="firebrick", s=60, marker="^", alpha=0.7)
                ax3d.text(o[0], o[1], 2.0, f"O{j}", fontsize=8, color="firebrick")

        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        ax3d.set_title(f"3D view {tag}")
        ax3d.legend(loc="upper left", fontsize=8)

        # ----- Top-down view -----
        ax_top.plot(pts[:, 0], pts[:, 1], "-", color="tab:green", lw=2.0, label="trajectory")

        if len(violation_pts):
            ax_top.scatter(
                violation_pts[:, 0],
                violation_pts[:, 1],
                c="red",
                s=22,
                marker="x",
                label=f"violations ({len(violation_pts)})",
            )

        ax_top.scatter(
            wps[:, 0],
            wps[:, 1],
            c="orange",
            marker="D",
            s=70,
            edgecolor="k",
            label=f"waypoints ({len(wps)})",
            zorder=5,
        )
        for i, p in enumerate(wps):
            ax_top.annotate(
                str(i), (p[0], p[1]), textcoords="offset points", xytext=(5, 5), fontsize=7
            )

        if len(wps):
            ax_top.scatter(
                wps[0, 0], wps[0, 1], c="black", s=100, marker="o", label="start", zorder=6
            )

        # Gates in xy (outer + opening lines)
        for k, (gc, gy) in enumerate(zip(pGLL_array, y_GBL_array)):
            w_dir = np.array([-np.sin(gy), np.cos(gy)])
            g_xy = gc[:2]
            outer_a = g_xy - (FRAME_WIDTH / 2) * w_dir
            outer_b = g_xy + (FRAME_WIDTH / 2) * w_dir
            open_a = g_xy - (FRAME_OPENING / 2) * w_dir
            open_b = g_xy + (FRAME_OPENING / 2) * w_dir
            ax_top.plot(
                [outer_a[0], outer_b[0]], [outer_a[1], outer_b[1]], color="tab:blue", lw=2.5
            )
            ax_top.plot([open_a[0], open_b[0]], [open_a[1], open_b[1]], color="tab:cyan", lw=3.5)
            ax_top.scatter(
                [g_xy[0]], [g_xy[1]], c="blue", s=60, marker="o", edgecolor="k", zorder=6
            )
            ax_top.annotate(
                f"G{k}",
                (g_xy[0], g_xy[1]),
                textcoords="offset points",
                xytext=(8, -8),
                fontsize=9,
                color="tab:blue",
                weight="bold",
            )

        # Obstacles in xy
        for j, o in enumerate(p_oll_array):
            ax_top.add_patch(
                plt.Circle((o[0], o[1]), R_OBSTACLE + CLEARANCE, color="orange", alpha=0.2)
            )
            ax_top.add_patch(plt.Circle((o[0], o[1]), 0.15, color="firebrick", alpha=0.6))
            ax_top.annotate(
                f"O{j}",
                (o[0], o[1]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                color="firebrick",
            )

        ax_top.set_xlabel("x")
        ax_top.set_ylabel("y")
        ax_top.set_title(f"Top-down view {tag}")
        ax_top.set_aspect("equal")
        ax_top.grid(alpha=0.3)
        ax_top.legend(loc="upper left", fontsize=8)

        fig.suptitle(f"Avoid gates tree — {tag}", fontsize=12)
        fig.tight_layout()

        plt.show(block=block)
