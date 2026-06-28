"""Subclass definition of new planner."""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R



from lsy_drone_racing.control.env_obs import EnvState_t
from lsy_drone_racing.control.Planner.planner import Planner, Trajectory, R_OBSTACLE

_MAX_AVOID_ITER = 20     # maximum number of iterations to avoid obstacles


class SplinePlanner(Planner):
    """Class to generate smooth Drone Trajectory for MPC."""
    
    def __init__(
        self,
        obs: EnvState_t,
        info: dict,
        config: dict,
        t_total: float
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

    def plan(
            self,
            obs: EnvState_t,
            t_elapsed: float
    ) -> Trajectory:
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
            self,
            p_WLL_array: np.ndarray,
            t_elapsed: float
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
        t_sample = np.linspace(0, t_remaining, int(np.round(t_remaining*self.freq)))
        
        # Cubic Spline
        spline_ref_array = CubicSpline(t_gates, p_WLL_array, axis=0)

        return spline_ref_array, t_sample
    
    def _build_waypoints(
            self,
            obs: EnvState_t,
            t_elapsed: float
    ) -> np.ndarray:
        """Creates waypoints to avoid hindrances and complete gates.
        
        Args:
            obs:                Observed environment states.
            t_elapsed:          Time passed in the race so far.

        Returns:
            p_WLL_array:        N-dim array of waypoints for the cubic spline.
        """
        # Current drone position
        pDLL = obs.pBLL

        # Read out gates
        pGLL_array, y_GBL_array = self._gate(obs)

        # Read out obstacles
        pOLL_array = obs.pOLL_array

        # Parameter defined to set helping points in front and behind the gates
        Distance = 0.05

        # Create waypoint matrix
        p_WLL_array = pDLL

        pPrevLL = np.zeros(3)
        pNextLL = np.zeros(3)

        for i in range(len(pGLL_array)):
            pPrevLL[0] = pGLL_array[i,0] - Distance*np.cos(y_GBL_array[i])
            pPrevLL[1] = pGLL_array[i,1] - Distance*np.sin(y_GBL_array[i])
            pPrevLL[2] = pGLL_array[i,2]

            p_WLL_array = np.vstack([p_WLL_array, pPrevLL])

            p_WLL_array = np.vstack([p_WLL_array, pGLL_array[i]])

            if i == len(pGLL_array)-1:
                Distance = 1
            
            pNextLL[0] = pGLL_array[i,0] + Distance*np.cos(y_GBL_array[i])
            pNextLL[1] = pGLL_array[i,1] + Distance*np.sin(y_GBL_array[i])
            pNextLL[2] = pGLL_array[i,2]

            p_WLL_array = np.vstack([p_WLL_array, pNextLL])
        #p_WLL_array = self._180_degree_turn(p_WLL_array, 
        #                                    pOLL_array, pGLL_array, y_GBL_array, t_elapsed, obs)
        #p_WLL_array = self._avoid_collisions(p_WLL_array, pOLL_array, pGLL_array, y_GBL_array, t_elapsed)

        p_WLL_array = self._avoidance_tree(p_WLL_array, pOLL_array, pGLL_array, y_GBL_array, t_elapsed)

        return p_WLL_array
    
    def _avoid_collisions(
            self,
            p_WLL_array: np.ndarray,
            pOLL_array: np.ndarray,
            pGLL_array: np.ndarray,
            y_GBL_array: np.ndarray,
            t_elapsed: float
    ) -> np.ndarray:
        """Avoids obsticles or gate frames by setting waypoints around them.
        
        Args:
            p_WLL_array:            Waypoints to be passed through.
            pOLL_array:             Obsticle positions.
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
            t_dense = np.linspace(0, t_sample[-1], len(t_sample) * 4) # change later
            pts = spline(t_dense)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            t_gates  = spline.x
            kept_wps = spline(t_gates)
            s_wp     = np.interp(t_gates, t_dense, cum)

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
                hit_obsticle, obsticle_centre = self._check_obsticle2(p, pOLL_array)
                hit_gate, gate_centre, gate_yaw = self._check_gate3(p, pGLL_array, y_GBL_array)

                if hit_obsticle:
                    if not inside_obst:
                        inside_obst = True
                        entry_i = i
                        entry_obst_c = obsticle_centre[:2]
                    continue

                if False:#hit_gate:
                    if not inside_gate:
                        inside_gate = True
                        entry_i = i
                        entry_gate_c = gate_centre
                        entry_gate_yaw = gate_yaw
                        #print('this gate was hit', entry_gate_c, entry_gate_yaw)
                    continue

                if inside_obst:
                    inside_obst = False
                    p_in, p_out = pts[entry_i], p
                    p_mid = (p_out + p_in)/2
                    mid_idx = (entry_i + i) // 2
                    #p_mid = pts[mid_idx]
                    #breakpoint()
                    # 2D radial push around obsticle
                    bis = (p_in[:2] - entry_obst_c) + (p_out[:2] - entry_obst_c)
                    nb = np.linalg.norm(bis)

                    push_vector = bis/nb
                    #breakpoint()
                    push_length = self._get_obsticle_push(p_mid.copy(), entry_obst_c, push_vector)
                    #print(push_length)
                    #breakpoint()
                    new_xy = p_mid[:2] + push_length * push_vector
                    #new_xy = p_mid[:2] + 0.2 * push_vector
                    new_wp = [new_xy[0], new_xy[1], 0.5 * (p_in[2] + p_out[2])]
                    
                    # ---- Local spline check: would the resulting trajectory hit a gate? ----
                    s_detour = 0.5 * (cum[entry_i] + cum[i])
                    local_hits = self._local_spline_hits_gate(
                        wps, new_wp, s_detour, t_elapsed, pGLL_array, y_GBL_array,
                        local_radius=0.5
                    )

                    if local_hits:
                        #breakpoint()
                        # Flip push to the opposite side of the obstacle
                        push_vector = -push_vector
                        push_length = self._get_obsticle_push(p_mid.copy(), entry_obst_c, push_vector)
                        new_xy = p_mid[:2] + push_length * push_vector
                        new_wp = np.array([new_xy[0], new_xy[1], 0.5 * (p_in[2] + p_out[2])])
                        #breakpoint()
                    detours.append((0.5 * (cum[entry_i] + cum[i]), new_wp))

                if inside_gate:
                    inside_gate = False
                    p_in, p_out = pts[entry_i], p
                    p_mid = (p_out + p_in)/2
                    #breakpoint()
                    # 2D radial push around obsticle
                    push_vector = (p_mid - entry_gate_c)/np.linalg.norm(p_mid - entry_gate_c)
                    #push_vector = self._get_gate_push_vector(entry_gate_c, entry_gate_yaw, p_in, p_out, p_mid)
                    push_length = self._get_gate_push(p_mid.copy(), entry_gate_c, entry_gate_yaw, push_vector)
                    #push_length = 0.72
                    #print(push_length)
                    new_xy = p_mid + push_length * push_vector
                    new_wp = np.array([new_xy[0], new_xy[1], 0.5 * (p_in[2] + p_out[2])])
                    detours.append((0.5 * (cum[entry_i] + cum[i]), new_wp))

            if not detours:
                    return wps
            #print(detours)
            items = [(s_wp[k], kept_wps[k]) for k in range(len(t_gates))] + detours
            items.sort(key=lambda it: it[0])
            wps = np.array([pt for _, pt in items])

        return wps
    
    def _local_spline_hits_gate(self,
                                wps: np.ndarray,
                                new_wp: np.ndarray,
                                s_detour: float,
                                t_elapsed: float,
                                pGLL_array: np.ndarray,
                                y_GBL_array: np.ndarray,
                                local_radius: float = 0.5,
                                n_samples: int = 30
                                ) -> bool:
        """Insert new_wp into wps and check if the resulting spline violates a gate
        in a window of arc-length around the insertion point.

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
            hit, _, _ = self._check_gate3(p, pGLL_array, y_GBL_array)
            if hit:
                return True

        return False
    



    def _avoidance_tree(self,
                    p_WLL_array: np.ndarray,
                    pOLL_array: np.ndarray,
                    pGLL_array: np.ndarray,
                    y_GBL_array: np.ndarray,
                    t_elapsed: float
                    ) -> np.ndarray:
        """Avoid obstacles by trying both detour sides and picking the better one.

        For each obstacle violation:
        1. Compute the two candidate push directions (radially out from obstacle:
            "side A" along the bisector, "side B" the opposite direction).
        2. For each side, build a trial spline with the detour inserted.
        3. Count remaining violations (gates + obstacles) and measure arc length.
        4. Pick the candidate with fewer violations; tie-break on shorter length.

        Args:
            p_WLL_array:        Current waypoint list.
            pOLL_array:         Obstacle centers.
            pGLL_array:         Gate centers.
            y_GBL_array:        Gate yaws.
            t_elapsed:          Time passed during the race.

        Returns:
            p_WLL_array:        Waypoint list with detours inserted.
        """
        wps = p_WLL_array.copy()

        for outer_iter in range(_MAX_AVOID_ITER):
            # Build the current spline and find the FIRST obstacle violation
            spline, t_sample = self._create_spline(wps, t_elapsed)
            t_dense = np.linspace(0, t_sample[-1], len(t_sample) * 4)
            pts = spline(t_dense)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])

            # Find the first violation segment
            entry_i, exit_i, entry_obst_c = self._find_first_obstacle_violation(
                pts, pOLL_array
            )

            if entry_i is None:
                # No violations remaining — done
                return wps

            # Use the midpoint sample of the violation as the detour anchor
            mid_idx = (entry_i + exit_i) // 2
            p_mid = pts[mid_idx].copy()    # NOTE: .copy() so we don't mutate pts

            # Compute two candidate push directions
            push_A, push_B = self._candidate_push_directions(
                pts[entry_i], pts[exit_i], p_mid, entry_obst_c
            )

            # Evaluate both candidates
            candidate_A = self._evaluate_candidate(
                wps, p_mid, entry_obst_c, push_A, entry_i, exit_i, cum,
                pOLL_array, pGLL_array, y_GBL_array, t_elapsed
            )
            candidate_B = self._evaluate_candidate(
                wps, p_mid, entry_obst_c, push_B, entry_i, exit_i, cum,
                pOLL_array, pGLL_array, y_GBL_array, t_elapsed
            )

            # Pick the better one
            chosen = self._pick_better(candidate_A, candidate_B)
            if chosen is None:
                # Neither produced a valid spline — bail
                return wps

            new_wp, s_detour = chosen['new_wp'], chosen['s_detour']

            # Insert into wps at the correct arc-length position
            wps = self._insert_at_arc_length(wps, new_wp, s_detour, t_elapsed)

        return wps


    def _find_first_obstacle_violation(self, pts, pOLL_array):
        """Walk through dense samples and return (entry_i, exit_i, obstacle_center)
        for the first obstacle violation segment. Returns (None, None, None) if no
        violation.
        """
        inside = False
        entry_i = None
        entry_obst_c = None

        for i, p in enumerate(pts):
            hit, obst_c = self._check_obsticle2(p, pOLL_array)
            if hit:
                if not inside:
                    inside = True
                    entry_i = i
                    entry_obst_c = obst_c[:2].copy()
                continue
            if inside:
                # Exit detected
                return entry_i, i, entry_obst_c

        # If still inside at end, treat last sample as exit
        if inside:
            return entry_i, len(pts) - 1, entry_obst_c
        return None, None, None


    def _candidate_push_directions(self, p_in, p_out, p_mid, obst_c):
        """Return two unit push vectors in xy: 'side A' (bisector outward from obstacle)
        and 'side B' (opposite direction).
        """
        # Side A: bisector direction (away from obstacle through midpoint)
        bis = (p_in[:2] - obst_c) + (p_out[:2] - obst_c)
        nb = np.linalg.norm(bis)
        if nb < 1e-9:
            # Degenerate; use perpendicular to chord
            tv = (p_out - p_in)[:2]
            bis = np.array([-tv[1], tv[0]])
            nb = np.linalg.norm(bis) + 1e-9

        push_A = bis / nb
        push_B = -push_A
        return push_A, push_B


    def _evaluate_candidate(self, wps, p_mid, obst_c, push_vec,
                            entry_i, exit_i, cum,
                            pOLL_array, pGLL_array, y_GBL_array, t_elapsed):
        """Build a trial spline with the detour inserted in `push_vec` direction.
        Returns a dict with the candidate's metrics, or None on failure.
        """
        # Use _get_obsticle_push on a COPY so we don't mutate p_mid
        push_length = self._get_obsticle_push(p_mid.copy(), obst_c, push_vec)
        if push_length == 0:
            # The midpoint is already outside the obstacle — push a fixed safe amount
            push_length = R_OBSTACLE + 0.1

        new_xy = p_mid[:2] + push_length * push_vec
        new_wp = np.array([new_xy[0], new_xy[1], p_mid[2]])

        s_detour = 0.5 * (cum[entry_i] + cum[exit_i])

        # Build trial spline with new_wp inserted
        try:
            trial_wps = self._insert_at_arc_length(wps, new_wp, s_detour, t_elapsed)
            trial_spline, trial_t_sample = self._create_spline(trial_wps, t_elapsed)
            trial_t_dense = np.linspace(0, trial_t_sample[-1], len(trial_t_sample) * 4)
            trial_pts = trial_spline(trial_t_dense)
        except Exception:
            return None

        # Count violations
        n_violations = self._count_violations(trial_pts, pOLL_array, pGLL_array, y_GBL_array)

        # Measure arc length
        trial_seg = np.linalg.norm(np.diff(trial_pts, axis=0), axis=1)
        arc_length = trial_seg.sum()

        return {
            'new_wp': new_wp,
            's_detour': s_detour,
            'n_violations': n_violations,
            'arc_length': arc_length,
            'push_vec': push_vec,
        }


    def _count_violations(self, pts, pOLL_array, pGLL_array, y_GBL_array):
        """Count distinct violation segments (obstacles + non-target gates)."""
        n = 0
        inside_obst = False
        inside_gate = False

        for p in pts:
            hit_o, _ = self._check_obsticle2(p, pOLL_array)
            hit_g, _, _ = self._check_gate3(p, pGLL_array, y_GBL_array)

            if hit_o:
                if not inside_obst:
                    n += 1
                    inside_obst = True
            else:
                inside_obst = False

            if hit_g:
                if not inside_gate:
                    n += 1
                    inside_gate = True
            else:
                inside_gate = False

        return n


    def _pick_better(self, candidate_A, candidate_B):
        """Pick the better candidate: fewer violations first, shorter length to break ties."""
        if candidate_A is None and candidate_B is None:
            return None
        if candidate_A is None:
            return candidate_B
        if candidate_B is None:
            return candidate_A

        if candidate_A['n_violations'] < candidate_B['n_violations']:
            return candidate_A
        if candidate_B['n_violations'] < candidate_A['n_violations']:
            return candidate_B
        # Tie: pick shorter
        if candidate_A['arc_length'] <= candidate_B['arc_length']:
            return candidate_A
        return candidate_B


    def _insert_at_arc_length(self, wps, new_wp, s_target, t_elapsed):
        """Insert new_wp into wps at the position matching s_target in arc-length."""
        seg = np.linalg.norm(np.diff(wps, axis=0), axis=1)
        cum_wps = np.concatenate([[0.0], np.cumsum(seg)])
        insert_idx = int(np.searchsorted(cum_wps, s_target))
        return np.insert(wps, insert_idx, new_wp, axis=0)












































    
    def _180_degree_turn(self,
                        p_WLL_array: np.ndarray,
                        pOLL_array: np.ndarray,
                        pGLL_array: np.ndarray,
                        y_GBL_array: np.ndarray,
                        t_elapsed: float,
                        obs: EnvState_t
                    ) -> np.ndarray:
        """Handles 180 degree turns by setting waypoints around the turn.

        Args:
            p_WLL_array:        Waypoints to be passed through.
            pOLL_array:         Obsticles to be avoided.
            pGLL_array:         Gates to be passed through.
            y_GBL_array:        Yaw angles of the gates.
            t_elapsed:          Time passed

        Returns:
            p_WLL_array:        Waypoints to be passed through, avoiding obsticles.
        """
        def wrap_to_pi(a):
            return (a + np.pi) % (2 * np.pi) - np.pi
        def find_row(arr, target, tol=1e-6):
            """Return the index of the first row equal to target, or -1."""
            diffs = np.linalg.norm(arr - target, axis=1)
            idx = int(np.argmin(diffs))
            return idx if diffs[idx] < tol else -1
        D = 0.05
        PUSH = 0.2
        YAW_TOL = 0.6
        OBS_PUSH = 0.2
        MAX_OBS_ITER = 5

        def to_body(p, gate_pos, yaw):
            rel = p - gate_pos
            return np.array([
                np.cos(yaw) * rel[0] + np.sin(yaw) * rel[1],
                -np.sin(yaw) * rel[0] + np.cos(yaw) * rel[1],
                rel[2]
                    ])
        drone_pos = obs.pBLL
        gate_idx = obs.pTLL_index
        check_old_gate = False
        all_gates = obs.pTLL_array
        is_180_turn_old = False
        is_close_old = False
        aligned = False
        if gate_idx >= 1:
            prev_gate = all_gates[gate_idx - 1]
            prev_quads = obs.qTLT_array[gate_idx - 1]
            prev_yaw = R.from_quat(prev_quads).as_euler('ZYX')[0]
            yaw_diff = wrap_to_pi(y_GBL_array[0] - prev_yaw)
            gate_distance = np.linalg.norm((pGLL_array[0] - prev_gate)[:2])
            is_180_turn_old = np.isclose(np.abs(yaw_diff), np.pi, atol=YAW_TOL)
            is_close_old = gate_distance < 1.1
            Next_gate_body = to_body(pGLL_array[0], prev_gate, prev_yaw)

            # Aligned check: gate i+1 should be roughly on gate i's through-axis
            # (small lateral offset) and reasonably close in the through-direction
            aligned = (abs(Next_gate_body[0]) < 0.35      # within 0.5 m of the axis line
                    and abs(Next_gate_body[1]) < 1.3)
            #breakpoint()
            
        #print('ifstatements for the replan', is_180_turn_old, is_close_old, aligned)
        if is_180_turn_old and is_close_old and aligned:
            #print('test')
            # Anchors: drone position itself, and pre-gate-0
            d_prev = np.array([np.cos(prev_yaw), np.sin(prev_yaw), 0.0])
            d_gate0 = np.array([np.cos(y_GBL_array[0]), np.sin(y_GBL_array[0]), 0.0])
            anchor_a = pGLL_array[0] - prev_gate                         
            anchor_b = pGLL_array[0] - D * d_gate0

            # Same geometry as in the gate-to-gate case
            mid = 0.5 * (anchor_a + anchor_b)
            chord_xy = (anchor_b - anchor_a)[:2]
            chord_norm = np.linalg.norm(chord_xy)
            if chord_norm > 1e-6:
                chord_unit = chord_xy / chord_norm
                perp_xy = np.array([-chord_unit[1], chord_unit[0]])
                if np.dot(perp_xy, d_prev[:2]) < 0:
                    perp_xy = -perp_xy

                turn_wp_xy = mid[:2] + (0.5 + PUSH) * perp_xy
                turn_wp = np.array([turn_wp_xy[0], turn_wp_xy[1], mid[2]])

                # Push out from obstacles
                if len(pOLL_array) > 0:
                    for _ in range(MAX_OBS_ITER):
                        d_to_obs = np.linalg.norm(turn_wp[:2] - pOLL_array[:, :2], axis=1)
                        if d_to_obs.min() >= R_OBSTACLE + 0.1:
                            break
                        turn_wp[:2] += OBS_PUSH * perp_xy

                idx_drone = find_row(p_WLL_array, drone_pos)
                

                drone_body = to_body(drone_pos, pGLL_array[0], y_GBL_array[0])
                wp_body = to_body(turn_wp, pGLL_array[0], y_GBL_array[0])
                #breakpoint()
                if idx_drone >= 0 and abs(drone_body[1]) > abs(wp_body[1]):
                    p_WLL_array = np.insert(p_WLL_array, idx_drone + 1, turn_wp, axis=0)
                    print(f'drone-side turn waypoint inserted: {turn_wp}')

        for i in reversed(range(len(pGLL_array) - 1)):
            yaw_diff = wrap_to_pi(y_GBL_array[i + 1] - y_GBL_array[i])
            gate_distance = np.linalg.norm((pGLL_array[i + 1] - pGLL_array[i])[:2])
            is_180_turn = np.isclose(np.abs(yaw_diff), np.pi, atol=YAW_TOL)
            is_close    = gate_distance < 1.3
            Next_gate_body = to_body(pGLL_array[i + 1], pGLL_array[i], y_GBL_array[i])

            # Aligned check: gate i+1 should be roughly on gate i's through-axis
            # (small lateral offset) and reasonably close in the through-direction
            aligned = (abs(Next_gate_body[0]) < 0.35      # within 0.5 m of the axis line
                    and abs(Next_gate_body[1]) < 1.3)  # in front, but not too far
            #print('this is hereasdfasdfasdfasdf', Next_gate_body)
            #print('ifstatements', is_180_turn, is_close, aligned)
            #breakpoint()
            if not (is_180_turn and is_close and aligned):
                continue
            
            # Compute the post-gate-i and pre-gate-(i+1) anchor points
            d_i = np.array([np.cos(y_GBL_array[i]),   np.sin(y_GBL_array[i]),   0.0])
            d_ip1 = np.array([np.cos(y_GBL_array[i+1]), np.sin(y_GBL_array[i+1]), 0.0])
            p_post_i  = pGLL_array[i]   + D * d_i
            p_pre_ip1 = pGLL_array[i+1] - D * d_ip1
            # Build the turn waypoint geometry
            mid = 0.5 * (p_post_i + p_pre_ip1)
            chord_xy = (p_pre_ip1 - p_post_i)[:2]
            chord_norm = np.linalg.norm(chord_xy)
            chord_unit = chord_xy / chord_norm
            perp_xy = np.array([-chord_unit[1], chord_unit[0]])
            if np.dot(perp_xy, d_i[:2]) < 0:
                perp_xy = -perp_xy
            turn_wp_xy = mid[:2] + (0.5 + PUSH) * perp_xy
            turn_wp = np.array([turn_wp_xy[0], turn_wp_xy[1], mid[2]])
            if len(pOLL_array) > 0:
                for _ in range(MAX_OBS_ITER):
                    d_to_obs = np.linalg.norm(turn_wp[:2] - pOLL_array[:, :2], axis=1)
                    if d_to_obs.min() >= R_OBSTACLE + 0.1:
                        break
                    turn_wp[:2] += OBS_PUSH * perp_xy
            idx_post = find_row(p_WLL_array, p_post_i)
            #print(idx_post)
            if idx_post < 0:
                continue
            p_WLL_array = np.insert(p_WLL_array, idx_post + 1, turn_wp, axis=0)
            #breakpoint()
        return p_WLL_array