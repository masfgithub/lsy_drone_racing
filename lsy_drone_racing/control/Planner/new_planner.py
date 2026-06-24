"""Subclass definition of new planner."""

import numpy as np
from scipy.interpolate import CubicSpline
from lsy_drone_racing.control.Planner.planner import FRAME_WIDTH, CLEARANCE
from scipy.spatial.transform import Rotation as R

# In _U_turn:
HALF_FRAME = FRAME_WIDTH / 2                     # 0.36, distance from gate center to frame edge
DETOUR_MARGIN = CLEARANCE + 0.1                  # extra clearance past the frame edge
DETOUR_DIST = HALF_FRAME + DETOUR_MARGIN         # how far from gate center to place detour

from lsy_drone_racing.control.env_obs import EnvState_t
from lsy_drone_racing.control.Planner.planner import Planner, Trajectory

OBSTACLE_RADIUS = 0.15  # radius of the obstacle to avoid
_MAX_AVOID_ITER = 10     # maximum number of iterations to avoid obstacles


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
        # Current dron position
        pDLL = obs.pBLL


        # Read out gates
        pGLL_array, y_GBL_array = self._gate(obs)

        # Read out obsticles
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

        #p_WLL_array = self._obst_near_gate(p_WLL_array, pOLL_array, pGLL_array, y_GBL_array, t_elapsed)
        #print('obst_near_gate done', p_WLL_array)
        p_WLL_array = self._U_turn(p_WLL_array, pOLL_array, pGLL_array, y_GBL_array, t_elapsed, obs)
        #print('U_turn done', p_WLL_array)
        p_WLL_array = self._180_degree_turn(p_WLL_array, 
                                            pOLL_array, pGLL_array, y_GBL_array, t_elapsed, obs)
        #print('180_degree_turn done', p_WLL_array)
        p_WLL_array = self._avoid_obstacles(p_WLL_array, pOLL_array, t_elapsed)
        #print('avoid_obstacles done', p_WLL_array)
        #p_WLL_array = self._detour_gates1(obs, p_WLL_array, pGLL_array, y_GBL_array, t_elapsed)

        return p_WLL_array
    
    def _avoid_obstacles(self,
                        p_WLL_array: np.ndarray,
                        pOLL_array: np.ndarray,
                        t_elapsed: float
                        ) -> np.ndarray:
        """Avoids obsticles by setting waypoints around obsticles.
        
        Args:
            p_WLL_array:        Waypoints to be passed through.
            pOLL_array:         Obsticles to be avoided.
            t_elapsed:          Time passed during race.
            
        Returns:
            p_WLL_array:        Waypoints to be passed through, avoiding obsticles.
        """
        wps = p_WLL_array.copy()
        for _ in range(_MAX_AVOID_ITER):
            # Make a dense Spline
            #spline_test_array, t_sample = self._create_spline(wps, t_elapsed)
            #t_dense = np.linspace(0, t_sample[-1], len(t_sample) * 4)
            #pts = spline_test_array(t_dense)

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
            inside = False
            entry_i = None
            entry_c = None
            entry_push = None

            # Check each point from dense Spline for collision with obsticle
            for i, p in enumerate(pts):
                hit_obst, c_xy_obst, push_obst = self._check_obsticle(p, pOLL_array)

                if hit_obst:
                    if not inside:
                        inside = True
                        entry_i = i
                        entry_c = c_xy_obst
                        entry_push = push_obst
                        
                    continue

                if inside:
                    inside = False
                    p_in, p_out = pts[entry_i], p
                    
                    # 2D radial push around obsticle
                    bis = (p_in[:2] - entry_c) + (p_out[:2] - entry_c)
                    nb = np.linalg.norm(bis)
                    if nb < 1e-6:
                        tv = p_out[:2] - p_in[:2]
                        bis = np.array([-tv[1], tv[0]])
                        nb = np.linalg.norm(bis) + 1e-9
                    new_xy = entry_c + 1.2*entry_push * bis / nb
                    #print('this works', entry_push)
                    new_wp = np.array([new_xy[0], new_xy[1], 0.5 * (p_in[2] + p_out[2])])
                    detours.append((0.5 * (cum[entry_i] + cum[i]), new_wp))

            if not detours:
                    return wps
                
            items = [(s_wp[k], kept_wps[k]) for k in range(len(t_gates))] + detours
            items.sort(key=lambda it: it[0])
            wps = np.array([pt for _, pt in items])


        return wps

    
    def _obst_near_gate(self,
                    p_WLL_array: np.ndarray,
                    pOLL_array: np.ndarray,
                    pGLL_array: np.ndarray,
                    y_GBL_array: np.ndarray,
                    t_elapsed: float
                    ) -> np.ndarray:
        """Inserts detour waypoints around obstacles that sit close to a gate.

        For each gate, checks every obstacle. If an obstacle's xy distance to the
        gate center is below a threshold, inserts two detour waypoints on the
        opposite side of the gate axis from the obstacle. The detour goes either
        before pre-gate or after post-gate depending on whether the obstacle is
        on the entry or exit side of the gate.
        """
        def find_row(arr, target, tol=1e-6):
            diffs = np.linalg.norm(arr - target, axis=1)
            idx = int(np.argmin(diffs))
            return idx if diffs[idx] < tol else -1

        D = 0.05
        NEAR_THRESHOLD = 0.5
        SIDE_PUSH = 0.4
        TRANSITION_FACTOR = 0.5

        if len(pOLL_array) == 0:
            return p_WLL_array

        for i in reversed(range(len(pGLL_array))):
            gate_pos = pGLL_array[i]
            yaw = y_GBL_array[i]

            # Gate body-frame basis vectors expressed in world coordinates
            x_axis = np.array([np.cos(yaw),  np.sin(yaw),  0.0])
            y_axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0])

            for o in range(len(pOLL_array)):
                obs_pos = pOLL_array[o]
                d_xy = np.linalg.norm(obs_pos[:2] - gate_pos[:2])
                if d_xy >= NEAR_THRESHOLD:
                    continue

                # Obstacle in gate body frame
                rel = obs_pos - gate_pos
                body_x = np.dot(rel, x_axis)
                body_y = np.dot(rel, y_axis)

                on_exit_side = body_x > 0.0

                # Place detour on the opposite side of the gate axis from the obstacle
                side_y_body = (-np.sign(body_y) if abs(body_y) > 1e-6 else 1.0) * SIDE_PUSH
                transition_y_body = side_y_body * TRANSITION_FACTOR

                # Build waypoints in world frame
                # (gate_pos is world; x_axis, y_axis are world-frame unit vectors)
                side_wp = gate_pos + body_x * x_axis + side_y_body * y_axis
                side_wp[2] = obs_pos[2]

                transition_wp = gate_pos + body_x * x_axis + transition_y_body * y_axis
                transition_wp[2] = gate_pos[2]

                # Insert relative to pre-gate or post-gate depending on which side
                if on_exit_side:
                    anchor = gate_pos + D * x_axis
                    idx_anchor = find_row(p_WLL_array, anchor)
                    if idx_anchor < 0:
                        continue
                    p_WLL_array = np.insert(p_WLL_array, idx_anchor + 1, side_wp, axis=0)
                    p_WLL_array = np.insert(p_WLL_array, idx_anchor + 2, transition_wp, axis=0)
                else:
                    anchor = gate_pos - D * x_axis
                    idx_anchor = find_row(p_WLL_array, anchor)
                    if idx_anchor < 0:
                        continue
                    p_WLL_array = np.insert(p_WLL_array, idx_anchor, transition_wp, axis=0)
                    p_WLL_array = np.insert(p_WLL_array, idx_anchor, side_wp, axis=0)

        return p_WLL_array
    
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
        # Iterate from the end so earlier insertions don't shift later positions
        drone_yaw = self._drone_yaw(obs)
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
            
        print('ifstatements for the replan', is_180_turn_old, is_close_old, aligned)
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
                        if d_to_obs.min() >= OBSTACLE_RADIUS + 0.1:
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
            print('ifstatements', is_180_turn, is_close, aligned)
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
                    if d_to_obs.min() >= OBSTACLE_RADIUS + 0.1:
                        break
                    turn_wp[:2] += OBS_PUSH * perp_xy
            idx_post = find_row(p_WLL_array, p_post_i)
            #print(idx_post)
            if idx_post < 0:
                continue
            p_WLL_array = np.insert(p_WLL_array, idx_post + 1, turn_wp, axis=0)
            #breakpoint()
        return p_WLL_array
    
    def _drone_yaw(self, obs):
        """Extract drone's yaw from its quaternion."""
        return R.from_quat(obs.qBLB).as_euler('ZYX')[0]

    def _U_turn(self,
            p_WLL_array: np.ndarray,
            pOLL_array: np.ndarray,
            pGLL_array: np.ndarray,
            y_GBL_array: np.ndarray,
            t_elapsed: float,
            obs: EnvState_t
            ) -> np.ndarray:
        """Handles the case where the next gate sits behind the current gate.

        If gate i+1 lies in the negative-x direction (entry side) of gate i and is
        laterally aligned with gate i's axis within Y_MARGIN, the drone would
        otherwise fly back through gate i to reach gate i+1. Instead, insert a
        detour waypoint that routes the drone around (or over) gate i:

        - If gate i+1 is higher than gate i: detour goes UP (over the gate)
        - Else if gate i+1 is to the LEFT (body +y): detour goes LEFT
        - Else (gate i+1 is to the RIGHT, body -y): detour goes RIGHT

        The detour is inserted between post-gate-i and pre-gate-(i+1).

        Args:
            p_WLL_array:        Current waypoint list.
            pOLL_array:         Obstacle centers (unused here, kept for signature).
            pGLL_array:         Gate centers.
            y_GBL_array:        Gate yaws.
            t_elapsed:          Time passed (unused here, kept for signature).

        Returns:
            p_WLL_array:        Waypoint list with detours inserted where needed.
        """
        def find_row(arr, target, tol=1e-3):  # Changed from 1e-6 to 1e-3
            diffs = np.linalg.norm(arr - target, axis=1)
            idx = int(np.argmin(diffs))
            return idx if diffs[idx] < tol else -1

        D = 0.05  # Changed from 0.2 to match your _build_waypoints loop
        Y_MARGIN = 1         # gate i+1 must be within this lateral distance of gate i's axis
        HEIGHT_THRESHOLD = 0.2 # gate i+1 must be at least this much higher to use UP detour
        MIN_GATE_DISTANCE = 1.1

        # Calculate safe detour distance based on physical gate dimensions
        HALF_FRAME = FRAME_WIDTH / 2                     
        DETOUR_MARGIN = CLEARANCE + 0.3                  
        DETOUR_DIST = HALF_FRAME + DETOUR_MARGIN         

        gate_idx = obs.pTLL_index
        all_gates = obs.pTLL_array
        drone_pos = obs.pBLL

        is_uturn_old = False
        if gate_idx >= 1 and len(pGLL_array) > 0:
            prev_gate = all_gates[gate_idx - 1]
            prev_quads = obs.qTLT_array[gate_idx - 1]
            prev_yaw = R.from_quat(prev_quads).as_euler('ZYX')[0]
            next_gate_pos = pGLL_array[0]

            # 3D distance from previous gate to current target gate
            gate_dist = np.linalg.norm(next_gate_pos - prev_gate)
            if gate_dist >= MIN_GATE_DISTANCE:
                # Previous gate's body frame
                prev_x_axis = np.array([np.cos(prev_yaw),  np.sin(prev_yaw),  0.0])
                prev_y_axis = np.array([-np.sin(prev_yaw), np.cos(prev_yaw), 0.0])
                prev_z_axis = np.array([0.0, 0.0, 1.0])

                # Express current target gate in previous gate's body frame
                rel = next_gate_pos - prev_gate
                body_x = np.dot(rel, prev_x_axis)
                body_y = np.dot(rel, prev_y_axis)
                body_z = rel[2]

                # Same triggers as the gate-to-gate U-turn case
                if body_x < 0.0 and abs(body_y) < Y_MARGIN:
                    is_uturn_old = True

        if is_uturn_old:
            # Pick detour direction (same priority logic as gate-to-gate)
            if body_z >= HEIGHT_THRESHOLD:
                detour_offset = DETOUR_DIST * prev_z_axis
            elif body_y > 0:
                detour_offset = DETOUR_DIST * prev_y_axis
            else:
                detour_offset = -DETOUR_DIST * prev_y_axis

            detour_wp = prev_gate + detour_offset

            # Decide whether to insert: only if drone hasn't already passed the
            # previous gate's plane (i.e., drone is still on the entry side of the
            # previous gate, where the detour makes sense)
            rel_drone = drone_pos - prev_gate
            drone_body_x = np.dot(rel_drone, prev_x_axis)

            # Distance check: detour should be closer to drone than the next gate is
            # (i.e., the drone hasn't passed the detour point yet)
            dist_detourtogate = np.linalg.norm(next_gate_pos - detour_wp)
            dist_dronetogate = np.linalg.norm(next_gate_pos - drone_pos)

            idx_drone = find_row(p_WLL_array, drone_pos)
            #breakpoint()
            if (idx_drone >= 0
                and dist_detourtogate < dist_dronetogate):
                p_WLL_array = np.insert(p_WLL_array, idx_drone + 1, detour_wp, axis=0)
                print(f'drone-side U-turn detour inserted: {detour_wp}')

        # Iterate in reverse so insertion indices stay valid
        for i in reversed(range(len(pGLL_array) - 1)):
            gate_pos = pGLL_array[i]
            next_gate_pos = pGLL_array[i + 1]
            yaw = y_GBL_array[i]
            # 1. DISTANCE CONSTRAINT CHECK
            # Calculate total 3D distance between gate centers
            gate_dist = np.linalg.norm(next_gate_pos - gate_pos)
            if gate_dist < MIN_GATE_DISTANCE:
                # Gates are too close; skip U-turn detour and let _180_degree_turn handle it
                continue

            # Gate i's body-frame basis vectors in world coordinates
            x_axis = np.array([np.cos(yaw),  np.sin(yaw),  0.0])  # through-direction
            y_axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0])  # left
            z_axis = np.array([0.0, 0.0, 1.0])                   # up

            # Express next gate in gate i's body frame
            rel = next_gate_pos - gate_pos
            body_x = np.dot(rel, x_axis)
            body_y = np.dot(rel, y_axis)
            body_z = rel[2]

            #print(f"\n[DEBUG U-TURN] Checking Gate {i} -> Gate {i+1}")
            #print(f"  -> World Pos Gate {i}: {gate_pos}")
            #print(f"  -> World Pos Gate {i+1}: {next_gate_pos}")
            #print(f"  -> Relative Local Frame: X (forward)={body_x:.3f}, Y (left)={body_y:.3f}, Z (up)={body_z:.3f}")

            # Detection: next gate is behind gate i and roughly on-axis
            if body_x >= 0.0:
                continue                                  # next gate is in front, not a U-turn
            if abs(body_y) >= Y_MARGIN:
                continue                                  # next gate is off to one side, not a U-turn
            #print('test the uturn function')
            # Decide detour direction
            # Priority: height > left/right
            if body_z >= HEIGHT_THRESHOLD:
                # Detour goes UP
                detour_offset = DETOUR_DIST * z_axis
            elif body_y > 0:
                # Next gate is to the left → detour LEFT
                detour_offset = DETOUR_DIST * y_axis
            else:
                # Next gate is to the right (or directly on-axis with zero lateral) → detour RIGHT
                detour_offset = -DETOUR_DIST * y_axis

            # --- KEY CHANGE HERE ---
            # Place the detour waypoint perfectly within the current gate's 2D plane.
            # Because y_axis and z_axis have 0 component in the gate's x (through) direction,
            # adding them directly to gate_pos locks the waypoint in the plane.
            detour_wp = gate_pos + detour_offset

            # Insert between post-gate-i and pre-gate-(i+1)
            post_gate_i = gate_pos + D * x_axis
            idx_post = find_row(p_WLL_array, post_gate_i)
            if idx_post < 0:
                continue
            
            p_WLL_array = np.insert(p_WLL_array, idx_post + 1, detour_wp, axis=0)

        return p_WLL_array

    def _detour_gates1(
            self,
            obs: EnvState_t,
            p_WLL_array: np.ndarray,
            pGLL_array: np.ndarray,
            y_GBL_array: np.ndarray,
            t_elapsed: float
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
        base_push = FRAME_WIDTH / 2 + CLEARANCE*2

        wps = self._assemble_with_detours1(p_WLL_array, detours)
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
                detour = c + base_push * d
                detours.append((n_passed, detour))
                hit = True
                break

        if not hit:
            return wps
        
        new_p_WLL_array = self._assemble_with_detours1(p_WLL_array, detours)

        return new_p_WLL_array
    
    def _assemble_with_detours1(
            self,
            p_WLL_array: np.ndarray,
            detours: list
        ) -> np.array:
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

    def _push_dir_frame1(
            self,
            local: np.ndarray,
            yaw: np.array
        ) -> np.array:
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
        if   k == 0:
            ld = np.array([0.0,  1.0, 0.0])
        elif k == 1:
            ld = np.array([0.0, -1.0, 0.0])
        elif k == 2:
            ld = np.array([0.0,  0.0, 1.0])
        else:
            ld = np.array([0.0,  0.0, -1.0])

        cos, sin = np.cos(yaw), np.sin(yaw)

        push_dir = np.array([ld[0]*cos - ld[1]*sin,
                        ld[0]*sin + ld[1]*cos,
                        ld[2]])
        return push_dir