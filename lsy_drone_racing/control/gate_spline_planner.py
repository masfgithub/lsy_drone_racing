"""Gate-centred spline planner (obstacle-free).

A minimal racing-line planner that stays compatible with the SplinePlanner
interface (subclasses ``Planner``; ``plan(obs, t_elapsed) -> Trajectory``). It
ignores obstacles and builds waypoints purely from the gate poses:

    * start  : the current drone position
    * per gate i (from ``pTLL_index`` onward):
          - the gate centre
          - a point ``exit_offset`` (default 5 cm) past the centre along the
            gate normal  ("5 cm in the direction of the orientation")
    * between gate i-1 and gate i: a "detour" waypoint on gate i's approach
      axis, set back by half the gate-to-gate distance:
          detour_i = centre_i - 0.5 * ||centre_i - centre_{i-1}|| * n_i

All gate normals are oriented along the travel direction so ``+n`` is the exit
side (and ``-n`` the approach side). The waypoints are fitted with a
time-parameterised cubic spline (same scheme as ``SplinePlanner._create_spline``)
and returned as a ``Trajectory(positions, velocities, ts)``.

The gate normal convention matches the rest of the stack
(``_gate_normals_from_quats``): the gate-frame local x-axis,
``R.from_quat(q_xyzw).apply([1, 0, 0])``, with ``obs.qTLT_array`` in xyzw order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

# Stay compatible with the real interface, but fall back to lightweight stand-ins
# so this module (and its test) run without the lsy_drone_racing package.
try:
    from lsy_drone_racing.control.planner import Planner, Trajectory
except Exception:  # pragma: no cover - standalone / test fallback
    from dataclasses import dataclass

    @dataclass
    class Trajectory:  # type: ignore[no-redef]
        """Minimal stand-in matching Trajectory(positions, velocities, ts)."""

        positions: np.ndarray
        velocities: np.ndarray
        ts: np.ndarray

    class Planner:  # type: ignore[no-redef]
        """Minimal stand-in base planner."""

        def __init__(self, obs, info, config):
            self._obs, self._info, self._config = obs, info, config


if TYPE_CHECKING:
    from lsy_drone_racing.control.env_obs import EnvState_t


# Extra space (m) left between a gate's outer frame edge and the arc that wraps
# AROUND it on a re-crossing. The wrap waypoints sit at outer_half + this radially,
# so larger = the return loop gives the gate a wider berth. Tune here.
FRAME_WRAP_CLEARANCE = 0.15

# Distance (m) along the gate normal between consecutive wrap waypoints: the first
# and third of the three purple wrap points sit at +/- this from the middle one (so
# this is also the first<->middle and middle<->third spacing). Keep it above
# thickness/2 (~0.175 m) so the outer two clear the gate's depth. Tune here.
FRAME_WRAP_SPACING = 0.30

# When True, the drone-racing pipeline pops up the planned gates / waypoints /
# spline / obstacles 3D figure once at simulation start, via plot_plan() (defined
# at the bottom of this module). It is a blocking window -- close it to let the sim
# run. Set False to disable the feature entirely.
SHOW_PLAN_PLOT = True

# Debug: when True, plan() prints the full obs state (paste-ready) whenever a real
# violation survives planning -- a residual obstacle hit, or more than a trivial
# 1-2 sample frame bow. Lets a failing replan be captured from the sim console and
# reproduced offline. Set False once captured.
DUMP_VIOLATION_STATE = True

# Offset (m) along each gate normal used to build the inter-gate detour waypoint:
# the detour is the midpoint between the previous gate's exit-side point
# (centre + this * normal) and the next gate's approach-side point
# (centre - this * normal), so it leaves on the exit side and aims at the approach
# side instead of cutting straight centre-to-centre. Tune here.
DETOUR_GATE_OFFSET = 0.20


class GateCenterSplinePlanner(Planner):
    """Spline planner that threads gate centres, ignoring obstacles."""

    def __init__(
        self,
        obs: "EnvState_t",
        info: dict,
        config: dict,
        t_total: float,
        max_speed: float = 2.0,
        exit_offset: float = 0.05,
        attitude_offset: float = 0.05,
        lead_speed_eps: float = 0.1,
        gate_thickness: float = 0.35,
        gate_hole_half: float = 0.115,
        gate_outer_half: float = 0.40,
        frame_margin: float = 0.05,
        drone_radius: float = 0.0,
        thread_gates: bool = True,
        thread_separation: float = 0.10,
        frame_keepout: bool = True,
        obstacle_d_min: float = 0.15,
        obstacle_margin: float = 0.15,
        avoid_obstacles: bool = True,
        max_avoid_iters: int = 8,
    ):
        """Initialise the planner.

        Args:
            obs:             Initial observation.
            info:            Additional environment information (unused).
            config:          Environment configuration (uses config.env.freq).
            t_total:         Assumed total trajectory duration (s).
            max_speed:       Kept for interface compatibility (not used for timing).
            exit_offset:     Exit-helper offset (m) along the gate normal, used only
                             when ``thread_gates`` is False.
            attitude_offset: Distance (m, default 5 cm) for the lead waypoint placed
                             just in front of the drone, along its velocity (obs.vBLL)
                             when moving or its thrust direction (body +z from
                             obs.qBLB) when still, so the spline leaves the drone
                             along its current motion without a large lead that
                             fights the first gate on a close replan.
            lead_speed_eps:  Speed (m/s) above which the lead waypoint follows the
                             velocity direction (replan mid-flight) instead of the
                             thrust direction (initial plan).
            gate_thickness:  Gate frame depth (m) along the normal -- the Rahmen
                             "tunnel" length the drone must thread straight.
            gate_hole_half:  Half the gate opening (m); the hole is 2*this on a side.
            gate_outer_half: Half the outer frame (m); frame material lives between
                             the hole and this outer extent.
            frame_margin:    Safety margin (m). Shrinks the in-plane safe hole and
                             extends the frame depth slab used for clip detection.
            drone_radius:    Extra in-plane inset (m) subtracted from the safe hole.
            thread_gates:    Option A. If True, add an approach and an exit waypoint
                             on the gate normal, one on each side of the centre, so
                             the spline crosses each gate straight and threads the
                             hole by construction.
            thread_separation: Distance (m) between the approach and exit waypoints
                             (they sit at +/- half this along the gate normal).
            frame_keepout:   Option C. If True, count gate-frame clips alongside
                             obstacle violations when scoring avoidance candidates,
                             so an obstacle reroute can't quietly clip a frame.
            obstacle_d_min:  Obstacle keep-out radius (m, XY).
            obstacle_margin: Extra clearance (m) the avoidance waypoint is pushed
                             beyond d_min.
            avoid_obstacles: If True, run the push-and-reroute obstacle avoidance.
            max_avoid_iters: Max greedy avoidance iterations (one obstacle/iter).
        """
        super().__init__(obs, info, config)
        self._t_total = float(t_total)
        self._max_speed = float(max_speed)
        self._exit_offset = float(exit_offset)
        self._attitude_offset = float(attitude_offset)
        self._lead_speed_eps = float(lead_speed_eps)
        self._gate_thickness = float(gate_thickness)
        self._gate_hole_half = float(gate_hole_half)
        self._gate_outer_half = float(gate_outer_half)
        self._frame_margin = float(frame_margin)
        self._drone_radius = float(drone_radius)
        self._thread_gates = bool(thread_gates)
        self._frame_keepout = bool(frame_keepout)
        # Option A: approach/exit straddle the centre, +/- half thread_separation
        # along the normal. The depth slab (Option C clip detection) stays tied to
        # the real frame thickness -- it is the physical frame depth, independent
        # of how far apart the approach/exit waypoints are placed.
        self._thread_offset = 0.5 * float(thread_separation)
        self._depth_slab_half = 0.5 * self._gate_thickness + self._frame_margin
        self._safe_hole_half = max(
            0.01, self._gate_hole_half - self._frame_margin - self._drone_radius
        )
        self._obstacle_d_min = float(obstacle_d_min)
        self._obstacle_margin = float(obstacle_margin)
        self._avoid_obstacles = bool(avoid_obstacles)
        self._max_avoid_iters = int(max_avoid_iters)
        self._avoid_log: list[dict] = []
        self._frame_log: list[dict] = []
        self._gate_centers: np.ndarray | None = None
        self._gate_rot: np.ndarray | None = None
        # Passed gates (index < pTLL_index): no longer threaded, but still physical
        # frame obstacles to avoid. Populated by _build_waypoints on each plan.
        self._past_gate_centers: np.ndarray = np.zeros((0, 3))
        self._past_gate_rot: np.ndarray = np.zeros((0, 3, 3))
        self._freq = config.env.freq
        self.trajectory: Trajectory | None = None
        self._waypoints: np.ndarray | None = None
        self._frame_violations: list[tuple[int, int]] = []
        self._plan_fig = None  # handle to the live diagnostic figure (reused)

    # ----------------------------------------------------------------------

    def _plan_once(self, obs, t_elapsed, include_attitude):
        """Run one full planning pass (build -> obstacle avoid -> frame wrap ->
        spline -> violation check). Returns a result dict; does not mutate the
        public trajectory. Per-pass avoid/frame logs are captured so the chosen
        pass can restore them."""
        wps, labels = self._build_waypoints(obs, return_labels=True,
                                            include_attitude=include_attitude)
        obstacles = np.asarray(getattr(obs, "pOLL_array", np.zeros((0, 3))),
                               dtype=float)
        wps, labels = self._avoid(wps, labels, obstacles, t_elapsed)
        wps, labels = self._avoid_frame_recrossings(wps, labels, obstacles, t_elapsed)
        spline, t_sample = self._create_spline(wps, t_elapsed)
        positions = spline(t_sample)
        velocities = spline(t_sample, nu=1)
        frame_viol = self._check_frame_violations(positions)
        obs_xy = obstacles[:, :2] if obstacles.size else np.zeros((0, 2))
        obst_viol = (self._count_violations(positions[:, :2], obs_xy,
                     self._obstacle_d_min) if obs_xy.size else 0)
        return {
            "wps": wps, "labels": labels, "positions": positions,
            "velocities": velocities, "t_sample": t_sample,
            "frame_viol": frame_viol, "obst_viol": obst_viol,
            "obstacles": obstacles,
            "avoid_log": list(self._avoid_log),
            "frame_log": list(self._frame_log),
        }

    @staticmethod
    def _violation_score(r) -> tuple[int, int]:
        """(#obstacles violated, total frame-material samples). Lower is cleaner."""
        return (int(r["obst_viol"]), int(sum(n for _, n in r["frame_viol"])))

    @staticmethod
    def _is_real_violation(r) -> bool:
        """True if a pass left a residual obstacle hit or more than a trivial
        1-2 sample frame bow (the threshold used for the debug dump)."""
        ov, fv = GateCenterSplinePlanner._violation_score(r)
        return ov > 0 or fv > 2

    def plan(self, obs: "EnvState_t", t_elapsed: float = 0.0) -> Trajectory:
        """Build waypoints, reroute around obstacles, fit the spline, return Trajectory."""
        # Primary pass: with the velocity/thrust lead (attitude) waypoint.
        best = self._plan_once(obs, t_elapsed, include_attitude=True)
        used_attitude = True
        # If the lead waypoint led to a real violation (e.g. a replan whose velocity
        # points into a gate hole, which launches the spline through the frame and
        # tangles the wrap), retry without it and keep whichever pass is cleaner.
        if self._is_real_violation(best):
            alt = self._plan_once(obs, t_elapsed, include_attitude=False)
            if self._violation_score(alt) < self._violation_score(best):
                best, used_attitude = alt, False

        # Commit the chosen pass (restore its avoid/frame logs for the diagnostics).
        self._avoid_log = best["avoid_log"]
        self._frame_log = best["frame_log"]
        self._waypoints = best["wps"]
        self._waypoint_labels = best["labels"]
        positions = best["positions"]
        obstacles = best["obstacles"]
        self.trajectory = Trajectory(positions, best["velocities"], best["t_sample"])

        # Always check whether the final centerline actually enters any gate's
        # frame MATERIAL (strict: inside the half-thickness and between the hole and
        # outer edges -- no safety margin). Stored on self._frame_violations and
        # warned about, so it is visible in the console even with the figure off.
        self._frame_violations = best["frame_viol"]
        if self._frame_violations:
            detail = ", ".join(f"gate {gi} ({n} samples)"
                               for gi, n in self._frame_violations)
            note = "" if used_attitude else " [lead waypoint dropped]"
            print(f"[GateCenterSplinePlanner] WARNING: trajectory enters gate "
                  f"frame material: {detail}{note}")

        # One-shot debug dump: if a real violation survived planning (a residual
        # obstacle hit, or more than a trivial 1-2 sample frame bow), print the full
        # obs in paste-ready form so this exact (often replan) state can be
        # reproduced offline. Toggle DUMP_VIOLATION_STATE off once captured.
        if DUMP_VIOLATION_STATE:
            obst_viol = best["obst_viol"]
            frame_total = sum(n for _, n in self._frame_violations)
            if obst_viol > 0 or frame_total > 2:
                print("=== PLANNER VIOLATION STATE (paste back to reproduce) ===")
                print(f"# pTLL_index={int(getattr(obs, 'pTLL_index', 0))}  "
                      f"obst_viol={obst_viol}  frame_viol={self._frame_violations}  "
                      f"lead_dropped={not used_attitude}")
                print(f"drone      = {np.asarray(obs.pBLL, float).tolist()}")
                print(f"vel        = {np.asarray(getattr(obs, 'vBLL', np.zeros(3)), float).tolist()}")
                print(f"qBLB       = {np.asarray(getattr(obs, 'qBLB', [0, 0, 0, 1]), float).tolist()}")
                print(f"centers    = {np.asarray(obs.pTLL_array, float).tolist()}")
                print(f"quats      = {np.asarray(obs.qTLT_array, float).tolist()}")
                print(f"obstacles  = {np.asarray(obs.pOLL_array, float).tolist()}")
                print(f"pTLL_index = {int(getattr(obs, 'pTLL_index', 0))}")
                print("=== END VIOLATION STATE ===")

        # After EACH plan (init and every sim replan), show the diagnostic figure
        # if the flag is on. Blocking -- the run halts here until you close the
        # window, like the standalone test (otherwise the sim loop races on and the
        # window is torn down before it renders). prev_fig closes any earlier figure
        # first so windows don't pile up across replans.
        if SHOW_PLAN_PLOT:
            self._plan_fig = plot_plan(self, obs, self.trajectory,
                                       block=True, prev_fig=self._plan_fig)
        return self.trajectory

    # ----------------------------------------------------------------------

    def _gate_normals(
        self, centers: np.ndarray, quats_xyzw: np.ndarray, drone: np.ndarray
    ) -> np.ndarray:
        """Yaw-defined unit gate normals (gate-frame +x axis in world).

        The direction each gate must be flown through is fixed by its yaw -- the
        gate-frame x-axis, ``R.from_quat(q).apply([1, 0, 0])`` -- and is NOT
        inferred from the racing-line geometry. yaw and yaw+pi are different gates
        here (opposite required directions), so the quaternion already carries the
        sign and we use it directly: the drone enters from the ``-normal`` side and
        exits the ``+normal`` side. ``centers``/``drone`` are unused, kept only for
        call-site compatibility.
        """
        normals = np.atleast_2d(R.from_quat(quats_xyzw).apply([1.0, 0.0, 0.0]))
        return normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)

    def _build_waypoints(
        self, obs: "EnvState_t", return_labels: bool = False,
        include_attitude: bool = True,
    ) -> np.ndarray | tuple[np.ndarray, list[str]]:
        """Construct the waypoint list (start, attitude, per-gate approach/centre/exit, detours).

        ``include_attitude=False`` drops the velocity/thrust lead waypoint. plan()
        uses that fallback when the lead waypoint drives the spline into a violation
        (e.g. a mid-flight replan whose velocity points straight into a gate hole).
        """
        drone = np.asarray(obs.pBLL, dtype=float).reshape(3)
        gi = int(getattr(obs, "pTLL_index", 0))
        centers = np.asarray(obs.pTLL_array, dtype=float)[gi:].reshape(-1, 3)
        quats = np.asarray(obs.qTLT_array, dtype=float)[gi:].reshape(-1, 4)
        normals = self._gate_normals(centers, quats, drone)
        self._gate_centers = centers
        self._gate_rot = R.from_quat(quats).as_matrix().reshape(-1, 3, 3)
        # Passed gates stay physical obstacles: keep their centres/orientations so
        # the frame-material check and the wrap handler still avoid them, even
        # though we no longer thread them. A post-pass replan (e.g. triggered by a
        # nearby obstacle) can otherwise route the line straight back through the
        # frame of the gate we just cleared.
        all_centers = np.asarray(obs.pTLL_array, dtype=float).reshape(-1, 3)
        all_quats = np.asarray(obs.qTLT_array, dtype=float).reshape(-1, 4)
        if gi > 0:
            self._past_gate_centers = all_centers[:gi]
            self._past_gate_rot = R.from_quat(all_quats[:gi]).as_matrix().reshape(-1, 3, 3)
        else:
            self._past_gate_centers = np.zeros((0, 3))
            self._past_gate_rot = np.zeros((0, 3, 3))

        # Lead waypoint: leave the drone along its current motion so the spline
        # doesn't snap sideways out of the gate. On a replan mid-flight the drone
        # is moving, so we point the lead along the velocity direction (obs.vBLL);
        # at the initial plan the drone is essentially still, so we fall back to
        # the thrust direction (body +z rotated by obs.qBLB) -- where the attitude
        # points, "like from the start".
        vel = np.asarray(getattr(obs, "vBLL", np.zeros(3)), dtype=float).reshape(3)
        speed = float(np.linalg.norm(vel))
        if speed > self._lead_speed_eps:
            lead_dir = vel / speed
        else:
            q_body = np.asarray(getattr(obs, "qBLB", [0.0, 0.0, 0.0, 1.0]),
                                dtype=float).reshape(4)
            lead_dir = R.from_quat(q_body).apply([0.0, 0.0, 1.0])
            lead_dir = lead_dir / (np.linalg.norm(lead_dir) + 1e-12)

        wps: list[np.ndarray]
        labels: list[str]
        if include_attitude:
            wps = [drone, drone + self._attitude_offset * lead_dir]
            labels = ["start", "attitude"]
        else:
            wps = [drone]
            labels = ["start"]
        for i in range(len(centers)):
            if i > 0:
                # detour = midpoint between the previous gate's exit-side point
                # (centre + off * normal) and the next gate's approach-side point
                # (centre - off * normal), off = DETOUR_GATE_OFFSET. This leaves
                # along the previous gate's exit and aims at the next gate's
                # approach instead of cutting straight centre-to-centre.
                prev_exit = centers[i - 1] + DETOUR_GATE_OFFSET * normals[i - 1]
                next_appr = centers[i] - DETOUR_GATE_OFFSET * normals[i]
                wps.append(prev_exit + 0.5 * (next_appr - prev_exit))
                labels.append("detour")
            if self._thread_gates:
                # Option A: approach and exit on the gate normal, set back by
                # (thickness/2 + margin), so the spline crosses the frame depth
                # straight -- bounding the crossing angle and threading the hole.
                wps.append(centers[i] - self._thread_offset * normals[i])  # approach
                labels.append("approach")
                wps.append(centers[i].copy())                              # gate centre
                labels.append("gate")
                wps.append(centers[i] + self._thread_offset * normals[i])  # exit
                labels.append("exit")
            else:
                wps.append(centers[i].copy())                              # gate centre
                labels.append("gate")
                wps.append(centers[i] + self._exit_offset * normals[i])    # short exit
                labels.append("exit")

        wps_arr = np.asarray(wps, dtype=float)
        self._waypoints = wps_arr
        if return_labels:
            return wps_arr, labels
        return wps_arr

    def _create_spline(
        self, wps: np.ndarray, t_elapsed: float
    ) -> tuple[CubicSpline, np.ndarray]:
        """Fit a time-parameterised cubic spline (SplinePlanner scheme)."""
        wps = np.asarray(wps, dtype=float)
        seg = np.linalg.norm(np.diff(wps, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])

        # drop points that don't advance arc length (coincident waypoints)
        keep = np.concatenate([[True], seg > 1e-6])
        wps = wps[keep]
        cum = cum[keep]
        if len(wps) < 2:
            raise ValueError("need at least two distinct waypoints to fit a spline")

        d_total = cum[-1] if cum[-1] > 1e-9 else 1.0
        t_remaining = max(self._t_total - t_elapsed, 1e-3)
        t_gates = cum / d_total * t_remaining
        n_samp = max(int(np.round(t_remaining * self._freq)), 2)
        t_sample = np.linspace(0.0, t_remaining, n_samp)

        spline = CubicSpline(t_gates, wps, axis=0)
        return spline, t_sample

    # ----------------------------------------------------------------------
    # Obstacle avoidance (push perpendicular to the path, pick fewest violations)
    # ----------------------------------------------------------------------

    def _avoid(self, wps, labels, obstacles, t_elapsed):
        """Greedily reroute the spline around violated obstacle cylinders.

        Each iteration: find the trajectory sample that most deeply violates an
        obstacle, build a LEFT and a RIGHT candidate by inserting a waypoint
        pushed perpendicular to the path to (d_min + margin) from the obstacle
        centre, re-spline both, and keep the better candidate by priority:
        (1) fewest violations -- obstacles violated plus gates whose real frame
        MATERIAL is entered (strict, no margin); (2) straightest, i.e. least total
        curvature. Repeat until clear or max_avoid_iters is hit. ``self._avoid_log``
        records both candidates per decision for inspection/plotting.
        """
        self._avoid_log = []
        wps = [np.asarray(w, dtype=float) for w in wps]
        labels = list(labels)
        if not self._avoid_obstacles or len(obstacles) == 0:
            return np.asarray(wps), labels

        obs_xy = obstacles[:, :2]
        r_keep = self._obstacle_d_min
        r_push = self._obstacle_d_min + self._obstacle_margin

        for _ in range(self._max_avoid_iters):
            spline, t_sample = self._create_spline(np.asarray(wps), t_elapsed)
            pos = spline(t_sample)
            hit = self._worst_violation(pos[:, :2], obs_xy, r_keep)
            if hit is None:
                break
            o, k = hit

            # horizontal path tangent at the deepest violating sample
            v = np.asarray(spline(t_sample[k], nu=1)).reshape(-1)
            t_xy = np.array([v[0], v[1], 0.0])
            if np.linalg.norm(t_xy) < 1e-9:
                radial = pos[k, :2] - obs_xy[o]
                t_xy = np.array([radial[1], -radial[0], 0.0])
            t_xy = t_xy / (np.linalg.norm(t_xy) + 1e-12)
            perp = np.array([-t_xy[1], t_xy[0], 0.0])    # left of travel

            z = float(pos[k, 2])
            centre = np.array([obstacles[o, 0], obstacles[o, 1], z])
            cand = {"left": centre + r_push * perp, "right": centre - r_push * perp}

            # Drop the detour waypoint for the sector being rerouted so the placed
            # avoid waypoint REPLACES it instead of fighting it. A detour is dropped
            # if it brackets the path segment nearest the violation (its sector is
            # the one being avoided) OR it sits inside this obstacle's keep-out. The
            # old keep-out-only rule missed sector detours that pin the path onto the
            # obstacle from just outside the keep-out -- e.g. the Gate3->Gate4 detour,
            # 0.32 m from the obstacle but well past the 0.15 m radius.
            seg_j = self._insertion_index(wps, pos[k]) - 1   # nearest segment (seg_j, seg_j+1)
            drop = {
                j for j, (w, lab) in enumerate(zip(wps, labels))
                if lab == "detour"
                and (j in (seg_j, seg_j + 1)
                     or np.linalg.norm(np.asarray(w)[:2] - obs_xy[o]) < r_keep)
            }
            red_wps = [w for j, w in enumerate(wps) if j not in drop]
            red_labels = [lab for j, lab in enumerate(labels) if j not in drop]

            ins = self._insertion_index(red_wps, pos[k])

            evals = {}
            for side, wp in cand.items():
                cwps = red_wps[:ins] + [wp] + red_wps[ins:]
                cspline, ct = self._create_spline(np.asarray(cwps), t_elapsed)
                cpos = cspline(ct)
                # Feasibility = things actually hit: obstacles violated + gates whose
                # real frame MATERIAL is entered (strict, no margin -- so a side that
                # only grazes the safety margin still counts as clean, unlike the old
                # margin-padded _frame_clips that tied a feasible side with a violating
                # one). frame term respects the frame_keepout toggle.
                fviol = len(self._check_frame_violations(cpos)) if self._frame_keepout else 0
                evals[side] = {
                    "wps": cwps,
                    "pos": cpos,
                    "nviol": self._count_violations(cpos[:, :2], obs_xy, r_keep) + fviol,
                    "curv": self._path_curvature(cpos),
                }
            # Priority: (1) no violation, (2) straightest (least total curvature).
            chosen = min(evals, key=lambda s: (evals[s]["nviol"], evals[s]["curv"]))

            self._avoid_log.append({
                "obstacle": int(o),
                "chosen": chosen,
                "dropped_detours": len(drop),
                "left_cand": cand["left"], "right_cand": cand["right"],
                "left_pos": evals["left"]["pos"], "right_pos": evals["right"]["pos"],
                "left_nviol": evals["left"]["nviol"], "right_nviol": evals["right"]["nviol"],
                "left_curv": evals["left"]["curv"], "right_curv": evals["right"]["curv"],
            })

            wps = evals[chosen]["wps"]
            labels = red_labels[:ins] + ["avoid"] + red_labels[ins:]

        return np.asarray(wps), labels

    def _frame_gates(self):
        """Gates the path must respect as frame obstacles, as (centre, Rg,
        is_threaded) tuples: the remaining gates (threaded, from pTLL_index on)
        first, then the PASSED gates (obstacle-only -- no longer threaded, but their
        frame material is still a collision risk). Used by every frame check and the
        wrap handler so a gate keeps avoiding even after it has been passed.
        """
        gates = []
        if self._gate_centers is not None:
            for c, Rg in zip(self._gate_centers, self._gate_rot):
                gates.append((c, Rg, True))
        for c, Rg in zip(self._past_gate_centers, self._past_gate_rot):
            gates.append((c, Rg, False))
        return gates

    def _frame_material_mask(self, pos: np.ndarray) -> np.ndarray:
        """Boolean mask of samples that sit inside any gate's frame MATERIAL.

        Strict, no safety margin: a sample is in material if, in some gate's local
        frame, |x| <= thickness/2 (inside the gate's depth) and the in-plane offset
        is between the hole edge and the outer edge (hole_half < max(|y|,|z|) <=
        outer_half). This is the true collision region, unlike the margin-padded
        ``_frame_clips``. Covers threaded AND passed gates. Used by the always-on
        plan check and the figure highlight.
        """
        pos = np.asarray(pos, dtype=float)
        mask = np.zeros(len(pos), dtype=bool)
        fh, oh, hw = 0.5 * self._gate_thickness, self._gate_outer_half, self._gate_hole_half
        for c, Rg, _ in self._frame_gates():
            local = (pos - c) @ Rg
            lat = np.maximum(np.abs(local[:, 1]), np.abs(local[:, 2]))
            mask |= (np.abs(local[:, 0]) <= fh) & (lat > hw) & (lat <= oh)
        return mask

    def _check_frame_violations(self, pos: np.ndarray) -> list[tuple[int, int]]:
        """Per-gate count of samples inside that gate's frame material (strict).

        Returns a list of (gate_index, sample_count) for the gates that are hit;
        empty if clean. Indexes the combined frame-gate list (threaded gates first,
        then passed gates), so a clipped PASSED gate is reported too. Same material
        test as ``_frame_material_mask``.
        """
        pos = np.asarray(pos, dtype=float)
        fh, oh, hw = 0.5 * self._gate_thickness, self._gate_outer_half, self._gate_hole_half
        hits = []
        for gi, (c, Rg, _) in enumerate(self._frame_gates()):
            local = (pos - c) @ Rg
            lat = np.maximum(np.abs(local[:, 1]), np.abs(local[:, 2]))
            n = int(np.sum((np.abs(local[:, 0]) <= fh) & (lat > hw) & (lat <= oh)))
            if n:
                hits.append((gi, n))
        return hits

    def _frame_clips(self, pos: np.ndarray) -> int:
        """Count gates whose frame (Rahmen) the trajectory pos (N x 3) clips.

        A sample clips gate g if, in g's local frame, it lies within the depth
        slab (|x| <= thickness/2 + margin) AND is laterally outside the safe hole
        (|y| or |z| > safe_hole_half) while still inside the outer frame extent
        (|y|, |z| <= outer_half) -- i.e. it sits where the frame material is, at a
        depth where the frame exists. Covers threaded AND passed gates. Returns the
        number of gates clipped.
        """
        if not self._frame_keepout:
            return 0
        sh, oh, slab = self._safe_hole_half, self._gate_outer_half, self._depth_slab_half
        pos = np.asarray(pos, dtype=float)
        count = 0
        for c, Rg, _ in self._frame_gates():
            local = np.abs((pos - c) @ Rg)        # |x| depth, |y| width, |z| height
            in_slab = local[:, 0] <= slab
            in_outer = (local[:, 1] <= oh) & (local[:, 2] <= oh)
            outside_safe = (local[:, 1] > sh) | (local[:, 2] > sh)
            if np.any(in_slab & in_outer & outside_safe):
                count += 1
        return count

    def _avoid_frame_recrossings(self, wps, labels, obstacles, t_elapsed):
        """Wrap the path around the OUTSIDE of any gate frame it re-crosses.

        After a gate is threaded once, a sharp turn back toward the next gate can
        make the line cross the same gate's plane a second time. A single pushed
        waypoint does not fix this: the spline still sags back through the frame on
        either side of that point. Instead we insert a short CHAIN of three
        waypoints that routes the return path AROUND the frame box. In gate-local
        axes (x = through-normal, y/z = in plane), on a chosen in-plane side, all
        at radius R = outer_half + FRAME_WRAP_CLEARANCE:

            1. (+/-D, R)   out to R on the entry side, D = FRAME_WRAP_SPACING along
                           the normal (kept beyond the frame depth);
            2. ( 0,  R)    around the side, crossing the gate plane at radius R, well
                           outside the 0.40 m frame;
            3. (-/+D, R)   continue to the exit side, still at radius R.

        The frame can be wrapped on EITHER in-plane side. We build both: the
        ``near`` wrap on the side the re-crossing is already on (the natural, short
        way round) and the ``far`` wrap on the opposite side. Each is re-splined and
        scored, and we keep the feasible one -- fewest obstacle violations first,
        then fewest frame clips, then shortest -- so a clean near side wins normally
        and we only swing to the far side when the near one runs into an obstacle
        (as Gate 2 does on the real track). Both candidates are recorded in
        ``self._frame_log``. The tie-break when both are feasible is provisional.
        The intended thread is protected. Greedy: fix the worst re-crossing,
        re-spline, repeat.
        """
        self._frame_log = []
        wps = [np.asarray(w, dtype=float) for w in wps]
        labels = list(labels)
        if not self._frame_keepout or self._gate_centers is None:
            return np.asarray(wps), labels

        obstacles = np.asarray(obstacles, dtype=float)
        obs_xy = obstacles[:, :2] if obstacles.size else np.zeros((0, 2))
        R = self._gate_outer_half + FRAME_WRAP_CLEARANCE     # radius past the frame
        D = FRAME_WRAP_SPACING                                # depth between wrap points
        frame_gates = self._frame_gates()                    # threaded + passed gates

        for _ in range(self._max_avoid_iters):
            spline, t_sample = self._create_spline(np.asarray(wps), t_elapsed)
            pos = spline(t_sample)
            hit = self._worst_recrossing(pos)
            if hit is None:
                break
            gi, k = hit
            c, Rg, threaded = frame_gates[gi]
            normal = Rg[:, 0] / (np.linalg.norm(Rg[:, 0]) + 1e-12)
            yaxis = Rg[:, 1] / (np.linalg.norm(Rg[:, 1]) + 1e-12)
            zaxis = Rg[:, 2] / (np.linalg.norm(Rg[:, 2]) + 1e-12)
            rel = pos[k] - c
            ly, lz = float(rel @ yaxis), float(rel @ zaxis)
            m = max(abs(ly), abs(lz))          # dominant in-plane component
            if m < 1e-9:                       # crossing on the axis: pick the +y side
                ly, lz, m = 1.0, 0.0, 1.0
            # The frame is a SQUARE ring: clear it by pushing the DOMINANT component
            # to R (Euclidean scaling leaves a diagonal crossing at R/sqrt(2) < oh,
            # still inside the frame -- which never converges). off then has
            # max(|y|, |z|) = R, outside the outer edge on any heading.
            off = R * (ly / m * yaxis + lz / m * zaxis)
            # which side along the normal the line enters from / exits to (a couple
            # samples either side of the plane crossing avoid the near-zero ambiguity)
            n = len(pos)
            x_in = float((pos[max(0, k - 2)] - c) @ normal)
            x_out = float((pos[min(n - 1, k + 2)] - c) @ normal)
            s_in = 1.0 if x_in >= 0.0 else -1.0
            s_out = -s_in if (x_out >= 0.0) == (x_in >= 0.0) else (1.0 if x_out >= 0.0 else -1.0)
            ins = self._insertion_index(wps, pos[k])

            def arc_for(o: np.ndarray) -> list:
                return [c + s_in * D * normal + o, c + o, c + s_out * D * normal + o]

            cand = {"near": off, "far": -off}     # near = re-crossing side, far = opposite
            evals = {}
            for side, o in cand.items():
                cwps = wps[:ins] + arc_for(o) + wps[ins:]
                cspline, ct = self._create_spline(np.asarray(cwps), t_elapsed)
                cpos = cspline(ct)
                nviol = (self._count_violations(cpos[:, :2], obs_xy, self._obstacle_d_min)
                         if obs_xy.size else 0)
                evals[side] = {"wps": cwps, "obst": nviol,
                               "frame": self._frame_clips(cpos),
                               "length": self._path_length(cpos)}
            # feasibility first (obstacle-clear, then frame-clean); on a tie keep
            # the natural near side (provisional -- final tie-break TBD).
            rank = {"near": 0, "far": 1}
            chosen = min(evals, key=lambda s: (evals[s]["obst"], evals[s]["frame"],
                                               rank[s], evals[s]["length"]))
            wps = evals[chosen]["wps"]
            labels = labels[:ins] + ["frame"] * 3 + labels[ins:]
            self._frame_log.append({
                "gate": int(gi), "from_offset": round(m, 3), "wrapped_to": round(R, 3),
                "points": 3, "side": chosen, "past": not threaded,
                "near_obst": evals["near"]["obst"], "far_obst": evals["far"]["obst"],
            })
        return np.asarray(wps), labels

    def _worst_recrossing(self, pos):
        """Worst gate-plane crossing to wrap -- across threaded AND passed gates.

        Threaded gate: each must be threaded exactly once. Among the plane crossings
        within the outer frame extent, the one nearest the gate centre is the
        intended thread; every *other* crossing (a second pass through the hole or a
        frame clip) is a re-crossing to wrap.

        Passed gate (obstacle-only): there is NO intended thread, so any plane
        crossing that clips the frame MATERIAL (lateral between the hole and outer
        edges) is a collision to wrap. Passing through its open hole is harmless and
        ignored.

        Returns (gate_idx, sample_idx) -- gate_idx into ``_frame_gates()`` -- for the
        crossing sitting closest to the gate plane (the clearest pass), or None.
        """
        gates = self._frame_gates()
        if not gates:
            return None
        oh, hw = self._gate_outer_half, self._gate_hole_half
        worst, worst_depth = None, np.inf
        for gi, (c, Rg, threaded) in enumerate(gates):
            L = (pos - c) @ Rg
            x = L[:, 0]                                   # signed depth along normal
            lat = np.maximum(np.abs(L[:, 1]), np.abs(L[:, 2]))
            crossing = np.sign(x[:-1]) != np.sign(x[1:])  # plane crossing k -> k+1
            if threaded:
                sc = crossing & (lat[:-1] <= oh) & (lat[1:] <= oh)
                cross = np.where(sc)[0]
                if len(cross) <= 1:                       # threaded once (or not): fine
                    continue
                kc = int(np.argmin(np.linalg.norm(pos - c, axis=1)))
                intended = cross[int(np.argmin(np.abs(cross - kc)))]
                recross = [int(k) for k in cross if k != intended]
            else:
                # passed gate: wrap any crossing through the frame material
                mat = (lat > hw) & (lat <= oh)
                sc = crossing & (mat[:-1] | mat[1:])
                recross = [int(k) for k in np.where(sc)[0]]
            for k in recross:
                ksamp = k if abs(x[k]) <= abs(x[k + 1]) else k + 1   # sample on the plane
                depth = abs(float(x[ksamp]))
                if depth < worst_depth:                   # nearest the plane = clearest pass
                    worst_depth, worst = depth, (gi, int(ksamp))
        return worst

    @staticmethod
    def _worst_violation(pos_xy, obs_xy, r):
        """Return (obstacle_idx, sample_idx) of the deepest violation, or None."""
        worst, worst_depth = None, 1e-6
        for o in range(len(obs_xy)):
            d = np.linalg.norm(pos_xy - obs_xy[o], axis=1)
            k = int(np.argmin(d))
            depth = r - float(d[k])
            if depth > worst_depth:
                worst_depth, worst = depth, (o, k)
        return worst

    @staticmethod
    def _count_violations(pos_xy, obs_xy, r):
        """Number of obstacles the trajectory comes within r of (XY)."""
        return int(sum(
            np.min(np.linalg.norm(pos_xy - obs_xy[o], axis=1)) < r - 1e-6
            for o in range(len(obs_xy))
        ))

    @staticmethod
    def _insertion_index(wps, point):
        """Index at which to insert `point`: after the nearest waypoint segment."""
        wps = np.asarray(wps)
        point = np.asarray(point)
        best_j, best_d = 0, np.inf
        for j in range(len(wps) - 1):
            a, b = wps[j], wps[j + 1]
            ab = b - a
            tt = float(np.clip((point - a) @ ab / (float(ab @ ab) + 1e-12), 0.0, 1.0))
            d = float(np.linalg.norm(point - (a + tt * ab)))
            if d < best_d:
                best_d, best_j = d, j
        return best_j + 1

    @staticmethod
    def _path_length(pos):
        return float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))

    @staticmethod
    def _path_curvature(pos):
        """Total absolute turning of the path (integral of |curvature| ds).

        Sum of the angles between consecutive segment tangents -- a sampling-robust
        "straightness" measure: 0 for a straight line, larger the more the path
        bends. Used to pick the straightest detour side once feasibility ties.
        """
        pos = np.asarray(pos, dtype=float)
        d = np.diff(pos, axis=0)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        u = d / (n + 1e-12)
        dots = np.clip(np.sum(u[:-1] * u[1:], axis=1), -1.0, 1.0)
        return float(np.sum(np.arccos(dots)))

    # ----------------------------------------------------------------------

    def get_pos_traj(self, n: int = 200) -> np.ndarray:
        """Return the planned position samples (call plan() first)."""
        if self.trajectory is None:
            raise RuntimeError("call plan() before get_pos_traj()")
        return self.trajectory.positions


def plot_plan(planner, obs, traj=None, *, show=True, block=True, prev_fig=None,
              save_path=None,
              title="GateCenterSplinePlanner -- gates, waypoints, spline"):
    """Render the planned gates / waypoints / spline / obstacles as a 3D figure.

    Call AFTER ``planner.plan(obs)`` so the waypoints and logs are populated. Draws
    the fitted spline, every labelled waypoint (start / attitude / approach / gate /
    exit / detour / avoid / frame-wrap arc), any samples that enter gate frame
    material (highlighted red), each obstacle reroute's chosen and alternative
    candidate trajectories, each gate's outer / hole / safe-hole squares with its
    normal, and the obstacle keep-out cylinders.

    show/block control an interactive window (blocking by default -- close it to
    continue; pass block=False for a non-blocking live window, e.g. the per-plan
    refresh in the sim). prev_fig, if given, is closed first so a repeatedly-replanned
    sim reuses a single window instead of piling them up. save_path also writes the
    figure to disk. This is the single plotting implementation shared by the test and
    the planner's per-plan SHOW_PLAN_PLOT hook. matplotlib is imported lazily so the
    planner itself never depends on it.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if prev_fig is not None:
        try:
            plt.close(prev_fig)
        except Exception:
            pass

    if traj is None:
        traj = planner.trajectory
    P = np.asarray(traj.positions, float)

    centers = np.asarray(obs.pTLL_array, float)
    quats = np.asarray(obs.qTLT_array, float)
    normals = planner._gate_normals(centers, quats, np.asarray(obs.pBLL, float))
    wps = planner._waypoints
    labels = planner._waypoint_labels

    fig = plt.figure(figsize=(11, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(P[:, 0], P[:, 1], P[:, 2], "-", color="0.45", lw=1.6, label="spline")

    # Highlight any samples that enter gate frame material (the always-on check).
    viol = planner._frame_material_mask(P)
    if viol.any():
        ax.scatter(P[viol, 0], P[viol, 1], P[viol, 2], c="red", s=34,
                   depthshade=False, label="frame violation", zorder=10)

    colors = {"start": "green", "attitude": "magenta", "approach": "tab:cyan",
              "gate": "black", "exit": "tab:blue", "detour": "tab:red",
              "avoid": "darkorange", "frame": "purple"}
    seen = set()
    for p, lab in zip(wps, labels):
        ax.scatter(*p, color=colors.get(lab, "gray"), s=48,
                   label=lab if lab not in seen else None, depthshade=False)
        seen.add(lab)

    # Obstacle reroute candidates. For EACH violated obstacle we compute a LEFT
    # and a RIGHT trajectory and keep the better one; draw both for every obstacle
    # so the calculated ALTERNATIVE (the side we did not take) is always visible.
    # Colour is per obstacle decision; the alternative is dashed/bold, the chosen
    # side dotted/faint (the grey spline above already follows the chosen route).
    avoid_log = getattr(planner, "_avoid_log", [])
    cand_cmap = plt.get_cmap("tab10")
    for idx, d in enumerate(avoid_log):
        o = int(d["obstacle"])
        col = cand_cmap(idx % 10)
        chosen = d["chosen"]
        alt = "right" if chosen == "left" else "left"
        ap = np.asarray(d[f"{alt}_pos"], float)
        ax.plot(ap[:, 0], ap[:, 1], ap[:, 2], "--", lw=1.5, alpha=0.9, color=col,
                label=f"alt obs {o} ({alt})")
        cp = np.asarray(d[f"{chosen}_pos"], float)
        ax.plot(cp[:, 0], cp[:, 1], cp[:, 2], ":", lw=1.0, alpha=0.45, color=col,
                label=f"chosen obs {o} ({chosen})")

    drone = np.asarray(obs.pBLL, float)
    tdir = R.from_quat(np.asarray(obs.qBLB, float)).apply([0.0, 0.0, 1.0])
    ax.quiver(drone[0], drone[1], drone[2], tdir[0], tdir[1], tdir[2],
              length=0.35, color="magenta", lw=1.6)

    def _square(c, yax, zax, h):
        return np.array([
            c + h * yax + h * zax, c - h * yax + h * zax,
            c - h * yax - h * zax, c + h * yax - h * zax, c + h * yax + h * zax,
        ])

    oh, hw, sh = planner._gate_outer_half, planner._gate_hole_half, planner._safe_hole_half
    first = True
    for c, q, n in zip(centers, quats, normals):
        Rm = R.from_quat(q).as_matrix()
        yax, zax = Rm[:, 1], Rm[:, 2]
        outer, hole, safe = _square(c, yax, zax, oh), _square(c, yax, zax, hw), _square(c, yax, zax, sh)
        ax.plot(outer[:, 0], outer[:, 1], outer[:, 2], "-", color="0.6", lw=1.0,
                label="gate frame" if first else None)
        ax.plot(hole[:, 0], hole[:, 1], hole[:, 2], "-", color="saddlebrown", lw=1.4)
        ax.plot(safe[:, 0], safe[:, 1], safe[:, 2], "--", color="tab:green", lw=1.0,
                label="safe hole" if first else None)
        ax.quiver(c[0], c[1], c[2], n[0], n[1], n[2], length=0.35, color="orange", lw=1.6)
        first = False

    d_min = planner._obstacle_d_min
    obstacles = np.asarray(getattr(obs, "pOLL_array", np.zeros((0, 3))), float)
    for k, o in enumerate(obstacles):
        th = np.linspace(0, 2 * np.pi, 40)
        Th, Zc = np.meshgrid(th, np.linspace(0.0, float(o[2]), 2))
        ax.plot_surface(o[0] + d_min * np.cos(Th), o[1] + d_min * np.sin(Th), Zc,
                        color="tab:purple", alpha=0.20, linewidth=0, shade=False)
        ax.plot(o[0] + d_min * np.cos(th), o[1] + d_min * np.sin(th),
                np.full_like(th, o[2]), color="tab:purple", lw=0.8, alpha=0.6,
                label="obstacle" if k == 0 else None)

    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(title)
    ax.legend(loc="upper left")
    try:
        ax.set_box_aspect((1, 1, 0.5))
    except Exception:
        pass
    ax.view_init(elev=28, azim=-60)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=130)
    if show:
        plt.show(block=block)
        if not block:
            plt.pause(0.001)   # flush GUI events so the window actually renders
    else:
        plt.close(fig)
    return fig