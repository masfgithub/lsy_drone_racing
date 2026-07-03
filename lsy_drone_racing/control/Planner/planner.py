"""Planner interface: abstract base class for all trajectory planners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:

    from lsy_drone_racing.control.env_obs import EnvState_t

__all__ = ["Trajectory", "Planner", "DEFAULT_MAX_SPEED"]

DEFAULT_MAX_SPEED = 12.0  # m/s
FRAME_WIDTH = 0.72
FRAME_OPENING = 0.1
FRAME_THICK = 0.4
POST_WIDTH = 0.2
CLEARANCE = 0.1
R_OBSTACLE = 0.15
DRONE_RADIUS = 0.05



@dataclass
class Trajectory:
    """Class to represent trajectory from planner with positions, velocities and timestamps."""

    positions: np.ndarray
    velocities: np.ndarray
    timestamps: np.ndarray

    #p_ref_LL: np.ndarray
    #v_ref_LL: np.ndarray
    #t_ref: np.ndarray


class Planner(ABC):
    """Abstract base class for drone trajectory planners."""

    def __init__(
            self,
            obs: EnvState_t,
            info: dict,
            config: dict
    ):
        """Initialize Planner Class.
        
        Arg:
            obs:                Environment state observation.
            info:               Additional environment information.
            config:             Environment configuration.
        """
        self.freq = config.env.freq
        self._gates_information = {
            "total_length": 0.72,
            "total_height": 0.72,
            "hole_width": 0.25,
            "hole_height": 0.25,
            "thickness": 0.3,
            "margin": 0.05,
        }

    @abstractmethod
    def plan(
        self,
        obs: EnvState_t,
        info: dict,
        config: dict
    ) -> Trajectory:
        """Compute a trajectory through the gates. Subclasses must implement.

        Args:
            obs:                Environment state observation.
            info:               Additional environment information.
            config:             Environment configuration.

        Returns:
            trajectory:         pos, vel, time in a trajectory class.
        """

    def _gate(
            self,
            obs: EnvState_t
        ) -> tuple[np.ndarray, np.ndarray]:
        """Returns gate yaw and gate centre position from environment observation.

        Args:
            obs:                Environment state observation.

        Returns:
            y_GBL_array:        Gate orientation yaw of body relative to Local frame.
            pGLL_array:         Gate center position of Target relative to Local frame in Local 
                                coordinates.
        """
        # Quaternion of gate frames
        qTLT = obs.qTLT_array
        pTLL_index = obs.pTLL_index
        
        # Extracted rotation matrix/Euler angles from the quaternion
        y_GBL_array = R.from_quat(qTLT[pTLL_index:]).as_euler('ZYX')[:, 0]

        # Centre position of gate frames
        pGLL_array = obs.pTLL_array[pTLL_index:]

        return pGLL_array, y_GBL_array
        # Quaternion of gate frames
        qTLT = obs.qTLT_array
        pTLL_index = obs.pTLL_index
        
        # Extracted rotation matrix/Euler angles from the quaternion
        y_GBL_array = R.from_quat(qTLT[pTLL_index:]).as_euler('ZYX')[:, 0]

        # Centre position of gate frames
        if pTLL_index + 4 > 4:
            pGLL_array = obs.pTLL_array[pTLL_index:]
            # ADD THIS: Slice the yaw array exactly the same way!
            y_GBL_array = y_GBL_array[pTLL_index:] 
        else:
            pGLL_array = obs.pTLL_array[pTLL_index:pTLL_index + 2]
            # ADD THIS: Slice the yaw array exactly the same way!
            y_GBL_array = y_GBL_array[pTLL_index:pTLL_index + 2]

        return pGLL_array, y_GBL_array
    
    def _check_gate(
        self,
        p_ref_LL: np.array, 
        pGLL_array: np.ndarray, 
        y_GBL_array: np.ndarray
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None, float | None, float | None]:
        """Checks if a trajectory point is colliding with any gate frame.
        
        Args:
            p_ref_LL:           Trajectory point to check.
            pGLL_array:         Center positions of all gates.
            y_GBL_array:        Yaws of all the gates.

        Returns:
            is_inside_frame:    True if the point is colliding with a gate's solid frame structure.
            centre:             Centre of the gate that's been violated.
            local:              Distance from the point to the centre of the violated gate.
            yaw:                Yaw of the violated gate.
            half_outer:         Outer distance of the gate frame.

        """
        # Half-dimensions including clearance margins
        half_outer = (FRAME_WIDTH / 2)
        half_open  = FRAME_OPENING / 2
        half_thick = FRAME_THICK + CLEARANCE
        half_post_width = POST_WIDTH/2 + CLEARANCE

        # Difference trajectory point to all gate centers
        diff = p_ref_LL - pGLL_array
        
        # Rotate differences into each individual gate's local frame using its yaw
        lx =  diff[:, 0] * np.cos(y_GBL_array) + diff[:, 1] * np.sin(y_GBL_array)
        ly = -diff[:, 0] * np.sin(y_GBL_array) + diff[:, 1] * np.cos(y_GBL_array)
        lz =  diff[:, 2]

        # Calculate bounding box overlaps for all gates simultaneously
        in_depth = np.abs(lx) < half_thick
        in_outer = (np.abs(ly) < half_outer) & (np.abs(lz) < half_outer)
        in_open  = (np.abs(ly) < half_open) & (np.abs(lz) < half_open)

        # Gate Post
        # point is in the post zone when it sits BELOW the opening (local frame, consistent units)
        lz_post = lz < -half_open          # not: diff[:,2] < post_height
        in_post = np.abs(ly) < half_post_width
        post_collisions = lz_post & in_post & in_depth
        
        # Collision happens if it is inside gate frame but not the airhole
        gate_collisions = in_depth & in_outer & ~in_open
        
        is_inside_frame = bool(np.any(gate_collisions))
        post_collisions = lz_post & in_post & in_depth
        is_post_collisions = bool(np.any(post_collisions))

        if not is_inside_frame | is_post_collisions:
            return False, None, None, None, None
        
        if is_post_collisions:
            lz = np.abs(lz*10)

        #print(is_post_collisions, is_inside_frame)

        #if is_inside_frame:
            #print(in_depth, in_outer, ~in_open)
        #breakpoint()
        g = int(np.argmax(gate_collisions))
        centre = pGLL_array[g]
        local = np.array([lx[g], ly[g], lz[g]])
        yaw = y_GBL_array[g]
        return True, centre, local, yaw, half_outer

    def _check_gate2(
        self,
        p_ref_LL: np.array, 
        pGLL_array: np.ndarray, 
        y_GBL_array: np.ndarray
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None, float | None, float | None]:
        """Checks if a trajectory point is colliding with any gate frame.
        
        Args:
            p_ref_LL:           Trajectory point to check.
            pGLL_array:         Center positions of all gates.
            y_GBL_array:        Yaws of all the gates.

        Returns:
            is_inside_frame:    True if the point is colliding with a gate's solid frame structure.
            centre:             Centre of the gate that's been violated.
            local:              Distance from the point to the centre of the violated gate.
            yaw:                Yaw of the violated gate.
            half_outer:         Outer distance of the gate frame.

        """
        # Half-dimensions including clearance margins
        half_outer = (FRAME_WIDTH / 2)
        half_open  = FRAME_OPENING / 2
        half_thick = FRAME_THICK
        half_post_width = POST_WIDTH/2 + CLEARANCE

        # Difference trajectory point to all gate centers
        diff = p_ref_LL - pGLL_array
        
        # Rotate differences into each individual gate's local frame using its yaw
        lx =  diff[:, 0] * np.cos(y_GBL_array) + diff[:, 1] * np.sin(y_GBL_array)
        ly = -diff[:, 0] * np.sin(y_GBL_array) + diff[:, 1] * np.cos(y_GBL_array)
        lz =  diff[:, 2]

        # Calculate bounding box overlaps for all gates simultaneously
        in_depth = np.abs(lx) < half_thick
        in_outer = (np.abs(ly) < half_outer) & (np.abs(lz) < half_outer)

        # Gate Post
        # point is in the post zone when it sits BELOW the opening (local frame, consistent units)
        lz_post = lz < -half_open          # not: diff[:,2] < post_height
        in_post = np.abs(ly) < half_post_width
        post_collisions = lz_post & in_post & in_depth
        
        # Collision happens if it is inside gate frame but not the airhole
        gate_collisions = in_depth & in_outer
        
        is_inside_frame = bool(np.any(gate_collisions))
        post_collisions = lz_post & in_post & in_depth
        is_post_collisions = bool(np.any(post_collisions))

        if not is_inside_frame | is_post_collisions:
            return False, None, None, None, None
        
        if is_post_collisions:
            lz = np.abs(lz*10)

        #print(is_post_collisions, is_inside_frame)

        #if is_inside_frame:
            #print(in_depth, in_outer, ~in_open)
        #breakpoint()
        g = int(np.argmax(gate_collisions))
        centre = pGLL_array[g]
        local = np.array([lx[g], ly[g], lz[g]])
        yaw = y_GBL_array[g]
        return True, centre, local, yaw, half_outer
    
    def _check_obsticle(
            self,
            p_ref_LL: np.array,
            pOLL_array: np.ndarray
    ) -> tuple[bool, np.ndarray|None, float|None]:
        """Checks if a trajectory point is inside an obsticle.
        
        Args:
            p_ref_LL:       Trajectory point to be checked.
            pOLL_array:     Centre point of obsticles.

        Returns:
            is_inside_obsticle:     Boolian value if point is inside true if its outside false.
            d_O:                    Distance the point needs to clear the obsticle.
        """
        # Hardcoded physical pillar radius + safety margin (match to your env configuration)
        r_obstacle = R_OBSTACLE

        # consider only x and y coordinates because it is a pillar
        p_ref_xy = p_ref_LL[0:2]
        pOLL_xy  = pOLL_array[:, 0:2]

        # xy diff to all obsticles 
        diff_xy = p_ref_xy - pOLL_xy

        # xy norm distance to the obsticle
        distances_xy = np.linalg.norm(diff_xy, axis=1)

        # A collision occurs if the distance is less than the obstacle radius threshold
        obstacle_collisions = distances_xy < r_obstacle
        is_inside_obstacle = bool(np.any(obstacle_collisions))

        if not is_inside_obstacle:
            return False, None, None
        
        # Return obsticle centre of violation and push for the detour point
        o = int(np.argmin(distances_xy))
        return True, pOLL_array[o, :2], r_obstacle
    
    def get_pos_traj(self) -> np.ndarray:
        """TBD: for Ruff.

        Args:
            TBD: for Ruff.

        Returns:
            TBD: for Ruff.
        """
        return self.trajectory.positions
    
    def setpoint_at(self, t: float, lookahead_t: float = 0.15) -> np.ndarray:
        """Setpoint for sim."""
        ts = self.trajectory.timestamps
        tq = min(t + lookahead_t, ts[-1])
        return np.array([np.interp(tq, ts, self.trajectory.positions[:, k])
                         for k in range(3)])

    def _check_obsticle2(
            self,
            p_ref_LL: np.array,
            pOLL_array: np.ndarray
    ) -> tuple[bool, np.array]:
        """Checks if a trajectory point is inside an obsticle.
        
        Args:
            p_ref_LL:       Trajectory point to be checked.
            obs:            Environment state observation.

        Returns:
            is_inside_obsticle:     Boolian value if point is inside true if its outside
            obsticle:               Centre point of obsticle that is violated.
        """
        if len(pOLL_array) == 0:
           return False, None
        r_obsticle = R_OBSTACLE

        distance = np.linalg.norm(p_ref_LL[0:2] - pOLL_array[:, 0:2], axis=1)
        
        is_inside_obsticle = np.any(distance < r_obsticle)
        obsticle = pOLL_array[np.argmin(distance)]
        
        return is_inside_obsticle, obsticle
    
    def _get_obsticle_push(
            self,
            p_ref_LL: np.array,
            obsticle: np.array,
            push_vector: np.array
    ) -> float:
        """Get the push vector to avoid the obsticle.
        
        Args:
            p_ref_LL:       Trajectory point from which to push away from the obsticle.
            obsticle:       Centre point of obsticle that is violated.
            push_vector:    Push vector to avoid the obsticle.

        Returns:
            push:           Length of push vector to avoid the obsticle.
        """
        r_obsticle = R_OBSTACLE
        push_steps = 0.01
        i = 0
        p_ref_LL = p_ref_LL.copy()
        while np.linalg.norm(p_ref_LL[:2] - obsticle[:2]) < r_obsticle:
            p_ref_LL[0:2] += push_steps * push_vector[0:2]
            i += 1
        distance = p_ref_LL[0:2] - obsticle[0:2]
        #breakpoint()
        push = i * push_steps + 0.03
        return push
    
    def _check_gate3(
            self,
            p_ref_LL: np.array,
            pGLL_array: np.ndarray,
            y_GBL_array: np.ndarray
    ) -> tuple[bool, np.array, float]:
        """Checks if a trajectory point is colliding with any gate frame.
        
        Args:
            p_ref_LL:           Trajectory point to check.
            pGLL_array:         Center positions of all gates.
            y_GBL_array:        Yaws of all the gates.
            
        Returns:
            is_inside_frame:    True if the point is colliding with a gate's solid frame
                                structure.
            centre:             Centre of the gate that's been violated.
            yaw:                Yaw of the violated gate.
        """
        # Transform ref point into each gate's local frame
        diff = p_ref_LL - pGLL_array
        x = diff[:, 0] * np.cos(y_GBL_array) + diff[:, 1] * np.sin(y_GBL_array)
        y = -diff[:, 0] * np.sin(y_GBL_array) + diff[:, 1] * np.cos(y_GBL_array)
        z = diff[:, 2]

        z0 = FRAME_OPENING / 2 - DRONE_RADIUS
        y0 = FRAME_OPENING / 2 - DRONE_RADIUS

        z1 = FRAME_WIDTH / 2 + DRONE_RADIUS
        y1 = FRAME_WIDTH / 2 + DRONE_RADIUS

        xmax = FRAME_THICK / 2
        m = xmax/(z1 - z0)

        check_z_x_axis = (abs(z) > z0) & (abs(z) < z1) & (abs(x) < m * (abs(z) - z0)) & (abs(y) < y1)
        check_y_x_axis = (abs(y) > y0) & (abs(y) < y1) & (abs(x) < m * (abs(y) - y0)) & (abs(z) < z1)

        is_inside_frame = np.any(check_z_x_axis | check_y_x_axis)

        if not is_inside_frame:
            return False, None, None
        
        g = int(np.argmax(check_z_x_axis | check_y_x_axis))
        centre = pGLL_array[g]
        yaw = y_GBL_array[g]
        return True, centre, yaw

    def _get_gate_push(
            self,
            p_ref_LL: np.array,
            centre: np.array,
            yaw: float,
            push_vector: np.array
    ) -> np.array:
        """Get the push length to avoid the gate frame.
        
        Args:
            p_ref_LL:       Trajectory point from which to push away from the gate frame.
            centre:         Centre of the gate that's been violated.
            yaw:            Yaw of the violated gate.
            push_vector:    Push vector to avoid the gate frame.

        Returns:
            p_ref_LL:       New trajectory point after pushing away from the gate frame.
        """
        inside_gate = True
        push_steps = 0.01
        i = 0
        p_ref_LL = p_ref_LL.copy()

        #p_ref_LL += (FRAME_OPENING + 0.02) * push_vector

        while inside_gate:
            p_ref_LL += push_steps * push_vector
            inside_gate, _, _ = self._check_gate3(p_ref_LL, np.array([centre]), np.array([yaw]))
            i += 1
        #breakpoint()
        push = i * push_steps + 0.1
        return push
    
    def _get_gate_push_vector(
            self,
            centre: np.array,
            yaw: float,
            p_in: np.array,
            p_out: np.array,
            p_mid: np.array
    ) -> np.array:
        """Find the vector to push away from the gate frame.
        
        Args:
            centre:                 centre of the hit gate.
            yaw:                    yaw of the hit gate.
            p_in:                   entering point of the hit gate.
            p_out:                  Exiting point of the hit gate.
            
        Returns:
            push_vector:            Vector that pushes away from the gate.
        """
        chord = p_out - p_in
        chord_norm = np.linalg.norm(chord)
        
        chord_unit = chord / chord_norm

        # Vector from centre to mid (the "away from centre" direction)
        away = p_mid - centre

        # Component of `away` along the chord direction
        proj_on_chord = np.dot(away, chord_unit) * chord_unit

        # The perpendicular component (vector rejection)
        perp = away - proj_on_chord

        # Normalize
        perp_norm = np.linalg.norm(perp)

        return perp / perp_norm