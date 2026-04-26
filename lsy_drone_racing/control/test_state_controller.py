"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class StateController(Controller):
    """State controller following a pre-defined trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        
        self.estimated_sector_times = np.array([3.0, 2.25, 3.0, 2.0])

        self.nominal_gates_rpy = np.array([[0.0, 0.0, -0.78],
                                        [0.0, 0.0, 2.35],
                                        [0.0, 0.0, 3.14],
                                        [0.0, 0.0, 0.0]])
        self.nominal_gates_position = np.array([[0.5 , 0.25, 0.7], 
                                         [1.05, 0.75, 1.2], 
                                         [-1.0, -0.25, 0.7],
                                         [0.0, -0.75, 1.2]]) # four gates
        
        self.nominal_obst_position = np.array([[0.0, 0.75, 1.55], 
                                         [1.0, 0.25, 1.55], 
                                         [-1.5, -0.25, 1.55],
                                         [-0.5, -0.75, 1.55]]) 

        self._t_total = 25  # s
        #t = np.linspace(0, self._t_total, len(waypoints))
        self._des_pos_spline = None#CubicSpline(t, waypoints)

        self._tick = 0
        self._finished = False

        self._first_iteration = True
        #test command
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]

        self.kp = np.array([0.55, 0.55, 1.55])
        self.ki = np.array([0.045, 0.045, 0.05])
        self.kd = np.array([0.5, 0.5, 0.5])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)
        self.g = 9.81

        self._prev_action = []
        self.save_point = None

        self._old_gate_index = -1
        self._sector_times = np.array([0.0, 0.0, 0.0, 0.0])

        self.max_rec_depth = 5
        self.current_rec_depth = 0

    def is_element(self, elem, arr):
        for i in range(0, len(arr)):
            if elem == arr[i]:
                return True

        return False

    def check_spline_obst_collision(self, spline, t_start, t_end, obst_index, trigger_distance=0.2):
        t_arr = np.linspace(t_start, t_end, 40) # 40 rule of thumb
        obst = self.nominal_obst_position[obst_index]
        
        delta_min_norm = 200 # super high value for init
        delta_min = None

        for i in t_arr:
            des_pos = spline(i)
            delta = obst[:2] - des_pos[:2]
            delta_norm = np.linalg.norm(delta)
            if delta_norm < delta_min_norm:
                delta_min = delta
                delta_min_norm = delta_norm
        #print(f'Minimum distance: {delta_min_norm}')

        if delta_min_norm < trigger_distance:
            d = (delta_min/(delta_min_norm+0.001))*trigger_distance # move trigger distance away
            adj_pos = np.array([obst[0] - d[0], obst[1] - d[1], des_pos[2]])
            #print(f'Pos Adjusted: old pos: {des_pos}, new pos: {adj_pos}')
            return adj_pos
        return None

    def compute_waypoints_sector_0(self, t, current_gate_index, gate_position_array, pBL, vel_vec, add_waypoint=None):
        if self._old_gate_index != current_gate_index:
            self._sector_times[0] = t
            self._old_gate_index = current_gate_index

        pGL_prev, pGL_next = self.compute_prev_and_next_gate_waypoint(gate_position_array[0], self.nominal_gates_rpy[0], 0.4, 0.4)

        new_waypoints = np.array([[-1.5, 0.75, 0.05]])

        if add_waypoint is not None:
            new_waypoints = np.append(new_waypoints, [add_waypoint], axis=0)

        pGL_next[2] = gate_position_array[0][2]+0.2
        self.save_point = pGL_next
        new_waypoints = np.append(new_waypoints, [gate_position_array[0]], axis=0)
        new_waypoints = np.append(new_waypoints, [pGL_next], axis=0)

        t_array = np.linspace(self._sector_times[0], self._sector_times[0]+self.estimated_sector_times[0], len(new_waypoints))
        des_pos_spline = CubicSpline(t_array, new_waypoints)

        return des_pos_spline

    def compute_waypoints_sector_1(self, t, current_gate_index, gate_position_array, pBL, vel_vec, add_waypoint=None):
        if self._old_gate_index != current_gate_index:
            self._sector_times[1] = t
            self._old_gate_index = current_gate_index

        pGL_prev, pGL_next = self.compute_prev_and_next_gate_waypoint(gate_position_array[1], self.nominal_gates_rpy[1])
        
        new_waypoints = np.array([gate_position_array[0]])

        new_waypoints = np.append(new_waypoints, [[1.2, -0.15, 1.1]], axis=0)
        if add_waypoint is not None:
            new_waypoints = np.append(new_waypoints, [add_waypoint], axis=0)

        new_waypoints = np.append(new_waypoints, [gate_position_array[1]], axis=0)
        new_waypoints = np.append(new_waypoints, [pGL_next], axis=0)

        t_array = np.linspace(self._sector_times[1], self._sector_times[1]+self.estimated_sector_times[1], len(new_waypoints))
        des_pos_spline = CubicSpline(t_array, new_waypoints)

        return des_pos_spline

    def compute_waypoints_sector_2(self, t, current_gate_index, gate_position_array, pBL, vel_vec, add_waypoint=None):
        if self._old_gate_index != current_gate_index:
            self._sector_times[2] = t
            self._old_gate_index = current_gate_index
        
        pGL_prev, pGL_next = self.compute_prev_and_next_gate_waypoint(gate_position_array[2], self.nominal_gates_rpy[2], 0.4, 0.3)
        
        new_waypoints = np.array([gate_position_array[1]])

        if add_waypoint is not None:
            new_waypoints = np.append(new_waypoints, [add_waypoint], axis=0)

        new_waypoints = np.append(new_waypoints, [[-0.5, -0.05, 0.8]], axis=0)
        new_waypoints = np.append(new_waypoints, [gate_position_array[2]], axis=0)
        new_waypoints = np.append(new_waypoints, [pGL_next], axis=0)
        
        t_array = np.linspace(self._sector_times[2], self._sector_times[2]+self.estimated_sector_times[2], len(new_waypoints))
        des_pos_spline = CubicSpline(t_array, new_waypoints)

        return des_pos_spline

    def compute_waypoints_sector_3(self, t, current_gate_index, gate_position_array, pBL, vel_vec, add_waypoint=None):
        if self._old_gate_index != current_gate_index:
            self._sector_times[3] = t
            self._old_gate_index = current_gate_index
        
        pGL0_prev, pGL0_next = self.compute_prev_and_next_gate_waypoint(gate_position_array[2], self.nominal_gates_rpy[2], 0.1, 0.1)
        pGL_prev, pGL_next = self.compute_prev_and_next_gate_waypoint(gate_position_array[3], self.nominal_gates_rpy[3], 0.2, 0.4)
        new_waypoints = np.array([pGL0_prev])
        new_waypoints = np.append(new_waypoints, [[-0.5, -0.4, 0.9]], axis=0)

        if add_waypoint is not None:
            new_waypoints = np.append(new_waypoints, [add_waypoint], axis=0)

        new_waypoints = np.append(new_waypoints, [gate_position_array[3]], axis=0)
        new_waypoints = np.append(new_waypoints, [pGL_next], axis=0)

        t_array = np.linspace(self._sector_times[3], self._sector_times[3]+self.estimated_sector_times[3], len(new_waypoints))
        des_pos_spline = CubicSpline(t_array, new_waypoints)
        
        return des_pos_spline


    def compute_waypoint_trajectory(self, t, current_gate_index, gate_position_array, pBL, vel_vec, add_waypoint=None):
        if self.is_element(current_gate_index, [0]):
            des_pos_spline = self.compute_waypoints_sector_0(t, current_gate_index, gate_position_array, pBL, vel_vec, add_waypoint)
            feedback = self.check_spline_obst_collision(des_pos_spline, self._sector_times[0], self._sector_times[0]+self.estimated_sector_times[0], 0)

        elif self.is_element(current_gate_index, [1]):
            des_pos_spline = self.compute_waypoints_sector_1(t, current_gate_index, gate_position_array, pBL, vel_vec, add_waypoint)
            feedback = self.check_spline_obst_collision(des_pos_spline, self._sector_times[1], self._sector_times[1]+self.estimated_sector_times[1], 0)

        elif self.is_element(current_gate_index, [2]):
            des_pos_spline = self.compute_waypoints_sector_2(t, current_gate_index, gate_position_array, pBL, vel_vec, add_waypoint)
            feedback = self.check_spline_obst_collision(des_pos_spline, self._sector_times[2], self._sector_times[2]+self.estimated_sector_times[2], 1)

        elif self.is_element(current_gate_index, [3]):
            des_pos_spline = self.compute_waypoints_sector_3(t, current_gate_index, gate_position_array, pBL, vel_vec, add_waypoint)
            feedback = self.check_spline_obst_collision(des_pos_spline, self._sector_times[3], self._sector_times[3]+self.estimated_sector_times[3], 3)

        if (feedback is not None) and (self.current_rec_depth < self.max_rec_depth): # obstacle violation
            self.current_rec_depth += 1
            des_pos_spline = self.compute_waypoint_trajectory(t, current_gate_index, gate_position_array, pBL, vel_vec, feedback)
            self.current_rec_depth = 0

        return des_pos_spline

    def compute_prev_and_next_gate_waypoint(self, pTL, rpy, offset_prev = 0.2, offset_next = 0.4):
        
        alpha_yaw = rpy[2] # for now use yaw angle only
        pDelta = np.array([np.cos(alpha_yaw), np.sin(alpha_yaw), 0])
        pTL_prev = pTL - offset_prev*pDelta
        pTL_next = pTL + offset_next*pDelta

        return pTL_prev, pTL_next

    def check_violated_obstacle(self, pBL, pGL_array, delta=0.2):
        for i in range(0, len(pGL_array)):
            pGL = pGL_array[i]
            dGB_2D = np.linalg.norm(pGL[:2] - pBL[:2])

            if dGB_2D < delta:
                return [pGL, i]

        return None

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:  # Maximum duration reached
            self._finished = True
        
        pTL_index = obs['target_gate']
        pTL_array = obs['gates_pos']
        qTL_array = obs['gates_quat']
        pTL_visited = obs['gates_visited']
        pGL_array = obs['obstacles_pos']
        pBL = obs['pos']
        vBLL = obs['vel']
        quat = obs['quat']
        pTL = pTL_array[pTL_index]
        
        pGL_visited = obs['obstacles_visited']
        
        rotations = R.from_quat(qTL_array)
        self.nominal_gates_rpy = rotations.as_euler('xyz', degrees=False)

        if pTL_index == -1:
            self._finished = True
            return self._prev_action

        dTB_2D = np.linalg.norm(pTL[:2] - pBL[:2])
        violation = self.check_violated_obstacle(pBL, pGL_array)

        if (self._first_iteration or 
            (np.linalg.norm(self.nominal_gates_position[pTL_index] - pTL) > 0.01 and dTB_2D < 0.65) or 
            (self._old_gate_index != pTL_index)): #(violation is not None)):
        #    # compute new trajectory
        #    #violated_obstacle = violation[0] 
        #    #i_violated_obstacle = violation[1]
        #    pos_des = self._des_pos_spline(t)
            new_spline = self.compute_waypoint_trajectory(t, pTL_index, pTL_array, pBL, vBLL) # TBD: calculate new trajectory once new information about the gates-position is available
            self._des_pos_spline = new_spline
            self._first_iteration = False
            self.nominal_gates_position = pTL_array

        action = self.PID(self._des_pos_spline, pBL, vBLL, quat, t)
        self._prev_action = action
        #action = np.concatenate((des_pos, np.zeros(10)), dtype=np.float32)
        return action

    def PID(self, spline, pBL, vBLL, quat, t):
        vel_spline = spline.derivative()
        
        des_pos = spline(t)
        des_vel = vel_spline(t)
        des_yaw = 0.0

        # Calculate the deviations from the desired trajectory
        pos_error = des_pos - pBL
        vel_error = des_vel - vBLL

        # Update integral error
        self.i_error += pos_error * (1 / self._freq)
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        # Compute target thrust
        target_thrust = np.zeros(3)
        target_thrust += self.kp * pos_error
        target_thrust += self.ki * self.i_error
        target_thrust += self.kd * vel_error
        target_thrust[2] += self.drone_mass * self.g

        # Update z_axis to the current orientation of the drone
        z_axis = R.from_quat(quat).as_matrix()[:, 2]

        # update current thrust
        thrust_desired = target_thrust.dot(z_axis)

        # update z_axis_desired
        z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
        x_c_des = np.array([np.cos(des_yaw), np.sin(des_yaw), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired)
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)

        action = np.concatenate([euler_desired, [thrust_desired]], dtype=np.float32)

        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        setpoint = self._des_pos_spline(self._tick / self._freq).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        trajectory = self._des_pos_spline(np.linspace(0, self._t_total, 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
