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
from scipy.interpolate import CubicSpline

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

        # Same waypoints as in the attitude controller. Determined by trial and error.
        waypoints = np.array(
            [
                [-1.5, 0.75, 0.05],
                [0.13, 0.5, 0.6],
                [0.63, 0.13, 0.6],
                [1.25, -0.5, 0.6],
                [1.75, 0.0, 1.0],
                [0.7, 0.8, 1.1],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.3, 0.7],
                [-1.2, -0.3, 1.2],
                [-0.5, -0.5, 1.2],
                [0.5, -1.1, 0.9],
            ]
        )

        self._gates_rpy = np.array([-0.78, 2.35, 3.14, 0.0])
        self._gates_position = np.array([[0.5 , 0.25, 0.7], 
                                         [1.05, 0.75, 1.2], 
                                         [-1.0, -0.25, 0.7],
                                         [0.0, -0.75, 1.2]]) # four gates
        self._t_total = 15  # s
        t = np.linspace(0, self._t_total, len(waypoints))
        self._des_pos_spline = CubicSpline(t, waypoints)

        self._tick = 0
        self._finished = False

        self._first_iteration = True

    def compute_waypoint_trajectory(self, t, current_gate_index, gate_position_array, pBL, vel_vec, alpha=0.3):
        pT = gate_position_array[current_gate_index]
        pBL_next = (0.1*vel_vec/(np.linalg.norm(vel_vec))+0.001) + pBL # add 0.001 for numerical stability

        pGL_prev, pGL_next = self.compute_prev_and_next_gate_waypoint(gate_position_array[0], -0.78)
        if current_gate_index == 0 and self._first_iteration == False:
            pGL_prev = alpha*pBL_next + (1-alpha)*pGL_prev

        new_waypoints = np.array([[-1.5, 0.75, 0.05]])
        new_waypoints = np.append(new_waypoints, [pGL_prev], axis=0)
        new_waypoints = np.append(new_waypoints, [pGL_next], axis=0)
        new_waypoints = np.append(new_waypoints, [[1.25, -0.5, 0.7]], axis=0)
        new_waypoints = np.append(new_waypoints, [[2.0, 0.0, 1.2]], axis=0)

        pGL_prev, pGL_next = self.compute_prev_and_next_gate_waypoint(gate_position_array[1], 2.35)
        if current_gate_index == 1:
            pGL_prev = alpha*pBL_next + (1-alpha)*pGL_prev
        
        new_waypoints = np.append(new_waypoints, [pGL_prev], axis=0)
        new_waypoints = np.append(new_waypoints, [gate_position_array[1]], axis=0)
        new_waypoints = np.append(new_waypoints, [[0.0, 1.0, 0.9]], axis=0)
        new_waypoints = np.append(new_waypoints, [[-0.5, -0.05, 0.7]], axis=0)

        pGL_prev, pGL_next = self.compute_prev_and_next_gate_waypoint(gate_position_array[2], 3.14, 0.4, 0.1)
        if current_gate_index == 2:
            pGL_prev = alpha*pBL_next + (1-alpha)*pGL_prev

        new_waypoints = np.append(new_waypoints, [pGL_prev], axis=0)
        new_waypoints = np.append(new_waypoints, [pGL_next], axis=0)
        new_waypoints = np.append(new_waypoints, [[pGL_next[0], pGL_next[1], 1.2]], axis=0)
        new_waypoints = np.append(new_waypoints, [[-0.5, -0.5, 1.2]], axis=0)
        
        pGL_prev, pGL_next = self.compute_prev_and_next_gate_waypoint(gate_position_array[3], 0.0, 0.2, 0.4)
        if current_gate_index == 3:
            pGL_prev = alpha*pBL_next + (1-alpha)*pGL_prev

        new_waypoints = np.append(new_waypoints, [pGL_prev], axis=0)
        new_waypoints = np.append(new_waypoints, [pGL_next], axis=0)

        t_array = np.linspace(0, self._t_total, len(new_waypoints))
        des_pos_spline = CubicSpline(t_array, new_waypoints)

        return des_pos_spline

    def compute_prev_and_next_gate_waypoint(self, pTL, alpha, offset_prev = 0.4, offset_next = 0.4):
        pDelta = np.array([np.cos(alpha), np.sin(alpha), 0])
        pTL_prev = pTL - offset_prev*pDelta
        pTL_next = pTL + offset_next*pDelta

        return pTL_prev, pTL_next


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
        pBL = obs['pos']
        vBLL = obs['vel']
        pTL = pTL_array[pTL_index]
        
        dTB_2D = np.linalg.norm(pTL[:2] - pBL[:2])

        if (self._first_iteration == True) or (np.linalg.norm(self._gates_position[pTL_index] - pTL) > 0.01 and (dTB_2D < 0.65)):
            # compute new trajectory
            pos_des = self._des_pos_spline(t)
            new_spline = self.compute_waypoint_trajectory(t, pTL_index, pTL_array, pBL, vBLL) # TBD: calculate new trajectory once new information about the gates-position is available
            self._des_pos_spline = new_spline
            self._first_iteration = False
            self._gates_position = pTL_array

        des_pos = self._des_pos_spline(t)
        action = np.concatenate((des_pos, np.zeros(10)), dtype=np.float32)
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
