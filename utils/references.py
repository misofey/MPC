from utils.continuous_dynamics import Dynamics
import numpy as np


class ReferenceGenerator:
    """generates lateral velocity, yaw rate and steering references from a waypoint/heading array
    this file uses steering as the wheel angle, right positive"""

    def __init__(self, N, dt, target_vel):
        self.N = N
        self.dt = dt

        self.dynamics = Dynamics(N, dt)
        # self.dcgains = [-0.7314, -1.6784]
        self.target_vel = target_vel
        self.dcgains = self.get_dcgains(self.target_vel)

    def get_dcgains(self, target_vel):
        A, B = self.dynamics.linear_steering_model(target_vel)
        return np.linalg.inv(A) @ B

    def get_vy_steer_default_speed(self, r):
        steer = r / self.dcgains[1]
        vy = steer * self.dcgains[0]
        return vy, steer

    def get_vy_steer_custom_speed(self, r, vel):
        dcgains = self.get_dcgains(vel)
        steer = r / dcgains[1]
        vy = steer * dcgains[0]
        return vy, steer

    def waypoints_to_references_linear(self, waypoints, headings, speeds):
        """
        assumes fields pos_x, pos_y, cos_head, sin_head
        add fields vy, r, steering
        removes fields cos_heading, pos_x"""
        references = np.zeros(self.N + 1, 6)

        yawrates = headings / speeds
        steerings = yawrates / self.dcgains[1]
        vys = steerings * self.dcgains[0]

        references[:, 0:3] = waypoints[:, 1:4]  # pos_y, head_cos, head_sin
        references[:, 3] = vys
        references[:, 4] = yawrates
        references[:, 5] = steerings
        return references
