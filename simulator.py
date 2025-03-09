from ava_planning.steering_test import waypoints

from NLMPC import NLSolver, NLOcp
import matplotlib.pyplot as plt
import numpy as np
from utils.path_planning import SkidpadPlanner


class Dynamics:
    def __init__(self):
        self.m = 180  # Car mass [kg]
        self.I_z = 294  # TODO: unit
        self.wbase = 1.53  # wheel base [m]
        self.x_cg = 0.57  # C.G x location [m]
        self.lf = self.x_cg * self.wbase  # Front moment arm [m]
        self.lr = (1 - self.x_cg) * self.wbase  # Rear moment arm [m]

        [self.Cf, self.Cr] = self.get_tyre_stiffness()
        self.dt = 0.002

    def get_tyre_stiffness(self) -> (float, float):
        C_data_y = np.array(
            [
                1.537405752168591e04,
                2.417765976460659e04,
                3.121158998819641e04,
                3.636055041362088e04,
            ]
        )
        C_data_x = [300, 500, 700, 900]

        Cf = np.interp((9.81 * self.m / 2) * (1 - self.x_cg), C_data_x, C_data_y)
        Cr = np.interp((9.81 * self.m / 2) * self.x_cg, C_data_x, C_data_y)

        return Cf, Cr

    def single_track_model(self, x, u) -> np.ndarray:
        x_dot = np.zeros_like(x)
        x_dot[0] = x[2] * x[5] - x[3] * x[6]  # px
        x_dot[1] = x[3] * x[5] + x[2] * x[6]  # py
        x_dot[2] = -x[6] * x[3]  # cos_head
        x_dot[3] = x[6] * x[2]  # sin_head
        x_dot[4] = 0  # vx
        x_dot[5] = (
            -(self.Cf + self.Cr) * x[5]
            + (-x[4] + (self.Cr * self.lr - self.Cf * self.lf) / (self.m * x[4])) * x[6]
        ) - self.Cf * x[
            7
        ]  # vy
        x_dot[6] = (
            (self.lr * self.Cr - self.lf * self.Cf) / self.I_z * x[5]
            - (self.lf * self.lf * self.Cf + self.lr * self.lr * self.Cr)
            / (self.I_z * x[5])
            * x[6]
            - (self.Cr * self.lr) * x[7]
        )  # r
        x_dot[7] = u
        return x_dot

    def rk4_integraton(self, xk, u) -> np.ndarray:
        k1 = self.single_track_model(xk, u)
        k2 = self.single_track_model(xk + self.dt / 2 * k1, u)
        k3 = self.single_track_model(xk + self.dt / 2 * k2, u)
        k4 = self.single_track_model(xk + self.dt * k3, u)

        return xk + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class Simulator:
    def __init__(self):
        self.N = 20  # number of prediction timesteps
        self.Tf = 0.2  # final time

        self.ocp = NLOcp(self.N, self.Tf)
        self.MPC_controller = NLSolver(self.ocp)


        starting_pose = [0, 0, 1, 0]
        starting_velocity = [8, 0, 0]
        starting_steering_angle = [0]
        starting_state = np.array(
            starting_pose + starting_velocity[1:] + starting_steering_angle
        )

        self.pose = np.array(starting_state)
        self.vel = np.array(starting_velocity)
        self.steering = np.array(starting_steering_angle)


        self.waypoint_generator = SkidpadPlanner(
            target_vel=starting_velocity[0],
        )

        waypoints = self.waypoint_generator.request_waypoints()
        self.NLSolver.solve_problem(starting_state)

    def step(self):
