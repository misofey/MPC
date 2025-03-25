# import os
#
# if "LD_LIBRARY_PATH" not in os.environ:
#     os.environ["LD_LIBRARY_PATH"] = (
#         "/home/miso/DUT/DUT25-Autonomous/src/controllers/acados/lib"
#     )
#     if "ACADOS_SOURCE_DIR" not in os.environ:
#         # If it doesn't exist, export it with a default value
#         os.environ["ACADOS_SOURCE_DIR"] = (
#             "/home/miso/DUT/DUT25-Autonomous/src/controllers/acados"
#         )
# else:
#     os.environ["LD_LIBRARY_PATH"] += (
#         os.pathsep + "/home/miso/DUT/DUT25-Autonomous/src/controllers/acados/lib"
#     )
#     if "ACADOS_SOURCE_DIR" not in os.environ:
#         # If it doesn't exist, export it with a default value
#         os.environ["ACADOS_SOURCE_DIR"] = (
#             "/home/miso/DUT/DUT25-Autonomous/src/controllers/acados"
#         )
#

from NLMPC import NLSolver, NLOcp
from LMPC import LOcp


import matplotlib.pyplot as plt
import numpy as np
from utils.path_planning import SkidpadPlanner
from utils import path_planning
from utils import plotting


class Dynamics:
    def __init__(self, dt=0.002):
        self.m = 180  # Car mass [kg]
        self.I_z = 294  # TODO: unit
        self.wbase = 1.53  # wheel base [m]
        self.x_cg = 0.57  # C.G x location [m]
        self.lf = self.x_cg * self.wbase  # Front moment arm [m]
        self.lr = (1 - self.x_cg) * self.wbase  # Rear moment arm [m]

        [self.Cf, self.Cr] = self.get_tyre_stiffness()
        self.dt = dt

    def get_tyre_stiffness(self) -> tuple[float, float]:
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

        return float(Cf), float(Cr)

    def single_track_model(self, x, u) -> np.ndarray:
        x_dot = np.zeros_like(x)
        x_dot[0] = x[2] * x[4] - x[3] * x[5]  # px
        x_dot[1] = x[3] * x[4] + x[2] * x[5]  # py
        x_dot[2] = -x[6] * x[3]  # cos_head
        x_dot[3] = x[6] * x[2]  # sin_head
        x_dot[4] = 0  # vx
        x_dot[5] = (
            -(self.Cf + self.Cr) / (self.m * x[4]) * x[5]
            + (-x[4] + (self.Cr * self.lr - self.Cf * self.lf) / (self.m * x[4])) * x[6]
        ) - self.Cf / self.m * x[
            7
        ]  # vy
        x_dot[6] = (
            (self.lr * self.Cr - self.lf * self.Cf) / (self.I_z * x[4]) * x[5]
            - (self.lf * self.lf * self.Cf + self.lr * self.lr * self.Cr)
            / (self.I_z * x[4])
            * x[6]
            - (self.Cf * self.lf) / self.I_z * x[7]
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
    def __init__(
        self,
        N,
        Tf,
        acados_print_level=0,
        starting_state=None,
        starting_lap=None,
        figures=False,
        nonlin=False
    ):
        self.N = N  # number of prediction timesteps
        self.Tf = Tf  # final time
        self.dt = self.Tf / self.N
        if nonlin:
            print("Simulator Started with Nonlinear Model")
            self.ocp = NLOcp(self.N, self.Tf)
            self.MPC_controller = NLSolver(self.ocp, acados_print_level)
        else:
            print("Simulator Started with Linear Model")
            self.ocp = LOcp(self.N, self.Tf)
            self.MPC_controller = self.ocp

        if starting_state is None:
            starting_pose = [15.0, 0.1, 1.0, 0]
            starting_velocity = [15.0, 0.0, 0.0]
            starting_steering_angle = [0.0]
        else:
            starting_pose = starting_state[:4]
            starting_velocity = starting_state[4:7]
            starting_steering_angle = starting_state[7]
        if starting_lap is None:
            starting_lap = 0

        self.pose = np.array(starting_pose)
        self.vel = np.array(starting_velocity)
        self.steering = np.array(starting_steering_angle)
        self.lap = starting_lap

        self.waypoint_generator = SkidpadPlanner(
            target_vel=starting_velocity[0], Nt=self.N, dt=self.dt
        )

        # self.planned_trajectory = np.zeros([self.N, self.ocp.n_states])
        self.planned_references = np.zeros([self.N, 4])

        self.dynamics = Dynamics(self.dt)
        print("Simulator created!")

        self.figures = figures

        if figures:
            self.steering_ax = plt.figure().get_axes()[0]
            plt.figure()

    @property
    def full_state(self):
        return np.hstack((self.pose, self.vel, self.steering))

    @property
    def red_state(self):
        return np.hstack((self.pose, self.vel[1:], self.steering))

    def lapcounter(self, new_pose_x):
        if new_pose_x > path_planning.center and self.pose[0] <= path_planning.center:
            self.lap += 1

    @full_state.setter
    def full_state(self, new_state):
        self.pose = new_state[:4]
        self.vel = new_state[4:7]
        self.steering = new_state[7]

    def get_waypoints(self):
        x = self.pose[0]
        y = self.pose[1]
        heading = np.arctan2(self.pose[3], self.pose[2])
        lap = self.lap
        # print(self.full_state)
        return self.waypoint_generator.request_waypoints(x, y, heading, lap)

    def step(self):
        plt.clf()

        waypoints, speeds, progress, heading_derotation = self.get_waypoints()

        # plotting.plot_path_and_heading(waypoints)
        # plt.draw()
        # print(speeds)
        disturbance = 0#np.random.normal(0, 0.1, size=self.red_state.shape)
        disturbed_state = self.red_state + disturbance
        status, trajectory, inputs = self.MPC_controller.optimize(
            disturbed_state, waypoints, speeds
        )
        steer = trajectory[1, 6]

        steer = inputs[0]
        # inputs = np.append(inputs, [0])
        print("steer: ", steer)

        new_state = self.dynamics.rk4_integraton(self.full_state, steer)

        self.lapcounter(new_state[0])
        self.full_state = new_state
        self.planned_references = waypoints
        self.planned_trajectory = trajectory
        trig_viol = np.linalg.norm(self.planned_trajectory[:, 2:4], axis=1)-1
        print(f"Trigonometric const violation: {trig_viol}")

        # # THIS INCLUDES THE STEERING RATE AS WELL
        # self.planned_trajectory = np.concatenate(
        #     [trajectory, inputs.reshape([-1, 1])], axis=1
        # )

        plotting.plot_path_and_heading(self.planned_trajectory, self.planned_references)

        # t = np.linspace(0, Tf, N + 1)
        # plotting.plot_steering(simulator.planned_trajectory[:, :7], inputs, t)
        plt.draw()
        plt.show(block=False)

    def dynamics_step(self, input):
        new_state = self.dynamics.rk4_integraton(self.full_state, input)
        self.lapcounter(new_state[0])
        self.full_state = new_state

    def test_pathplanning(self):
        x = self.pose[0]
        y = self.pose[1]
        heading = np.arctan2(self.pose[3], self.pose[2])
        lap = self.lap

        waypoints, speeds, progress, heading_derotation = (
            self.waypoint_generator.request_waypoints(x, y, heading, lap)
        )

        plotting.plot_path_and_heading(waypoints)
        plt.show()


if __name__ == "__main__":
    N = 25
    Tf = 0.5
    acados_print_level = 2
    # starting_state = [
    #     5.0,
    #     0.1,
    #     1.0,
    #     0,
    # ]ng_state[7]
    simulator = Simulator(N, Tf, acados_print_level)
    # simulator.step()
    # simulator.step()
    # print(simulator.planned_trajectory)
    # plotting.plot_directions(simulator.planned_trajectory, simulator.planned_references)
    # plt.show()
    # t = np.linspace(0, Tf, N + 1)
    # plotting.plot_steering(
    #     simulator.planned_trajectory[:, :6], simulator.planned_trajectory[:, 7], t
    # )
    # plt.show()
    # simulator.test_pathplanning()
    plt.ion()
    history = [simulator.full_state]
    for i in range(1000):
        input("do a button press to optimize")
        # print(simulator.full_state)
        simulator.step()
        # print(simulator.planned_references)
        history.append(simulator.full_state)
        plt.show()
    history = np.array(history)
    print(history)
    # plotting.plot_path_and_heading(history)
    plt.show()