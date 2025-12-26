from NLMPC import NLOcp
from LMPC2 import LOcp
from LPVMPC import LPVOcp
from EKF import CarEKF
from OFLMPC2 import OFLOcp

from utils.continuous_dynamics import Dynamics, indices

import matplotlib.pyplot as plt
import numpy as np
from utils.skidpad_waypoints import SkidpadPlanner
from utils import skidpad_waypoints
from utils import plotting
import logging
import yaml


class SkidpadSimulator:
    def __init__(
        self,
        N,
        Tf,
        acados_print_level=0,
        starting_state=None,
        starting_lap=None,
        figures=False,
        model="LPV",
    ):

        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s][%(levelname)s] - %(message)s",
            handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
        )

        self.N = N  # number of prediction timesteps
        self.Tf = Tf  # final time
        self.dt = self.Tf / self.N

        if model == "NL":
            logging.info("Simulator Started with Nonlinear Model")
            self.ocp = NLOcp(self.N, self.Tf)
            self.MPC_controller = self.ocp
            self.disturbed = False
        elif model == "L":
            logging.info("Simulator Started with Linear Model")
            self.ocp = LOcp(self.N, self.Tf)
            self.MPC_controller = self.ocp
            self.disturbed = False
        elif model == "LPV":
            logging.info("Simulator Started with LPV Model")
            self.ocp = LPVOcp(self.N, self.Tf)
            self.MPC_controller = self.ocp
            self.disturbed = False
        elif model == "OFL":
            logging.info("Simulator Started with offset-free output feedback LPV Model")
            self.ocp = OFLOcp(self.N, self.Tf)
            self.MPC_controller = self.ocp
            self.disturbed = True
        elif model == "none":
            logging.info("Simulator Starter without controller")
            self.disturbed = True
        else:
            logging.error("Model not recognized. Please choose 'NL', 'L', or 'LPV'.")
            return

        if starting_state is None:
            starting_pose = [15.0, 0.1, 1.0, 0]
            starting_velocity = [8.0, 0.0, 0.0]
            starting_steering_angle = [0.0]
            if self.disturbed:
                starting_disturbances = [0.0, 0.0]
        else:
            starting_pose = starting_state[:4]
            starting_velocity = starting_state[4:7]
            starting_steering_angle = starting_state[7]
            if self.disturbed:
                starting_disturbances = [
                    starting_state[indices["steering_dist"]],
                    starting_state[indices["d_f"]],
                ]

        self.pose = np.array(starting_pose)
        self.vel = np.array(starting_velocity)
        self.steering = np.array(starting_steering_angle)
        if self.disturbed:
            self.disturbances = np.array(starting_disturbances)
        self.lap = starting_lap

        self.waypoint_generator = SkidpadPlanner(
            target_vel=starting_velocity[0], Nt=self.N, dt=self.dt
        )

        # self.planned_trajectory = np.zeros([self.N, self.ocp.n_states])
        self.planned_references = np.zeros([self.N, 4])

        self.dynamics = Dynamics(self.dt)
        logging.info(f"initial state: {self.red_state}")
        print("Simulator created!")

        self.figures = figures

        if figures:
            self.steering_ax = plt.figure().get_axes()[0]
            plt.figure()

    @property
    def full_state(self):
        if self.disturbed:
            return np.hstack((self.pose, self.vel, self.steering, self.disturbances))
        else:
            return np.hstack((self.pose, self.vel, self.steering))

    @property
    def red_state(self):
        return np.hstack((self.pose, self.vel[1:], self.steering))

    def lapcounter(self, new_pose_x):
        if (
            new_pose_x > skidpad_waypoints.center
            and self.pose[0] <= skidpad_waypoints.center
        ):
            self.lap += 1

    @full_state.setter
    def full_state(self, new_state):
        self.pose = new_state[:4]
        self.vel = new_state[4:7]
        self.steering = new_state[7]
        if self.disturbed:
            self.disturbances = new_state[[indices["steering_dist"], indices["d_f"]]]

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
        disturbance = 0  # np.random.normal(0, 0.1, size=self.red_state.shape)
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
        trig_viol = np.linalg.norm(self.planned_trajectory[:, 2:4], axis=1) - 1
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

    def simulate(self, n_steps) -> tuple[np.ndarray, np.ndarray]:

        simulated_state_trajectory = np.zeros((n_steps, self.dynamics.nx))
        simulated_input_trajectory = np.zeros((n_steps, 1))
        reference = np.zeros((n_steps, 4))

        for i in range(n_steps):

            waypoints, speeds, progress, heading_derotation, absolute_waypoints = (
                self.get_waypoints()
            )

            # logging.info(f"state before optimizing: {self.red_state}")
            # noisy_red_state = (
            #     self.red_state
            # )  # + np.random.normal(0, 0.01, size=self.red_state.shape)
            status, trajectory, inputs = self.MPC_controller.optimize(
                self.red_state, waypoints, speeds
            )
            steer = trajectory[1, 6]

            steer = inputs[0]
            print("steer: ", steer)

            self.dynamics_step(steer)

            self.planned_references = waypoints
            self.planned_trajectory = trajectory

            reference[i, :] = absolute_waypoints[0, :]
            simulated_state_trajectory[i, :] = self.full_state
            simulated_input_trajectory[i, :] = steer

        return simulated_state_trajectory, simulated_input_trajectory, reference

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

    def plot_skidpad_run():
        pass


if __name__ == "__main__":
    # step or skidpad
    simulate = "skidpad"
    if simulate == "skidpad":
        N = 10
        Tf = 0.5
        acados_print_level = 2
        starting_state = [
            0.0,
            0.0,
            1.0,
            0,  # starting pose
            8.0,
            0.0,
            0.0,  # starting velocity
            0.0,  # starting steering angle
            0.0,  # starting steering disturbance
        ]

        simulator = SkidpadSimulator(N, Tf, acados_print_level, starting_state, 0)
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
    elif simulate == "step":
        N = 50
        Tf = 0.5
        acados_print_level = 0

        starting_state = [
            0.0,
            0.0,
            1.0,
            0,  # starting pose
            10.0,
            0.0,
            0.0,  # starting velocity
            0.0,  # starting steering angle
        ]

        sim = SkidpadSimulator(N, Tf, acados_print_level, starting_state)
        sim.simulate(500)

        # plt.ion()
        # history = [simulator.full_state]
        # for i in range(1000):
        #     input("do a button press to optimize")
        #     # print(simulator.full_state)
        #     simulator.step()
        #     # print(simulator.planned_references)
        #     history.append(simulator.full_state)
        #     plt.show()
        # history = np.array(history)
        # print(history)
        # plt.show()
    # history = np.array(history)
    # print(history)
    # # plotting.plot_path_and_heading(history)
    # plt.show()
