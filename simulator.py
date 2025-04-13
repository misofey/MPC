import os

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
# os.environ["ACADOS_SOURCE_DIR"] = (
#     "/home/miso/DUT/DUT25-Autonomous/src/controllers/acados"
# )
# os.environ["LD_LIBRARY_PATH"] = (
#     "/home/miso/DUT/DUT25-Autonomous/src/controllers/acados/lib"
# )


from NLMPC import NLOcp
from LMPC2 import LOcp
from LPVMPC import LPVOcp
from OFLMPC2 import OFLOcp
from EKF import CarEKF
from continuous_dynamics import Dynamics, indices

import matplotlib.pyplot as plt
import numpy as np
from utils.path_planning import SkidpadPlanner
from utils.step_planning import StepPlanner
from utils import path_planning
from utils import plotting
import logging
from skidpad_simulator import SkidpadSimulator
import yaml


class StepSimulator:
    def __init__(
        self,
        N,
        Tf,
        acados_print_level=0,
        starting_state=None,
        figures=False,
        model="LPV",
    ):
        self.N = N  # number of prediction timesteps
        self.Tf = Tf  # final time
        self.dt = self.Tf / self.N

        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s][%(levelname)s] - %(message)s",
            handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
        )

        self.N = N  # number of prediction timesteps
        self.Tf = Tf  # final time
        self.dt = self.Tf / self.N
        self.model = model

        if model == "NL":
            logging.info("Simulator Started with Nonlinear Model")
            self.ocp = NLOcp(self.N, self.Tf)
            self.MPC_controller = self.ocp
            self.disturbed = False
        elif model == "L":
            logging.info("Simulator Started with Linear Model")
            self.ocp = LOcp(self.N, self.Tf)
            self.MPC_controller = self.ocp
            self.disturbed = True
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

        self.waypoint_generator = StepPlanner(
            target_vel=starting_velocity[0], Nt=self.N, dt=self.dt
        )
        self.planned_references = np.zeros([self.N, 4])

        self.dynamics = Dynamics(self.dt, disturbance=self.disturbed)
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

    @full_state.setter
    def full_state(self, new_state):
        self.pose = new_state[:4]
        self.vel = new_state[4:7]
        self.steering = new_state[7]
        if self.disturbed:
            self.disturbances = new_state[[indices["steering_dist"], indices["d_f"]]]

    # def get_waypoints(self, x=None, y=None, heading=None):
    #     if x is None:
    #         x = self.pose[0]
    #     if y is None:
    #         y = self.pose[1]
    #     if heading is None:
    #         heading = np.arctan2(self.pose[3], self.pose[2])
    #     return self.waypoint_generator.request_waypoints(x, y, heading)

    def get_waypoints(self, x=None, y=None, heading=None):
        if x is None:
            x = self.pose[0]
        if y is None:
            y = self.pose[1]
        if heading is None:
            heading = np.arctan2(self.pose[3], self.pose[2])
        # print(self.full_state)
        return self.waypoint_generator.request_waypoints(x, y, heading)

    def simulate(self, n_steps) -> tuple[np.ndarray, np.ndarray]:

        simulated_state_trajectory = np.zeros((n_steps, self.dynamics.nx))
        simulated_input_trajectory = np.zeros((n_steps, 1))
        reference = np.zeros((n_steps, 4))

        for i in range(n_steps):

            waypoints, speeds, progress, heading_derotation, absolute_waypoints = (
                self.get_waypoints()
            )

            # logging.info(f"state before optimizing: {self.red_state}")
            status, trajectory, inputs = self.MPC_controller.optimize(
                self.red_state, waypoints, speeds
            )
            steer = trajectory[1, 6]

            steer = inputs[0]
            print("steer: ", steer)

            new_state = self.dynamics.rk4_integraton(self.full_state, steer)

            self.full_state = new_state
            self.planned_references = waypoints
            self.planned_trajectory = trajectory

            reference[i, :] = absolute_waypoints[0, :]
            simulated_state_trajectory[i, :] = new_state
            simulated_input_trajectory[i, :] = steer

        return simulated_state_trajectory, simulated_input_trajectory, reference

    def simulate_of(
        self, n_steps, initial_state_estimate: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        simulated_state_trajectory = np.zeros((n_steps, 10))
        simulated_input_trajectory = np.zeros((n_steps, 1))
        estimated_state_trajectory = np.zeros((n_steps, 10))
        planned_path = np.zeros((n_steps, 2))

        if initial_state_estimate is None:
            initial_pose_est = [-5.0, 0.0, 1.0, 0.0]
            initial_velocity_est = [15.0, 0.0, 0.0]
            initial_steering_angle_est = [0.0]
            initial_disturbances_est = [0.0, 0.0]
            initial_state_estimate = (
                initial_pose_est
                + initial_velocity_est
                + initial_steering_angle_est
                + initial_disturbances_est
            )

        SE = CarEKF(self.dt, True, inital_state=initial_state_estimate)

        for i in range(n_steps):

            x_est = SE.x_est
            pos_x_est = x_est[indices["pos_x"]]
            pos_y_est = x_est[indices["pos_y"]]
            heading_est = np.arctan2(
                x_est[indices["heading_sin"]], x_est[indices["heading_cos"]]
            )
            waypoints, speeds, progress, heading_derotation, absolute_waypoints = (
                self.get_waypoints(pos_x_est, pos_y_est, heading_est)
            )
            # waypoints, speeds, progress, heading_derotation, absolute_waypoints = (
            #     self.get_waypoints()
            # )

            planned_path[i, :] = [
                absolute_waypoints[0, 0],
                absolute_waypoints[0, 1],
            ]
            estimated_IC = SE.estimated_red_state()
            d_est = estimated_IC[-1]
            status, trajectory, inputs = self.MPC_controller.optimize(
                estimated_IC, waypoints, speeds, d_est
            )

            steer = inputs[0]
            print("steer: ", steer)

            new_state = self.dynamics.rk4_integraton(self.full_state, steer)

            self.full_state = new_state
            self.planned_references = waypoints
            self.planned_trajectory = trajectory

            SE.time_update(steer)
            # print(
            #     "measured_state: ",
            #     self.dynamics.measure_state_noise(self.full_state),
            # )
            # select noise here
            SE.measurement_update(
                self.dynamics.measure_state_noiseless(self.full_state)
            )

            estimated_state_trajectory[i, :] = SE.estimate_full_state()
            simulated_state_trajectory[i, :] = new_state
            simulated_input_trajectory[i, :] = steer

        return (
            simulated_state_trajectory,
            simulated_input_trajectory,
            estimated_state_trajectory,
            planned_path,
        )

    def step(self):
        plt.clf()

        waypoints, speeds, progress, heading_derotation = self.get_waypoints()

        # plotting.plot_path_and_heading(waypoints)
        # plt.draw()
        # print(speeds)
        status, trajectory, inputs = self.MPC_controller.optimize(
            self.red_state, waypoints, speeds
        )
        steer = trajectory[1, 6]

        steer = inputs[0]
        # inputs = np.append(inputs, [0])
        print("steer: ", steer)

        new_state = self.dynamics.rk4_integraton(self.full_state, steer)

        self.full_state = new_state
        self.planned_references = waypoints
        self.planned_trajectory = trajectory

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
        self.full_state = new_state

    def test_pathplanning(self):
        x = self.pose[0]
        y = self.pose[1]
        heading = np.arctan2(self.pose[3], self.pose[2])

        waypoints, speeds, progress, heading_derotation = (
            self.waypoint_generator.request_waypoints(x, y, heading)
        )

        plotting.plot_path_and_heading(waypoints)
        plt.show()

    def lsim(self, u, n_steps, initial_state_estimate: np.ndarray = None):
        if isinstance(u, float):
            u = np.ones(n_steps) * u

        simulated_state_trajectory = np.zeros((n_steps, self.dynamics.nx))
        simulated_input_trajectory = np.zeros((n_steps, 1))
        estimated_state_trajectory = np.zeros((n_steps, self.dynamics.nx))

        if initial_state_estimate is None:
            initial_pose_est = [0.0, 0.0, 1.0, 0.0]
            initial_velocity_est = [8.0, 0.0, 0.0]
            initial_steering_angle_est = [0.0]
            if self.disturbed:
                initial_disturbances_est = [0.0, 0.0]
            initial_state_estimate = (
                initial_pose_est
                + initial_velocity_est
                + initial_steering_angle_est
                + initial_disturbances_est
            )

        SE = CarEKF(self.dt, True, inital_state=initial_state_estimate)

        for i in range(n_steps):
            self.dynamics_step(u[i])
            simulated_state_trajectory[i, :] = self.full_state
            simulated_input_trajectory[i, :] = u[i]

            SE.time_update(u[i])
            # print(
            #     "measured_state: ",
            #     self.dynamics.measure_state_noiseless(self.full_state),
            # )
            SE.measurement_update(
                self.dynamics.measure_state_noiseless(self.full_state)
            )
            estimated_state_trajectory[i, :] = SE.estimate_full_state()

        return (
            simulated_state_trajectory,
            simulated_input_trajectory,
            estimated_state_trajectory,
        )

    def dlqr_sim(self, n_steps, K: np.ndarray):

        simulated_state_trajectory = np.zeros((n_steps, self.dynamics.nx))
        simulated_input_trajectory = np.zeros((n_steps, 1))
        reference = np.zeros((n_steps, 4))
        with open("parameters_NL.yaml", "r") as file:
            params = yaml.safe_load(file)

        rate_limit = params["model"]["max_steering_rate"]
        angle_limit = params["model"]["max_steering_angle"]

        for i in range(n_steps):

            waypoints, speeds, progress, heading_derotation, absolute_waypoints = (
                self.get_waypoints()
            )

            # logging.info(f"state before optimizing: {self.red_state}")
            status, trajectory, inputs = self.MPC_controller.optimize(
                self.red_state, waypoints, speeds
            )

            dt = self.dynamics.dt
            y_ref = waypoints[1]
            ref_state
            ref_state = np.zeros(5)
            ref_state[0] = y_ref
            effect_state = self.full_state[[1, 3, 5, 6, 7]]
            steer = -K @ (ref_state - effect_state)
            steer = np.clip(steer, -rate_limit, angle_limit)
            current_steer = self.full_state[-1]
            steer = np.clip(
                steer,
                (-angle_limit - current_steer) / dt,
                (angle_limit - current_steer) / dt,
            )

            new_state = self.dynamics.rk4_integraton(self.full_state, steer)

            self.full_state = new_state
            self.planned_references = waypoints
            self.planned_trajectory = trajectory

            reference[i, :] = absolute_waypoints[0, :]
            simulated_state_trajectory[i, :] = new_state
            simulated_input_trajectory[i, :] = steer

        return simulated_state_trajectory, simulated_input_trajectory, reference


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

        simulator = SkidpadSimulator(N, Tf, acados_print_level, starting_state)
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
        Tf = 1
        acados_print_level = 2

        starting_state = [
            -1.0,
            0.0,
            1.0,
            0,  # starting pose
            8.0,
            0.0,
            0.0,  # starting velocity
            0.0,  # starting steering angle
        ]

        simulator = StepSimulator(N, Tf, acados_print_level, starting_state)

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
        plt.show()
    history = np.array(history)
    print(history)
    # plotting.plot_path_and_heading(history)
    plt.show()
