import numpy as np
import scipy as sp
from continuous_dynamics import Dynamics, indices


class CarEKF:
    def __init__(
        self,
        dt,
        disturbance: bool,
        inital_state=None,
    ):
        if inital_state is None:
            self.x_est = np.zeros(9)
        else:
            self.x_est = np.array(inital_state)

        self.dynamics = Dynamics(dt, disturbance)
        self.disturbed = disturbance
        if self.disturbed:
            self.nx = self.dynamics.nx
        else:
            self.nx = 8

        print("number of state estimated are: ", self.nx)
        # self.P = np.diag(np.ones(self.nx))  # bad initial state estimate

        self.P = np.diag([1, 1, 1, 1, 1, 10.0, 1.0, 1.0, 0.0, 1000000])
        self.Q = np.diag(
            [0.05, 0.05, 0.01, 0.01, 0.001, 0.1, 0.01, 0.01, 0.0, 0.0]
        )  # assume no process noise
        self.R = np.diag(self.dynamics.measurement_noises)

        self.x_real = np.zeros(self.nx)

        self.dt = dt

    def time_update(self, u):
        x_dot = self.dynamics.single_track_model(self.x_est, u)
        A, B, F = self.dynamics.jacobian_forward_euler(self.x_est)
        self.x_est = x_dot * self.dt + self.x_est
        # print("covariance is: ", self.P)
        self.P = F @ self.P @ F.T + self.Q

    def measurement_update(self, x_meas):
        assert len(x_meas) == 7
        gain = self.kalman_gain()
        self.x_est = self.x_est + gain @ (x_meas - self.measure_x_est())
        self.P = self.P - gain @ self.dynamics.measurement_matrix @ self.P

    def estimate_full_state(self):
        return self.x_est

    def measure_x_est(self):
        return self.dynamics.measurement_matrix @ self.x_est

    def estimated_red_state(self):
        # red state is codeword for the states which are used by the mpc solver
        state_indices = [0, 1, 2, 3, 5, 6, 7]
        if self.disturbed:
            state_indices.append(8)
            state_indices.append(9)
        return self.x_est[state_indices]

    def kalman_gain(self):
        left_side = (
            self.dynamics.measurement_matrix
            @ self.P
            @ self.dynamics.measurement_matrix.T
            + self.R
        )
        right_side = self.P @ self.dynamics.measurement_matrix.T
        # return right_side @ np.linalg.inv(left_side)
        return np.linalg.solve(left_side.T, right_side.T).T
