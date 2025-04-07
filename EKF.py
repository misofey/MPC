import numpy as np
import scipy as sp
from continuous_dynamics import Dynamics


class CarEKF:
    def __init__(
        self,
        dt,
        disturbance: bool,
        inital_state=None,
    ):
        if inital_state is None:
            self.x_est = np.zeros(8)
        else:
            self.x_est = np.array(inital_state)

        self.dynamics = Dynamics(dt, disturbance)
        if disturbance:
            self.nx = 9
        else:
            self.nx = 8

        print("number of state estimated are: ", self.nx)
        self.P = np.diag(np.ones(self.nx))  # bad initial state estimate
        self.Q = np.zeros((self.nx, self.nx))  # assume no process noise
        self.R = np.diag(self.dynamics.measurement_noises)

        self.x_real = np.zeros(self.nx)

        self.dt = dt

    def time_update(self, u):
        x_dot = self.dynamics.single_track_model(self.x_est, u)
        A, B, F = self.dynamics.jacobian_forward_euler(self.x_est)
        self.x_est = x_dot * self.dt + self.x_est
        print("covariance is: ", self.P)
        self.P = F @ self.P @ F.T + self.Q

    def measurement_update(self, x_meas):
        assert len(x_meas) == self.nx - 1
        gain = self.kalman_gain()
        self.x_est = self.x_est + gain @ (x_meas - self.measure_x_est())
        self.P = self.P - gain @ self.dynamics.measurement_matrix @ self.P

    def estimated_state(self):
        return self.x_est

    def measure_x_est(self):
        return self.dynamics.measurement_matrix @ self.x_est

    def kalman_gain(self):
        left_side = (
            self.dynamics.measurement_matrix
            @ self.P
            @ self.dynamics.measurement_matrix.T
            + self.R
        )
        right_side = self.P @ self.dynamics.measurement_matrix.T

        return np.linalg.solve(left_side.T, right_side.T).T
