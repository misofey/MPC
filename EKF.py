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
            self.x_est = inital_state

        self.dynamics = Dynamics(disturbance, dt)
        if disturbance:
            self.nx = 9
        else:
            self.nx = 8

        self.P = np.diag(np.ones(self.nx))  # bad initial state estimate
        self.Q = np.zeros((self.nx, self.nx))  # assume no process noise
        self.R = self.dynamics.measurement_covariance

        self.x_real = np.zeros(self.nx)
        self.kalman_gain_left = self.dynamics.measurement_covariance

        self.dt = dt

    def time_update(self, u):
        A, B, F = self.dynamics.jacobian_at_state(self.x_est)
        self.x_est = (A @ self.x_est + B @ u) * self.dt + self.x_est
        self.P = F @ self.P @ F.T + self.Q

    def measurement_update(self, x_meas):
        assert x_meas.len == self.nx - 1
        gain = self.kalman_gain()
        self.x_est = self.x_est + gain @ (x_meas - self.measure_x_est())
        self.P = self.P - gain @ self.dynamics.measurement_matrix @ self.P

    def estimated_state(self):
        return self.x_est

    def measure_x_est(self):
        return self.dynamics.measurement_covariance @ self.x_est

    def kalman_gain(self):
        left_side = (
            self.dynamics.measurement_covariance
            @ self.P
            @ self.dynamics.measurement_covariance.T
            + self.R
        )
        right_side = (
            self.P
            @ self.dynamics.measurement_covariance
            @ self.dynamics.measurement_covariance.T
        )
        return np.linalg.solve(left_side, right_side)
