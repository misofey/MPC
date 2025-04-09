import numpy as np

indices = {
    "pos_x": 0,
    "pos_y": 1,
    "heading_cos": 2,
    "heading_sin": 3,
    "vx": 4,
    "vy": 5,
    "r": 6,
    "steering": 7,
    "steering_dist": 8,
    "d_f": 9,
}


class Dynamics:
    """class responsible for the dynamics of the car and the measurement of the state for the output feedback controller"""

    def __init__(self, dt=0.002, disturbance: bool = False):
        self.m = 180  # Car mass [kg]
        self.I_z = 294  # TODO: unit
        self.wbase = 1.53  # wheel base [m]
        self.x_cg = 0.57  # C.G x location [m]
        self.lf = self.x_cg * self.wbase  # Front moment arm [m]
        self.lr = (1 - self.x_cg) * self.wbase  # Rear moment arm [m]

        [self.Cf, self.Cr] = self.get_tyre_stiffness()
        self.dt = dt

        if disturbance:
            self.nx = 10
            self.disturbed = True

            self.measurement_matrix = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                ]
            )
            print("Dynamics are with disturbance")
        else:
            self.nx = 8
            self.disturbed = False
            self.measurement_matrix = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )
            print("Dynamics are without disturbance")

        # noises for the partial state measurement
        self.measurement_noises = np.array([0.1, 0.1, 0.03, 0.03, 0.0001, 0.001, 0.001])

        # self.measurement_covariance = np.diag(1 / self.measurement_noises)

        seed = 1
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

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
        if self.disturbed:
            steering_disturbance = x[indices["steering_dist"]]
            force_disturbance = x[indices["d_f"]]
        else:
            steering_disturbance = 0
            force_disturbance = 0
        x_dot[0] = x[2] * x[4] - x[3] * x[5]  # px
        x_dot[1] = x[3] * x[4] + x[2] * x[5]  # py
        x_dot[2] = -x[6] * x[3]  # cos_head
        x_dot[3] = x[6] * x[2]  # sin_head
        x_dot[4] = 0  # vx
        x_dot[5] = (
            (
                -(self.Cf + self.Cr) / (self.m * x[4]) * x[5]
                + (-x[4] + (self.Cr * self.lr - self.Cf * self.lf) / (self.m * x[4]))
                * x[6]
            )
            - self.Cf / self.m * (x[7] + steering_disturbance)
            + force_disturbance
        )  # vy
        x_dot[6] = (
            (self.lr * self.Cr - self.lf * self.Cf) / (self.I_z * x[4]) * x[5]
            - (self.lf * self.lf * self.Cf + self.lr * self.lr * self.Cr)
            / (self.I_z * x[4])
            * x[6]
            - (self.Cf * self.lf) / self.I_z * (x[7] + steering_disturbance)
        )  # r
        x_dot[7] = u

        if self.disturbed:
            x_dot[8] = 0  # disturbance derivatives are zero
            x_dot[9] = 0
        return x_dot

    def rk4_integraton(self, xk, u) -> np.ndarray:
        k1 = self.single_track_model(xk, u)
        k2 = self.single_track_model(xk + self.dt / 2 * k1, u)
        k3 = self.single_track_model(xk + self.dt / 2 * k2, u)
        k4 = self.single_track_model(xk + self.dt * k3, u)

        return xk + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # return xk + self.dt * self.single_track_model(xk, u)

    def steering_dist_jacobian(self):
        return np.array(
            [
                0,
                0,
                0,
                0,
                0,
                -self.Cf / self.m,
                -(self.Cf * self.lf) / self.I_z,
                0,
                0,
                0,
            ]
        )

    def side_force_dist_jacobian(self):
        return np.array(
            [
                0,
                0,
                0,
                0,
                0,
                1 / self.m,
                0,
                0,
                0,
                0,
            ]
        )

    def jacobian_forward_euler(self, x):
        """continous time derivative linearization at the current state
        returns: A, B matrices"""
        nx = len(x)

        # linearize tire forces /wr vx
        tf11dvx = -(self.Cf + self.Cr) / self.m * x[5] * np.log(x[4])
        tf11dvy = -(self.Cf + self.Cr) / (self.m * x[4])
        tf12dvx = -x[6] + (self.Cr * self.lr - self.Cf * self.lf) / self.m * np.log(
            x[4]
        )
        tf12dr = -x[4] + (self.Cr * self.lr - self.Cf * self.lf) / (self.m * x[4])
        tf1ddelta = -self.Cf / self.m
        tf21dvx = (
            (self.lr * self.Cr - self.lf * self.Cf) / (self.I_z) * x[5] * np.log(x[4])
        )
        tf21dvy = (self.lr * self.Cr - self.lf * self.Cf) / (self.I_z * x[4])
        tf22dvx = (
            (self.lf * self.lf * self.Cf + self.lr * self.lr * self.Cr)
            / self.I_z
            * x[6]
        ) * np.log(x[4])
        tf22dr = (self.lf * self.lf * self.Cf + self.lr * self.lr * self.Cr) / (
            self.I_z * x[4]
        )
        tf2ddelta = -(self.Cf * self.lf) / self.I_z

        A = np.zeros([nx, nx])
        A[0, :8] = np.array([0, 0, x[4], -x[5], x[2], -x[3], 0, 0])
        A[1, :8] = np.array([0, 0, x[5], x[4], x[3], x[2], 0, 0])
        A[2, :8] = np.array([0, 0, 0, -x[6], 0, 0, -x[3], 0])
        A[3, :8] = np.array([0, 0, x[6], 0, 0, 0, x[2], 0])
        A[4, :8] = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        A[5, :8] = np.array([0, 0, 0, 0, tf11dvx + tf12dvx, tf11dvy, tf12dr, tf1ddelta])
        A[6, :8] = np.array([0, 0, 0, 0, tf21dvx + tf22dvx, tf21dvy, tf22dr, tf2ddelta])
        A[7, :8] = np.array([0, 0, 0, 0, 0, 0, 0, 0])

        B = np.zeros(nx)
        B[7] = 1
        if self.disturbed:
            A[:, 8] = self.steering_dist_jacobian()
            A[:, 9] = self.side_force_dist_jacobian()

        return A, B, self.dt * A + np.eye(nx)

    def measurement_jacobian(self):
        """measurement measures everything but vy, linear measurement model, so"""
        return self.measurement_matrix

    def measure_state_noise(self, x):
        """measure state and add gaussian noise"""
        return (
            self.measurement_matrix @ x
        ) + self.measurement_noises * self.rng.normal(len(x))

    def measure_state_noiseless(self, x):
        return self.measurement_matrix @ x
