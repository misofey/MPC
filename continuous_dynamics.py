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
            self.steering_angle_disturbance = 0.01
            self.nx = 9
            self.disturbed = True
        else:
            self.steering_angle_disturbance = 0
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

        # noises for the partial state measurement
        self.measurement_noises = np.array(
            [0.3, 0.3, 0.01, 0.01, 0.00001, 0.01, 0.00001]
        )

        self.measurement_covariance = np.diag(self.measurement_noises)

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
        x_dot[0] = x[2] * x[4] - x[3] * x[5]  # px
        x_dot[1] = x[3] * x[4] + x[2] * x[5]  # py
        x_dot[2] = -x[6] * x[3]  # cos_head
        x_dot[3] = x[6] * x[2]  # sin_head
        x_dot[4] = 0  # vx
        x_dot[5] = (
            -(self.Cf + self.Cr) / (self.m * x[4]) * x[5]
            + (-x[4] + (self.Cr * self.lr - self.Cf * self.lf) / (self.m * x[4])) * x[6]
        ) - self.Cf / self.m * (
            x[7] + self.steering_angle_disturbance
        )  # vy
        x_dot[6] = (
            (self.lr * self.Cr - self.lf * self.Cf) / (self.I_z * x[4]) * x[5]
            - (self.lf * self.lf * self.Cf + self.lr * self.lr * self.Cr)
            / (self.I_z * x[4])
            - (self.Cf * self.lf) / self.I_z * (x[7] + self.steering_angle_disturbance)
        )  # r
        print(self.steering_angle_disturbance)
        x_dot[7] = u
        return x_dot

    def rk4_integraton(self, xk, u) -> np.ndarray:
        k1 = self.single_track_model(xk, u)
        k2 = self.single_track_model(xk + self.dt / 2 * k1, u)
        k3 = self.single_track_model(xk + self.dt / 2 * k2, u)
        k4 = self.single_track_model(xk + self.dt * k3, u)

        return xk + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def steering_dist_jacobian(self):
        np.array(
            [
                [0],
                [0],
                [0],
                [0],
                [0],
                [-self.Cf / self.m],
                [-(self.Cf * self.lf) / self.I_z],
                [0],
            ]
        )

    def side_force_dist_jacobian(self):
        np.array(
            [
                [0],
                [0],
                [0],
                [0],
                [0],
                [1],
                [0],
                [0],
            ]
        )

    def jacobian_at_state(self, x):
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
        tfx22dvx = (
            (self.lf * self.lf * self.Cf + self.lr * self.lr * self.Cr)
            / self.I_z
            * x[6]
        ) * np.log(x[4])
        tf22dr = (self.lf * self.lf * self.Cf + self.lr * self.lr * self.Cr) / (
            self.I_z * x[4]
        )
        tf2ddelta = -(self.Cf * self.lf) / self.I_z

        A = np.zeros([nx, nx])
        A[0, :7] = np.array([0, 0, x[4], -x[5], x[2], -x[3], 0, 0])
        A[1, :7] = np.array([0, 0, x[5], x[4], x[3], x[2], 0, 0])
        A[2, :7] = np.array([0, 0, 0, -x[6], 0, 0, -x[3], 0])
        A[3, :7] = np.array([0, 0, x[6], 0, 0, 0, x[2], 0])
        A[4, :7] = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        A[5, :7] = np.array([0, 0, 0, 0, tf11dvx + tf12dvx, tf11dvy, tf12dr, tf1ddelta])
        A[6, :7] = np.array(
            [0, 0, 0, 0, tf21dvx + tfx22dvx, tf21dvy, tf22dr, tf2ddelta]
        )
        A[7, :7] = np.array([0, 0, 0, 0, 0, 0, 0, 0])

        B = np.zeros([1, nx])
        B[0, 7] = 1
        # if self.disturbed:
        #     A = np.vstack((A, self.steering_dist_jacobian()))
        #     A = np.hstack((A, np.zeros((A.shape(0), 1))))
        F = np.vstack((A, B))

        return A, B, F

    def measurement_jacobian(self):
        """measurement measures everything but vy, linear measurement model, so"""
        return self.measurement_matrix

    def measure_state(self, x):
        """measure state and add gaussian noise"""
        return self.measurement_matrix @ x + self.measurement_noises * self.rng.normal(
            len(x)
        )

    # class Dynamics:
    #     def __init__(self, dt=0.002):
    #         self.m = 180  # Car mass [kg]
    #         self.I_z = 294  # TODO: unit
    #         self.wbase = 1.53  # wheel base [m]
    #         self.x_cg = 0.57  # C.G x location [m]
    #         self.lf = self.x_cg * self.wbase  # Front moment arm [m]
    #         self.lr = (1 - self.x_cg) * self.wbase  # Rear moment arm [m]
    #
    #         [self.Cf, self.Cr] = self.get_tyre_stiffness()
    #         self.dt = dt
    #
    #     def get_tyre_stiffness(self) -> tuple[float, float]:
    #         C_data_y = np.array(
    #             [
    #                 1.537405752168591e04,
    #                 2.417765976460659e04,
    #                 3.121158998819641e04,
    #                 3.636055041362088e04,
    #             ]
    #         )
    #         C_data_x = [300, 500, 700, 900]
    #
    #         Cf = np.interp((9.81 * self.m / 2) * (1 - self.x_cg), C_data_x, C_data_y)
    #         Cr = np.interp((9.81 * self.m / 2) * self.x_cg, C_data_x, C_data_y)
    #
    #         return float(Cf), float(Cr)
    #
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


#
#     def rk4_integraton(self, xk, u) -> np.ndarray:
#         k1 = self.single_track_model(xk, u)
#         k2 = self.single_track_model(xk + self.dt / 2 * k1, u)
#         k3 = self.single_track_model(xk + self.dt / 2 * k2, u)
#         k4 = self.single_track_model(xk + self.dt * k3, u)
#
#         return xk + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
