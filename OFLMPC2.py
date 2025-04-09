from matplotlib import pyplot as plt
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
import casadi as cs
import numpy as np
import control as ct
import sympy as sp
from pprint import pprint as pr
import yaml
import logging

from continuous_dynamics import indices


class OFLOcp(AcadosOcp):

    def __init__(self, N: int, Tf: float, discrete: bool = True, debug: bool = False):
        AcadosOcp.__init__(self)

        # Initialize logging
        if debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format="[%(asctime)s][%(levelname)s] - %(message)s",
                handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s][%(levelname)s] - %(message)s",
                handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
            )
        # Load parameters from a YAML file
        with open("parameters.yaml", "r") as file:
            params = yaml.safe_load(file)

        # Assign parameters to class attributes
        self.params = params
        ### Constants ###
        self.discrete = discrete
        # Simulation constants
        self.N = N
        self.Tf = Tf
        self.n_states = 6
        self.n_inputs = 1
        self.n_outputs = 5
        self.n_parameters = 1
        self.n_disturbances = 1
        # Dynamics constants
        self.m = params["model"]["m"]  # Car mass [kg]
        self.I_z = params["model"]["I_z"]  # TODO: unit
        self.wbase = params["model"]["wbase"]  # wheel base [m]
        self.x_cg = params["model"]["x_cg"]  # C.G x location [m]
        self.lf = self.x_cg * self.wbase  # Front moment arm [m]
        self.lr = (1 - self.x_cg) * self.wbase  # Rear moment arm [m]

        [self.Cf, self.Cr] = self.get_tyre_stiffness()

        self.max_steering = params["model"]["max_steering_angle"]
        self.max_steering_rate = params["model"][
            "max_steering_rate"
        ]  # one second from full left to full right

        # Linearization point
        self.x_lin_point = np.array([0, 0, 0, 0, 0, 0])
        self.u_lin_point = np.array([0])
        self.p_lin_point = np.array([15.0])
        # Model setup
        self.model = AcadosModel()

        # Model name
        self.model.name = "NonlinearDynamicBycicleModel"

        ### Decision variables ###
        self.model.u = cs.MX.sym("u", self.n_inputs)
        self.model.x = cs.MX.sym("x", self.n_states)
        self.model.xdot = cs.MX.sym("xdot", self.n_states)

        ### Parameters ###
        self.model.p = cs.MX.sym("p", self.n_parameters + self.n_disturbances)

        # Set model dynamics
        self.get_dynamics()
        if self.discrete:
            print("Discrete dynamics is being used")
            self.set_discrete_dynamics()
        else:
            self.set_cont_dynamics()

        # Set cost
        self.set_cost()
        # Perform stability analysis
        self.stability()
        # Set terminal cost
        self.set_terminal_cost()
        # Set constraints
        self.set_constraints()
        # Set solver options
        self.set_solver_options()

        self.solver = AcadosOcpSolver(self)

        self.metrics = {"runtime": []}

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

        return Cf, Cr

    def get_dynamics(self):

        p_x = self.model.x[0, 0]
        p_y = self.model.x[1, 0]
        heading = self.model.x[2, 0]
        v_y = self.model.x[3, 0]
        omega = self.model.x[4, 0]
        steering_angle = self.model.x[5, 0]

        steering_rate = self.model.u[0, 0]

        v_x = self.model.p[0, 0]

        d_p_x = v_x * 1  # - v_y * np.sin(heading)

        d_p_y = v_x * heading + v_y * 1

        d_heading = omega

        d_v_y = -(self.Cf + self.Cr) / (self.m * v_x + 0.001) * v_y
        d_v_y += (
            (-v_x + (self.Cr * self.lr - self.Cf * self.lf))
            / (self.m * v_x + 0.001)
            * omega
        )
        d_v_y -= self.Cf / self.m * steering_angle

        d_omega = (
            (self.lr * self.Cr - self.lf * self.Cf) / (self.I_z * v_x + 0.001) * v_y
        )
        d_omega += (
            -(self.lf * self.lf * self.Cf + self.lr * self.lr * self.Cr)
            / (self.I_z * v_x + 0.001)
            * omega
        )
        d_omega -= self.lf * self.Cf / self.I_z * steering_angle

        d_steering = steering_rate

        self.f = cs.vertcat(d_p_x, d_p_y, d_heading, d_v_y, d_omega, d_steering)

        self.A = cs.jacobian(self.f, self.model.x)
        self.A = cs.Function("A", [self.model.x, self.model.u, self.model.p], [self.A])
        self.A = self.A(self.x_lin_point, self.u_lin_point, self.model.p)
        self.B = cs.jacobian(self.f, self.model.u)
        self.B = cs.Function("B", [self.model.x, self.model.u, self.model.p], [self.B])
        self.B = self.B(self.x_lin_point, self.u_lin_point, self.model.p)

        # self.B_d = np.ones((self.n_states, self.n_disturbances))
        self.B_d = np.zeros((self.n_states, self.n_disturbances))
        self.B_d[3, 0] = 1 / self.m
        self.C_d = np.eye(self.n_outputs, self.n_disturbances)

        self.f += self.B_d @ self.model.p[self.n_parameters :]

    def set_discrete_dynamics(self, type: str = "fe") -> None:

        if type == "rk4":
            self.apply_rk4_with_linearized_matrices()

        elif type == "fe":
            self.model.disc_dyn_expr = self.model.x + self.f * self.Tf / self.N

    def apply_rk4_with_linearized_matrices(self):

        dt = self.Tf / self.N  # Time step

        # TODO: The linearization point should be from x and u
        # idk why it works like this but not like that
        x_dot = (
            lambda x, u: self.A @ (x - self.x_lin_point)
            + self.B @ (u - self.u_lin_point)
            - self.x_lin_point
        )

        # Compute RK4 steps using linearized system
        k1 = x_dot(self.model.x, self.model.u)
        k2 = x_dot(self.model.x + (dt / 2) * k1, self.model.u)
        k3 = x_dot(self.model.x + (dt / 2) * k2, self.model.u)
        k4 = x_dot(self.model.x + dt * k3, self.model.u)

        # Compute the RK4 update step
        x_next = self.model.x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Set the discretized dynamics
        self.model.disc_dyn_expr = x_next

    def set_cont_dynamics(self) -> None:

        # TODO: same comment as at the ddiscrete model
        # self.model.f_expl_expr = self.A @ (self.model.x) + self.B @ (self.model.u)
        self.model.f_expl_expr = self.f

    def set_constraints(self) -> None:
        # Bounds for self.model.x
        self.constraints.idxbx = np.array([5])
        self.constraints.lbx = np.array([-self.max_steering])
        self.constraints.ubx = np.array([self.max_steering])

        # Intial condition
        self.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4, 5])
        self.constraints.lbx_0 = np.array([0, 0, 0, 0, 0, 0])
        self.constraints.ubx_0 = np.array([0, 0, 0, 0, 0, 0])

        # Bounds for input
        self.constraints.idxbu = np.array(
            [0]
        )  # the 0th input has the constraints, so J_bu = [1]
        self.constraints.lbu = np.array([-self.max_steering_rate])
        self.constraints.ubu = np.array([self.max_steering_rate])

        # Set terminal constraints
        # x_ref = cs.vcat((-1.9, -0.6, 0, 0, self.cost.yref[3]))
        # x_terminal_shifted = self.model.x[1:] - x_ref
        # self.model.con_h_expr_e = 1/2 * x_terminal_shifted.T @ self.P @ x_terminal_shifted
        # print(self.model.con_h_expr_e)
        # self.constraints.lh_e = np.array([-1000000000000])
        # self.constraints.uh_e = np.array([self.c])

    def set_cost(self) -> None:

        # Output selection matrix
        Vx = np.eye(self.n_states + self.n_inputs, self.n_states)

        self.cost.Vx = Vx
        self.cost.Vx_e = Vx

        Vu = np.zeros((self.n_states + self.n_inputs, self.n_inputs))
        Vu[-1, 0] = 1
        self.cost.Vu = Vu
        self.cost.Vu_e = Vu

        self.cost.cost_type = "LINEAR_LS"
        self.cost.cost_type_e = "LINEAR_LS"

        # Cost matrices
        Q = np.array(self.params["controller"]["Q"])
        R = np.array(self.params["controller"]["R"])
        q = self.params["controller"]["q"]
        r = self.params["controller"]["r"]
        self.cost.W = np.zeros(
            (self.n_states + self.n_inputs, self.n_states + self.n_inputs)
        )
        self.cost.W[: self.n_states, : self.n_states] = Q * q
        self.cost.W[self.n_states :, self.n_states :] = R * r
        self.cost.W_e = self.cost.W

        # Reference trajectory
        self.cost.yref = np.zeros(self.n_states + self.n_inputs)
        self.cost.yref_e = np.zeros(self.n_states + self.n_inputs)

        # Initial parameter trajectory
        self.parameter_values = np.array([9.0, 0.5])

    def set_terminal_cost(self) -> None:
        self.cost.W_e = np.zeros(
            (self.n_states + self.n_inputs, self.n_states + self.n_inputs)
        )
        # Terminal constraint does not constrain x position and input
        self.cost.W_e[1:-1, 1:-1] = 1 / 2 * self.P

    def set_solver_options(self) -> None:
        # set QP solver and integration
        self.dims.N = self.N
        self.solver_options.tf = self.Tf
        self.solver_options.N_horizon = self.N
        if self.discrete:
            # self.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
            self.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
            self.solver_options.nlp_solver_type = "SQP"
            self.solver_options.hessian_approx = "EXACT"
            self.solver_options.integrator_type = "DISCRETE"
            self.solver_options.globalization = "MERIT_BACKTRACKING"
        else:
            self.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
            self.solver_options.nlp_solver_type = "SQP"
            self.solver_options.hessian_approx = "EXACT"
            self.solver_options.integrator_type = "ERK"
            self.solver_options.globalization = "MERIT_BACKTRACKING"

        self.solver_options.nlp_solver_max_iter = 200
        self.solver_options.tol = 1e-6
        # self.solver_options.nlp_solver_tol_comp = 1e-2

        self.solver_options.print_level = 3
        # ocp.solver_options.nlp_solver_exact_hessian = True
        self.solver_options.qp_solver_warm_start = 0
        self.solver_options.regularize_method = "MIRROR"

    def waypoints_to_references(self, waypoints: np.ndarray) -> np.ndarray:
        references = np.zeros([self.N + 1, self.n_states + self.n_inputs])
        # TODO: comment here the sates given in waypoints
        references[:, :3] = np.concatenate((waypoints[:, :2], waypoints[:, 3:]), axis=1)
        return references

    def optimize(self, x0, waypoints, p, d_hat=np.array([[1.0]])):
        # print("x0: ", x0)
        starting_state = np.array([0, 0, 0, x0[4], x0[5], x0[6]])
        ref_points = self.waypoints_to_references(waypoints)

        for i in range(self.N):
            self.solver.cost_set(i, "yref", ref_points[i, :])
            self.solver.set(i, "p", np.concatenate((p[i].reshape((1, -1)), d_hat)))

        x_ref_e = np.array(
            [ref_points[self.N, 1], ref_points[self.N, 3], 0, 0, ref_points[self.N, 4]]
        )
        self.solver.set(self.N, "yref", ref_points[self.N, :])
        logging.debug(f"Ref: {ref_points[self.N, :]}")
        # self.set(0, "lbx", starting_state)
        # self.set(0, "ubx", starting_state)

        # status = self.solve()
        # trajectory = self.get_flat("x")
        # inputs = self.get_flat("u")

        u0 = self.solver.solve_for_x0(starting_state)

        runtime = self.solver.get_stats("time_tot")
        self.metrics["runtime"].append(runtime)

        logging.info(f"Solver runtime: {runtime * 1000} ms")

        # fish out the results from the solver
        trajectory = np.zeros([self.N + 1, self.n_states])
        inputs = np.zeros([self.N, self.n_inputs])
        for i in range(self.N):
            trajectory[i, :] = self.solver.get(i, "x")
            inputs[i, :] = self.solver.get(i, "u")
        trajectory[self.N, :] = self.solver.get(self.N, "x")

        logging.debug(f"Steering rate: \n {inputs[:15]}")
        logging.debug(f"Steering angle: \n {trajectory[:15, -1]}")
        logging.debug(f"Error: {trajectory[:15, 2] - waypoints[:15, 3]}")

        status = 0
        # Reconstruct trajectory in the original 7 state form
        trajectory = np.concatenate(
            (
                trajectory[:, :2],
                np.cos(trajectory[:, 2:3]),
                np.sin(trajectory[:, 2:3]),
                trajectory[:, 3:],
            ),
            axis=1,
        )

        return status, trajectory, inputs

    def stability(self):
        # np.set_printoptions(precision=1)
        if self.discrete:
            # This uses the specified discretization so either Runge Kutta 4 or Forward Euler
            A = cs.jacobian(self.model.disc_dyn_expr, self.model.x)
            B = cs.jacobian(self.model.disc_dyn_expr, self.model.u)
            A = cs.Function("A", [self.model.x, self.model.u, self.model.p], [A])
            B = cs.Function("B", [self.model.x, self.model.u, self.model.p], [B])
            A = np.array(
                A(self.x_lin_point, self.u_lin_point, self.p_lin_point),
                dtype=np.float32,
            )
            B = np.array(
                B(self.x_lin_point, self.u_lin_point, self.p_lin_point),
                dtype=np.float32,
            )
            A = A[
                1:, 1:
            ]  # remove x position because it is uncontrollable, it is onl there for the simulation
            B = B[1:, :]
        else:
            # This method uses Forward Euler discretization
            A = cs.jacobian(self.f, self.model.x)
            B = cs.jacobian(self.f, self.model.u)
            A = cs.Function("A", [self.model.x, self.model.u, self.model.p], [self.A])
            B = cs.Function("B", [self.model.x, self.model.u, self.model.p], [self.B])
            A = np.array(
                A(self.x_lin_point, self.u_lin_point, self.p_lin_point),
                dtype=np.float32,
            )
            B = np.array(
                B(self.x_lin_point, self.u_lin_point, self.p_lin_point),
                dtype=np.float32,
            )
            C = np.zeros((self.n_outputs, self.n_states))
            D = np.zeros((self.n_outputs, self.n_inputs))
            dsys = ct.ss(A, B, C, D, self.Tf / self.N)
            dsys = ct.sample_system(dsys, self.Tf / self.N)

        print(f"Cont state matrix:\n {A}")
        print(f"Cont input matric:\n {B}")
        print(f"Sampling time: {self.Tf / self.N}")

        # Get Q and R matrices from W matrix
        W = self.cost.W
        Q = np.array(
            [
                [W[0, 0], 0, 0, 0, 0, 0],
                [0, W[1, 1], 0, 0, 0, 0],
                [0, 0, W[2, 2], 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, W[3, 3]],
            ]
        )
        Q = Q[1:, 1:]
        R = self.cost.W[4, 4]

        # Check Conditions for existance of ARE solutio
        controllable = np.linalg.matrix_rank(ct.ctrb(A, B)) == A.shape[0]
        tmp = np.linalg.matrix_rank(ct.ctrb(A.T, Q)) == A.shape[0]
        # find Q such that modes on im axis are controllable
        eigenvalues = np.linalg.eigvals(A)

        first_nonzero_index = lambda array: (
            np.flatnonzero(array)[0] if np.any(array) else -1
        )

        pivot_indeces = np.apply_along_axis(
            first_nonzero_index, axis=1, arr=ct.ctrb(A, B)
        )
        print(f"Controllability matrix: \n {sp.Matrix(ct.ctrb(A, B)).rref()}")
        print(f"Eigenvalues: {eigenvalues}")
        print(f"System is controllable: {controllable}")
        print(f"System is stabilizable: ")
        print(f"Controllability of (A.T, Q): {tmp}".format())
        print(f"Solution to ARE exists: {tmp & controllable}")

        ### TERMINAL COST ###
        # STEP 1: Obtain terminal cost for (non reference tracking) quadratic cost
        #         using the solution of ARE
        if tmp & True:
            K, P, E = ct.dlqr(A, B, Q, R)
        else:
            print("Not attempting to solve ARE, using fake K matrix")
            return

        print(f"Solution of ARE: {P}")
        self.P = P


if __name__ == "__main__":
    N = 50
    Tf = 1
    ocp = OFLOcp(N, Tf)
    ocp.stability()
