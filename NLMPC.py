from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
import logging
import yaml


class NLOcp(AcadosOcp):
    def __init__(self, N, Tf, debug=False):
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
        with open("parameters_NL.yaml", "r") as file:
            params = yaml.safe_load(file)

        # Assign parameters to class attributes
        self.params = params
        ### Constants ###
        # Simulation constants
        self.N = N
        self.Tf = Tf
        self.n_states = 7
        self.n_inputs = 1
        self.n_outputs = 8
        # Dynamics constants
        self.m = params["model"]["m"]
        self.I_z = params["model"]["I_z"]
        self.wbase = params["model"]["wbase"]
        self.x_cg = params["model"]["x_cg"]
        self.lf = self.x_cg * self.wbase
        self.lr = (1 - self.x_cg) * self.wbase

        [self.Cf, self.Cr] = self.get_tyre_stiffness()

        self.max_steering = params["model"]["max_steering_angle"]
        self.max_steering_rate = params["model"]["max_steering_rate"]

        # Model setup
        self.model = AcadosModel()

        # Model name
        self.model.name = "NonlinearDynamicBycicleModel"

        ### Decision variables ###
        self.model.u = cs.MX.sym("u", self.n_inputs, 1)
        self.model.x = cs.MX.sym("x", self.n_states, 1)
        self.model.xdot = cs.MX.sym("xdot", self.n_states, 1)

        ### Parameters ###
        self.model.p = cs.MX.sym("p", 1, 1)

        # Set model dynamics
        self.set_dynamics()
        # Set constraints
        self.set_constraints()
        # Set cost
        self.set_cost()
        # Set solver options
        self.set_solver_options()
        # Create solver
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

    def set_dynamics(self) -> None:
        p_x = self.model.x[0, 0]
        p_y = self.model.x[1, 0]
        cos_heading = self.model.x[2, 0]
        sin_heading = self.model.x[3, 0]
        v_x = self.model.p[0, 0]
        v_y = self.model.x[4, 0]
        r = self.model.x[5, 0]
        wheel_angle = self.model.x[6, 0]

        steering_rate = self.model.u[0, 0]

        # d_p_x = v_x * cos_heading - v_y * sin_heading
        d_p_x = v_x * cos_heading - v_y * sin_heading

        d_p_y = v_x * sin_heading + v_y * cos_heading

        d_cos_heading = -sin_heading * r
        d_sin_heading = cos_heading * r

        d_v_y = -(self.Cf + self.Cr) / (self.m * v_x + 0.1) * v_y
        d_v_y += (
            (-v_x + (self.Cr * self.lr - self.Cf * self.lf)) / (self.m * v_x + 0.1) * r
        )
        d_v_y -= self.Cf / self.m * wheel_angle

        d_r = (self.lr * self.Cr - self.lf * self.Cf) / (self.I_z * v_x + 0.1) * v_y
        d_r += (
            -(self.lf * self.lf * self.Cf + self.lr * self.lr * self.Cr)
            / (self.I_z * v_x + 0.1)
            * r
        )
        d_r -= self.lf * self.Cf / self.I_z * wheel_angle

        d_steering = steering_rate

        f = cs.vertcat(
            d_p_x, d_p_y, d_cos_heading, d_sin_heading, d_v_y, d_r, d_steering
        )

        self.model.f_expl_expr = f
        self.model.f_impl_expr = self.model.xdot - f

    def set_constraints(self) -> None:
        # Bounds for self.model.x
        self.constraints.idxbx = np.array([6])
        self.constraints.lbx = np.array([-self.max_steering])
        self.constraints.ubx = np.array([self.max_steering])

        # Intial condition
        # self.constraints.x0 = np.array([0, 0, 1, 0, 0, 0, 0])

        self.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4, 5, 6])
        self.constraints.lbx_0 = np.array([0, 0, 1, 0, 0, 0, 0])
        self.constraints.ubx_0 = np.array([0, 0, 1, 0, 0, 0, 0])

        # Bounds for input
        self.constraints.idxbu = np.array(
            [0]
        )  # the 0th input has the constraints, so J_bu = [1]
        self.constraints.lbu = np.array([-self.max_steering_rate])
        self.constraints.ubu = np.array([self.max_steering_rate])

    def set_cost(self) -> None:

        # Output selection matrix
        Vx = np.eye(self.n_outputs, self.n_states)
        # Vx[[(4, 4), (5, 5)]] = 0
        # Vx[4, 6] = 1

        self.cost.Vx_e = Vx
        self.cost.Vx = Vx

        Vu = np.array([[0], [0], [0], [0], [0], [0], [0], [1]])
        self.cost.Vu = Vu

        self.cost.cost_type = "LINEAR_LS"
        self.cost.cost_type_e = "LINEAR_LS"
        Q = np.array([0, 1e5, 1e-10, 1e0, 0, 1e0, 1e0])
        R = np.array([1e0]) * 10
        Qe = np.array([0, 1e5, 1e-10, 1e0, 0, 1e0, 1e0]) * 100
        # Cost matrices
        self.cost.W = np.diag(np.append(Q, R))
        self.cost.W_e = np.diag(np.append(Qe, 0))

        # Reference trajectory
        self.cost.yref = np.array([0, 0, 1, 0, 0, 0, 0, 0])
        self.cost.yref_e = np.array([0, 0, 1, 0, 0, 0, 0, 0])

        # Initial parameter trajectory
        self.parameter_values = np.array([9.0])

    def set_solver_options(self, acados_print_level=0) -> None:
        # set QP solver and integration
        self.solver_options.tf = self.Tf
        self.solver_options.N_horizon = self.N
        # ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        self.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.solver_options.nlp_solver_type = "SQP"
        self.solver_options.hessian_approx = "EXACT"
        self.solver_options.integrator_type = "ERK"
        self.solver_options.globalization = "MERIT_BACKTRACKING"

        self.solver_options.nlp_solver_max_iter = 200
        self.solver_options.tol = 1e-4
        # self.solver_options.nlp_solver_tol_comp = 1e-2

        self.solver_options.print_level = acados_print_level
        # ocp.solver_options.nlp_solver_exact_hessian = True
        self.solver_options.qp_solver_warm_start = 0
        self.solver_options.regularize_method = "MIRROR"

    def waypoints_to_references(self, waypoints):
        references = np.zeros([self.N + 1, self.n_outputs])
        references[:, :4] = waypoints[:, :]
        # references[:, 4] = 0.00
        return references

    def optimize(self, x0, waypoints, p):
        # print("x0: ", x0)
        starting_state = np.array([0, 0, 1, 0, x0[4], x0[5], x0[6]])
        ref_points = self.waypoints_to_references(waypoints)
        # print(ref_points)

        for i in range(self.N):
            self.solver.cost_set(i, "yref", ref_points[i, :])
            self.solver.set(i, "p", p[i])

        self.solver.set(self.N, "yref", ref_points[self.N, :])
        # self.set(0, "lbx", starting_state)
        # self.set(0, "ubx", starting_state)

        # status = self.solve()
        # trajectory = self.get_flat("x")
        # inputs = self.get_flat("u")

        u0 = self.solver.solve_for_x0(starting_state)

        runtime = self.solver.get_stats("time_tot")
        self.metrics["runtime"].append(runtime)
        logging.info(f"Solver runtime: {runtime*1000} ms")

        # fish out the results from the solver
        trajectory = np.zeros([self.N + 1, self.n_states])
        inputs = np.zeros([self.N, self.n_inputs])
        for i in range(self.N):
            trajectory[i, :] = self.solver.get(i, "x")
            inputs[i, :] = self.solver.get(i, "u")
        trajectory[self.N, :] = self.solver.get(self.N, "x")
        logging.debug(inputs[:15])
        logging.debug(trajectory[:15, -1])

        # plt.plot(trajectory[:, -1], label="angle")
        # plt.plot(inputs, label="rate")
        # plt.show()
        status = 0
        return status, trajectory, inputs
