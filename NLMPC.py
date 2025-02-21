from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
import casadi as cs
import types
import numpy as np
import matplotlib.pyplot as plt


class NLOptimizer(AcadosOcp):

    def __init__(self, filename="config.yaml"):
        AcadosOcp.__init__()

        ### Constants ###
        # Simulation constants
        self.N = 30
        self.Tf = 2
        self.n_states = 7
        self.n_inputs = 1
        self.n_outputs = 6
        # Dynamics constants
        self.m = 180  # Car mass [kg]
        self.I_z = 294  # TODO: unit
        self.wbase = 1.53  # wheel base [m]
        self.x_cg = 0.57  # C.G x location [m]
        self.lf = self.x_cg * self.wbase  # Front moment arm [m]
        self.lr = (1 - self.x_cg) * self.wbase  # Rear moment arm [m]

        [self.Cf, self.Cr] = self.get_tyre_stiffness()

        # Model name
        self.model.name = "NonlinearDynamicBycicleModel"

        ### Decision variables ###
        self.model.u = cs.MX.sym("Steering_rate", self.n_inputs, self.N)
        self.model.x = cs.MX.sym(self.n_states, self.N)
        self.model.xdot = cs.MX.sym(self.n_states, self.N)

        ### Parameters ###
        self.model.p = cs.DM.sym("x_speed", 1, self.N)

        # Set model dynamics
        self.set_dynamics()
        # Set constraints
        self.set_constraints()
        # Set cost
        self.set_cost()
        # Set solver options
        self.set_solver()

    def set_solver(self) -> None:

        # set QP solver and integration
        self.solver_options.tf = self.Tf
        self.solver_options.N_horizon = self.N
        # self.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        self.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.solver_options.nlp_solver_type = "SQP"
        self.solver_options.hessian_approx = "EXACT"
        self.solver_options.integrator_type = "ERK"

        self.solver_options.nlp_solver_max_iter = 200
        self.solver_options.tol = 1e-4
        # self.solver_options.nlp_solver_tol_comp = 1e-2

        self.solver_options.print_level = 0
        self.solver_options.nlp_solver_exact_hessian = True
        self.solver_options.qp_solver_warm_start = 0

    def set_cost(self) -> None:

        # Output selection matrix
        Vx = np.eye(self.n_outputs, self.n_states)
        Vx[[(4, 4), (5, 5)]] = 0
        Vx[4, 6] = 1
        self.cost.Vx_e = Vx
        self.cost.Vx = Vx

        Vu = np.array([[0], [0], [0], [0], [0], [1]])
        self.cost.Vu = Vu
        self.cost.Vu_e = Vu

        self.cost.cost_type = "LINEAR_LS"
        self.cost.cost_type_e = "LINEAR_LS"

        # Cost matrices
        self.cost.W = np.array([1.0, 1.0, 1e-3, 1e-1, 0, 1e-5, 0.0]) * 0.1
        self.cost.W_e = np.array([1, 1, 0.7, 0.7, 0, 0, 0]) * 0.01

    def set_constraints(self) -> None:

        # Intial condition
        self.constraints.x0 = np.array([0, 0, 1, 0, 0, 0, 0])

        # Bounds for self.model.x
        self.constraints.idxbx = np.array([6])
        self.constraints.lbx = np.array([-0.4])
        self.constraints.ubx = np.array([0.4])

        # Bounds for input
        self.constraints.idxbu = np.array(
            [0]
        )  # the 0th input has the constraints, so J_bu = [1]
        self.constraints.lbu = np.array([-10])
        self.constraints.ubu = np.array([10])

    def set_dynamics(self) -> None:
        p_x = self.model.x[0, :]
        p_y = self.model.x[1, :]
        cos_heading = self.model.x[2, :]
        sin_heading = self.model.x[3, :]
        v_y = self.model.x[4, :]
        v_x = self.model.p[1, :]
        r = self.model.x[5, :]
        steering = self.model.x[6, :]

        steering_rate = self.model.u[0, :]

        d_p_x = (
            v_x * cos_heading + v_y * sin_heading
        )  # TODO check sign here because it conflicts with nuclino
        d_p_y = -v_x * sin_heading + v_y * cos_heading

        d_cos_heading = -sin_heading * r
        d_sin_heading = cos_heading * r

        d_v_y = -(self.Cf * self.Cr) * v_y
        d_v_y += (v_x - (self.Cf * self.lf - self.Cr * self.lr) / (self.m * v_x)) * r
        d_v_y += self.Cf * self.m * steering

        d_r = (self.lf * self.Cf - self.lr * self.Cr) / self.I_z * v_y
        d_r += (
            -(self.lf * self.lf * self.Cf + self.lr * self.lr * self.Cr)
            / (self.I_z * v_x)
            * r
        )
        d_r += self.lf * self.Cf / self.I_z * steering

        d_steering = steering_rate

        f = cs.vcat(d_p_x, d_p_y, d_cos_heading, d_sin_heading, d_v_y, d_r, d_steering)

        self.model.f_expl_expr = f
        self.model.f_impl_expr = self.model.xdot - f

    def get_tyre_stiffness(self) -> (float, float):
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
