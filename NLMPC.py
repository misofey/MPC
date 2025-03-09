from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
import casadi as cs
import numpy as np


class NLOcp(AcadosOcp):
    def __init__(self, N, Tf):
        AcadosOcp.__init__()

        ### Constants ###
        # Simulation constants
        self.N = N
        self.Tf = Tf
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

        self.max_steering = 0.4
        self.max_steering_rate = (
            2 * self.max_steering
        )  # one second from full left to full right

        # Model setup
        self.model = AcadosModel()

        # Model name
        self.model.name = "NonlinearDynamicBycicleModel"

        ### Decision variables ###
        self.model.u = cs.MX.sym(self.n_inputs, 1)
        self.model.x = cs.MX.sym(self.n_states, 1)
        self.model.xdot = cs.MX.sym(self.n_states, 1)

        ### Parameters ###
        self.model.p = cs.MX.sym(1, 1)

        # Set model dynamics
        self.set_dynamics()
        # Set constraints
        self.set_constraints()
        # Set cost
        self.set_cost()

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

    def set_dynamics(self) -> None:
        p_x = self.model.x[0, 0]
        p_y = self.model.x[1, 0]
        cos_heading = self.model.x[2, 0]
        sin_heading = self.model.x[3, 0]
        v_y = self.model.x[4, 0]
        v_x = self.model.p[0, 0]
        r = self.model.x[5, 0]
        wheel_angle = self.model.x[6, 0]

        steering_rate = self.model.u[0, 0]

        d_p_x = v_x * cos_heading - v_y * sin_heading
        d_p_y = v_x * sin_heading + v_y * cos_heading

        d_cos_heading = -sin_heading * r
        d_sin_heading = cos_heading * r

        d_v_y = -(self.Cf * self.Cr) * v_y
        d_v_y += (-v_x + (self.Cr * self.lr - self.Cf * self.lf) / (self.m * v_x)) * r
        d_v_y -= self.Cf * self.m * wheel_angle

        d_r = (self.lr * self.Cr - self.lf * self.Cf) / self.I_z * v_y
        d_r += (
            -(self.lf * self.lf * self.Cf + self.lr * self.lr * self.Cr)
            / (self.I_z * v_x)
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

        # Intial condition
        self.constraints.x0 = np.array([0, 0, 1, 0, 0, 0, 0])

        # Bounds for self.model.x
        self.constraints.idxbx = np.array([6])
        self.constraints.lbx = np.array([-self.max_steering])
        self.constraints.ubx = np.array([self.max_steering])

        # Bounds for input
        self.constraints.idxbu = np.array(
            [0]
        )  # the 0th input has the constraints, so J_bu = [1]
        self.constraints.lbu = np.array([-self.max_steering])
        self.constraints.ubu = np.array([self.max_steering])

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

        # Reference trajectory
        self.cost.yref = np.array([0, 0, 1, 0, 0, 0])
        self.cost.yref_e = np.array([0, 0, 1, 0, 0, 0])

        # Initial parameter trajectory
        self.parameter_values = np.array([7.0])


class NLSolver(AcadosOcpSolver):
    def __init__(self, ocp: NLOcp):
        # Set solver options in OCP
        self.ocp = ocp
        self.set_solver_options(self.ocp)
        super().__init__(self.ocp)

    def set_solver_options(self, ocp: NLOcp) -> None:
        # set QP solver and integration
        ocp.solver_options.tf = ocp.Tf
        ocp.solver_options.N_horizon = ocp.N
        # self.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.integrator_type = "ERK"

        ocp.solver_options.nlp_solver_max_iter = 200
        ocp.solver_options.tol = 1e-4
        # self.solver_options.nlp_solver_tol_comp = 1e-2

        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_exact_hessian = True
        ocp.solver_options.qp_solver_warm_start = 0

    def solve_problem(self, x0, ref_points, p):
        self.set(0, "lbx", x0)
        self.set(0, "ubx", x0)

        for i in range(self.ocp.N):
            self.cost_set(i, "y_ref", ref_points[i, :])
            self.set(i, "p", p[i])

        status = self.solve()
        trajectory = self.get_flat("x")

        return status, trajectory
