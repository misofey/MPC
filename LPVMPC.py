from matplotlib import pyplot as plt
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
import casadi as cs
import numpy as np
import logging
import yaml


class LPVOcp(AcadosOcp):
    def __init__(self, N:int, Tf:float, discrete:bool = True, debug:bool = False):
        AcadosOcp.__init__(self)

        # Initialize logging
        if debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format="[%(asctime)s][%(levelname)s] - %(message)s",
                handlers=[
                    logging.FileHandler("debug.log"),
                    logging.StreamHandler()
                ]
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s][%(levelname)s] - %(message)s",
                handlers=[
                    logging.FileHandler("debug.log"),
                    logging.StreamHandler()
                ]
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
        self.n_params = 1
        self.n_outputs = 5

        # Dynamics constants
        self.m = params["model"]['m']
        self.I_z = params["model"]['I_z']
        self.wbase = params["model"]['wbase']
        self.x_cg = params["model"]['x_cg']
        self.lf = self.x_cg * self.wbase
        self.lr = (1 - self.x_cg) * self.wbase 

        [self.Cf, self.Cr] = self.get_tyre_stiffness()

        self.max_steering = params["model"]['max_steering_angle']
        self.max_steering_rate = (
            3 * self.max_steering
        )  # one second from full left to full right

        # Linearization point
        self.x_lin_point = np.array([0, 0, 0, 0, 0, 0])
        self.u_lin_point = np.array([0])
        self.p_lin_point = np.array([15.0])
        self.prev_x = np.array([[0, 0, 0, 0, 0, 0]])
        self.prev_x = np.repeat(self.prev_x, self.N+1, 0).T
        self.prev_u = np.array([[0]])
        self.prev_u = np.repeat(self.prev_u, self.N+1, 1)

        # Model setup
        self.model = AcadosModel()

        # Model name
        self.model.name = "NonlinearDynamicBycicleModel"

        ### Decision variables ###
        self.model.u = cs.MX.sym("u", self.n_inputs)
        self.model.x = cs.MX.sym("x", self.n_states)
        self.model.xdot = cs.MX.sym("xdot", self.n_states)

        ### Parameters ###
        self.model.p = cs.MX.sym("p", self.n_states+self.n_inputs+self.n_params)

        # Set model dynamics
        self.set_dynamics()
        # Set constraints
        self.set_constraints()
        # Set cost
        self.set_cost()
        # Set solver options
        self.set_solver_options()
        self.solver = AcadosOcpSolver(self)

        self.metrics = {
            "runtime": [],
            "cost": [],
            "status": [],
        }

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
        heading = self.model.x[2, 0]
        v_y = self.model.x[3, 0]
        omega = self.model.x[4, 0]
        steering_angle = self.model.x[5, 0]

        steering_rate = self.model.u[0, 0]

        v_x = self.model.p[self.n_states, 0]
        logging.info(v_x)

        # Nonlinear dynamics
        d_p_x = v_x * np.cos(heading) - v_y * np.sin(heading)

        d_p_y = v_x * np.sin(heading) + v_y * np.cos(heading)

        d_heading = omega

        d_v_y = -(self.Cf + self.Cr) / (self.m * v_x + 0.001) * v_y
        d_v_y += (
            (-v_x + (self.Cr * self.lr - self.Cf * self.lf)) / (self.m * v_x + 0.001) * omega
        )
        d_v_y -= self.Cf / self.m * steering_angle

        d_omega = (self.lr * self.Cr - self.lf * self.Cf) / (self.I_z * v_x + 0.001) * v_y
        d_omega += (
            -(self.lf * self.lf * self.Cf + self.lr * self.lr * self.Cr)
            / (self.I_z * v_x + 0.001)
            * omega
        )
        d_omega -= self.lf * self.Cf / self.I_z * steering_angle

        d_steering = steering_rate

        self.f = cs.vertcat(
            d_p_x, d_p_y, d_heading, d_v_y, d_omega, d_steering
        )

        self.A = cs.jacobian(self.f, self.model.x)
        self.A = cs.Function("A", [self.model.x, self.model.u, self.model.p], [self.A])
        self.A = self.A(self.x_lin_point, self.u_lin_point, self.model.p)
        self.B = cs.jacobian(self.f, self.model.u)
        self.B = cs.Function("B", [self.model.x, self.model.u, self.model.p], [self.B])
        self.B = self.B(self.x_lin_point, self.u_lin_point, self.model.p) 


        self.model.f_expl_expr = self.A @ (self.model.x) + self.B @ (self.model.u) + cs.vertcat(v_x, 0, 0, 0, 0, 0)
        self.model.f_impl_expr = self.model.xdot - (self.A @ self.model.x + self.B @ self.model.u)


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
        self.constraints.idxbu = np.array([0])  # the 0th input has the constraints, so J_bu = [1]
        self.constraints.lbu = np.array([-self.max_steering_rate])
        self.constraints.ubu = np.array([ self.max_steering_rate])

    def set_cost(self) -> None:

        # Output selection matrix
        Vx = np.eye(self.n_states+self.n_inputs, self.n_states)
        
        self.cost.Vx = Vx
        self.cost.Vx_e = Vx

        Vu = np.zeros((self.n_states+self.n_inputs, self.n_inputs))
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
        self.cost.W = np.zeros((self.n_states+self.n_inputs, self.n_states+self.n_inputs))
        self.cost.W[:self.n_states, :self.n_states] = Q*q
        self.cost.W[self.n_states:, self.n_states:] = R*r
        self.cost.W_e = self.cost.W

        # Reference trajectory
        self.cost.yref = np.zeros(self.n_states+self.n_inputs)
        self.cost.yref_e =  np.zeros(self.n_states+self.n_inputs)

        # Initial parameter trajectory
        self.parameter_values = np.concatenate((self.prev_x[:, 0], np.array([15.0]), self.prev_u[:, 0]))


    def set_solver_options(self) -> None:
        # set QP solver and integration
        self.solver_options.tf = self.Tf
        self.solver_options.N_horizon = self.N
        # self.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        self.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.solver_options.nlp_solver_type = "SQP"
        self.solver_options.hessian_approx = "EXACT"
        self.solver_options.integrator_type = "ERK"
        self.solver_options.globalization = "MERIT_BACKTRACKING"

        self.solver_options.nlp_solver_max_iter = 200
        self.solver_options.tol = 1e-4
        # self.solver_options.nlp_solver_tol_comp = 1e-2

        self.solver_options.print_level = 0
         # ocp.solver_options.nlp_solver_exact_hessian = True
        self.solver_options.qp_solver_warm_start = 0
        self.solver_options.regularize_method = "MIRROR"

    def waypoints_to_references(self, waypoints:np.ndarray) -> np.ndarray:
        references = np.zeros([self.N + 1, self.n_states + self.n_inputs])
        #TODO: comment here the sates given in waypoints
        references[:, :3] = np.concatenate((
            waypoints[:, :2],
            waypoints[:, 3:]), axis=1)
        return references

    def optimize(self, x0, waypoints, p, lin_mode:str = "prev_iter"):
        starting_state = np.array([0, 0, 0, x0[4], x0[5], x0[6]])
        ref_points = self.waypoints_to_references(waypoints)

        if lin_mode == "prev_iter":
        # Use previous iteration as lin point
            for i in range(self.N):
                self.solver.cost_set(i, "yref", ref_points[i, :])
                self.solver.set(i, "p", np.array([
                                                self.prev_x[0, i+1], # x
                                                self.prev_x[1, i+1], # y
                                                self.prev_x[2, i+1], # sin heading 
                                                self.prev_x[3, i+1], # v_y
                                                self.prev_x[4, i+1], # r
                                                self.prev_x[5, i+1], # steering angle
                                                p[i],                # v_x
                                                self.prev_u[0, i]  # steering rate
                                                
                    ]))
                
        elif lin_mode == "reference":
            for i in range(self.N):
                self.solver.cost_set(i, "yref", ref_points[i, :])
                self.solver.set(i, "p", np.array([
                                                ref_points[i, 0],    # x
                                                ref_points[i, 1],    # y
                                                ref_points[i, 3],    # sin heading
                                                self.prev_x[3, i+1], # v_y
                                                self.prev_x[4, i+1], # r
                                                self.prev_x[5, i+1], # steering angle
                                                p[i],                # v_x
                                                self.prev_u[0, i]
                    ]))


        # self.set(self.ocp.N, "yref", ref_points[self.ocp.N, :])
        # self.set(0, "lbx", starting_state)
        # self.set(0, "ubx", starting_state)
        #
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
        self.prev_x = np.array([
            trajectory[: ,0], # x
            trajectory[: ,1], # y
            trajectory[: ,2], # cos heading
            trajectory[: ,3], # sin heading 
            trajectory[:, 4], # v_y
            trajectory[:, 5], # r# steering angle
        ])
        self.prev_u = np.array([
            inputs[:, 0],     # steering rate
        ])

        status = 0

        trajectory = np.concatenate((
            trajectory[:, :2],
            np.cos(trajectory[:, 2:3]),
            np.sin(trajectory[:, 2:3]),
            trajectory[:, 3:]
            ), axis=1)

        return status, trajectory, inputs

    
if __name__ == "__main__":
    N = 100
    Tf = 0.5
    ocp = NLOcp(N, Tf)
    x0 = np.array([0, 0, 1, 0, 0, 0, 0])
    ref_points = np.ones((N, 6))*0.01
    p = np.array([1.0])
    status, trajectory = ocp.solve_problem(x0, ref_points, p)
    plt.plot(trajectory[0, :], trajectory[1, :])
    plt.show()