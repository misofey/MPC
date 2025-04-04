from matplotlib import pyplot as plt
import osqp
import scipy.sparse as sparse
import casadi as cs
import numpy as np
import control as ct
import sympy as sp
from pprint import pprint as pr
import yaml
import logging
import pandas as pd


class TargetSelector:

    def __init__(self, N:int, Tf:float, debug:bool = False):

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
        # Simulation constants
        self.N = N
        self.Tf = Tf
        #  All states: x, y, heading, v_y, omega, steering_angle
        self.n_states = 6
        self.n_disturbances = 1
        self.n_inputs = 1
        # Measured states: x, y, omega, steering_angle
        self.n_outputs = 5
        # Dynamics constants
        self.m = params["model"]['m']  # Car mass [kg]
        self.I_z = params["model"]['I_z']  # TODO: unit
        self.wbase = params["model"]['wbase']  # wheel base [m]
        self.x_cg = params["model"]['x_cg']  # C.G x location [m]
        self.lf = self.x_cg * self.wbase  # Front moment arm [m]
        self.lr = (1 - self.x_cg) * self.wbase  # Rear moment arm [m]

        [self.Cf, self.Cr] = self.get_tyre_stiffness()

        self.max_steering = params["model"]['max_steering_angle']
        self.max_steering_rate = (
            3 * self.max_steering
        )  # one second from full left to full right

        # Linearization point
        self.x_lin_point = np.array([0, 0, 0, 0, 0, 0])
        self.u_lin_point = np.array([0])
        self.p_lin_point = np.array([15.0])

        self.d_hat = 0.5*np.ones((self.n_disturbances, 1))
        self.y_ref = np.zeros((self.N, self.n_outputs))

        # Set cost
        self.set_cost()
        # Set dynamics
        self.set_dynamics()
        # Set constraints
        self.set_constraints()
        # Set solver
        self.solver = osqp.OSQP()
        self.solver.setup(self.P, self.q, self.A_eq, self.l_eq, self.u_eq, verbose=False)
        self.solver.warm_start()
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


    def set_dynamics(self):
        x = cs.MX.sym("x", self.n_states)
        u = cs.MX.sym("u", self.n_inputs)
        p = cs.MX.sym("p", 1)
        p_x = x[0, 0]
        p_y = x[1, 0]
        heading = x[2, 0]
        v_y = x[3, 0]
        omega = x[4, 0]
        steering_angle = x[5, 0]

        steering_rate = u[0, 0]

        v_x = p[0, 0]

        d_p_x = v_x * 1 #- v_y * np.sin(heading)

        d_p_y = v_x * heading + v_y * 1

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

        self.A = cs.jacobian(self.f, x)
        self.A = cs.Function("A", [x, u, p], [self.A])
        self.A = self.A(self.x_lin_point, self.u_lin_point, self.p_lin_point) * (self.Tf/self.N) + np.eye(self.n_states)

        self.B = cs.jacobian(self.f, u)
        self.B = cs.Function("B", [x, u, p], [self.B])
        self.B = self.B(self.x_lin_point, self.u_lin_point, self.p_lin_point) * (self.Tf/self.N)

        self.B_d = np.eye(self.n_states, self.n_disturbances)
        self.C_d = np.zeros((self.n_outputs, self.n_disturbances))
        self.C = np.ones((self.n_outputs, self.n_states))

    def apply_rk4_with_linearized_matrices(self):
        
        dt = self.Tf / self.N # Time step

        # TODO: The linearization point should be from x and u 
        # idk why it works like this but not like that
        x_dot = lambda x, u: self.A @ (x-self.x_lin_point) + self.B @ (u-self.u_lin_point) - self.x_lin_point

        # Compute RK4 steps using linearized system
        k1 = x_dot(self.model.x, self.model.u)
        k2 = x_dot(self.model.x + (dt / 2) * k1, self.model.u)
        k3 = x_dot(self.model.x + (dt / 2) * k2, self.model.u)
        k4 = x_dot(self.model.x + dt * k3, self.model.u)
    

        # Compute the RK4 update step
        x_next = self.model.x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Set the discretized dynamics
        self.model.disc_dyn_expr = x_next


    def set_constraints(self) -> None:
        
        first_row = np.hstack((self.A, self.B, np.eye(self.n_states)))
        second_row = np.hstack([np.zeros((self.n_outputs, self.n_states)), self.C, np.zeros((self.n_outputs, self.n_inputs))])
        E = sparse.vstack([first_row, second_row])
        A_eq = sparse.csc_matrix((self.N * (self.n_states + self.n_outputs), self.N * (2*self.n_states + self.n_inputs)))
        for i in range(self.N):
            if i == 0:
                tmp = 0
            else:
                tmp = 1
            i_start = i * (self.n_states + self.n_outputs)
            i_end = (i + 1) * (self.n_states + self.n_outputs)
            j_start = i * (2*self.n_states + self.n_inputs) - self.n_states*tmp
            j_end = (i + 1) * (2*self.n_states + self.n_inputs) - self.n_states*tmp
            print(j_end - j_start)
            print(i_end - i_start)
            print(E.shape)
            A_eq[i_start: i_end, j_start: j_end] = E
            
        # Transform matrices in sparse form for the horizon
        df = pd.DataFrame(A_eq[:5*self.n_states, :5*self.n_states].toarray())
        print(df)
        initial_condition = np.hstack([np.eye(self.n_states), np.zeros((self.n_states, self.N*(2*self.n_states + self.n_inputs)-self.n_states))])
        self.A_eq = sparse.vstack([initial_condition, A_eq])
        

        # RHS
        y = np.zeros((self.n_states, self.n_outputs))
        E_rhs = sparse.vstack([self.B_d@self.d_hat, -self.C_d@self.d_hat])
        print(E_rhs.shape)
        E_rhs = np.repeat(E_rhs, self.N, 0)
        print(E_rhs.shape)

        
        u_bounds_min = np.tile(-self.max_steering_rate, (self.N, 1))
        u_bounds_max = np.tile(self.max_steering_rate, (self.N, 1))

        #self.l_eq = np.hstack([rhs, x_bounds_min_statesn, u_bounds_min])
        #self.u_eq = np.hstack([rhs, x_bounds_max, u_bounds_max])    
        self.l_eq = np.vstack((E_rhs))
        self.u_eq = np.vstack([E_rhs])    

        print(f"A_eq shape: {self.A_eq.shape}")
        print(f"l_eq shape: {self.l_eq.shape}")
        print(f"u_eq shape: {self.u_eq.shape}")

    def set_cost(self) -> None:

        Q = np.eye(self.n_states)  # Identity matrix for output cost
        R = np.eye(self.n_inputs)  # Identity matrix for input cost


        # Quadratic cost matrix
        Q_big = sparse.kron(sparse.eye(self.N), Q)
        R_big = sparse.kron(sparse.eye(self.N), R)
        self.P = sparse.block_diag([Q_big, R_big])

        self.q = np.zeros(self.N * (self.n_states + self.n_inputs))

        return None

    def optimize_target(self, x_0_hat, waypoints, d_hat=[0.5]) -> np.ndarray:

        # Update RHS of equality costraint to ensure initial conditions and disturbance value
        # Compute new right-hand side for equality constraints
        # Add A @ x0 to first stage dynamics (only the first nx rows)
        y_ref = self.waypoints_to_references(waypoints)
        x0_term = np.zeros(self.N * self.n_states)
        x0_term[:self.n_x] = self.A @ x_0_hat

        # Compute new right-hand side for equality constraints
        rhs = np.hstack([self.Bd_big @ d_hat + x0_term, y_ref - self.Cd_big @ d_hat])

        # Bounds
        l = np.hstack([rhs, self.u_bounds_min])
        u = np.hstack([rhs, self.u_bounds_max])

        # Update solver constraints
        self.solver.update(l=l, u=u)

        sol = self.solver.solve()
        if sol.info.status_val != osqp.OSQP_SOLVED:
            logging.error(f"OSQP solver failed: {sol.info.status}")
            return None
        
        # Extract the solution  
        x_sol = sol.x[:self.N * self.n_states].reshape(self.N, self.n_states)
        u_sol = sol.x[self.N * self.n_states :].reshape(self.N, self.n_inputs)

        return x_sol, u_sol


    def waypoints_to_references(self, waypoints:np.ndarray) -> np.ndarray:
        references = np.zeros([self.N + 1, self.n_states + self.n_inputs])
        #TODO: comment here the sates given in waypoints
        references[:, :3] = np.concatenate((
            waypoints[:, :2],
            waypoints[:, 3:]), axis=1)
        return references


if __name__ == "__main__":
    N = 50
    Tf = 1
    ocp = LOcp(N, Tf)
    ocp.stability()
