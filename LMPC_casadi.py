import casadi as cs
import numpy as np

class LMPCCasAdi:
    def __init__(self):

        self.n_states = 7
        self.n_inputs = 2
        self.N = 0
        self.t_f = 1
        
        self.problem = cs.Opti

        # Parameters
        self.x_lin_point = self.problem.parameter(self.n_states, 1)
        self.x_0 = self.problem.parameter(self.n_states, self.N)
        self.reference_trajectory = self.problem.parameter(2, self.N)
        
        # Decision variables
        self.u = self.problem.variable(self.n_inputs, self.N)

    def set_cost(self) -> None:
        state_trajectory = cs.MX(self.n_states, self.N-1)
        #TODO: Dont constrain the initial state
        state_trajectory[:, 0] = self.x_0
        for i in range(self.N-1):
            state_trajectory[:, i+1] = self.A @ state_trajectory[:, i] + self.B @ self.u[:, i]

        cost = (state_trajectory[:2, :]-self.reference_trajectory)
        cost = cost[0, :]**2 + cost[1, :]**2
        cost = cost @ cost.T

        self.problem.minimize(cost)
        

    def set_constraints(self) -> None:
        constraints = []
        state_trajectory = cs.MX(self.n_states, self.N-1)
        #TODO: Dont constrain the initial state
        state_trajectory[:, 0] = self.x_0
        for i in range(self.N-1):
            state_trajectory[:, i+1] = self.A @ state_trajectory[:, i] + self.B @ self.u[:, i]

        # Constrain states so linearization holds
        constraints.append(state_trajectory[2, 2:] < self.x_lin_point[2]*self.error_tol) # longitudinal speed
        constraints.append(state_trajectory[3, 2:] < self.x_lin_point[3]*self.error_tol) # lateral speed
        constraints.append(state_trajectory[4, 2:] < self.x_lin_point[4]*self.error_tol) # sin of steering angle
        constraints.append(state_trajectory[5, 2:] < self.x_lin_point[5]*self.error_tol) # cos of steering angle
        constraints.append(state_trajectory[6, 2:] < self.x_lin_point[6]*self.error_tol) # angular acceleration

        # Constrain inputs 
        constraints.append(self.u[0, 2:] <  5/180*np.pi)
        constraints.append(self.u[0, 2:] > -5/180*np.pi)
        #constraints.append(self.u[1, 2:] < )
        #constraints.append(self.u[1, 2:] < )
        # Contrain rate of change of inputs
        #constraints.append(self.u[0, 2:-1] - self.u[0, 3:] < )
        #constraints.append(self.u[0, 2:-1] - self.u[0, 3:] > )
        #constraints.append(self.u[1, 2:-1] - self.u[1, 3:] < )
        #constraints.append(self.u[1, 2:-1] - self.u[1, 3:] > )

        # Trigonometric constraint
        constraints.append(state_trajectory[2, 2:]*state_trajectory[2, 2:] + state_trajectory[3, 2:]*state_trajectory[3, 2:] == 1)

        self.problem.subject_to(constraints)

    def set_dynamics(self) -> None:
        sin_delta = self.x_lin_point[2, :]
        cos_delta = self.x_lin_point[3, :] 
        v_tx = self.x_lin_point[4, :]
        v_ty = self.x_lin_point[5, :]
        omega = self.x_lin_point[6, :]

        self.A = cs.DM.zeros(self.n_states, self.n_states)

        self.A[0, 2] = -v_ty
        self.A[0, 3] = v_tx
        self.A[0, 4] = -sin_delta
        self.A[1, 2] = v_tx
        self.A[1, 3] = v_ty
        self.A[1, 4] = cos_delta
        self.A[2, 2] = -omega
        self.A[2, 6] = -sin_delta
        self.A[3, 3] = omega
        self.A[3, 6] = cos_delta



